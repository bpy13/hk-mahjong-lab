//! Fits the route table for the `route` bot ([`mahjong_search::route`]) from
//! self-play.
//!
//! `fit_routes <hands> <bot spec> <out dir> [seed base]`: plays `<hands>` hands
//! (seeds from `[seed base]`, default 0) with the bot in all four seats. After
//! every discard, the discarder's 13-tile hand is recorded under the key of the
//! route the bot follows (for `chain` and `route`, the route it picked; for
//! other bots, the best route by the chain numbers): route × hits needed ×
//! weighted-helper band × tiles-left bucket × melds ([`RouteKey`]). The hand's
//! end adds whether that seat won, the points per share of the win, and the
//! rounds from the record to the win. Thin entries are shrunk toward coarser
//! ones, as `fit_targets` does: drop the melds, then the helper band, then the
//! wall band (route × needed), then the route (needed only), then the overall
//! rate. So every cell has a value, even with no records of its own, and the
//! `route` bot reads nothing else.
//! Writes `<out dir>/table.json` (every cell), `<out dir>/helpers.json` (the
//! mean weighted helpers per route × needed, for the `chain2` bot) and
//! `<out dir>/summary.txt`.
//!
//! Progress is saved per chunk of 500 hands in `<out dir>`; a rerun skips
//! finished chunks. `RAYON_NUM_THREADS` sets the threads; `ROUTE_EXPLORE` makes
//! the route bots follow their second-best route at that share of discards.
//!
//! `fit_routes diff <table a> <table b>`: the largest and mean absolute change
//! of P(win) over the entries with at least 2000 records in both.
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::route::{
    Route, RouteBot, RouteKey, RouteTable, HELPER_BANDS, NEEDED_BANDS, ROUTES,
    ROUTE_CELLS,
};
use mahjong_search::target_value::TILES_LEFT_BUCKETS;
use mahjong_search::{Agent, SeatView};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

/// Pseudo-observations pulling an entry toward its parent.
const SHRINK: f64 = 50.0;
/// Hands per saved chunk.
const CHUNK: u64 = 500;
/// Entries with this many records count as well measured in the summary and the diff.
const SOLID: f64 = 2000.0;

/// Sums for one table entry.
#[derive(Clone, Copy, Debug, Default)]
struct Sums {
    n: f64,
    wins: f64,
    points: f64,
    rounds: f64,
    /// Sum of the weighted helpers of the records.
    helpers: f64,
}

impl Sums {
    fn add(&mut self, o: &Sums) {
        self.n += o.n;
        self.wins += o.wins;
        self.points += o.points;
        self.rounds += o.rounds;
        self.helpers += o.helpers;
    }
}

/// One finished chunk: hands, wins declared, and the sums per entry.
struct Chunk {
    hands: u64,
    wins: u64,
    cells: Vec<Sums>,
}

/// Plays one hand; returns the records (entry, sums of one record) and the wins.
fn play(seed: u64, agent: &Agent, probe: &RouteBot) -> (Vec<(usize, Sums)>, u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    // (seat, route, entry, wall tiles left, weighted helpers): one per route per discard.
    let mut seen: Vec<(usize, Route, usize, usize, f32)> = Vec::new();
    while let Some(p) = s.current_player() {
        let legal = s.legal_actions_vec();
        let wall = s.wall_remaining();
        let a = agent.choose(&s, &legal, &mut rng);
        if let Action::Discard(t) = a {
            // Every route the 13-tile hand could still take gets a record; the
            // win later counts only under the route of the finished hand.
            let view = SeatView::of(&s, p);
            let mut after = view.concealed;
            after.remove(t);
            let melds = s.melds(p).len();
            for r in probe.score(&view.with(after, view.melds.clone()), wall) {
                seen.push((
                    p,
                    r.eval.route,
                    RouteKey::of(&r.eval, wall, melds, s.rules()).cell(),
                    wall,
                    r.eval.helpers,
                ));
            }
        }
        s.apply(a);
    }
    let end_wall = s.wall_remaining();
    let result = s.result().expect("hand over");
    let rules = *s.rules();
    let records = seen
        .into_iter()
        .map(|(p, route, cell, wall, helpers)| {
            let win = result
                .wins
                .iter()
                .find(|w| w.seat == p && Route::of_win(&w.score) == route);
            let sums = Sums {
                n: 1.0,
                wins: win.is_some() as u8 as f64,
                points: win.map_or(0.0, |w| rules.points(w.score.faan) as f64),
                rounds: if win.is_some() {
                    (wall - end_wall) as f64 / 4.0
                } else {
                    0.0
                },
                helpers: helpers as f64,
            };
            (cell, sums)
        })
        .collect();
    (records, result.wins.len() as u64)
}

fn format_chunk(c: &Chunk) -> String {
    let mut s = format!("hands {} wins {}\n", c.hands, c.wins);
    for (i, e) in c.cells.iter().enumerate().filter(|(_, e)| e.n > 0.0) {
        let _ = writeln!(
            s,
            "{i} {} {} {} {} {}",
            e.n, e.wins, e.points, e.rounds, e.helpers
        );
    }
    s
}

fn parse_chunk(text: &str) -> Chunk {
    let mut lines = text.lines();
    let head: Vec<u64> = lines
        .next()
        .expect("chunk header")
        .split(' ')
        .filter_map(|w| w.parse().ok())
        .collect();
    let mut cells = vec![Sums::default(); ROUTE_CELLS];
    for l in lines {
        let f: Vec<f64> = l.split(' ').map(|x| x.parse().expect("number")).collect();
        cells[f[0] as usize] = Sums {
            n: f[1],
            wins: f[2],
            points: f[3],
            rounds: f[4],
            helpers: f[5],
        };
    }
    Chunk {
        hands: head[0],
        wins: head[1],
        cells,
    }
}

/// Plays the hands chunk by chunk (skipping saved chunks); returns the summed
/// chunks and how many hands this run played.
fn play_all(dir: &Path, spec: &str, seeds: std::ops::Range<u64>) -> (Chunk, u64) {
    std::fs::create_dir_all(dir).expect("create the output folder");
    let agent = Agent::parse(spec).unwrap_or_else(|e| panic!("bot spec {spec:?}: {e}"));
    let probe = RouteBot::chain();
    let mut total = Chunk {
        hands: 0,
        wins: 0,
        cells: vec![Sums::default(); ROUTE_CELLS],
    };
    let mut played = 0;
    let start = Instant::now();
    let mut lo = seeds.start;
    while lo < seeds.end {
        let hi = (lo + CHUNK).min(seeds.end);
        let path = dir.join(format!("chunk-{lo}-{hi}.txt"));
        let chunk = match std::fs::read_to_string(&path) {
            Ok(text) => parse_chunk(&text),
            Err(_) => {
                let hands: Vec<(Vec<(usize, Sums)>, u64)> = (lo..hi)
                    .into_par_iter()
                    .map(|seed| play(seed, &agent, &probe))
                    .collect();
                let mut chunk = Chunk {
                    hands: hi - lo,
                    wins: 0,
                    cells: vec![Sums::default(); ROUTE_CELLS],
                };
                for (records, wins) in hands {
                    chunk.wins += wins;
                    for (cell, sums) in records {
                        chunk.cells[cell].add(&sums);
                    }
                }
                let tmp = path.with_extension("tmp");
                std::fs::write(&tmp, format_chunk(&chunk)).expect("write a chunk");
                std::fs::rename(&tmp, &path).expect("finish a chunk");
                played += hi - lo;
                chunk
            }
        };
        total.hands += chunk.hands;
        total.wins += chunk.wins;
        for (t, c) in total.cells.iter_mut().zip(&chunk.cells) {
            t.add(c);
        }
        println!(
            "hands {}/{}, {:.0} s",
            hi - seeds.start,
            seeds.end - seeds.start,
            start.elapsed().as_secs_f64()
        );
        lo = hi;
    }
    (total, played)
}

/// Shrinks `num / den` per entry toward its parent (the same key with any
/// melds), the parent toward the same key with any helper band, that toward
/// its route × needed, that toward its needed, and that toward the overall
/// rate, each with `SHRINK` pseudo-observations. An entry and parents with no
/// records take the first coarser rate that has some.
fn shrink(num: &[f64], den: &[f64]) -> Vec<f64> {
    let coarser = |i: usize, level: usize| {
        let k = RouteKey::from_cell(i);
        let k = match level {
            0 => RouteKey {
                route: Route::ALL[0],
                helpers: 0,
                wall: 0,
                melds: 0,
                ..k
            },
            1 => RouteKey {
                helpers: 0,
                wall: 0,
                melds: 0,
                ..k
            },
            2 => RouteKey {
                helpers: 0,
                melds: 0,
                ..k
            },
            _ => RouteKey { melds: 0, ..k },
        };
        k.cell()
    };
    const LEVELS: usize = 4; // coarsest first: needed; route × needed; + wall; + helpers
    let overall = num.iter().sum::<f64>() / den.iter().sum::<f64>().max(1.0);
    let mut sums: Vec<HashMap<usize, (f64, f64)>> = vec![HashMap::new(); LEVELS];
    for i in 0..num.len() {
        for (level, map) in sums.iter_mut().enumerate() {
            let e = map.entry(coarser(i, level)).or_default();
            e.0 += num[i];
            e.1 += den[i];
        }
    }
    (0..num.len())
        .map(|i| {
            let mut prior = overall;
            for (level, map) in sums.iter().enumerate() {
                let (x, n) = map[&coarser(i, level)];
                prior = (x + SHRINK * prior) / (n + SHRINK);
            }
            (num[i] + SHRINK * prior) / (den[i] + SHRINK)
        })
        .collect()
}

fn fit(dir: &Path, spec: &str, hands: u64, seed_base: u64) {
    let start = Instant::now();
    let (total, played) = play_all(dir, spec, seed_base..seed_base + hands);
    let secs = start.elapsed().as_secs_f64();
    let col = |f: fn(&Sums) -> f64| total.cells.iter().map(f).collect::<Vec<f64>>();
    let (n, wins) = (col(|s| s.n), col(|s| s.wins));
    let p_win = shrink(&wins, &n);
    let points = shrink(&col(|s| s.points), &wins);
    let rounds = shrink(&col(|s| s.rounds), &wins);
    let mut json = format!(
        "{{\"bot\": {spec:?}, \"hands\": {}, \"seed_base\": {seed_base}, \"entries\": [\n",
        total.hands
    );
    let filled: Vec<usize> = (0..ROUTE_CELLS).filter(|&i| n[i] > 0.0).collect();
    for i in 0..ROUTE_CELLS {
        let k = RouteKey::from_cell(i);
        let _ = writeln!(
            json,
            "{{\"cell\": {i}, \"route\": \"{}\", \"needed\": {}, \"helpers\": {}, \"wall\": {}, \
             \"melds\": {}, \"n\": {}, \"p_win\": {:.6}, \"points\": {:.4}, \"rounds\": {:.4}}}{}",
            k.route.name(),
            k.needed,
            k.helpers,
            k.wall,
            k.melds,
            n[i],
            p_win[i],
            points[i],
            rounds[i],
            if i + 1 < ROUTE_CELLS { "," } else { "" }
        );
    }
    json.push_str("]}\n");
    std::fs::write(dir.join("table.json"), json).expect("write table.json");
    // Mean weighted helpers per route × needed.
    let mut by_route = vec![(0.0, 0.0); ROUTES * NEEDED_BANDS];
    for (i, c) in total.cells.iter().enumerate() {
        let k = RouteKey::from_cell(i);
        let e = &mut by_route[k.route as usize * NEEDED_BANDS + k.needed];
        e.0 += c.helpers;
        e.1 += c.n;
    }
    let lines: Vec<String> = by_route
        .iter()
        .enumerate()
        .filter(|(_, e)| e.1 > 0.0)
        .map(|(j, &(h, count))| {
            let (r, needed) = (j / NEEDED_BANDS, j % NEEDED_BANDS);
            format!(
                "{{\"route_index\": {r}, \"route\": \"{}\", \"needed\": {needed}, \"n\": {count}, \"helpers\": {:.4}}}",
                Route::ALL[r].name(),
                h / count
            )
        })
        .collect();
    std::fs::write(
        dir.join("helpers.json"),
        format!(
            "{{\"bot\": {spec:?}, \"means\": [\n{}\n]}}\n",
            lines.join(",\n")
        ),
    )
    .expect("write helpers.json");
    let records: f64 = n.iter().sum();
    let won: f64 = wins.iter().sum();
    let speed = if played > 0 {
        format!("{:.1}", played as f64 / secs)
    } else {
        "n/a (every chunk was already saved)".to_string()
    };
    let summary = format!(
        "bot {spec}\nhands {}\nrecords {records}\nrecords by a seat that went on to win {won}\n\
         wins declared {}\nhands per second {speed} ({played} hands played in {secs:.1} s, \
         {} threads)\nentries with records {}\nentries with n >= 500: {}\nentries with n >= 2000: {}\n\
         table cells {ROUTE_CELLS} (helper bands {HELPER_BANDS}, wall bands {TILES_LEFT_BUCKETS})\n",
        total.hands,
        total.wins,
        rayon::current_num_threads(),
        filled.len(),
        n.iter().filter(|&&x| x >= 500.0).count(),
        n.iter().filter(|&&x| x >= SOLID).count(),
    );
    std::fs::write(dir.join("summary.txt"), &summary).expect("write summary.txt");
    print!("{summary}");
}

fn diff(a: &str, b: &str) {
    let (ta, tb) = (
        RouteTable::load(a).unwrap_or_else(|e| panic!("{e}")),
        RouteTable::load(b).unwrap_or_else(|e| panic!("{e}")),
    );
    let mut changes: Vec<(f64, usize)> = ta
        .entries
        .iter()
        .filter(|(_, e)| e.n >= SOLID)
        .filter_map(|(cell, e)| {
            let f = tb.entries.get(cell).filter(|f| f.n >= SOLID)?;
            Some(((e.p_win - f.p_win).abs(), *cell))
        })
        .collect();
    changes.sort_by(|x, y| y.0.total_cmp(&x.0));
    println!("entries with n >= {SOLID} in both: {}", changes.len());
    if let Some(&(max, cell)) = changes.first() {
        let mean = changes.iter().map(|c| c.0).sum::<f64>() / changes.len() as f64;
        println!(
            "largest |change of p_win|: {max:.4} at {:?}",
            RouteKey::from_cell(cell)
        );
        println!("mean |change of p_win|: {mean:.4}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [mode, a, b] if mode == "diff" => diff(a, b),
        [hands, spec, out, rest @ ..] if rest.len() <= 1 => {
            let hands = hands.parse().expect("hands");
            let seed_base = rest.first().map_or(0, |s| s.parse().expect("seed base"));
            fit(Path::new(out), spec, hands, seed_base);
        }
        _ => {
            eprintln!(
                "usage: fit_routes <hands> <bot spec> <out dir> [seed base]\n       \
                 fit_routes diff <table a> <table b>"
            );
            std::process::exit(2);
        }
    }
}
