//! Fits the target value table ([`mahjong_search::target_tables`]) from games
//! between the rule-based bots and writes `crates/search/src/target_tables.rs`
//! (plan Task 9, hard-problems.md idea 1e).
//!
//! After every discard, the discarder's 13-tile hand is keyed by its best
//! target kind, faan-aware shanten, tiles-left bucket and melds
//! ([`TargetKey`]); the hand's end says whether that player won, how, and for
//! how many points per share. Cells with few observations are shrunk toward a
//! coarser cell (drop the melds, then the target kind), as `fit_danger` does,
//! and P(win) is made non-increasing in faan-shanten. A calibration report on
//! held-out hands (other seeds) follows.
//!
//! `cargo run --release -p mahjong-search --example fit_targets [hands] [holdout]`
//!
//! Progress is saved per chunk of 500 hands under `runs/targets/<hands>-<holdout>/`
//! (relative to the working directory); a rerun skips finished chunks. Set
//! `RAYON_NUM_THREADS` to limit the threads.
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::target_value::{
    TargetKey, TargetKind, FAAN_SHANTEN_BUCKETS, MELD_BUCKETS, TARGET_CELLS, TARGET_KINDS,
    TILES_LEFT_BUCKETS,
};
use mahjong_search::{Bot, SeatView};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Pseudo-observations pulling a cell toward its parent.
const SHRINK: f64 = 50.0;
/// Hands per saved chunk.
const CHUNK: usize = 500;
/// Seeds of held-out hands start here (fitting hands use 0..hands).
const HOLDOUT_SEED: u64 = 50_000_000;

/// The field: a value-heavy mix, rotated so each hand seats four of the five.
const BOTS: [&str; 5] = ["value", "defensive", "value", "efficiency", "fast"];

/// One discard's hand and how the hand ended for the discarder.
#[derive(Clone, Copy, Debug)]
struct Obs {
    cell: u16,
    hand: u32,
    won: bool,
    self_drawn: bool,
    points: f32,
}

fn play(seed: u64) -> Vec<Obs> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let bots: Vec<Bot> = (0..4)
        .map(|s| Bot::by_name(BOTS[(seed as usize + s) % BOTS.len()]).unwrap())
        .collect();
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    let mut seen: Vec<(usize, TargetKey)> = Vec::new();
    while let Some(p) = s.current_player() {
        let legal = s.legal_actions_vec();
        let a = bots[p].choose(&s, &legal, &mut rng);
        if let Action::Discard(t) = a {
            let view = SeatView::of(&s, p);
            let mut after = view.concealed;
            after.remove(t);
            let melds = view.melds.clone();
            let key = TargetKey::of_view(&view.with(after, melds), s.wall_remaining());
            seen.push((p, key));
        }
        s.apply(a);
    }
    let result = s.result().expect("hand over");
    let rules = *s.rules();
    seen.into_iter()
        .map(|(p, key)| {
            let win = result.wins.iter().find(|w| w.seat == p);
            Obs {
                cell: key.cell() as u16,
                hand: seed as u32,
                won: win.is_some(),
                self_drawn: win.is_some_and(|w| w.self_drawn),
                points: win.map_or(0.0, |w| rules.points(w.score.faan) as f32),
            }
        })
        .collect()
}

/// Plays hands `seeds` in chunks, saving each chunk's observations under `dir`.
fn play_all(dir: &Path, label: &str, seeds: std::ops::Range<u64>) -> Vec<Obs> {
    std::fs::create_dir_all(dir).expect("create the progress folder");
    let start = Instant::now();
    let total = seeds.end - seeds.start;
    let mut out = Vec::new();
    let mut lo = seeds.start;
    while lo < seeds.end {
        let hi = (lo + CHUNK as u64).min(seeds.end);
        let path = dir.join(format!("{label}-{lo}-{hi}.txt"));
        let obs = match std::fs::read_to_string(&path) {
            Ok(text) => parse_chunk(&text),
            Err(_) => {
                let obs: Vec<Obs> = (lo..hi).into_par_iter().flat_map(play).collect();
                let tmp = path.with_extension("tmp");
                std::fs::write(&tmp, format_chunk(&obs)).expect("write a chunk");
                std::fs::rename(&tmp, &path).expect("finish a chunk");
                obs
            }
        };
        out.extend(obs);
        println!(
            "{label}: hands {}/{total}, observations {}, {:.0} s",
            hi - seeds.start,
            out.len(),
            start.elapsed().as_secs_f64()
        );
        lo = hi;
    }
    out
}

fn format_chunk(obs: &[Obs]) -> String {
    let mut s = String::new();
    for o in obs {
        let _ = writeln!(
            s,
            "{} {} {} {} {}",
            o.cell, o.hand, o.won as u8, o.self_drawn as u8, o.points
        );
    }
    s
}

fn parse_chunk(text: &str) -> Vec<Obs> {
    text.lines()
        .map(|l| {
            let f: Vec<&str> = l.split(' ').collect();
            Obs {
                cell: f[0].parse().expect("cell"),
                hand: f[1].parse().expect("hand"),
                won: f[2] == "1",
                self_drawn: f[3] == "1",
                points: f[4].parse().expect("points"),
            }
        })
        .collect()
}

/// Shrinks `num / den` per cell toward its parent (the same key with any melds),
/// the parent toward its grandparent (any kind, any melds) and that toward the
/// overall rate, each with `SHRINK` pseudo-observations.
fn shrink(num: &[f64], den: &[f64]) -> Vec<f64> {
    let parent = |i: usize| i / MELD_BUCKETS;
    let grand = |i: usize| {
        let k = TargetKey::from_cell(i);
        k.faan_shanten * TILES_LEFT_BUCKETS + k.tiles_left
    };
    let overall = num.iter().sum::<f64>() / den.iter().sum::<f64>().max(1.0);
    let mut parents = HashMap::<usize, (f64, f64)>::new();
    let mut grands = HashMap::<usize, (f64, f64)>::new();
    for i in 0..num.len() {
        for (map, g) in [(&mut parents, parent(i)), (&mut grands, grand(i))] {
            let e = map.entry(g).or_default();
            e.0 += num[i];
            e.1 += den[i];
        }
    }
    (0..num.len())
        .map(|i| {
            let g = grands[&grand(i)];
            let g = (g.0 + SHRINK * overall) / (g.1 + SHRINK);
            let p = parents[&parent(i)];
            let p = (p.0 + SHRINK * g) / (p.1 + SHRINK);
            (num[i] + SHRINK * p) / (den[i] + SHRINK)
        })
        .collect()
}

/// Weighted least-squares fit of a non-increasing sequence (pool adjacent violators).
fn non_increasing(v: &mut [f64], w: &[f64]) {
    // Blocks of (mean, weight, length), merged while a later block is higher.
    let mut blocks: Vec<(f64, f64, usize)> = Vec::new();
    for (&x, &wx) in v.iter().zip(w) {
        blocks.push((x, wx, 1));
        while blocks.len() > 1 && blocks[blocks.len() - 2].0 < blocks[blocks.len() - 1].0 {
            let b = blocks.pop().unwrap();
            let a = blocks.last_mut().unwrap();
            a.0 = (a.0 * a.1 + b.0 * b.1) / (a.1 + b.1);
            a.1 += b.1;
            a.2 += b.2;
        }
    }
    let mut i = 0;
    for (m, _, len) in blocks {
        for x in &mut v[i..i + len] {
            *x = m;
        }
        i += len;
    }
}

struct Table {
    p_win: Vec<f64>,
    p_self_draw: Vec<f64>,
    points: Vec<f64>,
}

fn fit(obs: &[Obs]) -> (Table, Vec<f64>, f64) {
    let mut n = vec![0.0; TARGET_CELLS];
    let mut wins = vec![0.0; TARGET_CELLS];
    let mut selfs = vec![0.0; TARGET_CELLS];
    let mut points = vec![0.0; TARGET_CELLS];
    for o in obs {
        let c = o.cell as usize;
        n[c] += 1.0;
        if o.won {
            wins[c] += 1.0;
            selfs[c] += o.self_drawn as u8 as f64;
            points[c] += o.points as f64;
        }
    }
    let mut p_win = shrink(&wins, &n);
    // Non-increasing in faan-shanten for every (kind, tiles left, melds).
    for kind in 0..TARGET_KINDS {
        for tl in 0..TILES_LEFT_BUCKETS {
            for m in 0..MELD_BUCKETS {
                let cells: Vec<usize> = (0..FAAN_SHANTEN_BUCKETS)
                    .map(|fs| {
                        TargetKey {
                            kind: TargetKind::ALL[kind],
                            faan_shanten: fs,
                            tiles_left: tl,
                            melds: m,
                        }
                        .cell()
                    })
                    .collect();
                let mut v: Vec<f64> = cells.iter().map(|&c| p_win[c]).collect();
                let w: Vec<f64> = cells.iter().map(|&c| n[c] + SHRINK).collect();
                non_increasing(&mut v, &w);
                for (&c, x) in cells.iter().zip(v) {
                    p_win[c] = x;
                }
            }
        }
    }
    let table = Table {
        p_win,
        p_self_draw: shrink(&selfs, &wins),
        points: shrink(&points, &wins),
    };
    let total_wins = wins.iter().sum();
    (table, n, total_wins)
}

/// One calibration table: per group, the mean prediction against the observed
/// mean with a 95 % interval that treats each hand as one cluster (a hand's
/// observations are not independent). Returns (buckets within, buckets shown).
fn report(title: &str, labels: &[String], rows: &[(usize, f64, f64, u32)]) -> (usize, usize) {
    println!("{title}");
    let (mut ok, mut shown) = (0, 0);
    for (g, label) in labels.iter().enumerate() {
        let mut n = 0.0;
        let (mut pred, mut obs) = (0.0, 0.0);
        let mut clusters = HashMap::<u32, f64>::new();
        for &(_, p, y, hand) in rows.iter().filter(|r| r.0 == g) {
            n += 1.0;
            pred += p;
            obs += y;
            *clusters.entry(hand).or_default() += y - p;
        }
        if n < 30.0 {
            continue;
        }
        let se = clusters.values().map(|r| r * r).sum::<f64>().sqrt() / n;
        let within = ((obs - pred) / n).abs() <= 1.96 * se;
        shown += 1;
        ok += within as usize;
        println!(
            "  {label:<14} n={n:>8}  predicted {:>8.4}  observed {:>8.4} ± {:.4}  {}",
            pred / n,
            obs / n,
            1.96 * se,
            if within { "ok" } else { "OUTSIDE" }
        );
    }
    (ok, shown)
}

fn calibrate(table: &Table, obs: &[Obs]) {
    const EDGES: [f64; 11] = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8];
    let bin = |p: f64| EDGES.iter().rposition(|&e| p >= e).unwrap_or(0);
    let labels: Vec<String> = EDGES
        .iter()
        .enumerate()
        .map(|(i, lo)| format!("{lo:.2}–{:.2}", EDGES.get(i + 1).copied().unwrap_or(1.0)))
        .collect();
    let win_rows = |group: &dyn Fn(&Obs, f64) -> usize| -> Vec<(usize, f64, f64, u32)> {
        obs.iter()
            .map(|o| {
                let p = table.p_win[o.cell as usize];
                (group(o, p), p, o.won as u8 as f64, o.hand)
            })
            .collect()
    };
    let key = |o: &Obs| TargetKey::from_cell(o.cell as usize);
    let mut tally = (0, 0);
    let mut add = |r: (usize, usize)| {
        tally.0 += r.0;
        tally.1 += r.1;
    };
    add(report(
        "P(win) by predicted probability:",
        &labels,
        &win_rows(&|_, p| bin(p)),
    ));
    let fs_labels: Vec<String> = (0..FAAN_SHANTEN_BUCKETS)
        .map(|f| format!("faan-shanten {f}"))
        .collect();
    add(report(
        "P(win) by faan-shanten:",
        &fs_labels,
        &win_rows(&|o, _| key(o).faan_shanten),
    ));
    let tl_labels: Vec<String> = (0..TILES_LEFT_BUCKETS)
        .map(|t| format!("tiles left {t}"))
        .collect();
    add(report(
        "P(win) by tiles-left bucket:",
        &tl_labels,
        &win_rows(&|o, _| key(o).tiles_left),
    ));
    let kind_labels: Vec<String> = TargetKind::ALL.iter().map(|k| k.name().into()).collect();
    add(report(
        "P(win) by target kind:",
        &kind_labels,
        &win_rows(&|o, _| key(o).kind as usize),
    ));
    let won: Vec<&Obs> = obs.iter().filter(|o| o.won).collect();
    let self_rows: Vec<_> = won
        .iter()
        .map(|o| {
            let c = o.cell as usize;
            (
                key(o).kind as usize,
                table.p_self_draw[c],
                o.self_drawn as u8 as f64,
                o.hand,
            )
        })
        .collect();
    add(report(
        "P(self-draw | win) by target kind:",
        &kind_labels,
        &self_rows,
    ));
    let point_rows: Vec<_> = won
        .iter()
        .map(|o| {
            let c = o.cell as usize;
            (
                key(o).kind as usize,
                table.points[c],
                o.points as f64,
                o.hand,
            )
        })
        .collect();
    add(report(
        "Points per share | win by target kind:",
        &kind_labels,
        &point_rows,
    ));
    println!(
        "calibration: {} of {} buckets within their 95 % interval",
        tally.0, tally.1
    );
}

fn rust_table(name: &str, doc: &str, values: &[f64]) -> String {
    let body: Vec<String> = values
        .chunks(MELD_BUCKETS)
        .map(|ch| {
            let v: Vec<String> = ch.iter().map(|x| format!("{x:.5}")).collect();
            format!("    {},", v.join(", "))
        })
        .collect();
    format!(
        "/// {doc}\n#[rustfmt::skip]\npub const {name}: [f64; TARGET_CELLS] = [\n{}\n];\n",
        body.join("\n")
    )
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let hands = args.first().copied().unwrap_or(20_000);
    let holdout = args.get(1).copied().unwrap_or(4_000);
    let dir = PathBuf::from(format!("runs/targets/{hands}-{holdout}"));
    let fit_obs = play_all(&dir, "fit", 0..hands as u64);
    let test_obs = play_all(&dir, "holdout", HOLDOUT_SEED..HOLDOUT_SEED + holdout as u64);
    let (table, n, wins) = fit(&fit_obs);
    let sparse = n.iter().filter(|&&x| x < SHRINK).count();
    println!(
        "fitted on {} observations ({wins} from winning hands) in {hands} hands; \
         {sparse} of {TARGET_CELLS} cells have fewer than {SHRINK} observations",
        fit_obs.len()
    );
    println!("P(win), kind × faan-shanten, at tiles-left 1 and 0 melds:");
    for kind in TargetKind::ALL {
        let row: Vec<String> = (0..FAAN_SHANTEN_BUCKETS)
            .map(|fs| {
                let c = TargetKey {
                    kind,
                    faan_shanten: fs,
                    tiles_left: 1,
                    melds: 0,
                }
                .cell();
                format!("{:.3}", table.p_win[c])
            })
            .collect();
        println!("  {:<10} {}", kind.name(), row.join("  "));
    }
    println!("held-out calibration on {holdout} hands:");
    calibrate(&table, &test_obs);

    let src = format!(
        "//! Fitted tables for [`crate::target_value`]. Generated by\n\
         //! `cargo run --release -p mahjong-search --example fit_targets {hands} {holdout}`\n\
         //! ({} discards, {wins} of them by players who went on to win, from games\n\
         //! between the rule-based bots); do not edit. Index: [`crate::target_value::TargetKey::cell`].\n\n\
         use crate::target_value::TARGET_CELLS;\n\n{}\n{}\n{}",
        fit_obs.len(),
        rust_table(
            "P_WIN",
            "P(a legal win before the hand ends).",
            &table.p_win
        ),
        rust_table(
            "P_SELF_DRAW",
            "P(self-drawn | win).",
            &table.p_self_draw
        ),
        rust_table(
            "POINTS",
            "E[points per share | win].",
            &table.points
        ),
    );
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/target_tables.rs");
    std::fs::write(path, src).expect("write tables");
    println!("wrote {path}");
}
