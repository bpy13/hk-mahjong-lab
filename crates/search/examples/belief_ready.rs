//! Does the belief network read a ready hand from its discards? (plan
//! 2026-11-16 Task 18, `docs/study-results/belief-ready-read.md`)
//!
//! ```text
//! cargo run --release -p mahjong-search --example belief_ready -- \
//!     --hand C:\Users\brian\.orchestrator\notes\mahjong-flush-game-2860.json [--from 56]
//! cargo run --release -p mahjong-search --example belief_ready -- \
//!     --deals 20000 --seed 18 [--field stage] [--belief models/belief.mjnn]
//! ```
//!
//! `--hand`: replays a saved hand (`seed`, `actions` as `[seat, code]`) and, at
//! every turn decision from step `--from` on, prints for each opponent of the
//! player to act the true ready state, the network's P(ready) (`tenpai` head),
//! the race bot's risk-adjusted reading of it (`race::RISK_MODEL`) and the
//! danger table's threat.
//!
//! `--deals`: plays deals of four `--field` bots (`stage` by default; hand seeds from `--seed`, none
//! used to fit the network, which was fitted on a `value`/`defensive` field) and,
//! at every discard decision of every seat, logs each opponent's true ready
//! state (`State::wins_on` on any kind, the network's own training target), the
//! network's P(ready) and the table's. Prints markdown: ten bins by each
//! estimate, and the slices of the task (last third of the wall; a seat whose
//! discards were nearly all outside one suit and has since thrown that suit; a
//! seat that just threw a tile of the suit another seat shows three or four
//! open sets of; both; runs of discards that were the drawn tile thrown back),
//! each with count, true rate ± 95 %, mean network and mean table estimate.
//! Threads: `RAYON_NUM_THREADS`.

use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, Event, RuleSet, State};
use mahjong_search::passed::risk_scores;
use mahjong_search::race::RISK_MODEL;
use mahjong_search::{BeliefModel, Bot, Danger};
use mahjong_tiles::{Suit, Tile};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Tiles in the wall at the deal (flowers on).
const INITIAL_WALL: usize = 91;
/// Discards a seat must have made before its suit pattern is read.
const MIN_DISCARDS: usize = 8;
/// The first discards that must all lie outside the suspected suit.
const OUTSIDE_FIRST: usize = 6;

/// One discard of one seat.
#[derive(Clone, Copy)]
struct Thrown {
    tile: Tile,
    from_hand: bool,
}

fn discards_of(state: &State) -> [Vec<Thrown>; 4] {
    let mut out: [Vec<Thrown>; 4] = Default::default();
    for e in state.events() {
        if let Event::Discard {
            seat,
            tile,
            from_hand,
        } = *e
        {
            out[seat as usize].push(Thrown { tile, from_hand });
        }
    }
    out
}

/// Whether seat `o` is truly ready: some kind would win for it now (the
/// network's training target, `log::truth`).
fn truly_ready(state: &State, o: usize) -> bool {
    Tile::all_kinds().any(|k| state.wins_on(o, k))
}

/// The suit seat `o` seems to collect: its first [`OUTSIDE_FIRST`] discards hold
/// no tile of it, a later discard does, and every suited exposed meld of `o` is
/// in it. Returns the suit and whether `o`'s latest discard is in it.
fn flush_suspect(state: &State, o: usize, thrown: &[Thrown]) -> Option<(Suit, bool)> {
    if thrown.len() < MIN_DISCARDS {
        return None;
    }
    let last = thrown.last()?.tile.suit();
    for x in Suit::ALL {
        let early_clean = thrown[..OUTSIDE_FIRST]
            .iter()
            .all(|d| d.tile.suit() != Some(x));
        let since = thrown[OUTSIDE_FIRST..]
            .iter()
            .any(|d| d.tile.suit() == Some(x));
        let melds_fit = state
            .melds(o)
            .iter()
            .filter(|m| !m.concealed)
            .all(|m| m.tile.suit().is_none_or(|s| s == x));
        if early_clean && since && melds_fit {
            return Some((x, last == Some(x)));
        }
    }
    None
}

/// The suit that another seat than `o` shows three or four open sets of (all of
/// its exposed melds in that one suit), if any.
fn flush_shown_against(state: &State, o: usize) -> Option<Suit> {
    (0..4).filter(|&q| q != o).find_map(|q| {
        let melds = state.melds(q);
        let exposed: Vec<_> = melds.iter().filter(|m| !m.concealed).collect();
        if exposed.len() < 3 {
            return None;
        }
        let s = exposed[0].tile.suit()?;
        exposed
            .iter()
            .all(|m| m.tile.suit() == Some(s))
            .then_some(s)
    })
}

/// Whether `o`'s latest discard (and the one before, for `two`) is a tile of a
/// suit another seat shows three or four open sets of. Returns (feeds, feeds
/// from hand 入章, feeds twice in a row).
fn feeds_flush(state: &State, o: usize, thrown: &[Thrown]) -> (bool, bool, bool) {
    let Some(y) = flush_shown_against(state, o) else {
        return (false, false, false);
    };
    let Some(last) = thrown.last() else {
        return (false, false, false);
    };
    let feeds = last.tile.suit() == Some(y);
    let two = feeds && thrown.len() >= 2 && thrown[thrown.len() - 2].tile.suit() == Some(y);
    (feeds, feeds && last.from_hand, two)
}

/// One opponent at one discard decision.
#[derive(Clone, Copy)]
struct Row {
    ready: bool,
    net: f64,
    table: f64,
    race: f64,
    wall: usize,
    exposed: usize,
    suspect: bool,
    suspect_last: bool,
    feeds: bool,
    feeds_hand: bool,
    feeds_two: bool,
    /// Latest discards in a row that were the drawn tile thrown back (摸打).
    streak: usize,
}

/// Sum statistics of a set of rows.
#[derive(Clone, Copy, Default)]
struct Cell {
    n: u64,
    ready: u64,
    net: f64,
    table: f64,
    race: f64,
}

impl Cell {
    fn add(&mut self, r: &Row) {
        self.n += 1;
        self.ready += r.ready as u64;
        self.net += r.net;
        self.table += r.table;
        self.race += r.race;
    }
    fn merge(&mut self, o: &Cell) {
        self.n += o.n;
        self.ready += o.ready;
        self.net += o.net;
        self.table += o.table;
        self.race += o.race;
    }
    fn line(&self, name: &str) -> String {
        if self.n == 0 {
            return format!("| {name} | 0 | – | – | – | – |");
        }
        let n = self.n as f64;
        let p = self.ready as f64 / n;
        let pm = 1.96 * (p * (1.0 - p) / n).sqrt();
        format!(
            "| {name} | {} | {:.2} ± {:.2} % | {:.2} % | {:.2} % | {:.2} % |",
            self.n,
            100.0 * p,
            100.0 * pm,
            100.0 * self.net / n,
            100.0 * self.table / n,
            100.0 * self.race / n
        )
    }
}

const SLICES: &[&str] = &[
    "all",
    "last third of the wall (≤ 30 tiles left)",
    "suit-suspect: first 6 discards outside one suit, that suit thrown since",
    "suit-suspect, latest discard in that suit",
    "feeds a shown flush: latest discard in the suit another seat shows 3–4 open sets of",
    "feeds a shown flush, thrown from hand (入章)",
    "feeds a shown flush, drawn and thrown (摸打)",
    "feeds a shown flush twice in a row",
    "suit-suspect and feeds a shown flush",
    "suit-suspect (latest in suit) and feeds a shown flush",
    "last third and suit-suspect",
    "last third and feeds a shown flush",
    "last third, suit-suspect and feeds a shown flush",
    "opponent shows 0 exposed sets",
    "opponent shows 1 exposed set",
    "opponent shows 2 exposed sets",
    "opponent shows 3–4 exposed sets",
    "1 exposed set, suit-suspect",
    "1 exposed set, feeds a shown flush",
    "1 exposed set, suit-suspect and feeds a shown flush",
    "1 exposed set, last third, suit-suspect and feeds a shown flush",
    "last 3 discards all drawn and thrown (摸打)",
    "last 5 discards all drawn and thrown",
    "last 8 discards all drawn and thrown",
    "1 exposed set, last 5 drawn and thrown",
    "suit-suspect, last 5 drawn and thrown",
    "1 exposed set, suit-suspect, last 5 drawn and thrown",
    "1 exposed set, suit-suspect, feeds a shown flush, last 5 drawn and thrown",
];

fn slice_flags(r: &Row) -> [bool; 28] {
    let late = r.wall <= INITIAL_WALL / 3;
    let one = r.exposed == 1;
    let s5 = r.streak >= 5;
    [
        true,
        late,
        r.suspect,
        r.suspect_last,
        r.feeds,
        r.feeds_hand,
        r.feeds && !r.feeds_hand,
        r.feeds_two,
        r.suspect && r.feeds,
        r.suspect_last && r.feeds,
        late && r.suspect,
        late && r.feeds,
        late && r.suspect && r.feeds,
        r.exposed == 0,
        one,
        r.exposed == 2,
        r.exposed >= 3,
        one && r.suspect,
        one && r.feeds,
        one && r.suspect && r.feeds,
        one && late && r.suspect && r.feeds,
        r.streak >= 3,
        s5,
        r.streak >= 8,
        one && s5,
        r.suspect && s5,
        one && r.suspect && s5,
        one && r.suspect && r.feeds && s5,
    ]
}

#[derive(Clone)]
struct Stats {
    slices: Vec<Cell>,
    by_net: [Cell; 10],
    by_table: [Cell; 10],
    /// Ready rate against the table by the network's own reading: rows where the
    /// network is below the table and rows where it is above.
    net_vs_table: [Cell; 2],
    decisions: u64,
}

impl Stats {
    fn new() -> Stats {
        Stats {
            slices: vec![Cell::default(); SLICES.len()],
            by_net: [Cell::default(); 10],
            by_table: [Cell::default(); 10],
            net_vs_table: [Cell::default(); 2],
            decisions: 0,
        }
    }
    fn add(&mut self, r: &Row) {
        for (c, on) in self.slices.iter_mut().zip(slice_flags(r)) {
            if on {
                c.add(r);
            }
        }
        self.by_net[bin(r.net)].add(r);
        self.by_table[bin(r.table)].add(r);
        self.net_vs_table[(r.net > r.table) as usize].add(r);
    }
    fn merge(&mut self, o: &Stats) {
        for (a, b) in self.slices.iter_mut().zip(&o.slices) {
            a.merge(b);
        }
        for (a, b) in self.by_net.iter_mut().zip(&o.by_net) {
            a.merge(b);
        }
        for (a, b) in self.by_table.iter_mut().zip(&o.by_table) {
            a.merge(b);
        }
        for (a, b) in self.net_vs_table.iter_mut().zip(&o.net_vs_table) {
            a.merge(b);
        }
        self.decisions += o.decisions;
    }
}

fn bin(p: f64) -> usize {
    ((p * 10.0) as usize).min(9)
}

/// The three readings of opponent `o` for the player to act `me`.
fn readings(state: &State, me: usize, o: usize, model: &BeliefModel) -> (f64, f64, f64) {
    let out = model.run(state);
    let net = out.seat(o).map_or(f64::NAN, |b| b.tenpai);
    let table = Danger::estimate(state, me).threat(o);
    let risk = risk_scores(state);
    let race = (net * RISK_MODEL.tenpai[RISK_MODEL.bin(risk[o])]).min(1.0);
    (net, table, race)
}

/// Logs every opponent at this discard decision.
fn log_decision(state: &State, me: usize, model: &BeliefModel, stats: &mut Stats) {
    let thrown = discards_of(state);
    stats.decisions += 1;
    for o in (0..4).filter(|&o| o != me) {
        let (net, table, race) = readings(state, me, o, model);
        let suspect = flush_suspect(state, o, &thrown[o]);
        let (feeds, feeds_hand, feeds_two) = feeds_flush(state, o, &thrown[o]);
        stats.add(&Row {
            ready: truly_ready(state, o),
            net,
            table,
            race,
            wall: state.wall_remaining(),
            exposed: state.melds(o).exposed_count(),
            suspect: suspect.is_some(),
            suspect_last: suspect.is_some_and(|(_, last)| last),
            feeds,
            feeds_hand,
            feeds_two,
            streak: thrown[o].iter().rev().take_while(|d| !d.from_hand).count(),
        });
    }
}

fn play_deal(deal_seed: u64, bot: &Bot, model: &BeliefModel, stats: &mut Stats) {
    let mut state = State::new_deal(
        RuleSet::default(),
        &mut ChaCha8Rng::seed_from_u64(deal_seed),
    );
    let mut rng = ChaCha8Rng::seed_from_u64(deal_seed ^ 0x9e37_79b9);
    while let Some(me) = state.current_player() {
        let legal = state.legal_actions_vec();
        let action = bot.choose(&state, &legal, &mut rng);
        if matches!(action, Action::Discard(_)) {
            log_decision(&state, me, model, stats);
        }
        state.apply(action);
    }
}

fn header() -> &'static str {
    "| Slice | Opponent-decisions | True ready | Network mean | Table mean | Race-adjusted mean |\n|---|---|---|---|---|---|"
}

fn run_deals(deals: u32, seed: u64, field: &str, belief: &Path) {
    let chunk = 500u32;
    let chunks: Vec<u32> = (0..deals).step_by(chunk as usize).collect();
    let start = std::time::Instant::now();
    let total = chunks
        .into_par_iter()
        .map(|first| {
            let model = BeliefModel::load(belief).expect("belief model");
            let bot = Bot::by_name(field).unwrap_or_else(|| panic!("no bot {field:?}"));
            let mut stats = Stats::new();
            for d in first..(first + chunk).min(deals) {
                let deal_seed = seed.wrapping_mul(1_000_003).wrapping_add(d as u64);
                play_deal(deal_seed, &bot, &model, &mut stats);
            }
            stats
        })
        .reduce(Stats::new, |mut a, b| {
            a.merge(&b);
            a
        });
    println!(
        "{deals} deals of four `{field}` bots, seed {seed}, model {}: {} discard decisions, {} opponent rows, {:.0} s.\n",
        belief.display(),
        total.decisions,
        total.slices[0].n,
        start.elapsed().as_secs_f64()
    );
    println!("## Ten bins by the network's P(ready)\n\n{}", header());
    for (i, c) in total.by_net.iter().enumerate() {
        println!(
            "{}",
            c.line(&format!("network {}–{} %", i * 10, i * 10 + 10))
        );
    }
    println!("\n## Ten bins by the table's P(ready)\n\n{}", header());
    for (i, c) in total.by_table.iter().enumerate() {
        println!("{}", c.line(&format!("table {}–{} %", i * 10, i * 10 + 10)));
    }
    println!("\n## Where the two disagree\n\n{}", header());
    println!("{}", total.net_vs_table[0].line("network ≤ table"));
    println!("{}", total.net_vs_table[1].line("network > table"));
    println!("\n## Slices\n\n{}", header());
    for (name, c) in SLICES.iter().zip(&total.slices) {
        println!("{}", c.line(name));
    }
}

/// Reads `"seed": N` and `"actions": [[seat, code], ...]` from the saved hand.
fn read_hand(path: &Path) -> (u64, Vec<(usize, u16)>) {
    let text = std::fs::read_to_string(path).expect("hand file");
    let after = |key: &str| -> &str {
        let i = text.find(key).unwrap_or_else(|| panic!("no {key}"));
        &text[i + key.len()..]
    };
    let seed: u64 = after("\"seed\":")
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .expect("seed");
    let acts = after("\"actions\":");
    let end = acts.find("]]").expect("actions end");
    let actions = acts[..end]
        .split("],")
        .map(|pair| {
            let nums: Vec<u64> = pair
                .chars()
                .map(|c| if c.is_ascii_digit() { c } else { ' ' })
                .collect::<String>()
                .split_whitespace()
                .map(|n| n.parse().unwrap())
                .collect();
            (nums[0] as usize, nums[1] as u16)
        })
        .collect();
    (seed, actions)
}

fn hand_text(state: &State, seat: usize) -> String {
    let melds: Vec<String> = state
        .melds(seat)
        .iter()
        .map(|m| {
            m.tiles()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("")
        })
        .collect();
    format!("{} | sets {}", state.hand(seat), melds.join(" "))
}

fn replay(path: &Path, from: usize, belief: &Path) {
    let model = BeliefModel::load(belief).expect("belief model");
    let (seed, actions) = read_hand(path);
    // The same deal as `mahjong._core.HkState(seed)` (pybind `rng_from`: SmallRng).
    let mut state = State::new_deal(RuleSet::default(), &mut SmallRng::seed_from_u64(seed));
    println!(
        "Hand {} (seed {seed}, {} actions), model {}\n",
        path.display(),
        actions.len(),
        belief.display()
    );
    let seat_name = ["East", "South", "West", "North"];
    for (i, &(seat, code)) in actions.iter().enumerate() {
        let me = state.current_player().expect("hand in progress");
        assert_eq!(me, seat, "step {i}: seat {seat} in the file, {me} to act");
        let action = Action::from_code(code).expect("action code");
        if i >= from && state.phase_name() == "turn" {
            let thrown = discards_of(&state);
            println!(
                "step {i}, wall {}, {} to act ({action}); hand {}",
                state.wall_remaining(),
                seat_name[me],
                hand_text(&state, me)
            );
            for o in (0..4).filter(|&o| o != me) {
                let (net, table, race) = readings(&state, me, o, &model);
                let last = thrown[o].last().map_or("-".to_string(), |d| {
                    format!("{}{}", d.tile, if d.from_hand { " 入章" } else { " 摸打" })
                });
                let suspect = flush_suspect(&state, o, &thrown[o]);
                let (feeds, _, two) = feeds_flush(&state, o, &thrown[o]);
                println!(
                    "    {:<5} ready {:<5} network {:5.1} %  race-adjusted {:5.1} %  table {:5.1} %  last discard {last}  suspect {:?} feeds {feeds} twice {two}  hand {}",
                    seat_name[o],
                    truly_ready(&state, o),
                    100.0 * net,
                    100.0 * race,
                    100.0 * table,
                    suspect,
                    hand_text(&state, o)
                );
            }
        }
        state.apply(action);
    }
    if let Some(r) = state.result() {
        println!("\nresult: {r:?}");
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut hand: Option<PathBuf> = None;
    let mut from = 0usize;
    let mut deals = 0u32;
    let mut seed = 18u64;
    let mut field = "stage".to_string();
    let mut belief = PathBuf::from("models/belief.mjnn");
    let mut i = 0;
    while i < raw.len() {
        let v = raw.get(i + 1).cloned().unwrap_or_default();
        match raw[i].as_str() {
            "--hand" => hand = Some(PathBuf::from(&v)),
            "--from" => from = v.parse().expect("--from"),
            "--deals" => deals = v.parse().expect("--deals"),
            "--seed" => seed = v.parse().expect("--seed"),
            "--field" => field = v.clone(),
            "--belief" => belief = PathBuf::from(&v),
            other => panic!("unknown option {other}"),
        }
        i += 2;
    }
    if let Some(h) = hand {
        replay(&h, from, &belief);
    } else if deals > 0 {
        run_deals(deals, seed, &field, &belief);
    } else {
        eprintln!(
            "usage: --hand <file> [--from N] | --deals N [--seed S] [--field bot] [--belief file]"
        );
    }
}
