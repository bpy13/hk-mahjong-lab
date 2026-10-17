//! Fits the danger tables ([`mahjong_search::danger_tables`]) from games between
//! the rule-based bots and writes `crates/search/src/danger_tables.rs`.
//!
//! At every turn, for every opponent of the player to move: is the opponent
//! waiting with a legal win (some tile completes the hand with the minimum faan)?
//! And, if so, which of the tiles that could still be discarded would win? Cells
//! with few observations are shrunk toward a coarser cell.
//!
//! `cargo run --release -p mahjong-search --example fit_danger [hands]`
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::danger::{Opponent, PROGRESS_BUCKETS, THREAT_CELLS, WAIT_CELLS};
use mahjong_search::Bot;
use mahjong_tiles::Tile;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

#[derive(Clone)]
struct Counts {
    threat: Vec<(f64, f64)>,
    wait: Vec<(f64, f64)>,
    /// Real discards: per opponent (threat cell, wait cell), and whether it was
    /// won on.
    discards: Vec<(Vec<(usize, usize)>, bool)>,
}

impl Counts {
    fn new() -> Counts {
        Counts {
            threat: vec![(0.0, 0.0); THREAT_CELLS],
            wait: vec![(0.0, 0.0); WAIT_CELLS],
            discards: Vec::new(),
        }
    }
    fn add(mut self, o: Counts) -> Counts {
        for (a, b) in self.threat.iter_mut().zip(o.threat) {
            a.0 += b.0;
            a.1 += b.1;
        }
        for (a, b) in self.wait.iter_mut().zip(o.wait) {
            a.0 += b.0;
            a.1 += b.1;
        }
        self.discards.extend(o.discards);
        self
    }
}

const BOTS: [&str; 5] = ["defensive", "value", "efficiency", "defensive", "fast"];

fn play(hand: usize) -> Counts {
    let mut rng = SmallRng::seed_from_u64(1000 + hand as u64);
    let bots: Vec<Bot> = (0..4)
        .map(|s| Bot::by_name(BOTS[(hand + s) % BOTS.len()]).unwrap())
        .collect();
    let mut c = Counts::new();
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    let mut pending: Option<(Vec<(usize, usize)>, usize)> = None;
    loop {
        if !s.in_claim() {
            if let Some((cells, from)) = pending.take() {
                let hit = s
                    .result()
                    .is_some_and(|r| r.wins.iter().any(|w| !w.self_drawn && w.from == Some(from)));
                c.discards.push((cells, hit));
            }
        }
        let Some(p) = s.current_player() else { break };
        if s.phase_name() == "turn" {
            let unseen = s.unseen(p);
            for o in (0..4).filter(|&o| o != p) {
                let opp = Opponent::of(&s, o);
                let waiting = Tile::all_kinds().any(|t| s.wins_on(o, t));
                let cell = &mut c.threat[opp.threat_cell()];
                cell.0 += waiting as u8 as f64;
                cell.1 += 1.0;
                if waiting {
                    for t in Tile::all_kinds() {
                        if let Some(w) = opp.wait_cell(t, &unseen) {
                            let cell = &mut c.wait[w];
                            cell.0 += s.wins_on(o, t) as u8 as f64;
                            cell.1 += 1.0;
                        }
                    }
                }
            }
        }
        let legal = s.legal_actions_vec();
        let a = bots[p].choose(&s, &legal, &mut rng);
        if let Action::Discard(t) = a {
            let unseen = s.unseen(p);
            let cells = (0..4)
                .filter(|&o| o != p)
                .filter_map(|o| {
                    let opp = Opponent::of(&s, o);
                    opp.wait_cell(t, &unseen).map(|w| (opp.threat_cell(), w))
                })
                .collect();
            pending = Some((cells, p));
        }
        s.apply(a);
    }
    c
}

/// Raw edges for the calibration curve.
const RAW_EDGES: [f64; 11] = [
    0.0, 0.0025, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.25,
];

/// Maps raw estimates of real discards to observed deal-in rates: per bucket
/// (mean raw, smoothed observed rate), made non-decreasing.
fn calibration(raw: &[(f64, bool)]) -> Vec<(f64, f64)> {
    let mut out: Vec<(f64, f64)> = Vec::new();
    for (i, &lo) in RAW_EDGES.iter().enumerate() {
        let hi = RAW_EDGES.get(i + 1).copied().unwrap_or(1.01);
        let b: Vec<_> = raw.iter().filter(|r| r.0 >= lo && r.0 < hi).collect();
        if b.len() < 30 {
            continue;
        }
        let n = b.len() as f64;
        let mean = b.iter().map(|r| r.0).sum::<f64>() / n;
        let hits = b.iter().filter(|r| r.1).count() as f64;
        // Shrink toward the raw estimate with 20 pseudo-observations.
        let rate = (hits + 20.0 * mean) / (n + 20.0);
        out.push((mean, rate));
        println!(
            "  raw {lo:.4}–{hi:.4}: n={:>7}  mean raw {:.4}  observed {:.4}",
            b.len(),
            mean,
            hits / n
        );
    }
    let mut m = 0.0f64;
    for p in &mut out {
        m = m.max(p.1);
        p.1 = m;
    }
    out
}

/// Shrinks each cell toward its group's pooled rate (`group` maps a cell to its
/// parent), with `k` pseudo-observations.
fn shrink(cells: &[(f64, f64)], group: impl Fn(usize) -> usize, k: f64) -> Vec<f64> {
    let total: (f64, f64) = cells.iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    let overall = total.0 / total.1.max(1.0);
    let mut parents = std::collections::HashMap::<usize, (f64, f64)>::new();
    for (i, c) in cells.iter().enumerate() {
        let e = parents.entry(group(i)).or_default();
        e.0 += c.0;
        e.1 += c.1;
    }
    cells
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let p = parents[&group(i)];
            let parent = (p.0 + k * overall) / (p.1 + k);
            (c.0 + k * parent) / (c.1 + k)
        })
        .collect()
}

fn table(name: &str, len: &str, values: &[f64]) -> String {
    let body: Vec<String> = values
        .chunks(6)
        .map(|ch| {
            let v: Vec<String> = ch.iter().map(|x| format!("{x:.5}")).collect();
            format!("    {},", v.join(", "))
        })
        .collect();
    format!(
        "pub const {name}: [f64; {len}] = [\n{}\n];\n",
        body.join("\n")
    )
}

fn main() {
    let hands: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4000);
    let c = (0..hands)
        .into_par_iter()
        .map(play)
        .reduce(Counts::new, Counts::add);
    // Threat: shrink toward (exposed, progress), i.e. drop the meld tag.
    let threat = shrink(&c.threat, |i| i / 3, 40.0);
    // Wait: shrink toward (class, unseen, discarded), i.e. drop the flush relation.
    let wait = shrink(&c.wait, |i| i / 3, 60.0);

    println!(
        "threat (P waiting with a legal win), exposed × progress bucket, tag plain/value/flush:"
    );
    for e in 0..5 {
        let row: Vec<String> = (0..PROGRESS_BUCKETS)
            .map(|p| {
                let b = (e * PROGRESS_BUCKETS + p) * 3;
                format!("{:.2}/{:.2}/{:.2}", threat[b], threat[b + 1], threat[b + 2])
            })
            .collect();
        println!("  {e} melds: {}", row.join("  "));
    }
    let raw: Vec<(f64, bool)> = c
        .discards
        .iter()
        .map(|(cells, hit)| {
            let safe: f64 = cells
                .iter()
                .map(|&(a, b)| 1.0 - threat[a] * wait[b])
                .product();
            (1.0 - safe, *hit)
        })
        .collect();
    println!("calibration on {} real discards:", raw.len());
    let calib = calibration(&raw);
    let n: f64 = c.threat.iter().map(|x| x.1).sum();
    let w: f64 = c.wait.iter().map(|x| x.1).sum();
    println!("observations: {n} opponent-turns, {w} waiting tile checks");

    let src = format!(
        "//! Fitted tables for [`crate::danger`]. Generated by\n\
         //! `cargo run --release -p mahjong-search --example fit_danger {hands}`\n\
         //! ({n} opponent-turns from games between the rule-based bots); do not edit.\n\n\
         use crate::danger::{{THREAT_CELLS, WAIT_CELLS}};\n\n\
         /// P(opponent waits with a legal win), per [`crate::danger::Opponent::threat_cell`].\n\
         #[rustfmt::skip]\n{}\n\
         /// P(a waiting opponent wins on the tile), per [`crate::danger::Opponent::wait_cell`].\n\
         #[rustfmt::skip]\n{}\n\
         /// (raw estimate, observed deal-in rate) on real discards, raw increasing;\n\
         /// [`crate::danger::Danger::tile`] interpolates between the points.\n\
         #[rustfmt::skip]\npub const CALIBRATION: [(f64, f64); {}] = [\n{}\n];\n",
        table("THREAT", "THREAT_CELLS", &threat),
        table("WAIT", "WAIT_CELLS", &wait),
        calib.len(),
        calib
            .iter()
            .map(|(a, b)| format!("    ({a:.5}, {b:.5}),"))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/danger_tables.rs");
    std::fs::write(path, src).expect("write tables");
    println!("wrote {path}");
}
