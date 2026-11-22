//! Plan 2026-11-16 Task 23: the ceiling of "avoid the dealing tile", measured with
//! the cheating bot [`mahjong_search::oracle`].
//!
//! `defence_ceiling match <deals> [seed]`: each deal is played with four plain
//! `stage` bots (one game per seat, the set-up of `defence_match.rs`), then, for
//! each of the four ranges (`far`, `notready`, `all`, `far-big`), with the oracle
//! bot in each seat against three plain `stage`. Prints per range the paired
//! points a hand against plain `stage` in the same seat on the same wall (95 %
//! range over deals; the four seats of a deal are one sample), the rates beside
//! plain `stage`'s, the share of hands and decisions changed, and where the
//! points come from (outcome pairs, wall third and the threat's open sets at the
//! first change).
//!
//! `defence_ceiling reading <model> <deals> [seed]`: on the four-`stage` games of
//! the same deals, every discard decision of every seat: whether `stage`'s pick
//! truly deals in, and how the model's `dealin` head ranks it. Prints per range
//! the ranking quality on the decisions the oracle would change.
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::belief::BeliefModel;
use mahjong_search::oracle::{deal_in, discards, distance, DealIn, OracleBot, Range, BIG_FAAN};
use mahjong_search::stage::StageBot;
use mahjong_search::SeatView;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// How a hand ended for one seat.
const OUTCOMES: [&str; 5] = [
    "won",
    "dealt in",
    "other self-drew",
    "other's deal-in",
    "draw",
];

/// One seat's result in one game.
#[derive(Clone, Copy, Default)]
struct Hand {
    points: f64,
    outcome: usize,
    decisions: u32,
    covered_dealing: u32,
    changed: u32,
    /// Wall third and the threat's open sets at the first change.
    first: Option<(usize, usize)>,
}

fn deal_seed(seed: u64, i: u64) -> u64 {
    seed.wrapping_mul(1_000_003).wrapping_add(i)
}

/// Plays one deal with the oracle (`me`, or plain `stage` when `None`) in `seat`.
fn play(rules: &RuleSet, me: Option<&OracleBot>, deal: u64, seat: usize) -> Hand {
    let stage = StageBot::default();
    let mut state = State::new_deal(*rules, &mut ChaCha8Rng::seed_from_u64(deal));
    let wall0 = state.wall_remaining().max(1);
    let mut rng = ChaCha8Rng::seed_from_u64(deal.wrapping_mul(4).wrapping_add(seat as u64));
    let mut h = Hand::default();
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let p = state.current_player().expect("not over");
        let action = match me {
            Some(bot) if p == seat => {
                let w = state.wall_remaining();
                let (a, d) = bot.decide(&state, &legal, &mut rng);
                if let Some(d) = d {
                    h.decisions += 1;
                    if d.stage_deals && d.safe_exists && bot.range.covers(d.dist) {
                        h.covered_dealing += 1;
                    }
                    if d.changed() {
                        h.changed += 1;
                        if h.first.is_none() {
                            let third = ((wall0 - w.min(wall0)) * 3 / wall0).min(2);
                            h.first = Some((third, d.truth.melds));
                        }
                    }
                }
                a
            }
            _ => stage.choose(&state, &legal, &mut rng),
        };
        state.apply(action);
    }
    let r = state.result().expect("over");
    h.points = r.deltas[seat] as f64;
    h.outcome = if r.wins.iter().any(|w| w.seat == seat) {
        0
    } else if r.wins.iter().any(|w| !w.self_drawn && w.from == Some(seat)) {
        1
    } else if r.wins.iter().any(|w| w.self_drawn) {
        2
    } else if !r.wins.is_empty() {
        3
    } else {
        4
    };
    h
}

/// Mean and 95 % range of per-deal samples.
fn mean_ci(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let v = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, 1.96 * (v / n).sqrt())
}

fn rates(hands: &[Hand]) -> String {
    let n = hands.len() as f64;
    let pct = |k: usize| 100.0 * hands.iter().filter(|h| h.outcome == k).count() as f64 / n;
    format!(
        "win {:.2}%  deal-in {:.2}%  other self-drew {:.2}%  other's deal-in {:.2}%  draw {:.2}%",
        pct(0),
        pct(1),
        pct(2),
        pct(3),
        pct(4)
    )
}

fn run_match(deals: u64, seed: u64) {
    let rules = RuleSet::default();
    let bots = Range::ALL.map(OracleBot::new);
    let start = std::time::Instant::now();
    // Per deal and seat: [plain, far, notready, all, far-big].
    let rows: Vec<[[Hand; 5]; 4]> = (0..deals)
        .into_par_iter()
        .map(|i| {
            let deal = deal_seed(seed, i);
            let mut out = [[Hand::default(); 5]; 4];
            for (seat, row) in out.iter_mut().enumerate() {
                row[0] = play(&rules, None, deal, seat);
                for (k, b) in bots.iter().enumerate() {
                    row[k + 1] = play(&rules, Some(b), deal, seat);
                }
            }
            out
        })
        .collect();
    println!(
        "stage-oracle vs three stage, {deals} deals x 4 seats, seed {seed} ({:.0} s)",
        start.elapsed().as_secs_f64()
    );
    let hands_of = |k: usize| -> Vec<Hand> { rows.iter().flatten().map(|r| r[k]).collect() };
    let plain = hands_of(0);
    let (pm, pci) = mean_ci(
        &rows
            .iter()
            .map(|d| d.iter().map(|r| r[0].points).sum::<f64>() / 4.0)
            .collect::<Vec<_>>(),
    );
    println!("  plain stage: {pm:+.3} ± {pci:.3}/hand  {}", rates(&plain));
    let n_hands = plain.len() as f64;
    for (k, b) in bots.iter().enumerate() {
        let k = k + 1;
        let name = b.range.name();
        let hands = hands_of(k);
        let paired: Vec<f64> = rows
            .iter()
            .map(|d| d.iter().map(|r| r[k].points - r[0].points).sum::<f64>() / 4.0)
            .collect();
        let (m, ci) = mean_ci(&paired);
        let decisions: u32 = hands.iter().map(|h| h.decisions).sum();
        let changed: u32 = hands.iter().map(|h| h.changed).sum();
        let covered: u32 = hands.iter().map(|h| h.covered_dealing).sum();
        let changed_hands = hands.iter().filter(|h| h.changed > 0).count();
        let total_delta: f64 = paired.iter().sum::<f64>() * 4.0;
        // Hands with no change must be the plain game exactly.
        let drift = rows
            .iter()
            .flatten()
            .filter(|r| r[k].changed == 0 && r[k].points != r[0].points)
            .count();
        println!();
        println!("  == {name} ==");
        println!("  paired {name} − plain stage: {m:+.3} ± {ci:.3}/hand");
        println!("  {}", rates(&hands));
        println!(
            "  changed: {:.2}% of hands, {:.3}% of discard decisions ({changed} of {decisions}; {covered} in range with a dealing pick and a safe tile); {:+.2} points per changed decision; unchanged hands that differ from plain: {drift}",
            100.0 * changed_hands as f64 / n_hands,
            100.0 * changed as f64 / decisions.max(1) as f64,
            total_delta / changed.max(1) as f64,
        );
        // Outcome pairs over changed hands: points per hand (all hands).
        println!("  where the points come from (changed hands; plain outcome -> {name} outcome: hands, points per hand of all hands):");
        let mut cells = [[(0usize, 0.0f64); 5]; 5];
        let mut by_third = [(0usize, 0.0f64); 3];
        let mut by_melds = [(0usize, 0.0f64); 5];
        for r in rows.iter().flatten() {
            if r[k].changed == 0 {
                continue;
            }
            let d = r[k].points - r[0].points;
            let c = &mut cells[r[0].outcome][r[k].outcome];
            c.0 += 1;
            c.1 += d;
            if let Some((third, melds)) = r[k].first {
                by_third[third].0 += 1;
                by_third[third].1 += d;
                by_melds[melds.min(4)].0 += 1;
                by_melds[melds.min(4)].1 += d;
            }
        }
        for (a, row) in cells.iter().enumerate() {
            for (b, &(n, s)) in row.iter().enumerate() {
                if n > 0 {
                    println!(
                        "    {:<16} -> {:<16} {:>7}  {:+.4}",
                        OUTCOMES[a],
                        OUTCOMES[b],
                        n,
                        s / n_hands
                    );
                }
            }
        }
        // A change happens only where the plain game deals in at once (the two
        // games are the same up to it), so the gain splits into the deal-in
        // avoided (minus the plain points) and what the hand then brings the
        // oracle (its points, by its outcome).
        let mut avoided = 0.0;
        let mut after = [(0usize, 0.0f64); 5];
        for r in rows.iter().flatten().filter(|r| r[k].changed > 0) {
            avoided -= r[0].points;
            after[r[k].outcome].0 += 1;
            after[r[k].outcome].1 += r[k].points;
        }
        println!(
            "    deal-in avoided at the change: {:+.4}/hand; then the hand for {name}:",
            avoided / n_hands
        );
        for (o, &(n, s)) in after.iter().enumerate() {
            println!(
                "      {:<16} {n:>7} hands  {:+.4}/hand",
                OUTCOMES[o],
                s / n_hands
            );
        }
        for (t, &(n, s)) in by_third.iter().enumerate() {
            println!(
                "    first change in wall third {}: {n} hands, {:+.4}/hand",
                t + 1,
                s / n_hands
            );
        }
        for (o, &(n, s)) in by_melds.iter().enumerate() {
            println!(
                "    threat with {o} open sets: {n} hands, {:+.4}/hand",
                s / n_hands
            );
        }
    }
}

/// One discard decision of the four-`stage` games, for the reading check.
#[derive(Clone, Copy)]
struct Row {
    dist: i8,
    pick: DealIn,
    /// Truly safe tiles (no deal-in at all), and safe from a big deal-in.
    safe: u16,
    safe_big: u16,
    /// The model's deal-in chance of the pick.
    score: f64,
    /// Safe tiles the model rates below the pick (ties count half), out of `safe`;
    /// the same for big-safe tiles.
    below: f64,
    below_big: f64,
    /// The model's lowest-rated tile is truly safe / safe from a big deal-in.
    lowest_safe: bool,
    lowest_safe_big: bool,
}

fn run_reading(path: &str, deals: u64, seed: u64) {
    let model = BeliefModel::load(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    assert!(model.has_task19_heads(), "{path}: needs the dealin head");
    let rules = RuleSet::default();
    let stage = StageBot::default();
    let start = std::time::Instant::now();
    let rows: Vec<Row> = (0..deals)
        .into_par_iter()
        .flat_map_iter(|i| {
            let deal = deal_seed(seed, i);
            let mut state = State::new_deal(rules, &mut ChaCha8Rng::seed_from_u64(deal));
            let mut rng = ChaCha8Rng::seed_from_u64(deal.wrapping_mul(4));
            let mut out = Vec::new();
            while !state.is_terminal() {
                let legal = state.legal_actions_vec();
                let p = state.current_player().expect("not over");
                let (action, _) = stage.decide(&state, &legal, &mut rng);
                if let Action::Discard(pick) = action {
                    let cands = discards(&legal);
                    let view = SeatView::of(&state, p);
                    let dist = distance(&view, &cands);
                    let b = model.run_for(&state, p);
                    let danger = |t: mahjong_tiles::Tile| {
                        let i = t.index() as usize;
                        1.0 - b
                            .opponents
                            .iter()
                            .map(|o| 1.0 - o.dealin[i])
                            .product::<f64>()
                    };
                    let truth: Vec<(f64, DealIn)> = cands
                        .iter()
                        .map(|&t| (danger(t), deal_in(&state, p, t)))
                        .collect();
                    let me = cands.iter().position(|&t| t == pick).expect("pick");
                    let score = truth[me].0;
                    let (mut safe, mut safe_big, mut below, mut below_big) = (0, 0, 0.0, 0.0);
                    for (j, (s, d)) in truth.iter().enumerate() {
                        if j == me {
                            continue;
                        }
                        let cmp = if *s < score {
                            1.0
                        } else if *s == score {
                            0.5
                        } else {
                            0.0
                        };
                        if d.paid == 0 {
                            safe += 1;
                            below += cmp;
                        }
                        if !(d.paid > 0 && d.faan >= BIG_FAAN) {
                            safe_big += 1;
                            below_big += cmp;
                        }
                    }
                    let lowest = truth
                        .iter()
                        .min_by(|a, b| a.0.total_cmp(&b.0))
                        .map(|x| x.1)
                        .expect("a candidate");
                    out.push(Row {
                        dist,
                        pick: truth[me].1,
                        safe,
                        safe_big,
                        score,
                        below,
                        below_big,
                        lowest_safe: lowest.paid == 0,
                        lowest_safe_big: !(lowest.paid > 0 && lowest.faan >= BIG_FAAN),
                    });
                }
                state.apply(action);
            }
            out
        })
        .collect();
    println!(
        "reading {path} on {deals} deals of four stage, seed {seed}: {} discard decisions ({:.0} s)",
        rows.len(),
        start.elapsed().as_secs_f64()
    );
    for range in Range::ALL {
        let big = range == Range::FarBig;
        let deals_in = |r: &Row| range.deals(&r.pick);
        let in_range: Vec<&Row> = rows.iter().filter(|r| range.covers(r.dist)).collect();
        let pos: Vec<&Row> = in_range
            .iter()
            .copied()
            .filter(|r| deals_in(r) && if big { r.safe_big } else { r.safe } > 0)
            .collect();
        let mut neg: Vec<f64> = in_range
            .iter()
            .filter(|r| !deals_in(r))
            .map(|r| r.score)
            .collect();
        neg.sort_by(|a, b| b.total_cmp(a));
        // AUC: P(score of a dealing pick > score of a safe pick).
        let mut ps: Vec<f64> = pos.iter().map(|r| r.score).collect();
        ps.sort_by(|a, b| a.total_cmp(b));
        let mut asc = neg.clone();
        asc.sort_by(|a, b| a.total_cmp(b));
        let mut wins = 0.0;
        for &s in &ps {
            let lo = asc.partition_point(|&x| x < s);
            let hi = asc.partition_point(|&x| x <= s);
            wins += lo as f64 + 0.5 * (hi - lo) as f64;
        }
        let auc = wins / (ps.len().max(1) as f64 * asc.len().max(1) as f64);
        println!();
        println!(
            "  == {} == decisions in range {}, oracle would change {} ({:.2}%); safe picks {}",
            range.name(),
            in_range.len(),
            pos.len(),
            100.0 * pos.len() as f64 / in_range.len().max(1) as f64,
            neg.len()
        );
        println!("    AUC of the pick's read deal-in chance, dealing vs safe picks: {auc:.3}");
        for fa in [0.01, 0.05, 0.10, 0.20] {
            let k = ((fa * neg.len() as f64) as usize).min(neg.len().saturating_sub(1));
            let thr = neg.get(k).copied().unwrap_or(1.0);
            let caught = ps.iter().filter(|&&s| s > thr).count();
            // Points-weighted: the deal-in points of the caught picks whose
            // replacement (the model's safest tile) is truly safe, out of all.
            let paid_all: f64 = pos.iter().map(|r| r.pick.paid as f64).sum();
            let paid_ok: f64 = pos
                .iter()
                .filter(|r| {
                    r.score > thr
                        && if big {
                            r.lowest_safe_big
                        } else {
                            r.lowest_safe
                        }
                })
                .map(|r| r.pick.paid as f64)
                .sum();
            println!(
                "    flag the pick when its read chance > {thr:.4} ({:.0}% of safe picks flagged): catches {:.1}% of the dealing picks; caught with a truly safe replacement: {:.1}% of their deal-in points",
                100.0 * fa,
                100.0 * caught as f64 / ps.len().max(1) as f64,
                100.0 * paid_ok / paid_all.max(1.0)
            );
        }
        let within: f64 = pos
            .iter()
            .map(|r| {
                if big {
                    r.below_big / r.safe_big as f64
                } else {
                    r.below / r.safe as f64
                }
            })
            .sum::<f64>()
            / pos.len().max(1) as f64;
        let lowest = pos
            .iter()
            .filter(|r| {
                if big {
                    r.lowest_safe_big
                } else {
                    r.lowest_safe
                }
            })
            .count() as f64
            / pos.len().max(1) as f64;
        println!(
            "    within those decisions: the pick is read above {:.1}% of the safe tiles; the model's safest tile is truly safe in {:.1}%",
            100.0 * within,
            100.0 * lowest
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize, d: u64| args.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
    match args.first().map(String::as_str) {
        Some("match") => run_match(num(1, 100), num(2, 23)),
        Some("reading") => run_reading(&args[1], num(2, 100), num(3, 23)),
        _ => eprintln!("defence_ceiling match <deals> [seed] | reading <model> <deals> [seed]"),
    }
}
