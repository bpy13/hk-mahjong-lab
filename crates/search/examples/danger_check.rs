//! Are the deal-in numbers shown in study mode calibrated?
//!
//! 1. The per-tile danger heuristic ([`Danger`], the coloured bar under each tile):
//!    for every discard in real games, the estimate before the discard against
//!    whether that discard was won on, in buckets.
//! 2. The solver's 出銃率 (PIMC `deal_in_rate`, the chance of dealing in at any
//!    point in the rest of the hand): the estimate for the move actually played
//!    against whether that seat later dealt in.
//!
//! 3. With `--belief <model>` (plan Task 6): the payment-aware danger from the
//!    belief model ([`Danger::from_belief`]). For every discard by any seat, the
//!    predicted expected loss in points against the points that seat really paid
//!    on that discard, per wall-progress bucket (6 buckets, as `danger.rs`), with
//!    a 95 % interval of the observed mean; and the predicted deal-in chance
//!    against the observed rate. The PIMC part is off in this mode unless a
//!    second number asks for it.
//!
//! 4. With `--exact <deals> [seed]` (plan Task 13): four `stage` bots; every
//!    discard facing an opponent with 1 or 4 concealed tiles, the table figure
//!    and the exact-counting figure ([`mahjong_search::exact_wait`]) in ten bins
//!    side by side, and the check that a figure of exactly 1.0 always deals in to
//!    that opponent and 0.0 never does.
//!
//! Games: seat 0 plays `value`, the others `defensive` (the UI's 中級).
//! `cargo run --release -p mahjong-search --example danger_check -- [--belief <model>] [hands] [pimc_every]`
//! `cargo run --release -p mahjong-search --example danger_check -- --exact 40000 1`
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::danger::with_exact;
use mahjong_search::ev::HeuristicBelief;
use mahjong_search::exact_wait::exact_wait;
use mahjong_search::giveup;
use mahjong_search::{pimc, Belief, BeliefModel, Bot, Danger, PimcConfig};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const EDGES: [f64; 7] = [0.0, 0.01, 0.02, 0.04, 0.07, 0.10, 0.20];
const PROGRESS_BUCKETS: usize = 6;

/// Wall-progress bucket as in `danger.rs`.
fn progress_bucket(s: &State) -> usize {
    let initial = if s.rules().flowers { 91.0 } else { 83.0 };
    let progress = (1.0 - s.wall_remaining() as f64 / initial).clamp(0.0, 0.999);
    (progress * PROGRESS_BUCKETS as f64) as usize
}

/// Points `seat` paid for wins on its own discard.
fn paid_on_discard(s: &State, seat: usize) -> f64 {
    s.result().map_or(0.0, |r| {
        r.wins
            .iter()
            .filter(|w| !w.self_drawn && w.from == Some(seat))
            .map(|w| w.paid_by[seat] as f64)
            .sum()
    })
}

/// Per-bucket sums for the belief check.
#[derive(Clone, Copy, Default)]
struct LossBucket {
    n: f64,
    pred_loss: f64,
    obs_loss: f64,
    obs_loss_sq: f64,
    pred_p: f64,
    hits: f64,
    /// Over discards that were won on: Σ predicted points given a deal-in
    /// (expected loss / deal-in chance), and the points really paid.
    hit_pred_pay: f64,
    hit_obs_pay: f64,
}

fn bucket(p: f64) -> usize {
    EDGES.iter().rposition(|&e| p >= e).unwrap_or(0)
}

fn dealt_in(s: &State, seat: usize) -> bool {
    s.result()
        .is_some_and(|r| r.wins.iter().any(|w| !w.self_drawn && w.from == Some(seat)))
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().map(String::as_str) == Some("--exact") {
        let num = |i: usize, d: u64| raw.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
        run_exact(num(1, 400), num(2, 1));
        return;
    }
    let mut belief: Option<BeliefModel> = None;
    let mut args: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        if raw[i] == "--belief" {
            let path = raw.get(i + 1).expect("--belief needs a model path");
            belief = Some(BeliefModel::load(path).unwrap_or_else(|e| panic!("{e}")));
            i += 2;
            continue;
        }
        args.extend(raw[i].parse::<usize>().ok());
        i += 1;
    }
    let hands = args.first().copied().unwrap_or(400);
    let pimc_every = args
        .get(1)
        .copied()
        .unwrap_or(if belief.is_some() { 0 } else { 6 });
    let mut loss_buckets = [LossBucket::default(); PROGRESS_BUCKETS];
    let me_bot = Bot::by_name("value").unwrap();
    let opp = Bot::by_name("defensive").unwrap();
    let mut rng = SmallRng::seed_from_u64(11);
    // [bucket] -> (sum of estimates, deal-ins, count)
    let mut tile_buckets = [(0.0f64, 0u32, 0u32); EDGES.len()];
    // PIMC: per decision (predicted deal-in, predicted win, actual deal-in, actual win, stage)
    let mut pimc_rows: Vec<(f64, f64, bool, bool, usize)> = Vec::new();
    for h in 0..hands {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let mut pending: Vec<(f64, f64, usize)> = Vec::new();
        let mut turns = 0usize;
        while let Some(p) = s.current_player() {
            let legal = s.legal_actions_vec();
            let a = if p == 0 {
                me_bot.choose(&s, &legal, &mut rng)
            } else {
                opp.choose(&s, &legal, &mut rng)
            };
            if p == 0 && s.phase_name() == "turn" && legal.len() > 1 {
                turns += 1;
                if pimc_every > 0 && turns.is_multiple_of(pimc_every) {
                    let an = pimc(
                        &s,
                        &PimcConfig {
                            samples: 96,
                            candidates: 4,
                            seed: (h * 1000 + turns) as u64,
                            ..PimcConfig::default()
                        },
                    );
                    if let Some(m) = an.moves.iter().find(|m| m.action == a) {
                        let stage = (s.wall_remaining() < 45) as usize;
                        pending.push((m.deal_in_rate, m.win_rate, stage));
                    }
                }
            }
            if let Action::Discard(t) = a {
                let d = Danger::estimate(&s, p).tile(t);
                let predicted = belief.as_ref().map(|m| {
                    let db = Danger::from_belief(&s, p, m as &dyn Belief);
                    (progress_bucket(&s), db.expected_loss(t), db.tile(t))
                });
                let mut next = s.clone();
                next.apply(a);
                // Resolve the claim window on this discard.
                while next.in_claim() {
                    let q = next.current_player().unwrap();
                    let l = next.legal_actions_vec();
                    let c = if q == 0 {
                        me_bot.choose(&next, &l, &mut rng)
                    } else {
                        opp.choose(&next, &l, &mut rng)
                    };
                    next.apply(c);
                }
                let hit = dealt_in(&next, p);
                if let Some((pb, loss, prob)) = predicted {
                    let paid = paid_on_discard(&next, p);
                    let lb = &mut loss_buckets[pb];
                    lb.n += 1.0;
                    lb.pred_loss += loss;
                    lb.obs_loss += paid;
                    lb.obs_loss_sq += paid * paid;
                    lb.pred_p += prob;
                    lb.hits += hit as u8 as f64;
                    if hit && prob > 0.0 {
                        lb.hit_pred_pay += loss / prob;
                        lb.hit_obs_pay += paid;
                    }
                }
                let b = &mut tile_buckets[bucket(d)];
                b.0 += d;
                b.1 += hit as u32;
                b.2 += 1;
                s = next;
                continue;
            }
            s.apply(a);
            let _ = rng.gen::<u8>();
        }
        let won = s.result().unwrap().wins.iter().any(|w| w.seat == 0);
        let di = dealt_in(&s, 0);
        for (pd, pw, st) in pending {
            pimc_rows.push((pd, pw, di, won, st));
        }
    }

    println!("1. Per-tile danger heuristic vs immediate deal-in ({hands} hands)");
    println!("   bucket        discards   mean estimate   actual deal-in");
    for (i, (sum, hit, n)) in tile_buckets.iter().enumerate() {
        if *n == 0 {
            continue;
        }
        let hi = EDGES
            .get(i + 1)
            .map_or("+".to_string(), |e| format!("{:.0}%", e * 100.0));
        println!(
            "   {:>3.0}%–{:<5} {:>9}   {:>12.1}%   {:>8.1}% ± {:.1}",
            EDGES[i] * 100.0,
            hi,
            n,
            100.0 * sum / *n as f64,
            100.0 * *hit as f64 / *n as f64,
            196.0
                * ((*hit as f64 / *n as f64) * (1.0 - *hit as f64 / *n as f64) / *n as f64).sqrt(),
        );
    }
    let (s_all, h_all, n_all) = tile_buckets
        .iter()
        .fold((0.0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    println!(
        "   all           {:>9}   {:>12.1}%   {:>8.1}%",
        n_all,
        100.0 * s_all / n_all as f64,
        100.0 * h_all as f64 / n_all as f64
    );

    if belief.is_some() {
        println!(
            "\n3. Belief model: expected loss on the tile discarded vs points paid on it ({hands} hands, every seat)"
        );
        println!(
            "   bucket  discards   predicted   observed  95% interval          within   deal-in predicted  observed  95% interval   points | deal-in: predicted observed"
        );
        let mut all_within = true;
        for (b, lb) in loss_buckets.iter().enumerate() {
            if lb.n == 0.0 {
                continue;
            }
            let mean = lb.obs_loss / lb.n;
            let var = (lb.obs_loss_sq / lb.n - mean * mean).max(0.0);
            let half = 1.96 * (var / lb.n).sqrt();
            let pred = lb.pred_loss / lb.n;
            // With no observed loss the interval is [0, 0]; use the rule of
            // three's upper bound (3 / n deal-ins of the mean paid) instead.
            let (lo, hi) = if lb.obs_loss == 0.0 {
                (0.0, 3.0 / lb.n * 8.0)
            } else {
                (mean - half, mean + half)
            };
            let within = (lo..=hi).contains(&pred);
            all_within &= within;
            let rate = lb.hits / lb.n;
            let rhalf = 1.96 * (rate * (1.0 - rate) / lb.n).sqrt();
            println!(
                "   {b:>6}  {:>8}   {pred:>9.4}  {mean:>9.4}  [{lo:>7.4}, {hi:>7.4}]  {:>6}   {:>16.4}%  {:>7.4}%  [{:.4}, {:.4}]%   {:>8.1} {:>8.1}",
                lb.n as u64,
                if within { "yes" } else { "no" },
                100.0 * lb.pred_p / lb.n,
                100.0 * rate,
                100.0 * (rate - rhalf).max(0.0),
                100.0 * (rate + rhalf),
                lb.hit_pred_pay / lb.hits.max(1.0),
                lb.hit_obs_pay / lb.hits.max(1.0),
            );
        }
        println!(
            "   every bucket within its interval: {}",
            if all_within { "yes" } else { "no" }
        );
    }

    if pimc_every == 0 {
        return;
    }
    println!("\n2. Solver 出銃率 / 勝率 (rest of the hand) vs what happened");
    for (st, name) in [(0, "early (wall ≥ 45)"), (1, "late (wall < 45)")] {
        let rows: Vec<_> = pimc_rows.iter().filter(|r| r.4 == st).collect();
        let n = rows.len().max(1) as f64;
        let pd: f64 = rows.iter().map(|r| r.0).sum::<f64>() / n;
        let ad = rows.iter().filter(|r| r.2).count() as f64 / n;
        let pw: f64 = rows.iter().map(|r| r.1).sum::<f64>() / n;
        let aw = rows.iter().filter(|r| r.3).count() as f64 / n;
        println!(
            "   {name:<18} n={:>4}  deal-in predicted {:>5.1}% actual {:>5.1}%   win predicted {:>5.1}% actual {:>5.1}%",
            rows.len(),
            100.0 * pd,
            100.0 * ad,
            100.0 * pw,
            100.0 * aw
        );
    }
}

// ----- `--exact` (plan Task 13): exact counting against the tables. -----

/// Danger bins of the exact check (upper edges, in %), as `giveup_log`.
const EXACT_EDGES: [f64; 9] = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 25.0];
const EXACT_BINS: usize = EXACT_EDGES.len() + 1;

fn exact_bin(d: f64) -> usize {
    EXACT_EDGES
        .iter()
        .position(|&e| 100.0 * d < e)
        .unwrap_or(EXACT_EDGES.len())
}

#[derive(Clone, Copy, Default)]
struct Cell {
    n: f64,
    d: f64,
    hits: f64,
}

/// Which opponents a discard faced: 0 = one with 1 concealed tile, 1 = one with
/// 4 (and none with 1).
const GROUPS: usize = 2;

#[derive(Clone, Copy, Default)]
struct ExactLog {
    deals: f64,
    discards: f64,
    /// Discards facing an exactly counted opponent, by group.
    faced: [f64; GROUPS],
    /// Opponent-discard pairs by concealed count (1, 4).
    pairs: [f64; 2],
    /// [group][bin], table figure and exact figure.
    table: [[Cell; EXACT_BINS]; GROUPS],
    exact: [[Cell; EXACT_BINS]; GROUPS],
    /// Per opponent with 1 / 4 concealed tiles: pairs given exactly 1.0, and of
    /// those how many that opponent won on; given 0.0, and how many it won on.
    one: [f64; 2],
    one_hit: [f64; 2],
    zero: [f64; 2],
    zero_hit: [f64; 2],
    /// Discards where the exact figure differs from the table figure by more
    /// than 0.01, and where the discarded tile's figure moved to exactly 0 or 1.
    changed: f64,
}

impl ExactLog {
    fn add(mut self, o: ExactLog) -> ExactLog {
        self.deals += o.deals;
        self.discards += o.discards;
        self.changed += o.changed;
        for g in 0..GROUPS {
            self.faced[g] += o.faced[g];
            for b in 0..EXACT_BINS {
                for (a, x) in [
                    (&mut self.table[g][b], o.table[g][b]),
                    (&mut self.exact[g][b], o.exact[g][b]),
                ] {
                    a.n += x.n;
                    a.d += x.d;
                    a.hits += x.hits;
                }
            }
        }
        for k in 0..2 {
            self.pairs[k] += o.pairs[k];
            self.one[k] += o.one[k];
            self.one_hit[k] += o.one_hit[k];
            self.zero[k] += o.zero[k];
            self.zero_hit[k] += o.zero_hit[k];
        }
        self
    }
}

/// One deal of four `stage` bots; every discard facing an opponent with 1 or 4
/// concealed tiles is checked. A miss (a 1.0 that did not deal in, a 0.0 that
/// did) is printed to stderr with its deal seed.
fn exact_deal(rules: &RuleSet, deal_seed: u64) -> ExactLog {
    let bot = Bot::by_name("stage").unwrap();
    let mut state = State::new_deal(*rules, &mut ChaCha8Rng::seed_from_u64(deal_seed));
    let mut rng = ChaCha8Rng::seed_from_u64(deal_seed ^ 0x9e37_79b9);
    let mut log = ExactLog {
        deals: 1.0,
        ..ExactLog::default()
    };
    while let Some(seat) = state.current_player() {
        let legal = state.legal_actions_vec();
        let action = bot.choose(&state, &legal, &mut rng);
        let Action::Discard(t) = action else {
            state.apply(action);
            continue;
        };
        log.discards += 1.0;
        let off = Danger::from_belief(&state, seat, &HeuristicBelief);
        let on = with_exact(true, || Danger::from_belief(&state, seat, &HeuristicBelief));
        let mut next = state.clone();
        next.apply(action);
        while next.in_claim() {
            let l = next.legal_actions_vec();
            let c = bot.choose(&next, &l, &mut rng);
            next.apply(c);
        }
        let Some(split) = on.split(t) else {
            state = next;
            continue;
        };
        let winners: Vec<usize> = next.result().map_or(Vec::new(), |r| {
            r.wins
                .iter()
                .filter(|w| !w.self_drawn && !w.robbed && w.from == Some(seat))
                .map(|w| w.seat)
                .collect()
        });
        let hit = !winners.is_empty();
        let opps: Vec<usize> = (0..4).filter(|&o| o != seat && on.is_exact(o)).collect();
        let group = if opps.iter().any(|&o| state.concealed_count(o) == 1) {
            0
        } else {
            1
        };
        log.faced[group] += 1.0;
        for &o in &opps {
            let k = (state.concealed_count(o) == 4) as usize;
            log.pairs[k] += 1.0;
            let e = exact_wait(&state, seat, o).expect("exact opponent");
            let ready = on.threat(o);
            let p = ready * e.win[t.index() as usize];
            let won = winners.contains(&o);
            if p == 1.0 {
                log.one[k] += 1.0;
                log.one_hit[k] += won as u8 as f64;
                if !won {
                    eprintln!("MISS 1.0 not won: seed {deal_seed} seat {seat} opp {o} tile {t}");
                }
            } else if p == 0.0 {
                log.zero[k] += 1.0;
                log.zero_hit[k] += won as u8 as f64;
                if won {
                    eprintln!("MISS 0.0 won: seed {deal_seed} seat {seat} opp {o} tile {t}");
                }
            }
        }
        let table = giveup::danger_of(&off, t).0;
        let exact = 1.0 - (1.0 - giveup::calibrate(split.table_tile)) * (1.0 - split.exact_tile);
        if (exact - table).abs() > 0.01 {
            log.changed += 1.0;
        }
        for (cells, d) in [
            (&mut log.table[group], table),
            (&mut log.exact[group], exact),
        ] {
            let c = &mut cells[exact_bin(d)];
            c.n += 1.0;
            c.d += d;
            c.hits += hit as u8 as f64;
        }
        state = next;
    }
    log
}

fn print_exact_bins(title: &str, cells: &[Cell; EXACT_BINS]) {
    println!("  {title}");
    println!("  | predicted bin | discards | mean predicted | observed | 95 % range | observed / predicted |");
    println!("  |---|---|---|---|---|---|");
    for (i, c) in cells.iter().enumerate() {
        if c.n == 0.0 {
            continue;
        }
        let lo = if i == 0 { 0.0 } else { EXACT_EDGES[i - 1] };
        let name = match EXACT_EDGES.get(i) {
            Some(hi) => format!("{lo}–{hi} %"),
            None => format!("{lo} % +"),
        };
        let pred = c.d / c.n;
        let obs = c.hits / c.n;
        let half = 1.96 * (obs * (1.0 - obs) / c.n).sqrt();
        let ratio = if pred > 0.0 {
            format!("{:.2}", obs / pred)
        } else {
            "–".to_string()
        };
        println!(
            "  | {name} | {} | {:.2} % | {:.2} % | {:.2}–{:.2} % | {ratio} |",
            c.n,
            100.0 * pred,
            100.0 * obs,
            100.0 * (obs - half).max(0.0),
            100.0 * (obs + half).min(1.0),
        );
    }
}

fn run_exact(deals: u64, seed: u64) {
    let rules = RuleSet::default();
    let start = std::time::Instant::now();
    let log = (0..deals)
        .into_par_iter()
        .map(|i| exact_deal(&rules, seed.wrapping_mul(1_000_003).wrapping_add(i)))
        .reduce(ExactLog::default, ExactLog::add);
    println!(
        "exact counting check: {deals} deals of four `stage` bots, seed {seed} ({:?})",
        start.elapsed()
    );
    println!(
        "discards {}; facing an opponent with 1 concealed tile: {}; with 4 (none with 1): {}",
        log.discards, log.faced[0], log.faced[1]
    );
    println!(
        "opponent-discard pairs: 1 concealed tile {}, 4 concealed tiles {}",
        log.pairs[0], log.pairs[1]
    );
    for (k, name) in ["1 concealed tile", "4 concealed tiles"].iter().enumerate() {
        println!(
            "{name}: given exactly 1.0: {} (that opponent won on it: {}); given exactly 0.0: {} (won on it: {})",
            log.one[k], log.one_hit[k], log.zero[k], log.zero_hit[k]
        );
    }
    println!(
        "discards whose figure moved by more than 1 point of a percent: {}",
        log.changed
    );
    for (g, name) in [
        "facing an opponent with 1 concealed tile",
        "facing an opponent with 4 concealed tiles (none with 1)",
    ]
    .iter()
    .enumerate()
    {
        println!("\n{name}");
        print_exact_bins(
            "table figure (Danger::from_belief, heuristic, through giveup::danger_of)",
            &log.table[g],
        );
        print_exact_bins(
            "exact figure (same, exact opponents outside the curve)",
            &log.exact[g],
        );
    }
    let misses = log.one[0] - log.one_hit[0] + log.one[1] - log.one_hit[1]
        + log.zero_hit[0]
        + log.zero_hit[1];
    println!("\nmisses (1.0 not won on + 0.0 won on): {misses}");
}
