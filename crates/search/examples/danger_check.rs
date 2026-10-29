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
//! Games: seat 0 plays `value`, the others `defensive` (the UI's 中級).
//! `cargo run --release -p mahjong-search --example danger_check -- [--belief <model>] [hands] [pimc_every]`
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::{pimc, Belief, BeliefModel, Bot, Danger, PimcConfig};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

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
