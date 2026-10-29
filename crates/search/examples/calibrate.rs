//! How well does PIMC's model of the future match real play, per world sampler?
//! (Plan Task 7, `docs/solver.md` §4.)
//!
//! The field is the one `danger_check` uses: seat 0 plays `value`, seats 1–3 play
//! `defensive`. At seat 0's turns the example measures, for best-of-K and (with
//! `--belief`) the belief sampler without and with importance weights:
//!
//! 1. **Shanten**: the opponents' mean shanten in sampled worlds against their
//!    real hands, per wall-progress bucket (6 buckets, as `danger.rs`). Weighted
//!    samplers average with their clipped importance weights.
//! 2. **Rollouts**: at every `--every`-th turn, PIMC (`--samples`, 4 candidates)
//!    predicts seat 0's win and deal-in rate for the rest of the hand after the
//!    move `value` really plays; these are compared with what happened.
//!
//! ```text
//! cargo run --release -p mahjong-search --example calibrate -- \
//!     --belief models/belief.mjnn --hands 200 --every 3 --samples 96
//! ```
//!
//! `--every 0` skips the PIMC part; `--tilt0`, `--tilt1` and `--draws` set the
//! belief sampler's shanten tilt (without / with weights) and candidates, for
//! fitting [`mahjong_search::world::default_tilt`] on one `--seed` and checking
//! it on another.
use mahjong_analysis::shanten;
use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::pimc::{clipped_weights, pimc_sampled};
use mahjong_search::world::{default_tilt, sample_world_from, WorldSampler, BELIEF_DRAWS};
use mahjong_search::{BeliefModel, Bot, PimcConfig};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;

const PROGRESS_BUCKETS: usize = 6;

/// Wall-progress bucket as in `danger.rs`.
fn progress_bucket(s: &State) -> usize {
    let initial = if s.rules().flowers { 91.0 } else { 83.0 };
    let progress = (1.0 - s.wall_remaining() as f64 / initial).clamp(0.0, 0.999);
    (progress * PROGRESS_BUCKETS as f64) as usize
}

/// One PIMC prediction at a decision: (predicted win, predicted deal-in, late?).
type Prediction = (f64, f64, bool);
/// A prediction with what happened: (.., actual win, actual deal-in).
type Row = (f64, f64, bool, bool, bool);

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut belief: Option<Arc<BeliefModel>> = None;
    let (mut hands, mut every, mut samples, mut worlds) = (200usize, 3usize, 96usize, 16usize);
    let mut draws = BELIEF_DRAWS;
    // Shanten tilt without and with importance weights.
    let mut tilt = [default_tilt(false), default_tilt(true)];
    let mut seed = 7u64;
    let mut i = 0;
    while i < raw.len() {
        let value = raw.get(i + 1).cloned().unwrap_or_default();
        let num = || value.parse::<usize>().expect("a number");
        match raw[i].as_str() {
            "--belief" => {
                belief = Some(Arc::new(
                    BeliefModel::load(&value).unwrap_or_else(|e| panic!("{e}")),
                ))
            }
            "--hands" => hands = num(),
            "--every" => every = num(),
            "--samples" => samples = num(),
            "--worlds" => worlds = num(),
            "--draws" => draws = num(),
            "--tilt0" => tilt[0] = value.parse().expect("a number"),
            "--tilt1" => tilt[1] = value.parse().expect("a number"),
            "--seed" => seed = num() as u64,
            other => panic!("unknown argument {other:?}"),
        }
        i += 2;
    }
    let mut samplers: Vec<(&str, WorldSampler)> = vec![("best-of-K", WorldSampler::BestOfK)];
    if let Some(m) = &belief {
        for (name, weights) in [("belief, weights=0", false), ("belief, weights=1", true)] {
            samplers.push((
                name,
                WorldSampler::Belief {
                    model: Arc::clone(m),
                    weights,
                    draws,
                    tilt: tilt[weights as usize],
                },
            ));
        }
    }
    let ns = samplers.len();
    let me_bot = Bot::by_name("value").unwrap();
    let opp = Bot::by_name("defensive").unwrap();
    let mut rng = SmallRng::seed_from_u64(seed);
    // Shanten sums per bucket: [0] real, [1 + j] sampler j; and opponent counts.
    let mut shanten_sum = vec![[0.0f64; PROGRESS_BUCKETS]; ns + 1];
    let mut shanten_n = [0u32; PROGRESS_BUCKETS];
    // Per sampler: (predicted win, predicted deal-in, late, actual win, actual deal-in).
    let mut rows: Vec<Vec<Row>> = vec![Vec::new(); ns];
    let start = std::time::Instant::now();
    for h in 0..hands {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let mut pending: Vec<Vec<Prediction>> = vec![Vec::new(); ns];
        let mut turns = 0usize;
        while let Some(p) = s.current_player() {
            let legal = s.legal_actions_vec();
            let a = if p == 0 {
                me_bot.choose(&s, &legal, &mut rng)
            } else {
                opp.choose(&s, &legal, &mut rng)
            };
            if p == 0 && s.phase_name() == "turn" {
                let b = progress_bucket(&s);
                for o in 1..4 {
                    shanten_sum[0][b] += shanten(s.hand(o), s.melds(o).len()) as f64;
                }
                shanten_n[b] += 3;
                for (j, (_, sampler)) in samplers.iter().enumerate() {
                    let mut per_world = Vec::with_capacity(worlds);
                    let mut log_w = Vec::with_capacity(worlds);
                    for k in 0..worlds {
                        let mut wrng = SmallRng::seed_from_u64((h * 7919 + turns * 31 + k) as u64);
                        let (w, lw) = sample_world_from(&s, 0, sampler, &mut wrng);
                        per_world.push(
                            (1..4)
                                .map(|o| shanten(w.hand(o), w.melds(o).len()) as f64)
                                .sum::<f64>(),
                        );
                        log_w.push(lw);
                    }
                    let wts = if sampler.weighted() {
                        clipped_weights(&log_w)
                    } else {
                        vec![1.0; worlds]
                    };
                    let total: f64 = wts.iter().sum();
                    shanten_sum[j + 1][b] +=
                        per_world.iter().zip(&wts).map(|(x, w)| x * w).sum::<f64>() / total;
                }
                if legal.len() > 1 {
                    turns += 1;
                    if every > 0 && turns.is_multiple_of(every) {
                        for (j, (_, sampler)) in samplers.iter().enumerate() {
                            let config = PimcConfig {
                                samples,
                                candidates: 4,
                                seed: (h * 1000 + turns) as u64,
                                ..PimcConfig::default()
                            };
                            let an = pimc_sampled(&s, &config, sampler);
                            if let Some(m) = an.moves.iter().find(|m| m.action == a) {
                                pending[j].push((
                                    m.win_rate,
                                    m.deal_in_rate,
                                    s.wall_remaining() < 45,
                                ));
                            }
                        }
                    }
                }
            }
            s.apply(a);
        }
        let r = s.result().expect("finished");
        let won = r.wins.iter().any(|w| w.seat == 0);
        let dealt_in = r.wins.iter().any(|w| !w.self_drawn && w.from == Some(0));
        for (j, list) in pending.into_iter().enumerate() {
            rows[j].extend(
                list.into_iter()
                    .map(|(pw, pd, late)| (pw, pd, late, won, dealt_in)),
            );
        }
        if (h + 1) % 50 == 0 {
            eprintln!("{} hands, {:.0} s", h + 1, start.elapsed().as_secs_f64());
        }
    }

    println!("1. Opponents' mean shanten at seat 0's turns, real vs sampled ({hands} hands, seed {seed}; seat 0 value, three defensive; {worlds} worlds per turn; belief draws={draws}, tilt {} / {} without / with weights)", tilt[0], tilt[1]);
    let header: Vec<String> = samplers.iter().map(|(n, _)| format!("{n:>18}")).collect();
    println!("   bucket  opponents    real {}", header.join(""));
    for b in 0..PROGRESS_BUCKETS {
        let n = shanten_n[b].max(1) as f64;
        let real = shanten_sum[0][b] / n;
        let cols: Vec<String> = (0..ns)
            .map(|j| {
                // Each turn adds the (weighted) mean of the three opponents'
                // summed shanten and counts 3 opponents, as the real column does.
                let v = shanten_sum[j + 1][b] / n;
                format!("{v:>11.2} ({:+.2})", v - real)
            })
            .collect();
        println!(
            "   {b:>6}  {:>9}  {real:>6.2} {}",
            shanten_n[b],
            cols.join("")
        );
    }

    if every == 0 {
        return;
    }
    println!("\n2. PIMC's predicted win and deal-in rate for the rest of the hand vs what happened ({samples} samples, every {every}th turn)");
    for (j, (name, _)) in samplers.iter().enumerate() {
        for (label, filter) in [
            ("early (wall ≥ 45)", Some(false)),
            ("late (wall < 45)", Some(true)),
            ("all", None),
        ] {
            let sel: Vec<_> = rows[j]
                .iter()
                .filter(|r| filter.is_none_or(|late| r.2 == late))
                .collect();
            let n = sel.len().max(1) as f64;
            let pw = sel.iter().map(|r| r.0).sum::<f64>() / n;
            let pd = sel.iter().map(|r| r.1).sum::<f64>() / n;
            let aw = sel.iter().filter(|r| r.3).count() as f64 / n;
            let ad = sel.iter().filter(|r| r.4).count() as f64 / n;
            println!(
                "   {name:<18} {label:<18} n={:>5}  win predicted {:>5.1}% actual {:>5.1}% ({:+.1})   deal-in predicted {:>5.1}% actual {:>5.1}% ({:+.1})",
                sel.len(),
                100.0 * pw,
                100.0 * aw,
                100.0 * (pw - aw),
                100.0 * pd,
                100.0 * ad,
                100.0 * (pd - ad)
            );
        }
    }
    eprintln!("done in {:.0} s", start.elapsed().as_secs_f64());
}
