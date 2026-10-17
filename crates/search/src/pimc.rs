//! Perfect-information Monte Carlo (PIMC) search.
//!
//! To evaluate the player to act's options:
//!
//! 1. Shortlist the candidates: every special action (win, kong, claim, pass)
//!    plus the best `candidates` discards by the faan-aware measure.
//! 2. Repeatedly sample a **determinization**: the other hands and the wall
//!    re-dealt at random, consistent with what the player has seen.
//! 3. In each sample, play every candidate to the end of the hand with the
//!    [`Fast`] bot playing all four seats. All candidates share the same sample
//!    (common random numbers), so differences between them are measured with
//!    much less noise than their absolute values.
//! 4. Report, per candidate: mean points, a 95% confidence interval, and the
//!    rates of winning and of dealing in.
//!
//! Like all PIMC methods it assumes the hidden tiles are uniformly random (it does
//! not infer anything from how opponents have played) and lets each sample's
//! future be played as if everyone knew it ("strategy fusion"). Its numbers are
//! estimates under the rollout bot's style of play, not equilibrium values.

use crate::bots::{best_discard, Fast, Metric};
use crate::view::SeatView;
use mahjong_games::{play_out, Game};
use mahjong_hk::{Action, State};
use mahjong_tiles::Tile;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::time::{Duration, Instant};

/// PIMC settings.
#[derive(Clone, Debug)]
pub struct PimcConfig {
    /// Maximum number of determinizations.
    pub samples: usize,
    /// Stop early when this much time has passed (checked between batches).
    pub time_limit: Option<Duration>,
    /// How many discards to evaluate (best first by the faan-aware measure).
    pub candidates: usize,
    /// Seed for reproducible results.
    pub seed: u64,
}

impl Default for PimcConfig {
    fn default() -> Self {
        PimcConfig {
            samples: 400,
            time_limit: None,
            candidates: 6,
            seed: 0,
        }
    }
}

/// Results for one candidate action.
#[derive(Clone, Debug)]
pub struct ActionStats {
    pub action: Action,
    /// Samples played.
    pub n: u32,
    /// Mean points for the acting player.
    pub mean: f64,
    /// Half-width of the 95% confidence interval of `mean`.
    pub ci95: f64,
    /// Fraction of samples the acting player won.
    pub win_rate: f64,
    /// Fraction of samples the acting player paid for someone's discard win.
    pub deal_in_rate: f64,
    /// Fraction of samples that ended in an exhaustive draw.
    pub draw_rate: f64,
}

/// The analysis of one decision.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub seat: usize,
    /// Candidates, best mean first.
    pub moves: Vec<ActionStats>,
    /// Determinizations sampled.
    pub samples: usize,
    pub elapsed: Duration,
}

impl Analysis {
    /// The recommended action.
    pub fn best(&self) -> Action {
        self.moves[0].action
    }
}

#[derive(Clone, Copy, Default)]
struct Acc {
    n: u32,
    sum: f64,
    sum_sq: f64,
    wins: u32,
    deal_ins: u32,
    draws: u32,
}

impl Acc {
    fn add(&mut self, other: &Acc) {
        self.n += other.n;
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
        self.wins += other.wins;
        self.deal_ins += other.deal_ins;
        self.draws += other.draws;
    }

    fn record(&mut self, state: &State, seat: usize) {
        let r = state.result().expect("finished");
        let x = r.deltas[seat] as f64;
        self.n += 1;
        self.sum += x;
        self.sum_sq += x * x;
        self.wins += r.wins.iter().any(|w| w.seat == seat) as u32;
        self.deal_ins += r.wins.iter().any(|w| !w.self_drawn && w.paid_by[seat] > 0) as u32;
        self.draws += r.wins.is_empty() as u32;
    }
}

/// The candidates PIMC evaluates in `state` (see the module docs).
pub fn shortlist(state: &State, discards: usize) -> Vec<Action> {
    let legal = state.legal_actions_vec();
    let mut out: Vec<Action> = legal
        .iter()
        .copied()
        .filter(|a| !matches!(a, Action::Discard(_)))
        .collect();
    let tiles: Vec<Tile> = legal
        .iter()
        .filter_map(|a| {
            if let Action::Discard(t) = a {
                Some(*t)
            } else {
                None
            }
        })
        .collect();
    if !tiles.is_empty() {
        let view = SeatView::of_current(state);
        let mut remaining = tiles;
        for _ in 0..discards.min(remaining.len()) {
            let (t, _) = best_discard(&view, &remaining, Metric::Faan);
            out.push(Action::Discard(t));
            remaining.retain(|&x| x != t);
        }
    }
    out
}

/// Runs PIMC for the player to act.
///
/// # Panics
/// Panics if the hand is over.
pub fn pimc(state: &State, config: &PimcConfig) -> Analysis {
    let start = Instant::now();
    let seat = state.current_player().expect("hand in progress");
    let candidates = shortlist(state, config.candidates.max(1));
    let rollout = Fast::default();
    let mut totals = vec![Acc::default(); candidates.len()];
    let batch = rayon::current_num_threads().max(1) * 4;
    let mut done = 0;
    while done < config.samples {
        let n = batch.min(config.samples - done);
        let results: Vec<Vec<Acc>> = (done..done + n)
            .into_par_iter()
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(
                    config.seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let world = state.determinize(seat, &mut rng);
                candidates
                    .iter()
                    .map(|&a| {
                        let mut s = world.clone();
                        s.apply(a);
                        play_out(&mut s, &rollout, &mut rng);
                        let mut acc = Acc::default();
                        acc.record(&s, seat);
                        acc
                    })
                    .collect()
            })
            .collect();
        for r in results {
            for (t, a) in totals.iter_mut().zip(r.iter()) {
                t.add(a);
            }
        }
        done += n;
        if config
            .time_limit
            .is_some_and(|limit| start.elapsed() >= limit)
        {
            break;
        }
    }
    let mut moves: Vec<ActionStats> = candidates
        .iter()
        .zip(totals.iter())
        .map(|(&action, a)| {
            let n = a.n.max(1) as f64;
            let mean = a.sum / n;
            let var = (a.sum_sq / n - mean * mean).max(0.0);
            ActionStats {
                action,
                n: a.n,
                mean,
                ci95: 1.96 * (var / n).sqrt(),
                win_rate: a.wins as f64 / n,
                deal_in_rate: a.deal_ins as f64 / n,
                draw_rate: a.draws as f64 / n,
            }
        })
        .collect();
    moves.sort_by(|a, b| b.mean.total_cmp(&a.mean));
    Analysis {
        seat,
        moves,
        samples: done,
        elapsed: start.elapsed(),
    }
}
