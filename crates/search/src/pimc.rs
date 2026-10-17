//! Perfect-information Monte Carlo (PIMC) search.
//!
//! To evaluate the player to act's options:
//!
//! 1. Shortlist the candidates: every special action (win, kong, claim, pass),
//!    the best `candidates` discards by the faan-aware measure, and the
//!    [`Value`] bot's own choice, which serves as the **baseline**.
//! 2. Repeatedly sample a **determinization**: the other hands and the wall
//!    re-dealt consistently with what the player has seen, with opponents' hands
//!    as developed as real ones at this stage ([`crate::world`]).
//! 3. In each sample, play every candidate to the end of the hand with the
//!    [`Rollout`] policy: the [`Fast`] bot for the player's own later moves, and
//!    a defending `Fast` (it folds against threats, like the `defensive` bot)
//!    for the opponents. All candidates share the same sample (common random
//!    numbers). Opponents that never defend made the player's win rate look far
//!    higher than in real games (37–40% predicted against 22–29% observed; the
//!    `danger_check` example measures this).
//! 4. Report, per candidate: mean points with a 95% confidence interval, win,
//!    deal-in and draw rates, and the **paired difference** from the baseline:
//!    its mean over samples and standard error. Because both moves see the same
//!    futures, the difference is measured far more precisely than either mean.
//!
//! [`Analysis::choose`] turns this into a move: always declare a legal win;
//! otherwise play the candidate with the best mean only if it beats the baseline
//! by more than `z` standard errors of the paired difference, else the baseline.
//! With few samples the search therefore defers to the heuristic instead of
//! acting on noise.
//!
//! Like all PIMC methods it infers little from how opponents have played (beyond
//! how developed hands typically are at this stage) and lets each sample's
//! future be played as if everyone knew it ("strategy fusion"). Its numbers are
//! estimates under the rollout bot's style of play, not equilibrium values.

use crate::bots::{best_discard, Bot, Fast, Metric, Value};
use crate::view::SeatView;
use crate::world::sample_world;
use mahjong_games::{play_out, Game, Policy};
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
    /// How many standard errors a candidate must beat the baseline by to be
    /// chosen over it ([`Analysis::choose`]).
    pub z: f64,
}

impl Default for PimcConfig {
    fn default() -> Self {
        PimcConfig {
            samples: 400,
            time_limit: None,
            candidates: 6,
            seed: 0,
            z: 2.0,
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
    /// Mean of (this action's points − the baseline's points) over samples.
    pub diff: f64,
    /// Standard error of `diff`.
    pub diff_se: f64,
}

/// The analysis of one decision.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub seat: usize,
    /// Candidates, best mean first.
    pub moves: Vec<ActionStats>,
    /// The heuristic's choice that differences are measured against.
    pub baseline: Action,
    /// Determinizations sampled.
    pub samples: usize,
    pub elapsed: Duration,
}

impl Analysis {
    /// The candidate with the best mean (what study mode ranks first).
    pub fn best(&self) -> Action {
        self.moves[0].action
    }

    /// The move to play (see the module docs).
    pub fn choose(&self, z: f64) -> Action {
        if let Some(m) = self
            .moves
            .iter()
            .find(|m| matches!(m.action, Action::Win | Action::SelfDrawnWin))
        {
            return m.action;
        }
        self.moves
            .iter()
            .filter(|m| m.action != self.baseline && m.diff > z * m.diff_se)
            .max_by(|a, b| a.mean.total_cmp(&b.mean))
            .map_or(self.baseline, |m| m.action)
    }
}

/// One rollout's result for the acting player.
#[derive(Clone, Copy, Default)]
struct Sample {
    points: f64,
    win: bool,
    deal_in: bool,
    draw: bool,
}

impl Sample {
    fn of(state: &State, seat: usize) -> Sample {
        let r = state.result().expect("finished");
        Sample {
            points: r.deltas[seat] as f64,
            win: r.wins.iter().any(|w| w.seat == seat),
            deal_in: r.wins.iter().any(|w| !w.self_drawn && w.paid_by[seat] > 0),
            draw: r.wins.is_empty(),
        }
    }
}

/// The rollout policy: [`Fast`] for `seat`, a defending [`Fast`] for the others.
#[derive(Clone, Debug)]
pub struct Rollout {
    seat: usize,
    own: Bot,
    opponents: Bot,
}

impl Rollout {
    /// Rollouts from `seat`'s point of view.
    pub fn for_seat(seat: usize) -> Rollout {
        Rollout {
            seat,
            own: Bot::Fast(Fast::default()),
            opponents: Bot::Fast(Fast::new(1.0)),
        }
    }
}

impl Policy<State> for Rollout {
    fn choose<R: rand::Rng + ?Sized>(
        &self,
        state: &State,
        legal: &[Action],
        rng: &mut R,
    ) -> Action {
        if state.current_player() == Some(self.seat) {
            self.own.choose(state, legal, rng)
        } else {
            self.opponents.choose(state, legal, rng)
        }
    }
}

/// The candidates PIMC evaluates in `state` (see the module docs), and the
/// baseline action among them.
pub fn shortlist(state: &State, discards: usize) -> (Vec<Action>, Action) {
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
    let baseline = Value::default().choose(state, &legal, &mut SmallRng::seed_from_u64(0));
    if !out.contains(&baseline) {
        out.push(baseline);
    }
    (out, baseline)
}

/// Runs PIMC for the player to act.
///
/// # Panics
/// Panics if the hand is over.
pub fn pimc(state: &State, config: &PimcConfig) -> Analysis {
    let (candidates, baseline) = shortlist(state, config.candidates.max(1));
    let seat = state.current_player().expect("hand in progress");
    pimc_with(
        state,
        config,
        candidates,
        baseline,
        &Rollout::for_seat(seat),
    )
}

/// PIMC with given candidates, baseline (one of the candidates) and rollout
/// policy; [`pimc`] uses the heuristic shortlist and [`Rollout`].
///
/// # Panics
/// Panics if the hand is over or `baseline` is not among `candidates`.
pub fn pimc_with<P: Policy<State>>(
    state: &State,
    config: &PimcConfig,
    candidates: Vec<Action>,
    baseline: Action,
    rollout: &P,
) -> Analysis {
    let start = Instant::now();
    let seat = state.current_player().expect("hand in progress");
    let base = candidates
        .iter()
        .position(|&a| a == baseline)
        .expect("baseline is a candidate");
    let mut samples: Vec<Vec<Sample>> = Vec::new();
    let batch = rayon::current_num_threads().max(1) * 4;
    while samples.len() < config.samples {
        let from = samples.len();
        let n = batch.min(config.samples - from);
        let results: Vec<Vec<Sample>> = (from..from + n)
            .into_par_iter()
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(
                    config.seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let world = sample_world(state, seat, &mut rng);
                candidates
                    .iter()
                    .map(|&a| {
                        let mut s = world.clone();
                        s.apply(a);
                        play_out(&mut s, rollout, &mut rng);
                        Sample::of(&s, seat)
                    })
                    .collect()
            })
            .collect();
        samples.extend(results);
        if config
            .time_limit
            .is_some_and(|limit| start.elapsed() >= limit)
        {
            break;
        }
    }
    let n = samples.len().max(1) as f64;
    let mut moves: Vec<ActionStats> = candidates
        .iter()
        .enumerate()
        .map(|(c, &action)| {
            let pts: Vec<f64> = samples.iter().map(|s| s[c].points).collect();
            let diffs: Vec<f64> = samples
                .iter()
                .map(|s| s[c].points - s[base].points)
                .collect();
            let (mean, se) = mean_se(&pts);
            let (diff, diff_se) = mean_se(&diffs);
            let rate =
                |f: fn(&Sample) -> bool| samples.iter().filter(|s| f(&s[c])).count() as f64 / n;
            ActionStats {
                action,
                n: samples.len() as u32,
                mean,
                ci95: 1.96 * se,
                win_rate: rate(|s| s.win),
                deal_in_rate: rate(|s| s.deal_in),
                draw_rate: rate(|s| s.draw),
                diff,
                diff_se,
            }
        })
        .collect();
    moves.sort_by(|a, b| b.mean.total_cmp(&a.mean));
    Analysis {
        seat,
        moves,
        baseline,
        samples: samples.len(),
        elapsed: start.elapsed(),
    }
}

/// Mean and standard error of the mean.
fn mean_se(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}
