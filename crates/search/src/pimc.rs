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
//! **Anchor and rollout policy.** [`pimc_anchored`] replaces the [`Value`]
//! baseline with another bot's choice (the *anchor*, e.g. the `ev` bot,
//! [`shortlist_anchored`]) and the player's own rollout moves with another bot
//! ([`Rollout::with`]); the opponents keep the defending `Fast`. The `ev` bot in
//! the rollouts costs about 0.4–0.7 ms per decision against `Fast`'s few
//! microseconds, so such searches are many times slower per sample.
//!
//! [`Analysis::choose`] turns this into a move: always declare a legal win;
//! otherwise play the candidate with the best mean only if it beats the baseline
//! by more than `z` standard errors of the paired difference, else the baseline.
//! With few samples the search therefore defers to the heuristic instead of
//! acting on noise.
//!
//! Like all PIMC methods it lets each sample's future be played as if everyone
//! knew it ("strategy fusion"). Its numbers are estimates under the rollout bot's
//! style of play, not equilibrium values. With the default sampler it infers
//! little from how opponents have played (beyond how developed hands typically
//! are at this stage); [`pimc_sampled`] and [`pimc_with_sampler`] can instead
//! draw opponents' hands from the trained belief model
//! ([`WorldSampler::Belief`]). The sampler is a separate argument rather than a
//! [`PimcConfig`] field because the app's engine builds `PimcConfig` with every
//! field named, so a new field would break it.
//!
//! **Weighted samples.** When the sampler is weighted, each determinization
//! carries an importance weight; weights are clipped at their 95th percentile
//! (so one sample cannot dominate) and every mean, rate and paired difference
//! is a weighted one. Standard errors use the effective sample size
//! `(Σw)² / Σw²`. Weights are computed after all samples are in, in sample
//! order, so the result is the same on one thread as on many.
//!
//! **Leaf value** (plan Task 16, `netpimc:...,leaf=value,depth=6`).
//! [`pimc_with_leaf`] stops each rollout after `depth` discards (any seat's) and
//! scores the position with a network's four-seat value head instead of playing
//! to the end ([`LeafValue`]). The value head predicts each seat's final points,
//! log-scaled as in training (`sign(x) · ln(1 + |x| / 8)`); the leaf undoes the
//! scaling. A rollout that reaches the end of the hand before the leaf is scored
//! by its real result. Win, deal-in and draw rates count only the rollouts that
//! ended before their leaf, so they are lower bounds under `leaf=value`.

use crate::bots::{best_discard, Bot, Fast, Metric, Value};
use crate::net::Mlp;
use crate::view::SeatView;
use crate::world::sample_world_from;
pub use crate::world::WorldSampler;
use mahjong_games::{play_out, Game, Policy};
use mahjong_hk::{Action, State};
use mahjong_tiles::Tile;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Duration;
// `std::time::Instant` panics in the browser; `web_time` is `Instant` on native.
use web_time::Instant;

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

/// Scale of the value head's log transform (idea 5d; `labels.value_targets`).
pub const VALUE_LOG_SCALE: f64 = 8.0;

/// Leaf evaluation for [`pimc_with_leaf`]: after `depth` discards, the
/// network's `value` output for the searching seat replaces the rest of the
/// rollout.
#[derive(Clone, Debug)]
pub struct LeafValue {
    pub net: Arc<Mlp>,
    /// Discards (by any seat) played in the rollout before the leaf.
    pub depth: usize,
}

impl LeafValue {
    /// A leaf evaluator; `Err` if `net` has no four-wide `value` output.
    pub fn new(net: Arc<Mlp>, depth: usize) -> Result<LeafValue, String> {
        match net.output("value") {
            Some(r) if r.len() == 4 => Ok(LeafValue { net, depth }),
            _ => Err("leaf=value needs a network with a 4-wide `value` output (MJNN v3)".into()),
        }
    }

    /// The expected final points of `seat` in `state` (a hand in progress),
    /// read from the value head of the player to act (relative seats).
    ///
    /// # Panics
    /// Panics if the hand is over.
    pub fn points(&self, state: &State, seat: usize) -> f64 {
        let me = state.current_player().expect("hand in progress");
        let out = self.net.forward_state(state);
        let range = self.net.output("value").expect("checked in new");
        let y = out[range.start + (seat + 4 - me) % 4] as f64;
        y.signum() * VALUE_LOG_SCALE * y.abs().exp_m1()
    }

    /// Plays `state` with `rollout` until the hand ends or `depth` discards
    /// have been made, then scores it for `seat`.
    fn play<P: Policy<State>, R: rand::Rng + ?Sized>(
        &self,
        state: &mut State,
        rollout: &P,
        rng: &mut R,
        seat: usize,
    ) -> Sample {
        let mut legal = Vec::new();
        let mut discards = 0;
        while state.current_player().is_some() {
            if discards >= self.depth {
                return Sample {
                    points: self.points(state, seat),
                    ..Sample::default()
                };
            }
            state.legal_actions(&mut legal);
            let a = rollout.choose(state, &legal, rng);
            if matches!(a, Action::Discard(_)) {
                discards += 1;
            }
            state.apply(a);
        }
        Sample::of(state, seat)
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
            opponents: Rollout::default_opponents(),
        }
    }

    /// Rollouts from `seat`'s point of view with `own` playing `seat`'s later
    /// moves and `opponents` the other three seats (`pimc:...,rollout=ev`
    /// passes the `ev` bot as `own`).
    pub fn with(seat: usize, own: Bot, opponents: Bot) -> Rollout {
        Rollout {
            seat,
            own,
            opponents,
        }
    }

    /// The opponents' policy of [`Rollout::for_seat`]: a `Fast` bot that
    /// folds against threats.
    pub fn default_opponents() -> Bot {
        Bot::Fast(Fast::new(1.0))
    }

    /// The same policies seen from another seat.
    pub fn for_other_seat(&self, seat: usize) -> Rollout {
        Rollout {
            seat,
            ..self.clone()
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
    let baseline = Value::default().choose(state, &legal, &mut SmallRng::seed_from_u64(0));
    shortlist_around(state, &legal, discards, baseline)
}

/// [`shortlist`] with `anchor`'s choice as the baseline instead of the
/// [`Value`] bot's (`pimc:...,anchor=ev`). The anchor's choice is added to the
/// candidates when the faan-aware measure did not pick it.
pub fn shortlist_anchored(state: &State, discards: usize, anchor: &Bot) -> (Vec<Action>, Action) {
    let legal = state.legal_actions_vec();
    let baseline = anchor.choose(state, &legal, &mut SmallRng::seed_from_u64(0));
    shortlist_around(state, &legal, discards, baseline)
}

/// The special actions, the best `discards` discards by the faan-aware
/// measure, and `baseline` (added if missing).
fn shortlist_around(
    state: &State,
    legal: &[Action],
    discards: usize,
    baseline: Action,
) -> (Vec<Action>, Action) {
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
    pimc_sampled(state, config, &WorldSampler::BestOfK)
}

/// [`pimc`] with determinizations drawn by `sampler`.
///
/// # Panics
/// Panics if the hand is over.
pub fn pimc_sampled(state: &State, config: &PimcConfig, sampler: &WorldSampler) -> Analysis {
    let (candidates, baseline) = shortlist(state, config.candidates.max(1));
    let seat = state.current_player().expect("hand in progress");
    pimc_with_sampler(
        state,
        config,
        sampler,
        candidates,
        baseline,
        &Rollout::for_seat(seat),
    )
}

/// [`pimc_sampled`] with `anchor`'s choice as the baseline
/// ([`shortlist_anchored`]) and `rollout`'s policies (seen from the player to
/// act). With `anchor = value` and [`Rollout::for_seat`]'s policies it is
/// [`pimc_sampled`].
///
/// # Panics
/// Panics if the hand is over.
pub fn pimc_anchored(
    state: &State,
    config: &PimcConfig,
    sampler: &WorldSampler,
    anchor: &Bot,
    rollout: &Rollout,
) -> Analysis {
    let (candidates, baseline) = shortlist_anchored(state, config.candidates.max(1), anchor);
    let seat = state.current_player().expect("hand in progress");
    pimc_with_sampler(
        state,
        config,
        sampler,
        candidates,
        baseline,
        &rollout.for_other_seat(seat),
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
    pimc_with_sampler(
        state,
        config,
        &WorldSampler::BestOfK,
        candidates,
        baseline,
        rollout,
    )
}

/// [`pimc_with`] with determinizations drawn by `sampler`.
///
/// # Panics
/// Panics if the hand is over or `baseline` is not among `candidates`.
pub fn pimc_with_sampler<P: Policy<State>>(
    state: &State,
    config: &PimcConfig,
    sampler: &WorldSampler,
    candidates: Vec<Action>,
    baseline: Action,
    rollout: &P,
) -> Analysis {
    pimc_with_leaf(state, config, sampler, candidates, baseline, rollout, None)
}

/// [`pimc_with_sampler`] whose rollouts stop at `leaf`'s depth and take the
/// network's value there (see the module docs); `None` plays every rollout to
/// the end, exactly as [`pimc_with_sampler`].
///
/// # Panics
/// Panics if the hand is over or `baseline` is not among `candidates`.
pub fn pimc_with_leaf<P: Policy<State>>(
    state: &State,
    config: &PimcConfig,
    sampler: &WorldSampler,
    candidates: Vec<Action>,
    baseline: Action,
    rollout: &P,
    leaf: Option<&LeafValue>,
) -> Analysis {
    let start = Instant::now();
    let seat = state.current_player().expect("hand in progress");
    let base = candidates
        .iter()
        .position(|&a| a == baseline)
        .expect("baseline is a candidate");
    let mut samples: Vec<Vec<Sample>> = Vec::new();
    let mut log_weights: Vec<f64> = Vec::new();
    let batch = rayon::current_num_threads().max(1) * 4;
    while samples.len() < config.samples {
        let from = samples.len();
        let n = batch.min(config.samples - from);
        let results: Vec<(f64, Vec<Sample>)> = (from..from + n)
            .into_par_iter()
            .map(|i| {
                let mut rng = SmallRng::seed_from_u64(
                    config.seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                let (world, log_w) = sample_world_from(state, seat, sampler, &mut rng);
                let row = candidates
                    .iter()
                    .map(|&a| {
                        let mut s = world.clone();
                        s.apply(a);
                        match leaf {
                            Some(l) => l.play(&mut s, rollout, &mut rng, seat),
                            None => {
                                play_out(&mut s, rollout, &mut rng);
                                Sample::of(&s, seat)
                            }
                        }
                    })
                    .collect();
                (log_w, row)
            })
            .collect();
        for (log_w, row) in results {
            log_weights.push(log_w);
            samples.push(row);
        }
        if config
            .time_limit
            .is_some_and(|limit| start.elapsed() >= limit)
        {
            break;
        }
    }
    // `None`: every sample counts once (the unweighted arithmetic, unchanged).
    let weights = sampler.weighted().then(|| clipped_weights(&log_weights));
    let stat = |xs: &[f64]| match &weights {
        Some(w) => weighted_mean_se(xs, w),
        None => mean_se(xs),
    };
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
            let (mean, se) = stat(&pts);
            let (diff, diff_se) = stat(&diffs);
            let rate = |f: fn(&Sample) -> bool| match &weights {
                Some(w) => {
                    let total: f64 = w.iter().sum();
                    samples
                        .iter()
                        .zip(w)
                        .filter(|(s, _)| f(&s[c]))
                        .map(|(_, w)| w)
                        .sum::<f64>()
                        / total
                }
                None => samples.iter().filter(|s| f(&s[c])).count() as f64 / n,
            };
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

/// Importance weights from their logarithms: scaled so the largest is 1, then
/// clipped at the 95th percentile so one sample cannot dominate.
pub fn clipped_weights(log_weights: &[f64]) -> Vec<f64> {
    if log_weights.is_empty() {
        return Vec::new();
    }
    let top = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut w: Vec<f64> = log_weights.iter().map(|lw| (lw - top).exp()).collect();
    let mut sorted = w.clone();
    sorted.sort_by(f64::total_cmp);
    let cap = sorted[(sorted.len() * 95).div_ceil(100).saturating_sub(1)];
    for x in &mut w {
        *x = x.min(cap);
    }
    w
}

/// Weighted mean and its standard error, using the effective sample size
/// `(Σw)² / Σw²` (equal weights give [`mean_se`]).
fn weighted_mean_se(xs: &[f64], w: &[f64]) -> (f64, f64) {
    let total: f64 = w.iter().sum();
    if xs.is_empty() || total <= 0.0 {
        return mean_se(xs);
    }
    let mean = xs.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / total;
    let var = xs
        .iter()
        .zip(w)
        .map(|(x, w)| w * (x - mean).powi(2))
        .sum::<f64>()
        / total;
    let n_eff = total * total / w.iter().map(|w| w * w).sum::<f64>();
    (mean, (var / (n_eff - 1.0).max(1.0)).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::BeliefModel;
    use mahjong_hk::RuleSet;
    use std::path::Path;
    use std::sync::Arc;

    #[test]
    fn weighted_statistics_reduce_to_the_unweighted_ones() {
        let xs = [1.0, -3.0, 8.0, 0.5, 2.0];
        let (m, se) = mean_se(&xs);
        let (wm, wse) = weighted_mean_se(&xs, &[0.4; 5]);
        assert!((m - wm).abs() < 1e-12);
        assert!((se - wse).abs() < 1e-12);
        // A heavier weight pulls the mean towards its sample.
        let (pulled, _) = weighted_mean_se(&xs, &[1.0, 1.0, 5.0, 1.0, 1.0]);
        assert!(pulled > m);
    }

    #[test]
    fn weights_are_clipped_at_the_95th_percentile() {
        // One huge weight among 40 ordinary ones is cut to the 95th percentile.
        let mut lw = vec![0.0; 39];
        lw.push(50.0);
        let w = clipped_weights(&lw);
        assert_eq!(w.len(), 40);
        assert!(w.iter().all(|&x| x <= w[0] + 1e-12));
        let (mean, _) = weighted_mean_se(
            &(0..40)
                .map(|i| if i == 39 { 100.0 } else { 0.0 })
                .collect::<Vec<_>>(),
            &w,
        );
        assert!(mean < 5.0, "the outlier dominated: {mean}");
        assert!(clipped_weights(&[]).is_empty());
    }

    fn position(seed: u64, steps: usize) -> State {
        let bot = Bot::by_name("fast").unwrap();
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        for _ in 0..steps {
            if s.is_terminal() {
                break;
            }
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
        }
        s
    }

    #[test]
    fn belief_pimc_is_the_same_on_one_thread() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief.mjnn");
        if !p.exists() {
            eprintln!("models/belief.mjnn missing: skipped");
            return;
        }
        let model = Arc::new(BeliefModel::load(&p).unwrap());
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let mut checked = 0;
        for weights in [true, false] {
            let config = PimcConfig {
                samples: 24,
                candidates: 3,
                seed: 5,
                ..PimcConfig::default()
            };
            let sampler = WorldSampler::belief(Arc::clone(&model), weights);
            for (seed, steps) in [(1, 20), (2, 45)] {
                let s = position(seed, steps);
                if s.is_terminal() {
                    continue;
                }
                let many = pimc_sampled(&s, &config, &sampler);
                let single = one.install(|| pimc_sampled(&s, &config, &sampler));
                let key = |a: &Analysis| {
                    a.moves
                        .iter()
                        .map(|m| {
                            (
                                m.action,
                                m.mean.to_bits(),
                                m.diff.to_bits(),
                                m.diff_se.to_bits(),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(key(&single), key(&many), "weights {weights}, seed {seed}");
                for m in &many.moves {
                    assert!(m.mean.is_finite() && m.diff_se.is_finite());
                    assert!((0.0..=1.0).contains(&m.win_rate));
                }
                checked += 1;
            }
        }
        assert!(checked >= 2);
    }

    #[test]
    fn value_anchor_with_the_default_rollout_is_plain_pimc() {
        let config = PimcConfig {
            samples: 16,
            candidates: 3,
            seed: 9,
            ..PimcConfig::default()
        };
        let anchor = Bot::Value(Value::default());
        for (seed, steps) in [(3, 18), (4, 40)] {
            let s = position(seed, steps);
            if s.is_terminal() {
                continue;
            }
            assert_eq!(shortlist(&s, 3), shortlist_anchored(&s, 3, &anchor));
            let plain = pimc_sampled(&s, &config, &WorldSampler::BestOfK);
            // The rollout's seat is replaced by the player to act.
            let anchored = pimc_anchored(
                &s,
                &config,
                &WorldSampler::BestOfK,
                &anchor,
                &Rollout::for_seat(0),
            );
            let key = |a: &Analysis| {
                a.moves
                    .iter()
                    .map(|m| (m.action, m.mean.to_bits(), m.diff.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(key(&plain), key(&anchored), "seed {seed}");
            assert_eq!(plain.baseline, anchored.baseline);
        }
    }

    #[test]
    fn ev_anchor_and_rollout_are_the_same_on_one_thread() {
        use crate::ev::EvBot;
        let ev = Bot::Ev(EvBot::default());
        let config = PimcConfig {
            samples: 8,
            candidates: 2,
            seed: 3,
            ..PimcConfig::default()
        };
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let mut checked = 0;
        for (seed, steps) in [(5, 22), (6, 38)] {
            let s = position(seed, steps);
            if s.is_terminal() {
                continue;
            }
            let seat = s.current_player().unwrap();
            let rollout = Rollout::with(seat, ev.clone(), Rollout::default_opponents());
            let (list, baseline) = shortlist_anchored(&s, 2, &ev);
            let legal = s.legal_actions_vec();
            assert_eq!(
                baseline,
                ev.choose(&s, &legal, &mut SmallRng::seed_from_u64(1))
            );
            assert!(list.contains(&baseline));
            let sampler = WorldSampler::BestOfK;
            let many = pimc_anchored(&s, &config, &sampler, &ev, &rollout);
            let single = one.install(|| pimc_anchored(&s, &config, &sampler, &ev, &rollout));
            let key = |a: &Analysis| {
                a.moves
                    .iter()
                    .map(|m| (m.action, m.mean.to_bits(), m.diff.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(key(&single), key(&many), "seed {seed}");
            assert_eq!(many.baseline, baseline);
            checked += 1;
        }
        assert!(checked >= 1);
    }
}
