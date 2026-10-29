//! Sampling the hidden tiles for search: **structured determinization**.
//!
//! Re-dealing the hidden tiles uniformly at random (what
//! [`mahjong_games::Game::determinize`] does) gives opponents hands that are much
//! less developed than real ones: by mid-hand a real player has thrown away their
//! useless tiles and kept the useful ones. Measured with `value` bots
//! (`cargo run --release -p mahjong-search --example calibrate`), opponents' mean
//! shanten is
//!
//! | Tiles drawn from the wall | Real | Uniform | Best of K (K below) |
//! |---|---|---|---|
//! | 0–20 | 2.70 | 3.55 | 2.69 (K = 4) |
//! | 21–41 | 1.54 | 3.24 | 1.36–1.78 (K = 16–64) |
//! | 42–62 | 1.09 | 2.97 | 1.17 (K = 64) |
//! | 63–83 | 0.77 | 2.54 | 0.56–0.88 (K = 64–256) |
//!
//! With uniform hands a searcher believes nobody else is close to winning, so it
//! ignores defense; in tests it won 2.5% of hands against `value` bots while its
//! own rollouts predicted 50–70%. Here each opponent's hand is instead the best
//! (lowest shanten) of `K` random hands drawn from the hidden tiles, with
//! `K = 4 + used² / 40` (`used` = tiles drawn from the wall so far), which tracks
//! the real means above. It still uses only the observer's information.
//!
//! **Honours.** Players throw lone honours early, so an honour nobody has
//! discarded is more often held than a uniform deal suggests. Measured with the
//! `honours` example, the share of the observer's unseen honour copies that sit in
//! opponents' hands is
//!
//! | Tiles drawn | Real | Best of K | Best of K, honours × 1.6 |
//! |---|---|---|---|
//! | 0–22 | 33% | 25% | 37% |
//! | 23–45 | 39% | 25% | 38% |
//! | 46–68 | 50% | 34% | 48% |
//! | 69–91 | 71% | 60% | 72% |
//!
//! so candidate hands draw honours [`HONOUR_WEIGHT`] times as likely as suited
//! tiles. Without this the solver expected to pung dragons that opponents were in
//! fact holding, and advised breaking ready hands to chase them.
//!
//! **Belief sampling** (plan Task 7, [`WorldSampler::Belief`]). The trained belief
//! model ([`crate::belief`]) gives, per opponent and tile kind, a distribution
//! `p(k)` over holding `k = 0..4` copies, read from the public state (discards,
//! melds, timing). Its target for a determinization is `L = Π_o Π_t p_o,t(h_o,t)`
//! over the opponents' hands `h_o`, restricted to the hidden tiles. Each opponent
//! in turn (next seat first) draws a hand of exactly its size from the tiles still
//! hidden, every copy of kind `t` with odds `m_t / (a_t − m_t)`, where `m_t` is the
//! belief's expected count and `a_t` the copies left (Fisher's noncentral
//! hypergeometric: "proportional to the expected count", but with an exact
//! density). The draw is exact via a small table over kinds × tiles so far.
//!
//! The belief's per-kind marginals are independent, so such hands lack the pairs
//! and runs real hands have (about +0.3 to +0.8 shanten against real play). Each
//! opponent therefore gets `draws` candidates and keeps one with probability
//! proportional to `exp(−tilt × shanten)`, times the **importance weight**
//! `L / q` (belief likelihood over proposal density) when weights are on
//! (sampling-importance-resampling). The tilt ramps up from 0 at the deal
//! ([`tilt_ramp`]); its defaults ([`default_tilt`]) make sampled shanten match
//! real play. With weights on, the determinization's weight is the mean
//! candidate weight, returned as a logarithm; PIMC clips it at the 95th
//! percentile so one sample cannot dominate ([`crate::pimc`]). With weights off
//! the weight is 0 (all samples count equally). The rest of the hidden tiles
//! become the wall, shuffled. Like best-of-K it reads only what the observer can
//! see.

use crate::belief::{BeliefModel, BeliefOutput, MAX_COPIES};
use mahjong_analysis::shanten;
use mahjong_games::Game;
use mahjong_hk::State;
use mahjong_tiles::{Tile, TileCounts, NUM_KINDS};
use rand::seq::SliceRandom;
use rand::Rng;
use std::sync::Arc;

/// How PIMC samples the hidden tiles (see the module docs).
#[derive(Clone, Debug, Default)]
pub enum WorldSampler {
    /// Best of K random hands by shanten, honours weighted ([`sample_world`]).
    #[default]
    BestOfK,
    /// Hands drawn from the belief model's expected counts, `draws` per opponent,
    /// one kept by resampling with the shanten tilt `exp(−tilt × shanten)`; with
    /// `weights`, the resampling also uses the belief likelihood and each
    /// determinization carries its importance weight.
    Belief {
        model: Arc<BeliefModel>,
        weights: bool,
        draws: usize,
        tilt: f64,
    },
}

/// Candidate hands per opponent for the belief sampler (default).
pub const BELIEF_DRAWS: usize = 16;

/// The belief sampler's default shanten tilt, with and without importance
/// weights (fitted with the `calibrate` example so sampled opponents' mean
/// shanten matches real play, `docs/solver.md` §4). The tilt grows from 0 at the
/// deal to its full value once a sixth of the wall is drawn ([`tilt_ramp`]):
/// dealt hands are random, and only play makes them more developed.
pub fn default_tilt(weights: bool) -> f64 {
    if weights {
        0.8
    } else {
        1.2
    }
}

/// The share of the tilt that applies at this point of the hand: 0 at the
/// deal, 1 once a sixth of the wall is drawn.
pub fn tilt_ramp(state: &State) -> f64 {
    let initial = initial_wall(state) as f64;
    let used = initial - (state.wall_remaining() as f64).min(initial);
    (6.0 * used / initial).min(1.0)
}

impl WorldSampler {
    /// The belief sampler with the default draws and tilt.
    pub fn belief(model: Arc<BeliefModel>, weights: bool) -> WorldSampler {
        WorldSampler::Belief {
            model,
            weights,
            draws: BELIEF_DRAWS,
            tilt: default_tilt(weights),
        }
    }

    /// Whether samples carry importance weights.
    pub fn weighted(&self) -> bool {
        matches!(self, WorldSampler::Belief { weights: true, .. })
    }
}

/// A determinization for `observer` drawn by `sampler`, and the logarithm of its
/// (unnormalised) importance weight: 0 unless the sampler is weighted.
///
/// The belief model describes the player to act, so when `observer` is not the
/// player to act the belief sampler falls back to best-of-K (weight 0).
pub fn sample_world_from<R: Rng + ?Sized>(
    state: &State,
    observer: usize,
    sampler: &WorldSampler,
    rng: &mut R,
) -> (State, f64) {
    match sampler {
        WorldSampler::Belief {
            model,
            weights,
            draws,
            tilt,
        } if !state.is_terminal() && state.current_player() == Some(observer) => {
            let belief = model.run(state);
            let opts = BeliefDraw {
                weights: *weights,
                draws: (*draws).max(1),
                tilt: *tilt * tilt_ramp(state),
            };
            belief_world(state, observer, &belief, opts, rng)
        }
        _ => (sample_world(state, observer, rng), 0.0),
    }
}

/// Floor for belief probabilities inside logarithms.
const MIN_P: f64 = 1e-12;
/// How far an expected count is kept from 0 and from the copies left, so the
/// per-copy odds stay finite and positive.
const COUNT_MARGIN: f64 = 1e-4;

/// Hand sizes per seat never exceed this (14 concealed tiles).
const MAX_HAND: usize = 14;

/// Draws per-kind counts summing to exactly `n`, with P(counts) proportional to
/// `Π_t g[t][counts_t]` (the table is built once and drawn from many times).
struct FixedTotal {
    g: [[f64; MAX_COPIES]; NUM_KINDS],
    /// `suffix[i][s]`: total weight of kinds `i..` holding exactly `s` tiles.
    suffix: [[f64; MAX_HAND + 1]; NUM_KINDS + 1],
    n: usize,
}

impl FixedTotal {
    /// `g[t][k]` must be 0 for impossible `k`.
    fn new(g: [[f64; MAX_COPIES]; NUM_KINDS], n: usize) -> FixedTotal {
        assert!(n <= MAX_HAND, "a hand of {n} tiles");
        let mut suffix = [[0.0f64; MAX_HAND + 1]; NUM_KINDS + 1];
        suffix[NUM_KINDS][0] = 1.0;
        for i in (0..NUM_KINDS).rev() {
            for s in 0..=n {
                suffix[i][s] = (0..MAX_COPIES.min(s + 1))
                    .map(|k| g[i][k] * suffix[i + 1][s - k])
                    .sum();
            }
        }
        assert!(
            suffix[0][n] > 0.0,
            "no hand of {n} tiles fits the hidden tiles"
        );
        FixedTotal { g, suffix, n }
    }

    /// One draw and the log of its probability.
    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> ([u8; NUM_KINDS], f64) {
        let mut counts = [0u8; NUM_KINDS];
        let mut left = self.n;
        let mut log_q = -self.suffix[0][self.n].ln();
        for (i, count) in counts.iter_mut().enumerate() {
            let mut options = [0.0f64; MAX_COPIES];
            for (k, o) in options
                .iter_mut()
                .enumerate()
                .take(left.min(MAX_COPIES - 1) + 1)
            {
                *o = self.g[i][k] * self.suffix[i + 1][left - k];
            }
            let sum: f64 = options.iter().sum();
            let mut x = rng.gen::<f64>() * sum;
            // The last option with positive weight catches rounding at the top.
            let mut k = options.iter().rposition(|&w| w > 0.0).expect("feasible");
            for (j, &w) in options.iter().enumerate() {
                if x < w {
                    k = j;
                    break;
                }
                x -= w;
            }
            *count = k as u8;
            log_q += self.g[i][k].ln();
            left -= k;
        }
        (counts, log_q)
    }
}

/// Binomial coefficient for small arguments (`n ≤ 4`).
fn choose(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// The belief sampler's settings for one determinization.
#[derive(Clone, Copy)]
struct BeliefDraw {
    weights: bool,
    draws: usize,
    tilt: f64,
}

/// Index drawn with probability proportional to `exp(log_w[i])`, and the log of
/// the mean of those weights.
fn resample<R: Rng + ?Sized>(log_w: &[f64], rng: &mut R) -> (usize, f64) {
    let top = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = log_w.iter().map(|x| (x - top).exp()).collect();
    let sum: f64 = w.iter().sum();
    let mut x = rng.gen::<f64>() * sum;
    let mut pick = w.len() - 1;
    for (i, &wi) in w.iter().enumerate() {
        if x < wi {
            pick = i;
            break;
        }
        x -= wi;
    }
    (pick, top + (sum / w.len() as f64).ln())
}

/// The belief sampler (see the module docs).
///
/// Per opponent: `draws` candidate hands from the expected-count proposal `q`;
/// each gets the resampling weight `exp(−tilt × shanten)`, times `L / q` when
/// `weights` is on; one is kept with probability proportional to its weight
/// (sampling-importance-resampling). With weights on, the determinization's log
/// weight adds the log of the mean candidate weight, which makes the kept hands
/// properly weighted for the target `L × exp(−tilt × shanten)`.
fn belief_world<R: Rng + ?Sized>(
    state: &State,
    observer: usize,
    belief: &BeliefOutput,
    opts: BeliefDraw,
    rng: &mut R,
) -> (State, f64) {
    let (pool, sizes) = state.hidden_pool(observer);
    let (playing, mut bonus): (Vec<Tile>, Vec<Tile>) =
        pool.into_iter().partition(|t| !t.is_bonus());
    let mut left = [0usize; NUM_KINDS];
    for t in &playing {
        left[t.index() as usize] += 1;
    }
    let mut hands = [TileCounts::new(); 4];
    let mut log_w = 0.0;
    for seat in (1..4).map(|r| (observer + r) % 4) {
        let b = belief.seat(seat).expect("an opponent");
        let melds = state.melds(seat).len();
        let mut g = [[0.0f64; MAX_COPIES]; NUM_KINDS];
        for (i, gi) in g.iter_mut().enumerate() {
            let a = left[i].min(MAX_COPIES - 1);
            if a == 0 {
                gi[0] = 1.0;
                continue;
            }
            // Expected count among what is still possible (an earlier opponent
            // may have taken copies the belief gave this one).
            let p = &b.counts[i][..=a];
            let mass: f64 = p.iter().sum();
            let m = if mass > 0.0 {
                p.iter().enumerate().map(|(k, q)| k as f64 * q).sum::<f64>() / mass
            } else {
                0.0
            };
            let m = m.clamp(COUNT_MARGIN, a as f64 - COUNT_MARGIN);
            let odds = m / (a as f64 - m);
            for (k, w) in gi.iter_mut().enumerate().take(a + 1) {
                *w = choose(left[i], k) * odds.powi(k as i32);
            }
            // Scaling a kind's row changes no probability; it keeps the table's
            // products far from overflow.
            let top = gi.iter().copied().fold(0.0, f64::max);
            gi.iter_mut().for_each(|w| *w /= top);
        }
        let table = FixedTotal::new(g, sizes[seat]);
        let mut candidates = Vec::with_capacity(opts.draws);
        let mut cand_log_w = Vec::with_capacity(opts.draws);
        for _ in 0..opts.draws {
            let (counts, log_q) = table.draw(rng);
            let mut hand = TileCounts::new();
            for (i, &k) in counts.iter().enumerate() {
                for _ in 0..k {
                    hand.add(Tile::from_index(i as u8));
                }
            }
            let mut lw = -opts.tilt * shanten(&hand, melds) as f64;
            if opts.weights {
                let log_l: f64 = counts
                    .iter()
                    .enumerate()
                    .map(|(i, &k)| b.counts[i][k as usize].max(MIN_P).ln())
                    .sum();
                lw += log_l - log_q;
            }
            candidates.push((counts, hand));
            cand_log_w.push(lw);
        }
        let (pick, log_mean) = resample(&cand_log_w, rng);
        if opts.weights {
            log_w += log_mean;
        }
        let (counts, hand) = candidates.swap_remove(pick);
        for (i, &k) in counts.iter().enumerate() {
            left[i] -= k as usize;
        }
        hands[seat] = hand;
    }
    let mut wall: Vec<Tile> = (0..NUM_KINDS)
        .flat_map(|i| std::iter::repeat_n(Tile::from_index(i as u8), left[i]))
        .collect();
    wall.append(&mut bonus);
    wall.shuffle(rng);
    (state.redeal(observer, &hands, wall), log_w)
}

/// How much more likely an honour is to be in an opponent's hand than a suited
/// tile, when drawing candidate hands (see the module docs).
pub const HONOUR_WEIGHT: f64 = 1.6;

/// Tiles in the wall after the deal (before flower replacement).
fn initial_wall(state: &State) -> usize {
    if state.rules().flowers {
        144 - 53
    } else {
        136 - 53
    }
}

/// The number of candidate hands per opponent at this point of the hand.
pub fn candidates_per_opponent(state: &State) -> usize {
    let used = initial_wall(state).saturating_sub(state.wall_remaining()) as f64;
    (4.0 + used * used / 40.0).min(200.0) as usize
}

/// Draws `n` tiles from `pool` without replacement, honours `honour_weight`
/// times as likely as suited tiles (Efraimidis–Spirakis keys).
fn weighted_draw<R: Rng + ?Sized>(
    pool: &[Tile],
    n: usize,
    honour_weight: f64,
    rng: &mut R,
) -> Vec<Tile> {
    let mut keyed: Vec<(f64, Tile)> = pool
        .iter()
        .map(|&t| {
            let w = if t.is_honour() { honour_weight } else { 1.0 };
            (rng.gen::<f64>().powf(1.0 / w), t)
        })
        .collect();
    let k = n.min(keyed.len().saturating_sub(1));
    keyed.select_nth_unstable_by(k, |a, b| b.0.total_cmp(&a.0));
    keyed.into_iter().take(n).map(|(_, t)| t).collect()
}

/// A structured determinization for `observer` (see the module docs).
pub fn sample_world<R: Rng + ?Sized>(state: &State, observer: usize, rng: &mut R) -> State {
    sample_world_with(state, observer, HONOUR_WEIGHT, rng)
}

/// [`sample_world`] with a given weight for honour tiles in opponents' hands.
pub fn sample_world_with<R: Rng + ?Sized>(
    state: &State,
    observer: usize,
    honour_weight: f64,
    rng: &mut R,
) -> State {
    if state.is_terminal() {
        return state.clone();
    }
    let k = candidates_per_opponent(state);
    let (pool, sizes) = state.hidden_pool(observer);
    let (mut playing, mut bonus): (Vec<Tile>, Vec<Tile>) =
        pool.into_iter().partition(|t| !t.is_bonus());
    let mut hands = [TileCounts::new(); 4];
    for seat in (0..4).filter(|&p| p != observer) {
        let n = sizes[seat];
        let melds = state.melds(seat).len();
        let mut best: Option<(i8, Vec<Tile>)> = None;
        for _ in 0..k {
            let hand = if honour_weight == 1.0 {
                playing.partial_shuffle(rng, n).0.to_vec()
            } else {
                weighted_draw(&playing, n, honour_weight, rng)
            };
            let s = shanten(&TileCounts::from_tiles(&hand), melds);
            if best.as_ref().is_none_or(|(b, _)| s < *b) {
                best = Some((s, hand));
            }
        }
        let (_, chosen) = best.expect("k >= 1");
        for &t in &chosen {
            let i = playing.iter().position(|&x| x == t).expect("tile in pool");
            playing.swap_remove(i);
        }
        hands[seat] = TileCounts::from_tiles(&chosen);
    }
    let mut wall = playing;
    wall.append(&mut bonus);
    wall.shuffle(rng);
    state.redeal(observer, &hands, wall)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use mahjong_games::Policy;
    use mahjong_hk::RuleSet;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use std::path::Path;

    fn model() -> Option<Arc<BeliefModel>> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief.mjnn");
        p.exists().then(|| Arc::new(BeliefModel::load(&p).unwrap()))
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
    fn fixed_total_draws_have_the_right_size_and_density() {
        let mut g = [[0.0; MAX_COPIES]; NUM_KINDS];
        for row in g.iter_mut().take(3) {
            *row = [1.0, 2.0, 1.0, 0.0, 0.0];
        }
        for row in g.iter_mut().skip(3) {
            row[0] = 1.0;
        }
        // Totals of 2 over three kinds with weights 1, 2, 1 per kind: the exact
        // probability of (1, 1, 0) is 4 / Z with Z = 3 × 4 (two singles) + 3 × 1.
        let mut rng = SmallRng::seed_from_u64(1);
        let mut seen = 0;
        let table = FixedTotal::new(g, 2);
        for _ in 0..200 {
            let (c, log_q) = table.draw(&mut rng);
            assert_eq!(c.iter().map(|&k| k as usize).sum::<usize>(), 2);
            if c[..3] == [1, 1, 0] {
                assert!((log_q - (4.0f64 / 15.0).ln()).abs() < 1e-12);
                seen += 1;
            }
        }
        assert!(seen > 20);
    }

    #[test]
    fn belief_worlds_are_consistent_and_depend_only_on_the_observer_view() {
        let Some(m) = model() else {
            eprintln!("models/belief.mjnn missing: skipped");
            return;
        };
        for sampler in [
            WorldSampler::belief(Arc::clone(&m), true),
            WorldSampler::belief(Arc::clone(&m), false),
            WorldSampler::Belief {
                model: Arc::clone(&m),
                weights: true,
                draws: 1,
                tilt: 0.0,
            },
        ] {
            for (seed, steps) in [(1, 5), (2, 30), (3, 60), (4, 90)] {
                let s = position(seed, steps);
                let Some(p) = s.current_player() else {
                    continue;
                };
                let (w, log_w) =
                    sample_world_from(&s, p, &sampler, &mut SmallRng::seed_from_u64(3));
                assert!(log_w.is_finite());
                if !sampler.weighted() {
                    assert_eq!(log_w, 0.0);
                }
                // The observer's own view is unchanged; hand sizes are kept.
                assert_eq!(w.hand(p), s.hand(p));
                assert_eq!(w.wall_remaining(), s.wall_remaining());
                for o in 0..4 {
                    assert_eq!(w.concealed_count(o), s.concealed_count(o));
                }
                // Another arrangement of the same hidden tiles gives the same sample.
                let other = s.determinize(p, &mut SmallRng::seed_from_u64(99));
                let (w2, log_w2) =
                    sample_world_from(&other, p, &sampler, &mut SmallRng::seed_from_u64(3));
                assert_eq!(log_w, log_w2);
                for o in 0..4 {
                    assert_eq!(w.hand(o), w2.hand(o));
                }
            }
        }
    }
}
