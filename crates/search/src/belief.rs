//! The trained belief model in Rust (plan Tasks 5 and 6, `docs/belief.md`).
//!
//! [`BeliefModel`] loads an MJNN v2 file with the heads `counts`, `tenpai`,
//! `waits`, `faan` and `outcome`, runs it on the acting seat's public view and
//! turns the raw outputs into probabilities ([`BeliefOutput`]):
//!
//! * `counts`: per opponent and tile kind, a categorical over 0–4 copies, with
//!   copies beyond what the observer has not seen masked out (the same mask as
//!   `mask_counts` in `python/mahjong/belief/model.py`);
//! * `tenpai`, `waits`: sigmoids; `faan`, `outcome`: softmaxes.
//!
//! The network reads [`crate::features::encode`], which describes the decision of
//! the **player to act**, so a belief is always from that seat's point of view.
//! Results are cached per public state (a small ring keyed by a hash of what the
//! acting seat can see), so the danger model, the `ev` bot and a check that ask
//! about the same decision run the network once.
//!
//! The model implements [`crate::ev::Belief`], which is how the `ev` bot
//! (`ev:belief=models/belief.mjnn`), [`crate::Danger::from_belief`] and the
//! `defensive:belief=...` bot read it. No threads, no randomness: safe in the
//! one-thread (WebAssembly) path.

use crate::ev::{Belief, Beliefs, HeuristicBelief, SeatBelief, FAAN_BUCKETS};
use crate::net::Mlp;
use mahjong_games::Game;
use mahjong_hk::State;
use mahjong_tiles::{Tile, NUM_KINDS};
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Copies per tile kind in the `counts` head: 0, 1, 2, 3, 4.
pub const MAX_COPIES: usize = 5;
/// Opponents per decision (relative seats 1 = next, 2 = across, 3 = previous).
pub const OPPONENTS: usize = 3;
/// Outcome classes: no win, wins on a discard, self-draws (before the hand ends).
pub const OUTCOMES: usize = 3;
/// Entries kept in the per-state cache.
const CACHE_SIZE: usize = 64;

/// What the model says about one opponent, as probabilities.
#[derive(Clone, Debug, PartialEq)]
pub struct OpponentBelief {
    /// `counts[t][k]`: P(the opponent holds exactly `k` copies of kind `t`).
    pub counts: [[f64; MAX_COPIES]; NUM_KINDS],
    /// P(waiting with a legal win now).
    pub tenpai: f64,
    /// P(a discard of kind `t` wins now | tenpai), independent per kind.
    pub waits: [f64; NUM_KINDS],
    /// P(the best discard win now pays `3 + b` faan | tenpai), `b = 0..8`.
    pub faan: [f64; FAAN_BUCKETS],
    /// P(no win), P(wins on a discard), P(self-draws), before the hand ends.
    pub outcome: [f64; OUTCOMES],
}

/// The model's output for one decision: the acting seat's three opponents.
#[derive(Clone, Debug, PartialEq)]
pub struct BeliefOutput {
    /// The seat the belief is for (the player to act).
    pub me: usize,
    /// Relative seats 1, 2, 3 (index 0 = the next seat).
    pub opponents: [OpponentBelief; OPPONENTS],
}

impl BeliefOutput {
    /// The belief about absolute seat `seat` (`None` for `me`).
    pub fn seat(&self, seat: usize) -> Option<&OpponentBelief> {
        let r = (seat + 4 - self.me) % 4;
        (r != 0).then(|| &self.opponents[r - 1])
    }
}

/// A loaded belief model (see the module docs).
pub struct BeliefModel {
    net: Mlp,
    counts: Range<usize>,
    tenpai: Range<usize>,
    waits: Range<usize>,
    faan: Range<usize>,
    outcome: Range<usize>,
    cache: Mutex<Vec<(u64, Arc<BeliefOutput>)>>,
}

impl fmt::Debug for BeliefModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The weights are not worth printing.
        f.debug_struct("BeliefModel")
            .field("inputs", &self.net.input_len())
            .field("outputs", &self.net.output_len())
            .finish()
    }
}

/// Feeds formatted text straight into a hasher (no allocation).
struct HashWriter<'a, H: Hasher>(&'a mut H);

impl<H: Hasher> fmt::Write for HashWriter<'_, H> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

/// A hash of everything the player to act can see: the public event list, the
/// rules, the wall count, every seat's hand size and bonus tiles, and the
/// actor's own hand and drawn tile. Two states with the same fingerprint encode
/// to the same network input.
fn fingerprint(state: &State, me: usize) -> u64 {
    use fmt::Write as _;
    let mut h = DefaultHasher::new();
    me.hash(&mut h);
    state.phase_name().hash(&mut h);
    state.wall_remaining().hash(&mut h);
    for seat in 0..4 {
        state.concealed_count(seat).hash(&mut h);
        state.bonus(seat).hash(&mut h);
    }
    let hand = state.hand(me);
    for t in Tile::all_kinds() {
        hand[t].hash(&mut h);
    }
    let _ = write!(
        HashWriter(&mut h),
        "{:?}|{:?}|{:?}|{:?}",
        state.rules(),
        state.last_drawn(),
        state.claim_tile(),
        state.events()
    );
    h.finish()
}

fn sigmoid(x: f32) -> f64 {
    1.0 / (1.0 + (-(x as f64)).exp())
}

/// Softmax of `logits` into `out`; entries where `keep` is false get 0.
fn softmax(logits: &[f32], keep: impl Fn(usize) -> bool, out: &mut [f64]) {
    let max = (0..logits.len())
        .filter(|&i| keep(i))
        .map(|i| logits[i] as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for (i, o) in out.iter_mut().enumerate() {
        *o = if keep(i) {
            (logits[i] as f64 - max).exp()
        } else {
            0.0
        };
        sum += *o;
    }
    if sum > 0.0 {
        for o in out.iter_mut() {
            *o /= sum;
        }
    }
}

impl BeliefModel {
    /// Loads an MJNN v2 belief model.
    pub fn load(path: impl AsRef<Path>) -> Result<BeliefModel, String> {
        let path = path.as_ref();
        Self::from_mlp(Mlp::load(path)?).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Wraps a loaded network; it must name the five heads with their sizes.
    pub fn from_mlp(net: Mlp) -> Result<BeliefModel, String> {
        let head = |name: &str, per_opp: usize| {
            let r = net
                .output(name)
                .ok_or_else(|| format!("not a belief model: no {name:?} output slice"))?;
            if r.len() != OPPONENTS * per_opp {
                return Err(format!(
                    "output slice {name:?} has {} values, expected {}",
                    r.len(),
                    OPPONENTS * per_opp
                ));
            }
            Ok(r)
        };
        Ok(BeliefModel {
            counts: head("counts", NUM_KINDS * MAX_COPIES)?,
            tenpai: head("tenpai", 1)?,
            waits: head("waits", NUM_KINDS)?,
            faan: head("faan", FAAN_BUCKETS)?,
            outcome: head("outcome", OUTCOMES)?,
            net,
            cache: Mutex::new(Vec::with_capacity(CACHE_SIZE)),
        })
    }

    /// The model's beliefs for the player to act in `state` (cached).
    ///
    /// # Panics
    /// Panics if the hand is over.
    pub fn run(&self, state: &State) -> Arc<BeliefOutput> {
        let me = state.current_player().expect("hand in progress");
        let key = fingerprint(state, me);
        if let Ok(cache) = self.cache.lock() {
            if let Some((_, out)) = cache.iter().find(|(k, _)| *k == key) {
                return Arc::clone(out);
            }
        }
        let out = Arc::new(self.compute(state, me));
        if let Ok(mut cache) = self.cache.lock() {
            if cache.len() >= CACHE_SIZE {
                cache.remove(0);
            }
            cache.push((key, Arc::clone(&out)));
        }
        out
    }

    /// One forward pass and the conversion to probabilities, without the cache.
    fn compute(&self, state: &State, me: usize) -> BeliefOutput {
        let y = self.net.forward_state(state);
        let unseen = state.unseen(me);
        let opponents = std::array::from_fn(|r| {
            let mut b = OpponentBelief {
                counts: [[0.0; MAX_COPIES]; NUM_KINDS],
                tenpai: sigmoid(y[self.tenpai.start + r]),
                waits: [0.0; NUM_KINDS],
                faan: [0.0; FAAN_BUCKETS],
                outcome: [0.0; OUTCOMES],
            };
            let counts = &y[self.counts.start + r * NUM_KINDS * MAX_COPIES..];
            for t in Tile::all_kinds() {
                let i = t.index() as usize;
                let left = unseen[t] as usize;
                softmax(
                    &counts[i * MAX_COPIES..(i + 1) * MAX_COPIES],
                    |k| k <= left,
                    &mut b.counts[i],
                );
                b.waits[i] = sigmoid(y[self.waits.start + r * NUM_KINDS + i]);
            }
            let f = self.faan.start + r * FAAN_BUCKETS;
            softmax(&y[f..f + FAAN_BUCKETS], |_| true, &mut b.faan);
            let o = self.outcome.start + r * OUTCOMES;
            softmax(&y[o..o + OUTCOMES], |_| true, &mut b.outcome);
            b
        });
        BeliefOutput { me, opponents }
    }
}

impl Belief for BeliefModel {
    /// The model's numbers in the `ev` bot's shape. `x_o` (wins on someone's
    /// discard before the hand ends) is the `outcome` head's second class, so
    /// `r_disc` is not used (`docs/ev-bot.md` §9 Q1). A wait on a kind with no
    /// unseen copy is set to 0, as in [`crate::Danger::estimate`].
    ///
    /// The network describes the player to act; for any other `me` this falls
    /// back to [`HeuristicBelief`].
    fn estimate(&self, state: &State, me: usize) -> Beliefs {
        if state.current_player() != Some(me) {
            return HeuristicBelief.estimate(state, me);
        }
        let out = self.run(state);
        let unseen = state.unseen(me);
        let mut seats: [SeatBelief; 4] = std::array::from_fn(|_| SeatBelief::quiet());
        for (o, s) in seats.iter_mut().enumerate() {
            let Some(b) = out.seat(o) else { continue };
            s.tenpai = b.tenpai;
            s.faan = b.faan;
            s.self_draw = b.outcome[2];
            s.discard_win = Some(b.outcome[1]);
            for t in Tile::all_kinds() {
                let i = t.index() as usize;
                let c = &b.counts[i];
                s.wait[i] = if unseen[t] == 0 { 0.0 } else { b.waits[i] };
                s.has_one[i] = 1.0 - c[0];
                s.has_two[i] = c[2..].iter().sum();
                s.mean_count[i] = c.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
            }
        }
        Beliefs { seats }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use mahjong_games::Policy;
    use mahjong_hk::RuleSet;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn model() -> Option<BeliefModel> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief.mjnn");
        p.exists().then(|| BeliefModel::load(&p).unwrap())
    }

    #[test]
    fn outputs_are_probabilities_and_respect_the_unseen_mask() {
        let Some(m) = model() else {
            eprintln!("models/belief.mjnn missing: skipped");
            return;
        };
        let bot = Bot::by_name("fast").unwrap();
        let mut rng = SmallRng::seed_from_u64(5);
        let mut checked = 0;
        for _ in 0..4 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(p) = s.current_player() {
                if checked % 7 == 0 {
                    let out = m.run(&s);
                    let unseen = s.unseen(p);
                    assert!(out.seat(p).is_none());
                    for b in &out.opponents {
                        assert!((0.0..=1.0).contains(&b.tenpai));
                        assert!((b.outcome.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                        assert!((b.faan.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                        for t in Tile::all_kinds() {
                            let c = &b.counts[t.index() as usize];
                            assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                            for (k, &pk) in c.iter().enumerate() {
                                if k > unseen[t] as usize {
                                    assert_eq!(pk, 0.0, "{t}: {k} copies but {} unseen", unseen[t]);
                                }
                            }
                        }
                    }
                    // The cache returns the same numbers.
                    assert_eq!(*m.run(&s), *out);
                    assert_eq!(m.compute(&s, p), *out);
                }
                checked += 1;
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
            }
        }
        assert!(checked > 100);
    }

    #[test]
    fn hidden_tiles_do_not_change_the_belief() {
        let Some(m) = model() else {
            return;
        };
        let bot = Bot::by_name("fast").unwrap();
        let mut rng = SmallRng::seed_from_u64(9);
        for _ in 0..10 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            for _ in 0..40 {
                if s.is_terminal() {
                    break;
                }
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
            }
            let Some(p) = s.current_player() else {
                continue;
            };
            let other = s.determinize(p, &mut rng);
            assert_eq!(fingerprint(&s, p), fingerprint(&other, p));
            assert_eq!(m.estimate(&s, p), m.estimate(&other, p));
        }
    }

    #[test]
    fn a_policy_network_is_not_a_belief_model() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ppo.mjnn");
        if p.exists() {
            let err = BeliefModel::load(&p).unwrap_err();
            assert!(err.contains("not a belief model"), "{err}");
        }
    }
}
