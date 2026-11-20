//! The trained belief model in Rust (plan Tasks 5 and 6, `docs/belief.md`).
//!
//! [`BeliefModel`] loads an MJNN v2 file with the heads `counts`, `tenpai`,
//! `waits`, `faan` and `outcome`, runs it on the acting seat's public view and
//! turns the raw outputs into probabilities ([`BeliefOutput`]):
//!
//! * `counts`: per opponent and tile kind, a categorical over 0–4 copies, with
//!   copies beyond what the observer has not seen masked out (the same mask as
//!   `mask_counts` in `python/mahjong/belief/model.py`);
//! * `tenpai`, `waits`: sigmoids; `faan`, `outcome`: softmaxes;
//! * optional, the race heads (`models/race-belief.mjnn`, `docs/race-bot.md` §6–7):
//!   `state` (far / one away / ready / folded) and `mode` (the opponent's last discard:
//!   push / detour / fold), softmaxes. A file without them (`models/belief.mjnn`)
//!   loads as before and leaves [`OpponentBelief::state`] and `mode` empty;
//! * optional, the heads of plan 2026-11-16 Task 19 (Brian's required outputs):
//!   `dealin` (per opponent and kind, P(it wins on a discard of that kind now),
//!   sigmoids: the unconditional form of `tenpai × waits`), `claim` (per opponent
//!   and kind, P(it would claim a discard of that kind into a meld now), sigmoids),
//!   `winner` (who wins the hand and how: for each relative seat self-drawn then
//!   off a discard, then a draw; one softmax of nine, [`BeliefOutput::win_chances`]
//!   folds it to Brian's five) and `points` (the observer's expected net points
//!   for the hand, one value × [`POINTS_SCALE`]). Such a file may leave the
//!   `counts` head out; the copies then follow the uniform law of the unseen
//!   tiles ([`uniform_counts`]). [`BeliefOutput::dealin_or_claim`] merges the
//!   6 × 34 into one figure per kind.
//!
//! A network that reads the `order` block was fitted on positions seen by a
//! seat that is not to act as well (right after its discard, after the claim
//! window closed, after an opponent claimed), so [`BeliefModel::run_for`] reads
//! it from any seat; older files are read for the player to act only.
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
/// Classes of the race `state` head: far, one away, ready, folded.
pub const STATES: usize = 4;
/// Classes of the race `mode` head: push, detour, fold.
pub const MODE_CLASSES: usize = 3;
/// Classes of the `winner` head: for each relative seat (0 = me) a self-drawn
/// win then a win off a discard (`2 r`, `2 r + 1`), then a draw (8).
pub const WINNERS: usize = 9;
/// The `winner` head's class for an exhaustive draw.
pub const WINNER_DRAW: usize = 8;
/// The `points` head is the observer's net points for the hand divided by this.
pub const POINTS_SCALE: f64 = 32.0;
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
    /// P(far), P(one away), P(ready), P(folded), when the file has the race heads.
    pub state: Option<[f64; STATES]>,
    /// P(push), P(detour), P(fold) for the opponent's last discard, when the file has
    /// the race heads.
    pub mode: Option<[f64; MODE_CLASSES]>,
    /// P(the opponent would claim a discard of kind `t` into a meld now), when the
    /// file has the `claim` head (plan 2026-11-16 Task 19).
    pub claim: Option<[f64; NUM_KINDS]>,
    /// P(the opponent wins on a discard of kind `t` now), when the file has the
    /// `dealin` head; else `tenpai × waits[t]`. Kinds with no unseen copy are 0.
    pub dealin: [f64; NUM_KINDS],
}

/// The model's output for one decision: the acting seat's three opponents.
#[derive(Clone, Debug, PartialEq)]
pub struct BeliefOutput {
    /// The seat the belief is for (the player to act).
    pub me: usize,
    /// Relative seats 1, 2, 3 (index 0 = the next seat).
    pub opponents: [OpponentBelief; OPPONENTS],
    /// The nine classes of the `winner` head (see [`WINNERS`]), when the file has
    /// it; sums to 1.
    pub winner: Option<[f64; WINNERS]>,
    /// The observer's expected net points for the hand, when the file has the
    /// `points` head.
    pub points: Option<f64>,
}

impl BeliefOutput {
    /// The belief about absolute seat `seat` (`None` for `me`).
    pub fn seat(&self, seat: usize) -> Option<&OpponentBelief> {
        let r = (seat + 4 - self.me) % 4;
        (r != 0).then(|| &self.opponents[r - 1])
    }

    /// Per kind, P(some opponent wins on it or claims it into a meld if I throw it
    /// now), the 6 × 34 of plan 2026-11-16 Task 19 ([`OpponentBelief::dealin`] and
    /// `claim`, 0 without that head) merged as `1 − Π_o (1 − d_o − (1 − d_o) c_o)`.
    pub fn dealin_or_claim(&self) -> [f64; NUM_KINDS] {
        let mut out = [0.0; NUM_KINDS];
        for (i, o) in out.iter_mut().enumerate() {
            let mut none = 1.0;
            for b in &self.opponents {
                let d = b.dealin[i];
                let c = b.claim.map_or(0.0, |c| c[i]);
                none *= 1.0 - d - (1.0 - d) * c;
            }
            *o = 1.0 - none;
        }
        out
    }

    /// Brian's five win chances: P(me wins), P(next), P(across), P(previous),
    /// P(draw), folded from the `winner` head; `None` without it.
    pub fn win_chances(&self) -> Option<[f64; 5]> {
        self.winner
            .map(|w| [w[0] + w[1], w[2] + w[3], w[4] + w[5], w[6] + w[7], w[8]])
    }
}

/// A loaded belief model (see the module docs).
pub struct BeliefModel {
    net: Mlp,
    counts: Option<Range<usize>>,
    tenpai: Range<usize>,
    waits: Range<usize>,
    faan: Range<usize>,
    outcome: Range<usize>,
    state: Option<Range<usize>>,
    mode: Option<Range<usize>>,
    winner: Option<Range<usize>>,
    claim: Option<Range<usize>>,
    dealin: Option<Range<usize>>,
    points: Option<Range<usize>>,
    cache: Mutex<Vec<(u64, Arc<BeliefOutput>)>>,
}

/// `ln C(n, k)` for small integers (the tiles of one hand), by the log-gamma sum.
fn ln_choose(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    let f = |m: usize| (1..=m).map(|i| (i as f64).ln()).sum::<f64>();
    f(n) - f(k) - f(n - k)
}

/// The uniform law of the copies of each kind in a hand of `hand` tiles drawn
/// from the `unseen` tiles (hypergeometric; `docs/belief.md` §4.1's "uniform
/// re-dealing"): `out[t][k] = C(u_t, k) C(U − u_t, hand − k) / C(U, hand)`.
pub fn uniform_counts(
    unseen: &mahjong_tiles::TileCounts,
    hand: usize,
) -> [[f64; MAX_COPIES]; NUM_KINDS] {
    let total: usize = Tile::all_kinds().map(|t| unseen[t] as usize).sum();
    let mut out = [[0.0; MAX_COPIES]; NUM_KINDS];
    if total == 0 || hand > total {
        for row in &mut out {
            row[0] = 1.0;
        }
        return out;
    }
    let denom = ln_choose(total, hand);
    for t in Tile::all_kinds() {
        let u = unseen[t] as usize;
        let row = &mut out[t.index() as usize];
        for (k, p) in row.iter_mut().enumerate() {
            if k <= u && hand >= k && hand - k <= total - u {
                *p = (ln_choose(u, k) + ln_choose(total - u, hand - k) - denom).exp();
            }
        }
    }
    out
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
    state.current_player().hash(&mut h);
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
        state
            .last_drawn()
            .filter(|_| state.current_player() == Some(me)),
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

    /// Wraps a loaded network; it must name the heads `tenpai`, `waits`, `faan` and
    /// `outcome` with their sizes. `counts`, the race heads `state` and `mode`, and
    /// the Task 19 heads `winner` and `claim` are optional, with their sizes checked.
    pub fn from_mlp(net: Mlp) -> Result<BeliefModel, String> {
        let optional = |name: &str, len: usize| match net.output(name) {
            None => Ok(None),
            Some(r) if r.len() == len => Ok(Some(r)),
            Some(r) => Err(format!(
                "output slice {name:?} has {} values, expected {len}",
                r.len()
            )),
        };
        let head = |name: &str, per_opp: usize| {
            optional(name, OPPONENTS * per_opp)?
                .ok_or_else(|| format!("not a belief model: no {name:?} output slice"))
        };
        Ok(BeliefModel {
            counts: optional("counts", OPPONENTS * NUM_KINDS * MAX_COPIES)?,
            tenpai: head("tenpai", 1)?,
            waits: head("waits", NUM_KINDS)?,
            faan: head("faan", FAAN_BUCKETS)?,
            outcome: head("outcome", OUTCOMES)?,
            state: optional("state", OPPONENTS * STATES)?,
            mode: optional("mode", OPPONENTS * MODE_CLASSES)?,
            winner: optional("winner", WINNERS)?,
            claim: optional("claim", OPPONENTS * NUM_KINDS)?,
            dealin: optional("dealin", OPPONENTS * NUM_KINDS)?,
            points: optional("points", 1)?,
            net,
            cache: Mutex::new(Vec::with_capacity(CACHE_SIZE)),
        })
    }

    /// Whether the file has the race `state` head (the `mode` head may be missing:
    /// the race bot reads each on its own).
    pub fn has_race_heads(&self) -> bool {
        self.state.is_some()
    }

    /// Whether the file has a `counts` head (else the copies follow [`uniform_counts`]).
    pub fn has_counts(&self) -> bool {
        self.counts.is_some()
    }

    /// Whether the file has the Task 19 heads `dealin`, `claim`, `winner` and `points`.
    pub fn has_task19_heads(&self) -> bool {
        self.winner.is_some()
            && self.claim.is_some()
            && self.dealin.is_some()
            && self.points.is_some()
    }

    /// Whether the network may be read from a seat that is not to act (it reads
    /// the `order` block, so it was fitted on such positions too).
    pub fn reads_spectator(&self) -> bool {
        self.net.reads("order")
    }

    /// The model's beliefs for the player to act in `state` (cached).
    ///
    /// # Panics
    /// Panics if the hand is over.
    pub fn run(&self, state: &State) -> Arc<BeliefOutput> {
        let me = state.current_player().expect("hand in progress");
        self.run_for(state, me)
    }

    /// The model's beliefs from seat `me`'s point of view (cached), whoever is to
    /// act: right after `me`'s discard (a claim window, or the next seat's turn),
    /// after a claim of it, and so on (plan 2026-11-16 Task 19). Meaningful only
    /// for a file with [`BeliefModel::reads_spectator`] when `me` is not acting.
    ///
    /// # Panics
    /// Panics if the hand is over.
    pub fn run_for(&self, state: &State, me: usize) -> Arc<BeliefOutput> {
        assert!(state.current_player().is_some(), "hand in progress");
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
        let y = self.net.forward_state_for(state, me);
        let unseen = state.unseen(me);
        let opponents = std::array::from_fn(|r| {
            let mut b = OpponentBelief {
                counts: [[0.0; MAX_COPIES]; NUM_KINDS],
                tenpai: sigmoid(y[self.tenpai.start + r]),
                waits: [0.0; NUM_KINDS],
                faan: [0.0; FAAN_BUCKETS],
                outcome: [0.0; OUTCOMES],
                state: None,
                mode: None,
                claim: None,
                dealin: [0.0; NUM_KINDS],
            };
            if let Some(head) = &self.counts {
                let counts = &y[head.start + r * NUM_KINDS * MAX_COPIES..];
                for t in Tile::all_kinds() {
                    let i = t.index() as usize;
                    let left = unseen[t] as usize;
                    softmax(
                        &counts[i * MAX_COPIES..(i + 1) * MAX_COPIES],
                        |k| k <= left,
                        &mut b.counts[i],
                    );
                }
            } else {
                b.counts = uniform_counts(&unseen, state.concealed_count((me + r + 1) % 4));
            }
            for t in Tile::all_kinds() {
                let i = t.index() as usize;
                b.waits[i] = sigmoid(y[self.waits.start + r * NUM_KINDS + i]);
                if unseen[t] > 0 {
                    b.dealin[i] = match &self.dealin {
                        Some(head) => sigmoid(y[head.start + r * NUM_KINDS + i]),
                        None => b.tenpai * b.waits[i],
                    };
                }
            }
            if let Some(head) = &self.claim {
                let mut c = [0.0; NUM_KINDS];
                for (i, v) in c.iter_mut().enumerate() {
                    *v = sigmoid(y[head.start + r * NUM_KINDS + i]);
                }
                b.claim = Some(c);
            }
            let f = self.faan.start + r * FAAN_BUCKETS;
            softmax(&y[f..f + FAAN_BUCKETS], |_| true, &mut b.faan);
            let o = self.outcome.start + r * OUTCOMES;
            softmax(&y[o..o + OUTCOMES], |_| true, &mut b.outcome);
            if let Some(head) = &self.state {
                let mut p = [0.0; STATES];
                softmax(&y[head.start + r * STATES..][..STATES], |_| true, &mut p);
                b.state = Some(p);
            }
            if let Some(head) = &self.mode {
                let mut p = [0.0; MODE_CLASSES];
                let at = head.start + r * MODE_CLASSES;
                softmax(&y[at..at + MODE_CLASSES], |_| true, &mut p);
                b.mode = Some(p);
            }
            b
        });
        let winner = self.winner.as_ref().map(|head| {
            let mut p = [0.0; WINNERS];
            softmax(&y[head.clone()], |_| true, &mut p);
            p
        });
        let points = self
            .points
            .as_ref()
            .map(|head| y[head.start] as f64 * POINTS_SCALE);
        BeliefOutput {
            me,
            opponents,
            winner,
            points,
        }
    }
}

impl Belief for BeliefModel {
    /// The model's numbers in the `ev` bot's shape. `x_o` (wins on someone's
    /// discard before the hand ends) is the `outcome` head's second class, so
    /// `r_disc` is not used (`docs/ev-bot.md` §9 Q1). A wait on a kind with no
    /// unseen copy is set to 0, as in [`crate::Danger::estimate`].
    ///
    /// The network describes the player to act; for any other `me` this falls
    /// back to [`HeuristicBelief`], unless the file reads the `order` block
    /// ([`BeliefModel::reads_spectator`]).
    fn estimate(&self, state: &State, me: usize) -> Beliefs {
        if state.current_player() != Some(me) && !self.reads_spectator() {
            return HeuristicBelief.estimate(state, me);
        }
        let out = self.run_for(state, me);
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
                // A file with the `dealin` head (plan Task 19) gives the chance
                // of a deal-in itself: `wait = dealin / tenpai`, so that
                // `tenpai × wait` is that head (plan Task 20).
                s.wait[i] = if unseen[t] == 0 {
                    0.0
                } else if self.dealin.is_some() && b.tenpai > 1e-9 {
                    (b.dealin[i] / b.tenpai).min(1.0)
                } else {
                    b.waits[i]
                };
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

    /// A tiny MJNN v2 file with the given input and output slices (weights are a
    /// fixed pattern), as `net.rs`'s tests build them.
    fn tiny_file(inputs: &[(&str, usize)], outputs: &[(&str, usize)]) -> Vec<u8> {
        let mut b = b"MJNN".to_vec();
        b.extend(2u32.to_le_bytes());
        for slices in [inputs, outputs] {
            b.extend((slices.len() as u32).to_le_bytes());
            let mut at = 0usize;
            for &(name, len) in slices {
                b.extend((name.len() as u32).to_le_bytes());
                b.extend(name.as_bytes());
                b.extend((at as u32).to_le_bytes());
                b.extend((len as u32).to_le_bytes());
                at += len;
            }
        }
        let n_in: usize = inputs.iter().map(|s| s.1).sum();
        let n_out: usize = outputs.iter().map(|s| s.1).sum();
        b.extend(2u32.to_le_bytes());
        for (out, inp) in [(8usize, n_in), (n_out, 8usize)] {
            b.extend((out as u32).to_le_bytes());
            b.extend((inp as u32).to_le_bytes());
            for k in 0..out * inp {
                b.extend((((k % 11) as f32 - 5.0) * 0.01).to_le_bytes());
            }
            for k in 0..out {
                b.extend((k as f32 * 0.001).to_le_bytes());
            }
        }
        b
    }

    /// Plan 2026-11-16 Task 19: a file with the `order` input and the `dealin`,
    /// `claim`, `winner`, `points` and `state` heads but no `counts` head loads;
    /// its copies follow the uniform law, its winner chances sum to 1, the 34
    /// numbers lie in [0, 1], and it is read from a seat that is not to act.
    #[test]
    fn task19_heads_load_without_counts() {
        use crate::features::{ORDER_LEN, STATE_LEN};
        let bytes = tiny_file(
            &[("features", STATE_LEN), ("order", ORDER_LEN)],
            &[
                ("tenpai", 3),
                ("waits", 102),
                ("faan", 24),
                ("outcome", 9),
                ("state", 12),
                ("winner", 9),
                ("claim", 102),
                ("dealin", 102),
                ("points", 1),
            ],
        );
        let m = BeliefModel::from_mlp(Mlp::from_bytes(&bytes).unwrap()).unwrap();
        assert!(!m.has_counts() && m.has_task19_heads() && m.has_race_heads());
        assert!(m.reads_spectator());
        let bot = Bot::by_name("fast").unwrap();
        let mut rng = SmallRng::seed_from_u64(3);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        for _ in 0..30 {
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
        }
        let me = s.current_player().unwrap();
        let out = m.run(&s);
        let w = out.winner.unwrap();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let five = out.win_chances().unwrap();
        assert!((five.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(out.points.unwrap().is_finite());
        // Read from another seat: its own view, a different answer, finite.
        let other = m.run_for(&s, (me + 1) % 4);
        assert_eq!(other.me, (me + 1) % 4);
        assert!(other.seat(me).is_some() && other.seat((me + 1) % 4).is_none());
        assert_eq!(*m.run_for(&s, (me + 1) % 4), *other);
        let est = m.estimate(&s, (me + 1) % 4);
        assert_eq!(est.seats[me].tenpai, other.seat(me).unwrap().tenpai);
        let unseen = s.unseen(me);
        for (r, b) in out.opponents.iter().enumerate() {
            let hand = s.concealed_count((me + r + 1) % 4);
            for t in Tile::all_kinds() {
                let c = &b.counts[t.index() as usize];
                assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                let mean: f64 = c.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
                let total: usize = Tile::all_kinds().map(|t| unseen[t] as usize).sum();
                let want = hand as f64 * unseen[t] as f64 / total as f64;
                assert!((mean - want).abs() < 1e-9, "{t}: {mean} vs {want}");
            }
            assert!(b.claim.unwrap().iter().all(|&p| (0.0..=1.0).contains(&p)));
            assert!(b.state.unwrap().iter().sum::<f64>() - 1.0 < 1e-9);
            assert!(b.mode.is_none());
        }
        let risk = out.dealin_or_claim();
        for t in Tile::all_kinds() {
            let p = risk[t.index() as usize];
            assert!((0.0..=1.0).contains(&p));
            if unseen[t] == 0 {
                assert_eq!(out.opponents[0].dealin[t.index() as usize], 0.0);
            }
        }
        // The old consumers still get a `Beliefs` (counts from the uniform law).
        let est = m.estimate(&s, me);
        assert!(est.seats[(me + 1) % 4].mean_count.iter().sum::<f64>() > 0.0);
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
