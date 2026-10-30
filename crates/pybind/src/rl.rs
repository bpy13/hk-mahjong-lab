//! Bindings for learning (Phase 3): feature sizes, batched self-play
//! environments, and teacher data generated in parallel in Rust.
//!
//! Arrays cross the boundary as flat little-endian `bytes` that Python views
//! with `numpy.frombuffer` (features are `float32`, masks `uint8`).

use crate::convert::rules_from;
use crate::{value_error, PyState};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, HandResult, Pattern, RuleSet, State, WinRecord, NUM_ACTION_CODES};
use mahjong_search::features::{encode, encode_oracle, ORACLE_LEN, STATE_LEN};
use mahjong_search::{Agent, Mlp};
use mahjong_tiles::Tile;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

fn f32_bytes<'py>(py: Python<'py>, v: &[f32]) -> Bound<'py, PyBytes> {
    // Written straight into the Python object: no intermediate byte vector
    // (`observe` hands over ~2 MB per call).
    PyBytes::new_with(py, 4 * v.len(), |buf| {
        for (b, x) in buf.as_chunks_mut::<4>().0.iter_mut().zip(v) {
            *b = x.to_le_bytes();
        }
        Ok(())
    })
    .expect("allocating a bytes object")
}

/// `(STATE_LEN, ORACLE_LEN, NUM_ACTION_CODES)`.
#[pyfunction]
pub(crate) fn feature_sizes() -> (usize, usize, usize) {
    (STATE_LEN, ORACLE_LEN, NUM_ACTION_CODES)
}

/// Runs an MJNN network file (v1–v3) in Rust on feature rows (`features`:
/// little-endian `float32` bytes, `n × input_len`); returns the outputs as
/// `float32` bytes (`n × output_len`) and `(input_len, output_len)`. For
/// checking an export against PyTorch (plan Task 14).
#[pyfunction]
pub(crate) fn net_forward<'py>(
    py: Python<'py>,
    path: &str,
    features: &[u8],
) -> PyResult<(Bound<'py, PyBytes>, (usize, usize))> {
    let net = Mlp::load(path).map_err(value_error)?;
    let (inp, out) = (net.input_len(), net.output_len());
    if !features.len().is_multiple_of(4 * inp) {
        return Err(value_error(format!(
            "{} bytes is not a whole number of {inp}-float rows",
            features.len()
        )));
    }
    let x: Vec<f32> = features
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let y: Vec<f32> = py.detach(|| {
        x.par_chunks(inp)
            .flat_map_iter(|row| net.forward(row))
            .collect()
    });
    Ok((f32_bytes(py, &y), (inp, out)))
}

/// The encoding of the decision of the player to act: `(features, oracle, mask)`
/// as lists (for tests and one-off use; training uses [`VecEnv`]).
#[pyfunction]
pub(crate) fn encode_state(state: &PyState) -> PyResult<(Vec<f32>, Vec<f32>, Vec<u32>)> {
    if state.inner.is_terminal() {
        return Err(value_error("the hand is over"));
    }
    let mut f = vec![0.0; STATE_LEN];
    let mut o = vec![0.0; ORACLE_LEN];
    encode(&state.inner, &mut f);
    encode_oracle(&state.inner, &mut o);
    let mut mask = vec![0u32; NUM_ACTION_CODES];
    for a in state.inner.legal_actions_vec() {
        mask[a.code() as usize] = 1;
    }
    Ok((f, o, mask))
}

/// Role of a decision handed to Python: a learner seat whose moves are trained.
const ROLE_LEARNER: u8 = 0;
/// Role: the baseline seat of a paired game, played from Python (`baseline=None`).
const ROLE_BASELINE: u8 = 1;
/// Role: another seat played by the learner's network but not trained on (the
/// non-learner seats of a paired game when `opponents=None`).
const ROLE_OPPONENT: u8 = 2;

/// Who moves for one seat of one game.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mover {
    /// Python chooses; the decision is observed with this role.
    Python(u8),
    /// The `opponents` agent, in Rust.
    Opponent,
    /// The `baseline` agent, in Rust.
    Baseline,
}

/// The second copy of every game in paired mode (idea 5a): the same wall played
/// again with the baseline agent in the learner's seat.
///
/// Until the learner and the baseline first choose differently, the two copies
/// are the same position: the pair is *shared*, only copy 0 is observed (except
/// at the learner seat's decisions, where a Python baseline sees copy 1 too),
/// and every move is applied to both copies. So the other seats' choices
/// (sampled by the learner's network or drawn by Rust agents) are common to both
/// copies up to the first difference (common random numbers), and those
/// positions are encoded once instead of twice.
struct Paired {
    /// `None`: Python plays the baseline seat (decisions with role 1).
    baseline: Option<Agent>,
    /// The baseline's copy of each game.
    shadow: Vec<State>,
    /// One random stream per game and side for the Rust agents, both seeded from
    /// the wall and kept equal while the pair is shared.
    rngs: Vec<[SmallRng; 2]>,
    /// The copies have played different moves since the deal.
    diverged: Vec<bool>,
    /// A Rust baseline's move at a shared learner-seat decision, applied to copy
    /// 1 when Python's move for copy 0 arrives.
    pending_baseline: Vec<Option<Action>>,
    /// Python's learner move at a shared decision, waiting for the Python
    /// baseline's move for copy 1 (later in the same `step`).
    pending_learner: Vec<Option<Action>>,
}

/// One winner of a finished hand: seat, target class of the winning hand (as in
/// `labels::target_class`), faan paid, faan pattern bits (bit i = `Pattern::ALL[i]`).
type WinInfo = (u8, u8, u8, u32);

/// The details of one finished hand (see `finished(detail=True)`).
struct Finished {
    env: usize,
    seed: u64,
    min_faan: u8,
    deltas: [i32; 4],
    wins: Vec<WinInfo>,
    /// The seat the learner played, or `None` when it played all four.
    learner: Option<usize>,
    /// Paired mode: the baseline copy's deltas and wins.
    baseline: Option<([i32; 4], Vec<WinInfo>)>,
}

/// The target class (`kind * 9 + suit * 3 + structure`, see
/// `mahjong_search::labels::target_class`) of the hand a seat finally won with:
/// thirteen orphans, all honours, full flush, half flush or mixed, crossed with
/// all pungs / all chows / any.
fn win_class(state: &State, w: &WinRecord) -> u8 {
    let has = |p: Pattern| w.score.patterns.iter().any(|&(q, _)| q == p);
    // The suit of a flush: the suit of any suited tile of the winner.
    let suit = || {
        let concealed = state.hand(w.seat).tiles();
        let melded = state.melds(w.seat).iter().map(|m| m.tile);
        concealed
            .into_iter()
            .chain(melded)
            .chain(w.win_tile)
            .find_map(|t: Tile| t.suit())
            .map_or(0, |s| s as u8)
    };
    let (kind, s) = if has(Pattern::ThirteenOrphans) {
        (4, 0)
    } else if has(Pattern::AllHonours) {
        (3, 0)
    } else if has(Pattern::FullFlush) {
        (2, suit())
    } else if has(Pattern::HalfFlush) {
        (1, suit())
    } else {
        (0, 0)
    };
    let structure = if has(Pattern::AllPungs) {
        1
    } else if has(Pattern::AllChows) {
        2
    } else {
        0
    };
    kind * 9 + s * 3 + structure
}

fn win_infos(state: &State, r: &HandResult) -> Vec<WinInfo> {
    r.wins
        .iter()
        .map(|w| {
            let bits = w
                .score
                .patterns
                .iter()
                .filter_map(|(p, _)| Pattern::ALL.iter().position(|q| q == p))
                .fold(0u32, |acc, i| acc | 1 << i);
            (w.seat as u8, win_class(state, w), w.score.faan, bits)
        })
        .collect()
}

/// Many hands stepped in lockstep. The learner plays every seat (`opponents=None`)
/// or one seat against an agent spec, with the learner in seat `i % 4` of game
/// `i` and games in groups of four sharing a wall (duplicate evaluation).
///
/// With `paired=True` (plan Task 17, idea 5a) every game `i` has a second copy
/// on the same wall: copy 0 has the learner in seat `i % 4`, copy 1 the
/// `baseline` agent (or Python, role 1, when `baseline=None`) in that seat; the
/// other seats are the `opponents` agent in both copies, or, with
/// `opponents=None`, the learner's network (role 2, not trained on). A game is
/// redealt only when both copies are over, and `finished(detail=True)` reports
/// both copies' deltas, so the learner's reward can be the paired difference.
///
/// `observe()` encodes every game waiting on Python; `step(actions)` applies
/// one action code per observed game (in the same order), lets the Rust agents
/// move, and records and restarts finished games.
#[pyclass(name = "HkVecEnv", module = "mahjong._core")]
pub struct VecEnv {
    rules: RuleSet,
    opponents: Option<Agent>,
    rng: SmallRng,
    envs: Vec<State>,
    seeds: Vec<u64>,
    rounds: Vec<u64>,
    base_seed: u64,
    /// Observed slots: game `i` copy `side` is `2 * i + side` (side 0 only
    /// without pairing).
    observed: Vec<usize>,
    finished: Vec<(usize, [i32; 4], u64)>,
    details: Vec<Finished>,
    paired: Option<Paired>,
    /// `min_faan` of the hand each game is playing (`set_min_faan` changes
    /// `rules` for the next deals only).
    dealt_min_faan: Vec<u8>,
}

impl VecEnv {
    fn game(&self, slot: usize) -> &State {
        let i = slot / 2;
        match (slot % 2, &self.paired) {
            (1, Some(p)) => &p.shadow[i],
            _ => &self.envs[i],
        }
    }

    fn mover(&self, i: usize, side: usize, seat: usize) -> Mover {
        let learner_seat = seat == i % 4;
        match &self.paired {
            None => match self.opponents {
                None => Mover::Python(ROLE_LEARNER),
                Some(_) if learner_seat => Mover::Python(ROLE_LEARNER),
                Some(_) => Mover::Opponent,
            },
            Some(p) => match (learner_seat, side) {
                (true, 0) => Mover::Python(ROLE_LEARNER),
                (true, _) if p.baseline.is_some() => Mover::Baseline,
                (true, _) => Mover::Python(ROLE_BASELINE),
                (false, _) if self.opponents.is_some() => Mover::Opponent,
                (false, _) => Mover::Python(ROLE_OPPONENT),
            },
        }
    }

    fn deal(&mut self, i: usize) {
        // Game i plays round r of group g; in evaluation mode the four games of a
        // group play the same walls (duplicate format), whenever each finishes.
        // Paired games each have their own wall (the pairing is the duplicate).
        let grouped = self.opponents.is_some() && self.paired.is_none();
        let group = if grouped { i / 4 } else { i } as u64;
        let round = self.rounds[i];
        self.rounds[i] += 1;
        let seed = self.base_seed
            ^ group.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ round.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        self.seeds[i] = seed;
        self.dealt_min_faan[i] = self.rules.min_faan;
        self.envs[i] = State::new_deal(self.rules, &mut SmallRng::seed_from_u64(seed));
        if let Some(p) = &mut self.paired {
            p.shadow[i] = self.envs[i].clone();
            let r = SmallRng::seed_from_u64(seed ^ 0x5EED_CA11);
            p.rngs[i] = [r.clone(), r];
            p.diverged[i] = false;
            p.pending_baseline[i] = None;
            p.pending_learner[i] = None;
        }
    }

    /// Whether game `i` is a pair whose copies are still the same position.
    fn shared(&self, i: usize) -> bool {
        self.paired.as_ref().is_some_and(|p| !p.diverged[i])
    }

    /// Whether copy `side` of game `i` waits on Python.
    fn observable(&self, i: usize, side: usize) -> bool {
        let s = self.game(2 * i + side);
        if s.is_terminal() {
            return false;
        }
        if side == 0 || !self.shared(i) {
            return true;
        }
        // A shared pair shows copy 1 only to a Python baseline at the learner's seat.
        let seat = s.current_player().unwrap_or(0);
        self.mover(i, 1, seat) == Mover::Python(ROLE_BASELINE)
    }

    /// Lets the Rust agents move in game `i` (both copies) until Python must act
    /// or the hand ends.
    fn advance_all(&mut self, i: usize) {
        if self.shared(i) {
            self.advance_shared(i);
        } else {
            self.advance(i, 0);
            if self.paired.is_some() {
                self.advance(i, 1);
            }
        }
    }

    /// [`Self::advance_all`] for a shared pair: Rust opponents' moves go to both
    /// copies; at the learner seat a Rust baseline's move is chosen now and kept
    /// until Python's move for copy 0 arrives.
    fn advance_shared(&mut self, i: usize) {
        let mut legal = Vec::new();
        loop {
            let Some(p) = self.envs[i].current_player() else {
                return;
            };
            let mover = self.mover(i, 0, p);
            let opponent = self.opponents.clone();
            let pd = self.paired.as_mut().expect("paired");
            match mover {
                Mover::Opponent => {
                    let agent = opponent.expect("opponents");
                    self.envs[i].legal_actions(&mut legal);
                    let a = agent.choose(&self.envs[i], &legal, &mut pd.rngs[i][0]);
                    self.envs[i].apply(a);
                    pd.shadow[i].apply(a);
                    pd.rngs[i][1] = pd.rngs[i][0].clone();
                }
                Mover::Python(ROLE_LEARNER) => {
                    if let (Some(b), None) = (&pd.baseline, pd.pending_baseline[i]) {
                        pd.shadow[i].legal_actions(&mut legal);
                        let a = b.choose(&pd.shadow[i], &legal, &mut pd.rngs[i][1]);
                        pd.pending_baseline[i] = Some(a);
                    }
                    return;
                }
                _ => return,
            }
        }
    }

    /// Applies Python's move `a` to copy `side` of a shared pair.
    fn step_shared(&mut self, i: usize, side: usize, a: Action) {
        let focus = self.envs[i].current_player() == Some(i % 4);
        let pd = self.paired.as_mut().expect("paired");
        let learner_and_python_baseline = focus && pd.baseline.is_none();
        match side {
            0 if !focus => {
                // Another seat: the same move in both copies.
                self.envs[i].apply(a);
                pd.shadow[i].apply(a);
            }
            0 if learner_and_python_baseline => {
                // Wait for the baseline's move on copy 1 (later in this step).
                self.envs[i].apply(a);
                pd.pending_learner[i] = Some(a);
                return;
            }
            0 => {
                let b = pd.pending_baseline[i].take().expect("the baseline's move");
                self.envs[i].apply(a);
                pd.shadow[i].apply(b);
                if a == b {
                    pd.rngs[i][0] = pd.rngs[i][1].clone();
                } else {
                    pd.diverged[i] = true;
                }
            }
            _ => {
                let l = pd.pending_learner[i].take().expect("the learner's move");
                pd.shadow[i].apply(a);
                if a != l {
                    pd.diverged[i] = true;
                }
            }
        }
        self.advance_all(i);
        self.settle(i);
    }

    /// Lets the Rust agents move in game `i`, copy `side`, until Python must act
    /// or the hand ends.
    fn advance(&mut self, i: usize, side: usize) {
        if self.paired.is_none() && self.opponents.is_none() {
            return;
        }
        if self.shared(i) {
            self.advance_shared(i);
            return;
        }
        let mut legal = Vec::new();
        loop {
            let slot = 2 * i + side;
            let Some(p) = self.game(slot).current_player() else {
                return;
            };
            let agent = match self.mover(i, side, p) {
                Mover::Python(_) => return,
                Mover::Opponent => self.opponents.as_ref(),
                Mover::Baseline => self.paired.as_ref().and_then(|p| p.baseline.as_ref()),
            }
            .expect("a Rust agent")
            .clone();
            match &mut self.paired {
                None => {
                    // Unpaired evaluation: one shared stream (the original behaviour).
                    self.envs[i].legal_actions(&mut legal);
                    let a = agent.choose(&self.envs[i], &legal, &mut self.rng);
                    self.envs[i].apply(a);
                }
                Some(pd) => {
                    let s = if side == 0 {
                        &mut self.envs[i]
                    } else {
                        &mut pd.shadow[i]
                    };
                    s.legal_actions(&mut legal);
                    let a = agent.choose(s, &legal, &mut pd.rngs[i][side]);
                    s.apply(a);
                }
            }
        }
    }

    fn record(&mut self, i: usize) {
        let r = self.envs[i].result().expect("over");
        self.finished.push((i, r.deltas, self.seeds[i]));
        let learner = (self.opponents.is_some() || self.paired.is_some()).then_some(i % 4);
        let baseline = self.paired.as_ref().map(|p| {
            let s = &p.shadow[i];
            let rb = s.result().expect("over");
            (rb.deltas, win_infos(s, rb))
        });
        self.details.push(Finished {
            env: i,
            seed: self.seeds[i],
            min_faan: self.dealt_min_faan[i],
            deltas: r.deltas,
            wins: win_infos(&self.envs[i], r),
            learner,
            baseline,
        });
    }

    fn settle(&mut self, i: usize) {
        loop {
            let over = self.envs[i].is_terminal()
                && self
                    .paired
                    .as_ref()
                    .is_none_or(|p| p.shadow[i].is_terminal());
            if !over {
                return;
            }
            self.record(i);
            self.deal(i);
            self.advance_all(i);
        }
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (n, seed=0, opponents=None, paired=false, baseline=None, **rules))]
    fn new(
        n: usize,
        seed: u64,
        opponents: Option<&str>,
        paired: bool,
        baseline: Option<&str>,
        rules: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let rules = rules_from(rules)?;
        let opponents = opponents
            .map(Agent::parse)
            .transpose()
            .map_err(value_error)?;
        if opponents.is_some() && !paired && !n.is_multiple_of(4) {
            return Err(value_error("evaluation needs a multiple of 4 games"));
        }
        if baseline.is_some() && !paired {
            return Err(value_error("baseline needs paired=True"));
        }
        let blank = State::new_deal(rules, &mut SmallRng::seed_from_u64(0));
        let paired = if paired {
            Some(Paired {
                baseline: baseline
                    .map(Agent::parse)
                    .transpose()
                    .map_err(value_error)?,
                shadow: vec![blank.clone(); n],
                rngs: vec![[SmallRng::seed_from_u64(0), SmallRng::seed_from_u64(0)]; n],
                diverged: vec![false; n],
                pending_baseline: vec![None; n],
                pending_learner: vec![None; n],
            })
        } else {
            None
        };
        let mut env = VecEnv {
            rules,
            opponents,
            rng: SmallRng::seed_from_u64(seed ^ 0xABCD),
            envs: vec![blank; n],
            seeds: vec![0; n],
            rounds: vec![0; n],
            base_seed: seed.wrapping_mul(1_000_003),
            observed: Vec::new(),
            finished: Vec::new(),
            details: Vec::new(),
            paired,
            dealt_min_faan: vec![rules.min_faan; n],
        };
        for i in 0..n {
            env.deal(i);
            env.advance_all(i);
            // A hand can end before Python's first move (a heavenly win).
            env.settle(i);
        }
        Ok(env)
    }

    /// Sets the minimum faan for hands dealt from now on (the curriculum, idea
    /// 1b); hands in progress keep theirs.
    fn set_min_faan(&mut self, min_faan: u8) {
        self.rules.min_faan = min_faan;
    }

    /// The minimum faan of the next deals.
    #[getter]
    fn min_faan(&self) -> u8 {
        self.rules.min_faan
    }

    /// Whether games are paired with a baseline copy.
    #[getter]
    fn paired(&self) -> bool {
        self.paired.is_some()
    }

    /// `(features, oracle, masks, env_ids, seats)` for every game waiting on
    /// Python; features and oracle are float32 bytes, masks uint8 bytes. In
    /// paired mode `env_ids` are game indices and `observed_roles()` tells the
    /// copies apart.
    #[allow(clippy::type_complexity)]
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Bound<'py, PyBytes>,
        Vec<u32>,
        Vec<u8>,
    ) {
        let sides = if self.paired.is_some() { 2 } else { 1 };
        self.observed = (0..self.envs.len())
            .flat_map(|i| (0..sides).map(move |s| 2 * i + s))
            .filter(|&slot| self.observable(slot / 2, slot % 2))
            .collect();
        // Each row is encoded in parallel straight into its slice of the output
        // (no per-row vectors to gather afterwards).
        let n = self.observed.len();
        let mut feats = vec![0.0f32; n * STATE_LEN];
        let mut oracle = vec![0.0f32; n * ORACLE_LEN];
        let mut masks = vec![0u8; n * NUM_ACTION_CODES];
        let this = &*self;
        py.detach(|| {
            feats
                .par_chunks_mut(STATE_LEN)
                .zip(oracle.par_chunks_mut(ORACLE_LEN))
                .zip(masks.par_chunks_mut(NUM_ACTION_CODES))
                .zip(this.observed.par_iter())
                .for_each(|(((f, o), m), &slot)| {
                    let s = this.game(slot);
                    encode(s, f);
                    encode_oracle(s, o);
                    for a in s.legal_actions_vec() {
                        m[a.code() as usize] = 1;
                    }
                });
        });
        let seats = self
            .observed
            .iter()
            .map(|&slot| self.game(slot).current_player().unwrap_or(0) as u8)
            .collect();
        (
            f32_bytes(py, &feats),
            f32_bytes(py, &oracle),
            PyBytes::new(py, &masks),
            self.observed
                .iter()
                .map(|&slot| (slot / 2) as u32)
                .collect(),
            seats,
        )
    }

    /// `(sides, roles)` of the last `observe()`: copy 0 (learner) or 1 (baseline)
    /// of each game, and the role of each decision (0 = learner, trained; 1 =
    /// baseline seat played from Python; 2 = another seat played by the learner's
    /// network, not trained on).
    fn observed_roles(&self) -> (Vec<u8>, Vec<u8>) {
        self.observed
            .iter()
            .map(|&slot| {
                let (i, side) = (slot / 2, slot % 2);
                let seat = self.game(slot).current_player().unwrap_or(0);
                let role = match self.mover(i, side, seat) {
                    Mover::Python(r) => r,
                    _ => ROLE_OPPONENT,
                };
                (side as u8, role)
            })
            .unzip()
    }

    /// The moves the agent `spec` would choose in each observed game (in the
    /// order of the last `observe()`), with a random stream seeded by `seed`.
    /// For tests and teacher comparisons.
    #[pyo3(signature = (spec, seed=0))]
    fn agent_actions(&self, py: Python<'_>, spec: &str, seed: u64) -> PyResult<Vec<u16>> {
        let agent = Agent::parse(spec).map_err(value_error)?;
        let this = self;
        Ok(py.detach(|| {
            this.observed
                .par_iter()
                .map(|&slot| {
                    let s = this.game(slot);
                    let legal = s.legal_actions_vec();
                    let mut rng = SmallRng::seed_from_u64(seed ^ slot as u64);
                    agent.choose(s, &legal, &mut rng).code()
                })
                .collect()
        }))
    }

    /// Applies one action code per observed game.
    fn step(&mut self, actions: Vec<u16>) -> PyResult<()> {
        if actions.len() != self.observed.len() {
            return Err(value_error("one action per observed game"));
        }
        let observed = std::mem::take(&mut self.observed);
        for (&slot, &code) in observed.iter().zip(&actions) {
            let a =
                Action::from_code(code).ok_or_else(|| value_error(format!("bad action {code}")))?;
            let (i, side) = (slot / 2, slot % 2);
            if !self.game(slot).legal_actions_vec().contains(&a) {
                return Err(value_error(format!("illegal action {a} in game {i}")));
            }
            if self.shared(i) {
                self.step_shared(i, side, a);
                continue;
            }
            match (side, &mut self.paired) {
                (1, Some(p)) => p.shadow[i].apply(a),
                _ => self.envs[i].apply(a),
            }
            self.advance(i, side);
            self.settle(i);
        }
        Ok(())
    }

    /// Finished games since the last call: `(env_id, deltas, seed)`; the learner's
    /// seat in evaluation mode is `env_id % 4`.
    ///
    /// With `detail=True`, one dict per finished game instead: `env`, `seed`,
    /// `min_faan` (of that deal), `deltas` (the learner's copy), `learner` (its
    /// seat, or None when it played all four), `wins` (per winner: `(seat,
    /// target_class, faan, pattern_bits)`; the class is the hand's final plan, as
    /// `labels::target_class` numbers it, so per-pattern win counts follow), and
    /// in paired mode `baseline_deltas`, `baseline_wins` and `paired` (the
    /// learner seat's points minus the baseline's on the same wall).
    #[pyo3(signature = (detail=false))]
    fn finished(&mut self, py: Python<'_>, detail: bool) -> PyResult<Py<PyAny>> {
        let plain = std::mem::take(&mut self.finished);
        let details = std::mem::take(&mut self.details);
        if !detail {
            return Ok(plain.into_pyobject(py)?.into_any().unbind());
        }
        let out = PyList::empty(py);
        for f in details {
            let d = PyDict::new(py);
            d.set_item("env", f.env)?;
            d.set_item("seed", f.seed)?;
            d.set_item("min_faan", f.min_faan)?;
            d.set_item("deltas", f.deltas)?;
            d.set_item("learner", f.learner)?;
            d.set_item("wins", f.wins)?;
            if let Some((bd, bw)) = f.baseline {
                let seat = f.learner.expect("paired games have a learner seat");
                d.set_item("paired", f.deltas[seat] - bd[seat])?;
                d.set_item("baseline_deltas", bd)?;
                d.set_item("baseline_wins", bw)?;
            }
            out.append(d)?;
        }
        Ok(out.into_any().unbind())
    }
}

/// Teacher data: features, oracle, masks (bytes), actions, returns, hand index.
type TeacherData<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Vec<u16>,
    Vec<f32>,
    Vec<u32>,
);

/// One recorded decision: features, oracle, mask, action code, acting seat.
type Decision = (Vec<f32>, Vec<f32>, Vec<u8>, u16, usize);

/// Plays `hands` hands with `teacher` in every seat (in parallel) and returns,
/// for every decision with a choice: features, oracle, masks (bytes), the
/// teacher's action codes, the acting seat's final points, and the hand index.
#[pyfunction]
#[pyo3(signature = (hands, teacher="value", seed=0, **rules))]
pub(crate) fn teacher_data<'py>(
    py: Python<'py>,
    hands: usize,
    teacher: &str,
    seed: u64,
    rules: Option<&Bound<'py, PyDict>>,
) -> PyResult<TeacherData<'py>> {
    let rules = rules_from(rules)?;
    let agent = Agent::parse(teacher).map_err(value_error)?;
    type Row = (Vec<f32>, Vec<f32>, Vec<u8>, u16, f32, u32);
    let rows: Vec<Row> = py.detach(|| {
        (0..hands)
            .into_par_iter()
            .flat_map_iter(|h| {
                let mut rng =
                    SmallRng::seed_from_u64(seed ^ (h as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut s = State::new_deal(rules, &mut rng);
                let mut pending: Vec<Decision> = Vec::new();
                let mut legal = Vec::new();
                while let Some(p) = s.current_player() {
                    s.legal_actions(&mut legal);
                    let a = agent.choose(&s, &legal, &mut rng);
                    if legal.len() > 1 {
                        let mut f = vec![0.0; STATE_LEN];
                        let mut o = vec![0.0; ORACLE_LEN];
                        encode(&s, &mut f);
                        encode_oracle(&s, &mut o);
                        let mut m = vec![0u8; NUM_ACTION_CODES];
                        for x in &legal {
                            m[x.code() as usize] = 1;
                        }
                        pending.push((f, o, m, a.code(), p));
                    }
                    s.apply(a);
                    let _ = rng.gen::<u8>();
                }
                let deltas = s.result().expect("over").deltas;
                pending
                    .into_iter()
                    .map(move |(f, o, m, a, p)| (f, o, m, a, deltas[p] as f32, h as u32))
            })
            .collect()
    });
    let mut feats = Vec::with_capacity(rows.len() * STATE_LEN);
    let mut oracle = Vec::with_capacity(rows.len() * ORACLE_LEN);
    let mut masks = Vec::with_capacity(rows.len() * NUM_ACTION_CODES);
    let (mut acts, mut rets, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    for (f, o, m, a, r, h) in rows {
        feats.extend(f);
        oracle.extend(o);
        masks.extend(m);
        acts.push(a);
        rets.push(r);
        idx.push(h);
    }
    Ok((
        f32_bytes(py, &feats),
        f32_bytes(py, &oracle),
        PyBytes::new(py, &masks),
        acts,
        rets,
        idx,
    ))
}
