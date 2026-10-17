//! Bindings for learning (Phase 3): feature sizes, batched self-play
//! environments, and teacher data generated in parallel in Rust.
//!
//! Arrays cross the boundary as flat little-endian `bytes` that Python views
//! with `numpy.frombuffer` (features are `float32`, masks `uint8`).

use crate::convert::rules_from;
use crate::{value_error, PyState};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State, NUM_ACTION_CODES};
use mahjong_search::features::{encode, encode_oracle, ORACLE_LEN, STATE_LEN};
use mahjong_search::Agent;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

fn f32_bytes<'py>(py: Python<'py>, v: &[f32]) -> Bound<'py, PyBytes> {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    PyBytes::new(py, &bytes)
}

/// `(STATE_LEN, ORACLE_LEN, NUM_ACTION_CODES)`.
#[pyfunction]
pub(crate) fn feature_sizes() -> (usize, usize, usize) {
    (STATE_LEN, ORACLE_LEN, NUM_ACTION_CODES)
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

/// Many hands stepped in lockstep. The learner plays every seat (`opponents=None`)
/// or one seat against an agent spec, with the learner in seat `i % 4` of game
/// `i` and games in groups of four sharing a wall (duplicate evaluation).
///
/// `observe()` encodes every game waiting on the learner; `step(actions)` applies
/// one action code per observed game (in the same order), lets opponents move,
/// and records and restarts finished games.
#[pyclass(name = "HkVecEnv", module = "mahjong._core")]
pub struct VecEnv {
    rules: RuleSet,
    opponents: Option<Agent>,
    rng: SmallRng,
    envs: Vec<State>,
    seeds: Vec<u64>,
    rounds: Vec<u64>,
    base_seed: u64,
    observed: Vec<usize>,
    finished: Vec<(usize, [i32; 4], u64)>,
}

impl VecEnv {
    fn learner(&self, i: usize) -> Option<usize> {
        self.opponents.as_ref().map(|_| i % 4)
    }

    fn deal(&mut self, i: usize) {
        // Game i plays round r of group g; in evaluation mode the four games of a
        // group play the same walls (duplicate format), whenever each finishes.
        let group = if self.opponents.is_some() { i / 4 } else { i } as u64;
        let round = self.rounds[i];
        self.rounds[i] += 1;
        let seed = self.base_seed
            ^ group.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ round.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        self.seeds[i] = seed;
        self.envs[i] = State::new_deal(self.rules, &mut SmallRng::seed_from_u64(seed));
    }

    /// Lets opponents move in game `i` until the learner must act or the hand ends.
    fn advance(&mut self, i: usize) {
        let Some(learner) = self.learner(i) else {
            return;
        };
        let agent = self.opponents.clone().expect("opponents");
        let mut legal = Vec::new();
        while let Some(p) = self.envs[i].current_player() {
            if p == learner {
                break;
            }
            self.envs[i].legal_actions(&mut legal);
            let a = agent.choose(&self.envs[i], &legal, &mut self.rng);
            self.envs[i].apply(a);
        }
    }

    fn settle(&mut self, i: usize) {
        while self.envs[i].is_terminal() {
            let r = self.envs[i].result().expect("over");
            self.finished.push((i, r.deltas, self.seeds[i]));
            self.deal(i);
            self.advance(i);
        }
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (n, seed=0, opponents=None, **rules))]
    fn new(
        n: usize,
        seed: u64,
        opponents: Option<&str>,
        rules: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let rules = rules_from(rules)?;
        let opponents = opponents
            .map(Agent::parse)
            .transpose()
            .map_err(value_error)?;
        if opponents.is_some() && !n.is_multiple_of(4) {
            return Err(value_error("evaluation needs a multiple of 4 games"));
        }
        let mut env = VecEnv {
            rules,
            opponents,
            rng: SmallRng::seed_from_u64(seed ^ 0xABCD),
            envs: vec![State::new_deal(rules, &mut SmallRng::seed_from_u64(0)); n],
            seeds: vec![0; n],
            rounds: vec![0; n],
            base_seed: seed.wrapping_mul(1_000_003),
            observed: Vec::new(),
            finished: Vec::new(),
        };
        for i in 0..n {
            env.deal(i);
            env.advance(i);
        }
        Ok(env)
    }

    /// `(features, oracle, masks, env_ids, seats)` for every game waiting on the
    /// learner; features and oracle are float32 bytes, masks uint8 bytes.
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
        self.observed = (0..self.envs.len())
            .filter(|&i| !self.envs[i].is_terminal())
            .collect();
        let envs = &self.envs;
        let rows: Vec<(Vec<f32>, Vec<f32>, Vec<u8>)> = py.detach(|| {
            self.observed
                .par_iter()
                .map(|&i| {
                    let s = &envs[i];
                    let mut f = vec![0.0; STATE_LEN];
                    let mut o = vec![0.0; ORACLE_LEN];
                    encode(s, &mut f);
                    encode_oracle(s, &mut o);
                    let mut m = vec![0u8; NUM_ACTION_CODES];
                    for a in s.legal_actions_vec() {
                        m[a.code() as usize] = 1;
                    }
                    (f, o, m)
                })
                .collect()
        });
        let mut feats = Vec::with_capacity(rows.len() * STATE_LEN);
        let mut oracle = Vec::with_capacity(rows.len() * ORACLE_LEN);
        let mut masks = Vec::with_capacity(rows.len() * NUM_ACTION_CODES);
        for (f, o, m) in rows {
            feats.extend(f);
            oracle.extend(o);
            masks.extend(m);
        }
        let seats = self
            .observed
            .iter()
            .map(|&i| envs[i].current_player().unwrap_or(0) as u8)
            .collect();
        (
            f32_bytes(py, &feats),
            f32_bytes(py, &oracle),
            PyBytes::new(py, &masks),
            self.observed.iter().map(|&i| i as u32).collect(),
            seats,
        )
    }

    /// Applies one action code per observed game.
    fn step(&mut self, actions: Vec<u16>) -> PyResult<()> {
        if actions.len() != self.observed.len() {
            return Err(value_error("one action per observed game"));
        }
        let observed = std::mem::take(&mut self.observed);
        for (&i, &code) in observed.iter().zip(&actions) {
            let a =
                Action::from_code(code).ok_or_else(|| value_error(format!("bad action {code}")))?;
            if !self.envs[i].legal_actions_vec().contains(&a) {
                return Err(value_error(format!("illegal action {a} in game {i}")));
            }
            self.envs[i].apply(a);
            self.advance(i);
            self.settle(i);
        }
        Ok(())
    }

    /// Finished games since the last call: `(env_id, deltas, seed)`; the learner's
    /// seat in evaluation mode is `env_id % 4`.
    fn finished(&mut self) -> Vec<(usize, [i32; 4], u64)> {
        std::mem::take(&mut self.finished)
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
