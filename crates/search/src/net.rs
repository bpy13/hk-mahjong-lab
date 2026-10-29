//! Running a trained network in Rust (Phase 3; belief model in plan Task 5).
//!
//! Networks are trained in PyTorch (`python/mahjong/rl/`, `python/mahjong/belief/`)
//! and exported as a plain multilayer perceptron: Linear layers with ReLU
//! between them, in a small binary format (`MJNN`, little-endian):
//!
//! ```text
//! b"MJNN"  u32 version
//! version 2 only:
//!   u32 n_inputs   for each: u32 name_len  name (utf-8)  u32 offset  u32 len
//!   u32 n_outputs  for each: u32 name_len  name (utf-8)  u32 offset  u32 len
//! u32 n_layers
//! for each layer: u32 out  u32 in  f32[out * in] (row-major)  f32[out]
//! ```
//!
//! **Version 1** has no names: the input is [`crate::features::encode`]'s vector
//! (zero-padded if the first layer expects more, as for a network trained with
//! oracle features) and the output has one value per action code.
//!
//! **Version 2** names its input and output slices. Input names this module can
//! fill from a [`State`]: `features` ([`crate::features::encode`]), `oracle`
//! (zeros at play time) and `discards` ([`encode_discards`]: each opponent's last
//! eight discards, one-hot). Output names are the model's business (the belief
//! model's are listed in `docs/belief.md`); a `policy` slice, when present, is
//! what [`NetPolicy`] reads. A v2 network without a `policy` slice is not a
//! policy and [`Mlp::scores`] must not be called on it.

use crate::features::{encode, ORACLE_LEN, STATE_LEN};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, Event, State, NUM_ACTION_CODES};
use mahjong_tiles::NUM_KINDS;
use rand::Rng;
use std::io::Read;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

/// Discards per opponent in the `discards` input block (most recent first).
pub const DISCARD_HISTORY: usize = 8;
/// Length of [`encode_discards`]'s output: 3 opponents × 8 discards × 34 kinds.
pub const DISCARD_LEN: usize = 3 * DISCARD_HISTORY * NUM_KINDS;

#[derive(Debug)]
struct Layer {
    out: usize,
    inp: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

/// A named slice of the input or output vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Slice {
    pub name: String,
    pub range: Range<usize>,
}

/// Dot product with eight independent accumulators, which the compiler turns
/// into SIMD (a single running sum cannot be vectorized without reordering).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b[..a.len()].as_chunks::<8>();
    let mut acc = [0.0f32; 8];
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}

/// A loaded network.
#[derive(Debug)]
pub struct Mlp {
    version: u32,
    inputs: Vec<Slice>,
    outputs: Vec<Slice>,
    layers: Vec<Layer>,
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32s(r: &mut impl Read, n: usize) -> std::io::Result<Vec<f32>> {
    let mut b = vec![0u8; 4 * n];
    r.read_exact(&mut b)?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn read_slices(r: &mut impl Read) -> Result<Vec<Slice>, String> {
    let err = |e: std::io::Error| e.to_string();
    let n = read_u32(r).map_err(err)? as usize;
    if n > 1024 {
        return Err(format!("{n} named slices: not an MJNN v2 header"));
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let len = read_u32(r).map_err(err)? as usize;
        let mut name = vec![0u8; len];
        r.read_exact(&mut name).map_err(err)?;
        let name = String::from_utf8(name).map_err(|e| e.to_string())?;
        let offset = read_u32(r).map_err(err)? as usize;
        let len = read_u32(r).map_err(err)? as usize;
        out.push(Slice {
            name,
            range: offset..offset + len,
        });
    }
    Ok(out)
}

/// Each opponent's last [`DISCARD_HISTORY`] discards as one-hot tiles into `out`
/// (length [`DISCARD_LEN`]): relative seat `r` (1 = next, 2 = across,
/// 3 = previous), slot `k` (0 = most recent) at
/// `((r - 1) * 8 + k) * 34 + tile`; empty slots stay zero. The same block as the
/// self-play log's `discards` field for seats 1–3.
///
/// # Panics
/// Panics if the hand is over or `out` has the wrong length.
pub fn encode_discards(state: &State, out: &mut [f32]) {
    assert_eq!(out.len(), DISCARD_LEN);
    out.fill(0.0);
    let me = state.current_player().expect("hand in progress");
    let mut history: [Vec<u8>; 4] = Default::default();
    for e in state.events() {
        if let Event::Discard { seat, tile } = *e {
            history[(seat as usize + 4 - me) % 4].push(tile.index());
        }
    }
    for r in 1..4 {
        for (k, &t) in history[r].iter().rev().take(DISCARD_HISTORY).enumerate() {
            out[((r - 1) * DISCARD_HISTORY + k) * NUM_KINDS + t as usize] = 1.0;
        }
    }
}

impl Mlp {
    /// Loads an `MJNN` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Mlp, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Parses an `MJNN` buffer (version 1 or 2).
    pub fn from_bytes(bytes: &[u8]) -> Result<Mlp, String> {
        let mut r = bytes;
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"MJNN" {
            return Err("not an MJNN network file".into());
        }
        let err = |e: std::io::Error| e.to_string();
        let version = read_u32(&mut r).map_err(err)?;
        let (inputs, outputs) = match version {
            1 => (
                vec![Slice {
                    name: "features".into(),
                    range: 0..STATE_LEN,
                }],
                vec![Slice {
                    name: "policy".into(),
                    range: 0..NUM_ACTION_CODES,
                }],
            ),
            2 => (read_slices(&mut r)?, read_slices(&mut r)?),
            v => return Err(format!("unsupported MJNN version {v}")),
        };
        let n = read_u32(&mut r).map_err(err)? as usize;
        if n == 0 {
            return Err("a network needs at least one layer".into());
        }
        let mut layers = Vec::with_capacity(n);
        for _ in 0..n {
            let out = read_u32(&mut r).map_err(err)? as usize;
            let inp = read_u32(&mut r).map_err(err)? as usize;
            let w = read_f32s(&mut r, out * inp).map_err(err)?;
            let b = read_f32s(&mut r, out).map_err(err)?;
            layers.push(Layer { out, inp, w, b });
        }
        for pair in layers.windows(2) {
            if pair[0].out != pair[1].inp {
                return Err("layer sizes do not chain".into());
            }
        }
        let (first, last) = (&layers[0], &layers[n - 1]);
        if version == 1 && (first.inp < STATE_LEN || last.out != NUM_ACTION_CODES) {
            return Err(format!(
                "expected input >= {STATE_LEN} and {NUM_ACTION_CODES} outputs"
            ));
        }
        for s in &inputs {
            let want = match s.name.as_str() {
                "features" => STATE_LEN,
                "oracle" => ORACLE_LEN,
                "discards" => DISCARD_LEN,
                other => return Err(format!("unknown input slice {other:?}")),
            };
            if s.range.len() != want || s.range.end > first.inp {
                return Err(format!(
                    "input slice {} is {:?}, expected {want} values inside {}",
                    s.name, s.range, first.inp
                ));
            }
        }
        for s in &outputs {
            if s.range.end > last.out {
                return Err(format!(
                    "output slice {} is {:?}, beyond {} outputs",
                    s.name, s.range, last.out
                ));
            }
        }
        Ok(Mlp {
            version,
            inputs,
            outputs,
            layers,
        })
    }

    /// The file format version (1 or 2).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Length of the input vector.
    pub fn input_len(&self) -> usize {
        self.layers[0].inp
    }

    /// Length of the output vector.
    pub fn output_len(&self) -> usize {
        self.layers[self.layers.len() - 1].out
    }

    /// The named input slices, in file order.
    pub fn inputs(&self) -> &[Slice] {
        &self.inputs
    }

    /// The named output slices, in file order.
    pub fn outputs(&self) -> &[Slice] {
        &self.outputs
    }

    /// Where the output slice `name` sits in [`Mlp::forward`]'s result.
    pub fn output(&self, name: &str) -> Option<Range<usize>> {
        self.outputs
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.range.clone())
    }

    /// The input vector for `state`: every named input slice filled from public
    /// information (`features`, `discards`; `oracle` stays zero).
    ///
    /// # Panics
    /// Panics if the hand is over.
    pub fn encode_input(&self, state: &State) -> Vec<f32> {
        let mut x = vec![0.0f32; self.input_len()];
        for s in &self.inputs {
            match s.name.as_str() {
                "features" => encode(state, &mut x[s.range.clone()]),
                "discards" => encode_discards(state, &mut x[s.range.clone()]),
                _ => {}
            }
        }
        x
    }

    /// One forward pass on an input vector (zero-padded to the input size).
    pub fn forward(&self, features: &[f32]) -> Vec<f32> {
        let mut x: Vec<f32> = features.to_vec();
        x.resize(self.layers[0].inp, 0.0);
        for (i, l) in self.layers.iter().enumerate() {
            let mut y: Vec<f32> = (0..l.out)
                .map(|o| l.b[o] + dot(&l.w[o * l.inp..(o + 1) * l.inp], &x))
                .collect();
            if i + 1 < self.layers.len() {
                for v in &mut y {
                    *v = v.max(0.0);
                }
            }
            x = y;
        }
        x
    }

    /// [`Mlp::encode_input`] followed by [`Mlp::forward`].
    pub fn forward_state(&self, state: &State) -> Vec<f32> {
        self.forward(&self.encode_input(state))
    }

    /// Network scores of each legal action in `state` (same order as `legal`),
    /// read from the `policy` output slice.
    ///
    /// # Panics
    /// Panics if the network has no `policy` slice of one value per action code.
    pub fn scores(&self, state: &State, legal: &[Action]) -> Vec<f32> {
        let policy = self
            .output("policy")
            .filter(|r| r.len() == NUM_ACTION_CODES)
            .expect("a policy network (an MJNN v1 file or a v2 `policy` slice)");
        let out = self.forward_state(state);
        legal
            .iter()
            .map(|a| out[policy.start + a.code() as usize])
            .collect()
    }
}

/// Plays the legal action with the highest network score (always declaring a
/// legal win, like every other agent).
#[derive(Clone, Debug)]
pub struct NetPolicy(pub Arc<Mlp>);

impl Policy<State> for NetPolicy {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], _rng: &mut R) -> Action {
        if let Some(&w) = legal
            .iter()
            .find(|a| matches!(a, Action::Win | Action::SelfDrawnWin))
        {
            return w;
        }
        if legal.len() == 1 {
            return legal[0];
        }
        let s = self.0.scores(state, legal);
        let best = (0..legal.len())
            .max_by(|&a, &b| s[a].total_cmp(&s[b]))
            .expect("non-empty");
        legal[best]
    }
}

/// The `k` legal actions the network rates highest, best first.
pub fn top_actions(net: &Mlp, state: &State, k: usize) -> Vec<Action> {
    let legal = state.legal_actions_vec();
    let s = net.scores(state, &legal);
    let mut idx: Vec<usize> = (0..legal.len()).collect();
    idx.sort_by(|&a, &b| s[b].total_cmp(&s[a]));
    idx.into_iter().take(k).map(|i| legal[i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use mahjong_hk::RuleSet;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn push_layers(b: &mut Vec<u8>, shapes: &[(usize, usize)]) {
        b.extend((shapes.len() as u32).to_le_bytes());
        for &(out, i) in shapes {
            b.extend((out as u32).to_le_bytes());
            b.extend((i as u32).to_le_bytes());
            for k in 0..out * i {
                b.extend((((k % 7) as f32 - 3.0) * 0.01).to_le_bytes());
            }
            for k in 0..out {
                b.extend((k as f32 * 0.001).to_le_bytes());
            }
        }
    }

    fn tiny(in_extra: usize) -> Vec<u8> {
        let mut b = b"MJNN".to_vec();
        b.extend(1u32.to_le_bytes());
        let inp = STATE_LEN + in_extra;
        push_layers(&mut b, &[(4, inp), (NUM_ACTION_CODES, 4)]);
        b
    }

    fn push_slices(b: &mut Vec<u8>, slices: &[(&str, usize, usize)]) {
        b.extend((slices.len() as u32).to_le_bytes());
        for &(name, offset, len) in slices {
            b.extend((name.len() as u32).to_le_bytes());
            b.extend(name.as_bytes());
            b.extend((offset as u32).to_le_bytes());
            b.extend((len as u32).to_le_bytes());
        }
    }

    /// A v2 belief-shaped network: features + discards in, named slices out.
    fn tiny_v2() -> Vec<u8> {
        let mut b = b"MJNN".to_vec();
        b.extend(2u32.to_le_bytes());
        push_slices(
            &mut b,
            &[
                ("features", 0, STATE_LEN),
                ("discards", STATE_LEN, DISCARD_LEN),
            ],
        );
        push_slices(&mut b, &[("tenpai", 0, 3), ("counts", 3, 510)]);
        push_layers(&mut b, &[(6, STATE_LEN + DISCARD_LEN), (513, 6)]);
        b
    }

    #[test]
    fn loads_and_runs() {
        let net = Mlp::from_bytes(&tiny(10)).unwrap();
        assert_eq!(net.version(), 1);
        let y = net.forward(&vec![0.5; STATE_LEN]);
        assert_eq!(y.len(), NUM_ACTION_CODES);
        assert!(y.iter().all(|v| v.is_finite()));
        assert_eq!(net.output("policy"), Some(0..NUM_ACTION_CODES));
        assert!(Mlp::from_bytes(b"nope").is_err());
    }

    #[test]
    fn v2_names_its_slices() {
        let net = Mlp::from_bytes(&tiny_v2()).unwrap();
        assert_eq!(net.version(), 2);
        assert_eq!(net.input_len(), STATE_LEN + DISCARD_LEN);
        assert_eq!(net.output("tenpai"), Some(0..3));
        assert_eq!(net.output("counts"), Some(3..513));
        assert_eq!(net.output("policy"), None);
        // A slice past the last layer, or an input name Rust cannot fill, is an error.
        let mut bad = b"MJNN".to_vec();
        bad.extend(2u32.to_le_bytes());
        push_slices(&mut bad, &[("features", 0, STATE_LEN)]);
        push_slices(&mut bad, &[("x", 0, 9)]);
        push_layers(&mut bad, &[(4, STATE_LEN)]);
        assert!(Mlp::from_bytes(&bad).is_err());
        let mut bad = b"MJNN".to_vec();
        bad.extend(2u32.to_le_bytes());
        push_slices(&mut bad, &[("hidden", 0, STATE_LEN)]);
        push_slices(&mut bad, &[]);
        push_layers(&mut bad, &[(4, STATE_LEN)]);
        assert!(Mlp::from_bytes(&bad).is_err());
    }

    /// Plays `steps` moves of a seeded hand with the `fast` bot; `None` if it ended.
    fn mid_hand(seed: u64, steps: usize) -> Option<State> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let bot = Bot::by_name("fast").unwrap();
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        for _ in 0..steps {
            if s.is_terminal() {
                return None;
            }
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
        }
        (!s.is_terminal()).then_some(s)
    }

    #[test]
    fn discard_block_lists_the_last_eight_per_opponent() {
        let s = (0..50)
            .find_map(|seed| mid_hand(seed, 60))
            .expect("some hand still running after 60 moves");
        let me = s.current_player().unwrap();
        let mut x = vec![0.0; DISCARD_LEN];
        encode_discards(&s, &mut x);
        let mut seen: [Vec<u8>; 4] = Default::default();
        for e in s.events() {
            if let Event::Discard { seat, tile } = *e {
                seen[(seat as usize + 4 - me) % 4].push(tile.index());
            }
        }
        for r in 1..4 {
            let block = &x[(r - 1) * DISCARD_HISTORY * NUM_KINDS..r * DISCARD_HISTORY * NUM_KINDS];
            let recent: Vec<u8> = seen[r]
                .iter()
                .rev()
                .take(DISCARD_HISTORY)
                .copied()
                .collect();
            assert_eq!(block.iter().sum::<f32>() as usize, recent.len());
            for (k, &t) in recent.iter().enumerate() {
                assert_eq!(block[k * NUM_KINDS + t as usize], 1.0);
            }
        }
    }

    /// AC3's last clause: two positions that differ only in hidden tiles give
    /// identical belief outputs (the v2 input is public information only).
    #[test]
    fn hidden_tiles_do_not_change_v2_outputs() {
        let net = Mlp::from_bytes(&tiny_v2()).unwrap();
        let belief = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief.mjnn");
        let trained = belief.exists().then(|| Mlp::load(&belief).unwrap());
        let mut rng = SmallRng::seed_from_u64(11);
        let mut checked = 0;
        for seed in 0..40 {
            let Some(s) = mid_hand(seed, 10 + (seed as usize * 7) % 70) else {
                continue;
            };
            let other = s.determinize(s.current_player().unwrap(), &mut rng);
            assert_ne!(
                s.hand((s.current_player().unwrap() + 1) % 4).tiles().len(),
                0
            );
            assert_eq!(net.encode_input(&s), net.encode_input(&other));
            assert_eq!(net.forward_state(&s), net.forward_state(&other));
            if let Some(t) = &trained {
                let y = t.forward_state(&s);
                assert_eq!(y, t.forward_state(&other));
                assert!(y.iter().all(|v| v.is_finite()));
            }
            checked += 1;
        }
        assert!(checked >= 20);
    }
}
