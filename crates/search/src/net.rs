//! Running a trained network in Rust (Phase 3).
//!
//! Networks are trained in PyTorch (`python/mahjong/rl/`) and exported as a plain
//! multilayer perceptron: Linear layers with ReLU between them, in a small binary
//! format (`MJNN`, version 1, little-endian):
//!
//! ```text
//! b"MJNN"  u32 version  u32 n_layers
//! for each layer: u32 out  u32 in  f32[out * in] (row-major)  f32[out]
//! ```
//!
//! The input is [`crate::features::encode`]'s vector; if the first layer expects
//! more inputs (a network trained with oracle features), the extra inputs are
//! zero, as at play time. The output has one value per action code.

use crate::features::{encode, STATE_LEN};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, State, NUM_ACTION_CODES};
use rand::Rng;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug)]
struct Layer {
    out: usize,
    inp: usize,
    w: Vec<f32>,
    b: Vec<f32>,
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

impl Mlp {
    /// Loads an `MJNN` file.
    pub fn load(path: impl AsRef<Path>) -> Result<Mlp, String> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Parses an `MJNN` buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Mlp, String> {
        let mut r = bytes;
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != b"MJNN" {
            return Err("not an MJNN network file".into());
        }
        let err = |e: std::io::Error| e.to_string();
        let version = read_u32(&mut r).map_err(err)?;
        if version != 1 {
            return Err(format!("unsupported MJNN version {version}"));
        }
        let n = read_u32(&mut r).map_err(err)? as usize;
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
        match (layers.first(), layers.last()) {
            (Some(f), Some(l)) if f.inp >= STATE_LEN && l.out == NUM_ACTION_CODES => {
                Ok(Mlp { layers })
            }
            _ => Err(format!(
                "expected input >= {STATE_LEN} and {NUM_ACTION_CODES} outputs"
            )),
        }
    }

    /// One forward pass on encoded features (zero-padded to the input size).
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

    /// Network scores of each legal action in `state` (same order as `legal`).
    pub fn scores(&self, state: &State, legal: &[Action]) -> Vec<f32> {
        let mut f = vec![0.0; STATE_LEN];
        encode(state, &mut f);
        let out = self.forward(&f);
        legal.iter().map(|a| out[a.code() as usize]).collect()
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

    fn tiny(in_extra: usize) -> Vec<u8> {
        let mut b = b"MJNN".to_vec();
        b.extend(1u32.to_le_bytes());
        b.extend(2u32.to_le_bytes());
        let inp = STATE_LEN + in_extra;
        for (out, i) in [(4usize, inp), (NUM_ACTION_CODES, 4)] {
            b.extend((out as u32).to_le_bytes());
            b.extend((i as u32).to_le_bytes());
            for k in 0..out * i {
                b.extend((((k % 7) as f32 - 3.0) * 0.01).to_le_bytes());
            }
            for k in 0..out {
                b.extend((k as f32 * 0.001).to_le_bytes());
            }
        }
        b
    }

    #[test]
    fn loads_and_runs() {
        let net = Mlp::from_bytes(&tiny(10)).unwrap();
        let y = net.forward(&vec![0.5; STATE_LEN]);
        assert_eq!(y.len(), NUM_ACTION_CODES);
        assert!(y.iter().all(|v| v.is_finite()));
        assert!(Mlp::from_bytes(b"nope").is_err());
    }
}
