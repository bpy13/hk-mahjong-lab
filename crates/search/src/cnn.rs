//! The suit-shared 1-D CNN (plan Task 14), stored as MJNN version 3.
//!
//! Trained in PyTorch (`python/mahjong/rl/cnn.py`, which documents the design);
//! this module runs its forward pass on one thread with no allocation-heavy
//! framework, for the solver, the tournaments and (later) the app.
//!
//! **Input**: [`crate::features::encode`]'s vector, viewed through
//! [`encode_planes`]: each suit's 9 tiles × 26 planes, and a rest block (the
//! 7 honours × 26 planes, then the scalars).
//!
//! **Network**:
//!
//! 1. *Honour branch*: two dense layers with ReLU on the rest block → `h`.
//! 2. *Suit tower*: a convolution (kernel `k`, 26 → `C` channels) and
//!    `blocks` residual blocks, the **same weights for every suit**. Each block
//!    first adds a context vector to every position: a dense layer of the sum
//!    and the maximum over suits of each suit's mean-pooled channels, and `h`.
//!    That is how suits see each other and the honours, and because sum and max
//!    do not care about suit order, swapping two suits in the input swaps them in
//!    the output.
//! 3. *Merge*: `z = relu(dense([sum, max of the pooled suits, h]))`.
//! 4. *Heads*: the per-tile actions of suited tiles (discard, concealed kong,
//!    added kong, chow from that tile) come from a 1×1 convolution over each
//!    suit's positions after adding a projection of `z`; honour-tile and global
//!    actions (self-drawn win, pass, win, pung, kong) from a dense layer on `z`.
//!    The **target** head scores 45 plan classes ([`crate::labels::target_class`]):
//!    flush classes per suit from that suit's pooled channels and `z`, the
//!    others from `z`; the 18 unused (suit-less plans with suit 1 or 2) get a
//!    fixed −10⁴. The **value** head gives four relative-seat values summing to
//!    zero (the mean is subtracted).
//!
//! **File** (after `b"MJNN"`, `u32 3` and the v2 named input and output
//! slices): `u32 len` + architecture name ([`ARCH`]); `u32 n` hyper-parameters
//! as `u32 len` + name + `u32 value`; `u32 n` tensors as `u32 len` + name,
//! `u32 dtype` (0 = f32, 1 = int8 with one f32 scale per output row), `u32
//! ndim`, `u32` dims, then (int8 only) `f32` scales for each of `dims[0]` rows
//! and the values. Tensor names and shapes are PyTorch's (`stem.weight`
//! `[C, 26, k]`, `blocks.0.conv1.weight`, …). Int8 weights are expanded to f32
//! when loaded, so int8 only shrinks the file.

use crate::features::{
    encode_planes, HONOUR_KINDS, PLANES, REST_LEN, STATE_LEN, SUIT_BLOCK_LEN, SUIT_LEN,
};
use std::collections::HashMap;
use std::io::Read;

/// Architecture name in the file.
pub const ARCH: &str = "suitcnn-1";
/// Output lengths: policy (one per action code), target classes, values.
pub const POLICY_LEN: usize = 134;
pub const TARGET_LEN: usize = 45;
pub const VALUE_LEN: usize = 4;
/// Total output length: policy, then target, then value.
pub const OUTPUT_LEN: usize = POLICY_LEN + TARGET_LEN + VALUE_LEN;
/// Logit of the target classes no plan uses.
pub const UNUSED_TARGET: f32 = -1.0e4;
/// Outputs of the honour-and-global policy layer: discard, concealed kong and
/// added kong of each honour, then self-drawn win, pass, win, pung, kong.
const HONOUR_POLICY: usize = 3 * HONOUR_KINDS + 5;
const GLOBAL_CODES: [usize; 5] = [34, 103, 104, 105, 106];

use crate::net::dot;

fn read_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

fn read_name(r: &mut impl Read) -> Result<String, String> {
    let len = read_u32(r)? as usize;
    if len > 4096 {
        return Err(format!("name of {len} bytes"));
    }
    let mut b = vec![0u8; len];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    String::from_utf8(b).map_err(|e| e.to_string())
}

fn read_f32s(r: &mut impl Read, n: usize) -> Result<Vec<f32>, String> {
    let mut b = vec![0u8; 4 * n];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

/// A tensor read from the file (int8 already expanded).
struct Tensor {
    dims: Vec<usize>,
    data: Vec<f32>,
}

/// A dense layer, row-major `[out][inp]`.
#[derive(Debug)]
struct Dense {
    out: usize,
    inp: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Dense {
    fn apply(&self, x: &[f32], y: &mut [f32], relu: bool) {
        debug_assert_eq!(x.len(), self.inp);
        for (o, yo) in y.iter_mut().enumerate().take(self.out) {
            let v = self.b[o] + dot(&self.w[o * self.inp..(o + 1) * self.inp], x);
            *yo = if relu { v.max(0.0) } else { v };
        }
    }
}

/// A 1-D convolution with zero padding `k / 2` on each side; weights stored
/// `[out][tap][inp]` so each output is one dot product with a contiguous
/// window of a position-major input.
#[derive(Debug)]
struct Conv {
    out: usize,
    inp: usize,
    k: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Conv {
    /// `padded`: `(len + k - 1) × inp`, position-major, zeros at both ends;
    /// `y`: `len × out`.
    fn apply(&self, padded: &[f32], len: usize, y: &mut [f32]) {
        let win = self.k * self.inp;
        for i in 0..len {
            let x = &padded[i * self.inp..i * self.inp + win];
            for o in 0..self.out {
                y[i * self.out + o] = self.b[o] + dot(&self.w[o * win..(o + 1) * win], x);
            }
        }
    }
}

#[derive(Debug)]
struct Block {
    ctx: Dense,
    conv1: Conv,
    conv2: Conv,
}

/// A loaded suit-shared CNN.
#[derive(Debug)]
pub struct SuitCnn {
    channels: usize,
    kernel: usize,
    honour1: Dense,
    honour2: Dense,
    stem: Conv,
    blocks: Vec<Block>,
    merge: Dense,
    zproj: Dense,
    suit_policy: Dense,
    honour_policy: Dense,
    flush_target: Dense,
    other_target: Dense,
    value1: Dense,
    value2: Dense,
}

struct Tensors(HashMap<String, Tensor>);

impl Tensors {
    fn take(&mut self, name: &str, dims: &[usize]) -> Result<Vec<f32>, String> {
        let t = self
            .0
            .remove(name)
            .ok_or_else(|| format!("tensor {name} missing"))?;
        if t.dims != dims {
            return Err(format!("tensor {name} is {:?}, expected {dims:?}", t.dims));
        }
        Ok(t.data)
    }

    fn dense(&mut self, name: &str, out: usize, inp: usize) -> Result<Dense, String> {
        Ok(Dense {
            out,
            inp,
            w: self.take(&format!("{name}.weight"), &[out, inp])?,
            b: self.take(&format!("{name}.bias"), &[out])?,
        })
    }

    fn conv(&mut self, name: &str, out: usize, inp: usize, k: usize) -> Result<Conv, String> {
        let torch = self.take(&format!("{name}.weight"), &[out, inp, k])?;
        // PyTorch's [out][inp][tap] → [out][tap][inp].
        let mut w = vec![0.0; out * k * inp];
        for o in 0..out {
            for c in 0..inp {
                for j in 0..k {
                    w[(o * k + j) * inp + c] = torch[(o * inp + c) * k + j];
                }
            }
        }
        Ok(Conv {
            out,
            inp,
            k,
            w,
            b: self.take(&format!("{name}.bias"), &[out])?,
        })
    }
}

impl SuitCnn {
    /// Reads the part of an MJNN v3 file after the named slices.
    pub(crate) fn read(r: &mut impl Read) -> Result<SuitCnn, String> {
        let arch = read_name(r)?;
        if arch != ARCH {
            return Err(format!("unknown MJNN v3 architecture {arch:?}"));
        }
        let n = read_u32(r)? as usize;
        let mut hp = HashMap::new();
        for _ in 0..n.min(64) {
            let name = read_name(r)?;
            hp.insert(name, read_u32(r)? as usize);
        }
        let get = |k: &str| {
            hp.get(k)
                .copied()
                .filter(|&v| v > 0)
                .ok_or_else(|| format!("hyper-parameter {k} missing"))
        };
        let (c, blocks, k) = (get("channels")?, get("blocks")?, get("kernel")?);
        let (hh, m, vh) = (get("honour_hidden")?, get("merge")?, get("value_hidden")?);
        if k % 2 == 0 || k > SUIT_LEN {
            return Err(format!("kernel {k} must be odd and at most {SUIT_LEN}"));
        }
        let n = read_u32(r)? as usize;
        if n > 4096 {
            return Err(format!("{n} tensors"));
        }
        let mut tensors = HashMap::new();
        for _ in 0..n {
            let name = read_name(r)?;
            let dtype = read_u32(r)?;
            let ndim = read_u32(r)? as usize;
            if ndim == 0 || ndim > 4 {
                return Err(format!("tensor {name}: {ndim} dimensions"));
            }
            let dims: Vec<usize> = (0..ndim)
                .map(|_| read_u32(r).map(|d| d as usize))
                .collect::<Result<_, _>>()?;
            let len: usize = dims.iter().product();
            if len > 64 << 20 {
                return Err(format!("tensor {name}: {len} values"));
            }
            let data = match dtype {
                0 => read_f32s(r, len)?,
                1 => {
                    let scales = read_f32s(r, dims[0])?;
                    let mut q = vec![0u8; len];
                    r.read_exact(&mut q).map_err(|e| e.to_string())?;
                    let row = len / dims[0];
                    q.iter()
                        .enumerate()
                        .map(|(i, &v)| v as i8 as f32 * scales[i / row])
                        .collect()
                }
                d => return Err(format!("tensor {name}: unknown dtype {d}")),
            };
            tensors.insert(name, Tensor { dims, data });
        }
        let mut t = Tensors(tensors);
        let net = SuitCnn {
            channels: c,
            kernel: k,
            honour1: t.dense("honour1", hh, REST_LEN)?,
            honour2: t.dense("honour2", hh, hh)?,
            stem: t.conv("stem", c, PLANES, k)?,
            blocks: (0..blocks)
                .map(|i| {
                    Ok(Block {
                        ctx: t.dense(&format!("blocks.{i}.ctx"), c, 2 * c + hh)?,
                        conv1: t.conv(&format!("blocks.{i}.conv1"), c, c, k)?,
                        conv2: t.conv(&format!("blocks.{i}.conv2"), c, c, k)?,
                    })
                })
                .collect::<Result<_, String>>()?,
            merge: t.dense("merge", m, 2 * c + hh)?,
            zproj: t.dense("zproj", c, m)?,
            suit_policy: {
                // A 1×1 convolution is a dense layer over channels.
                let w = t.take("suit_policy.weight", &[4, c, 1])?;
                Dense {
                    out: 4,
                    inp: c,
                    w,
                    b: t.take("suit_policy.bias", &[4])?,
                }
            },
            honour_policy: t.dense("honour_policy", HONOUR_POLICY, m)?,
            flush_target: t.dense("flush_target", 6, c + m)?,
            other_target: t.dense("other_target", 9, m)?,
            value1: t.dense("value1", vh, m)?,
            value2: t.dense("value2", VALUE_LEN, vh)?,
        };
        // Training-only tensors (the oracle critic) may follow; anything else is
        // a file this code does not understand.
        if let Some(extra) = t.0.keys().find(|k| !k.starts_with("oracle")) {
            return Err(format!("unexpected tensor {extra}"));
        }
        Ok(net)
    }

    /// Learnable values used at play time.
    pub fn parameters(&self) -> usize {
        let d = |l: &Dense| l.w.len() + l.b.len();
        let cv = |l: &Conv| l.w.len() + l.b.len();
        d(&self.honour1)
            + d(&self.honour2)
            + cv(&self.stem)
            + self
                .blocks
                .iter()
                .map(|b| d(&b.ctx) + cv(&b.conv1) + cv(&b.conv2))
                .sum::<usize>()
            + d(&self.merge)
            + d(&self.zproj)
            + d(&self.suit_policy)
            + d(&self.honour_policy)
            + d(&self.flush_target)
            + d(&self.other_target)
            + d(&self.value1)
            + d(&self.value2)
    }

    /// Pooled channels per suit (mean over positions), and the context input
    /// `[sum over suits, max over suits, h]`.
    fn pool(&self, y: &[Vec<f32>; 3], h: &[f32]) -> ([Vec<f32>; 3], Vec<f32>) {
        let c = self.channels;
        let pooled: [Vec<f32>; 3] = std::array::from_fn(|s| {
            let mut p = vec![0.0f32; c];
            for i in 0..SUIT_LEN {
                for (pc, v) in p.iter_mut().zip(&y[s][i * c..(i + 1) * c]) {
                    *pc += v;
                }
            }
            p.iter_mut().for_each(|v| *v /= SUIT_LEN as f32);
            p
        });
        let mut ctx = Vec::with_capacity(2 * c + h.len());
        ctx.extend((0..c).map(|j| pooled[0][j] + pooled[1][j] + pooled[2][j]));
        ctx.extend((0..c).map(|j| pooled[0][j].max(pooled[1][j]).max(pooled[2][j])));
        ctx.extend_from_slice(h);
        (pooled, ctx)
    }

    /// One forward pass: policy (134), target (45) and value (4) outputs.
    ///
    /// # Panics
    /// Panics if `features` is not [`STATE_LEN`] long.
    pub fn forward(&self, features: &[f32]) -> Vec<f32> {
        let (c, k) = (self.channels, self.kernel);
        let pad = k / 2;
        let mut suits = vec![0.0f32; SUIT_BLOCK_LEN];
        let mut rest = vec![0.0f32; REST_LEN];
        encode_planes(&features[..STATE_LEN], &mut suits, &mut rest);

        let mut h1 = vec![0.0f32; self.honour1.out];
        self.honour1.apply(&rest, &mut h1, true);
        let mut h = vec![0.0f32; self.honour2.out];
        self.honour2.apply(&h1, &mut h, true);

        // Stem, per suit (shared weights).
        let padded_len = |inp: usize| (SUIT_LEN + k - 1) * inp;
        let mut y: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; SUIT_LEN * c]);
        let mut buf = vec![0.0f32; padded_len(PLANES.max(c))];
        for s in 0..3 {
            let b = &mut buf[..padded_len(PLANES)];
            b.fill(0.0);
            b[pad * PLANES..(pad + SUIT_LEN) * PLANES]
                .copy_from_slice(&suits[s * SUIT_LEN * PLANES..(s + 1) * SUIT_LEN * PLANES]);
            self.stem.apply(b, SUIT_LEN, &mut y[s]);
            y[s].iter_mut().for_each(|v| *v = v.max(0.0));
        }

        // Residual blocks with the cross-suit context.
        let mut ctx = vec![0.0f32; c];
        let mut t1 = vec![0.0f32; SUIT_LEN * c];
        let mut t2 = vec![0.0f32; SUIT_LEN * c];
        for blk in &self.blocks {
            let (_, ctx_in) = self.pool(&y, &h);
            blk.ctx.apply(&ctx_in, &mut ctx, false);
            for ys in y.iter_mut() {
                let b = &mut buf[..padded_len(c)];
                b.fill(0.0);
                for i in 0..SUIT_LEN {
                    for j in 0..c {
                        b[(pad + i) * c + j] = (ys[i * c + j] + ctx[j]).max(0.0);
                    }
                }
                blk.conv1.apply(b, SUIT_LEN, &mut t1);
                b.fill(0.0);
                for i in 0..SUIT_LEN * c {
                    b[pad * c + i] = t1[i].max(0.0);
                }
                blk.conv2.apply(b, SUIT_LEN, &mut t2);
                for (v, d) in ys.iter_mut().zip(&t2) {
                    *v = (*v + d).max(0.0);
                }
            }
        }

        // Merge.
        let (pooled, merge_in) = self.pool(&y, &h);
        let mut z = vec![0.0f32; self.merge.out];
        self.merge.apply(&merge_in, &mut z, true);

        let mut out = vec![0.0f32; OUTPUT_LEN];
        // Policy: suited tiles.
        let mut zp = vec![0.0f32; c];
        self.zproj.apply(&z, &mut zp, false);
        let mut cell = vec![0.0f32; c];
        let mut logits = [0.0f32; 4];
        for (s, ys) in y.iter().enumerate() {
            for i in 0..SUIT_LEN {
                for j in 0..c {
                    cell[j] = (ys[i * c + j] + zp[j]).max(0.0);
                }
                self.suit_policy.apply(&cell, &mut logits, false);
                let tile = s * SUIT_LEN + i;
                out[tile] = logits[0];
                out[35 + tile] = logits[1];
                out[69 + tile] = logits[2];
                out[107 + tile] = logits[3];
            }
        }
        // Policy: honour tiles and global actions.
        let mut hp = [0.0f32; HONOUR_POLICY];
        self.honour_policy.apply(&z, &mut hp, false);
        for j in 0..HONOUR_KINDS {
            out[27 + j] = hp[j];
            out[62 + j] = hp[HONOUR_KINDS + j];
            out[96 + j] = hp[2 * HONOUR_KINDS + j];
        }
        for (g, &code) in GLOBAL_CODES.iter().enumerate() {
            out[code] = hp[3 * HONOUR_KINDS + g];
        }

        // Target.
        let target = &mut out[POLICY_LEN..POLICY_LEN + TARGET_LEN];
        target.fill(UNUSED_TARGET);
        let mut fin = Vec::with_capacity(c + z.len());
        let mut f6 = [0.0f32; 6];
        for (s, p) in pooled.iter().enumerate() {
            fin.clear();
            fin.extend_from_slice(p);
            fin.extend_from_slice(&z);
            self.flush_target.apply(&fin, &mut f6, false);
            for (j, &v) in f6.iter().enumerate() {
                let kind = 1 + j / 3;
                target[kind * 9 + s * 3 + j % 3] = v;
            }
        }
        let mut o9 = [0.0f32; 9];
        self.other_target.apply(&z, &mut o9, false);
        for (j, &v) in o9.iter().enumerate() {
            let kind = [0, 3, 4][j / 3];
            target[kind * 9 + j % 3] = v;
        }

        // Value: four relative seats, zero-sum.
        let mut vh = vec![0.0f32; self.value1.out];
        self.value1.apply(&z, &mut vh, true);
        let mut v = [0.0f32; VALUE_LEN];
        self.value2.apply(&vh, &mut v, false);
        let mean = v.iter().sum::<f32>() / VALUE_LEN as f32;
        for (o, x) in out[POLICY_LEN + TARGET_LEN..].iter_mut().zip(v) {
            *o = x - mean;
        }
        out
    }
}
