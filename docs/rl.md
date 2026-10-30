# Reinforcement learning

Phase 3 trains networks to play Hong Kong mahjong and compares them with the
Phase 1 bots and solver. **Status:** the full study ran on 2026-10-18;
results are in [study-results/phase3-rl.md](study-results/phase3-rl.md). Code:
`crates/search/src/features.rs` (observation encoding), `crates/search/src/net.rs`
(running networks in Rust), `crates/pybind/src/rl.rs` (batched environments,
teacher data) and `python/mahjong/rl/` (training).

## 1. Why not supervised learning first?

The strongest mahjong programs (Suphx, NAGA, Mortal, LuckyJ) are Japanese Riichi
programs that start by imitating millions of human games from Tenhou. No such
collection exists for Hong Kong rules, and Riichi games teach the wrong lessons
(no 3-faan minimum, no flowers, different scoring and claiming). So the
"human" stage is replaced by imitating the best rule-based bot (`value`), and
everything after that is self-play.

## 2. What the network sees

Features are computed in Rust from the acting seat's point of view (seats are
relative: me, next, across, previous), 933 floats:

* 26 tile planes of 34: own hand (one plane per copy count); melds, discards and
  the last three discards of each seat; unseen copies; the tile being claimed;
  the tile just drawn; scoring tiles (dragons and the two value winds); the
  heuristic danger of each tile; and, for each tile that could be discarded,
  plain and faan-aware shanten after it, plain and faan-aware acceptance after
  it, and the faan its target guarantees;
* 49 scalars: decision phase, who discarded the claimed tile, wall left, winds,
  bonus tiles, exposed melds and hand sizes, liability, each opponent's threat,
  own shanten, faan-aware shanten, target faan, whether still concealed, and the
  minimum faan.

The hand-made planes (danger, shanten and acceptance after each discard) give the network the
results of the Phase 1 analysis for free; it can learn to trust or ignore them.

**Oracle features** (136 floats: each opponent's concealed tiles and the next 16
wall tiles, as counts per kind) are appended for training only, and are zero at play time.

A test checks that two positions that differ only in hidden tiles produce
identical features, so the network cannot see what it should not.

## 3. Network

```
features (933) + oracle (136) → 512 → 512 → 256 ─┬─ q   (134)  points per action (DMC)
                                                 ├─ pi  (134)  policy logits (imitation, PPO)
                                                 └─ v   (1)    points (PPO baseline)
```

Every action is one of 134 codes (discard each tile, kongs, pass, win, pung,
chow by lowest tile), so one forward pass scores every legal action; illegal ones
are masked. Points are divided by 32 for training. After training, the torso and
one head are exported to a small binary format (`MJNN`) that Rust runs directly,
so networks play in the same duplicate tournaments as the bots and can drive
search. Every agent, networks included, always declares a legal win.

### 3.1 The suit-shared CNN (build run, plan Task 14)

`python/mahjong/rl/cnn.py` (design in its module docs; Rust: `crates/search/src/cnn.rs`,
file format MJNN v3). The 26 tile planes are cut into three suits of 9 tiles and the
7 honours. One convolution tower (kernel 3, 128 channels, 6 residual blocks) runs on
every suit with the **same weights**, so a shape learned in characters is known in dots
and bamboo. Before each block every position gets a context vector built from the sum
and the maximum over suits of each suit's pooled channels plus a dense honour-and-scalar
branch; sum and max ignore suit order, so swapping two suits in the input swaps the
outputs exactly (tests in Python and Rust).

```
features (933) ─┬─ suits 3 × (26 planes × 9) ─ stem conv ─ 6 × [context + conv + conv] ─┐
                └─ honours 26 × 7 + 49 scalars ─ 512 ─ 512 (h) ─────────── context ──────┤
                                       merge [Σ suits, max suits, h] → 512 (z) ─┬─ policy 134
                                                                                ├─ target 45
                                                                                └─ value 4 (zero-sum)
```

Heads: **policy** (the 134 action codes; per-tile actions of suited tiles from a 1×1
convolution per position, honour and global actions from `z`), **target** (45 plan
classes = plan kind 5 × suit 3 × structure 3, `labels::target_class`; flush classes per
suit; the 18 unused slots fixed at −10⁴), **value** (final points of the four relative
seats, zero-sum). An optional oracle critic (`CnnConfig(oracle=True)`) feeds the hidden
oracle block into the value head in training only; the export leaves it out.
Augmentation: `augment()` applies a random suit permutation per row to features, legal
masks, actions, target classes and the oracle block.

Measured on this PC (random weights, `python -m mahjong.rl.cnn --out models/cnn-test.mjnn
--int8`, `net_speed`, one thread, CPU shared with other jobs): **2,187,569 parameters**;
file **8.75 MB fp32, 2.22 MB int8** (one scale per output row; Rust expands int8 to
fp32 when loading); forward pass **2.1–2.5 ms**; int8 top-1 agreement with fp32 on 1,000
held-out positions 99.8 %, fp32 Rust vs PyTorch within 1e-4 (`python/tests/test_cnn.py`).

### 3.2 Search labels (plan Task 13)

`cargo run --release -p mahjong-search --example label -- --decisions 1000000 --out
runs/labels --chunk 20000 --seed 3 --teacher "<agent spec>"` plays single hands with
the teacher in one seat (hand index mod 4) and `ev:belief=models/belief.mjnn,floor=none`
in the other three, and writes one row per teacher decision with a choice: features,
oracle block, legal mask, up to 12 search candidates (mean, interval, paired difference
with the baseline and its standard error, rollout win and deal-in rates), baseline and
chosen move, the target class after the move, the final four-seat deltas and how the
hand ended for the labelled seat (layout: `crates/search/src/labels.rs`, reader
`python/mahjong/rl/labels.py`, 2,654 bytes per row). Hands are seeded from (seed,
chunk, hand), so a rerun resumes after the last complete chunk with the same rows, and
a directory refuses a different teacher. The detached job is
`experiments/task13_label.ps1` (it waits for Task 11's matches and picks the teacher).

### 3.3 Distilling the labels (plan Task 15)

`python -m mahjong.rl.distill_cnn --labels runs/labels --out runs/distill --epochs 12
--minutes 110 [--resume]` (or `python experiments/phase4_build.py --stage distill`)
trains the CNN of §3.1 on the GPU. The last label chunk is held out. All rows sit on the
GPU (features as float16, about 1.9 GB per million rows); each batch gets one random
suit permutation per row, as GPU gathers. Loss: policy cross-entropy against the
search's move, weighted by its margin (`diff / diff_se` over the best other candidate,
clipped to [1, 3]); 0.5 × target cross-entropy (rows without a feasible plan skipped);
0.5 × value MSE against the log-scaled four-seat deltas. AdamW, lr 2e-3, 300 warm-up
steps, cosine decay to 5 %, batch 2,048, gradient clip 5. A checkpoint
(`runs/distill/ckpt.pt`) every 15 minutes and after each epoch; `--resume` continues
with the same batch order. Outputs: `models/cnn-distill.pt` (PyTorch, PPO's `--init`),
`models/cnn-distill-fp32.mjnn`, `models/cnn-distill.mjnn` (int8), and
`runs/distill/result.json` (held-out top-1 agreement, Rust int8-vs-fp32 top-1 agreement
on 1,000 held-out positions, sizes). One epoch over 540 k rows took 45 s on the
RTX 4060 Ti, so 12 epochs over 1 M rows take about 15 minutes. On that slice held-out
agreement levelled off near 0.90 after about 7 epochs, while the held-out value error
rose after epoch 2 (the value target is one noisy hand result shared by about 22 rows).

`experiments/phase4_build.py` then runs `--stage evaluate` (the bare int8 network vs
three `value` bots on Task 7's 1,000 walls, `runs/matches/cnn-distill.json`) and
`--stage netpimc` (Task 16: bare network, network + PIMC-48 with full rollouts, network
+ PIMC-48 with the value head at depth 6, and belief-PIMC-64, all 24 seatings;
`runs/matches/netpimc.json` and `netpimc-verdict.{json,md}`, which apply Brian's rule:
a challenger replaces the search teacher only if challenger minus teacher has a 95 %
interval above 0). The detached job `experiments/phase4_job.ps1` waits for the labels
and runs all three stages from a frozen copy in `runs/phase4/pkg/`.

## 4. Methods

**Distillation** (`distill.py`). Hands are played by the `value` bot in all four
seats, generated in Rust in parallel, in chunks so they fit in memory. The policy
head learns the bot's action (cross-entropy), the Q head the final points after
the action taken, and the value head the final points. This network starts every
other run.

**Deep Monte Carlo** (`dmc.py`, DouZero, Zha et al. 2021). The Q head plays all
four seats, ε-greedily (ε = 0.05). Every decision of a finished hand is labelled
with the acting seat's final points, a Monte Carlo return with no bootstrapping,
and Q(state, action taken) is fitted to it by mean squared error over the last
three iterations' hands.

**PPO with oracle guiding** (`ppo.py`, Suphx, Li et al. 2020). The policy head
plays all four seats by sampling. Early in training each decision keeps the
oracle features with probability *keep*, which falls linearly from 1 to 0 over the
first half of training: the network first learns with perfect information (which
is far easier: it can see who is about to win and on which tile), then is weaned
off it. Advantages are final points minus the value estimate (rewards come only at
the end of a hand, so no discounting is needed). Clipped objective (ε = 0.2),
3 epochs per iteration, entropy bonus 0.005.

**Network + search** (`netpimc` agent). The network proposes the candidates
and its top choice is the baseline that PIMC must beat by two standard errors of
the paired difference (as in Phase 1). Rollouts are played either by the `fast`
bot or by the network itself in every seat.

## 5. Running it

```sh
python experiments/phase3_rl.py --quick     # a few minutes, checks everything runs
python experiments/phase3_rl.py             # the full study (about 5.3 hours on an M1)
```

Stages resume: a stage whose model already exists in `models/` is skipped.
