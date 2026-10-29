# The belief model (plan Task 5)

What each opponent holds, read from public information only. One model serves both
designs of the build run: the EV bot (`docs/ev-bot.md` §1, trait `Belief`) and the
belief-driven PIMC sampler (Task 7). Code: `python/mahjong/belief/` (`model.py`,
`train.py`, `evaluate.py`, `logs.py`), `crates/search/src/net.rs` (MJNN v2 loader),
tests `python/tests/test_belief.py` and `cargo test -p mahjong-search net::`.

## 1. Inputs

| Block | Size | Source |
|---|---|---|
| public features | 933 | `features::encode` (the observer's view: own hand, melds, discards per seat with the last three weighted, unseen counts, heuristic danger, shanten after each discard, scalars) |
| discard order (**off**, §5) | 3 × 8 × 34 = 816 | each opponent's last eight discards one-hot, most recent first (`net.rs::encode_discards`; the log's `discards` field for relative seats 1–3). Trained with `--discards`; the committed model does not use it |

Nothing hidden enters: the Rust test `net::tests::hidden_tiles_do_not_change_v2_outputs`
(40 mid-hand positions, each re-dealt behind the observer's back, also run on the
committed `models/belief.mjnn`) and the Python test `test_hidden_tiles_do_not_change_outputs`
check that two positions differing only in hidden tiles give identical outputs (AC3, last
clause).

## 2. Outputs (per opponent; relative seats 1 = next, 2 = across, 3 = previous)

| Head | Values | Meaning | Loss |
|---|---|---|---|
| `counts` | 34 × 5 | categorical over 0–4 copies of each kind in the concealed hand | cross-entropy, summed over kinds |
| `tenpai` | 1 | P(waiting with a legal win now) | binary cross-entropy |
| `waits` | 34 | P(a discard of that kind wins now \| tenpai), independent sigmoids | binary cross-entropy, tenpai rows only |
| `faan` | 8 | the faan of the best **discard win now** (3 … 10, the log's `opp_faan_now`) \| tenpai | cross-entropy, tenpai rows only |
| `outcome` | 3 | {no win, wins on a discard, self-draws} before the hand ends (`opp_won`) | cross-entropy |

The output vector is the heads in this order, each with its three opponents contiguous
(`counts`: opponent, kind, copies). Total 648 values. The MJNN v2 file names each head as
an output slice and `features` (and `discards` when used) as input slices, so the Rust
side looks them up by name (`Mlp::output("tenpai")`, `Mlp::forward_state(&state)`).

**For the EV bot** (`docs/ev-bot.md` §1 and §9): `sd_o` is `outcome[2]`; open question
1's `x_o` (wins on *someone's* discard before the hand ends) is now a model output,
`outcome[1]`, so `r_disc = 2.0` is no longer needed once the bot reads the model. The
faan buckets mean a discard win now (§9 Q2: keep the `+1` faan for `Pay_sd`). On the
hold-out log the observed rates are 15.7 % discard wins and 4.7 % self-draws per
opponent per hand early in the hand (ratio 3.3, not 2.0).

**Count mask.** Copies beyond what the observer has not seen are impossible; the
Python side sets those logits to `−1e9` before the softmax (`mask_counts`), in training
and at prediction. The exported network writes raw logits, so a Rust consumer applies
the same mask from `State::unseen` (Task 6's `belief.rs`).

## 3. Architecture and training

MLP: 933 → 384 → ReLU → 288 → ReLU → 648; **656,808 parameters** (≤ 1 M; the variant
with the discard block has 970,152). Adam, learning rate 1e‑3 with cosine decay to 0,
batch 2,048, 5 epochs over the 2,092,494 training rows (`runs/logs/train`, field
{`value` ×3, `defensive` ×2, `fast`, `efficiency`, one `pimc:16` seat in ten hands}).
Each epoch visits the 50 chunks in a fresh random order, four chunks (≈ 2,000 hands)
loaded and shuffled together. Checkpoints `runs/belief/ckpt-<epoch>.pt`; `--resume`;
log `runs/belief/train.log` (one JSON line per epoch with train and hold-out loss parts).
The run is reproducible: two runs of the command below gave identical losses.

```powershell
python -m mahjong.belief.train --logs runs/logs/train --out models/belief.pt --epochs 5
python -m mahjong.belief.evaluate --logs runs/logs/holdout --model models/belief.pt --out runs/belief/eval.json --bestof runs/belief/holdout-bok
cargo run --release -p mahjong-search --example selfplay_log -- --hands 2000 --field holdout --out runs/belief/holdout-bok --seed 2 --bestof 64 --threads 6
```

Machine time (2026-10-29, RTX 4060 Ti, ≈ 3 GB of GPU memory, CPU shared with a 14-worker
job): training **349 s** for 5 epochs (≈ 70 s per epoch; the plan allowed 3 GPU-h);
evaluation 10 s; the best-of-64 sidecar 42 min on 6 threads (2,000 hands, 173,567 rows).
A 15-epoch run lowered the hold-out count loss from 18.74 to 18.62 nats per opponent
hand (0.6 %) but raised the waits loss (4.61 → 4.70), so the 5-epoch model is the one kept.

Hold-out loss parts after 5 epochs (nats, per opponent; `runs/belief/train.log`):
counts 18.74 (sum over 34 kinds), tenpai 0.175, waits 4.61 (sum over 34 kinds, tenpai
rows), faan 1.61, outcome 0.583.

## 4. Evaluation (AC3) on the hold-out field

Hold-out: `runs/logs/holdout`, 435,334 rows from 5,000 hands of {`value`, `closed`,
`net:path=models/ppo.mjnn`, `defensive`}, a field the model never saw. All numbers are
in `runs/belief/eval.json` (not committed; rerun the `evaluate` command, 10 s).

### 4.1 Count log-likelihood (nats per tile kind, lower is better)

| Rows | Model | Uniform re-dealing | Best-of-64 (add-0.5 smoothing) |
|---|---|---|---|
| all 435,334 | **0.5505** | 0.6344 | — |
| the first 2,000 hands (173,567 rows) | **0.5511** | 0.6347 | 0.6623 |

Uniform re-dealing is the exact hypergeometric law of a hand drawn from the observer's
unseen tiles. Best-of-K (`world::sample_world`, 64 draws per row, each opponent the
lowest-shanten of `k` candidate hands with honours weighted 1.6) is *worse than uniform*
in likelihood: it concentrates mass on a few plausible hands, so it is often confidently
wrong about a single kind, even though its hands have realistic shanten (which is what it
was built for). **AC3's first clause holds.**

### 4.2 Calibration per wall-progress bucket (6 buckets as `danger.rs`; Wilson 95 % interval of the observed rate)

Tenpai (n = rows × 3 opponents):

| Bucket | n | Observed | Predicted | Interval | Within |
|---|---|---|---|---|---|
| 0 | 266,658 | 0.0003 | 0.0011 | [0.0002, 0.0003] | no |
| 1 | 294,696 | 0.0070 | 0.0077 | [0.0067, 0.0073] | no |
| 2 | 271,629 | 0.0426 | 0.0446 | [0.0419, 0.0434] | no |
| 3 | 227,304 | 0.1190 | 0.1331 | [0.1177, 0.1203] | no |
| 4 | 156,903 | 0.2050 | 0.2245 | [0.2030, 0.2070] | no |
| 5 | 88,812 | 0.2873 | 0.2943 | [0.2844, 0.2903] | no |

Deal-in, "the tile I discarded was won on" (346,603 discards read back from the next
row of the hand; predicted `1 − Π_o (1 − tenpai_o × wait_o(t))`):

| Bucket | n | Observed | Predicted | Interval | Within |
|---|---|---|---|---|---|
| 0 | 68,751 | 0.00000 | 0.00008 | [0.00000, 0.00006] | no |
| 1 | 77,645 | 0.0007 | 0.0007 | [0.0005, 0.0009] | yes |
| 2 | 73,409 | 0.0046 | 0.0045 | [0.0041, 0.0051] | yes |
| 3 | 61,738 | 0.0132 | 0.0138 | [0.0123, 0.0141] | yes |
| 4 | 42,150 | 0.0247 | 0.0252 | [0.0233, 0.0262] | yes |
| 5 | 22,910 | 0.0371 | 0.0346 | [0.0348, 0.0397] | no |

**AC3's calibration clause does not hold.** The model over-predicts tenpai on the
hold-out field in every bucket, by 0.0007 (bucket 1) to 0.02 (bucket 4) absolute, about
5–12 % relative past bucket 1. Deal-in is within the interval in four of six buckets; the
misses are bucket 0 (no deal-in in 68,751 early discards, the model says 1 in 10,000)
and bucket 5 (0.0346 against a lower bound of 0.0348).

Where the gap comes from. On the **training field** the same model is close to
calibrated: tenpai observed 0.0532 / predicted 0.0504 (bucket 2), 0.1331 / 0.1392 (3),
0.2206 / 0.2268 (4), 0.2890 / 0.2971 (5), so the late buckets carry a 3–5 % relative
over-prediction even in-sample, and the hold-out bots (`closed`, the PPO network) reach
tenpai less often than the training bots at the same public signals. Both effects push
the same way. The intervals are tight (±1 % relative at these counts), so "within the
interval" here means a bias under about 0.002. (Bucket 0's deal-in: predicted
0.000079 against a Wilson upper bound of 0.000056 with no event in 68,751 discards.)

Other heads, for Task 6 and the EV bot (hold-out, observed / predicted): self-draw before
the hand ends 0.047 / 0.057 (bucket 0) … 0.022 / 0.032 (bucket 5), over-predicted
throughout; discard win 0.157 / 0.157 (0), 0.149 / 0.155 (3), 0.072 / 0.097 (5); waits
among tenpai opponents 0.044 / 0.044 per kind (all buckets within 0.002).

### 4.3 What this means for the run

The plan's risk row "AC3 fails" is about the belief being *no better than best-of-K*;
here the counts are far better than both baselines and the deal-in chance is calibrated
mid-hand, while tenpai and the outcome head run 5–12 % high on an unseen field. Task 6's
`danger_check --belief` (expected loss per bucket) and Task 7's calibration example will
show whether that bias matters for play; a per-bucket temperature fitted on a *second
training-field* log (not the hold-out) would be the honest fix, and belongs to a later
revision.

## 5. The discard-order block

Trained both ways with the same seed and schedule (hold-out, 5 epochs):

| Input | Parameters | Count loss (nats per opponent hand) | Count NLL per kind | Tenpai | Waits | Faan | Outcome |
|---|---|---|---|---|---|---|---|
| features only (**kept**) | 656,808 | 18.740 | 0.5505 | 0.1754 | 4.612 | 1.610 | 0.5831 |
| + discard order | 970,152 | 18.716 | 0.5501 | 0.1738 | 4.597 | 1.603 | 0.5836 |

The block improves the hold-out count NLL by **0.13 %**, under the plan's 2 % bar, so it
is dropped: the features already hold each seat's full discard set and its last three
discards weighted by recency, which is most of the information. The code keeps the
option (`--discards`, `BeliefConfig(discards=True)`; `net.rs` fills a `discards` input
slice when a file names one) for a later revision.

## 6. In Rust: `belief.rs` and payment-aware danger (plan Task 6)

`crates/search/src/belief.rs` (`BeliefModel`) loads the MJNN v2 file, checks the five
heads, runs the network for the player to act and turns the logits into probabilities:
the count mask from `State::unseen` (as `mask_counts`), sigmoids for `tenpai` and
`waits`, softmaxes for `faan` and `outcome`. Results are cached per public state (64
entries, keyed by a hash of what the actor sees), so the danger model, the `ev` bot and
`danger_check` asking about one decision run the network once. It implements the `ev`
bot's `Belief` trait: `sd_o = outcome[2]`, `x_o = outcome[1]`, `P1`, `P2` and `E[cnt]`
from the counts, and a wait on a kind with no unseen copy set to 0. Tests:
`cargo test -p mahjong-search belief::` (probabilities, mask, cache, hidden tiles).

`Danger::from_belief(state, me, &belief)` (`danger.rs`; the old `Danger::estimate` is
unchanged) gives `tile(t) = 1 − Π_o (1 − tenpai_o × wait_o(t))`, `threat(o) = tenpai_o`,
and three new numbers: `expected_loss(t) = Σ_o P_o(t) × Pay(o) × shares(o)` in points
(shares 1, 3 when I am liable for `o`, 0 when another seat is), `points_if_win(o) =
Pay(o)` and `liability_tile(t)`: the seat with three exposed melds that could claim `t`
as its fourth (a pung when two copies are unseen, a chow for my next seat). With
`HeuristicBelief` it is the no-model fallback. `defensive:belief=models/belief.mjnn`
folds to the tile with the smallest expected loss (default `defensive` unchanged).

`cargo run --release -p mahjong-search --example danger_check -- --belief
models/belief.mjnn 2000` (2026-10-29, 111 s; seat 0 `value`, three `defensive`; every
discard by every seat; interval = 95 % normal interval of the observed mean):

| Bucket | Discards | Predicted loss (points) | Observed | Interval | Within | Deal-in predicted | Observed | Points per deal-in predicted / observed |
|---|---|---|---|---|---|---|---|---|
| 0 | 28,296 | 0.0023 | 0.0011 | [−0.0006, 0.0029] | yes | 0.009 % | 0.007 % | 23.5 / 16.0 |
| 1 | 31,535 | 0.0169 | 0.0155 | [0.0055, 0.0255] | yes | 0.070 % | 0.063 % | 21.9 / 24.4 |
| 2 | 29,178 | 0.0882 | 0.0883 | [0.0680, 0.1086] | yes | 0.39 % | 0.42 % | 21.0 / 19.9 |
| 3 | 24,136 | 0.2734 | 0.2841 | [0.2439, 0.3243] | yes | 1.17 % | 1.33 % | 21.4 / 19.5 |
| 4 | 16,232 | 0.5246 | 0.4426 | [0.3848, 0.5004] | **no** | 2.12 % | 2.12 % | 21.9 / 18.5 |
| 5 | 9,358 | 0.7686 | 0.8780 | [0.7535, 1.0025] | yes | 2.92 % | 3.52 % | 23.0 / 22.6 |

**Five of six buckets are within their interval; bucket 4 is not** (predicted 0.525
against an upper bound of 0.500), so Task 6's "each within its interval" does not hold.
The deal-in chance in bucket 4 is exact (2.12 % / 2.12 %); the miss is the hand value:
the `faan` head expects 21.9 points per deal-in where 18.5 were paid (about 344
deal-ins in that bucket). Bucket 5 shows the
other bias of §4.2: deal-ins under-predicted (2.9 % against 3.5 %). The old heuristic's
table (part 1 of the same run) over-predicts in its 4–20 % buckets.

## 7. Files

- `models/belief.pt` (PyTorch, not committed) and `models/belief.mjnn` (MJNN v2, 2.6 MB,
  committed).
- `runs/belief/eval.json`: every table above; `runs/belief/train.log`: the epoch log.
- `runs/belief/holdout-bok/`: the first 2,000 hold-out hands replayed with
  `selfplay_log --bestof 64` (the rows are byte-identical to `runs/logs/holdout`), plus
  the `bestof-NNNNN.bin` histograms the best-of-K baseline is read from.
- `runs/belief/ablation/`, `runs/belief/long/`: the discard-block and 15-epoch runs.
