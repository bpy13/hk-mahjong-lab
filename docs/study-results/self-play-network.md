# A self-play network that starts from the three-stage bot

Plan: `docs/plans/2026-11-16-self-play-network.md`. Baseline: `stage` (`StageBot::new(3, 1)`),
+2.54 ± 0.21 per hand against three `value` bots over 40,000 deals (Task 1).

## Task 4: imitation start (old bot, now `stage-gate0`)

**Status (2026-11-16): done, both pass marks met.** Jobs 151 (cpu lane, data, 19:50-20:00)
and 152 (gpu lane, training and the match, 20:01-20:26) ended with exit code 0. Both ran from
the worker clone `repo\.worktrees\hk-mahjong-lab-workers`; the network is
`models\stage-imitation.pt` / `.mjnn` there (not in git).

### Data
- 3,000,000 decisions of `stage`, written by the `label` example (frozen copy
  `runs\rl-stage\bin\label.exe`) into `runs\rl-stage\labels`: 30 chunks of 100,000 rows,
  seed 16, all four seats `stage`, one labelled seat per hand (hand index mod 4), only
  decisions with more than one legal move. About 20 rows per hand, so about 150,000 hands.
- Held out by hand: chunks 28 and 29 (200,000 rows); chunks hold whole hands, so no hand is
  in both sets.
- Each row carries the 933 features, the passed block (488) and the **full discard history**
  in label layout 3: 100 bytes per row (`labels::pack_history`; one byte per seat and
  discard slot, `1 + tile + 34 · from-hand + 68 · claimed`, then one overflow count per seat).
  Training unpacks it on the card to the 3,460-float `history` block, so the 2.8 M training
  rows (about 9 GB as float16) fit in the card's 16 GB.
- Checks: the Rust test `labels::tests::history_in_the_row_equals_encode_history` reads the
  history bytes of the written rows of 1,000 `stage` decisions and finds them equal to
  `features::encode_history` of the same positions (all three slot kinds occur: 摸打, 入章,
  claimed); `python/tests/test_label_history.py` checks that both Python unpackers (numpy and
  the on-card torch one) give back `encode_history` exactly on 1,000 positions of `stage`
  play, that the suit swap moves the block like the network's permutation, and that a tiny
  network trains on layout-3 labels and runs in Rust.
- Speed: 10,000 rows in 2 s on 20 threads.

### Network and training
- Suit-shared CNN (`mahjong.rl.cnn`, default size: 128 channels, 6 blocks), inputs features +
  passed + history (`--passed --history`), losses as `distill_cnn` (policy cross-entropy, all
  rows weight 1 for a bot teacher; target head; value head on the final deltas).
- 6 epochs, at most 150 minutes, batch 2,048, AdamW 2e-3 with cosine decay, random suit swap
  per row. Output: `models\stage-imitation.pt`, `-fp32.mjnn`, `.mjnn` (int8, what the match
  uses); log and `result.json` in `runs\rl-stage\imitation`.
- CPU smoke (2026-11-16, tiny settings, 20,000 rows, 39 steps): the whole path works —
  labels, training with the history block, int8 export (99.9 % top-1 agreement with fp32),
  and the network plays in `compare` against three `stage` bots.

### Results (job 152, 2026-11-16)
| Measure | Pass mark | Result | 95 % range |
|---|---|---|---|
| Held-out agreement with `stage` (200,000 decisions) | ≥ 90 % | **93.1 %** (0.93121) | ± 0.11 points (normal approximation; rows of one hand are correlated, so the true range is a little wider) |
| Network vs three `stage`, 4,000 deals | ≥ −1.0 per hand | **−0.24 per hand** | ± 0.71 (−0.95 to +0.47) |

- Training: 6 epochs of 2.8 M rows in 21.8 minutes on the card (not timed out); held-out
  agreement by epoch 87.7, 90.1, 91.7, 92.6, 92.9, 93.1 %; target head 98.6 %; 2,730,545
  parameters; int8 export agrees with fp32 on 99.5 % of 1,000 rows.
- Match (int8 network in one seat, three `stage`, seats rotated, draws 5.2 %):

  | Seat | Points per hand | Win | Deal-in | Avg faan |
  |---|---|---|---|---|
  | network | −0.24 ± 0.71 | 23.6 % | 19.1 % | 5.1 |
  | stage | +0.05 ± 0.71 | 24.1 % | 18.8 % | 5.1 |
  | stage | +0.01 ± 0.71 | 24.3 % | 18.6 % | 5.1 |
  | stage | +0.18 ± 0.70 | 24.3 % | 18.0 % | 5.1 |

- Reading: the network copies `stage` closely and plays within noise of it (its range holds
  zero); it wins a little less and deals in a little more. Task 6 starts from this network.

## Task 4b: imitation start again, on the strict bot

**Status (2026-11-17): done, both pass marks met.** Jobs 165 (cpu lane, labels, 00:20-00:28)
and 166 (gpu lane, training and the match, 00:28-00:48) ended with exit code 0, from the worker
clone. `stage` is the strict gate-1 bot (Task 9). Settings unchanged from Task 4. The network is
`models\stage-imitation.pt` / `.mjnn`; Task 4's outputs are kept as `runs\rl-stage\labels-gate0`,
`imitation-gate0` and `models\stage-imitation-gate0.*`. (The queue job log with its first line
`mahjong.__file__` was not found by the worker; the orchestrator read both jobs' results.)

| Measure | Pass mark | Task 4 (old bot) | Task 4b (strict `stage`) | 95 % range (4b) |
|---|---|---|---|---|
| Held-out agreement (200,000 decisions) | >= 90 % | 93.1 % | **93.1 %** (0.93124) | +- 0.11 points |
| Network vs three strict `stage`, 4,000 deals | >= -1.0 per hand | -0.24 | **-0.11 per hand** | +- 0.71 (-0.82 to +0.60) |

- Training: 6 epochs of 2.8 M rows in 17.7 minutes; agreement by epoch 87.4, 89.8, 91.6, 92.5,
  92.9, 93.1 %; target head 98.6 %; int8 agrees with fp32 on 99.6 % of 1,000 rows.
- Match (draws 5.7 %): network -0.11 +- 0.71 (win 23.9 %, deal-in 18.9 %, faan 5.1); the three
  `stage` seats -0.21, -0.17, +0.49 (+- 0.69, 0.70, 0.70). Inside noise of `stage`; Task 6 starts from it.
- Task 9's match did not show the strict bot clearly weaker than `stage-gate0`: no inbox note.

## Task 7: final measurement of the best network (cpu job 182, 2026-11-17)

Network: `runs/rl-stage/best.pt` (hour 6, iteration 797 of the 10.00 h run, job 167) exported
as `best.mjnn` (int8). Outputs in `runs/rl-stage/final/` (`final.ps1`, exit 0, ended 11:32).

### 40,000 deals, best network + three `stage` bots (seats rotated, draws 5.6 %)
| Seat | Points per hand | 95 % range | Win | Deal-in | Avg faan |
|---|---|---|---|---|---|
| network | **-0.09** | ± 0.22 (-0.31 to +0.13) | 23.7 % | 18.3 % | 5.1 |
| `stage` | -0.20 | ± 0.22 | 23.9 % | 18.8 % | 5.1 |
| `stage` | +0.31 | ± 0.22 | 24.1 % | 18.4 % | 5.1 |
| `stage` | -0.01 | ± 0.22 | 24.1 % | 18.5 % | 5.1 |

### 4,000 deals at one table (draws 6.2 %)
| Seat | Points per hand | 95 % range | Win | Deal-in | Avg faan |
|---|---|---|---|---|---|
| best network | +0.65 | ± 0.70 | 23.7 % | 18.6 % | 5.1 |
| teacher (imitation net) | -0.17 | ± 0.67 | 23.8 % | 18.4 % | 5.0 |
| `stage` | +1.33 | ± 0.70 | 25.0 % | 18.3 % | 5.1 |
| `value` | -1.81 | ± 0.64 | 23.0 % | 18.3 % | 4.8 |

### Deal-in breakdown (`dealin`, 20,000 deals, strict `stage` vs three `value`)
Full table in `runs/rl-stage/final/dealin.txt`. `stage` deal-in 18.52 % per hand (value 18.57 %),
22.3 points paid per deal-in (value 23.9). By stage: stage 1 discards give 9.2 % of hands a
deal-in (12.4 discards per hand), stage 3 give 8.5 % (2.5 discards per hand, 34 per 1000
discards, the riskiest), stage 2 0.8 %. By tile: honours 30.2 per 1000 discards (value 24.7), 3-7
tiles 9.4 (value 10.7).

### The 2-hourly curve (job 167, 2,000 deals each vs three `stage`, each ± 2.0)
| Hour | Iteration | Points per hand | Win | Deal-in | Draws | Self-play draws |
|---|---|---|---|---|---|---|
| 2 | 292 | -0.26 ± 2.02 | 23.0 % | 17.0 % | 5.4 % | 6.4 % |
| 4 | 556 | +0.31 ± 2.02 | 24.5 % | 17.3 % | 5.6 % | 6.4 % |
| 6 (best) | 797 | +0.48 ± 2.03 | 22.9 % | 17.2 % | 6.4 % | 6.5 % |
| 8 | 1020 | -0.54 ± 1.86 | 22.0 % | 17.1 % | 5.9 % | 6.8 % |
| 10 | 1262 | -0.88 ± 2.03 | 23.6 % | 17.6 % | 5.6 % | 6.6 % |

(From `runs/rl-stage/log.json`, filled in by the Task 8 review. `stopped.txt` says "4 results in
a row without a gain on the best (-0.26 ± 2.02 per hand)": the stall rule compares the four
newest results with the best result *before* them, which is hour 2's; `best.pt` is hour 6's.)

### Pass mark
**Not met.** The best network scores -0.09 ± 0.22 per hand against three `stage` bots over
40,000 deals: its range holds zero, so it is level with `stage`, not above it. The run's 1.29 M
hands brought no measurable gain over the imitation start (-0.11 ± 0.71).

## Low-noise run (Task 10, gpu job 252, 2026-11-19 14:44 to 18:48)
Why: the analysis `2026-11-17-self-play-analysis.md` (faults M1, M2; experiment 4) found the
10-hour run held at its start by the fading pull, its stall rule blind to any realistic gain, and
its training score (learner minus the frozen start on the same tiles) flat at about -0.7.

Settings (`experiments/stage/lownoise.ps1`, frozen copy `runs/rl-lownoise/pkg`): same start
(`models/stage-imitation.pt`, the same weights as Task 6's teacher) and pool (current 50 %, older
30 %, `stage` 20 %, a member every 2 h); pull toward the start constant at 0.02 (`--no-pull-fade`);
4,096 hands per update; one pass; learning rate 3e-5; stall rule off (`--no-stall`; draw and
collapse rules on, a result of 2,000 deals against three `stage` every hour); 4 hours of training.
Each hour's row: the mean paired difference over that hour's learner hands (raw points a hand,
95 % range = 1.96 standard errors), the mean entropy and the distance from the start (mean
`KL(start || network)` per decision), from `runs/rl-lownoise/hourly.txt`.

(Job 189, started 2026-11-17, was cancelled 28 minutes in on Brian's word; its folder is
`runs/rl-lownoise-job189`. Job 252 is a fresh run with the same settings, exit 0, 4.02 h,
161 updates, 659,682 hands.)

| Hour | Hands | Learner minus frozen start, points a hand | 95 % range | Entropy | Distance from start |
|---|---|---|---|---|---|
| 1 | 176,198 | -0.86 | ± 0.25 | 0.213 | 0.101 |
| 2 | 167,986 | -0.51 | ± 0.26 | 0.208 | 0.097 |
| 3 | 151,600 | +0.04 | ± 0.28 | 0.207 | 0.096 |
| 4 | 163,898 | +0.01 | ± 0.26 | 0.198 | 0.089 |

Meaning: the paired difference rose from -0.86 in hour 1 to level with the start (+0.04, +0.01)
in hours 3 and 4, past the -0.3 mark, with ranges that do not overlap hour 1's: once the noise is
cut, the training signal is there, and the learner climbed back to the start's level in 4 hours.
It did not go above the start. The hourly results of 2,000 deals against three `stage` bots
(-0.58, -1.11, -0.92, -1.21 per hand at hours 1 to 4, each ± 2.0; win 22.6-23.4 %, deal-in
18.4-19.2 %) are too noisy to show either change.

End network (`runs/rl-lownoise/final.mjnn`, hour 4) against three strict `stage` bots over
40,000 deals, beside Task 7's -0.09 ± 0.22: pending cpu job 253 `rl-lownoise-final` (queued
behind another project's cpu job 199; writes `runs/rl-lownoise/final/vs-stage-40000.txt`).
