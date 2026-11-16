# A self-play network that starts from the three-stage bot

Plan: `docs/plans/2026-11-16-self-play-network.md`. Baseline: `stage` (`StageBot::new(3, 1)`),
+2.54 ± 0.21 per hand against three `value` bots over 40,000 deals (Task 1).

## Task 4: imitation start

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
