# A self-play network that starts from the three-stage bot

Plan: `docs/plans/2026-11-16-self-play-network.md`. Baseline: `stage` (`StageBot::new(3, 1)`),
+2.54 ± 0.21 per hand against three `value` bots over 40,000 deals (Task 1).

## Task 4: imitation start

**Status (2026-11-16 19:40): data and training queued, no result yet.** Jobs 151 (cpu lane,
data) and 152 (gpu lane, after 151, training and the match). Both run from the worker clone
`repo\.worktrees\hk-mahjong-lab-workers`.

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

### Results (to be filled in when job 152 has ended)
| Measure | Pass mark | Result | 95 % range |
|---|---|---|---|
| Held-out agreement with `stage` | ≥ 90 % | pending (`result.json`: `held_out.top1_agreement`) | `held_out.top1_ci95` (normal approximation; rows of one hand are correlated, so the true range is a little wider) |
| Network vs three `stage`, 4,000 deals | ≥ −1.0 per hand | pending (`vs-stage.txt`, first line) | ± on that line |
