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
40,000 deals (cpu job 253 `rl-lownoise-final`, exit 0, 2026-11-19 20:05;
`runs/rl-lownoise/final/vs-stage-40000.txt`), beside Task 7's best network of the 10-hour run:

| Bot | Points a hand | 95 % range | Win | Deal-in | Avg faan |
|---|---|---|---|---|---|
| Low-noise end network (`final.mjnn`) | +0.15 | ± 0.23 | 24.0 % | 18.5 % | 5.1 |
| `stage` (seat 2) | -0.01 | ± 0.22 | 24.1 % | 18.8 % | 5.1 |
| `stage` (seat 3) | -0.06 | ± 0.22 | 24.0 % | 18.5 % | 5.1 |
| `stage` (seat 4) | -0.08 | ± 0.23 | 23.9 % | 18.3 % | 5.1 |
| Task 7: 10-hour run's best (`best.mjnn`, hour 6) | -0.09 | ± 0.22 | 23.7 % | 18.3 % | 5.1 |

Draws 5.5 %. Meaning: the low-noise network is level with three `stage` bots (+0.15 ± 0.23, the
range includes 0), as Task 7's network was; it is 0.24 a hand above Task 7's figure, inside the
two ranges, so neither run beat the three-stage bot. Together with the hourly table: cutting the
noise lets the learner recover from its early drop, but 4 hours of self-play at this speed bring
it back to the start's level, not above it.

## Luck taken out of the reward (Task 29, 2026-11-23)
Why: Brian, 2026-11-23, after asking how Suphx and LuckyJ use reinforcement learning: "then try
again? with the luck removal technique?". Both earlier runs ended level with `stage`, and the
analysis (`2026-11-17-self-play-analysis.md`, fault M2) found the learning signal far smaller than
the noise of each update.

**What the earlier runs already had.** The plan's premise was that the reward was the hand's raw
points. It was not: the trainer's paired mode (idea 5a, on by default since Task 17 and on in the
10-hour run and the low-noise run) already plays every wall twice, once with the learner in a
seat and once with the frozen start network in that seat, and the reward is the learner's points
minus the start's points on the same deal with the same wall (log-scaled). That is the plan's
"same-deal baseline" (switch 2). What the runs did not have is a critic: in paired mode the
advantage of every decision of a hand was that one paired number, with nothing subtracted and no
credit per decision (`ppo_cnn.py`, `update`, before this task). So what is new here is the oracle
critic and the per-decision credit.

**Built (part 1), each a switch of `python -m mahjong.rl.ppo_cnn`, default off:**
1. `--oracle-baseline`: a scalar value head (`SuitCNN.oracle_paired`) on the same hidden layer as
   the four-seat oracle critic reads the position and the hidden view (the three concealed hands
   and the next 16 wall tiles, `features::encode_oracle`) and is trained on the paired reward; the
   advantage is the paired reward minus its value. In unpaired mode (`--no-paired-reward`) the
   four-seat critic's own-seat value is subtracted, as before; `--no-oracle-baseline` subtracts
   nothing. The policy and target heads never read the hidden view, and the export leaves every
   `oracle*` tensor out, so play never uses it (`test_luck.py`: the same weights and seed give
   the same rollout with the switches on or off; after an update with them on the exported
   network's policy equals PyTorch's within 1e-3).
2. The same-deal baseline is the paired mode itself (above): no new switch.
3. `--per-decision-credit` (a third way, Tencent's and Suphx's per-step credit in spirit): the
   advantage of a decision is the critic's value at the learner's next decision of that hand (the
   reward after its last one) minus its value now, so a decision is credited with what it
   changed, not with the whole hand. It needs the critic; its sum over a hand is the whole hand's
   advantage.
4. `--keep-hourly`: a copy of the network at every hourly row (`hour-NN.pt` / `.mjnn`).

**Measurement** (`python -m mahjong.rl.luck`, job script `experiments/stage/luck_variance.ps1`):
one fixed set of 50,000 self-play hands from the start network `models/stage-imitation.pt`,
played exactly as the trainer's rollout does (paired mode, the opponent pool with no older member
yet: current network and `stage`, min faan 3). Per learner decision the trunk output, the hidden
view, the paired reward and the learner's own points are kept. The critic heads are fitted on the
first 80 % of the hands with the trunk frozen (5 epochs; the trainer fits them online with the
trunk, so a long run's critic is at least this good), and every setting is measured on the last
20 %: the standard deviation of the advantage per decision and per hand (the mean over the hand's
decisions, the unit that adds up over an hour), in raw points where the advantage is a whole
hand's result, and the critic's held-out R². Hands per hour: the paired rollout's own timing; an
unpaired rollout of 4,000 hands for the two unpaired settings (the learner plays all four seats:
four seat-hands a hand); one update per setting timed on the last chunk. The last two columns
give the noise of one hour's mean advantage (the per-hand std over the square root of the
seat-hands an hour gives) and that noise relative to Task 10's setting: the lower, the more an
hour of training tells.

**The short of it (gpu job 312, 2026-11-23 06:20 to 06:38, exit 0).** The luck-removal parts cut
no noise. The oracle critic explains nothing of the paired reward (held-out R² −0.006 with the
hidden view, −0.002 without), so subtracting it changes no advantage and the `both` setting is
Task 10's setting (noise 0.99 of Task 10's). The per-decision row's 0.05 is an artefact of that
flat critic, not luck taken out (below). The one honest gain in the table is the unpaired
`oracle` row, 0.47 of Task 10's noise, which comes from playing the learner in all four seats
(five times the decisions an hour), not from the critic; it would cost the opponent pool (the pool
needs paired games) and the hourly learner-minus-start figure. So the 24-hour run below is, in
effect, Task 10's setting played six times longer, on the rules of Tasks 27, 31 and 32, with a
copy of the network every hour; the `oracle` setting is the candidate for a next run.

| Setting | Advantage std per decision | Std of a hand's summed advantage | Raw points per hand | Critic R² | Hands / h | Learner seat-hands / h | Decisions / h | Noise of an hour's mean | vs Task 10's |
|---|---|---|---|---|---|---|---|---|---|
| neither | 1.206 | 24.287 | 40.1 |  | 216,650 | 866,598 | 16,962,077 | 0.00133 | 0.49 |
| same-deal baseline | 1.069 | 22.092 | 38.6 |  | 169,039 | 169,039 | 3,320,953 | 0.00274 | 1.00 |
| oracle critic | 1.166 | 23.249 |  | 0.064 | 218,015 | 872,059 | 17,068,966 | 0.00127 | 0.47 |
| both | 1.072 | 22.096 |  | -0.006 | 170,950 | 170,950 | 3,358,481 | 0.00272 | 0.99 |
| both, per-decision credit | 0.247 | 1.040 |  | -0.006 | 171,016 | 171,016 | 3,359,791 | 0.00013 | 0.05 |
| oracle critic, no hidden view | 1.167 | 23.259 |  | 0.064 | 218,015 | 872,059 | 17,068,966 | 0.00127 | 0.47 |
| both, no hidden view | 1.070 | 22.087 |  | -0.002 | 170,950 | 170,950 | 3,358,481 | 0.00272 | 0.99 |
| both, per-decision credit, no hidden view | 0.240 | 1.040 |  | -0.002 | 171,016 | 171,016 | 3,359,791 | 0.00013 | 0.05 |

Rows: `neither` and `oracle critic` are unpaired (the learner in all four seats, the raw hand
result as reward, four seat-hands a hand); `same-deal baseline` is Task 10's setting; `both` adds
the critic; the `no hidden view` rows fit the same critic head on the position alone. Timing
under load from other jobs: the paired rollout 979 s for 50,037 hands, the unpaired 43 s for
4,004; one update 0.0015–0.0060 s a hand. The critic fitted on 40,028 hands with the trunk frozen
(5 epochs) and was scored on 10,008 held-out hands.

**What the table says.**
1. *The critic.* On the raw reward the hidden view explains 6.4 % of the variance; the position
   alone explains the same 6.4 %, so the three concealed hands and the next 16 wall tiles add
   nothing the position does not already say. On the paired reward (the learner's points minus
   the start network's on the same deal) the critic explains nothing at all: pairing has already
   removed what the hidden view could see. A critic that predicts a constant subtracts a constant
   from every advantage, and PPO normalises the batch, so `both` is Task 10's setting.
2. *Per-decision credit: what it credits.* The advantage of a decision is V(next own decision)
   minus V(now), with the hand's reward in place of V after the last decision (the one-step
   actor-critic estimator). The sum over a hand telescopes to reward − V(first decision): the
   hand's advantage, counted once. In the paired setting every one of the hand's 19.6 decisions
   carries the whole hand's advantage, so the "summed over a hand" column holds 19.6 copies of
   it (22.1) against one copy (1.04): that ratio, 0.047, is the 0.05, and it is bookkeeping, not
   less luck in the signal.
3. *Per-decision credit: can it bias the gradient?* Yes, and here it does. With a flat critic the
   advantage of every decision but the hand's last is V(next) − V(now) ≈ 0 plus the critic's
   noise, and the last decision carries the whole paired reward. Check: 1.072 / √19.6 = 0.242,
   the table's 0.247. The gradient would then train only the hand's last decision (the winning
   claim or the deal-in), with the earlier ones credited with nothing: the estimator's bias is
   the whole of the critic's error, and the critic has not learned anything here. The estimator
   is unbiased only when the critic is the true value, which it is not. Not run.
4. *Unpaired play.* The honest gain: 0.47–0.49 of Task 10's noise, from 4 learner seat-hands a
   hand and a faster rollout (no second game a wall); the per-hand std is a little higher (24.3
   against 22.1). It drops the pool and the paired hourly figure.

**Verdict on the setting:** `both` (paired reward, oracle critic as baseline), which is Task 10's
setting plus a critic that trains online with the trunk and can only help if it learns over the
24 hours; per-decision credit is not run (point 3). Orchestrator's call, 2026-11-23, after this
table; it was put to Brian the same day with the note that the 0.05 reported earlier was not a
real gain. The `oracle` setting is the candidate for a next run if this one does not move.

**The run (part 2).** Script `experiments/stage/luckless.ps1 -Setting both`, gpu lane, 24 hours of
training across restarts, from `models/stage-imitation.pt` and Task 10's pool, with Task 10's
other settings (pull 0.02 constant, 4,096 hands an update, one pass, learning rate 3e-5, stall
rule off), a copy of the network every hour (`runs/luckless/hour-NN.pt` / `.mjnn`), and every
hour the learner minus the frozen start on the same tiles with its 95 % range, entropy and
distance from the start (`runs/luckless/hourly.txt`). The engine is the one on main after Task 32
(two-share discard payment, 食換 ban, flower choice, ADR 0004–0007), built into this checkout and
frozen with the package. Then the end network against three strict `stage` bots over 40,000
deals (cpu lane, `-Final`). gpu job 316 `hk-luckless` (queued 2026-11-23 15:10) and cpu job 317
`hk-luckless-final` were queued for it from checkout `hk-mahjong-lab-2`.

**The `both` run was stopped at 1.3 hours.** Brian, 2026-11-23 about 16:30, after the orchestrator
laid out both settings (`both` keeps the hourly paired figure and the pool at Task 10's noise;
unpaired `oracle` has 0.47 of that noise and drops both) and recommended the switch: "so you
recommend to swtich?", then "okay". The orchestrator cancelled job 316 (it ran 15:07 to 16:28,
52 updates, 213,069 hands, 160,000 hands an hour; its one hourly row: hour 1, −1.80 ± 0.32 points
a hand against the frozen start, entropy 0.216, distance 0.104) and job 317. Its partial output is
not a result; it is kept at `runs\luckless-both-stopped` in checkout `hk-mahjong-lab-2`.

**The run redone with the `oracle` setting (slot 1, 2026-11-23).** Script
`experiments/stage/luckless.ps1 -Setting oracle` from checkout `hk-mahjong-lab-workers`: the same
start network `models/stage-imitation.pt` (byte-identical to the one job 316 used), the same other
settings (24 hours of training across restarts, pull 0.02 constant, 4,096 hands an update, one
pass, learning rate 3e-5, stall rule off, a copy of the network every hour), the engine on main
after Task 32 (plus Task 33's read-only result field), rebuilt into the checkout and frozen with
the package. What differs from Task 10's setting:
- *Unpaired reward with the oracle critic* (`--no-paired-reward --oracle-baseline`): no second
  game a wall with the frozen start; the reward is the learner's own log-scaled points and the
  advantage is the reward minus the four-seat oracle critic's own-seat value (the critic reads
  the hidden view; play never does, the test in `test_luck.py` holds).
- *What the learner plays against:* the pool needs paired games, so there is none. The learner's
  current network plays all four seats of every hand (four learner seat-hands a hand, which is
  where the 0.47 comes from); no `stage` seat and no older copy is at the table.
- *No hourly paired figure:* nothing plays the same deals as the learner, so the hourly row
  (`runs\luckless\hourly.txt`) keeps hands, entropy and distance from the start only (the trainer
  was changed to write that row, the hourly copies and `stopped.txt` in unpaired mode too; a new
  test holds it). The hourly result is instead `evaluate()`: 2,000 hands against three `value`
  bots and 2,000 against three frozen start networks (`--eval-walls 500`, in `log.json`).
- *Hourly copies matched afterwards* (`-Hourly`, cpu lane): hours 4, 8, 12, 16, 20, 24 and the
  start each plus three strict `stage` bots on the same 20,000 deals (seed 2), so the report can
  say whether the network was still rising at hour 24 (each copy's gain minus the start's).
Smoke run before queueing (`-DryRun -Setting oracle`, 6 minutes on 4 CPU threads, 256 hands an
update): exit 0, 11 updates, 2,819 hands (24,000 hands an hour on 4 CPU threads), three 2-minute "hourly" rows with n/a for the paired figure and entropy and distance filled in, copies hour-01 to hour-03 (.pt and .mjnn), two results against `value` and the frozen start, and `stopped.txt` ("time: 0.1 h of training reached") at the end; a second start on the same folder ends at once.

Jobs: gpu job 318 `hk-luckless` (queued 2026-11-23 16:44, started at once), then cpu job 319
`hk-luckless-final` and cpu job 320 `hk-luckless-hourly` after it, all in checkout
`hk-mahjong-lab-workers`. The run was paused three times on Brian's word and resumed from its last
checkpoint each time (the trainer's `--resume`; the same `-Setting oracle`): jobs 321 to 323
(2026-11-23 18:49, after "in game right now, pause mahjong for a while" / "mahjong still paused? if
so please resume"), 325 to 327 (2026-11-24 01:22, cancelled 16:15 on "pause mahjong training") and
329 to 331 (2026-11-25 01:48, "also mahjong is paused? please resume" / "Yes, resume", from
ckpt-000919 at 18.86 h). Training ended 2026-11-25 07:19 (gpu job 329, exit 0, `stopped.txt`:
"time: 24 h of training reached"): 24.01 h of training, 1,232 updates, 5,048,304 hands
(20,193,216 learner seat-hands), 233,000 hands an hour over the whole run (200,000–250,000 by
hour, under other jobs' load). One CUDA out-of-memory warning at 19:12 on 2026-11-23 (hour 1 of
job 321, the allocator's retry message; the update went through and the log has no gap). The
final match is cpu job 330 (exit 0, 1,020 s on 8 threads); the hourly-copy matches are cpu job
331 (started 2026-11-25 07:36, about 2.8 CPU h).

**Results (slot 1, 2026-11-25).**

*Hourly table* (`runs\luckless\hourly.txt`; no paired figure in unpaired mode, so the third and
fourth columns are the trainer's own `evaluate()` reading of that hour's network: 2,000 hands,
500 walls × 4 seatings, the learner in one seat against three frozen start networks, and the same
against three `value` bots, raw points a hand, each ± about 2.3; `log.json`):

| Hour | Hands | vs frozen start (± 2.3) | vs three `value` (± 2.3) | Entropy | Distance from start |
|---|---|---|---|---|---|
| 1 | 241,774 | −0.20 | +3.28 | 0.196 | 0.088 |
| 2 | 221,267 | −0.10 | +3.15 | 0.179 | 0.093 |
| 3 | 225,363 | +1.32 | +4.57 | 0.170 | 0.090 |
| 4 | 225,390 | +0.68 | +4.75 | 0.166 | 0.090 |
| 5 | 249,954 | +1.06 | +3.55 | 0.164 | 0.091 |
| 6 | 245,855 | +1.52 | +4.94 | 0.167 | 0.091 |
| 7 | 204,892 | +2.68 | +3.71 | 0.165 | 0.090 |
| 8 | 241,762 | +2.14 | +4.23 | 0.164 | 0.090 |
| 9 | 241,759 | +2.18 | +4.22 | 0.168 | 0.091 |
| 10 | 188,487 | +0.56 | +3.81 | 0.167 | 0.092 |
| 11 | 163,898 | +2.12 | +4.08 | 0.165 | 0.093 |
| 12 | 167,989 | +1.53 | +4.43 | 0.171 | 0.090 |
| 13 | 172,113 | +1.68 | +3.82 | 0.172 | 0.094 |
| 14 | 176,204 | +2.50 | +3.90 | 0.169 | 0.093 |
| 15 | 176,191 | +1.78 | +4.32 | 0.169 | 0.092 |
| 16 | 184,399 | +2.35 | +3.70 | 0.172 | 0.093 |
| 17 | 188,476 | +1.95 | +4.29 | 0.173 | 0.097 |
| 18 | 172,091 | +1.76 | +4.31 | 0.172 | 0.097 |
| 19 | 204,872 | +1.52 | +3.93 | 0.174 | 0.098 |
| 20 | 241,780 | +2.16 | +4.00 | 0.177 | 0.099 |
| 21 | 237,664 | +0.43 | +4.36 | 0.178 | 0.099 |
| 22 | 237,665 | +1.78 | +4.60 | 0.189 | 0.109 |
| 23 | 208,984 | +2.26 | +4.46 | 0.190 | 0.107 |
| 24 | 229,475 | +2.29 | +2.95 | 0.186 | 0.104 |

Reading it: each row's ± 2.3 is too wide for one hour to mean anything, but 22 of the 24 readings
against the frozen start are positive, and the means of six hours (± 0.9 each) are +0.71 (hours
1–6), +1.87 (7–12), +2.00 (13–18) and +1.74 (19–24): the network moved above its start in the
first six hours and then held about +2 a hand against it, with no clear rise after hour 7 at this
noise. The 20,000-deal paired matches of the hourly copies (job 331, below) are the figure that
decides. Entropy fell from 0.196 to 0.164 by hour 5 and rose again to 0.186–0.190 in hours 22–24;
the distance from the start went from 0.088 to 0.104, so the pull at 0.02 held it near the start
throughout. In its own four-seat self-play the learner's win rate went from 23.9 % (hour 1) to
23.2 % (hour 24) and the draw rate from 6.0 % to 8.2 %: the network plays slower, more careful
hands against itself as training goes on. The oracle critic's held-out R² on the raw reward rose
from 0.08 to 0.11 over the run (the frozen-trunk measurement gave 0.064).

*Hourly copies against the start* (`-Hourly`, cpu job 331: hours 4, 8, 12, 16, 20, 24 and the
start, each plus three strict `stage` on the same 20,000 deals, seed 2, into
`runs\luckless\hourly-match\<name>-vs-stage-20000.txt`; job 331 exit 0, 07:36 to 08:35, about
8.4 minutes a match on the CPU lane). Each row is one match: the copy in seat 1 plus three
`stage`; the gain is the copy's points minus the start's points on the same deals (the two
matches share the walls but not the play, so the gain's range is at most ± 0.56, the
independent-samples bound, and smaller in truth):

| Copy | Points a hand | 95 % range | Gain over the start | Win | Deal-in | `stage` seats |
|---|---|---|---|---|---|---|
| Start (`teacher.mjnn`, the imitation network) | −0.40 | ± 0.39 | — | 23.6 % | 18.6 % | +0.13, +0.19, +0.08 |
| Hour 4 | +0.67 | ± 0.40 | +1.07 | 24.5 % | 17.9 % | −0.32, +0.09, −0.44 |
| Hour 8 | +0.77 | ± 0.40 | +1.17 | 24.4 % | 17.6 % | −0.38, +0.06, −0.44 |
| Hour 12 | +0.47 | ± 0.39 | +0.87 | 24.5 % | 17.7 % | −0.04, −0.01, −0.42 |
| Hour 16 | +0.82 | ± 0.40 | +1.22 | 24.6 % | 17.4 % | −0.45, −0.05, −0.32 |
| Hour 20 | +1.00 | ± 0.39 | +1.40 | 24.6 % | 16.9 % | −0.35, −0.18, −0.47 |
| Hour 24 (= `final.mjnn`) | +1.19 | ± 0.39 | +1.59 | 24.8 % | 16.7 % | −0.51, −0.34, −0.35 |

Draws 5.5–6.1 %, average faan 5.1 in every seat. The hour-24 copy's +1.19 ± 0.39 on seed 2 agrees
with the 40,000-deal final match on seed 1 below (+1.21 ± 0.27). Reading it: the start network is
a little below `stage` (−0.40, the same level as Task 4b's imitation start); by hour 4 the copy
is already +1.07 over it, two thirds of the final gain, so most of the gain came in the first
four hours, as the trainer's own readings said. After that the copies rise slowly: a straight
line through the six copies climbs 0.026 points a hand per hour (95 % range 0.002 to 0.050,
from each match's ± 0.40), about +0.6 a hand per 24 hours, and the last two copies are the
highest; the deal-in rate falls almost every step, 17.9 % at hour 4 to 16.7 % at hour 24, while
the win rate barely moves (24.4–24.8 %), so the later hours bought defence, not offence. The
hour-12 dip (+0.47) is within one match's range of its neighbours. **Still rising at hour 24:
yes, slowly** (the slope's range just excludes 0; a further 24 hours at this rate would be worth
about +0.6 a hand, if the rise holds).

*End network against three strict `stage` bots, 40,000 deals* (cpu job 330, `-Final`, seats
rotated, seed 1, `runs\luckless\final\vs-stage-40000.txt`; the two-share payment rule of Tasks 31
and 32, so the points scale is larger than in Task 7's and Task 10's tables above, which were
played under the old rule):

| Bot | Points a hand | 95 % range | Win | Deal-in | Avg faan |
|---|---|---|---|---|---|
| End network (`runs\luckless\final.mjnn`, hour 24) | **+1.21** | ± 0.27 | 24.8 % | 16.6 % | 5.0 |
| `stage` (seat 2) | −0.35 | ± 0.28 | 23.5 % | 19.0 % | 5.1 |
| `stage` (seat 3) | −0.34 | ± 0.28 | 23.5 % | 18.6 % | 5.1 |
| `stage` (seat 4) | −0.52 | ± 0.28 | 23.4 % | 18.9 % | 5.1 |

Draws 6.1 %. The gain comes from both sides of the table: the network wins 1.3 points more often
than a `stage` seat (24.8 % against 23.4–23.5 %) and deals in 2.0–2.4 points less often (16.6 %
against 18.6–19.0 %), at the same average faan. For scale, under the same rule the best
hand-made defence bot, `stage-safe:t=1`, gains +0.52 ± 0.09 a hand over plain `stage` (Task 31,
`defence-false-alarms.md`); the network's gain is more than twice that. The earlier self-play
networks were level (Task 7, −0.09 ± 0.22; Task 10, +0.15 ± 0.23, old rule).

**Verdict: above `stage`.** +1.21 ± 0.27 points a hand over 40,000 deals, the range well clear of
0: the first self-play network to beat the three-stage bot. Whether the gain over the start was
still rising at hour 24: yes, slowly. The 20,000-deal matches of the hourly copies (job 331, table
above) put two thirds of the gain in the first four hours (+1.07 over the start at hour 4) and
then a climb of about 0.026 a hand per hour (95 % range 0.002 to 0.050) to +1.59 at hour 24, the
deal-in rate falling from 17.9 % to 16.7 % on the way; the trainer's own 2,000-hand readings
(± 0.9 per six hours) were too coarse to see it.
What made the difference against the two level runs: six times the training
(24 h), four learner seat-hands a hand with no second game a wall (the 0.47 noise of part 1's
table), the critic explaining a tenth of the raw reward, and the new payment rule, where a
deal-in costs double and careful play is worth more; this run cannot say which of these counts
most. Nothing is a default and no network is in the app: `final.mjnn` and `final.pt` and the
hourly copies are in `runs\luckless` of checkout `hk-mahjong-lab-workers` (not in git). What next
is Brian's question in `docs/inbox.md` (another 24 h of `oracle` from hour 24, 24 h of `both`,
stop and take it, each with its hours).

## 24 more hours of `oracle` (Task 34, queued 2026-11-27)

Brian, 2026-11-27: "honestly mahjong needs the continuation training more than big2, but I guess it
can wait behind big2 given our timeframe"; and "during the queued continuation training it should
also match against its older version like in big2 to gain a more comprehensive view of its gain
over the course of training".

Setup (slot 1):
- The run goes on from Task 29's hour-24 checkpoint (`ckpt-001232.pt`, 5,048,304 hands) with the
  same settings (`luckless.ps1 -Setting oracle`: unpaired, oracle critic, no pool, learning rate
  3e-5, pull 0.02, 4,096 hands per update, an hourly copy), in its own folder `runs\luckless48`
  (checkout `hk-mahjong-lab-workers`), so Task 29's `runs\luckless` and its `final.mjnn` (the
  network on the game server) stay as they are. Its hour-24 network is copied in as
  `hour-24.mjnn`; copies go on from `hour-25`.
- Engine: unchanged since Task 29. The only commits to `crates/` or `python/` after Task 29's
  freeze (612fdd0) touch `crates/app-api` (hk-games Tasks N1 and N4), which the trainer does not
  use; the trainer code is byte for byte the frozen one. So the run is frozen again from main
  (`-Run luckless48 -Freeze`) and plays the same game as hours 0 to 24: the figures compare
  directly.
- Resume checked on a copy of the folder (CPU, 0.02 h more): it loads the checkpoint, clears the
  time stop, trains, writes `hour-25` and stops on its new time.
- Six gpu jobs of 4 hours (`-Hours 28`, `32`, ... `48`), each after the one before, the first
  after big2's last training job (queue job 364), so other gpu jobs can pass between them.
- After each gpu job, one cpu job (`-Copy <hour>`): the copy plus three strict `stage` on Task 29's
  20,000 deals (seed 2, comparable with its hourly-copy figures), and head to head, networks only,
  against the copy 4 hours before and against hour 24: two of each, both seatings (A B A B and
  A A B B, each rotated), the same 7,500 walls (seed 4), new example `h2h` (95 % range over walls).
  Size from a 300-wall pilot (hour 24 against hour 20: +1.04 ± 1.49 a hand, so the spread of a
  wall's score is about 13 points): 7,500 walls gives about ± 0.3 a hand. After the last gpu job,
  `-Final`: the end network plus three strict `stage` on Task 29's 40,000 deals (seed 1).
- Cost: about 24.3 GPU h and 22.5 CPU h (copy jobs about 2.2 h for hour 28, 4 h for each other,
  final 0.3 h); local only, no paid API calls.

Results: RESULT-PLACEHOLDER-T34 (the table: hour, vs three `stage` (20,000 deals), vs the copy 4
hours before, vs hour 24; the end network vs three `stage` over 40,000 deals; verdict: still
rising / flat).
