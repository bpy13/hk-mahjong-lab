# The ready hand's push-or-fold choice learned from play-outs (Task 12, 2026-11-19)

Brian, 2026-11-19, on Task 11's result (`stage3-giveup.md`: the give-up rule lost 0.63 ± 0.17
points a hand): "so the setup is incorrect just as I suspected. How to improve the give-up
role"; "you compared offensive EV to the deal-in EV. realistically the opponent if ready also
should have their own offensive EV. I think that's the part that has not been taken into
account, overestimating the deal-in EV as it should only be the difference between the deal-in
EV and the opponent offensive EV."; "sure, go ahead and train and experiment and see if it
improves rather than getting worse".

Code: `crates/search/src/pushfold.rs` (the decision, the play-out label, the model's forward
pass, the bot's verdict), `crates/search/src/pushfold_tables.rs` (the fitted weights, written by
`experiments/stage/fit_pushfold.py`), `crates/search/examples/pushfold.rs` (`label` = step 1,
`match` = step 5's detail columns), the `pushfold` switch on `StageBot` (`stage.rs`, off by
default: `stage` plays exactly as before) and the bot name `stage-fold-learned` (`bots.rs`).
Outputs: `runs\stage\t12-*.txt|csv|md` in the worker checkout `hk-mahjong-lab-2`. All runs went
directly on 12 threads at below-normal priority (the plan allows runs under 2 hours; the cpu lane
held other projects' jobs).

## Step 1: the labels
No formula for the parts. At every stage 3 decision of four `stage` bots where the chosen
discard `t` leaves the hand ready and a tile `s` with a smaller calibrated expected loss exists
(Task 11's calibration, `giveup::danger_of`), both choices are played out from the true state of
the logged deal: every seat keeps the tiles it really holds and only the order of the tiles still
in the wall is shuffled again, one shuffle per sample; the push branch throws `t`, the fold
branch throws `s`, both then play `stage` for all four seats with the same shuffle and the same
random stream, so the two play-outs differ only in my discard. (The orchestrator's correction of
its first design, 2026-11-19: re-dealing the opponents' hands at random would throw away the
hands they built, so the label would not be the real cost of that tile; averaged over many
positions that look alike from the seat, labels from true hands are the expected result given
what the seat sees. Brian: "The probability you feed is not the probability of losing with that
tile".) The folded hand keeps throwing its smallest-loss tile until a stage 3 discard leaves it
ready again, then plays `stage` as before (Task 11's after-fold behaviour). Label = mean over the
samples of (my points after the push − my points after the fold). An opponent's self-draw, a
third player's deal-in and a drawn hand land in the label by themselves.

One decision per deal is drawn at random among the deal's candidates (so a deal is not
over-represented by a long ready spell); the deal's count of candidates is kept as the row's
weight. Beside the label, the row records for every tile in the seat's hand whether an
opponent's real hand wins on it by discard and what that would cost (`pushfold::truth`): the
true risk of each tile, against which the table's calibrated danger is checked below.

Runs: `pushfold label 40000 24 1` (seed 1; at below-normal priority it was starved by the other
projects' jobs, 4,500 deals in 100 min, and the shell's 2-hour limit stopped it: 4,094 rows kept)
and `pushfold label 16000 24 2` (seed 2, normal priority, 10 threads, 32 min, 14,433 rows). 24 wall
shuffles per position; 91 % of deals have at least one candidate decision. Together 18,527
labelled positions (`t12-labels-seed1.csv`, `t12-labels-seed2.csv`). The standard error of one
position's label is 9.4 points on average (the play-outs of big hands swing by hundreds of
points); the mean over all positions is +18.78 ± 0.50 (95 %), so pushing is worth about 19 points
more than folding at the average candidate decision.

Fit, tables and check: `python experiments/stage/fit_pushfold.py runs/stage/t12-labels-seed1.csv
runs/stage/t12-labels-seed2.csv` (report `runs/stage/t12-step2.md`, weights
`crates/search/src/pushfold_tables.rs`).

## Step 2: the labels read without a model
Mean label +18.78 ± 0.50 points (push − fold, weighted by the deal's candidate decisions); mean standard error of one position's label 9.44 over 24 play-out pairs. Fold better (label < 0) in 16.5 % of positions; clearly better (label < −1.96 × its standard error) in 6.3 %. The chosen tile `t` truly deals in at 5.72 % of the decisions (mean calibrated danger 5.29 %), the safest tile `s` at 1.68 % (mean calibrated danger 0.42 %). The last column of each table is the share of decisions where `t` truly deals in.

#### The table's calibrated danger against the truth (100,223 tiles in hand at the labelled decisions)

| predicted bin | tiles | mean predicted | truly deals in | 95 % range | observed ÷ predicted | points paid per tile |
|---|---|---|---|---|---|---|
| < 0.5 % | 38,906 | 0.18 % | 0.41 % | 0.35–0.48 % | 2.34 | 0.11 |
| 0.5 %–1 % | 20,486 | 0.67 % | 0.57 % | 0.47–0.67 % | 0.86 | 0.14 |
| 1 %–2 % | 10,948 | 1.46 % | 1.49 % | 1.26–1.72 % | 1.02 | 0.37 |
| 2 %–3 % | 5,892 | 2.50 % | 2.55 % | 2.14–2.95 % | 1.02 | 0.60 |
| 3 %–5 % | 8,519 | 3.81 % | 3.97 % | 3.55–4.38 % | 1.04 | 0.97 |
| 5 %–7 % | 4,729 | 5.84 % | 6.47 % | 5.77–7.17 % | 1.11 | 1.63 |
| 7 %–10 % | 3,515 | 8.39 % | 8.99 % | 8.04–9.94 % | 1.07 | 2.16 |
| 10 %–15 % | 3,578 | 12.11 % | 12.35 % | 11.28–13.43 % | 1.02 | 2.96 |
| 15 %–25 % | 2,420 | 18.98 % | 21.32 % | 19.69–22.95 % | 1.12 | 6.52 |
| ≥ 25 % | 1,230 | 33.25 % | 30.41 % | 27.84–32.98 % | 0.91 | 11.06 |

Of all tiles, 2.88 % truly deal in; the table gives one figure to tiles that are in truth 0 or 1, so within a bin the truth is spread as the observed rate says, and no model reading only the table can tell the two apart.

#### The labels by what the seat sees

- `t` truly deals in: 1,025 positions, mean label -15.11 ± 1.43
- `t` is truly safe: 17,502 positions, mean label +20.84 ± 0.49

#### By calibrated danger of the chosen tile `t`

| bin | positions | mean label (push − fold) | 95 % range | fold better | fold clearly better | `t` truly deals in |
|---|---|---|---|---|---|---|
| < 0.5 % | 3,304 | +21.49 | ±1.17 | 10.8 % | 1.8 % | 0.2 % |
| 0.5 %–1 % | 4,036 | +21.09 | ±1.05 | 12.0 % | 2.2 % | 0.6 % |
| 1 %–2 % | 1,872 | +19.41 | ±1.48 | 14.2 % | 3.5 % | 1.7 % |
| 2 %–3 % | 1,572 | +20.45 | ±1.62 | 13.8 % | 3.7 % | 2.6 % |
| 3 %–5 % | 2,270 | +19.37 | ±1.39 | 15.4 % | 5.2 % | 4.2 % |
| 5 %–7 % | 1,298 | +17.90 | ±1.77 | 16.9 % | 6.3 % | 7.4 % |
| 7 %–10 % | 1,078 | +16.06 | ±1.85 | 20.5 % | 9.8 % | 10.0 % |
| 10 %–15 % | 1,295 | +15.21 | ±1.84 | 23.3 % | 13.0 % | 12.9 % |
| 15 %–25 % | 1,080 | +12.73 | ±2.26 | 31.0 % | 19.4 % | 23.0 % |
| ≥ 25 % | 722 | +8.38 | ±3.05 | 39.8 % | 28.4 % | 31.6 % |

#### By my hand's points on a discard win (8 = 3 faan, 16 = 4, 24–32 = 5–6, 48–64 = 7–8)

| bin | positions | mean label (push − fold) | 95 % range | fold better | fold clearly better | `t` truly deals in |
|---|---|---|---|---|---|---|
| 8 | 2,427 | +9.30 | ±0.68 | 17.3 % | 6.5 % | 5.3 % |
| 16 | 6,306 | +12.65 | ±0.51 | 16.6 % | 5.5 % | 5.2 % |
| 24–32 | 6,831 | +16.76 | ±0.64 | 17.3 % | 6.5 % | 6.1 % |
| 48–64 | 2,239 | +36.94 | ±1.97 | 14.7 % | 8.1 % | 6.6 % |
| ≥ 96 | 724 | +64.88 | ±5.35 | 11.3 % | 5.9 % | 5.5 % |

#### By live winning tiles

| bin | positions | mean label (push − fold) | 95 % range | fold better | fold clearly better | `t` truly deals in |
|---|---|---|---|---|---|---|
| 1–2 | 4,437 | +19.23 | ±1.11 | 19.9 % | 8.3 % | 7.2 % |
| 3–4 | 12,171 | +17.95 | ±0.59 | 16.0 % | 5.8 % | 5.3 % |
| 5–8 | 1,856 | +23.61 | ±1.48 | 10.8 % | 4.6 % | 5.0 % |
| 9+ | 63 | +30.03 | ±11.36 | 9.2 % | 3.1 % | 2.3 % |

#### By rounds left

| bin | positions | mean label (push − fold) | 95 % range | fold better | fold clearly better | `t` truly deals in |
|---|---|---|---|---|---|---|
| 0–2 | 1,005 | +5.61 | ±2.14 | 27.3 % | 19.8 % | 16.6 % |
| 3–5 | 2,543 | +14.56 | ±1.37 | 25.7 % | 11.2 % | 10.7 % |
| 6–8 | 4,619 | +18.11 | ±0.97 | 18.7 % | 7.0 % | 6.5 % |
| 9–11 | 5,484 | +20.57 | ±0.88 | 13.4 % | 3.9 % | 3.6 % |
| 12+ | 4,876 | +23.59 | ±0.95 | 8.7 % | 1.8 % | 1.3 % |

#### By the largest opponent ready chance (the belief's `tenpai`)

| bin | positions | mean label (push − fold) | 95 % range | fold better | fold clearly better | `t` truly deals in |
|---|---|---|---|---|---|---|
| < 10 % | 6,904 | +23.74 | ±0.84 | 10.0 % | 2.2 % | 1.2 % |
| 10–30 % | 7,337 | +17.61 | ±0.76 | 17.6 % | 6.7 % | 5.8 % |
| 30–60 % | 3,570 | +14.38 | ±1.08 | 23.0 % | 11.2 % | 11.2 % |
| ≥ 60 % | 716 | +16.45 | ±2.72 | 19.5 % | 8.2 % | 9.6 % |

#### By the largest opponent's expected points per share if it wins on a discard

| bin | positions | mean label (push − fold) | 95 % range | fold better | fold clearly better | `t` truly deals in |
|---|---|---|---|---|---|---|
| ≤ 8 | 1,175 | +24.30 | ±2.12 | 10.1 % | 1.6 % | 0.5 % |
| 9–16 | 341 | +24.48 | ±4.04 | 12.3 % | 3.3 % | 2.4 % |
| 17–32 | 10,333 | +20.64 | ±0.67 | 14.0 % | 5.0 % | 4.1 % |
| 33–64 | 2,443 | +15.59 | ±1.25 | 20.8 % | 8.6 % | 9.0 % |
| ≥ 65 | 4,235 | +15.23 | ±1.00 | 20.6 % | 9.0 % | 8.5 % |

Task 11's rule (from the row's own figures) folds in 13.4 % of these positions; the mean label where it folds is +7.13 ± 1.06 (a positive value means those folds lose points).

**Reading.** Fold is better (label below 0) in 16.5 % of the candidate decisions, clearly better
(below −1.96 standard errors) in 6.3 %: well above the plan's 2 % line, so there is value to take.
But it sits where the seat cannot see it: when `t` truly deals in (5.7 % of decisions) the label
is −15.1 ± 1.4, when it is truly safe +20.8 ± 0.5, and the table's calibrated danger, though it
is calibrated (observed ÷ predicted 0.86–1.12 from 0.5 % up; the safest bin stays 2.3× too low,
as in Task 11), gives the same figure to tiles that are in truth 0 or 1. Even at danger 25 % and
above the push still wins +8.4 ± 3.1 on average, and only at 0–2 rounds left with a cheap hand
does the mean label come near 0 (+5.6 ± 2.1). Task 11's rule, re-read from these rows, folds in
13.4 % of the decisions, and those folds lose 7.1 ± 1.1 points each: that is its −0.63 a hand.
Brian's point stands in the numbers: the deal-in's cost is far below the table's loss figure
because the opponents win anyway (the label already nets their self-draws and other deal-ins), so
a fold pays only when the tile really deals in, which the belief does not know.

## Step 3: the model
Training rows 14,818, held-out rows 3,709 (deals with number divisible by 5).

| model | held-out MSE | says fold | mean label where it folds | 95 % range | gain per decision |
|---|---|---|---|---|---|
| ridge linear | 548.0 | 2.4 % | +1.65 | ±4.73 | -0.040 |
| network 32×32 + linear skip | 534.0 | 1.2 % | -0.61 | ±5.19 | +0.008 |
| network + true deal-in of `t` and `s` (oracle, not for the bot) | 462.8 | 5.6 % | -15.63 | ±2.43 | +0.872 |
| always push (`stage`) | 1079.4 | 0.0 % | – | – | -0.000 |
| Task 11's rule (approximate) | 1084.1 | 14.0 % | +8.43 | ±2.23 | -1.178 |

`gain per decision`: points per candidate decision gained on the held-out rows by folding where the model says so (−label summed over those rows ÷ all rows' weight); the fold set must have a mean label below 0 for the model to pass.

The choice: a small network (two hidden layers of 32, ReLU, a linear skip on standardised inputs,
Adam with weight decay, early stopping on the held-out deals) beside a ridge-linear baseline, both
fitted to the label with the deal's candidate count as weight; a table or tree over 31 inputs of
this noise would need far more rows, and the network's weights go into Rust as constants
(`pushfold_tables.rs`), with a parity test on held-out rows. The network is chosen (larger
held-out gain per decision), but it barely acts: it says fold in 1.2 % of held-out positions,
with a mean label there of −0.61 ± 5.19. **The held-out check (mean label of the fold set below
0, 95 % range clear of it) is not met**: the range includes 0. The oracle row shows why: the same
network given the true deal-in of `t` and `s` folds in 5.6 % of positions at −15.6 ± 2.4 and gains
+0.87 points per candidate decision (about +0.55 a hand at 0.63 candidate decisions per hand: 51,935 over 20,500 deals), while the
network reading the table's danger gains +0.008. Of the error the bot's model could remove,
nearly all comes from the blunt danger input, none from the model's shape.

## Step 4: the bot
`stage-fold-learned` is `stage` with the `pushfold` switch: at a stage 3 discard that leaves
the hand ready, when a safer tile exists and the model predicts `push − fold` below 0, it throws
the safer tile; the hand then keeps throwing its smallest-loss tile until a stage 3 discard
leaves it ready again, when the model decides again (Task 11's fold memory, `giveup::remember`).
The switch is off by default, so `stage` plays exactly as today.

## Step 5: matches
All three on 40,000 deals, seed 1, run directly (5 threads each, side by side, about 1 h each).
`compare` plays each deal four times with the seats rotated (the plan's command; its ± treats
hands as independent); `pushfold match` (Task 11's `giveup_log match` with the same columns)
plays each deal four times with the bot in every seat and takes a deal's four seatings as one
sample. Outputs `runs\stage\t12-compare-vs-stage.txt`, `t12-compare-vs-value.txt`,
`t12-match-vs-stage.txt`.

| bot | against | tool | points per hand | win | deal-in | points paid per deal-in | ready hands that gave up | stage 3 deal-ins per 100 hands |
|---|---|---|---|---|---|---|---|---|
| `stage-fold-learned` | three `stage` | `compare` | **−0.06 ± 0.22** (the `stage` seats +0.04, −0.04, +0.06) | 23.7 % | 18.3 % | – | – | – |
| `stage-fold-learned` | three `stage` | `pushfold match` | **+0.03 ± 0.17** | 23.8 % | 17.7 % | 27.2 | 2.1 % | 7.69 |
| `stage-fold-learned` | three `value` | `compare` | **+2.64 ± 0.21** (`value` −0.81, −0.80, −1.03) | 24.9 % | 18.2 % | – | – | – |
| plain `stage` (Task 9, same deals) | three `value` | `compare` | +2.62 ± 0.21 | 25.1 % | 18.5 % | 22.2 | 0 | 8.0 (Task 11 step 1, four `stage`) |
| Task 11's `stage-fold` | three `stage` | `giveup_log match` | −0.63 ± 0.17 | 21.9 % | 17.0 % | 27.0 | 17.2 % | 7.04 |
| Task 11's `stage-fold` | three `value` | `giveup_log match` | +1.64 ± 0.19 | 23.0 % | 17.1 % | 22.5 | 16.8 % | 7.23 |

**Does `stage-fold-learned` gain points per hand over `stage` (95 % range clear of 0)? No: it is
level.** −0.06 ± 0.22 and +0.03 ± 0.17 against three `stage`, +2.64 ± 0.21 against three `value`
where plain `stage` scores +2.62 ± 0.21. It gives up in 2.1 % of its ready hands (Task 11's rule:
17 %) and loses nothing by it, which is what the held-out check predicted (a fold set with a mean
label of −0.6 ± 5.2 and a gain of +0.008 a decision). The learned choice repairs Task 11's loss
(−0.63) but finds no gain, because the gain is not visible from the seat: the labels say a fold
pays on average only when the tile truly deals in (−15.1 ± 1.4 against +20.8 ± 0.5 when it is
safe), and the belief's danger spreads one figure over tiles that are in truth 0 or 1. The oracle
with the true deal-in would gain about +0.55 a hand, so the next step, if any, is a better read
of which opponent waits on which tile (the trained belief model instead of the heuristic tables,
or the discard history as input), not another rule shape. Nothing becomes the default bot: Brian
decides.

## Tests
`pushfold.rs` (5 tests): the decision needs a ready push and a safer tile (Task 11's test
position: a 3-faan half flush drawing 7p against red and green pungs; throwing West breaks the
wait, a dead North is itself the safest tile); the push and fold play-outs of one sample use the
same re-dealt tiles (the same seed gives the same wall and hands, another seed another world; a
pair replays); a position where every play-out ends the same (empty wall, nobody wins on either
tile) gives label 0 with standard error 0; `predict` matches the fit script's held-out rows;
`judge` applies only to a ready push. `stage.rs`: every existing stage test unchanged (the
switch is off); the bot names carry the switch (`stage`: off, `stage-fold-learned`: on).
