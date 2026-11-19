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

PLACEHOLDER_STEP1

## Step 2: the labels read without a model
PLACEHOLDER_STEP2

## Step 3: the model
PLACEHOLDER_STEP3

## Step 4: the bot
`stage-fold-learned` is `stage` with the `pushfold` switch: at a stage 3 discard that leaves
the hand ready, when a safer tile exists and the model predicts `push − fold` below 0, it throws
the safer tile; the hand then keeps throwing its smallest-loss tile until a stage 3 discard
leaves it ready again, when the model decides again (Task 11's fold memory, `giveup::remember`).
The switch is off by default, so `stage` plays exactly as today.

## Step 5: matches
PLACEHOLDER_STEP5

## Tests
`pushfold.rs` (5 tests): the decision needs a ready push and a safer tile (Task 11's test
position: a 3-faan half flush drawing 7p against red and green pungs; throwing West breaks the
wait, a dead North is itself the safest tile); the push and fold play-outs of one sample use the
same re-dealt tiles (the same seed gives the same wall and hands, another seed another world; a
pair replays); a position where every play-out ends the same (empty wall, nobody wins on either
tile) gives label 0 with standard error 0; `predict` matches the fit script's held-out rows;
`judge` applies only to a ready push. `stage.rs`: every existing stage test unchanged (the
switch is off); the bot names carry the switch (`stage`: off, `stage-fold-learned`: on).
