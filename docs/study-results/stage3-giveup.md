# A give-up rule for the three-stage bot's ready hand (Task 11, 2026-11-19)

Brian, 2026-11-19: "can you calculate like the stage 3, let's say the probability of deal in is
accurate, then at what point one needs to give up given a bunch of scenarios"; on the rule that
came out: "ok, so if we equip this to the stage bot, what happens? can you try".

Code: `crates/search/src/giveup.rs` (the rule, figures, calibration), the `giveup` switch on
`StageBot` (`crates/search/src/stage.rs`, off by default: `stage` plays exactly as before),
bot names `stage-fold`, `stage-fold-half`, `stage-fold-double` (`bots.rs`), and
`crates/search/examples/giveup_log.rs` (`log` = step 1, `match` = step 3, `table` = the
break-even table). Outputs: `runs\stage\t11-*.txt` in the worker clone. All runs went directly
on 8 threads at below-normal priority, not through the queue (orchestrator's call,
2026-11-19: the cpu lane was held by long jobs and Brian waits on this).

## The rule
When the bot is in stage 3 and its chosen discard `t` leaves the hand ready: push when
`(1 − d) × V − loss(t) > −loss(s)`, else throw `s`, the tile with the smallest expected loss.
`d` and `loss` come from `Danger::from_belief` with the heuristic belief (the fitted tables; no
model file). `V` sums over the rounds left: (chance I draw a winning tile × 3 shares of the
self-drawn points + chance an opponent throws one × 1 share − the extra deal-in chance of my
later discard × the points it would pay), each round weighted by the chance the hand is still
going. After a give-up the hand throws its smallest-loss tile until a discard leaves it ready
again; then the rule decides again.

Corrections of the parts (mine; the shape is the orchestrator's):
* `d` and `loss` are calibrated (below): the raw chance was too low on stage 3 discards.
* The later-discard term is the ready hand's deal-in rate **minus** that of the fold's
  smallest-loss tile at the same decisions: a folded hand also discards every round.
* The points a later deal-in costs: the chosen tile's calibrated loss ÷ its calibrated chance.
* If `s` also keeps the hand ready, the fold side is `s`'s own push value; if no tile is safer
  than `t`, there is no give-up.
* Self-drawn wins carry the 自摸 faan.
* The give-up memory: the bot has none, so each give-up is stored per thread, keyed by the seat
  and the public event history (`giveup::remember`, `giveup::folded`).

## Step 1: measured before building
40,000 deals of four `stage` bots, seed 1 (fit, `t11-step1-pass1.txt`, 9 min) and seed 2
(check, `t11-step1-seed2.txt`, 13 min). A "ready round" starts at a stage 3 discard that leaves
the hand ready and ends at that seat's next discard. Deal-ins per 100 hands: 18.1 in all, 8.0 on
stage 3 discards (the plan's 12.4 of 18.6 came from the `dealin` table against `value` bots and
counts stage 3 by another cut; here only the last discard's own stage counts).

### Danger check (every stage 3 discard, `Danger::tile`, seed 1, raw)
| predicted bin | discards | mean predicted | observed | 95 % range | observed ÷ predicted |
|---|---|---|---|---|---|
| 0–0.5 % | 182,189 | 0.12 % | 0.41 % | 0.38–0.44 % | 3.58 |
| 0.5–1 % | 49,441 | 0.71 % | 0.79 % | 0.71–0.87 % | 1.11 |
| 1–2 % | 44,277 | 1.48 % | 1.98 % | 1.85–2.11 % | 1.33 |
| 2–3 % | 30,481 | 2.36 % | 3.44 % | 3.23–3.64 % | 1.45 |
| 3–5 % | 34,624 | 3.90 % | 5.87 % | 5.62–6.12 % | 1.51 |
| 5–7 % | 18,938 | 5.95 % | 10.48 % | 10.05–10.92 % | 1.76 |
| 7–10 % | 13,732 | 8.53 % | 15.84 % | 15.23–16.45 % | 1.86 |
| 10–15 % | 8,677 | 11.96 % | 25.50 % | 24.59–26.42 % | 2.13 |
| 15–25 % | 3,337 | 18.07 % | 37.25 % | 35.61–38.89 % | 2.06 |
| 25 % + | 78 | 27.63 % | 73.08 % | 63.23–82.92 % | 2.64 |

Six of ten bins are off by more than a factor 1.5 (the plan's limit), all too low, so the
danger was rescaled first: a piecewise-linear curve through (mean predicted, observed) of these
bins (`giveup::CALIBRATION`). The expected loss in points was much closer (paid ÷ predicted 0.87
to 1.24 in bins 2 to 9; 4.2 in the safest bin, 1.6 in the last): the model's chance is too low
but its points per deal-in too high for big threats, so the loss got its own factor
(`giveup::LOSS_FACTOR`). This is the reverse of the 9 Nov race bot, whose danger was too high.

Check on fresh deals (seed 2, calibrated):

| predicted bin | discards | mean predicted | observed | 95 % range | observed ÷ predicted |
|---|---|---|---|---|---|
| 0–0.5 % | 147,056 | 0.15 % | 0.40 % | 0.37–0.43 % | 2.60 |
| 0.5–1 % | 76,039 | 0.68 % | 0.57 % | 0.52–0.62 % | 0.83 |
| 1–2 % | 32,227 | 1.44 % | 1.54 % | 1.40–1.67 % | 1.07 |
| 2–3 % | 25,080 | 2.56 % | 2.28 % | 2.10–2.47 % | 0.89 |
| 3–5 % | 34,040 | 3.82 % | 3.89 % | 3.69–4.10 % | 1.02 |
| 5–7 % | 18,274 | 5.95 % | 5.90 % | 5.56–6.25 % | 0.99 |
| 7–10 % | 15,709 | 8.45 % | 7.32 % | 6.91–7.73 % | 0.87 |
| 10–15 % | 16,906 | 12.19 % | 11.92 % | 11.44–12.41 % | 0.98 |
| 15–25 % | 13,163 | 19.16 % | 19.84 % | 19.16–20.52 % | 1.04 |
| 25 % + | 7,532 | 32.83 % | 34.01 % | 32.94–35.08 % | 1.04 |

Nine of ten bins within 0.83–1.07. The safest bin stays 2.6 times too low, but the gap is
0.25 points of a percent (0.15 % predicted, 0.40 % observed); it only touches the choice
between near-safe tiles and the fold's later-discard correction, so it was left.

### Per-round figures for a ready hand (seed 2, calibrated; `giveup::MEASURED`)
"own" and "theirs" are multiples of live winning tiles ÷ unseen tiles (the rough model: 1 and 3).

| rounds left | ready rounds | live ÷ unseen | own win | win on a discard | my discard deals in | another player ends it | wall runs out | own × | theirs × | extra deal-in vs the fold |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–2 | 12,324 | 0.103 | 2.54 % | 11.98 % | 8.50 % | 12.65 % | 17.25 % | 0.25 | 1.17 | 7.85 % |
| 3–5 | 27,756 | 0.073 | 3.62 % | 13.46 % | 7.01 % | 12.43 % | 0 | 0.50 | 1.85 | 6.40 % |
| 6–8 | 43,240 | 0.058 | 3.55 % | 12.82 % | 4.90 % | 9.64 % | 0 | 0.61 | 2.20 | 4.29 % |
| 9–11 | 42,116 | 0.049 | 3.59 % | 12.51 % | 2.84 % | 6.35 % | 0 | 0.73 | 2.53 | 2.34 % |
| 12+ | 30,874 | 0.042 | 3.46 % | 12.35 % | 1.19 % | 3.15 % | 0 | 0.82 | 2.91 | 0.91 % |

Against the rough model (own 1, theirs 3, 6 % another player ends it, 2 % per later discard):
my own draws win only a quarter to four fifths as often as live ÷ unseen says (the other
players hold my tiles, and late in the hand they hold more of the unseen ones), opponents throw
my tile 1.2 to 2.9 times a round instead of 3, and late in the hand both another player's win
(12–13 %) and my own later deal-in (6–8 % above the fold's tile) are far more likely.

### Break-even danger (give up above this chance), one threatening opponent
My hand's faan on a discard × live winning tiles, against their hand's faan; unseen tiles =
4 a round + 39 in the three hands; nothing lost by the fold's tile (`giveup_log table`,
`runs\stage\t11-breakeven.txt`).

6 rounds left:

| mine | tiles | vs 3, measured | vs 5 | vs 8 | vs 10 | vs 3, rough | vs 5 | vs 8 | vs 10 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 23.3 % | 0 | 0 | 0 | 49.1 % | 21.5 % | 5.4 % | 0 |
| 3 | 4 | 41.5 % | 11.4 % | 0 | 0 | 60.1 % | 31.8 % | 12.3 % | 4.1 % |
| 3 | 8 | 55.0 % | 25.1 % | 5.5 % | 0 | 65.5 % | 37.9 % | 17.2 % | 8.0 % |
| 5 | 2 | 50.0 % | 16.8 % | 0 | 0 | 70.1 % | 42.3 % | 18.7 % | 7.3 % |
| 5 | 4 | 65.7 % | 34.7 % | 8.9 % | 0 | 78.2 % | 53.7 % | 28.6 % | 14.8 % |
| 5 | 8 | 75.4 % | 48.7 % | 22.3 % | 8.2 % | 81.8 % | 59.5 % | 34.7 % | 19.9 % |
| 8 | 2 | 76.2 % | 48.3 % | 18.5 % | 1.4 % | 87.2 % | 69.1 % | 44.2 % | 26.4 % |
| 8 | 4 | 85.1 % | 64.3 % | 36.5 % | 17.1 % | 91.2 % | 77.4 % | 55.5 % | 37.3 % |
| 8 | 8 | 89.9 % | 74.3 % | 50.4 % | 31.2 % | 92.8 % | 81.0 % | 61.2 % | 43.6 % |

2 rounds left:

| mine | tiles | vs 3, measured | vs 5 | vs 8 | vs 10 | vs 3, rough | vs 5 | vs 8 | vs 10 |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 2 | 5.6 % | 0 | 0 | 0 | 38.8 % | 15.9 % | 4.7 % | 0.7 % |
| 3 | 4 | 19.8 % | 0 | 0 | 0 | 54.1 % | 27.1 % | 10.7 % | 4.2 % |
| 3 | 8 | 36.7 % | 10.1 % | 0 | 0 | 65.2 % | 37.8 % | 17.5 % | 8.6 % |
| 5 | 2 | 25.5 % | 2.3 % | 0 | 0 | 60.5 % | 32.7 % | 13.8 % | 5.9 % |
| 5 | 4 | 44.2 % | 15.0 % | 0 | 0 | 73.6 % | 47.6 % | 24.3 % | 12.7 % |
| 5 | 8 | 61.4 % | 31.0 % | 8.4 % | 0 | 81.5 % | 59.2 % | 34.6 % | 20.1 % |
| 8 | 2 | 54.8 % | 23.9 % | 3.1 % | 0 | 81.6 % | 59.3 % | 34.4 % | 19.6 % |
| 8 | 4 | 71.2 % | 42.4 % | 16.2 % | 3.0 % | 88.9 % | 72.7 % | 49.4 % | 32.1 % |
| 8 | 8 | 82.5 % | 59.8 % | 32.5 % | 15.3 % | 92.7 % | 80.7 % | 60.9 % | 43.4 % |

The orchestrator's rough figures for comparison (6 rounds left): 16 % (3 faan, 2 tiles, against
5 faan), 7 % (against 8), 52 % (5 faan, 4 tiles, against 5), 29 % (against 8), 53 % (8 faan,
4 tiles, against 8). My rough column gives 21.5, 5.4, 53.7, 28.6 and 55.5 %: the same model,
with 自摸 worth one faan more and 63 unseen tiles. A 0 means the measured `V` is below zero: a
cheap ready hand against a big hand loses more to its own later discards than it wins, so it
should give up even a tile that is perfectly safe right now.

## Step 3: matches
`giveup_log match <bot> <opponent> 40000 1`: each deal played four times with the bot in each
seat; points per hand with a 95 % range over deals (a deal's four seatings are one sample).
Outputs `runs\stage\t11-match-*.txt`. Run time 20–40 min each on 8 threads.

| bot | against | points per hand | win | deal-in | points paid per deal-in | ready hands that gave up | stage 3 deal-ins per 100 hands |
|---|---|---|---|---|---|---|---|
| `stage-fold` | three `stage` | **−0.63 ± 0.17** | 21.9 % | 17.0 % | 27.0 | 17.2 % | 7.04 |
| `stage-fold-half` (V × 0.5) | three `stage` | −0.85 ± 0.17 | 21.3 % | 16.6 % | 26.9 | 21.7 % | 6.71 |
| `stage-fold-double` (V × 2) | three `stage` | −0.51 ± 0.17 | 22.3 % | 17.2 % | 27.0 | 14.5 % | 7.27 |
| `stage-fold` | three `value` | +1.64 ± 0.19 | 23.0 % | 17.1 % | 22.5 | 16.8 % | 7.23 |

For comparison, plain `stage`: in step 1 (four `stage` bots) 18.1 deal-ins per 100 hands, 8.0 of
them on stage 3 discards; against three `value` (Task 9, `compare`, 40,000 deals, seed 1) +2.62 ±
0.21, win 25.1 %, deal-in 18.5 %, 22.2 points paid per deal-in.

**Does the rule gain points per hand over `stage` (95 % range clear of 0)? No.** It loses
0.63 ± 0.17 a hand. Giving up in 17 % of ready hands cuts deal-ins by about 1 per 100 hands (stage
3: 8.0 to 7.0) but costs about 3 wins per 100 hands, and against `value` it scores about 1 point a
hand less than plain `stage`. The less the bot folds, the less it loses (V × 0.5: −0.85,
V × 1: −0.63, V × 2: −0.51), so the measured `V` is too low. The likely cause: the later-discard
term charges every later round with the deal-in rate of a hand that never folds, and `V` leaves
out the value of a hand that improves while ready. A version that folds less (V × 4, or `V` from
the rough figures) would be the next test; none of these bots becomes the default (Brian decides).

## Tests
`crates/search/src/giveup.rs` (4 tests): a 3-faan half flush (an exposed chow, a flower that is
not its own) waiting on West with 2 live, facing an opponent with red and green pungs at 6 rounds
left, gives up a drawn 7p (calibrated danger 14.6 %, above its break-even) and pushes a dead
North; an 8-faan full flush pushes the same 7p; the `StageBot` switch changes the move only with
the rule on and remembers the give-up for that seat only; `V` grows with tiles and rounds and
scales. `stage.rs`: every existing stage test unchanged (the switch is off), and the bot names
carry the switch (`stage`: off).
