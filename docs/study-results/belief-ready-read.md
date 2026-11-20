# Does the belief network read a ready hand from its discards? (plan Task 18)

Plan: `docs/plans/2026-11-16-self-play-network.md`, Task 18. Brian, 2026-11-20, on hand 2860:
"yea but you do see West discarding two dangerous tiles in a row", "If bots have no defensive
mechanism, dangerous discard should say everything about the readiness", "please investigate".
Tool: `crates/search/examples/belief_ready.rs` (`--hand` replays a saved hand; `--deals` measures
on fresh deals). No bot's default changed; no retraining.

## 1. What the network is given about an opponent's discards

`models/belief.mjnn` reads one input block, `features` (933 floats, `features::encode`; the file
names no other slice). Of an opponent's discards it holds:

* **The set of kinds thrown**, one plane per seat: count of each kind thrown ÷ 4 (so the suit
  pattern "every 筒 and 索 thrown, no 萬" is visible).
* **The last three discards**, one plane per seat, weighted 1, ⅔, ⅓ by recency.
* **The exposed sets** of each seat (one plane each, with the "exposed sets" and "hand size"
  scalars), and the **unseen copies** of each kind from the observer's view.
* **The danger table's own reading** of each opponent: the threat scalar (`Danger::threat`, the
  table cell by exposed sets × wall progress × set kind) and the per-tile danger plane. These are
  table outputs, not discard facts.
* The wall count, phase, winds, bonus tiles, liability flags, and the observer's own hand with its
  shanten and acceptance figures.

What is **missing**:

* **The order of discards** beyond the last three (the first 24 in order exist as the `history`
  block since Task 3, but this file does not name it).
* **摸打 or 入章** for any discard: whether the tile was the one just drawn, thrown straight back,
  or came out of the hand. The `passed` block carries it; this file does not use it. A run of
  drawn-and-thrown discards is the classic sign of a ready hand, and the network cannot see it.
* **Tiles thrown since the opponent's last claim** (not in any block as such; the history block
  would let a network infer it).
* **Whether a discard was dangerous when thrown**: the danger plane is the observer's danger
  now, not the thrower's at the time. The race bot patches this from outside with
  `passed::risk_scores` (the thrower's public danger of each tile it threw, discounted) and the
  multiplier `race::RISK_MODEL.tenpai` (0.79 to 1.31 by risk bin) on the `tenpai` head.
* **Which seat's open sets the thrown tile is live against** is not a feature; it is derivable
  from the planes (the tile, the other seats' sets) only if the fit learnt it.

The discard-order block (`discards`, each opponent's last eight) was tried when the model was
built and dropped for a 0.13 % gain (`docs/belief.md` §5). The fit field was `value` ×3,
`defensive` ×2, `fast`, `efficiency`, one `pimc:16` seat in ten hands, 2.09 M rows
(`docs/belief.md` §3): no `stage` bot, which did not exist then.

## 2. Hand 2860 reproduced

`belief_ready --hand C:\Users\brian\.orchestrator\notes\mahjong-flush-game-2860.json --from 50`
(the deal is `HkState(2860)`'s, SmallRng; the 80 recorded actions replay without a mismatch;
South wins 7 faan on West's 4索 at step 79, as recorded). West (seat 2) holds `12377889m GG` with
`567m` open and is **truly ready from step 50 (wall 54) to the end**. From step 50 every one of
West's eight discards (白, 1索, 2萬, 8萬, 白, 8索, 3筒, 4索) was the drawn tile thrown straight back
(摸打). South (seat 1) is ready from step 57 with three open 索 sets, four from step 72. North is
never ready.

Readings of West at the other seats' turns after West's 8索 (step 69):

| Step | Wall | Seat asking | True | Network P(ready) | Race-adjusted | Table |
|---|---|---|---|---|---|---|
| 72 | 37 | South | ready | 10.1 % | 13.2 % | 4.3 % |
| 74 | 35 | North | ready | 6.7 % | 6.1 % | 4.3 % |
| 75 | 34 | East | ready | 20.6 % | 18.8 % | 4.3 % |

Brian's note had 9 / 6 / 19 % and "tables 4–6 %": the same picture (the model file is the same,
md5 `17e05cb2…`, in both checkouts; the one-point differences come from which decision in the
turn was asked). The table cell is "1 exposed set, one suit only, bucket 3" = 4.3 %; before wall
45 it was 1.5 %.

Readings along the way (the same run, West's row): 5.9 % (step 50, North asking) … 4.9 % (step
58, right after West's 2萬) … 12.3 % (step 63, East) … 17.3 % (67) … 10.1 % (72) … 20.6 % (75)
… 10.1 % (76, South). The East seat reads West 1.5 to 3 times higher than South and North do at
the same moment: the network's reading depends on the asking seat's own hand, not only on
West's public record. Meanwhile North, never ready, with two open 索 sets and throwing honours
from hand, is read at 19 to 37 % late in the hand; South, ready with three open 索 sets, at 30 to
54 % (table 28 to 38 %), then 82 to 96 % with four.

## 3. Held-out deals of four `stage` bots

`belief_ready --deals 20000 --seed 18 --field stage` (deal seeds `18 × 1,000,003 + i`; the network
was fitted on a `value`/`defensive` field with other seeds, so none of these deals, and no deal of
`stage` at all, was in the fit). 1239514 discard decisions, 3718542 opponent rows, 365 s on 16 threads. A row is one opponent at one
discard decision (turn phase, before the discard). "True ready" is the network's own training
target: some kind would win for the opponent now (`State::wins_on`). ± is the 95 % range of the
true rate. "Race-adjusted" is the `tenpai` head times the race bot's risk multiplier
(`RISK_MODEL`), as `race:belief=models/belief.mjnn` reads it.

### Ten bins by the network's estimate

| Slice | Opponent-decisions | True ready | Network mean | Table mean | Race-adjusted mean |
|---|---|---|---|---|---|
| network 0–10 % | 3041133 | 1.97 ± 0.02 % | 1.78 % | 1.79 % | 1.76 % |
| network 10–20 % | 306240 | 21.27 ± 0.14 % | 14.28 % | 12.44 % | 14.26 % |
| network 20–30 % | 147876 | 34.57 ± 0.24 % | 24.51 % | 19.32 % | 24.69 % |
| network 30–40 % | 86197 | 45.01 ± 0.33 % | 34.60 % | 26.12 % | 34.99 % |
| network 40–50 % | 56115 | 54.39 ± 0.41 % | 44.66 % | 32.95 % | 45.38 % |
| network 50–60 % | 36117 | 61.04 ± 0.50 % | 54.60 % | 38.40 % | 55.88 % |
| network 60–70 % | 20027 | 67.47 ± 0.65 % | 64.41 % | 42.58 % | 66.62 % |
| network 70–80 % | 9118 | 78.96 ± 0.84 % | 74.38 % | 52.18 % | 77.00 % |
| network 80–90 % | 8235 | 97.26 ± 0.35 % | 85.49 % | 79.07 % | 84.00 % |
| network 90–100 % | 7484 | 99.97 ± 0.04 % | 93.23 % | 85.89 % | 87.49 % |

### Ten bins by the table's estimate

| Slice | Opponent-decisions | True ready | Network mean | Table mean | Race-adjusted mean |
|---|---|---|---|---|---|
| table 0–10 % | 3120358 | 2.37 ± 0.02 % | 2.26 % | 1.79 % | 2.26 % |
| table 10–20 % | 286099 | 23.23 ± 0.15 % | 16.77 % | 14.02 % | 16.90 % |
| table 20–30 % | 174351 | 41.66 ± 0.23 % | 29.01 % | 23.97 % | 28.90 % |
| table 30–40 % | 58855 | 56.54 ± 0.40 % | 43.34 % | 37.68 % | 45.26 % |
| table 40–50 % | 61336 | 65.07 ± 0.38 % | 53.26 % | 44.78 % | 53.11 % |
| table 50–60 % | 0 | – | – | – | – |
| table 60–70 % | 19 | 100.00 ± 0.00 % | 70.81 % | 68.81 % | 72.48 % |
| table 70–80 % | 2364 | 100.00 ± 0.00 % | 81.95 % | 74.45 % | 81.31 % |
| table 80–90 % | 14509 | 99.89 ± 0.05 % | 87.48 % | 85.63 % | 84.30 % |
| table 90–100 % | 651 | 100.00 ± 0.00 % | 91.86 % | 93.90 % | 84.89 % |

### Slices

| Slice | Opponent-decisions | True ready | Network mean | Table mean | Race-adjusted mean |
|---|---|---|---|---|---|
| all | 3718542 | 8.17 ± 0.03 % | 6.52 % | 5.44 % | 6.54 % |
| last third of the wall (≤ 30 tiles left) | 452055 | 30.84 ± 0.13 % | 24.74 % | 20.58 % | 24.12 % |
| suit-suspect: first 6 discards outside one suit, that suit thrown since | 477418 | 35.05 ± 0.14 % | 19.52 % | 16.75 % | 19.37 % |
| suit-suspect, latest discard in that suit | 200763 | 34.45 ± 0.21 % | 17.20 % | 15.31 % | 16.69 % |
| feeds a shown flush: latest discard in the suit another seat shows 3–4 open sets of | 71436 | 13.75 ± 0.25 % | 12.24 % | 10.17 % | 12.78 % |
| feeds a shown flush, thrown from hand (入章) | 24497 | 5.90 ± 0.30 % | 7.35 % | 6.13 % | 7.63 % |
| feeds a shown flush, drawn and thrown (摸打) | 46939 | 17.85 ± 0.35 % | 14.79 % | 12.28 % | 15.47 % |
| feeds a shown flush twice in a row | 31370 | 7.21 ± 0.29 % | 8.40 % | 7.26 % | 8.88 % |
| suit-suspect and feeds a shown flush | 17414 | 33.91 ± 0.70 % | 20.88 % | 17.94 % | 21.59 % |
| suit-suspect (latest in suit) and feeds a shown flush | 5589 | 25.07 ± 1.14 % | 15.82 % | 13.89 % | 16.18 % |
| last third and suit-suspect | 188395 | 47.53 ± 0.23 % | 29.02 % | 24.74 % | 28.25 % |
| last third and feeds a shown flush | 19084 | 30.07 ± 0.65 % | 24.01 % | 20.98 % | 24.55 % |
| last third, suit-suspect and feeds a shown flush | 8503 | 44.76 ± 1.06 % | 28.02 % | 24.74 % | 28.50 % |
| opponent shows 0 exposed sets | 2125290 | 0.70 ± 0.01 % | 1.17 % | 1.21 % | 1.13 % |
| opponent shows 1 exposed set | 892315 | 6.41 ± 0.05 % | 5.67 % | 3.90 % | 5.64 % |
| opponent shows 2 exposed sets | 500288 | 22.95 ± 0.12 % | 16.39 % | 12.90 % | 16.63 % |
| opponent shows 3–4 exposed sets | 200649 | 58.27 ± 0.22 % | 42.34 % | 38.37 % | 42.68 % |
| 1 exposed set, suit-suspect | 117930 | 27.77 ± 0.26 % | 10.97 % | 8.26 % | 10.83 % |
| 1 exposed set, feeds a shown flush | 21755 | 8.97 ± 0.38 % | 8.32 % | 5.96 % | 8.59 % |
| 1 exposed set, suit-suspect and feeds a shown flush | 4037 | 30.47 ± 1.42 % | 12.50 % | 9.78 % | 12.84 % |
| 1 exposed set, last third, suit-suspect and feeds a shown flush | 1995 | 40.35 ± 2.15 % | 16.36 % | 14.30 % | 16.60 % |
| last 3 discards all drawn and thrown (摸打) | 351242 | 22.51 ± 0.14 % | 14.99 % | 12.90 % | 14.35 % |
| last 5 discards all drawn and thrown | 96318 | 31.07 ± 0.29 % | 19.53 % | 16.70 % | 18.36 % |
| last 8 discards all drawn and thrown | 14989 | 40.46 ± 0.79 % | 23.62 % | 20.11 % | 21.77 % |
| 1 exposed set, last 5 drawn and thrown | 21798 | 22.01 ± 0.55 % | 11.71 % | 8.80 % | 10.81 % |
| suit-suspect, last 5 drawn and thrown | 33482 | 50.90 ± 0.54 % | 26.39 % | 23.27 % | 24.79 % |
| 1 exposed set, suit-suspect, last 5 drawn and thrown | 6732 | 48.02 ± 1.19 % | 14.39 % | 11.20 % | 13.50 % |
| 1 exposed set, suit-suspect, feeds a shown flush, last 5 drawn and thrown | 379 | 43.54 ± 4.99 % | 13.48 % | 10.80 % | 13.40 % |

Where the two sources disagree: when the network reads lower than the table (1.04 M rows) the
true rate is 8.86 % against the network's 5.54 % and the table's 7.56 %; when it reads higher
(2.68 M rows) the truth is 7.90 % against 6.91 % and 4.61 %. The network ranks better than the
table (its top two bins hold 97 to 100 % ready hands at 85 to 93 % stated) but it is low in every
bin from 10 % up, by 7 to 10 points, and the table is lower still.

## 4. The same on four `value` bots (the network's own kind of field)

`belief_ready --deals 20000 --seed 18 --field value` (1248564 discard decisions, 3745692 opponent rows, 288 s). The fit
field was `value` ×3 with `defensive`, `fast`, `efficiency` seats; four `value` bots is the nearest
single-bot field. If the network were right on its own field and wrong only on `stage`, the slices
would be calibrated here. They are not:

| Slice | `stage`: rows | true | network | table | `value`: rows | true | network | table |
|---|---|---|---|---|---|---|---|---|
| all | 3718542 | 8.17 ± 0.03 % | 6.52 % | 5.44 % | 3745692 | 8.78 ± 0.03 % | 7.00 % | 6.08 % |
| last third of the wall (≤ 30 tiles left) | 452055 | 30.84 ± 0.13 % | 24.74 % | 20.58 % | 450462 | 33.25 ± 0.14 % | 25.70 % | 21.93 % |
| opponent shows 1 exposed set | 892315 | 6.41 ± 0.05 % | 5.67 % | 3.90 % | 1196560 | 4.68 ± 0.04 % | 3.89 % | 2.64 % |
| suit-suspect: first 6 discards outside one suit, that suit thrown since | 477418 | 35.05 ± 0.14 % | 19.52 % | 16.75 % | 416188 | 38.51 ± 0.15 % | 19.96 % | 17.74 % |
| 1 exposed set, suit-suspect | 117930 | 27.77 ± 0.26 % | 10.97 % | 8.26 % | 97922 | 29.49 ± 0.29 % | 10.02 % | 7.50 % |
| feeds a shown flush: latest discard in the suit another seat shows 3–4 open sets of | 71436 | 13.75 ± 0.25 % | 12.24 % | 10.17 % | 104994 | 15.56 ± 0.22 % | 13.32 % | 11.59 % |
| suit-suspect and feeds a shown flush | 17414 | 33.91 ± 0.70 % | 20.88 % | 17.94 % | 20876 | 40.48 ± 0.67 % | 21.94 % | 19.88 % |
| last 5 discards all drawn and thrown | 96318 | 31.07 ± 0.29 % | 19.53 % | 16.70 % | 113017 | 34.03 ± 0.28 % | 20.68 % | 18.05 % |
| last 8 discards all drawn and thrown | 14989 | 40.46 ± 0.79 % | 23.62 % | 20.11 % | 19343 | 44.95 ± 0.70 % | 25.30 % | 22.21 % |
| suit-suspect, last 5 drawn and thrown | 33482 | 50.90 ± 0.54 % | 26.39 % | 23.27 % | 38307 | 56.25 ± 0.50 % | 27.07 % | 24.59 % |
| 1 exposed set, suit-suspect, last 5 drawn and thrown | 6732 | 48.02 ± 1.19 % | 14.39 % | 11.20 % | 6528 | 54.79 ± 1.21 % | 13.61 % | 10.55 % |
| 1 exposed set, suit-suspect, feeds a shown flush, last 5 drawn and thrown | 379 | 43.54 ± 4.99 % | 13.48 % | 10.80 % | 474 | 59.49 ± 4.42 % | 13.79 % | 10.74 % |

The full `value` tables are in `runs/belief-ready/value-20000.md` beside the `stage` ones
(`stage-20000.md`, not in git; reproduce with the commands above, 5 to 6 minutes each).

## 5. Verdict

**The network is too low in those slices, and West's hand is not an exception.** On 20,000
held-out deals of four `stage` bots:

* A seat whose first six discards avoided one suit and that has since thrown that suit is ready
  35 % of the time; the network says 20 %, the table 17 %. With one open set (West's case) the
  truth is 28 % against 11 % and 8 %.
* A seat whose last five discards were all the drawn tile thrown back is ready 31 % of the time
  (network 20 %); eight in a row, 40 % (network 24 %). West had done it eight times.
* Both signs together, with one open set, West's exact situation: **ready 48 % of the time;
  network 14 %, table 11 %** (6,732 rows, ± 1.2). Adding "the latest discard feeds a shown flush"
  leaves it at 44 % (379 rows, ± 5). West's 6 to 20 % was a typical reading for a situation
  where the seat is ready about half the time.
* "Throws a tile into a shown flush" on its own is **not** a sign of readiness for these bots:
  13.8 % ready (network 12 %), and only 5.9 % when the tile came from the hand (入章), because a
  seat still building its hand throws from the hand and one that is ready throws what it drew.
  Brian's reading "a dangerous discard should say everything" holds for the drawn-and-thrown
  kind (17.9 %, and 44 % with the other signs), not for every dangerous discard.
* The network is low everywhere above 10 %: in every network bin from 10 % up the true rate is 7
  to 10 points above the stated figure, on both fields, while below 10 % (82 % of rows) it is
  right. The table is lower still and has no discard input at all: its cell for West (one set,
  one suit, wall bucket 3) is 4.3 % whatever West threw.

**Cause.** Not the field: four `value` bots, the kind the network was fitted on, give the same
gaps (West's situation 55 % ready, network 14 %). Not the data count alone: the suit-suspect
slice is 13 % of all rows (477,418 here, so about 270,000 in the 2.09 M-row fit set). Two causes
with evidence:

1. **The input lacks the fact** for the drawn-and-thrown run: no 摸打/入章 flag and no order
   beyond the last three discards are in `features` (§1). The network cannot separate "five
   drawn tiles thrown back" from "five tiles thrown from the hand", and the slices show it: with
   the suit pattern fixed, the run of five raises the truth from 35 % to 51 % and the network's
   reading from 19.5 % to 26.4 % (the rise it does show comes from the wall count and set
   counts that go with it).
2. **The fit under-uses a fact it does have** for the suit pattern: the discard-set planes hold
   "nothing of one suit thrown early, that suit thrown later" exactly (order aside), yet the
   network reads 11 % where the truth is 28 % (one set, suit-suspect). The `tenpai` head is one
   output of 648 trained together, with the `counts` head's loss 100 times larger (18.7 against
   0.175 nats, `docs/belief.md` §3) through a shared 384/288 trunk for five epochs; the head is
   calibrated on average and misses the minority patterns. The reading also moves with the asking
   seat's own hand (East read West at 1.5 to 3 times South's figure in §2), which a public-record
   estimate should not do.

**Cheapest change that would fix it (not built; no retraining in this task).** A ready-state
table keyed by public discard facts, fitted from these same logs: exposed sets (0 to 4) × wall
bucket (6) × suit-suspect (yes/no) × drawn-and-thrown run (0, 1–2, 3–4, 5+) = 240 cells, filled
from 20,000 `stage` deals (3.7 M rows; the thinnest cells above hold hundreds of rows), with
smoothing toward the existing `THREAT` cell. It would read West at about 48 % instead of 4 to
20 %, needs no network, no GPU and no new input, and `Danger::estimate` already has the cell
lookup shape (`danger.rs`, `Opponent::threat_cell`). Cost: about half a worker day: the fit in
the same example, the table in `danger_tables.rs` behind a switch (off by default), a
calibration check on fresh seeds, and a 40,000-deal match of `stage` reading it against plain
`stage` (one cpu job of about 5 minutes). The dearer fix is a refit of the belief network with
the `passed` block (which carries 摸打/入章 and already loads: `--passed`, `docs/belief.md` §1)
or the `history` block, the `tenpai` loss up-weighted, on a logged `stage` field: about 25,000
logged hands (one cpu job, an hour) and a 10-minute GPU fit, plus the same checks; it would also
mend the under-reading in every bin, which the table cannot.

## 6. The strict suit collector (plan Task 18a)

Brian, 2026-11-20: "If the player is going for 萬子 and the player plays 萬子, that means it's
definitely has no irrelevant tiles and is at most 1 from ready if not already ready"; on §3's
34 %: "are you sure 34% is the correct number?"; and "just count all stage 3 hands that ended up
hitting its flush and check if it's already ready, pretty straightforward". The 34 % was right
for §3's pattern, but that pattern was loose (no open-set condition, only six early discards).
This section measures it strictly. Same run as §3 (`belief_ready --deals 20000 --seed 18 --field
stage`, 445 s on 16 threads; §3's tables came out digit for digit). "Ready" is §3's target (some
kind wins now); "one away" is the faan-aware distance (`faan_shanten`: exchanges to a ready hand
that meets the 3-faan minimum) at most 1. No 摸打/入章 split anywhere. ± is the 95 % range
(rows of one seat in one hand are not independent, so the true ranges are somewhat wider).

### 6.1 The simple count: a stage 3 flush hand throws its own suit

From the bot's own state: the seat's `StageBot` decision is stage 3 (within one step of ready,
faan-aware, after its best discard; with the default gate, no tile outside the flush), its hand
is a flush before the discard (every concealed suited tile and every suited set in one suit;
half if any honour, full if none), and the tile it throws is of that suit. "Ready" is measured
after the discard. "Readings" are the other three seats' network and table readings of it at
each one's next discard decision (before it acts again). "First own-suit discard" counts only a
discard where the seat had thrown no tile of that suit earlier in the hand.

| Discards | Count | Ready after it | Ready or one away | Readings | Network mean | Table mean |
|---|---|---|---|---|---|---|
| every stage 3 discard | 193655 | 40.6 ± 0.2 % | 100 % | 495294 | 21.8 % | 17.4 % |
| stage 3, flush hand, any tile | 130315 | 42.7 ± 0.3 % | 100 % | 329727 | 23.4 % | 18.6 % |
| **own suit: all** | 45371 | **59.8 ± 0.5 %** | 100 % | 118280 | 25.9 % | 22.6 % |
| own suit: half flush | 42847 | 58.0 ± 0.5 % | 100 % | 111773 | 25.4 % | 22.0 % |
| own suit: full flush | 2524 | 90.2 ± 1.2 % | 100 % | 6507 | 34.9 % | 33.1 % |
| own suit: 0 open sets | 2675 | 46.4 ± 1.9 % | 100 % | 6973 | 6.0 % | 6.4 % |
| own suit: 1 open set | 10968 | 49.1 ± 0.9 % | 100 % | 28435 | 11.4 % | 8.7 % |
| own suit: 2 open sets | 18694 | 56.2 ± 0.7 % | 100 % | 48770 | 21.7 % | 17.6 % |
| own suit: 3–4 open sets | 13034 | 76.6 ± 0.7 % | 100 % | 34102 | 48.1 % | 44.7 % |
| own suit: first third (61–91 left) | 1650 | 42.7 ± 2.4 % | 100 % | 4644 | 5.6 % | 8.9 % |
| own suit: middle third (31–60 left) | 25379 | 56.9 ± 0.6 % | 100 % | 68105 | 21.8 % | 19.3 % |
| own suit: last third (≤ 30 left) | 18342 | 65.3 ± 0.7 % | 100 % | 45531 | 34.1 % | 29.0 % |
| **first own-suit discard: all** | 22008 | **49.7 ± 0.7 %** | 100 % | 58318 | 21.7 % | 18.5 % |
| first: half flush | 20300 | 46.5 ± 0.7 % | 100 % | 53846 | 20.8 % | 17.5 % |
| first: full flush | 1708 | 87.6 ± 1.6 % | 100 % | 4472 | 32.9 % | 30.5 % |
| first: 0 open sets | 1605 | 38.1 ± 2.4 % | 100 % | 4224 | 5.7 % | 5.7 % |
| first: 1 open set | 6136 | 40.7 ± 1.2 % | 100 % | 16075 | 10.6 % | 7.5 % |
| first: 2 open sets | 9300 | 47.7 ± 1.0 % | 100 % | 24663 | 19.8 % | 15.5 % |
| first: 3–4 open sets | 4967 | 68.4 ± 1.3 % | 100 % | 13356 | 43.8 % | 41.3 % |
| first: first third | 1322 | 38.9 ± 2.6 % | 100 % | 3711 | 5.3 % | 8.3 % |
| first: middle third | 14215 | 49.0 ± 0.8 % | 100 % | 38318 | 19.4 % | 16.8 % |
| first: last third | 6471 | 53.5 ± 1.2 % | 100 % | 16289 | 30.9 % | 25.0 % |

**When a stage 3 flush hand throws its own suit, it is already ready 60 % of the time (50 % at
its first own-suit discard of the hand; 90 % for a full flush) and one away the rest of the
time; the other seats' network reads it at 26 % (22 % at the first), the table at 23 % (19 %).**
"Ready or one away" is 100 % by the definition of stage 3, so for these bots Brian's "at most one
from ready" holds by construction once the bot is in stage 3; the question an opponent faces is
whether the discards show that, which §6.2 measures. The gap is widest with one open set (West's
case): ready 49 %, network 11 %, table 9 %.

### 6.2 The same seen from the discards: the strict collector

A strict collector of suit x at a decision: it threw at least N tiles before its first x tile,
those hold tiles of both other suits (no x by definition), every suited open set is in x, and it
has thrown at least one x since. This uses only public facts. "Concealed all x or honours" is the
share whose concealed tiles hold no tile of the other two suits (the "no irrelevant tiles" part
of Brian's claim). Rows: opponent at another seat's discard decision (as §3).

| Rows | Count | True ready | Ready or one away | Concealed all x or honours | Network mean | Table mean |
|---|---|---|---|---|---|---|
| N = 6, all | 437724 | 38.0 ± 0.1 % | 69.8 ± 0.1 % | 76.5 ± 0.1 % | 20.8 % | 17.7 % |
| N = 6, 0 open sets | 96499 | 8.4 ± 0.2 % | 23.6 ± 0.3 % | 28.1 ± 0.3 % | 4.2 % | 4.8 % |
| N = 6, 1 open set | 113320 | 28.6 ± 0.3 % | 64.5 ± 0.3 % | 80.5 ± 0.2 % | 10.8 % | 8.0 % |
| N = 6, 2 open sets | 142865 | 45.1 ± 0.3 % | 87.7 ± 0.2 % | 93.2 ± 0.1 % | 22.6 % | 17.8 % |
| N = 6, 3–4 open sets | 85040 | 72.3 ± 0.3 % | 99.3 ± 0.1 % | 97.9 ± 0.1 % | 49.9 % | 45.2 % |
| N = 6, first third | 7259 | 15.5 ± 0.8 % | 38.6 ± 1.1 % | 52.7 ± 1.1 % | 3.2 % | 3.8 % |
| N = 6, middle third | 250390 | 30.4 ± 0.2 % | 61.0 ± 0.2 % | 70.8 ± 0.2 % | 14.8 % | 12.7 % |
| N = 6, last third | 180075 | 49.5 ± 0.2 % | 83.4 ± 0.2 % | 85.4 ± 0.2 % | 29.8 % | 25.3 % |
| N = 8, all | 328240 | 44.4 ± 0.2 % | 78.8 ± 0.1 % | 87.1 ± 0.1 % | 24.3 % | 20.5 % |
| N = 8, 0 open sets | 49510 | 14.1 ± 0.3 % | 34.2 ± 0.4 % | 45.0 ± 0.4 % | 5.3 % | 6.0 % |
| N = 8, 1 open set | 85946 | 33.3 ± 0.3 % | 71.6 ± 0.3 % | 89.2 ± 0.2 % | 12.0 % | 9.0 % |
| N = 8, 2 open sets | 118698 | 47.4 ± 0.3 % | 89.6 ± 0.2 % | 95.9 ± 0.1 % | 24.2 % | 18.9 % |
| N = 8, 3–4 open sets | 74086 | 72.7 ± 0.3 % | 99.4 ± 0.1 % | 98.7 ± 0.1 % | 51.4 % | 46.0 % |
| N = 8, first third | 254 | 40.9 ± 6.0 % | 72.8 ± 5.5 % | 78.3 ± 5.1 % | 11.5 % | 12.3 % |
| N = 8, middle third | 168695 | 37.1 ± 0.2 % | 71.4 ± 0.2 % | 83.7 ± 0.2 % | 18.0 % | 15.1 % |
| N = 8, last third | 159291 | 52.2 ± 0.2 % | 86.5 ± 0.2 % | 90.8 ± 0.1 % | 31.0 % | 26.1 % |
| **N = 10, all** | 231872 | **49.6 ± 0.2 %** | **85.5 ± 0.1 %** | 93.5 ± 0.1 % | 27.7 % | 23.0 % |
| N = 10, 0 open sets | 25317 | 21.2 ± 0.5 % | 48.9 ± 0.6 % | 64.1 ± 0.6 % | 6.5 % | 7.3 % |
| N = 10, 1 open set | 60927 | 38.2 ± 0.4 % | 78.3 ± 0.3 % | 94.3 ± 0.2 % | 13.6 % | 10.4 % |
| N = 10, 2 open sets | 87889 | 50.9 ± 0.3 % | 91.8 ± 0.2 % | 97.7 ± 0.1 % | 26.6 % | 20.3 % |
| N = 10, 3–4 open sets | 57739 | 72.3 ± 0.4 % | 99.5 ± 0.1 % | 99.1 ± 0.1 % | 53.5 % | 47.1 % |
| N = 10, middle third | 95848 | 43.2 ± 0.3 % | 80.0 ± 0.3 % | 91.9 ± 0.2 % | 21.7 % | 17.6 % |
| N = 10, last third | 136023 | 54.2 ± 0.3 % | 89.3 ± 0.2 % | 94.6 ± 0.1 % | 31.9 % | 26.7 % |

(Network and table ± are below 0.2 points in every row with more than 10,000 rows; N = 10 has one
row in the first third.)

**Brian's claim holds mostly, not always: a strict collector (N = 10) is ready or one away 85 % of
the time and ready 50 %; with two or more open sets 92 to 99.5 %; with no open set only 49 %
(a concealed hand that skipped one suit is often still building). The network reads these
seats at about half the true ready rate in every open-set bin (13.6 % against 38 % with one set).**

### 6.3 The swing at the first own-suit discard

One row per seat and hand that becomes a strict collector at its first x discard: the last
decision before that discard, the first after it, and the first after its second x discard
(fewer: not every seat throws a second).

| Rows | Count | True ready | Ready or one away | Network mean | Table mean |
|---|---|---|---|---|---|
| N = 6, before the first x | 31612 | 12.4 ± 0.4 % | 52.8 ± 0.6 % | 12.1 ± 0.2 % | 9.4 ± 0.1 % |
| N = 6, after the first x | 31612 | 33.4 ± 0.5 % | 67.9 ± 0.5 % | 16.2 ± 0.2 % | 13.6 ± 0.2 % |
| N = 6, after the second x | 16485 | 49.2 ± 0.8 % | 75.9 ± 0.7 % | 22.2 ± 0.3 % | 19.2 ± 0.3 % |
| N = 8, before the first x | 26562 | 14.3 ± 0.4 % | 59.9 ± 0.6 % | 14.2 ± 0.2 % | 10.9 ± 0.1 % |
| N = 8, after the first x | 26562 | 38.1 ± 0.6 % | 76.1 ± 0.5 % | 18.9 ± 0.2 % | 15.7 ± 0.2 % |
| N = 8, after the second x | 12944 | 56.8 ± 0.9 % | 85.2 ± 0.6 % | 26.1 ± 0.4 % | 22.3 ± 0.3 % |
| N = 10, before the first x | 21157 | 16.2 ± 0.5 % | 66.4 ± 0.6 % | 16.9 ± 0.2 % | 12.7 ± 0.2 % |
| N = 10, after the first x | 21157 | 42.5 ± 0.7 % | 82.8 ± 0.5 % | 22.2 ± 0.3 % | 18.1 ± 0.2 % |
| N = 10, after the second x | 9624 | 62.0 ± 1.0 % | 91.1 ± 0.6 % | 29.6 ± 0.4 % | 24.9 ± 0.4 % |

**The first own-suit discard lifts the true ready rate by 21 to 26 points (N = 10: 16 % → 43 %,
and 62 % after the second); the network's reading moves 4 to 5 points (17 % → 22 % → 30 %), the
table's about the same.** Before that discard the network is right (17 % against 16 %); it
misses the jump, not the level.

### 6.4 An honour from a seat that has thrown no x

A seat with 8 or more discards, tiles of both other suits among them, no x, every suited open
set in x, now throws an honour: the last decision before the honour and the first after.

| Rows | Count | True ready | Ready or one away | Network mean | Table mean |
|---|---|---|---|---|---|
| before the last third, before | 64911 | 3.6 ± 0.1 % | 31.6 ± 0.4 % | 7.4 ± 0.1 % | 6.1 ± 0.1 % |
| before the last third, after | 64911 | 13.3 ± 0.3 % | 55.1 ± 0.4 % | 12.6 ± 0.1 % | 8.7 ± 0.1 % |
| in the last third, before | 21210 | 6.3 ± 0.3 % | 41.5 ± 0.7 % | 16.9 ± 0.2 % | 13.7 ± 0.1 % |
| in the last third, after | 21210 | 17.1 ± 0.5 % | 63.5 ± 0.6 % | 23.8 ± 0.2 % | 17.0 ± 0.2 % |
| all, before | 86121 | 4.2 ± 0.1 % | 34.1 ± 0.3 % | 9.8 ± 0.1 % | 8.0 ± 0.1 % |
| all, after | 86121 | 14.2 ± 0.2 % | 57.2 ± 0.3 % | 15.4 ± 0.1 % | 10.7 ± 0.1 % |

The honour is a real but smaller clue: ready 4 % → 14 %, ready or one away 34 % → 57 % (98 % of
these seats already hold only x and honours). Here the network is about right after the honour
(15 % against 14 %) and too high before it; the table barely moves.

### 6.5 Hand 2860 under these rows

(`belief_ready --hand … --from 0`, output `runs/belief-ready/hand-2860-t18a.txt`.)

* **West's 2萬 (step 57, wall 49)**: West had thrown 10 tiles first (6索 5筒 7筒 8筒 東 6筒 3筒 4索
  白 1索), both other suits, no 萬, its one open set `567m`: a strict collector at N = 6, 8 and
  10 from this discard, one open set, middle third. Its rows: ready 38 %, ready or one away 78 %
  (N = 10, one set); the swing row 16 % → 43 %. West was truly ready (since step 24). The
  network read it at 5.4 % just before and 4.9 / 9.2 / 5.9 % after; the table stayed at 1.5 %.
* **West's 8萬 (step 61, wall 45)**: the second x discard; swing row 62 % ready (N = 10). Readings
  after it 4.5 / 12.3 / 7.1 %; the table moved to 4.3 % (its wall bucket, not the discard).
* **South's 3索 (step 72, wall 37)**: South had thrown 14 tiles before its first 索, both other
  suits; all four open sets in 索 (it had just claimed 8索 for the fourth): N = 10, 3–4 open sets,
  ready 72 %, ready or one away 99.5 %. South was ready; the network read it at 82 to 96 %
  (table 88 %), up from 46 to 54 % before, mostly from the fourth open set.

West is the case the network misses: a strict collector with one open set, read at 5 to 12 %
where such seats are ready 38 % of the time (and West's own record, 10 clean discards then two
萬, puts it in the 62 % row).

## Files

* `crates/search/examples/belief_ready.rs`: the tool (`--hand`, `--deals`, `--field`; Task 18a's
  tables print after §3's for `--field stage`).
* `runs/belief-ready/` (not in git): `hand-2860.txt`, `stage-20000.md`, `value-20000.md`,
  `stage-20000-t18a.md` (all tables of §3 and §6), `hand-2860-t18a.txt`.
* No model, bot, table or default changed.


