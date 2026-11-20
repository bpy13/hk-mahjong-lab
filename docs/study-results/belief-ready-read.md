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

## 7. Retrained on the discard order (plan Task 19)

Brian, 2026-11-20: "you don't care if South plays 3s or if West plays 2w and 8w, those are
obvious clues that should swing the estimation massively. Less obvious clues includes honour
discards"; 摸打 against 入章 is "completely irrelevant, especially in this mobile app setting".
This section builds the model that §5 asked for, fits it on fresh games of four `stage` bots and
checks it on the same 20,000 held-out deals as §3 and §6. Nothing becomes a default: the old
`models/belief.mjnn` stays the reading of every bot; the new file is `models/belief-order.mjnn`
(not in git; it is in the workers clone, 11 MB).

### 7.1 What was built

* **Logs.** `selfplay_log --field stage` writes rows of layout 3: besides layout 2's fields, the
  packed discard and claim order (`order`: every seat's first 24 discards in the order thrown,
  the tile and whether it was claimed, plus an overflow count; nothing says whether a tile was
  drawn and thrown or thrown from the hand), each opponent's state (far / one away / ready, the
  faan-aware distance), the kinds it could claim into a meld now, the hand's winner and how it
  won, and the observer's net points for the hand. New since Task 18: rows are also written from
  the discarder's view when it is not to act (right after its discard, after the claim window
  closed without a claim, after an opponent claimed), so the model can be read on the position
  after a discard, which Task 20's bot needs. Training log: 40,000 hands, seed 190, 6,236,796
  rows (156 a hand), 20 GB, 1 h 51 min on 14 threads (1.45 s a hand on one thread, most of it
  the labels). Hold-out log: 4,000 hands, seed 191, 623,958 rows. Both in `runs/belief-order/`
  of the workers clone.
* **Inputs.** The 933 public `features` as before, the `order` block of Task 3 with the 摸打 /
  入章 scalar set to 1 for every discard (`features::encode_order`, 3,460 values), and, in the
  chosen model, a per-seat suit summary of the same order (`features::encode_suits`, 56 values:
  per seat its discard count, the share of each suit and of honours among its discards, whether
  each suit has been thrown, the index of its first discard of each suit, and each suit's count
  among its last five discards). The summary is computed from the logged `order` field
  (`log::suits_from_packed`, `model.suits_block`), so no new logging was needed; a test checks
  the Rust and Python values on a hand-made discard list, and `log::tests::order_and_claims_in_the_row`
  checks the live encoder against the packed field on played hands. No input tells a drawn tile
  from a hand tile (`features::tests::spectator_encoding_hides_the_draw_and_the_decision`,
  `order_block_hides_how_a_tile_was_thrown`, `suits_block_summarises_each_seat`).
* **Outputs** (Brian's list, 2026-11-20): per opponent P(ready) (`tenpai`, 3), the state
  (`state`, 3 × 3), the faan spread of its win (`faan`, 3 × 8), the win chances (`winner`, 9
  classes: each seat self-drawn or off a discard, and the draw; Brian's five numbers are the
  pairs summed, `BeliefOutput::win_chances`), per tile and opponent the deal-in chance
  (`dealin`, 3 × 34) and the claim chance (`claim`, 3 × 34; merged to his 34 numbers by
  `BeliefOutput::dealin_or_claim`), the expected points for the hand (`points`, 1), plus the old
  `waits` and `outcome` heads and, as a training aid only, the copies-per-kind head `counts`
  (3 × 34 × 5). The exported file names every slice; `crates/search/src/belief.rs` reads them
  (`BeliefModel::has_task19_heads`, `run_for` for the spectator view).
* **Fit.** One MLP, hidden 512 and 384 (ReLU), Adam at 1e-3 with a cosine schedule, batch
  2,048, each epoch a fresh pass over the 80 chunks in random order; hold-out by whole deals.
  An epoch takes 60 s on the GPU with the log in the page cache and about 500 s when it is read
  from disk again. Five fits (`python -m mahjong.belief.train --order --task19 ...`):

| Fit | Inputs, heads, settings | Epochs | Held-out ready (nats) | state | winner | dealin | claim |
|---|---|---|---|---|---|---|---|
| constant / frequencies | the hold-out log's own rates | – | 0.2502 | 0.676 | 2.041 | 0.728 | 6.611 |
| A `order-nocounts` | features + order, no `counts` head | 8 | 0.1405 | 0.366 | 2.032 | 0.617 | 5.096 |
| B `order-weighted` | as A, loss weights ready × 10, state × 5, winner × 3 | 5 | 0.1658 | 0.397 | 2.603 | 0.514 | 5.122 |
| C `order-counts` | as A with the `counts` head | 6 | 0.1359 | 0.358 | 1.960 | 0.539 | 4.961 |
| **E `order-suits`** | as C plus the `suits` summary | 6 | **0.1346** | 0.355 | 1.967 | 0.540 | 4.963 |
| F `order-suits-dropout` | as E with dropout 0.2 and weight decay 1e-4 | 6 | 0.1395 | 0.367 | 1.905 | 0.491 | 5.081 |

  Held-out losses on the first hold-out chunk (77,905 rows); `winner` is cross-entropy over 9
  classes (the frequencies alone give 2.041, a uniform guess 2.197), `dealin` and `claim` are
  binary cross-entropy summed over the 34 kinds. Weighting the ready head (B) makes it overfit
  within two epochs (train 0.089, held-out 0.166); the `counts` head helps every other head as a
  regulariser (C against A), so it stays in the file as a training aid, as the plan allowed;
  the suit summary adds a little more (E). The chosen model is E: 2,811,882 parameters, trained
  2026-11-20 21:03–21:18 (`runs/belief-order/fit-suits.log`, checkpoints `ckpt-suits/`).
  All fits overfit the 40,000 hands to some degree: E's training ready loss is 0.118 against
  0.135 held out, and its `dealin` held-out loss rises after epoch 1 while `tenpai` still falls.
  Fit F (dropout 0.2 after each hidden layer, AdamW weight decay 1e-4, both added to the trainer
  for this task) trades the ready head for the rest: ready 0.1395 against E's 0.1346, but `winner`
  1.905, `dealin` 0.491, `faan` 1.57 and `waits` 4.93 against E's 1.967, 0.540, 2.14 and 5.85,
  with train and held-out ready losses close (0.131 / 0.140). It is kept beside E as
  `models/belief-order-reg.mjnn` for Task 20, whose bot reads the deal-in and win heads as much
  as the ready head; this task's bar is the ready log loss, so E is the file named above.

### 7.2 The 20,000 held-out deals of four `stage` bots (seed 18)

The same deals, decisions and 3,718,542 opponent rows as §3 (every seat's discard decisions,
each opponent's true ready state against the reading of the seat to act).

| Reading | Log loss on ready (nats) |
|---|---|
| danger table | 0.17868 |
| old network `belief.mjnn` | 0.17155 |
| fit A (order, no counts) | 0.15787 |
| fit C (order, counts) | 0.15348 |
| **fit E (order, suits, counts)** | **0.14986** |
| fit F (as E, dropout and weight decay) | 0.15295 |

Ten bins by fit E's P(ready) (compare §3's table for the old network):

| Slice | Opponent-decisions | True ready | Network mean | Table mean |
|---|---|---|---|---|
| network 0–10 % | 3025387 | 1.28 ± 0.01 % | 1.04 % | 2.00 % |
| network 10–20 % | 209548 | 17.51 ± 0.16 % | 14.42 % | 10.86 % |
| network 20–30 % | 121819 | 27.32 ± 0.25 % | 24.66 % | 14.51 % |
| network 30–40 % | 85517 | 36.14 ± 0.32 % | 34.76 % | 17.85 % |
| network 40–50 % | 67272 | 43.78 ± 0.37 % | 44.81 % | 21.17 % |
| network 50–60 % | 56199 | 50.73 ± 0.41 % | 54.87 % | 24.76 % |
| network 60–70 % | 48965 | 58.49 ± 0.44 % | 64.90 % | 28.73 % |
| network 70–80 % | 42696 | 66.21 ± 0.45 % | 74.89 % | 33.61 % |
| network 80–90 % | 36546 | 75.46 ± 0.44 % | 84.84 % | 40.93 % |
| network 90–100 % | 24593 | 88.46 ± 0.40 % | 94.16 % | 58.29 % |

The new reading is within 3 points of the truth up to 50 % and over-confident by 6 to 10 points
in the 60–90 % bins (the old network was 7 to 10 points under-confident there, §3): the fit's
overfitting shows as too-sharp high readings. It puts 2.3 times as many rows above 50 % as the
old network (209,000 against 81,000) and those rows are ready 68 % of the time.

Slices (true rate, then the mean reading of fit E, the old network and the table):

| Slice | Rows | True ready | Fit E | Old network | Table |
|---|---|---|---|---|---|
| all | 3718542 | 8.17 ± 0.03 % | 8.08 % | 6.52 % | 5.44 % |
| last third of the wall | 452055 | 30.84 ± 0.13 % | 30.79 % | 24.74 % | 20.58 % |
| suit-suspect (first 6 discards outside one suit, that suit thrown since) | 477418 | 35.05 ± 0.14 % | 31.38 % | 19.52 % | 16.75 % |
| suit-suspect, latest discard in that suit | 200763 | 34.45 ± 0.21 % | 28.97 % | 17.20 % | 15.31 % |
| last 5 discards all drawn and thrown (no input carries this) | 96318 | 31.07 ± 0.29 % | 27.94 % | 19.53 % | 16.70 % |
| West's case: 1 open set, suit-suspect, feeds a shown flush, last 5 drawn and thrown | 379 | 43.54 ± 4.99 % | 29.52 % | 13.48 % | 10.80 % |

### 7.3 Task 18a's rows (§6) under the new reading

True rate, then the mean reading of fit E and of the old network (both from the other seats'
next decisions; the table's figures are in §6). Rows within 5 points of the truth are marked.

| Row (§6) | Count | True ready | Fit E | Old | Within 5 |
|---|---|---|---|---|---|
| 6.1 stage 3 flush hand, own-suit discard: all | 45371 | 59.8 % | 45.3 % | 25.9 % | no (−14.5) |
| 6.1 own-suit discard, full flush | 2524 | 90.2 % | 66.1 % | 34.9 % | no (−24) |
| 6.1 own-suit discard, 0 open sets | 2675 | 46.4 % | 13.2 % | 6.0 % | no (−33) |
| 6.1 own-suit discard, 1 open set | 10968 | 49.1 % | 24.2 % | 11.4 % | no (−25) |
| 6.1 own-suit discard, 2 open sets | 18694 | 56.2 % | 43.5 % | 21.7 % | no (−13) |
| 6.1 own-suit discard, 3–4 open sets | 13034 | 76.6 % | 72.0 % | 48.1 % | yes |
| 6.1 own-suit discard, last third | 18342 | 65.3 % | 54.2 % | 34.1 % | no (−11) |
| 6.1 first own-suit discard of the hand: all | 22008 | 49.7 % | 39.4 % | 21.7 % | no (−10) |
| 6.1 first own-suit discard, 1 open set | 6136 | 40.7 % | 21.8 % | 10.6 % | no (−19) |
| 6.1 first own-suit discard, 3–4 open sets | 4967 | 68.4 % | 68.0 % | 43.8 % | yes |
| 6.2 strict collector N = 6, all | 437724 | 38.0 % | 34.0 % | 20.8 % | yes |
| 6.2 N = 6, 1 open set | 113320 | 28.6 % | 19.9 % | 10.8 % | no (−9) |
| 6.2 N = 6, 2 open sets | 142865 | 45.1 % | 42.0 % | 22.6 % | yes |
| 6.2 N = 6, 3–4 open sets | 85040 | 72.3 % | 72.1 % | 49.9 % | yes |
| 6.2 N = 8, all | 328240 | 44.4 % | 39.7 % | 24.3 % | yes |
| 6.2 N = 10, all | 231872 | 49.6 % | 44.3 % | 27.7 % | no (−5.3) |
| 6.2 N = 10, 0 open sets | 25317 | 21.2 % | 10.5 % | 6.5 % | no (−11) |
| 6.2 N = 10, 1 open set | 60927 | 38.2 % | 25.2 % | 13.6 % | no (−13) |
| 6.2 N = 10, 2 open sets | 87889 | 50.9 % | 47.3 % | 26.6 % | yes |
| 6.2 N = 10, 3–4 open sets | 57739 | 72.3 % | 74.7 % | 53.5 % | yes |
| 6.2 N = 10, last third | 136023 | 54.2 % | 49.0 % | 31.9 % | no (−5.2) |
| 6.3 swing, N = 10: last decision before the first x | 21157 | 16.2 % | 14.6 % | 16.9 % | yes |
| 6.3 N = 10: first decision after the first x | 21157 | 42.5 % | 38.3 % | 22.2 % | yes |
| 6.3 N = 10: after the second x | 9624 | 62.0 % | 50.3 % | 29.6 % | no (−12) |
| 6.3 N = 6: before / after first x / after second x | 31612 / 31612 / 16485 | 12.4 / 33.4 / 49.2 % | 10.8 / 29.3 / 39.3 % | 12.1 / 16.2 / 22.2 % | yes / yes / no (−10) |
| 6.4 honour from a seat with no x, all: before / after | 86121 | 4.2 / 14.2 % | 6.9 / 14.4 % | 9.8 / 15.4 % | yes / yes |
| 6.4 in the last third: before / after | 21210 | 6.3 / 17.1 % | 10.5 / 18.4 % | 16.9 / 23.8 % | yes / yes |

The swing Brian asked for is now read: a strict N = 10 collector goes from 14.6 % to 38.3 % at
its first own-suit discard and to 50.3 % at the second (truth 16.2 → 42.5 → 62.0 %; the old
network moved 16.9 → 22.2 → 29.6 %), and a late honour from a seat that has shown no x is read
at 18.4 % against 17.1 % true (old 23.8 %). Where the reading is still short of 5 points, and why:

* **The rows of §6.1** condition on the discarder's own stage and hand (a stage 3 flush hand
  throwing its suit), which no public reading can see; the public counterpart of each row is
  §6.2's strict collector, where the reading is within 5 points at every N for two or more open
  sets. With 0 or 1 open sets the public pattern is weak evidence (such seats are ready 8–38 %
  of the time, and the same discards come from many hands that are not collecting), and there
  the model stays 9 to 13 points low on the strict rows and 19 to 33 points low on the own-stage
  rows. Full-flush hands (2,524 rows in 1.2 M) are too rare in the log for the fit to learn
  their 90 % rate (read 66 %).
* **The second own-suit discard** (6.3, truth 49–62 %) is read 10–12 points low at every N: the
  suit summary carries the first discard of each suit and the share since, but not the count of
  own-suit discards since the first, and the one-hot order block learns that conjunction only
  partly from 40,000 hands.
* These are fit limits, not input limits: every row above is a function of public facts the
  model is given. The remedy is more hands (the fit overfits at 40,000: train 0.118 against
  0.135 held out on ready), a feature for "own-suit discards since the first", or both; neither
  was in this task's budget.

### 7.4 Hand 2860 step by step

P(ready) read by the seat to act (§2 for the hand; the old network's figures as in §2).

| Step | Viewer | South (ready from 57) old → fit E | West (ready from 54) old → fit E |
|---|---|---|---|
| 57 | West | 32.2 → 72.0 % | – |
| 58 | North | 23.6 → 77.0 % | 4.9 → 12.4 % |
| 59 | East | 42.7 → 70.8 % | 9.2 → 21.4 % |
| 60 | South | – | 5.9 → 11.5 % |
| 61 | West | 46.3 → 66.8 % | – |
| 62 | North | 29.7 → 80.8 % | 4.5 → 19.2 % |
| 63 | East | 52.6 → 76.0 % | 12.3 → 33.7 % |
| 64 | South | – | 7.1 → 11.8 % |
| 65 | West | 49.1 → 63.7 % | – |
| 66 | North | 30.6 → 78.2 % | 6.3 → 32.6 % |
| 67 | East | 54.2 → 77.6 % | 17.3 → 49.5 % |
| 68 | South | – | 9.7 → 14.0 % |
| 69 | West | 49.7 → 67.4 % | – |
| 72 | South | – | 10.1 → 15.7 % |
| 73 | West | 93.8 → 97.5 % | – |
| 74 | North | 81.6 → 98.4 % | 6.7 → 33.7 % |
| 75 | East | 95.7 → 98.4 % | 20.6 → 52.2 % |
| 76 | South | – | 10.1 → 9.3 % |
| 78 | West | 92.1 → 97.5 % | – |

South is read above 50 % at every decision from step 57 on (63.7 to 98.4 %; the old network
crossed 50 % only at step 63 and fell back). West is not: 9 to 52 %, above 50 % only at step 75
(East's view, after the fourth own-suit discard). Why not: West's public record is the strict
one-set collector of §6.2 (10 clean discards, then 2萬 and 8萬); such seats are ready 38 % of the
time at N = 10 with one open set and 62 % after the second own-suit discard, and the model reads
that class 10 to 13 points under the truth (§7.3). Within the class West is read lowest by South
(9–16 %) and highest by East (21–52 %): the three viewers hold different hands, so the unseen
counts and the `features` block differ, and South, who holds four 索 sets and sees West's 1索
and 8索 go by, reads West as a 索 feeder rather than a 萬 collector. East's reading at step 67
(49.5 %) and 75 (52.2 %) is the behaviour Brian asked for; South's is the remaining gap. Fit F
reads this hand lower (South 38–96 % from step 57, under 50 % at four of its first seven
decisions; West 12–40 %) and its Task 18a rows sit 2 to 9 points under fit E's.

### 7.5 Time per call

One thread (`RAYON_NUM_THREADS=1`, 100 deals, the first `BeliefModel::run` call at each of
6,310 decisions; the two later calls per decision hit the model's cache):

| Model | Inputs | p50 | p90 | p99 |
|---|---|---|---|---|
| old `belief.mjnn` | features | 0.13 ms | 0.21 ms | 0.26 ms |
| fit C `order-counts` | features + order | 1.25 ms | 3.22 ms | 26 ms |
| fit E `belief-order.mjnn` | features + order + suits | 0.48 ms | 0.92 ms | 2.4 ms |
| fit F `belief-order-reg.mjnn` | the same | 0.75 ms | 11 ms | 56 ms (beside a 20-thread run) |

Fit C's figures were taken while a GPU fit's loader shared the CPU; fit E's on a quieter
machine; the forward pass of the 2.8 M-parameter network is about 0.4 ms, under the 10 ms bar
by a wide margin. With 20 rayon threads on the 20,000-deal run the p50 was 4.8 ms.

### 7.6 The other required outputs

Checked on the hold-out log (fit E, epoch 6) against the trivial reading from the log's own rates:

* Win chances (`winner`): a softmax over 9 classes, so the five numbers (four seats and the
  draw) sum to 1 by construction (Rust test `belief::tests::task19_heads_load_without_counts`);
  held-out cross-entropy 1.967 nats against 2.041 for the class frequencies (self-drawn 5.8 %,
  off a discard 18 % per seat, draw 4.7 %). The gain is modest because most rows are early in
  the hand, when the result is open; no per-stage breakdown was made.
* State (far / one away / ready): 0.355 nats against 0.676 for the frequencies (77 / 16 / 7 %).
* Deal-in per tile and opponent (`dealin`, 3 × 34 sigmoids): 0.540 nats summed over the 34 kinds
  against 0.728 for the per-kind rates (0.11 positive kinds a row); this head overfits first
  (held-out loss rises from epoch 1) and is the one fit F's regularisation improves most.
* Claim per tile and opponent (`claim`): 4.963 against 6.611 (1.67 claimable kinds a row).
* Faan spread (`faan`, 8 buckets over ready opponents): 2.14 nats in fit E, slightly worse than
  a uniform 2.08: this head is over-fitted in E (fit A 2.66, fit F 1.57 at its third epoch).
  Fit F's 1.57 nats is the figure to use when the faan spread matters.
* Expected points (`points`, Huber on the value / 32): 0.433 against a label spread of 1.32.
* Copies per kind (`counts`): 18.98 nats summed over 3 × 34 kinds, kept as a training aid only
  (it lifted every other head, §7.1); Brian's doubt that the solver still needs it stands.

### 7.7 Verdict

* The retrained reading beats the old network and the table on ready log loss over the 20,000
  held-out deals (0.14986 against 0.17155 and 0.17868), is calibrated within 3 points up to
  50 % and over-confident above, and reads the obvious clues Brian named: a strict collector's
  first own-suit discard swings it from 15 % to 38 %, South in hand 2860 is read at 64–98 %
  from the moment it is ready, and a late honour from a one-suit seat is read at its true rate.
* It still under-reads collectors with 0 or 1 open sets and the second own-suit discard by 10 to
  25 points, and West in hand 2860 stays under 50 % for every viewer but East. Both are fit
  limits on 40,000 hands, stated in §7.3.
* One call costs 0.5 ms on one thread. The file carries every output Brian listed and can be
  read on the position after a discard or a claim (`BeliefModel::run_for`).
* For Task 20: `race:belief=models/belief-order.mjnn` and the fold bots' readings can take this
  file; nothing changed by default.

## 8. Defence on the new reading (plan Task 20)

Brian, 2026-11-20: "if it show gains it already passes Milestone 1". Every defence trial so far
read the old figure and showed no gain over plain `stage` (Tasks 11, 12, 14). This section reruns
them with Task 19's model (`models/belief-order.mjnn`, fit E) as the reading, and adds the bot
Brian described: "the sole purpose of the defensive strategy is to figure out the correct move
that gives maximum EV ... realistically your loss EV is your deal in rate + their self-draw rate".
Nothing became a default; every bot is a spec option.

### 8.1 What was built

* **A reading switch.** `danger::with_reading` sets a belief model for the thread and
  `Danger::of_reading` reads it instead of the fitted tables; the give-up rule (Task 11), the
  learned push-or-fold choice (Task 12) and the new bot read their danger through it. A stage
  bot gets a reading with the spec option `belief=` (`stage-fold:belief=...`,
  `stage-fold-learned:belief=...`, `stage-ev:belief=...`; `crate::Agent`). Without it every
  figure is as before (a test checks the tables in force with and without a reading).
* **The model's deal-in head as the danger.** For a file with Task 19's `dealin` head the
  belief's per-tile wait is `dealin ÷ tenpai`, so `tenpai × wait` is that head
  (`BeliefModel::estimate`); older files are read as before.
* **Task 11's rule refitted** (`giveup::READ_CALIBRATION`, `READ_LOSS_FACTOR`, `READ_MEASURED`,
  picked when a reading is set). On the stage 3 discards of 40,000 deals of four `stage` (seed 1,
  `BELIEF=models/belief-order.mjnn giveup_log log 40000 1`, 7.8 min) the new deal-in figure ranks
  far better than the table but is too sure at both ends:

| Raw new reading | Discards | Predicted | Observed | Observed ÷ predicted | Table's estimate |
|---|---|---|---|---|---|
| 0–0.5 % | 249,990 | 0.08 % | 0.54 % | 6.65 | 0.69 % |
| 0.5–1 % | 30,189 | 0.72 % | 2.37 % | 3.29 | 2.93 % |
| 1–2 % | 28,630 | 1.43 % | 3.88 % | 2.71 | 4.07 % |
| 2–3 % | 15,021 | 2.46 % | 5.93 % | 2.41 | 5.26 % |
| 3–5 % | 16,705 | 3.89 % | 7.42 % | 1.91 | 6.31 % |
| 5–7 % | 9,465 | 5.92 % | 9.17 % | 1.55 | 7.82 % |
| 7–10 % | 8,828 | 8.37 % | 11.71 % | 1.40 | 8.80 % |
| 10–15 % | 8,212 | 12.23 % | 14.32 % | 1.17 | 10.17 % |
| 15–25 % | 7,844 | 19.31 % | 17.52 % | 0.91 | 12.17 % |
| 25 % and up | 10,752 | 45.96 % | 27.81 % | 0.61 | 17.07 % |

  The table puts only 17 discards above 25 % (its own bins, 8,000 deals, seed 3) where the new
  reading puts 10,752 at a true 28 %: the reading separates dangerous tiles much better, but its
  level must be mapped. The calibration (these points) and the loss factor (points paid ÷
  predicted loss per bin, 6.5 in the safest bin to 0.54 in the top one) were fitted on seed 1; on
  seed 2 (40,000 deals) the calibrated figure is within 0.81–1.06 of the truth in 9 of 10 bins,
  the safest bin still 4.2 times low (0.09 % against 0.37 %; below the first point the map is
  proportional, as in Task 11). The push side's figures (`READ_MEASURED`) were measured on seed 2
  with this calibration in force. The regularised fit F (`belief-order-reg.mjnn`) gives the same
  picture (safest bin 5.5 times low, top bin 0.77; 8,000 deals, seed 3), so fit E stays the
  reading throughout.
* **Task 12's push-or-fold refitted.** Labels as in Task 12 (play-outs on the true hands with the
  wall reshuffled, 24 pairs a position), with the danger, the choice of the safer tile and the
  folded hand's discards read from the new reading: `BELIEF=... pushfold label 4500 24 1` and
  `16000 24 2` (9 + 32 min on 14 threads), 18,571 positions. Mean label +19.03 ± 0.50 (Task 12:
  +18.78); fold better in 16.5 %, clearly in 6.0 %. Fit (`fit_pushfold.py --tables
  crates/search/src/pushfold_read_tables.rs`, 14,856 training and 3,715 held-out rows):

| Model | Held-out MSE | Says fold | Mean label where it folds | Gain per decision |
|---|---|---|---|---|
| ridge linear (chosen) | 574.3 | 1.9 % | +4.33 ± 5.77 | −0.083 |
| network 32×32 + linear skip | 562.9 | 2.3 % | +4.62 ± 5.53 | −0.105 |
| network + true deal-in (oracle, not for the bot) | 484.0 | 5.5 % | −14.72 ± 2.70 | +0.809 |
| Task 11's rule on the new reading (approximate) | 1121.9 | 13.2 % | +9.12 ± 2.46 | −1.202 |

  As in Tasks 12 and 14 the check is not met: where the model folds, pushing was better on
  average. The fit script now writes a linear model with zero hidden units of the size the Rust
  side is built with (it wrote one unit before, which did not compile).
* **Brian's bot, `stage-ev:belief=models/belief-order.mjnn`** (`crates/search/src/defence.rs`).
  In stage 3, and in stage 0 (no line), for each legal discard `t`:
  `EV(t) = P(I self-draw) × 3 × my self-drawn points + P(I win off a discard) × my discard points
  − deal-in(t) − Σ_o P(o self-draws) × o's self-drawn points × my share − liability(t)`.
  The deal-in term is the model's `dealin` head read before the discard (who wins on `t` now,
  their `faan` head's points, the 12-tile shares), times the loss factor refitted above (my
  addition: the raw head is 6.6 times too low on safe tiles). The "after" chances are the
  `winner` head read from my seat on the position right after the discard. Two calls of mine,
  agreed by the orchestrator on 2026-11-20: (a) the reading right after the discard stands for
  the weighting over no action and each claim, because Task 19 logged exactly that row with the
  hand's real outcome as its label (the position after a claim cannot be built without the
  claimer's hidden hand); the claim chances are printed beside each candidate; (b) the race
  bot's "old reading" is its shipped `models/race-belief.mjnn`. My points: the live winning
  tiles' mean points when ready after `t`, else the minimum hand's. The liability term pays the
  two other shares when `t` could be an opponent's fourth exposed meld. Later deal-ins of mine
  are left out, as in Brian's sum. The stage bot's own pick is kept within 0.05 points.

### 8.2 Brian's scenario

Next seat (1) on a 萬子 full flush, three 萬子 melds shown (2萬 and 7萬 pungs, 345萬) and truly
ready on 7萬; across (2) and previous (3) on 筒子 (5筒 and 白 pungs; 9筒 pung and 123筒), no 萬子
in seat 1's nine discards and no 筒子 in seats 2 and 3's; me on turn with
`147筒 123579索 東西北發中`, 47 tiles in the wall (`defence_match scenario`,
`runs/t20/scenario.txt`). Terms in points, best EV first; "after" chances read right after the
discard:

| Tile | EV | My win term | Deal-in | Their self-draw | Liability | Next / across / previous win after | Claim next / across / previous |
|---|---|---|---|---|---|---|---|
| 9索 | −15.51 | 0.11 | 0.12 | 15.21 | 0.30 | 40.4 / 26.6 / 28.8 % | 0.3 / 0.6 / 0.0 % |
| 3索 | −15.72 | 0.10 | 0.16 | 15.44 | 0.22 | 39.5 / 22.2 / 34.1 % | 0.4 / 1.4 / 0.1 % |
| 北 | −15.80 | 0.10 | 0.00 | 15.90 | 0.00 | 39.3 / 25.3 / 32.0 % | 0.0 / 0.1 / 0.1 % |
| 東 | −15.90 | 0.11 | 0.09 | 15.91 | 0.00 | 36.7 / 26.0 / 33.3 % | 0.2 / 1.0 / 0.4 % |
| 1筒 | −16.16 | 0.08 | 0.13 | 15.96 | 0.16 | 39.4 / 28.3 / 29.4 % | 0.4 / 3.3 / 3.1 % |
| 2索 | −16.23 | 0.08 | 0.12 | 16.03 | 0.16 | 39.3 / 24.9 / 32.3 % | 0.5 / 5.5 / 0.1 % |
| 4筒 | −17.33 | 0.10 | 0.21 | 15.94 | 1.27 | 41.6 / 27.3 / 27.5 % | 0.8 / 2.7 / 5.8 % |
| 7筒 | −17.81 | 0.08 | 0.14 | 16.74 | 1.01 | 39.2 / 25.6 / 32.1 % | 0.7 / 2.7 / 4.9 % |

(Six more candidates in the file.) The bot throws 9索: here it prefers 索 to 筒, and not by
Brian's second effect. A 筒 discard is claimed more often (3–6 %) and the liability term
charges it, while the opponents' self-draw chances after a 筒 discard do not fall. The reading
is weak in this position: it reads the truly ready flush seat at 3.6 % ready (fit F: 11.3 %)
yet gives it a 39 % chance to win the hand, and the self-draw term moves by up to 1.5 points
between tiles that change nothing visible about that seat (16.74 for 7筒 against 15.21 for
9索). That term is the largest in every candidate, so its noise, not the deal-in, decides most
choices.

### 8.3 The matches

Each bot against three plain `stage` over 40,000 deals (seed 20): every deal played with the
bot in each of the four seats, then the same with the bot on the old reading, and once with
four plain `stage` (`defence_match match <bot> <old bot> 40000 20`, run directly on 14 threads
on the orchestrator's word). The ± is the 95 % range over deals (the four seatings of a deal
are one sample); paired differences are on the same deals and seats; "plain" is the seat's
score in the all-`stage` game of the same deal.

| Bot (new reading: `belief=models/belief-order.mjnn`) | Points per hand | Win | Deal-in | Changed a discard | Paired − plain `stage` | Same bot on the old reading | Old − plain | Paired new − old |
|---|---|---|---|---|---|---|---|---|
| `stage-ev` (Brian's EV bot) | −2.16 ± 0.18 | 19.5 % | 16.1 % | 54.0 % of hands | **−2.220 ± 0.176** | – (needs Task 19's heads) | – | – |
| `stage-fold` (Task 11's rule, refitted) | −0.61 ± 0.17 | 22.0 % | 17.4 % | 8.2 % | **−0.667 ± 0.072** | `stage-fold`: −0.54 ± 0.17, win 22.1 %, deal-in 17.3 % | −0.606 ± 0.069 | −0.061 ± 0.072 |
| `stage-fold-learned` (Task 12's choice, refitted) | +0.05 ± 0.17 | 23.8 % | 18.0 % | 1.6 % | **−0.008 ± 0.027** | `stage-fold-learned`: +0.10 ± 0.17, win 24.0 %, deal-in 18.0 % | +0.039 ± 0.020 | −0.048 ± 0.028 |
| `race:belief=...` | −3.52 ± 0.19 | 15.0 % | 15.6 % | – | **−3.586 ± 0.253** | `race:belief=models/race-belief.mjnn`: −3.63 ± 0.19, win 16.2 %, deal-in 17.2 % | −3.687 ± 0.256 | +0.101 ± 0.170 |
| plain `stage` (same deals and seats) | +0.06 ± 0.17 | 24.1 % | 18.3 % | – | – | – | – | – |

Times on 14 threads: `stage-ev` 51 min, `stage-fold` 64 min, `stage-fold-learned` 61 min, race 93 min
(the matches with a network ran on about 4 cores: the model's shared per-state cache is behind
one lock). Outputs `runs/t20/match-*.txt`. Task 11 measured `stage-fold` at −0.63 ± 0.17 and
Task 12 `stage-fold-learned` at +0.03 ± 0.17 against three `stage` on other deals; both agree.

One comparison came out above zero: Task 12's `stage-fold-learned` on its **old** reading (the
fitted tables), +0.039 ± 0.020 against plain `stage` in the same seats. That is one of six
paired differences, so it was played again on fresh deals (seed 21, 40,000 deals, 68 min,
`runs/t20/match-stage-fold-learned-old-seed21.txt`): +0.024 ± 0.022 (bot +0.07 ± 0.17, win
24.0 %, deal-in 17.9 %, changed a discard in 0.9 % of hands; plain `stage` +0.05 ± 0.17, win
24.1 %, deal-in 18.2 %). Both seeds together (80,000 deals): **+0.032 ± 0.015 points a hand**.
That is about 1 % of plain `stage`'s edge over three `value` bots (+2.62).

### 8.4 Verdict

* **On the new reading, no bot gains over plain `stage` beyond its range.** Brian's EV bot loses
  2.22 ± 0.18 points a hand. Task 11's rule loses 0.67 ± 0.07. The learned push-or-fold choice
  is level (−0.008 ± 0.027). The race bot loses 3.59 ± 0.25. Against the same bots on the old
  reading, the new reading changes nothing beyond the range for the rule (−0.06 ± 0.07) or the
  race bot (+0.10 ± 0.17), and makes the learned choice slightly worse (−0.048 ± 0.028).
* **The one measured gain is the old reading's learned choice**: Task 12's
  `stage-fold-learned`, +0.032 ± 0.015 points a hand over 80,000 deals on two seeds, with the
  whole range above zero. By Brian's words ("if it show gains it already passes Milestone 1")
  this is a gain over plain `stage`, but a very small one (it changes a discard in 1 % of hands).
  It comes from Task 12's bot, not from the new reading.
* **Why the EV bot loses.** It overrides the stage bot in 54 % of hands, and its win rate falls
  from 24.1 % to 19.5 % while its deal-ins fall only from 18.3 % to 16.1 %. Its largest term,
  the opponents' self-draw chances read after each candidate discard, moves by 1 to 2 points
  between tiles that change nothing visible (§8.2). So the network's noise between near-equal
  positions decides, not the danger. The reading is also wrong where Brian's scenario needs it
  most: the truly ready full-flush seat reads 3.6 % ready. Making the EV bot work needs "after"
  readings that are steady between near-equal positions (for example, read the after-discard
  difference only through the deal-in and claim heads, or average several fits), not a
  different weighting.
* **Why the reruns change nothing.** The new deal-in figure ranks tiles much better than the
  table (§8.1), but the decisions these bots change are rare (1–8 % of hands). The labels say
  pushing is right in about 84 % of the candidate fold decisions whatever the reading (mean
  push − fold +19 points). The learned choice's held-out check fails on every reading (Tasks
  12, 14, 20): the visible inputs cannot pick out the 16 % of positions where folding is better.
  Only the true deal-in (the oracle, +0.81 a decision) does.

## Files

* `crates/search/examples/belief_ready.rs`: the tool (`--hand`, `--deals`, `--field`; Task 18a's
  tables print after §3's for `--field stage`).
* `runs/belief-ready/` (not in git): `hand-2860.txt`, `stage-20000.md`, `value-20000.md`,
  `stage-20000-t18a.md` (all tables of §3 and §6), `hand-2860-t18a.txt`.
* No model, bot, table or default changed.
* Task 19: `crates/search/examples/selfplay_log.rs` (`--field stage`, layout 3 with the
  spectator rows), `crates/search/src/log.rs` (layout 3, `pack_order`, `suits_from_packed`),
  `crates/search/src/features.rs` (`encode_order`, `encode_suits`, the `_for` forms),
  `crates/search/src/net.rs` (input slices `order`, `suits`), `crates/search/src/belief.rs`
  (Task 19 heads, `run_for`, `win_chances`, `dealin_or_claim`), `python/mahjong/belief/`
  (`model.py`: `order_block`, `suits_block`, the heads; `train.py`: `--order --suits --task19
  --no-counts --weight --dropout --weight-decay --hours`), `belief_ready` prints the ready log
  loss and the one-call timing.
* `runs/belief-order/` (not in git): `train/` and `holdout/` (the logs), `fit-*.log` and
  `ckpt-*/` (the five fits), `order-*.pt` / `.mjnn`, `checks/` (`old-`, `A-`, `C-`, `E-`
  `deals20k.md`, `hand2860.md`, `timing-1thread.md`). `models/belief-order.mjnn` is fit E.
* Task 20: `crates/search/src/defence.rs` (the bot), `danger.rs` (`with_reading`,
  `Danger::of_reading`), `giveup.rs` (`READ_*`, `calibration`, `loss_factor`, `measured`),
  `pushfold.rs` (`READ_MODEL`) and `pushfold_read_tables.rs`, `stage.rs` (`reading`, `defence`),
  `agent.rs` (`stage*:belief=`), `bots.rs` (`stage-ev`), `belief.rs` (the `dealin` head as the
  wait); `examples/defence_match.rs` (`match`, `scenario`), `BELIEF=` in `giveup_log` and
  `pushfold`; `experiments/stage/fit_pushfold.py` (`--tables ...read...`, zero-padded linear).
* `runs/t20/` (not in git): `giveup-read-seed1|2.txt`, `giveup-reg-seed3.txt`,
  `giveup-table-seed3.txt`, `labels-read-seed1|2.csv`, `fold-learned-step2.md`, `scenario.txt`,
  `scenario-reg.txt`, `match-*.txt` (with `match-stage-fold-learned-old-seed21.txt`).
