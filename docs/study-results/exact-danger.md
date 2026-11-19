# Exact danger by counting (plan Task 13)

Plan: `docs/plans/2026-11-16-self-play-network.md`, Task 13. Brian, 2026-11-19: "but it should be
possible to reach 100%. let's say full flush all pung 12 exposed, you can count the available
tiles and sometimes it just boil down to a single option"; on the counting step: "Yes, add it".
Scope narrowed the same day: "but we're redoing version B, so it doesn't make sense to build the
new test on top of version B". So there is no bot and no match here; the match is Task 14 (the
learned push-or-fold choice of Task 12 with this figure as its input).

## What was built

* `crates/search/src/exact_wait.rs`: for one opponent seen from one seat, every concealed hand it
  could hold when it holds 1 or 4 concealed tiles, and from them the tiles it can win on. A hand
  counts when (a) its tiles are among the tiles this seat cannot see, (b) with the exposed sets it
  is ready, (c) the win on the tile by discard reaches the 3-faan minimum under the engine's own
  scoring (`score_win`, with the opponent's flowers and seat and prevailing winds), (d) the tile
  is not one the opponent was offered since its own last action.
* The pass rule the engine has: the passed-win lock 過水 (rules §4.4, `RuleSet::passed_win_lock`,
  on by default). A seat offered a win on a tile kind, on a discard or a robbed kong, that does not
  take it cannot win on that kind from any player until its next turn starts (its draw or its own
  claim); a self-draw is not refused. Seen from outside: a hand that wins on a tile discarded since
  the opponent last acted either won (the hand would be over) or passed and is locked. So those
  tiles give that opponent 0.
* Weights: each ready hand weighs the number of ways to hold it from the unseen copies,
  `Π C(unseen, held)`.
  * One concealed tile: the opponent is always ready (if any hand is possible). Danger of `k` =
    weight of `k` ÷ the weight of all kinds that pass (a) to (d). One kind left gives exactly 1.0;
    a kind that fails gives exactly 0.0.
  * Four concealed tiles: the table's chance that the opponent is ready, spread over the possible
    ready hands by weight. A tile no possible hand wins on gets exactly 0.0.
  * More concealed tiles: the table, as before.
* `Danger` (`danger.rs`): the switch `danger::with_exact(true, ...)` (thread-local, off by
  default) makes `Danger::estimate` and `Danger::from_belief` use the exact figure where it
  exists, with the points paid from the same list (the faan of each possible hand). The table
  opponents still go through the calibration curve; the exact ones stay outside it. Two read-only
  methods: `Danger::is_exact(seat)` and `Danger::split(tile)` (table part and exact part apart, for
  Task 14). Every existing public function is unchanged, and with the switch off every figure is
  bit for bit the same (tests).
* `crates/search/examples/danger_check.rs --exact <deals> [seed]`: the measurement below.

Tests (`exact_wait::tests`): Brian's case (four exposed pungs of one suit, one kind left: 1.0,
every other tile 0.0, points 128 = the 10-faan cap); two kinds with 2 and 1 unseen copies: 2/3
and 1/3; a tile seat 1 threw after the opponent's last discard: 0.0 (the lock); a hand that
reaches the minimum only as a flush (one chow, three pungs of bamboos): 9s and E 0.5 each, 2m
0.0, points 48 and 8; three exposed sets: every tile no possible hand wins on is 0.0, the two
winning tiles keep the table's ready chance; the 4-tile shapes equal the engine's waits for every
ready 4-tile hand; switch on with no short hand changes nothing.

## Measurement: 40,000 deals of four `stage` bots (seed 1)

`danger_check --exact 40000 1`, run directly (14 minutes on a loaded PC), output
`runs/stage/t13-exact-40000.txt` in the worker clone. Every discard by any seat was checked; a
discard counts below when at least one opponent held 1 or 4 concealed tiles. The table figure is
Task 11's: `Danger::from_belief` with the fitted tables, through `giveup::danger_of`'s
calibration. The exact figure is the same with the switch on: the table opponents through that
calibration, the exact ones not.

| | count |
|---|---|
| discards | 2,478,426 |
| facing an opponent with 1 concealed tile | 39,328 (1.6 %) |
| facing one with 4 concealed tiles (none with 1) | 323,376 (13.0 %) |
| opponent-discard pairs, 1 concealed tile | 39,919 |
| opponent-discard pairs, 4 concealed tiles | 378,775 |
| discards whose figure moved by more than 1 point of a percent | 135,335 |

### The 1.0 and 0.0 check

| opponent holds | given exactly 1.0 | of those it won on | given exactly 0.0 | of those it won on |
|---|---|---|---|---|
| 1 concealed tile | 4 | 4 | 22,629 | 0 |
| 4 concealed tiles | 0 | – | 172,122 | 0 |

Zero misses. 1.0 is rare: a bot holding the only live kind usually keeps it.

### Facing an opponent with 1 concealed tile (39,328 discards)

| predicted bin | table: discards | predicted | observed | obs ÷ pred | exact: discards | predicted | observed | obs ÷ pred |
|---|---|---|---|---|---|---|---|---|
| 0–0.5 % | 16,222 | 0.07 % | 0.75 % | 11.43 | 16,052 | 0.06 % | 0.44 % | 7.91 |
| 0.5–1 % | 3,821 | 0.69 % | 0.94 % | 1.37 | 2,184 | 0.68 % | 1.60 % | 2.36 |
| 1–2 % | 3,690 | 1.51 % | 1.30 % | 0.86 | 2,240 | 1.56 % | 1.96 % | 1.26 |
| 2–3 % | 1,577 | 2.46 % | 3.74 % | 1.52 | 2,488 | 2.51 % | 2.53 % | 1.01 |
| 3–5 % | 2,628 | 4.04 % | 4.38 % | 1.08 | 5,263 | 3.96 % | 3.63 % | 0.92 |
| 5–7 % | 1,396 | 5.90 % | 6.23 % | 1.06 | 3,336 | 5.94 % | 7.79 % | 1.31 |
| 7–10 % | 1,450 | 8.58 % | 7.03 % | 0.82 | 3,056 | 8.38 % | 11.94 % | 1.43 |
| 10–15 % | 1,965 | 13.02 % | 11.76 % | 0.90 | 2,525 | 12.16 % | 19.80 % | 1.63 |
| 15–25 % | 2,812 | 18.94 % | 13.73 % | 0.72 | 1,629 | 18.85 % | 33.39 % | 1.77 |
| 25 % + | 3,767 | 37.42 % | 32.15 % | 0.86 | 555 | 33.01 % | 58.20 % | 1.76 |

### Facing an opponent with 4 concealed tiles, none with 1 (323,376 discards)

| predicted bin | table: discards | predicted | observed | obs ÷ pred | exact: discards | predicted | observed | obs ÷ pred |
|---|---|---|---|---|---|---|---|---|
| 0–0.5 % | 124,758 | 0.10 % | 0.97 % | 9.41 | 122,532 | 0.11 % | 0.32 % | 2.88 |
| 0.5–1 % | 51,546 | 0.67 % | 0.57 % | 0.85 | 44,797 | 0.72 % | 0.92 % | 1.28 |
| 1–2 % | 30,741 | 1.47 % | 1.88 % | 1.28 | 47,049 | 1.47 % | 2.24 % | 1.52 |
| 2–3 % | 17,338 | 2.46 % | 3.67 % | 1.49 | 28,385 | 2.48 % | 4.39 % | 1.77 |
| 3–5 % | 29,860 | 3.93 % | 5.06 % | 1.29 | 37,210 | 3.89 % | 7.31 % | 1.88 |
| 5–7 % | 19,526 | 5.87 % | 7.93 % | 1.35 | 18,678 | 5.89 % | 11.93 % | 2.03 |
| 7–10 % | 17,641 | 8.52 % | 9.71 % | 1.14 | 13,305 | 8.26 % | 16.47 % | 1.99 |
| 10–15 % | 15,571 | 12.39 % | 13.45 % | 1.09 | 7,497 | 12.07 % | 24.32 % | 2.01 |
| 15–25 % | 10,965 | 19.21 % | 19.59 % | 1.02 | 3,394 | 18.39 % | 35.92 % | 1.95 |
| 25 % + | 5,430 | 29.64 % | 33.70 % | 1.14 | 529 | 29.94 % | 53.31 % | 1.78 |

(95 % ranges are in the output file.)

## What the tables say

* **The exact figure sorts better than the table.** Against four concealed tiles its safest bin
  deals in 0.32 % of the time against the table's 0.97 % (392 against 1,210 deal-ins in bins of
  the same size), and its top bins reach 36 % and 53 % against the table's 20 % and 34 %. Against
  one concealed tile the top bin reaches 58 % against 32 %. The tiles it calls safe are safer and
  the ones it calls dangerous are more dangerous.
* **Its level is about 2 times too low** (obs ÷ pred 1.3 to 2.0 in most bins). The weights treat
  every possible hand as equally likely per way of holding it, but players keep the tiles that
  make a ready hand, so the real hand is more often one that wins on the tile thrown. The table
  part kept Task 11's calibration; the exact part has none. A calibration of the exact part (or
  feeding it as an input to Task 12's learned choice, which is Task 14's plan) is the next step,
  not part of this task.
* **Against four concealed tiles, compared with the table alone:** the ready chance is the
  table's, so the total danger per opponent is the same size, but 45 % of the opponent-tile pairs
  (172,122 of 378,775) become exactly 0, and the rest of the chance moves onto the tiles a possible
  hand can win on.
* **The table's own danger is far off in the safest bin here** (9 to 11 times too low): Task 11's
  calibration was fitted on stage 3 discards of the bot itself, and these are every seat's
  discards facing a short hand.

Verdict: exact counting is correct where it is certain (zero misses on 4 discards at 1.0 and
194,751 opponent-discard pairs at 0.0) and ranks tiles better than the table, but its chances
need a calibration before a bot acts on their level. Nothing becomes the default; Task 14 tests it
on the learned push-or-fold choice. No paid API calls.
