# The three-stage bot's early throws (plan 2026-11-16, Task 30)

Brian, 2026-11-23, after stepping through deal 99000316: "The early game algorithm needs
fixing though"; on the tie-break that kept the 萬 tile: "well it's a tiebreaker for all pungs
but not for half flush".

Built by slot 3, 2026-11-23. Code: `crates/search/src/stage.rs` (`StageBot::early`, bits
`EARLY_SUIT`, `EARLY_FAR`, `EARLY_VALUE`, all off by default), bot names in
`crates/search/src/bots.rs`, scan `crates/search/examples/early_scan.rs`, match job
`experiments/stage/early_game.ps1`. Every match here is played under the corrected 12-tile rule
(liability on self-draw only, Task 27).

## 1. What went wrong in the two positions

In both positions the leading line is all pungs (stage 1), not the flush. Stage 1 ranks a throw
by: the leading line's distance after it, the target's faan, the helping tiles, then **the
other lines' distances** (flush, thirteen orphans, all chows, in the lines' order), then
connection, copies visible, and a random draw.

- **Position 1** (deal 99000316, action 2, seat 2, first throw, `1133m 6m 16p 466s E N W Wh`):
  6m, 1p, 6p and four honours tie on everything down to the other lines' distances. A lone 6m
  does not lower the 萬 flush distance, so throwing it costs the flush nothing by that measure,
  and 1p and the honours count for thirteen orphans. What's left is 6m or 6p, decided by copies
  visible and the random draw: the bot threw 6m. The orchestrator's reading of the code is
  right: in-suit and off-suit singles tie on the flush figure.
- **Position 2** (action 24, seat 3, fifth throw, `55m 9m 22588s E S W N G Wh`): 9m, 5s and the
  six honours tie down to the other lines. Throwing 9m or an honour puts thirteen orphans
  (6 away) one further; 5s touches no line's distance, so 5s goes, 64 random draws out of 64.
  Thirteen orphans at 6 away weighs the same as the half flush at 6 away.
- The `stage_value` figures in the example files are **not** the stage 1 order: they are the
  value bot's ranking (distance, acceptance, faan, connectedness) that `discard_figures`
  logs. E and Wh's acceptance of 26 against 25 plays no part in a stage 1 pick. This is a
  logging gap, not a play fault. A table page that wants to explain stage 1 picks needs the
  stage 1 key logged (not built here).

Better throws: position 1 is an off-suit single, 6p or 1p (6p is what Brian named). Position 2
is 9m, a tile no line but thirteen orphans wants; every honour still serves the 索 half flush
and all pungs.

## 2. The variants (switches, default off)

| Bot | Bits | What changes, only among stage 1 throws tied on distance, faan and helpers |
|---|---|---|
| `stage-early-suit` | A `EARLY_SUIT` | a suited tile outside the best flush suit goes before a tile of that suit or an honour |
| `stage-early-far` | B `EARLY_FAR` | in the other lines' distances a line counts only within 1 tile of the leading line, each read with a handicap: thirteen orphans +2 (`ORPHANS_HANDICAP`, the builder's figure), all chows +4 (its lead rule) |
| `stage-early-value` | C `EARLY_VALUE` | an honour that scores as a pung (dragon, seat wind, round wind) goes after the other tied throws |
| `stage-early-ab` | A + B | both |
| `stage-early` | A + B + C | all three |

C comes from the scan in section 3. Picks over 64 random streams (test `show_early_picks`):

| Position | plain | A | B | C | A+B | A+B+C |
|---|---|---|---|---|---|---|
| 1 | 6m, 6p | 6p | 1p, 6m, 6p, N, W | 6m, 6p | 1p, 6p | 1p, 6p |
| 2 | 5s | 9m | 5s, 9m, E, G, N, S, W, Wh | 5s | 9m | 9m |

B alone does not fix position 2. With thirteen orphans out of the key, nothing else in the key
tells 9m from 5s (a lone 5s doesn't lower the 索 flush distance either), so the pick falls to
the random draw among eight tiles. A is the fix for both positions. B only removes the
thirteen-orphans pull.

Tests: `early_suit_throws_an_off_suit_single_first` and
`early_switches_keep_5s_while_9m_is_held` (both positions, plain fault shown, fixed with A,
A+B, A+B+C); `early_switch_off_leaves_stage_as_before` (`StageBot::default()` and `stage` play
200 seeded deals with the same action hash as before, and the bot names set the bits).

## 3. Other early-game faults: a fresh 40-deal scan

The 22 other example files were never committed (only Task 25's test replayed them;
orchestrator, 2026-11-23). So `early_scan 40 30` played 40 fresh deals (ids 30000000–30000039)
with four plain `stage` and checked every stage 1 discard: 1,941 of them. Output is in
`runs/early/scan-40-seed30.txt` (this checkout). Variant A changes 89 picks (4.6 %), B 197,
C 32, A+B 252, all three 262.

| Fault | Example (deal, action, seat, hand before the throw) | Plain throws | Fixed by |
|---|---|---|---|
| a tile of the best flush suit goes while an off-suit single ties | 30000000, 3, seat 2, `3339m 4678p 11s EE W N` (pungs lead) | 4p | A (9m) |
| thirteen orphans keeps a terminal and a connected tile goes | 30000002, 31, seat 1, `156779m 1789p S WhWh R` (萬 flush lead) | 7p | B (1p) |
| a scoring honour single goes while a non-scoring one is held | 30000012, 39, seat 3, `99p 122566s W N R` (pungs lead, seat 3 = North) | R | C (W) |
| the same | 30000026, 16, seat 2, `44p 9p 77s E N WhWh G R` (pungs lead) | R | C (9p); A (9p) |
| the same | 30000003, 13, seat 3, `333p 2447788s NN G R` (pungs lead) | G | C (2s) |
| B alone throws a dragon where plain kept it | 30000003, 8, seat 3, `3m 333p 244788s NN G R` | 3m | B throws G; A+B+C keeps 3m |
| a pair of the third suit goes while a lone tile of the second kept suit stays | 30000003, 7, seat 2, `23488m 66p 3s 78s S W N G` (萬 flush) | 6p | not built (see below) |

The scan's checks flagged 12 scoring-honour throws by plain `stage`, 101 broken pairs, and 500
throws of a connected tile while an isolated one was held. Most of the last two are the flush
rule doing what Brian set: the two best suits are kept while the third suit has tiles, and the
third suit is thrown even when it holds a pair. In 20 of the first three throws a flush hand
broke a pair of the third suit. That comes from Brian's rule (the third suit goes first), not
from a tie-break, so no switch was built for it here.

## 4. Matches (cpu job 311, pending)

One variant against three plain `stage`, rotated through the four seats; every rotation is
played beside an all-plain twin on the same wall and random stream (`flush_neighbours one`,
Task 22's set-up). 40,000 deals, seed 30, corrected 12-tile rule. The gain is the variant's
points a hand minus the same seat in the twin, with its 95 % range over deals. The tool's
"changed the suit" line is the neighbour rule's counter and reads 0 for these bots.

Smoke run, 200 deals (direct, 38 s, ranges too wide to read): A +1.92 ± 1.43, B −0.87 ± 2.13,
C +0.69 ± 0.76, A+B +1.29 ± 2.61, all +1.30 ± 2.52.

| Bot | Gain a hand | Verdict |
|---|---|---|
| `stage-early-suit` | pending (job 311) | |
| `stage-early-far` | pending | |
| `stage-early-value` | pending | |
| `stage-early-ab` | pending | |
| `stage-early` | pending | |

Output: `runs/early/one-40000-seed30.txt` in checkout `hk-mahjong-lab-3`.

## 5. Example file with the switches on

`docs/study-results/examples/early-99000316-seat2.json` (format 2): deal 99000316 with every
seat playing `stage-early` (all three switches). The tested seat 2 is `stage-safe:t=1` with the
same switches (`defence_false_alarms examples-json runs/early/examples 40 99
models/belief-order.mjnn safe1 only=99000316:2 base=stage-early`; new option `base=`). The
game leaves the old one at the first throw: seat 0 throws 9p rather than 8p, seat 1 pungs it,
and seat 2 draws a different tile. So position 1 does not come up again as such; seat 2's first
throws are 1p, 6p, 4p, 4s. Seat 0's own file has no change by the false-alarm bot under the
switches, so none was written; seat 3's is in `runs/early/examples`. The `stage_value`
figures in the file are still the value bot's ranking (section 1).
