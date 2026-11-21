# Flush suit by the neighbours (plan Task 21, 2026-11-21)

Brian, 2026-11-21, after the defence trials of Task 20 showed no gain: "Let's try a different
direction - for the 3 stage bot, during the first stage, when considering which flush for half
flush, if top two choices were close (within 1 tiles), then always choose: 1. The one that is
different from your 上家 2. The one that is the same as your 下家"; on their order: "Prefer 1
over 2, and choose both if possible"; on how a neighbour's suit is known: "No need at least 2
tiles to know neighbor. If never discarded while others discarded (so thrown least by 1 at
minimum) then that is good enough indication"; on the match: "I kinda of assume if 2v2 and
adjacent config, the original version that has the new version as previous seat would have a
lower EV than it's copy at its next sear", then "Can just use the second one?" (only the
two-against-two match runs).

上家 is the seat that plays just before the bot (the bot may chow its discards); 下家 the seat
that plays just after it (it may chow the bot's discards). Seat `s` plays before `s + 1`.

## What changed (`crates/search/src/stage.rs`)

* `StageBot` has a switch `neighbour`: 0 off (every existing bot, `stage` included); 1 = close
  means the two best suits' flush distances are at most 1 apart (bot `stage-neighbour`, the
  orchestrator's reading (a)); 2 = close means their tile counts (concealed and melded) are at
  most 1 apart (bot `stage-neighbour-tiles`, the variant).
* Where it acts: stage 1 of a leading flush only (the flush leads and the hand still holds a
  tile outside the kept suits), at discards and at claims, anew each turn. Both of the two best
  suits must still have tiles in the concealed hand, so the hand stays in stage 1 after the
  swap. The order the bot already had (`suit_order`: flush distance, tiles, pairs and
  neighbours, copies visible) picks which two suits are "the top two".
* The rule (Brian's order): of the two close suits, a suit that is not 上家's and is 下家's
  beats one that is neither; when the two rules point to different suits, rule 1 (not 上家's)
  wins; an unknown neighbour settles nothing. With both suits held and the third suit still in
  the hand, the swap changes the main suit (the one stage 1 may chow and the pung rule 3
  counts); once the third suit is gone it changes which suit is kept and which is thrown.
* A neighbour's suit (`stage::read_suit`, public facts only): the suit of its open suited sets
  when they are all one suit (mixed: unknown; concealed kongs do not count); with no open
  suited set, the one suit it has thrown no tile of while it has thrown at least one tile of
  each of the other two (Brian's words); otherwise unknown.
* Tests (`stage::tests::neighbour_*`): two suits one tile apart (`1235789m234679pW`), 上家 on
  the better suit: the other is kept; 上家 unknown and 下家 on the second suit: the second is
  kept; both rules on one suit: kept; rules split: rule 1 wins; suits two or more apart
  (`123456789m2346pW`): unchanged under both readings; both neighbours unknown or on the third
  suit: unchanged; switch off: unchanged; the public reading of a neighbour's suit. Switch
  off: `compare stage value stage-fold stage-gate0 200 7` prints the same lines digit for digit
  before and after the change.
* Tool: `crates/search/examples/flush_neighbours.rs` (`flush_neighbours pair <deals> [seed] [bot...]`).
* Three suits (`neighbour = 3`, bot `stage-neighbour-three`; Brian, 2026-11-21: "The new stage
  1 rules could also be split between 3 flushes right?", then "Yes"; "Second and third are to
  be considered when only 1 less tile than the flush with most tiles"): every suit whose tile
  count (hand and open sets) is at most 1 below the suit with most tiles is a candidate, if it
  still has concealed tiles; 上家's suit is dropped, then 下家's suit is preferred among the
  rest; the chosen suit moves to the front and the others keep their order (so with the third
  suit still in the hand, the chosen suit and the next one are kept). Nothing settled: the
  present order. Test `neighbour_rule_three_suits` (`1357m1357p157sEEWh`, three suits within
  1): unknown neighbours leave it; 下家 on the third suit 索 makes 索 lead (the two-suit rule
  could not); 上家 萬 and 下家 索: 索; both neighbours on 索: dropped, unchanged; a suit 3 tiles
  below the most is no candidate. Switch off and modes 1 and 2: the same output digit for digit
  after this change (`compare` 200 deals; `flush_neighbours pair 200 21` modes 1 and 2).
* `Note::neighbour` carries flags (`NB_CHANGED`, `NB_THREE`, `NB_DIFFERS`), so the match tool
  can count changed hands, hands with three candidates and hands where the three-suit choice
  differed from the two-suit rule's.

## The match

Brian's set-up: two test bots and two plain `stage`, seated new, new, old, old in playing order,
the table rotated through the four winds on each deal (four games a deal; the four games of a
deal are one sample for the ± ranges, 95 %). Each rotation is also played by four plain `stage`
on the same wall and random stream (the twin), so every seat has a plain counterpart holding
the same tiles. Seat roles: **N1** new bot whose 上家 is plain, **N2** new bot whose 上家 is
new, **O1** plain bot whose 上家 is new, **O2** plain bot whose 上家 is plain. Gain = mean of
N1 and N2 minus mean of O1 and O2, paired over deals.

### Two suits, 40,000 deals (seed 21)

`flush_neighbours pair 40000 21` (12 games a deal, 1,686 s on 15–16 threads), log
`runs/flush-neighbours/pair-40000-seed21.txt` in the workers clone. Neighbour reading: Brian's
(open sets, or the one suit never thrown while both others were).

| | `stage-neighbour-tiles` (close = tile counts within 1; Brian's rule) | `stage-neighbour` (close = flush distances within 1; comparison only) |
|---|---|---|
| N1 new, 上家 plain | +0.313 ± 0.182 (win 24.45 %, deal-in 18.13 %) | +0.324 ± 0.186 (24.35 %, 18.03 %) |
| N2 new, 上家 new | −0.216 ± 0.178 (23.51 %, 18.22 %) | −0.586 ± 0.181 (22.91 %, 18.15 %) |
| O1 plain, 上家 new | −0.331 ± 0.182 (23.65 %, 18.22 %) | −0.270 ± 0.187 (23.66 %, 18.23 %) |
| O2 plain, 上家 plain | +0.234 ± 0.181 (24.44 %, 18.05 %) | +0.531 ± 0.189 (24.80 %, 17.86 %) |
| **gain** (new − plain) | **+0.097 ± 0.210** | **−0.261 ± 0.217** |
| plain bots: 上家 new − 上家 plain (O1 − O2) | −0.565 ± 0.295 | −0.801 ± 0.306 |
| new bots: 上家 new − 上家 plain (N2 − N1) | −0.529 ± 0.293 | −0.910 ± 0.296 |
| new-bot hands where the rule changed the suit | 18.15 % | 25.38 % |
| neighbour reading at stage 1 flush discards: right / neighbour not on a flush line / other suit | 84.8 % / 14.5 % / 0.7 % (4.0 M readings) | 84.0 % / 14.9 % / 1.1 % |
| changed hands: points, this bot / same seat in the twin | −6.52 / −6.60 (diff +0.08 ± 0.33) | −6.27 / −5.47 (diff −0.80 ± 0.31) |
| changed hands: win | 13.5 % / 14.4 % | 13.7 % / 16.1 % |
| changed hands: chows claimed from 上家 | 0.408 / 0.403 a hand | 0.378 / 0.437 |
| changed hands: melds 下家 made with the bot's discards | 0.784 / 0.948 a hand | 0.780 / 0.966 |
| changed hands: ended in a flush | 62.1 % / 70.2 % | 59.2 % / 73.7 % |

Each role against the same seat in the all-plain twin (tile counts): N1 +0.264 ± 0.118, N2
−0.253 ± 0.118, O1 −0.265 ± 0.128, O2 +0.254 ± 0.121 (flush distances: +0.276, −0.623, −0.204,
+0.551, each about ± 0.15).

* **Brian's guess holds**: the plain bot whose 上家 is a new bot scores less than the plain bot
  whose 上家 is plain, by 0.57 ± 0.30 points a hand (tile counts) and 0.80 ± 0.31 (flush
  distances). The same holds for the two new bots (0.53 and 0.91). Whoever sits after a new bot
  loses, new or plain: rule 2 makes the new bot collect its 下家's suit, so it throws fewer of
  that suit; 下家 makes 0.78 melds a hand from its discards in the changed hands against 0.95 in
  the twin. The seat before (上家) of a new bot gains (O2 and N1 at about +0.25 each against the
  twin).
* In this seating each pair has one member after a new bot (N2 and O1), so the starving effect
  falls on both pairs alike and the gain measures what the rule does for the bot's own hand.
  With tile counts it is +0.10 ± 0.21, level; with flush distances −0.26 ± 0.22, a loss just
  beyond its range (it changes the suit in a quarter of hands and those hands end in a flush
  less often, 59 % against 74 %).
* The neighbour reading is right in 85 % of readings; when wrong, the neighbour is almost always
  not on a flush line at all (15 %) rather than on another suit (under 1 %), so the 60 % mark of
  the plan is passed and no second reading was tried.

### Tile counts, two suits and three suits, 200,000 deals (seed 21)

Brian, 2026-11-21: "Then increase the number of games for the 3-suit case" (after +0.10 ± 0.21
above). `flush_neighbours pair 200000 21 stage-neighbour-tiles stage-neighbour-three` (the same
walls as above for the first 40,000 deals; 12 games a deal, 3,865 s on 15–16 threads, run from
the checkout with Windows power throttling lifted on the process), log
`runs/flush-neighbours/pair-200000-seed21.txt` in the workers clone.

| | two suits `stage-neighbour-tiles` | three suits `stage-neighbour-three` |
|---|---|---|
| N1 new, 上家 plain | +0.260 ± 0.081 (win 24.46 %, deal-in 18.12 %) | +0.266 ± 0.081 (24.45 %, 18.12 %) |
| N2 new, 上家 new | −0.221 ± 0.079 (23.43 %, 18.18 %) | −0.231 ± 0.079 (23.42 %, 18.17 %) |
| O1 plain, 上家 new | −0.246 ± 0.081 (23.66 %, 18.17 %) | −0.251 ± 0.081 (23.65 %, 18.16 %) |
| O2 plain, 上家 plain | +0.207 ± 0.081 (24.50 %, 18.04 %) | +0.215 ± 0.081 (24.50 %, 18.03 %) |
| **gain** (new − plain) | **+0.040 ± 0.093** | **+0.036 ± 0.093** |
| plain bots: 上家 new − 上家 plain (O1 − O2) | −0.453 ± 0.132 | −0.466 ± 0.132 |
| new bots: 上家 new − 上家 plain (N2 − N1) | −0.481 ± 0.130 | −0.497 ± 0.130 |
| against the twin: N1 / N2 / O1 / O2 | +0.215 / −0.254 / −0.210 / +0.249 (each ± 0.05–0.06) | +0.220 / −0.264 / −0.214 / +0.257 |
| new-bot hands where the rule changed the suit | 18.16 % | 18.38 % |
| new-bot hands with three candidate suits at least once | | 13.94 % |
| new-bot hands where the choice differed from the two-suit rule's at least once | | 0.44 % |
| neighbour reading right / not on a flush line / other suit | 84.8 % / 14.4 % / 0.7 % (20.1 M readings) | the same |
| changed hands: points, this bot / same seat in the twin | −6.51 / −6.45 (diff −0.06 ± 0.15) | −6.52 / −6.45 (diff −0.07 ± 0.15) |
| changed hands: win | 13.5 % / 14.5 % | 13.5 % / 14.5 % |
| changed hands: chows claimed from 上家 | 0.408 / 0.411 a hand | 0.407 / 0.411 |
| changed hands: melds 下家 made with the bot's discards | 0.780 / 0.957 a hand | 0.779 / 0.958 |
| changed hands: ended in a flush | 62.4 % / 70.7 % | 62.3 % / 70.5 % |

* A third suit is a candidate in 14 % of the new bots' hands, but the three-suit choice differs
  from the two-suit rule's in only 0.44 %: when three suits are within one tile, the neighbours'
  reading almost always settles on one of the top two anyway (or settles nothing). The two bots
  play almost the same games, and every figure agrees within a few hundredths.

## Verdict

**Does the rule gain over plain `stage`?** **No: level.** Over 200,000 deals of Brian's
new, new, old, old table, the two new bots score +0.040 ± 0.093 points a hand more than the two
plain bots with the two-suit rule and +0.036 ± 0.093 with the three-suit rule, both inside
their ranges (the flush-distance reading, 40,000 deals: −0.26 ± 0.22). In the hands where it
changes the suit, the bot ends in a flush less often (62 % against 71 %) and wins less often
(13.5 % against 14.5 %) for the same points (−0.06 ± 0.15). What the rule does measurably is
starve the seat after it: 下家 makes 0.78 melds a hand from the bot's discards against 0.96, and
**Brian's guess is right**: the plain bot whose 上家 is a new bot scores 0.45 ± 0.13 points a
hand less than the plain bot whose 上家 is plain (three suits 0.47 ± 0.13); a new bot after a
new bot loses the same (0.48 ± 0.13). The seat before a new bot gains about as much (+0.22 to
+0.26 against the twin). In a table of one new bot and three plain ones that starving effect
would hurt only the plain 下家, which this set-up does not measure. Nothing becomes a default.

## One against three (plan Task 22, 2026-11-21)

Brian, 2026-11-21, on the +0.040 ± 0.093 above: "I think this is good evidence that the
advantage is here although quite small"; offered one new bot against three plain ones (the full
rule, and rule 2 alone, on the same deals): "1v3 is good, go ahead." With one new bot only a plain
bot sits after it, so the points 下家 loses no longer cancel out of the gain.

* New bot `stage-neighbour-next` (switch value 4): rule 2 alone. Close as in
  `stage-neighbour-tiles` (tile counts at most 1 apart); of the two close suits 下家's is kept;
  上家's suit plays no part. Tests `neighbour_rule_two_alone` and
  `neighbour_switch_off_leaves_stage_as_before` (plain `stage` plays 200 seeded deals with the
  same action hash as on the Task 21 code).
* Match: `flush_neighbours one 200000 21` (both bots in one pass on the same deals): one new bot
  and three plain `stage`, the new bot rotated through the four winds, each rotation beside its
  all-plain twin on the same wall and random stream. 12 games a deal, 4,081 s, run directly from
  the checkout with power throttling lifted. Log `runs/flush-neighbours/one-200000-seed21.txt` in
  the workers clone. Ranges are 95 % over deals (the four rotations of a deal are one sample).

| against the same seat in the all-plain twin | full rule `stage-neighbour-tiles` | rule 2 alone `stage-neighbour-next` |
|---|---|---|
| **new bot (the gain)** | **−0.019 ± 0.037** | **−0.051 ± 0.029** |
| new bot: win / twin | 23.95 % / 24.11 % | 23.80 % / 24.11 % |
| new bot: deal-in / twin | 18.25 % / 18.15 % | 18.07 % / 18.15 % |
| 下家 (plain, plays after the new bot) | −0.227 ± 0.044 (win 23.62 % / 24.05 %) | −0.272 ± 0.035 (23.54 % / 24.05 %) |
| opposite (plain) | +0.012 ± 0.043 (24.07 % / 24.07 %) | +0.156 ± 0.034 (24.35 % / 24.07 %) |
| 上家 (plain, plays before the new bot) | +0.235 ± 0.042 (24.49 % / 23.97 %) | +0.168 ± 0.033 (24.27 % / 23.97 %) |
| new-bot hands where the rule changed the suit | 18.01 % | 10.70 % |
| changed hands: points, new bot / twin | −6.56 / −6.45 (diff −0.11 ± 0.21) | −7.26 / −6.78 (diff −0.48 ± 0.27) |
| changed hands: win | 13.5 % / 14.4 % | 11.9 % / 14.7 % |
| changed hands: melds 下家 made with the bot's discards | 0.775 / 0.964 a hand | 0.602 / 1.080 |
| changed hands: ended in a flush | 62.4 % / 70.5 % | 60.4 % / 70.3 % |

Raw points a hand (deal-in and win rates beside them in the log): full rule +0.026 ± 0.079,
下家 −0.194, opposite −0.025, 上家 +0.192; rule 2 alone −0.005 ± 0.077, 下家 −0.239, opposite
+0.119, 上家 +0.126 (each ± 0.08).

* **Full rule: no gain.** −0.019 ± 0.037 is level with plain `stage`. 下家 is starved (0.23
  points a hand), but the points go to 上家 (+0.24), not to the new bot.
* **Rule 2 alone: a small loss.** −0.051 ± 0.029 is below zero beyond its range. It starves
  下家 harder (−0.27; 0.60 melds from the bot's discards against 1.08 in the changed hands), but
  keeping 下家's suit costs the bot its own hand: in the changed hands it wins 11.9 % against
  14.7 % and ends in a flush 60 % against 70 %. The points go to opposite and 上家 (+0.16, +0.17).
* **The two compared** on the same deals and seats: rule 2 alone − full rule −0.032 ± 0.029,
  so rule 2 alone is a little worse, just beyond its range.

### Verdict

Neither rule gains with one new bot against three plain ones: the full rule is level
(−0.019 ± 0.037) and rule 2 alone loses a little (−0.051 ± 0.029). The rule hurts 下家, as
Brian said, but what 下家 loses goes to the two other plain seats, not to the bot that starves it,
and the bot gives up some of its own flushes and wins to do it. Nothing becomes a default.
