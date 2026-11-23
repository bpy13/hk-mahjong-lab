# Where the defence bots lose points to false alarms (plan 2026-11-16 Task 24)

Brian, 2026-11-23: "well maybe we should figure out how to deal with the false alarm. You should be
able to figure out which hands' EV were lost because of false alarm, and we can investigate those".
This study logs, for four defence bots, each hand's first changed discard, labels it a true or a
false alarm from the true hands, and splits the points by kind.

## Set-up

* Bots, all reading `models/belief-order.mjnn` (Task 19), each one against three plain `stage`:
  * `safe1` = `stage-safe:t=1` and `safe5` = `stage-safe:t=5` (new switch `StageBot::safe`, off by
    default): for a hand that is not ready, when the reading's deal-in chance of `stage`'s pick
    (`1 - prod(1 - dealin_o)` over the three opponents, as in Task 23) is above the threshold, throw
    `stage`'s pick among the tiles read below it; no such tile, no change. Thresholds 0.1347 (`t=1`)
    and 0.0220 (`t=5`) flag 1 % and 5 % of the safe picks of not-ready hands in
    `runs/defence-ceiling/reading-40000-seed23.txt` (seed 23, so this match on seed 24 is out of
    sample).
  * `ev` = `stage-ev` (Brian's EV bot, Task 20) and `fold` = `stage-fold` (the give-up rule of
    Task 11 for a ready hand), as they are.
* Match: `crates/search/examples/defence_false_alarms.rs match 40000 24`, cpu job 308 (script
  `experiments/stage/false_alarms.ps1`), 2,744 s, exit 0. Each deal is played with four plain
  `stage` and with each bot in each seat against three plain `stage` on the same wall; the two games
  are equal until the bot's first changed decision (checked: 0 unchanged hands differ from the plain
  game, for every bot). Paired = the bot's points minus plain `stage`'s in the same seat on the same
  wall; the 95 % range is over deals. Output `runs/defence-false-alarms/match-40000-seed24.txt` in
  the workers clone.
* Label: at the first changed discard, a **true alarm** if `stage`'s pick would have been won on at
  once (the engine settles the claim window with every possible winner declaring, liability
  included), a **false alarm** otherwise; a first change that is not a discard (a claim or a win
  passed) is its own row. Test `false_alarm::tests::false_alarm_label_agrees_with_the_engine`
  checks the label against the engine's own settlement on 10 deals × 4 seats per bot.
* Rule: the main table, the split tables and the example hands were played under the 12-tile
  liability rule as it stood on 2026-11-23 (liability for a discard win too). Task 27 changed it to
  self-draw only (ADR 0004) and replayed this match: cpu job 310, 2,466 s, exit 0, output
  `runs/defence-false-alarms/match-40000-seed24-selfdraw-rule.txt` in the slot 2 worktree (with its
  own split tables and example hands); its headline figures are in the table "Under the amended
  rule" beside the old ones. Task 31 then made a deal-in cost the discarder 2 shares (ADR 0005)
  and queued the match a third time: cpu job 313, output
  `runs/defence-false-alarms/match-40000-seed24-two-share-rule.txt` in the workers clone
  (figures to follow, under "Under the two-share rule").
* "From ready" is the faan-aware distance after the best discard (0 = ready with a legal win).
  Plain `stage` deals in in 18.20 % of its hands, 1.174 % of its discards (15.5 discards a hand).

## Results (40,000 deals × 4 seats)

| Bot | Paired − plain `stage` (a hand) | Win | Deal-in | Hands changed | First changes: true / false | True-alarm points | False-alarm points | Not a discard | Sum | Mean false alarm | Mean true alarm caught |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain `stage` | — | 24.08 %¹ | 18.20 % | — | — | — | — | — | — | — | — |
| `safe1` | **+0.242 ± 0.074** | 23.93 % | 16.77 % | 13.96 % | 16.1 % / 83.3 % | +0.2296 | +0.0059 | +0.0061 | +0.2416 | +0.05 | +10.21 |
| `safe5` | −0.164 ± 0.139 | 22.03 % | 13.51 % | 45.09 % | 7.2 % / 92.3 % | +0.3546 | −0.5165 | −0.0022 | −0.1641 | −1.24 | +10.89 |
| `ev` | **−2.293 ± 0.178** | 19.56 % | 16.19 % | 53.87 % | 4.2 % / 94.2 % | +0.2952 | −2.5711 | −0.0166 | −2.2925 | −5.06 | +13.16 |
| `fold` | **−0.675 ± 0.069** | 22.08 % | 17.36 % | 8.12 % | 12.4 % / 87.2 % | +0.2140 | −0.8843 | −0.0049 | −0.6752 | −12.50 | +21.32 |

Points are a hand over all 160,000 hands; the three parts sum to the paired gain in every row
(the job prints the sum beside the gain; unchanged hands add +0.0000). ¹ plain `stage`'s win rate is
Task 23's (seed 23); this job prints only its deal-in rate.

### Under the amended rule (self-draw only, Task 27, job 310)

Same deals (seed 24), same bots and reading, binaries built from the engine after ADR 0004.

| Bot | Paired, old rule | Paired, amended rule | Win, new | Deal-in, new | First changes true / false, new | True-alarm points, new | False-alarm points, old → new | Mean false alarm, old → new | "Liable for a win" false alarms, old → new |
|---|---|---|---|---|---|---|---|---|---|
| plain `stage` | — | — | — | 18.20 % | — | — | — | — | — |
| `safe1` | +0.242 ± 0.074 | **+0.160 ± 0.067** | 23.93 % | 16.77 % | 16.9 % / 82.5 % | +0.2398 | +0.0059 → −0.0846 | +0.05 → −0.73 | 158 → 43 |
| `safe5` | −0.164 ± 0.139 | **−0.354 ± 0.130** | 22.03 % | 13.51 % | 7.4 % / 92.1 % | +0.3709 | −0.5165 → −0.7247 | −1.24 → −1.74 | 332 → 111 |
| `ev` | −2.293 ± 0.178 | **−2.310 ± 0.169** | 19.74 % | 16.23 % | 4.3 % / 94.1 % | +0.2992 | −2.5711 → −2.5916 | −5.06 → −5.12 | 349 → 123 |
| `fold` | −0.675 ± 0.069 | **−0.675 ± 0.063** | 22.09 % | 17.30 % | 13.0 % / 86.5 % | +0.2119 | −0.8843 → −0.8829 | −12.50 → −12.51 | 69 → 15 |

`safe1` still gains under the amended rule (+0.160 ± 0.067, 2.4 times its range from zero) but by
a third less: its true alarms bring the same (+0.24 a hand) and its false alarms, level under the old
rule, now cost −0.085 a hand. Under the old rule a changed pick often also escaped a liability the
plain game paid on another seat's discard win (its "same ending" false alarms were worth +0.073 a
hand, now +0.012); that part of the gain came from the wrong rule. `safe5` loses twice as much for
the same reason; `ev` and `fold` are unchanged. The split tables below are the old rule's; in the
new run the one-away hands carry the whole false-alarm loss of `safe1` (9,816 hands, −0.086 a hand,
−1.40 each; hands two or three away level), which supports the proposal in the verdict.

Per hand, false alarms are common: `safe1` raises one in 11.6 % of hands (0.75 % of discards),
`safe5` in 41.6 % (2.6 %), `ev` in 50.8 % (3.2 %), `fold` in 7.1 % (0.45 %). True alarms are 2.3 %,
3.3 %, 2.2 % and 1.0 % of hands.

### What the split tables say

* **`safe1` gains, and its false alarms cost nothing on average** (+0.0059 a hand, +0.05 each). The
  gain is the true alarms: +10.21 a catch, a third of the 28.94 its pick would have paid at once
  (the rest goes to later deal-ins and others' self-draws, as in Task 23). Inside the false alarms,
  the losing kinds are: the bot one away from ready (9,913 hands, −0.048 a hand, −0.78 each; hands
  two or more away gain +0.054), the flagged opponent truly far (−0.052, −3.08 each) or truly ready
  (−0.026), and an opponent with one or two open sets (−0.071). These are cancelled by gains
  elsewhere, so the "share of the sum" column of `safe1` (shares of a near-zero sum) is not
  meaningful; read its points column. Task 23 expected at most +0.38 a hand at this setting with free
  false flags; `safe1` keeps 64 % of that.
* **`safe5` loses because its false alarms are cheap one by one but many**: 66,619 of them (41.6 %
  of hands), −1.24 each, −0.517 a hand, more than its true alarms bring (+0.355). The loss sits in:
  the bot **one away from ready** (−0.463, 90 % of the loss; with the reading below 0.05 alone
  −0.321, 62 %); the reading's figure **0.022–0.050** (−0.398, 77 %; above 0.135 the false alarms
  are level, as in `safe1`); the flagged opponent **truly far** (−0.391, 76 %, −3.24 each); one or
  two open sets (−0.574, 111 %); the middle third of the wall (91 %). The ending: wins lost (−2.25)
  and others' self-draws (−0.79) outweigh the better endings (+2.88). Hands three or more from ready
  are level or gain at any reading.
* **`ev`'s loss is not the reading's false alarms but its own rule.** 65 % of its false alarms
  change a pick the reading rates **below 0.010**, and 91 % are in one-away hands; mean −5.06 each.
  Its loss is wins lost: 12,523 hands, −4.81 a hand against all of `ev`'s −2.29 (plain won → bot
  did not: −4.95). The bot gives up speed for safety it does not need.
* **`fold` loses −12.50 per false alarm**, all on ready hands (that is its range); 76 % of them break
  the ready hand (one step of progress lost, −13.10 each) and 34 % turn a win into no win (−1.11 a
  hand). 42 % of its folds come when the reading rates `stage`'s pick below 0.010: its trigger is
  the danger of the opponent, not of the tile.

## Split tables, per bot

Each table splits the first changes that are false alarms. Columns: hands, share of the false
alarms, points a hand (over all 160,000 hands), share of the false-alarm sum, mean points each.
"Progress the replacement cost" compares the distance from ready after the thrown tile with that
after `stage`'s pick. "Ending beside the plain game" sorts each hand by how the bot's game ended
against the plain game (liable for a win = the bot paid for a 12-tile liability).

### `safe1`

```text
  by the bot's distance from ready: bin | hands | share | points a hand | share of the sum | mean each
    1                         9913   53.3%  -0.0483  -819.5%  -0.78
    2                         5130   27.6%  +0.0336   568.6%  +1.05
    3                         2217   11.9%  +0.0200   339.8%  +1.45
    4+                        1349    7.2%  +0.0006    11.0%  +0.08
  by wall third: bin | hands | share | points a hand | share of the sum | mean each
    1                          265    1.4%  +0.0019    33.1%  +1.18
    2                        11390   61.2%  +0.0083   141.5%  +0.12
    3                         6954   37.4%  -0.0044   -74.6%  -0.10
  by the flagged opponent's open sets: bin | hands | share | points a hand | share of the sum | mean each
    0                          417    2.2%  +0.0017    28.8%  +0.65
    1                         2774   14.9%  -0.0211  -357.6%  -1.22
    2                         7791   41.9%  -0.0495  -838.1%  -1.02
    3                         6685   35.9%  +0.0665  1127.1%  +1.59
    4                          942    5.1%  +0.0083   139.8%  +1.40
  by the flagged opponent's true state: bin | hands | share | points a hand | share of the sum | mean each
    ready                     8164   43.9%  -0.0256  -433.9%  -0.50
    one away                  7769   41.7%  +0.0830  1406.8%  +1.71
    far (2+)                  2676   14.4%  -0.0515  -872.9%  -3.08
  by the reading's figure for the pick: bin | hands | share | points a hand | share of the sum | mean each
    0.135–0.250            10377   55.8%  -0.0019   -31.4%  -0.03
    0.250–0.500             5996   32.2%  +0.0130   220.3%  +0.35
    >= 0.500                  2236   12.0%  -0.0053   -89.0%  -0.38
  by progress the replacement cost (steps from ready): bin | hands | share | points a hand | share of the sum | mean each
    none or better           17658   94.9%  -0.0083  -139.8%  -0.07
    1 step                     951    5.1%  +0.0141   239.8%  +2.38
  by ending beside the plain game: bin | hands | share | points a hand | share of the sum | mean each
    liable for a win           158    0.8%  -0.0922  -1563.6%  -93.42
    win lost                  1111    6.0%  -0.4256  -7214.4%  -61.30
    later deal-in             1598    8.6%  -0.1246  -2111.9%  -12.48
    other's self-draw         1219    6.6%  -0.1576  -2671.2%  -20.69
    same ending              11295   60.7%  +0.0729  1235.6%  +1.03
    better ending             3228   17.3%  +0.7331  12425.4%  +36.34
  ending pairs (plain -> bot): hands, points a hand
    won              -> won                 1270  +0.0056
    won              -> dealt in             297  -0.1461
    won              -> other self-drew      191  -0.0927
    won              -> other's deal-in      444  -0.1423
    won              -> draw                 221  -0.0789
    dealt in         -> won                  187  +0.0881
    dealt in         -> dealt in            1796  +0.0026
    dealt in         -> other self-drew      504  -0.0211
    dealt in         -> other's deal-in     1025  +0.1805
    dealt in         -> draw                 291  +0.0553
    other self-drew  -> won                  140  +0.0788
    other self-drew  -> dealt in             467  +0.0336
    other self-drew  -> other self-drew     1894  -0.0026
    other self-drew  -> other's deal-in      639  +0.1627
    other self-drew  -> draw                 122  +0.0319
    other's deal-in  -> won                  261  +0.0825
    other's deal-in  -> dealt in             954  -0.1368
    other's deal-in  -> other self-drew      637  -0.1306
    other's deal-in  -> other's deal-in     4981  +0.0473
    other's deal-in  -> draw                 292  +0.0096
    draw             -> won                  100  +0.0349
    draw             -> dealt in             200  -0.0311
    draw             -> other self-drew       95  -0.0180
    draw             -> other's deal-in      211  -0.0070
    draw             -> draw                1390  +0.0000
  distance from ready × the reading's figure (hands, points a hand):
    dist 0  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]      0 -0.0000
    dist 1  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]   9913 -0.0483
    dist 2  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]   5130 +0.0336
    dist 3  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]   2217 +0.0200
    dist 4+  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]   1349 +0.0006
```

### `safe5`

```text
  by the bot's distance from ready: bin | hands | share | points a hand | share of the sum | mean each
    1                        27391   41.1%  -0.4633    89.7%  -2.71
    2                        20180   30.3%  -0.1063    20.6%  -0.84
    3                        11558   17.3%  -0.0035     0.7%  -0.05
    4+                        7490   11.2%  +0.0566   -11.0%  +1.21
  by wall third: bin | hands | share | points a hand | share of the sum | mean each
    1                         4154    6.2%  -0.0503     9.7%  -1.94
    2                        50003   75.1%  -0.4722    91.4%  -1.51
    3                        12462   18.7%  +0.0059    -1.1%  +0.08
  by the flagged opponent's open sets: bin | hands | share | points a hand | share of the sum | mean each
    0                         3399    5.1%  -0.0126     2.4%  -0.59
    1                        17179   25.8%  -0.2828    54.7%  -2.63
    2                        29619   44.5%  -0.2913    56.4%  -1.57
    3                        15157   22.8%  +0.0703   -13.6%  +0.74
    4                         1265    1.9%  -0.0001     0.0%  -0.02
  by the flagged opponent's true state: bin | hands | share | points a hand | share of the sum | mean each
    ready                    18808   28.2%  -0.0209     4.1%  -0.18
    one away                 28508   42.8%  -0.1043    20.2%  -0.59
    far (2+)                 19303   29.0%  -0.3912    75.7%  -3.24
  by the reading's figure for the pick: bin | hands | share | points a hand | share of the sum | mean each
    0.022–0.050            35407   53.1%  -0.3982    77.1%  -1.80
    0.050–0.135            21013   31.5%  -0.1239    24.0%  -0.94
    0.135–0.250             5966    9.0%  -0.0083     1.6%  -0.22
    0.250–0.500             3175    4.8%  +0.0187    -3.6%  +0.94
    >= 0.500                  1058    1.6%  -0.0049     0.9%  -0.73
  by progress the replacement cost (steps from ready): bin | hands | share | points a hand | share of the sum | mean each
    none or better           59409   89.2%  -0.4066    78.7%  -1.10
    1 step                    7209   10.8%  -0.1109    21.5%  -2.46
    2 steps                      1    0.0%  +0.0010    -0.2%  +160.00
  by ending beside the plain game: bin | hands | share | points a hand | share of the sum | mean each
    liable for a win           332    0.5%  -0.1852    35.8%  -89.23
    win lost                  5851    8.8%  -2.2513   435.9%  -61.56
    later deal-in             4135    6.2%  -0.3550    68.7%  -13.74
    other's self-draw         5424    8.1%  -0.7875   152.5%  -23.23
    same ending              36911   55.4%  +0.1781   -34.5%  +0.77
    better ending            13966   21.0%  +2.8844  -558.5%  +33.04
  ending pairs (plain -> bot): hands, points a hand
    won              -> won                 4785  +0.0195
    won              -> dealt in            1087  -0.5101
    won              -> other self-drew     1237  -0.6604
    won              -> other's deal-in     2490  -0.7778
    won              -> draw                1116  -0.3670
    dealt in         -> won                  645  +0.3035
    dealt in         -> dealt in            4194  +0.0032
    dealt in         -> other self-drew     2301  -0.1290
    dealt in         -> other's deal-in     4850  +0.8188
    dealt in         -> draw                1248  +0.2322
    other self-drew  -> won                  482  +0.2632
    other self-drew  -> dealt in            1203  +0.0911
    other self-drew  -> other self-drew     6556  +0.0186
    other self-drew  -> other's deal-in     2858  +0.6785
    other self-drew  -> draw                 565  +0.1477
    other's deal-in  -> won                  974  +0.3095
    other's deal-in  -> dealt in            2536  -0.3852
    other's deal-in  -> other self-drew     2830  -0.6176
    other's deal-in  -> other's deal-in    17989  +0.0935
    other's deal-in  -> draw                1297  +0.0301
    draw             -> won                  308  +0.0797
    draw             -> dealt in             416  -0.0737
    draw             -> other self-drew      337  -0.0759
    draw             -> other's deal-in      832  -0.0089
    draw             -> draw                3483  +0.0000
  distance from ready × the reading's figure (hands, points a hand):
    dist 0  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]      0 -0.0000
    dist 1  [<0.05]  13471 -0.3210  [0.05-0.135]   8978 -0.1148  [>=0.135]   4942 -0.0275
    dist 2  [<0.05]  10833 -0.0830  [0.05-0.135]   6344 -0.0240  [>=0.135]   3003 +0.0008
    dist 3  [<0.05]   6698 -0.0306  [0.05-0.135]   3459 +0.0081  [>=0.135]   1401 +0.0189
    dist 4+  [<0.05]   4405 +0.0364  [0.05-0.135]   2232 +0.0069  [>=0.135]    853 +0.0134
```

### `ev`

```text
  by the bot's distance from ready: bin | hands | share | points a hand | share of the sum | mean each
    0 (ready)                 7042    8.7%  -0.3621    14.1%  -8.23
    1                        74187   91.3%  -2.2090    85.9%  -4.76
  by wall third: bin | hands | share | points a hand | share of the sum | mean each
    1                        10987   13.5%  -0.3604    14.0%  -5.25
    2                        53431   65.8%  -1.8019    70.1%  -5.40
    3                        16811   20.7%  -0.4088    15.9%  -3.89
  by the flagged opponent's open sets: bin | hands | share | points a hand | share of the sum | mean each
    0                        18301   22.5%  -0.5916    23.0%  -5.17
    1                        25930   31.9%  -0.8919    34.7%  -5.50
    2                        24148   29.7%  -0.7329    28.5%  -4.86
    3                        11653   14.3%  -0.3200    12.4%  -4.39
    4                         1197    1.5%  -0.0349     1.4%  -4.66
  by the flagged opponent's true state: bin | hands | share | points a hand | share of the sum | mean each
    ready                    16333   20.1%  -0.3548    13.8%  -3.48
    one away                 24656   30.4%  -0.7326    28.5%  -4.75
    far (2+)                 40240   49.5%  -1.4837    57.7%  -5.90
  by the reading's figure for the pick: bin | hands | share | points a hand | share of the sum | mean each
    < 0.010                  52825   65.0%  -1.5969    62.1%  -4.84
    0.010–0.022             7914    9.7%  -0.2359     9.2%  -4.77
    0.022–0.050             7950    9.8%  -0.2864    11.1%  -5.77
    0.050–0.135             7027    8.7%  -0.2477     9.6%  -5.64
    0.135–0.250             2751    3.4%  -0.0999     3.9%  -5.81
    0.250–0.500             1901    2.3%  -0.0678     2.6%  -5.71
    >= 0.500                   861    1.1%  -0.0365     1.4%  -6.77
  by progress the replacement cost (steps from ready): bin | hands | share | points a hand | share of the sum | mean each
    none or better           64778   79.7%  -1.7898    69.6%  -4.42
    1 step                   16450   20.3%  -0.7812    30.4%  -7.60
    2 steps                      1    0.0%  -0.0001     0.0%  -24.00
  by ending beside the plain game: bin | hands | share | points a hand | share of the sum | mean each
    liable for a win           349    0.4%  -0.2348     9.1%  -107.64
    win lost                 12523   15.4%  -4.8084   187.0%  -61.44
    later deal-in             4044    5.0%  -0.3451    13.4%  -13.65
    other's self-draw         3819    4.7%  -0.5877    22.9%  -24.62
    same ending              46888   57.7%  +0.0193    -0.8%  +0.07
    better ending            13606   16.8%  +3.3855  -131.7%  +39.81
  ending pairs (plain -> bot): hands, points a hand
    won              -> won                15554  -0.0581
    won              -> dealt in            2481  -1.1865
    won              -> other self-drew     2606  -1.4158
    won              -> other's deal-in     5370  -1.6224
    won              -> draw                2223  -0.7273
    dealt in         -> won                 1329  +0.6375
    dealt in         -> dealt in            5172  +0.0088
    dealt in         -> other self-drew     1603  -0.1070
    dealt in         -> other's deal-in     3223  +0.5402
    dealt in         -> draw                1002  +0.1847
    other self-drew  -> won                 1042  +0.5452
    other self-drew  -> dealt in            1243  +0.0946
    other self-drew  -> other self-drew     6355  +0.0043
    other self-drew  -> other's deal-in     2101  +0.4981
    other self-drew  -> draw                 462  +0.1124
    other's deal-in  -> won                 2236  +0.6281
    other's deal-in  -> dealt in            2348  -0.3689
    other's deal-in  -> other self-drew     1979  -0.4476
    other's deal-in  -> other's deal-in    15708  +0.0331
    other's deal-in  -> draw                 950  +0.0193
    draw             -> won                  648  +0.2044
    draw             -> dealt in             478  -0.0852
    draw             -> other self-drew      268  -0.0594
    draw             -> other's deal-in      683  -0.0039
    draw             -> draw                4165  +0.0000
  distance from ready × the reading's figure (hands, points a hand):
    dist 0  [<0.05]   5242 -0.2379  [0.05-0.135]    786 -0.0493  [>=0.135]   1014 -0.0750
    dist 1  [<0.05]  63447 -1.8814  [0.05-0.135]   6241 -0.1984  [>=0.135]   4499 -0.1293
    dist 2  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]      0 -0.0000
    dist 3  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]      0 -0.0000
    dist 4+  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]      0 -0.0000
```

### `fold`

```text
  by the bot's distance from ready: bin | hands | share | points a hand | share of the sum | mean each
    0 (ready)                11320  100.0%  -0.8843   100.0%  -12.50
  by wall third: bin | hands | share | points a hand | share of the sum | mean each
    1                          215    1.9%  -0.0200     2.3%  -14.88
    2                         4641   41.0%  -0.3891    44.0%  -13.41
    3                         6464   57.1%  -0.4753    53.7%  -11.76
  by the flagged opponent's open sets: bin | hands | share | points a hand | share of the sum | mean each
    0                         2518   22.2%  -0.2460    27.8%  -15.63
    1                         2813   24.8%  -0.2059    23.3%  -11.71
    2                         3399   30.0%  -0.2497    28.2%  -11.75
    3                         2211   19.5%  -0.1479    16.7%  -10.70
    4                          379    3.3%  -0.0349     3.9%  -14.73
  by the flagged opponent's true state: bin | hands | share | points a hand | share of the sum | mean each
    ready                     3632   32.1%  -0.2624    29.7%  -11.56
    one away                  3955   34.9%  -0.2945    33.3%  -11.92
    far (2+)                  3733   33.0%  -0.3275    37.0%  -14.03
  by the reading's figure for the pick: bin | hands | share | points a hand | share of the sum | mean each
    < 0.010                   4733   41.8%  -0.3911    44.2%  -13.22
    0.010–0.022              959    8.5%  -0.0725     8.2%  -12.09
    0.022–0.050             1157   10.2%  -0.0728     8.2%  -10.07
    0.050–0.135             1562   13.8%  -0.0949    10.7%  -9.72
    0.135–0.250             1012    8.9%  -0.0738     8.3%  -11.67
    0.250–0.500             1046    9.2%  -0.0866     9.8%  -13.25
    >= 0.500                   851    7.5%  -0.0927    10.5%  -17.44
  by progress the replacement cost (steps from ready): bin | hands | share | points a hand | share of the sum | mean each
    none or better            2744   24.2%  -0.1822    20.6%  -10.63
    1 step                    8575   75.8%  -0.7019    79.4%  -13.10
    2 steps                      1    0.0%  -0.0001     0.0%  -24.00
  by ending beside the plain game: bin | hands | share | points a hand | share of the sum | mean each
    liable for a win            69    0.6%  -0.0469     5.3%  -108.64
    win lost                  3800   33.6%  -1.1121   125.8%  -46.83
    later deal-in              429    3.8%  -0.0394     4.4%  -14.68
    other's self-draw          440    3.9%  -0.0654     7.4%  -23.76
    same ending               5034   44.5%  +0.0149    -1.7%  +0.47
    better ending             1548   13.7%  +0.3645   -41.2%  +37.67
  ending pairs (plain -> bot): hands, points a hand
    won              -> won                 1127  +0.0061
    won              -> dealt in             427  -0.1876
    won              -> other self-drew      614  -0.3059
    won              -> other's deal-in     1422  -0.3311
    won              -> draw                1385  -0.3252
    dealt in         -> won                   72  +0.0383
    dealt in         -> dealt in             242  +0.0036
    dealt in         -> other self-drew      186  -0.0103
    dealt in         -> other's deal-in      362  +0.0607
    dealt in         -> draw                 206  +0.0428
    other self-drew  -> won                   96  +0.0553
    other self-drew  -> dealt in             107  +0.0112
    other self-drew  -> other self-drew      762  -0.0011
    other self-drew  -> other's deal-in      205  +0.0534
    other self-drew  -> draw                  75  +0.0227
    other's deal-in  -> won                  179  +0.0620
    other's deal-in  -> dealt in             186  -0.0277
    other's deal-in  -> other self-drew      226  -0.0497
    other's deal-in  -> other's deal-in     1590  +0.0030
    other's deal-in  -> draw                 204  +0.0083
    draw             -> won                   61  +0.0183
    draw             -> dealt in             140  -0.0244
    draw             -> other self-drew       30  -0.0069
    draw             -> other's deal-in       95  -0.0001
    draw             -> draw                1321  +0.0000
  distance from ready × the reading's figure (hands, points a hand):
    dist 0  [<0.05]   6849 -0.5363  [0.05-0.135]   1562 -0.0949  [>=0.135]   2909 -0.2531
    dist 1  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]      0 -0.0000
    dist 2  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]      0 -0.0000
    dist 3  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]      0 -0.0000
    dist 4+  [<0.05]      0 -0.0000  [0.05-0.135]      0 -0.0000  [>=0.135]      0 -0.0000
```

## Example hands (20 a bot)

The costliest false alarms of each bot, drawn across the kinds of ending. Tiles in brackets in a
discard list were claimed by another seat; `[1111m]` is a concealed kong. "read deal-in on X" is the
reading's chance that this opponent wins on `stage`'s pick X. The same hands can be opened as files in
the app (`defence_false_alarms examples-json`, Task 25).

### `safe1`: the costliest false alarms across the endings

#### Example 1: liable for a win, -456 points (deal 24007403, seat 0; 57 tiles left in the wall, wall third 2)

* bot (seat 0): `7m 1122556s S WhWh`, open `234s`; 2 from ready
  discards: 9m (9p) 1p 3p W 5m (4p) (9s) R
* next (下家) (seat 1): `34m 6p 1s`, open `777p 444s 999s`; truly 2 from ready; read deal-in on 7m: 0.003
  discards: 8m 4p N W 3s 3p E 9s G
* opposite (對家) (seat 2): `114779m Wh`, open `666m 222m`; truly 2 from ready; read deal-in on 7m: 0.132
  discards: 3s (7p) 5p (4s) 3p W E G E 8p R
* previous (上家) (seat 3): `12368p SS`, open `999p 444p`; truly 0 from ready; read deal-in on 7m: 0.000
  discards: (4s) 2s 8m (6m) 5m 2m 3m G 5p (2m) R

`stage`'s pick 7m, read 0.135 (nobody wins on it); thrown instead S (costs 0 steps of progress).
Plain game: bot self-drew 5 faan (bot +72). Bot game: opposite (對家) won off previous (上家) with 10 faan (bot -384).

#### Example 2: liable for a win, -400 points (deal 24036308, seat 2; 43 tiles left in the wall, wall third 2)

* bot (seat 2): `77m 556p 11s G`, open `444p RRR`; 1 from ready
  discards: 4s (W) 7s 6s 1p (5m) (4m) 2p 2m (9p) 3p (E) 8m
* next (下家) (seat 3): `2357p 233s NN R`, open `789p`; truly 3 from ready; read deal-in on G: 0.005
  discards: 7s 6m 8m 9m 8s 6s 3s 9m 4s
* opposite (對家) (seat 0): `23m GG`, open `WWW 444m EEE`; truly 0 from ready; read deal-in on G: 0.246
  discards: 7s 5p 9p 4s 2s Wh 8m 7s 9m (S) 1p (R) 9m
* previous (上家) (seat 1): `3m 7789p 55s`, open `555m SSS`; truly 2 from ready; read deal-in on G: 0.002
  discards: 8s 9s (4p) 2p 2m 6m W 1m 1m 3p S 8s E

`stage`'s pick G, read 0.251 (nobody wins on it); thrown instead 6p (costs 0 steps of progress).
Plain game: bot won off opposite (對家) with 4 faan (bot +16). Bot game: opposite (對家) won off previous (上家) with 10 faan (bot -384).

#### Example 3: liable for a win, -296 points (deal 24011168, seat 3; 31 tiles left in the wall, wall third 2)

* bot (seat 3): `44556m 577s`, open `333m 666p`; 1 from ready
  discards: 2m (1s) Wh 9m 2s 2s R 6s (W) (8s) 3s 9p 9p (S) N 3s
* next (下家) (seat 0): `13556s E R`, open `GGG 678s`; truly 2 from ready; read deal-in on 5s: 0.137
  discards: (3m) 5m 7m (6p) 5p 7p 6m 2p 1p 9p 2p 8p Wh 2m 9m
* opposite (對家) (seat 1): `1344559p 4s W N`, open `777p`; truly 3 from ready; read deal-in on 5s: 0.000
  discards: 7m 4m 3m 1m 9s 8m 8s 9m 1m 6s 2s 3s 4s 4s
* previous (上家) (seat 2): `89s EE`, open `111s WWW SSS`; truly 0 from ready; read deal-in on 5s: 0.163
  discards: 2p 2m 1p 2m 9m 8m Wh (G) 8p 1m 1p 4s (7p) 8p N N G

`stage`'s pick 5s, read 0.278 (nobody wins on it); thrown instead 6m (costs 0 steps of progress).
Plain game: previous (上家) won off next (下家) with 4 faan; bot won off next (下家) with 3 faan (bot +8). Bot game: previous (上家) won off bot with 9 faan (bot -288).

#### Example 4: win lost, -384 points (deal 24006868, seat 0; 37 tiles left in the wall, wall third 2)

* bot (seat 0): `2456678899p E Wh GG`; 2 from ready
  discards: 7s 3s (9m) 8s 6m 4m 2m 5s 2m W S (4m) 8m
* next (下家) (seat 1): `146678m NN Wh G`, open `234m`; truly 2 from ready; read deal-in on Wh: 0.012
  discards: 3s 6s 8s 4s 2s 5p 9s 4p 2p 1p W S
* opposite (對家) (seat 2): `3m 37779p Wh`, open `555m 999m`; truly 2 from ready; read deal-in on Wh: 0.130
  discards: 8s 6s 1m 7m E 7s W 4s 7m 4p (2p) 1p S 4m
* previous (上家) (seat 3): `12688p 9s E S NN`, open `123p`; truly 3 from ready; read deal-in on Wh: 0.001
  discards: (5m) 1m 1m 8s 1s 6s 5s 5m 5s 1s 4s 4s 1s 7m

`stage`'s pick Wh, read 0.142 (nobody wins on it); thrown instead 2p (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: draw (wall ran out).

#### Example 5: win lost, -384 points (deal 24008539, seat 0; 50 tiles left in the wall, wall third 2)

* bot (seat 0): `69p 1235699s W G`, open `456s`; 2 from ready
  discards: 2m 6m 4m 8m 5m 8p 7p 5p 7p 4p
* next (下家) (seat 1): `19m 899p 1s SS W N Wh G R`; truly 2 from ready; read deal-in on 6p: 0.000
  discards: 5m 5s 7m 2s 7m (4p) 3m 6m 6p 3p
* opposite (對家) (seat 2): `4m 22589s E WhWh G`, open `777s`; truly 3 from ready; read deal-in on 6p: 0.000
  discards: 5m 7m 3m 2m 8p 8p 5p 2p (2p) 7m
* previous (上家) (seat 3): `559p S G RR`, open `444p 123p`; truly 2 from ready; read deal-in on 6p: 0.359
  discards: 3m 6m 3s 2m 9m (6s) (7s) W N 1p N

`stage`'s pick 6p, read 0.359 (nobody wins on it); thrown instead W (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: opposite (對家) won off previous (上家) with 6 faan (bot +0).

#### Example 6: win lost, -384 points (deal 24012344, seat 1; 9 tiles left in the wall, wall third 3)

* bot (seat 1): `679p WW NN Wh`, open `333p RRR`; 1 from ready
  discards: 6m 2m 9m 7s 5s 4s 1s (6s) 6s 5s E 1m 1p 8s 5m (S) 2s 3m 9m 2p
* next (下家) (seat 2): `11245s N WhWh GG`, open `678s`; truly 2 from ready; read deal-in on Wh: 0.189
  discards: 6m 2m 9m 8m 3p 6p 4m 7p 5m 8p 9p 2p 9m 8s (R) 6p 9p 5p 1m
* opposite (對家) (seat 3): `12345677m Wh G`, open `SSS`; truly 1 from ready; read deal-in on Wh: 0.053
  discards: 5p 8p 7s 4p 9s 8p 1p 4s 5p 9s 4p 6s 6s 3s 3s 3s 8p 7p E
* previous (上家) (seat 0): `1m 1129p 19s E S W N G R`; truly 1 from ready; read deal-in on Wh: 0.001
  discards: 4m 7m 7p 4p 2s 8s 2s 4m 9s E 8m 3m (3p) 7m 4s 2p 7s 5m 3m 3s

`stage`'s pick Wh, read 0.233 (nobody wins on it); thrown instead 9p (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: draw (wall ran out).

#### Example 7: win lost, -384 points (deal 24013370, seat 2; 31 tiles left in the wall, wall third 2)

* bot (seat 2): `122334445689s W Wh`; 1 from ready
  discards: 8m 1m (2p) 9p 3m 3m 6m (8p) 5p 4p 5p 6p E 3p
* next (下家) (seat 3): `99p 22567789s S N G`; truly 3 from ready; read deal-in on W: 0.000
  discards: 7m 3m 2m 4p 1m 7m 2m 6p 1m 4p 7p 9m
* opposite (對家) (seat 0): `113678p SS WW`, open `222p`; truly 1 from ready; read deal-in on W: 0.017
  discards: 7s 3s 2m 5m 8m 5m 9m N (4m) 6m 6m 7p 5m G
* previous (上家) (seat 1): `1p 9s E S W Wh R`, open `888p 444m`; truly 4 from ready; read deal-in on W: 0.151
  discards: 8s 3s 1s 1s 8m 5m 5p 4s 3m 4m N 9p 2m 7p 4p

`stage`'s pick W, read 0.166 (nobody wins on it); thrown instead Wh (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: draw (wall ran out).

#### Example 8: win lost, -384 points (deal 24030327, seat 2; 35 tiles left in the wall, wall third 2)

* bot (seat 2): `122445699m W N`, open `123m`; 1 from ready
  discards: 2s 7s 9s 6p 2p (3p) 4p 9p 5p 6p (7p) G 8p (6s)
* next (下家) (seat 3): `12789s SSS W R`, open `456s`; truly 1 from ready; read deal-in on W: 0.015
  discards: 3m 4m 2p 9m 8p 9p 9p 5p 1p 1p N 7s
* opposite (對家) (seat 0): `666777m 889s EE RR`; truly 1 from ready; read deal-in on W: 0.000
  discards: 5p 6p 8p 5s 5s 4p 4s 2p 9p 6s 3p 1m 5m
* previous (上家) (seat 1): `46p W WhWhWh R`, open `333p 777p`; truly 1 from ready; read deal-in on W: 0.192
  discards: 8m 3m 5m 6s 8s 9m 9s 3s 3m 1s 1s 1p G 7s (1m)

`stage`'s pick W, read 0.204 (nobody wins on it); thrown instead N (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: next (下家) won off previous (上家) with 3 faan (bot +0).

#### Example 9: win lost, -384 points (deal 24033063, seat 0; 15 tiles left in the wall, wall third 3)

* bot (seat 0): `367789m 3s R`, open `234m 567m`; 1 from ready
  discards: 7s 2s 1s 9s 4p 9p 5p 2p 1p 8p 9p 2p 8p (9s) W 4s E (6s)
* next (下家) (seat 1): `3s N RR`, open `789s GGG 456s`; truly 1 from ready; read deal-in on 3s: 0.111
  discards: 2m 6m 8m 1m 6p 5p 3p 1p 1m 9m S 9m 4m 1s (W) 9p 1s 9s 8s
* opposite (對家) (seat 2): `123p NN WhWh`, open `777p WWW`; truly 0 from ready; read deal-in on 3s: 0.013
  discards: 6s 9s 6m 8m E 8p 5m 9p 4m 4s S 6s 7m (7s) (G) 5p S 3p 2s 6p
* previous (上家) (seat 3): `234555s N WhWh R`, open `678s`; truly 1 from ready; read deal-in on 3s: 0.039
  discards: 2p 6p 4p (7p) 7p 4p 8m 2m 4m (3m) 1p 3m (7m) 1m 5p G 7s 8p 1s

`stage`'s pick 3s, read 0.156 (nobody wins on it); thrown instead 3m (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: draw (wall ran out).

#### Example 10: win lost, -384 points (deal 24035691, seat 2; 52 tiles left in the wall, wall third 2)

* bot (seat 2): `179m 118899p 1s S N G R`; 3 from ready
  discards: 6p 7s 6m 6s 5p 3p 7p 9p
* next (下家) (seat 3): `5m 1234569s W NN Wh R`; truly 3 from ready; read deal-in on 7m: 0.000
  discards: 4p 3p 6p 3m 8p 6p 7p 5p
* opposite (對家) (seat 0): `122334678s EE WW`; truly 0 from ready; read deal-in on 7m: 0.000
  discards: 3p 6p 8m 2p (4m) 9m G R 1m
* previous (上家) (seat 1): `77899m 77888s`, open `444m`; truly 1 from ready; read deal-in on 7m: 0.170
  discards: 8p 1p 5s 2p S 1m G 4m Wh

`stage`'s pick 7m, read 0.170 (nobody wins on it); thrown instead 1p (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: opposite (對家) won off next (下家) with 4 faan (bot +0).

#### Example 11: win lost, -384 points (deal 24037135, seat 1; 27 tiles left in the wall, wall third 3)

* bot (seat 1): `19m 19p 1s EE SS WW N G R`; 1 from ready
  discards: 7s 5m 7p 5s 4p 7p 9p 7m 9m 4s 2s 2s R 8m
* next (下家) (seat 2): `13456p EE S W N`, open `222p`; truly 2 from ready; read deal-in on E: 0.001
  discards: 8m 6m 3s 1s 8m 4s 1s 8s 4s 4s 8s Wh 6s (4m) 9p
* opposite (對家) (seat 3): `1255678p 566778s`; truly 4 from ready; read deal-in on E: 0.000
  discards: 8m 1m 5m G 6m 8p Wh 3s 1s 1p 8p 8p 9m 1m
* previous (上家) (seat 0): `223335m 3344p`, open `444m`; truly 1 from ready; read deal-in on E: 0.141
  discards: 2s 6s 5s 9s 3s 9p R S Wh N R (2p) 6p 8s 5m 3s

`stage`'s pick E, read 0.142 (nobody wins on it); thrown instead S (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: draw (wall ran out).

#### Example 12: later deal-in, -152 points (deal 24016456, seat 0; 20 tiles left in the wall, wall third 3)

* bot (seat 0): `246899m 1s E S W Wh GGG`; 3 from ready
  discards: 2p 2p 1p 1p 2s 5s 3p 7p (4s) 8p (8s) (6s) 6s 3s 9s 5p R
* next (下家) (seat 1): `5688s`, open `234s 789s 456s`; truly 0 from ready; read deal-in on 1s: 0.165
  discards: 1p 7m 4p 7p 7m 4p 4p 5m 4m 3m R 2s 3m 7m 8m S 8m
* opposite (對家) (seat 2): `1899m 9p 9s E S W N Wh G R`; truly 1 from ready; read deal-in on 1s: 0.011
  discards: 7s 6s 3p 7s 8p 4p 6m 2p 5m 2p 5p 6p 6m 7p 3m (5s) 6p
* previous (上家) (seat 3): `112389s NNN Wh`, open `345s`; truly 1 from ready; read deal-in on 1s: 0.016
  discards: 6m 4m 4m 3p 8p 5p 6p 9p 1p 8p 1m R 2m S 5m E 9p

`stage`'s pick 1s, read 0.187 (nobody wins on it); thrown instead S (costs 0 steps of progress).
Plain game: next (下家) won off opposite (對家) with 10 faan (bot +0). Bot game: next (下家) won off bot with 10 faan; previous (上家) won off bot with 5 faan (bot -152).

#### Example 13: later deal-in, -128 points (deal 24000158, seat 3; 31 tiles left in the wall, wall third 2)

* bot (seat 3): `345666m 6p W Wh G R`, open `222m`; 2 from ready
  discards: 7s 2s 1s 5s 9p 7s (7p) 7s 6p 2s 9s S 2s
* next (下家) (seat 0): `13467p WW`, open `789p GGG`; truly 1 from ready; read deal-in on 6p: 0.005
  discards: 3m 8m 9m 4s 5m 4m 5s 8s 8s 6s 8m 3s 9m 3s 4s
* opposite (對家) (seat 1): `19m 19p 19s EE S W N Wh R`; truly 0 from ready; read deal-in on 6p: 0.000
  discards: 5m (2m) 4m 6s 5m 5p 4s 3s 4s 2p (8p) (5p) 2s 3p 2p
* previous (上家) (seat 2): `567p EE RR`, open `888p 345p`; truly 0 from ready; read deal-in on 6p: 0.191
  discards: 9m 7m 6s 8s 1s 9s 1p 2m 3m S S (G) 1m 8s

`stage`'s pick 6p, read 0.195 (nobody wins on it); thrown instead G (costs 0 steps of progress).
Plain game: previous (上家) won off opposite (對家) with 5 faan (bot +0). Bot game: opposite (對家) won off bot with 10 faan (bot -128).

#### Example 14: later deal-in, -128 points (deal 24000851, seat 1; 47 tiles left in the wall, wall third 2)

* bot (seat 1): `2344556789p 3s N Wh R`; 2 from ready
  discards: 9s 4m 7m 4m 6m (7s) 8m (4s) 5s (4s)
* next (下家) (seat 2): `123s R`, open `789s 234s 345s`; truly 0 from ready; read deal-in on 3s: 0.152
  discards: 5m 6p 7p 7p 8m 9p W Wh 3m G
* opposite (對家) (seat 3): `256799m 11s E S W N G`; truly 5 from ready; read deal-in on 3s: 0.000
  discards: 5p 5p 7s 1p 7s 5s 2p 2s 4s 6s
* previous (上家) (seat 0): `19m 1139p 19s E S W N G`; truly 1 from ready; read deal-in on 3s: 0.000
  discards: 6s 4s 5m 3m 8m 1m S W 4m 1m 8m

`stage`'s pick 3s, read 0.152 (nobody wins on it); thrown instead Wh (costs 0 steps of progress).
Plain game: previous (上家) won off next (下家) with 10 faan (bot +0). Bot game: previous (上家) won off bot with 10 faan (bot -128).

#### Example 15: later deal-in, -128 points (deal 24002309, seat 2; 15 tiles left in the wall, wall third 3)

* bot (seat 2): `1333455789p EE N Wh`; 1 from ready
  discards: 5s 2s 2s 5s 8s 5s (4m) (9m) 6m (3m) 3m 3m 7s 1m 7m 1s 5m 8m
* next (下家) (seat 3): `7m NN G`, open `456m 789m 123m`; truly 1 from ready; read deal-in on N: 0.301
  discards: 9s 3s 3s 8p 3p 9p 2p 2p R W 1p 4p 1s 1p Wh 2m S 6s
* opposite (對家) (seat 0): `19m 1p 199s E S W N Wh G R`; truly 0 from ready; read deal-in on N: 0.036
  discards: 5s 6m 4s 8p 7p 6p 5m 7p 7m (4s) 3m 8p 4p R 8m E (8s) S 8s
* previous (上家) (seat 1): `1244678s`, open `234s 678s`; truly 0 from ready; read deal-in on N: 0.308
  discards: 6p 5p 6p 9p 2m 6p 5m 2p 4m 4m W 1m 8m R Wh S G G 6m

`stage`'s pick N, read 0.534 (nobody wins on it); thrown instead Wh (costs 0 steps of progress).
Plain game: previous (上家) won off next (下家) with 10 faan (bot +0). Bot game: previous (上家) won off bot with 10 faan (bot -128).

#### Example 16: other's self-draw, -128 points (deal 24001149, seat 0; 37 tiles left in the wall, wall third 2)

* bot (seat 0): `145567p 8s E`, open `9999p 567p`; 1 from ready
  discards: 9s 1s 3m (9m) 5m 9s N (R) 1s 6s 5m G Wh
* next (下家) (seat 1): `66m 8s E`, open `888p 555s 222p`; truly 1 from ready; read deal-in on 8s: 0.071
  discards: 8m 3s N 3m 7p 8m 4p 4m 6s 7s 4s 2s 9m G
* opposite (對家) (seat 2): `22457m E SS WW`, open `[1111m]`; truly 2 from ready; read deal-in on 8s: 0.000
  discards: 6p (8p) 2s (5s) 9s (2p) 4s 6p 4s Wh G 3p Wh 2s
* previous (上家) (seat 3): `77m 15p 33s W`, open `999m RRR`; truly 2 from ready; read deal-in on 8s: 0.101
  discards: 3p 8p (9p) 3m 8m 1s N 6s 7s 1s 4m (7p) G

`stage`'s pick 8s, read 0.164 (nobody wins on it); thrown instead E (costs 0 steps of progress).
Plain game: next (下家) won off previous (上家) with 3 faan; opposite (對家) won off previous (上家) with 5 faan (bot +0). Bot game: opposite (對家) self-drew 10 faan (bot -128).

#### Example 17: other's self-draw, -128 points (deal 24002990, seat 1; 40 tiles left in the wall, wall third 2)

* bot (seat 1): `1224456678m 7p N G R`; 2 from ready
  discards: 8p 3p 3p (1p) 4s 5p 1p 3s 6p 1s 5s 5s
* next (下家) (seat 2): `12333455m 2p E W Wh R`; truly 3 from ready; read deal-in on 7p: 0.000
  discards: 8p 6p (4p) (3p) 6s 2p 7s 3s 2s 1s 9s
* opposite (對家) (seat 3): `6789p`, open `456p 234p WhWhWh`; truly 0 from ready; read deal-in on 7p: 0.093
  discards: 8m 7m 3s 6s 2s 9m 8s 4s S 4s 9m E
* previous (上家) (seat 0): `5589p NN GG RR`, open `111p`; truly 2 from ready; read deal-in on 7p: 0.159
  discards: 1m 9m 5m 2s 4s 1m 8s 8s 8s S 3p 7s (Wh) 9s

`stage`'s pick 7p, read 0.237 (nobody wins on it); thrown instead G (costs 0 steps of progress).
Plain game: opposite (對家) won off previous (上家) with 4 faan (bot +0). Bot game: previous (上家) self-drew 10 faan (bot -128).

#### Example 18: other's self-draw, -128 points (deal 24004247, seat 2; 37 tiles left in the wall, wall third 2)

* bot (seat 2): `113345677789p 6s Wh`; 1 from ready
  discards: 4s 7m 2m 3m 4m 4m 4m 3m G W
* next (下家) (seat 3): `224555p Wh`, open `SSS 888p`; truly 1 from ready; read deal-in on 6s: 0.000
  discards: 6m 8m 2m 8s 5s 7s 8m 6s 1s G N 9p
* opposite (對家) (seat 0): `1355778m`, open `999m 666m`; truly 1 from ready; read deal-in on 6s: 0.000
  discards: 8s 5s 4s (4s) 1s 7s 2s 2s 1s G G (S) 3s 6p Wh
* previous (上家) (seat 1): `67899s Wh R`, open `345s [EEEE]`; truly 1 from ready; read deal-in on 6s: 0.285
  discards: 6p 7p 5m (9m) 9m 8m 4m 2s W 3s 7s (8p) (6m) 2s

`stage`'s pick 6s, read 0.285 (nobody wins on it); thrown instead Wh (costs 0 steps of progress).
Plain game: previous (上家) won off next (下家) with 5 faan (bot +0). Bot game: opposite (對家) self-drew 10 faan (bot -128).

#### Example 19: other's self-draw, -128 points (deal 24004699, seat 3; 50 tiles left in the wall, wall third 2)

* bot (seat 3): `24458p 134s E W N Wh G R`; 6 from ready
  discards: 1m 7m 9m 9m 7m 7s 8m 1m 2s
* next (下家) (seat 0): `33346m 4455s E WhWh R`; truly 3 from ready; read deal-in on 4s: 0.000
  discards: 5p G 3p 2p (2s) S S 6p 7p 1p
* opposite (對家) (seat 1): `1346699s E WW`, open `123s`; truly 2 from ready; read deal-in on 4s: 0.188
  discards: 8p 8p 7m (7p) 5m 2m G N 9m R
* previous (上家) (seat 2): `2446m 2455p 55s`, open `777p`; truly 3 from ready; read deal-in on 4s: 0.000
  discards: 7s 1p 3p 8s 5m S 1m S 8m 8m

`stage`'s pick 4s, read 0.188 (nobody wins on it); thrown instead 3s (costs 0 steps of progress).
Plain game: opposite (對家) won off next (下家) with 7 faan (bot +0). Bot game: opposite (對家) self-drew 10 faan (bot -128).

#### Example 20: same ending, -352 points (deal 24021277, seat 1; 50 tiles left in the wall, wall third 2)

* bot (seat 1): `126699s SS W N Wh RRR`; 2 from ready
  discards: 7p 7s 4p 3p 6m 2m (3m) 7m (6p) E
* next (下家) (seat 2): `12457799p 2234s W`; truly 4 from ready; read deal-in on Wh: 0.000
  discards: 5m 7m 2m 1m 8s 6s 9m 8s
* opposite (對家) (seat 3): `2335788p 3s W N`, open `666p`; truly 3 from ready; read deal-in on Wh: 0.000
  discards: 5m 8m (8m) 1m 7s 2s 1s 3s 6m
* previous (上家) (seat 0): `46m WhWh GGG`, open `789m 333m`; truly 0 from ready; read deal-in on Wh: 0.144
  discards: 9s 4s 4p 1p 2p R E 1m 8m E 7s

`stage`'s pick Wh, read 0.144 (nobody wins on it); thrown instead W (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: bot won off next (下家) with 6 faan (bot +32).

### `safe5`: the costliest false alarms across the endings

#### Example 1: liable for a win, -384 points (deal 24020177, seat 1; 55 tiles left in the wall, wall third 2)

* bot (seat 1): `7m 888p 23s WW G RR`, open `333m`; 2 from ready
  discards: 8m 2p 5s 4s 4p 9p 7p 2p
* next (下家) (seat 2): `77m 455p 78s S W R`, open `999s`; truly 4 from ready; read deal-in on 7m: 0.105
  discards: (3m) 5s (7p) E 1s N 3s N N
* opposite (對家) (seat 3): `7m 13p G`, open `777p 666p 999p`; truly 1 from ready; read deal-in on 7m: 0.027
  discards: 4s E 4s 1s 6s (9m) 1m 1m 8s 7s
* previous (上家) (seat 0): `12244568m WhWh`, open `999m`; truly 1 from ready; read deal-in on 7m: 0.015
  discards: 8s 7s (9s) (6p) 2p 3p (9p) 7s 1s G 8s

`stage`'s pick 7m, read 0.142 (nobody wins on it); thrown instead 3s (costs 0 steps of progress).
Plain game: opposite (對家) won off next (下家) with 10 faan (bot +0). Bot game: opposite (對家) self-drew 10 faan (bot -384).

#### Example 2: liable for a win, -384 points (deal 24025461, seat 3; 35 tiles left in the wall, wall third 2)

* bot (seat 3): `2455788m 5s E S R`, open `EEE`; 3 from ready
  discards: 3p 1p 6p 7p 9p 6p (6s) 5s (3s) (4s) 6p 2p 4s
* next (下家) (seat 0): `W`, open `666s 333s 888s SSS`; truly 0 from ready; read deal-in on 5s: 0.026
  discards: (1m) 8p 5p 2p 3m R 3m W 4m (E) (Wh) 9s 2s 5p 1s
* opposite (對家) (seat 1): `677m 677s G`, open `111m 444s`; truly 2 from ready; read deal-in on 5s: 0.001
  discards: N 1p 3p 2p 7p R 2m 1s 5p 2s (S) 3s 4m 3p
* previous (上家) (seat 2): `12466899m NN`, open `WhWhWhWh`; truly 2 from ready; read deal-in on 5s: 0.000
  discards: 1p 4p 5p 8p 4p 2s 9p 2p 5s 7s (8s) 3p 8s 1p

`stage`'s pick 5s, read 0.027 (nobody wins on it); thrown instead S (costs 0 steps of progress).
Plain game: next (下家) won off opposite (對家) with 7 faan (bot +0). Bot game: previous (上家) self-drew 10 faan (bot -384).

#### Example 3: liable for a win, -360 points (deal 24028561, seat 1; 59 tiles left in the wall, wall third 1)

* bot (seat 1): `445s E W NN G RRR`, open `666s`; 2 from ready
  discards: 3m 6p (2m) 4m 4m 9p (8s)
* next (下家) (seat 2): `778p G`, open `222m 888s 777s`; truly 1 from ready; read deal-in on G: 0.035
  discards: 6m 2s 3p N 8m 5m 5s 9s
* opposite (對家) (seat 3): `2445889p 19s E S W Wh`; truly 5 from ready; read deal-in on G: 0.000
  discards: 6m 7m 3m (6s) (8m) 5m (7s) 5s
* previous (上家) (seat 0): `11289m E WW GG`, open `789m`; truly 2 from ready; read deal-in on G: 0.001
  discards: 3s 2s 6p 3p 2p 1p S

`stage`'s pick G, read 0.037 (nobody wins on it); thrown instead E (costs 0 steps of progress).
Plain game: previous (上家) won off bot with 5 faan (bot -24). Bot game: opposite (對家) self-drew 10 faan (bot -384).

#### Example 4: win lost, -480 points (deal 24014513, seat 1; 37 tiles left in the wall, wall third 2)

* bot (seat 1): `12555789m 5p S G`, open `234m`; 1 from ready
  discards: 3p 4p 2s 6m 5s N (R) 4s 4s 4s 4s 4p 1s
* next (下家) (seat 2): `234668889p G`, open `111p`; truly 1 from ready; read deal-in on 5p: 0.035
  discards: 6s 1s 2m 7m 8m 9m 9m 1m E W 6m S Wh
* opposite (對家) (seat 3): `33m 556677p 7889s N`; truly 3 from ready; read deal-in on 5p: 0.003
  discards: 7m 5s 9p 9m E 3p W 4m (1p) 2s 1s 8m (3s)
* previous (上家) (seat 0): `356789s Wh`, open `RRR 123s`; truly 1 from ready; read deal-in on 5p: 0.000
  discards: 5m 2m 7p 7p 4m 1m 2p 9p 2p (4m) 9p 2p E W

`stage`'s pick 5p, read 0.038 (nobody wins on it); thrown instead S (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: next (下家) self-drew 9 faan (bot -96).

#### Example 5: win lost, -480 points (deal 24028981, seat 3; 30 tiles left in the wall, wall third 3)

* bot (seat 3): `677888p EE W WhWh`, open `RRR`; 1 from ready
  discards: 2m 4m 4s 4s 7s 6m 5s 4m 6s 6s 3p (1p) N 4s
* next (下家) (seat 0): `677m 89p 99s SS G`, open `111p`; truly 3 from ready; read deal-in on W: 0.170
  discards: 6p 9m 3m 8s 4m 8s 7s 4s 9m 3p 8m 3s N 1s 1m
* opposite (對家) (seat 1): `1m 1p 19s EE S WW N Wh G R`; truly 1 from ready; read deal-in on W: 0.003
  discards: 7m 4m 8s 6s 5p 2p 2p 8m (4p) 5p 5m (3p) 3m 3s 4p
* previous (上家) (seat 2): `5667799p`, open `234p 345p`; truly 0 from ready; read deal-in on W: 0.096
  discards: 9m 2m 3m 7s 5m 5s 9m 3m 1s 2s N (R) 6m 6m 2s

`stage`'s pick W, read 0.252 (nobody wins on it); thrown instead 8p (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: previous (上家) won off bot with 9 faan (bot -96).

#### Example 6: win lost, -432 points (deal 24000777, seat 1; 43 tiles left in the wall, wall third 2)

* bot (seat 1): `122233444678s N R`; 1 from ready
  discards: 4p 4p 2p 1p 8m 7m 3m 2m 6p 1m 6m
* next (下家) (seat 2): `122456789m S W N G`; truly 3 from ready; read deal-in on N: 0.004
  discards: 6s 1s 8s (9s) (3p) 8p 8p 9p 9p Wh 4p
* opposite (對家) (seat 3): `789p EE S WW N G`, open `123p`; truly 2 from ready; read deal-in on N: 0.017
  discards: 3m 8m 5m 6m 4p 1p 3m 8p 7s Wh
* previous (上家) (seat 0): `135588s E S G R`, open `999s`; truly 3 from ready; read deal-in on N: 0.004
  discards: 3p 7m 4m 4m 2m 4m 1m 9p 5p 6p 7p 9m

`stage`'s pick N, read 0.025 (nobody wins on it); thrown instead R (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: previous (上家) self-drew 7 faan (bot -48).

#### Example 7: win lost, -432 points (deal 24040024, seat 1; 48 tiles left in the wall, wall third 2)

* bot (seat 1): `114s N R`, open `777s WhWhWh GGG`; 1 from ready
  discards: 4m 8p 5p 8m 7p 7m 6p (1p) (S) 3p E 9m
* next (下家) (seat 2): `1233456789m W N G`; truly 2 from ready; read deal-in on N: 0.003
  discards: 8p 4p 3p 2s 5s (7s) 9s 7p 9s E
* opposite (對家) (seat 3): `22455679p RR`, open `SSS`; truly 1 from ready; read deal-in on N: 0.024
  discards: 2s 5s 5m 8m 6m 1m E (Wh) 1p 5m
* previous (上家) (seat 0): `22469p WWW N R`, open `111p`; truly 2 from ready; read deal-in on N: 0.010
  discards: 1m 2s 7m 5m 8s 8s 9s 6s 6s E (G)

`stage`'s pick N, read 0.037 (nobody wins on it); thrown instead R (costs 0 steps of progress).
Plain game: bot won off previous (上家) with 10 faan; opposite (對家) won off previous (上家) with 5 faan (bot +384). Bot game: opposite (對家) self-drew 7 faan (bot -48).

#### Example 8: win lost, -416 points (deal 24009071, seat 2; 53 tiles left in the wall, wall third 2)

* bot (seat 2): `77p 123346799s S Wh R`; 3 from ready
  discards: 3m 9m 9m 7m 4p 2p 1p
* next (下家) (seat 3): `1234459p N G R`, open `EEE`; truly 3 from ready; read deal-in on 7p: 0.003
  discards: 4s 8m 1m 3m 5m 5m 5m S
* opposite (對家) (seat 0): `11m 33667p WWW`, open `222m`; truly 1 from ready; read deal-in on 7p: 0.006
  discards: 8s 3m 9m 7s 5s 5m 3s 2s 1s 6m
* previous (上家) (seat 1): `556788899p Wh GG R`; truly 2 from ready; read deal-in on 7p: 0.018
  discards: 8m 7m 7s 6m 5s (2m) 6s 1s (E) S

`stage`'s pick 7p, read 0.027 (nobody wins on it); thrown instead S (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: previous (上家) self-drew 6 faan (bot -32).

#### Example 9: win lost, -416 points (deal 24021277, seat 1; 50 tiles left in the wall, wall third 2)

* bot (seat 1): `126699s SS W N Wh RRR`; 2 from ready
  discards: 7p 7s 4p 3p 6m 2m (3m) 7m (6p) E
* next (下家) (seat 2): `12457799p 2234s W`; truly 4 from ready; read deal-in on Wh: 0.000
  discards: 5m 7m 2m 1m 8s 6s 9m 8s
* opposite (對家) (seat 3): `2335788p 3s W N`, open `666p`; truly 3 from ready; read deal-in on Wh: 0.000
  discards: 5m 8m (8m) 1m 7s 2s 1s 3s 6m
* previous (上家) (seat 0): `46m WhWh GGG`, open `789m 333m`; truly 0 from ready; read deal-in on Wh: 0.144
  discards: 9s 4s 4p 1p 2p R E 1m 8m E 7s

`stage`'s pick Wh, read 0.144 (nobody wins on it); thrown instead 2s (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: previous (上家) self-drew 6 faan (bot -32).

#### Example 10: win lost, -416 points (deal 24026370, seat 1; 56 tiles left in the wall, wall third 2)

* bot (seat 1): `9m 11333p 133777s G R`; 2 from ready
  discards: 4m 7p (2s) 4s Wh (4p) 4s 4m S
* next (下家) (seat 2): `23567899p EE Wh G R`; truly 3 from ready; read deal-in on G: 0.000
  discards: 1s 1s (7m) 2m 1m 8s 3m
* opposite (對家) (seat 3): `568p 66s E W`, open `777m 444p`; truly 3 from ready; read deal-in on G: 0.012
  discards: 3s 3m 6m 8m 5s 2m 1m 1s
* previous (上家) (seat 0): `799m 789p 99s E G`, open `222s`; truly 4 from ready; read deal-in on G: 0.089
  discards: 3m 8m 6m 2p 5s S N W N 5p

`stage`'s pick G, read 0.100 (nobody wins on it); thrown instead 9m (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: next (下家) self-drew 6 faan (bot -32).

#### Example 11: win lost, -416 points (deal 24030033, seat 0; 55 tiles left in the wall, wall third 2)

* bot (seat 0): `9m 7788p 3445678s Wh G`; 4 from ready
  discards: 4m 4m 1m 4p 1p 5p 9p 5m (6m)
* next (下家) (seat 1): `1m 1p 669s EE SS W Wh G R`; truly 3 from ready; read deal-in on 9m: 0.001
  discards: 7p 3s 3p 3m 5m 5p 4s 8s
* opposite (對家) (seat 2): `3445689m 899p W N R`; truly 4 from ready; read deal-in on 9m: 0.000
  discards: 6p 4p 3p 1s 2s 3p 8p 9s
* previous (上家) (seat 3): `1223378m N Wh R`, open `666m`; truly 2 from ready; read deal-in on 9m: 0.027
  discards: 4p 4s 7s 5s 7s W 3s 6p E

`stage`'s pick 9m, read 0.028 (nobody wins on it); thrown instead Wh (costs 0 steps of progress).
Plain game: bot self-drew 9 faan (bot +288). Bot game: next (下家) won off bot with 10 faan (bot -128).

#### Example 12: win lost, -408 points (deal 24028342, seat 2; 59 tiles left in the wall, wall third 2)

* bot (seat 2): `2m 9s S NN WhWh G`, open `EEE 111s`; 2 from ready
  discards: 6p 3p 8m 5s 5m 5m (1p) 3m
* next (下家) (seat 3): `345679p SS Wh R`, open `123p`; truly 2 from ready; read deal-in on 2m: 0.000
  discards: 2s 6s 7s (4m) (5m) 3s (7m) 9m
* opposite (對家) (seat 0): `11m W N`, open `345m 567m 789m`; truly 1 from ready; read deal-in on 2m: 0.039
  discards: 4s 2s 9s 6p 1p 1p 9p 9p 8p
* previous (上家) (seat 1): `227789m 556788s G`; truly 4 from ready; read deal-in on 2m: 0.000
  discards: 3p 7p 4p 5p 4m 3m 9p (E) (1s)

`stage`'s pick 2m, read 0.039 (nobody wins on it); thrown instead 9s (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: opposite (對家) self-drew 5 faan (bot -24).

#### Example 13: later deal-in, -224 points (deal 24014489, seat 2; 31 tiles left in the wall, wall third 2)

* bot (seat 2): `77m 1477789p 2555s Wh`; 3 from ready
  discards: 4m 4s E 4m 9s S N 2m 1s N 2s (R) 3m G
* next (下家) (seat 3): `666m 11p 33477s`, open `888s`; truly 1 from ready; read deal-in on 2s: 0.086
  discards: 4p 9m 9m 8p 8m 5m 9p 1m 3m S N E (G) 9s
* opposite (對家) (seat 0): `12233346899p WW`; truly 1 from ready; read deal-in on 2s: 0.000
  discards: 3s (8s) 2m 3m 1m 5m 5m 7s 6m 1s S E 9s 8s
* previous (上家) (seat 1): `56678p S W`, open `RRR GGG`; truly 1 from ready; read deal-in on 2s: 0.000
  discards: 8m 4m 9m 1m 6s 8m 7s 4s 4m 3s 2s 6s 1s 9s 6s

`stage`'s pick 2s, read 0.086 (nobody wins on it); thrown instead 4p (costs 0 steps of progress).
Plain game: draw (wall ran out). Bot game: opposite (對家) won off bot with 9 faan; previous (上家) won off bot with 10 faan (bot -224).

#### Example 14: later deal-in, -144 points (deal 24028435, seat 0; 51 tiles left in the wall, wall third 2)

* bot (seat 0): `11366689m 7s N R`, open `222m`; 2 from ready
  discards: 7p 5p 9s 4s 7s 7s 7p W 1s
* next (下家) (seat 1): `1m 1223339p E Wh`, open `999m`; truly 3 from ready; read deal-in on 7s: 0.000
  discards: 8s 5p 3m 7m G 7m 4m 6s W 9s
* opposite (對家) (seat 2): `23488p 22233s SS G`; truly 3 from ready; read deal-in on 7s: 0.000
  discards: 6m 4m (6s) (9m) 9s 8m 6p (2s) 3m 5p
* previous (上家) (seat 3): `157s SS Wh R`, open `456s 123s`; truly 2 from ready; read deal-in on 7s: 0.030
  discards: 1p 6p 8p (2m) 4m 5p 5m W 9p

`stage`'s pick 7s, read 0.031 (nobody wins on it); thrown instead 3m (costs 0 steps of progress).
Plain game: next (下家) won off previous (上家) with 3 faan (bot +0). Bot game: opposite (對家) won off bot with 4 faan; previous (上家) won off bot with 10 faan (bot -144).

#### Example 15: later deal-in, -128 points (deal 24000158, seat 3; 31 tiles left in the wall, wall third 2)

* bot (seat 3): `345666m 6p W Wh G R`, open `222m`; 2 from ready
  discards: 7s 2s 1s 5s 9p 7s (7p) 7s 6p 2s 9s S 2s
* next (下家) (seat 0): `13467p WW`, open `789p GGG`; truly 1 from ready; read deal-in on 6p: 0.005
  discards: 3m 8m 9m 4s 5m 4m 5s 8s 8s 6s 8m 3s 9m 3s 4s
* opposite (對家) (seat 1): `19m 19p 19s EE S W N Wh R`; truly 0 from ready; read deal-in on 6p: 0.000
  discards: 5m (2m) 4m 6s 5m 5p 4s 3s 4s 2p (8p) (5p) 2s 3p 2p
* previous (上家) (seat 2): `567p EE RR`, open `888p 345p`; truly 0 from ready; read deal-in on 6p: 0.191
  discards: 9m 7m 6s 8s 1s 9s 1p 2m 3m S S (G) 1m 8s

`stage`'s pick 6p, read 0.195 (nobody wins on it); thrown instead G (costs 0 steps of progress).
Plain game: previous (上家) won off opposite (對家) with 5 faan (bot +0). Bot game: opposite (對家) won off bot with 10 faan (bot -128).

#### Example 16: other's self-draw, -168 points (deal 24034865, seat 0; 59 tiles left in the wall, wall third 1)

* bot (seat 0): `11339m 18p 19s S W GG R`; 3 from ready
  discards: (2p) (3p) 5p 2s 3s 9s 5m
* next (下家) (seat 1): `57p N Wh`, open `222p 345p 9999p`; truly 1 from ready; read deal-in on 8p: 0.030
  discards: 2s 8s 8s 7s (6s) 6m 1s E
* opposite (對家) (seat 2): `7m 168s WhWh GG RR`, open `456s`; truly 2 from ready; read deal-in on 8p: 0.000
  discards: 4m 7m 8m 5m 6p 1p 2p 1p
* previous (上家) (seat 3): `6799m 2334567s E W`; truly 4 from ready; read deal-in on 8p: 0.000
  discards: 3p 7p (9p) 6p 1m 4p 4m 5m

`stage`'s pick 8p, read 0.030 (nobody wins on it); thrown instead G (costs 0 steps of progress).
Plain game: next (下家) won off opposite (對家) with 3 faan (bot -24). Bot game: next (下家) self-drew 8 faan (bot -192).

#### Example 17: other's self-draw, -128 points (deal 24000349, seat 3; 26 tiles left in the wall, wall third 3)

* bot (seat 3): `688m 1345677p E SS N`; 3 from ready
  discards: 4s 3m 7s N 1m 9s 2s 7s 8s 8s 4m 7m (9m) (8m)
* next (下家) (seat 0): `12456m W G`, open `999m 678m`; truly 1 from ready; read deal-in on 6m: 0.172
  discards: 6s 9s 9s 4s 2s 2p 1p 3p 3p 4s 7p 8p 5p R E
* opposite (對家) (seat 1): `1559m 66999p 1s N Wh R`; truly 4 from ready; read deal-in on 6m: 0.000
  discards: 4s 6s 3s 3s 6s 1s 7s 4p 8p 2s 2m 2p 8m 3m 3p
* previous (上家) (seat 2): `E SS WW WhWh GG R`, open `[5555s]`; truly 2 from ready; read deal-in on 6m: 0.000
  discards: 2m 3m 1s 1m 4p 8p 9p 8p 9s 6m N 8s 1s 3s 4m

`stage`'s pick 6m, read 0.173 (nobody wins on it); thrown instead 8m (costs 0 steps of progress).
Plain game: draw (wall ran out). Bot game: previous (上家) self-drew 10 faan (bot -128).

#### Example 18: other's self-draw, -128 points (deal 24000569, seat 3; 38 tiles left in the wall, wall third 2)

* bot (seat 3): `22568p N GG`, open `SSS 999p`; 1 from ready
  discards: 2m 3m 7s 7m 4s 4s 8m 9s 1s 9s W 6m E
* next (下家) (seat 0): `11m 333p 228889s NN`; truly 1 from ready; read deal-in on N: 0.035
  discards: 4m 7m 8m 6s E 5m (R) Wh W 1p W 6m 9m (9p)
* opposite (對家) (seat 1): `2346779s WhWh G`, open `111s`; truly 1 from ready; read deal-in on N: 0.008
  discards: 4m 9m (3p) (5p) 7m 6p 6p 5m 2m 7p 1p W (S)
* previous (上家) (seat 2): `2459p`, open `123p 456p RRR`; truly 1 from ready; read deal-in on N: 0.016
  discards: 6s (1s) 8m 2m 3m 3s 5m 4m 5m 3s 1p E 5s

`stage`'s pick N, read 0.058 (nobody wins on it); thrown instead 6p (costs 1 step of progress).
Plain game: next (下家) won off previous (上家) with 4 faan (bot +0). Bot game: next (下家) self-drew 10 faan (bot -128).

#### Example 19: other's self-draw, -128 points (deal 24000726, seat 0; 28 tiles left in the wall, wall third 3)

* bot (seat 0): `6p 8s N Wh G`, open `999p 777s 222p`; 2 from ready
  discards: 4m 4s 6m 4m 4p 8p 2s 8m 2m 7m 7p W S 7m 7s
* next (下家) (seat 1): `1123456p E N R`, open `555p`; truly 2 from ready; read deal-in on 8s: 0.000
  discards: 5s 3m 1m 9s 1s 1s 8m 4s 5m 5m 2s 8m (7s) W 2s 8s
* opposite (對家) (seat 2): `119m 19p 19s E S W Wh G R`; truly 0 from ready; read deal-in on 8s: 0.000
  discards: 3p (5p) 3p 8p 2m 8s 5s 3m (6s) 3m S 6p 3m 4p 8m
* previous (上家) (seat 3): `133356669s G`, open `456s`; truly 2 from ready; read deal-in on 8s: 0.045
  discards: 7p 2m W 4p 5m 2m 9m 7m 7m 9m (9p) R (2p) E

`stage`'s pick 8s, read 0.045 (nobody wins on it); thrown instead 6p (costs 0 steps of progress).
Plain game: opposite (對家) won off next (下家) with 10 faan (bot +0). Bot game: opposite (對家) self-drew 10 faan (bot -128).

#### Example 20: other's self-draw, -128 points (deal 24000977, seat 2; 43 tiles left in the wall, wall third 2)

* bot (seat 2): `46788s E SS N Wh R`, open `333s`; 3 from ready
  discards: 6p 9p 4p (8m) 3m 3p 3m 4m 7p W G
* next (下家) (seat 3): `56m N WhWhWh GG RR`, open `888m`; truly 1 from ready; read deal-in on Wh: 0.007
  discards: (8p) (5p) 2p 4p 9s 2s (1s) 7s 6s 5s 5s
* opposite (對家) (seat 0): `12467p 67s E W R`, open `999p`; truly 4 from ready; read deal-in on Wh: 0.000
  discards: 4m 7m 9m 1m 1m 5m 2s 9s 5s 9s
* previous (上家) (seat 1): `22m 4s S`, open `888p 5555p 111s`; truly 1 from ready; read deal-in on Wh: 0.035
  discards: 2p 6p 9s 5s W 3m 4m 2s 6m (9p) 5m (3s) 7m

`stage`'s pick Wh, read 0.041 (nobody wins on it); thrown instead R (costs 0 steps of progress).
Plain game: draw (wall ran out). Bot game: next (下家) self-drew 10 faan (bot -128).

### `ev`: the costliest false alarms across the endings

#### Example 1: liable for a win, -576 points (deal 24023160, seat 2; 48 tiles left in the wall, wall third 2)

* bot (seat 2): `459p N G`, open `123p 789p 123p`; 1 from ready
  discards: 3m 5m 6m (3s) (6s) (5s) 7s 9s 8s S (W)
* next (下家) (seat 3): `1389s`, open `123s 456s 567s`; truly 1 from ready; read deal-in on N: 0.066
  discards: 7m (8m) 5p 4p E S E R 3p 8m
* opposite (對家) (seat 0): `44557m NN`, open `789m WWW`; truly 1 from ready; read deal-in on N: 0.057
  discards: 6s 5p 1p 8p 6p 1m 1m E 6p 6p 8p (Wh)
* previous (上家) (seat 1): `234566m E W N R`, open `WhWhWh`; truly 3 from ready; read deal-in on N: 0.007
  discards: 4s 2s 2s 6s 7p 8p 4p (2p) 4s 8s (8p) (3p)

`stage`'s pick N, read 0.126 (nobody wins on it); thrown instead 4p (costs 1 step of progress).
Plain game: bot self-drew 9 faan (bot +288). Bot game: next (下家) won off bot with 9 faan (bot -288).

#### Example 2: liable for a win, -576 points (deal 24030481, seat 0; 13 tiles left in the wall, wall third 3)

* bot (seat 0): `112233567899s S W`; 1 from ready
  discards: 1m 6m 6m 6p (3p) 2p (6p) 3p 1p N 8p (7p) 6p R 7m 4m 1m 8m
* next (下家) (seat 1): `8p`, open `999p 234p 456p 567p`; truly 0 from ready; read deal-in on S: 0.001
  discards: 3s 8s 6s 9m 1m 2m 4m 5m 9m (7s) 6m E R 5p R 9m 2p Wh N
* opposite (對家) (seat 2): `1245579s`, open `789s GGG`; truly 1 from ready; read deal-in on S: 0.000
  discards: 3p 7p (9p) 7m 8m 7m 8m 4m 3m 1m 4m 3m N 4p W R 9m 5p 6m Wh
* previous (上家) (seat 3): `123445678s EE WW`; truly 1 from ready; read deal-in on S: 0.000
  discards: 3p 1p 8p 2m 2p 1p 7m 7p 3m 3m 9p 5m 5m 7p 5m N S (G) S

`stage`'s pick S, read 0.001 (nobody wins on it); thrown instead 9s (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: next (下家) self-drew 8 faan (bot -192).

#### Example 3: win lost, -512 points (deal 24020434, seat 2; 43 tiles left in the wall, wall third 2)

* bot (seat 2): `11222s E W WhWhWh G`, open `666s`; 1 from ready
  discards: 5m 3m 3m 5p 7p 6m 7p 5p 8p 9p 4s
* next (下家) (seat 3): `7m 14557s W N Wh G`, open `888s`; truly 4 from ready; read deal-in on E: 0.000
  discards: 2m 1m 1m 8p 2p 8m 5m 3p 4p 6p 6p 8p
* opposite (對家) (seat 0): `113445678m 2s E G R`; truly 3 from ready; read deal-in on E: 0.000
  discards: 8p 2p 5p 4p 6p (3s) 4s (8s) 3s (6s) 7p 2p (7s)
* previous (上家) (seat 1): `35s EE NN R`, open `345s 789s`; truly 1 from ready; read deal-in on E: 0.000
  discards: 6m 9m 8m 9m 4m 7p 9p 3p 4p 2p 1p

`stage`'s pick E, read 0.001 (nobody wins on it); thrown instead G (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: opposite (對家) self-drew 10 faan (bot -128).

#### Example 4: win lost, -480 points (deal 24005539, seat 2; 43 tiles left in the wall, wall third 2)

* bot (seat 2): `19m 19p 19s S WW NN GG R`; 1 from ready
  discards: 5p 6p 7m 2p 8s 1p (1s) 8m 2s 4m 3m
* next (下家) (seat 3): `55p 34567789s E N R`; truly 3 from ready; read deal-in on W: 0.000
  discards: 5m 8m 4m 4m 1m 7m 4p 8p 9p 8m
* opposite (對家) (seat 0): `2345s EE WhWh GG`, open `111s`; truly 1 from ready; read deal-in on W: 0.002
  discards: 6m 9m 1m 1p 8p 6p 7p 7p 2m 8s (5m) N
* previous (上家) (seat 1): `12223567m Wh R`, open `345m`; truly 1 from ready; read deal-in on W: 0.001
  discards: 3s 5s 6s 8p 2p 5p 3p 4p 4p 6p S 8s

`stage`'s pick W, read 0.003 (nobody wins on it); thrown instead 9s (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: next (下家) self-drew 9 faan (bot -96).

#### Example 5: win lost, -448 points (deal 24023464, seat 1; 43 tiles left in the wall, wall third 2)

* bot (seat 1): `1m 1239p 9s EE S W N Wh G R`; 1 from ready
  discards: 2s 5s 6s 4p 8m 5m 6m 5p 1m (7s) (7p)
* next (下家) (seat 2): `777m 1156p SS G`, open `777s`; truly 2 from ready; read deal-in on 3p: 0.002
  discards: 5s 7m 6s Wh 4s 2m 8m W 2s 4p
* opposite (對家) (seat 3): `224557p N`, open `888p 777p`; truly 1 from ready; read deal-in on 3p: 0.006
  discards: 5m 9m 4s 3s 8m 2s 1s 8s 9s 9s W E
* previous (上家) (seat 0): `1234445m S N WhWh RR`; truly 2 from ready; read deal-in on 3p: 0.000
  discards: 3p 9m 8p 9m 3s 4s 6p (8p) 6s 6s 8s 7s 6p

`stage`'s pick 3p, read 0.007 (nobody wins on it); thrown instead 9s (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: opposite (對家) self-drew 8 faan (bot -64).

#### Example 6: win lost, -448 points (deal 24023666, seat 0; 28 tiles left in the wall, wall third 3)

* bot (seat 0): `347778899p W N`, open `123p`; 1 from ready
  discards: 4s 1s (7s) 8s 1m 1s 3s 6m 7m 8m 9m 8s Wh 3m 3m
* next (下家) (seat 1): `2234468p GG R`, open `111p`; truly 1 from ready; read deal-in on W: 0.000
  discards: 9s (4s) 6s 7m 4m 5s 4m 2m 1m Wh E 7m 2m 9s Wh
* opposite (對家) (seat 2): `SS GG`, open `345s 7777s WWW`; truly 0 from ready; read deal-in on W: 0.000
  discards: 8p 8m 9m 5s 5m 6s 9s 7m 2p (1p) 9p 6s 2m 9p 4m (1s) (R)
* previous (上家) (seat 3): `1236s SS N`, open `123s RRR`; truly 1 from ready; read deal-in on W: 0.000
  discards: 4p 6p 8m 6m 5m 1m 3m 3m 6p 6m Wh E 5p (1p) (W) 8s

`stage`'s pick W, read 0.000 (nobody wins on it); thrown instead 9p (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: previous (上家) self-drew 8 faan (bot -64).

#### Example 7: win lost, -448 points (deal 24028676, seat 1; 50 tiles left in the wall, wall third 2)

* bot (seat 1): `566899s S N`, open `333s 111s`; 1 from ready
  discards: 7m 1m 2m 6p 9p 1p 8p 1p (1m) Wh
* next (下家) (seat 2): `22567m S W N Wh R`, open `111m`; truly 3 from ready; read deal-in on S: 0.001
  discards: 3s 4s 3p 8p 6p 3p 5p 5p 4s 8p
* opposite (對家) (seat 3): `55666m 77p 679s SS N`; truly 3 from ready; read deal-in on S: 0.000
  discards: 2s 3m 5s 4s 6p 8m 3m 5p (3s) 2p
* previous (上家) (seat 0): `12379p EE W N Wh G RR`; truly 3 from ready; read deal-in on S: 0.000
  discards: 2s 7s 5s 8m 5p 3m 4m 6p 8s (1s)

`stage`'s pick S, read 0.001 (nobody wins on it); thrown instead N (costs 0 steps of progress).
Plain game: bot won off previous (上家) with 10 faan (bot +384). Bot game: next (下家) won off bot with 8 faan (bot -64).

#### Example 8: win lost, -432 points (deal 24007592, seat 0; 39 tiles left in the wall, wall third 2)

* bot (seat 0): `13456789s W G R`, open `123s`; 1 from ready
  discards: 1m 8p 2p 3m 5p 4p (4m) 1m E 2m N Wh
* next (下家) (seat 1): `45677899m W R`, open `345m`; truly 1 from ready; read deal-in on W: 0.008
  discards: 5p 3p 2p 9p 5s E S 6p 1p 2s N 9s
* opposite (對家) (seat 2): `1234556m S W N Wh RR`; truly 3 from ready; read deal-in on W: 0.003
  discards: 7p 9p 6s 2s 4s 5s 1p 8p 1s 4s (6p) E
* previous (上家) (seat 3): `166777p 68s Wh G`, open `456p`; truly 3 from ready; read deal-in on W: 0.004
  discards: 7m 2s 6m 3m 4m 2m 1m (1s) 9s 8m 6s 8m

`stage`'s pick W, read 0.016 (nobody wins on it); thrown instead 3s (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: next (下家) won off bot with 7 faan (bot -48).

#### Example 9: win lost, -432 points (deal 24017186, seat 1; 55 tiles left in the wall, wall third 2)

* bot (seat 1): `23334789p S W N`, open `567p`; 1 from ready
  discards: 6s 1s 7m 4m 1m 7s 4s (G)
* next (下家) (seat 2): `2347889m 3s E SS N G`; truly 3 from ready; read deal-in on W: 0.000
  discards: 2p 6p 9p 1p (6s) 5s 2s
* opposite (對家) (seat 3): `3m 89s EE Wh R`, open `567s GGG`; truly 2 from ready; read deal-in on W: 0.007
  discards: 2s 1p 1p 8m 9m 3m 7p 3m
* previous (上家) (seat 0): `249m 134678s E S W R`; truly 5 from ready; read deal-in on W: 0.000
  discards: 7p 4p 4p 7m 6p 9p (7p) 6p 8p

`stage`'s pick W, read 0.007 (nobody wins on it); thrown instead S (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: opposite (對家) self-drew 7 faan (bot -48).

#### Example 10: win lost, -432 points (deal 24021146, seat 1; 35 tiles left in the wall, wall third 2)

* bot (seat 1): `35p S W WhWhWh GG RR`, open `111p`; 1 from ready
  discards: 8m 7m 8s 4s 5s 2s 9s 9m 9s 4m 9p 2s 8p
* next (下家) (seat 2): `256789p EEE WW NN`; truly 1 from ready; read deal-in on S: 0.003
  discards: 2m 7m 1m 9m 2s 9m 8s 5s 6s 3s 6s 7s (1p)
* opposite (對家) (seat 3): `134445567m 11s W R`; truly 3 from ready; read deal-in on S: 0.000
  discards: 6p 1p 9p 8s 6s 3s 9s 4s 4s 7s 7s (4p)
* previous (上家) (seat 0): `45679p 1s SS Wh R`, open `234p`; truly 2 from ready; read deal-in on S: 0.000
  discards: 8m 2m 3m 5m 8s 5s 5s 4s 6m 3s 1m 8m 2s

`stage`'s pick S, read 0.004 (nobody wins on it); thrown instead W (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: next (下家) self-drew 7 faan (bot -48).

#### Example 11: win lost, -432 points (deal 24033498, seat 0; 42 tiles left in the wall, wall third 2)

* bot (seat 0): `124579m EEE WhWhWh GG`; 1 from ready
  discards: 9s 1s 6s 1s 6p 6s 4s 5p 3p 1p R
* next (下家) (seat 1): `19m 1223348s E S W R`; truly 4 from ready; read deal-in on 1m: 0.000
  discards: 4p 2p 7p 8p 2m 4m 5m (2p) 8m 2m 8m
* opposite (對家) (seat 2): `44589p S WW N G`, open `123p`; truly 3 from ready; read deal-in on 1m: 0.000
  discards: 3s 4s 4m 7m 9m 9m 5s Wh 3m 5m 2s
* previous (上家) (seat 3): `1235677889s NN G`; truly 1 from ready; read deal-in on 1m: 0.000
  discards: 6m 6m 6m 2p 8p 6p 6p 1m 6p 5p 5p

`stage`'s pick 1m, read 0.000 (nobody wins on it); thrown instead 7m (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: previous (上家) self-drew 7 faan (bot -48).

#### Example 12: win lost, -416 points (deal 24006868, seat 0; 28 tiles left in the wall, wall third 3)

* bot (seat 0): `244566788999p GG`; 0 from ready
  discards: 7s 3s (9m) 8s 6m 4m 2m 5s 2m W S (4m) 8m Wh E
* next (下家) (seat 1): `2466788m NN G`, open `234m`; truly 1 from ready; read deal-in on G: 0.017
  discards: 3s 6s 8s 4s 2s 5p 9s 4p 2p 1p W S 1m Wh
* opposite (對家) (seat 2): `3m 337779p`, open `555m 999m`; truly 1 from ready; read deal-in on G: 0.023
  discards: 8s 6s 1m 7m E 7s W 4s 7m 4p (2p) 1p S 4m 9m Wh
* previous (上家) (seat 3): `12688p 9s E S NN`, open `123p`; truly 3 from ready; read deal-in on G: 0.001
  discards: (5m) 1m 1m 8s 1s 6s 5s 5m 5s 1s 4s 4s 1s 7m 7s 5s

`stage`'s pick G, read 0.040 (nobody wins on it); thrown instead 8p (costs -1 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: next (下家) self-drew 6 faan (bot -32).

#### Example 13: win lost, -416 points (deal 24009144, seat 0; 31 tiles left in the wall, wall third 2)

* bot (seat 0): `3556999p Wh`, open `111p 345p`; 1 from ready
  discards: 4m (9m) 9m 1s 3s 5s 5s 2s E 5m 7m G 9s 4s N
* next (下家) (seat 1): `5577s SS N WhWh R`, open `888s`; truly 2 from ready; read deal-in on Wh: 0.030
  discards: 7m 3m 1m 1m (2m) 4p 3p E G 3m 1s 5p G 2p 4m
* opposite (對家) (seat 2): `1246677899s SS W`; truly 2 from ready; read deal-in on Wh: 0.000
  discards: 2p 8p 4p 6p 5m 7p 8p 6p 6m (8m) 8m 1p 1m N
* previous (上家) (seat 3): `4566m`, open `999m 222m 888m`; truly 0 from ready; read deal-in on Wh: 0.015
  discards: 8p 3s (1p) (5p) 4s (8s) 3s E 3m 6s 6s N 1s R 1m 7m

`stage`'s pick Wh, read 0.044 (nobody wins on it); thrown instead 5p (costs 0 steps of progress).
Plain game: bot self-drew 9 faan (bot +288). Bot game: previous (上家) self-drew 10 faan (bot -128).

#### Example 14: win lost, -416 points (deal 24012484, seat 3; 72 tiles left in the wall, wall third 1)

* bot (seat 3): `19m 1239p 11s S W N Wh G R`; 1 from ready
  discards: 7s 7m (3s) 2m
* next (下家) (seat 0): `1445679m 57s N Wh G R`; truly 5 from ready; read deal-in on 3p: 0.000
  discards: 3p 8p 8p 2s
* opposite (對家) (seat 1): `24789m 239s E SS Wh G`; truly 4 from ready; read deal-in on 3p: 0.000
  discards: 5p 5s 4s 5p
* previous (上家) (seat 2): `34m 2299p 6s S N R`, open `333s`; truly 4 from ready; read deal-in on 3p: 0.000
  discards: 7s 6m 4s 9m 6m

`stage`'s pick 3p, read 0.000 (nobody wins on it); thrown instead R (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: opposite (對家) self-drew 6 faan (bot -32).

#### Example 15: later deal-in, -160 points (deal 24035882, seat 3; 34 tiles left in the wall, wall third 2)

* bot (seat 3): `123678s W R`, open `567s SSS`; 0 from ready
  discards: 5m 4m 1m 9m 7p 9p (4m) 4p 4p 6p 6p 9p 3m
* next (下家) (seat 0): `66889m WW`, open `345m EEE`; truly 1 from ready; read deal-in on W: 0.020
  discards: 3s 8s 3s 9s 7p 5p 2s 3p 4s 3p 8s 1m 4p 1s (S)
* opposite (對家) (seat 1): `779m WhWh GGG RR`, open `222m`; truly 1 from ready; read deal-in on W: 0.060
  discards: 1p 2p 1p 3s 7p 9s 5p 4s 5s 6s (E) 4m 4m 9s 2p
* previous (上家) (seat 2): `56789m 789p 11789s`; truly 5 from ready; read deal-in on W: 0.014
  discards: 1p 5s S 5s (2m) 1m 5p 2s N N 2p 6s E (7s)

`stage`'s pick W, read 0.092 (nobody wins on it); thrown instead 8s (costs 1 step of progress).
Plain game: next (下家) won off opposite (對家) with 6 faan (bot +0). Bot game: next (下家) won off bot with 6 faan; opposite (對家) won off bot with 10 faan (bot -160).

#### Example 16: later deal-in, -152 points (deal 24011186, seat 1; 20 tiles left in the wall, wall third 3)

* bot (seat 1): `4m 6p 333s WW G`, open `555p 555m`; 1 from ready
  discards: 8p 1p 8m 3m 4p Wh R 9m 7s 8p 7p 1m 7s 1p R (6s) (5s)
* next (下家) (seat 2): `3s W NN`, open `888s 456s 567s`; truly 1 from ready; read deal-in on 4m: 0.000
  discards: 8m 2m 1m 9p 7p 7m 6m 2p 9p 6m 7m 3p 3p 6m 8m R E 1s
* opposite (對家) (seat 3): `19m 19p 19s E S W WhWhWh R`; truly 1 from ready; read deal-in on 4m: 0.000
  discards: 4m 7m 6p 2s 2s 5s 6s 1p 9p S 2m 8s 7s E 1s 2s (5m) (4m)
* previous (上家) (seat 0): `345789m SS GG`, open `234m`; truly 0 from ready; read deal-in on 4m: 0.001
  discards: 2s (8s) 2p 8p 2p 7p 2p 4p (5p) 4s 5p E 3p 9s 8p N 4s 2m

`stage`'s pick 4m, read 0.002 (nobody wins on it); thrown instead 6p (costs 0 steps of progress).
Plain game: draw (wall ran out). Bot game: opposite (對家) won off bot with 10 faan; previous (上家) won off bot with 5 faan (bot -152).

#### Example 17: later deal-in, -152 points (deal 24012213, seat 2; 39 tiles left in the wall, wall third 2)

* bot (seat 2): `1477899p N`, open `234p WWW`; 1 from ready
  discards: 6m 4m 8s 6s 5s (3m) 2s 7s 1s 2s (5m) G
* next (下家) (seat 3): `8m EEE`, open `123m 345m WhWhWh`; truly 0 from ready; read deal-in on 4p: 0.000
  discards: 3s 3p 3s 8p 5s 6p 6p 7s 5p 7s G (S) 3s
* opposite (對家) (seat 0): `1135899m RRR`, open `SSS`; truly 1 from ready; read deal-in on 4p: 0.000
  discards: 5p 8p 6s 4s 2p 3s 9s 1s 8s G 5p 9s Wh 7p
* previous (上家) (seat 1): `12566777889m NN`; truly 0 from ready; read deal-in on 4p: 0.000
  discards: 7s 6p 9p 7p 4p 3p (2p) 2p E G (W) (Wh) 6s 5m

`stage`'s pick 4p, read 0.000 (nobody wins on it); thrown instead 7p (costs 1 step of progress).
Plain game: draw (wall ran out). Bot game: opposite (對家) won off bot with 5 faan; previous (上家) won off bot with 10 faan (bot -152).

#### Example 18: other's self-draw, -336 points (deal 24036876, seat 3; 49 tiles left in the wall, wall third 2)

* bot (seat 3): `2889p 66s RR`, open `666m 333p`; 1 from ready
  discards: 2m 3s 8s 5m 9m E 6m (G) 1m 5m
* next (下家) (seat 0): `122p E`, open `888m GGG 777m`; truly 1 from ready; read deal-in on 2p: 0.026
  discards: 2s 7p 9s 4m W 9s 8s (7p) 9p 1m (S) 5m (4p)
* opposite (對家) (seat 1): `4689p N WhWh`, open `567p 345p`; truly 1 from ready; read deal-in on 2p: 0.168
  discards: 2m (8m) 7s (3s) 2s 3m (7s) 9s 1s W (7m) E
* previous (上家) (seat 2): `155s Wh`, open `345s 777s SSS`; truly 1 from ready; read deal-in on 2p: 0.000
  discards: 8p 7p 5p 7p 9m 4p (6m) W G N (3p)

`stage`'s pick 2p, read 0.190 (nobody wins on it); thrown instead 9p (costs 0 steps of progress).
Plain game: next (下家) won off previous (上家) with 4 faan (bot -48). Bot game: next (下家) self-drew 10 faan (bot -384).

#### Example 19: other's self-draw, -144 points (deal 24016128, seat 3; 57 tiles left in the wall, wall third 2)

* bot (seat 3): `6m 2p Wh RR`, open `111m 888p 777m`; 1 from ready
  discards: W 1s 6s S 5s 7p 3p N
* next (下家) (seat 0): `1p 2444458s E Wh`, open `999s`; truly 3 from ready; read deal-in on 6m: 0.000
  discards: 5m 8m 2m 3m (7p) (6p) 9m 5p 4p (1p)
* opposite (對家) (seat 1): `229p G`, open `789p 456p 111p`; truly 1 from ready; read deal-in on 6m: 0.000
  discards: (1m) 6s 3m 3s (1s) S 2m (9s) N E
* previous (上家) (seat 2): `35677889s W G`, open `123s`; truly 1 from ready; read deal-in on 6m: 0.000
  discards: 4m 7p 3p 5p (8p) 5p 4p (7m)

`stage`'s pick 6m, read 0.000 (nobody wins on it); thrown instead Wh (costs 0 steps of progress).
Plain game: opposite (對家) won off bot with 7 faan (bot -144). Bot game: opposite (對家) self-drew 9 faan (bot -288).

#### Example 20: other's self-draw, -128 points (deal 24000569, seat 3; 38 tiles left in the wall, wall third 2)

* bot (seat 3): `22568p N GG`, open `SSS 999p`; 1 from ready
  discards: 2m 3m 7s 7m 4s 4s 8m 9s 1s 9s W 6m E
* next (下家) (seat 0): `11m 333p 228889s NN`; truly 1 from ready; read deal-in on N: 0.035
  discards: 4m 7m 8m 6s E 5m (R) Wh W 1p W 6m 9m (9p)
* opposite (對家) (seat 1): `2346779s WhWh G`, open `111s`; truly 1 from ready; read deal-in on N: 0.008
  discards: 4m 9m (3p) (5p) 7m 6p 6p 5m 2m 7p 1p W (S)
* previous (上家) (seat 2): `2459p`, open `123p 456p RRR`; truly 1 from ready; read deal-in on N: 0.016
  discards: 6s (1s) 8m 2m 3m 3s 5m 4m 5m 3s 1p E 5s

`stage`'s pick N, read 0.058 (nobody wins on it); thrown instead 6p (costs 1 step of progress).
Plain game: next (下家) won off previous (上家) with 4 faan (bot +0). Bot game: next (下家) self-drew 10 faan (bot -128).

### `fold`: the costliest false alarms across the endings

#### Example 1: liable for a win, -384 points (deal 24021391, seat 2; 35 tiles left in the wall, wall third 2)

* bot (seat 2): `6789p R`, open `1111p GGG WWW`; 0 from ready
  discards: 3m 9m 6m 4s 4s 9m (2s) 1s (3s) (3s) E 4p (9s) 7p E
* next (下家) (seat 3): `5s`, open `123s 123s 345s 789s`; truly 0 from ready; read deal-in on R: 0.492
  discards: 4m 3m 7m 7m 5p 4p 2p (1p) Wh (W) 2m 5m N 9m 4s
* opposite (對家) (seat 0): `25566789p 88s E S Wh`; truly 4 from ready; read deal-in on R: 0.005
  discards: 1m 4m 4m 6m 5m 4m 5s 1m 2m (9s) 2m 3s 6m 2s
* previous (上家) (seat 1): `888m 333589p S`, open `999s`; truly 2 from ready; read deal-in on R: 0.547
  discards: 6s 2p 9m 5m 7s 6s 7s 8s (G) E 7s N 6s 6s

`stage`'s pick R, read 0.771 (nobody wins on it); thrown instead 7p (costs 1 step of progress).
Plain game: bot self-drew 8 faan (bot +192). Bot game: next (下家) won off previous (上家) with 8 faan (bot -192).

#### Example 2: liable for a win, -264 points (deal 24010891, seat 1; 26 tiles left in the wall, wall third 3)

* bot (seat 1): `67899p WhWh R`, open `456p 567p`; 0 from ready
  discards: 6m 2s 7s 6s 6s N 9s (1s) 9s 6m 1s 4m 1m 7m 2m 1p
* next (下家) (seat 2): `1223337789m E W R`; truly 3 from ready; read deal-in on R: 0.007
  discards: 8p (5s) (8s) 6m (7s) 2p 3p 5s 4m 1p 2s 9p N 2s S
* opposite (對家) (seat 3): `4s`, open `345s 678s 789s 111s`; truly 0 from ready; read deal-in on R: 0.337
  discards: 5m 4p 4p 8p 7p N 5p E 2p N 9m G 6s 6m S 2p
* previous (上家) (seat 0): `23358s E S W Wh GGG R`; truly 4 from ready; read deal-in on R: 0.000
  discards: 5p 1p 3m (5p) (6p) 7m 9m 2p 5m 5m 4m 9m 8m 3p 8m 1m 6p

`stage`'s pick R, read 0.342 (nobody wins on it); thrown instead 6p (costs 1 step of progress).
Plain game: bot self-drew 5 faan (bot +72). Bot game: opposite (對家) won off bot with 8 faan (bot -192).

#### Example 3: win lost, -416 points (deal 24004615, seat 2; 46 tiles left in the wall, wall third 2)

* bot (seat 2): `4m 44p 33s`, open `888s 555p 333p`; 0 from ready
  discards: 9m (2m) 7m 9p 9s 9p 5m 1m S 6s (R) 8p (5s)
* next (下家) (seat 3): `12467779s N Wh`, open `345s`; truly 2 from ready; read deal-in on 4m: 0.000
  discards: 3m (8m) 8m 8p 7p 2m 4m (5p) (3p) 3p S
* opposite (對家) (seat 0): `34m WW`, open `222m 678m RRR`; truly 0 from ready; read deal-in on 4m: 0.759
  discards: 7s 9s 7p 2p 9p 2s 8p 6m 2p G E 1s
* previous (上家) (seat 1): `12266789p EE W N Wh`; truly 3 from ready; read deal-in on 4m: 0.000
  discards: 5m 7m 8m 9m (8s) 1m 2s 4s 5s 5s 6m G

`stage`'s pick 4m, read 0.759 (nobody wins on it); thrown instead 4p (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: opposite (對家) self-drew 6 faan (bot -32).

#### Example 4: win lost, -416 points (deal 24010219, seat 2; 19 tiles left in the wall, wall third 3)

* bot (seat 2): `222m 22244p 66s W`, open `[5555m]`; 0 from ready
  discards: 8s 4s 8p 1p 1s 1s 6p 1s 9m G (N) 6m 1m S 7s N 4m
* next (下家) (seat 3): `3m 156789p E G R`, open `NNN`; truly 3 from ready; read deal-in on W: 0.000
  discards: 5s (9s) 1s 6m 7m 8s 5s 7s 6m 8s 8m (3s) 4s 3s 8m 8m 2s
* opposite (對家) (seat 0): `123345679p WW WhWh`; truly 1 from ready; read deal-in on W: 0.023
  discards: 3m 4s 2s 9m 5s G S R 9s 9p 7m 1m (5s) 2s 6m 9m
* previous (上家) (seat 1): `EE Wh R`, open `999s 333s 567s`; truly 1 from ready; read deal-in on W: 0.304
  discards: 4p 7p 9p 1p 3m 1m 5p 9m 4m 7m 8m G R 6p S 5p 4s 2s

`stage`'s pick W, read 0.321 (nobody wins on it); thrown instead 2m (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: opposite (對家) self-drew 6 faan (bot -32).

#### Example 5: win lost, -416 points (deal 24030033, seat 0; 12 tiles left in the wall, wall third 3)

* bot (seat 0): `2245567789s Wh`, open `234s`; 0 from ready
  discards: 4m 4m 1m 4p 1p 5p 9p 5m (6m) 9m 8m 8p 8p 7p 7p 7m G 1s E
* next (下家) (seat 1): `19m 1p 19s E S W N WhWh G R`; truly 0 from ready; read deal-in on Wh: 0.024
  discards: 7p 3s 3p 3m 5m 5p 4s 8s 2p E 7m R 6s 7p 6p 6s 5p S
* opposite (對家) (seat 2): `344556789m R`, open `NNN`; truly 0 from ready; read deal-in on Wh: 0.000
  discards: 6p 4p 3p 1s 2s 3p 8p 9s 8s 2p 4p 8p 1s 3s 9p 9p 2p W 5s
* previous (上家) (seat 3): `122337889m Wh`, open `666m`; truly 1 from ready; read deal-in on Wh: 0.042
  discards: 4p 4s 7s 5s 7s W 3s 6p E 2p G 1p W 8s R 6p 1p (N) 9s (2s)

`stage`'s pick Wh, read 0.065 (nobody wins on it); thrown instead 2s (costs 1 step of progress).
Plain game: bot self-drew 9 faan (bot +288). Bot game: next (下家) self-drew 10 faan (bot -128).

#### Example 6: win lost, -400 points (deal 24022286, seat 2; 38 tiles left in the wall, wall third 2)

* bot (seat 2): `6s R`, open `WhWhWh NNN 555s GGG`; 0 from ready
  discards: 7p 8p 5p 1p 7m 2m 7p 9m 6p 1m 1m 6m 7s 4s
* next (下家) (seat 3): `1m 155999p 19s E S W N`; truly 3 from ready; read deal-in on 6s: 0.000
  discards: 6m 2m 4s (3m) 7s 8s (8p) 1s 8m 4s 7s (5s) 5m S
* opposite (對家) (seat 0): `333p 2266699s`, open `333m`; truly 0 from ready; read deal-in on 6s: 0.033
  discards: 7m W 6p 4m 1m 8s 5m 4m 7s 6p S (G) 6m
* previous (上家) (seat 1): `122344569p R`, open `888p`; truly 1 from ready; read deal-in on 6s: 0.000
  discards: 1s 8s 8s 5m 3m 8m 9m (Wh) 9m W (N) 4s 3s

`stage`'s pick 6s, read 0.033 (nobody wins on it); thrown instead R (costs 0 steps of progress).
Plain game: bot won off previous (上家) with 10 faan (bot +384). Bot game: opposite (對家) won off bot with 4 faan (bot -16).

#### Example 7: win lost, -384 points (deal 24007677, seat 3; 40 tiles left in the wall, wall third 2)

* bot (seat 3): `13357p`, open `222p 999p 888p`; 0 from ready
  discards: 2m 8m 9s 8m 9m 1m (5s) 4m Wh E G Wh
* next (下家) (seat 0): `16667889s N R`, open `345s`; truly 2 from ready; read deal-in on 1p: 0.000
  discards: 7m 3m 1m 1p 7m 7p 6p 5p 2m 4p 5p 4p E
* opposite (對家) (seat 1): `12889s E SSS W N G R`; truly 4 from ready; read deal-in on 1p: 0.000
  discards: 8m 9m 5s 7p 6p 2p 5m 7m (2p) 3m (9p) 2m 5s
* previous (上家) (seat 2): `9m 19p 234s E S WW N Wh R`; truly 3 from ready; read deal-in on 1p: 0.000
  discards: 3m 4p 3p 6p 7s 6m 6m 3s 4m 1p (8p)

`stage`'s pick 1p, read 0.000 (nobody wins on it); thrown instead 5p (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: draw (wall ran out).

#### Example 8: win lost, -384 points (deal 24012344, seat 1; 9 tiles left in the wall, wall third 3)

* bot (seat 1): `679p WW`, open `333p RRR NNN`; 0 from ready
  discards: 6m 2m 9m 7s 5s 4s 1s (6s) 6s 5s E 1m 1p 8s 5m (S) 2s 3m 9m 2p (Wh)
* next (下家) (seat 2): `11245s GG`, open `678s WhWhWh`; truly 1 from ready; read deal-in on 9p: 0.000
  discards: 6m 2m 9m 8m 3p 6p 4m 7p 5m 8p 9p 2p 9m 8s (R) 6p 9p 5p 1m (N)
* opposite (對家) (seat 3): `12345677m Wh G`, open `SSS`; truly 1 from ready; read deal-in on 9p: 0.000
  discards: 5p 8p 7s 4p 9s 8p 1p 4s 5p 9s 4p 6s 6s 3s 3s 3s 8p 7p E
* previous (上家) (seat 0): `1m 1129p 19s E S W N G R`; truly 1 from ready; read deal-in on 9p: 0.008
  discards: 4m 7m 7p 4p 2s 8s 2s 4m 9s E 8m 3m (3p) 7m 4s 2p 7s 5m 3m 3s

`stage`'s pick 9p, read 0.008 (nobody wins on it); thrown instead 7p (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: draw (wall ran out).

#### Example 9: win lost, -384 points (deal 24017644, seat 2; 41 tiles left in the wall, wall third 2)

* bot (seat 2): `89p`, open `234p 123p 789p 567p`; 0 from ready
  discards: 4m 9m 3s 8s 6s 6s 2s 5s E 5m W 1p
* next (下家) (seat 3): `4m 124567889s E N R`; truly 3 from ready; read deal-in on 8p: 0.000
  discards: 5p 4p (7m) 3p 8m 7m 1p 6p 1m (3m) (3m) 4m
* opposite (對家) (seat 0): `78m RR`, open `678m 123m 234m`; truly 0 from ready; read deal-in on 8p: 0.002
  discards: 5p 7s 2p 2p 4s 9s 4s 4p 5s 3p Wh G 2m
* previous (上家) (seat 1): `9m 19p 9s E S W N WhWhWh G R`; truly 1 from ready; read deal-in on 8p: 0.002
  discards: 2s 7s 3m 5m (4p) (3p) 6m 7m (9p) G N 2m (6p)

`stage`'s pick 8p, read 0.004 (nobody wins on it); thrown instead 9p (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: opposite (對家) won off next (下家) with 5 faan (bot +0).

#### Example 10: win lost, -384 points (deal 24018593, seat 2; 61 tiles left in the wall, wall third 1)

* bot (seat 2): `2455s G`, open `333s 999s 888s`; 0 from ready
  discards: 4m (9p) 5p 6p 6p N Wh 8p 1m
* next (下家) (seat 3): `123458p 9s E S Wh G RR`; truly 4 from ready; read deal-in on G: 0.000
  discards: (3s) 2s 5m 3m 3m 7m 6s (8s)
* opposite (對家) (seat 0): `67m 12344567p S WW`; truly 2 from ready; read deal-in on G: 0.000
  discards: 4s 7s 6s 3m 2m 7m (8m)
* previous (上家) (seat 1): `4667m 77p G`, open `999p 888m`; truly 2 from ready; read deal-in on G: 0.002
  discards: 3s (9s) 1s 5s 3m 6p W 3p

`stage`'s pick G, read 0.002 (nobody wins on it); thrown instead 5s (costs 1 step of progress).
Plain game: bot won off next (下家) with 10 faan (bot +384). Bot game: draw (wall ran out).

#### Example 11: win lost, -384 points (deal 24020914, seat 0; 44 tiles left in the wall, wall third 2)

* bot (seat 0): `12379p`, open `123p 456p 789p`; 0 from ready
  discards: 4m 6m 8m 1s 3s 3s 5s 2s 7s 7s W S
* next (下家) (seat 1): `5677m 44455p 1s GG R`; truly 3 from ready; read deal-in on 7p: 0.000
  discards: 3s 6s 6s 1p N 4s E 9s 3m 1m 7s 7s
* opposite (對家) (seat 2): `3p 12444556s E S W R`; truly 4 from ready; read deal-in on 7p: 0.000
  discards: 4m 4m (1m) 9m 3m 6p 9m 6m 5m 2p 9p 3p
* previous (上家) (seat 3): `2225799m Wh G R`, open `111m`; truly 2 from ready; read deal-in on 7p: 0.000
  discards: 6s 3s (3p) 8p 1s 9s 6p 7p (6p) (7p) N W

`stage`'s pick 7p, read 0.000 (nobody wins on it); thrown instead 9p (costs 0 steps of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: draw (wall ran out).

#### Example 12: win lost, -384 points (deal 24037113, seat 1; 27 tiles left in the wall, wall third 3)

* bot (seat 1): `122255666m 66677p`; 0 from ready
  discards: 7s 3s 5s 3p 4p 1p 4p (3s) 9p (6s) N (9s) 9m 8m 2p 8s 7m
* next (下家) (seat 2): `6s`, open `999s 123s 456s 789s`; truly 0 from ready; read deal-in on 1m: 0.001
  discards: 8m 5m 7p 7p 4m 1m R Wh (E) S (N) (W) (G) Wh 9p 7m 1s 8p
* opposite (對家) (seat 3): `23345589p E S`, open `WWW`; truly 2 from ready; read deal-in on 1m: 0.000
  discards: 7s 4s 1s 4m (7m) 1m (9s) 4s 2s 9m 8m G 3m 8s 3s
* previous (上家) (seat 0): `6m`, open `789m EEE NNN GGG`; truly 0 from ready; read deal-in on 1m: 0.000
  discards: 2p 6p 1p 5p 3s 6s 8s R 3m 7s S W 9m 7m Wh 5s 4s 1s

`stage`'s pick 1m, read 0.001 (nobody wins on it); thrown instead 6p (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: draw (wall ran out).

#### Example 13: win lost, -384 points (deal 24037396, seat 2; 27 tiles left in the wall, wall third 3)

* bot (seat 2): `7899m S`, open `456m 678m 456m`; 0 from ready
  discards: 8s 6s 5s 4p 5p 6p 7p E (3s) G 1m G 2s 3p R
* next (下家) (seat 3): `5677889s`, open `123s 444s`; truly 0 from ready; read deal-in on S: 0.041
  discards: 6p G 9p W E 3p 2p (1p) 3m 1m 2m 4p W Wh 2s 7m
* opposite (對家) (seat 0): `9m 179p 19s E SS N Wh G R`; truly 1 from ready; read deal-in on S: 0.269
  discards: 4s 3m 5m 4m 6m 2m 6p Wh 2s R 7p (4s) 6p 6s (4p) 7p
* previous (上家) (seat 1): `255888p N`, open `111p 234p`; truly 1 from ready; read deal-in on S: 0.001
  discards: 1s 8s 7s 6s 7s 3s (5m) (7m) 8m 1m 1m 5m 8m 4m R (6m)

`stage`'s pick S, read 0.300 (nobody wins on it); thrown instead 8m (costs 1 step of progress).
Plain game: bot self-drew 10 faan (bot +384). Bot game: next (下家) won off previous (上家) with 8 faan (bot +0).

#### Example 14: win lost, -352 points (deal 24014248, seat 0; 35 tiles left in the wall, wall third 2)

* bot (seat 0): `79p`, open `888p 678p 456p 345p`; 0 from ready
  discards: 4s 3m 9m 5m 4m 9s 9s 8m 8m 8m G S (E) 5p N
* next (下家) (seat 1): `1233567889s`, open `EEEE`; truly 1 from ready; read deal-in on 7p: 0.000
  discards: (8p) 4p 5m 2m 3p 2m 2m 6p 6m R 7p (6m) W Wh S
* opposite (對家) (seat 2): `1345556678s N G R`; truly 2 from ready; read deal-in on 7p: 0.000
  discards: 1m 5p 3p (1p) 3p 2p 2p 2p 2m 9p 9p (4m) S
* previous (上家) (seat 3): `77m 7s W`, open `111p 666m 444m`; truly 1 from ready; read deal-in on 7p: 0.010
  discards: 4s 2s 7s 9s 1m 1s 3m 9m 8m (8p) (6p) (4p) 5m S

`stage`'s pick 7p, read 0.010 (nobody wins on it); thrown instead 9p (costs 0 steps of progress).
Plain game: bot won off previous (上家) with 9 faan (bot +288). Bot game: opposite (對家) won off bot with 8 faan (bot -64).

#### Example 15: win lost, -320 points (deal 24001132, seat 0; 57 tiles left in the wall, wall third 2)

* bot (seat 0): `25s`, open `123s 567s 567s 234s`; 0 from ready
  discards: 3p 4m 5m 6m E G E 1p
* next (下家) (seat 1): `112m 19p 19s S W N Wh G R`; truly 1 from ready; read deal-in on 2s: 0.000
  discards: 8m 6p 8s 3s 2p 2m 7m W
* opposite (對家) (seat 2): `24456779p SS WhWhWh`; truly 1 from ready; read deal-in on 2s: 0.000
  discards: 3s 9m 4s 6m 5m 5m 8m R
* previous (上家) (seat 3): `4679m 357788p W N R`; truly 5 from ready; read deal-in on 2s: 0.000
  discards: (3s) (6s) (5s) (4s) 1s 8s 3m 3m

`stage`'s pick 2s, read 0.000 (nobody wins on it); thrown instead 5s (costs 0 steps of progress).
Plain game: bot won off next (下家) with 9 faan (bot +288). Bot game: opposite (對家) self-drew 6 faan (bot -32).

#### Example 16: later deal-in, -128 points (deal 24006095, seat 1; 65 tiles left in the wall, wall third 1)

* bot (seat 1): `E R`, open `666s 888s 234s 555s`; 0 from ready
  discards: 2p 4p 4m 6m 9m 8m 6p
* next (下家) (seat 2): `111m 19p 19s S W N Wh G R`; truly 1 from ready; read deal-in on E: 0.000
  discards: 7p 2s 7m 6s 5m 5m (5p)
* opposite (對家) (seat 3): `23m 11257p SSS`, open `567p`; truly 2 from ready; read deal-in on E: 0.000
  discards: 1s (6s) (8s) 3s 5s 6m 8m
* previous (上家) (seat 0): `134678m 34689p N Wh`; truly 5 from ready; read deal-in on E: 0.000
  discards: 4s 4s 1s (3s) 3s (5s)

`stage`'s pick E, read 0.000 (nobody wins on it); thrown instead R (costs 0 steps of progress).
Plain game: opposite (對家) won off previous (上家) with 4 faan (bot +0). Bot game: next (下家) won off bot with 10 faan (bot -128).

#### Example 17: later deal-in, -128 points (deal 24007383, seat 0; 18 tiles left in the wall, wall third 3)

* bot (seat 0): `789s E G`, open `234s 567s WWW`; 0 from ready
  discards: 8p 8p (2m) 6s 5m 7m 7p 3p 4p (4m) 3p S 3m N (Wh) 4p 6m 6s 2p
* next (下家) (seat 1): `111677m EE GG`, open `234m`; truly 1 from ready; read deal-in on E: 0.001
  discards: 1s 5s 5p 4p 6p 5p 7s 9p 1p 1p 1s N (R) 3s 4s 2s 9m
* opposite (對家) (seat 2): `555888m G`, open `RRR WhWhWh`; truly 0 from ready; read deal-in on E: 0.246
  discards: 7s 1s 7p 2p 3p 9p 4s 2s 6p 4m 1s 8s N S 1m 3p 8s (W)
* previous (上家) (seat 3): `34466799m E S`, open `222m`; truly 2 from ready; read deal-in on E: 0.001
  discards: 6s 8s 9p 5s 4s 8p 7p 5p 5p 1p 6p 6p (3s) R Wh 2s (7s) N

`stage`'s pick E, read 0.248 (nobody wins on it); thrown instead 7s (costs 1 step of progress).
Plain game: draw (wall ran out). Bot game: opposite (對家) won off bot with 10 faan (bot -128).

#### Example 18: other's self-draw, -128 points (deal 24001372, seat 1; 28 tiles left in the wall, wall third 3)

* bot (seat 1): `67p WWW`, open `999p 444m 555p`; 0 from ready
  discards: 2s 6s 6m 3p 8s 9s 9m 1p 4m 3p 1m 2m 3m Wh N
* next (下家) (seat 2): `11456667p GG`, open `WhWhWh`; truly 1 from ready; read deal-in on 6p: 0.038
  discards: 6m 3m 1m 8s 9s 3s (2s) 1s 1s 1s N R E R R 2p
* opposite (對家) (seat 3): `445667s SSS G`, open `123s`; truly 1 from ready; read deal-in on 6p: 0.000
  discards: 8m 3m 5m 1m 2m 8p (9p) 9p 8m 9s 3m (Wh) E W 7m 8p
* previous (上家) (seat 0): `677m 3346789s N G R`; truly 4 from ready; read deal-in on 6p: 0.000
  discards: 8p 2p 2p 3p 3p 2m 1m (4m) 2m 2p 5m 8p 8m (5p) 6m

`stage`'s pick 6p, read 0.038 (nobody wins on it); thrown instead W (costs 1 step of progress).
Plain game: next (下家) won off opposite (對家) with 6 faan (bot +0). Bot game: next (下家) self-drew 10 faan (bot -128).

#### Example 19: other's self-draw, -128 points (deal 24001422, seat 3; 27 tiles left in the wall, wall third 3)

* bot (seat 3): `12346p W NN`, open `WhWhWh 456p`; 0 from ready
  discards: 6s 7s 1s (9m) 4m 3s 3m 2s 5m 1m 5m 5m 5m 7s 9p
* next (下家) (seat 0): `457789p EE WW`, open `GGG`; truly 1 from ready; read deal-in on W: 0.230
  discards: 9s 9s 5s 5s 6m 7m 1m 1s 1m S R 2p (9p) 8m R 8s
* opposite (對家) (seat 1): `11125566p SS`, open `789p`; truly 1 from ready; read deal-in on W: 0.020
  discards: 2s 3s 2s 8m 4m 8s 8m 3m 2m (6m) 5s 2m (Wh) 4s 6m 6m
* previous (上家) (seat 2): `23347m E N`, open `999m 678m`; truly 2 from ready; read deal-in on W: 0.128
  discards: 8p 3p 3p 8p 9s 8s 2s 6s 3s 4s 2p S (G) 3s Wh (4p)

`stage`'s pick W, read 0.342 (nobody wins on it); thrown instead 4p (costs 1 step of progress).
Plain game: opposite (對家) won off previous (上家) with 3 faan (bot +0). Bot game: opposite (對家) self-drew 10 faan (bot -128).

#### Example 20: other's self-draw, -128 points (deal 24002870, seat 2; 16 tiles left in the wall, wall third 3)

* bot (seat 2): `1345666678s G`, open `EEE`; 0 from ready
  discards: 6m 9m 6p 7p 4m 4p 3p 8m 5p 2p 1p 6m 7m 2m 9p Wh
* next (下家) (seat 3): `44s WW`, open `3333m 111m SSS`; truly 0 from ready; read deal-in on 1s: 0.391
  discards: 6p 4p 5p 5m 3s 9p 8s 2m 8m N 9s Wh 4m 2m 3p 9m (R) 3s 5m
* opposite (對家) (seat 0): `66789p NN Wh GG`, open `RRR`; truly 1 from ready; read deal-in on 1s: 0.000
  discards: 4m 6m 7m 2s 5s 2m 2s 7s 8s 1s 9s (3m) 2s 9m 7m 5m 2p (S) 5s 5s
* previous (上家) (seat 1): `19m 19p 1s E S WW N Wh G R`; truly 0 from ready; read deal-in on 1s: 0.064
  discards: 8s 4p 7s 6m 8m 4m 4s 2p 8p 8p 4p (1m) 5p 2p 3p (E) 8p 7p

`stage`'s pick 1s, read 0.430 (nobody wins on it); thrown instead 5s (costs 1 step of progress).
Plain game: previous (上家) won off opposite (對家) with 10 faan (bot +0). Bot game: previous (上家) self-drew 10 faan (bot -128).

## Limits

* "False alarm" means only that `stage`'s pick would not have been won on at once. A changed pick
  can still save points later (an opponent who becomes ready next turn), which is why many false
  alarms end better than the plain game (`safe1`: 17.3 % of them, +36 each).
* Each hand is labelled by its first change only; later changes in the same hand are counted in its
  points but not labelled.
* One seed (24), one reading (`belief-order.mjnn`). The thresholds came from seed 23, so `safe1`'s
  gain is out of sample, but it is one match: +0.242 ± 0.074 under the old liability rule (3.3 times
  its range from zero), +0.160 ± 0.067 under the amended rule (2.4 times).
* The ready-state outputs of the reading (far / one away / ready per opponent) were not used by any
  bot here.

## Verdict

Most of the false-alarm loss comes from changing the pick of a hand **one away from ready** on a
**low reading (below 0.05)** against an opponent who is **truly far**, usually one with one or two
open sets: in `safe5` those kinds carry 62–90 % of the loss, `ev` changes picks rated below 0.01
and loses the wins (−4.81 a hand), and `fold` breaks ready hands (−12.50 each). At the 1 % setting
(`safe1`, threshold 0.135) false alarms cost nothing on average under the old 12-tile rule and the bot
**gains +0.242 ± 0.074 points a hand** over plain `stage`, the first defence bot to beat it; under
the amended rule (self-draw only, the rule now in force) it **gains +0.160 ± 0.067**, its false
alarms costing −0.085 a hand, all of it in one-away hands. Proposal (not built): keep
`safe1`'s rule, and never change a one-away hand's pick on a reading below 0.135 (the 1 % threshold
for one-away hands, the 5 % one only for hands three or more away, where its false alarms are level or gain), and leave `ev` and `fold` aside;
then confirm on a fresh seed under the amended liability rule.
