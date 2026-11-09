# Race bot: passed-tile safety and risk read (plan 2026-11-09, Phase 6, Task 16)

Status: job 34 failed at once (PowerShell parse error: `"$name:"`); fixed and requeued as job 35 on 2026-11-10; results not in yet.

What the job measures (`experiments/passed-tiles/job.ps1`, cpu lane, after job 32):

- Race bot v0 (`race:belief=models/belief.mjnn`) with the passed-tile safety (Task 14) and the
  risk read (Task 15) against the same bot with `passed=off,risk=off`, and each switch alone.
  Solo rows play each variant with three `value` bots. Head-to-head rows seat two variants
  and two `value` bots; their wall-paired bootstrap interval is the verdict.
  1,000 walls x 4 seatings, seed 2026.
- Sanity row: the teacher (`runs/labels-job/teacher-used.txt`) against three `value` with the
  passed-win rule on and off (`--no-passed-win-lock`), 300 walls. The rule changes the game,
  so this pair is not wall-paired; it only shows what the rule costs the teacher.

Where results land: `runs/matches/passed-*.json` (one tournament report each) and
`runs/matches/passed-summary.json` (from `experiments/passed-tiles/summary.py`).

## Task 7 (AC5): race bot v0 against three `value` bots

Source: `experiments/race-v0/job.ps1`, queue job 32 (exit 0, 2026-11-09 20:11, about 16 s),
output `runs/matches/race-v0-belief.json`. `race:belief=models/belief.mjnn` against three
`value`, 1,000 walls x 4 seatings (4,000 hands), seed 2026, passed-win rule on, 2,000-sample
wall bootstrap. The write-up was done 2026-11-10 by the hk-games worker.

| Agent | Points per hand | 95 % bootstrap | Win rate | Deal-in rate | Mean faan of wins |
|---|---|---|---|---|---|
| race (v0) | -9.128 | [-9.794, -8.496] | 0.4 % (16 wins) | 10.9 % | 6.88 |
| value (mean of the three) | about +2.6 to +3.4 each | e.g. [1.38, 3.83] | 25.9 % to 29.3 % | 16.8 % to 17.3 % | about 4.8 |

Draw rate 17.1 % (684 draws).

Reading (measured): the race bot plays complete, legal hands (4,000 hands, no failure) but is
far weaker than `value`. It almost never wins (16 of 4,000), so it folds or stalls nearly all the
time and still pays 437 deal-ins. Its few wins are big (mean 6.9 faan, 2 thirteen orphans).
AC5 asks for a measured strength with an interval: that is met. It sets no bar on the number.

Speed (inferred, not measured): the whole job took about 16 s on 8 threads for 4,000 hands, which
puts the average cost per race-bot decision well under 5 ms. A direct per-decision timing is
not recorded.

Consequence for Task 8 and 9: the race bot's heuristic estimates are a poor starting point
(-9 points per hand); the loop's first generation will start far below `value`.


## Task 16: what the passed-tile safety and the risk read are worth

Source: `experiments/passed-tiles/job.ps1`, queue job 35 (exit 0, 2026-11-10 03:40 to 04:04),
output `runs/matches/passed-*.json`, summary `runs/matches/passed-summary.json`. Race bot with
Tasks 14 and 15 ("full") against three variants, 1,000 walls x 4 seatings, seed 2026. The
write-up was done 2026-11-10 by the hk-games worker.

Head to head (full race bot against the variant, paired by wall, bootstrap; points per hand):

| Comparison | Mean | 95 % bootstrap |
|---|---|---|
| full - off (no Tasks 14-15 changes) | +0.228 | [-0.134, +0.600] |
| full - nopassed (no passed-tile safety) | +0.058 | [-0.140, +0.264] |
| full - norisk (no risk read) | +0.222 | [-0.142, +0.616] |

Alone against three `value` bots, all four variants score about -9.1 points per hand
(full -9.112, off -9.128, nopassed -9.054, norisk -9.176), win rate 0.4 %. All differences
have intervals that include zero.

Reading (measured): every interval contains zero. The data do not show that Tasks 14 and 15
help or hurt. The race bot is so weak against `value` (-9 points per hand) that a small
safety gain cannot show up in this match.

Teacher sanity row (300 walls): the rule-on and rule-off rows are identical to every digit
(mean 1.593, win rate 25.25 %, deal-in 19 %). Inferred, not checked: either the teacher never
meets a passed-win case in 300 walls, or `--no-passed-win-lock` does not change the teacher's
games. Left to check: run the two rows with a different seed, or count passed-win locks in the
rule-on logs. This row is therefore not evidence that the rule is neutral for the teacher.