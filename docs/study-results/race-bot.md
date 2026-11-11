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
## Tasks 8 and 9: the race loop (queue job 90, ended 2026-11-11 15:50, exit 0)

Source: standalone checkout `runs/race/` (`report.md`, `report.json`, `g*/a0/fit.json`,
`g*/a0/result.json`); the loop is `experiments/race_loop.ps1` + `race_loop.py`. Four
generations (0 = the heuristic bot, 1–3 trained), each 4,000 hands per match (1,000 walls ×
4 seatings); stop rule fired: "last two gains inside ±1.2 and the score against value rose
+0.27 over the last three generations". Best generation: 3.

| Gen | Gain vs previous | vs 3 value | vs teacher (race − teacher) | Win | Deal-in | Draw | Mean faan |
|---|---|---|---|---|---|---|---|
| 0 | — | -9.63 [-10.31, -8.96] | — | 0.002 | 0.107 | 0.164 | 7.00 |
| 1 | -0.00 [-0.25, +0.25] | -9.93 [-10.59, -9.27] | -11.31 [-12.85, -9.77] | 0.002 | 0.120 | 0.162 | 6.56 |
| 2 | +0.06 [-0.23, +0.35] | -9.13 [-9.79, -8.47] | -12.41 [-13.93, -10.89] | 0.003 | 0.111 | 0.158 | 7.69 |
| 3 | +0.18 [-0.08, +0.44] | -9.37 [-9.98, -8.76] | -10.67 [-12.06, -9.28] | 0.003 | 0.115 | 0.166 | 5.83 |

**Result (AC6): no generation is measurably better than the one before; none comes near the
teacher or `value`.** The race bot stays about 9 points per hand behind three `value` bots.
Its win rate is 0.2–0.3 % (the `value` bots win far more), so the loop converged on a bot
that almost never wins; self-play between race generations is 98 % draws. This is a
finding for Brian and the next step (the hazard table below is the likeliest cause): the
review keeps the teacher, as the plan said "the report decides".

### Held-out calibration of the state and mode heads (generation 3, `fit.json`)
Observed share vs mean predicted, per wall-progress bucket 0 (wall full) – 5 (wall nearly
empty), on a 5,000-hand hold-out with its own seed. "Within" means the observed share lies
inside the 95 % interval of the prediction (it rarely does with 250,000 rows per bucket;
read the gaps, not the flags).

| Head / class | b0 obs / pred | b1 | b2 | b3 | b4 | b5 |
|---|---|---|---|---|---|---|
| state far | .9992 / .9977 | .9928 / .9917 | .9768 / .9772 | .9556 / .9566 | .9399 / .9420 | .9418 / .9408 |
| state one away | .0007 / .0020 | .0062 / .0066 | .0180 / .0165 | .0297 / .0279 | .0320 / .0322 | .0297 / .0298 |
| state ready | .0001 / .0003 | .0011 / .0015 | .0050 / .0056 | .0131 / .0126 | .0188 / .0181 | .0209 / .0213 |
| state folded | .0000 / .0000 | .0000 / .0002 | .0001 / .0008 | .0016 / .0029 | .0093 / .0077 | .0076 / .0080 |
| mode push | 1.000 / 1.000 | 1.000 / .9998 | .9998 / .9993 | .9984 / .9978 | .9939 / .9948 | .9937 / .9938 |
| mode detour | .0000 / .0000 | .0000 / .0001 | .0001 / .0002 | .0005 / .0004 | .0007 / .0009 | .0016 / .0016 |
| mode fold | .0000 / .0000 | .0000 / .0001 | .0001 / .0005 | .0011 / .0017 | .0054 / .0044 | .0047 / .0045 |

The state head is calibrated to within 0.2 percentage points everywhere except the
rare early classes (a few hundred rows of "ready" or "folded" in buckets 0–2, where the
prediction is 2–5× too high in relative terms but under 0.1 point in absolute terms). The
mode head is nearly always "push" (99.4–100 %): self-play bots almost never fold or detour
(mode labels: push 29.9 M, detour 15 k, fold 55 k), so mode calibration is easy and says
little. Hold-out loss per epoch (state 0.1024, mode 0.0144 at epoch 4) fell monotonically.

### The race bot's own win chance (hazard check, §8.2)
The predicted chance that an opponent wins within the bot's horizon is 3–20× the observed
rate in every bucket (e.g. bucket 3: predicted 0.048, observed 0.0124; bucket 0: 0.028 vs
0.0014); every bucket is flagged `refit`. The loop refits `a`, `b`, `q` per generation, but
this prediction stays far too high, so the bot overrates every opponent and plays far too
defensively, which fits its tiny win rate. Not fixed here.

## Task 20: why the race bot folded, the fix, and the strength check (2026-11-11)

Method: `race_log` logs of 20,000 to 24,000 hands, one race seat against three `value` bots
(`--fixed value --shares 0,0,100`), `belief.mjnn`; the hazard check as in §8.2 but with
intervals that resample whole hands (rows of one hand are not independent; the old per-row
intervals are about 3 times too narrow). Scripts: `crates/search/examples/race_offence.rs`
(new), `race_log`, `compare`. Fit logs and held-out logs use different seeds (7; 11-16).

**1. The "3-20 times too high" reading was an artifact of where it was measured.** The loop's
hold-out logs are 60 % race-versus-race hands, and race bots almost never win (0.3 % of hands),
so the opponents' observed win rate was close to zero whatever the model said. Against `value`
opponents, the observed share of "opponent wins within the horizon":

| Bot | b0 obs / pred | b1 | b2 | b3 | b4 | b5 |
|---|---|---|---|---|---|---|
| v0 defaults, vs value | .023 / .149 | .082 / .155 | .167 / .188 | .235 / .234 | .232 / .235 | .135 / .118 |
| generation 3 tables, vs value | .025 / .018 | .081 / .038 | .168 / .096 | .238 / .169 | .234 / .186 | .131 / .083 |
| v0 defaults, vs race (self-play) | .0002 / .149 | .0005 / .154 | .001 / .168 | .002 / .184 | .003 / .152 | .002 / .051 |

Inputs: v0's constants are right from bucket 2 on and too high in buckets 0 and 1 (opponents
far from ready early); the fitted generation-3 tables, which were fitted on race self-play,
are too low everywhere against real opponents. The state chain `a`, `b` and the feed rates
`q` therefore do not inflate the chance; the self-play data they were fitted on were the problem.

**2. The real cause of the folding: far-hand discards.** On 1,000 early positions (wall at
least 70, every opponent with P(ready) + P(one away) < 0.5) the race bot picked the
`rank_discards` (pure offence) discard in only **25.3 %** (21.6 % with generation-3 tables).
In the cases checked, offence's discard lowers faan-aware shanten by one with few accepting
tiles; the race value prefers a discard that keeps shanten but accepts 40 to 80 tiles, because
its far-hand model (§11 item 7) rewards acceptance count and gains of 5 to 20 points swamp
any danger term (deal-in terms are under 0.01 points early). Hence a bot that plays nearly
at random early, folds late, and wins 0.1 %.

**3. The fix** (`docs/race-bot.md` §14): (a) discard decisions take a tie margin of
`0.5 + 100 × w` points, `w = Π_o (1 − P(ready_o) − P(one away_o))` (spec `calm=100,calm_power=1`),
so offence decides when no opponent is likely close and the margin shrinks smoothly as any
opponent's chance of being close grows (no switch; claims keep 0.5); (b) `a`, `b`, `q`
refitted on the `value` field and a per-bucket hazard multiplier `hz` fitted until the
predicted win chance matches (defaults in `RaceConfig`).

**4. Checks (fresh seeds, never used for fitting).**

| Check | Result |
|---|---|
| Offence agreement, 1,000 calm early positions, seeds 5 and 99 (belief and heuristic) | 1000 / 1000 (was 253 / 1000); test `calm_positions_take_the_offence_discard` ≥ 95 of 100 |
| Hazard check, seed 16, 24,000 hands, hands resampled | observed inside the interval in **5 of 6** buckets (b3 observed .2207 vs predicted .2175, difference interval [+.0012, +.0052]) |
| Same check on the earlier fit rounds (12,000 hands, seeds 13, 14) | 3 of 6 and 2 of 6: the multiplier was tuned in 3 rounds; 12,000 hands give a ±5 % ratio, so one round of fitting is noise-limited |
| Duplicate match, 1,000 walls × 4 seatings = 4,000 hands vs 3 `value` | **-1.97 ± 1.16 per hand** (95 %), win 16.4 %, deal-in 16.5 %; the same tables with `calm=0`: -9.94 ± 0.63 (win 0.2 %); `calm=100` with v0 tables: -1.39 ± 1.11 |

Reading: the bot was -9.4 [-10.0, -8.8] (job 90 best generation) and is now -1.97
[-3.13, -0.81]: better by 7.4 points, far outside either interval, but still behind `value`
(who score +1.1 to +2.2 in those matches; win 25.6 % against 16.4 %). The calibration alone
does not move the score (`calm=0` stays at -9.9); the offence margin does it. Also tried:
a smaller margin (`calm=10` -5.30, `calm=40,power=4` -3.11, `calm=40,power=2` -1.93,
`calm=1000` -1.4), so the gain saturates near -1.4 to -2.0 once early offence is restored; the
remaining gap is in the race value's mid-game and claim choices, not measured here.
The loop is re-queued in the gpu lane (job 97, `experiments/race_loop.ps1 -Repo <this clone>
-Root runs\race-t20`, because it trains `race-belief` heads and refits the tables on the new bot).

## Task 21: taking or passing a win

Method: `cargo run --release -p mahjong-search --example pass_win -- 1000` (new). Each of 1,000
walls is played four times (race seat rotated) against three `value` bots, once with `race`
(passing a win allowed, `docs/race-bot.md` §4.5) and once with `race:passwin=off`, same random
stream; hands without a pass are identical, so the paired difference is what the passes earned.

| | Result |
|---|---|
| With passing | -2.92 ± 1.10 per hand (95 %) |
| Without passing | -2.89 ± 1.09 per hand |
| Paired difference | **-0.03 ± 0.16** per hand (no measurable effect) |
| Hands with a pass | 33 of 4,000 (0.8 %), 35 passes; mean win value passed 30.3 points |
| Those 33 hands | with passing +24.2 ± 19.8 (I won 17); without +28.1 ± 7.0 (I won 33); paired -3.9 ± 18.9 |

Reading: the bot passes a win in under 1 % of hands, because it needs a much bigger wait still
live and every opponent far. Where it passes it wins half as often as when it takes the win,
and the points are lower on average, but the interval (±18.9 over 33 hands) includes zero, so
passing neither helps nor hurts measurably. The default stays on, as Brian asked
(2026-11-11); `passwin=off` is the safe setting. (The match with passing is -2.92, not the
Task 20 figure -1.97: that was a different set of walls, 1,000 walls seeded 1 there, 1000+ here.)
Unit tests (`crates/search/tests/race_bot.rs`): a 5-faan discard win with an 8-faan wait live,
60 tiles left, every opponent far: passed; same hand with an opponent 90 % ready, or 6 tiles
left: taken; after the pass the tile is locked for seat 0 while seat 2 is on turn.
