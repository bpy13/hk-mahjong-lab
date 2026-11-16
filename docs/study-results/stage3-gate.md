# Stage 3 gate: no tile outside the leading line (Task 9, 2026-11-16)

Brian, 2026-11-16, to the Mahjong watch session: "then make phase 3 the extra requirement that
is found in phase 2 as well, no irrelevant tile for top priority route". Later the same day,
through the orchestrator: "use strict version as default". Brief:
`C:\Users\brian\.orchestrator\mahjong-watch\brief-stage3-gate.md`.

## What changed (`crates/search/src/stage.rs`)
`StageBot` has a third switch, `gate`:

| gate | rule | bot name |
|---|---|---|
| 0 | stage 3 whenever the hand is within `ready` of ready (the first build) | `stage-gate0` (also the older names `stage-pung`, `stage1`, ...) |
| 1 | stage 3 only when the leading-line check finds no tile outside the line; a hand with an outside tile plays stage 1, even when ready (Brian's words read strictly) | `stage` (default, `StageBot::new(3, 1)`), alias `stage-gate1` |
| 2 | like 1, but a ready hand stays in stage 3 (the watch session's addition, not Brian's words) | `stage-gate2` |

* The check is the one stage 2 uses: the flush leads and a candidate tile lies outside the
  kept suits (`off` in `pick_discard`; in `claim`, a concealed tile outside the kept suits).
  It applies to the discard and to the claim decision.
* For leads other than the flush there is no "outside tile" notion, so stage 3 starts there
  as before (the watch session's reading, kept here).
* "Ready" for gate 2: on a discard, the best discard leaves the hand ready; on a claim, the
  13-tile hand is ready (the builder's reading).
* The older bot names (`stage1`, `stage2`, `stage-orig`, `stage-pung`, `stage-ready2`,
  `stage-both`) keep gate 0, since they name builds that were measured (the builder's call).
* Brian's example `11123456789s1m5p` (one from ready, holds 1m and 5p): offered a 4s, gate 0
  chows 456s (stage 3), gates 1 and 2 pass (stage 1: the chow does not bring the flush
  closer). With a drawn 9m, gate 0 discards in stage 3, gates 1 and 2 in stage 1 (an outside
  tile goes).
* Check that gate 0 is the old bot: `compare stage-gate0 value value value 1000 7` from the
  new build printed the same rows as `compare stage ...` from the build before the change
  (+2.22 ± 1.32, win 24.8 %, deal-in 17.9 %, avg faan 5.1); the 40,000-deal match (job 161)
  must reproduce +2.54 ± 0.21 digit for digit.

## Results (cpu lane, 40,000 deals, seed 1; outputs in `runs\stage\t9-*.txt` of the worker clone)
Baseline (`stage` before this change = `stage-gate0`, job 149): +2.54 ± 0.21 per hand, win
25.1 %, deal-in 18.4 %, avg faan 5.1.

Jobs (all ended with exit code 0 on 16 Nov; job 161 failed on a queue fault before it ran and
was repeated as job 163): 160 (`stage` vs three value), 163 (`stage-gate0` vs three value),
157 (`stage-gate2` vs three value), 162 (one table `stage stage-gate0 stage-gate2 stage-gate0`),
159 (stage counts, `dealin 10000 1 <gate>` for gates 0, 1, 2). Binaries:
`C:\Users\brian\.cache\hk-mahjong-t9\bin` (built from commit acabd24 with a scratch
`CARGO_TARGET_DIR`).

### Each gate against three value bots (40,000 deals, the same deals for each)
| Bot | Gate | Points per hand | Win rate | Deal-in | Average faan |
|---|---|---|---|---|---|
| `stage` (default) | 1 | +2.62 ± 0.21 | 25.1 % | 18.5 % | 5.1 |
| `stage-gate0` | 0 | +2.54 ± 0.21 | 25.1 % | 18.4 % | 5.1 |
| `stage-gate2` | 2 | +2.62 ± 0.21 | 25.1 % | 18.5 % | 5.1 |

`stage-gate0` equals the baseline (job 149) digit for digit, so gate 0 is the old bot.
The three value bots scored -0.81/-0.80/-1.01 (gate 1), -0.82/-0.75/-0.97 (gate 0) and
-0.78/-0.81/-1.03 (gate 2). Draws 5.9 % in all three.

### One table, the gates against each other (40,000 deals, seats rotated)
| Seat | Gate | Points per hand | Win rate | Deal-in | Average faan |
|---|---|---|---|---|---|
| `stage` | 1 | -0.19 ± 0.22 | 23.8 % | 18.7 % | 5.1 |
| `stage-gate0` | 0 | +0.03 ± 0.22 | 24.1 % | 18.6 % | 5.1 |
| `stage-gate2` | 2 | -0.01 ± 0.22 | 24.1 % | 18.6 % | 5.1 |
| `stage-gate0` | 0 | +0.16 ± 0.22 | 24.2 % | 18.4 % | 5.1 |

Draws 5.4 %.

### Stage counts (`dealin`, 10,000 deals, seed 1, the bot in each seat in turn = 40,000 hands against three value bots)
Discards per hand in each stage, and deal-ins per 100 hands from discards made in that stage:

| Gate | Stage 1 | Stage 2 | Stage 3 | Deal-ins in stage 1 / 2 / 3 per 100 hands | Win | Deal-in | Avg faan |
|---|---|---|---|---|---|---|---|
| 0 | 10.94 | 0.73 | 3.94 | 5.26 / 0.84 / 12.37 | 24.96 % | 18.51 % | 5.09 |
| 1 | 12.42 | 0.73 | 2.46 | 9.28 / 0.84 / 8.40 | 24.92 % | 18.56 % | 5.11 |
| 2 | 11.91 | 0.73 | 2.96 | 7.44 / 0.84 / 10.21 | 24.94 % | 18.54 % | 5.11 |

Gate 1 moves about 1.5 discards per hand (38 % of stage 3) back into stage 1, all of them in
flush-led hands (flush-led discards per hand 9.28, 10.77, 10.26 for gates 0, 1, 2; the other
leads are unchanged, as the gate does not apply to them). Gate 2 moves about 1.0. The deal-ins
move with the discards; the total deal-in rate does not change (18.51 to 18.56 %). Stage 2 is
unchanged.

### Which gate looks better
None measurably. Against value bots gates 1 and 2 score +0.08 above gate 0, well inside the
±0.21 range; at one table gate 1 is lowest (-0.19) and gate 0 highest, again all inside
each other's ±0.22 ranges. Gate 2 is level with or above gate 1 in both measures, so if one
is to be preferred on these numbers it is gate 2, but the data do not separate them. (The
matches share their deals, so the true noise of a difference is smaller than the ranges shown,
but the outputs give no paired range.) The strict default (`stage`, gate 1) is not clearly
weaker than `stage-gate0`: the ranges overlap, so Task 4b's "clearly weaker" note does not
apply. Brian decides the default; nothing else changes on these numbers.
