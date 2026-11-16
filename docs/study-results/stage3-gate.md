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

Pending: jobs 160 (`stage` vs three value), 161 (`stage-gate0` vs three value), 157
(`stage-gate2` vs three value), 162 (one table `stage stage-gate0 stage-gate2 stage-gate0`),
159 (stage counts, `dealin 10000 1 <gate>` for gates 0, 1, 2). Binaries:
`C:\Users\brian\.cache\hk-mahjong-t9\bin` (built from this commit with a scratch
`CARGO_TARGET_DIR`; md5 compare 6CD4755D..., dealin E9FDADE6...).

Quick look (1,000 deals, seed 7, not the result): `stage` (gate 1) +2.42 ± 1.32, `stage-gate0`
+2.22 ± 1.32, `stage-gate2` +2.01 ± 1.30; all inside each other's ranges.
