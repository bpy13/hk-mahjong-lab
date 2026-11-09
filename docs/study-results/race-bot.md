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
