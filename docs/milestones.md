# Milestones

This repo is the Hong Kong mahjong game of hk-games (submodule `games/hk-mahjong-lab`). Workers
build only toward the current milestone.

## Current: the 2026-11-09 plan built, with its strength checks
Registered with the orchestrator on Brian's word (2026-11-10: "Yes, register it"), after the
strength-check tasks were added (2026-11-10: "Yes, add all three").
Done when:
- Every task in `docs/plans/2026-11-09-review-fix-and-race-bot.md` is ticked.

Rules from hk-games: GPU or long CPU jobs go through the job queue
(`C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add`), never
started directly. Never touch the standalone checkout `repo\hk-mahjong-lab` (Brian and the
hk-games sessions work there, and queue jobs run from it); workers use the clone
`repo\.worktrees\hk-mahjong-lab-workers` and its worktrees only. The trained models
(`models/*`) are not in git: copy the ones a task needs from `repo\hk-mahjong-lab\models\`,
read-only, into your own checkout. Before a queue job runs from your checkout, build
`mahjong._core` there or point `PYTHONPATH` at your checkout's `python` folder with a copied
`_core.pyd` (as big2's Task 32 did).
