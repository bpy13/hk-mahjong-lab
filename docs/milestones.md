# Milestones

This repo is the Hong Kong mahjong game of hk-games (submodule `games/hk-mahjong-lab`). Workers
build only toward the current milestone.

## Current: a self-play network that starts from the three-stage bot
Brian, 2026-11-16: "wrap up this session and send everything to the orchestrator, let it start
working on the reinforcement learning with 3-stage bot as baseline, queued after the existing
DouDiZhu task in the GPU lane"; and "Yes, merge first" (the three-stage bot goes into this
repository before the training). The wording of this milestone is the orchestrator's.
Done when:
- Every task in `docs/plans/2026-11-16-self-play-network.md` is ticked, and Brian has the final
  measurement against the three-stage bot.
- Brian, 2026-11-20, on the first round of "retrain the reading on the full discard order, then a bot that defends with it" (plan Tasks 19 and 20): "yes, of course the first round must show a gain before we consider the future rounds. In fact I would say if it show gains it already passes Milestone 1, and future finetuning can be modified later on". So a measured gain of the defending bot over the plain three-stage bot passes the milestone; later rounds are planned only after that.

## Earlier (reached): the 2026-11-09 plan built, with its strength checks
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
