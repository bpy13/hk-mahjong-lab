# 0003 — A self-play network that starts from the three-stage bot

**Status:** accepted (Brian, 2026-11-16)

## Decision
A network learns to copy the three-stage rule bot (`stage`), then improves by self-play for up
to 48 hours against a pool of itself, older saved networks and the rule bot. Its input holds the
full discard history. The rule bot is merged into this repository first and its numbers stay.
The graphics card is used after Dou Dizhu's training job. Plan:
`docs/plans/2026-11-16-self-play-network.md`.

## Brian's words
- "let's try reinforcement learning. Use the current 3-stage solver as baseline, and let the
  solver evolve as we go. The solver should include full discard history, not just the last 3.
  You may use techniques such as including a previous checkpoint, older version randomly in
  games to compare their learning outcomes etc."
- "wrap up this session and send everything to the orchestrator, let it start working on the
  reinforcement learning with 3-stage bot as baseline, queued after the existing DouDiZhu task
  in the GPU lane"
- "in general the rules number shouldn't be changed"
- "Yes, merge first" (to: "Merge the three-stage bot into the real mahjong lab first, then train
  there?")

## Not decided
Putting the bot or the network into the game; defence for the rule bot; the one-from-ready
scoring gap (see `docs/study-results/three-stage-bot.md` once Task 1 has copied it).

## Stage 3 gate (2026-11-16)

- Stage 3 of the three-stage bot gets the extra requirement of stage 2: no tile outside the leading line. Two test bots were planned (plan Task 9). (Brian, 2026-11-16, to the Mahjong watch session: "then make phase 3 the extra requirement that is found in phase 2 as well, no irrelevant tile for top priority route")
- The strict version becomes the default bot `stage`, decided before the match results exist; the old behaviour stays as `stage-gate0` for comparison. (Brian, 2026-11-16, to the Mahjong watch session: "use strict version as default")
- The 48-hour self-play run is held and its start (labels, imitation network) is redone on the strict bot; job 154 was cancelled before it started (plan Tasks 4b and 6). Asked "Hold the 48-hour run and redo its start on the strict bot?" and told the redo costs about 35 minutes of preparation, he answered as quoted; reading it as a yes is the orchestrator's. (Brian, 2026-11-16: "that's fine")
- The 48-hour self-play run does not pause while he uses the PC. Asked "Mahjong 48-hour run. Should it pause during hours when you use the PC, and if so which hours?", told it uses the GPU heavily as the job that made the PC lag did. (Brian, 2026-11-17: "keep it running")
- One 4-hour low-noise self-play run (plan Task 10) is made before deciding whether to train further or stop; it runs ahead of big2's jobs. (Brian, 2026-11-17: "yes sure, go ahead with the 5 test runs, higher priority than big2", to the five runs proposed in `docs/study-results/2026-11-17-self-play-analysis.md`)
- Resuming the long self-play run is put aside for now. The 4-hour low-noise run (Task 10, job 189, started 14:08 on 17 Nov) was already running; the orchestrator let it go on and asked Brian whether he meant it too. (Brian, 2026-11-17: "update priorities: new solver for doudizhu using search method, then big2 highest-winrate-bot, then 5 test runs for doudizhu. Put aside resuming the doudizhu and mahjong self-play runs for now")

- The 4-hour low-noise run is paused: the orchestrator cancelled job 189 at 14:36 on 17 Nov, 28 minutes in; nothing is queued for this project. It starts again only on Brian's word. (Brian, 2026-11-17: "what mahjong test run? please pause and explain what it is")