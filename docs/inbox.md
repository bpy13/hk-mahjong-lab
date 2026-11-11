# Inbox

Questions only Brian can answer. Format: a bold title that is the question itself, then What it is,
Example, Why it matters, Options with the Default marked. The orchestrator turns each into a card.

## Open

## Answered

- **Task 20 is done: the race bot now scores -1.97 per hand against three `value` bots (was -9.4). Should the loop stay queued?**
  - What it is: the "3-20 times too high" win chance came from measuring in race-versus-race logs, where nobody wins; against `value` the old constants were nearly right. The real cause was that in quiet early hands the race bot picked pure offence's discard in only 25 % of positions (target 95 %). It now takes a wider tie margin when no opponent is likely close (smooth, no switch), and `a`, `b`, `q` and a hazard multiplier were refitted on the `value` field. Details: `docs/study-results/race-bot.md` (Task 20), `docs/race-bot.md` §14.
  - Example: 4,000-hand duplicate match vs three `value`: -1.97 ± 1.16 (win 16.4 %, `value` 25.6 %). Offence agreement on 1,000 calm early positions: 1000 of 1000. Hazard check on 24,000 fresh hands: 5 of 6 buckets inside the interval.
  - Why it matters: it beats -9.4 by far more than the interval, so the loop was re-queued as job 97 (gpu lane, runs from the worker clone, `runs/race-t20`); it is still behind `value`, and Task 21 (taking or passing a win) builds on this.
  - Options: (a) keep job 97 queued; (b) cancel it and go on to Task 21 with the hand-built tables.
  - Default: (a).
  - Card sent 2026-11-11 (orchestrator checked the match, agreement and hazard numbers in `docs/study-results/race-bot.md`; job 97 is in the gpu lane behind jobs 68, 69 and 70, and the safety check refused the orchestrator's move to the cpu lane, where job 90 ran).
  - Brian (2026-11-11): "default" (accepts (a): keep the loop). Orchestrator: Brian moved it to the cpu lane himself; it runs as job 98 since 17:54.

- **Should milestone 2 train five copies of the belief model to give honest confidence ranges?**
  - What it is: Task 18c's cheap confidence range (a last-layer Laplace approximation on
    `belief-v2`) failed its check: the observed rate fell inside mean ± 1 sd in 26 % of probability
    groups (95 % interval 13–30 %), where 60–76 % passes. The ranges are far too narrow because the
    model's bias, not its weight noise, is the main error. Details: `docs/belief.md` §8.
  - Example: a tenpai guess of 0.39 with sd 0.010 has an observed rate of 0.33; the range excludes it.
  - Why it matters: you asked for each guess to carry its own confidence, and "if fails add to
    Milestone 2". Nothing else waits on it.
  - Options: (a) add "train five copies of belief-v2 and use their spread" to milestone 2 (about 5 × 6 GPU
    minutes plus an evaluation); (b) drop the confidence ranges.
  - Default: (a), as you said.
  - Brian (2026-11-11, before the result, D101): "(a), and if fails add to Milestone 2". Recorded as (a): milestone 2 line added in hk-games `docs/milestones.md`.

