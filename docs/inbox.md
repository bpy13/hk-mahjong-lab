# Inbox

Questions only Brian can answer. Format: a bold title that is the question itself, then What it is,
Example, Why it matters, Options with the Default marked. The orchestrator turns each into a card.

## Open

- **The belief network reads a ready hand at 14 % where such hands are ready 48 % of the time. Fix it with a small public-facts table, refit the network with the drawn-and-thrown flag, or leave it?**
  - What it is: Task 18 measured, on 20,000 fresh deals of four `stage` bots (3.7 M opponent readings), how often an opponent is truly ready against what the belief network (`models/belief.mjnn`) and the danger table say. West's situation in hand 2860 (one open set, early discards all outside one suit then that suit thrown, five or more discards in a row that were the drawn tile thrown straight back) is ready 48 % of the time; the network says 14 %, the table 11 %. The network is 7 to 10 points low in every bin above 10 % and its input has no flag for "drawn and thrown" (摸打) against "thrown from the hand" (入章), nor the order of discards beyond the last three. The same gaps appear on four `value` bots, the kind it was fitted on, so it is the fit and the input, not the new bot. A tile thrown into a shown flush is, on its own, not a sign of readiness for these bots (14 % ready; 6 % when it came from the hand). Report: `docs/study-results/belief-ready-read.md`.
  - Example: hand 2860, West ready from wall 54 with `12377889m GG` and `567m` open, throws eight drawn tiles in a row (白, 1索, 2萬, 8萬, 白, 8索, 3筒, 4索); the other seats read West at 5 to 21 % throughout, the table at 1.5 to 4.3 %.
  - Why it matters: every bot that defends (`race`, the push-or-fold bots, the hint in the app) reads this figure; a seat that is ready half the time read as one in seven makes them push into it. Nothing in the game changes until a fix is built and measured.
  - Options: (a) a ready-state table keyed by public discard facts (open sets × wall bucket × suit pattern × drawn-and-thrown run, 240 cells) fitted from these logs, behind a switch, then a 40,000-deal match of `stage` reading it against plain `stage`: about half a worker day, no GPU, reads West at about 48 %; (b) refit the belief network on a logged `stage` field with the `passed` block (it carries the 摸打/入章 flag and already loads) and the ready loss weighted up: one cpu hour of logging, a 10-minute GPU fit, the same match; mends the under-reading in every bin too; (c) both, (a) first; (d) leave it, the figure stays as it is.
  - Default: (a).
  - Asked by the Task 18 worker (slot 1), 2026-11-20; figures from `runs/belief-ready/stage-20000.md` and `value-20000.md` in the workers clone.
  - Card sent 2026-11-20 (orchestrator; figures checked against `docs/study-results/belief-ready-read.md` sections 2 and 3).

- FYI (no answer needed): the shared `mahjong` env's editable install still points at `hk-mahjong-lab-3\python`, not the workers clone (a permission check refused the repoint, 2026-11-16). Harmless while jobs set `PYTHONPATH`. Tell the orchestrator if you want it repointed after job 154 ends.

- **The 4-hour low-noise run ended level with the three-stage bot (+0.15 ± 0.23 per hand over 40,000 deals). Build per-decision credit and train long, or stop?**
  - What it is: Task 10's run (gpu job 252, 4 h, 659,682 hands; pull to the start fixed at 0.02, 4,096 hands per update, learning rate 3e-5, stall rule off). Its training score (learner minus the frozen start on the same tiles) per hour: -0.86, -0.51, +0.04, +0.01 (each ± 0.26): it rose past the -0.3 mark set beforehand, back to the start's level, not above it. End network against three strict `stage` bots, 40,000 deals (cpu job 253): +0.15 ± 0.23, win 24.0 %, deal-in 18.5 % (the three `stage` seats -0.01, -0.06, -0.08). Task 7's 10-hour network: -0.09 ± 0.22.
  - Example: report `docs/study-results/self-play-network.md`, section "Low-noise run". The analysis (`docs/study-results/2026-11-17-self-play-analysis.md`, experiment 4) said: a rise to -0.3 or better means the signal is there once the noise is cut, so the next step is per-decision credit (each decision gets its own advantage from the critic already in the network, instead of the whole hand's result) and a long run.
  - Why it matters: neither run has beaten `stage`; a long run costs many GPU hours in the shared lane (mahjong has priority). The rise says learning works; the end level says 4 hours was not enough to gain.
  - Options: (a) new plan: build per-decision credit with the critic, test it on a 4-hour run, then a long run if its hourly score goes above the start; (b) stop self-play and keep `stage` as the best bot; (c) run the same low-noise settings longer (e.g. 24 h) without code changes.
  - Default: (a), as the analysis set out beforehand.
  - Card sent 2026-11-19 (orchestrator; figures checked against `runs/rl-lownoise/final/vs-stage-40000.txt` and the hourly table).

## Answered

- **Flower wins: the engine scores 七搶一 and 八仙過海 as limit hands, with a robbing win. Follow Hong Kong play instead?** (asked in chat by the orchestrator, 2026-11-20)
  - Options given: (a) 花糊 at 3 faan and 大花糊 at 8 faan, no robbing (recommended); (b) no flower wins at all; (c) leave it as it is.
  - Brian (2026-11-20): "wait, seven rob one is hong kong rule?", "I am pretty sure that's not the case", then on the options: "a". Plan Task 17 (self-play-network plan, Phase 8) builds it.

- **The self-play network did not beat the three-stage bot (-0.09 ± 0.22 per hand over 40,000 deals). Keep training, put it into the game, or stop?**
  - What it is: Task 7's final measurement. Best network (hour 6 of a 10 h run that stopped itself on its stall rule, 1.29 M hands) against three `stage` bots: -0.09 ± 0.22 per hand, win 23.7 % (stage 23.9-24.1 %), deal-in 18.3 % (stage 18.4-18.8 %). At one table of 4,000 deals: network +0.65, teacher -0.17, `stage` +1.33, `value` -1.81 (each ± 0.7). Pass mark (above zero against `stage`) is not met.
  - Example: the 2-hourly results were -0.26, +0.31, +0.48, -0.54, -0.88 (each ± 2.0): no trend. Report: `docs/study-results/self-play-network.md`.
  - Why it matters: GPU time is a shared lane; putting a bot level with `stage` into the game adds nothing.
  - Options: (a) stop here and keep `stage` as the best bot; (b) train further with changed settings (e.g. lower learning rate, larger deal count per result) as a new plan; (c) put the network into the game anyway.
  - Default: (a).
  - Card sent 2026-11-17, card 3 (facts checked against `runs/rl-stage/final/*.txt` and `stopped.txt`).
  - Brian (2026-11-17): "yes sure, go ahead with the 5 test runs, higher priority than big2" (to the five runs proposed by the self-play analysis; for mahjong: one 4-hour low-noise run, plan Task 10; the keep / stop choice follows its result).

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

