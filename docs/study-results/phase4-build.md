# Build-run results (B25): draft

Results of the mahjong bot build run (plan `docs/plans/2026-10-29-mahjong-bot-build.md`).
Every comparison is a duplicate tournament. Points are per hand under the default rules
(3-faan minimum, 10-faan limit, half-spicy, flowers on, liability on). "±" is a 95 %
interval. The candidate plays against three `value` bots unless stated otherwise, so 0
means "as good as `value`". This file is a draft: sections are added as the run's matches
finish, and the final comparison (plan AC9) is written at the end of the run.

## 1. The PIMC anchor: `value` or `ev` (plan Task 11)

**Question.** Belief-PIMC-64 must beat a baseline move by two standard errors before it
plays anything else; the baseline (the *anchor*) is the `value` bot's choice. Does the
`ev` bot (Task 10, `floor=none`) make a better anchor, and a better policy for the
searcher's own moves in the rollouts? The winner labels Phase 4's training data (Task 13).

**Matches.** 1,000 walls × 4 seatings (4,000 hands each), seeds 7000–7019 in 20 chunks of
50 walls, against three `value` bots, all on the same walls:

| Match | Spec (after `pimc:samples=64,candidates=4,belief=models/belief.mjnn,weights=1`) | Report |
|---|---|---|
| Task 7 reference | (none: anchor `value`, rollout `fast`) | `runs/matches/pimc64-belief.json` |
| Task 11 | `anchor=ev,rollout=ev` | `runs/matches/pimc64-ev.json` |
| Task 11, anchor only | `anchor=ev,rollout=fast` | `runs/matches/pimc64-ev-anchor.json` |

Intervals: each chunk's interval combined over chunks (Task 7's job's merge); the
difference between two matches uses an interval that treats them as independent, which
is conservative because they share walls (`runs/matches/task11/compare.json`).

**Results.** Pending: the job `runs/matches/task11/` starts after Task 7's job ends.

| Agent | Points / hand | Win | Deal-in | Minus Task 7 reference |
|---|---|---|---|---|
| Belief-PIMC-64, anchor `value` (Task 7) | pending | | | 0 |
| Belief-PIMC-64, anchor `ev`, rollout `ev` | pending | | | pending |
| Belief-PIMC-64, anchor `ev`, rollout `fast` | pending | | | pending |

**Decision rule (plan).** Whichever anchor scores higher becomes the Phase 4 labeller. If
the interval of the difference includes 0, this section says so; what to pick then is an
open question for Brian (`docs/open-items.md`), because anchor `ev` with `ev` rollouts
costs about 7 times as much machine time per label.
