# Open items (mahjong bot build run)

Plan: `docs/plans/2026-10-29-mahjong-bot-build.md`. Its "Open questions for Brian" still apply
with their defaults.

## Waiting on Brian
- **EV bot: the `Pay(me)` floor** (`docs/ev-bot.md`, Task 10, a2d1b05). The doc sets my payout
  to at least the target hand's guaranteed points. In both test hands `faan_shanten` picks a
  10-faan full flush, so the floor makes my payout 128 points even when the flush is far away.
  Effect: test position 2 does not deal in (safe −13.45 vs deal-in −15.87), and position 1
  chases the flush. Options: keep the floor; drop it (`floor=none`, which passes all three
  positions); or use it only near the target (e.g. faan-shanten ≤ 1). Default until Brian
  answers: the code keeps the doc's floor; tests check the positions with `floor=none`.
- **EV bot provisional strength** (heuristic belief, 4,000 hands each): floor on −7.44 vs 3 ×
  value, −5.23 vs 3 × defensive; `floor=none` −2.46 and −0.37. Neither meets AC5 yet. The real
  belief model (Task 5) comes next; the matches will be rerun with it.
- **Task 11: decided by the result (2026-10-30).** Anchor `value` beats anchor `ev` by 1.5
  (`ev` rollouts) and 2.0 (`fast` rollouts) points per hand; both intervals exclude 0
  (study results §1). The `floor=none` and "intervals overlap" questions no longer matter
  for the labeller. Task 13 labels with Task 7's spec.

- **Task 7: AC4 missed.** Belief-PIMC-64 scores +1.24 ± 1.21 per hand vs `value` (bar
  +3); weights add −0.02 points but cut deal-ins from 18.8 % to 17.8 %. The run goes on
  (proposal §9). Default: keep belief sampling with weights as the reference teacher, not
  the best-of-K fallback, because no result shows best-of-K is stronger.
- **Task 13: the teacher's labels mostly copy `value`.** In the smoke run 3 % of the
  teacher's moves differ from the `value` anchor (the search keeps the anchor unless a
  move beats it by two standard errors). Distillation will learn mostly `value`'s play.
  Default: label as planned; Task 15 weights rows by the search margin. Alternative:
  lower `z` (e.g. `z=1`) for labelling, which needs a new label run.
- **Task 13: the teacher pick.** Settled: the job uses Task 7's spec
  (`runs/labels-job/teacher-used.txt`), which Task 11's result confirms.

## Pending work
- Task 5 belief model (425a7d2): count NLL beats both baselines (0.551 vs 0.634 and 0.662);
  deal-in calibration 4 of 6 buckets; **tenpai over-predicted in all 6 buckets** (by 5–12 %
  relative). AC3 not fully met; `docs/belief.md` §4.3 proposes a per-bucket temperature fitted
  on a second training-field log. Task 6's `danger_check --belief` shows whether it matters.
  The discard-order block was dropped (+0.13 % NLL, bar 2 %). A new `outcome` head gives the
  EV bot's discard-win chance, replacing the `2.0 × self-draw` guess.
- Task 9's target table: 31 of 37 held-out buckets within their 95 % interval; every miss
  under-predicts by 0.4–2 points (limit hands 2.1). Refit with a bias correction, or accept it.
- Task 10: the EV-versus-realised decile plot needs a per-decision log the tournament runner
  does not write yet.
- Task 12 (benchmark mode) was interrupted by a usage limit; its work is uncommitted in the tree.
- Repo-wide mypy finds 19 errors in 4 test files (one is Task 12's test_benchmark.py).
- `make` is not installed on this PC; run the Makefile lines directly (CLAUDE.md).
- Two app-api wall-clock tests fail under outside CPU load; asked the app stream in bot-notes
  (hk-games 5342a50).
