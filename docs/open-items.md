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
