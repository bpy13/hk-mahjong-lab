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
- **Task 16: 400 walls instead of 1,000.** Closed (2026-11-01): the interval is not a near
  miss, see "Tasks 15–16 result" below.
- **Task 16: the leaf value is slower than full rollouts.** The CNN forward pass at every
  leaf (2–2.5 ms) costs more than playing the hand out with the `fast` bot, so `leaf=value`
  only pays if it plays better. Default: measure both in the match, as planned.
- **Task 15: value-head overfitting.** On a 540 k-row slice the held-out value error was
  lowest after 2 epochs while move agreement kept rising to 7 epochs. The full run uses 12
  epochs, a compromise. On the full 1 M rows the error was 2.4 % above its best at the end;
  this is not why the leaf-value search lost (study results §3). Default: keep.
- **Task 17: PPO throughput — encoding sped up (2026-10-30); above 100 K at above-normal
  priority, load-dependent at normal.** `features::encode` went from 653 to 76 µs per
  position with bit-identical output. The 3-minute GPU smoke (random network, 256 paired
  envs, learner hands per hour): old code 46 K; new code 74 K and 132 K in two runs at
  normal priority; new code 218 K at above-normal priority.
  Finding: on this shared PC the rollout is limited mostly by Windows scheduling, not by
  work. Other jobs keep every core busy at normal priority, so each wake-up of the rollout's
  threads waits for a time slice. In one back-to-back test at that load, old and new code
  both reached ~47 K at normal priority; at above-normal priority, old 121 K, new 219 K.
  New flag: `python -m mahjong.rl.ppo_cnn ... --priority above_normal`.
  **Question for Brian:** may Task 18's long PPO runs use `--priority above_normal`? It
  takes CPU from other jobs (labelling, other streams' solvers) only while the rollout
  works (the encoding burst, 7.6 of ~25 ms per step). Default (changed 2026-11-01, working
  agreement): normal priority until you say otherwise. The 60-minute run reached 370 K hands
  per hour at normal priority (Task 17 ticked), so above-normal is not needed yet.
- **Search RNG moved to ChaCha8 (hk-games D28, 2026-10-30).** PIMC samples, `solo`, the
  tournament walls and bootstrap now use `ChaCha8Rng`, and random-bot draws use u32 ranges,
  so the app's WebAssembly `expert` matches the server. Effect here: every tournament now
  deals different walls for the same seed than before; earlier results (Tasks 7, 10, 11)
  stay valid but cannot be replayed bit for bit with today's code. The labeller keeps
  `SmallRng` (`labels.rs`) so the running label directory stays resumable.
- **Brian's rule (2026-10-30), how it is applied.** "Use teacher unless distilled model
  wins significantly": the job compares each challenger (bare CNN, CNN + PIMC-48, CNN +
  PIMC-48 leaf value) with belief-PIMC-64 in the same 24-seating match, and a challenger
  replaces the teacher only if its wall-paired 95 % bootstrap interval of the difference
  lies above 0 (`runs/matches/netpimc-verdict.md`).

- **Tasks 15–16 result (2026-11-01).** The teacher stays: CNN + PIMC-48 minus teacher is
  +0.02 [−1.28, +1.31] on 400 walls. That interval is not a near miss, so the 400-wall
  question is closed: more walls would not lift it above 0, and no extension runs.
- **Round 2 of expert iteration (a second labelling with the new searcher).** Task 16's note
  starts it when the search moved the network by ≥ +1 with an interval excluding 0. That
  holds (+1.31 [+0.04, +2.58]). But the new searcher only ties the teacher, so your rule
  keeps the teacher. Round 2 costs about 1.2 CPU-days plus 20 GPU minutes. Options: (a) run
  it as the plan says, after Task 18's GPU runs; (b) skip it. Default: (a), and nothing
  starts before you answer. Task 18 runs from today's network either way.
- **Leaf value: rerun the 24-seating match?** The −4.16 came from a bug, now fixed. A short
  check gives +0.88 ± 3.04 vs `value` (800 hands), against −5.12 ± 2.36 before. The full
  match costs 16 h and only one row would change. Default: no rerun now; Task 19 measures
  the fixed leaf.
- **Task 19's search: leaf value only, or also full rollouts?** The plan names PIMC-48 with
  leaf value for Tasks 19, 21 and 22. Full rollouts tied the teacher; the fixed leaf is not
  yet shown to beat the bare network, and it costs twice the time. Example: Task 19's table
  would seat PPO + PIMC-48 with full rollouts in place of `value`. Default: Task 19 plays
  both search variants, and the stronger one goes on to Tasks 21 and 22.
- **Task 18: a 6-hour control run for the two ablations (runs with one feature removed).**
  The ablations train 6 hours; the main run trains 48. AC8 asks for their gain, but a
  6-hour run compared with a 48-hour run mostly shows the extra training. A 6-hour run with
  all features on would give a fair comparison. It costs 6 GPU-hours and is not in the plan
  or the queued job. Default: add it after the job's ablations if you say yes; until then
  the report compares each ablation with the main network and says this limit.
- **Task 18: restart after Windows Update.** The job does not restart by itself; last time
  that cost 19 h. A scheduled task at logon could restart it, but it is a standing change on
  your PC. Default: no scheduled task; rerun the job's command after a restart.

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

## From the review-fix and race-bot run (plan 2026-11-09)
- **`faan_shanten` calls four pin pungs plus a lone Green "ready for a full flush"**; the true
  distance is one step more. A fix passes the analysis tests but changes the feature encoding,
  so every model would need retraining. Not applied; patch kept outside the repo
  (coordinator's scratchpad, `analysis-leftover-fix.patch`). Likely also why `ev` swaps the live
  4p for honour singles after Brian's pung (not traced). Default: fix it in Phase 4, when the
  race bot's network is trained anyway.
- **The danger tables call a tile safe when the player holds its last unseen copy.** Only the
  defending `fast` bot works around it (Task 2); `defensive` and `ev` still have the flaw.
  Default: the race bot counts live copies itself; fix the tables when they are refitted.
- **Passed-win rule (Task 12–13) limits.** The engine counts tiles by kind, so throwing a
  tile of the same kind as the one just drawn counts as 摸打 (no observable difference at a
  table either, unless players watch which copy). States built from a position have no kong
  events in their log, so a re-deal there misses a lock from a passed kong rob (rare).
  Job 32 (race v0 match) now runs with the rule on, since the Python extension was rebuilt.
- **Requeue the poker benchmark after all hk-games tasks** (Brian, 2026-11-09). Job 28 was
  cancelled before it started (no progress lost) to unblock jobs 29–32. Requeue it unchanged:
  `hkq.ps1 add -Lane quiet -Script C:\Users\brian\OneDrive\Documents\workspace\repo\holdem\scripts\jobs\flop-h2h.ps1 -Label holdem-flop-h2h-ts`
  with script arguments `-TsOnly` (check `hkq.ps1` usage for how it passes them).
- `fast` is about 10–15 % slower per hand after Tasks 1–2 (noisy timings).
