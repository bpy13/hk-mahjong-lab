# Open items (mahjong bot build run)

Plan: `docs/plans/2026-10-29-mahjong-bot-build.md`. Its "Open questions for Brian" still apply
with their defaults.

## Waiting on Brian
- **Review-fix Task 4: AC3 failed (2026-11-10 write-up).** Teacher with the fixed `fast` minus old: -0.556 points per hand, 95 % interval [-1.61, +0.50] (bar: lower end above -0.5). Not a measured loss, but not ruled out. The fix stays for the review's grade; the bots' own play is unproven. Decided: accept, no longer match (Brian, 2026-11-10: "ok, apply the fix"; hk-games D73). Details: `docs/study-results/review-fix.md` section 5.
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
  distance is one step more. **Applied 2026-11-10 (Task 18)** with the passed-tile block; the
  encoding's fast acceptance count needed a matching fix (an isolated tile can now lower the
  measure), checked against the plain count on 600 hands. Whether `ev` still swaps the live 4p
  for honour singles after Brian's pung is not traced.
- **Task 18, code and data half (2026-11-10).** Done: the `faan_shanten` fix; a new public
  input block `passed` (`features::encode_passed`, 488 values: passed tiles per opponent with
  their 入章 since, 入章 discards per seat, own passed-win lock, 摸打 / 入章 of each seat's last
  three discards); the self-play, race and label logs carry it (layouts 2, 103, 2); the
  belief model reads it with `--passed`. Old networks still load: `passed` is a separate
  named input, not more `features` planes. Left for the training half:
  (1) the suit-shared CNN (`cnn.rs`, `rl/cnn.py`; the distilled network and PPO) reads only
  `features`, so it does not see the passed block yet; extend it before distilling;
  (2) the danger tables and `PASSED_FACTOR` refit on the new logs (`fit_danger`, `fit_passed`);
  (3) `features::encode` speed after the fix is not measured (no benchmark while jobs ran).
- **Task 18, training half (2026-11-10).** Jobs: 41 distill (gpu, 13:09–13:21; Dou Dizhu job 39
  cancelled after a 28-minute-old save and requeued as job 42), 43 refit (cpu), 44 match (cpu).
  (1) Distilled network `models/cnn-distill-v2.*` (passed block, labels-v2 from belief-v2's
  teacher, `experiments/task18-data/distill.ps1`, 12 epochs): held-out top-1 agreement with its
  teacher 0.917 (old network with its teacher: 0.924), target 0.988 (0.989), value MSE 1.185
  (1.255), int8 vs fp32 0.992 (0.996). `runs/distill-v2/result.json`.
  (2) Danger tables: refitted with the rule (`fit_danger 8000`, `runs/task18-refit/`). Calibration
  on 2,000 hands is the same as today's tables (every bucket within 0.1–0.6 points of the old
  row; both over-predict the 4–7 % and 10–20 % buckets a little). `defensive` against three
  `value` on the same walls: -1.77 ± 0.94 new vs -1.86 ± 0.95 old (unpaired). **Not adopted:**
  the encoding reads the tables (danger plane, threat scalars), so new tables would shift the
  input of belief-v2 and cnn-distill-v2, which were trained on the old tables; no gain was
  measured. The refitted file is kept in `runs/task18-refit/danger_tables-refit.rs`.
  (3) Belief danger calibration (`danger_check --belief`, 2,000 hands): every bucket within its
  interval for both the old model and belief-v2; belief-v2's expected loss is closer in buckets
  3–4 (0.290 vs observed 0.289; old 0.273).
  (4) `fit_passed --belief models/belief-v2.mjnn` (8,000 hands): the factor is 1.0 for N ≥ 1
  (belief-v2 already reads the passed tiles); N = 0 stays 0 (true rate 0 against an estimate of
  0.0018). If the race bot moves to belief-v2, set `PASSED_FACTOR_MODEL` to [0, 1, 1, 1, 1].
  (5) Duplicate match (job 44, 1,000 walls × 4 seatings, seed 2026, wall-paired bootstrap):
  cnn-distill-v2 minus cnn-distill head to head -0.642 [-2.016, +0.802]; solo against three
  `value`: +0.576 [-0.246, +1.454] new, +0.458 [-0.414, +1.414] old.
  Teacher (belief-PIMC-64) with belief-v2 minus with the old belief model, head to head:
  -0.098 [-1.782, +1.670] (`runs/matches/task18-*.json`).
  Reading (measured): neither retrained model is shown weaker or stronger; both point
  estimates are slightly below zero. **The hk-games pin was not moved:** the app loads no
  belief model or network (its review, drills and bots use the danger tables, which stay
  unchanged), so no retrained model would reach the app anyway.
- **Task 18: switch the lab's default models to the retrained ones?**
  **What it is:** the labeller, the race bot and later loops read `models/belief.mjnn` and
  `models/cnn-distill.mjnn`; the retrained `-v2` files sit beside them. **Example:** the race
  loop (Tasks 8–9) would start from `belief-v2` and `cnn-distill-v2` (passed-tile input).
  **Why it matters:** the match shows no difference (intervals above), but only the v2 files
  match today's encoding (`faan_shanten` fix, passed block). **Options:** (a) use the v2
  files from now on; (b) keep the old ones. **Default: (a)** for new lab work; nothing in the
  app changes either way.
  Separately, the hk-games pin still sits before the `faan_shanten` fix (ba47227); moving it
  changes the review's grading and needs the drills re-solved (orchestrator's call).
- **Task 18: label with the old belief model now, or wait for the retrained one?** Decided: wait (Brian, 2026-11-10: "4 ok"; hk-games D74).
  **What it is:** the teacher that writes the distillation labels (64-sample search) and the
  three other seats read a belief model, a network that guesses opponents' hidden tiles.
  **Example:** the job `experiments\task18-data\labels.ps1 -Belief 'models/belief.mjnn'`
  would start today with the old model; with the retrained model it waits for its GPU
  training, which queues behind Dou Dizhu (about 2026-11-12).
  **Why it matters:** the job takes about 14 hours of CPU. Labels from the old model teach the
  network moves chosen without the passed-tile record; a relabel later costs the 14 hours again.
  **Options:** (a) wait for the retrained belief model, then queue the labels; (b) label now
  with the old model. **Default: (a)**; the script is ready and not queued.
- **The danger tables call a tile safe when the player holds its last unseen copy.** Only the
  defending `fast` bot works around it (Task 2); `defensive` and `ev` still have the flaw.
  Default: the race bot counts live copies itself; fix the tables when they are refitted.
- **Passed-win rule (Task 12–13) limits.** The engine counts tiles by kind, so throwing a
  tile of the same kind as the one just drawn counts as 摸打 (no observable difference at a
  table either, unless players watch which copy). States built from a position have no kong
  events in their log, so a re-deal there misses a lock from a passed kong rob (rare).
  Job 32 (race v0 match) now runs with the rule on, since the Python extension was rebuilt.
- **Tasks 14–15 findings (67cbf71).** (1) Risk-taking does not predict hand value in this
  self-play field (value multipliers within 4 % of 1): Brian's "how hard they push narrows
  what they aim to win" is not seen yet; it predicts ready-or-not and fold only. Refit after
  the race loop, whose players weigh value. (2) The fold multipliers are not monotone (near-zero
  risk bins fold least); used as fitted. (3) The 換章 test reads exposed melds only, so a
  concealed flush never resets passed-tile safety. (4) Refit all tables after Task 18
  (`cargo run --release -p mahjong-search --example fit_passed -- --belief ...`).
  (5) 換章 as "a middle tile 3–7" carried no information (39.0 % changed vs 39.5 % for any
  入章); "a tile of their visible flush suit" did (55 %).
- **Requeue the poker benchmark after all hk-games tasks** (Brian, 2026-11-09). Job 28 was
  cancelled before it started (no progress lost) to unblock jobs 29–32. Requeue it unchanged:
  `hkq.ps1 add -Lane quiet -Script C:\Users\brian\OneDrive\Documents\workspace\repo\holdem\scripts\jobs\flop-h2h.ps1 -Label holdem-flop-h2h-ts`
  with script arguments `-TsOnly` (check `hkq.ps1` usage for how it passes them).
- `fast` is about 10–15 % slower per hand after Tasks 1–2 (noisy timings).
- **Task 18 not ticked: belief calibration reads "not within" in most buckets (slot 1, 2026-11-10).**
  I read the raw results (`runs/belief-v2/eval.json`, `runs/belief/eval.json`, `runs/distill-v2/result.json`,
  `runs/matches/task18-*.json` in the standalone checkout). They match the notes above:
  distill top-1 0.917 (old 0.924), matches inside noise, danger calibration within intervals.
  But `eval.json` marks tenpai calibration `within: false` in 5 of 6 buckets for belief-v2
  (old belief: 6 of 6) and deal_in `within: false` in 2 of 6 for both. belief-v2 is no worse than the
  old model, but the strict "within" check is not passed by either. Missing: a decision whether
  "passes its existing checks" means "no worse than the old model" (then tick) or "every bucket within".
- **Task 18 closed on Brian's ruling (slot 1, 2026-11-11).** A recorded calibration miss counts as done (hk-games reply 2026-11-11). Held-out calibration of belief-v2 (old belief model in brackets), predicted vs observed per wall-progress bucket (uns/belief-v2/eval.json; within = observed inside the 95 % interval of the prediction):

  | head | bucket | n | predicted | observed | within | old: predicted / observed |
  |---|---|---|---|---|---|---|
  | tenpai | 0 | 266658 | 0.10% | 0.03% | False | 0.11% / 0.03% |
  | tenpai | 1 | 294567 | 0.70% | 0.68% | True | 0.77% / 0.70% |
  | tenpai | 2 | 271521 | 4.28% | 4.19% | False | 4.46% / 4.26% |
  | tenpai | 3 | 227463 | 13.82% | 11.85% | False | 13.31% / 11.90% |
  | tenpai | 4 | 156246 | 23.23% | 20.51% | False | 22.45% / 20.50% |
  | tenpai | 5 | 87690 | 30.52% | 28.85% | False | 29.43% / 28.73% |
  | deal_in | 0 | 68730 | 0.01% | 0.00% | False | 0.01% / 0.00% |
  | deal_in | 1 | 77626 | 0.06% | 0.07% | True | 0.07% / 0.07% |
  | deal_in | 2 | 73422 | 0.42% | 0.45% | True | 0.45% / 0.46% |
  | deal_in | 3 | 61806 | 1.40% | 1.32% | True | 1.38% / 1.32% |
  | deal_in | 4 | 42021 | 2.55% | 2.53% | True | 2.52% / 2.47% |
  | deal_in | 5 | 22670 | 3.48% | 3.75% | False | 3.46% / 3.71% |

  Tenpai is over-predicted in buckets 3-5 (by 1-3 points) and inside its interval in 1 of 6; deal-in is inside in 4 of 6 (the misses are buckets 0 and 5, off by under 0.3 points). Distilled network top-1 0.917 (old 0.924); matches inside noise (see above). Nothing else waits on this.

## From the race loop (Tasks 8-9, 2026-11-11)
- The race bot's chance that an opponent wins is 3–20× the observed rate in every wall bucket
  (hazard check flags all six for refit), and the bot wins only 0.2–0.3 % of hands: it is far
  too defensive. Four generations gain nothing measurable (gen 3 vs gen 2 +0.18 [-0.08, +0.44]);
  -9.4 vs three `value`. Fixing `hazard` before more loops is the likeliest gain. Details:
  `docs/study-results/race-bot.md` (Tasks 8 and 9).
