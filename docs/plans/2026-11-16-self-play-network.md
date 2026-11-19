# Plan: a self-play network that starts from the three-stage bot (2026-11-16)

Status: approved by Brian (words below). Written by the orchestrator from the Mahjong watch
session's hand-over (`C:\Users\brian\.orchestrator\mahjong-watch\handover-to-orchestrator.md`,
`plan-self-play-network.md`, `findings.md`). Those three files are the background; read them
before a task, never write to that folder.

## Goal
A network that first learns to copy the three-stage rule bot, then improves by playing itself
for up to 48 hours, and is measured against that bot. The three-stage bot itself moves from the
watch session's plain work copy into this repository first, so the work has a history.

## Approach
1. Bring the three-stage bot and its test tools into the lab, unchanged, and reproduce its
   measured strength digit for digit.
2. Give the network the full discard history as input (a new, separate input version; every
   existing model keeps loading with the old one).
3. Imitation: train the network on the three-stage bot's decisions.
4. Self-play against a pool: the current network, saved older networks and the three-stage bot.
5. Measure, report, and stop. Nothing goes into the game without Brian's word.

## Brian's decisions (verbatim, 2026-11-16)
- "let's try reinforcement learning. Use the current 3-stage solver as baseline, and let the
  solver evolve as we go. The solver should include full discard history, not just the last 3.
  You may use techniques such as including a previous checkpoint, older version randomly in
  games to compare their learning outcomes etc."
- "wrap up this session and send everything to the orchestrator, let it start working on the
  reinforcement learning with 3-stage bot as baseline, queued after the existing DouDiZhu task
  in the GPU lane"
- What evolves: "the network mainly"; "in general the rules number shouldn't be changed".
- Baseline: bot name `stage` (= `stage-pung`, `StageBot::new(3, 1)`), "ok" to that default:
  +2.54 +- 0.21 per hand against three value bots over 40,000 deals.
- Length: "ok" to 48 hours of self-play, a result every 2 hours, an early stop if it stalls.
- Where it runs: "Yes, merge first" (to the orchestrator's question "Merge the three-stage bot
  into the real mahjong lab first, then train there?").

## Readings that are not Brian's (the watch session's or the orchestrator's; he may change them)
- "Full discard history": every discard of every seat in order, each marked drawn-and-thrown or
  thrown-from-hand, and claimed or not.
- Opponent pool per hand: current network 50 %, a saved older network 30 %, the three-stage bot
  20 %.
- Pass mark for "the network is better": it scores above zero against three `stage` bots, with
  the whole 95 % range above zero, over 40,000 deals.
- "Stalls" (orchestrator's reading): four results in a row (8 hours) with no gain against three
  `stage` bots beyond the 95 % range of the best earlier result; or draws above 15 % of hands.
- Only the bot and its test tools are merged. The work copy's other changes (`crates/app-api`:
  `api.rs`, `solver.rs`, `tests/api.rs`, the `whatif`, `timeit`, `route_check` examples, `pkg`)
  stay out unless the bot does not build without them; whatever is left out is listed in
  `docs/open-items.md`.

## Constraints (from `docs/milestones.md` and `CLAUDE.md`, they hold for every task)
- GPU or long CPU jobs go through the job queue
  (`C:\Users\brian\OneDrive\Documents\workspace\repo\.claude\job-queue\jobq.ps1 add`), never
  started directly. A task that ends on a queued job is marked waiting on that job the usual way
  and handed back; the next worker finishes it when the job has ended.
- The graphics card runs one job at a time and Dou Dizhu's job 120 holds it until about
  18 Nov 00:11. Until then nothing of this plan uses `cuda`, not even a smoke run: use
  `--device cpu`.
- Never touch the standalone checkout `repo\hk-mahjong-lab`. Models are copied from
  `repo\hk-mahjong-lab\models\`, read-only. Build `mahjong._core` in your own checkout before a
  queue job runs from it.
- The watch session's folder `C:\Users\brian\.orchestrator\mahjong-watch\` is read-only.
- The `mahjong` conda env is shared by all workers and queue jobs. Never run
  `pip install -e` into it or into the base env from a worker checkout (2026-11-16: a worker
  repointed it at its own checkout). Build `_core` into your checkout, set `PYTHONPATH` to your
  checkout's `python` folder, call `C:\Users\brian\miniconda3\envs\mahjong\python.exe` by full path, and start
  every job script with a check that `mahjong.__file__` lies under the script's own checkout
  (exit 1 if not).
- The rule bot's numbers do not change in this plan.
- Nothing is put into the game (hk-games) by this plan.

## Acceptance criteria
1. `stage` plays in the lab and reproduces +2.54 +- 0.21 against three value bots on the watch
   session's 40,000 deals (Task 1).
2. A network input with the full discard history exists in Rust and Python and they agree
   (Task 3); every earlier model still loads and plays (Task 3).
3. An imitation network picks the three-stage bot's move on at least 90 % of held-out
   decisions and scores no worse than -1.0 per hand against three `stage` bots over 4,000 deals
   (Task 4).
4. The self-play job ran its 48 hours or stopped early by the stall rule, with a result every
   2 hours in `runs/rl-stage/log.json` (Task 6).
5. Brian has the final measurement against the pass mark, in plain numbers (Task 7).

## Phase 1 – The three-stage bot in the lab (sonnet)

- [x] **Task 1: bring the three-stage bot into the lab and reproduce its strength**
  - Files: from `C:\Users\brian\.orchestrator\mahjong-watch\lab\crates` copy or merge
    `search/src/stage.rs`, `search/src/route.rs`, the changes in `search/src/bots.rs`,
    `search/src/lib.rs`, `search/src/agent.rs`, `analysis/src/lib.rs`, `analysis/src/target.rs`,
    and the examples `search/examples/compare.rs`, `dealin.rs`, `stage_speed.rs`,
    `fit_routes.rs`, `race_offence.rs`; create `experiments/stage/versus.ps1` (from the watch
    session's `jobs\versus.ps1`, paths changed to this checkout);
    `docs/study-results/three-stage-bot.md` (a copy of the watch session's `findings.md`, with
    one line on top saying where it came from); `docs/open-items.md`.
  - Done when: `cargo test --workspace` passes; a cpu-lane queue job of
    `compare stage value value value 40000 1` from this checkout prints the same points per
    hand, win rate, deal-in and average faan as
    `C:\Users\brian\.orchestrator\mahjong-watch\runs\stage-pung-vs-value.txt` (+2.54 +- 0.21,
    25.1 %, 18.4 %, 5.1), digit for digit; `git diff --stat` shows no file under
    `crates/app-api`; and `docs/open-items.md` lists each work-copy difference left out.
  - Notes: diff each file against this checkout first (`diff -rq`, leaving out `target`). If a
    digit differs, the merge is wrong: find the missing change, do not adjust the bot. If the
    bot cannot build without an `app-api` change, stop and put the question in the inbox.

- [x] **Task 2: the Python binding knows the three-stage bot** — needs: Task 1
  - Files: `crates/pybind/src/rl.rs` (and `lib.rs` there if needed), `python/mahjong/rl/`
    only where a bot name is listed, `python/tests/` (one new test).
  - Done when: `pip install -e ".[dev]"` builds `mahjong._core` in the worker's checkout; a new
    pytest plays 200 hands with `stage` in three seats through the binding and passes; the
    teacher-data call returns decisions of `stage` (test asserts the logged move equals the
    bot's move on 1,000 decisions); and
    `python -m mahjong.rl.ppo_cnn --init none --minutes 3 --out runs/ppo/smoke --device cpu`
    finishes with exit code 0.

- [x] **Task 2b: lint and one failing model test, no behaviour change** `[parallel-ok]` — needs: Task 4 (job 155 ended exit 0, 16 Nov 23:43: `stage` from the old binary +2.54 +- 0.21, 25.1 %, 18.4 %, 5.1, read by the orchestrator: the check passes; fmt, clippy, cargo test and the model swap were done earlier; the env repoint stays left out; tick the task and write the board line)
  - Files: `crates/search/src/stage.rs`, `crates/search/examples/dealin.rs`,
    `crates/search/examples/fit_routes.rs`, `docs/open-items.md`; the worker clone's
    `models/race-belief.mjnn` (not in git).
  - Done when: `cargo clippy --workspace -- -D warnings` and `cargo fmt --check` pass;
    `cargo test --workspace` passes, `crates/search/tests/model_files.rs` included; and a
    cpu-lane job of `compare stage value value value 40000 1` still prints +2.54 +- 0.21,
    25.1 %, 18.4 %, 5.1, digit for digit.
  - Notes: found by the Task 4 worker, 2026-11-16. Clippy: `stage.rs:530` (`sort_by` to
    `sort_by_key`); fmt: the three files above. The model test fails in the worker clone
    because its `models/race-belief.mjnn` was written there by the race loop on 11 Nov and
    differs from the one in `repo/hk-mahjong-lab/models/`. Rename the clone's file to
    `race-belief.mjnn.loop-2026-11-11` (keep it), copy the standalone checkout's file in
    (read-only source), and note both md5 sums in `docs/open-items.md`. Do not touch any file
    a running queue job reads: check `jobq.ps1 status` first. Also, once and only while no
    mahjong queue job runs: point the shared `mahjong` env's editable install back at
    `hk-mahjong-lab-workers` (a worker left it pointing at `hk-mahjong-lab-3` on 16 Nov) and
    print `mahjong.__file__` from a neutral folder as proof.

## Phase 2 – Full discard history as input (opus)

- [x] **Task 3: a new input version with every discard in order** (opus) `[parallel-ok]` (input `history`, 3,460 floats; Rust and Python agree on 1,000 positions, largest output difference 1.2e-7; all 15 `.mjnn` and 3 CNN `.pt` files in `models/` give the same output as before)
  - Files: `crates/search/src/features.rs`, `crates/search/src/cnn.rs`,
    `python/mahjong/rl/cnn.py`, tests beside them, `docs/rl.md` (section 2).
  - Done when: `cargo test -p mahjong-search` and `pytest python/tests -k "features or cnn"`
    pass with new tests written first: (a) per seat, up to 24 discards in order, each with its
    tile, drawn-and-thrown or thrown-from-hand, and claimed or not; a 25th discard is handled
    by a rule the test states; (b) two positions that differ only in hidden tiles give the same
    input; (c) Rust and Python produce the same input and the same network output on 1,000
    positions (largest difference below 1e-5); (d) every model file in `models/` that loaded
    before still loads and gives the same output on 100 positions as before the change.
  - Notes: the new input is a separate version chosen by name or by a field in the model
    file; the old 933-float input stays as it is. The work copy's own `features.rs` and
    `cnn.rs` are not copied (Task 1 leaves them alone). Check the crate's real package name in
    `crates/search/Cargo.toml` before running the test command.

## Phase 3 – Imitation, then self-play (opus; graphics card through the queue)

- [x] **Task 4: imitation start** (opus) (checkpoint) — needs: Task 2, Task 3 (jobs 151 and 152 exit 0; held-out top-1 93.1 % ± 0.11; -0.24 ± 0.71 per hand vs three `stage` over 4,000 deals; both pass)
  - Status (slot 1, 2026-11-16 19:40): built and pushed in 42f828e (label layout 3 with the
    packed history, `distill_cnn --history`, both job scripts; tests pass). Queued from the
    worker clone `repo\.worktrees\hk-mahjong-lab-workers`: job 151 (cpu,
    `imitation_data.ps1`, 3 M rows into `runs\rl-stage\labels`) and job 152 (gpu, after 151,
    behind gpu job 150; `imitation.ps1`: training, then 4,000 deals vs three `stage`).
    Check: `jobq.ps1 status`, `jobq.ps1 log 152`; each log's first line prints
    `mahjong.__file__` (must lie under the worker clone). Results in that clone:
    `runs\rl-stage\imitation\result.json` (`held_out.top1_agreement`, `held_out.top1_ci95`)
    and `runs\rl-stage\imitation\vs-stage.txt` (first line = the network). Finish: fill the
    table in `docs/study-results/self-play-network.md`, board line, tick or inbox.
  - Files: `experiments/stage/imitation_data.ps1` (cpu lane), `experiments/stage/imitation.ps1`
    (gpu lane), `python/mahjong/rl/distill_cnn.py` only as far as the new input needs,
    `docs/study-results/self-play-network.md` (new), `docs/board.md`.
  - Done when: about 3 million decisions of `stage` are logged (cpu-lane job), split by hand
    into training and held-out sets; the gpu-lane training job has ended with exit code 0; the
    network picks the bot's move on at least 90 % of held-out decisions; a match of the network
    against three `stage` bots over 4,000 deals scores no worse than -1.0 per hand; both
    numbers with their 95 % ranges are in the study file and on the board. If either number
    misses, the task is not ticked: the result and one proposed next step go to the inbox.
  - Notes: add the gpu job with `jobq.ps1 add -Lane gpu`; it waits behind job 120 by itself.
    Estimate from the watch session: about 2 hours on the graphics card.

- [x] **Task 4b: imitation start again, on the strict bot** (checkpoint) `[parallel-ok]` — needs: Task 2b, Task 9 (jobs 165 and 166 ended exit 0, 17 Nov 00:48: held-out top-1 93.1 %, -0.11 +- 0.71 per hand against three strict `stage` over 4,000 deals, read by the orchestrator: both checks pass; write the results up, tick; old outputs are in `runs\rl-stage\labels-gate0`, `imitation-gate0`, `models\stage-imitation-gate0.*`; finish: read `runs\rl-stage\imitation\result.json` and `vs-stage.txt`, add the strict row to docs/study-results/self-play-network.md and the board)
  - Why: Brian, 2026-11-16: "use strict version as default" (Task 9), then, asked whether to
    hold the 48-hour run and redo its start on the strict bot at a cost of about 35 minutes
    of preparation: "that's fine". Task 4's network copies the old bot (now `stage-gate0`).
  - Do: in your own checkout, after Task 2b and Task 9 are ticked (their queue jobs use the old
    binaries, so never rebuild before): pull, rebuild the Rust binaries and `mahjong._core`,
    confirm `stage` is the strict bot. Keep Task 4's outputs under new names
    (`runs\rl-stage\labels-gate0`, `runs\rl-stage\imitation-gate0`,
    `models\stage-imitation-gate0.*`), then run Task 4's two jobs again unchanged:
    `imitation_data.ps1` (cpu lane, about 10 minutes) and `imitation.ps1` (gpu lane, after
    it, about 25 minutes).
  - Done when: both jobs ended with exit code 0 and each log's first line shows
    `mahjong.__file__` under your checkout; held-out agreement with the strict `stage` is at
    least 90 %; the network scores no worse than -1.0 per hand against three strict `stage`
    bots over 4,000 deals; both numbers with 95 % ranges are added to
    `docs/study-results/self-play-network.md` beside Task 4's (old bot) row and on the board.
    If either misses, do not tick: result and one proposed next step go to the inbox.
  - Notes: follow the shared conda env rule (PYTHONPATH, full python path, no `pip install -e`).
    Change no training setting. If Task 9's match shows the strict `stage` clearly weaker than
    `stage-gate0` against three value bots (ranges do not overlap), finish this task anyway
    and say so in the inbox: Brian decides the baseline.

- [x] **Task 5: the self-play job with a pool of opponents** (opus) `[parallel-ok]` — needs: Task 2, Task 3 (`ppo_cnn --pool`, `mahjong.rl.pool`, `experiments/stage/selfplay.ps1`; 20-min cpu dry run: 4 results, killed after the 2nd, resumed from ckpt 86 with the log unchanged, stopped at its time limit; `HkVecEnv.observe_passed()`/`observe_history()` feed Task 4's passed+history network)
  - Files: `python/mahjong/rl/ppo_cnn.py`, `python/mahjong/rl/pool.py` (new),
    `experiments/stage/selfplay.ps1` (gpu lane, resumable), tests in `python/tests/`.
  - Done when: tests written first pass for: opponents drawn per hand 50 % current network,
    30 % a random saved older network, 20 % `stage` (measured shares within 2 points over
    20,000 draws; with no older network saved yet its share goes to the current network); a
    network saved every 30 minutes and one kept in the pool per 2 hours; every 2 hours a result
    of 2,000 deals against three `stage` bots and against each earlier pool member appended to
    `runs/rl-stage/log.json` with points per hand, 95 % range, win rate, deal-in and draw
    share; a pull toward the imitation network that fades to zero by a set hour; a stop with a
    written reason when draws pass 15 % or when the stall rule of this plan's header holds; a
    stop at 48 hours of training time. And a 20-minute dry run with `--device cpu`, times scaled
    down (save every 2 minutes, result every 5), writes two results, is killed, restarted, and
    continues from the last saved network with the log intact.
  - Notes: the dry run starts from a random network if Task 4's network does not exist yet.
    Total budget is 48 hours of training time across restarts, not per start.
  - Result (slot 3, 2026-11-16): settings chosen by the worker (Brian may change them before
    Task 6 starts, never between restarts): pull `0.1 * KL(imitation || net)` on every decision,
    fading linearly to 0 by hour 24; the per-pattern anchor off (`--anchor-weight 0`); no
    min-faan curriculum; paired reward against the imitation network greedy in the same seat.
    The dry run started from `models/cnn-distill.pt` (a trained net without history; a random
    net trips the 15 % draw stop at its first result): results at 5/10/15/20 min against
    `stage` -0.36, +1.52, +1.96, -3.08 per hand (200 deals each, ranges about ±5); self-play
    draws 7-9 %. A tiny `distill_cnn --passed --history` net loads through `load_init` and
    gives the same output from env rows as from label rows (difference below 1e-5 once the
    env rows are rounded to float16 as label files store them; 1.1e-5 unrounded).
    For Task 6: build `_core` into the checkout that runs the job (no `pip install -e`), then
    `experiments\stage\selfplay.ps1 -Freeze`, then the queue line in that script's header
    (`-Init` defaults to `models\stage-imitation.pt`). The script exits 1 unless
    `mahjong.__file__` lies under its checkout.

- [x] **Task 5b: a stop on collapse, and the best network kept** `[parallel-ok]` — needs: Task 5
  - Files: `python/mahjong/rl/pool.py`, `python/mahjong/rl/ppo_cnn.py`, tests in `python/tests/`.
  - Done when: tests written first pass for: (a) the run stops with a written reason when two
    results in a row against three `stage` bots lie below the best earlier result by more than
    the sum of the two 95 % ranges; (b) the network of the best result so far is kept as
    `runs/rl-stage/best.pt` and a worse result never replaces it; (c) `log.json` names the best
    result and its file. A 10-minute scaled-down dry run with `--device cpu` shows `best.pt`
    written.
  - Notes: reason (orchestrator, 2026-11-16): Dou Dizhu's run 2c fell from -0.3 to -0.7
    against its benchmark at the step where its imitation pull reached zero, never recovered in
    115 M steps, and kept only its latest checkpoint. This run fades the same kind of pull by
    hour 24. If the Dou Dizhu study (`docs/study-results/phase2.md`, section "2c collapse", in
    the doudizhu repository) names a cause by then, say in the board line whether it applies
    here. Do not change the fade or other training settings in this task.

- [x] **Task 6: the 48-hour self-play run** — needs: Task 4, Task 5, Task 5b, Task 4b (done 2026-11-17: gpu job 167, checkout `hk-mahjong-lab-workers`, `runs/rl-stage/`, ended 10.00 h of 48 by its stall rule, exit 0; do not remove that checkout; result on the board)
  - CHANGE 2026-11-16 22:10: job 154 (queued from `hk-mahjong-lab-3`, never started) was
    cancelled by the orchestrator on Brian's word, because bot `stage` changed (Task 9,
    strict stage 3). Queue the run anew from your own checkout once Task 4b has passed: pull,
    build `mahjong._core` into the checkout, check `stage` is the strict bot (Task 9's unit
    test), run `selfplay.ps1 -Freeze` (remove an old `runs\rl-stage\pkg` first if the folder
    has no `state.json`), then add the gpu job with `-Init` = Task 4b's network. Mark the task
    waiting on the new job number and name the checkout it runs from.
  - Files: `runs/rl-stage/` (not in git), `docs/board.md`.
  - Done when: the gpu-lane job of `experiments/stage/selfplay.ps1`, started from Task 4's
    network, has ended by the 48-hour limit or by a stop rule; `runs/rl-stage/log.json` holds
    a result for every 2 hours it ran; the board has one line with the hours run, the reason it
    ended, and the best result against three `stage` bots with its range.
  - Notes: add the job, mark the task waiting on it and hand back. If the job dies, re-add it:
    it resumes. Do not change the training settings between restarts.

## Phase 4 – Measure and report (sonnet)

- [x] **Task 7: final measurement and report to Brian** (checkpoint) — needs: Task 6 (cpu job 182 ended 17 Nov 11:32, exit 0: 40,000 deals, network -0.09 +- 0.22 per hand against three `stage` bots (win 23.7 %, deal-in 18.3 %, 5.1 faan), so the pass mark is not met; outputs in `runs/rl-stage/final/` (`vs-stage-40000.txt`, `table-4000.txt`, `dealin.txt`). Finish: add best-network numbers (best.pt = hour 6, iteration 797; best.mjnn = its int8 export), the 2-hourly curve (log.json: -0.26, +0.31, +0.48, -0.54, -0.88, each +- 2.0, win 22-24 %, deal-in 17-18 %), the pass-mark sentence to docs/study-results/self-play-network.md, post the inbox question, board line, tick)
  - Files: `experiments/stage/final.ps1` (cpu lane), `docs/study-results/self-play-network.md`,
    `docs/inbox.md`, `docs/board.md`.
  - Done when: the study file holds, for the best saved network: 40,000 deals against three
    `stage` bots (points per hand, 95 % range, win rate, deal-in, average faan); 4,000 deals at
    one table with the teacher, `stage` and the value bot; the deal-in breakdown from
    `dealin`; the 2-hourly curve as a table; and one sentence saying whether the pass mark is
    met. An inbox question gives Brian these numbers and asks what happens next (keep training,
    put it into the game, or stop). Nothing is copied into hk-games.

- [x] **Task 8: end-of-run review** (fable: end-of-run review) — needs: Task 7 (done 2026-11-17: review in `docs/study-results/review-self-play-network.md`; one doc figure fixed, job-script path checks hardened, docs name the bot, the input and the pool; judgment calls in open-items; tests and lint pass)
  - Done when: the review of this plan's commits is in `docs/study-results/` or the repo's
    review folder, plain bugs are fixed and verified, judgment calls are in
    `docs/open-items.md`, and `docs/architecture.md` and `docs/rl.md` name the three-stage bot,
    the new input version and the pool.

## Phase 5 – One low-noise run: can self-play show a gain at all (opus)

- [ ] **Task 10: a 4-hour self-play run with less noise and an accurate hourly score** (opus) (checkpoint) `[parallel-ok]` (waiting: gpu job 252 `rl-lownoise`, started 2026-11-19 14:44, 4 h, first log line names `runs/rl-lownoise/pkg`; then cpu job 253 `rl-lownoise-final` (-After 252); the cancelled run's folder is kept as `runs/rl-lownoise-job189`; queued by slot 1, 2026-11-19) (RESUMED 2026-11-19 on Brian's word "sure, go ahead and train and experiment and see if it improves rather than getting worse": queue a fresh 4 h run now, gpu lane, job name `rl-lownoise`, then the 40,000-deal match in the cpu lane, and tell the orchestrator the job ids so they go ahead of other projects' jobs; history: paused 2026-11-17 14:36 on his word "what mahjong test run? please pause and explain what it is": the orchestrator cancelled gpu job 189 `rl-lownoise` 28 minutes into its 4 h, so cpu job 191 did not run; code, tests and job script done by slot 1 in 2021de9 and the next commit: `--no-pull-fade`, `--no-stall`, `--score-minutes`, hourly rows in `runs/rl-lownoise/hourly.txt`; frozen copy `runs/rl-lownoise/pkg`, first log line checked; check: `jobq.ps1 status`, `runs/rl-lownoise/hourly.txt`, `runs/rl-lownoise/stopped.txt`, `runs/rl-lownoise/final/vs-stage-40000.txt`; left: the study section's table and sentence, the inbox question, the tick)
  - Why: (Brian, 2026-11-17: "yes sure, go ahead with the 5 test runs, higher priority than big2"). The analysis `docs/study-results/2026-11-17-self-play-analysis.md` (faults M1, M2; experiment 4): the 10-hour run was held at its start by the fading pull, its stop rule needed a jump of 2 points a hand, and its training score (learner minus the frozen start on the same tiles) stayed flat at about -0.7.
  - What: same start network and pool as Task 6. Changes, each a setting with a test where code changes: the pull toward the start constant at 0.02 (no fade); 4,096 hands per update; one pass per update; learning rate 3e-5; the stall rule off for this run (the collapse rule stays); every hour print and store the mean paired difference (learner minus frozen start, points a hand) with its 95 % range over that hour's hands, plus entropy and distance from the start. Run 4 hours through `jobq.ps1` (gpu lane, job name `rl-lownoise`), from a frozen code copy as Task 6 did; check `mahjong.__file__` in its first log line.
  - Files: `python/mahjong/rl/ppo_cnn.py`, `python/mahjong/rl/pool.py`, their tests, a new job script beside `selfplay.ps1`, `docs/study-results/self-play-network.md` (new section "Low-noise run").
  - Done when: the section holds the hourly table (the start is about -0.7) and one sentence on its meaning: a rise to -0.3 or better = the signal is there once noise is cut; flat = self-play at this speed will not lift this network. Then a 40,000-deal match of the end network against three strict `stage` bots (cpu lane) beside the earlier -0.09 +- 0.22. Post the result as an inbox question for Brian (train long with per-decision credit, or stop). Tests pass.
  - Queue: this job runs ahead of big2's gpu jobs (Brian's words above); it shares the gpu lane with the Dou Dizhu test jobs (`ddz-12*`), one at a time, in the order they were added.

## Phase 6 – A give-up rule for the ready hand (opus)

- [x] **Task 11: the three-stage bot gives up a ready hand when the tile is too dangerous (test bots, measured)** (opus) (checkpoint) `[parallel-ok]` (done by slot 1, 2026-11-19: step 1 on 2 × 40,000 deals: `Danger::tile` 1.1-3.6× too low on stage 3 discards, recalibrated (9 of 10 bins within 0.83-1.07 on fresh deals); matches 40,000 deals, run directly on the orchestrator's word: `stage-fold` -0.63 ± 0.17 per hand vs three `stage` (half -0.85, double -0.51, each ± 0.17), +1.64 ± 0.19 vs three `value`; no gain; report `docs/study-results/stage3-giveup.md`; fmt, clippy, cargo test pass)
  - Why: Brian, 2026-11-18: "for me realistically the task left is the defense part of the mahjong solver as well as a doudizhu solve with playable speed". On weighing one discard at a time: "but I think that's not right. If you go offensive and do not deal in it's through once and for all", "so it's not trivial". 2026-11-19: "can you calculate like the stage 3, let's say the probability of deal in is accurate, then at what point one needs to give up given a bunch of scenarios"; on the rule that came out: "ok, so if we equip this to the stage bot, what happens? can you try". Stage 3 gives 12.4 of the bot's 18.6 deal-ins per 100 hands.
  - The rule (orchestrator's rough model; the worker may correct the formula's parts but not its shape without saying so). When the bot is in stage 3 and ready, for its chosen discard `t`: push when `(1 - d) × V - loss(t) > -loss(s)`, else throw `s`, where `d` = the chance `t` deals in (`Danger::tile`), `loss(t)` = `Danger::expected_loss(t)` (chance times that opponent's points, summed over opponents), `s` = the tile in hand with the smallest expected loss, and `V` = what staying ready is worth over the rest of the hand, in points: summed over the rounds left, (chance I draw a winning tile × 3 shares + chance an opponent throws one × 1 share − the average expected loss of one later discard), each round weighted by the chance the hand is still going. With one threatening opponent this is "give up when d > V ÷ (V + L)". After a give-up the hand keeps throwing its smallest-loss tile until it is ready again without having thrown a tile the rule refused.
  - Step 1, measure before building (cpu lane): from 40,000 deals of four `stage` bots log, for ready hands, per round left: the chance per round that I win by my own draw and by a discard (against live winning tiles ÷ unseen tiles, so the "opponents throw my tile at random" factor becomes a measured one), the chance per round another player ends the hand, and the average expected loss of a later discard. Check `Danger::tile` against what happened: ten bins of predicted chance against the observed deal-in rate for stage 3 discards (`examples/danger_check.rs` is the starting point). If a bin is off by more than a factor 1.5, refit or rescale first and say so (the 9 Nov race bot folded far too often because its danger was 3 to 20 times too high).
  - Step 2, build: a switch on `StageBot`, off by default so `stage` plays exactly as today. Bot names `stage-fold` (the rule with step 1's measured figures), `stage-fold-half` and `stage-fold-double` (the push side's `V` times 0.5 and 2, to show how sensitive the result is).
  - Tests first: with the switch off the existing stage tests give the same moves; a ready 3-faan hand with 2 live winning tiles against an opponent showing a large hand throws the safest tile when its chosen tile's danger is above the break-even and pushes when below; an 8-faan ready hand pushes the same tile.
  - Step 3, measure (cpu lane, queue and end the run): `compare <bot> stage stage stage 40000 1` for the three bots, and `compare stage-fold value value value 40000 1`; per bot: points per hand with its 95 % range, win rate, deal-in rate, points paid per deal-in, share of ready hands that gave up at least once, and deal-ins per 100 hands in stage 3.
  - Files: `crates/search/src/stage.rs` (the switch), a new `crates/search/src/giveup.rs`, `crates/search/examples/` (the step 1 log and the match switch), tests, `docs/study-results/stage3-giveup.md`.
  - Done when: `cargo test --workspace`, fmt and clippy pass; `docs/study-results/stage3-giveup.md` has step 1's measured figures and the danger check table, the break-even table (my hand 3, 5, 8 faan × 2, 4, 8 winning tiles × their hand 3, 5, 8, 10 faan, at 6 and 2 rounds left) recomputed with the measured figures beside the orchestrator's rough one, and step 3's result table with one line: does the rule gain points per hand over `stage` (95 % range clear of 0), yes or no; pushed. Nothing becomes the default bot: Brian decides after the numbers.
  - Notes: build with `CARGO_TARGET_DIR` set to a scratch folder outside every checkout and run job binaries from there; never rebuild `target\release` or `_core.pyd` inside a checkout. The orchestrator's rough table (assumed 2 % danger per later discard, 6 % per round that another player ends the hand, opponents throwing my tile at random): 6 rounds left, give up above 16 % (3 faan, 2 tiles, against 5 faan), 7 % (against 8 faan), 52 % (5 faan, 4 tiles, against 5 faan), 29 % (against 8 faan), 53 % (8 faan, 4 tiles, against 8 faan).

## Phase 7 – Push or fold learned from played-out hands (fable)

- [ ] **Task 12: the ready hand's push-or-fold choice learned from play-outs, measured against the plain bot** (fable: label and model design open) (checkpoint) `[parallel-ok]` (taken: slot 2, 2026-11-19 14:44)
  - Why: Task 11's rule lost 0.63 ± 0.17 points a hand. Brian, 2026-11-19: "so the setup is incorrect just as I suspected. How to improve the give-up role"; "you compared offensive EV to the deal-in EV. realistically the opponent if ready also should have their own offensive EV. I think that's the part that has not been taken into account, overestimating the deal-in EV as it should only be the difference between the deal-in EV and the opponent offensive EV."; "and I assume the model somewhat estimates the parameters with its network?"; "sure, go ahead and train and experiment and see if it improves rather than getting worse". This is decision B15 (`docs/ev-bot.md`): defence counts what the opponents would win anyway, their self-draws included. Task 11's rule (`docs/study-results/stage3-giveup.md`, "The rule") set the fold side to `-loss(s)` only and built `V` from a formula.
  - What: no formula for the parts. Play both choices out and learn the difference.
    1. Labels (cpu): from deals of four `stage` bots, at every stage 3 decision where the chosen discard `t` leaves the hand ready and a tile with a smaller expected loss exists: play the rest of the hand out many times with `t` thrown (push) and many times with the smallest-loss tile `s` thrown (fold; afterwards the hand keeps throwing its smallest-loss tile until a discard leaves it ready again). Both sides use the same re-dealt hidden tiles (opponents' concealed tiles and the wall drawn again from what this seat cannot see, the same draws for push and for fold), all four seats playing `stage` after the first discard. Label = mean of (my points after push − my points after fold). An opponent's self-draw, a third player's deal-in and a drawn hand all land in the label by themselves. You choose the number of positions and of play-outs per position; report the label's standard error.
    2. First read the labels without any model: mean label by danger bin of `t` (calibrated, from Task 11) × my hand's faan × winning tiles left × rounds left × size of the visible threat, with the share of positions where fold is better. If fold is better in under 2 % of positions, say so: then no rule of this kind can gain much and step 3 is a short check, not a build.
    3. Model: fit a predictor of the label from what the seat can see (the bot's own figures: `t`'s and `s`'s calibrated danger and loss, my faan, live winning tiles, rounds left; per opponent: melds shown, discards made, whether its belief says ready, its expected faan). Your choice between a small network and a fitted table or tree; say why. Held-out check: mean label of the positions the model says fold, with its 95 % range (it must be below 0).
    4. Bot `stage-fold-learned`: `stage`, except that at these decisions it throws `s` when the model predicts push − fold below 0. The switch is off by default, so `stage` plays exactly as today.
    5. Measure: `compare stage-fold-learned stage stage stage 40000 1` and the same against three `value` bots; per bot: points per hand with its 95 % range, win rate, deal-in rate, points paid per deal-in, share of ready hands that gave up at least once. Beside them: plain `stage` and Task 11's `stage-fold`.
  - Files: a new `crates/search/src/pushfold.rs` and its example under `crates/search/examples/`, `crates/search/src/stage.rs` (the switch only), tests, a fitting script under `python/mahjong/` or `experiments/stage/` if the model is fitted in Python (the bot reads the fitted file in Rust), `docs/study-results/stage3-pushfold.md`. Do not edit `giveup.rs` beyond reading its calibration.
  - Tests first: with the switch off the existing stage tests give the same moves; push and fold play-outs from one position use identical hidden tiles (same seed gives the same wall for both); a position where every play-out ends the same for both choices gives label 0.
  - Done when: `cargo test --workspace`, fmt and clippy pass; `docs/study-results/stage3-pushfold.md` holds step 2's table, the held-out check, step 5's result table and one line: does `stage-fold-learned` gain points per hand over `stage` (95 % range clear of 0), is it level, or does it lose; pushed. Nothing becomes the default bot: Brian decides after the numbers.
  - Notes: build with `CARGO_TARGET_DIR` set to a scratch folder outside every checkout and run binaries from there; never rebuild `target\release` or `_core.pyd` inside a checkout. A run under 2 hours may run directly, as Task 11's did; a longer one goes through `jobq.ps1` (cpu lane; gpu lane only for a network fit), and you tell the orchestrator the job id so it is placed ahead of other projects' jobs (Brian, 2026-11-19: "mahjong and doudizhu always get priority"). Task 10 may be running in the gpu lane at the same time; it shares no files with this task.

- [ ] **Task 13: exact danger by counting when an opponent has few concealed tiles** (opus) (checkpoint) `[parallel-ok]` (taken: slot 1, 2026-11-19 15:45)
  - CHANGE 2026-11-19 (orchestrator's call on slot 1's question; Brian may change it; overrides the lines below where they differ): plain `stage` never reads `Danger` (only the give-up rule does), so there is no `stage-exact` bot. The switch goes on the give-up bot: `stage-fold-exact` = `stage-fold` with the exact switch on (a thread-local flag set from `bots.rs`). Matches: `stage-fold-exact` vs three `stage` and vs three `value`, beside `stage-fold` and plain `stage`. Verdict line: `stage-fold-exact` minus `stage-fold` (same seeds, with its 95 % range) is what exact counting adds; also say whether it gains over plain `stage`. Also report how many of the bot's decisions the exact figure changed.
  - Why: the danger figure comes from fitted tables, so it averages over similar-looking situations and its top bin deals in 73 % of the time (`docs/study-results/stage3-giveup.md`, danger check), even when the tile is certain. Brian, 2026-11-19: "is there scenario where the deal-in chance is 100%, across your simulations and scenarios encountered?"; "but it should be possible to reach 100%. let's say full flush all pung 12 exposed, you can count the available tiles and sometimes it just boil down to a single option"; "how to fix that?"; on the exact counting step: "Yes, add it".
  - What: a function that, for one opponent seen from one seat, lists every concealed hand that opponent could hold and from it the tiles that can win. A concealed hand is possible only if (a) its tiles are among those this seat cannot see (its own hand, all discards, all exposed sets and flowers counted out), (b) with the exposed sets it forms a ready hand, (c) the win on that tile by discard reaches the faan minimum under the engine's own scoring (the opponent's flowers and seat/round winds included), (d) it does not win on a tile that opponent let pass since its last draw where the engine's rules make that a refused win; state in the report which pass rule the engine has.
    1. One concealed tile (four sets exposed): the opponent is always ready. Danger of tile `k` = (unseen copies of `k` that it could be holding, as weight) ÷ (sum of weights over all kinds that pass a to d). One kind left gives exactly 1.0; a kind that fails gives exactly 0.0.
    2. Four concealed tiles (three sets exposed): enumerate the possible four-tile hands. Tiles that win for none of them get exactly 0.0. For the others keep the table's chance that the opponent is ready and spread it over the possible ready hands by their weight (the number of ways to hold them from the unseen tiles); say in the report how this compares with the table alone.
    3. More than four concealed tiles: no change, the table as today.
    `Danger` uses the exact figure where it exists, behind a switch that is off by default, so `stage` plays exactly as today. Bot name `stage-exact` (the switch on, nothing else changed). The points paid follow the same list (the faan of each possible hand), replacing the table's guess for that opponent.
  - Tests first: Brian's case, an opponent with four exposed pungs of one suit and no other faan, where the visible tiles leave one kind: that kind 1.0, every other tile 0.0; the same with two kinds left and 2 and 1 unseen copies: 2/3 and 1/3; a kind with all other copies visible is 0.0; a hand that only reaches the minimum as a flush gives 0.0 to every tile outside the suit; three exposed sets: a tile no possible hand wins on is 0.0; with the switch off every existing danger and stage test gives the same numbers.
  - Measure (cpu, runs under 2 hours may run directly; longer through `jobq.ps1` and tell the orchestrator the job id): (1) on 40,000 deals of four `stage` bots: how many discards face an opponent with one concealed tile and with four, and for those the danger check table (ten bins, predicted against observed) for the table figure and for the exact figure side by side; count the discards given 1.0 and 0.0 and how many of each dealt in (1.0 must deal in every time when thrown, 0.0 never: any miss is a bug, find it). (2) `compare stage-exact stage stage stage 40000 1` and against three `value`: points per hand with its 95 % range, win rate, deal-in rate, points paid per deal-in.
  - Files: a new `crates/search/src/exact_wait.rs`, `crates/search/src/danger.rs` (the switch and the call only), `crates/search/src/bots.rs` (the bot name), `crates/search/examples/danger_check.rs`, tests, `docs/study-results/exact-danger.md`. Task 12 (slot 2) is building `pushfold.rs` and touches `stage.rs`: do not edit those; it reads `Danger`, so keep `Danger`'s public functions unchanged.
  - Done when: `cargo test --workspace`, fmt and clippy pass; `docs/study-results/exact-danger.md` holds the counts, both danger check tables, the 1.0 and 0.0 check with zero misses, the match table and one line: does `stage-exact` gain points per hand over `stage` (95 % range clear of 0), is it level, or does it lose; pushed. Nothing becomes the default: Brian decides after the numbers. Tell the orchestrator at the end, so Task 12's learned choice can be run again with the exact figure as its input.
  - Notes: build with `CARGO_TARGET_DIR` set to a scratch folder outside every checkout; never rebuild `target\release` or `_core.pyd` inside a checkout. A concealed kong hides its tiles: treat them as the rules show them to the other seats.

## Shared files and safety
- Orchestrator's calls on worker questions, 2026-11-16 (Brian may change them): Task 5 adds one
  additive method `HkVecEnv.observe_history()` to `crates/pybind/src/rl.rs` and owns it; Task 4
  adds label layout 3 (layout 2 plus `history`) in `crates/search/src/labels.rs` and
  `examples/label.rs` and does not touch `crates/pybind`; the opponent pool draws each of the
  three other seats independently every hand (50 / 30 / 20), so tables are mixed.
- `crates/search/src/bots.rs`, `lib.rs`, `agent.rs`: Task 1 only. `features.rs`, `cnn.rs`,
  `cnn.py`: Task 3 only. `ppo_cnn.py`: Task 5 only. `distill_cnn.py`: Task 4 only.
  `crates/pybind`: Task 2 only, except one additive function `encode_history` (and its
  registration line) that Task 3 adds for its Rust-against-Python check; Task 3 may also touch
  `crates/search/src/net.rs` so a model file can name the new input (orchestrator's call,
  2026-11-16, on the Task 3 worker's question).
- Tasks 1 and 3 run side by side in separate worktrees; each builds its own `target` folder.
- Queue lanes: cpu for matches and data, gpu for the two training jobs. Run folders:
  `runs/rl-stage/` and `runs/ppo/smoke` only.

- [x] **Task 9: stage 3 only without a tile outside the leading line (two test bots)** (opus) `[parallel-ok]` (all jobs ended exit 0, 16 Nov; job 161 failed on a queue fault before it ran and was repeated as job 163; outputs `runs/stage/t9-*.txt` in the checkout `hk-mahjong-lab-workers`: `stage` +2.62 +- 0.21, `stage-gate0` +2.54 +- 0.21 (identical to the old bot), `stage-gate2` +2.62 +- 0.21, table `stage` -0.19, `stage-gate0` +0.03 and +0.16, `stage-gate2` -0.01, each +- 0.22, stage counts in `t9-stage-counts.txt`; read by the orchestrator. Done by slot 2, 2026-11-17: tables, stage counts and verdict (no gate measurably better; gate 1 not clearly weaker than gate 0) in `docs/study-results/stage3-gate.md`; fmt, clippy, cargo test pass. The append to the watch session's findings.md was refused by the permission check and left out (orchestrator's call); the full tables are in the study file)
  - Why: Brian, 2026-11-16, to the Mahjong watch session: "then make phase 3 the extra
    requirement that is found in phase 2 as well, no irrelevant tile for top priority route".
    Full brief (read it first): `C:\Users\brian\.orchestrator\mahjong-watch\brief-stage3-gate.md`. Today
    `11123456789s1m5p` enters stage 3 while it still holds 1m and 5p.
  - CHANGE 2026-11-16 21:40 (overrides the lines below where they differ). Brian, to the
    Mahjong watch session: "use strict version as default". So bot `stage`,
    `StageBot::default()` and `StageBot::new(3, 1)` use the gate 1 rule; today's behaviour
    stays as bot `stage-gate0`; `stage-gate2` as below. Matches: `stage` (new),
    `stage-gate0` (must reproduce +2.54 +- 0.21 digit for digit) and `stage-gate2`, each
    against three value bots, and one table `stage stage-gate0 stage-gate2 stage-gate0`.
    Tests that pin the old moves move to `stage-gate0`. Job 154 runs from a frozen package
    and keeps the old bot; whether it is redone on the new bot is Brian's decision (asked).
  - Files: `crates/search/src/stage.rs`, the bot name list (`bots.rs`), tests beside them,
    `experiments/stage/` for the job scripts, `docs/study-results/stage3-gate.md` (new).
  - Build: add a third switch to `StageBot`. Gate 0 = today's rule, and `StageBot::new(3, 1)`,
    `StageBot::default()` and the bot name `stage` keep today's behaviour exactly (the 48-hour
    run, job 154, plays against `stage`). Gate 1 (Brian's words read strictly): stage 3 only
    when the leading-line check finds no tile outside the line (the same `off` check stage 2
    uses); a hand with an outside tile plays stage 1 even when ready. Gate 2 (the watch
    session's addition for comparison, not Brian's words): like gate 1, but an already ready
    hand stays in stage 3. Apply the gate to the stage 3 claim decision as well as the discard.
    For leads other than the flush there is no "outside tile" notion, so stage 3 starts as
    today (the watch session's reading; say so in the report). Bot names `stage-gate1` =
    new(3,1,1), `stage-gate2` = new(3,1,2).
  - Tests first: the example hand does not enter stage 3 under gate 1; a ready hand with one
    outside tile is in stage 1 under gate 1 and in stage 3 under gate 2; `stage` gives the
    same moves as before on the existing stage tests.
  - Measure (cpu lane, about 5 minutes each, queue them and end the run; they start once the
    quiet job ahead is over): (1) `compare stage-gate1 value value value 40000 1`; (2) the
    same for `stage-gate2`; (3) one table `stage stage-gate1 stage-gate2 stage`, 40,000 deals;
    (4) discards per hand in stage 1/2/3 for gate 0, 1 and 2 (`examples/dealin.rs` reads
    `StageBot::default()`: give it a switch rather than changing the default).
  - Done when: `cargo test --workspace`, fmt and clippy pass; the three result tables and the
    stage counts are in `docs/study-results/stage3-gate.md` beside the baseline (`stage`
    against three value bots +2.54 +- 0.21, win 25.1 %, deal-in 18.4 %), with one line on
    which gate looks better; the same tables are appended to
    `C:\Users\brian\.orchestrator\mahjong-watch\findings.md` under a dated heading (append only, change
    nothing else in that folder); pushed.
  - Notes: build with `CARGO_TARGET_DIR` set to a scratch folder outside every checkout and
    run the job binaries from there. Never rebuild `target\release` or `_core.pyd` inside a
    checkout, and change nothing in `hk-mahjong-lab-3` (job 154 runs from it) while jobs 154
    and 155 are queued or running. The brief's "no git commands" rule was for the watch
    session's work copy; here commit and push as usual. Nothing becomes the default bot:
    Brian decides after reading the numbers (the orchestrator tells him).
