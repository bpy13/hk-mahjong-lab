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

- [ ] **Task 2b: lint and one failing model test, no behaviour change** `[parallel-ok]` — needs: Task 4
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

- [ ] **Task 5b: a stop on collapse, and the best network kept** `[parallel-ok]` — needs: Task 5 (taken: slot 1, 2026-11-16 12:00)
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

- [ ] **Task 6: the 48-hour self-play run** — needs: Task 4, Task 5, Task 5b
  - Files: `runs/rl-stage/` (not in git), `docs/board.md`.
  - Done when: the gpu-lane job of `experiments/stage/selfplay.ps1`, started from Task 4's
    network, has ended by the 48-hour limit or by a stop rule; `runs/rl-stage/log.json` holds
    a result for every 2 hours it ran; the board has one line with the hours run, the reason it
    ended, and the best result against three `stage` bots with its range.
  - Notes: add the job, mark the task waiting on it and hand back. If the job dies, re-add it:
    it resumes. Do not change the training settings between restarts.

## Phase 4 – Measure and report (sonnet)

- [ ] **Task 7: final measurement and report to Brian** (checkpoint) — needs: Task 6
  - Files: `experiments/stage/final.ps1` (cpu lane), `docs/study-results/self-play-network.md`,
    `docs/inbox.md`, `docs/board.md`.
  - Done when: the study file holds, for the best saved network: 40,000 deals against three
    `stage` bots (points per hand, 95 % range, win rate, deal-in, average faan); 4,000 deals at
    one table with the teacher, `stage` and the value bot; the deal-in breakdown from
    `dealin`; the 2-hourly curve as a table; and one sentence saying whether the pass mark is
    met. An inbox question gives Brian these numbers and asks what happens next (keep training,
    put it into the game, or stop). Nothing is copied into hk-games.

- [ ] **Task 8: end-of-run review** (fable: end-of-run review) — needs: Task 7
  - Done when: the review of this plan's commits is in `docs/study-results/` or the repo's
    review folder, plain bugs are fixed and verified, judgment calls are in
    `docs/open-items.md`, and `docs/architecture.md` and `docs/rl.md` name the three-stage bot,
    the new input version and the pool.

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
