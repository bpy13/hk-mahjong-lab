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

- [ ] **Task 2: the Python binding knows the three-stage bot** — needs: Task 1 (taken: slot 1, 2026-11-16 19:00)
  - Files: `crates/pybind/src/rl.rs` (and `lib.rs` there if needed), `python/mahjong/rl/`
    only where a bot name is listed, `python/tests/` (one new test).
  - Done when: `pip install -e ".[dev]"` builds `mahjong._core` in the worker's checkout; a new
    pytest plays 200 hands with `stage` in three seats through the binding and passes; the
    teacher-data call returns decisions of `stage` (test asserts the logged move equals the
    bot's move on 1,000 decisions); and
    `python -m mahjong.rl.ppo_cnn --init none --minutes 3 --out runs/ppo/smoke --device cpu`
    finishes with exit code 0.

## Phase 2 – Full discard history as input (opus)

- [ ] **Task 3: a new input version with every discard in order** (opus) `[parallel-ok]` (taken: slot 2, 2026-11-16 18:39)
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

- [ ] **Task 4: imitation start** (opus) (checkpoint) — needs: Task 2, Task 3
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

- [ ] **Task 5: the self-play job with a pool of opponents** (opus) `[parallel-ok]` — needs: Task 2, Task 3
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

- [ ] **Task 6: the 48-hour self-play run** — needs: Task 4, Task 5
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
- `crates/search/src/bots.rs`, `lib.rs`, `agent.rs`: Task 1 only. `features.rs`, `cnn.rs`,
  `cnn.py`: Task 3 only. `ppo_cnn.py`: Task 5 only. `distill_cnn.py`: Task 4 only.
  `crates/pybind`: Task 2 only, except one additive function `encode_history` (and its
  registration line) that Task 3 adds for its Rust-against-Python check; Task 3 may also touch
  `crates/search/src/net.rs` so a model file can name the new input (orchestrator's call,
  2026-11-16, on the Task 3 worker's question).
- Tasks 1 and 3 run side by side in separate worktrees; each builds its own `target` folder.
- Queue lanes: cpu for matches and data, gpu for the two training jobs. Run folders:
  `runs/rl-stage/` and `runs/ppo/smoke` only.
