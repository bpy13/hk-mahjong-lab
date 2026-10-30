# Plan: the mahjong bot build run (B25, HANDOFF item 1)

Repo: `C:\Users\brian\OneDrive\Documents\workspace\repo\hk-mahjong-lab` (standalone checkout,
branch `main`; `git pull --rebase` before every session). Design sources:
`docs/research/strategy-proposal.md` (§2 method, §5 phases B–C, §6 measures, §7 compute, §9
risks), `docs/research/hard-problems.md` (ideas 1a–8e), the bot-stream decisions
`C:\Users\brian\OneDrive\Documents\workspace\repo\bot-stream\docs\decisions.md` (B1–B25) and
`C:\Users\brian\OneDrive\Documents\workspace\repo\bot-stream\docs\plans\HANDOFF.md`.
The app's side of the engine is `crates/app-api/` (app stream, `crates/app-api/README.md`).

## Goal

A stronger Hong Kong mahjong bot, measured, in two designs built side by side:

1. **Brian's explicit EV bot** (B15): each move valued as offence EV minus defence EV loss,
   with opponents' waits and hand values read from their discards and melds.
2. **The proposal's path**: a belief model → a better PIMC teacher → a network copied from it
   (expert iteration) → PPO self-play → the network inside the search.

Both rest on one **shared belief model** (opponents' tenpai chance, waiting tiles, hand value,
hidden tile counts, from public information). The run ends with a duplicate match between the
two designs on the same walls, all 24 seatings, and the tooling for Brian's 400-hand human
benchmark (B21). Single hands only (B21); no matches with dealer retention.

What the app already has (bot-notes 2026-10-29): `bot_action`, `hint`, `review`, `grade`,
WebAssembly, reason keys. So the proposal's Phase A is **not** in this plan. The app stream's
open request to us, `State::from_position`, is Task 2.

## Approach and order

CPU work first, GPU work last: big2 holds the GPU for about 3 GPU-days before mahjong's turn.

| Phase | What | Needs the GPU | Machine time |
|---|---|---|---|
| 0 | Environment, `from_position`, the 24-seating harness, self-play logs with hidden hands | no | ~3 h CPU |
| 1 | Belief model, payment-aware danger, belief-driven PIMC sampling | small net: CPU (≤ 6 h) or 3 GPU-h | ~1 CPU-day |
| 2 | Brian's EV bot, target value table, EV bot as PIMC anchor | no | ~1 CPU-day |
| 3 | Human benchmark tooling: fixed walls, logging, replay, label export | no | hours |
| 4 | PIMC labels (1 M decisions), suit-shared CNN, distillation, network inside search | distillation 2 GPU-h | ~3 CPU-days + 0.1 GPU-day |
| 5 | PPO self-play with the proposal's additions; main run, second seed, two ablations | yes | 3.5 GPU-days |
| 6 | The comparison (EV bot vs network + search, 24 seatings), docs, app handover | no | ~1.5 CPU-days |

Totals: about **6.5 CPU-days** (labelling and matches, in the background) and about
**3.9 GPU-days** (0.15 belief if on GPU, 0.2 distillation, 3.5 PPO). The proposal's 8 GPU-days
for mahjong also covered Phase D (exploiter ladder, style pool, drill mining), which HANDOFF
item 1 does not list; those stay for a later run (open question 7).

Every job over ~15 minutes: runs headless (`conhost.exe --headless powershell.exe -NoProfile
-File <script>`), writes its own results file under `runs/<job>/`, saves progress or a
checkpoint at least every 30 minutes, resumes from it, and has a stated check command. Jobs
over 12 hours are announced before they start (proposal §7).

## Baseline on this PC (measured 2026-10-29)

- The conda env `mahjong` exists but `make dev` was **never run on this PC**: no `mahjong`
  package, no `pytest`, `ruff`, `mypy`, `torch`. The env's own `cargo`
  (`envs\mahjong\Library\bin\cargo.exe`, GNU toolchain, rustc 1.97.1) cannot link on Windows
  (`dlltool.exe` not found). rustup's MSVC toolchain (`%USERPROFILE%\.cargo\bin`, rustc 1.99.1,
  wasm32 target installed) is what the app stream used and what this plan uses: put
  `C:\Users\brian\.cargo\bin` first on `PATH` and set `PYO3_PYTHON` to the env's python.
- `cargo fmt --all --check`: passes (3 s). `cargo clippy --workspace --all-targets -- -D warnings`:
  passes (20 s, `dev` profile already cached).
- `cargo test --workspace` (rustup MSVC, `PYO3_PYTHON` set): **passes**, 112 s including a
  20 s compile; every suite `ok` (hk 14 tests + scenarios, analysis, search, app-api 14 + 1
  ignored timing test, pybind, tiles); the slowest suites take 20–30 s each.
- `pytest`, `ruff`, `mypy`, `make test-web`: not runnable until Task 0 (nothing installed).
- **After Task 0 (measured 2026-10-29, other projects' jobs loading the CPU; `make` is not
  installed, so each recipe's lines ran by hand):** `make test` ≈ **11 min** (`cargo test
  --workspace` 277 s, `pytest` 368 s with 18 passed, `vitest` 3 s with 7 passed); `make lint`
  ≈ **2 min** (fmt 1 s, clippy 1 s cached, ruff check 4 s, ruff format 1 s, mypy 105 s cold,
  `tsc -b` 4 s, oxlint 3 s), all exit 0. Two of three `cargo test --workspace` runs failed
  two app-api wall-clock budget tests under that load; they pass alone
  (`docs/known-issues.md`). torch 2.14.1+cu130, `cuda.is_available()` True.
- GPU: RTX 4060 Ti, 1.6 GB used, 0 % busy at planning time (big2 training not yet started).

## Shared resources (named so parallel helpers stay safe)

- **Cargo build directory**: `target/` in the main tree. Helpers in worktrees get their own.
  App Control sometimes blocks a freshly built executable (os error 4551); run the same
  command again (app-api README).
- **The conda env `mahjong`**: only Task 0 installs anything. Nobody else runs `pip install`.
- **The CPU**: Task 13 (labelling, ~2.3 days) and every 4,000-hand match with a search seat
  use all 20 threads. Matches use fixed sample counts (`samples=`), never `ms=` budgets, so
  their results do not depend on what else is running. No benchmarks while a labeller runs.
- **The GPU**: Tasks 5 (optional), 15, 17–19 and 21. Each checks `nvidia-smi` first and does
  not start while big2 training holds more than 4 GB.
- **`crates/app-api/`**: never edited by this run (app stream, D52). If an engine change
  breaks its tests, the engine change is wrong, not the test.
- **`crates/search/src/agent.rs`** (spec parsing, tournaments): Tasks 3, 7, 10, 16 add to it
  in sequence, never in parallel.
- **`crates/search/src/net.rs`** (MJNN format): Tasks 5 and 14 both extend it; Task 5 first.
- **Outputs**: `runs/logs/` (self-play logs), `runs/labels/` (PIMC labels), `runs/benchmark/`
  (Brian's walls and hands), `runs/matches/` (match reports), `runs/ppo/` (checkpoints); all
  gitignored except small JSON summaries copied into `docs/study-results/`.
- **Models**: `models/*.mjnn` committed only when under 10 MB; `.pt` files stay out of git.
- **Port 8000**: the lab web UI (Task 11 only). Ports 5180 and 8020 belong to hk-games.
- **Plan file and `docs/open-items.md`**: the coordinator edits them; helpers report instead.

## Constraints (verbatim from bot-stream `docs/decisions.md`, plus repo rules)

- **B1 Research on Fable, questions at the end.** […] Amended the same day after a usage
  limit: Fable only for difficult tasks (analysing research results, formulating outcomes and
  strategies for the app); other helpers run on opus. (Brian, 2026-10-28)
- **B5 Compute and outside code as in holdem:** local PC only; no commercial solvers; open
  source agents (e.g. DouZero, Mortal, RLCard) may be read and run as benchmarks, never
  linked or shipped. (Claude's default, carried from holdem D8/D9, 2026-10-28)
- **B7 App first.** Mahjong and big2 ship `bot_action`, `hint`, `review`, `grade` with reason
  keys from today's bots before training; 13 cards and Dou Dizhu do the same once their rules
  engines exist. (Brian: default)
- **B14 Outside code as local test opponents only** (Charlesworth's big2 network, the 13
  cards repos without licence, PerfectDou and AlphaDou through WSL2); never copied or
  shipped. (Brian: default)
- **B15 Mahjong: Brian's explicit EV design is a second approach, compared with the
  proposal.** Each move is valued as offence EV (P(win) × payout from opponents) minus
  defence EV loss (P(deal-in or other win) × payout to that opponent), with opponents' win
  chances and hand values estimated from their discards and melds. Defence minimises EV
  loss, so it may deliberately deal into a small-faan hand to stop a large-faan self-draw.
  The build phase builds it beside the proposal's network-and-search path and compares
  them in duplicate matches on the same walls (all 24 seatings). Reason: the two systems
  differ in structure, and Brian wants a measured comparison. (Brian, 2026-10-28)
- **B16 Other mahjong variants later.** Japanese (riichi) and Chinese Official (MCR) may be
  added in a later phase, using their strong published bots as references and benchmarks.
  (Brian, 2026-10-28)
- **B18 Scope.** Dou Dizhu up to release 2a (beat DouZero, approach PerfectDou; no 2b, no
  full belief net). Mahjong, big2 and 13 cards follow their proposals (mahjong plus B15).
  Holdem follows the fix-and-retrain plan once Brian restarts it. (Brian: default)
- **B19 Compute on this PC only.** No cloud, no new hardware for now. (Brian)
- **B20 `quality` and `interval` fields** are requested from the app stream. (Brian: default)
- **B21 Mahjong:** single hands first; Brian plays 400 hands as the human benchmark. (Brian)
- **B25 Go for the build run** (mahjong, big2, 13 cards, Dou Dizhu as in B18; holdem on
  hold), to start in a fresh session from `docs/plans/HANDOFF.md`. (Brian, 2026-10-29)
- Repo rules (hk-games CLAUDE.md, bot-notes 2026-10-29): never edit `crates/app-api/`; keep
  every faan key from `Pattern::english` unchanged; keep a search path that runs with
  `threads = 1` and no thread pool (WebAssembly); action codes 0–133 stay fixed (networks
  depend on them); every comparison of agents is a duplicate tournament; a rules change
  updates `docs/rules/hk.md`, the Rust engine and the Python reference oracle together (none
  is planned here); engines return keys, never display text (D42). In hk-games, edit only
  `docs/streams/bot-notes.md` and commit it alone.

## Acceptance criteria

Hands are counted per agent seat; "vs `value`" means one seat against three `value` bots in
the 4-seating duplicate format (proposal §6: 1,600 hands → about ±1.5; 4,000 hands → about
±1.3 for search agents). "24 seatings" means every permutation of four agents on each wall,
paired by wall, with a bootstrap 95 % interval.

- **AC1 Environment.** `make test` and `make lint` pass on this PC from the activated
  `mahjong` env with rustup's cargo; their durations are written in this file's header.
- **AC2 `State::from_position`.** A mid-hand `State` is built from hands, melds, river,
  bonus tiles, wall count, phase and rules without a wall; for 200 positions taken from
  seeded hands, the rebuilt state gives the same legal actions, scores, `Danger`, features
  and `shortlist` as the original; malformed input returns an error, never a panic. The app
  stream is told in bot-notes.
- **AC3 Belief model.** On held-out self-play games (a field the model never saw in
  training: a different bot mix), the per-tile count log-likelihood beats best-of-K
  sampling and uniform sampling; predicted tenpai chance and predicted deal-in chance are
  within their 95 % intervals of the observed rates in every one of 6 wall-progress buckets;
  two positions that differ only in hidden tiles give identical outputs (test).
- **AC4 Better teacher.** PIMC-64 with belief-drawn determinizations scores ≥ +3 points per
  hand vs `value` with a 95 % CI excluding 0 over 4,000 hands and a deal-in rate ≤ 17.9 %
  (`value`'s). Its rollouts' win and deal-in rates are within 3 points of real play
  (today 27 % vs 22 % and 10 % vs 15 %). If the bar fails, §9 of the proposal applies: keep
  best-of-K with discard weights and record the result; the run continues.
- **AC5 EV bot.** `ev` beats `defensive` (−1.9) with a CI excluding 0 over 4,000 hands vs
  `value` and is not below `value` by more than its interval; it takes < 5 ms per decision on
  one thread; the constructed "deal into the small hand to stop the big self-draw" position
  (Task 9) is answered as B15 describes, with the reason key `reason.ev.stops_big_hand`.
- **AC6 Human benchmark tooling.** 400 fixed walls exist; Brian can play them in the lab UI
  with hints off and every hand logged; `mahjong hk benchmark replay --agent <spec>` plays
  the same walls in the same seat against the same bots and prints the paired margin with
  its 95 % CI (expected ±4 at 400 hands); 200 disagreement positions export for labelling.
- **AC7 Network.** The distilled suit-shared CNN (≤ 5 M parameters) scores ≥ 0 vs `value`
  with CI half-width ≤ 2 over 4,000 hands; int8 top-1 agreement with fp32 ≥ 98 %; the network
  inside PIMC beats the bare network with a CI excluding 0 over 4,000 hands (24 seatings
  with `value` and PIMC filling the table).
- **AC8 PPO.** The PPO network scores ≥ +5 vs `value` with a CI excluding 0 over 4,000
  hands and beats belief-PIMC-128 head to head over 24 seatings with a CI excluding 0; win
  share per faan pattern within ±30 % of belief-PIMC's; draw rate 5–10 % in self-play; two
  seeds seated together (2 + 2 and 1 + 3) score within each other's intervals; the 5a and
  1a ablations report their gain with intervals.
- **AC9 The comparison (B15).** EV bot, network + search, belief-PIMC-128 and `value` play
  4,000 hands in all 24 seatings; `docs/study-results/phase4-build.md` reports points per
  hand with bootstrap CIs, win, deal-in, draw and per-pattern shares, and both designs'
  paired margins on Brian's 400 walls (when played). The report names the stronger design
  or says the interval does not separate them.
- **AC10 App safety.** `cargo test -p mahjong-app-api` passes unchanged at every commit; a
  snapshot test pins every `Pattern::english` string; every new search path runs with
  `threads = 1`; `cargo clippy -p mahjong-app-api --target wasm32-unknown-unknown` passes;
  the bot-notes entry (Task 24) tells the app stream what is new.
- **AC11 Docs.** `docs/solver.md`, `docs/rl.md`, `docs/architecture.md`, `README.md`,
  `CLAUDE.md` (commands) and `docs/known-issues.md` describe what was built.

## Phase 0 – Environment, `from_position`, harness, logs (opus)

- [x] **Task 0: Make the env work on this PC**
  - Files: modify `CLAUDE.md` (Environment section: rustup cargo, `PYO3_PYTHON`, Windows
    notes), `Makefile` (a `dev-windows` target or a note), `docs/known-issues.md` (create:
    the GNU-cargo trap, App Control retries).
  - Steps: in an activated `mahjong` shell with `C:\Users\brian\.cargo\bin` first on `PATH`
    and `PYO3_PYTHON=C:\Users\brian\miniconda3\envs\mahjong\python.exe`: `pip install -e
    ".[dev]"`, `pip install torch --index-url https://download.pytorch.org/whl/cu124` (or the
    CUDA build torch's site names for this driver), `npm install --prefix web`.
  - Done when: `python -c "import mahjong._core"` works; `make test` passes (Rust + Python +
    web) and `make lint` passes; `python -c "import torch; print(torch.cuda.is_available())"`
    prints `True`; the durations of `make test` and `make lint` are written in this file's
    header and in `CLAUDE.md`.
  - Notes: do not change `environment.yml`'s `rust` line (macOS uses it); document the
    Windows override instead. Machine time: 10–20 min.

- [x] **Task 1: Faan-key snapshot test and a one-thread search test** `[parallel-ok]`
  - Files: create `crates/hk/tests/pattern_names.rs`, `crates/search/tests/one_thread.rs`.
  - Done when: `cargo test -p mahjong-hk --test pattern_names` compares every
    `Pattern::english` string with a literal list in the test (AC10); `cargo test -p
    mahjong-search --test one_thread` runs `pimc` inside a `rayon::ThreadPoolBuilder::new()
    .num_threads(1)` pool and checks the result equals the default pool's for the same seed.
  - Notes: the one-thread test is the guard every later search change must keep green.

- [x] **Task 2: `State::from_position`** `[parallel-ok]`
  - Files: modify `crates/hk/src/state.rs`; create `crates/hk/tests/from_position.rs`; modify
    `crates/pybind/src/lib.rs` (`HkState.from_position(...)`), `python/mahjong/_core.pyi`,
    `python/tests/test_from_position.py`; modify `docs/architecture.md` (one paragraph).
  - Signature (Rust): `State::from_position(rules: RuleSet, p: &PositionSpec) ->
    Result<State, PositionError>` with `PositionSpec { hands: [TileCounts; 4], melds:
    [Vec<Meld>; 4], bonus: [Vec<Tile>; 4], river: Vec<(seat, Tile, claimed: bool)>,
    wall_remaining: usize, turn: usize, phase: Phase (turn | claim{tile, from} | rob{tile}),
    last_drawn: Option<Tile>, liable: [Option<usize>; 4] }`. The constructor: checks every
    tile kind totals 4 across hands, melds, river and the hidden wall count (bonus tiles
    once each); checks hand sizes (14 for the mover in `turn` phase after a draw, 13
    otherwise, minus 3 per meld); fills the undrawn wall with the remaining tiles in index
    order (the sampler sorts the hidden pool, so the order is invisible to every observer);
    synthesises `Event::Discard` entries in river order so `Danger` and the features see
    discard history; sets `discards`, `dealer_first_turn = false`, `kong_chain = 0`.
  - Done when: `cargo test -p mahjong-hk --test from_position` passes: (a) for 200 positions
    (50 seeds × 4 stop points, including claim windows) the state rebuilt from the original's
    public view plus hands gives identical `legal_actions_vec`, `self_drawn_score`,
    `claim_score`, `Danger::estimate` for every seat, `features::encode` and `shortlist`;
    (b) 5 malformed specs (a fifth copy, a wrong hand size, a meld tile also in the river
    unclaimed, `wall_remaining` that does not balance, a claim phase with no claim tile) each
    return `Err`, never panic; `pytest python/tests/test_from_position.py` builds one
    position from Python and analyses it.
  - Notes: the app's `Position` keeps its full-wall shape; the app stream decides how to
    expose this in `crates/app-api` (Task 24 tells them the signature). Hidden claim
    declarations of other seats are not part of a position (they are asked again).

- [x] **Task 3: The 24-seating harness** `[parallel-ok]`
  - Files: create `crates/search/src/tournament.rs` (move `duplicate` there from
    `agent.rs`, keep the old function as a wrapper); modify `crates/search/src/lib.rs`,
    `crates/pybind/src/search.rs` (`tournament(..., seatings=4|24)`),
    `python/mahjong/cli/main.py` (`mahjong hk tournament --seatings 24 --out <json>`),
    `python/mahjong/_core.pyi`; create `crates/search/tests/tournament.rs`,
    `python/tests/test_tournament.py`.
  - What it adds: all 24 permutations of four agents per wall (or 4 rotations as today);
    per-agent points per hand with (a) the independent-hands CI as today and (b) a
    wall-paired bootstrap CI (resample walls, 2,000 draws); pairwise paired differences
    between agents (same wall, same seat); win, self-drawn, deal-in, draw rates; mean faan;
    win share per faan pattern (keyed by `Pattern::english`); deal-ins split by victim
    agent; points lost to liability. One JSON shape written by every match in this plan.
  - Done when: `cargo test -p mahjong-search --test tournament` checks that 24 seatings of
    four identical `value` bots give per-agent means within the bootstrap CI of 0 and that
    deltas sum to zero per hand; `mahjong hk tournament value defensive efficiency random
    --deals 50 --seatings 24 --out runs/matches/smoke.json` writes the JSON and prints a
    table. Machine time: < 2 min.
  - Notes: results depend on `seed` and `deals` only; the per-pattern shares are what AC8's
    ±30 % check reads.

- [x] **Task 4: Self-play logs with hidden hands** `[parallel-ok]`
  (done: train 2,092,494 rows, hold-out 435,334 rows; one row per decision from the actor's
  view, not four; `test_logs.py` 2 passed)
  - Files: create `crates/search/src/log.rs` (row layout and writer), `crates/search/examples/
    selfplay_log.rs`, `python/mahjong/belief/__init__.py`, `python/mahjong/belief/logs.py`
    (numpy reader), `python/tests/test_logs.py`; modify `.gitignore` (`runs/logs/`).
  - Row (one per opponent-turn seen from each observer, so four rows per decision): the
    observer's public features (`features::encode`, 933 floats as float16), the discard
    order per seat (last 8 discards each, as tile indices), and the **hidden truth** for each
    of the three opponents: 34 tile counts, tenpai-with-legal-win flag, the set of winning
    tiles (34 bits), points per share if they won now (0 if not tenpai), and whether they
    actually won this hand and how (discard / self-draw) with the points.
  - Field: a mixture so the model does not learn one style: per hand, four seats drawn from
    {`value` ×3, `defensive` ×2, `fast`, `efficiency`, `pimc:samples=16,candidates=4`} with
    `pimc` at most one seat in 10 hands. Hold-out field for AC3: {`value`, `closed`,
    `net:path=models/ppo.mjnn`, `defensive`} with its own seed range.
  - Done when: `cargo run --release -p mahjong-search --example selfplay_log -- --hands 25000
    --out runs/logs/train --seed 1` and `--hands 5000 --field holdout --out runs/logs/holdout
    --seed 2` write chunked files (`*.bin` + `meta.json`) with ≥ 2 M and ≥ 0.4 M rows;
    `pytest python/tests/test_logs.py` reads a chunk and checks shapes and that counts per
    tile kind across hidden hands plus the observer's view never exceed 4.
  - Notes: machine time about 1–2 h on 20 threads (`fast` plays a hand in 1.7 ms; `value`
    in 40 ms; `pimc:16` seats dominate). Headless; check command: `Get-Content
    runs\logs\train\meta.json`.

## Phase 1 – The shared belief model (fable for design, opus for the rest)

- [ ] **Task 5: Belief model design, training and export** (fable)
  - Files: create `python/mahjong/belief/model.py`, `python/mahjong/belief/train.py`,
    `python/mahjong/belief/evaluate.py`, `python/tests/test_belief.py`; modify
    `crates/search/src/net.rs` (MJNN v2: named output slices; keep v1 readable),
    `python/mahjong/rl/model.py` (`export` writes v2 when asked); create `docs/belief.md`.
  - The model (design is this task's job; the shape below is the default): input = the 933
    public features plus the discard-order block (3 opponents × 8 × 34 one-hot, or an
    embedding of it); ≤ 1 M parameters; outputs per opponent (relative seat): 34 count
    logits (trained as a per-tile categorical over 0–4 copies, 170 outputs), tenpai logit,
    34 wait logits (conditional on tenpai), expected points per share as 8 faan-bucket logits
    (3 → 10 faan), and P(self-draw win before the hand ends). Loss: sum of cross-entropies
    with the tenpai-conditional terms masked. Compare against two baselines on the hold-out
    log: uniform re-dealing and best-of-K (`world::sample_world`, 64 samples → empirical
    count distribution per tile). Report calibration per wall-progress bucket (6 buckets, as
    `danger.rs`) for tenpai and for "the tile I discarded was won on".
  - Done when: `python -m mahjong.belief.train --logs runs/logs/train --out
    models/belief.pt --epochs 5` writes `models/belief.pt` and `models/belief.mjnn`;
    `python -m mahjong.belief.evaluate --logs runs/logs/holdout --model models/belief.pt --out
    runs/belief/eval.json` prints a table meeting AC3 (count NLL below both baselines;
    tenpai and deal-in calibration within CI in all 6 buckets); `cargo test -p mahjong-search
    net::` loads the v2 file and a test checks identical outputs for two positions differing
    only in hidden tiles; `pytest python/tests/test_belief.py` passes.
  - Notes: GPU if free (≈ 2–3 GPU-h), else CPU with 20 threads (≤ 6 h; `torch.set_num_threads
    (20)`). Headless; checkpoint each epoch to `runs/belief/ckpt-*.pt`; resume with
    `--resume`; check command `Get-Content runs\belief\train.log -Tail 5`. Machine time
    marked: **≤ 6 h CPU or 3 GPU-h**. If the discard-order block does not improve the
    hold-out NLL by ≥ 2 %, drop it and say so in `docs/belief.md`.
  - Status (2026-10-29): built; `models/belief.mjnn` (MJNN v2, 2.6 MB) committed; 349 s of GPU for
    5 epochs. AC3: count NLL 0.551 beats uniform 0.634 and best-of-64 0.662 per kind; hidden-tile
    test passes; deal-in within the interval in 4 of 6 buckets (misses: bucket 0 at 1 in 10,000 vs
    none seen, bucket 5 at 0.0346 vs ≥ 0.0348); tenpai over-predicted 5–12 % in every bucket on the
    hold-out field. Not ticked (`docs/belief.md` §4). Discard block dropped (+0.13 %, §5).
    Added the `outcome` head (`x_o` for the EV bot, ev-bot.md §9 Q1).

- [ ] **Task 6: Payment-aware danger from the belief model** `[parallel-ok after Task 5]`
  - Files: modify `crates/search/src/danger.rs` (add `Danger::from_belief(state, me, &Belief)`,
    `Danger::expected_loss(t)` in points, `Danger::liability_tile(t) -> Option<seat>` for a
    tile that would complete seat X's fourth exposed meld, `Danger::points_if_win(seat)`);
    create `crates/search/src/belief.rs` (loads MJNN v2, runs it, caches per state);
    `crates/search/examples/danger_check.rs` (extend: expected-loss calibration by bucket);
    modify `crates/search/src/bots.rs` (`defensive` folds on expected loss when a belief
    model is given; default behaviour unchanged).
  - Done when: `cargo test -p mahjong-search danger::` passes (old tests unchanged plus:
    expected loss is 0 for dead tiles; liability flag fires on a constructed position from
    `from_position` where seat X shows three exposed melds and I hold their fourth); `cargo
    run --release -p mahjong-search --example danger_check -- --belief models/belief.mjnn
    2000` prints predicted vs observed expected loss per bucket, each within its interval;
    `cargo test -p mahjong-app-api` still passes (the old `Danger::estimate` is untouched).
  - Notes: the heuristic tables stay as the no-model fallback. Machine time: danger_check
    ≈ 10 min.
  - Status (2026-10-29): built (`belief.rs` with a per-state cache, `Danger::from_belief`,
    `expected_loss`, `points_if_win`, `liability_tile`, `defensive:belief=`); `danger::` tests,
    app-api tests pass. `danger_check --belief models/belief.mjnn 2000` (111 s): 5 of 6 buckets
    within the interval; bucket 4 misses (0.525 predicted vs [0.385, 0.500]: deal-in rate exact,
    points per deal-in 21.9 predicted vs 18.5 paid). Not ticked (`docs/belief.md` §6).

- [ ] **Task 7: Belief-driven determinization and discard weights in PIMC**
  - Files: modify `crates/search/src/world.rs` (`WorldSampler::{BestOfK, Belief{model,
    weights: bool}}`; `sample_world_from(state, observer, &sampler, rng) -> (State, f64
    weight)`), `crates/search/src/pimc.rs` (`PimcConfig.sampler`; weighted means and paired
    differences when weights are on; still one-thread safe), `crates/search/src/agent.rs`
    (`pimc:...,belief=models/belief.mjnn,weights=1`), `crates/search/examples/calibrate.rs`
    (rollout win and deal-in rates vs real play, per sampler); modify `docs/solver.md` §4.
  - Sampling: for each opponent draw tiles one at a time without replacement with probability
    proportional to the belief's expected count of each kind remaining in the hidden pool;
    the determinization's weight (4b) is the belief likelihood of the sampled hands divided
    by the proposal density, clipped at the 95th percentile so one sample cannot dominate.
  - Done when: `cargo test -p mahjong-search pimc::` and `--test one_thread` pass; `cargo run
    --release -p mahjong-search --example calibrate -- --belief models/belief.mjnn` shows
    rollout win and deal-in rates within 3 points of real play (AC4 second half) and sampled
    opponents' mean shanten within 0.15 of real per bucket; the match `mahjong hk tournament
    "pimc:samples=64,candidates=4,belief=models/belief.mjnn,weights=1" value value value
    --deals 1000 --out runs/matches/pimc64-belief.json` reports ≥ +3 with CI excluding 0 and
    deal-in ≤ 17.9 % (AC4). A second match with `weights=0` isolates the weights' gain.
  - Notes: machine time **≈ 8 h per 4,000-hand match** on 20 threads (0.2 s per search
    decision, ~20 per hand); headless; check command `Get-Content
    runs\matches\pimc64-belief.progress`. If AC4 fails, record it and go on (proposal §9).
  - Status (2026-10-29): built (`WorldSampler::{BestOfK, Belief}`, `sample_world_from`, weighted
    PIMC via `pimc_sampled`/`pimc_with_sampler`; the sampler is an argument, not a `PimcConfig`
    field, because app-api names every field; `pimc:...,belief=,weights=,draws=,tilt=`). Belief
    marginals alone gave hands +0.3 to +0.8 shanten too high, so each opponent resamples 16
    candidates with a shanten tilt (0.8 weighted, 1.2 unweighted, ramped from the deal).
    `calibrate` (seed 7): weighted shanten within 0.15 in 5 of 6 buckets (bucket 5 +0.19); win
    −1.5 points, deal-in −3.2 points vs real play (best-of-K +0.9 / −4.4). Both 1,000-deal
    matches run in `runs/matches/task7/` (waits for thirteen-solver). Not ticked until AC4 is read.
  - Status (2026-10-30, matches read): **AC4 not met; run continues (proposal §9).** 4,000
    hands each, 18 threads, ≈ 55 min per match. `weights=1` +1.24 ± 1.21 (bootstrap
    [+0.40, +2.07]), deal-in 17.8 % (bar ≤ 17.9 % met); `weights=0` +1.25 ± 1.22, deal-in
    18.8 %. Weights minus no weights, paired by chunk: −0.02 [−1.09, +1.06]. Misses: +3 bar,
    rollout deal-in −3.2 (bar 3), shanten bucket 5 +0.19. First PIMC with a significant
    edge over `value`; best-of-K is not kept as the fallback (no evidence it is stronger).
    `weights=1` stays the teacher reference until Task 11. Not ticked (bars missed);
    `docs/study-results/phase4-build.md` §2, summaries in `docs/study-results/data/`.

## Phase 2 – Brian's EV bot (fable for the formula, opus for the rest)

- [x] **Task 8: The EV formula, written down** (fable)
  - Files: create `docs/ev-bot.md`.
  - Content: the exact value of a candidate action `a` for seat `me` over the rest of the
    hand, with every term named and sourced:
    `EV(a) = Offence(a) − Defence(a)`, where
    `Offence(a) = Σ_k P(me wins at my k-th future turn | a) × Pay(me) × Shares(win type)`,
    with P(win) from the target value table (Task 9: P(legal win | target, faan-shanten,
    tiles left, exposed melds)) split into self-draw (3 shares) and discard win (1 share,
    weighted by how often opponents discard my waits: the belief's expected counts), and
    `Pay(me)` from the target's guaranteed faan plus the table's E[extra faan];
    `Defence(a) = Σ_o P(o wins on the tile a discards now) × Pay(o) × L(o) +
    Σ_o Σ_k P(o self-draws at future turn k before the hand ends | a) × Pay(o) +
    Σ_o P(o wins on one of my later discards | a, my fold plan) × Pay(o) × L(o)`, with
    `L(o) = 3` when `me` is liable for `o` (12-tile rule) or `a` would make me liable, else 1;
    `Pay(o)` and the win probabilities from the belief model (tenpai, waits, faan buckets,
    self-draw chance). The hand ends at the first win: a deal-in now removes every later term,
    which is what lets the bot deal into an 8-point hand to stop a 128 × 3 self-draw. Horizon:
    the remaining draws for `me`, capped at 8, with the per-turn probabilities from the
    tables; claims and kongs valued the same way on the hand after the claim and its best
    discard; `pass` valued as the current hand. State which terms are approximations and
    how each is checked (a calibration plot for each probability source). Name the reason
    keys: `reason.ev.offence` (args: `p_win`, `points`), `reason.ev.deal_in_now` (`seat`,
    `p`, `points`, `shares`), `reason.ev.stops_big_hand` (`seat`, `p_self_draw`, `points`),
    `reason.ev.liability` (`seat`), `reason.ev.margin` (`points`). Give the three constructed
    test positions of Task 10 with their expected answers and the arithmetic.
  - Done when: the file exists with every term's source named, the two worked examples
    computed by hand (one ordinary discard, one deliberate deal-in), and the hyper-parameters
    listed with defaults (horizon 8, fold-plan = "safest tile each turn", clip of P at 0.95).
  - Notes: discussion-free: the builder (Task 10) implements this file as written. Anything
    the Fable helper cannot settle goes into "Open questions" with a default, not into the
    code.

- [ ] **Task 9: Target value table** `[parallel-ok]`
  (table from the 20,000-hand fit committed, 7e18101. Not ticked: 31 of 37 held-out buckets
  within their 95 % interval; the misses all under-predict by 0.4–2 points (limit hands 2.1).
  Open item: refit with a bias check, or accept for the EV bot.)
  - Files: create `crates/search/examples/fit_targets.rs` (fits from self-play, writes
    `crates/search/src/target_tables.rs`), `crates/search/src/target_value.rs` (lookup:
    `P(legal win before the hand ends | target kind, faan-shanten 0–4, tiles-left bucket 0–5,
    exposed melds 0–4)`, `P(self-draw | win)`, `E[points per share | win]`); modify
    `crates/search/src/lib.rs`; create `python/tests/test_target_table.py` (reads the fitted
    table through pybind and checks monotonic in faan-shanten).
  - Done when: `cargo run --release -p mahjong-search --example fit_targets 20000` writes the
    table (cells with < 50 observations shrink to the parent cell, as `fit_danger` does) and
    prints a calibration table on 4,000 held-out hands with every bucket within its interval;
    `cargo test -p mahjong-search target_value::` passes (every probability in [0, 1];
    faan-shanten 0 ≥ faan-shanten 1 for every other key).
  - Notes: machine time ≈ 1 h (20,000 `value`-mix hands at 40 ms plus bookkeeping).
    Idea 1e of hard-problems.md; the hint `progress` can show it later (app stream's call).

- [ ] **Task 10: The `ev` bot**
  - Files: create `crates/search/src/ev.rs` (`EvBot`, `EvTerms` per candidate with every
    term of Task 8 as a field, `EvBot::explain(state) -> Vec<(Action, EvTerms)>`); modify
    `crates/search/src/bots.rs` (`Bot::Ev`, `Bot::by_name("ev")`), `crates/search/src/agent.rs`
    (`ev:belief=models/belief.mjnn,horizon=8`), `crates/pybind/src/search.rs` (`ev_terms`
    for the lab UI and tests), `python/mahjong/_core.pyi`; create
    `crates/search/tests/ev_bot.rs`, `python/tests/test_ev.py`; modify `docs/solver.md`
    (§2 bots table) and `docs/ev-bot.md` (measured numbers).
  - Done when: `cargo test -p mahjong-search --test ev_bot` passes three constructed positions
    (built with `from_position`, belief outputs stubbed through a `Belief` trait so the test
    does not depend on model weights): (1) a plain efficiency discard when nobody threatens;
    (2) **the deliberate deal-in**: seat A shows three one-suit exposed melds with a dragon
    pung (E[points] ≥ 96, self-draw chance high) and seat B is tenpai on a cheap hand whose
    wait I hold; the bot discards B's wait and `explain` lists `reason.ev.stops_big_hand`;
    (3) the same as (2) but I am liable for B (12-tile rule): the bot does not deal in.
    `cargo bench`-style timing in the test: < 5 ms per decision on one thread. The match
    `mahjong hk tournament "ev:belief=models/belief.mjnn" value value value --deals 1000 --out
    runs/matches/ev-vs-value.json` meets AC5; a second match vs `defensive defensive
    defensive` is reported too.
  - Notes: machine time ≈ 15 min per 4,000-hand match (no search). Every number the bot uses
    is exposed in `EvTerms` so hints can show it (the app stream maps terms to keys).
  - Status (2026-10-29): built with `HeuristicBelief`; 0.37 ms/decision; §6 answers hold only with `ev:floor=none`; provisional matches miss AC5 (`docs/ev-bot.md` §10.1); real-belief matches wait on Task 5.
  - Status (2026-10-29, real belief): `ev:belief=models/belief.mjnn` works (`x_o` from the
    `outcome` head); 0.65 ms/decision with the network. 4,000 hands, seed 2026, points per hand
    (bootstrap 95 %): floor on vs `value` −7.23 [−8.12, −6.36], vs `defensive` −5.49; floor off
    vs `value` −0.99 [−2.02, +0.07], vs `defensive` +0.11 [−0.98, +1.22]. AC5 not met (floor off:
    above −1.9 but the interval does not clear it). Not ticked; `docs/ev-bot.md` §10.2.

- [x] **Task 11: EV bot as the PIMC anchor and rollout policy**
  - Files: modify `crates/search/src/pimc.rs` (`Rollout::with(own: Bot, opponents: Bot)`;
    the baseline action may come from `ev`), `crates/search/src/agent.rs`
    (`pimc:...,anchor=ev,rollout=ev`), `docs/solver.md` §4.
  - Done when: `cargo test -p mahjong-search` and `--test one_thread` pass; a 4,000-hand
    match of `pimc:samples=64,candidates=4,belief=...,anchor=ev,rollout=ev` vs `value`
    is written to `runs/matches/pimc64-ev.json` and compared with Task 7's `anchor=value`
    result in `docs/study-results/phase4-build.md` (draft section). Machine time ≈ 8 h,
    headless.
  - Notes: whichever anchor scores higher becomes the Phase 4 labeller (Task 13).
  - Status (2026-10-29): built (`pimc_anchored`, `shortlist_anchored`, `Rollout::with`;
    `Agent::PimcAnchored`; spec `anchor=value|ev,rollout=fast|ev,ev_<key>=`; the `ev` bot there
    uses the `belief=` model and `floor=none`; specs without the new options build the same
    agents as before). Tests: `pimc::`, `agent::`, `--test one_thread` (new `ev` case). Cost:
    7.6 s per hand vs 1.0 s for Task 7's spec (4 threads). Matches `pimc64-ev` (anchor and
    rollout `ev`) and `pimc64-ev-anchor` (anchor only) on Task 7's walls in
    `runs/matches/task11/`, detached, waiting for Task 7's job to exit, then ≈ 2.5–3.5 h on
    18 threads; uses the frozen extension `runs/matches/task11/pkg/` (Task 7 imports
    `python/mahjong/_core.pyd`, left untouched). Check: `Get-Content
    runs\matches\pimc64-ev.progress, runs\matches\task11\run.log -Tail 5`. Resume: rerun the
    same `Invoke-CimMethod ... run.ps1` command (finished chunks are skipped). Draft section:
    `docs/study-results/phase4-build.md` §1. Not ticked until the result is in.
  - Status (2026-10-30, result): **anchor `value` wins.** 4,000 hands each vs `value`:
    `anchor=ev,rollout=ev` −0.30 ± 1.22, `anchor=ev,rollout=fast` −0.80 ± 1.18, Task 7's
    `anchor=value` +1.24 ± 1.21. Paired by chunk the differences are −1.54 ± 1.33 and
    −2.03 ± 1.42 (both exclude 0). The `ev` anchor cuts deal-ins (14.4 % vs 17.8 %) but wins
    far less (18.9 % vs 24.6 %). Task 13 labels with Task 7's spec (already running).
    Study results §1; summaries in `docs/study-results/data/`. Ticked.

## Phase 3 – Brian's 400 hands: tooling (opus)

- [x] **Task 12: Benchmark mode in the lab UI, replay and label export**
  - Files: create `runs/benchmark/walls.json` (generated: 400 walls, seed 2026, Brian's seat
    = wall index mod 4; committed, ~300 KB), `python/mahjong/benchmark.py` (walls, session
    order, replay, paired margin, label export), `python/tests/test_benchmark.py`; modify
    `python/mahjong/server/session.py` and `app.py` (`POST /api/benchmark/next` creates the
    next unplayed wall with hints disabled and tag `benchmark`; `GET /api/benchmark/progress`),
    `web/src/` (a 基準測試 button that starts the next benchmark hand and shows "n / 400";
    hides the study panel in benchmark hands), `web/tests/` (unit test), `python/mahjong/cli/
    main.py` (`mahjong hk benchmark replay --agent <spec> --out runs/benchmark/replay-<name>
    .json`, `mahjong hk benchmark labels --out runs/benchmark/labels.json`), `docs/ui.md`.
  - Opponents: three bots of one spec read from `runs/benchmark/config.json` (default
    `value`; open question 1), seeded so the bots' moves depend only on the wall and the
    actions so far (the app-api's rule: generator from `(seed, actions so far)`), so the
    replay meets the same bot moves until the first point where the agent and Brian differ.
  - Done when: `pytest python/tests/test_benchmark.py` passes: 20 hands played by `value` in
    Brian's seat through the API are logged with the tag, `replay --agent value` reproduces
    the same results (paired margin exactly 0), and `replay --agent defensive` prints a
    margin with a CI; `labels` exports the positions where the human move and the agent's
    first choice differ (target 200 from 400 hands), as `Position` snapshots plus both
    moves; `make test-e2e` plays one benchmark hand in the browser; `make serve` shows the
    button.
  - Notes: Brian's 400 hands are his time (~20 h); the plan only makes them possible. The
    replay of both designs is Task 23. Logs stay in `runs/benchmark/hands/` (gitignored
    until Brian says otherwise; they are his data).

## Phase 4 – Labels, the network, expert iteration (opus; model design fable)

- [ ] **Task 13: Label 1 M decisions with the best teacher** (long job)
  - Files: create `crates/search/examples/label.rs` (plays mixed-field hands; at every
    decision with a choice runs the Task 11 winner — belief-PIMC-64 with the better anchor —
    and writes features, legal mask, the search's per-candidate means and diffs, the chosen
    action, the chosen target (plan kind + suit + structure) and the final four-seat deltas);
    `python/mahjong/rl/labels.py` (reader); modify `.gitignore` (`runs/labels/`).
  - Done when: `cargo run --release -p mahjong-search --example label -- --decisions
    1000000 --out runs/labels --chunk 20000 --seed 3` writes 50 chunks, resuming from the
    last complete chunk when restarted (test: kill after chunk 2, rerun, chunk 3 continues);
    `pytest python/tests/test_labels.py` reads a chunk. Machine time **≈ 2.3 days on 20
    threads** (0.2 s per decision); headless, announced (> 12 h); check command:
    `(Get-ChildItem runs\labels\chunk-*.bin).Count`.
  - Notes: the seats not being labelled play the belief-PIMC-16 or `ev` bot so the positions
    look like strong play, not `fast` play (one search seat per hand keeps the cost at one
    labelled decision stream). Start this the moment Task 11 is decided; Phase 3 runs in
    parallel on the few spare threads (it is light).
  - Status (2026-10-30): built (`crates/search/src/labels.rs`, `examples/label.rs`,
    `python/mahjong/rl/labels.py`, `python/tests/test_labels.py`; `Agent::search` exposes
    the search's `Analysis`; `LogWriter::open_layout` reuses the log machinery; 2,654
    bytes per row, features as float16; other seats `ev:belief=models/belief.mjnn,floor=none`).
    Resume test: one 300-row run and a stop-after-2-chunks run plus resume give
    byte-identical chunks; a different teacher is refused. Smoke with Task 7's spec: 200
    rows, ≈ 22 rows per hand, 3 % of moves differ from the `value` anchor. Queued detached
    job `experiments/task13_label.ps1` (pid 27120, affinity 0xFFFF0, 8 threads, frozen
    `runs/labels-job/bin/label.exe`): waits for Task 11's job, then takes the teacher from
    `runs/labels-job/teacher.txt` (`auto` = higher mean of `pimc64-belief` and `pimc64-ev`).
    Estimate: Task 7's teacher ≈ 15–25 h (start ≈ 09:00–10:00 today); `anchor=ev,rollout=ev`
    ≈ 7.6 × that (4–8 days). Check: `Get-Content runs\labels\progress.txt;
    Get-Content runs\labels-job\run.log -Tail 5`. Resume: rerun the `Invoke-CimMethod ...
    experiments\task13_label.ps1` command (kept teacher in `teacher-used.txt`). Dry run:
    `powershell -File experiments\task13_label.ps1 -DryRun`. Not ticked until 50 chunks exist.

- [ ] **Task 14: Suit-shared 1-D CNN with policy, target and four-seat value heads** (fable)
  - Files: create `python/mahjong/rl/cnn.py` (model, suit-swap augmentation, int8 export),
    `python/tests/test_cnn.py`; modify `crates/search/src/net.rs` (MJNN v3: 1-D conv layers
    shared across suits, dense honour branch, named heads; int8 weights with per-channel
    scales; fp32 still accepted), `crates/search/src/features.rs` (a `encode_planes` view
    that lays the 26 planes out as 3 suits × 9 + honours × 7 for the conv input; the flat
    `encode` stays for the MLP agents), `crates/search/examples/net_speed.rs`; create
    `crates/search/tests/cnn_io.rs`.
  - Design (this task's job; defaults): 2–5 M parameters; tower of 6–10 residual 1-D conv
    blocks (kernel 3, 128–192 channels) applied to each suit with shared weights, a 2-layer
    dense branch for honours and scalars, a merge, heads: policy 134, target (plan kind 5 ×
    suit 3 × structure 3 = 45 classes), value 4 (zero-sum by subtracting the mean), oracle
    critic inputs as an optional extra block used in training only.
  - Done when: `pytest python/tests/test_cnn.py` (forward shapes; suit-swap invariance of the
    policy up to the permuted action codes; export then reload in Rust gives the same logits
    within 1e-4 for fp32 and top-1 agreement ≥ 98 % for int8 on 1,000 held-out positions);
    `cargo test -p mahjong-search --test cnn_io`; `cargo run --release -p mahjong-search
    --example net_speed -- models/cnn-test.mjnn` prints the forward time on one thread
    (target ≤ 5 ms; record the number).
  - Notes: action codes stay 0–133 (CLAUDE.md). The `.wasm` size matters to the app: report
    the int8 file size (target ≤ 5 MB).
  - Status (2026-10-30): designed and built **on opus, not fable** (this session has no
    fable). `python/mahjong/rl/cnn.py` (design in its docstring and `docs/rl.md` §3.1),
    `crates/search/src/cnn.rs` (MJNN v3 inside `Mlp`, so `net:`/`netpimc:` take it),
    `features::encode_planes`, `pybind net_forward`, tests `test_cnn.py` and `tests/cnn_io.rs`.
    Defaults C = 128, 6 blocks, 512-wide dense parts: 2,187,569 parameters; fp32 8.75 MB,
    int8 2.22 MB; forward 2.1–2.5 ms on one thread (`net_speed`, CPU shared). fp32 Rust vs
    PyTorch within 1e-4 and int8 top-1 agreement 99.8 % on 1,000 held-out positions (random
    weights). Suit-swap equivariance is exact by construction and tested. Not ticked: the
    plan's bars are met on untrained weights; training waits for Task 13's labels (Task 15).

- [ ] **Task 15: Distil the labels** (GPU)
  - Files: create `python/mahjong/rl/distill_cnn.py`; modify `python/mahjong/rl/evaluate.py`
    (24-seating option); create `experiments/phase4_build.py` (stages resume like
    `phase3_rl.py`: distil → evaluate → netpimc evaluate).
  - Loss: policy cross-entropy against the search's choice, weighted by the search's margin
    (`diff / diff_se`, clipped at 3) so clear decisions count more; target head
    cross-entropy against the labelled target; value heads MSE against the four deltas
    (log-scaled: `sign(x) · log(1 + |x| / 8)`, idea 5d).
  - Done when: `python experiments/phase4_build.py --stage distill` trains on the GPU in
    ≤ 2 h (checkpoint every 15 min to `runs/distill/`, `--resume`), exports
    `models/cnn-distill.mjnn` (fp32 and int8); `--stage evaluate` runs 4,000 hands vs `value`
    and meets AC7's first two bars; top-1 agreement with the labels on a held-out chunk is
    printed (expect ≥ 80 %). Machine time: **≈ 2 GPU-h** plus 15 min CPU; headless; check
    command `Get-Content runs\distill\train.log -Tail 3`.
  - Notes: starts only when `nvidia-smi` shows big2 is not training or holds ≤ 4 GB.
  - Status (2026-10-30): built (`python/mahjong/rl/distill_cnn.py`, `experiments/phase4_build.py`
    stages `distill`, `evaluate`, `netpimc`, `python/tests/test_distill_cnn.py`). Labels sit on
    the GPU as float16; suit swaps are GPU gathers; checkpoint every 15 min plus each epoch;
    `--resume`. Slice test (27 chunks = 540 k rows, chunk 27 held out, 14 epochs, 8.7 min):
    held-out top-1 agreement 0.910 (bar 0.80), target accuracy 0.986, Rust int8 vs fp32 top-1
    0.998 (bar 0.98), 2,187,569 parameters, int8 2.22 MB. Held-out value MSE is lowest after
    epoch 2 and rises after it, so the full run uses 12 epochs (about 15 min on 1 M rows).
    The full distillation and the 4,000-hand `evaluate` match run in the queued job
    `experiments/phase4_job.ps1` once Task 13 has 1 M rows. Not ticked until the job's
    numbers are in.

- [ ] **Task 16: The network inside PIMC: candidates, rollouts, leaf value**
  - Files: modify `crates/search/src/agent.rs` (`netpimc` gains `belief=`, `leaf=value|
    rollout`, `depth=N` for a short rollout then the value head), `crates/search/src/pimc.rs`
    (leaf evaluation hook: after `depth` turns, score the state with the network's four-seat
    value for the acting seat instead of playing to the end), `docs/solver.md`, `docs/rl.md`.
  - Done when: tests pass (`one_thread` included); `python experiments/phase4_build.py
    --stage netpimc` runs the 24-seating match {bare CNN, CNN + PIMC-48 (full rollouts),
    CNN + PIMC-48 (leaf value, depth 6), belief-PIMC-64} × 1,000 walls and writes
    `runs/matches/netpimc.json`; AC7's third bar holds for at least one of the two search
    variants. Machine time **≈ 8–12 h** on 20 threads, headless; check command as Task 7.
  - Notes: if the search moved the network by ≥ +1 point with CI excluding 0, round 2:
    relabel 0.5 M decisions with the new searcher (Task 13's example, `--decisions 500000`,
    ~1.2 days) and re-distil (Task 15, `--stage distill2`); otherwise skip to Phase 5 and say
    why in the study results.
  - Status (2026-10-30): built. `netpimc` takes `leaf=value|rollout` and `depth=N` (default 6;
    `belief=` it already had); `LeafValue` and `pimc_with_leaf` in `pimc.rs`
    (`pimc_with_sampler` = `pimc_with_leaf(..., None)`, so no other search changes). Tests in
    `tests/cnn_io.rs`: a leaf deeper than the hand gives the full-rollout numbers bit for bit,
    a 2-discard leaf changes them, one thread = the pool, spec parsing and a full hand.
    Measured on 8 threads with the slice network (32 hands each, labeller running): bare CNN
    0.03 s per hand, CNN + PIMC-48 full rollouts 1.1 s, leaf value depth 6 2.4 s (the CNN
    forward pass at every leaf costs more than a `fast` rollout), belief-PIMC-64 2.8 s. So the
    plan's 1,000 walls × 24 seatings would take ~40 h on 8 threads; the queued job plays
    **400 walls × 24 seatings** (9,600 hands, ≈ 16 h), seeds 7100–7119, and writes
    `runs/matches/netpimc.json`, `netpimc-verdict.{json,md}` (Brian's rule and AC7 bar 3) and
    a `netpimc-partial-verdict.md` after every chunk. Not ticked until the match is in.

## Phase 5 – PPO self-play past the teacher (opus; GPU)

- [ ] **Task 17: `train-core` PPO with the proposal's additions**
  - Files: create `python/mahjong/rl/ppo_cnn.py`, `python/tests/test_ppo_cnn.py`; modify
    `crates/pybind/src/rl.rs` (`HkVecEnv`: (a) `paired=True` plays every wall twice with
    common random numbers, once by the learner and once by the baseline agent in the same
    seat, and reports the paired difference as the reward (idea 5a, the duplicate-baseline
    reward); (b) returns the four-seat deltas and the oracle features for the critic (5b);
    (c) `min_faan` settable per batch for the curriculum (1b); (d) per-pattern win counts
    in `finished()`), `python/mahjong/_core.pyi`.
  - Training loop: batched GPU inference for all envs; clipped PPO as today; critic = the
    oracle four-seat head (sees hidden tiles; outputs sum to zero), actor never sees them;
    rewards log-scaled (5d); curriculum min_faan 0 → 1 → 3 at 20 % / 40 % of the run;
    target head trained on the pattern the hand finally won with (1a); per-pattern KL anchor
    to the distilled network whose weight rises for any pattern whose win share falls below
    80 % of the teacher's (1c); who-pays value head (2b) = the four-seat critic; entropy
    bonus as today. Checkpoint every 30 min to `runs/ppo/<run>/ckpt-*.pt` plus
    `state.json` (hands, iteration, curriculum stage, optimizer); `--resume` continues from
    the newest checkpoint; a 24-seating evaluation of 200 walls vs `value` and the teacher
    every 2 h, logged with per-pattern shares and draw rate to `runs/ppo/<run>/log.json`.
  - Done when: `pytest python/tests/test_ppo_cnn.py` runs a 2-minute smoke run on the CPU
    with 16 envs, checks the paired reward is exactly 0 when learner and baseline are the
    same agent, the critic outputs sum to zero, a checkpoint is written and `--resume`
    continues the iteration count; `python -m mahjong.rl.ppo_cnn --init models/cnn-distill.pt
    --minutes 60 --out runs/ppo/throughput` prints hands per hour (proposal §7 expects
    300 K–1 M; record it and, if below 100 K, apply the §9 row "throughput 10× below").
  - Notes: the baseline agent for the paired reward is the teacher (`netpimc` is too slow;
    use the distilled CNN itself frozen, or `ev` — the faster of the two that scores higher
    in Phase 2). Machine time of the throughput run: **1 GPU-h**.
  - Status (2026-10-30): built (`python/mahjong/rl/ppo_cnn.py`, `python/tests/test_ppo_cnn.py`,
    `HkVecEnv` gains `paired=`, `baseline=`, `set_min_faan`, `finished(detail=True)` with
    per-pattern wins, `observed_roles`, `agent_actions`; defaults unchanged, old `ppo.py`
    works). Test: paired reward exactly 0 for three same-agent setups, critic sums to zero,
    a 40-second 16-env CPU smoke passes all curriculum stages, checkpoint, `--resume`
    continues the iteration count. A 3-minute GPU smoke from a random network (256 envs,
    paired): **33,744 learner hands per hour** (about 67 K simulated); the limit is Rust
    feature encoding (0.93 ms per position on one thread), not the GPU. That is below 100 K,
    so the §9 row "throughput 10× below" applies (open item). Not built: the 24-seating
    evaluation with the teacher (it evaluates 4 seatings vs `value`). Not ticked: the 60-minute
    throughput run from `models/cnn-distill.pt` waits for Task 15.
  - Status (2026-10-30, encoding speed-up): `features::encode` 653 → 76 µs per position on one
    thread (`examples/encode_speed.rs`), output bit-identical (golden hashes over 1,003 and
    36,213 positions stored from the old code; `HkVecEnv` outputs hash-equal old vs new over
    150,711 rows). Changes: `faan_shanten` shares the suit frontiers between its targets and
    allocates nothing; the shanten cache uses a cheap hasher; `encode` skips acceptance tiles
    that provably cannot count; `observe()` encodes into one buffer. Profile of a 256-env paired
    rollout (random network, GPU): per step, `observe` 27 → 7.6 ms, GPU forward ~17 ms, step
    < 1 ms. Throughput, same machine load (other jobs on every core), back to back, 60–150 s:
    normal priority 47.8 K (old) vs 47.3 K (new) learner hands per hour — both limited by
    wake-up latency, not CPU; above-normal priority 121 K (old) vs **219 K** (new). Added
    `--priority above_normal` (default normal; open item). The 3-minute CLI smoke
    (`--init none --minutes 3`): old code 46,264 learner hands per hour; new code 73,926 and
    131,589 (two runs at normal priority, load varied), **217,749** at above-normal priority.
    Still not ticked: the 60-minute run from `models/cnn-distill.pt` waits for Task 15.

- [ ] **Task 18: Main PPO run, second seed, two ablations** (GPU, long)
  - Files: create `runs/ppo/main/`, `runs/ppo/seed2/`, `runs/ppo/ablate-5a/`,
    `runs/ppo/ablate-1a/` (results and checkpoints; gitignored); `docs/study-results/
    phase4-build.md` (training section); `models/cnn-ppo.mjnn` (int8) when done.
  - Done when: `python -m mahjong.rl.ppo_cnn --init models/cnn-distill.pt --hours 48 --out
    runs/ppo/main` finishes or is stopped at 48 h with the log showing ≥ 15 M hands
    (otherwise the throughput shortfall is recorded); `--seed 2 --hours 24` for cross-play;
    `--no-paired-reward --hours 6` and `--no-target-head --hours 6` from the same start;
    `python experiments/phase4_build.py --stage ppo-eval` runs the AC8 matches (4,000
    hands vs `value`; 24 seatings vs belief-PIMC-128, 1,000 walls; 2 + 2 and 1 + 3 seed
    cross-play, 1,000 walls each; ablation runs vs `value`, 1,600 hands each) and writes
    them to the study results with per-pattern shares and draw rates.
  - Notes: machine time **2 + 1 + 0.25 + 0.25 = 3.5 GPU-days**, announced (> 12 h), headless;
    check command `Get-Content runs\ppo\main\log.json | ConvertFrom-Json | Select-Object
    -Last 1`. Status note to Brian at 12 h (proposal). Order: main, then seed 2, then the
    ablations; if the GPU is needed by another game, stop at a checkpoint and resume later.
    Evaluation matches with search seats: **≈ 8 h CPU each**, in the background while the
    next GPU run trains. If pattern collapse shows (shares drift > 30 %, draws > 10 %): raise
    the anchor weight ×2 and shorten the min_faan-0 stage to 10 %, once; record it.

- [ ] **Task 19: Round 3: the PPO network inside the search; re-distil if the search moved**
  - Files: modify `experiments/phase4_build.py` (`--stage round3`); `models/cnn-final.mjnn`.
  - Done when: the 24-seating match {PPO net, PPO net + PIMC-48 leaf value, belief-PIMC-128,
    `value`} × 1,000 walls is in the study results; if PPO + search beats bare PPO with a CI
    excluding 0, the labeller runs 0.5 M decisions with it (Task 13's example, ~1.2 days) and
    a fresh distillation (Task 15, **2 GPU-h**) is evaluated the same way; the better of the
    two networks is `models/cnn-final.mjnn`. Machine time: ≈ 10 h CPU (+ 1.2 CPU-days and
    2 GPU-h if round 3 happens).

## Phase 6 – The comparison, docs, handover (opus)

- [x] **Task 20: Reason keys for the two designs, in the engine's vocabulary** `[parallel-ok]`
  - Status (2026-10-30): done. `crates/search/src/reasons.rs` (`ReasonKey`, `ReasonArgs`,
    `Reason`; `plan_target`, `plan_target_class`, `expected_loss`, `expected_loss_delta` for
    the network path), `EvTerms::reasons()` for every candidate (returns `Vec<Reason>`, keys
    with their arguments), `tests/reasons.rs` (8 tests: key pattern, the three EV positions'
    keys, one thread = pool). `EvDecision::reasons` keeps its type. Keys:
    `reason.ev.{offence,deal_in_now,stops_big_hand,liability,margin}`, `reason.plan.target`,
    `reason.risk.expected_loss` (shown when ≥ 0.5 points).
  - Files: modify `crates/search/src/ev.rs` (`EvTerms::reasons() -> Vec<ReasonKey>` with the
    five `reason.ev.*` keys of Task 8), create `crates/search/src/reasons.rs` (a plain enum
    + args struct, no text; `features`-delta reasons for the network path: the plan head's
    target as `reason.plan.target`, danger delta as `reason.risk.expected_loss` with points);
    `crates/search/tests/reasons.rs`.
  - Done when: tests pass: every reason key string matches `^reason\.[a-z_.]+$`; the three
    EV test positions produce the expected keys; the one-thread test still passes.
  - Notes: the app stream maps these onto `crates/app-api/src/reasons.rs` (their file).

- [ ] **Task 21: The B15 comparison match** (long job)
  - Files: create `experiments/phase4_compare.py`; `docs/study-results/phase4-build.md`
    (comparison section); `runs/matches/compare-*.json`.
  - Done when: `python experiments/phase4_compare.py` plays 1,000 walls × 24 seatings of
    {`ev` (belief), `cnn-final` + PIMC-48 leaf value (the network + search), belief-PIMC-128
    (anchor = the Phase 2 winner), `value`}, reports every AC9 number with wall-paired
    bootstrap CIs, plus two 4,000-hand 1-vs-3 matches (each design vs three `value`), and the
    section states which design is stronger or that the interval does not separate them;
    the stronger design's deal-in rate, mean faan of wins and per-pattern shares are compared
    with the other's in one table. Machine time **≈ 16–20 h on 20 threads** (two search
    seats), headless, announced; check command `Get-Content runs\matches\compare.progress`.
  - Notes: both designs use the same belief model, so the match compares the decision rule,
    not the information. Also report `ev` with the heuristic danger only (no belief) as a
    fifth row in a 1-vs-3 match, so the belief model's own contribution is visible.

- [ ] **Task 22: Replay Brian's benchmark hands with both designs** (needs Brian's hands)
  - Files: `runs/benchmark/replay-ev.json`, `runs/benchmark/replay-net.json`; the study
    results (human benchmark section).
  - Done when: `mahjong hk benchmark replay --agent "ev:belief=models/belief.mjnn"` and
    `--agent "netpimc:path=models/cnn-final.mjnn,samples=48,leaf=value"` print each design's
    paired margin on Brian's walls with CI (AC6, bar 4), and `mahjong hk benchmark labels`
    exports the disagreement positions for Brian's 200 labels (agreement scored by
    `python -m mahjong.benchmark agreement --labels runs/benchmark/labels-brian.json`).
  - Notes: if Brian has played fewer than 400 hands by the end of Phase 6, run on what exists,
    say the interval, and put the rest in open-items. Machine time: minutes.

- [ ] **Task 23: Docs and study results**
  - Files: modify `docs/solver.md`, `docs/rl.md`, `docs/architecture.md` (new crates files,
    the speed table on this PC), `README.md`, `CLAUDE.md` (commands: benchmark, label,
    belief, ppo_cnn; the Windows toolchain note), `docs/known-issues.md`;
    `docs/study-results/phase4-build.md` complete (belief, teacher, EV bot, network, PPO,
    comparison, human benchmark, machine times measured vs this plan's estimates).
  - Done when: `make lint` passes (docs are checked by eye: every command in CLAUDE.md was
    run once in this session); every number in the study results points at a
    `runs/matches/*.json` file committed as a summary under `docs/study-results/data/`.

- [ ] **Task 24: Tell the app stream (bot-notes entry)**
  - Files: modify `C:\Users\brian\OneDrive\Documents\workspace\repo\hk-games\docs\streams\
    bot-notes.md` only (new entry at the top, committed alone in that checkout with
    `git pull --rebase` first).
  - Content: (1) `State::from_position(rules, &PositionSpec) -> Result<State, PositionError>`
    with the field list and the error cases, and the pybind form; (2) new agents and how to
    reach them from `crates/app-api`: `Bot::by_name("ev")` (needs a `Belief` loaded from
    bytes: `Belief::from_bytes`), `Mlp::from_bytes` for MJNN v2/v3, the `netpimc` options
    (`belief=`, `leaf=value`, `depth=`), all with `threads = 1`; suggested level mapping
    (`hard` = `cnn-final` int8, `expert` = `cnn-final` + PIMC leaf value, `medium` = `ev`,
    `easy` and `random` unchanged) as a proposal, their call; (3) the model byte sizes and
    the one-thread forward time measured in Task 14; (4) the reason keys of Task 20 and the
    `EvTerms` numbers a hint can show; (5) faan keys unchanged (the snapshot test); (6)
    the measured comparison result in two lines; (7) the still-open requests `quality`,
    `interval`, `"unclear"` (B20) restated, and whether `from_position` meets their drill
    need. Commit message: `bot-notes: mahjong from_position, ev bot, cnn models, reason keys`.
  - Done when: the entry is pushed in the hk-games checkout and `git -C hk-games status` is
    clean apart from the app stream's own work.

## Order and parallel groups

- Phase 0: Task 0 first (alone: it installs). Then Tasks 1, 2, 3, 4 in parallel (disjoint
  files; Task 3 moves `duplicate` out of `agent.rs`, nobody else touches `agent.rs` in this
  phase).
- Phase 1: Task 5 (fable) alone; then Tasks 6 and 7 in parallel (6 touches `danger.rs`,
  `belief.rs`, `bots.rs`; 7 touches `world.rs`, `pimc.rs`, `agent.rs`, `calibrate.rs`).
  Task 7's 8-hour match runs headless while Phase 2 starts.
- Phase 2: Task 8 (fable) and Task 9 in parallel; then Task 10; then Task 11 (its match in
  the background).
- Phase 3: Task 12, in parallel with Phase 2's matches (it is light on CPU).
- Phase 4: Task 13 starts as soon as Task 11's result picks the anchor (2.3 days, headless);
  Task 14 (fable) in parallel with it (model design needs no labels; it tests on Task 4's
  logs); Task 15 when labels and a GPU gap exist; Task 16 after 15.
- Phase 5: Task 17 (its smoke test on the CPU) while Task 13 or 16 runs; Task 18 when the
  GPU is free of big2; Task 19 after.
- Phase 6: Task 20 in parallel with Task 18's GPU time; Task 21 after Task 19; Task 22 when
  Brian's hands exist; Tasks 23–24 last.
- At most 4 helpers at once; none new after a usage limit (working agreement).

## Risks and the change of course (from proposal §9, made concrete)

| Signal | Course |
|---|---|
| AC3 fails (belief no better than best-of-K) | Keep best-of-K + discard weights (Task 7 `weights=1`); EV bot reads the heuristic danger plus the target table; record in study results |
| AC4 fails (teacher stays near +2) | Label with the best searcher anyway; the network path then rests on PPO (AC8); say so |
| AC5 fails (EV bot below `defensive`) | A Fable review of `docs/ev-bot.md` against 50 logged decisions (where did it lose points?), one revision, one re-match; then report as measured |
| PPO throughput < 100 K hands/h | Smaller CNN for training (2 M), paired baseline = frozen CNN, accept 5 M hands and report |
| Pattern collapse in PPO | Anchor ×2, min_faan-0 stage 10 %, once; else ship the Phase 4 network inside the search |
| GPU contention with another game | Stop at a checkpoint, resume later; the plan file names the week |
| Brian cannot play 400 hands in time | Report with what exists; the bot's "human level" claim waits |

## Not in this run (for `docs/open-items.md`)

- Proposal Phase D: exploiter ladder (bar 3), style pool for `easy`/`medium`, drill miner
  and seed ratings, counterfactual replay, luck ledger, reveal-at-the-end belief in review.
- Proposal Phase E (ported recipes, riichi proxy); B16 variants.
- The phone benchmark on a named Android (the app stream measures its WebAssembly build).
- Full matches with dealer retention (rules §10).

## Open questions for Brian (defaults apply until answered)

1. **Who do you play against in the 400 hands?** Three `value` bots are the field every
   number in this repo is measured against, so your margin is comparable with the bots'.
   Default: three `value` bots. (Alternative: three `ev` bots, once Task 10 passes.)
2. **Hints during the benchmark?** With hints on, the score measures you plus the solver.
   Default: hints and the study panel off in benchmark hands; you can still use study mode
   in ordinary hands.
3. **Where do you play?** The lab web UI (`make serve`, Traditional Chinese, logs every
   hand) is ready now; the hk-games app would need the app stream to fix walls and log
   hands. Default: the lab UI.
4. **The EV bot's deliberate deal-in rule.** Pure EV as B15 says: the bot deals in whenever
   the expected loss of continuing exceeds the deal-in cost, including when it is itself
   close to a win (its own win chance is in the offence term). Default: pure EV, no extra
   rule; the test positions check it.
5. **Phase bars.** The teacher must reach +3 and the PPO network +5 points per hand vs
   `value` (proposal Q3). Default: as written; a failed bar is recorded, not hidden.
6. **GPU sharing with big2.** The belief model (≈ 3 GPU-h) and distillation (≈ 2 GPU-h) fit
   beside a big2 run that uses ≤ 12 GB. Default: no sharing; they wait for a gap or run on
   the CPU (belief ≤ 6 h), so big2's timings stay clean.
7. **Phase D later.** Exploiter ladder, style bot levels and drill mining (≈ 3.5 GPU-days,
   two weeks) are not in HANDOFF item 1. Default: a separate run after this one.
8. **Labelling 200 positions** after your 400 hands (about 3 hours: for each position where
   you and the bot disagreed, which move you think is right). Default: yes, after the
   comparison is in, so the positions come from the final bots.

## Record of the planning session's baseline run (2026-10-29, 03:55–04:00 HKT)

Commands run from the repo root with `C:\Users\brian\.cargo\bin` first on `PATH` and
`PYO3_PYTHON=C:\Users\brian\miniconda3\envs\mahjong\python.exe`:

| Command | Result | Time |
|---|---|---|
| `cargo test --workspace` | exit 0; all suites `ok`; 1 ignored (app-api timing) | 112 s (20 s compile) |
| `cargo clippy --workspace --all-targets -- -D warnings` | exit 0 | 5 s (cached; 20 s cold) |
| `cargo fmt --all --check` | exit 0 | < 1 s |
| `conda run -n mahjong cargo test --workspace` (env's GNU cargo) | exit 101: `dlltool.exe` not found | — |
| `conda run -n mahjong pytest` / `ruff` / `mypy` | not installed | — |

Task 0 replaces this with the full `make test` and `make lint` timings.
