# End-of-run review: the self-play network plan (2026-11-17, Task 8)

Scope: commits `7b03646..3d20935` of `docs/plans/2026-11-16-self-play-network.md` (45 files,
about 6,800 added lines), reviewed by two helpers on 2026-11-17 (correctness on the main paths,
on Fable; scripts, docs and secrets, on Sonnet), fixes by the Task 8 worker. The two large
files copied unchanged from the watch session's work copy (`stage.rs`, `route.rs`) were read
only where Task 9 changed them; their strength was reproduced digit for digit (Tasks 1, 2b, 9).

## Fixed in this task
- `docs/study-results/self-play-network.md`, Task 4b: the three `stage` seats' ranges were
  written "each ± 0.70"; `runs/rl-stage/imitation/vs-stage.txt` gives ± 0.69, 0.70, 0.70.
  Corrected. Every other figure quoted in the study file was checked against its file
  (`final/vs-stage-40000.txt`, `table-4000.txt`, `dealin.txt`, `log.json`, `stopped.txt`,
  `imitation/result.json`) and matches.
- `docs/rl.md` §4 and §5 and `docs/architecture.md` did not name the three-stage bot, the
  `history` input or the opponent pool; `docs/solver.md`'s bot table had no `stage` row.
  Added, with the flags `--pool`, `--pool-minutes`, `--eval-deals`, `--pull`,
  `--pull-fade-hours`, `--draw-limit`, `--opponents`, `--priority`, the stop rules, `best.pt`.
- `docs/study-results/self-play-network.md`: the 2-hourly curve table had empty win and
  deal-in cells for hours 4-10; filled from `runs/rl-stage/log.json`, with a note on why
  `stopped.txt` names hour 2's result as "the best" (the stall rule compares with the best
  result before the four-result window) while `best.pt` is hour 6's.
- `experiments/stage/final.ps1`, `imitation.ps1`, `imitation_data.ps1`: the checkout guard
  compared paths with a culture-sensitive, case-sensitive `StartsWith`; now the same
  ordinal, case-insensitive check on the full path as `selfplay.ps1` (checked: passes for this
  checkout and a lower-cased path, fails for `repo\hk-mahjong-lab`).
- `experiments/stage/versus.ps1`: the header's queue line was not `jobq.ps1` syntax; corrected,
  and `$ErrorActionPreference = 'Stop'` added. `selfplay.ps1`'s header now also names the
  collapse stop and `best.pt`.

## Judgment calls (not changed; also in `docs/open-items.md`)
- **Stall rule reference point** (`python/mahjong/rl/pool.py`, `stop_reason`): the "best"
  the stall counter compares against is the last result that beat the running best by that
  best's 95 % range, not the result with the highest mean. With results 0.0, 0.4, 0.7 (each
  ± 0.5) the code counts 0.7 as a gain (over 0.0 + 0.5) where the plan's wording ("beyond the
  95 % range of the best earlier result", 0.4 + 0.5) would not: more lenient. `test_pool.py`
  pins the behaviour. The real run stops the same way under both readings (no result ever
  gained), and `best.pt` is chosen by the highest mean, which is why `stopped.txt` names
  -0.26 ± 2.02 (hour 2) as the best while `best.pt` is hour 6's +0.48. Not changed: a training
  setting of a run that has ended (plan: no changes between restarts); a new plan decides.
- **One fixed tournament seed for every 2-hourly result** (`ppo_cnn.py`, `_match`, seed 99):
  all five results share the same 500 walls × 4 seatings, so they are correlated across
  hours (a duplicate design, intentional). Fine for comparing hours; it does not shrink the
  ± 2.0 range of any one result, which is why the stall rule could not see a gain of the
  size hoped for (board, Task 6).
- **`versus.ps1` and `gate.ps1` have no `mahjong.__file__` check**: they run Rust binaries only
  (no Python), so there is nothing to check; `versus.ps1`'s header now says so.
- **Golden-model coverage in this checkout**: `models/` here holds only `belief.mjnn` and
  `race-belief.mjnn` of the 15 files `model_files.rs` knows, and none of the three old `.pt`
  checkpoints `test_cnn_history.py` checks, so those "old input still loads" tests pass here
  mostly by skipping (10 pytest skips). Task 3 ran them in full against the standalone
  checkout's models (board line: all 15 `.mjnn` and 3 `.pt` files unchanged).
- **`stage` opponents in the pool are seeded from a persisted global step counter**
  (`self.steps`), so a pool run's random stream differs from a non-pool run's; harmless.

## Clean (no issues found)
- `features.rs` `encode_history` (3,460 = 96 × 34 + 196; relative seats as in `encode`; a
  claim marks the latest discard; overflow ¼ per discard past the 24th, capped at 1) and the
  Python reference `cnn.py` `encode_history` (same semantics, including a claim on an
  overflow discard); `cnn.rs` planes and input slices; `net.rs` slice table
  (`features` → `passed` → `history`; 933-input networks untouched); `cnn_io.rs`,
  `model_files.rs`.
- `labels.rs` layout 3 (`history` 100 bytes before `legal`, `1 + tile + 34·from-hand +
  68·claimed`, overflow byte), `label.rs` layout check on resume, `labels.py`
  `unpack_history`, `distill_cnn.py` (`HistoryUnpack`, `SuitSwap` permuting the history,
  `--history`, `--passed`).
- `pool.py` and `ppo_cnn.py --pool`: per-seat 50/30/20 draw with the older share folded
  into the current network while the pool is empty; the same opponents in both copies of a
  paired game; the result is taken before `pool.add` (so only earlier members); the KL pull
  fades over total elapsed time; draw, collapse and stall stops are written to `stopped.txt`
  and the state; `best.pt` written only when the latest result is the strict maximum;
  checkpoint forced after a result or a new member; resume restores the history, the next
  pool time, the stop flag and the step counter; the 48-hour budget spans restarts.
- `pybind/src/rl.rs` (`observe_history`, `observe_passed`, `encode_history`), `_core.pyi`.
- Task 9's gate in `stage.rs` (gate 0 = old early return; gate 1 = stage 3 only when no tile
  lies outside the kept flush suits, else stage 1; gate 2 exempts a ready hand; the same test
  in `claim`), bot names in `bots.rs`, `agent.rs`, `target.rs`.
- Job scripts: all six parse; the five that run Python check `mahjong.__file__` and exit 1,
  call the env's Python by full path, set `PYTHONPATH` to the checkout, never
  `pip install -e`, read `HKQ_THREADS`; no path points at `hk-mahjong-lab` or
  `hk-mahjong-lab-3`; `selfplay.ps1 -Freeze` and `--resume` behave as the header says.
- `docs/open-items.md` current (Task 1 left-out items, Task 2b md5 sums); ADR 0003,
  `stage3-gate.md`, `three-stage-bot.md`'s source line, CLAUDE.md's command table.
- Secrets: none in the diff (no tokens, keys, passwords or e-mail addresses). Leftovers:
  no TODOs, debug prints, committed run outputs, `.pyc` or `pkg` folders; `.gitignore`
  covers `runs/` and `models/`; `git status` clean.

## Acceptance criteria
| # | Verdict | Evidence |
|---|---|---|
| 1 `stage` reproduces +2.54 ± 0.21 | met | jobs 149 and 155 (`compare stage value value value 40000 1`): +2.54 ± 0.21, 25.1 %, 18.4 %, 5.1; after Task 9 the digits live on as `stage-gate0` (`stage3-gate.md`) |
| 2 `history` input, Rust = Python, old models load | met | Task 3: 1,000 positions, largest output difference 1.2e-7; `model_files.rs` passes on all 18 model files |
| 3 imitation ≥ 90 %, ≥ -1.0 vs `stage` | met | 93.1 % ± 0.11; -0.24 ± 0.71 (old bot) and -0.11 ± 0.71 (strict bot) over 4,000 deals |
| 4 self-play ran to 48 h or a stop rule, result per 2 h | met | job 167: 10.00 h, stopped by the stall rule; 5 results in `log.json` |
| 5 Brian has the final measurement | met | -0.09 ± 0.22 over 40,000 deals vs three `stage`; pass mark not met; inbox card 3 |

## Tests and lint
Run by the correctness reviewer on this tree before the fixes above (which touch docs and
script comments or the path check only), with `CARGO_TARGET_DIR` outside the checkout, the
conda env's Python by full path and `PYTHONPATH` = this checkout's `python`:
- `cargo fmt --check` and `cargo clippy --workspace -- -D warnings`: pass.
- `cargo test --workspace`: pass (search lib 73 passed, 2 ignored; cnn_io 13; model_files 1
  passed, 1 ignored; app-api 18 passed, 2 ignored; every other suite 0 failed).
- pytest `python/tests`: 120 passed, 10 skipped (missing old `.pt` checkpoints and a smoke
  label folder), 267 s.
The Task 8 worker re-ran pytest and the six scripts' parse check after the fixes (below).
