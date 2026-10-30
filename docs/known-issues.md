# Known issues

Limits and traps found so far, newest first within each section.

## Windows build and environment

- **The env's own cargo cannot link.** `environment.yml` installs conda-forge `rust`
  (`envs\mahjong\Library\bin\cargo.exe`, GNU toolchain). On Windows its link step fails with
  `dlltool.exe` not found. Fix: put rustup's MSVC toolchain first on `PATH`
  (`$env:PATH = "C:\Users\brian\.cargo\bin;" + $env:PATH`) after `conda activate mahjong`.
  The `rust` line stays in `environment.yml` because macOS uses it.
- **PyO3 needs `PYO3_PYTHON`.** With rustup's cargo, set
  `$env:PYO3_PYTHON = "C:\Users\brian\miniconda3\envs\mahjong\python.exe"` before
  `cargo test`, `cargo clippy` or `pip install -e`, so the `pybind` crate builds against the
  env's Python and not whichever one PyO3 finds first.
- **App Control blocks some fresh executables.** A just-built build script or test binary can
  fail with os error 4551. Running the same command again passes (see
  `crates/app-api/README.md`).
- **`npm install --prefix web` fails** with npm 10.8.2 from the env: npm looks for
  `package.json` in the repo root (ENOENT). Run `npm install` inside `web\` instead.
- **`make` is not installed** on this PC. Run each recipe's lines from the Makefile by hand;
  `CLAUDE.md` lists them for Windows.
- **CUDA torch:** with driver 610.88, `pip install torch --index-url
  https://download.pytorch.org/whl/cu130` gives torch 2.14.1+cu130 with
  `torch.cuda.is_available() == True`.

## Tests

- **Two app-api budget tests fail under outside CPU load.**
  `rule_based_bots_and_expert_answer_within_budget` and
  `review_has_one_item_per_decision_of_the_seat` (`crates/app-api/tests/api.rs`) check wall-clock
  time. Their test lock keeps other tests in the same binary away, but not other programs. On
  2026-10-29, with other projects' training jobs running, `cargo test --workspace` failed them
  twice (expert used 5.8 s and 6.6 s; the debug limit is 4.5 s). Run alone,
  `cargo test -p mahjong-app-api --test api` passed all 14 in 32 s, and a third full run passed. When they fail, rerun that
  binary alone before treating it as a real failure. `crates/app-api/` belongs to the app
  stream, so this repo's sessions do not change those tests.
- **Load slows the whole suite.** On an idle PC `cargo test --workspace` took 112 s; under load
  it took 255–636 s (277 s for the passing run) (`faan_targets` and `shanten_brute` alone took 1–5 min each). `pytest` took
  368 s under load, and 821 s on 2026-10-30 beside the Task 13 labeller (8 threads).

## Random streams

- **Use `ChaCha8Rng` and u32 index draws in anything the app runs.** `SmallRng` is a
  different generator on wasm32 than on 64-bit, and rand 0.8 reads 32 bits for a usize
  range on wasm32 but 64 on native. Either one made the app's WebAssembly `expert` play
  other moves than the server for the same seed (hk-games D28). Since 2026-10-30 PIMC,
  `solo`, the tournament walls and bootstrap use `ChaCha8Rng`, and the random bots draw
  `gen_range(0..len as u32)`. `shuffle` and `choose` are safe: rand 0.8 draws u32 for them.
  A check on 3 seeded hands × 30 moves (`expert`, budget 0): the native and Node builds
  agreed on 90 of 90 moves; before the change they differed on 5 of 24.
  Consequence: a tournament seed deals other walls than it did before that date.
