# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Environment

Everything runs inside the `mahjong` conda env (Python, Rust, Node, maturin):

```sh
conda env create -f environment.yml   # once
conda activate mahjong
make dev                              # build the Rust extension + install Python package (editable)
```

## Commands

| Task | Command |
|---|---|
| All tests | `make test` (Rust + Python) |
| Rust tests only | `cargo test` |
| Python tests only | `pytest` |
| Lint / format | `make lint`, `make fmt` |
| UI unit tests | `make test-web` |
| Browser test | `make test-e2e` (once: `cd web && npx playwright install chromium`) |
| Web UI + API | `make serve` (http://127.0.0.1:8000) |
| UI dev server | `uvicorn mahjong.server.app:app --reload` + `make web-dev` |
| Speeds | `make bench` |
| Studies | `python experiments/phase1_solver.py`, `python experiments/phase3_rl.py` |

## Layout

* `crates/` — Rust: `tiles` (vocabulary), `games` (Game/Policy traits), `hk`
  (rules engine), `analysis` (shanten, faan-aware shanten), `search` (bots, danger,
  PIMC, features, network inference, tournaments), `pybind` (`mahjong._core`).
* `python/mahjong/` — Python package: `hk.py` helpers (Chinese names), CLI,
  FastAPI server, `rl/` (PyTorch training).
* `web/` — React + TypeScript UI (Traditional Chinese labels).
* `docs/` — rules spec (`docs/rules/hk.md` is authoritative), decisions (ADRs),
  study results.

## Conventions

* The rules doc is the spec. A rules change updates `docs/rules/hk.md`, the Rust
  engine, and the Python reference oracle together, each with a test citing the
  section (`§4.1.3`) or faan-table row (`#17`).
* Tile index: `1m–9m = 0–8`, `1p–9p = 9–17`, `1s–9s = 18–26`, `E S W N = 27–30`,
  `Wh G R = 31–33`, `F1–F4 = 34–37`, `S1–S4 = 38–41`. Hands are `TileCounts`
  (`[u8; 34]`). Text notation: `123m 55p EEE WhWh F2`.
* Public Rust items get rustdoc; public Python functions get docstrings.
* Action codes (0–133, `mahjong_hk::Action::code`) are the Python interface and
  the RL action space; changing them invalidates trained networks.
* Agents are spec strings (`value`, `pimc:samples=64`, `net:path=...`), parsed in
  `crates/search/src/agent.rs`.
* Every comparison of agents is a duplicate tournament (same walls, seats rotated).
* Commit in small, descriptive steps.
* Node tools must run inside the conda env (`conda activate mahjong`).
