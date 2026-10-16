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

## Layout

* `crates/` — Rust: `tiles` (vocabulary), `pybind` (PyO3 module `mahjong._core`).
  Phase 1 adds `games`, `hk`, `analysis`, `search`.
* `python/mahjong/` — Python package: CLI; later agents, evaluation, server, RL.
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
* Commit in small, descriptive steps.
