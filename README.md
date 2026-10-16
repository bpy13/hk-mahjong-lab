# mahjong — Hong Kong mahjong solver & RL lab

A research codebase for studying **Hong Kong mahjong** (香港麻雀). It pairs a fast
Rust rules engine with:

1. **A search-based solver**: exact hand mathematics (faan-aware shanten, the
   expected value of each discard) plus Monte Carlo search over sampled walls, with
   a deal-in danger model. It recommends discards and claims with win rate, deal-in
   rate and expected points.
2. **A playable UI** against solver bots, with a study mode showing the top 3 actions.
3. **Reinforcement learning** agents, compared head-to-head with the solver.

> Status: **Phase 0** (rules spec and scaffold). The rules in
> [docs/rules/hk.md](docs/rules/hk.md) are a draft under review; no engine code yet.

## Quick start

```sh
conda env create -f environment.yml
conda activate mahjong
make dev     # builds the Rust extension and installs the `mahjong` Python package
make test
mahjong tiles "R 9m F2 1m E 5m R"    # → 159m E RR F2
```

## Documentation

| Document | What it covers |
|---|---|
| [docs/rules/hk.md](docs/rules/hk.md) | The exact rules implemented: claims, kongs, flowers, faan table, payment |
| [docs/decisions/](docs/decisions/) | Why the project is built the way it is |
| [CLAUDE.md](CLAUDE.md) | Commands, layout and conventions for contributors |

## Repository layout

```
crates/        Rust: tiles (vocabulary), pybind (PyO3 module mahjong._core)
               Phase 1 adds: games (traits), hk (engine), analysis (hand maths), search (solvers)
python/mahjong/  Python: CLI; later agents, eval (duplicate tournaments), server, rl
docs/          Rules spec, decisions, study results
experiments/   Scripts that produce the results in docs/study-results/
```
