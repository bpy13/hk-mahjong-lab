# 0001 — Standalone repo, Rust engine with a claim-aware game interface

**Status:** accepted (2026-10-17)

## Context

The sister project `work/poker` (Big Two) has a Rust engine, PyO3 bindings, Python
orchestration and a React UI, and its `Game` trait is meant to be game-agnostic.
But that trait assumes exactly one player acts at a time and that every action is
public at once. Mahjong breaks both assumptions: after every discard, the three
other players declare claims *simultaneously* (win > pung/kong > chow, rules §4.1),
and nobody sees the others' declarations until the claim is resolved. Several
players may win on the same discard.

As in Big Two, the solver's accuracy and RL throughput are bound by engine speed:
draws, claim checks, win detection and faan scoring run millions of times per study.

## Decision

* A standalone repository with its own conda env (`mahjong`). Patterns from poker
  are reused (the rules doc as spec, a Python reference oracle, duplicate
  tournaments, phases with check-ins), but no code is shared or imported.
* Rules, hand mathematics and solvers are in Rust (`crates/`), built into one
  extension module `mahjong._core`. Agents, evaluation, CLI, server and RL are in
  Python.
* The game interface models the claim window explicitly. The engine asks each seat
  for its declaration in turn, but the observation given to a seat never includes
  pending declarations from other seats. Resolution then applies the priority rules.

## Consequences

* Changes to the Big Two code never affect mahjong, and the reverse.
* Search and RL code must treat claim decisions as simultaneous. Information-set
  tests check that a seat's view (and so the solver's output) is unchanged by
  another seat's pending declaration.
