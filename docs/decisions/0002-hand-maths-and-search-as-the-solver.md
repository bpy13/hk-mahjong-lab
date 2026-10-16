# 0002 — Hand mathematics plus search is the solver; learning is the comparison

**Status:** accepted (2026-10-17)

## Context

"Solver" often means CFR-family equilibrium computation. Its guarantee holds only for
two-player zero-sum games, and mahjong has four players and enormous information
sets (the wall order and three hidden hands). Tabular CFR does not apply.

The strongest mahjong programs are all for Japanese Riichi: Suphx (Microsoft, 2020),
NAGA, Mortal (open source) and LuckyJ (Tencent, 2023). They learn first from
millions of human game logs on Tenhou, then refine with RL. Hong Kong mahjong has no
comparable public log collection and no strong public bot. Its 3-faan minimum also
changes what a good hand is: a hand one tile from completion that cannot reach 3 faan
is not ready, so Riichi-style tile efficiency transfers only partly.

## Decision

* **Phase 1 solver**, in layers:
  1. exact hand mathematics: shanten, tile acceptance, and a *faan-aware* distance
     to a hand that is complete and scores at least the minimum;
  2. for each discard, a single-player expectimax over the remaining draws, giving
     P(win) and expected payout while ignoring opponents;
  3. a deal-in danger model from public information;
  4. Monte Carlo search (PIMC/ISMCTS) over sampled walls and hands, with fast bots
     in the rollouts, for the full decision including claims and defense.
  Every number it reports has a simulation count and a 95% interval.
* **Phase 3 learning:** there are no human logs, so a network is warm-started by
  distilling the solver. It then trains by self-play: DouZero-style Deep Monte Carlo
  first, then PPO with Suphx-style oracle guiding. It is evaluated against the solver
  in duplicate matches and then combined with it (network-guided search).

## Consequences

* The solver's numbers are estimates under a model of the opponents (the rollout
  bots), not equilibrium values. The UI and docs say so.
* PIMC/ISMCTS suffer from strategy fusion. Measuring how much this costs against
  learned agents is one of the study's questions.
