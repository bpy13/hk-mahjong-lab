# Phase 3 results: reinforcement learning

Produced by `python experiments/phase3_rl.py` on an M1 (8 GB), 2026-10-18, under
the current rules (bonus-tile faan do not count toward the 3-faan minimum). Raw
numbers are in `runs/phase3/results.json`, training curves in `runs/phase3/*.json`.
Method and network: [rl.md](../rl.md).

Each network is evaluated as in Phase 1: a duplicate tournament against three
`value` bots, 1,000 hands (250 walls × 4 seatings) unless stated otherwise. So 0
means "as good as `value`", the best rule-based bot. "±" is a 95% confidence
interval.

## 1. Training

| Run | Starts from | Wall-clock | Hands played | Ends with |
|---|---|---|---|---|
| `distill` | – | 38 min | 80 chunks of ~47k decisions | 77% agreement with `value` (38% after the first chunk) |
| `dmc` | `distill` | 90 min | 131k | Q loss ≈ 0.3–0.4 |
| `dmc0` | random | 30 min | 87k | Q loss ≈ 0.03 (see §3) |
| `ppo` | `distill` | 90 min | 316k | entropy 0.65 → 0.31; oracle features removed by the half-way point |
| `ppo0` | `distill` | 40 min | 143k | entropy 0.66 → 0.37; no oracle features |

## 2. Results

| Network | Points / hand | Win | Deal-in | Avg faan of wins |
|---|---|---|---|---|
| `value` (the field, from Phase 1) | 0.0 | 23.9% | 17.9% | 4.8 |
| **`ppo`** (PPO + oracle guiding) | **−0.1 ± 2.9** | 13.8% | 22.0% | 5.7 |
| **`ppo0`** (PPO, no oracle) | **−0.2 ± 2.6** | 15.0% | 21.7% | 5.5 |
| `distill` (imitation of `value`) | −5.6 ± 1.8 | 12.2% | 20.1% | 4.9 |
| `dmc` (Deep Monte Carlo from `distill`) | −11.5 ± 1.5 | 1.0% | 24.6% | 7.5 |
| `dmc0` (Deep Monte Carlo from scratch) | −11.5 ± 1.4 | 0.2% | 20.9% | 3.0 |
| `ppo` + PIMC, `fast` rollouts (240 hands) | −3.4 ± 4.6 | 17.1% | 27.5% | 5.7 |
| `ppo` + PIMC, network rollouts (160 hands) | −1.1 ± 6.5 | 15.6% | 23.8% | 5.6 |

For comparison, the best Phase 1 agents: the PIMC solver +1.7 ± 3.2 (pooled),
`defensive` −1.9 ± 1.5.

### What this says

* **PPO is the only learning method that worked.** Starting from a network that
  was 5.6 points per hand worse than its teacher, PPO self-play closed the whole
  gap: both PPO networks are level with `value` within the interval. They play
  differently from it, though. They win much less often (14–15% against 24%) and
  deal in more (22% against 18%), but their wins are bigger (5.5–5.7 faan against
  4.8). That they still break even suggests their deal-ins cost less (to cheaper
  hands, or late in the hand). That is not yet measured. The interval also allows
  them to be up to 3 points worse than `value`.
* **Imitation alone falls well short.** 77% agreement with `value` costs 5.6
  points per hand. Under the 3-faan minimum a few wrong discards are enough to
  make a hand undeclarable, so small disagreements cost a lot. Which decisions
  the disagreements fall on has not been measured.
* **Oracle guiding made no measurable difference.** `ppo` and `ppo0` score the
  same. But `ppo0` trained for 40 minutes against `ppo`'s 90, so oracle guiding
  bought nothing for twice the compute, or else PPO had already plateaued by 40
  minutes. A run of equal length is needed to separate the two.
* **Search on top of the network did not help** in this budget. Both PIMC
  variants score below the bare network, inside wide intervals. With `fast`
  rollouts the searcher deals in more (27.5%). That is the Phase 1 problem again:
  a weak rollout bot makes opponents look harmless.

## 3. Why Deep Monte Carlo collapsed

Both DMC networks stopped winning (1% and 0.2% of hands). Played against
themselves, four copies of `dmc` drew **92% of 200 hands** (a quick check after
the run; `value` self-play draws 6%). This is a self-reinforcing loop:

1. DMC labels every decision with the hand's final points. When no seat wins,
   every label is 0.
2. In self-play all four seats share the network. Once it plays a little worse,
   more hands end in a draw, so more labels are 0. The Q values then shrink
   toward 0 for every action and stop telling a good discard from a bad one.
3. Under the 3-faan minimum, aiming at a declarable hand needs a long, precise
   plan (Phase 1 §2). A network that has lost the plan almost never stumbles
   into a 3-faan win by chance, so it gets no signal to relearn it.

`dmc0` never got out of step 3. From random play no seat wins, and its tiny loss
(0.03) is the model fitting all-zero returns. `dmc` started from the distilled
network and fell into the same state. ε-greedy exploration (5% random actions)
mostly ruins hands rather than finding wins.

Two likely reasons PPO avoided this (not tested separately): it samples from a policy that started close to
`value`'s and moves it in small, clipped steps, and its value baseline
turns "everyone draws" into zero advantage instead of a target to fit.

DouZero's DMC works in 斗地主 (Dou Dizhu) because every game has a winner. Mahjong under a
minimum-faan rule has many null outcomes. Things that might rescue DMC: keep
the teacher in some seats (league play), add a margin loss toward the teacher's
action (as in the distillation stage), or shape the reward with faan-aware
shanten progress. None is tried here.

## 4. Summary

* PPO from a distilled start reaches the level of the best rule-based bot
  (−0.1 ± 2.9) with a different, higher-variance style: fewer, bigger wins.
* It does not beat the PIMC solver (+1.7 ± 3.2). Neither result is separated
  from `value` at 95%.
* Deep Monte Carlo collapses to a never-win, mostly-draw policy in four-seat
  self-play under the 3-faan minimum, with or without a warm start.
* Oracle guiding showed no benefit at this scale, and network-guided PIMC did
  not improve on the bare network.
* The next steps that address these results are in
  [research/related-work.md](../research/related-work.md): distilling PIMC
  (expert iteration) as a stronger teacher than `value`, a learned opponent
  model for search, and a 4-seat zero-sum value with an oracle critic.
