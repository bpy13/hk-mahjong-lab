# Phase 1 results: bots and the solver

Produced by `python experiments/phase1_solver.py` (raw numbers in
`runs/phase1/results.json`). Every comparison is a **duplicate tournament**: each
wall is played four times with the agents rotated through the seats, so luck of
the deal cancels out. Points are per hand under the default rules (3-faan
minimum, 10-faan limit, half-spicy, flowers on). "±" is a 95% confidence interval.
Unless stated otherwise, the candidate plays against three `value` bots (the
reference field), so a score of 0 means "as good as `value`".

## 1. The bot ladder

1,600 hands per row (400 walls × 4 seatings).

| Bot | Points / hand | Win | Deal-in | Avg faan of wins |
|---|---|---|---|---|
| `value` (the field itself) | 0.0 | 24.2% | 17.9% | 3.85 |
| `defensive` | −2.3 ± 1.0 | 16.6% | **13.9%** | 3.67 |
| `fast` (rollout bot) | −2.3 ± 1.1 | 17.6% | 20.1% | 3.83 |
| `closed` (never claims) | −3.3 ± 1.1 | 15.2% | 19.1% | 3.68 |
| `efficiency` | −5.1 ± 0.9 | 10.6% | 21.6% | 3.27 |
| `random` | −8.7 ± 0.7 | 0% | 26.9% | – |

What this says about the game:

* **Playing for faan matters most.** The `efficiency` bot plays for any complete
  hand as fast as possible. It wins less than half as often as `value`, because
  many of its completed hands are worth fewer than 3 faan and cannot be declared.
  It loses 5.1 points per hand.
* **Claiming is worth about 3 points a hand.** `closed` keeps the 門前清 faan by
  never claiming, but wins far less often (15% against 24%). Under this table,
  speed from claimed melds outweighs the concealed-hand bonus.
* **A naive fold rule costs more than it saves.** `defensive` cuts deal-ins from
  17.9% to 13.9% of hands but gives up a third of its wins, for −2.3 points.
  Defense has to be selective, and this is the gap the solver and learning
  should close.
* The cheap rollout bot `fast` (no acceptance counting) is 2.3 points worse than
  `value`: acceptance counting is worth having when decisions are not rushed.

## 2. Why the 3-faan minimum changes the game

`efficiency` against three `value` bots under different minimums
(1,200 hands each):

| Minimum faan | `efficiency` points / hand | `efficiency` win | Avg faan of all wins (value self-play) |
|---|---|---|---|
| 0 | −0.1 ± 0.4 | 25.7% | 1.6 |
| 1 | −0.1 ± 0.4 | 25.3% | 1.9 |
| 3 | **−4.5 ± 0.9** | 11.9% | 3.8 |

With a minimum of 0 or 1, playing for speed alone is as good as playing for
value: almost every complete hand can be declared. At 3 faan it becomes a large
handicap. This is the core of HK strategy, and the reason for the faan-aware
measure in [solver.md](../solver.md).

## 3. Seats and flowers

`value` self-play, 1,600 hands each. Points per hand by seat (East is the
dealer, with 14 tiles at the start):

| Flowers | East (dealer) | South | West | North | Draws | Avg faan | Self-drawn share of wins |
|---|---|---|---|---|---|---|---|
| on | +0.8 | −0.8 | +2.3 | −2.2 | 5.5% | 3.8 | 24% |
| off | +0.6 | +0.6 | +1.2 | −2.4 | 8.5% | 4.2 | 22% |

The intervals here are about ±1.2. **North is consistently about 2 points per
hand worse off**, with or without flowers. North discards directly before the
dealer, who has an extra tile to start with, so North's discards are the ones the
dealer can chow. Flowers add faan for nothing (無花, seat flowers), so they make
the minimum easier to reach: fewer draws (5.5% against 8.5%), and wins that need
less from the hand itself (3.8 faan on average against 4.2).

## 4. The solver

*Filled in from `results.json` "solver" and "solver_naive".*
