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

PIMC against three `value` bots. The first version failed badly, and finding out
why was the most instructive result of this phase.

### 4.1 Naive PIMC

The textbook version re-deals hidden tiles uniformly at random, then plays the
candidate with the highest mean. 320 hands per row (200 for 128 samples):

| Samples per decision | Points / hand | Win | Deal-in |
|---|---|---|---|
| 16 | −6.1 ± 2.5 | 5.3% | 23.4% |
| 32 | −4.6 ± 2.8 | 7.5% | 24.4% |
| 64 | −2.4 ± 3.1 | 10.6% | 22.2% |
| 128 | −0.4 ± 4.9 | 9.0% | 25.0% |

It wins less than half as often as the bots it is guided by, and deals in more.
Two causes:

1. **Uniform sampling makes opponents look harmless.** In its own rollouts the
   solver won 50–70% of hands, while real play gave 5–10%. Measured
   (`examples/calibrate.rs`), opponents' mean shanten is:

   | Tiles drawn from the wall | Real opponents | Uniform re-deal |
   |---|---|---|
   | 0–20 | 2.70 | 3.55 |
   | 21–41 | 1.54 | 3.24 |
   | 42–62 | 1.09 | 2.97 |
   | 63–83 | 0.77 | 2.54 |

   A real player discards what they don't need, so by mid-hand their hand is far
   more developed than a random one. A searcher that samples uniformly believes no
   one is close to winning, so it ignores defense.
2. **Noise.** The real difference between good discards is often below a point,
   while a single estimate's noise is ±4 points. Taking the argmax picks at random
   among the shortlist, and even declined a few legal wins by chance.

### 4.2 Fixes

* **Structured determinization** ([`world.rs`](../../crates/search/src/world.rs)):
  each opponent's hand is the best (lowest shanten) of K random hands from the
  hidden tiles, with K = 4 + used²/40, where *used* is the number of tiles drawn
  so far. This reproduces the real means above at every stage, and it still uses
  only the observer's information.
* **Anchoring**: every candidate is compared with the `value` bot's choice *on
  the same samples*. The paired difference is several times less noisy. The
  solver deviates only when a candidate is better by more than two standard
  errors, and it always declares a legal win.

240 hands per row:

| Samples per decision | Points / hand | Win | Deal-in | Avg faan |
|---|---|---|---|---|
| 32 | +2.9 ± 5.2 | 25.0% | 18.3% | 4.1 |
| 64 | +1.8 ± 4.5 | 25.8% | 18.8% | 4.2 |
| 128 | +3.6 ± 5.5 | 25.0% | 18.3% | 4.5 |

Win and deal-in rates are now those of the `value` field (24.2% and 17.9%), and
wins are bigger (4.1–4.5 faan against 3.9). Pooled over the three rows the
solver scores **+2.8 ± 2.9 points per hand**. That is a likely gain, but not yet
significant at 95%. More samples barely help, which points at the remaining
limits:
* the rollout bot (`fast`, itself 2.3 points worse than `value`) plays out every
  future;
* sampled opponents are shaped by shanten alone, not by what they discarded.

### 4.3 What it costs

At 32 samples × 4 candidates a decision takes about 0.2 s on 8 threads; 128
samples about 0.8 s. The UI's 高級 bot uses 96 samples with a 0.5 s cap.

## 5. Summary

* The 3-faan minimum is what makes Hong Kong strategy different: playing for speed
  alone costs 4.5 points per hand at 3 faan and nothing at 0–1.
* Claiming melds is worth about 3 points per hand under this table, more than the
  concealed bonus it gives up.
* Simple folding gives up more than it saves.
* PIMC needs opponent hands sampled as developed as real ones, and needs to be
  anchored to a good heuristic. Naive PIMC is far worse than the heuristic it
  rolls out with.
* Corrected PIMC is about as good as the best heuristic, probably slightly
  better. Phase 3 asks whether learning does better.
