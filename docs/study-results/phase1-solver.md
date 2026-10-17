# Phase 1 results: bots and the solver

Produced by `python experiments/phase1_solver.py` (raw numbers in
`runs/phase1/results.json`). Every comparison is a **duplicate tournament**: each
wall is played four times with the agents rotated through the seats, so luck of
the deal cancels out. Points are per hand under the default rules (3-faan
minimum, 10-faan limit, half-spicy, flowers on). "±" is a 95% confidence interval.
Unless stated otherwise, the candidate plays against three `value` bots (the
reference field), so a score of 0 means "as good as `value`".

> **Re-run 2026-10-18** under the current rules: bonus-tile faan (#3–#5) no
> longer count toward the 3-faan minimum (docs/rules/hk.md §6). The re-run also
> includes the calibrated danger model, defending rollout opponents and honour-
> weighted determinization (docs/solver.md). The earlier numbers are kept in
> brackets where they help.

## 1. The bot ladder

1,600 hands per row (400 walls × 4 seatings).

| Bot | Points / hand | Win | Deal-in | Avg faan of wins | (before) |
|---|---|---|---|---|---|
| `value` (the field itself) | 0.0 | 23.9% | 17.9% | 4.77 | |
| `defensive` | −1.9 ± 1.5 | 18.9% | **14.8%** | 4.68 | (−2.3) |
| `fast` (rollout bot) | −5.4 ± 1.5 | 12.2% | 24.4% | 4.68 | (−2.3) |
| `closed` (never claims) | −7.4 ± 1.7 | 6.8% | 22.0% | 5.76 | (−3.3) |
| `efficiency` | −10.3 ± 1.1 | 2.9% | 22.3% | 4.02 | (−5.1) |
| `random` | −13.1 ± 1.2 | 0.1% | 25.0% | – | (−8.7) |

What this says about the game:

* **Playing for faan matters even more now.** With flower faan no longer helping
  to reach 3 faan, `efficiency` (any complete hand, as fast as possible) wins
  only 2.9% of hands and loses 10.3 points per hand; most of its complete hands
  cannot be declared. Wins are worth more on average (4.8 faan against 3.9
  before), because every win now needs 3 faan from the hand itself.
* **Claiming is worth about 7 points a hand.** `closed` keeps the 門前清 faan by
  never claiming, but wins a quarter as often (6.8% against 23.9%). The cost of
  staying concealed more than doubled under the new rule: claimed melds are the
  main way to build faan (dragon and wind pungs, flushes, all pungs).
* **Defense is now close to free.** With the calibrated danger model,
  `defensive` cuts deal-ins from 17.9% to 14.8% and gives up less than before
  (−1.9 ± 1.5 points, against −2.3 and, with the first fitted tables, −5.8).
* **The rollout bot is weaker.** `fast` (no acceptance counting) is 5.4 points
  worse than `value` (2.3 before). The solver's rollouts therefore under-play
  every seat, which is part of its remaining miscalibration (docs/solver.md).

## 2. Why the 3-faan minimum changes the game

`efficiency` against three `value` bots under different minimums
(1,200 hands each):

| Minimum faan | `efficiency` points / hand | `efficiency` win | Avg faan of all wins (value self-play) |
|---|---|---|---|
| 0 | −0.1 ± 0.4 | 25.7% | 1.6 |
| 1 | −0.1 ± 0.6 | 23.6% | 2.2 |
| 3 | **−11.0 ± 1.2** | 2.6% | 4.9 |

With a minimum of 0 or 1, playing for speed alone is as good as playing for
value: almost every complete hand can be declared. At 3 faan it is a disaster.
(Before the flower-rule change it cost 4.5 points per hand; flower faan used to
carry many cheap hands over the line.) This is the core of HK strategy, and the
reason for the faan-aware measure in [solver.md](../solver.md).

## 3. Seats and flowers

`value` self-play, 1,600 hands each. Points per hand by seat (East is the
dealer, with 14 tiles at the start):

| Flowers | East (dealer) | South | West | North | Draws | Avg faan | Self-drawn share of wins |
|---|---|---|---|---|---|---|---|
| on | −3.3 | +1.8 | +0.3 | +1.3 | 5.8% | 4.7 | 24% |
| off | +0.6 | +0.6 | +1.2 | −2.4 | 8.5% | 4.2 | 22% |

The intervals are about ±1.2 to ±1.5, so only differences of about 3 points
stand out. Without flowers, North is again about 2 points per hand worse off
(North discards directly before the dealer, who can chow them). With flowers the
dealer is now the one behind (−3.3). A plausible cause: on a self-drawn win the
dealer pays like everyone else, while flowers make self-drawn and bonus-heavy
wins more frequent and larger (4.7 faan against 4.2), so being the dealer buys
less. That is a hypothesis to test, not a result. Flowers still mean fewer draws
(5.8% against 8.5%).

## 4. The solver

PIMC against three `value` bots. The first version failed badly, and finding out
why was the most instructive result of this phase.

### 4.1 Naive PIMC

The textbook version re-deals hidden tiles uniformly at random, then plays the
candidate with the highest mean. Measured in the first run (old flower rule; not
re-run), 320 hands per row (200 for 128 samples):

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

240 hands per row (re-run with the current solver and rules; the earlier
numbers in brackets):

| Samples per decision | Points / hand | Win | Deal-in | Avg faan | (before) |
|---|---|---|---|---|---|
| 32 | +2.0 ± 5.6 | 22.1% | 19.6% | 5.0 | (+2.9 ± 5.2) |
| 64 | +1.0 ± 6.1 | 22.9% | 16.3% | 5.2 | (+1.8 ± 4.5) |
| 128 | +2.0 ± 5.1 | 22.9% | 17.9% | 4.7 | (+3.6 ± 5.5) |

Win and deal-in rates are close to those of the `value` field (23.9% and 17.9%),
and wins are slightly bigger (4.7–5.2 faan against 4.8). Pooled over the three
rows the solver scores **+1.7 ± 3.2 points per hand**: still positive, still not
significant at 95%, and as before more samples barely help. The limits are the
model of the future, not the sample count:
* the rollout bot (`fast`, now 5.4 points worse than `value`) plays out every
  future;
* sampled opponents are shaped by shanten (and, now, how often honours are
  held), not by what they discarded. A learned opponent model is the top item in
  [research/related-work.md](../research/related-work.md).

### 4.3 What it costs

At 32 samples × 4 candidates a decision takes about 0.2 s on 8 threads; 128
samples about 0.8 s. The UI's 高級 bot uses 96 samples with a 0.5 s cap.

## 5. Summary

* The 3-faan minimum is what makes Hong Kong strategy different: playing for speed
  alone costs 11 points per hand at 3 faan (flowers bonus-only) and nothing at
  0–1.
* Claiming melds is worth about 7 points per hand under this table, much more
  than the concealed bonus it gives up.
* With a calibrated danger model, folding costs little (−1.9 ± 1.5).
* PIMC needs opponent hands sampled as developed as real ones, and needs to be
  anchored to a good heuristic. Naive PIMC is far worse than the heuristic it
  rolls out with.
* Corrected PIMC is about as good as the best heuristic, probably slightly
  better (+1.7 ± 3.2). Phase 3 asks whether learning does better.
