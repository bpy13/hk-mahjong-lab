# The solver

The solver recommends an action for the player to move and estimates, for each
candidate, the expected points, the chance of winning the hand and the chance of
dealing in. It is built in layers, from exact hand mathematics up to Monte Carlo
search. Code: `crates/analysis` and `crates/search`.

## 1. Hand mathematics (`crates/analysis`)

**Shanten** counts how many tile exchanges a hand is from being ready (one tile
from complete): `-1` complete, `0` ready, `1` one exchange away, and so on. Each
suit is searched once per count vector (cached) and summarised as a frontier: for
each number of melds and whether a pair is set aside, the most partial melds. The
four frontiers combine in a few hundred operations, about 0.3 µs per hand.
Tests check it against the engine's win check: a 13-tile hand has shanten 0 exactly
when some tile completes it, a 14-tile hand has shanten −1 exactly when it is
complete, and one exchange never changes shanten by more than one.

**Faan-aware shanten** is the Hong Kong-specific measure. Under a 3-faan minimum
a hand one tile from complete may still be worthless: `123m 456p 789s 234s 9p` is
ready by tile count but scores only 門前清 (1 faan), so it cannot win. The measure
plans **targets** for the finished hand:

| Part of a target | Options |
|---|---|
| Tiles allowed | any · half flush of the main suit · full flush of the main suit · honours only · thirteen orphans |
| Meld shapes | any · pungs only (#15) · chows only (#14) |
| Scoring pungs | any subset of the dragons, seat wind and prevailing wind held as pairs (or already exposed) |

Each target has a **guaranteed faan**: what it scores before anything lucky
happens, including flowers already held and 門前清 while the hand is concealed.
Only targets reaching the minimum count; the faan-aware shanten is the smallest
shanten among them (tiles outside the target's set are treated as discards, and a
required pung's missing copies are counted, checking they are still unseen). It is
a lower bound on value: it never plans on 自摸 or other luck. A test checks that
whenever it says "complete", the engine's scorer agrees the hand can be declared.

**Acceptance** (ukeire) counts unseen tiles that would lower a measure.

## 2. Bots (`crates/search/src/bots.rs`)

| Bot | Plays for | Claims | Notes |
|---|---|---|---|
| `random` | nothing | randomly | always declares a legal win |
| `efficiency` | any complete hand (plain shanten, then acceptance) | when closer | the "ignore the minimum" baseline |
| `value` | a hand meeting the minimum (faan-aware shanten, acceptance, target faan) | when closer | the reference field in the study |
| `closed` | as `value` | never (except to win) | measures what claiming is worth |
| `defensive` | as `value` | when closer | folds to the safest tile when far from ready against a threat |
| `fast` | as `value`, no acceptance counting | when closer | the rollout policy, ~1.7 ms per hand for all four seats |
| `ev` | the largest expected points: offence (target value table) minus defence (deal-in now, opponents' self-draws, the fold plan, 12-tile liability) | when the claim's EV beats passing by more than `tie_margin` | Brian's B15 design, `docs/ev-bot.md`, `crates/search/src/ev.rs`. Opponents come from a `Belief`; without `belief=` it uses `HeuristicBelief` (the danger tables, §8). `ev:belief=<path>` errors until Task 5's loader exists. Options: `horizon`, `clip`, `tie_margin`, `r_disc`, `four_meld_win`, `feed_clip`, `warn_p`, `fold=safest`, `floor=target\|none`. `ev_terms(state, spec)` in Python returns every term. ~0.4 ms per decision |

## 3. Danger model (`crates/search/src/danger.rs`)

Fitted to simulated games, from public information only. For each opponent a
**threat**, the chance they are already waiting on a tile that gives a *legal*
win (complete and at least the minimum faan), is looked up by exposed melds,
wall progress and whether the exposed melds carry faan or lie in one suit. For
each tile, the chance a waiting opponent wins on it is looked up by tile class
(honour, terminal, 2/8, middle), copies unseen, whether that opponent discarded
it (Hong Kong has no furiten), and its relation to a visible flush. A tile's
danger is 1 − Π(1 − threat × rate) over opponents, then mapped through a
calibration curve fitted on tiles players actually discarded (they are not a
random sample of the unseen tiles).

The tables live in `danger_tables.rs`, written by
`cargo run --release -p mahjong-search --example fit_danger 8000` (1.6 M
opponent-turns from games between the rule-based bots). The `danger_check`
example measures calibration on held-out games. The first version of this model
used hand-set weights and overstated the danger about 8× (4.3% predicted against
0.5% observed per discard). The fitted model predicts 0.9% against 0.8%, and every
bucket matches within its error bar. Bots use it to fold and to break ties, and
study mode shows it (the coloured bar, with the percentage on hover).

## 4. PIMC search (`crates/search/src/pimc.rs`)

1. **Candidates**: every special action (win, kongs, claims, pass) plus the best
   few discards by the faan-aware measure.
2. **Determinizations**: the other hands and the wall are re-dealt consistently
   with everything the player has seen (their hand, all melds, all discards, all
   bonus tiles, the hand sizes). Each opponent's hand is the best (lowest
   shanten) of K random hands from the hidden tiles, with K growing over the hand
   (K = 4 + used²/40), calibrated so sampled hands are as developed as real
   opponents' hands ([`world.rs`](../crates/search/src/world.rs); uniform
   re-dealing made the solver ignore danger, see the Phase 1 results). Hidden
   claim declarations of other seats are forgotten and asked again.
   **Belief sampling** (plan Task 7, `pimc:...,belief=models/belief.mjnn`): opponents'
   hands come from the trained belief model ([belief.md](belief.md)) instead; see
   "Belief-driven determinization" below.
3. **Rollouts**: in each determinization every candidate is played to the end of
   the hand by the `fast` bot in all four seats. All candidates share the
   determinization (common random numbers), so their *differences* are measured
   with much less noise than their absolute values.
4. **Output**: per candidate, mean points with a 95% confidence interval, win
   rate, deal-in rate and draw rate, and the **paired difference** from the
   `value` bot's choice (the baseline) with its standard error. Batches run in
   parallel (rayon).
5. **Choice** (for the bots): always declare a legal win; otherwise play the best
   candidate only if it beats the baseline by more than two standard errors of the
   paired difference, else the baseline. With few samples the search defers to the
   heuristic instead of acting on noise.

### Belief-driven determinization (plan Task 7)

`WorldSampler::Belief` in [`world.rs`](../crates/search/src/world.rs); spec
`pimc:samples=64,candidates=4,belief=models/belief.mjnn[,weights=0|1][,draws=16][,tilt=..]`
(`netpimc` takes the same options). In Rust: `pimc_sampled(state, &config, &sampler)`
and `pimc_with_sampler(...)`; `pimc` and `pimc_with` keep best-of-K. The sampler is an
argument, not a `PimcConfig` field, because `crates/app-api` builds `PimcConfig` with
every field named.

1. **Proposal.** The belief model gives, per opponent and tile kind, P(holding 0–4
   copies). Each opponent in turn (next seat first) draws a hand of exactly its size from
   the tiles still hidden; each copy of kind `t` has odds `m_t / (a_t − m_t)` (`m_t` the
   belief's expected count among the `a_t` copies left). That is "proportional to the
   expected count" with an exact density `q` (Fisher's noncentral hypergeometric, drawn
   through a small table over kinds × tiles).
2. **Structure.** The belief's per-kind marginals are independent, so hands drawn from
   them alone are far less developed than real ones (+0.3 to +0.8 shanten, below). Each
   opponent therefore gets `draws = 16` candidate hands, and one is kept with probability
   proportional to `exp(−tilt × shanten)` (sampling-importance-resampling). The tilt ramps
   from 0 at the deal (dealt hands are random) to its full value once a sixth of the wall
   is drawn. Defaults, fitted with `calibrate` on seed 101: tilt 0.8 with weights, 1.2
   without.
3. **Weights** (`weights=1`, the default). The resampling weight also multiplies in the
   belief likelihood over the proposal density, `L / q` with `L = Π_t p_t(h_t)`, and the
   determinization carries the log of the mean candidate weight, so samples are properly
   weighted for the target `L × exp(−tilt × shanten)`. PIMC clips the weights at their 95th
   percentile and reports weighted means, rates and paired differences; standard errors
   use the effective sample size `(Σw)² / Σw²`. With `weights=0` every sample counts once.
4. **One thread.** Weights are combined after all samples are in, in sample order, so the
   result is bit-identical on one thread and on a pool (test
   `pimc::tests::belief_pimc_is_the_same_on_one_thread`). The sampler reads only the
   observer's view (test `world::tests::belief_worlds_are_consistent_...`).

**Calibration** (`cargo run --release -p mahjong-search --example calibrate -- --belief
models/belief.mjnn --hands 200 --every 3 --samples 96`, seed 7, a different seed from the
fit; seat 0 `value`, three `defensive`, the `danger_check` field).

Opponents' mean shanten at seat 0's turns, real against sampled (16 worlds per turn;
difference in brackets):

| Bucket | Opponents | Real | Best-of-K | Belief, weights=0 | Belief, weights=1 |
|---|---|---|---|---|---|
| 0 | 2,343 | 3.23 | 2.78 (−0.45) | 3.10 (−0.13) | 3.20 (−0.03) |
| 1 | 2,367 | 2.50 | 2.06 (−0.44) | 2.34 (−0.16) | 2.40 (−0.10) |
| 2 | 2,187 | 2.08 | 1.59 (−0.49) | 2.09 (+0.01) | 2.11 (+0.02) |
| 3 | 1,887 | 1.74 | 1.25 (−0.49) | 1.89 (+0.14) | 1.84 (+0.10) |
| 4 | 1,296 | 1.47 | 0.93 (−0.53) | 1.65 (+0.18) | 1.59 (+0.12) |
| 5 | 765 | 1.21 | 0.73 (−0.48) | 1.49 (+0.28) | 1.40 (+0.19) |

Best-of-K (K fitted against `value` opponents) makes `defensive` opponents about half a
shanten too developed in every bucket. The weighted belief sampler is within 0.15 in five
of six buckets; bucket 5 misses (+0.19). On the fit seed (101) the same bucket was +0.01
against a real mean of 1.38 (here 1.21), so part of the miss is noise in the real mean,
which comes from few, correlated late-hand observations. Without the shanten tilt
(tilt 0, seed 101) the belief hands were +0.33 to +0.54 shanten too high with weights and
+0.38 to +0.79 without.

PIMC's predicted win and deal-in rate for the rest of the hand, after the move `value`
really played (96 samples, every third turn, 1,138 decisions; 723 early with wall ≥ 45,
415 late), against what happened:

| Sampler | Win: predicted / actual | Deal-in: predicted / actual | Late win | Late deal-in |
|---|---|---|---|---|
| Best-of-K | 26.4 % / 25.6 % (+0.9) | 13.1 % / 17.5 % (−4.4) | 28.7 / 23.6 (+5.1) | 10.5 / 14.5 (−3.9) |
| Belief, weights=0 | 23.9 % / 25.6 % (−1.6) | 13.9 % / 17.5 % (−3.6) | 26.6 / 23.6 (+2.9) | 10.2 / 14.5 (−4.2) |
| Belief, weights=1 | 24.1 % / 25.6 % (−1.5) | 14.3 % / 17.5 % (−3.2) | 26.6 / 23.6 (+3.0) | 11.2 / 14.5 (−3.2) |

The belief sampler moves both rates towards real play: the late win rate from +5.1 to
+3.0 points and the deal-in rate from −4.4 to −3.2. Deal-ins are still under-predicted
by about 3 points with every sampler, so most of that gap is the rollout policy (`fast`
in every seat), not the sampler; Task 11 tries `ev` rollouts. Plan AC4's "within 3
points" holds for the win rate overall (−1.5) and is missed narrowly for deal-ins
(−3.2). Each decision's actual outcome is one hand, and the decisions within a hand are
correlated, so these rates carry about ±2–3 points of noise.

Strength (AC4: PIMC-64 with belief sampling against three `value`, 1,000 walls × 4
seatings, with and without weights): `runs/matches/pimc64-belief.json` and
`runs/matches/pimc64-belief-w0.json` (job `runs/matches/task7/`); results to be added
when the matches finish.

**No peeking.** Determinization sorts the pool of hidden tiles before shuffling, so
two positions that differ only in hidden tiles produce *identical* solver output
for the same seed; a test checks this, and that every bot's decision is unchanged.

**How far to trust it.** With best-of-K, PIMC infers little from how opponents
have played (only how developed hands typically are at this stage, not which
tiles they discarded); the belief sampler reads discards and melds through the
belief model. Either way it evaluates each sampled future
as if everyone could see it ("strategy fusion"). Its numbers are estimates *under
the rollout bot's style of play*. The confidence intervals cover sampling noise
only, not these model errors. Measured strength is in
[study-results/phase1-solver.md](study-results/phase1-solver.md).

## 5. Solo estimates (`crates/search/src/solo.rs`)

For each discard: how often a legal self-drawn win is reached within the
remaining draws if no one else did anything, with the mean faan. This is pure
offense, shown in study mode beside the solver's full numbers.
