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

## 3. Danger model (`crates/search/src/danger.rs`)

A documented heuristic over public information. Each opponent gets a **threat**
(the chance they are waiting) from their exposed melds, whether those melds
already carry faan, and how far the wall has run. Each tile gets a weight for how
likely a waiting opponent is to want it: zero if no copies are unseen, low for an
honour with one copy left, lower for terminals than middle tiles, reduced for a
tile the opponent discarded (Hong Kong has no furiten, so not zero), and strongly
shaped by a visible flush. Bots use it to fold and to break ties; study mode shows
it. Phase 3 trains a learned replacement.

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

**No peeking.** Determinization sorts the pool of hidden tiles before shuffling, so
two positions that differ only in hidden tiles produce *identical* solver output
for the same seed; a test checks this, and that every bot's decision is unchanged.

**How far to trust it.** PIMC infers little from how opponents have played
(only how developed hands typically are at this stage, not which tiles they
discarded) and evaluates each sampled future
as if everyone could see it ("strategy fusion"). Its numbers are estimates *under
the rollout bot's style of play*. The confidence intervals cover sampling noise
only, not these model errors. Measured strength is in
[study-results/phase1-solver.md](study-results/phase1-solver.md).

## 5. Solo estimates (`crates/search/src/solo.rs`)

For each discard: how often a legal self-drawn win is reached within the
remaining draws if no one else did anything, with the mean faan. This is pure
offense, shown in study mode beside the solver's full numbers.
