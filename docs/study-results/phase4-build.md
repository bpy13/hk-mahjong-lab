# Build-run results (B25): draft

Results of the mahjong bot build run (plan `docs/plans/2026-10-29-mahjong-bot-build.md`).
Every comparison is a duplicate tournament. Points are per hand under the default rules
(3-faan minimum, 10-faan limit, half-spicy, flowers on, liability on). "±" is a 95 %
interval. The candidate plays against three `value` bots unless stated otherwise, so 0
means "as good as `value`". This file is a draft: sections are added as the run's matches
finish, and the final comparison (plan AC9) is written at the end of the run.

## 1. The PIMC anchor: `value` or `ev` (plan Task 11)

**Question.** Belief-PIMC-64 must beat a baseline move by two standard errors before it
plays anything else; the baseline (the *anchor*) is the `value` bot's choice. Does the
`ev` bot (Task 10, `floor=none`) make a better anchor, and a better policy for the
searcher's own moves in the rollouts? The winner labels Phase 4's training data (Task 13).

**Matches.** 1,000 walls × 4 seatings (4,000 hands each), seeds 7000–7019 in 20 chunks of
50 walls, against three `value` bots, all on the same walls:

| Match | Spec (after `pimc:samples=64,candidates=4,belief=models/belief.mjnn,weights=1`) | Report |
|---|---|---|
| Task 7 reference | (none: anchor `value`, rollout `fast`) | `runs/matches/pimc64-belief.json` |
| Task 11 | `anchor=ev,rollout=ev` | `runs/matches/pimc64-ev.json` |
| Task 11, anchor only | `anchor=ev,rollout=fast` | `runs/matches/pimc64-ev-anchor.json` |

Intervals: each chunk's interval combined over chunks (Task 7's job's merge). The
difference between two matches is given two ways: treating them as independent
(`runs/matches/task11/compare.json`, conservative), and paired by chunk (the same 50
walls and seatings in each of the 20 chunks; t-interval over the 20 chunk differences).

**Results.** Job `runs/matches/task11/` (2026-10-30 03:00–09:41, 18 threads; about
20 min per chunk with `ev` rollouts, about 2 min per chunk with `fast` rollouts). Summaries: `data/pimc64-ev.json`,
`data/pimc64-ev-anchor.json`, `data/task11-compare.json`.

| Agent | Points / hand | Bootstrap 95 % | Win | Self-draw | Deal-in | Mean faan | Draws | Minus Task 7 (independent) | Minus Task 7 (paired by chunk) |
|---|---|---|---|---|---|---|---|---|---|
| Belief-PIMC-64, anchor `value`, rollout `fast` (Task 7) | +1.24 ± 1.21 | [+0.40, +2.07] | 24.6 % | 5.4 % | 17.8 % | 4.86 | 6.5 % | 0 | 0 |
| Belief-PIMC-64, anchor `ev`, rollout `ev` | −0.30 ± 1.22 | [−1.43, +0.82] | 18.9 % | 3.9 % | 14.4 % | 5.12 | 9.4 % | −1.54 ± 1.71 | −1.54 ± 1.33 |
| Belief-PIMC-64, anchor `ev`, rollout `fast` | −0.80 ± 1.18 | [−1.91, +0.31] | 19.1 % | 4.0 % | 14.4 % | 5.12 | 9.2 % | −2.03 ± 1.69 | −2.03 ± 1.42 |

**Decision rule (plan).** Whichever anchor scores higher becomes the Phase 4 labeller. If
the interval of the difference includes 0, this section says so.

**Verdict.** Anchor `value` wins. Both `ev`-anchored searchers score below Task 7's
reference, by 1.5 and 2.0 points per hand. Paired by chunk, both intervals exclude 0; the
conservative independent interval also excludes 0 for both. The `ev` anchor plays like
the `ev` bot: about 3.4 points fewer deal-ins, but about 6 points fewer wins, more drawn
hands (9.4 % vs 6.5 %) and fewer self-draws. Bigger hands (mean faan 5.12 vs 4.86) do not
pay for the lost wins. `ev` rollouts help a little over `fast` rollouts with the `ev`
anchor (+0.50 per hand, not separated). The search keeps the anchor's move unless another
move beats it by two standard errors, so a weaker anchor makes a weaker searcher. Task 13
labels with Task 7's spec (`anchor=value,rollout=fast`), which its job already chose
(`runs/labels-job/teacher-used.txt`).

## 3. The distilled network and the network inside the search (plan Tasks 15–16)

**Question.** Does a suit-shared CNN copied from the teacher's 1 M labelled decisions
play as well as the teacher, and does putting it inside PIMC beat the bare network and
the teacher? Brian's rule (2026-10-30): "use teacher unless distilled model wins
significantly": a challenger replaces the teacher only if challenger minus teacher has
a 95 % interval above 0.

**Pending.** The job `experiments/phase4_job.ps1` (`runs/phase4/`) starts when Task 13's
labels are complete, distils on the GPU (12 epochs), plays the bare int8 network against
three `value` bots on 1,000 walls (`runs/matches/cnn-distill.json`; seeds 7000–7019, but
since the search RNG moved to ChaCha8 these are other walls than Task 7's), then the
24-seating match {bare CNN, CNN + PIMC-48, CNN + PIMC-48 leaf value at depth 6,
belief-PIMC-64} on 400 walls (`runs/matches/netpimc.json`). It writes the rule's verdict
to `runs/matches/netpimc-verdict.md`. Slice test so far (540 k rows): held-out agreement
with the teacher's move 0.910, int8 vs fp32 top-1 0.998.

## 2. Belief-driven determinization (plan Task 7, AC4)

**Question.** Does PIMC-64 get stronger when each sampled world draws the opponents'
hidden hands from the belief model (Task 5) instead of best-of-K by shanten? And do the
importance weights (each world weighted by belief likelihood over proposal density) add
anything on top?

**Matches.** 1,000 walls × 4 seatings (4,000 hands each), seeds 7000–7019 in 20 chunks
of 50 walls, against three `value` bots. Both matches used the same walls. Job
`runs/matches/task7/` (2026-10-30 01:10–02:59, 18 threads; about 55 minutes per match,
far below the plan's 8-hour estimate). Summaries: `data/pimc64-belief.json`,
`data/pimc64-belief-w0.json` (copies of `runs/matches/*.json`).

| Agent (`pimc:samples=64,candidates=4,belief=models/belief.mjnn,…`) | Points / hand | Bootstrap 95 % | Win | Deal-in | Mean faan |
|---|---|---|---|---|---|
| `weights=1` | +1.24 ± 1.21 | [+0.40, +2.07] | 24.6 % | 17.8 % | 4.86 |
| `weights=0` | +1.25 ± 1.22 | [+0.33, +2.17] | 25.1 % | 18.8 % | 4.91 |
| `value` (the three opponents, `weights=1` match) | −0.41 | | 23.7 % | 18.9 % | 4.70 |

Weights on minus weights off, paired by chunk (20 chunks of the same walls): −0.02 per
hand, 95 % interval [−1.09, +1.06].

**Against AC4 and Task 7's done-when.**

| Bar | Result | Met? |
|---|---|---|
| ≥ +3 per hand vs `value`, CI excluding 0 | +1.24, CI [+0.40, +2.07] excludes 0 but stays below +3 | **no** |
| Deal-in ≤ 17.9 % (`value`'s rate in Phase 1) | 17.8 % with weights; 18.8 % without | yes (weights=1 only, by 0.1 point) |
| Rollout win rate within 3 points of real play | −1.5 points (`docs/solver.md` §4) | yes |
| Rollout deal-in rate within 3 points of real play | −3.2 points | **no** (by 0.2) |
| Sampled opponents' shanten within 0.15 per bucket | 5 of 6 buckets (bucket 5 +0.19) | **no** (one bucket) |

**Compared with earlier baselines.** Phase 1's best-of-K PIMC-64 scored +1.0 ± 6.1 over
240 hands, and +1.7 ± 3.2 pooled over 32–128 samples (`phase1-solver.md` §4.2). Those
intervals are too wide to rank the samplers. Belief-PIMC-64 is the first PIMC whose
margin over `value` is significant; it is not shown to beat best-of-K, because no
best-of-K match on these walls exists.

**Verdict.** AC4 fails on strength: about +1.2 per hand, not +3. The belief sampler makes
the sampled worlds more realistic (shanten and rollout win rate) and gives a small,
significant edge over `value`. The weights add no points (difference −0.02 ± 1.07) but
cut the deal-in rate by one point. Following proposal §9 the run continues. The
`weights=1` spec stays the reference teacher until Task 11's `ev` anchor result is in;
the plan's "keep best-of-K" fallback is not applied, since nothing shows best-of-K is
stronger. Most of the remaining rollout gap (deal-ins under-predicted by about 3
points with every sampler) comes from the `fast` rollout policy, which Task 11 replaces.
