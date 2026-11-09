# Review fix: the rollout bot (plan 2026-11-09, Tasks 1–4)

Results of Phase 1 of `docs/plans/2026-11-09-review-fix-and-race-bot.md`. The app's review
grades each move by PIMC search whose rollouts are played by the `fast` bot (the player's
own later moves) and a defending `fast` bot (the three opponents). On Brian's hand of
2026-11-09 (fixture `crates/search/tests/data/brian-2026-11-09.json`) the review called his
pung of 8p at history index 85 a blunder. "Before" is commit 75dae32, "after" is 2c23636.
Every number below was produced by `crates/app-api/examples/regrade.rs` (README section
"Regrade a kept hand") with the old and the new build; raw outputs are in `runs/review-fix/`.

## 1. What was wrong (found before changing the code)

- **Dead wait kept.** `fast` uses the `FaanQuick` measure, which never counts tile
  acceptance. On a ready hand its discards tie on distance, faan and connectedness, and the
  tie fell to tile order. After the pung (`45p` left, four pin pungs) it threw 4p and kept
  5p, whose four copies were all visible. Acceptance itself was never wrong: `fast` skips it.
- **Live flush swapped for an honour single.** `faan_shanten` sets aside the tiles a plan
  forbids, and the shanten count then assumes a leftover tile to pair. With four pin pungs
  and one Green, the full-flush plan keeps no tile at all, yet counts as ready at 10 faan.
  So `fast` saw "4p wait" and "Green wait" as equal and swapped. (The bug is in
  `crates/analysis`; see section 5 for why it was not fixed there.)
- **Opponents fed the flush.** In `Brain::turn` a seat ready or one away never reaches the
  fold. And the danger tables call a tile safe when the observer holds its last unseen copy.
  After a pass, the opponent holding the last 3p or 6p saw it as safe and threw it into the
  three exposed pin pungs.

## 2. The fix (`crates/search/src/bots.rs`)

- Ready ties under `FaanQuick`: rank by dead wait (no live winning tile) last, then the best
  faan any live winning tile gives (read from the completed hands, which `faan_shanten`
  scores correctly), then the number of live winning tiles. Only when two or more discards
  tie at ready, and only tiles near the hand are tried, to keep `fast` cheap.
- Defending `fast` only (`defense > 0` and `FaanQuick`; `value`, `defensive` and `ev` play as
  before): against a seat with three or more exposed melds in one suit and a table threat of
  at least 0.25, a tile of that suit counts as if one copy were still unseen; a seat ready or
  one away discards the safest tile outside that suit that keeps its distance or costs one
  step.
- Tests: `crates/search/tests/rollout_fixes.rs` (7 tests: Brian's pung replayed from the
  fixture; dead single wait; live flush wait against an honour; dead flush wait; ready,
  one-away and far defenders facing a pin flush, including the held-last-copy case). Each
  failed before the change except "dead flush wait", which passed by tile order.
- Speed (`cargo run --release -p mahjong-search --example speed`, one run each, noisy):
  `fast` self-play 565 µs per hand before, 600–640 µs after; one 200-sample PIMC decision
  105 ms before, 117–127 ms after.
- The feature encoding's golden hash changed because its test games are played by `fast`;
  the encoding code is untouched (`features.rs` comment), so the trained models stay valid.

## 3. AC1: decision 85 on Brian's hand

| Measure | Before | After |
|---|---|---|
| Phone-sized grade (428 ms, 400 samples) | **blunder**, pass better by 48.3 ± 5.9 (1 se) | **best**: pung better by 29.1 ± 10.8 (1 se) |
| Review method, 15,000 samples: pung minus pass | −52.25 ± 1.84 | +19.77 ± 3.43 |
| Belief sampler, 5,000 samples: pung minus pass | −43.89 ± 4.68 | +32.07 ± 8.36 |
| Belief sampler + `ev` rollouts, 1,500 samples: pung minus pass | +26.16 ± 12.26 | +27.39 ± 13.12 |
| Traced, pung then `fast`: first own move | discard 4p in 5,000 of 5,000 | discard 5p in 5,000 of 5,000 |
| Traced, pass then `fast`: win rate (5,000 futures) | 0.846 | **0.345** |
| Traced, pass then `ev`: win rate | 0.833 | 0.308 |
| Traced, pung then `fast`: win rate / mean points | 0.310 / +34.6 | 0.248 / +91.8 |

Intervals in the search rows are 95 % (1.96 se). The `ev`-rollout row did not move: it
never used `fast` for the player's own moves, and it already said pung.

After the fix the traced `ev` policy still swaps the live 4p wait for honour singles in
some futures after the pung (E, R, W, N among its common winning hands): the same
`faan_shanten` flaw likely reaches the `ev` bot (inference, not traced). Open item, section 6.

## 4. Every decision of the hand (`regrade all`), phone-sized

Grades that changed (loss in points; "best" means the played move was the best):

| Index | Move | Before | After |
|---|---|---|---|
| 0 | discard 3s | best | good (0.35; best 5s) |
| 33 | discard 4s | best | good (9.0; best S) |
| 36 | discard 9m | best | good (0.52; best 3m) |
| 40 | discard 3m | inaccuracy (15.1) | good (17.75) |
| 49 | discard S | good (13.0) | inaccuracy (26.14) |
| 62 | discard G | good (0.05) | best |
| 85 | pung 8p | **blunder (48.32)** | **best** |

Unchanged grades: 21 of the 28 decisions. The loss of a "good" move moves by a few points
at most; 40 and 49 sit near the inaccuracy threshold in both builds (400 samples per
decision, standard errors of several points), so their swap is within noise.

## 5. AC3: the teacher with the fixed rollouts (pending)

**Question.** Does the teacher spec (`pimc:samples=64,candidates=4,belief=models/belief.mjnn,weights=1`,
anchor `value`, rollout `fast`) play at least as well with the fixed `fast` as with the old
one? AC3: new minus old, 95 % interval, lower end above −0.5 points per hand.

**Match.** 1,000 walls × 4 seatings against three `value` bots, on Task 7's walls (20 chunks
of 50, seeds 7000–7019), once with the extension built from 75dae32 (`runs/review-fix/pkg-old`)
and once from the fix (`pkg-new`); the spec string is identical. Difference paired by chunk
(t-interval, 19 degrees of freedom). Job: `experiments/review-fix/job.ps1` through the cpu
lane, queue jobs 29 (old), 30 (new) and 31 (compare, after both).

**Result.** Pending. Check: `Get-Content runs\review-fix\old.progress, runs\review-fix\new.progress`;
the verdict lands in `runs\review-fix\compare.json` (`new_minus_old`, `ac3_pass`).

## 6. Open items

- **`faan_shanten` counts a plan with no kept tile to pair as ready** (`crates/analysis`,
  `shanten_of`). Example: four pin pungs and a Green, toward a full flush, report shanten 0
  at 10 faan; the truth is 1. A fix (count the tiles the plan keeps; without a pair and with
  no leftover kept tile, add one) passes the analysis tests, but it changes the feature
  encoding (checked: the hash of 37,249 positions played by `efficiency` differs), so every
  trained network would need retraining. Not applied; the patch is kept outside the repo.
  It also affects `value`, `ev` and the review's baseline in such hands.
- **The danger tables treat a tile as safe when the observer holds its last unseen copy**
  (`Opponent::wait_cell` returns no cell). Only the defending `fast` bot works around it
  against obvious flushes; `defensive`, `value` with defence and the `ev` bot's
  table-based danger still have it.
