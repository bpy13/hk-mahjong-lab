# The rule points of the second and last rounds (plan 2026-11-16 Task 32)

Built 2026-11-23 by slot 1, from ADR 0006 and ADR 0007 (all 20 points of plan Task 28's
check are now settled). The rules file (`docs/rules/hk.md`), the engine (`crates/hk`) and
the Python oracle (`python/tests/reference/hk_reference.py`) changed together. Nothing
was retrained.

## What changed

| Point | Rule | Where in `docs/rules/hk.md` | Test |
|---|---|---|---|
| 13 | 食換 forbidden: right after a pung, not the claimed kind; after a chow, also the kind that re-forms the chow (claim 4 with 5-6: not 4, not 7) | §4.1.4 | `second_round.rs`: `chow_bans_the_claimed_tile_and_the_re_forming_tile_4_1_4`, `pung_bans_the_claimed_tile_4_1_4`, `a_claim_that_would_leave_no_legal_discard_is_not_offered_4_1_4` |
| 9 | A seat that places its seventh bonus tile may decline 花糊, replace it and play on; the eighth still wins 大花糊 at once | §3.3, §5.3, §7.5.1 | `a_seven_flower_win_may_be_passed_7_5_1`, `after_a_passed_seven_the_eighth_wins_at_once_7_5_31`, `a_seven_flower_win_in_the_deal_may_be_passed_3_3`; `scenarios.rs` seven-flower tests now declare the win |
| 10 | A flower win adds 海底撈月 (+1), 槓上 (+1) or 槓上槓 (10) when they hold; no 自摸 | §7.5.2 | `a_flower_win_on_a_kong_replacement_adds_kong_faan_7_5_2` (4 faan), `flower_win_scores_with_last_tile_and_double_kong_7_5_2` |
| 15 | 九子連環 on a self-draw: the final 14 tiles are enough; a discard win still needs the nine-sided wait | §7.4 #26 | `nine_gates_by_final_shape_on_a_self_draw_26`; `faan.rs` unit test |
| 14 | 包大三元: the feeder of the third dragon pung (or kong) pays the whole self-draw; no liability on a kong's replacement win; a later liability replaces an earlier | §8.3.4, §8.3.5 | `big_dragons_feeder_pays_the_whole_self_draw_8_3_4`, `big_dragons_no_liability_on_a_kong_replacement_win_8_3_4`, `a_later_liability_replaces_the_earlier_8_3` |
| 19 | 人糊 (#33), a limit hand: a non-dealer self-draws a win in the first go-round | §7.4 #33 | `hand_of_man_on_the_first_go_round_33`, `a_claim_ends_the_first_go_round_33`; `scenarios.rs` `self_drawn_win_everyone_pays_8_2` now scores it (seat 2 wins on its first draw) |
| 19 | 12-tile exception: no liability when the fourth open set is a pung of a tile the fed player discarded earlier | §8.3.3 | `no_twelve_tile_liability_for_a_pung_of_the_players_own_discard_8_3_3` |
| 18 | After an exhaustive draw the deal passes on | §10 (rules file only; matches are not built) | none |
| 5 (kept) | 槓上開花 counts when the kong's replacement is a flower | §5.4 | `kong_replacement_still_counts_after_a_flower_replacement_5_4` |
| 7 (kept) | 小四喜 is a limit hand, 10 faan | §7.4 #21 | `small_four_winds_is_a_10_faan_limit_21` (and `faan_spec.rs` `rows_21_to_28_limit_hands`) |
| 20 (kept) | 地糊 counts after the dealer's concealed kong | §7.4 #30 | `earthly_hand_still_counts_after_the_dealers_concealed_kong_30` |

**What ends the first go-round (人糊):** the first claim of a discard (chow, pung or kong).
Without a claim, seat `s` has its first turn after exactly `s` discards, so the engine
counts a non-dealer's self-draw as 人糊 when nothing has been claimed and the number of
discards equals its seat. Concealed kongs do not end the first go-round (as for 地糊,
point 20), and a win on the replacement tile of the winner's own first-turn kong still
counts. The rules-check row (zh.wikipedia: "閒家於開局第一輪即自摸") does not say more.

## Calls that are not Brian's (he may change them)

- Orchestrator, 2026-11-23: a pung or chow whose 食換 ban would leave no legal discard
  is not offered. The case: three open sets and `4p 5p 6p 7p` in hand; chowing a
  discarded 4p with 5-6 would leave 4p and 7p, both banned. Only a chow can run into
  this; a pung always leaves a legal tile.
- Orchestrator, 2026-11-23: the flower choice is a decision phase of its own, `flower`
  (`State::in_flower`, `phase_name() == "flower"`). It reuses codes 34 (take 花糊) and
  103 (play on), so the move space stays 134 codes. **Code 103 now has a second
  meaning:** "pass" in a claim or rob window, and "replace the seventh flower and play
  on" in the flower phase.
- Orchestrator (ADR 0007): 人糊 is 10 faan, like 天糊 and 地糊.
- Slot 1's readings: the 12-tile exception covers a claimed **pung** only (a claimed
  kong as the fourth set gets no exception: Brian, "12-tile exception only for pung").
  "Earlier" means any earlier discard of that kind in the hand. 包大三元 is always on,
  not tied to the `twelve_tile_liability` variant. HKMA's "槓後自摸不用包" applies to
  both of its liabilities, but this task applies it to 包大三元 only: the 12-tile rule
  keeps its liability on a kong's replacement win, as before, since nobody decided
  to change that.

## The move set: every place that assumes it

The action codes did not change (134). What changed is which codes are legal: fewer
discards right after a pung or chow, fewer chows offered, and a new phase with codes
34 and 103.

| Place | What it assumed | Now |
|---|---|---|
| `crates/hk/src/state.rs` | the legal-move list | the engine: `discard_ban()`, the claim check, the flower phase |
| `crates/search/src/bots.rs` (`Efficiency`, `Value`, `Fast`, `defensive`, and PIMC's baselines through them) | discards taken from the hand, not the legal list | candidates filtered to the legal discards; the flower choice takes 花糊 |
| `crates/search/src/route.rs` | the same | `pick_discard` takes the `discard_ban()` mask |
| `crates/search/src/race.rs` | with `pass_win` on, a win weighed against the best discard | the flower choice takes 花糊 |
| `stage.rs`, `ev.rs`, `net.rs`, `agent.rs`, `pimc.rs` | take an offered win, then choose from the legal list | unchanged; they take 花糊 |
| `crates/search/src/tournament.rs` | trusted the agents | asserts every move is legal, release builds included |
| `crates/search/src/features.rs` | the phase one-hot covers turn, claim and rob | unchanged: in the flower phase it reads all zero (see below) |
| `crates/search/src/labels.rs`, `crates/pybind/src/rl.rs` | pattern bits in a `u32` over the 32 rows | rows 1–32 keep their bits; row 33 has none (the layout is unchanged). `phase_code` stores the flower phase as 2, like rob |
| `crates/pybind` (`HkState.phase`), `python/mahjong/_core.pyi`, the server (`app.py`: labels 食花糊 and 補花), `web/src` (`api.ts`, the prompt in `App.tsx`) | four phases | five |
| `crates/app-api` | 32 patterns; any pass is a window declaration | 33 patterns; a flower-phase pass is not reported as a declaration |
| Example hand files (`docs/study-results/false-alarm-hand-format.md`) | a pass is a claim pass | the format doc names the flower pass. Both committed files (`early-99000316-seat2.json`, `safe1-99000316-seat0.json`) still replay legally to the end, with their recorded points |

Not covered: a position (`PositionSpec`, `HkState.from_position`) cannot describe the
flower phase. `to_position` writes a flower-phase state as a turn. The phase is rare
(17 times in 10,000 random deals), and no saved data holds one.

## Saved networks

All 21 `.mjnn` networks load, give their recorded outputs, and play a short
duplicate match (6 walls, 24 hands) against three `stage` bots with no illegal move
(`crates/search/tests/model_files.rs`). Policy networks play as `net:path=...`
(16 of them); belief networks play as the reading of `stage-safe:t=1` (5). The 13 files
not kept in this checkout were copied in from the standalone checkout's `models/` for
that run and removed afterwards; the test skips files that are not present. The output hashes
there, the features golden hash and the 200-deal `stage` fingerprint were recorded
again. Random and `stage` play now reaches other positions; the encoders did not
change.

What a saved network sees in the flower phase: the phase one-hot (turn, claim, rob)
is all zero, the "last drawn" plane is empty, and the per-discard planes hold their
off-turn filler. No network has seen that input. Every agent built on a network takes
an offered win before asking the network, so no network makes the flower choice in
play today. After a claim, the mask removes banned discards from what a network may
pick.

Which need retraining: none in order to load or play. All of them were trained under
the old move set and the old payments (Task 31), so they are slightly off-policy. Any
network meant to be used again should be retrained, and Task 29's run trains on the
new move set anyway.

## Checks

- Oracle against engine (`python/tests/test_differential.py` way, step by step: who
  acts, every legal list, every winner's faan, patterns and payments): **10,000 random
  deals** agree (1,116 wins; 17 flower choices; 16,281 turns under a 食換 ban), and
  3,000 flower-heavy walls agree (963 flower choices, 76 passed, 3 人糊). A new test,
  `test_engine_matches_reference_on_flower_walls`, keeps the flower walls in the suite.
- A 2,000-deal `stage` match (four `stage` bots, seed 32, duplicate, 8,000 hands per
  bot) ran to its end with no illegal move: cpu job 315, exit 0, 20 s; output
  `runs/stage/t32-stage-2000.txt` (each bot +0.09, −0.07, −0.11, +0.09 ± 1.3 a hand,
  win 23.7–24.5 %, draws 5.5 %).
- `cargo fmt --check`, `cargo clippy --workspace --all-targets -D warnings` and
  `cargo test --workspace` pass; pytest, ruff (files touched), mypy, tsc, oxlint and
  vitest pass.
