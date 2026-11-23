# The payment rule points of ADR 0005: what changed and what was made under the old rule

Plan 2026-11-16, Task 31 (ADR 0005, Brian 2026-11-23). Four rule points change, three are kept
and now have a test each.

| Point | Before | Now | Brian, 2026-11-23 |
|---|---|---|---|
| 1. Win on a discard (§8.2) | the discarder pays 1 share | the discarder pays **2 shares**; a self-draw stays 1 share from each of three | "出銃 pays double is correct" |
| 11. A robbed kong (§8.2) | the kong's maker pays 1 share, as a discarder | the kong's maker pays **3 shares**, the whole self-draw; nobody else pays | "搶槓 is paid as a full 自摸 by the player who made the kong" |
| 6. 槓上槓 (#8) | 8 faan | **10 faan** (the cap) | "槓上槓 and 大花糊 become 10 番" |
| 8. 大花糊 (#31) | 8 faan | **10 faan** | as above |
| 2. Flower faan toward the 3-faan minimum (§6) | do not count | kept | "we should refuse it" |
| 3. 門前清 (#2) with flowers, after a concealed kong | counts | kept | "we want people to learn both the flower and 門前清. Although 暗槓 should not break it" |
| 4. 坎坎糊 (#27) with a kong | concealed kong yes, exposed kong no | kept | "坎坎糊 is not allowed with a kong. 暗槓 allowed." |

The points table (§8.1) already has a 10-faan row (128 points a share) and the cap is 10, so #8
and #31 needed no change to the table. Several winners on one discard: the discarder pays each
winner 2 shares (3 each when a kong is robbed). The 12-tile liability stays self-draw only; when
the liable player is the discarder of a discard win it pays the discarder's 2 shares, nothing more.
The other 13 rule points of Task 28 are still open with Brian and are unchanged.

## What changed

* Rules: `docs/rules/hk.md` §7.1 row 8, §7.5 row 31 and point 3, §8.2 (table and a note with the
  Association's figures: at 3 faan 16 from the discarder against 8 from each of three on a
  self-draw), §8.3 point 2.
* Engine: `crates/hk/src/rules.rs` `DISCARD_SHARES = 2`, `ROBBED_KONG_SHARES = 3` (exported from
  `mahjong_hk`); `State::finish` (`state.rs`) pays them; `faan.rs` #8 and #31 at 10.
  `RuleSet::max_abs_score` is now 6 shares of a limit hand under `multiple_winners = all` (a
  discarder paying three winners 2 shares each), 3 under `head_bump`. No rule variant was added:
  the old payment is not selectable.
* Oracle: `python/tests/reference/hk_reference.py` (`_finish`, #8, #31).
* Tests:
  * `crates/hk/tests/scenarios.rs`: `discard_win_discarder_pays_two_shares_8_2` (5 faan, the
    discarder pays 48); `robbing_an_added_kong_5_2` (the kong's maker pays 3 × 8 = 24);
    `only_thirteen_orphans_robs_a_concealed_kong_5_1` (3 × 128 = 384); `double_kong_win_7_1_row_8`
    (10 faan, 128 from each of three); `all_eight_flowers_pays_10_faan_7_5_31` (128 from each of
    three); `several_winners_all_win_4_1_3` and `head_bump_variant_4_1_3` (2 shares to each
    winner); `earthly_hand_on_the_dealers_first_discard_7_4` (2 × 128);
    `twelve_tile_liability_not_on_a_discard_win_8_3` and
    `twelve_tile_liable_discarder_pays_two_shares_8_3` (the discarder's 2 shares, the liable
    seat nothing more).
  * `crates/hk/tests/faan_spec.rs`: `rows_7_8_kong_replacement` (#8 at 10) and the new
    `kept_rules_adr_0005_6_and_rows_2_27`, one check for each kept point with flowers in play: a
    hand at 3 faan only with a seat flower (2 without) is not a win; 門前清 counts with two
    flowers held and a concealed kong; 坎坎糊 with a concealed kong, not with an exposed one.
    `flowers_do_not_count_toward_the_minimum_6` and `row_27_claimed_pung_is_exposed` already
    covered parts of points 2 and 4.
  * The engine and the oracle agree on 10,000 random deals (seeds 0–9,999 of
    `test_differential.play_both`, rules drawn at random; 1,139 wins), run from this checkout's
    freshly built `_core.pyd` (`runs/t31/oracle10k.py`).
* Bots, wherever a value counts a payment (all follow the engine; nothing retrained):
  * `danger.rs`: a deal-in costs the discarder's 2 shares: `Danger::expected_loss` doubles, and
    `Danger::estimate`'s fixed deal-in cost goes from 8 to 16 points. The order of tiles by
    expected loss does not change (every opponent's term doubles), only how the loss weighs
    against other terms and thresholds.
  * `giveup.rs` (`stage-fold`): an opponent's discard that I win on is worth 2 shares; a later
    deal-in costs 2 shares. `LOSS_FACTOR` and `READ_LOSS_FACTOR` are ratios of points paid to
    points predicted and stay valid; the per-round chances (`Figures`) are not payments.
  * `pushfold.rs` (`stage-fold-learned`): the pay figure follows `giveup.rs`. The network in
    `pushfold_tables.rs` was fitted on play-out labels under the old rule; its inputs (loss,
    `V`, pay) are now twice as large for the same position, which it never saw. Not refitted.
  * `defence.rs` (`stage-ev`): my win on a discard counts 2 shares; deal-ins come from
    `danger.rs`.
  * `ev.rs` (`ev`): `L_disc = 2` always; `Shares(me) = 3 × P_sd + 2 × (1 − P_sd) × φ`.
    `docs/ev-bot.md` §1, §2, §5 examples 2 and 3 (B's stub lowered from 4 faan to 3 so the
    deal-in still costs 15.2 and the example still shows a deliberate deal-in; at 4 faan the
    deal-in would cost 30.4 and the bot would keep the safe tile), §6 rows 1 and 3. Tests
    `ev_bot.rs` and `reasons.rs` follow (`shares: 2`).
  * `race.rs` (`race`): `L_disc = 2`; my discard win 2 shares (a robbed kong 3) in the win-or-pass
    choice and the race; the opponents' fold cost counts deal-ins at 2 shares (1.5, 2.5, 1 instead
    of 1, 2, 0.5 shares of another seat's win; my wins on their discards 2 shares). `FOLD_SCALE`
    was fitted under the old rule, so the modelled opponents now fold more. The log label's
    "clearly safer" margin `LABEL_SAFER_POINTS` goes from 0.16 to 0.32 (still 2 percentage points
    of deal-in at `Danger::estimate`'s cost). `docs/race-bot.md` §3.2, §4.5, §8, §12 follow.
    Test `race_bot.rs` `takes_a_5_faan_discard_win_at_2_shares_even_when_far_ahead` (was
    `passes_a_small_win_when_far_ahead_and_the_wall_is_deep`): with the hand `123m 456m 789m 33m
    EE` the race bot passed a 5-faan discard win (24) for its 8-faan wait; the win is now worth
    48 and the pass about 34, so it takes it. The passed-win lock is still checked, with the pass
    applied by hand.
  * `bots.rs` (the rollout bots' belief danger): the "clearly safer" margin goes from 0.024 to
    0.048 points (still 0.3 % of a 3-faan deal-in).
  * `stage-safe:t=` thresholds are deal-in chances, not points: unchanged. Plain `stage` (`stage.rs`) reads no danger and no
    payment (it counts winning tiles by their points a share): its play is unchanged.
  * Search by play-out (PIMC, rollouts, tournaments) scores with the engine and follows it by
    itself. The value network's leaf values (`pimc` `leaf=value`) were trained under the old rule.

## How much the change moves points

`crates/search/examples/payment_count.rs`, 40,000 deals of four `stage` bots, seed 31, default
rules, run directly (562 s; output `runs/payment/count-40000.txt` in the workers clone). Every win
is settled twice, by the new rule (as played) and by the old one; `stage` plays the same under both
(it reads no payments), so the games are the same and only the settlement differs.

| | Count | Share of hands |
|---|---|---|
| Hands | 40,000 | |
| Wins | 38,361 | |
| Self-drawn (paid the same) | 8,907 | 22.3 % |
| **On a discard (now 2 shares)** | **29,388** | **73.5 %** |
| Robbed kongs (now 3 shares) | 66 | 0.17 % |
| 槓上槓 (now 10 faan) | 3 | 0.01 % |
| 大花糊 (now 10 faan) | 0 | 0 |

* Discarders paid 39.16 points a hand under the new rule against 19.58 under the old: the
  discard win is three quarters of all wins, so the change doubles most of the money that moves.
* Winners received 63.37 points a hand against 43.66 (+45 %).
* Points moved between seats by the change: 19.7 a hand, so each seat's result moves by 9.9
  points a hand on average, 16 to 330 times the paired gains measured between bots
  (±0.03 to ±0.6 a hand). Robbed kongs, 槓上槓 and 大花糊 are too rare to matter (69 hands in
  40,000).

What it means for results made under the old rule: matches between copies of one bot are not biased,
but any comparison where the bots differ in how often they deal in, or how much they give up to
avoid it, was measured on a table that charged a deal-in half of what it now costs. Defence is
worth more now; Task 24's false-alarm match below measures by how much.

## Made under the old rule

Every game played before this commit settled discard wins at 1 share, robbed kongs at 1 share, and
#8 and #31 at 8 faan. Nothing is retrained or rerun here, except the match below.

* Trained networks in `models/`: all of them, as listed in `liability-rule-change.md` ("Made under
  the old rule"): the value heads of `cnn-distill*`, `cnn-ppo`, `ppo`, `ppo0`, `dmc`, `dmc0`,
  `distill`, `cnn-test*` were fitted to final deltas, and a discard win's delta is now twice as
  large, so their value scale is off by up to a factor of 2 for discard wins; the PPO and DMC
  networks were trained on old-rule rewards; the belief networks (`belief*`, `race-belief`) have
  no payment targets (only the logged games' play, through the bots' old values). Also
  `stage-imitation*` (imitation of `stage`, whose play does not depend on the payment), and the
  push-or-fold network of `pushfold_tables.rs` (labels in points under the old rule) and the risk
  network (`risk_net.rs`: its points head is points a share, unaffected; its deal-in chance is
  not a payment).
* Self-play runs: Tasks 4–10 and Task 29's part 1 measurement (job 312, `runs\luck\variance.md`)
  used old-rule rewards. Task 29's 24-hour run waits for this task on purpose.
* Saved results: every match, tournament, label set and study before this commit, all of
  `docs/study-results/` except `rules-check.md`, `liability-rule-change.md` (rule history) and this
  file; in particular `defence-false-alarms.md`, `defence-ceiling.md`, `stage-early-game.md`
  (Task 30, job 311), `stage3-giveup.md`, `stage3-pushfold.md`, `belief-ready-read.md`,
  `self-play-network.md`, and the example hand files (their `paid_by` figures).

## Match re-run under the new rule

Task 24's false-alarm match (seed 24, 40,000 deals, the same four bots `stage-safe:t=1`,
`stage-safe:t=5`, `stage-ev`, `stage-fold` against plain `stage`, reading
`models/belief-order.mjnn`) is queued again as cpu job 313 (`hk-false-alarms-t31`, from the
workers clone; binary built from this engine in `C:\Users\brian\.cache\hk-mahjong-t31\release\examples`;
`experiments/stage/false_alarms.ps1 -Name match-40000-seed24-two-share-rule`; about 0.7 CPU hours,
as job 310 took 2,466 s). Output `runs/defence-false-alarms/match-40000-seed24-two-share-rule.txt`;
its figures go beside the old two in `defence-false-alarms.md`, each named with its rule.
