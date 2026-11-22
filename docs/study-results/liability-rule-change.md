# The 12-tile liability on self-draw only: what changed and what was made under the old rule

Plan 2026-11-16, Task 27 (ADR 0004, Brian 2026-11-23). Rules §8.3.2 now: when the player whose
fourth open set was fed by another player wins by **self-draw**, the feeder pays 3 shares and
nobody else pays; when that player wins on a **discard** (or by robbing a kong), the discarder pays
1 share as in §8.2 and the feeder pays nothing (also when the feeder is the discarder: 1 share).
Flower wins stay without liability. From 17 Oct to 23 Nov the rules file, the engine and the oracle
charged the feeder 3 shares on a discard win too.

## What changed

* Rules: `docs/rules/hk.md` §8.3 point 2.
* Engine: `State::finish` in `crates/hk/src/state.rs` (liability only on `self_drawn`).
* Oracle: `python/tests/reference/hk_reference.py` `_finish`.
* Tests: `crates/hk/tests/scenarios.rs` `twelve_tile_liability_not_on_a_discard_win_8_3` (seat 2
  feeds seat 1's fourth pung; seat 1 wins on seat 3's discard: seat 3 pays 8, seat 2 pays 0; the
  same hand self-drawn: seat 2 pays 48 = 3 × 16) and `twelve_tile_liable_discarder_pays_one_share_8_3`;
  `crates/hk/tests/properties.rs` asserts that every liable win in 3,000 random hands is
  self-drawn. The engine and the oracle agree on 10,000 random deals (seeds 0–9,999 of
  `test_differential.play_both`, rules drawn at random); 238 of those deals end in a discard win
  whose winner has a liable seat, so the comparison covers the changed case.
* Bots, wherever a value counts liability:
  * `danger.rs`: a discard win costs me 1 share whoever is liable (the `shares` multiplier, 1 / 3 /
    0 before, is gone). This is the expected loss `stage` and the defence bots read.
  * `defence.rs` (Brian's EV bot): the liability term counts only the opponent's self-draw
    (`P(o self-draws | after t) × o's self-drawn points × 2`).
  * `ev.rs`: `L_disc = 1` always, `L_sd = 3` when I am liable (0 when another seat is), `L_other
    = 0`; `Shares(me)` no longer changes with liability; the 12-tile lookahead keeps only its
    self-draw part. `docs/ev-bot.md` §1, §2, §5 example 3 and §6 row 3 follow. Example 3 (liable
    for B) now deals in, as example 2 does: tests `position_3_liability_does_not_protect_the_deal_in`
    (`ev_bot.rs`) and `position_3_gives_liability_and_deals_in` (`reasons.rs`).
  * `race.rs`: the same share multipliers; a win of mine is 1 share on a discard and 3 on a
    self-draw whoever is liable; the opponents' fold cost counts the liable seat's 3 shares on my
    self-draw only. `docs/race-bot.md` §3.2, §3.3, §4.3, §8 note 8; the worked example of §9 was
    made under the old rule and is marked so, not redone.
  * The `liable` field of the example hand files comes from the engine and follows it.
    `docs/study-results/examples/safe1-99000316-seat0.json` is rewritten: the bot's game now ends
    with seat 1, the discarder, paying 32 and the bot at 0 (it paid 96 before).

## How often the old rule moved points

`crates/search/examples/liability_count.rs`, 40,000 deals of four `stage` bots, seed 27, default
rules, run directly (380 s; output `runs/liability/count-40000.txt` in the slot 2 worktree):

| | Count | Share of hands |
|---|---|---|
| Hands | 40,000 | |
| Wins | 38,391 | |
| Wins on a discard | 29,606 | |
| Self-drawn wins with a liable seat (paid the same under both rules) | 381 | 0.95 % |
| **Discard wins with a liable seat (paid differently)** | **1,405** | **3.51 %** |
| of which the liable seat itself discarded (3 shares then, 1 now) | 518 | 1.30 % |

On those 1,405 discard wins the liable seats paid 129,624 points under the old rule (3.24 points
a hand over the table); under the new rule the discarders pay 43,208 (1.08 a hand). So in about one
hand in 28 the old rule put 3 shares (a mean of 92 points) on the wrong seat. Per seat that is
about 0.8 points a hand moved, which is larger than most of the paired gains measured between bots
(±0.03 to ±0.6 a hand) when the two bots differ in how often they feed a fourth open set or how
they play against a seat with one: Task 24's report puts 0.05–0.23 points a hand of the defence
bots' false-alarm loss on "liable for a win". Matches between copies of one bot are not biased
by it (every seat feeds as often), but their spread is wider.

## Made under the old rule

Every game played before 2026-11-23 (commit 5041a1a) settled discard wins with a liable seat the
old way. Nothing is retrained or rerun here, except the two matches below.

Trained networks in `models/` (all made under the old rule):

| File | What the old rule touched |
|---|---|
| `cnn-distill.pt` / `.mjnn`, `cnn-distill-fp32.mjnn`, `cnn-distill-v2.pt` / `.mjnn`, `cnn-distill-v2-fp32.mjnn` | the **value head's targets** are the hands' final deltas (`labels.rs` `deltas`, `labels.py` `value_targets`), wrong in about 3.5 % of hands; the policy imitates `stage`, whose danger read the old shares |
| `cnn-ppo.pt` / `.mjnn`, `ppo.pt` / `.mjnn`, `ppo0.pt` / `.mjnn` | rewards (final deltas, paired with the baseline copy) and the value head trained on them |
| `dmc.pt` / `.mjnn`, `dmc0.pt` / `.mjnn` | Monte Carlo returns (final deltas) |
| `distill.pt` / `.mjnn`, `cnn-test.mjnn`, `cnn-test-fp32.mjnn` | imitation of bots that valued liability the old way; value targets where the file has a value head |
| `belief.pt` / `.mjnn`, `belief-v2.pt` / `.mjnn`, `belief-order.mjnn`, `belief-order-reg.mjnn`, `race-belief.mjnn` | no payment targets (ready state, waits, faan, winner); only the logged games' play, through the bots' old liability values, which is a small effect |

The self-play runs of Tasks 4–10 (`runs/rl-stage/` and `runs/ppo/` in the standalone checkout and
the workers clone) and their checkpoints were trained on old-rule rewards in the same way.

Saved results (every match, tournament and label set before 2026-11-23): all of
`docs/study-results/` except `rules-check.md` and this file, in particular `defence-ceiling.md`
(Task 23), `defence-false-alarms.md` (Task 24), `belief-ready-read.md` (Task 20's defence trials),
`flush-suit-neighbours.md`, `stage3-gate.md`, `stage3-giveup.md`, `stage3-pushfold.md`,
`three-stage-bot.md`, `self-play-network.md`, `2026-11-17-self-play-analysis.md`,
`review-self-play-network.md`, `risk-network.md`, `exact-danger.md`, `race-bot.md`,
`exploiter.md`, `phase1-solver.md`, `phase3-rl.md`, `phase4-build.md`, `review-fix.md`,
`teacher-in-app.md`, the JSON files in `docs/study-results/data/`, the label sets and logs under
`runs/` in the checkouts, and the other example hand files Task 25 and 26 wrote outside git.

## Matches re-run under the new rule

Queued on the cpu lane from the slot 2 worktree, binaries built from the new engine in
`C:\Users\brian\.cache\hk-mahjong-t27\release\examples`:

* job 309, Task 23's ceiling match, `defence_ceiling match 40000 23`
  (`experiments/stage/defence_ceiling.ps1`), output `runs/defence-ceiling/match-40000-seed23-selfdraw-rule.txt`;
* job 310, Task 24's false-alarm match, `defence_false_alarms 40000 24` with the same four bots
  (`experiments/stage/false_alarms.ps1 -Bin ...`), output
  `runs/defence-false-alarms/match-40000-seed24-selfdraw-rule.txt`.

Check: `jobq.ps1 status`, `jobq.ps1 log 309`, `jobq.ps1 log 310`. When they end, the figures go
beside the old ones in `defence-ceiling.md` and `defence-false-alarms.md`, each named with its rule.
