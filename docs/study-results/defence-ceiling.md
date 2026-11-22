# The ceiling of defence before ready (plan 2026-11-16 Task 23)

Brian, 2026-11-22, asked "What to do about the defensive part" and chose (a): measure the upper
bound first. This study measures how many points a hand perfect knowledge of the deal-in is worth,
with a bot that cheats, and how much of it a reading as good as `models/belief-order.mjnn` could
reach.

## Set-up

* Bot `stage-oracle` (`crates/search/src/oracle.rs`, test-only: not an agent spec, never a default,
  never in the app). It plays as plain `stage`. At a discard decision it reads the true hands: if
  `stage`'s own pick would be won on at once (the engine settles the claim window with every
  possible winner declaring, liability included) and the hand holds a tile that would not, it throws
  `stage`'s pick among the tiles that would not (`StageBot::pick_discard` on the safe tiles only).
  The re-pick draws its tie-breaks from its own generator, so each oracle game is the plain game
  until the first change: checked, 0 unchanged hands differ from the plain game in every variant.
* "From ready" is the faan-aware distance of the 14-tile hand after its best discard, the measure
  `stage` uses for its stages (0 = ready with a legal win).
* Variants: `far` (two or more from ready), `notready` (one or more), `all` (every decision),
  `far-big` (as `far`, avoiding only deal-ins of 6 faan or more).
* Match: `crates/search/examples/defence_ceiling.rs match 40000 23`. Each deal is played with four
  plain `stage` (one game per seat) and with each variant in each seat against three plain `stage`,
  the set-up of `examples/defence_match.rs` (seed 23). Paired = the variant's points minus plain
  `stage`'s in the same seat on the same wall; the 95 % range is over deals (the four seats of a deal
  are one sample). Run directly on the orchestrator's word: 6,412 s (1 h 47 min) on 14 threads,
  sharing the cores with a poker cpu job. Output `runs/defence-ceiling/match-40000-seed23.txt` in the
  workers clone.
* Test: `oracle::tests::oracle_never_throws_a_dealing_tile_while_holding_a_safe_one` plays 10 deals
  per variant and checks every discard of the oracle seat from scratch.

## Results (40,000 deals × 4 seats)

| Bot | Paired − plain `stage` (a hand) | Win | Deal-in | Other self-drew | Other's deal-in | Draw | Hands changed | Decisions changed | Per changed decision |
|---|---|---|---|---|---|---|---|---|---|
| plain `stage` | — | 24.08 % | 18.03 % | 16.61 % | 35.86 % | 5.41 % | — | — | — |
| `far` | **+0.598 ± 0.036** | 24.11 % | 14.07 % | 17.77 % | 38.34 % | 5.70 % | 5.89 % | 0.740 % | +5.16 |
| `notready` | **+1.504 ± 0.057** | 24.24 % | 8.93 % | 19.16 % | 41.18 % | 6.49 % | 11.63 % | 1.532 % | +6.21 |
| `all` | **+3.254 ± 0.082** | 24.94 % | 1.02 % | 21.10 % | 44.79 % | 8.14 % | 17.40 % | 2.416 % | +8.41 |
| `far-big` | **+0.374 ± 0.030** | 24.09 % | 17.22 % | 16.86 % | 36.36 % | 5.48 % | 1.30 % | 0.162 % | +14.90 |

So the one-away hands add +0.91 a hand (`notready` − `far`) and the ready hands +1.75 (`all` −
`notready`). Task 12 put the ready hand's push-or-fold oracle at +0.87 a decision (about +0.55 a
hand); avoiding the dealing tile at every ready decision is worth +1.75 a hand, about three times
that, because it keeps pushing with a safe tile instead of folding. `far-big` keeps 63 % of `far`'s
gain from 22 % of its changes: the big hands carry much of it.

### Where the points come from

A change happens only where the plain game deals in at once (the two games are equal up to that
point), so the gain splits into the deal-in avoided and what the rest of the hand then brings the
oracle. Points a hand, over all hands:

| | `far` | `notready` | `all` | `far-big` |
|---|---|---|---|---|
| Deal-in avoided at the change | +1.598 | +3.222 | +4.888 | +0.815 |
| then: the oracle wins | +0.017 | +0.077 | +0.383 | +0.004 |
| then: the oracle deals in later | −0.540 | −0.716 | −0.092 | −0.269 |
| then: another seat self-draws | −0.438 | −0.997 | −1.779 | −0.161 |
| then: another seat's deal-in / draw | −0.040 | −0.083 | −0.145 | −0.016 |
| **Gain** | **+0.598** | **+1.504** | **+3.254** | **+0.374** |

Out of the deal-ins avoided, the oracle wins the hand rarely (60 of 9,431 changed hands in `far`);
the hand mostly goes to another seat's deal-in (no cost) or self-draw (it still pays). In `far` and
`notready` the oracle deals in later in 33 % and 22 % of the changed hands, on a later decision outside
its range or with no safe tile. No plain win is lost: plain had already dealt in.

By the first change (points a hand; hands):

| | `far` | `notready` | `all` | `far-big` |
|---|---|---|---|---|
| wall third 1 | +0.036 (682) | +0.051 (808) | +0.067 (857) | +0.020 (128) |
| wall third 2 | +0.333 (6,231) | +0.763 (10,893) | +1.468 (14,431) | +0.202 (1,252) |
| wall third 3 | +0.229 (2,518) | +0.690 (6,901) | +1.719 (12,551) | +0.152 (694) |
| threat with 0 open sets | +0.081 (473) | +0.174 (916) | +0.339 (1,360) | +0.071 (266) |
| 1 open set | +0.084 (1,809) | +0.234 (3,566) | +0.520 (5,287) | +0.039 (284) |
| 2 open sets | +0.210 (3,807) | +0.517 (7,492) | +1.146 (11,178) | +0.117 (715) |
| 3 open sets | +0.196 (3,206) | +0.497 (6,269) | +1.057 (9,383) | +0.125 (742) |
| 4 open sets | +0.028 (136) | +0.082 (359) | +0.192 (631) | +0.023 (67) |

The gain is in the middle and last thirds of the wall, against a threat with two or three open sets.

## What a reading could reach

`defence_ceiling reading models/belief-order.mjnn 40000 23`: the four-`stage` games of the same
deals (seat 0's plain games), every discard decision of every seat (2,483,684 decisions, 2,630 s).
At each, the reading's deal-in chance of a tile is `1 − Π_o (1 − dealin_o)` over the three
opponents (Task 19's `dealin` head). Positives are the decisions the oracle would change (the pick
deals in, a safe tile exists); negatives are the decisions whose pick is safe. Output
`runs/defence-ceiling/reading-40000-seed23.txt`.

| Range | Decisions | Oracle would change | AUC (dealing vs safe picks) | Pick read above … of the safe tiles | Reading's safest tile truly safe |
|---|---|---|---|---|---|
| `far` | 1,870,285 | 9,409 (0.50 %) | 0.865 | 71.2 % | 91.5 % |
| `notready` | 2,246,543 | 18,722 (0.83 %) | 0.873 | 73.8 % | 91.6 % |
| `all` | 2,483,684 | 27,998 (1.13 %) | 0.871 | 74.4 % | 89.7 % |
| `far-big` | 1,870,285 | 2,035 (0.11 %) | 0.864 | 72.0 % | 92.5 % |

A real bot can only flag a pick when its read chance is above a threshold. Share of the dealing
picks' deal-in points it catches with a truly safe replacement (the reading's safest tile), against
the share of safe picks it flags by mistake, and the false flags a hand that costs:

| Safe picks flagged | `far`: share caught | false flags a hand | `notready`: share caught | false flags a hand |
|---|---|---|---|---|
| 1 % | 29.5 % | 0.12 | 25.3 % | 0.14 |
| 5 % | 62.2 % | 0.58 | 58.3 % | 0.70 |
| 10 % | 76.2 % | 1.16 | 73.7 % | 1.39 |
| 20 % | 84.9 % | 2.33 | 84.3 % | 2.78 |

At best, if a false flag cost nothing, a reading as good as `belief-order.mjnn` reaches about 30 %
of the ceiling at 1 % false flags (`far` +0.18, `notready` +0.38 a hand) and about 60 % at 5 %
(`far` +0.37, `notready` +0.88). Each false flag throws a worse tile for offence; the gain survives
only if a false flag costs less than about 1.5 points (`far`, 1 %) or 2.7 points (`notready`, 1 %),
0.64 / 1.25 at 5 %. These shares take the reading check's positives (plain games) as standing for
the oracle's changes, which also include later decisions of hands the oracle kept alive.

## Limits

* This is the ceiling of "avoid the dealing tile" only. It does not see the cost of an opponent's
  self-draw that a discard leaves open (decision B15) or of feeding a claim; the self-draw losses it
  suffers after a change are in the measured points (they are why only 37 % (`far`) to 67 % (`all`) of the avoided
  deal-in remains). Variant 5 (never feed a claim that makes an opponent ready) was not built: it
  needs a ready check of each claimant's hand after every claim it could make, over an hour of work.
* The bot sees the true hands, so it is no evidence that any real bot gains; it says how much room
  there is.
* The cost of a false flag was not measured; Task 12 found that folding a ready hand costs +18.8
  points a decision, which is why the ready range is hard; for a hand two or more from ready the cost
  should be much smaller, but only a match can say.

## Verdict

Perfect knowledge of the deal-in is worth +0.60 ± 0.04 points a hand before one-away (+1.50 ± 0.06
for every hand that is not ready, +3.25 ± 0.08 with ready hands too), far more than the level
results so far; a reading as good as `belief-order.mjnn` (AUC 0.87 on those decisions) could reach
at best about 30 % of it (+0.18 / +0.38 a hand) flagging 1 % of safe picks and about 60 % (+0.37 /
+0.88) flagging 5 %, if its false flags cost little.
