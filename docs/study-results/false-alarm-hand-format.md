# The false-alarm example hands as files

Plan 2026-11-16, Task 25. `defence_false_alarms examples-json <out dir> <deals> [seed] [model]
[bot ...]` (`crates/search/examples/defence_false_alarms.rs`) writes one JSON file for each
example hand the report prints (the same choice of examples, per bot), named
`<bot>-<deal>-seat<n>.json`. The builder is `example_json` in `crates/search/src/false_alarm.rs`.
The tile names (`5p`, `E`, `Wh`, `F2`) and the actions (`action.*`, with the engine's action
`code`) are those of the app's engine (`crates/app-api/README.md`).

Each hand is one deal played twice on the same wall: the *plain* game (four plain `stage` bots)
and the *bot* game (the defence bot in `seat`, three plain `stage`). The two action lists are the
same up to `first_change_index`, where the bot throws something other than `stage`'s pick.

| Field | Meaning |
|---|---|
| `format` | File layout version, 1. |
| `deal` | The deal id (`seed * 1,000,003 + index`); the wall is its shuffle. |
| `bot` | The bot's short name (`safe1`, `safe5`, `ev`, `fold`). |
| `seat` | The bot's seat, 0–3. |
| `dealer` | The dealer's seat, always 0. |
| `seat_winds` | The wind of each seat: `["E","S","W","N"]`. |
| `prevailing_wind` | The round's wind (`E`). |
| `wall` | All 144 tiles in draw order from the front (replacements come off the back); `State::from_wall` in the engine starts the hand from it. |
| `plain_actions` | Every action of the plain game from the first draw to the end, all seats, in order, claim passes included. Each is `{"type": "action.discard", "tile": "5p", "code": 13}`; `action.pung` and `action.kong` and `action.pass` carry no tile; `action.chow` carries its lowest tile and `tiles` (the three tiles). |
| `bot_actions` | The same for the bot's game. |
| `first_change_index` | Index in both lists of the first action that differs. Before it both games are identical. |
| `label` | `false alarm` (`stage`'s pick would not have been won on) or `true alarm` (it would, with every possible winner declaring). The examples are false alarms. |
| `ending_kind` | How the hand ended beside the plain game: `liable for a win`, `win lost`, `later deal-in`, `other's self-draw`, `same ending`, `better ending` (the report's kinds). |
| `change` | The position at the first change (below). |
| `plain_ending`, `bot_ending` | The ending of each game (below). |
| `plain_points`, `bot_points` | The bot seat's points in each game. |

`change`:

| Field | Meaning |
|---|---|
| `stage_pick` | The tile plain `stage` throws. |
| `thrown_instead` | The tile the bot throws. |
| `progress_steps_cost` | Steps from ready (after the discard, 13 tiles) the replacement costs: distance after `thrown_instead` minus after `stage_pick`. 0: no progress lost. |
| `own_distance` | The bot's distance from ready (best discard). |
| `wall_remaining`, `wall_third` | Tiles left in the wall, and the wall third (1–3) the change falls in. |
| `paid_at_once_if_picked` | Points `stage`'s pick would have cost at once (0 for a false alarm). |
| `seats` | Four entries, from the bot's seat on (`bot`, `next`, `opposite`, `previous`), each: `seat`, `role`, `wind`, `hand` (concealed tiles, text notation), `open_sets`, `discards` (in order; `claimed` when another seat took it), `true_distance` (the seat's real faan-aware distance from ready, 0 = ready; for the bot its own distance), `read_dealin_on_pick` (the reading's deal-in figure for `stage`'s pick on that seat, null for the bot). |

An ending (`plain_ending`, `bot_ending`): `draw` (wall ran out), `points` (all four seats, sum 0),
and `wins`, one per winner: `winner`, `from` (the discarder, null for a self-draw), `self_drawn`,
`faan`, `win_tile`, `paid_by` (what each seat paid), `liable` (the seat that paid for the whole
table under the 12-tile rule, or null), `robbed`.

## Worked example: `examples/safe1-99000316-seat0.json`

Seed 99, 40 deals, bot `safe1` (`stage-safe:t=1`), deal 99000316, seat 0
(`defence_false_alarms examples-json out 40 99 models/belief-order.mjnn safe1`). It is the first
example the report prints:

* Position at action 37 of both lists (58 tiles left, wall third 2): the bot (seat 0, wind E) holds
  `119m 5p 12377899s E N`, 3 from ready; discards `8p 8p 9p(claimed) 4m 7m 1p 9p`.
  Next (seat 1): `5m 333466p S N Wh`, open `999p`, truly 3 from ready, read 0.001.
  Opposite (seat 2): `11m 46p WW R`, open `333m 666s`, truly 2 from ready, read 0.254.
  Previous (seat 3): `559m 122888s W N Wh G`, truly 4 from ready, read 0.000.
* `stage_pick` 5p, read 0.254 on the opposite seat (nobody wins on it: `label` false alarm,
  `paid_at_once_if_picked` 0); `thrown_instead` 9m, `progress_steps_cost` 0.
* Plain game: seat 2 wins off seat 1 with 5 faan on R; the bot's points 0.
  Bot game: seat 2 wins off seat 1 with 6 faan on R; the bot is liable (`liable`: 0) and pays
  96 (`bot_points` −96). `ending_kind`: `liable for a win`.
