# mahjong-app-api

The app-facing adapter over the Hong Kong mahjong engine. It implements the calls of hk-games
`docs/engine-interface.md` (contract v0) as JSON strings in and out, and builds to
WebAssembly with `wasm-bindgen`. It holds no game logic of its own: rules, claims, scoring,
bots, the danger model, hand mathematics and the PIMC solver come from `crates/hk`,
`crates/analysis` and `crates/search`.

Every response carries `"api": 0`. Every error is
`{"api": 0, "error": {"code": "error.*", "detail": "<developer text>"}}`.
The engine returns keys and numbers only, never display text: tiles are ids such as `"5p"`
(with a `tile.*` key for each), faan patterns `faan.*`, actions `action.*`, melds `meld.*`,
reasons `reason.*`, events `event.*`. The Chinese labels of the lab UI (`python/mahjong/hk.py`)
are the app's dictionary job.

## Build and test

```powershell
cargo test -p mahjong-app-api                       # the JSON calls end to end (debug: ~8 s)
cargo test -p mahjong-app-api --release -- --ignored --nocapture timing   # the timings below
cargo clippy -p mahjong-app-api --all-targets -- -D warnings
cargo clippy -p mahjong-app-api --target wasm32-unknown-unknown -- -D warnings
wasm-pack build crates/app-api --target web --release                      # -> crates/app-api/pkg (gitignored)
wasm-pack build crates/app-api --target nodejs --release --out-dir <dir>   # for a Node smoke run
```

On this PC, App Control sometimes blocks a freshly built executable (build script, test, os
error 4551); running the same command again passes. A `CARGO_TARGET_DIR` outside OneDrive
helps.

The WebAssembly class is `MahjongApi` (`new MahjongApi()`); its methods are the calls below
with the same names. Strings in, strings out; `seat`, `budget_ms` and `samples` are plain
numbers. Native Rust uses `mahjong_app_api::Api` with the same methods.

## Calls

| Call | Input | Output |
|---|---|---|
| `meta()` | — | `GameMeta` plus `tiles`, `faan` and `actions` key lists |
| `new_game(config)` | `GameConfig` | `{game_id}` |
| `view(game_id, seat)` | — | `SeatView` |
| `legal_actions(game_id)` | — | `{actions: [Action], seat, phase}` for the seat to act (empty once the hand is over) |
| `apply(game_id, action)` | `Action` | `{events: [Event], view: SeatView}` (the view is for the seat that acted) |
| `bot_action(game_id, level, budget_ms)` | level from `bot_levels` | `Action` plus `budget_used_ms` (not applied: call `apply` with it) |
| `hint(game_id, seat, budget_ms)` | seat must be to move | `Hint` with the study extras |
| `hint_with_samples(game_id, seat, budget_ms, samples)` | as `hint` | `Hint` with an explicit sample count (measurements, drills) |
| `review(game_id, seat, budget_ms)` | — | `{items: [ReviewItem], seat, budget_used_ms}` |
| `grade(position, action, budget_ms)` | `Position`, `Action` | `Grade` |
| `snapshot(game_id)` | — | `Position` |
| `restore(position)` | `Position` | `{game_id}` |
| `drop_game(game_id)` | — | `{}` |

Two shapes differ from the contract's table because every response must be an object with
`api`: `legal_actions` wraps the list in `{actions}` and `review` wraps it in `{items}`.

`budget_ms` is a ceiling in milliseconds; `0` means no limit. The solver gets 80% of it and
checks the clock between batches of 4 samples; the rest covers the last batch and the study
figures. The review also caps each item's samples from the speed of the previous item. Every
search runs on one thread: natively in a one-thread rayon pool, in the browser on rayon's
single-thread fallback (checked by the Node smoke run).

### Who acts when

A hand has three decision phases (`phase` in views and `legal_actions`):

- `turn`: the seat to move may win by self-draw, declare a kong or discard.
- `claim`: after a discard, each other seat that has a claim available declares one (`win`,
  `pung`, `kong`, `chow`, or `pass`), one seat at a time in turn order. Seats with nothing but
  `pass` are skipped. A declaration is hidden from the other seats until the window closes.
- `rob`: after a kong, seats that could win on its tile declare `win` or `pass`.

Draws happen inside `apply`; the drawn tile is in the mover's view as `last_drawn`.

### Randomness

Everything is seeded. `new_game` shuffles the wall from `seed`. Each decision draws its own
generator from `(seed, number of actions so far, call)`, so a hand restored from a snapshot
continues exactly as the original would: the random bot picks the same move, the solver runs
the same search.

## Shapes

### Tiles

A tile id is the engine's text notation: `1m`–`9m` characters 萬, `1p`–`9p` dots 筒, `1s`–`9s`
bamboo 索, `E S W N` winds, `Wh G R` dragons (white, green, red), `F1`–`F4` flowers, `S1`–`S4`
seasons. `meta().tiles` maps every id to its key: `tile.5_dots`, `tile.1_characters`,
`tile.9_bamboo`, `tile.east` … `tile.north`, `tile.white`, `tile.green`, `tile.red`,
`tile.flower_1` … `tile.season_4`. Hands are listed in index order (characters, dots, bamboo,
winds, dragons), repeats kept.

### GameMeta

```json
{ "api": 0, "game": "mahjong", "seats": [4, 4],
  "variants": { "flowers": ["on", "off"], "min_faan": ["3", "0", "1", "2", "4", "5"], "faan_cap": ["10", "13", "8"],
                "payout": ["half_spicy", "full_spicy"], "multiple_winners": ["all", "head_bump"],
                "twelve_tile_liability": ["on", "off"], "prevailing_wind": ["east", "south", "west", "north"] },
  "bot_levels": ["random", "easy", "medium", "hard", "expert"],
  "has_hint": true, "has_review": true, "has_grade": true,
  "tiles": { "1m": "tile.1_characters", "...": "..." },
  "faan": [ { "key": "faan.self_drawn", "number": 1, "limit": false }, "..." ],
  "actions": ["action.discard", "action.self_drawn_win", "action.concealed_kong", "action.added_kong",
              "action.pass", "action.win", "action.pung", "action.kong", "action.chow"] }
```

Each variant lists its options default first; `min_faan` and `faan_cap` list the usual choices
and accept any number 0–13 (with `min_faan ≤ faan_cap`). See `docs/rules/hk.md` for meanings.

### GameConfig

```json
{ "seats": 4, "variants": { "flowers": "off", "min_faan": 1, "prevailing_wind": "south" }, "seed": 12345, "human_seats": [0] }
```

`seats` must be 4 (`error.bad_config` otherwise). Variants left out keep the default; an
unknown key or value is `error.unknown_variant`. Switches accept `"on"`/`"off"` or
`true`/`false`; numbers accept a number or a numeric string; `prevailing_wind` accepts a wind
name or a seat index 0–3. `human_seats` is stored and echoed; the engine does not act on it.
Seat 0 is always the dealer (East).

### Action

```json
{ "type": "action.discard", "tile": "5p", "code": 13 }
{ "type": "action.self_drawn_win", "code": 34 }
{ "type": "action.concealed_kong", "tile": "E", "code": 62 }
{ "type": "action.added_kong", "tile": "E", "code": 96 }
{ "type": "action.pass", "code": 103 }
{ "type": "action.win", "code": 104 }
{ "type": "action.pung", "code": 105 }
{ "type": "action.kong", "code": 106 }
{ "type": "action.chow", "tile": "3p", "tiles": ["3p", "4p", "5p"], "code": 118 }
```

`code` is the engine's stable action code (0–133, `mahjong_hk::Action::code`, the same
numbers the Python server uses); on output it is always present, on input it is accepted on
its own when `type` is left out. A chow is named by its lowest tile; on input `tiles` (the
three tiles) may be given instead of `tile`. `action.kong` is the claim of a discard with three
in hand; the two kongs declared on one's own turn are `action.concealed_kong` and
`action.added_kong`.

### SeatView

Only what `seat` may see. Other hands appear only once the hand is over.

```json
{ "api": 0, "game_id": "g1", "seat": 0,
  "rules": { "flowers": "on", "min_faan": 3, "faan_cap": 10, "payout": "half_spicy",
             "multiple_winners": "all", "twelve_tile_liability": "on", "prevailing_wind": "east" },
  "seed": 12345, "human_seats": [0],
  "phase": "turn", "current_player": 0, "your_turn": true, "turn": 0, "wall_remaining": 90,
  "hand": ["3m", "6m", "8m", "1p", "..."], "last_drawn": "8m",
  "melds": [[ { "kind": "meld.pung", "tile": "E", "tiles": ["E", "E", "E"], "concealed": false, "from": 3 } ], [], [], []],
  "bonus": [["F1"], [], ["S2", "S3"], []],
  "concealed_counts": [14, 13, 13, 13],
  "river": [ { "player": 0, "tile": "9s", "claimed": false }, "..." ],
  "claim": null, "liable": [null, null, null, null],
  "legal": [ Action, "..." ],
  "history": [ { "player": 0, "action": Action }, "..." ],
  "over": false, "result": null, "final_hands": null }
```

- `turn` is the seat whose turn it is in turn order (the discarder during a claim window);
  `current_player` the seat to act (a declaring seat during a window).
- `last_drawn` is set only in the mover's own view, on a turn that began with a draw.
- `river`: every discard still lying on the table, in order; `claimed` marks one taken into a
  meld (it stays listed so the app can fade it, as the lab UI does).
- `claim`: `{tile, from}` while a claim or rob window is open.
- `liable`: for each seat, the seat liable for it under the 12-tile rule (§8.3), or `null`.
- `legal` is filled only when it is this seat's turn.
- `history`: every action so far. Another seat's declaration in a window that is still open
  is `{"player": p, "action": null}`.
- When `over`: `result` (below) and `final_hands` (all four concealed hands).

### Result

```json
{ "wins": [ { "seat": 2, "score": { "faan": 4, "total": 4, "qualifying": 3, "limit": false,
                                    "patterns": [ { "key": "faan.half_flush", "faan": 3 }, { "key": "faan.seat_flower", "faan": 1 } ] },
              "points_per_share": 16, "self_drawn": false, "from": 1, "win_tile": "7p",
              "paid_by": [0, 16, 0, 0], "liable": null, "robbed": false } ],
  "deltas": [0, -16, 16, 0] }
```

`wins` is empty after an exhaustive draw. `qualifying` is the faan without the bonus-tile
patterns (what must reach `min_faan`). `deltas` sums to zero.

### Event

`apply` returns the events in order, with the engine's public events (draws never reveal the
tile) plus two of this crate's own:

```json
{ "type": "event.declared", "player": 1, "action": Action }          // the acting seat's own window declaration
{ "type": "event.draw", "player": 1, "replacement": false }
{ "type": "event.bonus", "player": 1, "tile": "F2" }
{ "type": "event.discard", "player": 1, "tile": "5p" }
{ "type": "event.claim", "player": 2, "from": 1, "meld": { "kind": "meld.pung", "tile": "5p", "tiles": ["5p", "5p", "5p"], "concealed": false, "from": 1 } }
{ "type": "event.concealed_kong", "player": 2, "tile": "E" }
{ "type": "event.added_kong", "player": 2, "tile": "E" }
{ "type": "event.win", "player": 2 }
{ "type": "event.exhaustive_draw" }
{ "type": "event.hand_over", "result": Result }
```

A discard that nobody can claim is followed in the same `apply` by the next seat's
`event.draw`. A window's resolution (claim, win, next draw) comes with the last declaration.

### Position

The whole hidden state: the full tile set in deal order plus the history. The other fields
describe the resulting position and are ignored by `restore`.

```json
{ "api": 0, "game": "mahjong",
  "rules": { "...": "..." }, "seed": 12345, "human_seats": [0],
  "wall": ["4s", "E", "1m", "..."],
  "history": [ { "player": 0, "action": Action }, "..." ],
  "phase": "claim", "current_player": 2, "hands": [["..."], ["..."], ["..."], ["..."]],
  "wall_remaining": 61, "over": false, "result": null }
```

`wall` must hold exactly the tiles in play under `rules` (144, or 136 with flowers off): the
first 14 go to the dealer, the next 13 to each other seat, draws come from the front and
replacements from the back (`docs/rules/hk.md` §3). A wall that is not a full set or a history
that does not replay is `error.bad_position`. The engine has no constructor for a mid-hand
position without a wall, so a drill position is a snapshot of a hand (seeded deal plus
history); Task 18 can search seeds for the positions it wants.

### Hint

Top 3 candidates, best first, from PIMC (up to 400 samples within the budget, over the
engine's shortlist: every special action, the best 6 discards by the faan-aware measure, and
the `value` bot's choice as baseline), plus the study-mode figures the lab UI shows.

```json
{ "api": 0, "seat": 0, "phase": "turn", "samples": 220, "budget_used_ms": 2438,
  "baseline": Action,
  "moves": [
    { "action": { "type": "action.discard", "tile": "8m", "code": 7 }, "score": -0.98, "win_rate": 0.168,
      "numbers": { "deal_in_rate": 0.15, "draw_rate": 0.2, "score_ci95": 4.1, "samples": 220,
                   "diff": 1.3, "diff_se": 0.9, "baseline": false },
      "reasons": [ { "key": "reason.keeps_faan_path", "args": { "faan_shanten": 5, "faan": 6, "target": Target } },
                   { "key": "reason.loses_shanten", "args": { "shanten": 3, "best": 2 } },
                   { "key": "reason.score_margin", "args": { "points": 2.3, "se": 0.9 } } ] } ],
  "progress": { "shanten": 2, "faan_shanten": 5, "target": Target },
  "danger": { "tiles": { "3m": 0.004, "8m": 0.012, "...": 0 }, "threat": [0, 0.31, 0.05, 0.12] },
  "discards": [ { "tile": "8m", "shanten": 3, "waits": 47, "faan_shanten": 5, "faan_waits": 39, "target": Target, "danger": 0.012 }, "..." ] }
```

- `score` is the expected points for the seat; `win_rate`, `deal_in_rate` and `draw_rate` are
  fractions of the samples; `score_ci95` is the 95% half-width of `score`; `diff` and
  `diff_se` are the paired difference from the baseline and its standard error (far more
  precise than the means, because every candidate is played on the same samples).
- `progress`: `shanten` (exchanges to a ready hand; −1 complete), `faan_shanten` (the same
  toward a hand that meets `min_faan`; `null` when no plan reaches it) and its `target`:
  `{ "plan": { "kind": "plan.half_flush", "suit": "suit.bamboo" }, "structure": "structure.any",
  "pungs": ["Wh"], "faan": 6, "limit": false }`. Plan kinds: `plan.mixed`, `plan.half_flush`,
  `plan.full_flush`, `plan.honours`, `plan.orphans` (suit `suit.characters|dots|bamboo`);
  structures `structure.any`, `structure.all_pungs`, `structure.all_chows`; `pungs` are the
  scoring pungs the plan needs; `faan` what it guarantees.
- `danger`: the heuristic danger model, from public information only: for each tile in the
  hand, the chance that discarding it lets an opponent win; `threat`, each seat's chance of
  waiting on a legal win (0 for the seat itself).
- `discards`: the discard table (turn phase only, otherwise `[]`): for every discard, shanten
  and acceptance (`waits`, unseen tiles that lower it), faan-aware shanten and its acceptance,
  the target and the tile's danger.

A forced decision (one legal action) returns that one move with `score: null`,
`reason.forced` and no search.

### ReviewItem

One per decision the seat has made so far (the review may run before the hand ends).

```json
{ "turn": 17, "player": 0, "phase": "turn", "played": Action, "best": Action,
  "loss": 2.4, "grade": "good",
  "reasons": [ "..." ], "best_reasons": [ "..." ],
  "numbers": { "played_score": -3.1, "best_score": -0.7, "played_win_rate": 0.14, "best_win_rate": 0.18,
               "played_deal_in_rate": 0.21, "best_deal_in_rate": 0.16, "loss_se": 1.5, "samples": 120 } }
```

`turn` indexes `history`. `reasons` explain the move played (position facts plus the solver's
verdict); `best_reasons` explain the best move. The review uses PIMC with up to 300 samples
and 8 shortlisted discards per decision, with the played move added to the candidates, so it
is scored on the same samples as the best one; `loss` is the paired difference. A forced move
gets `grade: "best"`, `loss: 0`, `reasons: [reason.forced]`, empty `numbers` and no search. The
budget is shared equally among the decisions still to review.

### Grade

```json
{ "api": 0, "seat": 0, "grade": "inaccuracy", "loss": 4.2, "best": Action,
  "reasons": [ "..." ], "best_reasons": [ "..." ],
  "numbers": { "played_score": -2.0, "best_score": 2.2, "...": "...", "loss_se": 1.1, "samples": 220 },
  "budget_used_ms": 2400 }
```

`grade` runs the hint search (PIMC, up to 400 samples within the budget, the graded action
added to the candidates) on the position and compares `action` with the solver's first
choice. An illegal `action` is `error.illegal_action`. The position is not registered as a
game.

## Grade thresholds

`loss` is the best move's expected points minus the played move's, from the paired samples.
Under the default rules the smallest win (3 faan, half-spicy) is worth 8 points per share.
The grade is read from the part of the loss that is more than sampling noise:
`max(0, loss − 1.96 × loss_se)`, so a short search cannot call a move a blunder.

| Grade | Condition |
|---|---|
| `best` | the solver's first choice |
| `good` | the loss is within noise, or under 1 point |
| `inaccuracy` | under 3 points |
| `mistake` | under 8 points (one share of a minimum win) |
| `blunder` | 8 points or more |

`loss_se` is the standard error of the difference between the two moves (their paired
differences from the baseline added in quadrature: exact when one of them is the baseline,
otherwise an upper bound). The numbers are estimates under the rollout bot's play
(`docs/solver.md` §4 says how far to trust them).

## Bot levels

| Level | Engine bot | What it plays |
|---|---|---|
| `random` | `random` | a uniformly random legal move, except that it always declares a legal win |
| `easy` | `efficiency` | any complete hand as fast as possible, ignoring the faan minimum (the lab UI's 初級) |
| `medium` | `defensive` | hands that meet the minimum; folds to the safest tile when far from ready against a threat (the lab UI's 中級) |
| `hard` | `value` | hands that meet the minimum, never folds (the strongest rule-based bot in `docs/study-results/phase1-solver.md`) |
| `expert` | `pimc:samples=96,candidates=4` | the PIMC solver as the lab UI's 高級 bot: it leaves the `value` baseline only when a candidate beats it by 2 standard errors; stops early at the budget |

The rule-based bots take well under a millisecond per decision. Levels are ordered by points
per hand against the `value` field (`docs/study-results/phase1-solver.md`): random −13.1,
efficiency −10.3, defensive −1.9, value 0.

## Reason keys

Every reason is a fact the code checks on the position or the solver's numbers; none is
guessed. Shanten counts exchanges to a ready hand; "faan path" means the faan-aware plan
(`progress.target`) that reaches `min_faan`.

| Key | Args | Means |
|---|---|---|
| `reason.forced` | — | the only legal move |
| `reason.wins` | `faan`, `points` | declares a win worth this much (self-drawn: three shares) |
| `reason.declines_win` | — | passes although a win was legal |
| `reason.ready_for_faan` | `faan`, `waits` | after this discard the hand is ready and meets the minimum; `waits` unseen tiles complete it |
| `reason.keeps_faan_path` | `faan_shanten`, `faan`, `target` | the discard keeps the best faan-aware shanten among the discards, on this plan |
| `reason.breaks_faan_path` | `faan_shanten`, `best` | another discard would leave the hand closer to a declarable win |
| `reason.no_faan_path` | — | after this discard no plan reaches the minimum |
| `reason.ready_short_of_faan` | `waits` | ready by tile count but the hand cannot be declared (needs a lucky pattern) |
| `reason.best_shanten` | `shanten` | the discard keeps the best plain shanten among the discards |
| `reason.loses_shanten` | `shanten`, `best` | another discard would leave the hand closer to complete |
| `reason.more_waits` | `waits` | among discards with the same shanten, this one accepts the most tiles |
| `reason.fewer_waits` | `waits`, `best` | another discard with the same shanten accepts more tiles |
| `reason.safe_tile` | `danger` | deal-in chance ≤ 1% while an opponent is at least 40% likely to be waiting |
| `reason.dangerous_tile` | `danger` | deal-in chance ≥ 5% |
| `reason.opponent_threat` | `player`, `threat` | an opponent is at least 40% likely to be waiting on a legal win |
| `reason.claim_advances` | `faan_shanten_now`, `faan_shanten_after` | the claim (and best discard after it) lowers the faan-aware shanten |
| `reason.claim_no_gain` | `faan_shanten_now`, `faan_shanten_after` | the claim does not bring a declarable win closer |
| `reason.pass_misses_claim` | `faan_shanten_now`, `faan_shanten_after` | passing although a claim would bring a declarable win closer |
| `reason.pass_keeps_hand` | — | passing; no available claim would bring a declarable win closer |
| `reason.stays_concealed` | — | passing keeps the hand concealed (門前清 still possible) |
| `reason.opens_hand` | — | the first exposed meld: the hand loses 門前清 |
| `reason.scoring_pung` | `faan` | a pung of a dragon, the seat wind or the prevailing wind (#10–#12) |
| `reason.kong_extra_draw` | — | a kong gives a replacement draw (and a chance of 槓上開花) |
| `reason.score_margin` | `points`, `se` | the best move's expected-points lead over the runner-up, beyond 1.96 standard errors |
| `reason.score_loss` | `points`, `se` | how many expected points this move loses against the best, beyond 1.96 standard errors |
| `reason.close_call` | `points`, `se` | the gap to the best move (or the runner-up) is within sampling noise |
| `reason.high_deal_in` | `rate` | the move dealt in to an opponent's win in at least 15% of samples |

## Error codes

| Code | When |
|---|---|
| `error.bad_json` | the input string is not valid JSON of the expected shape |
| `error.bad_config` | `seats` is not 4 |
| `error.bad_seat` | a seat outside 0–3 |
| `error.unknown_game` | no game with that `game_id` |
| `error.unknown_variant` | an unknown rule key or value, or `min_faan` above `faan_cap` |
| `error.unknown_level` | a bot level not in `bot_levels` |
| `error.bad_tile` | a malformed tile id |
| `error.bad_action` | an unknown `type`, a missing tile, or an invalid `code` |
| `error.illegal_action` | a well-formed action the rules do not allow now (tile not in hand, claim outside a window, a win that is not complete or is below the minimum) |
| `error.game_over` | acting, hinting or asking a bot after the hand ended |
| `error.not_your_turn` | `hint` for a seat that is not to act |
| `error.bad_position` | a `Position` that cannot be rebuilt |

## Measured timings

Release builds, one thread, the dealer's opening decision of seed 100, Windows 11 desktop
(2026-10-28). WebAssembly: `wasm-pack --release` with `wasm-opt -O`, Node 22. The opening
decision is the slowest (the longest rollouts, 6 candidates); each sample costs about 11 ms
native and 11 ms in WebAssembly there.

| Call | Native | WebAssembly (Node) |
|---|---|---|
| `hint`, budget 300 ms | 273 ms, 28 samples | 291 ms, 24 samples |
| `hint`, budget 1,000 ms | 838 ms, 84 samples | 850 ms, 76 samples |
| `hint`, budget 3,000 ms | 2,420 ms, 224 samples | 2,438 ms, 220 samples |
| `hint`, 400 samples, no budget | 6,882 ms | — |
| `bot_action` hard (`value`) | < 1 ms | < 1 ms |
| `bot_action` expert, budget 500 ms | 450 ms | 431 ms |
| `bot_action` expert, no budget (96 samples) | 1,087 ms | — |
| `review`, budget 3,000 ms | 28 items: 2,377 ms | 20 items: 2,298 ms |
| `review`, no budget (300 samples per item) | 28 items: 79 s | — |
| a whole hand, `hard` against three `easy` | — | 75 actions: 35 ms |

A 3-second hint gets about 220 samples; the lab UI's default (2 s) gave the same order of
magnitude on 8 threads. The `.wasm` of the web target is 445,560 bytes (435 KB) before gzip.

Rayon in the engine's search crate compiles for `wasm32-unknown-unknown`; rayon-core's global
pool falls back to the calling thread there, so `hint`, `bot_action` and `review` run without
a worker (checked by the Node smoke run).
