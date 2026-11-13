# Plan: fix the review's playout bot, then build the race bot (2026-11-09)

**Goal.** (1) The app's mahjong review stops calling sound moves "critical mistakes" because of
flaws in the rollout bot. (2) A new bot, the **race bot**, values every move by a formula over
all four players' chances (Brian's design, 2026-11-09), learns its estimates from self-play
that improves round by round, and is measured against the teacher.

**Why (evidence, 2026-11-09).** Brian's expert hand (kept-hand JSON from the app; decision at
history index 85: pung the 8p with `4588p` + three pin pungs, 30 wall tiles). The phone review
graded the pung a blunder (pass better by 48 points). `crates/app-api/examples/regrade.rs`
showed: (a) 150,000 samples give the same verdict, so it is not noise; (b) with `ev` rollouts
the verdict flips (pung +20 ± 4); (c) traced rollouts: after the pung the `fast` bot discards
4p in 5,000 of 5,000 samples and keeps 5p, whose four copies are all visible (dead wait), then
swaps to honour singles that break the full flush; (d) after a pass, two thirds of the bot's
wins come on the unchanged `4588p` with one unseen 3p and one unseen 6p: the defending `fast`
opponents keep feeding pins to an obvious pin flush (in `Brain::turn` a bot that is ready or
one away never defends). The sampler itself deals the hidden tiles correctly (checked).

**Brian's decisions for this plan (2026-11-09, verbatim or close; record in bot-stream
`docs/decisions.md` as B28, Brian's call):**
- The real offence EV sums every player's chances: if other players are better placed, my EV
  goes negative. Drawing a good tile raises my side; compare it with the expected loss from
  later throwing a tile I don't need (opportunity cost).
- Decisions are nested: once a player plays defensively they give up winning equity, and
  opponents read it from the discards. A player's EV is worked out over the chances of their
  actions: **fold, detour, push**. A folded player's threat becomes (about) zero.
- Opponents one tile away who have not started defending matter; throw tiles that are
  dangerous later, but useless to my goal, early. That includes tiles that let opponents
  pung or chow (faster for them), and honour tiles that give faan (keeping them raises my
  chance and my risk; balance them).
- Two threats: fold temporarily against B, push again once B folds if I hold nothing that
  loses to A. The mode is chosen fresh every turn and against each threat separately.
- Pushing a cheap hand can be defence: it removes the opponents' chance of an expensive hand.
- Hand value enters only through the EV calculation (no extra rule).
- My own table image (opponents reading my folds): left out at first.
- Training: learn from bot play, self-improving over the training period until it converges.
- Never trust simulation blindly; let measured matches decide between methods (the
  coordinator's recommendation, accepted by Brian's go-ahead "go ahead with the mahjong
  changes", 2026-11-09).

**Constraints (bot-stream and hk-games decisions).** Engines return keys, never display text
(D42). Edit this repo only in its standalone checkout; hk-games moves the pin and rebuilds
`app/src/engine/wasm/mahjong/` in the same commit. CPU- or GPU-heavy jobs go through the job
queue (`.claude\job-queue\jobq.ps1`, B26); Dou Dizhu holds the GPU lane until about
2026-11-12. The app-api's `PimcConfig` literal names every field (a new field breaks it).
The app's review budget stays 12 s per hand (`REVIEW_BUDGET_MS`), 0.2–1.5 s per decision.

**Acceptance criteria.**
- AC1: on Brian's hand, the phone-sized grade of decision 85 is no longer `blunder`, and the
  regrade shows the rollout bot discards 5p after the pung.
- AC2: the fixed rollout bot never keeps a wait with zero live tiles when a discard with live
  tiles exists at the same distance (test), and a defending opponent with a ready or one-away
  hand still declines to feed an obvious flush when a safe tile costs it little (test).
- AC3: the teacher spec with the fixed rollouts is not weaker than with the old ones:
  duplicate match, 1,000 walls × 4 seatings, difference with a 95 % interval whose lower end
  is above −0.5 points per hand.
- AC4: `docs/race-bot.md` specifies the race bot term by term, with Brian's pung hand worked
  by hand, the training labels and the self-play loop.
- AC5: the race bot v0 (heuristic estimates) plays legal hands within 5 ms per decision on one
  thread and has a measured strength against three `value` bots (4,000 hands, interval).
- AC6: the self-play loop runs at least three generations with a league and a stop rule; the
  best generation's strength against the teacher is reported with an interval (no bar: the
  report decides what the review uses next, with Brian).
- AC7: the app's review wording for pass and decline reasons describes the move, not the
  player (all three languages), and the app ships the rebuilt engine.

## Phase 1 – Review fixes (opus; Tasks 1–3 [parallel-ok] apart from shared `bots.rs`)

- [x] **Task 1: Live-tile-aware waits in the rollout bot**
  - Files: modify `crates/search/src/bots.rs` (and `crates/analysis` if the acceptance count
    lives there); test `crates/search/tests/rollout_fixes.rs` (new), fixture from Brian's hand
    at index 85 (copy `hand.json` into `crates/search/tests/data/brian-2026-11-09.json`).
  - Done when: the new test fails before and passes after: after the pung the `fast` bot
    discards 5p; a synthetic hand with two equal-distance waits, one dead, keeps the live one.
    `cargo test -p mahjong-search --release` passes.
  - Notes: count live copies from the acting seat's view (own hand, all melds, all discards).
    Find why `best_discard`'s acceptance ignored visible copies before changing it. Also keep
    a full flush (or half flush) once its wait has live tiles: do not swap to an honour single
    that drops the flush unless the flush wait is dead.

- [x] **Task 2: Defending opponents stop feeding an obvious flush**
  - Files: `crates/search/src/bots.rs` (`Brain::turn`), test in `rollout_fixes.rs`.
  - Done when: test: a ready or one-away defending `fast` seat, facing a seat with three
    exposed melds in one suit, discards a safe tile instead of a live tile of that suit when
    the safe tile keeps its distance or costs at most one step; `cargo test` passes; the
    traced pass-rollouts of Brian's hand (regrade `trace`) show a pass win rate well below 85 %
    (report the number).
  - Notes: the danger model already has a flush relation (`danger.rs`). Keep `fast`'s speed
    (rollouts call it millions of times): no belief network inside rollouts.

- [x] **Task 3: Keep and document the regrade tool**
  - Files: `crates/app-api/examples/regrade.rs` (exists, uncommitted), README section in
    `crates/app-api/README.md`.
  - Done when: `cargo build --release -p mahjong-app-api --example regrade` passes; README
    shows the four modes (`<index> 0`, `<index> <scale>`, `all`, `<index> 1 trace|count`).

- [x] **Task 4: Measure the fix** (AC3 failed: -0.556 [-1.61, +0.50]; see study-results/review-fix.md)
  - Files: `docs/study-results/review-fix.md` (new); queue job script under `experiments/`.
  - Done when: AC1 (regrade output pasted) and AC3 (duplicate match through the cpu lane,
    teacher spec old vs new rollouts) are recorded; plus `regrade all` on Brian's hand before
    and after (which grades changed). If AC3 fails, record it in open-items and keep the fix
    only for the review's grade, not for the bots' own play.

## Phase 2 – Race bot design (fable: method open)

- [x] **Task 5: `docs/race-bot.md`**
  - Files: create `docs/race-bot.md`; add B28 to bot-stream `docs/decisions.md`.
  - Done when: the doc defines, with sources in this repo for every input: each player's state
    chain (far / one away / ready / folded) and per-turn hazard; the competing-risks race and
    my equity over all four players; a move's effects (deal-in now, feeding pung/chow by seat,
    my live and improvement tiles, holding cost of each tile as danger rises, fold reserve,
    honour tiles that give faan); modes push / detour / fold per threat, opponents' modes as
    probabilities, the mode choice each turn (best response over the mode combinations, with a
    cost bound); the training labels (state, mode, waits, value) from self-play with full
    information, the fold label's exact definition; the self-play loop with a league and a stop
    rule (generation vs previous and vs teacher, gain inside noise for 2–3 generations); and
    Brian's pung hand worked through by hand with every number. Lists what v0 takes from
    existing parts (belief heads, danger tables, target table) and what Phase 4 trains.

## Phase 3 – Race bot v0 with existing estimates (opus)

- [x] **Task 6: `crates/search/src/race.rs`**
  - Files: create `crates/search/src/race.rs`, `Bot::Race` in `bots.rs`, spec `race[:...]` in
    `agent.rs`; tests `crates/search/tests/race_bot.rs`.
  - Done when: tests: legal moves on 1,000 random hands; Brian's pung hand values pung ≥ pass;
    the deliberate deal-in position of `ev-bot.md` §5 example 2 still deals in; a two-threat
    position folds against the pushing seat and pushes when it folds; one-thread timing < 5 ms
    per decision (release); `cargo test` passes.
  - Notes: estimates from the existing belief model where it has heads (tenpai, waits, faan,
    counts); state and mode from heuristics until Phase 4. `mahjong hk tournament` must accept
    the spec.

- [x] **Task 7: v0 strength** (-9.128 [-9.794, -8.496] vs three value; see study-results/race-bot.md)
  - Files: `docs/study-results/race-bot.md` (new); job under `experiments/`.
  - Done when: AC5 recorded: `race` vs three `value`, 1,000 walls × 4 seatings, points per
    hand with a wall-bootstrap interval, win / deal-in / draw rates, mean faan.

## Phase 4 – Learned estimates and the self-play loop (opus; GPU training through the queue)

- [x] **Task 8: Labels and heads** (job 90; held-out calibration per bucket in study-results/race-bot.md)
  - Files: self-play logger (extend `examples/selfplay_log.rs` or a new example); Python
    training under `python/mahjong/belief/` (new heads: state, mode); `models/race-belief.mjnn`.
  - Done when: the logger writes, per decision and opponent, the true state, the mode label
    (Task 5's definition), waits and hand value; the extended network trains in the gpu lane
    and reports held-out calibration per wall-progress bucket for state and mode.

- [x] **Task 9: The loop** (job 90, 4 generations, stop rule fired; gains inside noise, bot ~9 points behind value; study-results/race-bot.md)
  - Files: `experiments/race_loop.ps1` (queue job, resumable), `docs/study-results/race-bot.md`.
  - Done when: AC6: at least three generations (play → label → train → new bot), each gated by
    a duplicate match against the previous generation and the teacher, with a league of older
    generations and fixed bots; the stop rule from Task 5 applied; the report lists each
    generation's numbers with intervals.

## Phase 5 – App (sonnet; in hk-games)

- [x] **Task 10: Neutral reason wording** (done in hk-games b6946d4)
  - Files: hk-games `app/src/i18n/ui/{en,zh-Hant,zh-Hans}.json` keys `mahjong.reason.declines_win`,
    `mahjong.reason.pass_keeps_hand`, `mahjong.reason.pass_misses_claim` (and any other
    `mahjong.reason.*` written as "you did …").
  - Done when: each sentence describes the move ("Passing gives up a win you could take"),
    not the player; the dictionary key check and `npm test` pass.

- [x] **Task 11: Ship the engine** (hk-games 77ea4a8: pin 9c3eb2b, wasm, the 10 drills re-solved on the same positions; browser review of Brian's hand: pung of 8p graded best, 0 blunders)
  - Files: hk-games `games/hk-mahjong-lab` pin, `app/src/engine/wasm/mahjong/`.
  - Done when: after Phase 1 is pushed here, hk-games moves the pin, runs
    `npm run engines -- mahjong`, `npm test`, `npm run build`, and the browser review of
    Brian's hand (loaded into sessionStorage `hkgames.keptDeal.mahjong`) no longer shows the
    pung as a critical mistake; committed together. If the race bot later wins Phase 4, the
    app switch is a separate decision for Brian.

## Phase 6 – The passed-win rule and passed-tile reading (Brian, 2026-11-09; waits for his go)

Brian's instruction (close to verbatim): add the passed-win rule (過水: a player who passes a
win on a tile can't win on that tile until their next turn). The defence must use it, not
only within this go-round: a dangerous tile another player recently let pass is less likely
to deal in, by how much that player has changed their hand since (入章 / 換章). A player
willing to throw dangerous tiles updates our estimate of their offence: how hard they push
narrows the range of what they are aiming for.

Brian's answers (2026-11-09): the lock covers only the passed tile. 換章 means a pushing
player throws a tile that was clearly useful to them, which shows their hand improved; every
tile they let pass before is no longer safe against them (a full reset, beyond the gradual
fade per 入章). The existing models (teacher network, belief model, danger tables) are
retrained with the rule (Task 18). Old recorded deals need no support: the app is not in
production, so the server and the app simply switch to the new rule together (no rules
version on deals).

- AC8: the engine applies the rule (on by default in `RuleSet`); positions and kept hands
  carry it; the server's replay and the app agree; old recorded deals still replay.
- AC9: every player's discard record says whether the discard was the tile just drawn
  (摸打) or a tile from the hand (入章), as players see at a real table.
- AC10: the race bot's deal-in chance for tile t against player P uses "P let t pass",
  with a safety that fades with each of P's 入章 since; tests and a match show its effect.
- AC11: the race bot's read of each opponent rises toward push, and toward a closer or
  bigger hand, when that player throws tiles that were dangerous when thrown; a test.

- [x] **Task 12: The rule in the engine** (opus)
  - Files: `crates/hk/src/{rules,state,position}.rs`, the rules doc, `crates/hk` tests.
  - Done when: tests: a seat that passed a win on t gets no Win on t from any discard or
    kong rob until its next turn (draw or own claim); other seats can still win; pung and
    chow are unaffected; `RuleSet::passed_win_lock` false restores today's play; the
    position spec round-trips the lock; `cargo test --workspace --release` passes.
- [x] **Task 13: 摸打 / 入章 in the record** (opus)
  - Files: `crates/hk/src/state.rs` (discard history), app-api view JSON, `SeatView`.
  - Done when: each discard in the view carries `from_hand: bool`; test.
- [x] **Task 14: Passed-tile safety in the race bot and the defending `fast` bot** (fable:
  a new estimate with no data yet)
  - Files: `crates/search/src/{race,bots}.rs`, a new `crates/search/src/passed.rs`,
    `docs/race-bot.md` (a new section), tests.
  - Done when: for each opponent and tile, the bot keeps "let it pass, with N 入章 since";
    N = 0 and the lock still active gives zero deal-in chance; after that the chance moves
    back toward the table's estimate per 入章, at a rate measured from self-play logs
    (true win-on-t rate by N), not guessed; a 換章 (a pushing player throws a tile that was
    clearly useful to them: by our estimate it connected to their likely hand) resets all
    their passed-tile safety; the 換章 test is checked against self-play truth (did their
    waits really change); tests; timing still under 5 ms per decision.
- [x] **Task 15: Reading an opponent's risk-taking** (fable: model choice open)
  - Files: `crates/search/src/race.rs`, `docs/race-bot.md`, tests.
  - Done when: each opponent's push probability and hand-value estimate update from the
    danger of the tiles they threw (danger measured from our view when they threw it); the
    size of the update is checked against self-play truth (pushers' real closeness and hand
    value); a test where a player throws two dangerous tiles moves them to push.
- [x] **Task 16: Measure** (opus) (job 35; write-up bc2b807; all intervals include zero)
  - Files: `docs/study-results/race-bot.md`; job under `experiments/`.
  - Done when: race bot with Tasks 14–15 against without, duplicate match through the cpu
    lane, 1,000 walls × 4 seatings, interval; teacher with and without the rule as a sanity
    row.
- [x] **Task 17: Ship** (sonnet; in hk-games) (hk-games 3ce8c6b + e187f17: pin f288f10, wasm, server module, 9 drills same positions, drill 0003 replaced (seed 601, gap 3.32), lock message in 3 languages; app 467 tests, server 193 pass; real-engine check through the shipped wasm: 3 locked windows, Win never offered; UI rendering checked by passedWinLock.test.tsx, not in a browser)
  - Files: pin, `app/src/engine/wasm/mahjong/`, the server's mahjong module
    (`server\engines\build.ps1`), `app/src/drills/mahjong.json` re-solved on the same
    positions, a ui string in three languages saying why Win is not offered.
  - Done when: `npm test`, `npm run build`, server `pytest` pass; a browser check of a hand
    where the player passes a win and then cannot win on the same tile.
- [x] **Task 18: Retrain the existing models with the rule** (opus; gpu lane after Dou Dizhu) (Brian 2026-11-11: a recorded calibration miss counts as done and blocks nothing; record each bucket's predicted and observed rate)
  - Files: feature encoding (add the passed-tile record and 摸打 / 入章), labels, the belief
    model, the danger tables, the teacher's distilled network; the `faan_shanten` fix from
    open-items rides along (it also changes the encoding).
  - Done when: each retrained model passes its existing checks (belief calibration, danger
    calibration, the distilled network against the teacher); the app's review and drills use
    the new models only if they are not weaker in a duplicate match.
  - Status (2026-11-10): code and data half done (open-items, "Task 18, code and data
    half"): `faan_shanten` fix, `features::encode_passed` in the logs and the belief model's
    `--passed`; self-play logs queued (`experiments\task18-data\selfplay.ps1`); labels script
    ready, waiting for the retrained belief model. Training, refits and matches remain.
  - Status (2026-11-10, training half): belief-v2 (job 38), labels-v2 (job 40), cnn-distill-v2
    (job 41, top-1 0.917), refits (job 43: danger tables refitted but not adopted, calibration
    unchanged), duplicate match (job 44): network -0.642 [-2.02, +0.80], teacher -0.098
    [-1.78, +1.67]. Not weaker by a measured margin, not stronger; the app loads neither
    model, so the hk-games pin stays. Open-items "Task 18, training half" has the details.
- Phase 4's loop (job 33) waits for this phase: its logger and features add the passed-tile
  record and the risk-taking read, and it is requeued after Task 16.

## Phase 7 – Strength checks (opus; added 2026-11-10 on Brian's word "Yes, add all three")
Every strength number so far is measured against our own `value` bot. No public bot or game
records exist for Hong Kong rules (Mortal plays Japanese rules only). Two checks follow. The
first is Brian playing his 400 fixed walls (AC6 tooling, Task 12): that is Task 22 of
`2026-10-29-mahjong-bot-build.md`, waiting on his hands; nothing new is planned for it.

- [x] **Task 18c: A confidence range on each belief guess** `[parallel-ok]` (FAILED check: 26.1 % inside ±1 sd overall [13.0, 30.4]; tenpai 13 %, deal-in 64 %, discard-win 0 %, self-draw 14 %; docs/belief.md §8; inbox question posted for milestone 2)
  - Why: Brian, 2026-11-11, wants each guess to carry its own confidence (hk-games D100, D101):
    "(a), and if fails add to Milestone 2. Honestly just from the number I think the guess is quite close, so it should not block subsequent mahjong tasks".
  - What: a Laplace approximation on today's `models/belief-v2.pt` (no retraining): fit the
    curvature of the training loss at the trained weights (last layer, or diagonal, whichever
    fits in memory), then give each tenpai, deal-in and outcome guess a mean and a standard
    deviation of the probability.
  - Files: `python/mahjong/belief/laplace.py`, its test, `evaluate.py` (`--laplace`: a
    table that groups held-out guesses by their stated probability and reports how often the
    observed rate falls inside mean Â± 1 standard deviation, with intervals that resample whole
    games), `runs/belief-v2/eval-laplace.json`, `docs/belief.md` (a short section).
  - Done when: the table exists for tenpai, deal-in and outcome; the share inside Â±1 standard
    deviation is reported per group. Pass: between 60 % and 76 % overall (68 % expected). On a
    fail, record the numbers here and post one inbox question proposing five trained copies
    in milestone 2 (Brian: "if fails add to Milestone 2"). Either way nothing else waits on it.
- [ ] **Task 19: An attacker bot trained against the shipped bot** `[parallel-ok]` **[GPU: 10 h, job queue gpu lane]** (waiting: queue job 70 (was 54, re-queued 2026-11-11), gpu lane; code, test and job done, results go in docs/study-results/exploiter.md)
  - What: train one seat from fresh weights with `train-core` PPO (2026-10-29 plan, Task 17)
    while three seats play the bot the app ships as its top mahjong level, frozen. The attacker
    looks only for that bot's weaknesses.
  - Files: `experiments/exploiter/run.ps1` (`jobq.ps1 add -Lane gpu`; resumes from its latest
    checkpoint); a frozen-opponents option in the PPO trainer if it lacks one, with a short test;
    `docs/study-results/exploiter.md`.
  - Done when: the results file names the bot attacked and its spec string, and records the
    attacker's gain per hand against it in a 24-seating duplicate match with a 95 % interval.
    It also records a check that the attacker learns at all: a 2-hour attacker gains more
    against three `value` than against the shipped bot. No bar is set: a gain whose interval
    excludes 0 goes to `docs/open-items.md` and to Brian.

## Phase 8 - Fix the race bot's over-folding (Brian, 2026-11-11: "Ok, need fixing then")

- [x] **Task 20: Calibrate the race bot's win chances, then rerun the loop** (fable: calibration method open) `(checkpoint)` (hazard check 5 of 6 buckets on 24,000 fresh hands; offence agreement 1000/1000; match -1.97 ± 1.16 vs 3 value, was -9.4; loop re-queued as job 97; inbox checkpoint posted; study-results/race-bot.md Task 20)
  - Why: the race loop (Tasks 8-9, job 90) showed each opponent's per-turn win chance is
    3-20 times the observed rate in all six wall buckets; the bot folds far too often, wins
    0.2-0.3 % of hands and trails three `value` bots by 9.4 points per hand
    (`docs/study-results/race-bot.md`, open-items). Brian, 2026-11-11, on the point that
    folding raises the opponents' chances: "Ok, need fixing then".
  - What: find which input inflates the chance (the state chain `a`/`b`, the feed rates `q`,
    `tenpai_o` from the belief, or the round-1 deal-in term); fix it so the predicted and
    observed win rates agree; measure on held-out self-play, never on the hands it was fitted on.
  - Files: `crates/search/src/race.rs`, the Phase 4 fitting code it uses, their tests,
    `docs/race-bot.md` (a deviations note), `docs/study-results/race-bot.md` (a new section).
  - Done when: the hazard check puts the observed rate inside its interval in at least 4 of 6
    wall buckets; on 1,000 held-out early-hand positions where every opponent is most likely two or
    more tiles from winning, the race bot picks the same discard as the pure-offence ranking`rank_discards` at least 95 % of the time (Brian, 2026-11-11: "Sounds good, go"); a duplicate match of the fixed race bot against three `value` bots (4,000 hands,
    seats rotated) is reported with its interval; the loop is re-queued in the cpu lane through
    `hkq.ps1` only if the match beats today's -9.4 by more than its interval. Post the match
    result to the inbox (checkpoint) whether or not it wins.
- [x] **Task 21: The race bot decides whether to take a win or pass it for a bigger hand** (fable: method open) needs: Task 20 (-0.03 ± 0.16 vs no passing, 33 of 4,000 hands pass; study-results/race-bot.md Task 21)
  - Why: today `race.rs` takes every legal win (`docs/race-bot.md` Â§4.5). Brian, 2026-11-11:
    "Yes, that's the whole point"; "So yes the race bot should also learn whether to take the win or pass it up more greedily especially when it is obviously ahead of other players (so almost no defensive consideration)".
  - What: score a pass of a discard win or a self-draw win as one more candidate. Win now:
    the payment of this win. Pass: the race value of the hand kept (or of the best discard after a
    passed self-draw), aiming at the bigger target, with the passed-win rule (a passed discard
    win locks that tile until my next turn, `crates/hk` Task 12). When the opponents are far from
    winning, their terms are small and the pass is judged on my own chances, as Brian asks.
  - Files: `crates/search/src/race.rs` and its tests, `docs/race-bot.md` Â§4.5, the app's
    review reason `mahjong.reason.declines_win` if its wording no longer fits (hk-games, separate commit).
  - Done when: tests: (a) a 3-faan discard win with a flush self-draw wait, many tiles left and all
    opponents far, is passed; (b) the same hand with an opponent likely ready, or few tiles left, is
    taken; (c) a passed tile is never won on before my next turn. A duplicate match (4,000 hands,
    seats rotated) of the race bot with and without passing, reported with its interval in
    `docs/study-results/race-bot.md`, plus how often it passed and what those passes earned.
## Phase 4 - The teacher in the app, for now (sonnet)
Brian, 2026-11-13: "just swap in the teacher model for now, it will get changed later during
Milestone 1, I need to better see for myself how it performs". Today the app's hint, review and
expert bot run plain PIMC: `crates/app-api/src/solver.rs` calls `pimc_with`, which deals guesses
with `WorldSampler::BestOfK` (a hand-written rule). The teacher (`pimc:samples=64,candidates=4,
belief=models/belief.mjnn,weights=1`) deals them with the belief model. This phase gives the app
the teacher's guesses. hk-games then ships it (its milestone-1 plan, Task T1).

- [x] **Task 22: The app-api can deal its guesses with the belief model** (commit 321c384; hint 6.85 vs 4.34 ms/sample with the model; study-results/teacher-in-app.md)
  - Files: `crates/app-api/src/solver.rs`, `api.rs`, `lib.rs` (wasm and Python bindings),
    `crates/app-api/tests/api.rs`; `crates/search` only if `BELIEF_DRAWS`, `default_tilt` or
    `BeliefModel::from_mlp` need exporting.
  - What: a new call `set_belief(bytes)` on the Api (wasm: `&[u8]`; Python: `bytes`) builds
    `BeliefModel::from_mlp(Mlp::from_bytes(bytes))` and keeps it; it returns the usual JSON
    (ok, or an error key for a bad blob). While a model is set, `hint`, `review` and the
    `expert` bot use `WorldSampler::Belief { weights: true, draws: BELIEF_DRAWS, tilt:
    default_tilt(true) }` (the teacher's `weights=1`) through `pimc_with_sampler`. The expert
    bot then uses the teacher's 64 samples and 4 candidates; hint and review keep their sample
    caps, candidates and time budgets. `grade` (drills) and every call without a model set stay
    exactly as today (`BestOfK`), so drills, the server module and the replay checks do not change.
    No model file goes into git here: hk-games ships the file and passes its bytes in.
  - Done when: tests: (a) a bad blob gives an error and hints keep working on `BestOfK`;
    (b) with a model set (a tiny test MLP with the five belief heads, or `models/belief.mjnn`
    copied read-only from `repo\hk-mahjong-lab\models\` when the test is `#[ignore]`d), `hint`
    and an `expert` move return legal answers; (c) every existing app-api test passes unchanged.
    `cargo test -p mahjong-app-api` and clippy pass. A native timing on 20 mid-hand positions,
    one thread: hint milliseconds per sample with and without the model (via
    `hint_with_samples`), written in `docs/study-results/teacher-in-app.md`. If
    `models/belief.mjnn` (the teacher's measured model, 2026-10-29) no longer loads with today's
    feature encoding, use `belief-v2.mjnn` and say so in that file.
  - Push to main. The hk-games task needs this task's commit hash: write it in the tick note.

## Shared resources
`crates/search/src/bots.rs` (Tasks 1, 2, 6: serial), the job queue (Tasks 4, 7, 8, 9), the GPU
lane (Task 8, after Dou Dizhu's block), `models/` (Task 8 adds a file; nothing is replaced).

## Self-review
AC1 → Tasks 1, 4, 11; AC2 → Tasks 1, 2; AC3 → Task 4; AC4 → Task 5; AC5 → Tasks 6, 7;
AC6 → Tasks 8, 9; AC7 → Tasks 10, 11. Every task has a "done when".
