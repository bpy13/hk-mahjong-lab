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
queue (`bot-stream\tools\queue\hkq.ps1`, B26); Dou Dizhu holds the GPU lane until about
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

- [ ] **Task 4: Measure the fix**
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

- [ ] **Task 7: v0 strength**
  - Files: `docs/study-results/race-bot.md` (new); job under `experiments/`.
  - Done when: AC5 recorded: `race` vs three `value`, 1,000 walls × 4 seatings, points per
    hand with a wall-bootstrap interval, win / deal-in / draw rates, mean faan.

## Phase 4 – Learned estimates and the self-play loop (opus; GPU training through the queue)

- [ ] **Task 8: Labels and heads**
  - Files: self-play logger (extend `examples/selfplay_log.rs` or a new example); Python
    training under `python/mahjong/belief/` (new heads: state, mode); `models/race-belief.mjnn`.
  - Done when: the logger writes, per decision and opponent, the true state, the mode label
    (Task 5's definition), waits and hand value; the extended network trains in the gpu lane
    and reports held-out calibration per wall-progress bucket for state and mode.

- [ ] **Task 9: The loop**
  - Files: `experiments/race_loop.ps1` (queue job, resumable), `docs/study-results/race-bot.md`.
  - Done when: AC6: at least three generations (play → label → train → new bot), each gated by
    a duplicate match against the previous generation and the teacher, with a league of older
    generations and fixed bots; the stop rule from Task 5 applied; the report lists each
    generation's numbers with intervals.

## Phase 5 – App (sonnet; in hk-games)

- [ ] **Task 10: Neutral reason wording**
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

- [ ] **Task 12: The rule in the engine** (opus)
  - Files: `crates/hk/src/{rules,state,position}.rs`, the rules doc, `crates/hk` tests.
  - Done when: tests: a seat that passed a win on t gets no Win on t from any discard or
    kong rob until its next turn (draw or own claim); other seats can still win; pung and
    chow are unaffected; `RuleSet::passed_win_lock` false restores today's play; the
    position spec round-trips the lock; `cargo test --workspace --release` passes.
- [ ] **Task 13: 摸打 / 入章 in the record** (opus)
  - Files: `crates/hk/src/state.rs` (discard history), app-api view JSON, `SeatView`.
  - Done when: each discard in the view carries `from_hand: bool`; test.
- [ ] **Task 14: Passed-tile safety in the race bot and the defending `fast` bot** (fable:
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
- [ ] **Task 15: Reading an opponent's risk-taking** (fable: model choice open)
  - Files: `crates/search/src/race.rs`, `docs/race-bot.md`, tests.
  - Done when: each opponent's push probability and hand-value estimate update from the
    danger of the tiles they threw (danger measured from our view when they threw it); the
    size of the update is checked against self-play truth (pushers' real closeness and hand
    value); a test where a player throws two dangerous tiles moves them to push.
- [ ] **Task 16: Measure** (opus)
  - Files: `docs/study-results/race-bot.md`; job under `experiments/`.
  - Done when: race bot with Tasks 14–15 against without, duplicate match through the cpu
    lane, 1,000 walls × 4 seatings, interval; teacher with and without the rule as a sanity
    row.
- [ ] **Task 17: Ship** (sonnet; in hk-games)
  - Files: pin, `app/src/engine/wasm/mahjong/`, the server's mahjong module
    (`server\engines\build.ps1`), `app/src/drills/mahjong.json` re-solved on the same
    positions, a ui string in three languages saying why Win is not offered.
  - Done when: `npm test`, `npm run build`, server `pytest` pass; a browser check of a hand
    where the player passes a win and then cannot win on the same tile.
- [ ] **Task 18: Retrain the existing models with the rule** (opus; gpu lane after Dou Dizhu)
  - Files: feature encoding (add the passed-tile record and 摸打 / 入章), labels, the belief
    model, the danger tables, the teacher's distilled network; the `faan_shanten` fix from
    open-items rides along (it also changes the encoding).
  - Done when: each retrained model passes its existing checks (belief calibration, danger
    calibration, the distilled network against the teacher); the app's review and drills use
    the new models only if they are not weaker in a duplicate match.
- Phase 4's loop (job 33) waits for this phase: its logger and features add the passed-tile
  record and the risk-taking read, and it is requeued after Task 16.

## Shared resources
`crates/search/src/bots.rs` (Tasks 1, 2, 6: serial), the job queue (Tasks 4, 7, 8, 9), the GPU
lane (Task 8, after Dou Dizhu's block), `models/` (Task 8 adds a file; nothing is replaced).

## Self-review
AC1 → Tasks 1, 4, 11; AC2 → Tasks 1, 2; AC3 → Task 4; AC4 → Task 5; AC5 → Tasks 6, 7;
AC6 → Tasks 8, 9; AC7 → Tasks 10, 11. Every task has a "done when".
