<!-- Copied from C:\Users\brian\.orchestrator\mahjong-watch\findings.md (the Mahjong watch session's findings), 2026-11-16, unchanged below. -->

# Mahjong watch: what is wrong with the bots and hints

Started 2026-11-14. Brian plays the hk-games app (local copy of hk-games-orch at 4efe7a5,
engine pin 03d6f7c, served on 127.0.0.1:5180 from the watcher's scratchpad). The watcher logs
each position Brian comments on, his words verbatim, and what every bot level does there.

Bot levels (engine `crates/app-api/src/api.rs`): easy = efficiency, medium = defensive,
hard = value, expert = PIMC teacher (64 samples, 4 candidates, belief model). The hint runs
the same search as expert (HINT_SAMPLES, belief model when loaded).

Evidence marks: **measured** (a tool run this session), **sourced** (a lab doc), **inferred**.

How positions reach the watcher: Brian plays a private claude.ai artifact build of the app
(https://claude.ai/artifact/DSymkxHW2HgCrha3kGF9sC, offline preview account, recorder in
`src/watch.ts` of the scratch copy). It saves every snapshot, every hint and typed comments to
the artifact db (collections `positions`, `hints`, `notes`).

## Logged positions

### P1. Opening discard, half flush obvious, hint omits 8m (hand h1794661141833, ply 0)

Brian (2026-11-14, verbatim): "why won't model consider 8w in this scenario, when it's clearly
混一色 with either 筒子 or 索子"

Position: East (dealer), first discard, prevailing East, 3-faan minimum.
Hand: 8m 1p 3p 5p 7p 1s 1s 2s 3s S S W Wh G (seed 1799087666).

| Who | Move | Evidence |
|---|---|---|
| easy (efficiency) | W | measured |
| medium (defensive) | 8m | measured |
| hard (value) | 8m | measured |
| expert (teacher, 64 samples) | 8m (keeps the anchor; no move beats it by the margin) | measured |
| Hint in the app (180 samples, 1.6 s) | top three: 7p (+8.25 ± 3.9 over 8m), 1p, Wh. 8m not shown at all | measured (recorded hint) |
| Hint at 2,000 samples | 7p +2.12 ± 0.94 over 8m, 1p +0.96 ± 0.81, W +0.61 ± 0.67; 8m still not in the top three | measured |

Rollout draw rate about 47 % for every move (the rollouts rarely finish a 3-faan hand).

Reading:
- The bots play 8m. The leak is in the hint: it ranks moves by raw average from few sampled
  futures and shows only the top three, so the anchor move can vanish. **measured**
- With 2,000 samples 7p still scores about 2 points above 8m. So noise is not the whole story:
  the rollout policy (`fast`) seems to value this hand wrongly. **inferred**, trace pending.
- Setup note: the artifact build for P1 came from hk-games 4efe7a5, one commit before Task T2
  (8653772, hint = `hard` bot move). So Brian saw the search hint, not the live site's value
  hint. Rebuilt from main's session.ts as version 3 of the artifact. On the live site this
  hint would have said 8m.
- Brian's follow-up (verbatim): "depends on incoming tile. I thought the hint should now use
  value bot, which is just calculations? It seems I am misunderstand how the value bot
  operates, can you please explain"

### P2. 1p or 6s: the value bot breaks a full tie by tile order (hand h1794661688864, ply 33)

Brian (2026-11-14, verbatim): "here, how does it choose between 1t and 6s"

Position: East seat, prevailing East. Brian's melds: chow 789m, chow 234m. Concealed (after the
draw): 2m 1p 6s S W W W R. Opponents: seat 2 has kong 8p (added) and pung 8s; seat 3 has chow
789s; seat 1 no melds. 1p: one discarded by seat 3 (ply 23). 6s: none visible except Brian's.

| Who | Move | Evidence |
|---|---|---|
| easy | S | measured |
| medium | 1p | measured |
| hard (the app's hint now) | 1p | measured |
| expert | 1p | measured |
| Search, 2,000 samples | 6s vs 1p +0.14 ± 0.69 (a tie); W (break the West pung) +1.82 ± 0.83 | measured |

Why the value bot says 1p (code read, `crates/search/src/bots.rs` `rank_discards`): both
discards give the same key (faan-aware shanten 2, acceptance 14, target 3 faan half flush in
characters, connectedness 0). The last sort key is the tile itself, `sort_by_key(|&(t, k)| (k, t))`,
and characters < dots < bamboo < honours. So 1p wins on tile order alone. The `hard` level has
defense 0: table safety, who is threatening, and dead tiles never enter. **measured** (code and
run)

Engine danger read: threat per seat [0, 0.002, 0.011, 0.003]; it sees no one as dangerous.

Brian's follow-up (verbatim): "value bot is tie-break? it only shows 1 option in the app".
Since Task T2 the hint shows one move (`moves: [action]`, no score, no reasons), so a pure
tile-order tie looks like a confident recommendation. **measured** (main session.ts)

### P3. Throws a lone double-wind East instead of West (hand h1794662102925, ply 13)

Brian (2026-11-14, verbatim): "why dong in this scenario"

Position: Brian is East seat, prevailing wind East, so a pung of East is worth 2 faan (seat
plus prevailing); West is worth nothing to him. Concealed: 3p 5p 6p 6p 7p 1s 3s 3s 6s 6s 7s 7s
E W. No melds. Early (ply 13; seat 2 chow 234s, seat 3 chow 234m and pung 5m).

| Who | Move | Evidence |
|---|---|---|
| easy, medium, hard (the hint), expert | E (all four) | measured |
| Search, 2,000 samples | top three: 7p +0.91 ± 1.27, E (anchor), 5p −0.01; W not in the top three | measured |

Value bot keys (from the recorded hint panel): E and W both give faan-aware shanten 4,
acceptance 22, target 4 faan (mixed all pungs), connectedness 0. Full tie again; E sorts
before W, so E goes. **measured**

Root (code read, inferred for the general case): the key counts only the faan of the single
nearest target plan. A lone honour that would add faan if it pairs or pungs (double wind
here: +2) adds nothing to that plan's distance or faan, so it is valued exactly like a dead
West. The search does not rescue it, because its rollouts are played by `fast`, which uses
the same measure. **inferred**

Brian's correction (verbatim): "no? you clearly throw away 3t/5t/7t first?" (the watcher had
said West). His read: seven bamboo tiles (1s 3s 3s 6s 6s 7s 7s) plus East point to a bamboo
half flush with the double wind; the pins (3p 5p 7p, then 6p 6p) are the tiles to shed.

Why the value bot never gets there (from the recorded panel): 3p, 7p, 1s, E and W all tie on
shanten 4, acceptance 22, target 4 faan (mixed all pungs). The next key is connectedness,
lower first: 3p and 7p touch other pins, E and W touch nothing, so the honours go first, then
tile order puts E before W. The bot's "isolated tile goes first" rule is the opposite of
flush play, where connected off-suit tiles go and honours that fit the plan stay. The search
leans Brian's way but cannot separate it: 7p +0.91 ± 1.27 over E, 5p level. **measured**

## Causes (draft, updated as positions come in)

1. **Hand-value reading counts only the nearest plan.** The faan-aware key ignores upside
   tiles (a lone double wind, a dragon) and treats them as worthless. Seen: P3 (all bots), P2.
   Root (code read, **measured**): `crates/analysis/src/target.rs`. `value_pung_candidates`
   plans a dragon or value-wind pung only when the hand already holds a pair of it (or it is
   melded); `for_each_target` tries flush plans only for the single suit with most tiles
   (concealed plus melded; on a tie Rust's `max_by_key` returns the last, so bamboo beats dots
   beats characters, and the other suit's flush is never even scored; Brian asked about this
   2026-11-14: "assuming at the beginning it has the same number of wan and tung, and both
   target has same distance and same faan. How does it break the tie?"); the
   target picked is the lowest shanten, then the highest *guaranteed* faan. A lone East has no
   effect on any target. Then `rank_discards` in `crates/search/src/bots.rs` sorts by that.
2. **Ties broken by isolation, then tile order.** When the key ties, the least connected tile
   goes, then suit order (characters, dots, bamboo, honours E S W N). Isolation favours
   throwing honours and keeping off-suit connected tiles: wrong for flush play. Seen: P2 (1p
   over 6s), P3 (E over 3p/5p/7p). **measured**
6. **No commitment to a flush direction early.** With seven bamboo and a double wind, the bot
   aims at "mixed all pungs" (the nearest plan) instead of the bamboo half flush. Seen: P3, and
   P1 (the search hint preferred breaking pins/keeping 8m). **inferred**
3. **The hint hides uncertainty.** Search hint: top three by raw mean from 180 samples, the
   anchor can vanish (P1). Value hint: one move, no reason, ties look confident (P2).
   **measured**
4. **Search inherits the rollout bot's blind spots.** `fast` plays every future with the same
   key, so the search cannot see value the key misses (P1 at 2,000 samples, P3). **inferred**
5. **No table reading in the hint's bot.** `hard` has defense 0: dead tiles, others'
   discards and threats never enter (P2). **measured** (code)

## Ranked fix strategy (draft 1, proposed to Brian 2026-11-14; not approved)

Brian asked (verbatim): "what solution would you recommend". Based on three positions, all
about building the hand; defence and opponent reading are not yet observed.

1. **Replace "nearest plan" with "every plan, weighed".** Score a discard by summing over all
   plans: chance to finish the plan times its points. Widen the plan list (both flush suits,
   single value honours, small three dragons). Count live tiles, so a pair waiting for a pung
   is worth less than a two-sided chow wait. New evaluator beside `faan_shanten`, not a change
   to it: changing `faan_shanten` changes the feature encoding and forces retraining of every
   network (sourced: review-fix.md §6). Cost: code only, no training. Value-bot matches of
   4,000 hands take seconds to minutes (sourced: race-bot.md, 16 s for 4,000 hands).
   Caveat: the existing `ev` bot lost to `value` by 1.5 points per hand as the search anchor
   (sourced: phase4-build.md §1). It also reads the hand through `faan_shanten`: `ev.rs`
   line ~549 looks up one cell of the target value table (`target_value.rs`: win chance,
   self-draw share, points, keyed by plan kind, distance 0-4, wall bucket, melds) for the
   single best target (**measured**, code read). So it shares the single-plan root; its loss
   does not test this fix. That table ignores live tiles: every hand of one kind and distance
   gets the same win chance.
   How to weigh (proposed to Brian 2026-11-14 after "how do you plan to weigh every plan?"):
   per plan, chance to finish = chain over the remaining steps of (live useful tiles / unseen
   tiles) per turn, with claimable tiles (pung from anyone, chow from the left) counted higher,
   over the turns left; worth = chance x points of the plan's faan; a discard's score = its
   best plan's worth plus a reduced share of the other plans (the share is tuned by matches).
   Later option: exact one-draw lookahead (34 draws x 14 discards) with this score at the leaf.
2. **Brian's positions become tests.** P1 to P3 (and later ones) as fixtures; a new evaluator
   must pick his move. Average points against `value` cannot see these leaks, because every
   bot shares them. Cost: small.
3. **Hint honesty.** Show tied moves as tied; show the plan the move serves. Cost: small, app.
4. **Then rerun the search with the new rollout bot**, and relabel and redistil only if the
   paired match shows a gain. Cost: PIMC match about 55 min per 4,000 hands on 18 threads
   (sourced: phase4-build.md §2); distillation 21.7 min GPU (sourced); labelling time for
   1 M decisions not measured here.
5. **Defence and table reading**: wait for positions from Brian before ranking.

## Brian's design questions and proposal (2026-11-15)

Brian (verbatim): "when opponent plays random moves, while already 1 step from winning, is it
always profitable to swap for higher faan when offered the chance?"
Watcher's answer (**inferred**, rough sums, not engine runs): no. Compare win chance x points
before and after. The swap loses when the faan gain is small and the new wait has far fewer
live tiles, when the wall is nearly empty, when the swap leaves ready, or at the points cap.

Brian's solver proposal (verbatim): "is the following solver possible?
* in principle, the best move is found to give the best EV (summation of available tiles to
  win and their respective point)
* when 1 step away from winning, there are non-winning tiles that helps to introduce higher EV
  by swapping to a better combination, therefore these tiles should counts towards the implied EV
* which applies to N-steps away from winning also"

Watcher's reading (**inferred**, nothing built): this is a look-ahead over draws. Value of a
hand = average over every tile it can draw of the value after the best discard; a winning draw
is worth its points. It replaces the hand-made "plan list plus share" score in fix 1 with one
rule and needs no plan list. Limits: cost grows about 500x per extra draw looked ahead without
pruning; the plain form ignores opponents (claims, others winning first, dealing in). Not
approved; Brian is still shaping it.

Brian's corrections to the watcher's reading (2026-11-15, verbatim):
"* value of a hand is not the average, it's the sum. Hand with more outs is worth more. For
example, 10 tiles left in the deck, 2 wins 4 points. It worths more than 1 wins 6 points, even
if it has higher average.
* the non-winning draw does not worth the value of the better hand. With the previous 2 wins 4
points with 10 tiles left example, given 1 tile that improves to 4 wins 4 points, there is a
1/8 chance to improve to 2 extra outs, so the effective out should be roughly 2.25 wins 4 points
* claims, others winning first, dealing in: not sure why need to take that into consideration,
we're considering which tile to best discard"

Watcher's reply (**inferred**, hand sums): points 1 and 2 match the look-ahead once "average"
is read as "sum of outs x points over tiles left". His example over two draws: 40.0 % to win
with the improving tile counted, 37.8 % without. Point 3: for one discard choice, others
winning first and dealing in can wait; claims change the out count per shape (a pair takes a
pung from three opponents, a chow shape only from the left player), so they can change the
best discard. Open decision for Brian: count claims in the first version or not.

## Brian's solver defaults (2026-11-15, his decisions, verbatim)

"* actually don't we have a feature in the mahjong to see which tile to hit with the current
combinations? does that not already incorporate the 1-draw version we're trying to build here?
if not what's the difference?
* by default we do want the 1-draw + method 1
* at early game we by default check by suit, and check 對對糊
* before building the feature, provide an estimate of how long the solver should take at any
stage of the game"

("Method 1" = only look at draws that can change the hand. "Check by suit" = score each suit
from a prepared table and combine. 對對糊 = all pungs.)

Existing feature versus the 1-draw solver (**measured**, code read at lab pin 03d6f7c):
`crates/analysis/src/lib.rs` `acceptance` and `crates/app-api/src/discards.rs` `best_discards`
(the hand checker's "Waiting tiles" and the hint panel's "Waits" column). For each discard
they try all 34 tiles and count the live copies of those that lower the distance. Differences:
(1) a tile that keeps the distance but improves the hand counts zero; (2) every useful tile
counts the same, no points (the hand checker ignores faan; the hint uses only the nearest
plan); (3) distance and count are sorted one after the other, never combined into one value.
The loop over 34 tiles already exists, so the 1-draw solver reuses it with a richer value.

Timing (**measured** this session, scratch tool `timeit`, one core, this PC, 2,000 repeats):
one value-bot move = 154 us (P1, opening, 12 tile kinds), 75 us (P3, ply 13), 108 us (P2,
ply 33, two melds). Efficiency bot: 14 to 26 us.

Solver time estimates (**inferred** from those timings; nothing built):
| Stage | Method | PC, per move | Phone (2 to 3 times slower, assumed) |
|---|---|---|---|
| Early, 4 or more steps away | by suit from a prepared table, plus all-pungs check | under 1 ms | a few ms |
| Middle, 2 to 3 steps away | 1 draw ahead, relevant draws only | 15 to 60 ms | 50 to 200 ms |
| 1 step away or ready | 1 draw ahead | 10 to 30 ms | 30 to 100 ms |
| Ready, optional 2 draws ahead | relevant draws only, remembered hands | 0.2 to 1 s | 0.5 to 3 s |
Basis: 1 draw ahead = about 12 discards x 34 draws = about 400 rankings of about 0.1 ms each
(40 to 60 ms), halved by skipping draws that change nothing; the points leaf may double it.
The per-suit table is built once (seconds to minutes, not estimated closely). Test match of
4,000 hands with the 1-draw solver: about 10 to 30 minutes on 18 threads.

Brian (2026-11-15, verbatim): "* how do you plan to build the prepared table? what should the
table include?
* does this take unavailable tiles into account? this also means prepared table becomes biased
* agree to leave out 2-steps for now"
Decision (Brian): no 2-draws-ahead look in the first build.
Watcher's table proposal (**inferred**, not built): one entry per holding in one suit (about
405,000 holdings, plus honours); per entry, for each count of sets (0 to 4) with or without
the pair: tiles still missing, and which of the 9 tiles help. No chances stored, so the table
holds no assumption about live tiles; live copies are counted at the table at lookup time.
Known remaining bias: the distance assumes each needed tile still exists; a suit whose helping
tile is fully dead needs an exact recount. The lab's current distance has the same bias
(`shanten.rs` header: shanten ignores availability, acceptance uses it; **measured**, code).
Open: how the early game turns distance plus live outs into one value.

Brian (2026-11-15, verbatim): "* the splitting the tiles into different components, is that
how to current feature (camera check) works? is it robust and proven working 100% of the time?
* for the early-game table you shouldn't need to look at all combinations because some would
be already very close to winning
* the 3367 example, I'm not sure what you mean, is this early game? I'd imagine 6 and 7 to be
effective also, and 2, 4 to be partially effective also
* not much you can do about the missing tiles, current strategy works, table remembers the
effective tiles from the current combinations, and deduct the missing tiles directly"
Decision (Brian): the table remembers the effective tiles; the solver deducts missing tiles
directly. No further handling of dead tiles.

Checked (**measured**, code read and tests run this session at lab pin 03d6f7c):
- The hand checker's waits (`crates/hk/src/hand.rs` `waits`) add each of the 34 tiles in turn
  and test whether the hand is complete. Its completeness test agrees with a brute-force split
  on random hands (test `table_agrees_with_brute_force_split`, passed).
- The distance (`crates/analysis/src/shanten.rs`) already splits each suit separately and
  remembers each suit holding once seen ("searched once per count vector and cached"). So the
  per-suit table exists in on-demand form; it stores set and half-set counts, not which tiles
  help. Tests `shanten_brute.rs` (63,000 random hands, three properties) passed in 2.8 s.
- Not a proof for every hand. Known limits: the distance ignores how many copies remain; the
  hand checker ignores faan and flowers. The camera's tile recognition was not checked.
Watcher's correction: the "3s 3s 6s 7s" example was too narrow. 3s also helps; 2s, 4s, 6s, 7s
are Brian's "partially effective" tiles, which the 1-draw look counts and the table does not.

Brian (2026-11-15, verbatim): "* I thought the feature includes a subscribed feature that looks
at a hand with 5/8/11/14 tiles and is looking to hit the tiles that gives the best result when
1 away from winning. I know the check for completeness is trivial
* how is that not 100% proof? You mean the check for completeness is not robust?
* those are not the same as partially effective tiles. Early game you do not have enough tiles
to form a full hand, so those tiles are legitimate tiles because your next discard are still
irrelevant tiles"
Watcher's reply: (1) yes, that feature is `best_discards` (`crates/app-api/src/discards.rs`,
5/8/11/14 tiles; per discard: steps to ready, each useful tile with unseen copies, total;
works at any distance; ignores faan and flowers) (**measured**, code read). (2) No error is
known; "not 100 %" only meant the tests sample random hands instead of checking every hand.
(3) Brian is right: while the hand still lacks half-sets, a tile that forms a new half-set
lowers the distance and is fully effective, and the existing count already includes it.
"Partially effective" applies only once the hand has enough half-sets (near ready).

Brian (2026-11-15, verbatim): "* this touches on the earlier problem why I mentioned I wasn't
sure about the effective tile. for the 2 + 1 effective tile when 10 left, the exact math
probably isn't really 1/8, because you can draw irrelevant tile then the effective tile and
still gain the same effect, but if you draw the potentially winning tile before you draw the
effective tile you lose the effectiveness. This already isn't so simple for 1-away scenario,
needless to say for multiple-away
* real games also play differently. Opponent can aim for the same suite and keep your tiles,
and you can only claim if they hit (different from draw, claiming kills potential combinations)
* meaning if your left opponent is playing the same suite, your claim is minimized and your
potential with that suite is compromised. This needs to be taken into account when determining
which suite to go for early game.
* whether to play the same suite with the right player also becomes a decision, as them holding
the same tiles and not being able to claim 碰 tiles from them is also a small issue (this is
without considering their reduced EV from your defensive play is worth giving yourself a lower
offensive EV, which comes into consideration in a separate stage. We first finalize an improved
version of the value bot first)
* is there an effective way to do the calculation without having to do simulations to provide
estimates for the tradeoff between closer cheap hand vs a further rich hand"
Decision (Brian): first finalise an improved value bot; defence trade-offs come in a separate,
later stage. New requirement (Brian): the early choice of suit must account for opponents
collecting the same suit (left player: fewer chow claims; any player: tiles held back).

Watcher's exact sums (**measured**, script `chain.py` in the scratchpad, exact arithmetic, no
simulation). Brian's example (10 tiles left, 2 winning, 1 improving tile that adds 2 winning):
win chance within 1/2/3/4 draws = 20.0/37.8/53.3/66.7 % without the improving tile and
20.0/40.0/58.1/73.0 % with it. Irrelevant draws drop out: only the order of the relevant
tiles matters.
Step-chain method (proposal): chance to finish a plan within the draws left = a small table
over (steps done, draws used), each step hit with chance live helping tiles / unseen tiles.
Illustration with 12 helping tiles at every step, 80 unseen: 5 hits needed vs 6 hits needed =
17.5 % vs 6.9 % with 18 draws left (2.5 times), 3.2 % vs 0.7 % with 12 left (4.8 times). So
the cost of one extra step is not a fixed factor; it grows as the wall shrinks.

### Chain method worked on P3 (Brian asked 2026-11-15: "please show with a real example")

Position P3: 108 tiles unseen by Brian, wall 82, so about 20 own draws left. Points table
(`crates/hk/src/rules.rs` HALF_SPICY, **measured**): 3 faan 8, 4 faan 16, 5 faan 24, 6 faan 32.
Own draws only, no claims. First-hit counts are the watcher's hand count from the visible
tiles (**measured** by hand; 22 matches the engine's acceptance); later counts are estimates
(**inferred**). Script: `chain2.py` in the scratchpad.

| Plan | Hits to win | Helping tiles per hit | Finish chance, 20 draws | Points | Score |
|---|---|---|---|---|---|
| A: all pungs, 4 faan | 5 | 22, 12, 8, 6, 4 | 2.34 % | 16 | 0.37 |
| B: bamboo half flush + East pung, 6 faan | 6 | 26, 20, 15, 11, 8, 5 | 2.59 % | 32 | 0.83 |
| B': bamboo half flush without East, 4 faan | 6 | 23, 18, 14, 10, 7, 5 | 1.80 % | 16 | 0.29 |

Throw East keeps A and B': best 0.37. Throw 3p keeps A and B: best 0.83. The chain prefers
3p, as Brian did. With 14 draws left: A 0.44 %, B 0.35 %, B' 0.23 %; B x 32 still leads A x 16.
Caveat: without claims all chances are small, and all pungs gains most from claims (a pung
comes from any opponent), so the gap would narrow once claims count.

Brian (2026-11-15, verbatim): "* pung can always receive from opponent if they don't need the
same suite, so your chance of getting there is much higher, which was why I previously
emphasized the problem
* you also need to consider full flush and half flush + all pungs (also 6 hits)"

P3 chain redone with claims and Brian's two extra plans (script `chain3.py` in the scratchpad;
**inferred**: first-hit counts by hand from visible tiles, later counts estimated, claim model
assumed). Model: per round, own draw plus three discards; a pung tile counts 1 + 3w, a chow
tile 1 + w (left player only), a tile that only pairs a single counts 1, a winning tile
1 + 3w; w = chance an opponent throws the tile. 20 rounds, 108 unseen. Claimed hands lose the
concealed faan.

| Plan | Faan (points) | Finish, w = 1 | Score | Finish, w = 0.5 | Score |
|---|---|---|---|---|---|
| A all pungs | 3 (8) | 52.3 % | 4.2 | 19.9 % | 1.6 |
| B half flush + East | 5 (24) | 43.5 % | 10.5 | 15.9 % | 3.8 |
| B' half flush, no East | 3 (8) | 38.4 % | 3.1 | 12.8 % | 1.0 |
| C full flush | 7 (48) | 36.2 % | 17.4 | 11.4 % | 5.5 |
| D half flush + all pungs + East | 8 (64) | 19.0 % | 12.2 | 3.9 % | 2.5 |
| D' half flush + all pungs, no East | 6 (32) | 13.6 % | 4.4 | 2.4 % | 0.8 |

Findings: (1) claims lift every plan many times over, so they cannot stay out of the first
build; (2) every strong plan keeps the bamboo and sheds the pins; (3) the full flush is the
top plan and does not need East, so "best plan only" ties East with 3p again. East's worth
shows only in the second plans (B 10.5 and D 12.2 with East, against 3.1 and 4.4 without).
So the score must count more than the single best plan.

Brian (2026-11-15, verbatim, sent while the rerun was in progress): "meaning your previous 1/8
correction is not necessarily sufficient either because it depends also on opponent hitting
your winning tile"
Brian (2026-11-15, verbatim): "instead of score, the prediction should give percentage of
winning, and the expected value of the winning hand"
Decision (Brian): the solver reports two numbers per discard: the chance of winning, and the
expected value of the winning hand. No blended score as the output.
Watcher's proposal for combining plans into those two numbers (**inferred**): group plans
that contain each other (every full flush is also on the half-flush path). The widest plan in
the group gives the win chance; the narrower plans say how the wins split by points. P3
illustration from the table above, opponents throw everything: keep East = win about 44 %,
average winning hand about 44 points; throw East = win about 38 %, about 46 points.

Brian (2026-11-15, verbatim): "(then the better move is simply its product)"
Decision (Brian): the solver picks the discard with the highest product of win chance and
expected value of the winning hand.
Brian (2026-11-15, verbatim, follow-up): "or expected faan, equivalently"
Watcher's note (**measured**, points table in `crates/hk/src/rules.rs`): not equivalent for
choosing. Points are not proportional to faan (3 faan 8, 4 faan 16, 7 faan 48). Example: 50 %
of a 3 faan hand = 1.5 faan or 4 points; 20 % of a 7 faan hand = 1.4 faan or 9.6 points. The
two rules pick different discards. Open: choose by points (and show faan in the hint), or
choose by faan.

Brian (2026-11-15, verbatim): "* does the chain consider pung and chow?
* you typically have half-flush, that contains half-flush + pung, as well as full-flush (which
then contains full-flush + pung)
* yes, use points because of how the pay jump works across faan

1. why roughly true?
2. see bullet points
3. not necessarily if you have good value (like the 2 faan E) and pung structure
4. yea but how to estimate?"
Decisions (Brian): the product uses points, not faan. Plans nest as a tree: half flush
contains half flush + all pungs and full flush; full flush contains full flush + all pungs.
The solver does not always push to the full flush: with good value tiles (a 2 faan East) and
a pung structure, the half flush + all pungs branch can be worth more (8 faan = 64 points
against 7 faan = 48).
Watcher's proposals for the estimates (**inferred**, not built): (a) helping-tile counts:
exact recount for the first two hits (the hand after each possible first hit is known); for
later hits, average counts by plan type and hits left, measured once from the lab's own
self-play games. (b) opponent throw rate: measured from games as "how often a player throws a
tile of a suit", split by whether that player shows a meld in the suit or has thrown none of
it; start with two fixed values and tune by test matches.

## Plan for the improved value bot (draft 2, summary given to Brian 2026-11-15; not approved)

Brian (verbatim): "too many things happening right now. Please take a step back and summarize
what you'd like to plan for the improved value bot"

Goal: for each discard the bot works out the chance of winning and the expected points of the
winning hand, and throws the tile with the highest product. The hint shows both numbers.

Decided by Brian: two numbers, product, points not faan; 1 draw ahead on draws that matter, no
2-draw look; early game by suit plus an all-pungs check; plans nest as a tree under the half
flush; missing tiles deducted directly; claims matter (pung from anyone, chow from the left);
opponents collecting the same suit affect the suit choice; improved value bot first, defence
trade-offs in a later stage.
Proposed by the watcher, not yet accepted: the chain as the way to get the win chance; exact
tile counts for the first two hits, measured averages after; throw rates measured from games.
Not yet estimated: the running time of one chain over the whole plan tree.

Build order (proposal): 1 Brian's positions as tests. 2 Plan tree plus chain, own draws and
claims with fixed weights. 3 The 1-draw look near ready. 4 Measure average tile counts and
throw rates from lab games. 5 Test matches against today's value bot. 6 Hint shows both
numbers. Later: reading opponents' suits, defence, rerun of search and phone model.

Brian (2026-11-15, verbatim): "but it's not that simple, some draw gives more future
possibilities and some less, some give more value and some less"
Watcher's reply: agreed; each possible next draw gets its own follow-up hand with its own two
numbers, weighed by live copies (exact for the next two draws, averages beyond).

Brian (2026-11-15, verbatim): "the solver should provide a few things for the later
offensive/defensive stage:
* best offensive move:
* expected offensive-only consideration winrate
* expected points from winning hand
* expected timings from winning / N steps away
the third is relevant for understanding the race condition against the opponents. The discard
should be taken into account to guess the expected timings to calculate the true winrate from
the race conditions using the offensive-only winrate"
Decision (Brian): the solver's outputs per discard are (1) the best offensive move, (2) the
win rate from offence alone, (3) the expected points of the winning hand, (4) the expected
timing of the win and the steps away. The later offence/defence stage uses (4) to turn the
offence-only win rate into a true win rate against the opponents' race.
Watcher's note (**inferred**): the chain already yields the chance of having won by each
round, so (4) costs nothing extra; the full curve can be output, not only its average.

## Brian's simpler outline: choose a route first (2026-11-15, verbatim)

"Responding to your previous message, the improved value bot should indeed be not too
complicated. Maybe it is worthwhile to adopt a simpler approach. Here is the outline.
Typically, there are a few strategies at the beginning:
* all pungs no suite
* all chow + no claim + self draw
* 13 orphans
* half-suite
these are distinctive strategies that dictate the flow of the offensive game. The chance of
winning should be calculated first, together with the expected value, to decide which route to
go for. If committing to the first three strategies, the offensive decision is linear and no
difficult decision to make in the future besides folding (can just use improved value bot).
When going for half-suite, it includes the possibility of developing into higher tier such as
full-suite or/and pung. The offensive-winrate multiplied by the expected points should be
compared to assess whether to favour speed or potential value.
Does that make any sense?"

Watcher's reading (**inferred** unless marked): it makes sense and is smaller to build. The
lab's hand reading already lists these routes as targets (`crates/analysis/src/target.rs`:
mixed, half flush, full flush, honours, orphans, each with any sets, all pungs or all chows;
**measured**, code read earlier). Today it picks the nearest; the change is to pick by win
chance x points. Gap to settle: hands that reach 3 faan through dragon or wind pungs (with
flowers, concealed hand or self-draw) fit none of the four routes.

Brian (2026-11-15, verbatim): "* same principle applies when getting enough pairs while going
for orphans to pivot to pungs
* yes consider fifth route, although when taking those into account, I suspect the hand offers
enough strength to directly go for half-flush. Let's test this hypothesis"
Decisions (Brian): routes are re-compared every turn, so a route can pivot (orphans to all
pungs once enough pairs arrive, as half flush can grow into full flush or all pungs). A fifth
route (3 faan through dragon or wind pungs) is included. Hypothesis to test (Brian): hands
that qualify for the fifth route are usually strong enough to go for the half flush directly.
Watcher's proposed tests (**inferred**): (1) once the route comparison exists, count over
many dealt hands how often the fifth route scores highest and how often the half flush beats
it; (2) play the bot with four routes against the bot with five over 4,000 hands. Counting
wins by route in today's bot games would be biased, because today's bot picks the nearest
route.

Brian (2026-11-15, verbatim): "1. yes you need new simulations
2. need to consider claims, otherwise massively downplay 對對糊. Can naively assume opponent
always hit tile, although need to also consider whether to take the claim or not with similar
principle (as claim often massively reduce helping tiles. Maybe not completely necessary for
estimation only"
Decisions (Brian): the fifth-route hypothesis is tested with new simulations, not with
today's bot games. The estimate counts claims; first version assumes opponents always throw
the tile. Whether to take a claim is decided by the same principle (a claim often cuts the
helping tiles a lot); the estimate itself may leave that out.
Watcher's note (**inferred**): "always throw" favours pung routes most (a pung tile counts
from four places, a chow tile from two, a pairing tile from one), so it will overstate all
pungs; the honest-percentages check will show by how much.

Brian (2026-11-15, verbatim): "the overestimation is indeed an issue. Maybe assume a certain
percentage the opponent hold back the tile?"
Decision (Brian): the estimate assumes opponents hold back the needed tile a certain share of
the time (one setting), not "always throw". Value open. Watcher's proposal (**inferred**):
start at 50 % held back, then set it by the honest-percentages check (the value at which
predicted win rates match the actual ones in simulated games). P3 illustration: all pungs
finishes 52 % at "always throw" and 20 % at half held back.

Brian (2026-11-15, verbatim): "can assume high possibility of keeping 番子 and roughly 1/3
possibility of keeping the same suite"
Decision (Brian): starting hold-back settings: opponents keep a suited tile you need about one
time in three; they keep an honour tile (番子: winds and dragons) with high probability.
Number for "high" open; watcher proposed two in three as the start, tuned later.

Brian (2026-11-15, verbatim): "but the winrate depends on how good the strategy itself is
(knowing which is the most effective offensive tile to hit), lesser so how often the bot is
holding back the tiles we need to claim"
Watcher's correction (**inferred**): Brian is right; matching predicted and actual win rates
mixes several causes (the bot's own later play, opponents winning first, typical tile counts,
hold-back). So: (1) set the hold-back values by a direct count in simulated games (how often
an opponent who gets a tile the bot needs throws it), never by win-rate matching; (2) check
the offence-only win rate in games where opponents draw and discard as usual but never
declare a win, so the race does not blur the comparison.

Brian (2026-11-15, verbatim): "I guess instead of setting a percentage, just turn the chain
calculation into a full simulation of 4 improved value bots? This way we get fourfold the data
and a real percentage"
Watcher's reading (**inferred** unless marked): two ways to do this. (a) Simulate ahead of
time: four bots play about a million hands; the lab records, per route, tiles needed, live
helping tiles, rounds left and sets shown, how often the hand won, its points and the round
of the win; the bot looks these up at play time. Claims, hold-back and the race are then real
counts, no settings. The lab already has such a table (`crates/search/src/target_value.rs`,
**measured**, code read) but it has no live-tile key and the `ev` bot used it for the nearest
route only. (b) Simulate at every move: this is what the expert level does today; a 4,000-hand
match takes about 55 minutes (sourced: phase4-build.md), too slow for the value bot.
Circular point: the table comes from bots that play by the old rule, so build it, let the new
bot use it, replay, rebuild; two or three rounds. Cost estimate: 4,000 value-bot hands take
16 s (sourced), so a million hands take about 70 minutes per round. Limit: the numbers fit
the lab's bots, not human opponents.

Brian (2026-11-15, verbatim): "old way as in value bot? do not use old way. Use as much as
possible the improved way"
Decision (Brian): the simulations that fill the table are played by improved bots, not by
today's value bot. Watcher's proposal for the start (**inferred**): round 0 is played by a
first improved bot that compares routes with the simple chain estimate and Brian's starting
assumptions (opponents keep a suited tile 1 time in 3, an honour tile with high probability);
each later round is played by the bot that uses the previous round's table.

Brian (2026-11-15, verbatim): "can you do an estimation as to how many hands is needed for
each entry's winrate to not change too much? how many entries are we expecting in the table
anyways?"
Watcher's estimate (**inferred**, arithmetic only): entries = 8 routes x 7 values of tiles
needed (0 to 6) x 6 bands of live helping tiles x 5 bands of rounds left x 5 counts of sets
shown = 8,400, of which perhaps 3,000 to 5,000 occur often. Error of a measured win rate =
sqrt(p(1-p)/n): at a 20 % win rate, 1,600 samples give about 1 point either way, 6,400 give
half a point; average points need about 400 wins per entry for about 1 point. One million
hands x 4 seats x about 18 turns = about 72 million records, about 8,500 per entry on
average, but uneven: common entries get far more, rare ones a few hundred. Rare entries
should be merged with neighbours or fall back to the simple estimate below about 500 samples.
Only the route a bot actually plays yields an outcome, so routes the bots avoid get thin data.

Brian (2026-11-15, verbatim): "how much CPU/GPU does that use and how much of those resources
are available on the PC? I suppose PC is training DouDiZhu still"
PC state (**measured** 2026-11-15, job queue status, nvidia-smi, CPU load): queue job 120
`train-2c` runs in the gpu lane since 00:11 (python from the `doudizhu` conda env); GPU 96 %
busy, 6.9 of 16.4 GB used; CPU load 20 %; one benchmark (`m1-throughput-4`) waits in the quiet
lane. While a gpu job runs the cpu lane allows 1 job of 8 threads; otherwise 2 jobs of 8.
Simulation need (**inferred**): CPU only, no GPU, little memory. At value-bot speed (4,000
hands in 16 s, sourced; thread count of that run not checked, assumed 16 to 18) a million
hands take about 70 minutes on all threads, about 2.5 hours on 8. The improved bot does more
work per move (factor not measured; guessed 2 to 10), so a round of a million hands is
roughly 5 to 25 hours on 8 threads. A first round of 200,000 hands (about 14 million
records) is a fifth of that.

## Overnight run started (2026-11-15, about 04:20)

Go-ahead (Brian, verbatim): "this task should have higher priority than any other tasks in the
queue (including the benchmark task) besides the DouDiZhu training that it should run along
side. Do go with the million as I am going to bed now and you have plenty of time to test (you
may even go beyond million if needed). And when the test converge and the improved value bot
is ready, test it against the old value bot. You may also test the initial chain method
against value bot."
Plan file: `C:\Users\brian\.orchestrator\mahjong-watch\plan-improved-value-bot.md` (tasks,
and the defaults the watcher took where Brian had left a value open). Work copy of the lab:
`C:\Users\brian\.orchestrator\mahjong-watch\lab` (pin 03d6f7c, not a git checkout; no project
checkout is touched). A helper is building the bots; nothing is measured yet.

### Overnight run, step 1 result (helper report, checked figures to follow)

Built in the work copy: route evaluation (all three flush suits, value pung from one copy),
bots `chain` and `route`, table fitter `fit_routes`, position tool `route_check`. Existing
bots give identical match numbers before and after (helper's run; to be rechecked).
Initial chain bot, 200 deals against three value bots: -9.28 +/- 2.13 points per hand, wins
3.1 % of hands (helper's run). It fails P1 (throws S for a concealed all-chows hand) and P3
(throws 6s). Cause found by the helper: the simple chain credits far hands with wide
acceptance at every later hit and ignores that others end the hand first; simulated hands
show all chows at 3 to 5 hits away wins 0.1 to 3 % where the chain said 50 to 90 %.
A table fitted from only 2,000 simulated hands already makes the table bot throw 3p on P3.
Speed: 238 simulated hands per second on 8 threads, so a million hands take about 70 minutes.
Next: table bot uses table numbers only; a second chain bot uses measured average helping
tiles for later hits.

### Overnight run, step 2 (helper report; the watcher reran the route tests only)

Changes: the table bot now uses table numbers only (thin entries borrow from coarser ones); a
second chain bot (`chain2`) uses measured average helping tiles for later hits; the table's
hits-needed range was widened from 0-6 to 0-9 by the watcher (far hands were lumped together).
200 deals against three value bots, tables from only 2,000 simulated hands (helper's runs):
chain2 -5.91 +/- 2.57 (wins 7.5 %), table bot -6.58 +/- 2.11 (wins 10.0 %), first chain
-9.28 +/- 2.13 (wins 3.1 %). All three lose clearly to today's value bot at this stage.
Positions: P1 both new bots throw 8m (pass). P3 chain2 throws 6s, table bot throws E (fail;
the far entries were lumped and tied). P2 chain2 throws W, table bot 1p.
Round 0 queued 2026-11-15 about 05:25 as queue job 122: four `chain2` bots, 1,000,000 hands,
a less likely route tried in 20 % of decisions. Speed at start about 280 hands per second.

## Overnight run, 05:57–06:10: round 0 result, a flaw in the table, and the fix

**Round 0 (queue job 122) finished.** Measured: 1,000,000 hands in 4,074 s (245 hands/s, 8 threads).
566,920 wins declared (57 % of hands end in a win). 1,887 entries held 2,000 or more records.

**The first table was wrong as a route chooser (measured on Brian's positions).**
- The recorder kept one record per discard, under the route the bot followed.
- It counted the record as a win when the seat won *with any hand*.
- So a far route's entry held wins that came from other routes. Example from the East hand:
  "half flush + all pungs, 6 tiles needed" showed a 28.8 % win rate at 21.9 points.
  The 21.9 points prove those wins were mostly cheap hands, not that route.
- With that table the table bot still threw East on the East hand.

**Fix (the watcher's choice, inside Brian's outline; Brian can overrule):**
- At every discard the recorder now writes one record for **every** route the hand can still take.
- A win counts only under the one route of the finished hand (full flush, half flush + all pungs, ...).
- An entry now means: "from this route state, how often do improved bots win **by this route**,
  and with how many points".
- Wins by different routes exclude each other. So win chance x points, summed over the routes,
  is the hand's expected points. The table bot now ranks discards by that sum.
  Setting `ROUTE_VALUE=max` ranks by the best single route instead (Brian's first wording);
  both get a match.
- Known limit (inferred): a route the bots rarely follow shows a low win rate, which keeps them
  off it. The 5 % exploration works against that; it may not remove it.

**Trial (3,000 hands only, far too thin to trust):** East hand: table bot throws West, keeps East.

**Rerun:** queue job 124 `mj-route-round0b`, started 06:04, about 190 hands/s, about 90 minutes.
Job 123 (round 1 on the flawed table) was cancelled by the watcher after 48 s. The round 0
files stay in `runs\round0` for reference; its `helpers.json` still feeds the second chain bot.

## Overnight run, 07:25–07:35: rerun of round 0 and first real matches (all measured)

**Round 0 rerun (job 124):** 1,000,000 hands, 208 hands/s, 80 minutes. 426 million records.
3,766 entries hold 2,000 or more records (was 1,887).

**Matches, 4,000 deals each, seats rotated (16,000 hands), one bot against three value bots:**

| Bot | Points per hand | Hands won | Deals in | Average faan of wins |
|---|---|---|---|---|
| First chain bot (job 125) | -9.89 ± 0.42 | 3.2 % | 20.7 % | 5.3 |
| Table bot, round 0 table, ranks by the sum over routes | +0.23 ± 0.73 | 17.0 % | 20.9 % | 5.6 |
| Table bot, round 0 table, ranks by the best single route | +0.35 ± 0.68 | 19.1 % | 20.4 % | 5.3 |
| (value bots in the same matches) | about 0 | 25–26 % | 17–18 % | 4.7–4.8 |

- The first chain bot loses heavily. It wins 3 hands in 100.
- The table bot is level with the value bot within the error. It wins fewer hands, but richer ones.
- The table still comes from chain bots' play. Round 1 (job 126, started 07:25) plays table bots.

**Brian's positions with the round 0 table:**
- First hand (8m expected): sum mode throws West; best-route mode throws 8m.
- East hand: sum mode throws West and keeps East; best-route mode throws East.
- Third hand: both throw 1p, same as the value bot.

## Overnight run, 08:15–08:20: round 1 (all measured)

**Round 1 (job 126):** four table bots, 1,000,000 hands, 343 hands/s, 48 minutes.
91 % of hands ended in a win (66 % among chain bots).

**Table change, round 0 to round 1** (3,277 entries with 2,000+ records in both):
mean change of win rate 1.8 points; largest 34 points (all pungs, 1 tile needed, 3 sets shown).
Not settled: the rule asks for no change above 1 point. Round 2 (job 127) is queued.

**Matches with the round 1 table, 4,000 deals each, against three value bots:**

| Table bot | Points per hand | Hands won | Deals in | Average faan |
|---|---|---|---|---|
| ranks by the sum over routes | +0.51 ± 0.74 | 16.8 % | 21.3 % | 5.8 |
| ranks by the best single route | +0.95 ± 0.73 | 18.5 % | 20.8 % | 5.6 |

Both are a little up on round 0 (+0.23, +0.35). Neither lead is clear of the error yet.
The three positions give the same throws as with the round 0 table.

## 2026-11-16: round 2, round 3 queued, and the comparison bar for Brian

**Round 2 (job 127, measured):** 1,000,000 hands, 329 hands/s. Change from round 1 (3,439 entries
with 2,000+ records in both): mean 0.6 points, largest 20 points (all pungs, 1 needed, 3 sets
shown). Not settled yet. Round 3 (job 128) queued from the round 2 table.

**Comparison bar (Brian asked to play locally and compare hints):** the local game
(http://127.0.0.1:5180, scratch copy of the app) now shows, on every turn of Brian's seat:
value bot's pick | table bot, sum mode | table bot, best-route mode | first chain bot.
Each table and chain pick names the route it follows, its win chance and its points.
The dev server runs `route_check` on the round 2 table for each position (about a second).
Checked in the browser on a fresh deal: Value 9p | Table (sum) W | Table (best route) 9p | Chain 9p.

## 2026-11-16: Brian on the West-versus-8p hand (decision, his words)

Hand: 678m 8p 12366s W N WhWh G, first discard. Table bot (sum mode) threw West; value bot 8p.
Brian, verbatim: "forget about the table. throwing 8p is fundamentally better than W. This has
nothing to do with them having the same group or anything. The table should only be used if there
is a dispute between different suites. Going for a half-suite, it's clear to keep fanzi and sokzi."

Meaning taken (to confirm with Brian): two levels. The table only chooses the route and the suit.
Once the route is chosen, the discard follows the route plainly: for a bamboo half flush, keep
honours (番子) and bamboos (索子); throw the other suits first.
Brian, verbatim (same day): "also the table is too coarse. This will not help if both suites have
the same number of tiles and fall within the same band"

## 2026-11-16: is the fifth route (value pungs) useful? (measured)

Brian's question: "I once questioned the need of separating into the 5th type. What's the result? is it useful at all?"

Method: `runs/agg5.py` sums the per-discard records by the route of the finished winning hand (`Route::of_win`).
"Value pungs" here means every win that is not a flush, not all pungs, not all chows and not a limit hand.
Shares are weighted by discards, so they approximate the share of winning hands.

| Who plays (four equal bots) | Value-pungs share of wins | Points per value-pungs win | Half flush points per win |
|---|---|---|---|
| Table bot, round 3, 1,000,000 hands | 0.4 % (0.6 % early in the hand) | 14.5 | 21.0 |
| Value bot, 20,000 hands (`runs/value20k`) | 3.1 % (3.4 % early) | 14.6 | 20.5 |

Value bot's wins by route: half flush 65.1 %, all pungs 23.4 %, full flush 6.2 %, value pungs 3.1 %,
half flush + all pungs 1.1 %, limit 1.0 %, all chows 0.0 %.

Reading (inferred): the fifth route brings about 3 % of wins at the lowest payout, so about 2 % of points.
Hands that start near value pungs mostly finish as another route.
Not tested: a head-to-head match of a bot with the route against a bot without it.
Round 3 (queue job 128) finished: 1,000,000 hands, 891,802 wins declared.

## 2026-11-16: Brian's three-stage plan (replaces the table; agreed in discussion, not built yet)

Brian's words: "Instead of using the table, consider a simpler plan". Value pungs is dropped as a route.

**Stage 1, beginning of the hand.** Routes: half flush, all pungs, thirteen orphans, all chows.
- Brian: "The orders are only for ties in terms of route distance. Except for all chows that needs +2, route with lowest distance gets highest priority".
- The discard is the tile least relevant to the leading route.
- Suit for half flush (Brian): "should contain both if the third fewer tile flush still exists, then your default".
  The default: most tiles; then more pairs and neighbours; then fewer copies already visible.
- Claims in stage 1 (Brian): "never kong"; "never chow", later relaxed; pung only if "all pung is highest priority route",
  "pung fanzi when half-suit highest priority", "and lone pair of the half-suit".
- Chow (Brian): "can chow if have at least 8 effective tiles for half-flush"; the count is
  "bamboos with honours that are paired or tripped" (flush-suit tiles plus honours held as a pair or pung).
  Also: "the chow cannot destroy the remaining tile". Agreed test: half flush must be closer after the chow ("that's fine").
- Win (Brian): "yes take legal win always".

**Stage 2, every tile fits the leading route.** Checks in order: distance, faan, helping tiles, connection,
number of exposed tiles of the same type (more exposed: discard first), random.

**Stage 3, two tiles from winning.** The 1-step and 2-step count.

Claims in stages 2 and 3: my proposal is the value bot's rule plus "never claim while all chows leads". Brian has not confirmed this line.
Status: Brian has not said "go". Test planned: 4,000-deal matches against three value bots.

## 2026-11-16: three-stage bot built and measured (work copy only)

Brian said "go" (twice; the second after a pause). Code: `lab/crates/search/src/stage.rs`, bot name `stage`.
Unit tests: 7 pass (the 8p hand, the pung rules, the chow rule with 1357s, the all-chows lead).
The comparison bar is removed from the local game, as Brian asked.

How the build reads the plan where the plan left room (my readings, not his decisions):
- Suits for half flush are ordered by flush distance first, then most tiles, pairs and neighbours, fewest visible.
- Stage 3 starts when the hand is ready or one tile from ready, for any legal win.
- Kongs after stage 1 follow the value bot's rule. Claims in stage 2 need the flush to be closer afterwards.
- Stage 3 claims compare the 1-step and 2-step count with and without the claim.

Matches (measured): one `stage` bot against three value bots, 4,000 deals, same deals each run.

| All-chows lead needed | Points per hand | Win rate | Deal-in | Average faan |
|---|---|---|---|---|
| 2 tiles (the plan) | +0.42 ± 0.58 | 23.5 % | 17.9 % | 4.6 |
| 3 tiles | +1.49 ± 0.64 | 24.6 % | 18.8 % | 4.9 |
| 4 tiles | +1.74 ± 0.64 | 24.3 % | 18.8 % | 5.1 |
| never | +1.71 ± 0.65 | 23.9 % | 19.1 % | 5.1 |

Value bots in the same runs: win 23 to 24 %, faan 4.8, deal-in about 16.5 %.
Earlier bots for comparison: table bot at best +0.95 ± 0.73 (win 18.5 %), chain bot -9.89.

Logged hands (measured, `route_check`):
- 8p against West hand (678m 8p 12366s W N WhWh G): throws 8p. Matches Brian.
- P3 (35667p 1336677s E W): all pungs leads (distance 4; flush 5; chows 3). Throws 3p and keeps East. Matches Brian.
- P2: throws 1p, same as the value bot.
- P1 (8m 1357p 1123s SS W Wh G): with the 2-tile lead, all chows leads (chows 3, flush 5) and the bot throws White.
  Brian wanted 8m for the half flush. With a 3-tile lead the bot throws 8m.

Open for Brian: keep the 2-tile lead, or raise it to 3 or 4. The setting is the env var `STAGE_CHOWS_LEAD`; the default in code is still 2.
Not done: no defence in the bot (deal-in is about 2 points above the value bots); not merged into the real lab or the app.

## 2026-11-16: 40,000-deal matches, three-stage bot against three value bots (measured)

Brian asked for more hands to reduce the noise. Queue jobs 129 and 130; files `runs/match-stage-40k.txt` and `runs/match-stage-40k-lead4.txt`.

| All-chows lead needed | Bot | Points per hand | Win rate | Deal-in | Average faan |
|---|---|---|---|---|---|
| 2 tiles (the plan) | stage | +0.55 ± 0.18 | 23.5 % | 18.3 % | 4.6 |
| 2 tiles | value (three) | -0.76, -0.08, +0.29 (± 0.18 to 0.19) | 23.1 to 24.3 % | 16.5 to 16.7 % | 4.8 |
| 4 tiles | stage | +1.84 ± 0.21 | 24.3 % | 18.6 % | 5.1 |
| 4 tiles | value (three) | -0.74, -0.53, -0.56 (± 0.19) | 23.7 to 23.9 % | 18.3 to 18.7 % | 4.7 to 4.8 |

Draws 6.4 % and 6.0 %. Reading (inferred): with the 4-tile lead the stage bot wins more often on a discard, so the value bots deal in more.

## 2026-11-16: all-chows lead set to 4; teacher matches queued overnight

Brian (verbatim): "use 4, and how does it perform against the teacher"; then "just run the test overnight".
- The code default for the all-chows lead is now 4 (`stage.rs`). Unit tests: 7 pass.
- Teacher = `pimc:samples=64,candidates=4,belief=models/belief.mjnn` (value as baseline, fast rollouts).
- Timing trial (measured): 24 deals against three teachers took 156 s on 6 threads.
- Queued in the PC job queue, cpu lane, script `jobs/teacher.ps1` (a finished part is skipped on a rerun):
  - Jobs 131 to 134: one stage bot, one teacher, two value bots; 4 parts of 1,000 deals, seeds 11 to 14.
    Results: `runs/match-teacher-part1.txt` to `part4.txt`.
  - Jobs 135 to 138: one stage bot against three teachers; 4 parts of 500 deals, seeds 21 to 24.
    Results: `runs/match-teacher3-part1.txt` to `part4.txt`.
- The "±" the match tool prints is a 95 % interval (`ci95`).

## Overnight teacher matches, all-chows lead 4 (2026-11-16, measured)

Brian chose lead 4 ("use 4"). Both matches ran through the job queue, jobs 131 to 138.
The ± is a 95 % interval. I combined four equal parts per match.

Match A: new three-stage bot, teacher, two value bots. 4,000 deals (seeds per part).

| Bot | Points per hand | Win rate | Deal-in | Average faan |
|---|---|---|---|---|
| Three-stage bot | +1.18 ± 0.67 | 24.0 % | 19.4 % | 5.1 |
| Teacher (pimc, 64 samples) | +0.74 ± 0.65 | 23.9 % | 17.9 % | 4.9 |
| Value bot 1 | -1.01 ± 0.58 | 23.6 % | 18.1 % | 4.8 |
| Value bot 2 | -0.90 ± 0.61 | 24.0 % | 18.7 % | 4.7 |

Match B: new three-stage bot against three teachers. 2,000 deals.

| Bot | Points per hand | Win rate | Deal-in | Average faan |
|---|---|---|---|---|
| Three-stage bot | +0.42 ± 0.97 | 23.5 % | 18.5 % | 5.1 |
| Teacher, average of three | -0.14 | 24.0 % | 18.4 % | 4.9 |

Reading (my inference): the three-stage bot is level with the teacher within the noise.
It is clearly ahead of the value bot. It wins bigger hands (5.1 faan against 4.9).
It deals in more than the teacher in match A (19.4 % against 17.9 %); it has no defence yet.

## Deal-in breakdown: three-stage bot against three value bots (2026-11-16, measured)

Tool: `lab/crates/search/examples/dealin.rs`, queue job 139, 20,000 deals x 4 seats, lead 4.
Full output: `runs/dealin-20k.txt`.

| | Three-stage bot | Value bot |
|---|---|---|
| Hands | 80,000 | 240,000 |
| Deal-in | 18.64 % | 18.46 % |
| Points paid per deal-in | 22.2 | 23.7 |
| Discards per hand | 15.61 | 15.64 |
| Win rate / average faan | 24.32 % / 5.08 | 23.80 % / 4.74 |

The gap is 0.18 points; the 95 % range is about +-0.31 (my calculation). So with lead 4 the
three-stage bot does not deal in measurably more than the value bot. The earlier larger gap
came from the lead-2 match and from small matches.

Where the two bots differ (deal-ins per 100 hands, by bucket):
- Honours: three-stage 7.00, value 6.27. It throws fewer honours (2.29 against 2.54 per hand)
  but each one is more dangerous (30.6 against 24.6 deal-ins per 1,000 discards). My inference:
  it keeps honours for the flush and all-pungs lines and releases them late.
- Tiles 3 to 7: three-stage 7.00, value 7.55. It throws more of them but earlier, when safer.
- Three or more steps from a win: three-stage 2.94, value 2.27. It spends more discards far
  from a win (9.1 against 8.6 per hand) and keeps pushing.
- Ready hands: three-stage 6.30, value 6.65.
- By stage: stage 3 gives 12.4 of its 18.6 deal-ins per 100 hands, on 3.8 discards per hand.

Game copy: the scratch game copy now plays the three-stage bot at level `hard` (hint and
opponents). Work-copy change: `bot_name("hard")` in `crates/app-api/src/api.rs`. The earlier
engine files are kept in the scratchpad folder `hkg\wasm-backup-mahjong-value`.

## 2026-11-16 afternoon: stage 1 pung rule corrected by Brian (measured)

Brian's words: "lone-pair is 邊張, 1 or 9, 2 or 8 when 1 or 9 absent respectively"; then
"also allow non-honour pung when ... half-suit 1st priority AND the suit fits the half-suit";
then "include pung but add instead of 2nd priority, there should be 3 pair/trip ready for the
half-suit". My readings: "absent" = not in the own hand; the claimed pair counts as one of the three.

Bot names in the work copy: `stage1` = first build (pair with no tile within two steps);
`stage2` = edge pair; `stage` = edge pair, or three pairs/triplets of flush tiles and honours.

Against three value bots, 40,000 deals, same deals each time, +- = 95 % range:
| Version | Points per hand | Win rate | Deal-in | Average faan |
|---|---|---|---|---|
| stage1 | +1.84 +- 0.21 | 24.3 % | 18.6 % | 5.1 |
| stage2 | +2.29 +- 0.21 | 24.9 % | 18.5 % | 5.1 |
| stage (version 3) | +2.54 +- 0.21 | 25.1 % | 18.4 % | 5.1 |

Queue jobs 142 and 143 (version 3 against three of stage1, three of stage2) write
`runs/v3-vs-v1.txt` and `runs/v3-vs-v2.txt`.

Brian, same day: "I mean 2 from ready, not 2 from winning". Stage 3 must start at two from
ready. The build starts it at one from ready. NOT changed yet. He expects one table test:
original, each change alone, both. Handed to the orchestrator at his request ("please ask the
orchestrator to do the task, because you stop responding to me when you work on the task yourself").

## 2026-11-16 evening: stage 3 from two from ready, and the four-version table (measured)

What was built (work copy only, `lab/crates/search/src/stage.rs` and `bots.rs`):
- `StageBot` now has two independent switches: `pung` (stage 1 pung rule version 1, 2 or 3)
  and `ready` (stage 3 starts at one from ready = 1, or two from ready = 2).
- New bot names: `stage-orig` (pung 1, one from ready), `stage-pung` (pung 3, one from ready),
  `stage-ready2` (pung 1, two from ready), `stage-both` (pung 3, two from ready).
  `stage`, `stage1` and `stage2` are unchanged (all one from ready).
- New example `examples/stage_speed.rs` times the decisions of one bot against three value bots.

My reading, not Brian's: how stage 3 scores a hand two from ready. Brian said only "2 from
ready". I extended the existing count one more step. The bot takes every draw that brings the
hand to one from ready. It weights each by its chance. It multiplies by the best
one-from-ready value after a discard that keeps the hand one from ready. Pruning: only draws
that lower the distance, only discards that keep the lower distance. A memo inside one decision
reuses one-from-ready values. Ready and one-from-ready hands score exactly as before.

Observation while testing (measured): the one-from-ready count is 0 for some hands the
distance measure calls one from ready. Example: `11123456789s1m5p` (distance 1, value 0);
`11123456789s5pE` (distance 1, value 6.6). No single draw makes the first hand ready.
This behaviour predates this change; I did not alter it.

Time per decision (measured, one thread, 300 deals, bot in seat 0 against three value bots,
other queue jobs running on the PC):
| Bot | Mean | Median | 99th percentile | Max |
|---|---|---|---|---|
| stage-orig | 0.20 ms | 0.14 ms | 1.3 ms | 10.6 ms |
| stage-pung | 0.18 ms | 0.14 ms | 1.0 ms | 3.2 ms |
| stage-ready2 | 1.34 ms | 0.18 ms | 14.8 ms | 179 ms |
| stage-both | 1.18 ms | 0.19 ms | 12.0 ms | 32.7 ms |
Match wall time (measured, queue jobs 144-148): one-from-ready bots against three value bots
about 4.3 minutes; two-from-ready bots about 13 minutes; the four-version table 25.7 minutes.
All under 30 minutes, so no extra caching was added.

Table of all four versions, 40,000 deals, bots rotated through all seats, +- = 95 % range
(`runs/stage-table4.txt`, job 144):
| Version | Points per hand | Win rate | Deal-in | Average faan |
|---|---|---|---|---|
| stage-orig (pung 1, one from ready) | -0.53 +- 0.22 | 23.5 % | 18.8 % | 5.1 |
| stage-pung (pung 3, one from ready) | +0.35 +- 0.22 | 24.4 % | 18.6 % | 5.1 |
| stage-ready2 (pung 1, two from ready) | -0.33 +- 0.22 | 23.5 % | 18.0 % | 5.1 |
| stage-both (pung 3, two from ready) | +0.52 +- 0.22 | 24.0 % | 18.0 % | 5.1 |

Each version against three value bots, 40,000 deals, same deals each time
(`runs/<name>-vs-value.txt`, jobs 145-148):
| Version | Points per hand | Win rate | Deal-in | Average faan |
|---|---|---|---|---|
| stage-orig | +1.84 +- 0.21 | 24.3 % | 18.6 % | 5.1 |
| stage-pung | +2.54 +- 0.21 | 25.1 % | 18.4 % | 5.1 |
| stage-ready2 | +1.89 +- 0.21 | 24.1 % | 18.2 % | 5.1 |
| stage-both | +2.50 +- 0.21 | 24.8 % | 18.1 % | 5.1 |

The stage-orig and stage-pung lines equal the earlier stage1 and stage lines digit for digit
(measured). So the one-from-ready path is unchanged.

My inference from the numbers: the pung rule gives most of the gain (about +0.7 points per
hand against value bots, +0.9 in the table). Two from ready lowers deal-in by about 0.3 to 0.8
points of percentage but does not change points per hand beyond the 95 % range. In the table
the two-from-ready versions beat their one-from-ready partners by about 0.2 points per hand,
inside the range of each line.

## Final summary and hand-over (2026-11-16 evening)

**What Brian found wrong with the old models (his comments while playing, earlier sections):**
the value bot and the teacher have no sense of direction early in a hand; they do not commit
to a winning pattern and they claim tiles that break the hand's shape.

**What fixed it (measured):** Brian's own three-stage plan as a rule bot.
| Step | Against three value bots, 40,000 deals |
|---|---|
| First build, all-chows lead 2 | +0.55 +- 0.18 |
| All-chows lead 4 | +1.84 +- 0.21 |
| Edge-pair pung rule | +2.29 +- 0.21 |
| Edge pair or three pairs (the baseline now, bot `stage`) | +2.54 +- 0.21 |
Against the teacher (lead-4 version, before the pung change): level within the noise.

**Tried and not adopted:** a fifth line for value pungs (3 % of wins); stage 3 from two from
ready (no change in points, deal-in 0.3-0.8 points lower, 6 times slower).

**Ranked next steps:**
1. Self-play network that starts from the baseline (approved; handed to the orchestrator;
   `plan-self-play-network.md`).
2. Defence in stage 3 from the lab's danger table (proposed, not approved, not built). Stage 3
   gives 12.4 of the bot's 18.6 deal-ins per 100 hands.
3. The one-from-ready count scores 0 for hands that no single draw makes ready (helper's
   finding, example `11123456789s1m5p`). Not fixed.
4. Merge the three-stage bot into the real lab and the game (not approved, not started).
   All code sits only in the work copy `C:\Users\brian\.orchestrator\mahjong-watch\lab`.
