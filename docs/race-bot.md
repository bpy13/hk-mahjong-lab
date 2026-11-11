# The race bot: equity over all four players' chances (B28)

Plan `docs/plans/2026-11-09-review-fix-and-race-bot.md`, Task 5. This file is the design
that Phase 3 (`crates/search/src/race.rs`, v0) and Phase 4 (learned estimates, the
self-play loop) build. Brian's decisions for it are B28 in bot-stream `docs/decisions.md`;
the plan header lists them. This file implements them; it does not re-argue them.

Every input names its source in this repo. Where a number is an estimate, the text says
"estimate" and gives the value used. Matches decide between methods; the formula's numbers
are never trusted on their own (B28, last point).

**Why a new bot.** The `ev` bot (`docs/ev-bot.md`) values offence as `P_win × Pay(me)` with
`P_win` from a table keyed by target kind, faan-shanten, wall bucket and melds. The table
knows nothing about live tiles or the other players' progress, and the `Pay(me)` floor made
the bot chase 10-faan hands it rarely completed (`ev-bot.md` §10: −7.23 points per hand with
the floor, −0.99 without). The race bot replaces `P_win` by a race: every player's chance
of winning, turn by turn, worked out from live tiles and from how each player is placed.

| Brian's decision (B28) | Where it lives here |
|---|---|
| Offence sums every player's chances; better-placed opponents make my EV negative | §3 (the race), §3.3 (equity) |
| A good draw raises my side; compare it with the later cost of throwing a useless tile | §4.3, §4.4 (improvement tiles and holding cost) |
| Nested decisions: fold, detour, push; a folded player's threat is about zero | §5 |
| One-away opponents who have not folded matter; throw dangerous-later, useless-now tiles early, including claimable tiles and faan honours | §4.2, §4.4, §4.6 |
| Two threats: fold against B, push again once B folds if nothing I hold loses to A | §5.3 (mode per threat, re-chosen every turn) |
| Pushing a cheap hand can be defence | §3.3 (my win ends the race, so it removes every opponent's later term) |
| Hand value enters only through the EV | §3.3: payout multiplies the hazard, nothing else |
| Table image left out at first | §5.2 (opponents' modes depend on public state only) |
| Learn from bot play, self-improving until it converges | §7, §8 |
| Never trust simulation blindly | §8.3 (gates by duplicate match), AC5–AC6 |

## 1. Notation and inputs

| Symbol | Meaning | Source |
|---|---|---|
| `me`, `o`, `i` | the acting seat; an opponent; any of the four players | `State` (`crates/hk/src/state.rs`) |
| `a`, `t(a)`, `h(a)` | a candidate action, the tile it discards now (none for a pass or a claim), my hand after it | `state.legal_actions_vec()`, `SeatView` (`crates/search/src/view.rs`) |
| `wall`, `U` | tiles left in the wall; tiles I have not seen (`wall` + the three concealed hands) | `state.wall_remaining()`, `State::unseen(me)` |
| `n_i` | draws left for player `i`: `wall` tiles dealt in turn order from the next drawer | derived from the turn order after `a` (a claim moves the turn) |
| `k` | a round: one draw by each player, `k = 1 … n` with `n = max_i n_i` | derived |
| `points(f)` | points per share for `f` faan (3 → 8 … 10 → 128) | `RuleSet::points`, rules §8.1 |
| `s_i` | player `i`'s state: far (F), one away (O), ready (R), folded (X) | §2 |
| `h_i(k)` | player `i`'s win hazard in round `k`: P(`i` wins in round `k` \| the hand is still running) | §2 |
| `S(k)` | P(the hand is still running at the start of round `k`) | §3 |
| `m_i` | player `i`'s mode: push, detour, fold | §5 |
| `L_i(k)` | expected live copies (tiles that complete `i`'s hand) among the unseen tiles | §2.2 |
| `W_i`, `H_{i,j}` | of those, the expected copies in the wall and in player `j`'s hand | §2.2, from the counts head |
| `φ_{j→i}` | P(player `j` discards a live tile of `i` that `j` holds or draws this round) | §4.2 |
| `q(m)` | P(one discard by a player in mode `m` completes a given ready opponent's hand) | §2.3 |

**From the belief model** (`docs/belief.md` §2, `crates/search/src/belief.rs`, trait
`Belief` in `ev.rs`), run once per decision, per opponent `o`: `tenpai_o` (P(ready now)),
`wait_o(t)` (P(`t` completes `o` \| ready)), `P_b(o)` (faan bucket of a discard win now),
`cnt_o(t)` (distribution of copies of `t` in `o`'s hand), `outcome_o` (no win / discard win /
self-draw). Derived as in `ev-bot.md` §1: `Pay(o) = Σ_b P_b(o) × points(b)`, `Pay_sd(o)` one
faan higher, `P_o(t) = min(tenpai_o × wait_o(t), 0.95)`, `P1_o(t)`, `P2_o(t)`, `E[cnt_o(t)]`.
Without a model, `HeuristicBelief` gives the same quantities from the danger tables
(`danger.rs`, `danger_tables.rs`; `ev-bot.md` §8).

**From the hand analysis** (`crates/analysis/src/target.rs`): `faan_shanten(h)` gives my
faan-aware shanten `s`, the best target `T` and its guaranteed faan `f_T`; the acceptance
count (`rank_discards`, `bots.rs`) gives the unseen tiles that lower `s`.

**Share multipliers** are exactly `ev-bot.md` §1's table (`L_disc`, `L_sd`, `L_other`,
`Shares(me)`; rules §8.2–8.3). The 12-tile rule matters twice in the worked example (§9).

## 2. Each player's state chain and per-turn hazard

### 2.1 States

Every player is in one of four states, from my point of view a probability vector
`p_i(k) = (p_F, p_O, p_R, p_X)`:

| State | Meaning | How I know it |
|---|---|---|
| far (F) | faan-aware shanten ≥ 2 | me: exact from `faan_shanten`; opponents: the rest after O and R |
| one away (O) | faan-aware shanten 1 | me: exact; opponents: v0 estimate `p_O = min(1 − p_R, 1.5 × p_R)` (estimate; `docs/solver.md` §4's calibration table puts mean shanten at 1.2–1.5 in the last two wall buckets); Phase 4: the `state` head (§7) |
| ready (R) | shanten 0 with a legal win available | me: exact; opponents: `tenpai_o` |
| folded (X) | the player has given up winning this hand and discards for safety | me: my mode (§5); opponents: v0 heuristic fold probabilities (§5.2); Phase 4: the `mode` head |

Transitions per round, for a player who is not folded: `F → O` with probability `a_i`,
`O → R` with `b_i`, `R` stays. A folded player stops moving. For me, `a_me` and `b_me` are
the acceptance count over `U` (exact counts, my view). For opponents, v0 uses
`a_o = b_o = 0.15` per round (estimate: a one-away hand accepts about 6–8 of 54 unseen tiles
and claims add a little); Phase 4 fits `a`, `b` per wall bucket and state from the logged
true states (§7).

### 2.2 Live tiles

For a ready player `i`, `L_i = Σ_t wait_i(t) × unseen(t)` (for me, exact: my waits times
unseen copies). Where the copies sit matters: a copy in the wall can be drawn by me; a copy
in `j`'s hand can only be fed. From the counts head, `H_{i,j} = Σ_t wait_i(t) × E[cnt_j(t)]`
and `W_i = L_i − Σ_j H_{i,j}`. For opponents v0 uses the waits head for `wait_o(t)`; the
copies of their waits in my hand are known exactly (I hold them).

### 2.3 Hazards

Player `i`'s win hazard in round `k`:

```
h_i(k) = p_R,i(k) × (1 − fold_i(k)) × [ d_i + g_i(k) ]
d_i      = W_i / wall                      self-draw: a live copy drawn from the wall
g_i(k)   = Σ_{j≠i} feed_{j→i}(k)           fed by another player's discard
feed_{j→i}(k) = φ_{j→i} × ( H_{i,j} × 1[k = first round j acts] + W_i / wall )
```

`W_i / wall` stays constant in expectation as the wall shrinks (every draw removes a live
copy with that same chance), so `d_i` is one number per path. Held copies are fed at most
once, in the first round the holder acts after the decision. The feed rate `φ_{j→i}` depends
on `j`'s mode toward `i` and on whether the tile is useful to `j` (§4.2).

For an opponent whose waits I do not know exactly, v0 collapses `feed` to a per-discard
rate: `g_o(k) = Σ_{j≠o} q(m_j)`, with `q(push) = 0.08`, `q(detour) = 0.04`, `q(fold) =
0.01` (estimates; `docs/belief.md` §4.2 measures 2.5 % deal-in per discard in wall bucket 4
with about 0.5 of some opponent being ready, so about 5 % per discard against a ready hand
averaged over all styles). Phase 4 measures `q(m)` from the mode labels. For the acting
seat's own discard `j = me`, round 1 uses the exact `P_o(t(a))` instead of `q`.

A player in O or F has hazard 0 this round and becomes a threat only through the chain.
That is how "one-away opponents who have not folded" enter: their `p_O` feeds `p_R` in the
next rounds, and nothing I throw later is safe against them unless they fold.

## 3. The race: competing risks and my equity

### 3.1 Competing risks

The hand ends at the first win (or when the wall runs out). Hazards within a round are
treated as exclusive (they are small):

```
S(1) = 1,   S(k + 1) = S(k) × (1 − Σ_i h_i(k))
P(i wins in round k) = S(k) × h_i(k)
P(draw) = S(n + 1)
```

This is the same bookkeeping as `ev-bot.md`'s `S(k)` for the fold plan, now applied to every
player, with each player's hazard built from live tiles and states instead of from one
table cell. It is the "sum every player's chances" of B28: if `Σ_o h_o` is large, `S(k)`
collapses before my hazard pays, and my equity goes negative even with a good hand.

### 3.2 Payments

When I win in round `k`: by a self-draw I collect `Pay_sd(me) × 3`; on `j`'s discard
`Pay(me) × 1`; if someone is liable for me (12-tile rule), `× 3` either way, paid by them.
When `o` wins: by self-draw I pay `Pay_sd(o) × L_sd(o)`; on my discard `Pay(o) × L_disc(o)`;
on another's discard `Pay(o) × L_other(o)` (0 unless I am liable for `o`). `Pay(me)` is
`points(f_T)` plus the faan my win way adds (self-draw +1; the cap applies). No floor and no
table mean: hand value enters only through these payments (B28).

### 3.3 My equity for action `a`

```
E(a) = Σ_k S(k) × [ d_me(k) × Pay_sd(me) × Shares_sd
                  + Σ_j feed_{j→me}(k) × Pay(me) × Shares_disc
                  − Σ_o p_R,o(k)(1 − fold_o(k)) × ( d_o × Pay_sd(o) × L_sd(o)
                                                  + feed_{me→o}(k) × Pay(o) × L_disc(o)
                                                  + Σ_{j≠me,o} feed_{j→o}(k) × Pay(o) × L_other(o) ) ]
```

where `feed_{me→o}(1)` is `P_o(t(a))` for the tile I throw now and `q(m_me)` or the holding
plan (§4.4) later. Two consequences Brian asked for fall out without extra rules:

- **A cheap push is defence.** My win in round `k` sets `S(k + 1)` to zero for everyone;
  the opponents' later terms vanish. A 3-faan hand with a high hazard can beat folding when
  an opponent's expected payment is large.
- **Better-placed opponents make my equity negative.** With `p_R,o` high and `Pay(o)` large,
  the negative sum dominates even when my own hazard is decent.

## 4. A move's effects

Each candidate `a` changes the inputs of §2–3 as follows. Everything else is shared.

### 4.1 Deal-in now

`feed_{me→o}(1) = P_o(t(a))` from the belief (0 for a pass or a claim). A dead tile (no
unseen copy) has `P_o = 0` exactly (`belief.rs` sets a wait with no unseen copy to 0).

### 4.2 Feeding a pung or chow, by seat

A tile `o` can claim speeds `o` up by one state step this round: in round 1 add
`c_o(t) = P_claim_o(t) × (1 − P_o(t))` to `o`'s transition (`F → O` or `O → R`, split by
`p_F : p_O`), where `P_claim_o(t) = min(1, P2_o(t) + 1[o is my next seat] × max_chow
P1_o(u) × P1_o(v))`, the same as `ev-bot.md`'s `P_claim4_o`. If `o` shows three exposed melds
and nobody is liable for `o`, the claim also makes me liable: `L_sd = L_disc = L_other = 3`
for `o` from round 2 on, weighted by `c_o(t)` (`ev-bot.md`'s `Liab` term, now inside the
race). Source: `Danger::liability_tile`, `Danger::liability_chance` (`danger.rs`).

### 4.3 My live and improvement tiles

After `a`, `s`, `T`, my waits and acceptance come from `faan_shanten` on `h(a)`. Ready:
`L_me`, `W_me`, `H_{me,j}` from §2.2 with exact `unseen` counts and the counts head. One
away or far: `b_me` (or `a_me`) = acceptance / `U`; the hazard after reaching R uses the
expected live count of the target's waits (v0: the mean over the acceptance tiles of the
waits they would leave, computed by one `faan_shanten` per accepting kind, bounded in §6).
A drawn tile that raises `f_T` (a faan honour completing a pung, the flush's suit) raises
`Pay(me)`: v0 scores it through `faan_shanten` on the hand after the draw, averaged over the
accepting kinds.

### 4.4 Holding cost: the fold reserve and dangerous-later tiles

Each later round I discard one tile. Which one depends on my mode (§5): pushing, I throw
the tile `rank_discards` (`bots.rs`, `Metric::Faan`) ranks lowest; folding, the safest tile
left. The cost of throwing tile `u` in round `k` is `Σ_o p_R,o(k)(1 − fold_o(k)) ×
P_o(u | k) × Pay(o) × L_disc(o)`, where `P_o(u | k)` rises with `o`'s progress: v0 takes
`P_o(u | k) = wait_o(u) × p_R,o(k) / p_R,o(1)` (the wait pattern of today scaled to the
later ready chance; estimate; Phase 4 fits a per-bucket rise from the logs). The sum of these
costs over the rounds I expect to keep playing is the **holding cost** of my hand after `a`.

Two tiles with the same effect on my hand differ in holding cost. A tile that is useless to
my target and dangerous later (a middle tile of a suit an opponent is collecting, a
claimable tile, a lone faan honour) is cheaper to throw now, while `p_R,o` is low, than in
round 4. That is B28's "throw dangerous-later, useless-now tiles early": it is the
difference between `E(discard u now)` and `E(discard u in round k)` through this term, not a
rule.

The **fold reserve** is the number of tile copies in `h(a)` with `Σ_o P_o(u | k) × Pay(o) <
1 point` at the round I would start folding. When I fold with an empty reserve, the fold
plan's later discards use `q(push)`-level danger: folding then costs almost as much as
pushing, which is what makes a thin hand push (§9 shows one: after the pung I hold one
concealed tile, reserve 0).

### 4.5 Claims and passes

A claim is valued as `max_d E(Discard(d))` over the hand after the claim (`after_claim`,
`bots.rs`), with the turn order after the claim (my draws `n_me` shrink or grow). A pass
keeps the hand and the current order. Kongs as in `ev-bot.md` §2.1 (replacement draw adds
nothing).

**A legal win is one more candidate** (Task 21, `passwin=on`, the default; `passwin=off` takes
every legal win as before). Win now is what the win pays: `points(faan) × shares`, with the
faan of the actual win (`claim_score` / `self_drawn_score`, so a self-draw includes its +1)
and one share for a discard win (three when I am liable, §8.3) or three for a self-draw.
Pass is the race value of the hand kept: for a discard win the `Pass` candidate of the claim
window; for a self-draw the best discard of the 14 tiles. The win is taken unless the pass is
worth more than `win + tie_margin`. Nothing special is added for "far ahead": the race's own
terms do it. When the opponents are far, their loss terms are small and the pass is judged on
my chances; when one is likely ready, or the wall is short, the pass loses its later wins and
the win is taken. The engine enforces the lock on a passed discard win (`crates/hk` Task 12);
the race value of the kept hand still counts the passed tile among its waits, which only
matters when copies of it are live (none are in the test hand).

### 4.6 Honour tiles that give faan

A lone dragon or value wind in my hand raises `Pay(me)` only through `faan_shanten` on the
hands that pair or pung it (§4.3). It raises my cost through §4.4: honours are the classic
late deal-in tile (`danger_tables.rs` `WAIT` class 0 rises from 0.011 to 0.27 as unseen
copies fall). The bot keeps the honour while the first term beats the second, and throws it
early otherwise. No separate rule.

## 5. Modes: push, detour, fold

### 5.1 My modes, per threat

A **threat** is an opponent with `p_O + p_R` above `τ_threat = 0.2` (estimate) who is not
folded. For each threat `o` I choose a mode:

| Mode | Discards allowed against `o` | My hazard |
|---|---|---|
| push | any tile | full |
| detour | tiles with `P_o(u) ≤ 0.05` **or** tiles whose discard keeps `s` within one step of the best; prefer the best tile among those | reduced: `b_me`, `L_me` recomputed on the detour hand |
| fold | only tiles with `P_o(u) ≤ 0.02` (the reserve); when the reserve is empty, the safest tile | zero after the fold starts (`p_X,me = 1`); my chain stops |

My combined discard set is the intersection over threats of the allowed sets; if it is
empty, the tile with the smallest total `Σ_o P_o(u) × Pay(o) × L_disc(o)` is used (B28:
fold against B, push against A only if nothing I hold loses to A; the intersection does
that). The mode vector is chosen afresh every turn (§5.3); a fold against B lasts exactly as
long as B threatens. `p_X,me` for the later rounds follows the chosen mode: push or detour
keep my chain running; fold sets my hazard to 0 from the next round.

### 5.2 Opponents' modes as probabilities

Each opponent's mode is a distribution `π_o(m | public state)`, not a guess at one mode. It
enters §2 as `fold_o(k)` (removes `o`'s hazard and progress) and §2.3 as `q(m_j)` (how freely
`j` feeds the others). Independence between opponents is assumed.

v0 heuristic (estimate, replaced by Phase 4's `mode` head): `o` folds in proportion to what
a visible threat would cost `o`, scaled by `o`'s state:

```
cost_o = Σ_{i≠o} P̂(i wins) × Pay(i) × shares o would pay      (my own hand enters as i = me)
π_fold(o) = min(0.9, cost_o / 64) × { F: 1.0, O: 0.6, R: 0.2 }
π_detour(o) = 0.5 × (1 − π_fold)  if state O or R, else 0;   π_push = the rest
```

Here `P̂(i wins)` is a one-pass estimate (`tenpai_i × 0.6` for opponents, my own ready
hazard summed over the rounds for me). Two things fall out: under the 12-tile rule a seat
that pays nothing if I win has `cost = 0` from me and keeps pushing, and the liable seat
folds hard. That is the real table dynamic (§9). Table image (opponents reading *my*
folds) is left out (B28); the public state the Phase 4 head sees includes my discards, so
it learns part of it without being told.

### 5.3 The mode choice each turn: best response with a cost bound

For every candidate action `a` and every mode vector `M = (m_o)` over my threats, compute
`E(a, M)` with the opponents' modes averaged over `π_o` (hazards are linear in `fold_o`, so
the average is one pass, not an enumeration). Choose `argmax_{a, M} E(a, M)`.

Bound: threats are capped at the two with the highest `p_O + p_R`; a third threat is
folded into the hazards only (no mode for it). So at most `3² = 9` mode vectors. Candidates:
at most 14 discards plus claims, kongs and pass. Rounds: `n ≤ 8` after the first sixth of the
wall; `n = ceil(wall / 4)` capped at 8 (as `ev-bot.md`'s `horizon`). Per `(a, M)` the chain
is 4 players × 4 states × 8 rounds and a handful of multiplications per entry, about 600
floating-point operations. Total: 16 candidates × 9 vectors × 600 ≈ 90,000 operations, plus
one `faan_shanten` per candidate (0.3 µs each, `docs/solver.md` §1) and §4.3's
per-accepting-kind calls (≤ 34 per candidate, cached by hand). The belief runs once
(0.5 ms, `belief.rs` cache). Estimate: under 1 ms per decision on one thread, the AC5 bar is
5 ms. Task 6's timing test measures it.

## 6. What v0 takes from existing parts, and what Phase 4 trains

| Quantity | v0 (Phase 3) | Phase 4 |
|---|---|---|
| `tenpai_o`, `wait_o(t)`, `P_b(o)`, `cnt_o(t)`, `outcome_o` | belief heads (`models/belief.mjnn`) | retrained on the race field (`models/race-belief.mjnn`) |
| `p_O,o` (one away) | `min(1 − p_R, 1.5 p_R)` (estimate) | the `state` head: softmax over F / O / R / X |
| `fold_o`, `π_o(m)` | §5.2 heuristic | the `mode` head: softmax over push / detour / fold, per opponent |
| `a_o`, `b_o` | 0.15 per round (estimate) | tables fitted from the logged true-state transitions, per wall bucket and visible melds |
| `q(m)`, `φ_{j→i}` | 0.08 / 0.04 / 0.01; `φ` = `q`-level scaled by tile usefulness (§4.2 note) | measured from the logs by mode label and tile usefulness |
| `P_o(u \| k)` rise | scaled wait pattern (§4.4) | per-bucket rise fitted from the logs |
| My `s`, `T`, `f_T`, acceptance, waits | `faan_shanten`, `rank_discards` (exact) | unchanged |
| `Pay(o)` without a model | `HeuristicBelief` from the danger tables | unchanged fallback |
| Target value table (`target_value.rs`) | **not used** in the formula; `TargetValue::points` only as the cross-check in Task 7's report | unchanged |
| Danger tables (`danger_tables.rs`) | the `HeuristicBelief` fallback; `Danger::liability_tile` | unchanged |

Tile usefulness for `φ`: v0 sets `φ_{j→i} = 1.0 × (1 − π_fold(j) toward i)` for a tile that
`j`'s visible plan cannot use (another suit than `j`'s flush, an honour `j` discarded) and
`0.3` of that for a tile `j` could use (estimate).

## 7. Training labels from full-information self-play

The logger (`examples/selfplay_log.rs`, extended in Task 8) writes, for every decision of
every seat, the observer's public features (`features::encode`) plus, for each other
player, these labels read from the **true** hands:

| Label | Definition | Used for |
|---|---|---|
| `state` | the player's faan-aware shanten of the true hand: F (≥ 2), O (1), R (0 with a legal win available); X overrides when the player is folded (below) | the `state` head, the `a`, `b` tables |
| `mode` | the mode of the player's **last discard** (definition below); push for a player who has not discarded yet | the `mode` head, `q(m)`, `φ` |
| `waits` | the 34-bit set of tiles that complete the true hand with a legal win (R only) | the `waits` head |
| `value` | points per share of the best discard win now (`opp_faan_now`, as `belief.md`) and of a self-draw now | the `faan` head |
| `counts` | true copies of each kind in the hand | the `counts` head |
| `outcome` | who won the hand, how (self-draw / discard, by whom), in which round from this decision | hazard calibration per wall bucket (§8.2) |

**The mode label, exactly.** For a discard `t*` by player `j` from hand `H` (14 tiles after
the draw, melds included), with `s(t) = faan_shanten(H − t)` for every legal discard `t`,
`s_best = min_t s(t)`, and `s_before` the faan-aware shanten before the draw:

1. **fold** if `s(t*) ≥ s_best + 2`, or if `s_before = 0` and `s(t*) ≥ 1` (breaking a ready
   hand is a fold even by one step).
2. **detour** if not fold and `s(t*) = s_best + 1`; or if `s(t*) = s_best = 0` and the wait set
   after `t*` differs from the wait set after `rank_discards`' first choice and
   `Danger::expected_loss(t*)` (the public estimate at that moment, from the log's own
   features) is lower than that choice's.
3. **push** otherwise (`s(t*) = s_best`, including every tie by acceptance).

A player is **folded** (state X) from the first turn after a fold-labelled discard, and stays
folded until a push-labelled discard; a detour does not end a fold. Claims: a pung or chow is
labelled push; a pass that would have lowered `s` by one is labelled by the player's next
discard (no mode of its own). The labels need the true hand for `s(t)` and the waits; the
public danger estimate is the log's own number, so every label is a function of the log and
nothing else, and a second run reproduces it.

Why this definition and not the bot's internal mode: the labels must apply to every seat
in the league (fixed bots, old generations, later Brian's own hands), none of which has a
race-bot mode. Faan-aware shanten before and after is observable for any player.

## 8. The self-play loop

### 8.1 One generation

Generation `g` has a race bot `R_g` (the formula of §2–5 with estimate heads `θ_g`; `R_0` is
v0 with the existing belief and the heuristics of §6). Steps, all through the job queue
(B26/B27, `.claude\job-queue\jobq.ps1`; cpu lane for play, gpu lane for training; the
GPU lane is Dou Dizhu's until about 2026-11-12):

1. **Play** (cpu lane): 100,000 hands with full-information logging (§7). Seats drawn from a
   **league**: 60 % of hands four copies of `R_g`; 20 % `R_g` with three seats from
   `{R_{g−1}, R_{g−2}, …}` (uniform over the generations kept); 20 % `R_g` with three seats from
   the fixed field `{value, defensive, fast, efficiency, closed}`. The fixed share keeps the
   heads calibrated on styles the loop does not produce (the belief's hold-out gap,
   `belief.md` §4.3); the old generations stop the loop from beating only its latest self.
   Estimate: the race bot under 1 ms per decision and about 70 decisions per hand gives
   about 100,000 hands in under an hour on 8 threads (`selfplay_log` reached 370,000 hands
   per hour with the rule-based field, commit `dabe54e`).
2. **Label**: §7, inside the logger.
3. **Train** (gpu lane): `python -m mahjong.belief.train` extended with the `state` and
   `mode` heads, warm-started from `θ_g`, 5 epochs (the belief took 349 s for 2.1 M rows,
   `belief.md` §3; estimate 10 minutes here). Fit the `a`, `b`, `q`, `φ` tables from the same
   logs. Output `θ_{g+1}`, `models/race-belief-g<g+1>.mjnn`.
4. **Gate** (cpu lane, duplicate matches, 1,000 walls × 4 seatings = 4,000 hands each, seeds
   as Task 7's 7000–7019 so the walls match the teacher's matches):
   - `R_{g+1}` in one seat against three `R_g`: the generation's **gain**, paired by wall.
   - `R_{g+1}` against three `value`: the common yardstick (the teacher scored +1.24 ± 1.21
     on these walls, `study-results/phase4-build.md` §2).
   - `R_{g+1}` against the **teacher** (belief-PIMC-64, `anchor=value, rollout=fast`): one
     seat each plus two `value`, 4 seatings so each agent sits in every seat. Estimate about
     1.2 hours (the teacher takes 1.0 s per hand on 4 threads, `solver.md` §4).
5. **Report**: `docs/study-results/race-bot.md` gets one row per generation: points per hand
   and 95 % wall-bootstrap interval for each match, win / self-draw / deal-in / draw rates,
   mean faan, held-out calibration of the `state` and `mode` heads per wall bucket.

### 8.2 Hazard calibration

From each log the `outcome` label gives, per wall bucket, the observed share of ready
opponents who won within one round, two rounds, …; the race model's `h_o` summed the same
way must match within the bucket's 95 % interval. A bucket that misses by more than 20 %
relative is reported and the `q`, `a`, `b` tables of that bucket are refitted before the
next generation. This is the "never trust the formula" check inside the loop.

### 8.3 Stop rule

Noise: a 4,000-hand duplicate match has a 95 % interval of about ±1.2 points per hand
(`phase4-build.md`: ±1.21, ±1.22, ±1.18 at 4,000 hands). The loop stops when:

- **inside noise for 2–3 generations**: the gain of `R_{g+1}` over `R_g` (step 4, first
  match) has |gain| < 1.2 for two consecutive generations, **and** the score against `value`
  has not risen by more than 1.2 over the last three generations; or
- **a hard cap** of 8 generations (estimate of the budget; Brian can raise it); or
- **a regression**: a generation loses to its predecessor by more than 1.2 (paired interval
  below 0). Then that generation is dropped, training is repeated once from the same logs
  with a new seed, and if it regresses again the loop stops.

At least three generations run before the first stop check (AC6). The **best generation** is
the one with the highest score against `value`; ties within 1.2 go to the later one. Its
match against the teacher (step 4, third match) is the AC6 report. No bar is set: the report
and Brian decide what the review uses next.

## 9. Worked example: Brian's pung hand (decision 85)

Fixture `crates/search/tests/data/brian-2026-11-09.json`, history index 85, printed by
`target\release\examples\regrade.exe <fixture> 85 0`. Rules: minimum 3 faan, cap 10,
half-spicy, flowers on, 12-tile liability on. Only seat 0's view is used until §9.5.

### 9.1 The scene from seat 0

- **Me (seat 0, East)**: concealed `4p 5p 8p 8p`, exposed pungs `333p 111p 777p`, one bonus
  tile (the seat flower, 正花, +1 faan). Ready on `3p` or `6p` (the `45p` chow plus the `88p`
  pair): 清一色 7 + 正花 1 = **8 faan**, `points(8) = 64` per share on a discard; a self-draw
  adds 自摸 → 9 faan, 96 per share from each of three seats.
- **Seat 2 just discarded `8p`.** Legal: pung or pass. A pung gives me four exposed melds
  (`333p 111p 777p 888p`); by rules §8.3 seat 2 becomes **liable** for me: if I win in any
  way, seat 2 pays 3 shares and nobody else pays. The hand becomes 清一色 7 + 對對糊 3 +
  正花 1 = 11 → **capped at 10 faan**, 128 per share, **384 from seat 2** however I win.
- **Unseen from my view (U = 54)**: 30 in the wall, 10 + 10 + 4 concealed in seats 1, 2, 3.
- **Pung then discard 5p**: wait `4p`, 2 copies unseen (one in my hand, one discarded by
  seat 2). The other discard, `4p` waiting on `5p`, is dead: seat 3 shows `555p` and I hold the
  fourth. `5p` itself is dead for every opponent: `P_o(5p) = 0`, so the pung's discard carries
  no deal-in risk at all.
- **Pass**: waits `3p` (1 unseen: my `333p` holds three) and `6p` (1 unseen: three discarded).
  Both paths have **2 live copies**.
- **Draws left**: pass → seat 3 draws next, then me; 30 tiles give me **8 draws**. Pung → the
  turn passes to seat 1 after my discard; I get **7 draws**.
- **Opponents' visible melds**: seat 1 `222s` (1 meld, no faan shown; discards shed
  characters and dots); seat 2 `111m` (1 meld; discards shed dots and bamboo); seat 3 `555p
  WhWhWh 789p` (3 melds, a dragon pung and all suited melds in dots: a dots flush tag with
  faan; discards shed characters and bamboo).

### 9.2 Inputs and the estimates I take

| Input | Value | Source |
|---|---|---|
| `tenpai_o`: seats 1, 2, 3 | 0.17, 0.17, 0.44 | danger `THREAT` cells (1 meld, bucket 4, plain) = 0.172; (3 melds, bucket 4, flush) = 0.436, `danger_tables.rs`. v0 reads the belief head instead; the table is the stand-in here |
| `p_O,o` | 0.26, 0.26, 0.56 | §2.1's estimate `min(1 − p_R, 1.5 p_R)` |
| `a_o = b_o` | 0.15 per round | estimate (§2.1) |
| `Pay(o)`, `Pay_sd(o)`: seats 1, 2, 3 | 24 / 32, 16 / 24, 48 / 64 | estimate: seat 3 from `ev-bot.md` §8's fallback (3 + 1 dragon + 3 flush = 7 faan); seats 1 and 2 a guess at the belief faan head (a half flush with honours for seat 1, a plain hand for seat 2) |
| `r_sd` of a ready opponent | 2 / 54 = 0.037 per round | estimate: about two live copies |
| `q(push)`, `q(fold)` | 0.08, 0.01 | estimate (§2.3); my own later discards use `q(push)` (I push on both paths: a ready 8–10 faan hand never folds here) |
| Copies of my live tiles in each hand, `H_{me,j}` | pung path (`4p`): 0.05, 0.15, 0.25; pass path (`3p` + `6p`): 0.05, 0.10, 0.40 | estimate of the belief counts head: seats 1 and 2 have shed dots for 60 discards; seat 3 collects dots and holds 4 concealed tiles |
| Live copies in the wall `W_me` | pung 1.55, pass 1.45 | `2 − Σ_j H_{me,j}` |
| My self-draw hazard `d_me = W_me / 30` | pung **0.052**, pass **0.048** | §2.3 |
| Feed rates `φ_{j→me}`, pung path | seat 1 0.8, seat 2 0.05, seat 3 0.5 | §5.2: seats 1 and 3 pay nothing if I win (seat 2 is liable), so they push; a `4p` is useless to seat 1 and possibly useful to seat 3; seat 2 would pay 384 and folds. Estimates |
| Feed rates, pass path | 0.3, 0.3, 0.2 | feeding my visible 3-meld dots flush costs the feeder 64; a dots tile is useful to seat 3. Estimates |
| Opponents' fold probabilities (F, O, R), pung path | seat 2: 0.9, 0.6, 0.2; seats 1 and 3: 0 | §5.2 with `cost_2 ≈ 0.6 × 384`, `cost_1 = cost_3 = 0` from me |
| Opponents' fold probabilities, pass path | seats 1, 2: 0.4, 0.25, 0.1; seat 3: 0.3, 0.2, 0.05 | §5.2 with `cost ≈ 26` from me (a 64-point feed at about 0.1 plus a 96-point self-draw at about 0.2) |
| Liability | pung path: seat 2 liable for me; nobody liable for any opponent on either path | rules §8.3 |

### 9.3 The race, round by round

`h_me(k) = g_me(k) + (1 − g_me(k)) × d_me`, with the held copies fed in the first round each
seat acts (pung path: seats 1, 2, 3 all act before my first draw; pass path: only seat 3).
Opponents' hazards: `p_R,o(k) × (1 − fold_R,o) × (0.037 + Σ_{j≠o} q(m_j) + 0.08)`, states
advanced by `a`, `b` each round for the unfolded share.

**Pung, then 5p** (7 rounds; my win pays 384 any way):

| k | S(k) | h_me | h_1 | h_2 | h_3 | Σ h |
|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.281 | 0.039 | 0.038 | 0.100 | 0.458 |
| 2 | 0.542 | 0.118 | 0.048 | 0.041 | 0.120 | 0.326 |
| 3 | 0.365 | 0.118 | 0.058 | 0.044 | 0.136 | 0.357 |
| 4 | 0.235 | 0.118 | 0.070 | 0.048 | 0.151 | 0.386 |
| 5 | 0.144 | 0.118 | 0.082 | 0.051 | 0.163 | 0.413 |
| 6 | 0.085 | 0.118 | 0.094 | 0.054 | 0.173 | 0.439 |
| 7 | 0.048 | 0.118 | 0.106 | 0.057 | 0.182 | 0.463 |

Round 1's `g_me = 0.8 × (0.05 + 0.052) + 0.05 × (0.15 + 0.052) + 0.5 × (0.25 + 0.052) =
0.243`; `h_me = 0.243 + 0.757 × 0.052 = 0.281`. From round 2: `g_me = (0.8 + 0.05 + 0.5) ×
0.052 = 0.070`, `h_me = 0.118`.

Sums: **P(I win) = 0.449**; P(seat 1) 0.127, P(seat 2) 0.102, P(seat 3) 0.297; draw 0.026.
Gain `Σ_k S(k) h_me(k) × 384 = 172.3`. Losses: opponents' self-draws `Σ_k S(k) Σ_o p_R,o(1 −
fold) × 0.037 × Pay_sd(o) = 4.1`; wins on my discards (round 1: `P_o(5p) = 0`; later `0.08`
per round) `6.5`; wins on others' discards cost me nothing. **E(pung) = 172.3 − 4.1 − 6.5 =
+161.7**.

**Pass** (8 rounds; a discard win pays 64, a self-draw 288):

| k | S(k) | h_me | h_1 | h_2 | h_3 | Σ h |
|---|---|---|---|---|---|---|
| 1 | 1.000 | 0.134 | 0.038 | 0.038 | 0.098 | 0.307 |
| 2 | 0.693 | 0.128 | 0.044 | 0.044 | 0.113 | 0.330 |
| 3 | 0.465 | 0.085 | 0.052 | 0.052 | 0.127 | 0.316 |
| 4 | 0.318 | 0.085 | 0.059 | 0.059 | 0.140 | 0.344 |
| 5 | 0.209 | 0.085 | 0.067 | 0.067 | 0.151 | 0.371 |
| 6 | 0.131 | 0.085 | 0.075 | 0.075 | 0.162 | 0.398 |
| 7 | 0.079 | 0.085 | 0.084 | 0.084 | 0.171 | 0.423 |
| 8 | 0.046 | 0.085 | 0.092 | 0.092 | 0.179 | 0.448 |

Round 1: only seat 3 acts before my draw: `g_me = 0.2 × (0.40 + 0.048) = 0.090`, `h_me =
0.090 + 0.910 × 0.048 = 0.134`. Round 2: seats 1 and 2 act for the first time and may feed
a held copy: `g_me = 0.3 × (0.05 + 0.048) + 0.3 × (0.10 + 0.048) + 0.2 × 0.048 = 0.084`,
`h_me = 0.128`. From round 3: `g_me = (0.3 + 0.3 + 0.2) × 0.048 = 0.039`, `h_me = 0.085`.

Sums: **P(I win) = 0.329**; P(seat 1) 0.146, P(seat 2) 0.146, P(seat 3) 0.354; draw 0.025.
Gain `Σ_k S(k) × [g_me × 64 + (1 − g_me) d_me × 288] = 50.8`. Losses: self-draws 4.8, my
feeds 7.6. **E(pass) = 50.8 − 4.8 − 7.6 = +38.4**.

**Pung minus pass: +123 points.** The race bot pungs.

### 9.4 What drives the result, and a sensitivity check

Three things, all from seat 0's view: (1) the pung's payout is 384 whichever way I win,
against 64 on a discard for the pass (a self-draw's 288 arrives only with `d_me = 0.048`);
(2) both paths have two live copies, so the pung loses no live tiles, only one draw; (3) the
12-tile rule turns seats 1 and 3 into free feeders and makes seat 2 fold, which both help
me; the pass path keeps every seat careful with dots.

Dropping every feed (nobody ever discards a live tile to me; `H = 0`, `φ = 0`, so `d_me =
2 / 30 = 0.067` on both paths): P(I win) 0.213 vs 0.221, **E(pung) = +67.3, E(pass) = +49.4,
difference +17.9**. That is the order of the `ev`-rollout regrade in the plan header (pung
+20 ± 4). The pung is ahead under both feed assumptions; the size of the lead depends on
the estimates in §9.2, and the matched numbers of Phase 3–4 will replace them.

The tables are §2–3's model run with §9.2's inputs (a 60-line script, not kept). Task 6's
test `Brian's pung hand values pung ≥ pass` reproduces the comparison in Rust with v0's
real inputs.

### 9.5 Comment with the real hidden hands (after the fact only)

The fixture shows: both `4p` are in the wall (no opponent holds one), the last `3p` is in
seat 3's concealed hand (`3p W N R`, a far hand, which keeps it only until it folds or
reshapes), and the last `6p` is in the wall. Seat 3 is **far**, not ready: the danger table's
0.44 for a 3-meld flush tag over-states this particular seat (the plan header's observation
that the review's opponents "keep feeding pins" was about the `fast` rollout bot, not about
the race model). With the true positions the pung's live copies are both drawable and the
pass's are one drawable and one stuck, which widens the pung's lead. None of this is
available to the bot; it is why the race bot's inputs are learned from logs (§7) rather than
read from one table.

## 10. Open questions (defaults apply; the builder does not wait)

1. **`τ_threat = 0.2`** for which opponents get a mode of their own. Default 0.2; Task 6 may
   report the share of decisions with 0, 1, 2 threats.
2. **The detour tile set** (`P_o(u) ≤ 0.05` or within one shanten step). Default as §5.1;
   the mode label's detour rule (§7) mirrors it so the trained `mode` head and the bot's own
   modes mean the same thing.
3. **Horizon cap 8 rounds.** The same as `ev-bot.md`; a longer cap costs linearly.
4. **League shares 60 / 20 / 20** and **100,000 hands per generation.** Estimates; the first
   generation's calibration table (§8.2) says whether more hands are needed.

## 11. Deviations in v0 (Task 6, `crates/search/src/race.rs`)

**Deviation** marks where the code departs from §2–§6; each is a v0 choice that Phase 4 or a
match may revisit. The rest is built as written.

1. **Deviation: the deal-in now is its own step before round 1.** §3.3 puts `P_o(t(a))` inside
   round 1, where it competes with the round's other hazards; the code applies it first
   (`S` after it is `1 − Σ_o P_o(t(a))`) because my discard comes before anyone else acts.
   It is scaled by `1 − fold_R,o`, as every other hazard of `o` (a folded seat does not win).
2. **Deviation: §4.4's `P_o(u | k)` is read as `wait_o(u)`, conditional on `o` being ready.**
   The outer factor `p_R,o(k)(1 − fold_o)` already supplies readiness; multiplying by
   `p_R,o(k) / p_R,o(1)` as well would count it twice.
3. **Deviation: `W_o`** (an opponent's live copies in the wall) is the unseen copies of its
   waits less the belief's expected copies in every hidden hand, `o`'s own included.
4. **Deviation: the push plan.** My later discards are one copy of each kind whose discard now
   reaches the same faan-aware shanten or better (the hand's useless tiles), the most
   dangerous first; then the per-discard rate `q(m)` of my mode toward each opponent. After a
   pass or a kong there is no 14-tile hand to rank, so only the rates apply.
5. **Deviation: the fold plan** throws the allowed tiles (§5.1's intersection) with the
   useless ones first, in the push order, then the rest safest first against the seats I fold
   against. A seat I push against is ignored, as §5.1 says; this is what makes "fold against B,
   push against A" differ from the reverse. When two mode vectors give equal equity, the less
   defensive one wins (fewer folds, then fewer detours).
6. **Deviation: detour** keeps my hazard at 0.5 of push (estimate) instead of recomputing
   `b_me` and `L_me` on a detour hand; its plan is the push plan filtered by `P_o(u) ≤ 0.05`
   when a fold is also chosen.
7. **Deviation: a far hand** (faan-aware shanten ≥ 2) moves `F → O` at `acceptance / U / (s − 1)`
   per draw and `O → R` at `acceptance / U`, and has 4 live copies once ready (estimate). A
   one-away hand's live copies once ready are computed as §4.3 says (per accepting kind, the
   best ready discard's live winning tiles), with the unseen counts before the draw.
8. **Deviation: §5.2's cost from me** is `P̂_sd(me) × Pay_sd(me) + P̂_fed by o(me) × Pay(me)` for
   an unliable `o`, three times all of my wins for the seat liable for me, 0 for the others.
   `P̂(me)` is my hazard over the rounds with nobody else racing and `φ` before folds. A seat
   counts as folded (no mode of its own, §5.1) when `π_fold ≥ 0.5`.
9. **Deviation: `φ`'s usefulness** for a hand that is not ready uses the accepting kinds in
   place of the waits. An honour is useless to `j` only if `j` has discarded it (§6).
10. **Additions not in the doc**: ties within `tie_margin` (0.5 point) go to `rank_discards`,
    and a claim or kong must beat the pass or the best discard by more than `tie_margin`, as in
    the `ev` bot; rounds past the horizon are dropped without `ev-bot.md`'s `κ` (so `p_draw` is
    "nobody wins within the horizon"); a round's hazards are scaled to sum to at most 1;
    `RaceBot::fold_override` replaces §5.2's fold heuristic per seat (the hook for Phase 4's
    `mode` head, used by the two-threat test).
11. **Spec**: `race` takes `horizon=`, `tie_margin=`, `threat=` (`τ_threat`) and `belief=`; the
    plain `race` uses `HeuristicBelief`, so §6's belief heads need `race:belief=models/belief.mjnn`.

## 12. Deviations in Phase 4 (Tasks 8–9: labels, heads, the loop)

Code: the labels in `crates/search/src/race.rs` (`true_state`, `discard_mode`, `LabelTracker`),
the logger `crates/search/examples/race_log.rs`, the heads in `python/mahjong/belief/model.py`
(`BeliefConfig(race=True)`, `train.py --race --init`), the tables and checks in
`python/mahjong/belief/race_fit.py`, the loop `experiments/race_loop.py` (+ `.ps1` queue job).
**Deviation** marks where the code departs from §6–§8.

1. **Deviation: the race log is the belief log plus race fields.** Each row is
   `mahjong_search::log`'s row (so the five belief heads train unchanged) followed by
   `opp_state`, `opp_mode`, `opp_shanten`, `opp_points_sd` (the self-draw value of §7's
   `value`), the observer's own `my_mode` / `my_feeds` / `ready_opps` (for `q`), `league`,
   `wall_at_end` (rounds to the outcome) and, for race seats, `race_rounds` / `race_pwin`
   (§8.2). Layout version 101.
2. **Deviation: the `mode` head enters only as P(fold)**, through the fold hook, the same in
   every state (§5.2's per-state split and the detour share stay heuristic). A fixed
   `fold_override` still wins. The `state` head gives only `p_O` (capped at `1 − tenpai`);
   `tenpai` stays the tenpai head's, and the head's P(folded) is not used (the fold comes from
   the mode head).
3. **Deviation: fitted tables.** `a`, `b` are per wall bucket only (not by visible melds), as
   spec options `a=…,b=…` (six values each); a round is the same seat's first later
   observation with at least 4 fewer wall tiles; folded seats are left out; a bucket with fewer
   than 200 cases keeps v0's 0.15. `q(m)` is the feeds per ready opponent over the discards
   labelled `m` (spec `q=push/detour/fold`). `φ`'s usefulness factor and §4.4's `P_o(u | k)`
   rise are **not** fitted (v0's values stay).
4. **Deviation: §8.2's check** compares the race bot's own P(opponent wins within its horizon)
   for the chosen action with "won before the wall fell by `4 × rounds` tiles", per bucket,
   and flags a miss over 20 % relative. The tables are refitted every generation anyway.
5. **Deviation: the gates' intervals** combine the 20 chunks' wall-paired bootstrap intervals
   (normal approximation); the teacher match also reports race minus teacher, paired by wall.
6. **Deviation: generation 0** (v0 on `models/belief.mjnn`) is measured only against three
   `value` (the stop rule's first reference); the best generation is chosen among the learned
   ones. The score "has not risen by more than 1.2 over the last three generations" is
   `v_g − v_{g−3}`. A regression retry trains with seed 1. Each generation also plays a
   5,000-hand hold-out log (seed 50,000 + g) for the hold-out loss and the calibration tables.
7. **Observation on §7's fold label** (not a deviation; for Brian): swapping one tile changes
   faan-aware shanten by at most one step, so `s(t*) ≥ s_best + 2` fires only when the discard
   leaves no target that meets the minimum faan. Fold labels come almost only from breaking a
   ready hand. In a 200-hand smoke log the race bot's own discards were push 7,525, detour
   8,367, fold 740 (`fast` bots in `race_labels.rs`: push 4,315, detour 21, fold 2).
8. **Deviation: the mode label redefined** (replaces §7's rules 1–3; Brian's meaning:
   push plays the best tile for winning, detour keeps a chance to win but throws a safer
   tile at some cost in speed, fold gives up winning for safe tiles). Code:
   `race::discard_label` and the `LABEL_*` constants beside it. For a discard `t*`:
   - **best for the hand**: `rank_discards`' best faan-aware distance, acceptance short of
     the best by at most `max(2 tiles, 15 %)`, target faan at most 1 below the best;
   - **push**: `t*` is best for the hand, whatever its danger;
   - **clearly safer**: `Danger::expected_loss(t*)` (the public estimate, as before) is at
     most half the lowest expected loss among the best-for-hand discards, and lower by at
     least 0.16 points;
   - **fold**: clearly safer, and the distance rises or the acceptance falls below half
     the best's; **detour**: clearly safer, same distance, the rest;
   - **mistake**: neither best for the hand nor clearly safer. The log writes it as push
     (`my_mode` 0, no defensive intent) and flags it in `my_mistake` (layout version
     102); `race_fit`'s `labels.by_agent` counts it apart, per agent, calm (no opponent
     ready) and under threat.

   The folded state still runs from a fold label to the seat's next push label or claim;
   a detour or a mistake keeps it. The compare is against the **safest** best-for-hand
   discard, so a defensive label needs a real trade: throwing a hand-worse tile when an
   equally safe best-for-hand tile was there is a mistake.

   Why these numbers (my inference, not fitted): one tile kind has at most 4 live
   copies, so a gap of 2 tiles is under one kind and noise among near-equal shapes; 15 %
   scales it for wide hands (20 tiles of acceptance allow 3). One faan is the gap between
   `faan_shanten`'s near-equal targets. Below half the acceptance, the next step takes
   more than twice as long: speed is mostly given up, which is a fold. "Clearly safer"
   halves the deal-in risk and, at `Danger::estimate`'s 8 points per deal-in, cuts it by
   at least 2 percentage points; early in a hand every tile is under 2 % there, so no
   discard there counts as defensive.

   Smoke log (200 hands, seed 1, `--main race:belief=models/belief.mjnn --old race`, as
   item 7): the race seats' own discards went from push 6,152 / detour 8,154 / fold 715
   to push 6,076 / detour 17 / fold 38 / mistake 8,890. **Observation for Brian:** about
   99 % of those mistakes come while no opponent is ready; in a 20-hand sample almost all
   were in far hands (faan-aware shanten 3–7), one step behind the best. The fixed bots
   make few (`value`: push 234, detour 1, fold 0, mistake 50). The race bot's far-hand model
   (§11 item 7) barely tells distances apart, so tiny danger gaps decide its discard.
   The labels now show this; the bot itself is unchanged.

## 13. Passed tiles and risk-taking (Phase 6, Tasks 14–15)

Code: `crates/search/src/passed.rs` (the record and the risk score), `race.rs` (`RiskRead`,
`Table::new`), `danger.rs` (`Danger::apply_passed`), the defending `fast` bot in `bots.rs`.
The numbers are fitted by `cargo run --release -p mahjong-search --example fit_passed --
--hands 8000 --seed 1` (full-information self-play, a field drawn per seat from `race`,
`value`, `defensive`, `fast`, `closed`, `efficiency`; 585,620 decisions; about 200 s on one
thread). Rerun it to refit; `--belief models/belief.mjnn` fits against the model's estimates.
Brian's words and answers are the Phase 6 header of the plan.

### 13.1 The passed-tile record (Task 14, AC10)

When another seat discards `t` and seat `P` does not win on it, `P` **let `t` pass**: from
our view `P` was either not waiting on `t` or is now locked on it (rules §4.4). `Passed::of`
scans the public events and keeps, per seat and kind: passed; whether `P` has had a turn
since (a discard, claim or kong); the number `N` of `P`'s 入章 since (a tile from the hand
went out; a 摸打, the drawn tile straight out, leaves the hand unchanged; a claim or kong
counts as one change). `Passed::factor(P, t)` multiplies the deal-in estimate `P_P(t)`:

| Record | Factor | Measured (truth / estimate, relative to tiles that did not pass) |
|---|---|---|
| did not pass | 1 | 1.000 (29.1 M cases; the raw ratio is 0.957) |
| passed, no turn since (the lock) | **0** | 0 legal wins in 1.50 M cases (the rule) |
| passed, turned, N = 0 | 0 | 0 waits in 3.80 M cases (the hand is as it was) |
| N = 1 | **0.546** | 5.34 M cases |
| N = 2 | **0.618** | 3.72 M |
| N = 3 | **0.763** | 2.63 M |
| N ≥ 4 | **0.973** | 5.02 M |

`PASSED_FACTOR` in `passed.rs`. Without any 換章 reset the fade is slower (0.595, 0.667,
0.828, 1.100), because the reset takes out exactly the cases where the hand did change.
Against `models/belief.mjnn`'s estimates (`--belief`) the same cases give 0.722, 0.804, 1.0,
1.0 (`PASSED_FACTOR_MODEL`, used by `race:belief=`): the model already rates passed tiles
lower, so less of the fade is left for the factor.

**換章.** Brian: a pushing player throws a tile that was clearly useful to them, so the hand
took a new shape, and every tile they let pass is no longer safe. We see no hand, so the
fitter tried public readings of "clearly useful" and checked each against the truth (did the
seat's waits or faan-aware shanten change against its hand after its previous discard):

| Test (on a discard from the hand) | Fires | Waits or distance changed | Passed tiles after it (truth / estimate) |
|---|---|---|---|
| any 入章 | 265,942 | 39.5 % | 0.873 |
| middle tile 3–7 | 109,404 | 39.0 % | 0.868 |
| inner tile 2–8 | 155,648 | 38.9 % | 0.861 |
| fresh middle (kind and neighbours never discarded) | 64,564 | 41.1 % | 0.926 |
| terminal or honour | 110,294 | 40.3 % | 0.998 |
| **a tile of the seat's visible flush suit** | 14,546 | **55.0 %** | **1.248** |
| (摸打, for reference) | 283,327 | 0 % | — |

A middle tile from the hand says nothing beyond "a 入章": every hand reshapes a little on
each one. The one reading that marks a real change is a tile of the suit the seat's
exposed melds show it collecting: after it the waits changed in 55 % of cases and the
tiles it had let pass were back at 1.25 of the unconditioned deal-in rate, so a **full reset
is right there** (even conservative). `passed::reshaped` is that test; a seat without a
visible flush never resets, its safety only fades by `N`. The "pushing" part of Brian's
definition is not checked separately: the fitter found no public reading of it, and the
reset bucket is above 1 without it.

**Where it enters.** The race bot: `Table::new` scales `P_o(t)` (deal-in now, the fold and
detour tile sets, the plan's danger order) by the factor; the later-round `wait_o(u)` is left
alone (by then the seat has drawn again). The defending `fast` bot: `Danger::apply_passed`
rescales each opponent's rate before the fold's safest-tile choice and the "clearly safer"
tie-break; `Danger::estimate` now keeps the per-opponent rates for that. The `defensive` bot
and the `ev` bot are unchanged. Spec: `race:passed=off` switches it off (Task 16's match).

### 13.2 The risk-taking read (Task 15, AC11)

`passed::risk_scores` gives each seat a score: for every tile it threw, the public danger of
that tile as the tables rated it at the time (the thrower's view of the other three seats:
threat by exposed melds and wall progress, wait rate by tile class, copies then unseen on
the table, whether the target had discarded it) **less the mean danger of every kind still
unseen then**, summed with 0.75 per later discard. Positive: the seat throws more dangerous
tiles than the table offers; negative: it picks safe ones. The plain danger was tried first
and only measured how threatening the table was: its high bins held both pushers and
folders late in the hand.

`RiskRead` (`race.rs`) turns the score into multipliers on the heuristic fold probability
(§5.2), `tenpai_o`, `p_O,o` and `Pay(o)`, by five bins, each relative to what the wall
bucket's own truth / estimate ratio predicts (folds and readiness both rise late, as the
score's spread does):

| Score | Share of observations | Fold | Ready | One away | Pay |
|---|---|---|---|---|---|
| below −0.02 (picks safe tiles) | 18 % | **1.685** | **0.776** | 0.988 | 0.998 |
| −0.02 to 0 | 36 % | 0.467 | 0.867 | 0.982 | 1.015 |
| 0 to 0.02 (nothing thrown, or no danger around) | 35 % | 0.435 | 1.112 | 1.005 | 0.960 |
| 0.02 to 0.05 | 5 % | 0.536 | 1.311 | 1.041 | 0.995 |
| 0.05 and up (throws dangerous tiles) | 6 % | **0.736** | **1.466** | 1.034 | 1.030 |

Raw rates behind it (all wall buckets together): seats below −0.02 were folded 1.23 % of the
time and ready 8.3 % against an estimated 13.0 %; seats at 0.05 and up were folded 0.55 %
and ready 18.1 % against 15.0 %. So Brian's reading holds for closeness (a risk-taker is
1.5× as likely to be ready as the table says, a safe-picker 0.78×) and for the push / fold
split between the two ends (2.3× apart). **Observations for Brian:** (1) the hand value
hardly moves (within 4 %): how hard a seat pushes did not narrow what it is aiming to win,
in this field; (2) the fold multipliers are not monotone: the seats near zero (nothing
dangerous thrown, usually because no tile was dangerous) fold least within their wall
bucket, less than the risk-takers. The table is used as fitted. The fold truth is rare
(0.6 % of observations) and comes from the §12 label.

**Where it enters.** `Table::new` multiplies `tenpai_o`, `p_O,o`, `Pay(o)`, `Pay_sd(o)`; the
heuristic fold of §5.2 gets the fold multiplier (a `mode` head's P(fold), a `state` head's
one-away estimate and a fixed `fold_override` are left alone). `RISK_HEURISTIC` applies to the plain `race`; a `race:belief=`
bot uses `RISK_MODEL`, fitted with `--belief` against that model's estimates (ready 0.787 …
1.313 across the bins, the same fold row, value and one-away within 7 %). Spec:
`race:risk=off`. `RaceTerms::risk` carries the scores.

### 13.3 Deviations and limits

1. The 換章 test reads the seat's melds, not its hand: a concealed flush never resets.
2. The risk score's "danger at the time" counts unseen copies from the table only (the
   thrower's hand is hidden) and reads wall progress back from the discards and claims that
   followed; positions built from a `PositionSpec` have no draw events, so this works for
   them too.
3. The deal-in factor applies to the tile thrown now; the later-round feed rates ignore the
   record.
4. The `defensive` bot and the `ev` bot do not use the record (the `fast` bot runs the
   rollouts and the league; the fixed field's play is kept as the belief model saw it).
5. Fitted on the heuristic belief and the rule-based field plus `race` v0; Task 18's
   retrained models should see the record and the score as features, and Task 16 measures
   the two reads in a match before anything is trusted.

## 14. Deviations in Task 20: calm margin, fitted tables, hazard multiplier

Code: `crates/search/src/race.rs` (`RaceConfig::calm_margin`, `calm_power`, `hz`, `FIT_*`,
`offence_discard`); evidence in `docs/study-results/race-bot.md` (Task 20).

1. **Deviation: discards use a wider tie margin when no opponent is likely close.** The tie
   rule of §11 item 10 keeps `tie_margin` (0.5 points); for discards the margin becomes
   `tie_margin + calm_margin × w^calm_power`, `w = Π_o (1 − P(ready_o) − P(one away_o))`
   (defaults 100 and 1). Candidates within the margin of the best go to `rank_discards`.
   Defence is still weighted by each opponent's chance of being close, with no switch
   (Brian, D107). Reason: the far-hand model of §11 item 7 (acceptance-driven) made the bot
   disagree with pure offence in 75 % of calm early positions. Claims and kongs keep
   `tie_margin`.
2. **Deviation: `hz`**, a multiplier per wall bucket on each opponent's per-round win
   hazard (all three parts, so the loss terms stay consistent), fitted by comparing
   predicted and observed win chances on held-out hands against `value`.
3. **The defaults are now the fitted tables** (`FIT_A`, `FIT_B`, `FIT_Q`, `FIT_HZ`, from the
   `value` field with `belief.mjnn`); v0's constants stay as `OPP_STEP`, `Q`.
   Spec: `calm=`, `calm_power=`, `hz=x0/…/x5`.
4. **Limit:** the tables describe `value`-like opponents. Against other styles the check was
   not run.
5. `RaceConfig::v0()` returns v0's constants; the hand-built scenarios of
   `crates/search/tests/race_bot.rs` use it, as they were designed against them.
