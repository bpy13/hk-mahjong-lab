# The EV bot: Brian's explicit expected-value design (B15), written down

Plan task 8. This file is the specification that Task 10 (`crates/search/src/ev.rs`)
implements as written. Decision B15 (bot-stream `docs/decisions.md`, verbatim):

> Each move is valued as offence EV (P(win) × payout from opponents) minus defence EV loss
> (P(deal-in or other win) × payout to that opponent), with opponents' win chances and hand
> values estimated from their discards and melds. Defence minimises EV loss, so it may
> deliberately deal into a small-faan hand to stop a large-faan self-draw.

Every term below names the code or table it comes from. Payment numbers follow
`docs/rules/hk.md` §8 (half-spicy, 1 share on a discard win, 3 shares on a self-draw,
3 shares from the liable player under the 12-tile rule). Anything not settled here is in
§9 "Open questions" with the default the builder uses.

## 1. Notation and sources

| Symbol | Meaning | Source |
|---|---|---|
| `me` | the acting seat | `state.current_player()` |
| `o` | one of the three opponents | seats ≠ `me` |
| `a` | a candidate legal action | `state.legal_actions_vec()` |
| `h(a)` | my concealed hand after `a` (13 tiles after a discard) | `SeatView` (`crates/search/src/view.rs`) |
| `t(a)` | the tile `a` discards now (`None` for pass or a kong) | `Action::Discard(t)` |
| `points(f)` | points per share for `f` faan: 3→8, 4→16, 5→24, 6→32, 7→48, 8→64, 9→96, 10→128 | `RuleSet::points` (`crates/hk/src/rules.rs`), rules §8.1 |
| `wall` | tiles left in the wall | `state.wall_remaining()` |
| `n` | draws left per seat, `n = max(1, ceil(wall / 4))` | derived (approximation A6) |
| `H` | horizon in my future turns, `H = min(n, horizon)`, `horizon = 8` | hyper-parameter |
| `κ` | horizon factor `κ = min(1, H / n)` (1 when `wall ≤ 32`) | derived |
| `liable_for(s)` | the seat liable for seat `s`, if any | `State::liable_for` (rules §8.3) |
| `exposed_o` | exposed melds shown by `o` (0–4) | `MeldList::exposed_count` |
| `unseen` | tile counts I have not seen (other hands + wall) | `State::unseen(me)` |

**From the target value table (Task 9, `crates/search/src/target_value.rs`)**:
`lookup(TargetKey::of_view(&view_after_a, wall))` returns `TargetValue { p_win, p_self_draw,
points }`; the key is `TargetKind::of(T)`, the faan-shanten bucket (0, 1, 2, 3, 4 = 4 or
more), `tiles_left_bucket(wall)` and `melds.len().min(4)`. `T` and `s` come from
`faan_shanten(h(a))` (`crates/analysis/src/target.rs`):

| Symbol | Meaning |
|---|---|
| `T` | the best target `FaanShanten::target`; `f_T = Target::faan` (guaranteed faan, capped) |
| `s` | `FaanShanten::shanten` (0 = waiting). If `s == UNREACHABLE`, `P_win = 0` |
| `P_win` | `TargetValue::p_win`: P(legal win before the hand ends \| key) |
| `P_sd` | `TargetValue::p_self_draw`: P(the win is self-drawn \| win) |
| `pts_T` | `TargetValue::points`: E[points per share \| win] |

**From the belief model (Task 5, `crates/search/src/belief.rs`, trait `Belief`)**, per
opponent `o`, from public information only:

| Symbol | Meaning | Belief output |
|---|---|---|
| `tenpai_o` | P(`o` is waiting with a legal win now) | tenpai logit |
| `wait_o(t)` | P(`t` is a winning tile for `o` \| tenpai), per tile, independent sigmoids | 34 wait logits |
| `P_b(o)` | P(`o`'s discard win pays faan bucket `b`), `b = 3..10` | 8 faan-bucket logits, trained on the log's `opp_faan_now` (a discard win now) |
| `sd_o` | P(`o` self-draws a win before the hand ends) | the self-draw output |
| `cnt_o(t)` | distribution of copies of `t` in `o`'s hand (0–4) | 34 × 5 count logits |

Derived from the belief:

- `Pay(o) = Σ_b P_b(o) × points(b)` — expected points per share if `o` wins on a discard.
- `Pay_sd(o) = Σ_b P_b(o) × points(min(b + 1, 10))` — a self-draw adds 自摸 (1 faan, rules
  §7.1 #1).
- `x_o = r_disc × sd_o` — P(`o` wins on *someone's* discard before the hand ends). The
  belief does not output this; `r_disc = 2.0` (§9 Q1).
- `win_o = sd_o + x_o`, clipped at 0.95.
- `P_o(t) = min(tenpai_o × wait_o(t), P_clip)`, `P_clip = 0.95` — P(`o` wins on `t` if I
  discard it now).
- `P2_o(t) = P(cnt_o(t) ≥ 2)`, `P1_o(t) = P(cnt_o(t) ≥ 1)`.

**Share multipliers** (rules §8.2–§8.3; engine `State::finish` in `crates/hk/src/state.rs`):

| Case | My shares paid |
|---|---|
| `o` wins on my discard, nobody liable for `o` | `L_disc(o) = 1` |
| `o` self-draws, nobody liable for `o` | `L_sd(o) = 1` |
| `o` wins on another opponent's discard, nobody liable for `o` | `L_other(o) = 0` |
| `liable_for(o) == me` (any way `o` wins) | `L_disc = L_sd = L_other = 3` |
| `liable_for(o)` is another seat (any way `o` wins) | `L_disc = L_sd = L_other = 0` (the liable player pays all three shares, nobody else pays) |

My own shares when I win: `Shares(me) = 3` if `liable_for(me)` is some seat; otherwise
`3` for a self-draw and `1` for a discard win.

## 2. The formula

```
EV(a) = Offence(a) − Defence(a)                       (points, for seat me)

cont(a) = Π_o (1 − P_o(t(a)))                         (nobody wins on my discard now; 1 if no tile)

Offence(a) = cont(a) × κ × P_win × Pay(me) × Shares(me)
  Pay(me)    = max(points(f_T), pts_T)                 (the guaranteed faan is a floor)
  Shares(me) = 3                                   if liable_for(me) is Some
             = 3 × P_sd + 1 × (1 − P_sd) × φ        otherwise
  φ = feed ratio, 1.0 unless s == 0 (see §3 A3)

Defence(a) = DealInNow(a) + cont(a) × [ SelfDraw + OtherFeeds + Later(a) + Liab(a) ]

DealInNow(a) = Σ_o P_o(t(a)) × Pay(o) × L_disc(o)
SelfDraw     = κ × Σ_o sd_o × Pay_sd(o) × L_sd(o)
OtherFeeds   = κ × Σ_o x_o  × Pay(o)    × L_other(o)
Later(a)     = Σ_{k=1..min(H, |h(a)|)} S(k) × Σ_o P_o(t_k) × Pay(o) × L_disc(o)
               t_k = the k-th safest tile copy of h(a) by Σ_o P_o(t)   (the fold plan)
               S(k) = (1 − e)^(k−1),  e = κ × P_win / n + Σ_o win_o / n
Liab(a)      = Σ_o 1[exposed_o == 3 ∧ liable_for(o) is None]
               × P_claim4_o(t(a)) × (1 − P_o(t(a)))
               × κ × [ sd4 × 2 × Pay_sd(o) + x4 × 3 × Pay(o) ]
               P_claim4_o(t) = min(1, P2_o(t) + 1[o is the next seat] × max_chow P1_o(u) × P1_o(v))
               sd4 = four_meld_win / 3, x4 = 2 × four_meld_win / 3   (four_meld_win = 0.5)
```

Why the shape is what B15 asks for: the hand ends at the first win, so a deal-in now
(`DealInNow`) removes every later term through `cont(a)`. When an opponent's self-draw
threat `sd_o × Pay_sd(o)` is large and another opponent waits on a cheap tile I hold,
`DealInNow + cont × SelfDraw` of that tile is smaller than `SelfDraw` of a safe tile, and
the bot deals in on purpose (§5, example 2).

Term by term:

- **Offence.** `P_win`, `P_sd`, `pts_T` from the table; `Pay(me)` floors the table's mean at
  the target's guaranteed faan so a cell averaged over several targets cannot undervalue a
  7-faan plan. The 1-share discard win is weighted by `φ`, the belief's expected counts of my
  waits in opponents' hands (A3). Bonus-tile faan I already hold are in `Target::faan` when
  the analysis crate counts them; the table's mean covers the rest.
- **DealInNow.** The sum over opponents, not `1 − Π`: under the default `multiple_winners`
  rule every winner is paid (rules §4.1.3, §8.2).
- **SelfDraw.** `sd_o` is already "before the hand ends", so it carries no survival factor;
  only the horizon cap `κ` applies.
- **OtherFeeds.** Costs me nothing unless I am liable for `o`; then 3 shares (rules §8.3.2).
- **Later.** The fold plan: each future turn I discard the safest tile left in `h(a)`
  (copies counted separately, so a pair gives two discards; the tile drawn each turn is
  unknown and adds nothing, A5). `S(k)` is the chance the hand is still running at my k-th
  future discard. The term is what makes "keep the dangerous tile" cost something when the
  hand holds few safe tiles, and nothing when it holds at least `H` safe tiles.
- **Liab.** The 12-tile lookahead (hard-problems 3b): discarding `t` to a seat with three
  exposed melds who can claim it as the fourth makes me liable; every later win by `o` then
  costs 3 shares (2 more than a self-draw would, 3 more than a win off another seat). A seat
  with four exposed melds waits on one tile, so its win chance is set to `four_meld_win`,
  split 1 : 2 self-draw : discard (§9 Q3).

Everything is in points for seat `me`, so an `EV` of −12 means "this action expects to lose
12 points over the rest of the hand".

### 2.1 Actions other than a plain discard

| Action | Value |
|---|---|
| `Win`, `SelfDrawnWin`, robbing a kong | always taken, as every bot in `bots.rs` does |
| `Discard(t)` | the formula above with `t(a) = t`, `h(a)` = hand minus `t` |
| `Pung`, `Chow`, `ClaimKong` (claim window) | `max_d EV(Discard(d))` over the hand after the claim (`after_claim` in `bots.rs`), with the claimed meld in `melds` for `faan_shanten`; if the claim gives me four exposed melds, `Shares(me) = 3` (the discarder becomes liable for me) |
| `Pass` (claim window) | the formula with no tile now: `cont = 1`, `DealInNow = 0`, `Liab = 0`, `Later` from the current hand's fold plan starting at `k = 1` |
| `ConcealedKong(t)`, `AddedKong(t)` (my turn) | `max_d EV(Discard(d))` over the hand after the kong (kong tiles moved to `melds`); the replacement draw is unknown and adds nothing (A8); compared against the best plain discard |

A claim is taken when its value beats `Pass` by more than `tie_margin`; a kong when its
value beats the best plain discard by more than `tie_margin`.

### 2.2 Ties

Two candidates whose `EV` differ by less than `tie_margin` (0.5 point) are ordered by
`rank_discards(view, candidates, Metric::Faan)` from `bots.rs` (faan-shanten, acceptance,
target faan, connectedness, tile index), so the `ev` bot plays like `value` when nothing is
at stake. The table has no acceptance axis, so this tie rule is what separates two discards
with the same faan-shanten and target.

### 2.3 Output: `EvTerms`

One per candidate, every number the bot used (Task 10: `EvBot::explain(state) ->
Vec<(Action, EvTerms)>`, sorted by `ev` descending):

`action, ev, offence, p_win, p_sd, pay_me, shares_me, cont, deal_in_now[4] (per seat; 0 for
me), self_draw[4], other_feeds[4], later, liability, fold_plan: Vec<Tile>` plus the belief
inputs used (`tenpai[4], sd[4], pay[4], pay_sd[4]`).

## 3. Approximations and how each is checked

| # | Approximation | Check (where) |
|---|---|---|
| A1 | `P_win`, `P_sd`, `pts_T` come from a table keyed by target kind, faan-shanten, tiles-left bucket and exposed melds, fitted on `value`-mix self-play; they ignore acceptance and the opponents' pressure under this action | Task 9's calibration table on 4,000 held-out hands, every cell within its interval; the tie rule (§2.2) restores acceptance |
| A2 | `tenpai_o`, `wait_o`, `P_b`, `sd_o` are the belief's point estimates | Task 5 `evaluate`: tenpai and deal-in calibration per wall-progress bucket (6 buckets, as `danger.rs`); Task 6 `danger_check --belief`: expected loss predicted vs observed per bucket |
| A3 | `φ` (feed ratio): when `s == 0`, `φ = clip(feed / feed_avg, 0.5, 1.5)`, with `feed = Σ_{t∈W} Σ_o E[cnt_o(t)] / Σ_{t∈W} unseen(t)` over my waiting tiles `W`, and `feed_avg` = (hidden hand tiles) / (hidden hand tiles + wall); otherwise `φ = 1` | Task 10's match: the fraction of my wins that were discard wins, by `φ` tercile, against `1 − P_sd` |
| A4 | `x_o = r_disc × sd_o` (no belief output for discard wins) | the log's `opp_won` field: observed discard-win / self-draw ratio per bucket; if it differs from 2.0 by more than 25 % in any bucket, change `r_disc` or add the output (§9 Q1) |
| A5 | the fold plan ignores the tiles I will draw, and `P_o(t)` stays at today's value over the horizon (tenpai rises in reality) | Task 10 `ev_calibrate` (optional, §9 Q5): among decisions where `Later ≥ 1 point`, predicted vs observed deal-ins over the next 8 turns |
| A6 | every seat has `n = ceil(wall / 4)` draws left; claims and kongs that shift turns are ignored | none needed: an error of one draw changes `κ` by at most 1 / n |
| A7 | `Pass` and a claim are compared as if they left the turn count unchanged | none; the claim's value difference is usually dominated by faan-shanten |
| A8 | the kong replacement draw adds nothing; a robbed added kong is ignored | none; kongs are rare |
| A9 | every `cont`, `S(k)` and the belief's probabilities treat the three opponents as independent | the EV-versus-realised check below |

**The whole formula**: Task 10's 4,000-hand match logs, for every decision, `EV(a*)` and the
realised final delta for `me`. The study results plot realised delta against predicted EV in
deciles; a calibrated bot lies near the diagonal. This is the one number that says the terms
add up, and it is reported with the AC5 result in this file's "Measured numbers" section
(Task 10 appends it).

## 4. Hyper-parameters (defaults; `ev:` spec keys in `agent.rs`)

| Key | Default | Meaning |
|---|---|---|
| `horizon` | 8 | `H`: my future turns valued |
| `fold` | `safest` | the fold plan: safest tile copy of `h(a)` each turn (the only plan in this run) |
| `clip` | 0.95 | `P_clip` for `P_o(t)` and `win_o` |
| `tie_margin` | 0.5 | points below which two candidates tie (§2.2) |
| `r_disc` | 2.0 | `x_o = r_disc × sd_o` |
| `four_meld_win` | 0.5 | win chance of a seat after it claims its fourth exposed meld |
| `feed_clip` | 0.5–1.5 | bounds on `φ` |
| `warn_p` | 0.10 | `reason.ev.deal_in_now` is emitted when some `P_o(t(a*)) ≥ warn_p` |
| `belief` | path | the MJNN v2 file; absent → the heuristic fallback (§8) |

Cost: 14 discards × (34 tiles × 3 opponents + one `faan_shanten` + one table lookup); well
under 5 ms on one thread (AC5). The belief runs once per decision, not per candidate.

## 5. Worked examples (by hand)

Rules: defaults (`min_faan` 3, half-spicy, flowers on, 12-tile liability on). Table values
marked *illustrative* are stand-ins until Task 9's fitted table exists; the tests of §6
stub the belief and assert orderings and reason keys, not these exact decimals.

### Example 1: an ordinary discard, nobody threatens (test position 1)

Seat `me` = 0 (dealer), turn phase, just drew `3p`. `wall = 32`, so `n = 8`, `H = 8`,
`κ = 1`. No liability anywhere. Concealed (14):

`1m 2m 3m 4m 5m 6m 7m 8m 8m R R 9p E 3p`

Target: half flush in characters with a 中 pung (混一色 3 + 中 1 = 4 faan guaranteed).
The off-plan tiles are `3p` and `9p`; `E` is in plan (an honour).

Belief stub, every opponent: `tenpai = 0.10`, `wait(t) = 0.05` for every tile, `sd = 0.04`,
bucket 3 with probability 1 (`Pay = 8`, `Pay_sd = 16`), `x = 0.08`, `P2 = 0`.

Per discarded tile: `P_o(t) = 0.10 × 0.05 = 0.005` each, `cont = 0.995³ = 0.985`.
Per turn of the fold plan: `Σ_o P_o(t_k) × Pay(o) = 3 × 0.005 × 8 = 0.12`.

Terms that are the same for every candidate:
- `DealInNow = 3 × 0.005 × 8 × 1 = 0.12`
- `SelfDraw = 1 × 3 × 0.04 × 16 × 1 = 1.92`; `OtherFeeds = 0` (not liable); `Liab = 0`
- `e = P_win / 8 + 3 × 0.12 / 8 = P_win / 8 + 0.045`

**Discard `3p`** (illustrative table: `s = 2`, `P_win = 0.20`, `P_sd = 0.30`, `pts_T = 20`):
- `Pay(me) = max(points(4) = 16, 20) = 20`; `Shares = 3 × 0.30 + 0.70 = 1.60`
- `Offence = 0.985 × 0.20 × 20 × 1.60 = 6.30`
- `e = 0.025 + 0.045 = 0.070`; `Σ_{k=1}^{8} 0.93^(k−1) = (1 − 0.93⁸) / 0.07 = 6.29`
- `Later = 6.29 × 0.12 = 0.75`
- `Defence = 0.12 + 0.985 × (1.92 + 0 + 0.75 + 0) = 0.12 + 2.63 = 2.75`
- `EV = 6.30 − 2.75 = +3.55`

**Discard `8m`** (breaks the pair; illustrative `s = 3`, `P_win = 0.10`, `P_sd = 0.30`,
`pts_T = 16`):
- `Offence = 0.985 × 0.10 × 16 × 1.60 = 2.52`
- `e = 0.0125 + 0.045 = 0.0575`; `Σ = (1 − 0.9425⁸) / 0.0575 = 6.56`; `Later = 0.79`
- `Defence = 0.12 + 0.985 × (1.92 + 0.79) = 2.79`
- `EV = 2.52 − 2.79 = −0.27`

`3p` and `9p` tie (same `s`, same target, same danger); the tie rule orders them by
`rank_discards`, which gives `3p` (same key, lower tile index). This is exactly `value`'s
choice. Reason keys: `reason.ev.offence {p_win: 0.197, points: 32.0}`,
`reason.ev.margin {points: 3.8}` (against `8m`, the best non-tied alternative). No
`deal_in_now` (every `P_o ≤ 0.005 < warn_p`).

### Example 2: the deliberate deal-in (test position 2)

Seat `me` = 0, turn phase, just drew `Wh`. `wall = 30`, `n = 8`, `κ = 1`. Nobody is liable
for anyone. Concealed (14), far from any target:

`1m 2m 4m 6m 9m 2p 3p 5p 7p 8p S W N Wh`

- **Seat A = 2 (across)**: exposed pungs `111s 999s RRR`; concealed `5s 5s 6s 6s` (hidden).
  A real self-draw on `5s` scores 對對糊 3 + 混一色 3 + 中 1 + 自摸 1 + 無花 1 = 9 faan = 96
  per share; a discard win 8 faan = 64 (rules §7.3 #15, #16; §7.1 #1, #3; §8.1).
- **Seat B = 1 (next)**: exposed chows `345m 567p 234s`; concealed `E E E 2m` (hidden), so
  B waits on `2m` for the pair. I hold one `2m`.
- **Seat C = 3**: 13 concealed tiles, no melds.

Belief stub:

| | `tenpai` | `wait` | `sd` | bucket → `Pay`, `Pay_sd` | `x` |
|---|---|---|---|---|---|
| A | 0.90 | 1.0 on `5s`, `6s`; 0 on every tile I hold | 0.30 | 8 → 64, 96 | 0.60 |
| B | 0.95 | 1.0 on `2m`; 0 elsewhere | 0.05 | 4 → 16, 24 | 0.10 |
| C | 0 | 0 | 0 | 3 → 8, 16 | 0 |

`P2 = 0` for every seat and tile (so `Liab = 0`). Illustrative table for my hand after any
discard: `s = 4`, `P_win = 0.03`, `P_sd = 0.30`, `pts_T = 16` → `Offence_base = 0.03 × 16 ×
1.60 = 0.77`.

**Discard a safe tile (`Wh`; every tile but `2m` has `P_o = 0`)**:
- `cont = 1`, `DealInNow = 0`
- `SelfDraw = 0.30 × 96 × 1 + 0.05 × 24 × 1 + 0 = 28.8 + 1.2 = 30.0`
- `OtherFeeds = 0` (nobody liable); `Later`: the 8 safest of the 13 remaining tiles all have
  `P_o = 0`, and `2m` is the 13th, so `Later = 0`; `Liab = 0`
- `Defence = 30.0`; `Offence = 0.77`; **`EV = −29.23`**

**Discard `2m` (B's wait)**:
- `P_B(2m) = min(0.95 × 1.0, 0.95) = 0.95`; `P_A = P_C = 0`; `cont = 0.05`
- `DealInNow = 0.95 × 16 × 1 = 15.2`
- `SelfDraw = 30.0` as above, `Later = 0`, `OtherFeeds = 0`, `Liab = 0`
- `Defence = 15.2 + 0.05 × 30.0 = 15.2 + 1.5 = 16.7`
- `Offence = 0.05 × 0.77 = 0.04`; **`EV = −16.66`**

The bot discards `2m`: a near-certain 16-point loss ends the hand and removes the expected
28.8-point loss to A's self-draw. Margin `−16.66 − (−29.23) = 12.6` points. Reason keys:
`reason.ev.stops_big_hand {seat: 2, p_self_draw: 0.30, points: 96}`,
`reason.ev.deal_in_now {seat: 1, p: 0.95, points: 16, shares: 1}`,
`reason.ev.margin {points: 12.6}`.

The plan's illustration ("deal into an 8-point hand to stop a 128 × 3 self-draw") is the
same arithmetic with B at bucket 3 (`Pay = 8`) and A at bucket 10 (`Pay_sd = 128`, A
collects 128 from each of three seats; I pay one share, 128): safe tile `EV ≈ −0.30 × 128 =
−38.4`; deal in `≈ −(0.95 × 8 + 0.05 × 38.4) = −9.5`.

### Example 3: the same table, but I am liable for B (test position 3)

As example 2, except B's `EEE` is an exposed pung claimed from me, so B shows four exposed
melds (`345m 567p 234s EEE`), holds the single `2m` (hidden), and `liable_for(1) == 0`
(rules §8.3.1). Every win by B now costs me 3 shares: `L_disc(B) = L_sd(B) = L_other(B) = 3`.

**Discard `Wh`**:
- `SelfDraw = 0.30 × 96 × 1 + 0.05 × 24 × 3 = 28.8 + 3.6 = 32.4`
- `OtherFeeds = 0.10 × 16 × 3 = 4.8`; `Later = 0`
- `Defence = 37.2`; `Offence = 0.77`; **`EV = −36.43`**

**Discard `2m`**:
- `DealInNow = 0.95 × 16 × 3 = 45.6`
- `Defence = 45.6 + 0.05 × (32.4 + 4.8) = 45.6 + 1.86 = 47.46`
- `Offence = 0.04`; **`EV = −47.42`**

The bot does **not** deal in (margin 11.0 in favour of `Wh`). Reason keys:
`reason.ev.liability {seat: 1}`, `reason.ev.margin {points: 11.0}`; no `stops_big_hand`
(the chosen action is the safe one), no `deal_in_now` (`P_o(Wh) = 0`).

## 6. The three Task 10 test positions

Built with `State::from_position` (Task 2) and a stub `Belief` returning the numbers of §5,
so the tests do not depend on model weights. The target table is the fitted one, so the
tests assert orderings and keys, not decimals. `wall_remaining` as in §5; `river` holds
enough discards (any safe-looking tiles consistent with the counts) for the tile totals to
balance; `turn = 0`, `phase = turn`, `last_drawn` as stated, `liable` as stated.

| # | Position | Expected answer | Assertions |
|---|---|---|---|
| 1 | §5 example 1 | `Discard(3p)` (or `9p`: whichever `rank_discards` puts first) | the choice equals `Bot::by_name("value")`'s choice on the same state; every candidate's `deal_in_now` sums to ≤ 0.2 points; the top reason key is `reason.ev.offence`; `reason.ev.margin` present |
| 2 | §5 example 2 | `Discard(2m)` | `explain` ranks `2m` first with `ev(2m) − ev(second) ≥ 8`; keys contain `reason.ev.stops_big_hand {seat: 2, …}` and `reason.ev.deal_in_now {seat: 1, …}` |
| 3 | §5 example 3 | not `2m` | `explain` ranks `2m` last with `ev(first) − ev(2m) ≥ 8`; keys contain `reason.ev.liability {seat: 1}` and not `reason.ev.stops_big_hand` |

Timing: 1,000 calls of `choose` on position 2, one thread, `< 5 ms` each on average (AC5).

Tile-count check for positions 2 and 3 (each kind ≤ 4 across hands, melds and river):
`2m` me 1 + B 1 = 2; `5p`, `7p` me 1 + B 1 = 2 each; `1s`, `9s`, `R` A 3 each; `5s`, `6s`
A 2 each; `E` B 3; `2s 3s 4s`, `3m 4m 5m`, `6p` B 1 each (`4m`: me 1 + B 1 = 2). The
builder fills C's 13 tiles, the river and the wall from the remainder.

## 7. Reason keys (Task 20 maps them to the app's vocabulary)

Emitted for the chosen action `a*`, from its `EvTerms`; `a_safe` is the legal discard with
the smallest `DealInNow`, and `a₂` the second-best candidate by `EV`.

| Key | Args | When |
|---|---|---|
| `reason.ev.offence` | `p_win` = `cont × κ × P_win`, `points` = `Pay(me) × Shares(me)` | `Offence(a*) > 0` and no `stops_big_hand` |
| `reason.ev.deal_in_now` | `seat`, `p` = `P_o(t(a*))`, `points` = `Pay(o)`, `shares` = `L_disc(o)` | some `P_o(t(a*)) ≥ warn_p`; one key per such seat, largest `p` first |
| `reason.ev.stops_big_hand` | `seat` = argmax_o `sd_o × Pay_sd(o) × L_sd(o)`, `p_self_draw` = `sd_o`, `points` = `Pay_sd(o)` | `DealInNow(a*) − DealInNow(a_safe) ≥ 1` and `Defence(a*) < Defence(a_safe)` |
| `reason.ev.liability` | `seat` | a 3-share multiplier applied to a non-zero term of `a*` (seat = that `o`), or `a*` avoided a tile with `Liab ≥ 1` point (seat = the seat with three exposed melds) |
| `reason.ev.margin` | `points` = `EV(a*) − EV(a₂)` (0 when tied) | always |

Every key matches `^reason\.[a-z_.]+$` (Task 20's test). Engines return keys, never text
(D42).

## 8. Fallback without a belief model (plan risk table: AC3 fails)

Used only when no `belief=` is given. Sources: `Danger` (`crates/search/src/danger.rs`) with
Task 6's additions.

- `tenpai_o = Danger::threat(o)`; `P_o(t) = min(threat(o) × WAIT[wait_cell], P_clip)` (the
  per-opponent factor inside `Danger::estimate`; Task 6 exposes it as
  `Danger::expected_loss` pieces).
- `Pay(o)` from `o`'s exposed melds: 3 faan, +1 per scoring pung (dragon, own wind,
  prevailing wind: the `has_value` test in `danger.rs`), +3 if all exposed melds are in one
  suit (the `flush` tag), +3 if `exposed_o ≥ 2` and all exposed melds are pungs; capped at 10;
  `Pay_sd` one faan higher.
- `sd_o = 0.25 × tenpai_o`; `x_o = r_disc × sd_o`; `P2_o(t) = 0.1` when `unseen(t) ≥ 2` else 0.
- The rest of the formula is unchanged. Reported as the fifth row of Task 21's 1-vs-3 match.

## 9. Open questions (defaults apply; the builder does not wait)

1. **`x_o`, the chance `o` wins on a discard.** The belief (Task 5) outputs the self-draw
   chance only. Default: `x_o = 2.0 × sd_o`, from real-play rates (about 15 % discard wins
   and 7 % self-draws per seat per hand; plan AC4 text). Better: add a "wins on a discard
   before the hand ends" output to MJNN v2 in a later revision and use it directly.
2. **Faan buckets mean a discard win.** `P_b(o)` is read as the points of a discard win now
   (the log's `opp_faan_now`), and a self-draw adds one faan. If Task 5 trains the buckets on
   the realised win (`opp_win_faan`, which mixes both), drop the `+1` for `Pay_sd`. Default:
   as written here; Task 5's `docs/belief.md` states which it is.
3. **`four_meld_win = 0.5`**, split 1 : 2 self-draw : discard, for a seat right after its
   fourth exposed meld. Default as stated; Task 9's table (exposed melds = 4, `s = 0`) gives
   the measured number, and the builder should read it from the table when that cell exists.
4. **`Pay(me) = max(points(f_T), pts_T)`.** Settled: Task 9 stores `E[points per share |
   win]` as `TargetValue::points`, so the max rule applies as written. Kept here so the
   builder sees the floor is deliberate (a cell averaged over several targets).
5. **The `ev_calibrate` example (A5)** is extra work not in Task 10's file list. Default: not
   built in this run; A5 is checked only through the EV-versus-realised plot.
6. **Pure EV, no extra rule** (plan open question 4): the bot deals in whenever
   `Defence(deal-in) + lost Offence < Defence(safe)`, even when its own hand is close. Default:
   pure EV; the test positions check it.
7. **`tie_margin = 0.5` point.** Larger values make the bot play more like `value`; smaller
   ones let table noise decide. Default 0.5; Task 10 may report a sweep over {0.25, 0.5, 1}
   if time allows.

## 10. Measured numbers

Filled in by Task 10 (the AC5 match, timing, the EV-versus-realised plot) and Task 21.
