# Hong Kong mahjong — rules specification

This document is the single source of truth for the rules implemented in `crates/hk`.
Every rule has at least one test, in `crates/hk/tests/` or in the Python reference
oracle (`python/tests/reference/`), and tests cite the section numbers used here. If
the code and this document disagree, the code is wrong.

Rules that differ between tables are collected in a `RuleSet`. The **default** value
of each variant is marked *(default)*.

> **Status: draft for review.** Points marked **[confirm]** are defaults I chose where
> house rules commonly differ. They are listed together in §11.

## 1. Tiles

144 tiles: 136 playing tiles (4 copies of each of 34 kinds) and 8 bonus tiles.

| Group | Kinds | Text | Index |
|---|---|---|---|
| Characters 萬 | 1–9 | `1m` … `9m` | 0–8 |
| Dots 筒 | 1–9 | `1p` … `9p` | 9–17 |
| Bamboo 索 | 1–9 | `1s` … `9s` | 18–26 |
| Winds 風 | East 東, South 南, West 西, North 北 | `E S W N` | 27–30 |
| Dragons 箭 | White 白, Green 發, Red 中 | `Wh G R` | 31–33 |
| Flowers 花 | 梅 蘭 菊 竹 (numbered 1–4) | `F1` … `F4` | 34–37 |
| Seasons 季 | 春 夏 秋 冬 (numbered 1–4) | `S1` … `S4` | 38–41 |

Terms used below:
* **Suited tiles:** characters, dots and bamboo. **Honours:** winds and dragons.
* **Terminals:** the 1s and 9s of each suit.
* **Chow 上:** three consecutive tiles of one suit, e.g. `345p`.
* **Pung 碰:** three identical tiles. **Kong 槓:** four identical tiles. A kong counts
  as a pung wherever a pattern asks for one.
* **Meld:** a chow, pung or kong. A meld is **exposed** if it was made with a
  claimed discard (§4), and **concealed** otherwise. Concealed kongs (§5.1) stay
  concealed even though they are placed on the table.

## 2. Seats and winds

1. Four players in seats 0–3. Play passes 0 → 1 → 2 → 3 → 0.
2. Seat 0 is the dealer 莊家. The seat winds are East (seat 0), South (1), West (2)
   and North (3).
3. The prevailing wind 圈風 is East *(default)*. A single hand is the unit of play and
   study. Full matches with dealer rotation are in §10.

## 3. Deal

1. The 144 tiles are shuffled into one wall. Tiles are drawn from the **front**.
   Replacement tiles (for bonus tiles and kongs) are drawn from the **back**.
2. The dealer takes 14 tiles and every other seat takes 13.
3. **Initial flower replacement.** Starting with the dealer and going in seat order,
   each player sets aside all bonus tiles in their hand and draws the same number of
   replacements from the back. This repeats until no hand holds a bonus tile.
4. Bonus tiles are always placed face up beside the hand and are never part of it.
5. With the `flowers = off` variant, the 8 bonus tiles are removed before the
   shuffle, and §3.3, §5.3 and the flower patterns of §7 do not apply.
   Flowers are *on* by default.

## 4. A turn

The player to move holds 14 tiles, counting each kong as 3. Their turn has three
parts:

1. **Draw.** They draw one tile from the front of the wall. (The dealer's first turn
   skips this; so does a turn that follows a claimed pung, chow or kong, §4.2.)
   A drawn bonus tile is replaced at once (§5.3).
2. **Self actions**, in any order, any number of times:
   * declare a **self-drawn win** if the hand is complete and scores at least the
     minimum (§6, §7);
   * declare a **concealed kong** or an **added kong** (§5.1–§5.2), then draw a
     replacement tile.
3. **Discard** one tile face up. The discard opens the **claim window** (§4.1).

### 4.1 Claims on a discard

Each of the other three players may declare at most one claim on the discard:

| Claim | Who may | Needs in hand | Result |
|---|---|---|---|
| **Win** 食糊 | any other player | the discard completes the hand with ≥ minimum faan | the hand ends |
| **Pung** 碰 | any other player | 2 tiles of the same kind | exposed pung |
| **Kong** 槓 | any other player | 3 tiles of the same kind | exposed kong, then a replacement draw |
| **Chow** 上 | only the next player | the 2 other tiles of a chow | exposed chow |

1. Claims are declared **simultaneously**: no player sees another's declaration
   before the claim is resolved. (The engine collects them one seat at a time, but
   hides earlier declarations from later seats.)
2. **Priority:** win > pung or kong > chow. Only one pung or kong is possible, since
   at most one other player can hold 2 copies of a tile.
3. **Several winners:** if two or three players declare a win, **all of them win**
   *(default)*, and each is paid by the discarder (§8). Variant: `head_bump`
   截糊, where only the first winner after the discarder in turn order wins.
4. After a pung or chow, the claimer **discards without drawing** and play continues
   from their seat. A player who skipped a turn because of a pung (or kong) on someone
   else's discard simply loses that turn.
5. If nobody claims, the next player in seat order draws.
6. There is no restriction on passing up a win and winning later on the same tile
   (Hong Kong mahjong has no *furiten* rule).
7. The **last discard** of the hand (after the wall is exhausted) can be claimed
   only for a win. **[confirm]**

## 5. Kongs and bonus tiles

### 5.1 Concealed kong 暗槓

On their turn, a player holding all four copies of a kind may lay them down as a
concealed kong, then draw a replacement tile from the back of the wall. The kong's
tile is shown when it is declared. Only a player waiting on that tile for
**thirteen orphans** (#25) may rob a concealed kong (§5.2).

### 5.2 Added kong 加槓

On their turn, a player with an exposed pung who holds the fourth copy may add it to
the pung, then draw a replacement. Before the replacement is drawn, the other players
may **rob the kong** 搶槓: claim the added tile for a win as if it had been
discarded (§4.1, win only). If anyone robs it, the kong is cancelled and the hand
ends. A concealed kong can be robbed only for thirteen orphans (§5.1).

### 5.3 Bonus tiles during play

A player who draws a bonus tile (including as a replacement) places it face up and
draws a replacement from the back. They cannot discard bonus tiles, and nobody can
claim them.

### 5.4 Kong replacement wins

A player who completes their hand with a kong's replacement tile wins by self-draw
(§7, patterns 7 and 8). If the replacement is a bonus tile, the tile drawn to replace
it still counts as the kong's replacement.

## 6. Winning hands

A complete hand has 14 tiles (each kong counted as 3) in one of these two forms:

1. **Standard:** four melds and a pair 眼, e.g. `123m 456m 789p EEE 55s`.
2. **Thirteen orphans** 十三么: one of each terminal and honour (13 kinds), plus a
   second copy of any one of them.

Other forms such as seven pairs do not exist under these rules.

A complete hand may be declared a win only if it scores **at least 3 faan**
*(default; variant `min_faan`)*. Flower faan count toward the minimum.

When a hand can be split into melds in more than one way, the split that scores the
most faan is used.

## 7. Faan table

The faan of every pattern that applies are added together, and the total is capped
at 10 *(default; variant `faan_cap`)*. A **limit hand** scores the cap directly; the
other patterns are then not added.

Where a row says *replaces* another, only the larger of the two is counted.

### 7.1 Situational and bonus patterns

| # | Pattern | Faan | Condition |
|---|---|---|---|
| 1 | 自摸 Self-drawn | 1 | The winning tile was drawn from the wall (a replacement draw counts) |
| 2 | 門前清 Concealed hand | 1 | No exposed melds. Concealed kongs are allowed. Applies to self-drawn and discard wins alike, so a concealed self-drawn hand scores 2 with #1 |
| 3 | 無花 No flowers | 1 | The winner has no bonus tiles. Only when flowers are in play |
| 4 | 正花 Seat flower | 1 each | Holding the flower and/or the season whose number matches the seat (East=1, South=2, West=3, North=4) |
| 5 | 一台花 Full set | 2 | Holding all four flowers, or all four seasons. Replaces the seat-flower faan from that set; both sets give 4 |
| 6 | 海底撈月 Last tile | 1 | Self-drawn win on the last tile of the wall. 自摸 also counts |
| 7 | 槓上開花 Kong replacement | 1 | Win on a kong's replacement tile. 自摸 also counts, so the two give 2 together |
| 8 | 槓上槓 Double kong | 8 | Win on the replacement tile of a second kong declared in the same turn. 8 in total: replaces #7 and #1 |
| 9 | 搶槓 Robbing the kong | 1 | Win on the tile someone else adds to a pung (§5.2), or on a concealed kong's tile with thirteen orphans (§5.1) |

### 7.2 Meld patterns

| # | Pattern | Faan | Example |
|---|---|---|---|
| 10 | 三元牌 Dragon pung | 1 each | `RRR` |
| 11 | 門風 Seat wind pung | 1 | South seat with `SSS` |
| 12 | 圈風 Prevailing wind pung | 1 | `EEE` in the East round. For the East seat, `EEE` scores both #11 and #12 (2 faan) |

### 7.3 Whole-hand patterns

| # | Pattern | Faan | Example | Notes |
|---|---|---|---|---|
| 13 | 雞糊 Chicken hand | 0 | `123m 555p 789s 234s 99p` | No pattern applies |
| 14 | 平糊 All chows | 1 | `234m 567m 345p 678s 99s` | Four chows and any pair. Bonus tiles are never part of the hand, so flowers do not affect it |
| 15 | 對對糊 All pungs | 3 | `222m 555p 888s EEE 99p` | Four pungs or kongs |
| 16 | 混一色 Half flush | 3 | `123p 456p 789p RRR NN` | One suit plus honours, with at least one honour |
| 17 | 清一色 Full flush | 7 | `111s 234s 567s 789s 55s` | One suit, no honours. Replaces #16 |
| 18 | 混么九 Terminals and honours | 1 | `111m 999p EEE GGG 11s` | Only terminals and honours, with both present. Such a hand is always also #15, so the two give 4 together |
| 19 | 小三元 Small three dragons | 5 | `RRR GGG WhWh 123m 456m` | Pungs of two dragons and a pair of the third. Includes the two dragon pungs (#10 is not added for them) |
| 20 | 大三元 Big three dragons | 8 | `RRR GGG WhWhWh 234p 55s` | Pungs of all three dragons. Includes the three dragon pungs |

### 7.4 Limit hands (score the cap, 10 faan)

| # | Pattern | Example | Notes |
|---|---|---|---|
| 21 | 小四喜 Small four winds | `EEE SSS WWW NN 123m` | Pungs of three winds and a pair of the fourth |
| 22 | 大四喜 Big four winds | `EEE SSS WWW NNN 55m` | Pungs of all four winds |
| 23 | 字一色 All honours | `EEE SSS RRR GGG WW` | |
| 24 | 清么九 All terminals | `111m 999m 111p 999s 99p` | |
| 25 | 十三么 Thirteen orphans | `1m 9m 1p 9p 1s 9s E S W N Wh G R R` | §6.2 |
| 26 | 九子連環 Nine gates | `1112345678999m` + any `m` | Fully concealed, one suit; complete with any further tile of the suit |
| 27 | 坎坎糊 Four concealed pungs | `222m 555p 888s EEE 99p`, all concealed | Four concealed pungs or kongs. A pung completed by claiming the winning discard is exposed |
| 28 | 十八羅漢 Four kongs | four kongs and a pair | |
| 29 | 天糊 Heavenly hand | | The dealer's hand is complete after the initial deal (and flower replacement) |
| 30 | 地糊 Earthly hand | | A non-dealer wins on the dealer's first discard |
| 31 | 八仙過海 Eight immortals | | A player collects all 8 bonus tiles; they win at once, as a self-draw |
| 32 | 七搶一 Seven rob one | | A player holding 7 bonus tiles wins at once when another player draws the 8th. Paid as a discard win by the player who drew it |

Patterns 29–32 do not require a complete hand form beyond what they state (29 and 30
do require a complete hand).

### 7.5 Worked examples

| Hand and situation | Faan |
|---|---|
| `123p 456p 789p RRR NN`, self-drawn, concealed, no flowers, South seat | 混一色 3 + 中 1 + 自摸 1 + 門前清 1 + 無花 1 = **7** |
| `234m 567m 345p 678s 99s`, won on a discard, `234m` was a claimed chow, holds F2 as South seat | 平糊 1 + 正花 1 = **2**: cannot win |
| `EEE 111m 999p GGG 11s`, East seat, East round, discard win, exposed `GGG`, no flowers | 對對糊 3 + 混么九 1 + 門風 1 + 圈風 1 + 發 1 + 無花 1 = **8** |
| `RRR GGG WhWhWh 111s 22s`, half flush, self-drawn | 大三元 8 + 混一色 3 + 對對糊 3 + 自摸 1 = 15 → capped at **10** |

## 8. Payment

### 8.1 Points per share (half-spicy 半辣上)

| Faan | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| Points | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |

Up to 4 faan the payment doubles with each faan. After that, each extra faan adds
half of the previous doubling. (Variant `payout = full_spicy` 全辣 keeps doubling:
3 → 8, 4 → 16, …, 10 → 1024.)

### 8.2 Who pays

| Win | Payment |
|---|---|
| On a discard (including robbing a kong, §5.2) | The discarder pays the winner **1 share**. The other two pay nothing |
| Self-drawn | Each of the three opponents pays the winner 1 share |
| Several winners on one discard (§4.1.3) | The discarder pays each winner 1 share of that winner's points |
| 七搶一 (#32) | The player who drew the 8th bonus tile pays 1 share |
| Exhaustive draw (§9) | Nobody pays |

Kongs carry no separate payment. Every hand is zero-sum.

### 8.3 Liability: the 12-tile rule 十二張落地

1. If a player's discard is claimed to form another player's **fourth exposed meld**,
   so that 12 of that player's tiles are showing, the discarder becomes **liable** for
   that player. (Concealed kongs do not count toward the four.)
2. If that player then wins the hand, by discard or by self-draw, the liable player
   pays **3 shares** and nobody else pays.
3. This is the only liability rule *(default; variant `twelve_tile_liability = off`)*.
   Feeding the third dragon pung or the fourth wind pung does not create liability.

## 9. Exhaustive draw 流局

When a draw is due and the wall is empty, the hand ends with no winner and no
payment. No tiles are held back from the wall.

## 10. Matches (later)

A match is four rounds (East, South, West, North prevailing wind). The dealer keeps
the deal after winning the hand or after an exhaustive draw; otherwise the deal
passes to the next seat. Phase 1 studies single hands; matches are added
when the study needs them.

## 11. Points to confirm

1. §4.1.7: the last discard of the hand (made after the last wall tile has been
   drawn) can be claimed only for a win, not for a pung, kong or chow.
