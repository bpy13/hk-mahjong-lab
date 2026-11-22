# The rules file checked line by line against Hong Kong sources

Plan Task 28 of `docs/plans/2026-11-16-self-play-network.md` (slot 3, 2026-11-23). Brian asked
for it after two rules of the 17 Oct draft were found wrong only by looking at hands (the flower
win by robbing, 20 Nov; the 12-tile liability on a discard win, 23 Nov). Read-only on code: nothing
in `docs/rules/hk.md` or the engine changes here. Brian decides each difference; the questions are
in `docs/inbox.md` under the title beginning "The rules file against Hong Kong sources".

## Counts

| Mark | Rows |
|---|---|
| agrees | 94 |
| differs | 14 |
| sources disagree among themselves | 10 |
| no source found | 5 |
| already decided (Brian) | 3 |
| **total** | **126** |

126 rows cover every numbered rule, every unnumbered rule sentence that sets play or payment,
and every table row of `docs/rules/hk.md` (§1 to §10, faan rows #1 to #32, the worked examples
and the points table). Section 12 below lists rules the sources have and our file lacks; they
are not counted above. The 24 rows marked differs or sources disagree make 20 numbered questions
in the inbox (several rows share one question).

## Sources

Every page was read in full on 2026-11-23. Quotes are copied as the page writes them. All the
Chinese sources below are in Traditional characters (the zh.wikipedia pages were read through
the zh-hk variant); none is in Simplified.

| Key | Source |
|---|---|
| **HKMA** | 香港麻雀協會, 香港麻雀總例 (© 2025), linked from https://www.hkmahjong.org/rules ([Google Doc](https://docs.google.com/document/d/1TgdYpE_5Qiht_lBco3BqN_ijZJDRWugwzRzv946raq0/edit)). The only rulebook here written by a Hong Kong body; section names are the document's own |
| **HKMA-EN** | Hong Kong Mahjong Association, "Hong Kong Mahjong (Classical Mahjong)", 17 May 2023 ([Google Doc](https://docs.google.com/document/d/1a66AhGRZE6IJmA_nd-C7gIQfSJ7sXo_2-c4XgVnMpTQ/edit)). Played without flowers |
| **W1** | zh.wikipedia 香港麻雀, https://zh.wikipedia.org/zh-hk/香港麻雀 (oldid 92218886) |
| **W2** | zh.wikipedia 香港麻雀胡牌列表 (shown as 香港麻雀食糊列表), https://zh.wikipedia.org/zh-hk/香港麻雀胡牌列表 (oldid 95740220). Its headline faan are "清章十三番底正統牌型"; footnotes give other tables' values |
| **W3** | zh.wikipedia 包牌, https://zh.wikipedia.org/zh-hk/包牌 (oldid 90651455; no sources, mostly Japanese mahjong) |
| **Y1** | zh-yue.wikipedia 香港蔴雀食糊牌型, https://zh-yue.wikipedia.org/wiki/香港蔴雀食糊牌型 (oldid 2469371; same content as W2 in Cantonese) |
| **Y2** | zh-yue.wikipedia 香港蔴雀, https://zh-yue.wikipedia.org/wiki/香港蔴雀 (oldid 1834319) |
| **E1** | en.wikipedia Hong Kong mahjong scoring rules, https://en.wikipedia.org/wiki/Hong_Kong_mahjong_scoring_rules (oldid 1385162043; marked "accuracy disputed") |
| **E2** | en.wikipedia Mahjong, section "Old Hong Kong mahjong rules", https://en.wikipedia.org/wiki/Mahjong |
| **S** | Tom Sloper, FAQ 17 Hong Kong Old Style, https://www.sloperama.com/mjfaq/mjfaq17.html |
| **K** | mahjong.wikidot.com, Hong Kong Old Style pages, https://mahjong.wikidot.com/rules:hong-kong-old-style-scoring |

Pages that would not load or held no rules: mahjong.com.hk and hkmahjong.com (socket closed),
mahjongtime.com (error page), dragontiles.com (DNS), mahjongpros.com (no Hong Kong page), Victoria
Mahjong Club cheat sheets (images only).

Where sources disagree, HKMA is given first: it is a written rulebook from Hong Kong, current
(2025), and every default proposed in the inbox follows it unless all other Chinese sources say
otherwise.

## 1. Tiles (§1)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 1 | 144 tiles: 136 playing and 8 bonus | W1: "標準的香港麻雀會使用上述144張牌。" | agrees |
| 2 | Characters 萬 1–9 | W1 tile list | agrees |
| 3 | Dots 筒 1–9 | W1 tile list | agrees |
| 4 | Bamboo 索 1–9 | W1 tile list | agrees |
| 5 | Winds 東南西北 | W1 tile list | agrees |
| 6 | Dragons 白發中 | W1 tile list | agrees |
| 7 | Flowers 梅蘭菊竹 numbered 1–4 | W1: "春、夏、秋、冬；梅、蘭、菊、竹。其中春、梅代表東位；夏、蘭代表南位…" | agrees |
| 8 | Seasons 春夏秋冬 numbered 1–4 | as row 7 | agrees |
| 9 | A kong counts as a pung wherever a pattern asks for one | HKMA 對對糊 2.2: "手牌有四個坎/槓﹐沒有順子。" | agrees |
| 10 | A meld made with a claimed discard is exposed, otherwise concealed | HKMA 坎坎糊 6.4: "對碰必須自摸" (a pung completed on a discard is not concealed) | agrees |
| 11 | Concealed kongs stay concealed though placed on the table | HKMA 門前清 1.12: "胡牌前沒有「吃」、「碰」、「槓」過任何牌。"; HKMA-EN 1.2: "(Concealed kan are considered as melded melds)"; W2 門前清: "沒有上、碰、槓（包括暗槓）任何牌而食糊". For 門前清 and 坎坎糊 a concealed kong breaks concealment; for the 12-tile liability it does not count (row 118) | differs (questions 3 and 4) |

## 2. Seats and winds (§2)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 12 | Four players; play passes 0 → 1 → 2 → 3 | W1: "玩家應牢記「逆時針行牌，順時針摸牌」這個規定。" (the order is the same; the seat numbers are ours) | agrees |
| 13 | Seat 0 is the dealer; seat winds E S W N | HKMA 配牌 4: "東家應有14張牌" (the dealer is East) | agrees |
| 14 | Prevailing wind East by default; a single hand is the unit | Y2: rounds "東圈、南圈、西圈、北圈", each "預設有四局". One hand of the East round is a hand of a real match | agrees |

## 3. Deal (§3)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 15 | One shuffled wall; draws from the front, replacements from the back | W1: "槓後均須即時在牌牆尾端補牌"; HKMA 開槓 7: "從「牌尾」摸一張". Wall-building and dice (W1) change nothing in a shuffled wall | agrees |
| 16 | Dealer 14 tiles, others 13 | HKMA 配牌 4: "東家應有14張牌﹐閒家應各有13張牌" | agrees |
| 17 | Initial flower replacement: dealer first, in seat order, each sets aside all bonus tiles and draws as many from the back; repeated until no hand holds one | HKMA 配牌 5: "從牌尾補花，由東家開始補，東家補完後再請示南家補花"; 6: a flower drawn as a replacement waits for a second round from East. W1 adds that the dealer may also keep replacing at once ("有些規則也會規定先讓南家、西家、北家依次補花後，才再次補花"). Ours is HKMA's order; W1's changes only which back tile each seat gets | agrees |
| 18 | Flower wins are checked at the set-aside in the deal; a hand dealt all eight wins 大花糊 at once | HKMA 花糊牌例 4: a non-dealer holding 7 flowers from the deal replaces them, makes no claims in the first go-round, and declares at its first draw instead of drawing. W2: 7 or more flowers in the opening hand counts as 天糊 for the dealer, 地糊 for others (custom) | sources disagree (question 17) |
| 19 | Bonus tiles face up beside the hand, never part of it | W1: "摸到花牌後，不論是否與自己的風位（門風）配合，都要翻開，並立即在牌牆尾端補牌" | agrees |
| 20 | `flowers = off` removes the 8 bonus tiles before the shuffle | W1: "若不打花牌，則洗牌前先抽起兩臺花，只用136張牌。" | agrees |

## 4. A turn and claims (§4, §4.1)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 21 | Draw from the front; the dealer's first turn and a turn after a claim skip it; a drawn bonus tile is replaced at once | W1 (no draw after a claim: "可以不從牌牆裏摸牌"); HKMA 配牌 3–4 | agrees |
| 22 | Self actions: self-drawn win, concealed or added kong with a replacement | HKMA 開槓, 食糊 | agrees |
| 23 | Discard one tile face up; it opens the claim window | HKMA 麻雀的流程 | agrees |
| 24 | Claim row: win 食糊 by any other player | HKMA 鳴牌優先規條 | agrees |
| 25 | Claim row: pung 碰 by any other player | W1: claims from "其他玩家的棄牌" | agrees |
| 26 | Claim row: kong 槓 (3 in hand) by any other player, then a replacement | HKMA 鳴牌優先規條 2 ("槓牌（大明槓）") | agrees |
| 27 | Claim row: chow 上 only by the next player | HKMA 上牌 1: "只能上上家打出的牌" | agrees |
| 28 | §4.1.1 claims are declared at once, hidden from each other | HKMA: "只有最優先者的鳴牌成立" (priority, not speed, settles claims); Y1: "亦有玩家選擇鬥快叫「糊牌」…但呢種糊牌方法並冇根據。" | agrees |
| 29 | §4.1.2 priority win > pung or kong > chow | HKMA 鳴牌優先規條 2: "食糊 > 碰牌﹑槓牌（大明槓）> 上牌"; HKMA-EN: "Ron (win on chuck) > Pong/kan > Chi" | agrees |
| 30 | §4.1.3 several winners on one discard all win (default); `head_bump` 截糊 a variant | W2: "有些玩家更會使用無截糊的準則（多在打全銃制中使用），打出一張牌多家可同時食糊，稱為「一炮多響」"; W2 十三么: "不能被其他牌型截糊" (so 截糊 is the page's default); E2: "the player with the most faan value wins. If that is equal, then the player closest to the player who discarded". HKMA: not stated beyond the priority line | sources disagree (question 12) |
| 31 | §4.1.4 after a pung or chow the claimer discards without drawing; a skipped player loses the turn | W1; HKMA 上牌 | agrees |
| 32 | §4.1.4 (unstated) any tile may be discarded after a claim | HKMA 簡則 15 forbids "碰牌後打出與該個明坎相同的牌或上牌後打出能跟上牌用的搭子形成暗順的牌" (食換, a yellow card); HKMA-EN: "Shifting Melds: Discarding a tile that is identical to the tile just melded … are forbidden."; W2: "上/碰牌之後不可打相同的牌" | differs (question 13) |
| 33 | §4.1.5 nobody claims: the next player draws | W1 | agrees |
| 34 | §4.1.6 a passed win locks that tile (pointer to §4.4) | see rows 36–42 | agrees |
| 35 | §4.1.7 the last discard can be claimed only for a win, otherwise a draw | HKMA 流局 1: "若牌牆上最後一張牌已被摸走，並且摸該牌者接着打出的牌無人食出，便作流局。" | agrees |

## 5. Passing a win 過水 (§4.4)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 36 | 1. A player offered a win (discard or robbed kong) who does not take it is locked on that tile kind | HKMA 簡則 15: "如果錯過了食糊的機會，在同巡內（即下一次摸牌前）不能食跟張牌" | agrees |
| 37 | 2. Locked: no win on that kind from any player, discard or robbed kong | W1: "在該巡內有另一名玩家打出同一張牌，也不可以喊食糊"; W2: "對家/上家再打出相同的牌不可胡" | agrees |
| 38 | 3. Only the passed kind is locked; two passes lock both | W1 gives a stricter variant (every tile of the same wait, except a tile worth more) and adds "但通常不作這種限制" | agrees |
| 39 | 4. A locked player may still pung, kong or chow it, and win on it by self-draw | HKMA 簡則 15: "但碰牌及上牌則沒有此限制（牽涉包牌責任的情況除外）". The exception concerns liability, which our §8.3 has only for the 12-tile case | agrees |
| 40 | 5. The lock ends when the player's next turn starts (draw or own claim) | HKMA: "即下一次摸牌前"; W2: "需要到自己再摸牌後才可解除過水"; W1 variant: "直至自己摸牌或上、碰、槓牌為止" | agrees |
| 41 | 6. With several offered, only those who do not declare are locked | follows from rows 36 and 30 | agrees |
| 42 | 7. Nobody else sees a pass | engine detail following §4.1.1; no source discusses it | no source found |

## 6. The discard record 摸打 / 入章 (§4.5)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 43 | 1. A discard of the tile just drawn (or one of its kind) is recorded 摸打 | none of the sources makes this a rule. Brian, 2026-11-20: no bot input may use it ("it's completely irrelevant, especially in this mobile app setting") | no source found |
| 44 | 2. Every other discard is 入章 | as row 43 | no source found |

## 7. Kongs and bonus tiles (§5)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 45 | §5.1 four in hand may be laid as a concealed kong, then a replacement from the back | HKMA 開槓 7 | agrees |
| 46 | §5.1 the concealed kong's tile is shown when declared | not stated by any source | no source found |
| 47 | §5.1 only a thirteen-orphans wait may rob a concealed kong | HKMA 搶槓 3: "只有加槓可以搶槓﹐暗槓不可以搶槓﹐十三么除外。"; W1: "但若其他玩家是暗槓，則只有十三幺才可以搶暗槓。" | agrees |
| 48 | §5.2 an exposed pung plus the fourth tile makes an added kong, then a replacement | HKMA 開槓 | agrees |
| 49 | §5.2 others may rob the added kong for a win before the replacement; the kong is cancelled and the hand ends | HKMA 1.9: "一般牌型只可以搶加槓，十三么才可以搶暗槓。" | agrees |
| 50 | §5.3 a drawn bonus tile is placed face up and replaced from the back; never discarded or claimed | W1 (row 19); W2: "必須立刻從牌墩的牌尾補牌" | agrees |
| 51 | §5.3 the seventh bonus tile wins at once | HKMA 花糊牌例 3: "要食花糊須在摸花後不補牌馬上宣告食糊，補花等同放棄花糊機會。" (replacing the flower passes the win); Y1: "可選擇唔食糊，補咗花後就唔可以食糊"; W2: "玩家於摸出花牌類糊的話可選擇放棄糊牌權利" | differs (question 9) |
| 52 | §5.3 another player's bonus tile never gives anyone a win | no source lists 七搶一 (only K, an English community wiki, as a "local" 8-faan item) | already decided (Brian, 2026-11-20) |
| 53 | §5.4 a win on a kong's replacement is a self-draw | HKMA 開槓 9: "「槓後補牌」自摸食糊﹐稱為「槓上自摸」，加一番。" | agrees |
| 54 | §5.4 if the kong's replacement is a bonus tile, the tile drawn for it still counts as the kong's replacement | HKMA 1.10: "槓後補牌馬上自摸，補花後不算。" and "簡單而言，摸花等於狀態重設。"; W2: "明/暗/加槓後自摸，補花後不算". E1 alone counts a replacement "due to a Kong or a Bonus Tile" | differs (question 5) |

## 8. Winning hands (§6)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 55 | Standard form: four melds and a pair | HKMA 食糊 | agrees |
| 56 | Thirteen orphans: one of each terminal and honour plus a second of one | HKMA: "特殊糊牌形：「十三么」" | agrees |
| 57 | No other forms (no seven pairs) | HKMA names only 十三么 as a special shape; W2 footnote: Hong Kong play "一般完全沒有七對子". E1 and E2 give seven pairs 4 faan in some variants | agrees |
| 58 | Minimum 3 faan (`min_faan`) | HKMA 基本計分原則 3: "三番起糊：手牌必須有三番以上才可以食糊。" | agrees |
| 59 | Bonus-tile faan (#3, #4, #5) do not count toward the minimum | HKMA lists 無花, 正花, 一台花 (1.2–1.4) as ordinary faan with no such exclusion; W2's 雞糊 entry reaches 3 with "手持一副三元牌刻子再加一張正花時自摸"; S: an all-chows, no-flower, self-drawn hand is "3 fan … if your group has already agreed … that having no flowers is worth 1 fan"; K: "Minimum 0-2, 3 with Flowers". Only E2 (old style) excludes them: "Bonus tiles and a few other elements are not included in the minimum faan value" | sources disagree (question 2) |
| 60 | Flower wins need no complete hand and no minimum | HKMA 3.4: "摸到七張花。" | agrees |
| 61 | Bonus-tile faan are added afterwards to the paid faan | HKMA 1.2–1.4 | agrees |
| 62 | When a hand splits several ways, the split with the most faan counts | W2 坎坎糊: "有幾大食幾大" (in passing); no rulebook states it | no source found |

## 9. Faan table (§7)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 63 | Faan add up; cap 10 (`faan_cap`); a limit hand scores the cap and nothing else | HKMA 基本計分原則 4: "以10番為上限，例牌一律算10番。"; W1: "最大以打八番或十番為上限（網絡遊戲較常見最大打十三番）" | agrees |
| 64 | #1 自摸 1; a replacement draw counts | HKMA 1.5: "食糊牌由自己摸牌所得。" | agrees |
| 65 | #2 門前清 1, concealed kongs allowed, with flowers in play too | HKMA 1.12: "門前清（「無花」規則限定)": "胡牌前沒有「吃」、「碰」、「槓」過任何牌。"; W2: "沒有上、碰、槓（包括暗槓）任何牌而食糊，打花牌一般不作計算。" Both: a concealed kong breaks it, and with flowers in play it is not counted | differs (question 3) |
| 66 | #3 無花 1 | HKMA 1.3: "沒有花牌。" | agrees |
| 67 | #4 正花 1 each | HKMA 1.2: "俗稱正花。…一張門風花牌算一番。" | agrees |
| 68 | #5 一台花 2, replacing that set's seat flower; both sets 4 | HKMA 1.4: 1 faan, "（另加計包含在內的門風花）" (1 + the seat flower = 2 a set); W2: 2, "不再計同系列的正花". The totals are the same | agrees |
| 69 | #6 海底撈月 1, self-drawn last tile; 自摸 also counts | HKMA 1.11: "自摸牌牆最後一張牌食糊。"; W2: "食全局最後一隻牌，若為自摸則另計再加一番。" (a win on the last discard scores it too); E1 and K also count the last discard | sources disagree (question 16) |
| 70 | #7 槓上開花 1, with 自摸 2 | HKMA 1.10 槓上自摸 1; W2: "外加自摸另計共兩番" | agrees |
| 71 | #8 槓上槓 8, replacing #7 and #1 | HKMA 6.10: "連續開兩個槓之後食糊。" among the 10-faan limit hands; W2 連槓開花 8, footnote: "不同規例中作二至八番甚至例牌計不等"; E1 8 + 自摸 | sources disagree (question 6) |
| 72 | #9 搶槓 1 (added kong, or concealed kong with thirteen orphans) | HKMA 1.9 | agrees |
| 73 | #10 三元牌 1 each | HKMA 1.6: "紅中、發財、白板的坎子" | agrees |
| 74 | #11 門風 1 | HKMA 1.7 | agrees |
| 75 | #12 圈風 1; both 2 for East in East | HKMA 1.7: "門風及圈風重疊時計兩番。" | agrees |
| 76 | #13 雞糊 0 | W2: "單純四搭一對、無任何其他組合的牌型" | agrees |
| 77 | #14 平糊 1, four chows and any pair; flowers do not matter | HKMA 1.1: "手牌有四個順子﹐沒有坎/槓。（以番子做眼或叫單扉均可算平糊。）" S (old style) asks for a valueless pair in two suits; W2 footnote calls the honour-pair ban Guangdong | agrees |
| 78 | #15 對對糊 3 | HKMA 2.2 | agrees |
| 79 | #16 混一色 3 | HKMA 2.1 | agrees |
| 80 | #17 清一色 7, replaces #16 | HKMA 5.1: "只有一種花式（筒、索或萬），沒有番子的糊牌。" | agrees |
| 81 | #18 混么九 1, always with #15 (4) | HKMA 1.8: "只有一、九、番子。加計對對糊。" | agrees |
| 82 | #19 小三元 5 including the two dragon pungs | HKMA 2.3: "另外加計三元牌，實為五番。" | agrees |
| 83 | #20 大三元 8 including the three dragon pungs | HKMA 4.1: "另外加計三元牌，實為八番。" | agrees |
| 84 | #21 小四喜 limit | HKMA 6.2 limit (10); W2 6 faan, footnote: "必然伴隨混一色。不可另計混一色…但可另計門圈風、對對糊。"; S: "Many Cantonese players award six fan for this." | sources disagree (question 7) |
| 85 | #22 大四喜 limit | HKMA 6.3 | agrees |
| 86 | #23 字一色 limit | HKMA 6.1 全番子: "手牌全由番子組成。" | agrees |
| 87 | #24 清么九 limit | HKMA 6.6 | agrees |
| 88 | #25 十三么 limit | HKMA 6.9 | agrees |
| 89 | #26 九子連環 limit: concealed, one suit, `1112345678999` waiting on any tile of the suit (the engine checks the 13 tiles before the win) | HKMA 6.11: "自摸集齊牌型就可以食，食出銃必定要叫九扉。" (on a self-draw the final shape is enough; only a discard win needs the nine-sided wait) | differs (question 15) |
| 90 | #27 坎坎糊 limit: four concealed pungs or kongs; a pung completed on the winning discard is exposed | HKMA 6.4 limit: "不可有槓，單吊可以食出銃，對碰必須自摸。"; W2 lists it under 8 faan ("沒有碰、槓過的對對糊"), its text adds "應該計10番/爆番". The exposed-on-a-discard part agrees; kongs are excluded in both | differs (question 4) |
| 91 | #28 十八羅漢 limit | HKMA 6.5: "手牌有四個槓。" | agrees |
| 92 | #29 天糊 limit: the dealer's hand is complete after the deal | HKMA 6.7: "東家配牌直接食糊，暗槓後不算天糊。" | agrees |
| 93 | #30 地糊 limit: a non-dealer wins on the dealer's first discard | HKMA 6.8: "閒家配牌的13張已叫糊﹐食出東家首張打出的牌。東家若開了暗槓﹐便不算數。" Our rule does not cancel it after the dealer's concealed kong. (Y1 swaps the names 地糊 and 人糊) | differs (question 20) |
| 94 | Patterns 29 and 30 need a complete hand | HKMA 6.7–6.8 | agrees |
| 95 | #31 大花糊 8 | HKMA 6.12 "摸到八張花。" among the 10-faan limit hands; W2: "摸齊八隻花可即時食糊，計八番自摸", footnote 4 to 8 or a limit; E1 13 | sources disagree (question 8) |
| 96 | #32 花糊 3 | HKMA 3.4 and 花糊牌例 1: "小花糊只算三番自摸，不加計自摸一番。"; W2: "計兩番加自摸一番" (3) | agrees |
| 97 | §7.5.1 the holder wins at once and may not pass | HKMA 花糊牌例 3 (row 51): replacing the flower gives the win up | differs (question 9) |
| 98 | §7.5.1 #31 is won only in the deal | follows from row 97's rule; with a pass allowed it can be won in play | differs (question 9) |
| 99 | §7.5.2 the hand's tiles add nothing; no other pattern is added | HKMA 花糊牌例 2: it can add 海底撈月, 槓上自摸 and 槓上槓自摸; W2: "可另計海底撈月、槓上開花、連槓開花、天地糊等" | differs (question 10) |
| 100 | §7.5.3 paid like a self-draw; no 12-tile liability | HKMA 花糊牌例 1: "三番自摸" | agrees |
| 101 | §7.5.4 no robbing flower win (七搶一 removed) | see row 52 | already decided (Brian, 2026-11-20) |

### Worked examples (§7.6)

| # | Our example | Under the sources | Mark |
|---|---|---|---|
| 102 | `123p 456p 789p RRR NN`, self-drawn, concealed, no flowers, South: 7 | HKMA: 門前清 counts only under the no-flower rule (row 65); with flowers in play it is 6 | differs (question 3) |
| 103 | `234m 567m 345p 678s 99s`, discard, claimed chow, F2 as South: 2, cannot win | 2 under every source; cannot win either way | agrees |
| 104 | `EEE 111m 999p GGG 11s`, East in East, discard, exposed `GGG`, no flowers: 8 | HKMA rows 1.3, 1.6, 1.7, 1.8, 2.2 give the same | agrees |
| 105 | `GGG WhWhWh 111s 22s` + exposed `RRR`, self-drawn: capped at 10 | same | agrees |

## 10. Payment (§8)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 106 | §8.1 points a share: 3 → 8, 4 → 16, 5 → 24, 6 → 32, 7 → 48, 8 → 64, 9 → 96, 10 → 128 | W2 table "分數計算（全銃制）" 出銃 column: 8, 16, 24, 32, 48, 64, 96, 128; HKMA 分數換算表 (discard): 32, 64, 96, 128, 192, 256, 384, 512 ("分數按100/底半辣上計算"), the same steps ×4 | agrees |
| 107 | §8.1 half-spicy: doubling up to 4 faan, then half steps | W2: "傳統麻雀以四番為滿糊，隨後以半辣上，每兩番才跳一倍。" | agrees |
| 108 | §8.1 variant `full_spicy` keeps doubling | W2: "賭得比較大的人亦會辣辣上，每一番跳一倍。" | agrees |
| 109 | §8.2 a discard win: the discarder pays **1 share**, the others nothing | HKMA 分數換算表: 3 faan "32" on a discard, "16 x 3" on a self-draw; HKMA-EN: "The discarding player pays for the whole hand for chucking."; W2 全銃 table: 3 faan 出銃 8, 自摸 12 "(4 x 3)"; E1: self-draw "points awarded are 1.5 times the table amount"; K: self-draw "x2" each, discard "x4". Every source: the discarder pays **twice** what each player pays on a self-draw | differs (question 1) |
| 110 | §8.2 robbing a kong: paid as a discard (1 share) | HKMA 搶槓 2: "搶槓視為開槓者出銃，加計一番，但支付分數時按「包自摸」計算" (the kong's maker pays the whole self-draw, 3 shares); W3: "跑馬仔被搶槓者亦同" (under 包自摸); W2: the kong's maker counts as the discarder | sources disagree (question 11) |
| 111 | §8.2 self-draw: each of the three pays 1 share | HKMA "16 x 3"; W2 "4 x 3" | agrees |
| 112 | §8.2 several winners: the discarder pays each winner | W2: "若有超過一人同時搶槓亦視同一炮多響、需包賠所有食糊玩家" (where several winners are allowed) | agrees (with question 12 on whether several win) |
| 113 | §8.2 flower win: each of the three pays 1 share, no liability | HKMA 花糊牌例 1 | agrees |
| 114 | §8.2 exhaustive draw: nobody pays | HKMA 流局 | agrees |
| 115 | Kongs carry no separate payment; zero-sum | no source gives a kong payment | agrees |
| 116 | §8.3.1 feeding a fourth exposed meld (12 tiles showing) makes the discarder liable | HKMA 得分授受法 1 (十二章包); W2: "玩家甲打出的一張牌令玩家乙翻開第四組共十二張牌（稱為十二張落地）" | agrees |
| 117 | §8.3.2 the liable player pays 3 shares on a self-draw only (Task 27) | HKMA: "包牌者一力承擔被包牌者自摸時三家合共的失分，其餘兩家免去失分的責任。"; HKMA-EN: "the discarding player pays for the whole hand when the melding player wins on self-drawn" | already decided (Brian, 2026-11-23, ADR 0004) |
| 118 | §8.3.1 concealed kongs do not count toward the four | HKMA: "若第四個明章組合為暗槓，則沒有人需要包。" | agrees |
| 119 | §8.3.3 the only liability: feeding the third dragon pung or fourth wind pung creates none | HKMA 簡則 14: "包牌規條包括十二章包（張張包）及大三元包，槓後自摸不用包。"; HKMA: "供給第三副三元牌的一家包付自摸分數"; HKMA: "包後不包先" (only the later liability holds); W2 and W3 also have 包大三元 (W3 and W2's 陪銃 version also 大四喜) | differs (question 14) |
| 120 | §8.3.3 (unstated) no liability for a kong-replacement win | HKMA 簡則 14: "槓後自摸不用包"; W2: "玩家甲打出一張生章牌…若果玩家乙因而槓上自摸，則玩家甲要包乙槓上自摸" (with the alternative of +1 faan and no liability); W3 also has it | sources disagree (question 14) |

## 11. Exhaustive draw and matches (§9, §10)

| # | Our rule | Sources | Mark |
|---|---|---|---|
| 121 | §9 a draw due with an empty wall ends the hand; nobody pays | HKMA 流局 1 | agrees |
| 122 | §9 no tiles held back | W1: "三番起糊、摸到尾（即不保留尾棟）這些規則" | agrees |
| 123 | §10 four rounds E S W N | Y2 | agrees |
| 124 | §10 the dealer keeps the deal after a dealer win | W1: "或者莊家自己勝出了，則要「連莊」"; HKMA-EN "unless in some specified circumstances (eg. dealer wins the game)" (HKMA 簡則 1 "不設「連莊」" is a tournament rule) | agrees |
| 125 | §10 the dealer keeps the deal after a draw | HKMA 流局 2: "流局時莊家不會連莊，即使是北四局亦如是。"; W1: "若沒有任何玩家得勝（即是流局）…則要「連莊」", adding "也有不少玩家在流局的時候會照樣過莊"; K: "If Non-East wins the Hand or if it's a Draw, the Seat Winds rotate" | sources disagree (question 18) |
| 126 | §10 dealer passes to the next seat otherwise | W1, Y2 | agrees |

## 12. Rules the sources have and our file lacks

Not counted above. Each is in question 19 unless named elsewhere.

| Rule | Sources | Note |
|---|---|---|
| 食換: no discarding the tile just claimed (row 32) | HKMA, HKMA-EN, W2 | question 13 |
| 包大三元 (row 119) | HKMA, W2, W3 | question 14 |
| 人糊: a non-dealer self-draw in the first go-round | W2: "閒家於開局第一輪即自摸", 13 faan, footnote "部份規例僅計三番。" Not in HKMA | question 19 |
| 12-tile liability lifted when the fed player threw the same tile first in the go-round | W2 (exception to 包十二張). Not in HKMA | question 19 |
| 九張包, 包生章, 五子包生 | W2 and E1, only in the 陪銃 (discarder pays half) system | not asked: our payment is the 全銃 kind |
| Penalties: 詐糊, 相公, yellow cards | HKMA, W2 | not asked: the engine allows only legal moves |

## 13. What each difference changes

Every question below is in `docs/inbox.md`. Each says what changes in play and in payment, with
a small example. Questions 1 to 3 are the big ones: they change the value of most hands.

1. **Discard win pays 1 share (ours) or 2 shares (every source).** Example: a 3-faan hand on a
   discard pays 8 now; under the sources 16 (self-draw stays 8 from each). Dealing in costs
   twice as much against a self-draw, so defence is worth more. Changes the engine, the oracle,
   every bot value, the value network's targets and every saved result.
2. **Flower faan toward the 3-faan minimum.** Example: `234m 567m 345p 678s 99s` with `234m`
   a claimed chow, self-drawn, no flowers: 平糊 1 + 自摸 1 + 無花 1 = 3. It cannot win now (2
   toward the minimum); it wins under W2, S and (by silence) HKMA. More hands become winnable, especially cheap ones.
3. **門前清 under flowers, and with a concealed kong.** Example: row 102 is 7 now, 6 under HKMA.
4. **坎坎糊 with a concealed kong.** Example: `2222m` (concealed kong) `555p 888s EEE 99p` all
   concealed, self-drawn, South seat in the East round, no flowers: a limit (10) now; under HKMA
   對對糊 3 + 圈風 1 + 自摸 1 + 無花 1 = 6 (no 門前清: the kong breaks it).
5. **槓上開花 after a flower replacement.** Example: kong, the replacement is F3, its replacement
   wins: 自摸 + 槓上開花 = 2 now; 自摸 only (1) under HKMA and W2.
6. **槓上槓: 8 (ours, W2) or a 10-faan limit (HKMA).** 64 against 128 points a share.
7. **小四喜: limit (ours, HKMA) or 6 plus 門風/圈風/對對糊 (W2, S).**
8. **大花糊: 8 (ours, W2) or a 10-faan limit (HKMA).**
9. **Passing a flower win.** Under HKMA a player with seven flowers may replace the seventh and
   play on (for a bigger hand or the eighth flower, 大花糊). Ours forces the 3-faan win.
10. **Flower win adding 海底 / 槓上.** Example: the seventh flower drawn as a kong replacement:
    3 now; 4 under HKMA.
11. **A robbed kong: 1 share (ours, W2) or 3 shares from the kong's maker (HKMA, W3).**
    Example: robbing for 5 faan costs the kong's maker 24 now, 72 under HKMA.
12. **Several winners on one discard (ours) or one (截糊, W2's default; E2 by faan).**
13. **食換 ban.** Example: chow `34p` + 5p, then throw 2p or 5p: allowed now, forbidden under
    HKMA. Changes the legal discards after a claim (a move-mask change; action codes stay).
14. **包大三元 and 包槓上開花.** Example: A feeds B's third dragon pung; B self-draws: now each
    pays 1 share; under HKMA A pays all 3. Kong-replacement liability: HKMA none (as ours).
15. **九子連環 on a self-draw from another 13.** Example: holding `1123456789999m`, self-draw
    1m: the hand is `11123456789999m`, a limit under HKMA; under ours 清一色 7 + 自摸 1 + 門前清 1
    (+ flower faan), since the 13 tiles before the win were not the nine-sided wait.
16. **海底 on the last discard.** Ours and HKMA: no faan; W2, E1, K: +1.
17. **Seven or eight flowers in the deal.** Ours wins at the deal; HKMA makes a non-dealer
    declare at its first draw; W2 (custom) calls it 天糊 / 地糊.
18. **Dealer on a draw (matches, not built).** Ours keeps the deal; HKMA passes it.
19. **Rules we lack:** 人糊, the 12-tile exception.
20. **地糊 after the dealer's concealed kong.** HKMA cancels it; ours does not.
