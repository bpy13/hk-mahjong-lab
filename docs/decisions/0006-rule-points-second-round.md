# 0006 — Rule points settled from the line-by-line check (second round)

**Status:** accepted (Brian, 2026-11-23)

## Decision
Further points of plan Task 28's 20 (ADR 0005 holds the first seven):

| Point | Rule | Decision | Change to the rules file and engine |
|---|---|---|---|
| 5 | 槓上開花 when the kong's replacement is a flower | keep: it counts | none |
| 7 | 小四喜 (#21) | keep: a limit hand, 10 faan | none |
| 9 | Passing a seven-flower win | allowed: the player may replace the seventh flower, give up that win and play on | yes (one more choice in the action space) |
| 10 | A flower win that also has 海底撈月, 槓上自摸 or 槓上槓自摸 | they add | yes |
| 13 | 食換 | forbidden: after a claim, not the claimed tile, nor the tile that would re-form the chow | yes (legal-move mask) |
| 15 | 九子連環 on a self-draw | the final shape is enough | yes |
| 12, 16, 17 | several winners on one discard; 海底 on the last discard; seven or eight flowers in the deal | no change (closed by the orchestrator: Brian, "I mean why ask me if no change?") | none |

Plan Task 32 builds points 9, 10, 13 and 15. The 24-hour self-play run (Task 29) waits for it,
because points 9 and 13 change which moves exist.

## Brian's words (2026-11-23)
- "食換 - yes"
- "包大三元 - I don't think it's a thing in Hong Kong mahjong?"
- "5. counts"
- "7. what is a limit hand? 小四喜 should be 10 faan if that's what you meant"
- "9. allowed"
- "10. yes they add"
- "15. final shape enough"
- "18. what dealer?"
- "19. 人糊, and the same-tile exception to the 12-tile rule. what?"
- "20. 地糊 after the dealer's 暗槓. what?"
- "I mean why ask me if no change?"

## Not decided
Point 14 (包大三元: Brian doubts it is a Hong Kong rule; the orchestrator showed him the
Association's text), and points 18, 19 and 20, which he asked to have explained. Nothing changes
for them without his word.
