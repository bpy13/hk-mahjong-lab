# 0004 — The 12-tile liability counts only when the winner self-draws

**Status:** accepted (Brian, 2026-11-23)

## Decision
The 12-tile liability (十二張落地) applies only when the player with four open sets wins by
self-draw: the player who fed the fourth open set then pays all 3 shares. On a win by discard the
discarder pays 1 share as usual and the feeder pays nothing. The draft rule of 17 Oct
(`docs/rules/hk.md` section 8.3 point 2, no source cited) charged the feeder 3 shares on a
discard win too.

Sources read by the orchestrator on 2026-11-23, all giving the rule as 包自摸: zh.wikipedia
香港麻雀胡牌列表 and 包牌; zh-yue.wikipedia 香港蔴雀食糊牌型; en.wikipedia Hong Kong mahjong
scoring rules.

Plan Task 27 changes the rules file, the engine and the oracle and replays the two matches
measured under the old rule (Tasks 23 and 24). Plan Task 28 checks the rest of the rules file
line by line against Hong Kong sources; each difference goes to Brian as a question, nothing
else changes without his word.

## Brian's words
- On deal 99000316, where the bot paid 96 after another seat dealt in: "what???? its a deal in,
  player didn't lose anything"
- "what do you think the answer is? do some web search and come back to me with an answer"
- "how could this happen?"
- "pleaswe" (to: "Shall I have the whole rules file checked line by line against Hong Kong
  sources and bring you each doubt: yes, or fix this rule only?")

## Not decided
The variant some tables play, where the liability holds only for named hands (清一色 and the
like). Whether results and networks made under the old rule are redone: Task 27 first counts how
often the old rule moved points.
