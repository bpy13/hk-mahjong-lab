# 0007 — Rule points settled from the line-by-line check (last four)

**Status:** accepted (Brian, 2026-11-23)

## Decision
The last open points of plan Task 28's 20 (ADR 0005 and 0006 hold the others). All 20 are now settled.

| Point | Rule | Decision | Change to the rules file and engine |
|---|---|---|---|
| 14 | 包大三元 | added: whoever feeds the third dragon pung pays the whole of that player's self-draw | yes |
| 18 | The dealer after an exhaustive draw | the deal passes on | rules file now; the engine when matches are built |
| 19 | 人糊 | added: a non-dealer self-draws a win in the first go-round | yes |
| 19 | The 12-tile exception | added, for a pung only: the feeder of the fourth open set is not liable when that set is a pung and the fed player had discarded the same tile earlier | yes |
| 20 | 地糊 after the dealer's concealed kong | keep: it still counts | none |

Orchestrator's calls where his words leave a detail open (Brian may change them): 人糊 is a limit
hand, 10 faan, like 天糊 and 地糊 (zh.wikipedia gives 13 faan, "some rules 3"); the other details
of 包大三元 follow the Association's text quoted in `docs/study-results/rules-check.md` row 119
(the later liability replaces the earlier; no liability for a win on a kong's replacement tile).

## Brian's words (2026-11-23)
- "18. just pass on"
- "19. include 人糊, and 12-tile exception only for pung"
- "20. 地糊 still counts after 暗槓 (暗槓 cannot 搶)"
- "14. include 包大三元"
