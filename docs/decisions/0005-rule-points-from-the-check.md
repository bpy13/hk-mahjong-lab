# 0005 — Rule points settled from the line-by-line check (first seven)

**Status:** accepted (Brian, 2026-11-23)

## Decision
From the 20 points of plan Task 28 (`docs/study-results/rules-check.md`, inbox question of
2026-11-23), Brian settled these on 2026-11-23:

| Point | Rule | Decision | Change to the rules file and engine |
|---|---|---|---|
| 1 | Win on a discard (section 8.2) | the discarder pays 2 shares; a self-draw stays 1 share from each of three | yes |
| 2 | Flower faan toward the 3-faan minimum (section 6) | keep: they do not count | none |
| 3 | 門前清 (#2) | keep: it counts with flowers in play, and a concealed kong does not break it | none |
| 4 | 坎坎糊 (#27) | keep: a concealed kong is allowed; an exposed kong is not | none (a test proves the exposed-kong case) |
| 6 | 槓上槓 (#8) | 10 faan (limit) | yes |
| 8 | 大花糊 (#31) | 10 faan (limit) | yes |
| 11 | A robbed kong (section 8.2) | the kong's maker pays the whole self-draw, 3 shares | yes |

Points 2, 3 and 4 go against the Association's rulebook on purpose: this is the table Brian
wants players to learn on.

Plan Task 31 changes the rules file, the engine and the oracle for points 1, 6, 8 and 11. The
24-hour self-play run of Task 29 waits for it, because its reward follows the payment rule.

## Brian's words (2026-11-23, in reply to the card's short list)
- "1. 出銃 pays double is correct"
- "2. we should refuse it"
- "3. that's why we want to refuse it, we want people to learn both the flower and 門前清. Although 暗槓 should not break it"
- "* 槓上槓 and 大花糊 become 10 番. Correct"
- "* 搶槓 is paid as a full 自摸 by the player who made the kong. Correct"
- "* what is 包大三元 and 食換?"
- "* 坎坎糊 is not allowed with a kong. 暗槓 allowed."
- "anything else? I think you said there are 20"

The orchestrator's reading of them, where the words are short: "we should refuse it" (2) = a
hand that reaches 3 faan only with flower faan is not a win, as today; "that's why we want to
refuse it" (3) = 門前清 keeps counting when flowers are in play.

## Not decided
Points 5, 7, 9, 10 and 12 to 20 (the orchestrator listed them to Brian again on 2026-11-23 with
their defaults; 13 食換 and 14 包大三元 explained at his request). Nothing changes for them
without his word.
