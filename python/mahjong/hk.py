"""Friendly helpers around the Hong Kong mahjong engine (``mahjong._core``).

Tiles are ``int`` indices (docs/rules/hk.md §1) and actions are ``int`` codes
(see :data:`ACTION_CODES`). This module names them, in English text notation and
in Chinese, and builds rule keyword arguments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mahjong._core import HkState, action_text, format_tiles, parse_tiles, pattern_names

__all__ = [
    "Rules",
    "HkState",
    "tile_text",
    "tile_chinese",
    "tiles_chinese",
    "action_chinese",
    "PATTERNS",
    "parse_tiles",
    "format_tiles",
    "action_text",
    "SELF_DRAWN_WIN",
    "PASS",
    "WIN",
    "PUNG",
    "CLAIM_KONG",
]

# Action codes (mahjong_hk::Action::code).
SELF_DRAWN_WIN, PASS, WIN, PUNG, CLAIM_KONG = 34, 103, 104, 105, 106
ACTION_CODES = 134

_NUMERALS = "一二三四五六七八九"
_HONOURS = ["東", "南", "西", "北", "白", "發", "中"]
_BONUS = ["梅", "蘭", "菊", "竹", "春", "夏", "秋", "冬"]

#: Faan-table rows: number -> (Chinese name, English name).
PATTERNS: dict[int, tuple[str, str]] = {n: (zh, en) for n, zh, en in pattern_names()}


@dataclass(frozen=True)
class Rules:
    """Rule variants; the defaults are docs/rules/hk.md's defaults."""

    flowers: bool = True
    min_faan: int = 3
    faan_cap: int = 10
    payout: str = "half_spicy"
    multiple_winners: str = "all"
    twelve_tile_liability: bool = True
    prevailing_wind: int = 0

    def kwargs(self) -> dict[str, object]:
        """Keyword arguments for ``mahjong._core`` functions."""
        return asdict(self)

    def new_deal(self, seed: int | None = None) -> HkState:
        """A freshly shuffled hand under these rules."""
        return HkState(seed, **self.kwargs())


def tile_text(t: int) -> str:
    """``5p``, ``Wh``, ``F2``."""
    return format_tiles([t])


def tile_chinese(t: int) -> str:
    """``五筒``, ``白``, ``蘭``."""
    if t < 27:
        return _NUMERALS[t % 9] + "萬筒索"[t // 9]
    if t < 34:
        return _HONOURS[t - 27]
    return _BONUS[t - 34]


def tiles_chinese(tiles: list[int]) -> str:
    """Chinese names, space-separated, in index order."""
    return " ".join(tile_chinese(t) for t in sorted(tiles))


def action_chinese(code: int) -> str:
    """The action in Chinese, e.g. ``打 五筒``, ``碰``, ``上 三四五索``."""
    if code < 34:
        return f"打 {tile_chinese(code)}"
    if code == SELF_DRAWN_WIN:
        return "自摸"
    if code < 69:
        return f"暗槓 {tile_chinese(code - 35)}"
    if code < 103:
        return f"加槓 {tile_chinese(code - 69)}"
    if code >= 107:
        low = code - 107
        return "上 " + "".join(_NUMERALS[(low + k) % 9] for k in range(3)) + "萬筒索"[low // 9]
    return {PASS: "過", WIN: "食糊", PUNG: "碰", CLAIM_KONG: "槓"}[code]


def describe_score(score: dict[str, Any]) -> str:
    """``7 faan: 混一色 3, 中 1, …`` from a score dict."""
    parts = []
    for n, f in score["patterns"]:
        zh = PATTERNS[n][0]
        parts.append(zh if score["limit"] else f"{zh} {f}")
    head = f"{score['faan']} faan" + (" (limit)" if score["limit"] else "")
    return head + ": " + ", ".join(parts)
