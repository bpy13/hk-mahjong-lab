"""Duplicate evaluation of exported networks with the Rust tournament runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mahjong._core import tournament


def evaluate(spec: str, deals: int = 100, seed: int = 99, field: str = "value") -> dict[str, Any]:
    """``spec`` against three ``field`` bots; returns the candidate's row."""
    res = tournament([spec, field, field, field], deals, seed)
    a = res["agents"][0]
    n = a["hands"]
    return {
        "agent": spec,
        "mean": a["mean"],
        "ci95": a["ci95"],
        "win": a["wins"] / n,
        "deal_in": a["deal_ins"] / n,
        "avg_faan": a["win_faan"] / max(1, a["wins"]),
    }


def net_spec(path: Path) -> str:
    return f"net:path={path}"
