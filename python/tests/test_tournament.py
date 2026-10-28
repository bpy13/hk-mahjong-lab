"""The 24-seating duplicate harness through Python and the CLI (plan Task 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mahjong._core import tournament
from mahjong.cli.main import app
from typer.testing import CliRunner

AGENT_KEYS = {
    "agent",
    "hands",
    "mean",
    "ci95",
    "boot_lo",
    "boot_hi",
    "wins",
    "win_rate",
    "self_drawn",
    "self_drawn_rate",
    "deal_ins",
    "deal_in_rate",
    "draws",
    "draw_rate",
    "win_faan",
    "mean_faan",
    "liability_points",
    "deal_ins_to",
    "pattern_wins",
    "pattern_shares",
}


def test_24_seatings_report_shape() -> None:
    res = tournament(["fast", "efficiency", "random", "fast"], 3, seed=1, seatings=24)
    assert res["format"] == "hk-tournament/1"
    assert res["seatings"] == 24 and res["deals"] == 3 and res["hands"] == 72
    assert len(res["agents"]) == 4 and len(res["pairs"]) == 6
    for a in res["agents"]:
        assert set(a) >= AGENT_KEYS
        assert a["hands"] == 72
        assert a["boot_lo"] <= a["mean"] <= a["boot_hi"]
        assert sum(a["deal_ins_to"]) == a["deal_ins"]
        if a["wins"]:
            assert sum(a["pattern_wins"].values()) >= a["wins"]
            assert all(0 < s <= 1 for s in a["pattern_shares"].values())
    # Zero-sum: the agents' points cancel.
    assert sum(a["mean"] for a in res["agents"]) == pytest.approx(0, abs=1e-9)
    for p in res["pairs"]:
        a, b = res["agents"][p["a"]], res["agents"][p["b"]]
        assert p["mean"] == pytest.approx(a["mean"] - b["mean"], abs=1e-9)


def test_seatings_must_be_4_or_24() -> None:
    with pytest.raises(ValueError):
        tournament(["fast"] * 4, 1, seatings=6)


def test_cli_writes_json_and_prints_a_table(tmp_path: Path) -> None:
    out = tmp_path / "m" / "smoke.json"
    r = CliRunner().invoke(
        app,
        [
            "hk",
            "tournament",
            "value",
            "defensive",
            "efficiency",
            "random",
            "--deals",
            "2",
            "--seatings",
            "24",
            "--out",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "wall bootstrap" in r.output and "paired differences" in r.output
    data = json.loads(out.read_text("utf-8"))
    assert data["seatings"] == 24 and data["hands"] == 48
    assert data["rules"]["min_faan"] == 3
    assert [a["agent"] for a in data["agents"]] == ["value", "defensive", "efficiency", "random"]
