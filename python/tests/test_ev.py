"""The ``ev`` bot through pybind (plan Task 10): ``ev_terms``, the agent spec and a tiny match.

The belief model is the heuristic fallback here (no stub from Python), so these tests check
shapes and invariants; the constructed positions with a stubbed belief are in
``crates/search/tests/ev_bot.rs``.
"""

import re
from typing import Any

import pytest
from mahjong._core import HkState, agent_action, ev_terms, parse_tiles, tournament

KEY = re.compile(r"^reason\.[a-z_.]+$")


def position() -> HkState:
    """Seat 0 discarded 9p, seat 1 punged it and discarded W, seat 2 drew 5s (no flowers)."""
    args: dict[str, Any] = {
        "hands": [
            parse_tiles("123m 456m 789m 1p 23s E"),
            parse_tiles("234p 567p 345s N"),
            parse_tiles("123s 456s 789s RR 66m 5s"),
            parse_tiles("111m 22m 33m 44m 55m WhWh"),
        ],
        "melds": [[], [("pung", parse_tiles("9p")[0], False, 0)], [], []],
        "river": [(0, parse_tiles("9p")[0], True), (1, parse_tiles("W")[0], False)],
        "wall_remaining": 82,
        "to_move": 2,
        "last_drawn": parse_tiles("5s")[0],
        "flowers": False,
    }
    return HkState.from_position(**args)


def test_ev_terms_lists_every_candidate_best_first() -> None:
    s = position()
    out = ev_terms(s)
    assert out["choice"] in s.legal_actions()
    cands = out["candidates"]
    assert cands, "a turn decision has candidates"
    evs = [c["ev"] for c in cands]
    assert evs == sorted(evs, reverse=True)
    for c in cands:
        assert c["ev"] == pytest.approx(c["offence"] - c["defence"])
        assert 0.0 <= c["cont"] <= 1.0
        for name in ("p_now", "deal_in_now", "self_draw", "other_feeds", "tenpai", "sd", "pay"):
            assert len(c[name]) == 4
            assert c[name][2] == 0.0, f"{name}: the acting seat's own entry is 0"
        assert len(c["fold_plan"]) <= 8
    keys = [r["key"] for r in out["reasons"]]
    assert "reason.ev.margin" in keys
    assert all(KEY.match(k) for k in keys)
    for r in out["reasons"]:
        if "seat" in r["args"]:
            assert isinstance(r["args"]["seat"], int)


def test_spec_options_and_errors() -> None:
    s = position()
    assert agent_action(s, "ev:horizon=4,tie_margin=1,floor=none", seed=1) in s.legal_actions()
    with pytest.raises(ValueError, match="unknown ev option"):
        ev_terms(s, "ev:nope=1")
    with pytest.raises(ValueError, match="not found"):
        ev_terms(s, "ev:belief=models/no-such-belief.mjnn")
    with pytest.raises(ValueError, match="ev spec"):
        ev_terms(s, "value")


def test_a_tiny_match_runs() -> None:
    report = tournament(["ev", "value", "value", "value"], deals=2, seed=3)
    names = [a["agent"] for a in report["agents"]]
    assert names[0] == "ev"
    assert sum(a["mean"] for a in report["agents"]) == pytest.approx(0.0, abs=1e-9)
