"""``HkState.from_position``: build a mid-hand position from Python and analyse it (plan Task 2)."""

from typing import Any

import pytest
from mahjong._core import HkState, analyze, danger, parse_tiles


def tile(text: str) -> int:
    """One tile index from its text (``"5s"``)."""
    (t,) = parse_tiles(text)
    return t


def position(**overrides: object) -> dict[str, Any]:
    """Seat 0 discarded 9p, seat 1 punged it and discarded W, seat 2 drew 5s.

    49 concealed tiles, the pung and the W leave 82 tiles in the wall (no flowers).
    """
    args: dict[str, Any] = {
        "hands": [
            parse_tiles("123m 456m 789m 1p 23s E"),
            parse_tiles("234p 567p 345s N"),
            parse_tiles("123s 456s 789s RR 66m 5s"),
            parse_tiles("111m 22m 33m 44m 55m WhWh"),
        ],
        "melds": [[], [("pung", tile("9p"), False, 0)], [], []],
        "river": [(0, tile("9p"), True), (1, tile("W"), False)],
        "wall_remaining": 82,
        "to_move": 2,
        "last_drawn": tile("5s"),
        "flowers": False,
    }
    args.update(overrides)
    return args


def test_build_and_analyse() -> None:
    s = HkState.from_position(**position())
    assert s.current_player() == 2
    assert s.phase() == "turn"
    obs = s.observation(2)
    assert obs["wall_remaining"] == 82
    assert obs["river"] == [(0, tile("9p"), True), (1, tile("W"), False)]
    assert tile("5s") in s.legal_actions()

    a = analyze(s, samples=16, candidates=3, seed=1)
    assert a["seat"] == 2
    assert a["samples"] == 16
    assert a["choice"] in s.legal_actions()
    d = danger(s, 2)
    assert d is not None

    # The rebuilt state plays to the end once its wall is re-dealt.
    t = s.determinize(2, seed=3)
    while not t.is_terminal():
        t.apply(t.legal_actions()[-1])
    assert t.result() is not None


def test_claim_window() -> None:
    s = HkState.from_position(
        **position(
            hands=[
                parse_tiles("123m 456m 789m RR 3s E"),
                parse_tiles("234p 567p 345s N"),
                parse_tiles("123s 456s 789s R 66m 5s"),
                parse_tiles("111m 22m 33m 44m 55m WhWh"),
            ],
            river=[(0, tile("9p"), True), (1, tile("W"), False), (2, tile("R"), False)],
            to_move=0,
            phase="claim",
            claim_tile=tile("R"),
            claim_from=2,
            last_drawn=None,
        )
    )
    assert s.phase() == "claim"
    assert s.current_player() == 0
    assert sorted(s.legal_actions()) == [103, 105]  # pass, pung


@pytest.mark.parametrize(
    "overrides",
    [
        {"wall_remaining": 83},
        {"to_move": 4},
        {"phase": "claim"},
        {"hands": [[], [], [], []]},
        {"melds": [[], [("pair", tile("9p"), False, 0)], [], []]},
        {"river": [(0, tile("9p"), False), (1, tile("W"), False)]},
    ],
)
def test_malformed_positions_raise(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        HkState.from_position(**position(**overrides))
