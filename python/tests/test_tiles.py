"""The Python view of tiles matches docs/rules/hk.md §1."""

import pytest
from mahjong._core import format_tiles, parse_tiles
from mahjong.cli.main import app
from typer.testing import CliRunner


def test_index_layout() -> None:
    assert parse_tiles("1m 9m 1p 1s 9s E N Wh R F1 S4") == [0, 8, 9, 18, 26, 27, 30, 31, 33, 34, 41]


def test_round_trip() -> None:
    assert format_tiles(parse_tiles("R 9m F2 1m E 5m R")) == "159m E RR F2"


def test_errors() -> None:
    with pytest.raises(ValueError):
        parse_tiles("12")
    with pytest.raises(ValueError):
        format_tiles([42])


def test_cli_tiles() -> None:
    result = CliRunner().invoke(app, ["tiles", "WhWh 321p"])
    assert result.exit_code == 0
    assert result.output.strip() == "123p WhWh"
