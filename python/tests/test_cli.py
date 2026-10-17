"""CLI smoke tests."""

from mahjong.cli.main import app, parse_command
from typer.testing import CliRunner

runner = CliRunner()


def test_tournament_runs() -> None:
    r = runner.invoke(
        app, ["hk", "tournament", "value", "efficiency", "random", "fast", "--deals", "3"]
    )
    assert r.exit_code == 0, r.output
    assert "value" in r.output


def test_analyze_runs() -> None:
    r = runner.invoke(app, ["hk", "analyze", "--seed", "3", "--steps", "5", "--samples", "16"])
    assert r.exit_code == 0, r.output
    assert "solver" in r.output


def test_parse_command() -> None:
    assert parse_command("5p", [13, 2]) == 13
    assert parse_command("pass", [104, 103]) == 103
    assert parse_command("win", [104, 103]) == 104
    assert parse_command("win", [34, 1]) == 34
    assert parse_command("chow from 3s", [127, 103]) == 127
    assert parse_command("zz", [1]) is None


def test_patterns_listed() -> None:
    r = runner.invoke(app, ["hk", "patterns"])
    assert "清一色" in r.output
