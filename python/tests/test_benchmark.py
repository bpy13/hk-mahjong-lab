"""Brian's benchmark (plan Task 12): fixed walls, the API, replay, label export.

`value` plays 20 hands in Brian's seat through the API, so the replay of `value`
must reproduce every hand exactly (margin 0), while `defensive` gives a margin
with an interval and disagreements to label.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mahjong import benchmark
from mahjong._core import HkState, agent_action
from mahjong.cli.main import app as cli
from mahjong.server.app import app, store
from typer.testing import CliRunner

client = TestClient(app)
HANDS = 20


def play_benchmark_hand(agent: str) -> dict[str, Any]:
    """Starts the next benchmark hand and plays Brian's seat with ``agent``,
    seeded exactly as the replay seeds it."""
    g: dict[str, Any] = client.post("/api/benchmark/next").json()
    for _ in range(2000):
        if g["result"] is not None:
            return g
        if g["bot_to_act"]:
            g = client.post(f"/api/games/{g['id']}/step").json()
            continue
        s = store.get(g["id"])
        seed = benchmark.decision_seed(g["benchmark_index"], s.actions)
        a = agent_action(s.snapshot(), agent, seed)
        r = client.post(f"/api/games/{g['id']}/act", json={"action": a})
        assert r.status_code == 200, r.text
        g = r.json()
    raise AssertionError("hand did not finish")


@pytest.fixture(scope="module")
def hands_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """20 hands played by `value` through the API, logged in a temporary folder
    (never Brian's runs/benchmark/hands)."""
    d = tmp_path_factory.mktemp("hands")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MAHJONG_BENCHMARK_HANDS_DIR", str(d))
        for i in range(HANDS):
            g = play_benchmark_hand("value")
            assert g["benchmark_index"] == i
        yield d


def test_committed_walls_match_the_generator() -> None:
    walls = benchmark.load_walls()
    assert walls == json.loads(json.dumps(benchmark.generate_walls()))
    assert len(walls["walls"]) == benchmark.TOTAL == 400
    assert walls["seed"] == 2026
    for w in walls["walls"][:8]:
        assert w["seat"] == w["index"] % 4
        HkState.from_wall(w["wall"], **walls["rules"])  # a full tile set
    assert benchmark.load_config()["opponents"] == "value"


def test_decision_seed_depends_only_on_wall_and_history() -> None:
    a = benchmark.decision_seed(3, [(0, 5), (1, 103)])
    assert a == benchmark.decision_seed(3, [[0, 5], [1, 103]])
    assert a != benchmark.decision_seed(4, [(0, 5), (1, 103)])
    assert a != benchmark.decision_seed(3, [(0, 5)])
    assert 0 <= a < 2**63


def test_hands_are_logged_with_the_tag(hands_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAHJONG_BENCHMARK_HANDS_DIR", str(hands_dir))
    logs = benchmark.finished_hands()
    assert [h["index"] for h in logs] == list(range(HANDS))
    for h in logs:
        assert h["tag"] == "benchmark"
        assert h["seat"] == h["index"] % 4
        assert h["opponents"] == "value"
        assert sum(h["result"]["deltas"]) == 0
    p = client.get("/api/benchmark/progress").json()
    assert p == {
        "played": HANDS,
        "total": 400,
        "next": HANDS,
        "in_progress": None,
        "opponents": "value",
    }


def test_no_hints_or_review_and_resume(
    hands_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A copy of the 20 logs, so this test's started hand stays out of the others.
    for f in hands_dir.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    monkeypatch.setenv("MAHJONG_BENCHMARK_HANDS_DIR", str(tmp_path))
    g = client.post("/api/benchmark/next").json()
    assert g["tag"] == "benchmark" and g["benchmark_index"] == HANDS
    assert g["human"] == HANDS % 4 and g["opponents"] == "value"
    assert client.get(f"/api/games/{g['id']}/hints").status_code == 403
    assert client.get(f"/api/games/{g['id']}/review").status_code == 403
    # A second click returns the same hand.
    assert client.post("/api/benchmark/next").json()["id"] == g["id"]
    for _ in range(6):
        g = client.post(f"/api/games/{g['id']}/step").json() if g["bot_to_act"] else g
        if not g["bot_to_act"]:
            g = client.post(
                f"/api/games/{g['id']}/act", json={"action": g["legal"][0]["action"]}
            ).json()
    moves = len(store.get(g["id"]).actions)
    assert benchmark.progress()["in_progress"] == HANDS
    # After a server restart (the session gone), the hand resumes from its log.
    store._sessions.clear()
    again = client.post("/api/benchmark/next").json()
    assert again["id"] == g["id"] and again["benchmark_index"] == HANDS
    assert len(store.get(again["id"]).actions) == moves
    assert again["observation"] == g["observation"]


def test_replay_value_reproduces_every_hand(
    hands_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAHJONG_BENCHMARK_HANDS_DIR", str(hands_dir))
    res = benchmark.replay("value")
    assert len(res["hands"]) == HANDS
    assert all(h["margin"] == 0 and h["diverged_at"] is None for h in res["hands"])
    assert res["margin"]["mean"] == 0 and res["margin"]["ci95"] == 0


def test_replay_defensive_prints_a_margin_with_an_interval(
    hands_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MAHJONG_BENCHMARK_HANDS_DIR", str(hands_dir))
    out = tmp_path / "replay-defensive.json"
    r = CliRunner().invoke(
        cli, ["hk", "benchmark", "replay", "--agent", "defensive", "--out", str(out)]
    )
    assert r.exit_code == 0, r.output
    assert "margin" in r.output and "±" in r.output
    res = json.loads(out.read_text("utf-8"))
    assert res["margin"]["n"] == HANDS
    assert math.isfinite(res["margin"]["ci95"]) and res["margin"]["ci95"] > 0
    # Bots meet the same moves until the agent first differs from Brian.
    assert any(h["diverged_at"] is not None for h in res["hands"])


def test_labels_export_disagreements(
    hands_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MAHJONG_BENCHMARK_HANDS_DIR", str(hands_dir))
    assert benchmark.labels("value")["positions"] == []  # value agrees with itself
    out = tmp_path / "labels.json"
    r = CliRunner().invoke(
        cli, ["hk", "benchmark", "labels", "--agent", "defensive", "--out", str(out)]
    )
    assert r.exit_code == 0, r.output
    res = json.loads(out.read_text("utf-8"))
    assert res["hands"] == HANDS and res["positions"]
    for item in res["positions"]:
        assert item["human"] != item["agent_move"]
        # The snapshot rebuilds a position where both moves are legal.
        state = benchmark.position_from_json(item["position"], item["rules"])
        assert state.current_player() == item["seat"]
        legal = state.legal_actions()
        assert item["human"] in legal and item["agent_move"] in legal
