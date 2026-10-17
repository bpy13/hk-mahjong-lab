"""The game API: a full hand against bots, hints, review."""

from fastapi.testclient import TestClient
from mahjong.server.app import app

client = TestClient(app)


def play_to_end(gid: str, max_steps: int = 2000) -> dict:
    g = client.get(f"/api/games/{gid}").json()
    for _ in range(max_steps):
        if g["result"] is not None:
            return g
        if g["bot_to_act"]:
            g = client.post(f"/api/games/{gid}/step").json()
        else:
            # The human always takes the first legal option (wins come first).
            g = client.post(
                f"/api/games/{gid}/act", json={"action": g["legal"][0]["action"]}
            ).json()
    raise AssertionError("hand did not finish")


def test_full_hand_and_review() -> None:
    r = client.post("/api/games", json={"level": "easy", "seat": 2, "seed": 4})
    assert r.status_code == 200
    g = r.json()
    assert g["human"] == 2
    assert len(g["observation"]["hand"]) in (13, 14)
    end = play_to_end(g["id"])
    assert sum(end["result"]["deltas"]) == 0
    assert len(end["hands"]) == 4
    rev = client.get(f"/api/games/{g['id']}/review").json()
    assert isinstance(rev["decisions"], list)


def test_hints_on_the_humans_turn() -> None:
    g = client.post("/api/games", json={"level": "easy", "seat": 0, "seed": 1}).json()
    assert not g["bot_to_act"]  # the dealer moves first
    h = client.get(f"/api/games/{g['id']}/hints", params={"ms": 200, "samples": 40}).json()
    assert 1 <= len(h["solver"]["moves"]) <= 7
    assert len(h["danger"]["tiles"]) == 34
    assert "discards" in h and h["discards"][0]["solo"] is not None


def test_errors() -> None:
    assert client.post("/api/games", json={"level": "nope"}).status_code in (400, 422)
    g = client.post("/api/games", json={"level": "easy", "seat": 0, "seed": 1}).json()
    assert client.post(f"/api/games/{g['id']}/act", json={"action": 133}).status_code == 400
    assert client.post(f"/api/games/{g['id']}/step").status_code == 400
    assert client.get("/api/games/missing").status_code == 404


def test_meta() -> None:
    m = client.get("/api/meta").json()
    assert m["tiles"][31] == "白" and m["patterns"]["17"] == "清一色"


def test_moving_while_hints_compute() -> None:
    """A move submitted while the solver is still thinking must not fail
    (requests run in a thread pool; regression for an 'Already borrowed' error)."""
    import threading

    g = client.post("/api/games", json={"level": "easy", "seat": 0, "seed": 5}).json()
    errors: list[int] = []

    def ask() -> None:
        r = client.get(f"/api/games/{g['id']}/hints", params={"ms": 800, "samples": 4000})
        if r.status_code not in (200, 400):
            errors.append(r.status_code)

    t = threading.Thread(target=ask)
    t.start()
    move = client.post(f"/api/games/{g['id']}/act", json={"action": g["legal"][-1]["action"]})
    t.join()
    assert move.status_code == 200
    assert not errors
