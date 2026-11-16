"""Self-play against a pool of opponents (plan 2026-11-16 Task 5)."""

from __future__ import annotations

from typing import Any

import numpy as np
from mahjong._core import HkState, HkVecEnv, encode_history
from mahjong.rl.cnn import HISTORY_LEN


def test_observe_history_matches_encode_history_of_each_position() -> None:
    # Random play in four games (Python plays every seat); each finished hand is
    # replayed from its seed and every observed history row must equal
    # `encode_history` of the same position.
    env = HkVecEnv(4, 11, min_faan=0)
    rng = np.random.default_rng(5)
    rows: dict[int, list[tuple[np.ndarray, int]]] = {i: [] for i in range(4)}
    checked = 0
    while checked < 8:
        _fb, _ob, mb, ids, _seats = env.observe()
        n = len(ids)
        h = np.frombuffer(env.observe_history(), dtype=np.float32).reshape(n, HISTORY_LEN)
        m = np.frombuffer(mb, dtype=np.uint8).reshape(n, 134)
        actions = [int(rng.choice(np.flatnonzero(m[k]))) for k in range(n)]
        for k, i in enumerate(ids):
            rows[i].append((h[k].copy(), actions[k]))
        env.step(actions)
        for d in env.finished(detail=True):
            s = HkState(seed=d["seed"], min_faan=0)
            for hist, a in rows[d["env"]]:
                assert np.array_equal(hist, np.asarray(encode_history(s), dtype=np.float32))
                s.apply(a)
            assert s.is_terminal()
            rows[d["env"]] = []
            checked += 1


def test_draw_seats_shares() -> None:
    from mahjong.rl.pool import CURRENT, OLDER, STAGE, draw_seats

    rng = np.random.default_rng(0)
    # 20,000 seats drawn as in play: three opponents per hand, each on its own.
    kinds, member = draw_seats(rng, 20_000, older=3)
    share = np.bincount(kinds, minlength=3) / len(kinds)
    assert abs(share[CURRENT] - 0.5) < 0.02
    assert abs(share[OLDER] - 0.3) < 0.02
    assert abs(share[STAGE] - 0.2) < 0.02
    assert set(member[kinds == OLDER].tolist()) == {0, 1, 2}
    assert (member[kinds != OLDER] == -1).all()
    # Every older member about equally often.
    per = np.bincount(member[kinds == OLDER], minlength=3) / (kinds == OLDER).sum()
    assert np.abs(per - 1 / 3).max() < 0.03
    # No older network yet: its share goes to the current network.
    kinds, member = draw_seats(rng, 20_000, older=0)
    share = np.bincount(kinds, minlength=3) / len(kinds)
    assert share[OLDER] == 0 and (member == -1).all()
    assert abs(share[CURRENT] - 0.8) < 0.02 and abs(share[STAGE] - 0.2) < 0.02
    # Seats of one hand are independent: mixed tables occur.
    kinds, _ = draw_seats(rng, 3 * 2000, older=2)
    tables = kinds.reshape(-1, 3)
    mixed = (tables.min(1) != tables.max(1)).mean()
    assert abs(mixed - (1 - 0.5**3 - 0.3**3 - 0.2**3)) < 0.03


def test_stop_reason() -> None:
    from mahjong.rl.pool import stop_reason

    def r(
        mean: float, ci: float = 0.5, draws: float = 0.06, train: float | None = 0.08
    ) -> dict[str, Any]:
        return {"mean": mean, "ci95": ci, "draw_rate": draws, "train_draw_rate": train}

    assert stop_reason([]) is None
    assert stop_reason([r(-1.0)]) is None
    # Draws above 15 %, at the stage table or in self-play.
    assert "draws" in (stop_reason([r(-1.0, draws=0.16)]) or "")
    assert "self-play" in (stop_reason([r(-1.0, train=0.2)]) or "")
    assert stop_reason([r(-1.0, draws=0.15, train=None)]) is None
    # Four results in a row without a gain beyond the best's range: stop.
    flat = [r(-1.0), r(-0.6), r(-0.9), r(-0.7)]
    assert stop_reason(flat) is None  # three without a gain
    assert "stalled" in (stop_reason([*flat, r(-0.55)]) or "")
    # A gain (mean above best + ci95) resets the count, and the new best counts.
    rising = [r(-1.0), r(-0.6), r(-0.9), r(-0.4), r(-0.3), r(-0.2), r(0.0)]
    assert stop_reason(rising) is None  # -0.4 > -1.0 + 0.5: best is -0.4, 3 since
    assert "stalled" in (stop_reason([*rising, r(0.05)]) or "")


def test_pull_scale() -> None:
    from mahjong.rl.pool import pull_scale

    assert pull_scale(0, 24) == 1.0
    assert pull_scale(12, 24) == 0.5
    assert pull_scale(24, 24) == 0.0 and pull_scale(30, 24) == 0.0
    assert pull_scale(1, 0) == 0.0


def test_pool_members_persist(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import torch
    from mahjong.rl.cnn import CnnConfig, SuitCNN
    from mahjong.rl.pool import Pool

    torch.manual_seed(0)
    small = CnnConfig(channels=8, blocks=1, honour_hidden=16, merge=16, value_hidden=8)
    pool = Pool(tmp_path)
    assert len(pool) == 0
    pool.add(SuitCNN(small), 2.0, 10)
    pool.add(SuitCNN(small), 4.0, 20)
    again = Pool(tmp_path)
    assert [m.name for m in again.members] == ["member-000", "member-001"]
    assert again.members[1].hours == 4.0 and again.mjnn(1).exists()
    x = torch.rand(2, 933)
    with torch.no_grad():
        a = again.net(0, torch.device("cpu"))(x)[0]
        b = SuitCNN.load(pool.pt(0))(x)[0]
    assert torch.equal(a, b)


def test_pool_run_with_history_network_resumes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    import torch
    from mahjong.rl.cnn import CnnConfig, SuitCNN
    from mahjong.rl.ppo_cnn import main

    torch.manual_seed(3)
    init = tmp_path / "init.pt"
    small = CnnConfig(channels=8, blocks=1, honour_hidden=16, merge=16, value_hidden=8)
    SuitCNN(CnnConfig(**{**small.__dict__, "history": True})).save(init)
    out = tmp_path / "run"
    common = ["--init", str(init), "--out", str(out), "--envs", "8", "--device", "cpu"]
    common += ["--hands-per-iter", "8", "--pool", "--no-curriculum", "--eval-deals", "4"]
    # Every iteration a result and a new pool member; no stop on draws in this test
    # (a random network draws most hands).
    fast = ["--eval-minutes", "0.0001", "--pool-minutes", "0.0001", "--pull", "0.1"]
    fast += ["--draw-limit", "1.0"]
    main([*common, *fast, "--minutes", "60", "--max-iterations", "2"])
    state = json.loads((out / "state.json").read_text())
    log = json.loads((out / "log.json").read_text())
    assert state["iteration"] == 2 and state["steps"] > 0  # `stage` seats were played
    evals = [e["eval"] for e in log if "eval" in e]
    assert len(evals) >= 1
    assert all(e["vs_stage"]["deals"] == 4 for e in evals)
    assert [len(e["vs_pool"]) for e in evals] == list(range(len(evals)))
    assert all(e["pull"] > 0 for e in log)
    assert state["stopped"] is None
    # Resume: continues from the saved network with the log intact.
    main([*common, *fast, "--minutes", "60", "--resume", "--max-iterations", "3"])
    log2 = json.loads((out / "log.json").read_text())
    assert log2[: len(log)] == log and len(log2) == 3
    assert len(json.loads((out / "pool" / "members.json").read_text())) == 3
    # The plan's draw limit stops a random network at its first result, for good.
    out2 = tmp_path / "drawn"
    common2 = [*common[:3], str(out2), *common[4:]]
    main([*common2, *fast[:-2], "--minutes", "60", "--max-iterations", "3"])
    state = json.loads((out2 / "state.json").read_text())
    assert state["stopped"] and "draws" in state["stopped"] and state["iteration"] == 1
    assert (out2 / "stopped.txt").read_text().startswith(state["stopped"])
    main([*common2, *fast[:-2], "--minutes", "60", "--resume", "--max-iterations", "3"])
    assert json.loads((out2 / "state.json").read_text())["iteration"] == 1
