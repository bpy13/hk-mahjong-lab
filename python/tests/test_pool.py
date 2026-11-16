"""Self-play against a pool of opponents (plan 2026-11-16 Task 5)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
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


def test_collapse_stop_and_best_result() -> None:
    from mahjong.rl.pool import best_result, stop_reason

    def r(mean: float, ci: float = 0.5) -> dict[str, Any]:
        return {"mean": mean, "ci95": ci, "draw_rate": 0.05, "train_draw_rate": 0.05}

    # (a) Two in a row below the best earlier one by more than the sum of two ranges (1.0).
    base = [r(-1.0), r(0.0), r(-0.2)]
    assert stop_reason([*base, r(-1.2)]) is None  # one low result
    assert stop_reason([*base, r(-0.9), r(-1.2)]) is None  # -0.9 is within 1.0 of 0.0
    why = stop_reason([*base, r(-1.2), r(-1.5)])
    assert why is not None and "collapse" in why
    assert stop_reason([r(0.0), r(-1.2), r(-0.5), r(-1.3)]) is None  # not in a row
    # Wider ranges forgive the drop.
    assert stop_reason([r(0.0, 1.0), r(-0.2), r(-1.5, 1.0), r(-1.5, 1.0)]) is None
    # (b) The best so far: the first of equal means; a worse result never replaces it.
    assert best_result([]) is None
    assert best_result([r(-1.0), r(0.0), r(-0.5), r(0.0)]) == 1


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


def test_pool_run_with_passed_and_history_network_resumes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    import torch
    from mahjong.rl.cnn import CnnConfig, SuitCNN
    from mahjong.rl.ppo_cnn import main

    torch.manual_seed(3)
    init = tmp_path / "init.pt"
    small = CnnConfig(channels=8, blocks=1, honour_hidden=16, merge=16, value_hidden=8)
    SuitCNN(CnnConfig(**{**small.__dict__, "passed": True, "history": True})).save(init)
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
    # (b), (c): best.pt is the network of the best result; log.json names both.
    assert (out / "best.pt").exists()
    best = [e["eval"]["best"] for e in log if "eval" in e][-1]
    means = [e["mean"] for e in (x["eval"]["vs_stage"] for x in log if "eval" in x)]
    assert best["file"] == "best.pt" and best["mean"] == max(means)
    assert best["iteration"] == [x["iter"] for x in log if "eval" in x][means.index(max(means))]
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


LABEL_EXE = Path(__file__).resolve().parents[2] / "target" / "release" / "examples" / "label.exe"
SMALL: dict[str, Any] = {
    "channels": 8,
    "blocks": 1,
    "honour_hidden": 16,
    "merge": 16,
    "value_hidden": 8,
}


def _env_rows(net_cfg: Any, rows: int, seed: int) -> tuple[np.ndarray, np.ndarray, HkVecEnv]:
    """``rows`` decisions of random play, encoded as ppo_cnn feeds them."""
    from mahjong.rl.ppo_cnn import observe

    env = HkVecEnv(16, seed, min_faan=0)
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ms: list[np.ndarray] = []
    while sum(len(x) for x in xs) < rows:
        f, _o, m, ids, _ = observe(env, net_cfg.passed, net_cfg.history)
        xs.append(f.copy())
        ms.append(m.copy())
        env.step([int(rng.choice(np.flatnonzero(m[k]))) for k in range(len(ids))])
    return np.concatenate(xs)[:rows], np.concatenate(ms)[:rows], env


def _pack_history(h: np.ndarray) -> np.ndarray:
    """The inverse of ``labels.unpack_history`` (``labels::pack_history``'s bytes)."""
    from mahjong.rl.labels import HISTORY_SLOTS

    n = h.shape[0]
    sc = HISTORY_SLOTS * 34
    out = np.zeros((n, HISTORY_SLOTS + 4), dtype=np.uint8)
    planes = h[:, :sc].reshape(n, HISTORY_SLOTS, 34)
    full = planes.max(2) > 0
    tile = planes.argmax(2)
    from_hand = h[:, sc : sc + 2 * HISTORY_SLOTS : 2] == 1.0
    claimed = h[:, sc + 1 : sc + 2 * HISTORY_SLOTS : 2] == 1.0
    out[:, :HISTORY_SLOTS] = np.where(full, 1 + tile + 34 * from_hand + 68 * claimed, 0)
    out[:, HISTORY_SLOTS:] = np.round(h[:, sc + 2 * HISTORY_SLOTS :] * 4).astype(np.uint8)
    return out


def test_env_inputs_match_the_rust_network_agent(tmp_path: Path) -> None:
    # A network reading the passed and history blocks: PyTorch on the rows ppo_cnn
    # builds from the vector env picks the move Rust's `net:` agent picks from its
    # own encoding of the same positions.
    import torch
    from mahjong.rl.cnn import CnnConfig, SuitCNN, export
    from mahjong.rl.ppo_cnn import masked, observe

    torch.manual_seed(4)
    net = SuitCNN(CnnConfig(**SMALL, passed=True, history=True)).eval()
    path = tmp_path / "net.mjnn"
    export(net, path)
    env = HkVecEnv(16, 21, min_faan=0)
    rng = np.random.default_rng(2)
    checked = 0
    while checked < 300:
        f, _o, m, ids, _ = observe(env, True, True)
        rust = np.asarray(env.agent_actions(f"net:path={path}"))
        with torch.no_grad():
            logits = masked(net(torch.from_numpy(f.copy()))[0], torch.from_numpy(m.copy()))
        top2 = logits.topk(2, 1)
        clear = (top2.values[:, 0] - top2.values[:, 1] > 1e-3).numpy() & (m.sum(1) > 1)
        assert np.array_equal(rust[clear], top2.indices[:, 0].numpy()[clear])
        checked += int(clear.sum())
        env.step([int(rng.choice(np.flatnonzero(m[k]))) for k in range(len(ids))])


@pytest.mark.skipif(not LABEL_EXE.exists(), reason="build the label example first")
def test_distilled_network_reads_env_rows_as_label_rows(tmp_path: Path) -> None:
    # The seam between Task 4 and Task 6: a network saved by `distill_cnn --passed
    # --history` loads through ppo_cnn's load_init, and its output on 200 positions
    # fed as the vector env hands them to ppo_cnn equals its output on the same
    # positions fed through distill_cnn's label path (float16 features and passed
    # block, packed history, distill_cnn.inputs).
    import torch
    from mahjong.rl.distill_cnn import Data, DistillConfig, HistoryUnpack, distill, inputs
    from mahjong.rl.labels import unpack_history
    from mahjong.rl.ppo_cnn import load_init

    labels = tmp_path / "labels"
    subprocess.run(
        [str(LABEL_EXE), "--decisions", "400", "--chunk", "200", "--seed", "5"]
        + ["--teacher", "stage", "--others", "stage", "--out", str(labels)],
        check=True,
        capture_output=True,
    )
    cfg = DistillConfig(
        labels=str(labels),
        out=str(tmp_path / "distill"),
        models=str(tmp_path / "models"),
        name="tiny",
        epochs=1,
        minutes=0.2,
        batch=128,
        warmup_steps=1,
        device="cpu",
        int8_check_rows=50,
        cnn={**SMALL, "passed": True, "history": True},
    )
    distill(cfg)
    net = load_init(tmp_path / "models" / "tiny.pt").eval()
    assert net.cfg.passed and net.cfg.history and net.cfg.oracle

    x, m, _env = _env_rows(net.cfg, 200, 9)
    from mahjong.rl.cnn import HISTORY_LEN, STATE_LEN

    front = x[:, : x.shape[1] - HISTORY_LEN]  # features and passed block
    hist = x[:, x.shape[1] - HISTORY_LEN :]
    packed = _pack_history(hist)
    assert np.array_equal(unpack_history(packed), hist)  # the packing is lossless here
    assert front.shape[1] > STATE_LEN
    n = len(x)
    data = Data(
        x=torch.from_numpy(front.astype(np.float16)),
        legal=torch.from_numpy(m > 0),
        chosen=torch.zeros(n, dtype=torch.int64),
        weight=torch.ones(n),
        target=torch.full((n,), 255, dtype=torch.int64),
        value=torch.zeros((n, 4)),
        hist=torch.from_numpy(packed),
    )
    # Label files store the features and passed block as float16 (some feature
    # columns are continuous, e.g. 4.5e-6), the env hands over float32: the seam is
    # checked on the env rows rounded the way a label row stores them, and the
    # unrounded difference is bounded separately.
    rounded = x.copy()
    rounded[:, : front.shape[1]] = front.astype(np.float16).astype(np.float32)
    with torch.no_grad():
        via_labels = net(inputs(data, slice(0, n), HistoryUnpack("cpu")).float())
        via_env = net(torch.from_numpy(rounded))
        via_env_f32 = net(torch.from_numpy(x.copy()))
    for a, b, c in zip(via_labels, via_env, via_env_f32, strict=True):
        assert (a - b).abs().max().item() < 1e-5
        assert (a - c).abs().max().item() < 1e-3
