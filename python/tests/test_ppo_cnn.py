"""Task 17: the paired environment and a short PPO run of the CNN on the CPU."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mahjong._core import HkVecEnv
from mahjong.rl.cnn import ORACLE_LEN, STATE_LEN, CnnConfig, SuitCNN
from mahjong.rl.ppo_cnn import force_wins, load_init, main, set_priority, squash, stage_of

# PyTorch's OpenMP threads and the environments' rayon threads slow each other
# down badly on a shared CPU (see ppo_cnn.Trainer).
torch.set_num_threads(1)


def _arrays(env: HkVecEnv) -> tuple[np.ndarray, np.ndarray]:
    fb, _ob, mb, ids, _seats = env.observe()
    f = np.frombuffer(fb, dtype=np.float32).reshape(len(ids), STATE_LEN)
    m = np.frombuffer(mb, dtype=np.uint8).reshape(len(ids), 134)
    return f, m


Chooser = Callable[[HkVecEnv, np.ndarray, np.ndarray], Any]


def _play(env: HkVecEnv, choose: Chooser, hands: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while len(out) < hands:
        f, m = _arrays(env)
        actions = choose(env, f, m)
        env.step([int(a) for a in actions])
        out.extend(env.finished(detail=True))
    return out


def test_paired_reward_is_zero_when_learner_and_baseline_agree() -> None:
    # The same network (greedy) plays the learner, the baseline and the other seats.
    torch.manual_seed(0)
    net = SuitCNN(CnnConfig(channels=16, blocks=1, honour_hidden=32, merge=32, value_hidden=16))

    def greedy(_env: HkVecEnv, f: np.ndarray, m: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            pi, _, _ = net(torch.from_numpy(f.copy()))
        a = pi.masked_fill(torch.from_numpy(m.copy()) == 0, -1e9).argmax(1).numpy()
        return force_wins(a, m)[0]

    res = _play(HkVecEnv(8, 5, paired=True, min_faan=0), greedy, 24)
    assert len(res) >= 24
    assert all(r["paired"] == 0 for r in res)
    assert all(r["deltas"] == r["baseline_deltas"] for r in res)

    # The other seats sample at random: while the two copies agree, each of their
    # moves is drawn once and played in both copies (common random numbers), so
    # the difference is still exactly zero.
    rng = np.random.default_rng(3)

    def mixed(env: HkVecEnv, f: np.ndarray, m: np.ndarray) -> np.ndarray:
        roles = np.frombuffer(env.observed_roles()[1], dtype=np.uint8)
        a = greedy(env, f, m)
        for k in np.flatnonzero(roles == 2):
            a[k] = rng.choice(np.flatnonzero(m[k]))
        return force_wins(a, m)[0]

    res = _play(HkVecEnv(8, 7, paired=True, min_faan=0), mixed, 24)
    assert all(r["paired"] == 0 for r in res)
    assert all(r["deltas"] == r["baseline_deltas"] for r in res)
    assert all(r["learner"] == r["env"] % 4 for r in res)

    # The same with Rust agents: `value` as the baseline and the opponents, and the
    # learner's moves taken from `value` too.
    env = HkVecEnv(8, 6, opponents="value", paired=True, baseline="value", min_faan=0)
    res = _play(env, lambda e, f, m: e.agent_actions("value"), 16)
    assert all(r["paired"] == 0 for r in res)

    # A random learner is not the baseline: the paired difference moves.
    rng = np.random.default_rng(0)
    env = HkVecEnv(8, 6, opponents="value", paired=True, baseline="value", min_faan=0)
    res = _play(env, lambda e, f, m: [rng.choice(np.flatnonzero(r)) for r in m], 16)
    assert any(r["paired"] != 0 for r in res)
    for r in res:
        seat = r["learner"]
        assert r["paired"] == r["deltas"][seat] - r["baseline_deltas"][seat]
        for _seat, cls, faan, _bits in r["wins"] + r["baseline_wins"]:
            assert 0 <= cls < 45 and faan >= 0


def test_unpaired_env_keeps_its_old_interface() -> None:
    env = HkVecEnv(4, 1)
    assert not env.paired and env.min_faan == 3
    env.set_min_faan(0)
    assert env.min_faan == 0
    f, m = _arrays(env)
    sides, roles = env.observed_roles()
    assert list(sides) == list(roles) == [0] * len(f)
    env.step([int(np.flatnonzero(r)[0]) for r in m])
    assert isinstance(env.finished(), list)


def test_critic_is_zero_sum_and_the_actor_ignores_the_oracle() -> None:
    torch.manual_seed(1)
    net = load_init(None)
    x = torch.rand(6, STATE_LEN)
    o1, o2 = torch.rand(6, ORACLE_LEN), torch.rand(6, ORACLE_LEN)
    with torch.no_grad():
        p1, t1, v1 = net(x, o1)
        p2, t2, v2 = net(x, o2)
    assert torch.allclose(v1.sum(1), torch.zeros(6), atol=1e-5)
    assert v1.shape == (6, 4)
    assert torch.equal(p1, p2) and torch.equal(t1, t2)
    assert not torch.allclose(v1, v2)


def test_helpers() -> None:
    assert squash([0.0])[0] == 0
    assert np.isclose(squash([8.0])[0], np.log(2)) and np.isclose(squash([-8.0])[0], -np.log(2))
    assert stage_of(0.1, True) == (0, 0)
    assert stage_of(0.3, True) == (1, 1)
    assert stage_of(0.5, True) == (2, 3)
    assert stage_of(0.1, False) == (2, 3)


def test_set_priority() -> None:
    import sys

    on_windows = sys.platform == "win32"
    try:
        assert set_priority("above_normal") is on_windows
    finally:
        assert set_priority("normal") is on_windows


def test_smoke_run_checkpoints_and_resume(tmp_path: Path) -> None:
    init = tmp_path / "init.pt"
    torch.manual_seed(2)
    SuitCNN().save(init)
    out = tmp_path / "run"
    common = ["--out", str(out), "--envs", "16", "--device", "cpu", "--hands-per-iter", "16"]
    # A 40-second run: passes all three curriculum stages, evaluates, checkpoints.
    main(
        ["--init", str(init), "--minutes", "0.67", *common]
        + ["--eval-minutes", "0.3", "--eval-walls", "1", "--ckpt-minutes", "0.25"]
    )
    state = json.loads((out / "state.json").read_text())
    log = json.loads((out / "log.json").read_text())
    n = state["iteration"]
    assert n >= 2 and len(log) == n
    assert (out / state["checkpoint"]).exists()
    assert state["stage"] == 2 and state["min_faan"] == 3
    assert {e["min_faan"] for e in log} >= {0, 3}
    assert any("eval" in e for e in log)
    assert all(np.isfinite(e["policy_loss"]) and np.isfinite(e["value_loss"]) for e in log)
    assert (out / "final.mjnn").exists() and (out / "teacher.pt").exists()

    # --resume continues the iteration count from the newest checkpoint.
    main(
        ["--init", str(init), "--minutes", "5", "--resume", "--max-iterations", str(n + 1)] + common
    )
    state2 = json.loads((out / "state.json").read_text())
    log2 = json.loads((out / "log.json").read_text())
    assert state2["iteration"] == n + 1
    assert state2["hands"] > state["hands"]
    assert [e["iter"] for e in log2] == list(range(1, n + 2))


def test_unpaired_run_with_random_init(tmp_path: Path) -> None:
    out = tmp_path / "unpaired"
    main(
        ["--init", "none", "--minutes", "5", "--out", str(out), "--envs", "16"]
        + ["--device", "cpu", "--hands-per-iter", "4", "--no-paired-reward"]
        + ["--no-target-head", "--max-iterations", "2"]
    )
    state = json.loads((out / "state.json").read_text())
    assert state["iteration"] == 2 and state["anchor_weight"] == 0.0
