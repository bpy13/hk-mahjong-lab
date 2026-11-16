"""The full discard history as a network input (plan 2026-11-16 Task 3).

(a) and (b) are checked in Rust as well (``features.rs``); here: the Python reference
encoder equals Rust's ``features::encode_history`` on 1,000 positions, a network with
the block gives the same output in PyTorch and in Rust (largest difference below
1e-5), it stays suit-equivariant, and the CNN checkpoints trained before the block
still load and give the same output as before the change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from mahjong._core import HkState, encode_history, encode_state, net_forward
from mahjong.rl import cnn
from mahjong.rl.cnn import (
    HISTORY_DEPTH,
    HISTORY_LEN,
    HISTORY_PLANES,
    OUTPUT_LEN,
    PASSED_LEN,
    PERMS,
    STATE_LEN,
    CnnConfig,
    SuitCNN,
    action_perm,
    export,
    feature_perm,
    input_len,
    outputs,
    target_perm,
)

MODELS = Path(__file__).resolve().parents[2] / "models"


def _positions(n: int, seed: int) -> list[HkState]:
    """``n`` live positions from random legal play (whole hands, every decision)."""
    rng = np.random.default_rng(seed)
    out: list[HkState] = []
    hand = 0
    while len(out) < n:
        s = HkState(seed=seed * 1000 + hand)
        hand += 1
        while not s.is_terminal() and len(out) < n:
            out.append(s.copy())
            legal = s.legal_actions()
            s.apply(int(legal[rng.integers(len(legal))]))
    return out


@pytest.fixture(scope="module")
def states() -> list[HkState]:
    return _positions(1000, 31)


@pytest.fixture(scope="module")
def rows(states) -> np.ndarray:
    """Features followed by Rust's history block, one row per position."""
    x = np.zeros((len(states), STATE_LEN + HISTORY_LEN), dtype=np.float32)
    for i, s in enumerate(states):
        x[i, :STATE_LEN] = encode_state(s)[0]
        x[i, STATE_LEN:] = encode_history(s)
    return x


def test_python_and_rust_encode_the_same_history(states, rows) -> None:
    assert HISTORY_LEN == 96 * 34 + 196 and HISTORY_PLANES == 4 * HISTORY_DEPTH
    claimed = drawn = 0
    for s, row in zip(states, rows, strict=True):
        me = s.current_player()
        assert me is not None
        mine = cnn.encode_history(s.events(), me)
        assert np.array_equal(mine, row[STATE_LEN:]), s.events()
        sc = row[STATE_LEN + HISTORY_PLANES * 34 :]
        claimed += int((sc[1 : 2 * HISTORY_PLANES : 2] == 1).sum() > 0)
        drawn += int((sc[0 : 2 * HISTORY_PLANES : 2] == 0.5).sum() > 0)
    # The positions cover claimed discards and æ‘¸æ‰“ as well as å…¥ç« .
    assert claimed > 50 and drawn > 50


def test_history_rule_beyond_24_discards() -> None:
    """The stated rule for a 25th discard: slots keep the first 24; each later
    one adds 1/4 to the seat's overflow value (at most 1)."""
    events = [{"type": "discard", "seat": 3, "tile": i % 27, "from_hand": True} for i in range(27)]
    h = cnn.encode_history(events, me=0)
    sc = HISTORY_PLANES * 34
    r = 3
    for k in range(HISTORY_DEPTH):
        assert h[(r * HISTORY_DEPTH + k) * 34 + k % 27] == 1.0
    assert h[:sc].sum() == HISTORY_DEPTH
    assert h[sc + 2 * HISTORY_PLANES + r] == 0.75


def test_hidden_tiles_do_not_change_the_history(states) -> None:
    for i, s in enumerate(states[::50]):
        me = s.current_player()
        other = s.determinize(me, seed=i)
        assert encode_history(other) == encode_history(s)


def _rust(path: Path, x: np.ndarray) -> np.ndarray:
    out, (inp, n_out) = net_forward(str(path), x.astype("<f4").tobytes())
    assert (inp, n_out) == (x.shape[1], OUTPUT_LEN)
    return np.frombuffer(out, dtype="<f4").reshape(len(x), n_out)


def test_history_network_runs_the_same_in_rust(rows, tmp_path) -> None:
    torch.manual_seed(2)
    net = SuitCNN(CnnConfig(channels=32, blocks=2, honour_hidden=64, merge=64, history=True))
    net.eval()
    x = torch.from_numpy(rows)
    assert x.shape[1] == input_len(False, True)
    with pytest.raises(ValueError):
        net(x[:, :STATE_LEN])
    want = outputs(net, x).numpy()
    # The block matters.
    zeroed = torch.cat([x[:, :STATE_LEN], torch.zeros(len(x), HISTORY_LEN)], 1)
    assert not np.allclose(outputs(net, zeroed).numpy()[:, :134], want[:, :134])
    path = tmp_path / "history.mjnn"
    export(net, path)
    raw = path.read_bytes()
    assert b"history" in raw
    got = _rust(path, rows)
    diff = float(np.abs(got - want).max())
    print(f"largest difference on {len(rows)} positions: {diff:.2e}")
    assert diff < 1e-5


def test_passed_and_history_network_runs_the_same_in_rust(rows, tmp_path) -> None:
    """Both blocks: passed first, history after it (random passed values)."""
    rng = np.random.default_rng(4)
    n = 200
    passed = rng.choice([0.0, 0.0, 0.0, 0.25, 0.5, 1.0], size=(n, PASSED_LEN))
    x = np.concatenate([rows[:n, :STATE_LEN], passed.astype(np.float32), rows[:n, STATE_LEN:]], 1)
    torch.manual_seed(3)
    cfg = CnnConfig(channels=16, blocks=1, honour_hidden=32, merge=32, passed=True, history=True)
    net = SuitCNN(cfg).eval()
    path = tmp_path / "both.mjnn"
    export(net, path)
    got = _rust(path, x)
    assert np.abs(got - outputs(net, torch.from_numpy(x)).numpy()).max() < 1e-5


@pytest.mark.parametrize("perm", PERMS[1:])
def test_history_network_is_suit_equivariant(rows, perm) -> None:
    torch.manual_seed(5)
    cfg = CnnConfig(channels=16, blocks=1, honour_hidden=32, merge=32, passed=True, history=True)
    net = SuitCNN(cfg).eval()
    rng = np.random.default_rng(6)
    n = 64
    passed = rng.choice([0.0, 0.25, 1.0], size=(n, PASSED_LEN)).astype(np.float32)
    x = torch.from_numpy(np.concatenate([rows[:n, :STATE_LEN], passed, rows[:n, STATE_LEN:]], 1))
    p, t, v = net(x)
    ps, ts, vs = net(x[:, torch.from_numpy(feature_perm(perm, passed=True, history=True))])
    assert torch.allclose(ps, p[:, torch.from_numpy(action_perm(perm))], atol=1e-4)
    assert torch.allclose(ts, t[:, torch.from_numpy(target_perm(perm))], atol=1e-4)
    assert torch.allclose(vs, v, atol=1e-4)


#: ``(file, sum of |outputs|, sum of outputs x column)`` on 100 fixed random inputs,
#: recorded with the code before Task 3 (commit 18b1b09; unused target classes
#: counted as 0).
OLD_CHECKPOINTS = [
    ("cnn-distill.pt", 77195.24206740136, -1088446.5485516284),
    ("cnn-distill-v2.pt", 86457.80385740838, 104144.83804514809),
    ("cnn-ppo.pt", 82706.61192672706, -8157041.419776829),
]


@pytest.mark.parametrize(("name", "abs_sum", "weighted"), OLD_CHECKPOINTS)
def test_old_cnn_checkpoints_load_and_give_the_same_output(name, abs_sum, weighted) -> None:
    path = MODELS / name
    if not path.exists():
        pytest.skip(f"{name} not present (models/ is not in git)")
    net = SuitCNN.load(path).eval()
    assert not net.cfg.history
    n_in = input_len(net.cfg.passed)
    g = torch.Generator().manual_seed(7)
    x = (torch.rand(100, n_in, generator=g) < 0.1).float() * torch.rand(100, n_in, generator=g)
    y = outputs(net, x).double()
    y = torch.where(y < -9e3, torch.zeros_like(y), y)
    assert float(y.abs().sum()) == pytest.approx(abs_sum, rel=1e-6)
    assert float((y * torch.arange(y.shape[1])).sum()) == pytest.approx(weighted, rel=1e-6)
