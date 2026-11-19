"""The risk network (plan 2026-11-16 Task 15): row layout, model, export, parity with
Rust, and the check tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from mahjong._core import HkState, encode_history, encode_state, net_forward
from mahjong.risk import check
from mahjong.risk.data import (
    FIELDS,
    HISTORY_LEN,
    INPUT_LEN,
    RISK_LEN,
    STATE_LEN,
    RiskData,
    row_dtype,
)
from mahjong.risk.model import (
    HEAD_LEN,
    OUTPUT_LEN,
    RiskConfig,
    RiskNet,
    alive_from_inputs,
    losses,
    numpy_forward,
)

ROOT = Path(__file__).resolve().parents[2]


def test_row_layout_is_fixed() -> None:
    dt = row_dtype()
    assert dt.names is not None and dt.names[0] == "deal" and dt.names[-1] == "exact_points"
    size = sum(
        np.dtype(d).itemsize * int(np.prod(s)) if s else np.dtype(d).itemsize for _, d, s in FIELDS
    )
    assert dt.itemsize == size
    assert RISK_LEN == 3 * (5 * 34 + 8) and INPUT_LEN == STATE_LEN + HISTORY_LEN + RISK_LEN
    data = ROOT / "runs" / "risk" / "data"
    if (data / "meta.json").exists():
        d = RiskData(data)
        assert d.dtype.itemsize == int(d.meta["row_bytes"])
        rows = d.chunk(d.chunks[0])[:50]
        assert (rows["win"] <= 1).all()
        assert ((rows["points"] == 0) == (rows["win"] == 0)).all()


def _positions(n: int, seed: int) -> list[HkState]:
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
def rows() -> np.ndarray:
    """1,000 real positions: Rust's features and history, then a sparse risk block
    (the Rust binding of this checkout may predate `encode_risk`; the block holds
    the same value set: 0, 1/4 … 1)."""
    states = _positions(1000, 15)
    rng = np.random.default_rng(15)
    x = np.zeros((len(states), INPUT_LEN), dtype=np.float32)
    for i, s in enumerate(states):
        x[i, :STATE_LEN] = encode_state(s)[0]
        x[i, STATE_LEN : STATE_LEN + HISTORY_LEN] = encode_history(s)
    x[:, STATE_LEN + HISTORY_LEN :] = rng.choice(
        [0.0, 0.0, 0.0, 0.25, 0.5, 1.0], size=(len(states), RISK_LEN)
    ).astype(np.float32)
    return x


def test_model_shapes_loss_and_prediction(rows) -> None:
    torch.manual_seed(1)
    net = RiskNet(RiskConfig(hidden=(32, 16)))
    x = torch.from_numpy(rows[:64])
    out = net(x)
    assert out.shape == (64, OUTPUT_LEN) and OUTPUT_LEN == 2 * HEAD_LEN == 204
    win = (torch.rand(64, 3, 34) < 0.05).float()
    points = win * 8.0 * 2 ** torch.randint(0, 5, (64, 3, 34)).float()
    al = alive_from_inputs(x)
    assert al.shape == (64, 34) and al.any()
    loss = losses(out, win, points, al)
    assert torch.isfinite(loss.total) and loss.win > 0 and loss.pay >= 0
    p, pts = net.predict(x, al)
    assert p.shape == (64, 3, 34) and ((p >= 0) & (p <= 1)).all()
    dead = ~al[:, None, :].expand_as(p)
    assert (p[dead] == 0).all() and (pts[dead] == 0).all()
    live = ~dead
    assert (pts[live] >= 8.0).all() and (pts[live] <= 128.0).all()
    # A wrong pay target does not count where nothing wins.
    with torch.no_grad():
        l0 = losses(out, torch.zeros_like(win), points, al)
    assert float(l0.pay) == 0.0


def test_export_round_trips_and_matches_rust(rows, tmp_path) -> None:
    """Rust and Python give the same network output on 1,000 positions (largest
    difference under 1e-4)."""
    torch.manual_seed(2)
    net = RiskNet(RiskConfig(hidden=(48, 24)))
    named = tmp_path / "risk.mjnn"
    net.export(named)
    with open(named, "rb") as f:
        head = f.read(200)
    assert head[:4] == b"MJNN" and b"risk" in head and b"win" in head and b"pay" in head
    with torch.no_grad():
        want = net(torch.from_numpy(rows)).numpy()
    # The numpy reader of the same file.
    got = numpy_forward(named, rows)
    assert got.shape == want.shape and np.abs(got - want).max() < 1e-4
    # Rust: through a file whose risk block is unnamed, so an older binding runs it too.
    unnamed = tmp_path / "risk-unnamed.mjnn"
    net.export(unnamed, named_risk=False)
    out, (inp, n_out) = net_forward(str(unnamed), rows.astype("<f4").tobytes())
    assert (inp, n_out) == (INPUT_LEN, OUTPUT_LEN)
    rust = np.frombuffer(out, dtype="<f4").reshape(len(rows), n_out)
    assert np.abs(rust - want).max() < 1e-4
    # Save and load keep the weights.
    net.save(tmp_path / "risk.pt")
    again = RiskNet.load(tmp_path / "risk.pt")
    with torch.no_grad():
        assert torch.equal(again(torch.from_numpy(rows[:5])), net(torch.from_numpy(rows[:5])))


def test_check_tables_score_a_perfect_and_a_flat_source() -> None:
    rng = np.random.default_rng(3)
    y = (rng.random(20000) < 0.05).astype(np.float64)
    perfect = np.where(y == 1, 0.999, 0.001)
    flat = np.full_like(y, 0.05)
    assert check.log_loss(perfect, y) < check.log_loss(flat, y)
    assert check.brier(perfect, y) < check.brier(flat, y)
    table, ok = check.table_a({"flat": flat, "perfect": perfect}, y)
    # The flat source sits in one bin at its observed rate; the "perfect" one puts
    # 0.1 % on the misses, so its safest bin reads 0 observed (ratio 0): not within.
    assert "| 3–5 %" in table and ok["flat"] and not ok["perfect"]
    assert check.bins_of(np.array([0.0, 0.004, 0.005, 0.3])).tolist() == [0, 0, 1, 9]
    assert check.bin_name(0) == "0–0.5 %" and check.bin_name(9) == "25 % +"
    c = check.combine(np.array([[[0.5, 0.0], [0.5, 0.0], [0.0, 0.0]]]))
    assert np.allclose(c, [[0.75, 0.0]])
