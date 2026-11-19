"""The risk network (plan 2026-11-16 Task 15): row layout, model, export, parity with
Rust, and the check tables."""

from __future__ import annotations

import json
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


def _synthetic_rows(n: int, seed: int, layout: int) -> np.ndarray:
    """Rows with the content of a logged decision, in layout 1 or 2 (Task 16)."""
    from mahjong.risk.data import FIELDS_V2, N_OPP, NUM_KINDS, RISK_BYTES, UNSEEN_PLANE

    rng = np.random.default_rng(seed)
    rows = np.zeros(n, dtype=row_dtype(FIELDS if layout == 1 else FIELDS_V2))
    rows["deal"] = np.arange(n) // 30
    rows["seat"] = np.arange(n) % 4
    rows["wall"] = rng.integers(0, 90, n)
    rows["thrown"] = rng.integers(0, NUM_KINDS, n)
    rows["concealed"] = rng.choice([1, 4, 7, 10, 13], size=(n, N_OPP))
    rows["shares"] = 1
    rows["hand"] = (rng.random((n, NUM_KINDS)) < 0.3).astype(np.uint8)
    feats = (rng.random((n, STATE_LEN)) < 0.2).astype(np.float16) * 0.25
    feats[:, UNSEEN_PLANE * NUM_KINDS : (UNSEEN_PLANE + 1) * NUM_KINDS] = (
        rng.integers(0, 5, (n, NUM_KINDS)) / 4.0
    )
    rows["features"] = feats
    hist = rng.integers(0, 137, (n, 100)).astype(np.uint8)
    hist[:, 96:] = rng.integers(0, 3, (n, 4))
    rows["history"] = hist
    risk = rng.integers(0, 5, (n, RISK_LEN)).astype(np.uint8)
    if layout == 1:
        rows["risk"] = risk
    else:
        padded = np.concatenate([risk, np.zeros((n, 2 * RISK_BYTES - RISK_LEN), np.uint8)], axis=1)
        rows["risk"] = padded[:, 0::2] | (padded[:, 1::2] << 4)
    win = (rng.random((n, N_OPP, NUM_KINDS)) < 0.02).astype(np.uint8)
    rows["win"] = win
    rows["points"] = win * 16
    for f in ["table_tile", "exact_tile", "belief_tile"]:
        rows[f] = rng.random((n, NUM_KINDS)) * 0.3
    for f in ["table_loss", "exact_loss", "belief_loss"]:
        rows[f] = rng.random((n, NUM_KINDS)) * 2
    for f in ["table_p", "belief_p", "exact_p"]:
        rows[f] = rng.random((n, N_OPP, NUM_KINDS)) * 0.2
    rows["is_exact"] = 0
    return rows


def _write_data(path: Path, layout: int, chunks: list[np.ndarray], chunk_deals: int) -> None:
    from mahjong.risk.data import FIELDS_V2

    fields = FIELDS if layout == 1 else FIELDS_V2
    path.mkdir(parents=True)
    for i, rows in enumerate(chunks):
        rows.tofile(path / f"chunk-{i:03d}.bin")
    meta = {
        "layout": layout,
        "row_bytes": row_dtype(fields).itemsize,
        "deals": chunk_deals * len(chunks),
        "chunk_deals": chunk_deals,
        "seed": 1,
        "belief": None,
        "bots": "stage",
        "fields": [[n, d, list(s)] for n, d, s in fields],
        "chunks": [{"chunk": i, "rows": len(r)} for i, r in enumerate(chunks)],
    }
    (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_layout_2_rows_read_like_layout_1(tmp_path) -> None:
    """Task 16's smaller rows (float16 check columns, two risk values a byte, 3,613 bytes)
    give the same network input as Task 15's; both layouts load."""
    from mahjong.risk.data import FIELDS_V2, inputs, risk_block

    assert row_dtype(FIELDS_V2).itemsize == 3613 and row_dtype(FIELDS).itemsize == 4288
    a = _synthetic_rows(120, 7, 1)
    b = _synthetic_rows(120, 7, 2)
    _write_data(tmp_path / "v1", 1, [a], 4)
    _write_data(tmp_path / "v2", 2, [b], 4)
    d1, d2 = RiskData(tmp_path / "v1"), RiskData(tmp_path / "v2")
    assert (d1.layout, d2.layout) == (1, 2)
    r1, r2 = d1.chunk(0), d2.chunk(0)
    assert np.array_equal(inputs(r1), inputs(r2))
    assert np.array_equal(risk_block(r1), risk_block(r2))
    assert set(np.unique(risk_block(r2))) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert (
        np.abs(r1["table_tile"].astype(np.float64) - r2["table_tile"].astype(np.float64)).max()
        < 2e-4
    )
    assert np.array_equal(r1["win"], r2["win"])


def test_fit_stops_on_held_out_loss_and_summary_tables_every_source(tmp_path) -> None:
    """``--patience`` ends the fit once the held-out ``win`` loss stops improving, the fit
    resumes as finished, and the summary scores every source on the same rows."""
    from mahjong.risk import fit, summary

    chunks = [_synthetic_rows(240, s, 2) for s in (1, 2, 3)]
    data = tmp_path / "data"
    _write_data(data, 2, chunks, 8)
    out = tmp_path / "fit"
    args = [
        "--data", str(data), "--out", str(out), "--holdout-chunks", "2", "--train-chunks", "0-1",
        "--holdout-rows", "200", "--epochs", "40", "--patience", "1", "--select", "win",
        "--hidden", "16", "8", "--device", "cpu", "--threads", "2", "--batch", "64",
    ]  # fmt: skip
    fit.train(fit.parse_args(args))
    log = [line for line in (out / "train.log").read_text(encoding="utf-8").splitlines() if line]
    events = [json.loads(line)["event"] for line in log]
    assert events[0] == "start" and "stopped" in events and events[-1] == "done"
    assert (
        events.count("epoch") < 40 and (out / "best.pt").exists() and (out / "risk.mjnn").exists()
    )
    fit.train(fit.parse_args([*args, "--resume"]))
    assert "resumed" in (out / "train.log").read_text(encoding="utf-8")
    text = summary.run(
        summary.parse_args(
            ["--data", str(data), "--chunks", "2", "--model", f"net={out}", "--threads", "2"]
        )
    )
    for src in ["| table |", "| exact |", "| belief |"]:
        assert text.count(src) == 2, src  # populations `all` and `hand`
    assert text.count("| net |") == 3  # and the fits table
    assert "#### The fits" in text and "| net | [16, 8] |" in text
    assert "held-out win loss not improved" in text
