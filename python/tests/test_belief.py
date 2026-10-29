"""The belief model (plan Task 5): shapes, export, invariance, loss, baselines."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from mahjong.belief.evaluate import buckets, recover_discards, uniform_count_nll
from mahjong.belief.logs import NUM_KINDS, LogDir
from mahjong.belief.model import (
    DISCARD_LEN,
    MAX_COPIES,
    N_OPP,
    OUTPUT_LEN,
    Batch,
    BeliefConfig,
    BeliefNet,
    batch_from_rows,
    discard_block,
    losses,
    mask_counts,
)
from mahjong.rl.model import STATE_LEN

ROOT = Path(__file__).resolve().parents[2]


def fake_batch(n: int, seed: int = 0) -> Batch:
    """Random rows with the log's value ranges."""
    g = torch.Generator().manual_seed(seed)
    features = torch.rand(n, STATE_LEN, generator=g)
    unseen = torch.randint(0, 5, (n, NUM_KINDS), generator=g)
    features[:, 16 * NUM_KINDS : 17 * NUM_KINDS] = unseen.float() / 4
    counts = torch.minimum(torch.randint(0, 3, (n, N_OPP, NUM_KINDS), generator=g), unseen[:, None])
    discards = torch.randint(0, NUM_KINDS, (n, N_OPP, 8), generator=g).to(torch.uint8)
    discards[:, :, 5:] = 255
    return Batch(
        features=features.half(),
        discards=discards,
        unseen=unseen,
        counts=counts,
        tenpai=torch.randint(0, 2, (n, N_OPP), generator=g).float(),
        waits=torch.randint(0, 2, (n, N_OPP, NUM_KINDS), generator=g).float(),
        faan=torch.randint(0, 8, (n, N_OPP), generator=g),
        outcome=torch.randint(0, 3, (n, N_OPP), generator=g),
    )


def test_shapes_and_budget() -> None:
    default = BeliefNet()
    b = fake_batch(5)
    assert default.inputs(b.features, b.discards).shape == (5, STATE_LEN)
    net = BeliefNet(BeliefConfig(discards=True))
    assert default.parameter_count() <= net.parameter_count() <= 1_000_000
    x = net.inputs(b.features, b.discards)
    assert x.shape == (5, STATE_LEN + DISCARD_LEN)
    out = net(x)
    assert out.shape == (5, OUTPUT_LEN)
    p = net.predict(out, b.unseen)
    assert p["counts"].shape == (5, N_OPP, NUM_KINDS, MAX_COPIES)
    assert torch.allclose(p["counts"].sum(-1), torch.ones(5, N_OPP, NUM_KINDS))
    # Copies beyond the unseen count have probability zero.
    copies = torch.arange(MAX_COPIES)
    impossible = (copies > b.unseen[:, None, :, None]).expand_as(p["counts"])
    assert float(p["counts"][impossible].max()) == 0.0
    assert p["tenpai"].shape == (5, N_OPP)
    assert p["waits"].shape == (5, N_OPP, NUM_KINDS)
    assert p["faan"].shape == (5, N_OPP, 8)
    assert torch.allclose(p["outcome"].sum(-1), torch.ones(5, N_OPP))
    small = BeliefNet(BeliefConfig(hidden=(16,), discards=False))
    assert small.inputs(b.features, b.discards).shape == (5, STATE_LEN)


def test_discard_block_is_one_hot_in_rust_order() -> None:
    d = torch.full((1, N_OPP, 8), 255, dtype=torch.uint8)
    d[0, 1, 0] = 7  # across, most recent: 8m
    d[0, 2, 3] = 33  # previous, fourth most recent: red dragon
    x = discard_block(d)
    assert x.shape == (1, DISCARD_LEN)
    assert float(x.sum()) == 2.0
    assert x[0, (1 * 8 + 0) * NUM_KINDS + 7] == 1.0
    assert x[0, (2 * 8 + 3) * NUM_KINDS + 33] == 1.0


def test_hidden_tiles_do_not_change_outputs() -> None:
    """Two rows with the same public view but different hidden truth: same output."""
    net = BeliefNet(BeliefConfig(hidden=(32,), discards=True))
    b = fake_batch(2, seed=3)
    b.features[1] = b.features[0]
    b.discards[1] = b.discards[0]
    assert not torch.equal(b.counts[0], b.counts[1])
    out = net(net.inputs(b.features, b.discards))
    assert torch.equal(out[0], out[1])


def read_mjnn(path: Path) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]], list]:
    data = path.read_bytes()
    assert data[:4] == b"MJNN"
    (version,) = struct.unpack_from("<I", data, 4)
    assert version == 2
    pos = 8

    def slices() -> list[tuple[str, int, int]]:
        nonlocal pos
        (n,) = struct.unpack_from("<I", data, pos)
        pos += 4
        out = []
        for _ in range(n):
            (ln,) = struct.unpack_from("<I", data, pos)
            pos += 4
            name = data[pos : pos + ln].decode()
            pos += ln
            off, length = struct.unpack_from("<II", data, pos)
            pos += 8
            out.append((name, off, length))
        return out

    ins, outs = slices(), slices()
    (n_layers,) = struct.unpack_from("<I", data, pos)
    pos += 4
    layers = []
    for _ in range(n_layers):
        o, i = struct.unpack_from("<II", data, pos)
        pos += 8
        w = np.frombuffer(data, dtype="<f4", count=o * i, offset=pos).reshape(o, i)
        pos += 4 * o * i
        bias = np.frombuffer(data, dtype="<f4", count=o, offset=pos)
        pos += 4 * o
        layers.append((w, bias))
    assert pos == len(data)
    return ins, outs, layers


def test_export_v2_round_trip(tmp_path: Path) -> None:
    plain = BeliefNet(BeliefConfig(hidden=(8,)))
    plain.export(tmp_path / "plain.mjnn")
    assert read_mjnn(tmp_path / "plain.mjnn")[0] == [("features", 0, STATE_LEN)]
    net = BeliefNet(BeliefConfig(hidden=(24, 16), discards=True))
    path = tmp_path / "b.mjnn"
    net.export(path)
    ins, outs, layers = read_mjnn(path)
    assert ins == [("features", 0, STATE_LEN), ("discards", STATE_LEN, DISCARD_LEN)]
    assert [o[0] for o in outs] == ["counts", "tenpai", "waits", "faan", "outcome"]
    assert outs[0] == ("counts", 0, N_OPP * NUM_KINDS * MAX_COPIES)
    assert outs[-1][1] + outs[-1][2] == OUTPUT_LEN
    b = fake_batch(3, seed=5)
    x = net.inputs(b.features, b.discards).numpy()
    for k, (w, bias) in enumerate(layers):
        x = x @ w.T + bias
        if k + 1 < len(layers):
            x = np.maximum(x, 0)
    torch_out = net(net.inputs(b.features, b.discards)).detach().numpy()
    assert np.allclose(x, torch_out, atol=1e-4)


def test_loss_falls_when_fitting_a_fixed_batch() -> None:
    torch.manual_seed(0)
    net = BeliefNet(BeliefConfig(hidden=(64,), discards=True))
    b = fake_batch(64, seed=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    first = None
    for _ in range(40):
        parts = losses(net, net(net.inputs(b.features, b.discards)), b)
        if first is None:
            first = float(parts["total"].detach())
        opt.zero_grad()
        parts["total"].backward()
        opt.step()
    assert first is not None
    assert float(parts["total"].detach()) < 0.9 * first
    assert set(parts) == {"counts", "tenpai", "waits", "faan", "outcome", "total"}


def test_mask_counts_blocks_impossible_copies() -> None:
    logits = torch.zeros(1, N_OPP, NUM_KINDS, MAX_COPIES)
    unseen = torch.full((1, NUM_KINDS), 2)
    m = mask_counts(logits, unseen)
    assert (m[..., :3] == 0).all() and (m[..., 3:] < -1e8).all()


def test_uniform_hypergeometric_sums_to_one() -> None:
    unseen = np.array([[4, 3, 0] + [2] * 31])
    # A 13-tile hand: P(c) over c = 0..4 of kind 0 sums to one, kind 2 is always 0.
    total = 0.0
    for c in range(5):
        counts = np.zeros((1, 3, NUM_KINDS), dtype=np.int64)
        counts[0, 0, 0] = c
        counts[0, 0, 1:14] = 1  # the other 13 - c tiles elsewhere
        counts[0, 0, 1 : 1 + c] = 0
        nll = uniform_count_nll(unseen, counts)
        total += float(np.exp(-nll[0, 0, 0]))
    assert abs(total - 1.0) < 1e-9
    counts = np.zeros((1, 3, NUM_KINDS), dtype=np.int64)
    counts[0, 0, 3:16] = 1
    # P(0 copies of a kind with no unseen copy) = 1.
    assert abs(float(uniform_count_nll(unseen, counts)[0, 0, 2])) < 1e-9


def test_buckets_cover_the_wall() -> None:
    b = buckets(np.array([91, 90, 76, 60, 45, 30, 15, 0]))
    assert b.tolist() == [0, 0, 0, 2, 3, 4, 5, 5]


def test_recover_discards_reads_the_next_row() -> None:
    dt = np.dtype(
        [
            ("hand", "<u4"),
            ("step", "<u2"),
            ("observer", "u1"),
            ("phase", "u1"),
            ("features", "<f2", (STATE_LEN,)),
            ("discards", "u1", (4, 8)),
            ("opp_waits", "u1", (3, NUM_KINDS)),
            ("opp_won", "u1", (3,)),
            ("opp_win_from", "u1", (3,)),
            ("deltas", "<i4", (4,)),
        ]
    )
    rows = np.zeros(4, dtype=dt)
    rows["discards"] = 255
    rows["opp_win_from"] = 255
    # Row 0: seat 0 (turn) discards tile 5; row 1 is seat 1's turn and sees it as
    # the most recent discard of its previous seat (relative 3).
    rows["hand"] = 7
    rows["step"] = [0, 1, 2, 3]
    rows["observer"] = [0, 1, 1, 2]
    rows["phase"] = [0, 0, 1, 0]
    rows["discards"][1, 3, 0] = 5
    # Row 1 → row 2: a claim window, seat 1 discarded nothing (history unchanged).
    rows["discards"][2, 0] = rows["discards"][1, 0]
    # Row 3: the final row; seat 2 held tile 9 and 12, opponent 1 waited on 9 and won.
    rows["features"][3, 9] = 1.0
    rows["features"][3, 12] = 1.0
    rows["opp_waits"][3, 0, 9] = 1
    rows["opp_won"][3, 0] = 1
    rows["opp_win_from"][3, 0] = 0
    idx, tile, info = recover_discards(rows)
    assert idx.tolist() == [0, 3]
    assert tile.tolist() == [5, 9]
    assert info["from_next_row"] == 1 and info["from_final_win"] == 1


@pytest.mark.skipif(
    not (ROOT / "runs" / "logs" / "holdout" / "chunk-00000.bin").exists(), reason="no log"
)
def test_real_rows_make_a_batch() -> None:
    rows = LogDir(ROOT / "runs" / "logs" / "holdout").chunk(0)[:512]
    b = batch_from_rows(rows)
    assert len(b) == 512
    assert (b.counts <= b.unseen[:, None, :]).all()
    assert set(b.outcome.unique().tolist()) <= {0, 1, 2}
    net = BeliefNet(BeliefConfig(hidden=(32,)))
    parts = losses(net, net(net.inputs(b.features, b.discards)), b)
    assert all(torch.isfinite(v) for v in parts.values())
    idx, tile, info = recover_discards(rows)
    assert len(idx) > 100 and (tile < NUM_KINDS).all()
    # A recovered discard is a tile the observer held (plane 0 of the features).
    held = rows["features"][idx, tile] > 0.5
    assert held.all()
