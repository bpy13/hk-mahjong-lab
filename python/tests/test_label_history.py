"""The discard history in label files (label layout 3, plan 2026-11-16 Task 4).

The labeller stores ``features::encode_history`` packed into 100 bytes per row
(``labels::pack_history``; the Rust test ``history_in_the_row_equals_encode_history``
checks the written bytes against ``encode_history`` on 1,000 decisions of ``stage``).
Here: the two Python unpackers (numpy for reading, torch for training on the card)
give back exactly ``encode_history`` on 1,000 positions of ``stage`` play; the suit
swap moves the unpacked block like the network's own permutation; and a tiny network
trains on layout-3 labels with the block when such a folder exists
(``runs/labels-v3-smoke``, written by
``cargo run --release -p mahjong-search --example label -- --decisions 400 --chunk 200
--out runs/labels-v3-smoke --teacher stage --others stage``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from mahjong._core import HkState, agent_action, encode_history
from mahjong.rl.cnn import HISTORY_DEPTH, HISTORY_LEN, PASSED_LEN, PERMS, STATE_LEN, feature_perm
from mahjong.rl.distill_cnn import DistillConfig, HistoryUnpack, SuitSwap, distill
from mahjong.rl.labels import HISTORY_BYTES, HISTORY_SLOTS, LabelDir, history, unpack_history

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "runs" / "labels-v3-smoke"


def _pack(block: np.ndarray) -> np.ndarray:
    """``labels::pack_history`` rebuilt from an ``encode_history`` block (test only).
    The overflow value keeps counts up to 4, which is all the block itself holds."""
    out = np.zeros(HISTORY_BYTES, dtype=np.uint8)
    planes = block[: HISTORY_SLOTS * 34].reshape(HISTORY_SLOTS, 34)
    sc = block[HISTORY_SLOTS * 34 :]
    for slot in range(HISTORY_SLOTS):
        if sc[2 * slot] == 0:
            assert planes[slot].sum() == 0
            continue
        tile = int(planes[slot].argmax())
        from_hand = sc[2 * slot] == 1.0
        out[slot] = 1 + tile + 34 * from_hand + 68 * int(sc[2 * slot + 1])
    for r in range(4):
        out[HISTORY_SLOTS + r] = round(sc[2 * HISTORY_SLOTS + r] * 4)
    return out


def _stage_positions(n: int) -> list[HkState]:
    out: list[HkState] = []
    hand = 0
    while len(out) < n:
        s = HkState(seed=7000 + hand)
        hand += 1
        while not s.is_terminal() and len(out) < n:
            if len(s.legal_actions()) > 1:
                out.append(s.copy())
            s.apply(agent_action(s, "stage", hand))
    return out


def test_unpacked_history_equals_encode_history_on_1000_positions() -> None:
    states = _stage_positions(1000)
    blocks = np.array([encode_history(s) for s in states], dtype=np.float32)
    assert blocks.shape == (1000, HISTORY_LEN)
    packed = np.stack([_pack(b) for b in blocks])
    # Every kind of slot occurs: æ‘¸æ‰“, å…¥ç« , claimed.
    used = packed[:, :HISTORY_SLOTS]
    assert ((used >= 1) & (used <= 34)).any()
    assert ((used >= 35) & (used <= 68)).any()
    assert (used > 68).any()
    assert np.array_equal(unpack_history(packed), blocks)
    on_device = HistoryUnpack("cpu")(torch.from_numpy(packed))
    assert torch.equal(on_device, torch.from_numpy(blocks))


def test_overflow_beyond_24_discards() -> None:
    packed = np.zeros((1, HISTORY_BYTES), dtype=np.uint8)
    packed[0, HISTORY_DEPTH] = 1 + 5 + 34  # next seat's first discard: 6m from the hand
    packed[0, HISTORY_SLOTS + 1] = 6  # six discards beyond the 24th: value capped at 1
    packed[0, HISTORY_SLOTS + 2] = 1
    out = unpack_history(packed)
    sc = HISTORY_SLOTS * 34
    assert out[0, HISTORY_DEPTH * 34 + 5] == 1.0
    assert out[0, sc + 2 * HISTORY_DEPTH] == 1.0
    assert out[0, sc + 2 * HISTORY_SLOTS + 1] == 1.0
    assert out[0, sc + 2 * HISTORY_SLOTS + 2] == 0.25
    assert torch.equal(HistoryUnpack("cpu")(torch.from_numpy(packed)), torch.from_numpy(out))


def test_suit_swap_with_the_history_block() -> None:
    rng = np.random.default_rng(2)
    n = 12
    x = rng.random((n, STATE_LEN + PASSED_LEN + HISTORY_LEN)).astype(np.float32)
    which = np.arange(n) % len(PERMS)
    swap = SuitSwap("cpu", passed=True, history=True)
    tx, _, _, _ = swap.apply(
        torch.from_numpy(which),
        torch.from_numpy(x),
        torch.zeros((n, 134), dtype=torch.bool),
        torch.zeros(n, dtype=torch.int64),
        torch.zeros(n, dtype=torch.int64),
    )
    for r in range(n):
        want = x[r, feature_perm(PERMS[which[r]], passed=True, history=True)]
        assert np.array_equal(tx[r].numpy(), want)


def _small(dst: Path, rows: int = 100) -> Path:
    labels = LabelDir(V3)
    first = np.asarray(labels.chunk(0))
    if len(first) < 2 * rows:
        pytest.skip("label chunk too small")
    dst.mkdir()
    meta = json.loads((V3 / "meta.json").read_text("utf-8"))
    meta["chunks"] = []
    for i in range(2):
        name = f"chunk-{i:05d}.bin"
        first[i * rows : (i + 1) * rows].tofile(dst / name)
        meta["chunks"].append({"file": name, "rows": rows})
    meta["rows"] = 2 * rows
    (dst / "meta.json").write_text(json.dumps(meta), "utf-8")
    return dst


def test_distill_with_the_history_block(tmp_path: Path) -> None:
    if not (V3 / "chunk-00000.bin").exists():
        pytest.skip("no layout-3 labels")
    labels = LabelDir(V3)
    assert labels.meta["layout_version"] == 3
    rows = labels.chunk(0)
    h = history(rows)
    assert h.shape == (len(rows), HISTORY_LEN) and np.isfinite(h).all()
    cfg = DistillConfig(
        labels=str(_small(tmp_path / "labels")),
        out=str(tmp_path / "distill"),
        models=str(tmp_path / "models"),
        name="tiny-history",
        epochs=1,
        minutes=5,
        batch=64,
        warmup_steps=2,
        device="cpu",
        int8_check_rows=100,
        cnn={
            "channels": 8,
            "blocks": 1,
            "honour_hidden": 16,
            "merge": 16,
            "value_hidden": 8,
            "passed": True,
            "history": True,
        },
    )
    res = distill(cfg)
    assert res["train_rows"] == 100
    assert 0.0 <= res["held_out"]["top1_agreement"] <= 1.0
    # The Rust int8 check read the 4,881-wide input (features + passed + history).
    assert 0.0 <= res["int8_top1_agreement_with_fp32"] <= 1.0
