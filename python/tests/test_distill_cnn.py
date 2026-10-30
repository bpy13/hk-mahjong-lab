"""Distillation of search labels into the suit-shared CNN (plan Task 15).

The training test copies 400 rows of a real label chunk (``runs/labels`` or
``runs/labels-smoke``, as ``test_labels.py``) into a two-chunk directory and trains a
tiny network on the CPU; it skips when no labels exist.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from mahjong.rl.cnn import (
    NUM_TARGETS,
    PERMS,
    action_map,
    action_perm,
    feature_perm,
    target_map,
)
from mahjong.rl.distill_cnn import DistillConfig, SuitSwap, distill
from mahjong.rl.labels import LabelDir

ROOT = Path(__file__).resolve().parents[2]


def test_suit_swap_on_the_gpu_matches_the_numpy_permutations() -> None:
    rng = np.random.default_rng(0)
    n = 60
    x = rng.random((n, 933)).astype(np.float32)
    legal = rng.random((n, 134)) < 0.3
    chosen = rng.integers(0, 134, n)
    target = rng.integers(0, NUM_TARGETS, n)
    target[::7] = 255
    which = np.arange(n) % len(PERMS)
    swap = SuitSwap("cpu")
    tx, tl, tc, tt = swap.apply(
        torch.from_numpy(which),
        torch.from_numpy(x),
        torch.from_numpy(legal),
        torch.from_numpy(chosen),
        torch.from_numpy(target),
    )
    for r in range(n):
        p = PERMS[which[r]]
        assert np.array_equal(tx[r].numpy(), x[r, feature_perm(p)])
        assert np.array_equal(tl[r].numpy(), legal[r, action_perm(p)])
        assert tc[r] == action_map(p)[chosen[r]]
        want = 255 if target[r] == 255 else target_map(p)[target[r]]
        assert tt[r] == want
        # A legal chosen move stays legal under the same permutation.
        if legal[r, chosen[r]]:
            assert tl[r, tc[r]]


def _labels() -> Path | None:
    for d in (ROOT / "runs" / "labels", ROOT / "runs" / "labels-smoke"):
        if (d / "meta.json").exists() and (d / "chunk-00000.bin").exists():
            return d
    return None


def _small_label_dir(src: Path, dst: Path, rows: int = 200) -> Path:
    """Two chunks of ``rows`` rows each, cut from the first chunk of ``src``."""
    labels = LabelDir(src)
    first = np.asarray(labels.chunk(0))
    if len(first) < 2 * rows:
        pytest.skip("label chunk too small")
    dst.mkdir()
    meta = json.loads((src / "meta.json").read_text("utf-8"))
    meta["chunks"] = []
    for i in range(2):
        name = f"chunk-{i:05d}.bin"
        first[i * rows : (i + 1) * rows].tofile(dst / name)
        meta["chunks"].append({"file": name, "rows": rows})
    meta["rows"] = 2 * rows
    (dst / "meta.json").write_text(json.dumps(meta), "utf-8")
    return dst


def test_distill_trains_exports_and_resumes(tmp_path: Path) -> None:
    src = _labels()
    if src is None:
        pytest.skip("no labels")
    labels = _small_label_dir(src, tmp_path / "labels")
    out = tmp_path / "distill"
    cfg = DistillConfig(
        labels=str(labels),
        out=str(out),
        models=str(tmp_path / "models"),
        name="tiny",
        epochs=1,
        minutes=5,
        batch=64,
        warmup_steps=2,
        device="cpu",
        int8_check_rows=100,
        cnn={"channels": 8, "blocks": 1, "honour_hidden": 16, "merge": 16, "value_hidden": 8},
    )
    res = distill(cfg)
    assert res["train_rows"] == 200 and res["held_out_rows"] == 200
    assert 0.0 <= res["held_out"]["top1_agreement"] <= 1.0
    assert 0.0 <= res["int8_top1_agreement_with_fp32"] <= 1.0
    for key in ("pt", "fp32", "int8"):
        assert Path(res["models"][key]).exists()
    assert res["int8_bytes"] < res["fp32_bytes"]
    assert (out / "ckpt.pt").exists() and (out / "train.log").exists()
    assert len(res["history"]) == 1

    # Resuming a finished run trains no more steps; a longer run continues from it.
    again = distill(cfg, resume=True)
    assert len(again["history"]) == 1
    cfg.epochs = 2
    longer = distill(cfg, resume=True)
    assert [h["epoch"] for h in longer["history"]] == [0, 1]


def test_distill_refuses_overlapping_chunks(tmp_path: Path) -> None:
    src = _labels()
    if src is None:
        pytest.skip("no labels")
    labels = _small_label_dir(src, tmp_path / "labels")
    cfg = DistillConfig(
        labels=str(labels),
        out=str(tmp_path / "o"),
        train_chunks=[0, 1],
        holdout_chunks=[1],
        device="cpu",
    )
    with pytest.raises(ValueError):
        distill(cfg)
    shutil.rmtree(tmp_path / "o", ignore_errors=True)
