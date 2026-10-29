"""Search-labelled decisions (plan Task 13).

Reads the first chunk written by the ``label`` example: the directory in
``MAHJONG_LABEL_DIR``, else ``runs/labels`` and ``runs/labels-smoke``. Skips when none
exists (write one with ``cargo run --release -p mahjong-search --example label --
--decisions 200 --chunk 100 --out runs/labels-smoke --teacher
"pimc:samples=8,candidates=2"``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from mahjong.rl.labels import (
    NO_ACTION,
    NO_TARGET,
    NUM_TARGETS,
    LabelDir,
    features,
    oracle,
    policy_targets,
    value_targets,
)

ROOT = Path(__file__).resolve().parents[2]


def _dirs() -> list[Path]:
    env = os.environ.get("MAHJONG_LABEL_DIR")
    if env:
        return [Path(env)]
    return [ROOT / "runs" / "labels", ROOT / "runs" / "labels-smoke"]


def _present() -> list[Path]:
    return [d for d in _dirs() if (d / "meta.json").exists() and (d / "chunk-00000.bin").exists()]


@pytest.mark.parametrize(
    "path", _present() or [pytest.param(None, marks=pytest.mark.skip("no labels"))]
)
def test_label_chunk(path: Path) -> None:
    labels = LabelDir(path)
    rows = labels.chunk(0)
    n = len(rows)
    assert n == labels.meta["chunks"][0]["rows"] > 0
    assert labels.teacher

    x = features(rows)
    assert x.shape == (n, 933)
    assert np.isfinite(x).all()
    assert oracle(rows).shape == (n, 136)

    # The move played is legal, and a labelled decision always had a choice.
    legal = rows["legal"].astype(bool)
    assert legal[np.arange(n), rows["chosen"]].all()
    assert (legal.sum(axis=1) >= 2).all()
    # Searched rows: the played move and the baseline are among the candidates.
    k = rows["n_candidates"].astype(np.int64)
    codes = rows["cand_code"].astype(np.int64)
    searched = k > 0
    for i in np.flatnonzero(searched)[:500]:
        cands = codes[i, : k[i]]
        assert (codes[i, k[i] :] == NO_ACTION).all()
        assert legal[i, cands].all()
        assert rows["chosen"][i] in cands
        assert rows["baseline"][i] in cands
    # Targets and outcomes.
    t = rows["target"]
    assert ((t < NUM_TARGETS) | (t == NO_TARGET)).all()
    assert (rows["deltas"].sum(axis=1) == 0).all()
    won = rows["won"]
    assert set(np.unique(won)) <= {0, 1, 2}
    assert (rows["deltas"][won > 0, 0] > 0).all()
    assert (rows["win_patterns"][won == 0] == 0).all()

    chosen, weight = policy_targets(rows)
    assert chosen.shape == (n,) and weight.shape == (n,)
    assert ((weight >= 1.0) & (weight <= 3.0)).all()
    v = value_targets(rows)
    assert v.shape == (n, 4) and np.isfinite(v).all()
