"""The self-play logs with hidden hands (plan Task 4).

Reads the first chunk of a log written by the ``selfplay_log`` example: the
directory in ``MAHJONG_LOG_DIR``, else ``runs/logs/train`` and ``runs/logs/holdout``.
Skips when no log exists (write one with
``cargo run --release -p mahjong-search --example selfplay_log -- --hands 100
--out runs/logs/train``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from mahjong.belief.logs import NUM_KINDS, LogDir, features, unseen_counts

ROOT = Path(__file__).resolve().parents[2]


def _log_dirs() -> list[Path]:
    env = os.environ.get("MAHJONG_LOG_DIR")
    if env:
        return [Path(env)]
    return [ROOT / "runs" / "logs" / "train", ROOT / "runs" / "logs" / "holdout"]


def _present() -> list[Path]:
    return [
        d for d in _log_dirs() if (d / "meta.json").exists() and (d / "chunk-00000.bin").exists()
    ]


@pytest.mark.parametrize(
    "path", _present() or [pytest.param(None, marks=pytest.mark.skip("no log"))]
)
def test_chunk_shapes_and_hidden_counts(path: Path) -> None:
    log = LogDir(path)
    rows = log.chunk(0)
    n = len(rows)
    assert n == log.meta["chunks"][0]["rows"] > 0

    x = features(rows)
    assert x.shape == (n, 933)
    assert np.isfinite(x).all()
    assert float(np.min(x)) >= -0.01 and float(np.max(x)) <= 1.01
    assert rows["discards"].shape == (n, 4, 8)
    assert rows["opp_counts"].shape == (n, 3, NUM_KINDS)
    assert rows["opp_waits"].shape == (n, 3, NUM_KINDS)
    assert rows["opp_tenpai"].shape == (n, 3)
    assert rows["deltas"].shape == (n, 4)
    assert int(np.max(rows["agents"])) < len(log.agents)

    # The opponents' hidden tiles fit inside what the observer has not seen, so
    # hidden hands plus the observer's view never exceed 4 copies of a kind.
    hidden = rows["opp_counts"].astype(np.int64).sum(axis=1)
    unseen = unseen_counts(rows)
    assert (hidden <= unseen).all()
    assert (unseen <= 4).all()
    # Each opponent holds 13 tiles minus 3 per meld, at most 13 (14 never: the
    # observer is the one to act).
    sizes = rows["opp_counts"].astype(np.int64).sum(axis=2)
    assert ((sizes >= 1) & (sizes <= 13) & (sizes % 3 == 1)).all()

    # Tenpai means some winning tile, and a win now pays something.
    waits_any = rows["opp_waits"].any(axis=2)
    assert (waits_any == rows["opp_tenpai"].astype(bool)).all()
    assert ((rows["opp_points_now"] > 0) == waits_any).all()

    # Outcomes: zero-sum deltas; winners collected points; non-winners did not.
    assert (rows["deltas"].sum(axis=1) == 0).all()
    won = rows["opp_won"]
    assert set(np.unique(won)) <= {0, 1, 2}
    assert (rows["opp_win_points"][won > 0] > 0).all()
    assert (rows["opp_win_points"][won == 0] == 0).all()
    from_ = rows["opp_win_from"]
    assert (from_[won == 2] == 255).all()
    assert (from_[won == 1] < 4).all()
