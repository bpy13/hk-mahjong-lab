"""Reader for search-labelled decisions (plan Task 13, the ``label`` example).

A label directory holds ``meta.json`` (layout, teacher spec, chunks) and chunk files
``chunk-NNNNN.bin`` of packed rows, read through the same machinery as the self-play
logs (:class:`mahjong.belief.logs.LogDir`). One row is one decision of the labelled
seat; seats inside a row are relative to it (0 = actor). Field meanings are listed in
``crates/search/src/labels.rs``.

Example::

    from mahjong.rl.labels import LabelDir, policy_targets
    labels = LabelDir("runs/labels")
    rows = labels.chunk(0)
    x = features(rows)                 # float32, (n, 933)
    chosen, weight = policy_targets(rows)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from mahjong.belief.logs import LogDir, features

__all__ = [
    "LabelDir",
    "NO_ACTION",
    "NO_TARGET",
    "NUM_TARGETS",
    "features",
    "oracle",
    "policy_targets",
    "value_targets",
]

#: Empty candidate slot.
NO_ACTION = 65535
#: No feasible target plan.
NO_TARGET = 255
#: Target classes: plan kind (5) x suit (3) x structure (3).
NUM_TARGETS = 45

Rows = np.ndarray[Any, np.dtype[Any]]


class LabelDir(LogDir):
    """A label directory: its metadata (``teacher``, ``others``, ``seed``) and chunks."""

    @property
    def teacher(self) -> str:
        """The agent spec that labelled the decisions."""
        return str(self.meta["teacher"])

    @property
    def agents(self) -> list[str]:
        """The teacher and the agent of the other three seats."""
        return [str(self.meta["teacher"]), str(self.meta["others"])]


def oracle(rows: Rows) -> npt.NDArray[np.float32]:
    """The hidden oracle block (``features::encode_oracle``) as float32, ``(n, 136)``."""
    return np.asarray(rows["oracle"], dtype=np.float32)


def policy_targets(
    rows: Rows, clip: float = 3.0
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """The move played and a weight per row for the policy loss (plan Task 15).

    The weight is the search's margin for that move, ``diff / diff_se`` of the chosen
    candidate against the best other candidate, clipped to ``[1, clip]`` so clear
    decisions count more; rows without a search (a bot teacher) weigh 1.
    """
    chosen = rows["chosen"].astype(np.int64)
    codes = rows["cand_code"].astype(np.int64)
    diff = rows["cand_diff"].astype(np.float64)
    se = rows["cand_diff_se"].astype(np.float64)
    weight = np.ones(len(rows), dtype=np.float32)
    has = rows["n_candidates"] > 0
    if has.any():
        # z of each candidate against the baseline; the chosen move's margin is its
        # own z minus the best other candidate's (both measured on the same samples).
        z = np.where((codes != NO_ACTION) & (se > 0), diff / np.maximum(se, 1e-9), 0.0)
        is_chosen = codes == chosen[:, None]
        z_chosen = np.where(is_chosen, z, -np.inf).max(axis=1)
        z_other = np.where(~is_chosen & (codes != NO_ACTION), z, -np.inf).max(axis=1)
        margin = np.where(np.isfinite(z_other), z_chosen - z_other, clip)
        weight[has] = np.clip(np.abs(margin[has]), 1.0, clip).astype(np.float32)
    return chosen, weight


def value_targets(rows: Rows, scale: float = 8.0) -> npt.NDArray[np.float32]:
    """Final four-seat deltas (relative seats), log-scaled as idea 5d:
    ``sign(x) * log(1 + |x| / scale)``; shape ``(n, 4)``."""
    d = rows["deltas"].astype(np.float64)
    return (np.sign(d) * np.log1p(np.abs(d) / scale)).astype(np.float32)
