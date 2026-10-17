"""Collecting self-play hands from ``HkVecEnv`` with a network choosing actions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, field

import numpy as np
import torch

from mahjong._core import HkVecEnv
from mahjong.rl.model import POINT_SCALE, to_arrays

WIN_CODES = (34, 104)


@dataclass
class Batch:
    """Decisions from finished hands, with each acting seat's final points."""

    feats: list[np.ndarray] = field(default_factory=list)
    oracle: list[np.ndarray] = field(default_factory=list)
    masks: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    extra: list[tuple[float, ...]] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    hands: int = 0

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "feats": np.stack(self.feats),
            "oracle": np.stack(self.oracle),
            "masks": np.stack(self.masks),
            "actions": np.array(self.actions, dtype=np.int64),
            "extra": np.array(self.extra, dtype=np.float32),
            "returns": np.array(self.returns, dtype=np.float32),
        }


Chooser = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, list[tuple[float, ...]]]]
"""(features, oracle, masks) -> (action codes, per-decision extra values to store)."""


def force_wins(actions: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Every agent declares a legal win; so does the learner."""
    for code in WIN_CODES:
        actions = np.where(masks[:, code] == 1, code, actions)
    return actions


def collect(
    env: HkVecEnv, choose: Chooser, hands: int, pending: dict[int, list[Any]] | None = None
) -> Batch:
    """Steps ``env`` until ``hands`` hands have finished; returns their decisions.

    ``pending`` holds unfinished hands' decisions between calls (pass the same
    dict every time)."""
    pending = pending if pending is not None else defaultdict(list)
    out = Batch()
    while out.hands < hands:
        fb, ob, mb, ids, seats = env.observe()
        f, o, m = to_arrays(fb, ob, mb)
        actions, extra = choose(f, o, m)
        actions = force_wins(actions, m)
        for k, i in enumerate(ids):
            pending[i].append(
                (
                    f[k].astype(np.float16),
                    o[k].astype(np.float16),
                    m[k],
                    int(actions[k]),
                    extra[k],
                    seats[k],
                )
            )
        env.step([int(a) for a in actions])
        for i, deltas, _seed in env.finished():
            for feat, orc, mask, a, ex, seat in pending.pop(i, []):
                out.feats.append(feat)
                out.oracle.append(orc)
                out.masks.append(mask)
                out.actions.append(a)
                out.extra.append(ex)
                out.returns.append(deltas[seat] / POINT_SCALE)
            out.hands += 1
    return out


def greedy_q(net: torch.nn.Module, eps: float, rng: np.random.Generator) -> Chooser:
    """ε-greedy on the Q head, without oracle inputs."""

    def choose(
        f: np.ndarray, o: np.ndarray, m: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[float, ...]]]:
        x = torch.from_numpy(np.concatenate([f, np.zeros_like(o)], axis=1))
        with torch.no_grad():
            q, _, _ = net(x)
        q = q.masked_fill(torch.from_numpy(m) == 0, -1e9).numpy()
        actions = q.argmax(1)
        explore = rng.random(len(actions)) < eps
        for k in np.flatnonzero(explore):
            actions[k] = rng.choice(np.flatnonzero(m[k]))
        return actions, [()] * len(actions)

    return choose
