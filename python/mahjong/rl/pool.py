"""The opponent pool of self-play against the three-stage bot (plan 2026-11-16 Task 5).

Every hand, each of the learner's three opponents is drawn on its own (the
orchestrator's call, 2026-11-16, so tables are mixed): the current network 50 %, a
random saved older network 30 %, the three-stage rule bot ``stage`` 20 %. While no
older network is saved, its share goes to the current network.

Older networks are kept in ``<run>/pool/`` (``member-<n>.pt`` and ``.mjnn``, listed in
``members.json``); :class:`Pool` adds one, loads them for play and lists them for the
2-hourly results. :func:`stop_reason` holds the plan's stop rules, :func:`pull_scale`
the fading pull toward the imitation network.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mahjong.rl.cnn import SuitCNN, export

#: Opponent kinds.
CURRENT, OLDER, STAGE = 0, 1, 2
#: Shares of the kinds (current, older, stage).
SHARES = (0.5, 0.3, 0.2)
#: The rule bot's agent spec (``stage`` = ``StageBot::new(3, 1)``, the baseline).
STAGE_SPEC = "stage"
#: Draw share above which the run stops (plan header, "Stalls").
DRAW_LIMIT = 0.15
#: Results in a row without a gain after which the run stops.
PATIENCE = 4


def draw_seats(
    rng: np.random.Generator, n: int, older: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Kinds of ``n`` opponent seats and, for ``OLDER`` seats, which of the ``older``
    saved networks plays (-1 for the others); each seat drawn independently."""
    kinds = rng.choice(3, size=n, p=SHARES).astype(np.int64)
    if older == 0:
        kinds[kinds == OLDER] = CURRENT
    member = np.full(n, -1, dtype=np.int64)
    k = kinds == OLDER
    member[k] = rng.integers(0, max(older, 1), size=int(k.sum()))
    return kinds, member


@dataclass
class Member:
    """A saved older network."""

    name: str
    hours: float
    iteration: int

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "hours": self.hours, "iteration": self.iteration}


class Pool:
    """The saved older networks of a run, in ``<folder>/pool``."""

    def __init__(self, folder: Path) -> None:
        self.dir = folder / "pool"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index = self.dir / "members.json"
        self.members: list[Member] = []
        if self.index.exists():
            self.members = [Member(**m) for m in json.loads(self.index.read_text())]
        self._nets: dict[str, SuitCNN] = {}

    def __len__(self) -> int:
        return len(self.members)

    def pt(self, i: int) -> Path:
        return self.dir / f"{self.members[i].name}.pt"

    def mjnn(self, i: int) -> Path:
        return self.dir / f"{self.members[i].name}.mjnn"

    def add(self, net: SuitCNN, hours: float, iteration: int) -> Member:
        """Saves a frozen copy of ``net`` as a new member (files first, then the list,
        so a crash in between leaves no member without files)."""
        m = Member(f"member-{len(self.members):03d}", round(hours, 4), iteration)
        net.save(self.dir / f"{m.name}.pt")
        export(net, self.dir / f"{m.name}.mjnn")
        self.members.append(m)
        tmp = self.index.with_suffix(".tmp")
        tmp.write_text(json.dumps([x.to_json() for x in self.members], indent=1))
        os.replace(tmp, self.index)
        return m

    def net(self, i: int, device: torch.device) -> SuitCNN:
        """Member ``i`` for play (loaded once, frozen, in eval mode)."""
        name = self.members[i].name
        if name not in self._nets:
            net = SuitCNN.load(self.pt(i)).to(device).eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self._nets[name] = net
        return self._nets[name]


def pull_scale(hours: float, fade_hours: float) -> float:
    """Factor of the pull toward the imitation network: 1 at the start, falling in a
    straight line to 0 at ``fade_hours`` of training (0 from then on)."""
    if fade_hours <= 0:
        return 0.0
    return max(0.0, 1.0 - hours / fade_hours)


def best_result(results: list[dict[str, Any]]) -> int | None:
    """Index of the result with the highest mean (the earliest of equals), or None."""
    if not results:
        return None
    return max(range(len(results)), key=lambda i: (results[i]["mean"], -i))


def stop_reason(results: list[dict[str, Any]], draw_limit: float = DRAW_LIMIT) -> str | None:
    """Why the run must stop after the latest of ``results`` (the 2-hourly results
    against three ``stage`` bots, oldest first: ``mean``, ``ci95``, ``draw_rate``,
    and optionally ``train_draw_rate``, the self-play draw share since the previous
    result), or None.

    * Draws: the latest result's draw share, or the self-play draw share, above
      ``draw_limit`` (15 %).
    * Collapse: each of the last two results lies below the best earlier result (before
      those two) by more than the sum of its own and that result's 95 % half-widths.
    * Stall: four results in a row none of which gains on the best earlier one, a
      gain being a mean above that result's mean + its 95 % half-width.
    """
    if not results:
        return None
    last = results[-1]
    for key, what in (
        ("draw_rate", "against three stage bots"),
        ("train_draw_rate", "in self-play"),
    ):
        x = last.get(key)
        if x is not None and x > draw_limit:
            return f"draws {x:.1%} {what}, above {draw_limit:.0%} (result {len(results)})"
    if len(results) >= 3:
        earlier = results[:-2]
        best = earlier[best_result(earlier) or 0]
        if all(r["mean"] < best["mean"] - (best["ci95"] + r["ci95"]) for r in results[-2:]):
            return (
                f"collapse: the last two results ({results[-2]['mean']:+.2f}, "
                f"{last['mean']:+.2f}) are below the best earlier one "
                f"({best['mean']:+.2f} +- {best['ci95']:.2f}) by more than the ranges"
            )
    best, since = results[0], 0
    for r in results[1:]:
        if r["mean"] > best["mean"] + best["ci95"]:
            best, since = r, 0
        else:
            since += 1
    if since >= PATIENCE:
        return (
            f"stalled: {since} results in a row without a gain on the best "
            f"({best['mean']:+.2f} +- {best['ci95']:.2f} per hand)"
        )
    return None
