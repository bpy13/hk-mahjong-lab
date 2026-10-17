"""A game against three bots, as kept by the server.

The session stores, for every decision the human makes, a copy of the state
before it and the action chosen, so the post-hand review can re-analyze them.
Finished hands are written to ``runs/games/<id>.json`` (seed, rules, actions).
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mahjong._core import HkState, agent_action, analyze
from mahjong.hk import Rules

#: Opponent levels shown in the UI. Bots think about 0.5 s per decision at most.
LEVELS: dict[str, dict[str, str]] = {
    "easy": {"label": "初級", "spec": "efficiency", "about": "只求快糊，不計番數"},
    "medium": {"label": "中級", "spec": "defensive", "about": "計番數，識防守"},
    "hard": {"label": "高級", "spec": "pimc:samples=96,candidates=4,ms=500", "about": "蒙地卡羅搜尋"},
}

LOG_DIR = Path(os.environ.get("MAHJONG_GAME_LOG_DIR", "runs/games"))


@dataclass
class Decision:
    """A human decision: the state before it and the action taken."""

    state: HkState
    action: int


@dataclass
class Session:
    rules: Rules
    level: str
    human: int
    seed: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: HkState = field(init=False)
    actions: list[tuple[int, int]] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    review: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.state = self.rules.new_deal(self.seed)
        self._rng = random.Random(self.seed ^ 0x5EED)

    @property
    def bot_spec(self) -> str:
        return LEVELS[self.level]["spec"]

    def human_to_act(self) -> bool:
        return self.state.current_player() == self.human

    def act(self, action: int) -> None:
        """Applies the human's action (raises ``ValueError`` if illegal)."""
        if not self.human_to_act():
            raise ValueError("not your turn")
        before = self.state.copy()
        self.state.apply(action)
        self.decisions.append(Decision(before, action))
        self.actions.append((self.human, action))
        self._maybe_save()

    def step_bot(self) -> int:
        """Lets the bot to act move once; returns its action."""
        p = self.state.current_player()
        if p is None or p == self.human:
            raise ValueError("no bot to move")
        a = agent_action(self.state, self.bot_spec, self._rng.randrange(2**63))
        self.state.apply(a)
        self.actions.append((p, a))
        self._maybe_save()
        return a

    def run_review(self, samples: int = 300) -> list[dict[str, Any]]:
        """Re-analyzes each human decision that had a choice."""
        if self.review is not None:
            return self.review
        out = []
        for i, d in enumerate(self.decisions):
            legal = d.state.legal_actions()
            if len(legal) < 2:
                continue
            a = analyze(d.state, samples=samples, candidates=8, seed=i)
            moves = {m["action"]: m for m in a["moves"]}
            best = a["moves"][0]
            chosen = moves.get(d.action)
            out.append(
                {
                    "index": i,
                    "phase": d.state.phase(),
                    "observation": d.state.observation(self.human),
                    "chosen": d.action,
                    "best": best,
                    "chosen_stats": chosen,
                    "loss": best["mean"] - chosen["mean"] if chosen else None,
                }
            )
        self.review = out
        return out

    def _maybe_save(self) -> None:
        if not self.state.is_terminal():
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log = {
            "id": self.id,
            "seed": self.seed,
            "rules": self.rules.kwargs(),
            "level": self.level,
            "human": self.human,
            "actions": self.actions,
            "result": self.state.result(),
            "finished": time.time(),
        }
        (LOG_DIR / f"{self.id}.json").write_text(json.dumps(log))


class SessionStore:
    """In-memory sessions, oldest dropped beyond ``limit``."""

    def __init__(self, limit: int = 200) -> None:
        self.limit = limit
        self._sessions: dict[str, Session] = {}

    def create(self, rules: Rules, level: str, human: int | None, seed: int | None) -> Session:
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}")
        seed = seed if seed is not None else random.randrange(2**31)
        human = human if human is not None else random.Random(seed).randrange(4)
        s = Session(rules=rules, level=level, human=human, seed=seed)
        self._sessions[s.id] = s
        while len(self._sessions) > self.limit:
            oldest = min(self._sessions.values(), key=lambda x: x.created)
            del self._sessions[oldest.id]
        return s

    def get(self, sid: str) -> Session:
        return self._sessions[sid]
