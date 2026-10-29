"""A game against three bots, as kept by the server.

The session stores, for every decision the human makes, a copy of the state
before it and the action chosen, so the post-hand review can re-analyze them.
Finished hands are written to ``runs/games/<id>.json`` (seed, rules, actions).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mahjong import benchmark
from mahjong._core import HkState, agent_action, analyze
from mahjong.hk import Rules

#: Opponent levels shown in the UI. Bots think about 0.5 s per decision at most.
LEVELS: dict[str, dict[str, str]] = {
    "easy": {"label": "初級", "spec": "efficiency", "about": "只求快糊，不計番數"},
    "medium": {"label": "中級", "spec": "defensive", "about": "計番數，識防守"},
    "hard": {
        "label": "高級",
        "spec": "pimc:samples=96,candidates=4,ms=500",
        "about": "蒙地卡羅搜尋",
    },
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
    #: Serializes changes to the state: requests run in a thread pool, and a
    #: hints request may still be computing on a copy when the player moves.
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: Benchmark hands (plan Task 12): the wall index, the fixed wall and the
    #: opponents' spec. Such a hand has no hints or review, seeds every bot move
    #: from the wall index and the actions so far, and logs after every action.
    benchmark_index: int | None = None
    wall: list[int] | None = None
    opponents: str | None = None

    def __post_init__(self) -> None:
        if self.wall is not None:
            self.state = HkState.from_wall(self.wall, **self.rules.kwargs())
        else:
            self.state = self.rules.new_deal(self.seed)
        self._rng = random.Random(self.seed ^ 0x5EED)

    @property
    def tag(self) -> str | None:
        return benchmark.TAG if self.benchmark_index is not None else None

    @property
    def bot_spec(self) -> str:
        if self.opponents is not None:
            return self.opponents
        return LEVELS[self.level]["spec"]

    def _bot_seed(self) -> int:
        if self.benchmark_index is not None:
            return benchmark.decision_seed(self.benchmark_index, self.actions)
        return self._rng.randrange(2**63)

    def replay(self, actions: list[list[int]]) -> None:
        """Re-applies logged ``(seat, action)`` pairs (resuming an interrupted hand)."""
        with self.lock:
            for p, a in actions:
                if p == self.human:
                    self.decisions.append(Decision(self.state.copy(), a))
                self.state.apply(a)
                self.actions.append((p, a))

    def human_to_act(self) -> bool:
        return self.state.current_player() == self.human

    def act(self, action: int) -> None:
        """Applies the human's action (raises ``ValueError`` if illegal)."""
        with self.lock:
            if not self.human_to_act():
                raise ValueError("not your turn")
            before = self.state.copy()
            self.state.apply(action)
            self.decisions.append(Decision(before, action))
            self.actions.append((self.human, action))
            self._maybe_save()

    def snapshot(self) -> HkState:
        """A copy of the current state, for slow analysis outside the lock."""
        with self.lock:
            return self.state.copy()

    def step_bot(self) -> int:
        """Lets the bot to act move once; returns its action."""
        with self.lock:
            p = self.state.current_player()
            if p is None or p == self.human:
                raise ValueError("no bot to move")
            a = agent_action(self.state, self.bot_spec, self._bot_seed())
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
            best = moves[a["choice"]]
            chosen = moves.get(d.action)
            out.append(
                {
                    "index": i,
                    "phase": d.state.phase(),
                    "observation": d.state.observation(self.human),
                    "chosen": d.action,
                    "best": best,
                    "chosen_stats": chosen,
                    # Paired difference: both moves were played on the same samples.
                    "loss": best["diff"] - chosen["diff"] if chosen else None,
                }
            )
        self.review = out
        return out

    def _maybe_save(self) -> None:
        if self.benchmark_index is not None:
            self._save_benchmark()
            return
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

    def _save_benchmark(self) -> None:
        """Benchmark hands are logged after every action, so an interrupted hand
        resumes from its log (``result`` stays ``None`` until the hand ends)."""
        assert self.benchmark_index is not None
        benchmark.write_hand(
            {
                "format": benchmark.HAND_FORMAT,
                "tag": benchmark.TAG,
                "index": self.benchmark_index,
                "seat": self.human,
                "opponents": self.bot_spec,
                "rules": self.rules.kwargs(),
                "id": self.id,
                "actions": self.actions,
                "result": self.state.result(),
                "started": self.created,
                "finished": time.time() if self.state.is_terminal() else None,
            }
        )


class SessionStore:
    """In-memory sessions, oldest dropped beyond ``limit``."""

    def __init__(self, limit: int = 200) -> None:
        self.limit = limit
        self._sessions: dict[str, Session] = {}
        #: Two quick clicks on 基準測試 must not deal the same wall twice.
        self._benchmark_lock = threading.Lock()

    def create(self, rules: Rules, level: str, human: int | None, seed: int | None) -> Session:
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}")
        seed = seed if seed is not None else random.randrange(2**31)
        human = human if human is not None else random.Random(seed).randrange(4)
        s = Session(rules=rules, level=level, human=human, seed=seed)
        return self._add(s)

    def _add(self, s: Session) -> Session:
        self._sessions[s.id] = s
        while len(self._sessions) > self.limit:
            oldest = min(self._sessions.values(), key=lambda x: x.created)
            del self._sessions[oldest.id]
        return s

    def benchmark_next(self) -> Session | None:
        """The next benchmark hand (``None`` once all are played): the live session
        if the hand is under way, else rebuilt from its log, else a new deal."""
        with self._benchmark_lock:
            index = benchmark.next_index()
            if index is None:
                return None
            log = benchmark.load_hand(index)
            for s in self._sessions.values():
                # The log names its session; a stale session (another hands
                # directory, a deleted log) is never reused.
                if s.benchmark_index == index and log and log["id"] == s.id:
                    return s
            walls = benchmark.load_walls()
            entry = walls["walls"][index]
            s = Session(
                rules=benchmark.rules_of(walls),
                level=benchmark.TAG,
                human=entry["seat"],
                seed=walls["seed"],
                benchmark_index=index,
                wall=entry["wall"],
                opponents=log["opponents"] if log else benchmark.load_config()["opponents"],
            )
            if log:
                s.id = log["id"]
                s.created = log["started"]
                s.replay(log["actions"])
            else:
                s._save_benchmark()  # marks the wall as started
            return self._add(s)

    def get(self, sid: str) -> Session:
        return self._sessions[sid]
