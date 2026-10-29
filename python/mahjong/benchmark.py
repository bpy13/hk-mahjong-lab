"""Brian's 400-hand human benchmark (decision B21, plan Task 12).

Brian plays 400 single hands on fixed walls against three bots of one spec. Each
hand is logged; afterwards any agent can replay the same walls from Brian's seat
against the same opponents, giving a paired margin (agent minus Brian, per hand),
and the positions where Brian and an agent disagree can be exported for labelling.

* **Walls** (``runs/benchmark/walls.json``, committed): 400 walls in draw order,
  shuffled from seed 2026; Brian sits in seat ``index % 4``.
* **Config** (``runs/benchmark/config.json``, committed): the opponents' spec
  (default ``value``).
* **Hand logs** (``runs/benchmark/hands/<index>.json``, gitignored: Brian's data):
  written after every action, so an interrupted hand resumes where it stopped.
* **Bot seeds**: every bot decision draws its generator seed from the wall index and
  the actions so far (:func:`decision_seed`), the app-api's rule. A replay therefore
  meets exactly the bot moves Brian met until the first point where the agent and
  Brian differ.

``MAHJONG_BENCHMARK_DIR`` moves the walls and config; ``MAHJONG_BENCHMARK_HANDS_DIR``
moves the hand logs (the browser test uses it so it never touches Brian's logs).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path
from typing import Any

from mahjong._core import HkState, agent_action
from mahjong.hk import Rules, action_chinese

TOTAL = 400
SEED = 2026
TAG = "benchmark"
WALLS_FORMAT = "hk-benchmark-walls/1"
HAND_FORMAT = "hk-benchmark-hand/1"
REPLAY_FORMAT = "hk-benchmark-replay/1"
LABELS_FORMAT = "hk-benchmark-labels/1"
DEFAULT_CONFIG: dict[str, Any] = {"opponents": "value"}

_REPO = Path(__file__).resolve().parents[2]


def benchmark_dir() -> Path:
    """Where ``walls.json`` and ``config.json`` live."""
    return Path(os.environ.get("MAHJONG_BENCHMARK_DIR", _REPO / "runs" / "benchmark"))


def hands_dir() -> Path:
    """Where the hand logs go (Brian's data; gitignored)."""
    default = benchmark_dir() / "hands"
    return Path(os.environ.get("MAHJONG_BENCHMARK_HANDS_DIR", default))


# ---------------------------------------------------------------------------
# Walls, seats, config


def seat_for(index: int) -> int:
    """Brian's seat on wall ``index``: every seat equally often, dealer every 4th hand."""
    return index % 4


def generate_walls(n: int = TOTAL, seed: int = SEED, rules: Rules | None = None) -> dict[str, Any]:
    """``n`` walls in draw order (front first; replacements from the end).

    Each wall is the full tile set shuffled by Python's ``random.Random`` seeded with
    ``"<seed>:<index>"``, so any wall can be regenerated on its own.
    """
    rules = rules or Rules()
    tiles = [t for t in range(34) for _ in range(4)]
    if rules.flowers:
        tiles += list(range(34, 42))
    walls = []
    for i in range(n):
        wall = list(tiles)
        random.Random(f"{seed}:{i}").shuffle(wall)
        walls.append({"index": i, "seat": seat_for(i), "wall": wall})
    return {"format": WALLS_FORMAT, "seed": seed, "rules": rules.kwargs(), "walls": walls}


def write_walls(path: Path | None = None) -> Path:
    """Writes the 400 benchmark walls (one wall per line, so diffs stay readable)."""
    path = path or benchmark_dir() / "walls.json"
    data = generate_walls()
    head = {k: v for k, v in data.items() if k != "walls"}
    lines = [json.dumps(w, separators=(",", ":")) for w in data["walls"]]
    text = json.dumps(head)[:-1] + ', "walls": [\n' + ",\n".join(lines) + "\n]}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, "utf-8")
    return path


def load_walls(path: Path | None = None) -> dict[str, Any]:
    """The committed walls file."""
    path = path or benchmark_dir() / "walls.json"
    data: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if data.get("format") != WALLS_FORMAT:
        raise ValueError(f"{path} is not a {WALLS_FORMAT} file")
    return data


def load_config(path: Path | None = None) -> dict[str, Any]:
    """The benchmark config (opponents' spec); defaults if the file is missing."""
    path = path or benchmark_dir() / "config.json"
    config = dict(DEFAULT_CONFIG)
    if path.exists():
        config.update(json.loads(path.read_text("utf-8")))
    return config


def rules_of(walls: dict[str, Any]) -> Rules:
    return Rules(**walls["rules"])


def new_state(walls: dict[str, Any], index: int) -> HkState:
    """A fresh deal from wall ``index``."""
    return HkState.from_wall(walls["walls"][index]["wall"], **walls["rules"])


def decision_seed(index: int, actions: list[tuple[int, int]] | list[list[int]]) -> int:
    """The generator seed for the next decision: a hash of the wall index and every
    ``(seat, action)`` so far. Equal histories give equal seeds."""
    text = f"{SEED}:{index}:" + ";".join(f"{p},{a}" for p, a in actions)
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") >> 1


# ---------------------------------------------------------------------------
# Hand logs and session order


def hand_path(index: int) -> Path:
    return hands_dir() / f"{index:03d}.json"


def load_hand(index: int) -> dict[str, Any] | None:
    path = hand_path(index)
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return data


def write_hand(log: dict[str, Any]) -> None:
    path = hand_path(log["index"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(log), "utf-8")
    tmp.replace(path)


def finished_hands() -> list[dict[str, Any]]:
    """Every finished hand log, in wall order."""
    out = []
    for i in range(TOTAL):
        log = load_hand(i)
        if log is not None and log.get("result") is not None:
            out.append(log)
    return out


def next_index() -> int | None:
    """Session order: the lowest wall whose hand is not finished (an interrupted
    hand comes back first); ``None`` once all 400 are played."""
    for i in range(TOTAL):
        log = load_hand(i)
        if log is None or log.get("result") is None:
            return i
    return None


def progress() -> dict[str, Any]:
    """Played hands out of 400 and the next wall."""
    played = len(finished_hands())
    nxt = next_index()
    started = nxt is not None and load_hand(nxt) is not None
    return {
        "played": played,
        "total": TOTAL,
        "next": nxt,
        "in_progress": nxt if started else None,
        "opponents": load_config()["opponents"],
    }


# ---------------------------------------------------------------------------
# Replay and paired margin


def play_hand(
    walls: dict[str, Any],
    index: int,
    seat: int,
    agent: str,
    opponents: str,
    reference: list[list[int]] | None = None,
) -> dict[str, Any]:
    """Plays wall ``index`` with ``agent`` in ``seat`` and ``opponents`` elsewhere,
    each decision seeded by :func:`decision_seed`.

    With ``reference`` (the logged hand's actions) it also reports the first
    decision where the agent's move differs from the logged one.
    """
    state = new_state(walls, index)
    actions: list[tuple[int, int]] = []
    diverged: int | None = None
    while not state.is_terminal():
        p = state.current_player()
        assert p is not None
        spec = agent if p == seat else opponents
        a = agent_action(state, spec, decision_seed(index, actions))
        k = len(actions)
        if (
            reference is not None
            and diverged is None
            and (k >= len(reference) or tuple(reference[k]) != (p, a))
        ):
            diverged = k
        state.apply(a)
        actions.append((p, a))
    if reference is not None and diverged is None and len(actions) != len(reference):
        diverged = len(actions)
    result = state.result()
    assert result is not None
    return {"actions": actions, "result": result, "diverged_at": diverged}


def margin_stats(margins: list[float]) -> dict[str, float | int]:
    """Mean of the per-hand margins with a 95% normal interval (half-width)."""
    n = len(margins)
    mean = statistics.fmean(margins) if margins else 0.0
    sd = statistics.stdev(margins) if n > 1 else 0.0
    return {"n": n, "mean": mean, "ci95": 1.96 * sd / math.sqrt(n) if n else 0.0, "sd": sd}


def replay(agent: str, logs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Replays every finished benchmark hand with ``agent`` in Brian's seat.

    Each hand's margin is the agent's points minus Brian's points on the same wall
    and seat, against the same opponents' spec as in Brian's hand.
    """
    walls = load_walls()
    logs = finished_hands() if logs is None else logs
    hands = []
    for log in logs:
        seat = log["seat"]
        out = play_hand(walls, log["index"], seat, agent, log["opponents"], log["actions"])
        human = log["result"]["deltas"][seat]
        bot = out["result"]["deltas"][seat]
        hands.append(
            {
                "index": log["index"],
                "seat": seat,
                "human": human,
                "agent": bot,
                "margin": bot - human,
                "diverged_at": out["diverged_at"],
            }
        )
    stats = margin_stats([h["margin"] for h in hands])
    return {
        "format": REPLAY_FORMAT,
        "agent": agent,
        "hands": hands,
        "human_mean": statistics.fmean(h["human"] for h in hands) if hands else 0.0,
        "agent_mean": statistics.fmean(h["agent"] for h in hands) if hands else 0.0,
        "margin": stats,
    }


# ---------------------------------------------------------------------------
# Label export


def position_snapshot(state: HkState) -> dict[str, Any]:
    """The ``HkState.from_position`` keyword arguments for ``state`` (all four
    hands, melds, river, wall count, the player to move and the claim window).

    ``kong_chain`` is left at its default 0: the observation does not carry it, so a
    position right after a kong replacement loses the kong-on-kong bonus."""
    to_move = state.current_player()
    if to_move is None:
        raise ValueError("the hand is over")
    # The mover's own view: public tiles are the same for everyone, and it
    # carries the mover's last drawn tile.
    obs = state.observation(to_move)
    claim = obs["claim"]
    return {
        "hands": [state.hand(p) for p in range(4)],
        "melds": [
            [(m["kind"], m["tile"], m["concealed"], m["from"]) for m in seat]
            for seat in obs["melds"]
        ],
        "river": [tuple(r) for r in obs["river"]],
        "wall_remaining": obs["wall_remaining"],
        "to_move": to_move,
        "phase": state.phase(),
        "claim_tile": claim[0] if claim else None,
        "claim_from": claim[1] if claim else None,
        "last_drawn": obs["last_drawn"],
        "bonus": obs["bonus"],
        "liable": obs["liable"],
    }


def position_from_json(snapshot: dict[str, Any], rules: dict[str, Any]) -> HkState:
    """Rebuilds a state from an exported snapshot (JSON turned its tuples into lists)."""
    kwargs = dict(snapshot)
    kwargs["melds"] = [[tuple(m) for m in seat] for seat in snapshot["melds"]]
    kwargs["river"] = [tuple(r) for r in snapshot["river"]]
    return HkState.from_position(**kwargs, **rules)


def labels(agent: str, logs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The decisions where Brian's move and ``agent``'s first choice differ.

    Walks each logged hand as played; at each of Brian's decisions with more than
    one legal action it asks ``agent`` (seeded as in :func:`replay`) and keeps the
    position when the choices differ.
    """
    walls = load_walls()
    logs = finished_hands() if logs is None else logs
    items = []
    decisions = 0
    for log in logs:
        index, seat = log["index"], log["seat"]
        state = new_state(walls, index)
        rules = walls["rules"]
        done: list[tuple[int, int]] = []
        for p, a in log["actions"]:
            if p == seat and len(state.legal_actions()) > 1:
                decisions += 1
                choice = agent_action(state, agent, decision_seed(index, done))
                if choice != a:
                    items.append(
                        {
                            "index": index,
                            "decision": len(done),
                            "seat": seat,
                            "position": position_snapshot(state),
                            "rules": rules,
                            "human": a,
                            "human_label": action_chinese(a),
                            "agent_move": choice,
                            "agent_label": action_chinese(choice),
                        }
                    )
            state.apply(a)
            done.append((p, a))
    return {
        "format": LABELS_FORMAT,
        "agent": agent,
        "hands": len(logs),
        "decisions": decisions,
        "positions": items,
    }
