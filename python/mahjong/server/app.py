"""FastAPI app: the play-against-bots API and the built UI (web/dist).

Run with ``mahjong serve`` (or ``uvicorn mahjong.server.app:app``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mahjong._core import analyze, danger, discard_table, hand_progress, solo
from mahjong.hk import PATTERNS, Rules, action_chinese, tile_chinese
from mahjong.server.session import LEVELS, Session, SessionStore

app = FastAPI(title="mahjong", version="0.1.0")
store = SessionStore()
WEB = Path(__file__).resolve().parents[3] / "web" / "dist"


class RulesIn(BaseModel):
    flowers: bool = True
    min_faan: int = Field(3, ge=0, le=10)
    faan_cap: int = Field(10, ge=1, le=13)
    payout: str = "half_spicy"
    multiple_winners: str = "all"
    twelve_tile_liability: bool = True
    prevailing_wind: int = Field(0, ge=0, le=3)


class NewGame(BaseModel):
    level: str = "medium"
    seat: int | None = Field(None, ge=0, le=3)
    seed: int | None = None
    rules: RulesIn = RulesIn()


class Act(BaseModel):
    action: int


def _session(sid: str) -> Session:
    try:
        return store.get(sid)
    except KeyError:
        raise HTTPException(404, "no such game") from None


def view(s: Session) -> dict[str, Any]:
    """Everything the human may see, plus labels for the UI."""
    obs = s.state.observation(s.human)
    legal = obs["legal"]
    result = s.state.result()
    out: dict[str, Any] = {
        "id": s.id,
        "seed": s.seed,
        "level": s.level,
        "human": s.human,
        "rules": s.rules.kwargs(),
        "observation": obs,
        "legal": [{"action": a, "label": action_chinese(a)} for a in legal],
        "bot_to_act": not s.state.is_terminal() and not s.human_to_act(),
        "events": s.state.events()[-12:],
        "result": result,
    }
    if result is not None:
        # At the end every hand is revealed.
        out["hands"] = [s.state.hand(p) for p in range(4)]
    return out


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    """Levels, faan-table names and tile names for the UI."""
    return {
        "levels": LEVELS,
        "patterns": {n: zh for n, (zh, _) in PATTERNS.items()},
        "tiles": [tile_chinese(t) for t in range(42)],
    }


@app.post("/api/games")
def new_game(body: NewGame) -> dict[str, Any]:
    try:
        rules = Rules(**body.rules.model_dump())
        s = store.create(rules, body.level, body.seat, body.seed)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return view(s)


@app.get("/api/games/{sid}")
def get_game(sid: str) -> dict[str, Any]:
    return view(_session(sid))


@app.post("/api/games/{sid}/act")
def act(sid: str, body: Act) -> dict[str, Any]:
    s = _session(sid)
    try:
        s.act(body.action)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return view(s)


@app.post("/api/games/{sid}/step")
def step(sid: str) -> dict[str, Any]:
    """Lets the next bot move once (the UI paces bots for readability)."""
    s = _session(sid)
    try:
        s.step_bot()
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return view(s)


@app.get("/api/games/{sid}/hints")
def hints(sid: str, ms: int = 2000, samples: int = 800) -> dict[str, Any]:
    """Study mode: the solver's top moves, danger per tile, hand progress, and
    (on a turn) the discard table with offense-only estimates."""
    s = _session(sid)
    if not s.human_to_act():
        raise HTTPException(400, "not your decision")
    a = analyze(s.state, samples=samples, candidates=6, time_ms=ms, seed=len(s.actions))
    for m in a["moves"]:
        m["label"] = action_chinese(m["action"])
    out: dict[str, Any] = {
        "solver": a,
        "danger": danger(s.state, s.human),
        "progress": hand_progress(s.state, s.human),
    }
    if s.state.phase() == "turn":
        offense = {r["tile"]: r for r in solo(s.state, samples=150, seed=len(s.actions))}
        rows = discard_table(s.state)
        for r in rows:
            r["solo"] = offense.get(r["tile"])
        out["discards"] = rows
    return out


@app.get("/api/games/{sid}/review")
def review(sid: str) -> dict[str, Any]:
    s = _session(sid)
    if not s.state.is_terminal():
        raise HTTPException(400, "the hand is not over")
    items = s.run_review()
    for it in items:
        it["chosen_label"] = action_chinese(it["chosen"])
        it["best"]["label"] = action_chinese(it["best"]["action"])
    return {"decisions": items}


if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        file = WEB / path
        return FileResponse(file if path and file.is_file() else WEB / "index.html")
