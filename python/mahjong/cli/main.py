"""The ``mahjong`` command line."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mahjong import __version__
from mahjong._core import (
    HkState,
    agent_action,
    analyze,
    discard_table,
    format_tiles,
    hand_progress,
    parse_tiles,
    solo,
    tournament,
)
from mahjong.hk import (
    PASS,
    PATTERNS,
    Rules,
    action_chinese,
    action_text,
    describe_score,
    tile_text,
)

app = typer.Typer(help="Hong Kong mahjong solver and RL lab.", no_args_is_help=True)
hk = typer.Typer(help="Hong Kong mahjong: play, analyze, run tournaments.", no_args_is_help=True)
app.add_typer(hk, name="hk")
console = Console()

NoFlowers = Annotated[bool, typer.Option("--no-flowers", help="Play without bonus tiles.")]
MinFaan = Annotated[int, typer.Option(help="Minimum faan to win.")]
FaanCap = Annotated[int, typer.Option(help="Faan cap (limit).")]
Payout = Annotated[str, typer.Option(help="half_spicy or full_spicy.")]
Multi = Annotated[str, typer.Option(help="all or head_bump.")]
NoLiability = Annotated[bool, typer.Option("--no-liability", help="Turn off the 12-tile rule.")]


def make_rules(
    no_flowers: bool,
    min_faan: int,
    faan_cap: int,
    payout: str,
    multiple_winners: str,
    no_liability: bool,
) -> Rules:
    return Rules(
        flowers=not no_flowers,
        min_faan=min_faan,
        faan_cap=faan_cap,
        payout=payout,
        multiple_winners=multiple_winners,
        twelve_tile_liability=not no_liability,
    )


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def tiles(text: str) -> None:
    """Parse tile text and print it back sorted, e.g. ``mahjong tiles "R 9m 1m E"``."""
    try:
        typer.echo(format_tiles(parse_tiles(text)))
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the web UI and API."""
    import uvicorn

    uvicorn.run("mahjong.server.app:app", host=host, port=port)


@hk.command("tournament")
def tournament_cmd(
    agents: Annotated[
        list[str],
        typer.Argument(help="Four agent specs, e.g. value fast efficiency pimc:samples=32"),
    ],
    deals: int = 100,
    seed: int = 0,
    output: Annotated[Path | None, typer.Option("--json", help="Write results as JSON.")] = None,
    no_flowers: NoFlowers = False,
    min_faan: MinFaan = 3,
    faan_cap: FaanCap = 10,
    payout: Payout = "half_spicy",
    multiple_winners: Multi = "all",
    no_liability: NoLiability = False,
) -> None:
    """Duplicate tournament: every wall played four times, agents rotated."""
    if len(agents) != 4:
        raise typer.BadParameter("give exactly four agents")
    rules = make_rules(no_flowers, min_faan, faan_cap, payout, multiple_winners, no_liability)
    res = tournament(agents, deals, seed, **rules.kwargs())
    table = Table(title=f"{deals} walls × 4 seatings, draws {res['draws'] / res['hands']:.1%}")
    for col in ("agent", "points/hand", "±95%", "win", "self-drawn", "deal-in", "avg faan"):
        table.add_column(col, justify="right")
    for a in res["agents"]:
        n = a["hands"]
        table.add_row(
            a["agent"],
            f"{a['mean']:+.2f}",
            f"{a['ci95']:.2f}",
            f"{a['wins'] / n:.1%}",
            f"{a['self_drawn'] / n:.1%}",
            f"{a['deal_ins'] / n:.1%}",
            f"{a['win_faan'] / max(a['wins'], 1):.2f}",
        )
    console.print(table)
    if output:
        output.write_text(
            json.dumps({"rules": rules.kwargs(), "deals": deals, "seed": seed, **res}, indent=2)
        )


def show_position(state: HkState, seat: int) -> None:
    obs = state.observation(seat)
    winds = "東南西北"
    console.print(f"[dim]wall {obs['wall_remaining']}[/dim]")
    for p in range(4):
        melds = " ".join(
            format_tiles(m["tiles"]) + ("*" if m["concealed"] else "") for m in obs["melds"][p]
        )
        river = " ".join(tile_text(t) + ("'" if c else "") for s, t, c in obs["river"] if s == p)
        bonus = format_tiles(obs["bonus"][p]) if obs["bonus"][p] else ""
        who = "you" if p == seat else f"{obs['concealed_counts'][p]} tiles"
        console.print(
            f"{winds[p]} [{who}] melds: {melds or '-'}  bonus: {bonus or '-'}  river: {river}"
        )
    drawn = f"  (drew {tile_text(obs['last_drawn'])})" if obs["last_drawn"] is not None else ""
    console.print(f"[bold]hand: {format_tiles(obs['hand'])}[/bold]{drawn}")


def show_analysis(state: HkState, samples: int) -> None:
    a = analyze(state, samples=samples, candidates=6)
    table = Table(title=f"solver ({a['samples']} samples, {a['elapsed_ms']:.0f} ms)")
    for col in ("action", "points", "±95%", "win", "deal-in"):
        table.add_column(col, justify="right")
    for m in a["moves"][:3]:
        table.add_row(
            f"{m['text']} ({action_chinese(m['action'])})",
            f"{m['mean']:+.1f}",
            f"{m['ci95']:.1f}",
            f"{m['win_rate']:.0%}",
            f"{m['deal_in_rate']:.0%}",
        )
    console.print(table)


def parse_command(text: str, legal: list[int]) -> int | None:
    """Turns typed input into a legal action code."""
    text = text.strip().lower()
    names = {action_text(a).lower(): a for a in legal}
    if text in names:
        return names[text]
    words = {
        "p": PASS,
        "pass": PASS,
        "w": 104,
        "win": 104,
        "hu": 104,
        "tsumo": 34,
        "pung": 105,
        "kong": 106,
    }
    if text in words and words[text] in legal:
        return words[text]
    if text in ("w", "win") and 34 in legal:
        return 34
    try:
        tiles = parse_tiles(
            text.replace("discard", "").strip().title() if text[:1].isalpha() else text
        )
    except ValueError:
        return None
    if len(tiles) == 1 and tiles[0] in legal:
        return tiles[0]
    return None


@hk.command("play")
def play_cmd(
    bots: str = "pimc:samples=48,candidates=4",
    seed: int | None = None,
    hints: bool = False,
    seat: int = 0,
) -> None:
    """Play a hand against three bots. Type a tile to discard, or pung / kong /
    chow from 3s / win / pass; "?" asks the solver."""
    rules = Rules()
    seed = seed if seed is not None else random.randrange(2**31)
    state = rules.new_deal(seed)
    rng = random.Random(seed)
    while not state.is_terminal():
        p = state.current_player()
        legal = state.legal_actions()
        if p != seat:
            state.apply(agent_action(state, bots, rng.randrange(2**31)))
            continue
        console.rule()
        show_position(state, seat)
        if hints:
            show_analysis(state, 200)
        options = ", ".join(action_text(a) for a in legal if a >= 34) or "discard a tile"
        while True:
            typed = typer.prompt(f"your move [{options}]")
            if typed.strip() == "?":
                show_analysis(state, 400)
                continue
            code = parse_command(typed, legal)
            if code is not None:
                state.apply(code)
                break
            console.print("[red]not a legal move[/red]")
    result = state.result()
    assert result is not None
    console.rule("hand over")
    for w in result["wins"]:
        console.print(f"seat {w['seat']} wins: {describe_score(w['score'])}")
    if not result["wins"]:
        console.print("exhaustive draw")
    console.print(f"points: {result['deltas']}  (you are seat {seat})")


@hk.command("analyze")
def analyze_cmd(
    seed: int = 0,
    steps: int = 0,
    samples: int = 600,
    bots: str = "fast",
) -> None:
    """Deal hand ``seed``, let ``bots`` play ``steps`` decisions, then analyze the
    next decision: discard table, offense-only estimates and the solver."""
    state = Rules().new_deal(seed)
    rng = random.Random(seed)
    for _ in range(steps):
        if state.is_terminal():
            break
        state.apply(agent_action(state, bots, rng.randrange(2**31)))
    if state.is_terminal():
        raise typer.BadParameter("the hand ended before that many steps")
    seat = state.current_player()
    assert seat is not None
    show_position(state, seat)
    prog = hand_progress(state, seat)
    console.print(
        f"shanten {prog['shanten']}, faan-aware shanten {prog['faan_shanten']}, "
        f"target {prog['target']}"
    )
    if state.phase() == "turn":
        rows = {r["tile"]: r for r in discard_table(state)}
        offense = {r["tile"]: r for r in solo(state, samples=200)}
        table = Table(title="discards")
        for col in (
            "tile",
            "shanten",
            "accept",
            "faan-shanten",
            "faan-accept",
            "solo win",
            "solo faan",
        ):
            table.add_column(col, justify="right")
        for t, r in sorted(
            rows.items(), key=lambda kv: (kv[1]["faan_shanten"], -kv[1]["faan_ukeire"])
        ):
            o = offense[t]
            table.add_row(
                tile_text(t),
                str(r["shanten"]),
                str(r["ukeire"]),
                str(r["faan_shanten"]),
                str(r["faan_ukeire"]),
                f"{o['win_rate']:.0%}",
                f"{o['mean_faan']:.1f}",
            )
        console.print(table)
    show_analysis(state, samples)


@hk.command("patterns")
def patterns_cmd() -> None:
    """List the faan table."""
    for n, (zh, en) in PATTERNS.items():
        typer.echo(f"#{n:>2} {zh} {en}")
