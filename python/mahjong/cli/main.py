"""The ``mahjong`` command line. Game commands arrive in Phase 1."""

import typer

from mahjong import __version__
from mahjong._core import format_tiles, parse_tiles

app = typer.Typer(help="Hong Kong mahjong solver and RL lab.", no_args_is_help=True)


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
