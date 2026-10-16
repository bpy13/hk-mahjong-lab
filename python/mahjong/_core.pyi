"""Type stubs for the Rust extension module `mahjong._core`.

A tile is an ``int`` index (docs/rules/hk.md §1): ``1m``–``9m`` = 0–8,
``1p``–``9p`` = 9–17, ``1s``–``9s`` = 18–26, ``E S W N`` = 27–30, ``Wh G R`` = 31–33,
``F1``–``F4`` = 34–37, ``S1``–``S4`` = 38–41.
"""

__version__: str

def parse_tiles(text: str) -> list[int]:
    """Parse tile text such as ``"123m 55p EEE Wh F2"``. Raises ``ValueError``."""

def format_tiles(tiles: list[int]) -> str:
    """Format tile indices in sorted, grouped notation (``"123m 55p EEE"``)."""
