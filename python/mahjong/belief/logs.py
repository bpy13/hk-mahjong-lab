"""Reader for the self-play logs with hidden hands (``selfplay_log`` example).

A log directory holds ``meta.json`` and chunk files ``chunk-NNNNN.bin``. Each chunk
is a packed array of fixed-size rows whose layout ``meta.json`` lists (field name,
little-endian NumPy type, shape), so the record type is built from the file, never
hard-coded. One row is one decision, seen from the player to act (the observer);
seats inside a row are relative to the observer (0 = observer, 1 = next, 2 = across,
3 = previous), and the ``opp_*`` fields hold the hidden truth about relative seats
1, 2 and 3.

Example::

    from mahjong.belief.logs import LogDir
    log = LogDir("runs/logs/train")
    rows = log.chunk(0)                  # structured NumPy array (memory-mapped)
    x = features(rows)                   # float32, shape (n, 933)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

#: Feature planes (34 wide) at the start of ``features``; plane 16 is unseen / 4.
UNSEEN_PLANE = 16
NUM_KINDS = 34


def row_dtype(meta: dict[str, Any]) -> np.dtype[Any]:
    """The packed record type described by ``meta["fields"]``."""
    fields = [(f["name"], f["dtype"], tuple(f["shape"])) for f in meta["fields"]]
    dt = np.dtype(fields)
    if dt.itemsize != meta["row_bytes"]:
        raise ValueError(
            f"layout is {dt.itemsize} bytes per row, meta.json says {meta['row_bytes']}"
        )
    return dt


class LogDir:
    """A log directory: its metadata and its chunks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta: dict[str, Any] = json.loads((self.path / "meta.json").read_text("utf-8"))
        self.dtype = row_dtype(self.meta)

    @property
    def rows(self) -> int:
        """Rows in all complete chunks."""
        return int(self.meta["rows"])

    @property
    def agents(self) -> list[str]:
        """Agent specs; the ``agents`` field holds indices into this list."""
        return list(self.meta["agents"])

    def __len__(self) -> int:
        return len(self.meta["chunks"])

    def chunk(self, i: int) -> np.ndarray[Any, np.dtype[Any]]:
        """Chunk ``i`` as a read-only memory-mapped structured array."""
        name = self.meta["chunks"][i]["file"]
        return np.memmap(self.path / name, dtype=self.dtype, mode="r")

    def chunks(self) -> Iterator[np.ndarray[Any, np.dtype[Any]]]:
        """Every complete chunk, in order."""
        for i in range(len(self)):
            yield self.chunk(i)


def features(rows: np.ndarray[Any, np.dtype[Any]]) -> npt.NDArray[np.float32]:
    """The observer's public features as float32, shape ``(n, 933)``."""
    return np.asarray(rows["features"], dtype=np.float32)


def unseen_counts(rows: np.ndarray[Any, np.dtype[Any]]) -> npt.NDArray[np.int64]:
    """Copies of each tile kind the observer has not seen, shape ``(n, 34)``.

    Read from the features' unseen plane (count / 4, exact in float16).
    """
    plane = features(rows)[:, UNSEEN_PLANE * NUM_KINDS : (UNSEEN_PLANE + 1) * NUM_KINDS]
    return np.rint(plane * 4).astype(np.int64)
