"""Rows of the ``risk_log`` example (``crates/search/examples/risk_log.rs``).

A data directory holds ``meta.json`` (``row_bytes``, the ``fields`` in order as
``[name, dtype, shape]``, the ``chunks``) and ``chunk-NNN.bin`` files of fixed-size
records, each chunk holding whole deals. One row is one discard decision of one seat:
the network input (``features`` as float16, the packed ``history``, the ``risk`` block
times 4 as bytes), the true answer for every kind and opponent (``win``, ``points``;
opponents are relative seats 1 = next, 2 = across, 3 = previous), and what the three
existing sources say for every kind (``table_*``: Task 11's calibrated tables,
``exact_*``: Task 13's exact counting with the switch on, ``belief_*``: the belief
model), per tile combined over the opponents and per opponent.

Row layout 1 (Task 15) stores the six per-tile check columns as float32 and the
``risk`` block one value per byte; layout 2 (Task 16, ``"layout": 2`` in ``meta.json``)
stores them as float16 and packs two ``risk`` values per byte (value ``2i`` in the low
half of byte ``i``, ``2i + 1`` in the high half): 3,613 bytes a row against 4,288. Both
load; :func:`inputs` gives the same floats either way.

Example::

    d = RiskData("runs/risk/data")
    rows = d.chunk(0)
    x = inputs(rows)                 # float32, (n, STATE_LEN + HISTORY_LEN + RISK_LEN)
    win, points = targets(rows)      # (n, 3, 34) float32 each
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mahjong.rl.labels import HISTORY_BYTES, unpack_history

NUM_KINDS = 34
N_OPP = 3
STATE_LEN = 26 * NUM_KINDS + 49
HISTORY_LEN = 96 * NUM_KINDS + 2 * 96 + 4
#: ``risk_net::RISK_PLANES``, ``RISK_SCALARS``, ``RISK_LEN``.
RISK_PLANES = 5
RISK_SCALARS = 8
RISK_LEN = N_OPP * (RISK_PLANES * NUM_KINDS + RISK_SCALARS)
INPUT_LEN = STATE_LEN + HISTORY_LEN + RISK_LEN
#: The plane of ``features`` holding the unseen copies / 4 (``features.rs``).
UNSEEN_PLANE = 16

#: Bytes of the packed ``risk`` block in layout 2.
RISK_BYTES = (RISK_LEN + 1) // 2

#: The fields of one row, in order, as the Rust example wrote them in layout 1.
FIELDS: list[tuple[str, str, tuple[int, ...]]] = [
    ("deal", "<u4", ()),
    ("seat", "u1", ()),
    ("wall", "u1", ()),
    ("thrown", "u1", ()),
    ("hit", "u1", ()),
    ("concealed", "u1", (3,)),
    ("shares", "u1", (3,)),
    ("is_exact", "u1", (3,)),
    ("pad", "u1", (3,)),
    ("hand", "u1", (NUM_KINDS,)),
    ("features", "<f2", (STATE_LEN,)),
    ("history", "u1", (HISTORY_BYTES,)),
    ("risk", "u1", (RISK_LEN,)),
    ("win", "u1", (3, NUM_KINDS)),
    ("points", "u1", (3, NUM_KINDS)),
    ("table_tile", "<f4", (NUM_KINDS,)),
    ("exact_tile", "<f4", (NUM_KINDS,)),
    ("belief_tile", "<f4", (NUM_KINDS,)),
    ("table_loss", "<f4", (NUM_KINDS,)),
    ("exact_loss", "<f4", (NUM_KINDS,)),
    ("belief_loss", "<f4", (NUM_KINDS,)),
    ("table_p", "<f2", (3, NUM_KINDS)),
    ("belief_p", "<f2", (3, NUM_KINDS)),
    ("exact_p", "<f2", (3, NUM_KINDS)),
    ("exact_points", "u1", (3, NUM_KINDS)),
]

#: Layout 2: the same fields, the check columns as float16 and ``risk`` packed.
FIELDS_V2: list[tuple[str, str, tuple[int, ...]]] = [
    (n, "<f2" if d == "<f4" else d, (RISK_BYTES,) if n == "risk" else s) for n, d, s in FIELDS
]

Rows = np.ndarray[Any, np.dtype[Any]]


def row_dtype(fields: list[tuple[str, str, tuple[int, ...]]] | None = None) -> np.dtype[Any]:
    """The packed record dtype of one row."""
    spec = [(n, d, tuple(s)) if s else (n, d) for n, d, s in (fields or FIELDS)]
    return np.dtype(spec)


class RiskData:
    """A data directory: ``meta.json`` and its chunks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with open(self.path / "meta.json", encoding="utf-8") as f:
            self.meta: dict[str, Any] = json.load(f)
        fields = [(str(n), str(d), tuple(int(x) for x in s)) for n, d, s in self.meta["fields"]]
        self.dtype = row_dtype(fields)
        self.layout = int(self.meta.get("layout", 1))
        if self.layout not in (1, 2):
            raise ValueError(f"row layout {self.layout}: this reader knows 1 and 2")
        if self.dtype.itemsize != int(self.meta["row_bytes"]):
            raise ValueError(
                f"row dtype has {self.dtype.itemsize} bytes, meta says {self.meta['row_bytes']}"
            )

    @property
    def chunks(self) -> list[int]:
        """Chunk numbers present, in order."""
        return sorted(int(c["chunk"]) for c in self.meta["chunks"])

    def rows_in(self, chunk: int) -> int:
        return next(int(c["rows"]) for c in self.meta["chunks"] if int(c["chunk"]) == chunk)

    def chunk(self, chunk: int) -> Rows:
        """The rows of one chunk (a read-only memory map)."""
        path = self.path / f"chunk-{chunk:03d}.bin"
        return np.memmap(path, dtype=self.dtype, mode="r")

    def load(self, chunks: list[int]) -> Rows:
        """The rows of several chunks, concatenated in memory."""
        return np.concatenate([np.asarray(self.chunk(c)) for c in chunks])


def risk_block(rows: Rows) -> npt.NDArray[np.float32]:
    """The ``risk`` block as floats, ``(n, RISK_LEN)``, from either layout (told apart
    by the stored width: one byte per value, or two values per byte)."""
    raw = np.asarray(rows["risk"], dtype=np.uint8)
    if raw.shape[1] == RISK_LEN:
        return raw.astype(np.float32) / 4.0
    if raw.shape[1] != RISK_BYTES:
        raise ValueError(f"risk block of {raw.shape[1]} bytes: expected {RISK_LEN} or {RISK_BYTES}")
    out = np.empty((raw.shape[0], 2 * RISK_BYTES), dtype=np.uint8)
    out[:, 0::2] = raw & 15
    out[:, 1::2] = raw >> 4
    return out[:, :RISK_LEN].astype(np.float32) / 4.0


def inputs(rows: Rows) -> npt.NDArray[np.float32]:
    """The network input, ``(n, INPUT_LEN)`` float32: features, history, risk block."""
    n = len(rows)
    x = np.empty((n, INPUT_LEN), dtype=np.float32)
    x[:, :STATE_LEN] = rows["features"].astype(np.float32)
    x[:, STATE_LEN : STATE_LEN + HISTORY_LEN] = unpack_history(np.asarray(rows["history"]))
    x[:, STATE_LEN + HISTORY_LEN :] = risk_block(rows)
    return x


def targets(rows: Rows) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """``win`` (0 or 1) and ``points`` per share, each ``(n, 3, 34)`` float32."""
    return (
        np.asarray(rows["win"], dtype=np.float32),
        np.asarray(rows["points"], dtype=np.float32),
    )


def alive(rows: Rows) -> npt.NDArray[np.bool_]:
    """Kinds with at least one unseen copy, ``(n, 34)`` (the others cannot deal in)."""
    unseen = rows["features"][:, UNSEEN_PLANE * NUM_KINDS : (UNSEEN_PLANE + 1) * NUM_KINDS]
    return np.asarray(unseen, dtype=np.float32) > 0.0
