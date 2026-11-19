"""The risk network: a multilayer perceptron from the public input to, per opponent and
tile kind, the logit that a discard of the kind wins for that opponent now (``win``,
3 × 34) and ``log2(points / 8)`` of such a win (``pay``, 3 × 34).

:meth:`RiskNet.export` writes the MJNN v2 file ``crates/search/src/risk_net.rs`` loads:
input slices ``features``, ``history``, ``risk``; output slices ``win``, ``pay``. The
Rust side applies the sigmoid, forces 0 on kinds with no unseen copy, and turns ``pay``
into points (``8 × 2^pay``, within the rules' range), as :meth:`RiskNet.predict` does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mahjong.risk.data import HISTORY_LEN, INPUT_LEN, N_OPP, NUM_KINDS, RISK_LEN, STATE_LEN
from mahjong.rl.model import write_mjnn

HEAD_LEN = N_OPP * NUM_KINDS
OUTPUT_LEN = 2 * HEAD_LEN
BASE_POINTS = 8.0
MAX_LOG_PAY = 4.0


@dataclass
class RiskConfig:
    hidden: tuple[int, ...] = (768, 384)
    input_len: int = INPUT_LEN

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Losses:
    win: torch.Tensor
    pay: torch.Tensor
    total: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.total = self.win + 0.25 * self.pay


class RiskNet(nn.Module):
    def __init__(self, cfg: RiskConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RiskConfig()
        sizes = [self.cfg.input_len, *self.cfg.hidden, OUTPUT_LEN]
        self.linears = nn.ModuleList(
            [nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:], strict=True)]
        )
        # Start at the base rate: about 0.4 % of (opponent, kind) pairs win, and a
        # win pays about 8 × 2^0.7 points. Adam moves a bias by about the learning
        # rate per step, so a start at 0.5 would cost thousands of steps.
        last = self.linears[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.bias[:HEAD_LEN].fill_(-5.5)
            last.bias[HEAD_LEN:].fill_(0.7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i + 1 < len(self.linears):
                x = F.relu(x)
        return x

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def split(out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``win`` logits and ``pay`` values, each ``(n, 3, 34)``."""
        return (
            out[:, :HEAD_LEN].reshape(-1, N_OPP, NUM_KINDS),
            out[:, HEAD_LEN:].reshape(-1, N_OPP, NUM_KINDS),
        )

    def predict(
        self, x: torch.Tensor, alive: torch.Tensor, max_points: float = 128.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Probabilities and points per share, each ``(n, 3, 34)``; dead kinds
        (``alive`` false, ``(n, 34)``) get 0 and 0, as the Rust side does."""
        logits, pay = self.split(self(x))
        p = torch.sigmoid(logits) * alive[:, None, :]
        points = (BASE_POINTS * torch.exp2(pay.clamp(0.0, MAX_LOG_PAY))).clamp(
            BASE_POINTS, max_points
        ) * alive[:, None, :]
        return p, points

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg.to_dict(), "state": self.state_dict()}, path)

    @staticmethod
    def load(path: Path) -> RiskNet:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        net = RiskNet(RiskConfig(hidden=tuple(cfg["hidden"]), input_len=int(cfg["input_len"])))
        net.load_state_dict(ckpt["state"])
        return net

    def export(self, path: Path, named_risk: bool = True) -> None:
        """Writes the MJNN v2 file. With ``named_risk`` false the ``risk`` block is left
        unnamed (the first layer is still as wide): for running the network through an
        older Rust binding that does not know the name."""
        inputs = [("features", STATE_LEN), ("history", HISTORY_LEN)]
        if named_risk:
            inputs.append(("risk", RISK_LEN))
        linears = [lin for lin in self.linears if isinstance(lin, nn.Linear)]
        write_mjnn(path, linears, inputs=inputs, outputs=[("win", HEAD_LEN), ("pay", HEAD_LEN)])


def losses(
    out: torch.Tensor,
    win: torch.Tensor,
    points: torch.Tensor,
    alive: torch.Tensor,
) -> Losses:
    """Binary cross-entropy of ``win`` over the alive kinds (mean per alive kind) and the
    squared error of ``pay`` against ``log2(points / 8)`` over the kinds that win."""
    logits, pay = RiskNet.split(out)
    mask = alive[:, None, :].expand_as(win).float()
    bce = F.binary_cross_entropy_with_logits(logits, win, reduction="none")
    l_win = (bce * mask).sum() / mask.sum().clamp(min=1.0)
    target = torch.log2(points.clamp(min=BASE_POINTS) / BASE_POINTS)
    se = (pay - target) ** 2 * win
    l_pay = se.sum() / win.sum().clamp(min=1.0)
    return Losses(win=l_win, pay=l_pay)


def alive_from_inputs(x: torch.Tensor) -> torch.Tensor:
    """The unseen-copies plane of ``features`` as a bool mask ``(n, 34)``."""
    from mahjong.risk.data import UNSEEN_PLANE

    return x[:, UNSEEN_PLANE * NUM_KINDS : (UNSEEN_PLANE + 1) * NUM_KINDS] > 0.0


def numpy_forward(path: Path, x: np.ndarray) -> np.ndarray:
    """Runs an MJNN v2 dense file in numpy (a reference for the export)."""
    import struct

    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"MJNN"
    (version,) = struct.unpack_from("<I", data, 4)
    assert version == 2
    pos = 8
    for _ in range(2):
        (n,) = struct.unpack_from("<I", data, pos)
        pos += 4
        for _ in range(n):
            (ln,) = struct.unpack_from("<I", data, pos)
            pos += 4 + ln + 8
    (layers,) = struct.unpack_from("<I", data, pos)
    pos += 4
    h = x.astype(np.float32)
    for i in range(layers):
        out, inp = struct.unpack_from("<II", data, pos)
        pos += 8
        w = np.frombuffer(data, dtype="<f4", count=out * inp, offset=pos).reshape(out, inp)
        pos += 4 * out * inp
        b = np.frombuffer(data, dtype="<f4", count=out, offset=pos)
        pos += 4 * out
        h = h @ w.T + b
        if i + 1 < layers:
            h = np.maximum(h, 0.0)
    return h
