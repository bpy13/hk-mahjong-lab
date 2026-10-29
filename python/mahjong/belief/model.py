"""The belief model: what each opponent holds, from public information (plan Task 5).

One MLP reads the observer's 933 public features (plus, with ``discards=True``, each
opponent's last eight discards one-hot, ``3 × 8 × 34``), and writes, for the three opponents
(relative seats 1 = next, 2 = across, 3 = previous), the heads in :data:`HEADS`:

* ``counts``: 34 × 5 logits, a categorical over 0–4 copies of each tile kind;
* ``tenpai``: one logit, P(waiting with a legal win now);
* ``waits``: 34 logits, P(a win on that tile | tenpai), independent sigmoids;
* ``faan``: 8 logits, the faan of the best discard win now (3 … 10) | tenpai;
* ``outcome``: 3 logits over {no win, wins on a discard, self-draws} before the
  hand ends (the EV bot's ``sd_o`` is class 2 and its ``x_o`` is class 1).

The output vector is the heads in that order, each holding its three opponents
contiguously (``counts`` first: opponent, tile, copies). :meth:`BeliefNet.export`
writes the MJNN v2 file ``crates/search/src/net.rs`` loads, with these names as
output slices and ``features`` / ``discards`` as input slices. Count logits for
more copies than the observer has unseen are masked to ``-1e9`` by
:func:`mask_counts` (in training and in :meth:`BeliefNet.predict`); the exported
network writes the raw logits, so a Rust consumer applies the same mask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mahjong.belief.logs import NUM_KINDS, UNSEEN_PLANE
from mahjong.rl.model import STATE_LEN, write_mjnn

N_OPP = 3
HISTORY = 8
DISCARD_LEN = N_OPP * HISTORY * NUM_KINDS
MAX_COPIES = 5
FAAN_MIN, FAAN_MAX = 3, 10
N_FAAN = FAAN_MAX - FAAN_MIN + 1
N_OUTCOME = 3
#: Head name → values per opponent, in output order.
HEADS: dict[str, int] = {
    "counts": NUM_KINDS * MAX_COPIES,
    "tenpai": 1,
    "waits": NUM_KINDS,
    "faan": N_FAAN,
    "outcome": N_OUTCOME,
}
OUTPUT_LEN = N_OPP * sum(HEADS.values())
#: Count logits beyond the observer's unseen copies get this logit.
MASK_LOGIT = -1e9


def head_slices() -> dict[str, slice]:
    """Where each head sits in the output vector (all three opponents)."""
    out: dict[str, slice] = {}
    start = 0
    for name, per_opp in HEADS.items():
        out[name] = slice(start, start + N_OPP * per_opp)
        start += N_OPP * per_opp
    return out


@dataclass
class BeliefConfig:
    hidden: tuple[int, ...] = (384, 288)
    #: Also feed the discard-order block. Off by default: on the hold-out log it
    #: improved the count log-likelihood by 0.1 %, under the plan's 2 % bar
    #: (`docs/belief.md` §5).
    discards: bool = False

    @property
    def input_len(self) -> int:
        return STATE_LEN + (DISCARD_LEN if self.discards else 0)


class BeliefNet(nn.Module):
    def __init__(self, cfg: BeliefConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BeliefConfig()
        layers: list[nn.Module] = []
        width = self.cfg.input_len
        for h in self.cfg.hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.torso = nn.Sequential(*layers)
        self.head = nn.Linear(width, OUTPUT_LEN)
        self.slices = head_slices()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits, shape ``(batch, OUTPUT_LEN)``."""
        y: torch.Tensor = self.head(self.torso(x))
        return y

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def inputs(self, features: torch.Tensor, discards: torch.Tensor) -> torch.Tensor:
        """The input vector: ``features`` (``(n, 933)``, any float type) and the
        discard indices ``(n, 3, 8)`` (255 = empty) as one-hot, when the config
        uses them."""
        x = features.float()
        if not self.cfg.discards:
            return x
        return torch.cat([x, discard_block(discards)], dim=1)

    def split(self, out: torch.Tensor) -> dict[str, torch.Tensor]:
        """The output vector as per-head tensors ``(n, 3, ...)``."""
        n = out.shape[0]
        parts: dict[str, torch.Tensor] = {}
        for name, per_opp in HEADS.items():
            block = out[:, self.slices[name]].reshape(n, N_OPP, per_opp)
            if name == "counts":
                block = block.reshape(n, N_OPP, NUM_KINDS, MAX_COPIES)
            elif name == "tenpai":
                block = block.reshape(n, N_OPP)
            parts[name] = block
        return parts

    def predict(self, out: torch.Tensor, unseen: torch.Tensor) -> dict[str, torch.Tensor]:
        """Probabilities from raw logits: ``counts`` ``(n, 3, 34, 5)`` (masked by the
        observer's ``unseen`` ``(n, 34)``), ``tenpai`` ``(n, 3)``, ``waits``
        ``(n, 3, 34)``, ``faan`` ``(n, 3, 8)``, ``outcome`` ``(n, 3, 3)``."""
        p = self.split(out)
        return {
            "counts": torch.softmax(mask_counts(p["counts"], unseen), dim=-1),
            "tenpai": torch.sigmoid(p["tenpai"]),
            "waits": torch.sigmoid(p["waits"]),
            "faan": torch.softmax(p["faan"], dim=-1),
            "outcome": torch.softmax(p["outcome"], dim=-1),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @staticmethod
    def load(path: Path) -> BeliefNet:
        ckpt = torch.load(path, map_location="cpu")
        cfg = BeliefConfig(hidden=tuple(ckpt["cfg"]["hidden"]), discards=ckpt["cfg"]["discards"])
        net = BeliefNet(cfg)
        net.load_state_dict(ckpt["state"])
        return net

    def export(self, path: Path) -> None:
        """Writes the MJNN v2 file (named input and output slices)."""
        linears = [m for m in self.torso if isinstance(m, nn.Linear)] + [self.head]
        inputs = [("features", STATE_LEN)]
        if self.cfg.discards:
            inputs.append(("discards", DISCARD_LEN))
        outputs = [(name, N_OPP * per_opp) for name, per_opp in HEADS.items()]
        write_mjnn(path, linears, inputs=inputs, outputs=outputs)


def discard_block(discards: torch.Tensor) -> torch.Tensor:
    """One-hot of the opponents' discard indices ``(n, 3, 8)`` (255 = empty) as
    ``(n, 3 × 8 × 34)`` float, in the same order as ``net.rs::encode_discards``."""
    idx = discards.long().clamp(max=NUM_KINDS)
    one_hot = F.one_hot(idx, NUM_KINDS + 1)[..., :NUM_KINDS]
    return one_hot.reshape(discards.shape[0], DISCARD_LEN).float()


def mask_counts(logits: torch.Tensor, unseen: torch.Tensor) -> torch.Tensor:
    """Masks copies the observer knows an opponent cannot hold: ``logits``
    ``(n, 3, 34, 5)``, ``unseen`` ``(n, 34)`` copies the observer has not seen."""
    copies = torch.arange(MAX_COPIES, device=logits.device)
    too_many = copies > unseen[:, None, :, None]
    return logits.masked_fill(too_many, MASK_LOGIT)


def unseen_from_features(features: torch.Tensor) -> torch.Tensor:
    """The unseen copies per kind ``(n, 34)`` from the features' unseen plane."""
    plane = features[:, UNSEEN_PLANE * NUM_KINDS : (UNSEEN_PLANE + 1) * NUM_KINDS].float()
    return torch.round(plane * 4).long()


@dataclass
class Batch:
    """Inputs and targets of a set of log rows, as tensors on one device."""

    features: torch.Tensor  # (n, 933) float16 or float32
    discards: torch.Tensor  # (n, 3, 8) uint8
    unseen: torch.Tensor  # (n, 34) int64
    counts: torch.Tensor  # (n, 3, 34) int64 in 0..4
    tenpai: torch.Tensor  # (n, 3) float
    waits: torch.Tensor  # (n, 3, 34) float
    faan: torch.Tensor  # (n, 3) int64 in 0..7 (faan − 3, clipped)
    outcome: torch.Tensor  # (n, 3) int64: 0 none, 1 discard win, 2 self-draw

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: torch.device | str) -> Batch:
        return Batch(**{k: v.to(device) for k, v in self.__dict__.items()})

    def index(self, idx: torch.Tensor) -> Batch:
        return Batch(**{k: v[idx] for k, v in self.__dict__.items()})


def batch_from_rows(rows: np.ndarray[Any, np.dtype[Any]]) -> Batch:
    """Targets and inputs from log rows (``mahjong.belief.logs``), on the CPU."""
    features = torch.from_numpy(np.ascontiguousarray(rows["features"]))
    discards = torch.from_numpy(np.ascontiguousarray(rows["discards"][:, 1:4, :]))
    faan = np.clip(rows["opp_faan_now"].astype(np.int64), FAAN_MIN, FAAN_MAX) - FAAN_MIN
    return Batch(
        features=features,
        discards=discards,
        unseen=unseen_from_features(features),
        counts=torch.from_numpy(np.minimum(rows["opp_counts"], MAX_COPIES - 1).astype(np.int64)),
        tenpai=torch.from_numpy(rows["opp_tenpai"].astype(np.float32)),
        waits=torch.from_numpy(rows["opp_waits"].astype(np.float32)),
        faan=torch.from_numpy(faan),
        outcome=torch.from_numpy(rows["opp_won"].astype(np.int64)),
    )


def losses(net: BeliefNet, out: torch.Tensor, b: Batch) -> dict[str, torch.Tensor]:
    """Per-head mean losses (nats) and their sum under ``"total"``.

    ``counts``: cross-entropy summed over the 34 kinds (masked logits), averaged
    over rows and opponents; ``tenpai``, ``outcome``: cross-entropy; ``waits`` and
    ``faan``: cross-entropy over tenpai opponents only (zero weight otherwise).
    """
    p = net.split(out)
    n = out.shape[0]
    counts_logits = mask_counts(p["counts"], b.unseen).reshape(-1, MAX_COPIES)
    nll_counts = F.cross_entropy(counts_logits, b.counts.reshape(-1), reduction="none")
    loss_counts = nll_counts.reshape(n, N_OPP, NUM_KINDS).sum(-1).mean()
    loss_tenpai = F.binary_cross_entropy_with_logits(p["tenpai"], b.tenpai)
    weight = b.tenpai
    denom = weight.sum().clamp(min=1.0)
    wait_nll = F.binary_cross_entropy_with_logits(p["waits"], b.waits, reduction="none").sum(-1)
    loss_waits = (wait_nll * weight).sum() / denom
    faan_nll = F.cross_entropy(p["faan"].reshape(-1, N_FAAN), b.faan.reshape(-1), reduction="none")
    loss_faan = (faan_nll.reshape(n, N_OPP) * weight).sum() / denom
    loss_outcome = F.cross_entropy(p["outcome"].reshape(-1, N_OUTCOME), b.outcome.reshape(-1))
    total = loss_counts + loss_tenpai + loss_waits + loss_faan + loss_outcome
    return {
        "counts": loss_counts,
        "tenpai": loss_tenpai,
        "waits": loss_waits,
        "faan": loss_faan,
        "outcome": loss_outcome,
        "total": total,
    }
