"""The belief model: what each opponent holds, from public information (plan Task 5).

One MLP reads the observer's 933 public features (plus, with ``discards=True``, each
opponent's last eight discards one-hot, ``3 × 8 × 34``; with ``passed=True``, the
passed-tile record and 摸打 / 入章 of ``features::encode_passed``, :data:`PASSED_LEN`
values, from logs of layout 2), and writes, for the three opponents
(relative seats 1 = next, 2 = across, 3 = previous), the heads in :data:`HEADS`:

* ``counts``: 34 × 5 logits, a categorical over 0–4 copies of each tile kind;
* ``tenpai``: one logit, P(waiting with a legal win now);
* ``waits``: 34 logits, P(a win on that tile | tenpai), independent sigmoids;
* ``faan``: 8 logits, the faan of the best discard win now (3 … 10) | tenpai;
* ``outcome``: 3 logits over {no win, wins on a discard, self-draws} before the
  hand ends (the EV bot's ``sd_o`` is class 2 and its ``x_o`` is class 1).

With ``race=True`` (the race bot's network, ``docs/race-bot.md`` §6–7) two heads follow:

* ``state``: 4 logits over the opponent's true state {far, one away, ready, folded};
* ``mode``: 3 logits over the mode of its last discard {push, detour, fold}.

The output vector is the heads in that order, each holding its three opponents
contiguously (``counts`` first: opponent, tile, copies). :meth:`BeliefNet.export`
writes the MJNN v2 file ``crates/search/src/net.rs`` loads, with these names as
output slices and ``features`` / ``discards`` / ``passed`` as input slices. Count logits for
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
#: Length of ``features::encode_passed`` (14 tile planes and 12 scalars).
PASSED_LEN = 14 * NUM_KINDS + 12
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
#: The race heads (``docs/race-bot.md`` §7): true state F / O / R / X and the mode
#: of the last discard push / detour / fold. They follow the five heads above.
N_STATES = 4
N_MODES = 3
RACE_HEADS: dict[str, int] = {"state": N_STATES, "mode": N_MODES}
#: Count logits beyond the observer's unseen copies get this logit.
MASK_LOGIT = -1e9


def head_slices(heads: dict[str, int] | None = None) -> dict[str, slice]:
    """Where each head sits in the output vector (all three opponents)."""
    out: dict[str, slice] = {}
    start = 0
    for name, per_opp in (heads or HEADS).items():
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
    #: Also feed the passed-tile block (``features::encode_passed``; plan 2026-11-09
    #: Task 18). Needs logs of layout 2, which carry a ``passed`` field.
    passed: bool = False
    #: Add the race heads (``state``, ``mode``) after the five belief heads.
    race: bool = False

    @property
    def input_len(self) -> int:
        return (
            STATE_LEN + (DISCARD_LEN if self.discards else 0) + (PASSED_LEN if self.passed else 0)
        )

    @property
    def heads(self) -> dict[str, int]:
        return {**HEADS, **RACE_HEADS} if self.race else HEADS

    @property
    def output_len(self) -> int:
        return N_OPP * sum(self.heads.values())


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
        self.head = nn.Linear(width, self.cfg.output_len)
        self.slices = head_slices(self.cfg.heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits, shape ``(batch, OUTPUT_LEN)``."""
        y: torch.Tensor = self.head(self.torso(x))
        return y

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def inputs(
        self,
        features: torch.Tensor,
        discards: torch.Tensor,
        passed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The input vector: ``features`` (``(n, 933)``, any float type), then the
        discard indices ``(n, 3, 8)`` (255 = empty) as one-hot and the passed block
        ``(n, PASSED_LEN)``, each when the config uses it."""
        parts = [features.float()]
        if self.cfg.discards:
            parts.append(discard_block(discards))
        if self.cfg.passed:
            if passed is None:
                raise ValueError("this network reads the passed block; the log has none")
            parts.append(passed.float())
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def split(self, out: torch.Tensor) -> dict[str, torch.Tensor]:
        """The output vector as per-head tensors ``(n, 3, ...)``."""
        n = out.shape[0]
        parts: dict[str, torch.Tensor] = {}
        for name, per_opp in self.cfg.heads.items():
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
        ``(n, 3, 34)``, ``faan`` ``(n, 3, 8)``, ``outcome`` ``(n, 3, 3)``; with the
        race heads also ``state`` ``(n, 3, 4)`` and ``mode`` ``(n, 3, 3)``."""
        p = self.split(out)
        probs = {
            "counts": torch.softmax(mask_counts(p["counts"], unseen), dim=-1),
            "tenpai": torch.sigmoid(p["tenpai"]),
            "waits": torch.sigmoid(p["waits"]),
            "faan": torch.softmax(p["faan"], dim=-1),
            "outcome": torch.softmax(p["outcome"], dim=-1),
        }
        for name in RACE_HEADS if self.cfg.race else ():
            probs[name] = torch.softmax(p[name], dim=-1)
        return probs

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @staticmethod
    def load(path: Path) -> BeliefNet:
        ckpt = torch.load(path, map_location="cpu")
        c = ckpt["cfg"]
        # Checkpoints from before the race heads have no "race" entry.
        cfg = BeliefConfig(
            hidden=tuple(c["hidden"]),
            discards=c["discards"],
            passed=c.get("passed", False),
            race=c.get("race", False),
        )
        net = BeliefNet(cfg)
        net.load_state_dict(ckpt["state"])
        return net

    def export(self, path: Path) -> None:
        """Writes the MJNN v2 file (named input and output slices)."""
        linears = [m for m in self.torso if isinstance(m, nn.Linear)] + [self.head]
        inputs = [("features", STATE_LEN)]
        if self.cfg.discards:
            inputs.append(("discards", DISCARD_LEN))
        if self.cfg.passed:
            inputs.append(("passed", PASSED_LEN))
        outputs = [(name, N_OPP * per_opp) for name, per_opp in self.cfg.heads.items()]
        write_mjnn(path, linears, inputs=inputs, outputs=outputs)


def warm_start(net: BeliefNet, src: BeliefNet) -> None:
    """Copies ``src``'s weights into ``net`` where they mean the same thing: the torso
    (same widths required) and the rows of every head both have. A race network
    warm-started from a plain belief model keeps the five heads and starts its race
    heads fresh (``docs/race-bot.md`` §8.1 step 3: warm start from the last generation)."""
    if (
        net.cfg.hidden != src.cfg.hidden
        or net.cfg.discards != src.cfg.discards
        or net.cfg.passed != src.cfg.passed
    ):
        raise ValueError(f"cannot warm-start {net.cfg} from {src.cfg}")
    with torch.no_grad():
        net.torso.load_state_dict(src.torso.state_dict())
        for name, sl in net.slices.items():
            if name in src.slices:
                net.head.weight[sl] = src.head.weight[src.slices[name]]
                net.head.bias[sl] = src.head.bias[src.slices[name]]


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
    # The race labels (race logs only, ``race_log`` example; None otherwise).
    state: torch.Tensor | None = None  # (n, 3) int64: 0 far, 1 one away, 2 ready, 3 folded
    mode: torch.Tensor | None = None  # (n, 3) int64: 0 push, 1 detour, 2 fold
    # The passed-tile block (logs of layout 2 only; None otherwise).
    passed: torch.Tensor | None = None  # (n, PASSED_LEN) float16

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def to(self, device: torch.device | str) -> Batch:
        moved: dict[str, Any] = {
            k: None if v is None else v.to(device) for k, v in self.__dict__.items()
        }
        return Batch(**moved)

    def index(self, idx: torch.Tensor) -> Batch:
        picked: dict[str, Any] = {
            k: None if v is None else v[idx] for k, v in self.__dict__.items()
        }
        return Batch(**picked)


def batch_from_rows(rows: np.ndarray[Any, np.dtype[Any]]) -> Batch:
    """Targets and inputs from log rows (``mahjong.belief.logs``), on the CPU. A race
    log's ``opp_state`` and ``opp_mode`` fill :attr:`Batch.state` and :attr:`Batch.mode`."""
    features = torch.from_numpy(np.ascontiguousarray(rows["features"]))
    discards = torch.from_numpy(np.ascontiguousarray(rows["discards"][:, 1:4, :]))
    faan = np.clip(rows["opp_faan_now"].astype(np.int64), FAAN_MIN, FAAN_MAX) - FAAN_MIN
    names = rows.dtype.names or ()
    race = {
        key: torch.from_numpy(rows[field].astype(np.int64))
        for key, field in (("state", "opp_state"), ("mode", "opp_mode"))
        if field in names
    }
    passed = torch.from_numpy(np.ascontiguousarray(rows["passed"])) if "passed" in names else None
    return Batch(
        **race,
        passed=passed,
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
    ``faan``: cross-entropy over tenpai opponents only (zero weight otherwise); with
    the race heads and labels, ``state`` and ``mode``: cross-entropy.
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
    parts = {
        "counts": loss_counts,
        "tenpai": loss_tenpai,
        "waits": loss_waits,
        "faan": loss_faan,
        "outcome": loss_outcome,
    }
    if net.cfg.race and b.state is not None and b.mode is not None:
        parts["state"] = F.cross_entropy(p["state"].reshape(-1, N_STATES), b.state.reshape(-1))
        parts["mode"] = F.cross_entropy(p["mode"].reshape(-1, N_MODES), b.mode.reshape(-1))
    parts["total"] = sum(parts.values(), torch.zeros((), device=out.device))
    return parts
