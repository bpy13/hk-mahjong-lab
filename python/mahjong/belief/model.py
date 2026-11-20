"""The belief model: what each opponent holds, from public information (plan Task 5).

One MLP reads the observer's 933 public features (plus, with ``discards=True``, each
opponent's last eight discards one-hot, ``3 × 8 × 34``; with ``passed=True``, the
passed-tile record and 摸打 / 入章 of ``features::encode_passed``, :data:`PASSED_LEN`
values, from logs of layout 2; with ``order=True``, every seat's discard and claim
order without the 摸打 / 入章 flag, ``features::encode_order``, :data:`ORDER_LEN`
values, from logs of layout 3; plan 2026-11-16 Task 19), and writes, for the three
opponents (relative seats 1 = next, 2 = across, 3 = previous), the heads in
:data:`HEADS`:

* ``counts`` (optional since Task 19, ``counts=False`` drops it): 34 × 5 logits, a
  categorical over 0–4 copies of each tile kind;
* ``tenpai``: one logit, P(waiting with a legal win now);
* ``waits``: 34 logits, P(a win on that tile | tenpai), independent sigmoids;
* ``faan``: 8 logits, the faan of the best discard win now (3 … 10) | tenpai;
* ``outcome``: 3 logits over {no win, wins on a discard, self-draws} before the
  hand ends (the EV bot's ``sd_o`` is class 2 and its ``x_o`` is class 1).

With ``race=True`` (the race bot's network, ``docs/race-bot.md`` §6–7) two heads follow:

* ``state``: 4 logits over the opponent's true state {far, one away, ready, folded};
* ``mode``: 3 logits over the mode of its last discard {push, detour, fold}.

With ``task19=True`` (Brian's required outputs, plan 2026-11-16 Task 19) the heads
``state`` (as above; a stage log never has a fold), ``winner`` (5 logits for the whole
row: who wins the hand, relative seat 0–3, or a draw) and ``claim`` (34 logits per
opponent: P(it would claim a discard of that kind into a meld now)) follow. Brian's
34 numbers, P(deal-in or claim) per kind, are read out of ``tenpai``, ``waits`` and
``claim`` (:func:`dealin_or_claim`), so either form (per opponent, or combined) is
available.

The output vector is the heads in that order, each holding its groups (three
opponents, or one for ``winner``) contiguously (``counts`` first: opponent, tile,
copies). :meth:`BeliefNet.export` writes the MJNN v2 file ``crates/search/src/net.rs``
loads, with these names as output slices and ``features`` / ``discards`` / ``passed`` /
``order`` as input slices. Count logits for more copies than the observer has unseen
are masked to ``-1e9`` by :func:`mask_counts` (in training and in
:meth:`BeliefNet.predict`); the exported network writes the raw logits, so a Rust
consumer applies the same mask.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
#: Discard slots per seat in ``features::encode_order`` (and ``encode_history``).
ORDER_DEPTH = 24
ORDER_SLOTS = 4 * ORDER_DEPTH
#: Length of ``features::encode_order``: one-hot tile per slot, two scalars per
#: slot (filled, claimed), one overflow value per seat.
ORDER_LEN = ORDER_SLOTS * NUM_KINDS + 2 * ORDER_SLOTS + 4
#: Bytes of the log's packed ``order`` field (``log::pack_order``).
ORDER_BYTES = ORDER_SLOTS + 4
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
#: The Task 19 heads: ``state`` (shared with the race heads), ``winner`` (one group
#: of five for the row), ``claim`` (34 per opponent).
N_WINNERS = 5
TASK19_HEADS: dict[str, int] = {"state": N_STATES, "winner": N_WINNERS, "claim": NUM_KINDS}
#: Heads with one group per row instead of one per opponent.
ROW_HEADS = frozenset({"winner"})
#: Count logits beyond the observer's unseen copies get this logit.
MASK_LOGIT = -1e9


def head_groups(name: str) -> int:
    """Groups of a head: 1 for a row head (``winner``), else the three opponents."""
    return 1 if name in ROW_HEADS else N_OPP


def head_slices(heads: dict[str, int] | None = None) -> dict[str, slice]:
    """Where each head sits in the output vector (all its groups)."""
    out: dict[str, slice] = {}
    start = 0
    for name, per in (heads or HEADS).items():
        out[name] = slice(start, start + head_groups(name) * per)
        start += head_groups(name) * per
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
    #: Also feed the discard and claim order block (``features::encode_order``; plan
    #: 2026-11-16 Task 19). Needs logs of layout 3, which carry an ``order`` field.
    order: bool = False
    #: Keep the copies-per-kind head (Brian, 2026-11-20, doubts it is still needed;
    #: Task 19 fits it both ways).
    counts: bool = True
    #: Add the Task 19 heads (``state``, ``winner``, ``claim``) after the belief heads.
    task19: bool = False
    #: Loss weight per head (1 where absent); the total loss is their weighted sum.
    weights: dict[str, float] = field(default_factory=dict)

    @property
    def input_len(self) -> int:
        return (
            STATE_LEN
            + (DISCARD_LEN if self.discards else 0)
            + (PASSED_LEN if self.passed else 0)
            + (ORDER_LEN if self.order else 0)
        )

    @property
    def heads(self) -> dict[str, int]:
        heads = {k: v for k, v in HEADS.items() if self.counts or k != "counts"}
        if self.race:
            heads.update(RACE_HEADS)
        if self.task19:
            heads.update(TASK19_HEADS)
        return heads

    @property
    def output_len(self) -> int:
        return sum(head_groups(k) * v for k, v in self.heads.items())

    def weight(self, name: str) -> float:
        return float(self.weights.get(name, 1.0))


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
        """Raw logits, shape ``(batch, output_len)``."""
        y: torch.Tensor = self.head(self.torso(x))
        return y

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def inputs(
        self,
        features: torch.Tensor,
        discards: torch.Tensor,
        passed: torch.Tensor | None = None,
        order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The input vector: ``features`` (``(n, 933)``, any float type), then the
        discard indices ``(n, 3, 8)`` (255 = empty) as one-hot, the passed block
        ``(n, PASSED_LEN)`` and the packed order ``(n, ORDER_BYTES)`` uint8 as
        :func:`order_block`, each when the config uses it."""
        parts = [features.float()]
        if self.cfg.discards:
            parts.append(discard_block(discards))
        if self.cfg.passed:
            if passed is None:
                raise ValueError("this network reads the passed block; the log has none")
            parts.append(passed.float())
        if self.cfg.order:
            if order is None:
                raise ValueError("this network reads the order block; the log has none")
            parts.append(order_block(order))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)

    def inputs_of(self, b: Batch) -> torch.Tensor:
        """:meth:`inputs` from a :class:`Batch`."""
        return self.inputs(b.features, b.discards, b.passed, b.order)

    def split(self, out: torch.Tensor) -> dict[str, torch.Tensor]:
        """The output vector as per-head tensors ``(n, groups, ...)``."""
        n = out.shape[0]
        parts: dict[str, torch.Tensor] = {}
        for name, per in self.cfg.heads.items():
            block = out[:, self.slices[name]].reshape(n, head_groups(name), per)
            if name == "counts":
                block = block.reshape(n, N_OPP, NUM_KINDS, MAX_COPIES)
            elif name == "tenpai":
                block = block.reshape(n, N_OPP)
            elif name == "winner":
                block = block.reshape(n, N_WINNERS)
            parts[name] = block
        return parts

    def predict(self, out: torch.Tensor, unseen: torch.Tensor) -> dict[str, torch.Tensor]:
        """Probabilities from raw logits: ``counts`` ``(n, 3, 34, 5)`` (masked by the
        observer's ``unseen`` ``(n, 34)``; absent without the head), ``tenpai`` ``(n, 3)``,
        ``waits`` ``(n, 3, 34)``, ``faan`` ``(n, 3, 8)``, ``outcome`` ``(n, 3, 3)``; with
        the race heads also ``state`` ``(n, 3, 4)`` and ``mode`` ``(n, 3, 3)``; with the
        Task 19 heads ``state``, ``winner`` ``(n, 5)`` and ``claim`` ``(n, 3, 34)``."""
        p = self.split(out)
        probs = {
            "tenpai": torch.sigmoid(p["tenpai"]),
            "waits": torch.sigmoid(p["waits"]),
            "faan": torch.softmax(p["faan"], dim=-1),
            "outcome": torch.softmax(p["outcome"], dim=-1),
        }
        if "counts" in p:
            probs["counts"] = torch.softmax(mask_counts(p["counts"], unseen), dim=-1)
        for name in ("state", "mode", "winner"):
            if name in p:
                probs[name] = torch.softmax(p[name], dim=-1)
        if "claim" in p:
            probs["claim"] = torch.sigmoid(p["claim"])
        return probs

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @staticmethod
    def load(path: Path) -> BeliefNet:
        ckpt = torch.load(path, map_location="cpu")
        net = BeliefNet(config_from_dict(ckpt["cfg"]))
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
        if self.cfg.order:
            inputs.append(("order", ORDER_LEN))
        outputs = [(name, head_groups(name) * per) for name, per in self.cfg.heads.items()]
        write_mjnn(path, linears, inputs=inputs, outputs=outputs)


def config_from_dict(c: dict[str, Any]) -> BeliefConfig:
    """A config from a saved dict; entries older checkpoints lack take their defaults."""
    return BeliefConfig(
        hidden=tuple(c["hidden"]),
        discards=c["discards"],
        passed=c.get("passed", False),
        race=c.get("race", False),
        order=c.get("order", False),
        counts=c.get("counts", True),
        task19=c.get("task19", False),
        weights=dict(c.get("weights", {})),
    )


def warm_start(net: BeliefNet, src: BeliefNet) -> None:
    """Copies ``src``'s weights into ``net`` where they mean the same thing: the torso
    (same widths required) and the rows of every head both have. A race network
    warm-started from a plain belief model keeps the five heads and starts its race
    heads fresh (``docs/race-bot.md`` §8.1 step 3: warm start from the last generation)."""
    if (
        net.cfg.hidden != src.cfg.hidden
        or net.cfg.discards != src.cfg.discards
        or net.cfg.passed != src.cfg.passed
        or net.cfg.order != src.cfg.order
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


def order_block(packed: torch.Tensor) -> torch.Tensor:
    """``features::encode_order``'s block ``(n, ORDER_LEN)`` from the log's packed
    ``order`` bytes ``(n, ORDER_BYTES)`` (``log::pack_order``: slot byte 0 = empty,
    else ``1 + tile + 34 × claimed``; then one overflow count per seat): the same
    numbers as ``log::unpack_order``."""
    n = packed.shape[0]
    slots = packed[:, :ORDER_SLOTS].long()
    filled = slots > 0
    v = (slots - 1).clamp(min=0)
    tile = v % NUM_KINDS
    claimed = (v // NUM_KINDS).float() * filled
    planes = F.one_hot(tile, NUM_KINDS).float() * filled[..., None]
    scalars = torch.stack([filled.float(), claimed], dim=-1).reshape(n, 2 * ORDER_SLOTS)
    over = (packed[:, ORDER_SLOTS:].float() / 4.0).clamp(max=1.0)
    return torch.cat([planes.reshape(n, ORDER_SLOTS * NUM_KINDS), scalars, over], dim=1)


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


def dealin_or_claim(
    tenpai: torch.Tensor,
    waits: torch.Tensor,
    claim: torch.Tensor | None,
    unseen: torch.Tensor,
) -> torch.Tensor:
    """Brian's 34 numbers per row: P(some opponent wins on the kind or claims it into
    a meld if the observer throws it now), ``1 − Π_o (1 − d_o − (1 − d_o) c_o)`` with
    ``d_o = tenpai_o × wait_o`` and ``c_o`` the claim head (0 without it); 0 for a kind
    with no unseen copy. ``tenpai`` ``(n, 3)``, ``waits`` and ``claim`` ``(n, 3, 34)``,
    ``unseen`` ``(n, 34)``; the same formula as ``BeliefOutput::dealin_or_claim``."""
    d = tenpai[:, :, None] * waits
    c = torch.zeros_like(d) if claim is None else claim
    none = (1.0 - d - (1.0 - d) * c).prod(dim=1)
    return (1.0 - none) * (unseen > 0)


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
    # Layout 3 (plan 2026-11-16 Task 19): the packed order and the new labels.
    order: torch.Tensor | None = None  # (n, ORDER_BYTES) uint8
    winner: torch.Tensor | None = None  # (n,) int64: relative seat 0..3, 4 draw
    claim: torch.Tensor | None = None  # (n, 3, 34) float: would claim the kind

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
    log's ``opp_state`` and ``opp_mode`` fill :attr:`Batch.state` and :attr:`Batch.mode`;
    a layout 3 log's ``order``, ``opp_state``, ``winner`` and ``opp_claim`` fill
    :attr:`Batch.order`, ``state``, ``winner`` and ``claim``."""
    features = torch.from_numpy(np.ascontiguousarray(rows["features"]))
    discards = torch.from_numpy(np.ascontiguousarray(rows["discards"][:, 1:4, :]))
    faan = np.clip(rows["opp_faan_now"].astype(np.int64), FAAN_MIN, FAAN_MAX) - FAAN_MIN
    names = rows.dtype.names or ()
    extra: dict[str, torch.Tensor] = {
        key: torch.from_numpy(rows[field].astype(np.int64))
        for key, field in (("state", "opp_state"), ("mode", "opp_mode"), ("winner", "winner"))
        if field in names
    }
    if "passed" in names:
        extra["passed"] = torch.from_numpy(np.ascontiguousarray(rows["passed"]))
    if "order" in names:
        extra["order"] = torch.from_numpy(np.ascontiguousarray(rows["order"]))
    if "opp_claim" in names:
        extra["claim"] = torch.from_numpy(rows["opp_claim"].astype(np.float32))
    return Batch(
        **extra,
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
    """Per-head mean losses (nats) and their weighted sum under ``"total"``
    (``BeliefConfig.weights``; the parts themselves are unweighted).

    ``counts``: cross-entropy summed over the 34 kinds (masked logits), averaged
    over rows and opponents; ``tenpai``, ``outcome``: cross-entropy; ``waits`` and
    ``faan``: cross-entropy over tenpai opponents only (zero weight otherwise); with
    the race heads and labels, ``state`` and ``mode``: cross-entropy; with the Task 19
    heads and labels, ``state`` and ``winner``: cross-entropy, ``claim``: binary
    cross-entropy summed over the 34 kinds.
    """
    p = net.split(out)
    n = out.shape[0]
    parts: dict[str, torch.Tensor] = {}
    if "counts" in p:
        counts_logits = mask_counts(p["counts"], b.unseen).reshape(-1, MAX_COPIES)
        nll_counts = F.cross_entropy(counts_logits, b.counts.reshape(-1), reduction="none")
        parts["counts"] = nll_counts.reshape(n, N_OPP, NUM_KINDS).sum(-1).mean()
    parts["tenpai"] = F.binary_cross_entropy_with_logits(p["tenpai"], b.tenpai)
    weight = b.tenpai
    denom = weight.sum().clamp(min=1.0)
    wait_nll = F.binary_cross_entropy_with_logits(p["waits"], b.waits, reduction="none").sum(-1)
    parts["waits"] = (wait_nll * weight).sum() / denom
    faan_nll = F.cross_entropy(p["faan"].reshape(-1, N_FAAN), b.faan.reshape(-1), reduction="none")
    parts["faan"] = (faan_nll.reshape(n, N_OPP) * weight).sum() / denom
    parts["outcome"] = F.cross_entropy(p["outcome"].reshape(-1, N_OUTCOME), b.outcome.reshape(-1))
    if "state" in p and b.state is not None:
        parts["state"] = F.cross_entropy(p["state"].reshape(-1, N_STATES), b.state.reshape(-1))
    if "mode" in p and b.mode is not None:
        parts["mode"] = F.cross_entropy(p["mode"].reshape(-1, N_MODES), b.mode.reshape(-1))
    if "winner" in p and b.winner is not None:
        parts["winner"] = F.cross_entropy(p["winner"], b.winner)
    if "claim" in p and b.claim is not None:
        claim_nll = F.binary_cross_entropy_with_logits(p["claim"], b.claim, reduction="none")
        parts["claim"] = claim_nll.sum(-1).mean()
    total = torch.zeros((), device=out.device)
    for k, v in parts.items():
        total = total + net.cfg.weight(k) * v
    parts["total"] = total
    return parts
