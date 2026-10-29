"""The suit-shared 1-D CNN with policy, target and four-seat value heads (plan Task 14).

Design (the Rust side, ``crates/search/src/cnn.rs``, runs the same network):

* **Input**: the 933 features of ``features::encode``. Its 26 tile planes are cut
  into the three suits (9 tiles each) and the 7 honours. The suits go through a
  convolution tower with **one set of weights shared by all suits**: a pattern
  learned on characters (a 3-4 wait, a run, a pair) is known on dots and bamboo
  too, and the network needs a third of the data a flat MLP would.
* **Honour branch**: the honour columns (26 x 7) and the 49 scalars through two
  dense layers -> ``h``.
* **Tower**: a stem convolution (kernel 3, 26 -> C channels) and ``blocks``
  residual blocks. Each block first adds a context vector to every position of
  every suit: a dense layer of ``[sum over suits, max over suits]`` of each suit's
  mean-pooled channels and ``h``. Through it a suit learns how far the other suits
  and the honours are along (needed for flush plans and for defence), and since sum
  and max ignore suit order the network stays *suit-equivariant*: swapping two suits
  in the input swaps the corresponding action logits and target classes in the
  output, exactly (a test checks it).
* **Merge**: ``z = relu(dense([sum, max of the pooled suits, h]))``.
* **Policy (134 logits, the fixed action codes)**: the four per-tile actions of each
  suited tile (discard, concealed kong, added kong, chow from that tile) from a 1x1
  convolution over the suit's positions after adding a projection of ``z``; the
  honour tiles' three actions and the five global actions (self-drawn win, pass,
  win, pung, kong) from a dense layer on ``z``.
* **Target (45 classes, ``labels::target_class``)**: plan kind (mixed, half flush,
  full flush, honours, orphans) x suit x structure (any, all pungs, all chows). The
  18 flush classes come from each suit's pooled channels and ``z`` (shared weights,
  so they too follow a suit swap); the 9 suit-less classes from ``z``; the 18 slots
  no plan uses are a fixed -1e4.
* **Value (4)**: final points of each relative seat (0 = me), log-scaled in
  training, zero-sum (the mean is subtracted). With ``oracle=True`` the value head
  also reads the hidden oracle block (opponents' hands, next wall tiles) in training
  only (Suphx-style oracle critic, idea 5b); the exported network leaves it out,
  which equals feeding zeros.

Defaults (C = 128, 6 blocks, 512-wide dense parts) give about 2.2 M parameters and
about 18 M multiply-adds per position, so one forward pass on one thread stays near
the 5 ms target; ``net_speed`` measures it. No normalisation layers: the export
stays a plain list of tensors and the residual branches start near zero instead.

Export: ``export(model, path, int8=False)`` writes MJNN version 3 (see
``cnn.rs``); ``int8=True`` stores every weight matrix as int8 with one scale per
output row (biases stay float32), about a quarter of the size.
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from mahjong._core import feature_sizes

STATE_LEN, ORACLE_LEN, NUM_ACTIONS = feature_sizes()
PLANES = 26
SCALARS = STATE_LEN - PLANES * 34
SUIT_LEN = 9
HONOURS = 7
REST_LEN = PLANES * HONOURS + SCALARS
NUM_TARGETS = 45
VALUE_LEN = 4
OUTPUT_LEN = NUM_ACTIONS + NUM_TARGETS + VALUE_LEN
UNUSED_TARGET = -1.0e4
ARCH = "suitcnn-1"
#: Codes of the global actions read from the honour-and-global layer.
GLOBAL_CODES = (34, 103, 104, 105, 106)

assert SCALARS == 49 and NUM_ACTIONS == 134


@dataclass
class CnnConfig:
    channels: int = 128
    blocks: int = 6
    kernel: int = 3
    honour_hidden: int = 512
    merge: int = 512
    value_hidden: int = 256
    #: The value head also reads the oracle block (training only, never exported).
    oracle: bool = False


def split_planes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``features::encode_planes``: ``(B, 933)`` -> suits ``(B, 3, 26, 9)`` and
    rest ``(B, 231)`` (honours plane-major, then the scalars)."""
    b = x.shape[0]
    planes = x[:, : PLANES * 34].reshape(b, PLANES, 34)
    suits = planes[:, :, : 3 * SUIT_LEN].reshape(b, PLANES, 3, SUIT_LEN).permute(0, 2, 1, 3)
    rest = torch.cat(
        [planes[:, :, 3 * SUIT_LEN :].reshape(b, PLANES * HONOURS), x[:, PLANES * 34 :]], 1
    )
    return suits, rest


class Block(nn.Module):
    """Residual block: add the cross-suit context, conv, ReLU, conv, add."""

    def __init__(self, c: int, ctx_in: int, k: int) -> None:
        super().__init__()
        self.ctx = nn.Linear(ctx_in, c)
        self.conv1 = nn.Conv1d(c, c, k, padding=k // 2)
        self.conv2 = nn.Conv1d(c, c, k, padding=k // 2)
        # Start each residual branch near zero so the deep tower trains without
        # normalisation layers.
        nn.init.normal_(self.conv2.weight, std=1e-3)
        assert self.conv2.bias is not None
        nn.init.zeros_(self.conv2.bias)

    def forward(self, y: torch.Tensor, ctx_in: torch.Tensor) -> torch.Tensor:
        # y: (B*3, C, 9); ctx_in: (B, 2C + H).
        ctx = self.ctx(ctx_in).repeat_interleave(3, dim=0)[:, :, None]
        t = F.relu(self.conv1(F.relu(y + ctx)))
        return F.relu(y + self.conv2(t))


def _pool(y: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each suit's mean over positions ``(B, 3, C)`` and ``[sum, max over suits, h]``."""
    pooled = y.mean(-1).reshape(-1, 3, y.shape[1])
    return pooled, torch.cat([pooled.sum(1), pooled.amax(1), h], 1)


class SuitCNN(nn.Module):
    def __init__(self, cfg: CnnConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg = cfg or CnnConfig()
        c, k, hh, m, vh = cfg.channels, cfg.kernel, cfg.honour_hidden, cfg.merge, cfg.value_hidden
        if k % 2 == 0 or k > SUIT_LEN:
            raise ValueError(f"kernel {k} must be odd and at most {SUIT_LEN}")
        self.honour1 = nn.Linear(REST_LEN, hh)
        self.honour2 = nn.Linear(hh, hh)
        self.stem = nn.Conv1d(PLANES, c, k, padding=k // 2)
        self.blocks = nn.ModuleList([Block(c, 2 * c + hh, k) for _ in range(cfg.blocks)])
        self.merge = nn.Linear(2 * c + hh, m)
        self.zproj = nn.Linear(m, c)
        self.suit_policy = nn.Conv1d(c, 4, 1)
        self.honour_policy = nn.Linear(m, 3 * HONOURS + len(GLOBAL_CODES))
        self.flush_target = nn.Linear(c + m, 6)
        self.other_target = nn.Linear(m, 9)
        self.value1 = nn.Linear(m, vh)
        self.value2 = nn.Linear(vh, VALUE_LEN)
        if cfg.oracle:
            self.oracle_value = nn.Linear(ORACLE_LEN, vh)

    def forward(
        self, x: torch.Tensor, oracle: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x``: features ``(B, 933)``; ``oracle``: ``(B, 136)`` or None (zeros).

        Returns policy logits ``(B, 134)``, target logits ``(B, 45)`` and values
        ``(B, 4)`` (relative seats, summing to zero).
        """
        b = x.shape[0]
        suits, rest = split_planes(x)
        h = F.relu(self.honour2(F.relu(self.honour1(rest))))
        y = F.relu(self.stem(suits.reshape(b * 3, PLANES, SUIT_LEN)))
        for blk in self.blocks:
            _, ctx_in = _pool(y, h)
            y = blk(y, ctx_in)
        pooled, merge_in = _pool(y, h)
        z = F.relu(self.merge(merge_in))

        # Policy.
        zp = self.zproj(z).repeat_interleave(3, dim=0)[:, :, None]
        suited = self.suit_policy(F.relu(y + zp)).reshape(b, 3, 4, SUIT_LEN)
        suited = suited.permute(0, 2, 1, 3).reshape(b, 4, 3 * SUIT_LEN)  # (B, action, tile)
        hp = self.honour_policy(z)
        policy = torch.cat(
            [
                suited[:, 0],  # 0-26 discard suited
                hp[:, 0:HONOURS],  # 27-33 discard honours
                hp[:, 3 * HONOURS : 3 * HONOURS + 1],  # 34 self-drawn win
                suited[:, 1],  # 35-61 concealed kong suited
                hp[:, HONOURS : 2 * HONOURS],  # 62-68
                suited[:, 2],  # 69-95 added kong suited
                hp[:, 2 * HONOURS : 3 * HONOURS],  # 96-102
                hp[:, 3 * HONOURS + 1 :],  # 103-106 pass, win, pung, kong
                suited[:, 3],  # 107-133 chow
            ],
            1,
        )

        # Target: kind * 9 + suit * 3 + structure.
        flush = self.flush_target(torch.cat([pooled, z[:, None, :].expand(b, 3, z.shape[1])], 2))
        other = self.other_target(z)
        target = x.new_full((b, NUM_TARGETS), UNUSED_TARGET)
        for s in range(3):
            for j in range(6):
                target[:, (1 + j // 3) * 9 + s * 3 + j % 3] = flush[:, s, j]
        for j in range(9):
            target[:, (0, 3, 4)[j // 3] * 9 + j % 3] = other[:, j]

        # Value.
        pre = self.value1(z)
        if self.cfg.oracle and oracle is not None:
            pre = pre + self.oracle_value(oracle)
        v = self.value2(F.relu(pre))
        v = v - v.mean(1, keepdim=True)
        return policy, target, v

    def parameters_at_play(self) -> int:
        """Parameters of the exported network (the oracle critic is training only)."""
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("oracle"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @staticmethod
    def load(path: Path) -> SuitCNN:
        ckpt = torch.load(path, map_location="cpu")
        net = SuitCNN(CnnConfig(**ckpt["cfg"]))
        net.load_state_dict(ckpt["state"])
        return net


# --- Suit-swap augmentation -------------------------------------------------------


def _tile_perm(perm: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """``new[i] = old[src[i]]`` for the 34 tile kinds when old suit ``s`` becomes
    suit ``perm[s]``."""
    src = np.arange(34)
    for s in range(3):
        src[perm[s] * 9 : perm[s] * 9 + 9] = np.arange(s * 9, s * 9 + 9)
    return src


def feature_perm(perm: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """Column order that swaps suits in a features row (``x[:, feature_perm(p)]``)."""
    t = _tile_perm(perm)
    planes = (np.arange(PLANES)[:, None] * 34 + t[None, :]).reshape(-1)
    return np.concatenate([planes, np.arange(PLANES * 34, STATE_LEN)])


def oracle_perm(perm: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """Column order that swaps suits in an oracle row (four 34-wide blocks)."""
    t = _tile_perm(perm)
    return (np.arange(4)[:, None] * 34 + t[None, :]).reshape(-1)


def action_perm(perm: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """Column order that swaps suits in a 134-wide policy row; ``action_perm(p)[a]`` is
    the old code that becomes code ``a``."""
    t = _tile_perm(perm)
    src = np.arange(NUM_ACTIONS)
    for base in (0, 35, 69):
        src[base : base + 34] = base + t
    src[107:134] = 107 + t[:27]
    return src


def action_map(perm: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """New code of each old action code (the inverse of :func:`action_perm`)."""
    src = action_perm(perm)
    out = np.empty_like(src)
    out[src] = np.arange(len(src))
    return out


def target_map(perm: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """New class of each old target class (255 = none stays 255 via indexing guard)."""
    out = np.arange(NUM_TARGETS)
    for kind in (1, 2):
        for s in range(3):
            for st in range(3):
                out[kind * 9 + s * 3 + st] = kind * 9 + perm[s] * 3 + st
    return out


def target_perm(perm: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """Column order that swaps suits in a 45-wide target row."""
    m = target_map(perm)
    src = np.empty_like(m)
    src[m] = np.arange(NUM_TARGETS)
    return src


#: The six suit permutations.
PERMS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def augment(
    x: npt.NDArray[np.float32],
    legal: npt.NDArray[np.uint8],
    chosen: npt.NDArray[np.int64],
    target: npt.NDArray[np.int64],
    rng: np.random.Generator,
    oracle: npt.NDArray[np.float32] | None = None,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.uint8],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32] | None,
]:
    """Applies one random suit permutation per row to features, legal masks, the
    chosen action codes, target classes (255 = none is kept) and the oracle block.
    Values need no change (suits do not change who wins)."""
    x = x.copy()
    legal = legal.copy()
    chosen = chosen.copy()
    target = target.copy()
    oracle = None if oracle is None else oracle.copy()
    which = rng.integers(0, len(PERMS), size=len(x))
    for i, p in enumerate(PERMS):
        rows = np.flatnonzero(which == i)
        if i == 0 or len(rows) == 0:
            continue
        x[rows] = x[rows][:, feature_perm(p)]
        legal[rows] = legal[rows][:, action_perm(p)]
        chosen[rows] = action_map(p)[chosen[rows]]
        has = target[rows] < NUM_TARGETS
        target[rows[has]] = target_map(p)[target[rows[has]]]
        if oracle is not None:
            oracle[rows] = oracle[rows][:, oracle_perm(p)]
    return x, legal, chosen, target, oracle


# --- Export (MJNN version 3) ------------------------------------------------------


def _name(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def _slices(slices: list[tuple[str, int, int]]) -> bytes:
    out = struct.pack("<I", len(slices))
    for name, offset, length in slices:
        out += _name(name) + struct.pack("<II", offset, length)
    return out


def _tensor(name: str, t: torch.Tensor, int8: bool) -> bytes:
    a = t.detach().cpu().numpy().astype(np.float32)
    head = _name(name)
    if int8 and a.ndim >= 2:
        rows = a.reshape(a.shape[0], -1)
        scale = np.abs(rows).max(axis=1) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(rows / scale[:, None]), -127, 127).astype(np.int8)
        dims = struct.pack("<II", 1, a.ndim) + struct.pack(f"<{a.ndim}I", *a.shape)
        return head + dims + scale.astype("<f4").tobytes() + q.tobytes()
    dims = struct.pack("<II", 0, a.ndim) + struct.pack(f"<{a.ndim}I", *a.shape)
    return head + dims + a.astype("<f4").tobytes(order="C")


def export(model: SuitCNN, path: str | Path, int8: bool = False) -> int:
    """Writes ``model`` as MJNN version 3 (see ``crates/search/src/cnn.rs``); returns
    the file size in bytes. The oracle critic is left out."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.cfg
    hp = {
        "channels": cfg.channels,
        "blocks": cfg.blocks,
        "kernel": cfg.kernel,
        "honour_hidden": cfg.honour_hidden,
        "merge": cfg.merge,
        "value_hidden": cfg.value_hidden,
    }
    tensors = [(n, p) for n, p in model.state_dict().items() if not n.startswith("oracle")]
    out = bytearray(b"MJNN" + struct.pack("<I", 3))
    out += _slices([("features", 0, STATE_LEN)])
    out += _slices(
        [
            ("policy", 0, NUM_ACTIONS),
            ("target", NUM_ACTIONS, NUM_TARGETS),
            ("value", NUM_ACTIONS + NUM_TARGETS, VALUE_LEN),
        ]
    )
    out += _name(ARCH)
    out += struct.pack("<I", len(hp))
    for k, v in hp.items():
        out += _name(k) + struct.pack("<I", v)
    out += struct.pack("<I", len(tensors))
    for n, t in tensors:
        out += _tensor(n, t, int8)
    path.write_bytes(bytes(out))
    return len(out)


def outputs(model: SuitCNN, x: torch.Tensor) -> torch.Tensor:
    """Policy, target and value concatenated as the Rust forward pass returns them
    (``(B, 183)``)."""
    with torch.no_grad():
        p, t, v = model(x)
    return torch.cat([p, t, v], 1)


def main(argv: list[str] | None = None) -> None:
    """``python -m mahjong.rl.cnn --out models/cnn-test.mjnn [--int8] [--seed 0]``:
    exports a randomly initialised default network (for ``net_speed`` and size checks)."""
    import argparse

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    model = SuitCNN()
    size = export(model, args.out, int8=args.int8)
    print(f"{args.out}: {model.parameters_at_play():,} parameters, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
