"""The network: one MLP torso with three heads over the 134 action codes.

* ``q``: predicted final points of the acting player for each action
  (Deep Monte Carlo), divided by :data:`POINT_SCALE`;
* ``pi``: policy logits (imitation and PPO);
* ``v``: predicted final points of the acting player (PPO baseline).

Inputs are ``mahjong._core`` features (``STATE_LEN``) followed by oracle features
(``ORACLE_LEN``, zero at play time). Networks are exported to the ``MJNN`` format
that ``crates/search/src/net.rs`` runs, as the torso followed by one head.
"""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mahjong._core import feature_sizes

STATE_LEN, ORACLE_LEN, NUM_ACTIONS = feature_sizes()
INPUT_LEN = STATE_LEN + ORACLE_LEN
POINT_SCALE = 32.0
"""Points are divided by this for training (one 6-faan share is 32 points)."""


@dataclass
class NetConfig:
    hidden: tuple[int, ...] = (512, 512, 256)


class Net(nn.Module):
    def __init__(self, cfg: NetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or NetConfig()
        layers: list[nn.Module] = []
        width = INPUT_LEN
        for h in self.cfg.hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.torso = nn.Sequential(*layers)
        self.q = nn.Linear(width, NUM_ACTIONS)
        self.pi = nn.Linear(width, NUM_ACTIONS)
        self.v = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.torso(x)
        return self.q(h), self.pi(h), self.v(h).squeeze(-1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict()}, path)

    @staticmethod
    def load(path: Path) -> Net:
        ckpt = torch.load(path, map_location="cpu")
        cfg = NetConfig(hidden=tuple(ckpt["cfg"]["hidden"]))
        net = Net(cfg)
        net.load_state_dict(ckpt["state"])
        return net

    def export(self, path: Path, head: str) -> None:
        """Writes the torso and one head (``"q"`` or ``"pi"``) as an MJNN file."""
        linears = [m for m in self.torso if isinstance(m, nn.Linear)] + [getattr(self, head)]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"MJNN" + struct.pack("<II", 1, len(linears)))
            for lin in linears:
                w = lin.weight.detach().cpu().numpy().astype("<f4")
                b = lin.bias.detach().cpu().numpy().astype("<f4")
                f.write(struct.pack("<II", w.shape[0], w.shape[1]))
                f.write(w.tobytes(order="C"))
                f.write(b.tobytes())


def masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Illegal actions get a very low score."""
    return logits.masked_fill(mask == 0, -1e9)


def to_arrays(
    feats: bytes, oracle: bytes, masks: bytes
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Views the bytes from ``mahjong._core`` as arrays."""
    f = np.frombuffer(feats, dtype="<f4").reshape(-1, STATE_LEN)
    o = np.frombuffer(oracle, dtype="<f4").reshape(-1, ORACLE_LEN)
    m = np.frombuffer(masks, dtype=np.uint8).reshape(-1, NUM_ACTIONS).copy()
    return f, o, m


def inputs(f: np.ndarray, o: np.ndarray, keep_oracle: float = 0.0) -> torch.Tensor:
    """Network input: features plus oracle features, each oracle row kept with
    probability ``keep_oracle`` (else zeroed, as at play time)."""
    x = np.concatenate([f, o * (np.random.random((len(o), 1)) < keep_oracle)], axis=1)
    return torch.from_numpy(x.astype(np.float32))
