"""RL plumbing: encoding, a tiny run of each learner, export to Rust."""

from pathlib import Path

import numpy as np
import torch
from mahjong._core import HkState, encode_state, feature_sizes, tournament
from mahjong.rl.distill import distill
from mahjong.rl.dmc import dmc
from mahjong.rl.model import INPUT_LEN, Net
from mahjong.rl.ppo import ppo


def test_encoding_shape_and_mask() -> None:
    s = HkState(3)
    f, o, m = encode_state(s)
    state_len, oracle_len, actions = feature_sizes()
    assert len(f) == state_len and len(o) == oracle_len and len(m) == actions
    assert sorted(np.flatnonzero(m).tolist()) == sorted(s.legal_actions())


def test_tiny_training_runs_and_exports(tmp_path: Path) -> None:
    d = distill(tmp_path / "d.pt", hands=20, chunk=10)
    x = torch.zeros(2, INPUT_LEN)
    q, pi, v = d(x)
    assert q.shape == pi.shape == (2, 134) and v.shape == (2,)
    dmc(
        tmp_path / "q.pt",
        init=tmp_path / "d.pt",
        minutes=0.05,
        envs=16,
        hands_per_iter=8,
        eval_every=1,
        eval_deals=1,
    )
    ppo(
        tmp_path / "p.pt",
        init=tmp_path / "d.pt",
        minutes=0.05,
        envs=16,
        hands_per_iter=8,
        eval_every=1,
        eval_deals=1,
    )
    for name in ("q", "p"):
        assert (tmp_path / f"{name}.mjnn").exists()
    # The Rust side loads and plays the exported networks.
    res = tournament(
        [
            f"net:path={tmp_path / 'q.mjnn'}",
            f"netpimc:path={tmp_path / 'p.mjnn'},samples=4",
            "fast",
            "fast",
        ],
        1,
    )
    assert res["hands"] == 4
    assert Net.load(tmp_path / "q.pt") is not None
