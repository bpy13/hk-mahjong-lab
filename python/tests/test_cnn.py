"""The suit-shared CNN (plan Task 14): shapes, suit-swap equivariance, augmentation,
and the MJNN v3 export run in Rust (fp32 within 1e-4; int8 top-1 agreement >= 98 %)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from mahjong._core import net_forward, teacher_data
from mahjong.rl.cnn import (
    NUM_TARGETS,
    OUTPUT_LEN,
    PERMS,
    STATE_LEN,
    CnnConfig,
    SuitCNN,
    action_map,
    action_perm,
    augment,
    export,
    feature_perm,
    outputs,
    target_map,
    target_perm,
)
from mahjong.rl.model import ORACLE_LEN, to_arrays


@pytest.fixture(scope="module")
def positions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """About 1,000 real decisions (features, legal masks, actions) from `value` self-play
    on seeds no training run uses (held out)."""
    feats, oracle, masks, actions, _, _ = teacher_data(48, teacher="value", seed=990_001)
    f, _, m = to_arrays(feats, oracle, masks)
    n = min(1000, len(f))
    return f[:n].copy(), m[:n].copy(), np.asarray(actions[:n], dtype=np.int64)


@pytest.fixture(scope="module")
def model() -> SuitCNN:
    torch.manual_seed(0)
    return SuitCNN().eval()


def test_shapes_parameters_and_zero_sum(model: SuitCNN, positions) -> None:
    f, _, _ = positions
    x = torch.from_numpy(f[:16])
    p, t, v = model(x)
    assert p.shape == (16, 134) and t.shape == (16, NUM_TARGETS) and v.shape == (16, 4)
    assert torch.allclose(v.sum(1), torch.zeros(16), atol=1e-5)
    # The 18 suit-less classes with suit 1 or 2 are fixed and never win.
    unused = [k * 9 + s * 3 + st for k in (0, 3, 4) for s in (1, 2) for st in range(3)]
    assert (t[:, unused] == -1.0e4).all()
    assert 1_500_000 <= model.parameters_at_play() <= 5_000_000
    # The oracle critic changes only the value head and is not counted at play time.
    om = SuitCNN(CnnConfig(channels=8, blocks=1, honour_hidden=16, merge=16, oracle=True))
    o = torch.rand(16, ORACLE_LEN)
    p0, t0, v0 = om(x)
    p1, t1, v1 = om(x, o)
    assert torch.equal(p0, p1) and torch.equal(t0, t1) and not torch.equal(v0, v1)


@pytest.mark.parametrize("perm", PERMS[1:])
def test_suit_swap_equivariance(model: SuitCNN, positions, perm) -> None:
    f, _, _ = positions
    x = torch.from_numpy(f[:64])
    xs = x[:, torch.from_numpy(feature_perm(perm))]
    p, t, v = model(x)
    ps, ts, vs = model(xs)
    assert torch.allclose(ps, p[:, torch.from_numpy(action_perm(perm))], atol=1e-4)
    assert torch.allclose(ts, t[:, torch.from_numpy(target_perm(perm))], atol=1e-4)
    assert torch.allclose(vs, v, atol=1e-4)
    # The maps are each other's inverses.
    assert (action_map(perm)[action_perm(perm)] == np.arange(134)).all()
    assert (target_map(perm)[target_perm(perm)] == np.arange(NUM_TARGETS)).all()


def test_augment_keeps_moves_legal_and_logits_attached(model: SuitCNN, positions) -> None:
    f, m, a = positions
    n = 200
    target = np.full(n, 255, dtype=np.int64)
    target[::2] = np.arange(0, n, 2) % NUM_TARGETS
    xa, ma, aa, ta, _ = augment(f[:n], m[:n], a[:n], target, np.random.default_rng(1))
    assert ma[np.arange(n), aa].all()
    assert ((ta == 255) == (target == 255)).all()
    p, _, _ = model(torch.from_numpy(f[:n]))
    pa, _, _ = model(torch.from_numpy(xa))
    rows = torch.arange(n)
    assert torch.allclose(
        pa[rows, torch.from_numpy(aa)], p[rows, torch.from_numpy(a[:n])], atol=1e-4
    )


def _rust(path: Path, x: np.ndarray) -> np.ndarray:
    out, (inp, n_out) = net_forward(str(path), x.astype("<f4").tobytes())
    assert (inp, n_out) == (STATE_LEN, OUTPUT_LEN)
    return np.frombuffer(out, dtype="<f4").reshape(len(x), n_out)


def test_export_runs_the_same_in_rust(model: SuitCNN, positions, tmp_path: Path) -> None:
    f, m, _ = positions
    assert len(f) >= 1000
    want = outputs(model, torch.from_numpy(f)).numpy()

    fp32 = tmp_path / "cnn.mjnn"
    size32 = export(model, fp32)
    got = _rust(fp32, f)
    assert np.abs(got - want).max() < 1e-4

    int8 = tmp_path / "cnn-int8.mjnn"
    size8 = export(model, int8, int8=True)
    assert size8 < 0.3 * size32
    q = _rust(int8, f)
    legal = m.astype(bool)
    top = np.where(legal, want[:, :134], -np.inf).argmax(1)
    top8 = np.where(legal, q[:, :134], -np.inf).argmax(1)
    agreement = float((top == top8).mean())
    print(f"fp32 {size32 / 1e6:.2f} MB, int8 {size8 / 1e6:.2f} MB, top-1 agreement {agreement:.4f}")
    assert agreement >= 0.98
