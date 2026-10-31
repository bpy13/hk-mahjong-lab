"""Task 18: the arithmetic of the AC8 report in experiments/phase4_build.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "experiments" / "phase4_build.py"
    spec = importlib.util.spec_from_file_location("phase4_build", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pooled_takes_side_a_minus_side_b_whichever_way_a_pair_is_stored() -> None:
    pairs = [
        {"a": 0, "b": 1, "mean": 2.0, "boot_lo": 1.0, "boot_hi": 3.0},
        {"a": 0, "b": 3, "mean": 4.0, "boot_lo": 1.0, "boot_hi": 7.0},
        {"a": 1, "b": 2, "mean": -6.0, "boot_lo": -7.0, "boot_hi": -5.0},
        {"a": 2, "b": 3, "mean": 8.0, "boot_lo": 7.0, "boot_hi": 9.0},
    ]
    p = _module().pooled({"pairs": pairs}, [0, 2], [1, 3])
    assert p["mean"] == pytest.approx(5.0)  # (2 + 4 + 6 + 8) / 4
    assert p["lo"] == pytest.approx(5.0 - 1.5) and p["hi"] == pytest.approx(5.0 + 1.5)


def test_share_drift_names_patterns_more_than_30_percent_away() -> None:
    drift = _module().share_drift
    pimc = {"a": 0.5, "b": 0.3, "c": 0.19, "rare": 0.01}
    net = {"a": 0.6, "b": 0.1, "c": 0.3}
    # a is +20 %, b is -67 %, c is +58 %; "rare" is below the 2 % floor.
    assert drift(net, pimc) == ["b", "c"]
    # A pattern the network never wins counts as -100 %.
    assert drift({}, {"a": 1.0}) == ["a"]
