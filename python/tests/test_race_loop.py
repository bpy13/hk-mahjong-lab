"""Task 9: the stop rule, the best generation and the match arithmetic of
experiments/race_loop.py (``docs/race-bot.md`` §8.3)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _module() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "experiments" / "race_loop.py"
    spec = importlib.util.spec_from_file_location("race_loop", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


loop = _module()
CFG = {"max_gens": 8, "min_gens": 3}


def m(mean: float, half: float = 1.0) -> dict[str, float]:
    return {"mean": mean, "lo": mean - half, "hi": mean + half}


def gen(k: int, value: float, gain: float | None = None) -> dict[str, Any]:
    g: dict[str, Any] = {"generation": k, "value": m(value)}
    if gain is not None:
        g["gain"] = m(gain)
    return g


def test_no_stop_before_three_generations() -> None:
    acc = [gen(0, 0.0), gen(1, 0.1, 0.1), gen(2, 0.2, 0.1)]
    assert loop.stop_reason(acc, CFG) is None


def test_stops_when_gains_and_value_stay_inside_the_noise() -> None:
    acc = [gen(0, 0.0), gen(1, 3.0, 3.0), gen(2, 3.5, 0.5), gen(3, 3.6, 0.2)]
    # The score against value rose 3.6 over the last three generations: keep going.
    assert loop.stop_reason(acc, CFG) is None
    acc.append(gen(4, 3.8, -0.3))
    reason = loop.stop_reason(acc, CFG)
    assert reason is not None and reason.startswith("converged")


def test_a_large_gain_keeps_the_loop_running_and_the_cap_stops_it() -> None:
    acc = [gen(0, 0.0)] + [gen(k, 0.1 * k, 2.0) for k in range(1, 8)]
    assert loop.stop_reason(acc, CFG) is None
    acc.append(gen(8, 0.9, 2.0))
    assert "hard cap" in loop.stop_reason(acc, CFG)


def test_regression_needs_a_loss_beyond_the_noise_with_the_interval_below_zero() -> None:
    assert loop.regressed(m(-1.5, 1.0))
    assert not loop.regressed(m(-1.5, 2.0))  # interval reaches 0
    assert not loop.regressed(m(-1.0, 0.1))  # inside the noise


def test_best_generation_takes_the_latest_within_the_noise_of_the_top() -> None:
    acc = [gen(0, 5.0), gen(1, 1.0), gen(2, 3.0), gen(3, 2.0), gen(4, -1.0)]
    assert loop.best_generation(acc)["generation"] == 3
    assert loop.best_generation(acc[:1]) is None


def _chunk(mean: float, lo: float, hi: float, d: float, hands: int = 200) -> dict[str, Any]:
    agent = {
        "mean": mean,
        "boot_lo": lo,
        "boot_hi": hi,
        "wins": 50,
        "self_drawn": 10,
        "deal_ins": 20,
        "win_faan": 250,
    }
    return {
        "hands": hands,
        "draws": 8,
        "agents": [agent, dict(agent)],
        "pairs": [{"a": 0, "b": 1, "mean": d, "boot_lo": d - 1.96, "boot_hi": d + 1.96}],
    }


def test_summarize_weights_chunks_and_combines_their_intervals() -> None:
    s = loop.summarize(
        "x", ["a", "b", "c", "d"], [_chunk(1.0, 0.0, 2.0, 2.0), _chunk(3.0, 2.0, 4.0, 4.0)]
    )
    assert s["hands"] == 400
    assert s["mean"] == pytest.approx(2.0)
    # Two chunks of standard error 1/1.96 each, half weight: combined se 0.5/1.96 × √2.
    assert s["ci95"] == pytest.approx(1.96 * (2**0.5) * 0.5 / 1.96)
    assert s["minus_agent1"]["mean"] == pytest.approx(3.0)
    assert s["minus_agent1"]["ci95"] == pytest.approx(2**0.5 * 0.5 * 1.96)
    assert s["win_rate"] == pytest.approx(100 / 400)
    assert s["mean_faan"] == pytest.approx(5.0)
