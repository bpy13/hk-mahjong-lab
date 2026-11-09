"""The race bot's learned estimates (plan 2026-11-09 Task 8, ``docs/race-bot.md`` Â§6â€“8):
the ``state`` and ``mode`` heads on the belief model, the race log's labels, the fitted
tables and the per-bucket calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
from mahjong.belief.logs import NUM_KINDS, LogDir
from mahjong.belief.model import (
    N_MODES,
    N_OPP,
    N_STATES,
    OUTPUT_LEN,
    BeliefConfig,
    BeliefNet,
    batch_from_rows,
    losses,
    warm_start,
)
from mahjong.belief.race_fit import (
    fit,
    fit_q,
    fit_steps,
    head_calibration,
    label_counts,
    spec_options,
)
from mahjong.rl.model import STATE_LEN

from tests.test_belief import fake_batch, read_mjnn

ROOT = Path(__file__).resolve().parents[2]


def race_batch(n: int, seed: int = 0):
    b = fake_batch(n, seed)
    g = torch.Generator().manual_seed(seed + 100)
    b.state = torch.randint(0, N_STATES, (n, N_OPP), generator=g)
    b.mode = torch.randint(0, N_MODES, (n, N_OPP), generator=g)
    return b


def test_race_heads_add_state_and_mode_after_the_old_heads(tmp_path: Path) -> None:
    net = BeliefNet(BeliefConfig(hidden=(16,), race=True))
    b = race_batch(4)
    out = net(net.inputs(b.features, b.discards))
    assert out.shape == (4, OUTPUT_LEN + N_OPP * (N_STATES + N_MODES))
    p = net.predict(out, b.unseen)
    assert p["state"].shape == (4, N_OPP, N_STATES)
    assert p["mode"].shape == (4, N_OPP, N_MODES)
    assert torch.allclose(p["state"].sum(-1), torch.ones(4, N_OPP))
    assert torch.allclose(p["mode"].sum(-1), torch.ones(4, N_OPP))
    net.export(tmp_path / "r.mjnn")
    _, outs, _ = read_mjnn(tmp_path / "r.mjnn")
    assert [o[0] for o in outs] == ["counts", "tenpai", "waits", "faan", "outcome", "state", "mode"]
    assert outs[5] == ("state", OUTPUT_LEN, N_OPP * N_STATES)
    assert outs[6][1] + outs[6][2] == OUTPUT_LEN + N_OPP * (N_STATES + N_MODES)
    # The old layout is unchanged without the flag.
    plain = BeliefNet(BeliefConfig(hidden=(16,)))
    assert plain(plain.inputs(b.features, b.discards)).shape == (4, OUTPUT_LEN)
    assert "state" not in plain.predict(plain(plain.inputs(b.features, b.discards)), b.unseen)


def test_old_checkpoints_load(tmp_path: Path) -> None:
    net = BeliefNet(BeliefConfig(hidden=(8,)))
    path = tmp_path / "old.pt"
    # A checkpoint written before the race flag existed.
    torch.save({"cfg": {"hidden": [8], "discards": False}, "state": net.state_dict()}, path)
    back = BeliefNet.load(path)
    assert not back.cfg.race
    race = BeliefNet(BeliefConfig(hidden=(8,), race=True))
    race.save(tmp_path / "race.pt")
    assert BeliefNet.load(tmp_path / "race.pt").cfg.race


def test_losses_include_the_race_heads_only_when_asked() -> None:
    b = race_batch(32, seed=2)
    race = BeliefNet(BeliefConfig(hidden=(16,), race=True))
    parts = losses(race, race(race.inputs(b.features, b.discards)), b)
    assert {"state", "mode"} <= set(parts)
    total = sum(float(v.detach()) for k, v in parts.items() if k != "total")
    assert abs(total - float(parts["total"].detach())) < 1e-4
    plain = BeliefNet(BeliefConfig(hidden=(16,)))
    assert "state" not in losses(plain, plain(plain.inputs(b.features, b.discards)), b)


def test_warm_start_copies_the_shared_weights() -> None:
    torch.manual_seed(0)
    old = BeliefNet(BeliefConfig(hidden=(16,)))
    new = BeliefNet(BeliefConfig(hidden=(16,), race=True))
    warm_start(new, old)
    assert torch.equal(new.torso[0].weight, old.torso[0].weight)
    assert torch.equal(new.head.weight[:OUTPUT_LEN], old.head.weight)
    assert torch.equal(new.head.bias[:OUTPUT_LEN], old.head.bias)
    b = race_batch(3)
    o_old = old(old.inputs(b.features, b.discards))
    o_new = new(new.inputs(b.features, b.discards))
    assert torch.allclose(o_new[:, :OUTPUT_LEN], o_old)
    # A race model warm-starts a race model completely.
    newer = BeliefNet(BeliefConfig(hidden=(16,), race=True))
    warm_start(newer, new)
    assert torch.equal(newer.head.weight, new.head.weight)


def race_rows(n: int) -> np.ndarray:
    """Structured rows with the race log's fields (only those the code reads)."""
    dt = np.dtype(
        [
            ("hand", "<u4"),
            ("step", "<u2"),
            ("observer", "u1"),
            ("wall_remaining", "u1"),
            ("agents", "u1", (4,)),
            ("features", "<f2", (STATE_LEN,)),
            ("discards", "u1", (4, 8)),
            ("opp_counts", "u1", (3, NUM_KINDS)),
            ("opp_tenpai", "u1", (3,)),
            ("opp_waits", "u1", (3, NUM_KINDS)),
            ("opp_faan_now", "u1", (3,)),
            ("opp_won", "u1", (3,)),
            ("opp_state", "u1", (3,)),
            ("opp_mode", "u1", (3,)),
            ("opp_shanten", "u1", (3,)),
            ("my_mode", "u1"),
            ("my_feeds", "u1"),
            ("ready_opps", "u1"),
            ("my_mistake", "u1"),
            ("wall_at_end", "u1"),
            ("race_rounds", "u1"),
            ("race_pwin", "<f2", (4,)),
        ]
    )
    rows = np.zeros(n, dtype=dt)
    rows["discards"] = 255
    rows["my_mode"] = 255
    return rows


def test_batch_reads_the_race_labels() -> None:
    rows = race_rows(5)
    rows["opp_state"] = [[0, 1, 2], [3, 0, 0], [2, 2, 2], [1, 1, 1], [0, 3, 1]]
    rows["opp_mode"][1] = [2, 1, 0]
    b = batch_from_rows(rows)
    assert b.state is not None and b.mode is not None
    assert b.state.tolist()[1] == [3, 0, 0]
    assert b.mode.tolist()[1] == [2, 1, 0]
    # A plain belief log has no race labels.
    plain = rows[[n for n in rows.dtype.names if n not in ("opp_state", "opp_mode")]]
    assert batch_from_rows(plain).state is None


def test_q_is_feeds_per_ready_opponent_by_mode() -> None:
    rows = race_rows(6)
    # Push discards: 1 feed over 4 ready opponents; detour: 0 of 2; fold: no discard.
    rows["my_mode"] = [0, 0, 1, 255, 0, 1]
    rows["ready_opps"] = [2, 1, 1, 3, 1, 1]
    rows["my_feeds"] = [1, 0, 0, 1, 0, 0]
    q = fit_q(rows, default=(0.08, 0.04, 0.01))
    assert q["q"] == [0.25, 0.0, 0.01]  # fold keeps its default (no evidence)
    assert q["n_ready"] == [4, 2, 0]


def test_label_counts_split_by_agent_and_threat() -> None:
    rows = race_rows(6)
    # Seats 0 and 2 are agent 0 (race), seats 1 and 3 agent 1 (value).
    rows["agents"] = [0, 1, 0, 1]
    rows["observer"] = [0, 2, 1, 0, 3, 0]
    rows["my_mode"] = [0, 1, 0, 255, 2, 0]
    rows["my_mistake"] = [0, 0, 1, 0, 0, 1]
    rows["ready_opps"] = [0, 1, 0, 0, 2, 1]
    out = label_counts(rows, ["race", "value", "fast"])
    assert set(out) == {"race", "value"}  # "fast" never played
    race = out["race"]
    assert race["all"] == {"push": 1, "detour": 1, "fold": 0, "mistake": 1}
    assert race["calm"] == {"push": 1, "detour": 0, "fold": 0, "mistake": 0}
    assert race["threat"]["detour"] == 1 and race["threat"]["mistake"] == 1
    assert out["value"]["all"] == {"push": 0, "detour": 0, "fold": 1, "mistake": 1}


def test_steps_per_round_from_one_seat_s_true_states() -> None:
    # One hand, absolute seat 1 seen by observers 0, 2, 3 in turn; its true shanten
    # goes 2, 2, 1, 1, 0 as the wall falls by 2 per row (a round is 4 wall tiles).
    walls = [80, 78, 76, 74, 72, 70, 68]
    shanten = [2, 2, 2, 1, 1, 1, 0]
    rows = race_rows(len(walls))
    rows["hand"] = 3
    rows["step"] = np.arange(len(walls))
    rows["observer"] = [0, 2, 3, 0, 2, 3, 0]
    rows["wall_remaining"] = walls
    for i, obs in enumerate(rows["observer"]):
        r = (1 - int(obs)) % 4  # seat 1 relative to the observer
        rows["opp_shanten"][i, r - 1] = shanten[i]
        rows["opp_tenpai"][i, r - 1] = int(shanten[i] == 0)
        rows["opp_state"][i, r - 1] = 2 if shanten[i] == 0 else (1 if shanten[i] == 1 else 0)
    steps = fit_steps(rows, default_a=[0.15] * 6, default_b=[0.15] * 6, min_n=1)
    # F observations at walls 80, 78, 76 â†’ one round later (â‰¤ 76, â‰¤ 74, â‰¤ 72): F, O, O.
    # O observations at 74, 72, 70 â†’ â‰¤ 70, â‰¤ 68, (none): O, R. Walls 70-80 of 91 fall
    # in buckets 0 and 1.
    a, b = steps["a"], steps["b"]
    assert steps["n_a"][0] + steps["n_a"][1] == 3
    assert abs((a[0] * steps["n_a"][0] + a[1] * steps["n_a"][1]) - 2) < 1e-9
    assert steps["n_b"][0] + steps["n_b"][1] == 2
    assert abs((b[0] * steps["n_b"][0] + b[1] * steps["n_b"][1]) - 1) < 1e-9


def test_head_calibration_has_a_row_per_bucket_and_class() -> None:
    rows = race_rows(40)
    rows["wall_remaining"] = np.linspace(90, 0, 40).astype(np.uint8)
    rows["opp_state"] = np.random.default_rng(0).integers(0, 4, (40, 3))
    rows["opp_mode"] = np.random.default_rng(1).integers(0, 3, (40, 3))
    probs_state = np.full((40, 3, N_STATES), 1 / N_STATES)
    probs_mode = np.full((40, 3, N_MODES), 1 / N_MODES)
    cal = head_calibration(rows, probs_state, probs_mode)
    assert set(cal) == {"state", "mode"}
    assert set(cal["state"]) == {"far", "one_away", "ready", "folded"}
    assert set(cal["mode"]) == {"push", "detour", "fold"}
    assert len(cal["state"]["far"]) == 6
    assert sum(r["n"] for r in cal["mode"]["push"]) == 120


RACE_LOG = ROOT / "runs" / "race-smoke" / "logs"


@pytest.mark.skipif(not (RACE_LOG / "meta.json").exists(), reason="no race log")
def test_a_real_race_log_trains_and_fits() -> None:
    """Rows of the ``race_log`` example (``cargo run --release -p mahjong-search
    --example race_log -- --hands 200 --out runs/race-smoke/logs``)."""
    log = LogDir(RACE_LOG)
    assert log.meta["layout"] == "race"
    rows = np.asarray(log.chunk(0))
    b = batch_from_rows(rows[:256])
    assert b.state is not None and int(b.state.max()) <= 3
    # Unfolded opponents are ready exactly when they have a legal win now.
    st, ten = rows["opp_state"], rows["opp_tenpai"].astype(bool)
    assert ((st == 2) == ten)[st != 3].all()
    net = BeliefNet(BeliefConfig(hidden=(16,), race=True))
    parts = losses(net, net(net.inputs(b.features, b.discards)), b)
    assert all(torch.isfinite(v) for v in parts.values())
    steps = fit_steps(rows, min_n=1)
    assert all(0.0 <= x <= 1.0 for x in steps["a"] + steps["b"])
    assert all(0.0 <= x <= 1.0 for x in fit_q(rows)["q"])
    # The whole fit, as the loop runs it (label fields only, chunk by chunk).
    out = fit(argparse.Namespace(logs=RACE_LOG, holdout=None, model=None, min_n=1))
    assert out["rows"] == log.rows
    assert out["spec"].startswith("a=") and len(out["hazard"]) == 6
    # Every own discard is counted once, under the agent that made it.
    by_agent = out["labels"]["by_agent"]
    assert sum(sum(p["all"].values()) for p in by_agent.values()) == sum(out["labels"]["my_mode"])


def test_spec_options_are_race_spec_text() -> None:
    s = spec_options({"a": [0.1] * 6, "b": [0.2] * 5 + [0.25]}, {"q": [0.07, 0.03, 0.005]})
    assert s == "a=0.1/0.1/0.1/0.1/0.1/0.1,b=0.2/0.2/0.2/0.2/0.2/0.25,q=0.07/0.03/0.005"
