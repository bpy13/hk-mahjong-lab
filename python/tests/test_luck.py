"""Plan 2026-11-16 Task 29: the luck-removal switches of the self-play trainer and the
variance measurement (``mahjong.rl.luck``)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mahjong._core import net_forward
from mahjong.rl.cnn import ORACLE_LEN, STATE_LEN, CnnConfig, SuitCNN, export, load_state
from mahjong.rl.luck import SETTINGS, LuckConfig, run, unsquash
from mahjong.rl.ppo_cnn import (
    PpoCnnConfig,
    Trainer,
    advantages,
    load_init,
    main,
    next_values,
    squash,
    uses_critic,
)

torch.set_num_threads(1)

SMALL = CnnConfig(channels=16, blocks=1, honour_hidden=32, merge=32, value_hidden=16)


def _cfg(out: Path, init: Path, **kw: Any) -> PpoCnnConfig:
    base = PpoCnnConfig(
        out=out,
        init=init,
        seconds=600.0,
        seed=11,
        envs=16,
        device="cpu",
        hands_per_iter=16,
        epochs=1,
        batch=256,
        curriculum=False,
        anchor_weight=0.0,
        eval_minutes=1e9,
        pool_minutes=1e9,
        score_minutes=1e9,
        ckpt_minutes=1e9,
    )
    return replace(base, **kw)


def test_uses_critic_follows_the_switches() -> None:
    out, init = Path("x"), None
    # As before: no critic in paired mode, the critic in unpaired mode.
    assert uses_critic(PpoCnnConfig(out=out, init=init)) is False
    assert uses_critic(PpoCnnConfig(out=out, init=init, paired=False)) is True
    assert uses_critic(PpoCnnConfig(out=out, init=init, oracle_baseline=True)) is True
    unpaired_off = PpoCnnConfig(out=out, init=init, paired=False, oracle_baseline=False)
    assert uses_critic(unpaired_off) is False
    assert uses_critic(PpoCnnConfig(out=out, init=init, per_decision=True)) is True
    for s in SETTINGS.values():
        PpoCnnConfig(out=out, init=init, **s)


def test_next_values_and_advantages() -> None:
    # Two hands, rows interleaved in time: hand 7 has rows 0, 2, 3; hand 9 has rows 1, 4.
    uid = np.array([7, 9, 7, 7, 9])
    value = np.array([1.0, 10.0, 2.0, 3.0, 20.0], dtype=np.float32)
    reward = np.array([0.5, -1.0, 0.5, 0.5, -1.0], dtype=np.float32)
    nxt = next_values(uid, value, reward)
    assert nxt.tolist() == [2.0, 20.0, 3.0, 0.5, -1.0]

    d = {
        "uid": uid,
        "reward": reward,
        "vpaired": value,
        "value": np.stack([value * 2, -value, value, -value * 2], 1),
    }
    cfg = PpoCnnConfig(out=Path("x"), init=None)
    adv, critic = advantages(d, cfg)
    assert adv.tolist() == reward.tolist() and not critic.any()
    adv, critic = advantages(d, replace(cfg, oracle_baseline=True))
    assert np.allclose(adv, reward - value) and np.allclose(critic, value)
    adv, _ = advantages(d, replace(cfg, paired=False))  # unpaired: the four-seat critic
    assert np.allclose(adv, reward - value * 2)
    adv, _ = advantages(d, replace(cfg, paired=False, oracle_baseline=False))
    assert np.allclose(adv, reward)
    adv, _ = advantages(d, replace(cfg, per_decision=True))
    assert np.allclose(adv, nxt - value)
    # Per-decision credit sums to the whole hand's advantage over the hand's rows.
    assert np.isclose(adv[[0, 2, 3]].sum(), reward[0] - value[0])

    x = np.array([-30.0, 0.0, 3.0, 100.0])
    assert np.allclose(unsquash(squash(x)), x, atol=1e-4)


def test_paired_head_is_training_only_and_old_files_load(tmp_path: Path) -> None:
    torch.manual_seed(3)
    SuitCNN(SMALL).save(tmp_path / "init.pt")  # a start network without the oracle block
    net = load_init(tmp_path / "init.pt")
    # The hidden view starts with no effect, the paired head at zero.
    assert net.oracle_value.weight.abs().sum() == 0 and net.oracle_paired.weight.abs().sum() == 0
    x = torch.rand(5, STATE_LEN)
    o1, o2 = torch.rand(5, ORACLE_LEN), torch.rand(5, ORACLE_LEN)
    torch.nn.init.normal_(net.oracle_paired.weight)
    torch.nn.init.normal_(net.oracle_value.weight)
    with torch.no_grad():
        p1, t1, v1, vp1 = net.forward_all(x, o1)
        p2, t2, v2, vp2 = net.forward_all(x, o2)
        p3, t3, v3 = net(x)
        z = net.trunk(x)
        v4, vp4 = net.critics(z, o1)
    assert torch.equal(p1, p2) and torch.equal(t1, t2) and torch.equal(p1, p3)
    assert not torch.allclose(vp1, vp2)
    assert torch.allclose(v1, v4) and torch.allclose(vp1, vp4) and torch.allclose(v1, v3) is False
    assert net.parameters_at_play() == SuitCNN(SMALL).parameters_at_play()

    # A file saved before the head existed loads, with the head at zero.
    state = {k: v for k, v in net.state_dict().items() if not k.startswith("oracle_paired")}
    other = SuitCNN(net.cfg)
    load_state(other, state)
    assert other.oracle_paired.weight.abs().sum() == 0
    with torch.no_grad():
        assert torch.equal(other(x, o1)[0], p1)


def test_switches_do_not_change_play(tmp_path: Path) -> None:
    """The same weights and seed give the same rollout with the switches on or off (the
    hidden view reaches only the training value heads), and after an update with them
    on, the exported network plays what the PyTorch policy says without the hidden view."""
    init = tmp_path / "init.pt"
    torch.manual_seed(5)
    SuitCNN(SMALL).save(init)
    batches = []
    settings: tuple[tuple[str, dict[str, Any]], ...] = (
        ("off", {}),
        ("both", {"oracle_baseline": True}),
        ("td", {"per_decision": True}),
    )
    for name, s in settings:
        tr = Trainer(_cfg(tmp_path / name, init, **s))
        tr.env.set_min_faan(0)
        batch, _ = tr.collect()
        batches.append((tr, batch))
    (_, b0), (tr1, b1), (_, b2) = batches
    for b in (b1, b2):
        assert np.array_equal(b0["actions"], b["actions"]) and np.array_equal(b0["uid"], b["uid"])
        assert np.allclose(b0["logp"], b["logp"]) and np.allclose(b0["reward"], b["reward"])
        assert np.allclose(b0["vpaired"], b["vpaired"])

    losses = tr1.update(b1)
    assert np.isfinite(losses["adv_std"]) and "critic_r2" in losses
    path = tmp_path / "net.mjnn"
    export(tr1.net, path)
    f = b1["feats"].astype(np.float32)
    out, (inp, n_out) = net_forward(str(path), f.astype("<f4").tobytes())
    rust = np.frombuffer(out, dtype="<f4").reshape(len(f), n_out)
    tr1.net.eval()
    with torch.no_grad():
        pi, _, _ = tr1.net(torch.from_numpy(f))
    assert np.abs(rust[:, :134] - pi.numpy()).max() < 1e-3


def test_run_with_the_switches_keeps_hourly_copies(tmp_path: Path) -> None:
    init = tmp_path / "init.pt"
    torch.manual_seed(6)
    SuitCNN(SMALL).save(init)
    out = tmp_path / "run"
    main(
        ["--init", str(init), "--minutes", "5", "--out", str(out), "--envs", "16"]
        + ["--device", "cpu", "--hands-per-iter", "8", "--max-iterations", "3"]
        + ["--no-curriculum", "--anchor-weight", "0", "--oracle-baseline"]
        + ["--per-decision-credit", "--keep-hourly", "--score-minutes", "0.001"]
    )
    log = json.loads((out / "log.json").read_text())
    assert len(log) == 3
    assert all(np.isfinite(e["policy_loss"]) and np.isfinite(e["critic_r2"]) for e in log)
    hours = [e["hour"] for e in log if "hour" in e]
    assert hours and all((out / h["network"]).exists() for h in hours)
    assert (out / hours[0]["network"]).with_suffix(".mjnn").exists()
    state = json.loads((out / "state.json").read_text())
    assert state["config"]["oracle_baseline"] is True and state["config"]["per_decision"] is True


def test_luck_measurement_smoke(tmp_path: Path) -> None:
    init = tmp_path / "init.pt"
    torch.manual_seed(7)
    SuitCNN(SMALL).save(init)
    res = run(
        LuckConfig(
            init=init,
            out=tmp_path / "luck",
            hands=40,
            envs=8,
            device="cpu",
            fit_share=0.5,
            epochs=1,
            chunk=8,
            time_hands=8,
        )
    )
    r = res["results"]
    assert set(r) >= {f"{k}" for k in SETTINGS} | {"both, no hidden view"}
    for row in r.values():
        assert row["decisions"] > 0 and np.isfinite(row["adv_std"]) and row["hands_per_hour"] > 0
    assert r["same-deal baseline"]["hour_noise"] > 0
    assert (tmp_path / "luck" / "variance.md").read_text(encoding="utf-8").count("|") > 40
    assert res["fit_hands"] + res["held_out_hands"] >= 40


def test_unpaired_run_keeps_hourly_copies_and_stops_at_time(tmp_path: Path) -> None:
    """Task 29's ``oracle`` setting (unpaired, no pool): the hourly row has no paired
    figure but still comes with its copy, and the run's end writes ``stopped.txt`` so a
    resume ends at once and the final match can start."""
    init = tmp_path / "init.pt"
    torch.manual_seed(8)
    SuitCNN(SMALL).save(init)
    out = tmp_path / "run"
    args = (
        ["--init", str(init), "--minutes", "0.001", "--out", str(out), "--envs", "16"]
        + ["--device", "cpu", "--hands-per-iter", "8", "--no-curriculum", "--anchor-weight", "0"]
        + ["--no-paired-reward", "--oracle-baseline", "--keep-hourly", "--score-minutes", "0.001"]
    )
    main(args)
    log = json.loads((out / "log.json").read_text())
    hours = [e["hour"] for e in log if "hour" in e]
    assert hours and hours[0]["paired_mean"] is None and hours[0]["ci95"] is None
    assert hours[0]["hands"] > 0 and hours[0]["updates"] == 1
    assert (out / hours[0]["network"]).exists()
    assert (out / hours[0]["network"]).with_suffix(".mjnn").exists()
    assert "n/a (unpaired)" in (out / "hourly.txt").read_text()
    assert (out / "stopped.txt").read_text().startswith("time:")
    assert (out / "final.mjnn").exists()
    state = json.loads((out / "state.json").read_text())
    assert state["config"]["paired"] is False
    main([*args, "--resume"])
    assert json.loads((out / "state.json").read_text())["iteration"] == state["iteration"]
