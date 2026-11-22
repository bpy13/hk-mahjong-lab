"""Plan 2026-11-16 Task 29, part 1: how much luck each baseline takes out of the reward.

Plays one fixed set of self-play hands with the start network exactly as the trainer's
rollout does (paired mode, the opponent pool with no older member yet: the current
network and ``stage``), and keeps per learner decision the trunk output ``z`` of
:class:`~mahjong.rl.cnn.SuitCNN`, the hidden view, the paired reward (learner minus the
frozen start on the same tiles, squashed), the learner's own squashed points, the hand
and the seat. The critic heads are fitted on the first ``fit_share`` of the hands with
the trunk frozen (the trainer fits them online, trunk included, so a long run's critic
is at least this good) and every setting is measured on the rest:

| Setting | Advantage of a decision before PPO's normalisation |
|---|---|
| neither | ``sq(own points)`` |
| oracle critic | ``sq(own points) - V(z, hidden view)`` (the four-seat critic's own seat) |
| same-deal baseline | ``sq(own points - the start's points on the same deal)`` (Task 10) |
| both | the paired reward ``- Vp(z, hidden view)`` (the paired-reward head) |
| both, per-decision credit | ``Vp`` at the next own decision (the reward after the last) ``- Vp`` |

Per-decision credit telescopes: a hand's summed advantage is the hand's result (the reward
minus the critic's first value) given once, where the whole-hand settings give the reward to
every decision of the hand. With a poor critic (R² near 0) that one credit lands on the last
decision alone: less spread, but also fewer decisions that learn anything. Read its rows with
the critic's R² beside them.

plus the two critic rows with the hidden view zeroed, to show what the hidden view adds.
For each: the standard deviation of the advantage per decision, per hand (the mean over
the hand's decisions), in raw points where the advantage is a whole hand's result, the
critic's held-out R², and the hands, learner seat-hands and decisions per hour of that
setting (the paired rollout's own timing; an unpaired rollout of ``--time-hands`` hands
for the two unpaired settings; plus one update per setting timed on the last chunk),
from which the noise of one hour's mean advantage follows (the lower, the more an hour
of training tells). ``variance.md``, ``variance.json`` and ``summary.json`` in ``--out``.

CLI::

    python -m mahjong.rl.luck --init models/stage-imitation.pt --out runs/luck
        [--hands 50000] [--envs 256] [--device cuda|cpu] [--seed 29] [--fit-share 0.8]
        [--epochs 5] [--chunk 2000] [--time-hands 4000]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mahjong.rl.cnn import SuitCNN
from mahjong.rl.ppo_cnn import PpoCnnConfig, Trainer, next_values, no_power_throttling

#: The settings as the trainer's switches spell them (``--no-paired-reward``,
#: ``--oracle-baseline`` / ``--no-oracle-baseline``, ``--per-decision-credit``).
SETTINGS: dict[str, dict[str, Any]] = {
    "neither": {"paired": False, "oracle_baseline": False, "per_decision": False},
    "oracle critic": {"paired": False, "oracle_baseline": True, "per_decision": False},
    "same-deal baseline": {"paired": True, "oracle_baseline": False, "per_decision": False},
    "both": {"paired": True, "oracle_baseline": True, "per_decision": False},
    "both, per-decision credit": {"paired": True, "oracle_baseline": True, "per_decision": True},
}


@dataclass
class LuckConfig:
    init: Path
    out: Path
    hands: int = 50_000
    envs: int = 256
    device: str = "cuda"
    seed: int = 29
    #: Share of the hands the critic heads are fitted on; the rest is measured.
    fit_share: float = 0.8
    epochs: int = 5
    #: Hands per rollout chunk (the rows of a chunk are reduced to ``z`` at once).
    chunk: int = 2_000
    #: Hands of the unpaired rollout that times the unpaired settings.
    time_hands: int = 4_000
    torch_threads: int = 1
    lr: float = 1e-3
    batch: int = 4096


def unsquash(y: np.ndarray) -> np.ndarray:
    """Inverse of :func:`mahjong.rl.ppo_cnn.squash`: raw points from the log scale."""
    a = np.asarray(y, dtype=np.float64)
    return np.sign(a) * (np.expm1(np.abs(a)) * 8.0)


class Critic(nn.Module):
    """A copy of a network's value heads, fitted on cached trunk outputs; ``hidden``
    False zeroes the hidden view (the critic then sees only what the player sees)."""

    def __init__(self, net: SuitCNN, hidden: bool) -> None:
        super().__init__()
        self.value1 = copy.deepcopy(net.value1)
        self.value2 = copy.deepcopy(net.value2)
        self.oracle_value = copy.deepcopy(net.oracle_value)
        self.oracle_paired = copy.deepcopy(net.oracle_paired)
        self.hidden = hidden

    def forward(self, z: torch.Tensor, oracle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.value1(z)
        if self.hidden:
            pre = pre + self.oracle_value(oracle)
        h = F.relu(pre)
        v = self.value2(h)
        return v - v.mean(1, keepdim=True), self.oracle_paired(h)[:, 0]


def trainer_config(cfg: LuckConfig, paired: bool) -> PpoCnnConfig:
    """The trainer as Task 10's run had it, for rollouts only (no result, no pool
    member, no hourly row in the time of this measurement)."""
    return PpoCnnConfig(
        out=cfg.out / ("paired" if paired else "unpaired"),
        init=cfg.init,
        seconds=math.inf,
        seed=cfg.seed,
        paired=paired,
        pool=paired,
        curriculum=False,
        anchor_weight=0.0,
        epochs=1,
        hands_per_iter=cfg.chunk,
        envs=cfg.envs,
        device=cfg.device,
        torch_threads=cfg.torch_threads,
        eval_minutes=1e9,
        pool_minutes=1e9,
        score_minutes=1e9,
        ckpt_minutes=1e9,
    )


def trainer(cfg: LuckConfig, paired: bool) -> Trainer:
    tr = Trainer(trainer_config(cfg, paired))
    tr.env.set_min_faan(3)  # the real rules, as the run plays them without the curriculum
    tr.min_faan = tr.stats.student_min_faan = 3
    return tr


def trunk_of(net: SuitCNN, feats: np.ndarray, device: torch.device, bs: int = 2048) -> np.ndarray:
    """``net.trunk`` of ``feats`` (float16 rows) in batches, as float16."""
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), bs):
            x = torch.from_numpy(feats[i : i + bs].astype(np.float32)).to(device)
            out.append(net.trunk(x).half().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, net.cfg.merge), dtype=np.float16)


def reduce_batch(tr: Trainer, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Keeps of a collected batch what the measurement needs, with ``z`` for the inputs."""
    tr.net.eval()
    return {
        "z": trunk_of(tr.net, batch["feats"], tr.device),
        "oracle": batch["oracle"],
        "reward": batch["reward"].astype(np.float32),
        "returns": batch["returns"].astype(np.float32),
        "uid": batch["uid"].astype(np.int64),
        "seat": batch["seat"].astype(np.int64),
    }


def collect(
    tr: Trainer, hands: int, keep: bool
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Plays at least ``hands`` hands with ``tr`` (chunks of ``hands_per_iter``);
    returns the reduced rows (when ``keep``) and the timing. The last chunk's full
    batch is kept in the timing dict for timing an update."""
    rows: list[dict[str, np.ndarray]] = []
    decisions = 0
    t0 = time.time()
    last: dict[str, np.ndarray] | None = None
    hands0 = tr.hands
    while tr.hands - hands0 < hands:
        batch, _info = tr.collect()
        decisions += len(batch["uid"])
        if keep:
            rows.append(reduce_batch(tr, batch))
        last = batch
    elapsed = time.time() - t0
    played = tr.hands - hands0
    seat_hands = played * (1 if tr.cfg.paired else 4)
    timing = {
        "hands": played,
        "seat_hands": seat_hands,
        "decisions": decisions,
        "collect_s": round(elapsed, 2),
        "collect_s_per_hand": elapsed / played,
        "decisions_per_seat_hand": decisions / seat_hands,
        "last_batch": last,
    }
    if not keep:
        return None, timing
    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    return out, timing


def time_update(tr: Trainer, batch: dict[str, np.ndarray], setting: dict[str, Any]) -> float:
    """Seconds of one ``update`` (one pass, as the run) on ``batch`` under ``setting``;
    the network's weights are restored after it."""
    state = copy.deepcopy(tr.net.state_dict())
    old = tr.cfg
    tr.cfg = replace(old, **{k: v for k, v in setting.items() if k != "paired"}, epochs=1)
    if tr.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    tr.update(batch)
    if tr.device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    tr.cfg = old
    tr.net.load_state_dict(state)
    return dt


def fit_critic(
    critic: Critic,
    rows: dict[str, np.ndarray],
    fit: np.ndarray,
    cfg: LuckConfig,
    device: torch.device,
) -> list[float]:
    """Fits the heads on the rows ``fit`` (a boolean mask): MSE of the four-seat values
    on the squashed returns and of the paired head on the paired reward; returns the
    loss per epoch."""
    z = torch.from_numpy(rows["z"][fit])
    o = torch.from_numpy(rows["oracle"][fit])
    r4 = torch.from_numpy(rows["returns"][fit])
    rp = torch.from_numpy(rows["reward"][fit])
    opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    n = len(z)
    losses = []
    critic.train()
    for _ in range(cfg.epochs):
        perm = torch.from_numpy(rng.permutation(n))
        total, steps = 0.0, 0
        for i in range(0, n, cfg.batch):
            idx = perm[i : i + cfg.batch]
            zb = z[idx].to(device).float()
            ob = o[idx].to(device).float()
            v, vp = critic(zb, ob)
            loss = F.mse_loss(v, r4[idx].to(device)) + F.mse_loss(vp, rp[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        losses.append(total / max(1, steps))
    return losses


def predict(
    critic: Critic,
    rows: dict[str, np.ndarray],
    mask: np.ndarray,
    device: torch.device,
    bs: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Own-seat value and paired value of the rows in ``mask``."""
    critic.eval()
    v0, vp = [], []
    z, o = rows["z"][mask], rows["oracle"][mask]
    with torch.no_grad():
        for i in range(0, len(z), bs):
            zb = torch.from_numpy(z[i : i + bs].astype(np.float32)).to(device)
            ob = torch.from_numpy(o[i : i + bs].astype(np.float32)).to(device)
            v, p = critic(zb, ob)
            v0.append(v[:, 0].cpu().numpy())
            vp.append(p.cpu().numpy())
    return np.concatenate(v0), np.concatenate(vp)


def per_hand(uid: np.ndarray, x: np.ndarray, mean: bool = False) -> np.ndarray:
    """Sum (or mean) of ``x`` over the rows of each hand."""
    _, inv = np.unique(uid, return_inverse=True)
    sums = np.bincount(inv, weights=x.astype(np.float64))
    return sums / np.bincount(inv) if mean else sums


def r2(target: np.ndarray, pred: np.ndarray) -> float:
    var = float(np.var(target))
    return 1.0 - float(np.var(target - pred)) / var if var > 0 else 0.0


def measure(
    rows: dict[str, np.ndarray],
    held: np.ndarray,
    critics: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    """Every setting's advantage on the held-out rows: its std per decision (the plan's
    figure) and the std of a hand's summed advantage (the block view: the rows of one
    hand share one result, and per-decision credit telescopes to the hand's result given
    once; see the module docstring), raw points where the
    advantage is a whole hand's result, the critic's R². ``critics``: name -> (own-seat
    value, paired value) on the held-out rows."""
    uid = rows["uid"][held]
    own = rows["returns"][held][:, 0]
    paired = rows["reward"][held]
    out: dict[str, dict[str, Any]] = {}

    def row(name: str, adv: np.ndarray, *, raw: np.ndarray | None = None, **more: Any) -> None:
        hand = per_hand(uid, adv)
        d: dict[str, Any] = {
            "decisions": int(len(adv)),
            "hands": int(len(hand)),
            "adv_mean": float(adv.mean()),
            "adv_std": float(adv.std()),
            "hand_sum_std": float(hand.std()),
            **more,
        }
        if raw is not None:
            d["raw_points_std_per_hand"] = float(per_hand(uid, raw, mean=True).std())
        out[name] = d

    row("neither", own, raw=unsquash(own))
    row("same-deal baseline", paired, raw=unsquash(paired))
    for cname, (v0, vp) in critics.items():
        tag = "" if cname == "hidden" else ", no hidden view"
        row(f"oracle critic{tag}", own - v0, critic_r2=r2(own, v0))
        row(f"both{tag}", paired - vp, critic_r2=r2(paired, vp))
        nxt = next_values(uid, vp, paired)
        row(f"both, per-decision credit{tag}", nxt - vp, critic_r2=r2(paired, vp))
    return out


def throughput(
    results: dict[str, dict[str, Any]],
    timing: dict[str, dict[str, Any]],
    update_s: dict[str, float],
) -> None:
    """Adds hands, learner seat-hands and decisions per hour of training to each
    setting, with the update time, and the noise of one hour's mean advantage per
    decision in the block view: the std of a hand's summed advantage over the decisions
    a seat-hand gives and over the square root of the seat-hands an hour gives."""
    for name, r in results.items():
        base = name.split(",")[0]
        paired = SETTINGS[base]["paired"] if base in SETTINGS else True
        t = timing["paired" if paired else "unpaired"]
        key = name.replace(", no hidden view", "")
        upd = update_s.get(key, update_s.get(base, 0.0))
        s_per_hand = t["collect_s_per_hand"] + upd
        hands_h = 3600.0 / s_per_hand
        seat_h = hands_h * (1 if paired else 4)
        r.update(
            {
                "paired_rollout": paired,
                "update_s_per_hand": round(upd, 6),
                "hands_per_hour": round(hands_h),
                "seat_hands_per_hour": round(seat_h),
                "decisions_per_hour": round(seat_h * t["decisions_per_seat_hand"]),
                "hour_noise": r["hand_sum_std"]
                / (t["decisions_per_seat_hand"] * math.sqrt(seat_h)),
            }
        )


def table(results: dict[str, dict[str, Any]]) -> str:
    ref = results["same-deal baseline"]["hour_noise"]
    lines = [
        "| Setting | Advantage std per decision | Std of a hand's summed advantage "
        "| Raw points per hand | Critic R² | Hands / h | Learner seat-hands / h | Decisions / h "
        "| Noise of an hour's mean | vs Task 10's |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        raw = r.get("raw_points_std_per_hand")
        q = r.get("critic_r2")
        lines.append(
            f"| {name} | {r['adv_std']:.3f} | {r['hand_sum_std']:.3f} | "
            f"{'' if raw is None else f'{raw:.1f}'} | {'' if q is None else f'{q:.3f}'} | "
            f"{r['hands_per_hour']:,} | {r['seat_hands_per_hour']:,} | {r['decisions_per_hour']:,} "
            f"| {r['hour_noise']:.5f} | {r['hour_noise'] / ref:.2f} |"
        )
    return "\n".join(lines) + "\n"


def run(cfg: LuckConfig) -> dict[str, Any]:
    cfg.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    tr = trainer(cfg, paired=True)
    print(f"luck: {cfg.hands:,} paired hands with {cfg.init} on {tr.device}", flush=True)
    rows, t_paired = collect(tr, cfg.hands, keep=True)
    assert rows is not None
    print(
        f"luck: paired rollout {t_paired['hands']:,} hands, {t_paired['decisions']:,} decisions "
        f"in {t_paired['collect_s']:.0f} s",
        flush=True,
    )
    update_s: dict[str, float] = {}
    last = t_paired.pop("last_batch")
    n_last = len(np.unique(last["uid"]))
    for name, s in SETTINGS.items():
        if s["paired"]:
            update_s[name] = time_update(tr, last, s) / n_last
    del last

    # The critics: with and without the hidden view, fitted on the first hands.
    uids = np.unique(rows["uid"])
    cut = uids[min(len(uids) - 1, int(len(uids) * cfg.fit_share))]
    fit = rows["uid"] < cut
    held = ~fit
    device = tr.device
    critics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fit_log: dict[str, list[float]] = {}
    for cname, hidden in (("hidden", True), ("no hidden view", False)):
        critic = Critic(tr.net, hidden).to(device)
        fit_log[cname] = fit_critic(critic, rows, fit, cfg, device)
        critics[cname] = predict(critic, rows, held, device)
        print(f"luck: critic ({cname}) fitted, loss per epoch {fit_log[cname]}", flush=True)
    results = measure(rows, held, critics)
    fit_hands = int((uids < cut).sum())
    del rows, tr

    # The unpaired rollout: the learner in all four seats, no baseline copy.
    tu = trainer(cfg, paired=False)
    _, t_unpaired = collect(tu, cfg.time_hands, keep=False)
    last = t_unpaired.pop("last_batch")
    n_last = len(np.unique(last["uid"]))
    for name, s in SETTINGS.items():
        if not s["paired"]:
            update_s[name] = time_update(tu, last, s) / n_last
    del last, tu
    print(
        f"luck: unpaired rollout {t_unpaired['hands']:,} hands, {t_unpaired['decisions']:,} "
        f"decisions in {t_unpaired['collect_s']:.0f} s",
        flush=True,
    )
    timing = {"paired": t_paired, "unpaired": t_unpaired}
    throughput(results, timing, update_s)

    md = table(results)
    print(md, flush=True)
    (cfg.out / "variance.md").write_text(md, encoding="utf-8")
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
        "fit_hands": fit_hands,
        "held_out_hands": results["neither"]["hands"],
        "timing": timing,
        "update_s_per_hand": update_s,
        "critic_fit_loss": fit_log,
        "device": str(device),
    }
    (cfg.out / "variance.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (cfg.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"results": results, **summary}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m mahjong.rl.luck",
        description="Task 29 part 1: the advantage's variance under each luck-removal setting.",
    )
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hands", type=int, default=50_000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--fit-share", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=2_000)
    ap.add_argument("--time-hands", type=int, default=4_000)
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args(argv)
    no_power_throttling()
    run(
        LuckConfig(
            init=args.init,
            out=args.out,
            hands=args.hands,
            envs=args.envs,
            device=args.device,
            seed=args.seed,
            fit_share=args.fit_share,
            epochs=args.epochs,
            chunk=args.chunk,
            time_hands=args.time_hands,
            torch_threads=args.torch_threads,
        )
    )


if __name__ == "__main__":
    main()
