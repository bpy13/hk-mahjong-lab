"""PPO self-play with Suphx-style oracle guiding.

The policy head plays all four seats by sampling. Early in training the network
also sees the oracle features (opponents' hands, the next wall tiles); each
decision keeps them with probability ``keep``, which falls linearly from 1 to 0
over the first ``oracle_fraction`` of training, so the network first learns with
perfect information and then learns to do without it (Li et al., *Suphx*, 2020).

Rewards come only at the end of a hand, so advantages are Monte Carlo:
(final points − value estimate), with no discounting.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mahjong._core import HkVecEnv
from mahjong.rl.evaluate import evaluate, net_spec
from mahjong.rl.model import Net
from mahjong.rl.selfplay import Chooser, collect


def sampler(net: Net, keep: float, rng: np.random.Generator) -> Chooser:
    def choose(
        f: np.ndarray, o: np.ndarray, m: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[float, ...]]]:
        kept = (rng.random(len(f)) < keep).astype(np.float32)
        x = torch.from_numpy(np.concatenate([f, o * kept[:, None]], axis=1))
        with torch.no_grad():
            _, pi, v = net(x)
        logits = pi.masked_fill(torch.from_numpy(m) == 0, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        extra: list[tuple[float, ...]] = [
            (float(lp), float(val), float(k)) for lp, val, k in zip(logp, v, kept, strict=True)
        ]
        return np.asarray(a.numpy()), extra

    return choose


def ppo(
    out: Path,
    init: Path | None = None,
    minutes: float = 120.0,
    envs: int = 256,
    hands_per_iter: int = 512,
    epochs: int = 3,
    clip: float = 0.2,
    lr: float = 1e-4,
    batch: int = 2048,
    entropy: float = 0.005,
    oracle_fraction: float = 0.5,
    eval_every: int = 25,
    eval_deals: int = 60,
    seed: int = 0,
    log: Path | None = None,
) -> Net:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net.load(init) if init else Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    env = HkVecEnv(envs, seed + 7)
    pending: dict[int, list[Any]] = defaultdict(list)
    history: list[dict[str, Any]] = []
    start = time.time()
    it = 0
    hands = 0
    while (elapsed := time.time() - start) < minutes * 60:
        keep = (
            max(0.0, 1.0 - elapsed / (oracle_fraction * minutes * 60))
            if oracle_fraction > 0
            else 0.0
        )
        net.eval()
        d = collect(env, sampler(net, keep, rng), hands_per_iter, pending).arrays()
        hands += hands_per_iter
        old_logp = d["extra"][:, 0]
        adv = d["returns"] - d["extra"][:, 1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        kept = d["extra"][:, 2]
        n = len(d["actions"])
        net.train()
        stats = []
        for _ in range(epochs):
            perm = rng.permutation(n)
            for i in range(0, n, batch):
                idx = perm[i : i + batch]
                f = d["feats"][idx].astype(np.float32)
                o = d["oracle"][idx].astype(np.float32) * kept[idx, None]
                x = torch.from_numpy(np.concatenate([f, o], axis=1))
                mask = torch.from_numpy(d["masks"][idx])
                a = torch.from_numpy(d["actions"][idx])
                _, pi, v = net(x)
                dist = torch.distributions.Categorical(logits=pi.masked_fill(mask == 0, -1e9))
                logp = dist.log_prob(a)
                ratio = torch.exp(logp - torch.from_numpy(old_logp[idx]))
                ad = torch.from_numpy(adv[idx])
                loss_pi = -torch.min(ratio * ad, ratio.clamp(1 - clip, 1 + clip) * ad).mean()
                loss_v = F.mse_loss(v, torch.from_numpy(d["returns"][idx]))
                ent = dist.entropy().mean()
                loss = loss_pi + 0.5 * loss_v - entropy * ent
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                stats.append((loss_pi.item(), loss_v.item(), ent.item()))
        it += 1
        s = np.mean(stats, axis=0)
        entry: dict[str, Any] = {
            "iter": it,
            "hands": hands,
            "keep_oracle": keep,
            "policy_loss": float(s[0]),
            "value_loss": float(s[1]),
            "entropy": float(s[2]),
            "minutes": elapsed / 60,
        }
        if it % eval_every == 0:
            net.export(out.with_suffix(".mjnn"), "pi")
            entry["eval"] = evaluate(net_spec(out.with_suffix(".mjnn")), eval_deals)
            net.save(out)
        history.append(entry)
        print("ppo", entry, flush=True)
        if log:
            log.write_text(json.dumps(history, indent=2))
    net.save(out)
    net.export(out.with_suffix(".mjnn"), "pi")
    return net
