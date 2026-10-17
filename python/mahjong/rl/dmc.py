"""Deep Monte Carlo (DouZero-style) self-play.

The network's Q head plays all four seats ε-greedily. Every decision of a
finished hand is labelled with the acting seat's final points (a Monte Carlo
return: no bootstrapping), and Q(state, action taken) is fitted to it by mean
squared error over a small replay window of recent iterations.
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
from mahjong.rl.selfplay import collect, greedy_q


def dmc(
    out: Path,
    init: Path | None = None,
    minutes: float = 120.0,
    envs: int = 256,
    hands_per_iter: int = 512,
    replay: int = 3,
    eps: float = 0.05,
    lr: float = 3e-4,
    batch: int = 1024,
    eval_every: int = 25,
    eval_deals: int = 60,
    seed: int = 0,
    log: Path | None = None,
) -> Net:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net.load(init) if init else Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    env = HkVecEnv(envs, seed)
    pending: dict[int, list] = defaultdict(list)
    window: list[dict[str, np.ndarray]] = []
    history: list[dict[str, Any]] = []
    start = time.time()
    it = 0
    hands = 0
    while time.time() - start < minutes * 60:
        net.eval()
        b = collect(env, greedy_q(net, eps, rng), hands_per_iter, pending)
        hands += b.hands
        window = [*window[-(replay - 1) :], b.arrays()] if replay > 1 else [b.arrays()]
        data = {
            k: np.concatenate([w[k] for w in window])
            for k in ("feats", "oracle", "actions", "returns")
        }
        net.train()
        n = len(data["actions"])
        perm = rng.permutation(n)
        losses = []
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            f = data["feats"][idx].astype(np.float32)
            x = torch.from_numpy(
                np.concatenate([f, np.zeros((len(idx), data["oracle"].shape[1]), np.float32)], 1)
            )
            a = torch.from_numpy(data["actions"][idx])
            g = torch.from_numpy(data["returns"][idx])
            q, _, _ = net(x)
            loss = F.mse_loss(q.gather(1, a[:, None]).squeeze(1), g)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        it += 1
        entry: dict[str, Any] = {
            "iter": it,
            "hands": hands,
            "loss": float(np.mean(losses)),
            "minutes": (time.time() - start) / 60,
        }
        if it % eval_every == 0:
            net.export(out.with_suffix(".mjnn"), "q")
            entry["eval"] = evaluate(net_spec(out.with_suffix(".mjnn")), eval_deals)
            net.save(out)
        history.append(entry)
        print("dmc", entry, flush=True)
        if log:
            log.write_text(json.dumps(history, indent=2))
    net.save(out)
    net.export(out.with_suffix(".mjnn"), "q")
    return net
