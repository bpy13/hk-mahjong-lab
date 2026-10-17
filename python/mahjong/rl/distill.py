"""Warm start: imitate a teacher bot (no human game logs exist for Hong Kong rules).

Teacher hands are generated in Rust in chunks (to bound memory) and the network is
trained on each chunk once: the policy head by cross-entropy on the teacher's
action; the Q head by mean squared error on the taken action plus a large-margin
loss (Hester et al., *DQfD*, 2018) that ranks the teacher's action above every
other legal one by ``margin``, so that Deep Monte Carlo starts from a Q head that
already plays like the teacher; and the value head by mean squared error to the
acting player's final points.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mahjong._core import teacher_data
from mahjong.rl.model import POINT_SCALE, Net, inputs, masked, to_arrays


def distill(
    out: Path,
    hands: int = 48_000,
    chunk: int = 600,
    teacher: str = "value",
    lr: float = 1e-3,
    batch: int = 512,
    margin: float = 0.1,
    seed: int = 0,
    log: Path | None = None,
) -> Net:
    torch.manual_seed(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    start = time.time()
    history = []
    for c in range(hands // chunk):
        f, o, m, acts, rets, _ = teacher_data(chunk, teacher, seed * 1_000_000 + c)
        f, o, m = to_arrays(f, o, m)
        a = torch.tensor(acts, dtype=torch.long)
        g = torch.tensor(rets, dtype=torch.float32) / POINT_SCALE
        mask = torch.from_numpy(m.copy())
        x = inputs(f, o, 0.0)
        perm = torch.randperm(len(a))
        losses = []
        agree = agree_q = 0
        for i in range(0, len(a), batch):
            idx = perm[i : i + batch]
            q, pi, v = net(x[idx])
            logits = masked(pi, mask[idx])
            loss_pi = F.cross_entropy(logits, a[idx])
            q_taken = q.gather(1, a[idx, None]).squeeze(1)
            loss_q = F.mse_loss(q_taken, g[idx])
            bonus = torch.full_like(q, margin).scatter(1, a[idx, None], 0.0)
            loss_margin = (masked(q + bonus, mask[idx]).max(1).values - q_taken).mean()
            loss_v = F.mse_loss(v, g[idx])
            loss = loss_pi + loss_q + loss_margin + 0.5 * loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss_pi.item())
            agree += int((logits.argmax(1) == a[idx]).sum())
            agree_q += int((masked(q, mask[idx]).argmax(1) == a[idx]).sum())
        entry = {
            "chunk": c,
            "samples": len(a),
            "policy_loss": float(np.mean(losses)),
            "train_agreement": agree / len(a),
            "q_agreement": agree_q / len(a),
            "seconds": time.time() - start,
        }
        history.append(entry)
        print("distill", entry, flush=True)
    net.save(out)
    if log:
        log.write_text(json.dumps(history, indent=2))
    return net
