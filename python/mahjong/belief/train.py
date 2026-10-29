"""Train the belief model on the self-play logs (plan Task 5).

::

    python -m mahjong.belief.train --logs runs/logs/train --out models/belief.pt --epochs 5
        [--holdout runs/logs/holdout] [--discards] [--resume] [--ckpt-dir runs/belief]
        [--log runs/belief/train.log]

Each epoch visits every chunk once, in a fresh random order, loading ``--group``
chunks at a time and shuffling the rows inside the group (a chunk is 500 hands,
so a group mixes about 2,000 hands). A checkpoint (model, optimizer, schedule,
epoch) goes to ``<ckpt-dir>/ckpt-<epoch>.pt`` after every epoch; ``--resume``
continues from the newest one. After the last epoch the model is saved to
``--out`` and exported as MJNN v2 next to it (``.mjnn``). The log is one JSON
line per epoch (train loss parts, hold-out loss parts on the first hold-out
chunk, seconds).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mahjong.belief.logs import LogDir
from mahjong.belief.model import Batch, BeliefConfig, BeliefNet, batch_from_rows, losses


def load_group(log: LogDir, chunks: list[int]) -> Batch:
    """The rows of several chunks as one batch (CPU)."""
    parts = [batch_from_rows(log.chunk(c)) for c in chunks]
    if len(parts) == 1:
        return parts[0]
    return Batch(**{k: torch.cat([getattr(p, k) for p in parts]) for k in parts[0].__dict__})


def evaluate_loss(net: BeliefNet, batch: Batch, device: torch.device, bs: int) -> dict[str, float]:
    """Mean loss parts over ``batch`` (no gradients)."""
    net.eval()
    sums: dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for i in range(0, len(batch), bs):
            b = batch.index(torch.arange(i, min(i + bs, len(batch)))).to(device)
            out = net(net.inputs(b.features, b.discards))
            parts = losses(net, out, b)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v) * len(b)
            n += len(b)
    net.train()
    return {k: v / n for k, v in sums.items()}


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    for p in ckpt_dir.glob("ckpt-*.pt"):
        m = re.fullmatch(r"ckpt-(\d+)\.pt", p.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), p)
    return best[1] if best else None


def train(args: argparse.Namespace) -> BeliefNet:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    log = LogDir(args.logs)
    holdout = LogDir(args.holdout) if args.holdout and Path(args.holdout).exists() else None
    cfg = BeliefConfig(hidden=tuple(args.hidden), discards=args.discards)
    net = BeliefNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    steps_per_epoch = max(1, log.rows // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log) if args.log else ckpt_dir / "train.log"
    start_epoch = 0
    rng = np.random.default_rng(args.seed)
    if args.resume:
        latest = latest_checkpoint(ckpt_dir)
        if latest is not None:
            ck = torch.load(latest, map_location="cpu")
            if ck["cfg"] != asdict(cfg):
                raise SystemExit(f"{latest} was trained with another config: {ck['cfg']}")
            net.load_state_dict(ck["state"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_epoch = int(ck["epoch"]) + 1
            rng = np.random.default_rng([args.seed, start_epoch])
            print(f"resumed from {latest} (next epoch {start_epoch})", flush=True)

    def write_log(entry: dict[str, Any]) -> None:
        line = json.dumps(entry)
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    write_log(
        {
            "event": "start",
            "params": net.parameter_count(),
            "cfg": asdict(cfg),
            "rows": log.rows,
            "device": str(device),
            "start_epoch": start_epoch,
            "epochs": args.epochs,
        }
    )
    val = load_group(holdout, [0]) if holdout is not None else None
    started = time.time()
    n_chunks = min(len(log), args.max_chunks) if args.max_chunks else len(log)
    for epoch in range(start_epoch, args.epochs):
        order = rng.permutation(n_chunks).tolist()
        sums: dict[str, float] = {}
        seen = 0
        for g in range(0, len(order), args.group):
            group = load_group(log, order[g : g + args.group]).to(device)
            perm = torch.randperm(len(group), device=device)
            for i in range(0, len(group) - args.batch + 1, args.batch):
                b = group.index(perm[i : i + args.batch])
                out = net(net.inputs(b.features, b.discards))
                parts = losses(net, out, b)
                opt.zero_grad(set_to_none=True)
                parts["total"].backward()  # type: ignore[no-untyped-call]
                opt.step()
                sched.step()
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
                seen += 1
            del group
        entry: dict[str, Any] = {
            "event": "epoch",
            "epoch": epoch,
            "steps": seen,
            "lr": sched.get_last_lr()[0],
            "train": {k: v / max(seen, 1) for k, v in sums.items()},
            "seconds": round(time.time() - started),
        }
        if val is not None:
            entry["holdout"] = evaluate_loss(net, val, device, args.batch)
        torch.save(
            {
                "cfg": asdict(cfg),
                "state": net.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
            },
            ckpt_dir / f"ckpt-{epoch}.pt",
        )
        write_log(entry)
    net = net.cpu()
    out = Path(args.out)
    net.save(out)
    net.export(out.with_suffix(".mjnn"))
    write_log({"event": "done", "out": str(out), "mjnn": str(out.with_suffix(".mjnn"))})
    return net


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--logs", default="runs/logs/train")
    p.add_argument(
        "--holdout", default="runs/logs/holdout", help="hold-out log for the epoch report"
    )
    p.add_argument("--out", default="models/belief.pt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[384, 288])
    p.add_argument(
        "--discards",
        action="store_true",
        help="also feed the discard-order block (docs/belief.md §5)",
    )
    p.add_argument("--group", type=int, default=4, help="chunks loaded and shuffled together")
    p.add_argument(
        "--max-chunks", type=int, default=0, help="use only the first N chunks (smoke runs)"
    )
    p.add_argument("--ckpt-dir", default="runs/belief")
    p.add_argument("--log", default=None, help="JSON-lines log (default <ckpt-dir>/train.log)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
