"""Fits the risk network on ``risk_log`` rows (plan 2026-11-16 Task 15).

::

    python -m mahjong.risk.fit --data runs/risk/data --out runs/risk/fit \
        --holdout-chunks 18 19 --epochs 8 --minutes 90 --threads 8 --device cpu

Chunks are visited one at a time in a fresh random order each epoch, rows shuffled
inside the chunk. After every epoch the held-out loss (the first ``--holdout-rows`` of
the held-out chunks) goes to ``train.log`` (one JSON line per epoch), the weights to
``ckpt.pt``; the best held-out epoch is kept as ``best.pt`` and exported next to it as
``risk.mjnn`` (and copied to ``--models/risk.mjnn`` when given). ``--resume`` continues
from ``ckpt.pt``.

Task 16: ``--train-chunks 0-289`` fits on a range of chunks (default: every chunk not held
out), ``--select win`` picks and keeps the best epoch by the held-out ``win`` loss alone
(default ``total``: ``win`` + 0.25 × ``pay``), and ``--patience N`` stops once ``N``
epochs in a row have not improved on the best (0: never; the cosine schedule still
spans ``--epochs``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mahjong.risk.data import RiskData, inputs, targets
from mahjong.risk.model import RiskConfig, RiskNet, alive_from_inputs, losses


def to_tensors(rows: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inputs as float16 (every value is exact in it), targets as float32."""
    x = torch.from_numpy(inputs(rows)).to(torch.float16)
    win, points = targets(rows)
    return x, torch.from_numpy(win), torch.from_numpy(points)


def evaluate(
    net: RiskNet,
    data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    bs: int,
    device: torch.device,
) -> dict[str, float]:
    net.eval()
    x, win, points = data
    sums = {"win": 0.0, "pay": 0.0, "total": 0.0}
    n = 0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x[i : i + bs].to(device).float()
            out = net(xb)
            loss = losses(
                out,
                win[i : i + bs].to(device),
                points[i : i + bs].to(device),
                alive_from_inputs(xb),
            )
            k = len(xb)
            sums["win"] += float(loss.win) * k
            sums["pay"] += float(loss.pay) * k
            sums["total"] += float(loss.total) * k
            n += k
    net.train()
    return {k: v / max(n, 1) for k, v in sums.items()}


def train(args: argparse.Namespace) -> Path:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    data = RiskData(args.data)
    holdout = [int(c) for c in args.holdout_chunks]
    train_chunks = [c for c in data.chunks if c not in holdout]
    if args.train_chunks:
        lo, hi = (int(v) for v in args.train_chunks.split("-"))
        wanted = set(range(lo, hi + 1))
        train_chunks = [c for c in train_chunks if c in wanted]
        if len(train_chunks) != len(wanted):
            raise SystemExit(f"--train-chunks {args.train_chunks}: not every chunk is present")
    if not train_chunks or not holdout:
        raise SystemExit(f"chunks {data.chunks}: need training and held-out chunks")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.log"

    def log(entry: dict[str, Any]) -> None:
        entry["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    held = to_tensors(data.load(holdout)[: args.holdout_rows])
    cfg = RiskConfig(hidden=tuple(args.hidden))
    net = RiskNet(cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * sum(data.rows_in(c) for c in train_chunks) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
    start_epoch = 0
    best = float("inf")
    since_best = 0
    ckpt = out / "ckpt.pt"
    if args.resume and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state["best"])
        since_best = int(state.get("since_best", 0))
        if state.get("stopped"):
            log({"event": "resumed", "note": "the fit had already stopped; nothing to do"})
            return out / "risk.mjnn"
    log(
        {
            "event": "start",
            "params": net.parameter_count(),
            "train_chunks": train_chunks,
            "holdout": holdout,
            "holdout_rows": len(held[0]),
            "cfg": cfg.to_dict(),
            "batch": args.batch,
            "lr": args.lr,
            "epochs": args.epochs,
            "threads": args.threads,
            "train_rows": sum(data.rows_in(c) for c in train_chunks),
            "select": args.select,
            "patience": args.patience,
            "resumed_at": start_epoch,
        }
    )
    t_start = time.time()
    stop = False
    for epoch in range(start_epoch, args.epochs):
        order = rng.permutation(train_chunks)
        sums = {"win": 0.0, "pay": 0.0}
        n_rows = 0
        t_epoch = time.time()
        for c in order:
            x, win, points = to_tensors(data.chunk(int(c)))
            perm = torch.from_numpy(rng.permutation(len(x)))
            for i in range(0, len(x), args.batch):
                idx = perm[i : i + args.batch]
                xb = x[idx].to(device).float()
                out_b = net(xb)
                loss = losses(
                    out_b, win[idx].to(device), points[idx].to(device), alive_from_inputs(xb)
                )
                opt.zero_grad(set_to_none=True)
                loss.total.backward()  # type: ignore[no-untyped-call]
                opt.step()
                sched.step()
                sums["win"] += float(loss.win.detach()) * len(idx)
                sums["pay"] += float(loss.pay.detach()) * len(idx)
                n_rows += len(idx)
            if (time.time() - t_start) / 60.0 > args.minutes:
                stop = True
                break
        ev = evaluate(net, held, 4096, device)
        entry = {
            "event": "epoch",
            "epoch": epoch,
            "rows": n_rows,
            "train_win": sums["win"] / max(n_rows, 1),
            "train_pay": sums["pay"] / max(n_rows, 1),
            "holdout_win": ev["win"],
            "holdout_pay": ev["pay"],
            "holdout_total": ev["total"],
            "epoch_minutes": (time.time() - t_epoch) / 60.0,
            "minutes": (time.time() - t_start) / 60.0,
            "lr": sched.get_last_lr()[0],
        }
        if ev[args.select] < best:
            best = ev[args.select]
            since_best = 0
            net.save(out / "best.pt")
            net.export(out / "risk.mjnn")
            entry["best"] = True
        else:
            since_best += 1
        reason = None
        if stop:
            reason = f"time limit {args.minutes} min"
        elif args.patience and since_best >= args.patience:
            reason = f"held-out {args.select} loss not improved in {args.patience} epochs"
        torch.save(
            {
                "net": net.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch,
                "best": best,
                "since_best": since_best,
                "stopped": reason is not None or epoch + 1 == args.epochs,
            },
            ckpt,
        )
        log(entry)
        if reason:
            log({"event": "stopped", "reason": reason})
            break
    if args.models:
        models = Path(args.models)
        models.mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "risk.mjnn", models / "risk.mjnn")
        shutil.copy(out / "best.pt", models / "risk.pt")
    log(
        {
            "event": "done",
            f"best_holdout_{args.select}": best,
            "minutes": (time.time() - t_start) / 60.0,
            "mjnn": str(out / "risk.mjnn"),
        }
    )
    return out / "risk.mjnn"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--models", default=None, help="copy risk.mjnn and risk.pt here at the end")
    p.add_argument("--holdout-chunks", nargs="+", default=["18", "19"])
    p.add_argument("--holdout-rows", type=int, default=40000)
    p.add_argument("--hidden", nargs="+", type=int, default=[768, 384])
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--minutes", type=float, default=90.0)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=15)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--train-chunks", default=None, help="a range 'lo-hi' of chunks to fit on")
    p.add_argument("--select", choices=["total", "win"], default="total")
    p.add_argument("--patience", type=int, default=0)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
