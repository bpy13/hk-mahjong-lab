"""Distil the search labels into the suit-shared CNN (plan Task 15).

Data: the label directory of Task 13 (``runs/labels``, :mod:`mahjong.rl.labels`).
One or more whole chunks are held out (default: the last one) for the agreement
numbers; the rest is training data. Everything is loaded onto the GPU once
(features as float16, about 1.9 GB for 1 M rows), so a training step never waits
for the CPU.

Loss per row (plan Task 15):

* **policy**: cross-entropy of the legal-masked logits against the move the search
  played, weighted by the search's margin (``labels.policy_targets``: ``diff /
  diff_se`` of the chosen move over the best other candidate, clipped to [1, 3]),
  so clear decisions count more;
* **target**: cross-entropy of the target head against the labelled target plan
  (rows with no feasible plan are skipped);
* **value**: mean squared error of the four-seat value head against the final
  deltas, log-scaled ``sign(x) · ln(1 + |x| / 8)`` (idea 5d).

Every batch gets one random suit permutation per row (:func:`mahjong.rl.cnn.augment`,
done on the GPU here). AdamW with a short warm-up and a cosine decay. A checkpoint
(model, optimiser, position in the run) is written every ``checkpoint_minutes`` to
``out/ckpt.pt``; ``resume=True`` continues from it with the same batch order.

At the end: ``models/<name>.pt`` (PyTorch, Task 17's ``--init``),
``models/<name>-fp32.mjnn`` and ``models/<name>.mjnn`` (int8, what the agents load),
and ``out/result.json`` with held-out top-1 agreement (PyTorch fp32), the Rust
fp32-vs-int8 top-1 agreement on held-out positions, parameter count and file sizes.

    python -m mahjong.rl.distill_cnn --labels runs/labels --out runs/distill --minutes 100
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mahjong.rl.cnn import (
    NUM_TARGETS,
    PERMS,
    CnnConfig,
    SuitCNN,
    action_map,
    action_perm,
    export,
    feature_perm,
    target_map,
)
from mahjong.rl.labels import LabelDir, policy_targets, value_targets

NEG = -1.0e9


@dataclass
class DistillConfig:
    labels: str = "runs/labels"
    out: str = "runs/distill"
    models: str = "models"
    name: str = "cnn-distill"
    #: Chunk indices used for training (None = every chunk not held out).
    train_chunks: list[int] | None = None
    #: Held-out chunk indices (None = the last chunk).
    holdout_chunks: list[int] | None = None
    epochs: int = 6
    #: Stop training after this many minutes (the schedule is fitted to it).
    minutes: float = 100.0
    batch: int = 2048
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 300
    target_weight: float = 0.5
    value_weight: float = 0.5
    margin_clip: float = 3.0
    checkpoint_minutes: float = 15.0
    seed: int = 0
    device: str = "cuda"
    #: Positions for the Rust fp32-vs-int8 agreement check.
    int8_check_rows: int = 1000
    cnn: dict[str, Any] = field(default_factory=dict)


def log_line(out: Path, msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%dT%H:%M:%S} {msg}"
    print(line, flush=True)
    with (out / "train.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- Data -----------------------------------------------------------------------


@dataclass
class Data:
    """Label rows as GPU tensors."""

    x: torch.Tensor  # (n, 933) float16
    legal: torch.Tensor  # (n, 134) bool
    chosen: torch.Tensor  # (n,) int64
    weight: torch.Tensor  # (n,) float32
    target: torch.Tensor  # (n,) int64, 255 = none
    value: torch.Tensor  # (n, 4) float32, log-scaled

    def __len__(self) -> int:
        return int(self.chosen.shape[0])


def load(labels: LabelDir, chunks: list[int], device: str, clip: float) -> Data:
    """The rows of ``chunks`` moved to ``device`` (one chunk at a time on the CPU)."""
    parts: dict[str, list[torch.Tensor]] = {k: [] for k in Data.__dataclass_fields__}
    for i in chunks:
        rows = labels.chunk(i)
        chosen, weight = policy_targets(rows, clip=clip)
        parts["x"].append(torch.from_numpy(np.ascontiguousarray(rows["features"])).to(device))
        parts["legal"].append(torch.from_numpy(rows["legal"] > 0).to(device))
        parts["chosen"].append(torch.from_numpy(chosen).to(device))
        parts["weight"].append(torch.from_numpy(weight).to(device))
        parts["target"].append(torch.from_numpy(rows["target"].astype(np.int64)).to(device))
        parts["value"].append(torch.from_numpy(value_targets(rows)).to(device))
        del rows
    return Data(**{k: torch.cat(v) for k, v in parts.items()})


class SuitSwap:
    """:func:`mahjong.rl.cnn.augment` as GPU gathers: one random suit permutation
    per row for features, legal masks, chosen codes and target classes."""

    def __init__(self, device: str) -> None:
        def stack(f: Any) -> torch.Tensor:
            return torch.as_tensor(np.stack([f(p) for p in PERMS]), device=device)

        self.feat = stack(feature_perm)  # (6, 933): new column <- old column
        self.act = stack(action_perm)  # (6, 134): new code <- old code
        self.act_map = stack(action_map)  # (6, 134): old code -> new code
        tm = np.stack([np.append(target_map(p), np.arange(NUM_TARGETS, 256)) for p in PERMS])
        self.tgt_map = torch.as_tensor(tm, device=device)  # (6, 256), 255 stays 255

    def __call__(
        self,
        g: torch.Generator,
        x: torch.Tensor,
        legal: torch.Tensor,
        chosen: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        which = torch.randint(0, len(PERMS), (x.shape[0],), generator=g, device=x.device)
        return self.apply(which, x, legal, chosen, target)

    def apply(
        self,
        which: torch.Tensor,
        x: torch.Tensor,
        legal: torch.Tensor,
        chosen: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Row ``r`` gets permutation ``PERMS[which[r]]``."""
        x = x.gather(1, self.feat[which])
        legal = legal.gather(1, self.act[which])
        chosen = self.act_map[which, chosen]
        target = self.tgt_map[which, target]
        return x, legal, chosen, target


# --- Training ---------------------------------------------------------------------


def losses(
    net: SuitCNN,
    x: torch.Tensor,
    legal: torch.Tensor,
    chosen: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Policy, target and value losses, and the masked policy logits."""
    p, t, v = net(x.float())
    logits = p.masked_fill(~legal, NEG)
    ce = F.cross_entropy(logits, chosen, reduction="none")
    loss_p = (ce * weight).sum() / weight.sum()
    has = target < NUM_TARGETS
    loss_t = F.cross_entropy(t[has], target[has]) if bool(has.any()) else p.new_zeros(())
    loss_v = F.mse_loss(v, value)
    return loss_p, loss_t, loss_v, logits


@torch.no_grad()
def evaluate(net: SuitCNN, data: Data, batch: int = 8192) -> dict[str, float]:
    """Held-out top-1 agreement with the search's move, target accuracy, value MSE."""
    net.eval()
    agree = n = t_ok = t_n = 0
    v_se = 0.0
    for i in range(0, len(data), batch):
        sl = slice(i, i + batch)
        p, t, v = net(data.x[sl].float())
        logits = p.masked_fill(~data.legal[sl], NEG)
        agree += int((logits.argmax(1) == data.chosen[sl]).sum())
        n += logits.shape[0]
        has = data.target[sl] < NUM_TARGETS
        t_ok += int((t[has].argmax(1) == data.target[sl][has]).sum())
        t_n += int(has.sum())
        v_se += float(((v - data.value[sl]) ** 2).sum())
    net.train()
    return {
        "top1_agreement": agree / max(1, n),
        "target_accuracy": t_ok / max(1, t_n),
        "value_mse": v_se / max(1, 4 * n),
        "rows": float(n),
    }


def lr_at(step: int, total: int, cfg: DistillConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    frac = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * frac)))


def int8_agreement(fp32: Path, int8: Path, data: Data, rows: int) -> float:
    """Top-1 agreement (legal moves only) of the Rust forward pass of the int8 file
    with the fp32 file on the first ``rows`` held-out positions."""
    from mahjong._core import net_forward

    n = min(rows, len(data))
    x = data.x[:n].float().cpu().numpy().astype("<f4")
    legal = data.legal[:n].cpu().numpy()

    def top1(path: Path) -> np.ndarray:
        out, (_, n_out) = net_forward(str(path), x.tobytes())
        y = np.frombuffer(out, dtype="<f4").reshape(n, n_out)[:, : legal.shape[1]]
        return np.where(legal, y, NEG).argmax(1)

    return float((top1(fp32) == top1(int8)).mean())


def distill(cfg: DistillConfig, resume: bool = False) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    labels = LabelDir(cfg.labels)
    n_chunks = len(labels)
    holdout = cfg.holdout_chunks if cfg.holdout_chunks is not None else [n_chunks - 1]
    train = cfg.train_chunks or [i for i in range(n_chunks) if i not in holdout]
    if set(train) & set(holdout):
        raise ValueError("a chunk is both training and held-out data")
    log_line(out, f"start: {len(train)} train chunks, held out {holdout}, device {device}")

    t0 = time.time()
    data = load(labels, train, device, cfg.margin_clip)
    held = load(labels, holdout, device, cfg.margin_clip)
    log_line(
        out, f"loaded {len(data)} train rows, {len(held)} held-out rows in {time.time() - t0:.0f} s"
    )

    torch.manual_seed(cfg.seed)
    net = SuitCNN(CnnConfig(**cfg.cnn)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    swap = SuitSwap(device)
    steps_per_epoch = max(1, len(data) // cfg.batch)
    total = steps_per_epoch * cfg.epochs
    state: dict[str, Any] = {"epoch": 0, "step": 0, "global_step": 0, "seconds": 0.0}
    history: list[dict[str, Any]] = []
    ckpt = out / "ckpt.pt"
    if resume and ckpt.exists():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        state = saved["state"]
        history = saved["history"]
        total = saved["total"]
        log_line(
            out,
            f"resumed at epoch {state['epoch']} step {state['step']} "
            f"({state['global_step']} steps)",
        )

    def save() -> None:
        tmp = ckpt.with_suffix(".tmp")
        torch.save(
            {
                "model": net.state_dict(),
                "opt": opt.state_dict(),
                "state": state,
                "history": history,
                "total": total,
                "config": asdict(cfg),
            },
            tmp,
        )
        tmp.replace(ckpt)

    start = time.time() - state["seconds"]
    last_ckpt = time.time()
    budget = cfg.minutes * 60.0
    timed_out = False
    while state["epoch"] < cfg.epochs and not timed_out:
        g = torch.Generator(device=device)
        g.manual_seed(cfg.seed * 1000 + state["epoch"])
        order = torch.randperm(len(data), generator=g, device=device)
        sums = np.zeros(4)
        count = 0
        for s in range(state["step"], steps_per_epoch):
            idx = order[s * cfg.batch : (s + 1) * cfg.batch]
            x, legal, chosen, target = swap(
                g, data.x[idx], data.legal[idx], data.chosen[idx], data.target[idx]
            )
            for group in opt.param_groups:
                group["lr"] = lr_at(state["global_step"], total, cfg)
            lp, lt, lv, logits = losses(
                net, x, legal, chosen, data.weight[idx], target, data.value[idx]
            )
            loss = lp + cfg.target_weight * lt + cfg.value_weight * lv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            state["step"] = s + 1
            state["global_step"] += 1
            if s % 50 == 0:
                acc = float((logits.argmax(1) == chosen).float().mean())
                sums += [lp.item(), lt.item(), lv.item(), acc]
                count += 1
            if s % 500 == 0:
                log_line(
                    out,
                    f"epoch {state['epoch']} step {s}/{steps_per_epoch} loss p {lp.item():.4f} "
                    f"t {lt.item():.4f} v {lv.item():.4f} lr {opt.param_groups[0]['lr']:.2e}",
                )
            state["seconds"] = time.time() - start
            if time.time() - last_ckpt > cfg.checkpoint_minutes * 60:
                save()
                last_ckpt = time.time()
            if state["seconds"] > budget:
                timed_out = True
                break
        ev = evaluate(net, held)
        entry = {
            "epoch": state["epoch"],
            "steps": state["step"],
            "seconds": round(state["seconds"], 1),
            "train_policy_loss": sums[0] / max(1, count),
            "train_target_loss": sums[1] / max(1, count),
            "train_value_loss": sums[2] / max(1, count),
            "train_top1": sums[3] / max(1, count),
            **{f"held_{k}": v for k, v in ev.items()},
        }
        history.append(entry)
        log_line(out, f"epoch {state['epoch']} done: {json.dumps(entry)}")
        if not timed_out:
            state["epoch"] += 1
            state["step"] = 0
        save()
        last_ckpt = time.time()
    if timed_out:
        log_line(
            out,
            f"time budget of {cfg.minutes} min reached at epoch {state['epoch']} "
            f"step {state['step']}",
        )

    models = Path(cfg.models)
    pt = models / f"{cfg.name}.pt"
    fp32 = models / f"{cfg.name}-fp32.mjnn"
    int8 = models / f"{cfg.name}.mjnn"
    net = net.cpu().eval()
    net.save(pt)
    size32 = export(net, fp32, int8=False)
    size8 = export(net, int8, int8=True)
    final = evaluate(net.to(device), held)
    agree8 = int8_agreement(fp32, int8, held, cfg.int8_check_rows)
    result = {
        "config": asdict(cfg),
        "train_rows": len(data),
        "held_out_rows": len(held),
        "teacher": labels.teacher,
        "parameters": net.parameters_at_play(),
        "fp32_bytes": size32,
        "int8_bytes": size8,
        "held_out": final,
        "int8_top1_agreement_with_fp32": agree8,
        "epochs_done": state["epoch"],
        "timed_out": timed_out,
        "train_seconds": round(state["seconds"], 1),
        "history": history,
        "models": {"pt": str(pt), "fp32": str(fp32), "int8": str(int8)},
    }
    (out / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    log_line(
        out,
        f"DONE: held-out top-1 {final['top1_agreement']:.3f}, "
        f"target acc {final['target_accuracy']:.3f}, "
        f"value mse {final['value_mse']:.3f}, int8 vs fp32 top-1 {agree8:.4f}, "
        f"{result['parameters']:,} parameters, int8 {size8 / 1e6:.2f} MB",
    )
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Distil search labels into the suit-shared CNN.")
    ap.add_argument("--labels", default="runs/labels")
    ap.add_argument("--out", default="runs/distill")
    ap.add_argument("--models", default="models")
    ap.add_argument("--name", default="cnn-distill")
    ap.add_argument("--train-chunks", type=int, nargs="*")
    ap.add_argument("--holdout-chunks", type=int, nargs="*")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=100.0)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    cfg = DistillConfig(
        labels=a.labels,
        out=a.out,
        models=a.models,
        name=a.name,
        train_chunks=a.train_chunks,
        holdout_chunks=a.holdout_chunks,
        epochs=a.epochs,
        minutes=a.minutes,
        batch=a.batch,
        lr=a.lr,
        device=a.device,
        seed=a.seed,
    )
    distill(cfg, resume=a.resume)


if __name__ == "__main__":
    main()
