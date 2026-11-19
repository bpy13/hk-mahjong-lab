"""One table over several fits of the risk network on the same held-out deals (plan
2026-11-16 Task 16), beside the table, the exact figure and the belief model.

::

    python -m mahjong.risk.summary --data runs/risk/data-300k --chunks 290-299 \\
        --model "network, 30,000 deals=runs/risk/300k/fit-30k" \\
        --model "network, 290,000 deals=runs/risk/300k/fit-290k" --out runs/risk/300k/summary.md

Each ``--model`` is ``label=fit dir`` (``best.pt`` and ``train.log`` inside). Rows are the
sources, columns the held-out log loss and Brier score, the ten-bin verdict of
``check.py`` (every bin with 500 or more pairs within 0.8 to 1.25 of observed, with the
count of bins within), how often the source says above 50 % and below 1 % and the observed
rate in each. Populations ``all`` (every kind with an unseen copy) and ``hand`` (the kinds
in the thrower's hand), as ``check.py``. Chunks stream one at a time, so the memory is one
chunk's, whatever the held-out size.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mahjong.risk.check import EDGES, EPS, bins_of, combine
from mahjong.risk.data import RiskData, alive, inputs, targets
from mahjong.risk.model import RiskNet

F = npt.NDArray[np.float64]
N_BINS = len(EDGES) + 1


@dataclass
class Tally:
    """Running sums for one source on one population."""

    n: int = 0
    obs: float = 0.0
    pred: float = 0.0
    log_loss: float = 0.0
    brier: float = 0.0
    bin_n: npt.NDArray[np.int64] = field(default_factory=lambda: np.zeros(N_BINS, np.int64))
    bin_pred: F = field(default_factory=lambda: np.zeros(N_BINS))
    bin_obs: F = field(default_factory=lambda: np.zeros(N_BINS))
    high_n: int = 0
    high_obs: float = 0.0
    low_n: int = 0
    low_obs: float = 0.0

    def add(self, p: F, y: F) -> None:
        p = np.clip(p, EPS, 1 - EPS)
        self.n += len(p)
        self.obs += float(y.sum())
        self.pred += float(p.sum())
        self.log_loss += float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())
        self.brier += float(((p - y) ** 2).sum())
        b = bins_of(p)
        self.bin_n += np.bincount(b, minlength=N_BINS)
        self.bin_pred += np.bincount(b, weights=p, minlength=N_BINS)
        self.bin_obs += np.bincount(b, weights=y, minlength=N_BINS)
        high = p > 0.5
        self.high_n += int(high.sum())
        self.high_obs += float(y[high].sum())
        low = p < 0.01
        self.low_n += int(low.sum())
        self.low_obs += float(y[low].sum())

    def verdict(self) -> tuple[bool, int, int]:
        """(every bin of 500+ within 0.8–1.25, bins within, bins counted)."""
        big = self.bin_n >= 500
        ratio = np.divide(self.bin_obs, self.bin_pred, out=np.zeros(N_BINS), where=big)
        within = big & (ratio >= 0.8) & (ratio <= 1.25)
        return bool(within.sum() == big.sum()), int(within.sum()), int(big.sum())

    def row(self, name: str) -> str:
        ok, within, counted = self.verdict()
        share_high = 100 * self.high_n / max(self.n, 1)
        share_low = 100 * self.low_n / max(self.n, 1)
        obs_high = 100 * self.high_obs / self.high_n if self.high_n else float("nan")
        obs_low = 100 * self.low_obs / self.low_n if self.low_n else float("nan")
        return (
            f"| {name} | {self.log_loss / max(self.n, 1):.5f} | {self.brier / max(self.n, 1):.6f} "
            f"| {'yes' if ok else 'no'} ({within} of {counted}) "
            f"| {self.high_n:,} / {share_high:.2f} % / {obs_high:.1f} % "
            f"| {self.low_n:,} / {share_low:.2f} % / {obs_low:.2f} % |"
        )


HEADER = (
    "| source | log loss | Brier | ten-bin verdict (bins within 0.8–1.25) "
    "| above 50 %: n / share / observed | below 1 %: n / share / observed |\n"
    "|---|---|---|---|---|---|"
)


def fit_facts(fit_dir: Path) -> dict[str, Any]:
    """What ``train.log`` says: rows fitted, epochs run, the best epoch, minutes, parameters."""
    facts: dict[str, Any] = {}
    log = fit_dir / "train.log"
    if not log.exists():
        return facts
    epochs = 0
    best_epoch = None
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") == "start":
            facts["params"] = e.get("params")
            facts["train_rows"] = e.get("train_rows")
            facts["train_chunks"] = len(e.get("train_chunks", []))
            facts["hidden"] = e.get("cfg", {}).get("hidden")
        elif e.get("event") == "epoch":
            epochs = max(epochs, int(e["epoch"]) + 1)
            if e.get("best"):
                best_epoch = int(e["epoch"])
            facts["minutes"] = float(e.get("minutes", 0.0))
        elif e.get("event") == "stopped":
            facts["stopped"] = e.get("reason")
        elif e.get("event") == "done":
            facts["minutes"] = float(e.get("minutes", facts.get("minutes", 0.0)))
    facts["epochs"] = epochs
    facts["best_epoch"] = best_epoch
    return facts


def parse_chunks(spec: list[str]) -> list[int]:
    out: list[int] = []
    for s in spec:
        if "-" in s:
            lo, hi = (int(v) for v in s.split("-"))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(s))
    return out


def run(args: argparse.Namespace) -> str:
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    data = RiskData(args.data)
    chunks = parse_chunks(args.chunks)
    models: list[tuple[str, Path, RiskNet]] = []
    for spec in args.model:
        label, _, path = spec.partition("=")
        fit_dir = Path(path)
        net = RiskNet.load(fit_dir / "best.pt").to(device)
        net.eval()
        models.append((label, fit_dir, net))
    sources = ["table", "exact", "belief"] + [m[0] for m in models]
    tallies = {pop: {s: Tally() for s in sources} for pop in ("all", "hand")}
    n_rows = 0
    deals: set[int] = set()
    for c in chunks:
        rows = np.asarray(data.chunk(c))
        n_rows += len(rows)
        deals.update(int(d) for d in np.unique(rows["deal"]))
        al = alive(rows)
        hand = al & (np.asarray(rows["hand"]) > 0)
        win, _ = targets(rows)
        y_tile = win.max(axis=1).astype(np.float64)
        tile: dict[str, F] = {
            "table": np.asarray(rows["table_tile"], dtype=np.float64),
            "exact": np.asarray(rows["exact_tile"], dtype=np.float64),
            "belief": np.asarray(rows["belief_tile"], dtype=np.float64),
        }
        if models:
            x = torch.from_numpy(inputs(rows))
            al_t = torch.from_numpy(al)
            for label, _, net in models:
                parts = []
                with torch.no_grad():
                    for i in range(0, len(x), 4096):
                        p, _ = net.predict(
                            x[i : i + 4096].to(device), al_t[i : i + 4096].to(device)
                        )
                        parts.append(p.cpu().numpy().astype(np.float64))
                tile[label] = combine(np.concatenate(parts))
        for pop, mask in (("all", al), ("hand", hand)):
            y = y_tile[mask]
            for s, v in tile.items():
                tallies[pop][s].add(v[mask], y)

    md: list[str] = []
    md.append(
        f"Held-out chunks {chunks[0]}–{chunks[-1]} of `{data.path}`: {len(deals):,} deals, "
        f"{n_rows:,} discard decisions, {tallies['all']['table'].n:,} (decision, alive kind) "
        f"pairs, {tallies['hand']['table'].n:,} in hand."
    )
    for pop in ("all", "hand"):
        t = tallies[pop]["table"]
        rate = 100 * t.obs / max(t.n, 1)
        md.append(f"\n#### Population `{pop}` ({t.n:,} pairs, observed {rate:.3f} %)\n")
        md.append(HEADER)
        for s in sources:
            md.append(tallies[pop][s].row(s))
    if models:
        md.append("\n#### The fits\n")
        md.append(
            "| fit | hidden | parameters | deals | rows | epochs run | best epoch "
            "| minutes | stopped by |"
        )
        md.append("|---|---|---|---|---|---|---|---|---|")
        per_chunk = int(data.meta.get("chunk_deals", 1000))
        for label, fit_dir, net in models:
            f = fit_facts(fit_dir)
            md.append(
                f"| {label} | {f.get('hidden')} | {net.parameter_count():,} "
                f"| {per_chunk * f.get('train_chunks', 0):,} | {f.get('train_rows') or 0:,} "
                f"| {f.get('epochs')} | {f.get('best_epoch')} | {f.get('minutes', 0.0):.1f} "
                f"| {f.get('stopped', 'the epoch count')} |"
            )
    text = "\n".join(md) + "\n"
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True)
    p.add_argument("--chunks", nargs="+", default=["290-299"], help="numbers or ranges lo-hi")
    p.add_argument("--model", action="append", default=[], help="label=fit dir (repeatable)")
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    print(run(parse_args()))
