"""Evaluate the belief model on a hold-out log against AC3 (plan Task 5).

::

    python -m mahjong.belief.evaluate --logs runs/logs/holdout --model models/belief.pt
        --out runs/belief/eval.json [--bestof runs/belief/holdout-bok]

Reports, on every row of ``--logs``:

* the per-tile count negative log-likelihood (nats; mean over rows, opponents
  and the 34 kinds) of the model and of **uniform re-dealing** (the exact
  hypergeometric law of a hand drawn uniformly from the observer's unseen tiles);
* with ``--bestof``, a log directory written by ``selfplay_log --bestof K`` (the
  same hands, plus ``bestof-NNNNN.bin`` histograms), the model's and
  **best-of-K**'s count NLL on those rows (add-``--alpha`` smoothing, else one
  unseen count would cost infinitely);
* calibration per wall-progress bucket (6 buckets, as ``danger.rs``): predicted
  mean against the observed rate, with the observed rate's 95 % interval, for
  tenpai, for **deal-in** ("the tile I discarded was won on"), and for the
  outcome head (self-draw and discard wins before the hand ends).

The discarded tile is read back from the next row of the hand (its discard
history gains one tile); a discard an opponent can win on always has a next
row, the claim window in which that opponent declares the win, so deal-ins are
covered. The one discard without a next row is a hand's last discard into a
wall-exhaustion draw, which is dropped and counted. (If a log ever ended a hand
without the claim row, the tile would be the one wait of the winner that the
observer held; rows holding two or more such waits are dropped and counted.)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mahjong.belief.logs import NUM_KINDS, LogDir
from mahjong.belief.model import MAX_COPIES, N_OPP, BeliefNet, batch_from_rows, mask_counts

PROGRESS_BUCKETS = 6
INITIAL_WALL = 91.0
Z95 = 1.959964


def buckets(wall_remaining: npt.NDArray[np.integer[Any]]) -> npt.NDArray[np.int64]:
    """The wall-progress bucket of each row (0 = early … 5 = late), as ``danger.rs``."""
    progress = np.clip(1.0 - wall_remaining.astype(np.float64) / INITIAL_WALL, 0.0, 0.999)
    return (progress * PROGRESS_BUCKETS).astype(np.int64)


_LOG_FACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, 160)))])


def _log_choose(n: npt.NDArray[Any], k: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    n = np.asarray(n, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    ok = (k >= 0) & (k <= n) & (n >= 0)
    nn, kk = np.where(ok, n, 0), np.where(ok, k, 0)
    out = _LOG_FACT[nn] - _LOG_FACT[kk] - _LOG_FACT[nn - kk]
    return np.where(ok, out, -np.inf)


def uniform_count_nll(
    unseen: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """−log P(count) per ``(row, opponent, kind)`` when each opponent's hand is a
    uniform draw of its size from the observer's unseen tiles: hypergeometric with
    population ``N = Σ unseen``, ``u_t`` copies of kind ``t`` and ``n_o`` draws."""
    pool = unseen.sum(axis=1)[:, None, None]  # (n, 1, 1)
    u = unseen[:, None, :]  # (n, 1, 34)
    size = counts.sum(axis=2)[:, :, None]  # (n, 3, 1)
    logp = _log_choose(u, counts) + _log_choose(pool - u, size - counts) - _log_choose(pool, size)
    return -logp


def recover_discards(
    rows: np.ndarray[Any, np.dtype[Any]],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], dict[str, int]]:
    """Rows where the observer discarded and the tile is known: ``(row index,
    tile)`` plus counts of what was dropped (see the module docs)."""
    n = len(rows)
    hand = rows["hand"].astype(np.int64)
    step = rows["step"].astype(np.int64)
    obs = rows["observer"].astype(np.int64)
    disc = rows["discards"].astype(np.int64)  # (n, 4, 8)
    turn = rows["phase"] == 0
    idx: list[npt.NDArray[np.int64]] = []
    tiles: list[npt.NDArray[np.int64]] = []
    # (a) A next row of the same hand: its history of the previous observer grew.
    i = np.arange(n - 1)
    follow = (hand[i] == hand[i + 1]) & (step[i] + 1 == step[i + 1]) & turn[i]
    rel = (obs[i] - obs[i + 1]) % 4
    new = disc[i + 1, rel]  # (n-1, 8)
    old = disc[i, 0]
    shifted = (new[:, 1:] == old[:, :7]).all(axis=1) & (new[:, 0] != 255)
    unchanged = (new == old).all(axis=1)
    found = follow & shifted & ~unchanged
    idx.append(i[found])
    tiles.append(new[found, 0])
    # (b) The last row of a hand, in turn phase, where an opponent won on the
    # observer's discard: the tile is the winner's wait the observer held.
    last = np.ones(n, dtype=bool)
    last[:-1] = ~((hand[:-1] == hand[1:]) & (step[:-1] + 1 == step[1:]))
    won_on_me = (rows["opp_won"] == 1) & (rows["opp_win_from"] == 0)  # (n, 3)
    cand_rows = np.flatnonzero(last & turn & won_on_me.any(axis=1))
    own = rows["features"][cand_rows, :NUM_KINDS] > 0.5  # plane 0: holds ≥ 1 copy
    waits = rows["opp_waits"][cand_rows].astype(bool)  # (m, 3, 34)
    winners_wait = (waits & won_on_me[cand_rows][:, :, None]).any(axis=1)  # (m, 34)
    cands = own & winners_wait
    unique = cands.sum(axis=1) == 1
    idx.append(cand_rows[unique])
    tiles.append(cands[unique].argmax(axis=1))
    draw_last = int((last & turn & ~won_on_me.any(axis=1) & (rows["deltas"][:, 0] <= 0)).sum())
    info = {
        "from_next_row": int(found.sum()),
        "from_final_win": int(unique.sum()),
        "dropped_ambiguous_final": int((~unique).sum()),
        "dropped_final_no_win": draw_last,
    }
    order = np.argsort(np.concatenate(idx), kind="stable")
    return np.concatenate(idx)[order], np.concatenate(tiles)[order], info


def wilson(successes: float, n: float) -> tuple[float, float]:
    """Wilson's 95 % interval for a proportion: unlike the normal interval it has
    width when no event was observed (``successes = 0``)."""
    if n <= 0:
        return 0.0, 1.0
    z2 = Z95 * Z95
    p = successes / n
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return centre - half, centre + half


class Calibration:
    """Per-bucket sums of a predicted probability and an observed 0/1 label."""

    def __init__(self) -> None:
        self.pred = np.zeros(PROGRESS_BUCKETS)
        self.obs = np.zeros(PROGRESS_BUCKETS)
        self.n = np.zeros(PROGRESS_BUCKETS)

    def add(
        self,
        bucket: npt.NDArray[np.int64],
        pred: npt.NDArray[np.floating[Any]],
        obs: npt.NDArray[np.floating[Any]],
    ) -> None:
        self.pred += np.bincount(bucket, weights=pred, minlength=PROGRESS_BUCKETS)
        self.obs += np.bincount(bucket, weights=obs, minlength=PROGRESS_BUCKETS)
        self.n += np.bincount(bucket, minlength=PROGRESS_BUCKETS)

    def table(self) -> list[dict[str, float | bool]]:
        rows = []
        for b in range(PROGRESS_BUCKETS):
            n = max(self.n[b], 1.0)
            p_obs = self.obs[b] / n
            p_pred = self.pred[b] / n
            lo, hi = wilson(self.obs[b], n)
            rows.append(
                {
                    "bucket": b,
                    "n": int(self.n[b]),
                    "observed": p_obs,
                    "predicted": p_pred,
                    "ci_low": lo,
                    "ci_high": hi,
                    "within": bool(lo <= p_pred <= hi),
                }
            )
        return rows

    def all_within(self) -> bool:
        return all(bool(r["within"]) for r in self.table() if int(r["n"]) > 0)


def _mean_nll(logp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """−log p at the target, over the last axis; ``logp`` ``(..., 5)``."""
    return -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    net = BeliefNet.load(Path(args.model)).to(device).eval()
    log = LogDir(args.logs)
    nll_model = nll_uniform = 0.0
    n_tiles = 0
    cal = {k: Calibration() for k in ["tenpai", "deal_in", "self_draw", "discard_win", "wait"]}
    dropped = {
        "from_next_row": 0,
        "from_final_win": 0,
        "dropped_ambiguous_final": 0,
        "dropped_final_no_win": 0,
    }
    with torch.no_grad():
        for rows in log.chunks():
            bucket = buckets(rows["wall_remaining"])
            b = batch_from_rows(rows)
            model_nll_rows, pred = _forward(net, b, device, args.batch)
            nll_model += float(model_nll_rows.sum())
            nll_uniform += float(uniform_count_nll(b.unseen.numpy(), b.counts.numpy()).sum())
            n_tiles += b.counts.numel()
            opp_bucket = np.repeat(bucket, N_OPP)
            cal["tenpai"].add(opp_bucket, pred["tenpai"].ravel(), b.tenpai.numpy().ravel())
            outcome = b.outcome.numpy().ravel()
            cal["self_draw"].add(opp_bucket, pred["outcome"][:, :, 2].ravel(), (outcome == 2) * 1.0)
            cal["discard_win"].add(
                opp_bucket, pred["outcome"][:, :, 1].ravel(), (outcome == 1) * 1.0
            )
            tenpai = b.tenpai.numpy().astype(bool)
            if tenpai.any():
                tb = np.repeat(np.repeat(bucket, N_OPP).reshape(-1, N_OPP)[tenpai], NUM_KINDS)
                cal["wait"].add(tb, pred["waits"][tenpai].ravel(), b.waits.numpy()[tenpai].ravel())
            idx, tile, info = recover_discards(rows)
            for k, v in info.items():
                dropped[k] += v
            p_win = pred["tenpai"][idx] * pred["waits"][idx, :, tile]  # (m, 3)
            p_deal_in = 1.0 - np.prod(1.0 - p_win, axis=1)
            observed = rows["opp_waits"][idx, :, tile].any(axis=1)
            cal["deal_in"].add(bucket[idx], p_deal_in, observed * 1.0)
    result: dict[str, Any] = {
        "model": str(args.model),
        "logs": str(args.logs),
        "rows": log.rows,
        "count_nll_per_tile": {"model": nll_model / n_tiles, "uniform": nll_uniform / n_tiles},
        "calibration": {k: v.table() for k, v in cal.items()},
        "deal_in_rows": dropped,
    }
    if args.bestof:
        result["bestof"] = _bestof(net, LogDir(args.bestof), device, args.batch, args.alpha)
    counts_ok = result["count_nll_per_tile"]["model"] < result["count_nll_per_tile"]["uniform"]
    if "bestof" in result:
        counts_ok = counts_ok and result["bestof"]["model"] < result["bestof"]["bestof"]
    result["ac3"] = {
        "count_nll_below_baselines": bool(counts_ok),
        "tenpai_calibrated": cal["tenpai"].all_within(),
        "deal_in_calibrated": cal["deal_in"].all_within(),
    }
    result["ac3"]["pass"] = all(result["ac3"].values())
    return result


def _forward(
    net: BeliefNet, b: Any, device: torch.device, bs: int
) -> tuple[npt.NDArray[np.float64], dict[str, npt.NDArray[np.float64]]]:
    """Per-row count NLL (summed over opponents and kinds) and the predictions."""
    nll: list[torch.Tensor] = []
    preds: dict[str, list[torch.Tensor]] = {"tenpai": [], "waits": [], "outcome": [], "counts": []}
    for i in range(0, len(b), bs):
        part = b.index(torch.arange(i, min(i + bs, len(b)))).to(device)
        out = net(net.inputs(part.features, part.discards))
        p = net.predict(out, part.unseen)
        logp = torch.log_softmax(mask_counts(net.split(out)["counts"], part.unseen), dim=-1)
        nll.append(_mean_nll(logp, part.counts).sum(dim=(1, 2)).cpu())
        for k in preds:
            preds[k].append(p[k].cpu())
    return (
        torch.cat(nll).double().numpy(),
        {k: torch.cat(v).double().numpy() for k, v in preds.items()},
    )


def _bestof(
    net: BeliefNet, log: LogDir, device: torch.device, bs: int, alpha: float
) -> dict[str, Any]:
    """Count NLL of the model, best-of-K and uniform on the rows of a ``--bestof`` log."""
    k = int(log.meta.get("bestof", 0))
    if k <= 0:
        raise SystemExit(f"{log.path} was not written with --bestof")
    sums = {"model": 0.0, "bestof": 0.0, "uniform": 0.0}
    n_tiles = 0
    with torch.no_grad():
        for i, rows in enumerate(log.chunks()):
            side = np.fromfile(log.path / f"bestof-{i:05}.bin", dtype=np.uint8)
            hist = side.reshape(len(rows), N_OPP, NUM_KINDS, MAX_COPIES).astype(np.float64)
            b = batch_from_rows(rows)
            model_nll, _ = _forward(net, b, device, bs)
            counts = b.counts.numpy()
            p = (hist + alpha) / (k + MAX_COPIES * alpha)
            p_true = np.take_along_axis(p, counts[..., None], axis=-1)[..., 0]
            sums["model"] += float(model_nll.sum())
            sums["bestof"] += float(-np.log(p_true).sum())
            sums["uniform"] += float(uniform_count_nll(b.unseen.numpy(), counts).sum())
            n_tiles += counts.size
    return {"k": k, "alpha": alpha, "rows": log.rows, **{k_: v / n_tiles for k_, v in sums.items()}}


def print_report(r: dict[str, Any]) -> None:
    c = r["count_nll_per_tile"]
    print(
        f"rows {r['rows']}  count NLL per tile: model {c['model']:.4f}  uniform {c['uniform']:.4f}"
    )
    if "bestof" in r:
        b = r["bestof"]
        print(
            f"best-of-{b['k']} rows {b['rows']} (alpha {b['alpha']}): model {b['model']:.4f}  "
            f"best-of-K {b['bestof']:.4f}  uniform {b['uniform']:.4f}"
        )
    for name, table in r["calibration"].items():
        print(f"\n{name}: bucket  n  observed  predicted  95% interval  within")
        for row in table:
            print(
                f"  {row['bucket']}  {row['n']:>8}  {row['observed']:.4f}  {row['predicted']:.4f}"
                f"  [{row['ci_low']:.4f}, {row['ci_high']:.4f}]  {'yes' if row['within'] else 'NO'}"
            )
    print(f"\ndeal-in rows: {r['deal_in_rows']}")
    print(f"AC3: {r['ac3']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--logs", default="runs/logs/holdout")
    p.add_argument("--model", default="models/belief.pt")
    p.add_argument("--out", default="runs/belief/eval.json")
    p.add_argument("--bestof", default=None, help="log dir written with selfplay_log --bestof K")
    p.add_argument("--alpha", type=float, default=0.5, help="add-alpha smoothing of best-of-K")
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = evaluate(args)
    print_report(result)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()
