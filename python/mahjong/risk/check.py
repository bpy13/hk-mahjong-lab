"""Checks (a) to (d) of plan 2026-11-16 Task 15 on held-out deals, for the table
(Task 11's calibration), exact counting (Task 13), the belief model and the network,
side by side; written as Markdown.

::

    python -m mahjong.risk.check --data runs/risk/data --model runs/risk/fit/best.pt \
        --chunks 18 19 --out runs/risk/check.md

Per tile (combined over the three opponents, as ``Danger::tile``) the observed answer is
"at least one opponent wins on the kind by discard now" from the real hands; the
population is every kind with an unseen copy (``all``), the kinds in the thrower's hand
(``hand``), and the kind really thrown (``thrown``). Per opponent (check d) the
population is every (decision, opponent, alive kind) triple; the table and belief
figures there are the raw ``tenpai × wait`` (no calibration curve exists per
opponent), exact counting is ``ready × share`` where it applies and the table figure
elsewhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mahjong.risk.data import RiskData, alive, inputs, targets
from mahjong.risk.model import RiskNet

#: Upper bin edges in percent, as ``danger_check --exact`` (Task 13).
EDGES = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 25.0]
EPS = 1e-6

F = npt.NDArray[np.float64]


def bins_of(p: F) -> npt.NDArray[np.int64]:
    return np.searchsorted(np.asarray(EDGES) / 100.0, p, side="right")


def bin_name(i: int) -> str:
    lo = 0.0 if i == 0 else EDGES[i - 1]
    return f"{lo:g}–{EDGES[i]:g} %" if i < len(EDGES) else f"{lo:g} % +"


def log_loss(p: F, y: F) -> float:
    q = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def brier(p: F, y: F) -> float:
    return float(np.mean((p - y) ** 2))


def table_a(sources: dict[str, F], y: F) -> tuple[str, dict[str, bool]]:
    """The ten-bin table and, per source, whether every bin with 500 or more entries is
    within 0.8 to 1.25 of observed."""
    lines = [
        "| predicted bin | "
        + " | ".join(f"{s}: n / pred / obs / obs÷pred" for s in sources)
        + " |",
        "|---|" + "---|" * len(sources),
    ]
    ok = {s: True for s in sources}
    binned = {s: bins_of(p) for s, p in sources.items()}
    for b in range(len(EDGES) + 1):
        cells = []
        for s, p in sources.items():
            m = binned[s] == b
            n = int(m.sum())
            if n == 0:
                cells.append("–")
                continue
            pred, obs = float(p[m].mean()), float(y[m].mean())
            ratio = obs / pred if pred > 0 else float("nan")
            if n >= 500 and not (0.8 <= ratio <= 1.25):
                ok[s] = False
            cells.append(f"{n:,} / {100 * pred:.2f} % / {100 * obs:.2f} % / {ratio:.2f}")
        lines.append(f"| {bin_name(b)} | " + " | ".join(cells) + " |")
    return "\n".join(lines), ok


def table_b(sources: dict[str, F], y: F) -> str:
    lines = [
        "| source | n | mean predicted | observed | log loss | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for s, p in sources.items():
        lines.append(
            f"| {s} | {len(p):,} | {100 * p.mean():.3f} % | {100 * y.mean():.3f} % "
            f"| {log_loss(p, y):.5f} | {brier(p, y):.6f} |"
        )
    return "\n".join(lines)


def table_c(sources: dict[str, F], y: F) -> str:
    lines = [
        "| source | above 50 %: n / share / observed | below 1 %: n / share / observed |",
        "|---|---|---|",
    ]
    for s, p in sources.items():
        hi, lo = p > 0.5, p < 0.01

        def cell(m: npt.NDArray[np.bool_]) -> str:
            obs = 100 * y[m].mean() if m.any() else float("nan")
            return f"{int(m.sum()):,} / {100 * m.mean():.2f} % / {obs:.2f} %"

        lines.append(f"| {s} | {cell(hi)} | {cell(lo)} |")
    return "\n".join(lines)


def combine(p: F) -> F:
    """``1 − Π (1 − p_o)`` over the opponent axis."""
    return 1.0 - np.prod(1.0 - p, axis=1)


def run(args: argparse.Namespace) -> str:
    torch.set_num_threads(args.threads)
    data = RiskData(args.data)
    net = RiskNet.load(Path(args.model))
    net.eval()
    parts: list[dict[str, Any]] = []
    for c in args.chunks:
        rows = np.asarray(data.chunk(int(c)))
        x = torch.from_numpy(inputs(rows))
        al = alive(rows)
        win, points = targets(rows)
        with torch.no_grad():
            p_list, pts_list = [], []
            for i in range(0, len(x), 4096):
                p, pts = net.predict(x[i : i + 4096], torch.from_numpy(al[i : i + 4096]))
                p_list.append(p.numpy())
                pts_list.append(pts.numpy())
        parts.append(
            {
                "rows": rows,
                "alive": al,
                "win": win,
                "points": points,
                "net_p": np.concatenate(p_list).astype(np.float64),
                "net_points": np.concatenate(pts_list).astype(np.float64),
            }
        )
    rows = np.concatenate([p["rows"] for p in parts])
    al = np.concatenate([p["alive"] for p in parts])
    win = np.concatenate([p["win"] for p in parts]).astype(np.float64)
    points = np.concatenate([p["points"] for p in parts]).astype(np.float64)
    net_p = np.concatenate([p["net_p"] for p in parts])
    net_points = np.concatenate([p["net_points"] for p in parts])
    n = len(rows)
    deals = len(np.unique(rows["deal"]))

    # Per-opponent sources, (n, 3, 34).
    table_p = np.asarray(rows["table_p"], dtype=np.float64)
    belief_p = np.asarray(rows["belief_p"], dtype=np.float64)
    exact_p = np.asarray(rows["exact_p"], dtype=np.float64)
    is_exact = np.asarray(rows["is_exact"], dtype=bool)
    exact_p_full = np.where(is_exact[:, :, None], np.nan_to_num(exact_p), table_p)
    # The network with the exact 0 and 1 override, as `Danger` applies it.
    over_p = net_p.copy()
    ex0 = is_exact[:, :, None] & (exact_p == 0.0)
    ex1 = is_exact[:, :, None] & (exact_p == 1.0)
    over_p[ex0] = 0.0
    over_p[ex1] = 1.0

    # Per-tile sources, (n, 34).
    tile = {
        "table": np.asarray(rows["table_tile"], dtype=np.float64),
        "exact": np.asarray(rows["exact_tile"], dtype=np.float64),
        "belief": np.asarray(rows["belief_tile"], dtype=np.float64),
        "network": combine(net_p),
        "network+exact 0/1": combine(over_p),
    }
    y_tile = win.max(axis=1)
    has_belief = not np.isnan(tile["belief"]).all()
    if not has_belief:
        tile.pop("belief")
    hand = np.asarray(rows["hand"]) > 0
    thrown = np.zeros_like(al)
    thrown[np.arange(n), rows["thrown"].astype(int)] = True
    pops = {"all": al, "hand": al & hand, "thrown": thrown}

    md: list[str] = []
    md.append(
        f"Held-out chunks {list(args.chunks)}: {deals:,} deals, {n:,} discard decisions, "
        f"{int(al.sum()):,} (decision, alive kind) pairs, {int((al & hand).sum()):,} in hand. "
        f"Network `{Path(args.model).name}`, {net.parameter_count():,} parameters."
    )
    verdict_a: dict[str, bool] = {}
    scores: dict[str, tuple[float, float]] = {}
    for name, m in pops.items():
        src = {s: v[m] for s, v in tile.items()}
        y = y_tile[m]
        md.append(
            f"\n### Population `{name}` ({int(m.sum()):,} pairs, observed {100 * y.mean():.3f} %)\n"
        )
        md.append(
            "(a) Ten bins of predicted against observed "
            "(n / mean predicted / observed / observed ÷ predicted):\n"
        )
        ta, ok = table_a(src, y)
        md.append(ta)
        if name == "all":
            verdict_a = ok
            scores = {s: (log_loss(p, y), brier(p, y)) for s, p in src.items()}
        md.append(
            "\n(b) Telling single tiles apart "
            "(log loss, natural, lower is better; Brier beside it):\n"
        )
        md.append(table_b(src, y))
        md.append("\n(c) Confident calls:\n")
        md.append(table_c(src, y))

    # Expected loss (points), per tile, population all.
    shares = np.asarray(rows["shares"], dtype=np.float64)[:, :, None]
    true_loss = (win * points * shares).sum(axis=1)
    net_loss = (net_p * net_points * shares).sum(axis=1)
    over_loss = (
        over_p
        * np.where(ex1, np.asarray(rows["exact_points"], dtype=np.float64), net_points)
        * shares
    ).sum(axis=1)
    loss_src = {
        "table": np.asarray(rows["table_loss"], dtype=np.float64),
        "exact": np.asarray(rows["exact_loss"], dtype=np.float64),
        "network": net_loss,
        "network+exact 0/1": over_loss,
    }
    if has_belief:
        loss_src["belief"] = np.asarray(rows["belief_loss"], dtype=np.float64)
    m = pops["all"]
    md.append("\n### Points it would cost (expected loss per discard, population `all`)\n")
    md.append("| source | mean predicted | mean true | ratio | mean abs error | corr |")
    md.append("|---|---|---|---|---|---|")
    t = true_loss[m]
    for s, v in loss_src.items():
        v = v[m]
        corr = float(np.corrcoef(v, t)[0, 1]) if v.std() > 0 else float("nan")
        md.append(
            f"| {s} | {v.mean():.4f} | {t.mean():.4f} | {v.mean() / t.mean():.2f} "
            f"| {np.abs(v - t).mean():.4f} | {corr:.3f} |"
        )
    # Points given a win: predicted points on the kinds that really win.
    w = win.astype(bool)
    md.append(
        f"\nOn the {int(w.sum()):,} (decision, opponent, kind) triples that really win: "
        f"true points per share "
        f"{points[w].mean():.1f}, network {net_points[w].mean():.1f} "
        f"(mean abs error {np.abs(net_points[w] - points[w]).mean():.1f})."
    )

    # (d) per opponent by concealed count.
    md.append(
        "\n### (d) Per opponent, by its concealed tiles "
        "(every (decision, opponent, alive kind) triple)\n"
    )
    conc = np.asarray(rows["concealed"], dtype=int)
    per = {
        "table (raw)": table_p,
        "exact": exact_p_full,
        "network": net_p,
        "network+exact 0/1": over_p,
    }
    if has_belief:
        per["belief (raw)"] = belief_p
    md.append(
        "| concealed | triples | observed | "
        + " | ".join(f"{s}: pred / log loss / Brier" for s in per)
        + " |"
    )
    md.append("|---|---|---|" + "---|" * len(per))
    alive3 = np.broadcast_to(al[:, None, :], win.shape)
    for c in [1, 4, 7, 10, 13]:
        sel = alive3 & (conc[:, :, None] == c)
        if not sel.any():
            continue
        y = win[sel]
        cells = [
            f"{100 * p[sel].mean():.3f} % / {log_loss(p[sel], y):.5f} / {brier(p[sel], y):.6f}"
            for p in per.values()
        ]
        md.append(
            f"| {c} | {int(sel.sum()):,} | {100 * y.mean():.3f} % | " + " | ".join(cells) + " |"
        )

    # Exact 0 and 1 against the network.
    md.append("\n### Where exact counting proves 0 or 1, what the network says\n")
    for val, m3 in [(0.0, ex0 & alive3), (1.0, ex1 & alive3)]:
        if not m3.any():
            md.append(f"Exactly {val:g}: no triples.")
            continue
        p = net_p[m3]
        y = win[m3]
        md.append(
            f"Exactly {val:g}: {int(m3.sum()):,} triples (observed {100 * y.mean():.3f} %); "
            f"network mean {100 * p.mean():.3f} %, median {100 * np.median(p):.3f} %, "
            f"{'max' if val == 0 else 'min'} {100 * (p.max() if val == 0 else p.min()):.2f} %; "
            f"share within 0.05 of the proof: {100 * np.mean(np.abs(p - val) < 0.05):.1f} %."
        )
        order = np.argsort(-np.abs(p - val))[:8]
        idx = np.argwhere(m3)[order]
        md.append(
            "\nWorst cases (deal, seat, wall left, opponent's concealed, kind index, network %):\n"
        )
        for (r, o, k), pv in zip(idx, p[order], strict=True):
            md.append(
                f"- deal {rows['deal'][r]}, seat {rows['seat'][r]}, wall {rows['wall'][r]}, "
                f"opponent +{o + 1} with {conc[r, o]} concealed, kind {k}: {100 * pv:.1f} %"
            )

    # Verdict.
    md.append("\n### Verdict line\n")
    right = verdict_a.get("network", False)
    better_table = scores["network"][0] < scores["table"][0]
    better_exact = scores["network"][0] < scores["exact"][0]
    md.append(
        f"Is the network right on average (every bin with 500 or more pairs within 0.8 to 1.25 "
        f"of observed, population `all`): **{'yes' if right else 'no'}**. "
        f"Does it tell single tiles apart better than the table (log loss "
        f"{scores['network'][0]:.5f} against {scores['table'][0]:.5f}): "
        f"**{'yes' if better_table else 'no'}**; "
        f"than the exact figure ({scores['exact'][0]:.5f}): **{'yes' if better_exact else 'no'}**."
    )
    md.append(
        "Bins within 0.8–1.25 (population `all`): "
        + ", ".join(f"{s}: {'yes' if ok else 'no'}" for s, ok in verdict_a.items())
        + "."
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
    p.add_argument("--model", required=True)
    p.add_argument("--chunks", nargs="+", type=int, default=[18, 19])
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args(argv)


if __name__ == "__main__":
    print(run(parse_args()))
