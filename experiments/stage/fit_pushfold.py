"""Fit the push-or-fold model from play-out labels (plan Task 12).

Reads the CSV written by ``pushfold label`` (``crates/search/examples/pushfold.rs``),
prints step 2's tables (mean label by danger bin, my hand's points, live winning
tiles, rounds left and the size of the visible threat, with the share of positions
where the fold is better), fits a ridge-linear model and a small network (two
hidden layers, ReLU, a linear skip) to the label, checks both on held-out deals
(mean label of the positions each says fold, with its 95 % range) and writes the
better one as Rust constants to ``crates/search/src/pushfold_tables.rs``.

Usage::

    python experiments/stage/fit_pushfold.py runs/stage/t12-labels-seed1.csv \
        --report runs/stage/t12-step2.md

Plan Task 14 (labels from ``pushfold label-exact``, the exact danger figure where it
is 0 or 1 as the input): the same fit into the second weight file, and a section on
where the exact figure applied::

    python experiments/stage/fit_pushfold.py runs/stage/t14-labels-seed1.csv \
        runs/stage/t14-labels-seed2.csv --report runs/stage/t14-step2.md \
        --tables crates/search/src/pushfold_exact_tables.rs

Plan Task 20 (labels from ``BELIEF=models/belief-order.mjnn pushfold label``, the
danger read from Task 19's model): the same fit into ``pushfold_read_tables.rs``::

    python experiments/stage/fit_pushfold.py runs/t20/labels-read-seed1.csv         runs/t20/labels-read-seed2.csv --report runs/t20/fold-learned-step2.md         --tables crates/search/src/pushfold_read_tables.rs

Rows are weighted by ``n_eligible`` (one decision was drawn per deal, so a deal
with many candidate decisions stands for all of them).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

FEATURES = [
    "d_t",
    "loss_t",
    "d_s",
    "loss_s",
    "my_points",
    "my_own_points",
    "live",
    "rounds_left",
    "unseen",
    "fold_ready",
    "fold_live",
    "fold_points",
    "v_measured",
    "pay",
    "t_is_drawn",
    "my_melds",
    "threat_next",
    "pay_next",
    "melds_next",
    "discards_next",
    "liable_next",
    "threat_across",
    "pay_across",
    "melds_across",
    "discards_across",
    "liable_across",
    "threat_prev",
    "pay_prev",
    "melds_prev",
    "discards_prev",
    "liable_prev",
]
DANGER_EDGES = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.25]


def load(paths: list[str]) -> dict[str, np.ndarray]:
    """All CSVs as one table of float columns (tiles kept as strings)."""
    cols: dict[str, list] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != len(header):
                    continue
                for h, v in zip(header, parts, strict=True):
                    cols.setdefault(h, []).append(v)
    out: dict[str, np.ndarray] = {}
    for h, v in cols.items():
        if h in ("push", "fold", "tiles", "s_table"):
            out[h] = np.array(v)
        else:
            out[h] = np.array(v, dtype=np.float64)
    return out


def danger_bins(title: str, dg: np.ndarray, hit: np.ndarray, pd: np.ndarray) -> list[str]:
    """One ten-bin table of a danger figure against the true deal-in."""
    out = [
        f"\n### {title} ({len(dg):,} tiles in hand at the labelled decisions)\n",
        "| predicted bin | tiles | mean predicted | truly deals in | 95 % range | "
        "observed ÷ predicted | points paid per tile |",
        "|---|---|---|---|---|---|---|",
    ]
    b = bins_of(dg, DANGER_EDGES)
    names = bin_labels(DANGER_EDGES, lambda v: f"{100 * v:g} %")
    for k, name in enumerate(names):
        m = b == k
        if not m.any():
            continue
        p = hit[m].mean()
        half = 1.96 * math.sqrt(p * (1 - p) / m.sum())
        out.append(
            f"| {name} | {int(m.sum()):,} | {100 * dg[m].mean():.2f} % | {100 * p:.2f} % | "
            f"{100 * max(p - half, 0):.2f}–{100 * (p + half):.2f} % | "
            f"{p / max(dg[m].mean(), 1e-9):.2f} | {pd[m].mean():.2f} |"
        )
    return out


def truth_table(d: dict[str, np.ndarray]) -> list[str]:
    """The danger input against the true deal-in of every tile in hand.

    Task 12's rows carry ``tile:table danger:points``; Task 14's
    ``tile:risk input:points:table danger``, where the risk input is the exact
    figure where it is exactly 0 or 1, else the table.
    """
    if "tiles" not in d:
        return []
    dangers, tables, hits, paid = [], [], [], []
    for row in d["tiles"]:
        for item in row.split(";"):
            if not item:
                continue
            parts = item.split(":")
            dangers.append(float(parts[1]))
            paid.append(float(parts[2]))
            hits.append(float(parts[2]) > 0)
            tables.append(float(parts[3]) if len(parts) > 3 else float(parts[1]))
    dg, tb = np.array(dangers), np.array(tables)
    hit, pd = np.array(hits, dtype=float), np.array(paid)
    with_exact = bool(np.any(dg != tb))
    out: list[str] = []
    if with_exact:
        out += danger_bins(
            "The risk input (the exact figure where it is 0 or 1, else the table) against "
            "the truth",
            dg,
            hit,
            pd,
        )
        out += danger_bins("The table's calibrated danger alone against the truth", tb, hit, pd)
        out.append("\n#### Where the exact figure applied\n")
        out.append("| tiles | count | share of tiles | truly deal in | mean table danger |")
        out.append("|---|---|---|---|---|")
        for name, m in [
            ("exactly 0 (the table part stays)", (dg != tb) & (dg < 1.0)),
            ("exactly 1", dg == 1.0),
            ("the table (no exact figure, or neither 0 nor 1)", dg == tb),
        ]:
            if not m.any():
                out.append(f"| {name} | 0 | 0.0 % | – | – |")
                continue
            out.append(
                f"| {name} | {int(m.sum()):,} | {100 * m.mean():.1f} % | "
                f"{100 * hit[m].mean():.2f} % | {100 * tb[m].mean():.2f} % |"
            )
    else:
        out += danger_bins("The table's calibrated danger against the truth", dg, hit, pd)
    share_in = hit.mean()
    out.append(
        f"\nOf all tiles, {100 * share_in:.2f} % truly deal in; the table gives one figure to "
        "tiles that are in truth 0 or 1, so within a bin the truth is spread as the "
        "observed rate says, and no model reading only the table can tell the two apart."
    )
    return out


def exact_section(d: dict[str, np.ndarray]) -> list[str]:
    """Task 14: where the exact figure changed the decision's inputs."""
    if "exact_t" not in d:
        return []
    y, se, w = d["label"], d["se"], d["n_eligible"]
    out = ["\n### The exact figure at the labelled decisions\n"]
    n_ex, n_tb = float(w.sum()), float(d["n_eligible_table"].sum())
    out.append(
        f"Candidate decisions in the labelled deals: {int(n_ex):,} with the exact layer on, "
        f"{int(n_tb):,} with the table alone (a tile the exact figure calls safe makes a "
        "safer `s` exist, or removes `t`'s loss)."
    )
    changed = (d["exact_t"] >= 0) | (d["exact_s"] >= 0)
    s_changed = d["s_table"] != d["fold"]
    not_table = d["table_eligible"] == 0
    out.append(
        f"Of the {len(y):,} labelled decisions, the exact figure applied to `t` or `s` in "
        f"{100 * float(np.sum(w * changed) / w.sum()):.1f} % (weighted; unweighted "
        f"{100 * changed.mean():.1f} %); the safest tile `s` differs from the table's in "
        f"{100 * s_changed.mean():.1f} %; the table alone would not have seen a decision in "
        f"{100 * not_table.mean():.1f} %."
    )
    out.append(
        "\n| group | positions | mean label (push − fold) | 95 % range | fold better | "
        "fold clearly better | `t` truly deals in | `s` truly deals in |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    groups = [
        ("`t` exact 1 (certain deal-in)", d["exact_t"] == 1),
        ("`t` exact 0", d["exact_t"] == 0),
        ("`t` from the table", d["exact_t"] < 0),
        ("`s` exact 0", d["exact_s"] == 0),
        ("`s` from the table", d["exact_s"] < 0),
        ("`s` differs from the table's", s_changed),
        ("no exact figure on `t` or `s`", ~changed),
    ]
    for name, m in groups:
        if not m.any():
            out.append(f"| {name} | 0 | – | – | – | – | – | – |")
            continue
        mean, s = wmean_se(y[m], w[m])
        better = float(np.sum(w[m] * (y[m] < 0)) / w[m].sum())
        clearly = float(np.sum(w[m] * (y[m] < -1.96 * se[m])) / w[m].sum())
        rt = float(np.sum(w[m] * d["true_t"][m]) / w[m].sum())
        rs = float(np.sum(w[m] * d["true_s"][m]) / w[m].sum())
        out.append(
            f"| {name} | {int(m.sum()):,} | {mean:+.2f} | ±{1.96 * s:.2f} | {100 * better:.1f} % | "
            f"{100 * clearly:.1f} % | {100 * rt:.1f} % | {100 * rs:.1f} % |"
        )
    return out


def wmean_se(y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Weighted mean and its standard error (effective sample size)."""
    if len(y) == 0 or w.sum() == 0:
        return math.nan, math.nan
    m = float(np.sum(w * y) / w.sum())
    var = float(np.sum(w * (y - m) ** 2) / w.sum())
    n_eff = float(w.sum() ** 2 / np.sum(w**2))
    return m, math.sqrt(var / max(n_eff, 1.0))


def bins_of(x: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.searchsorted(np.array(edges), x, side="right")


def bin_labels(edges: list[float], fmt) -> list[str]:
    names = []
    lo = None
    for e in edges:
        names.append(f"{fmt(lo)}–{fmt(e)}" if lo is not None else f"< {fmt(e)}")
        lo = e
    names.append(f"≥ {fmt(lo)}")
    return names


def table(
    title: str,
    key: np.ndarray,
    names: list[str],
    y: np.ndarray,
    se: np.ndarray,
    w: np.ndarray,
    real: np.ndarray,
    out: list[str],
) -> None:
    out.append(f"\n### {title}\n")
    out.append(
        "| bin | positions | mean label (push − fold) | 95 % range | fold better | "
        "fold clearly better | `t` truly deals in |"
    )
    out.append("|---|---|---|---|---|---|---|")
    for b, name in enumerate(names):
        m = key == b
        if not m.any():
            continue
        mean, s = wmean_se(y[m], w[m])
        better = float(np.sum(w[m] * (y[m] < 0)) / w[m].sum())
        clearly = float(np.sum(w[m] * (y[m] < -1.96 * se[m])) / w[m].sum())
        r, _ = wmean_se(real[m], w[m])
        out.append(
            f"| {name} | {int(m.sum()):,} | {mean:+.2f} | ±{1.96 * s:.2f} | "
            f"{100 * better:.1f} % | {100 * clearly:.1f} % | {100 * r:.1f} % |"
        )


def step2(d: dict[str, np.ndarray]) -> list[str]:
    y, se, w = d["label"], d["se"], d["n_eligible"]
    real = d["true_t"]
    out = [f"## Step 2: the labels read without a model ({len(y):,} positions)\n"]
    mean, s = wmean_se(y, w)
    better = float(np.sum(w * (y < 0)) / w.sum())
    clearly = float(np.sum(w * (y < -1.96 * se)) / w.sum())
    tin = float(np.sum(w * d["true_t"]) / w.sum())
    sin = float(np.sum(w * d["true_s"]) / w.sum())
    out.append(
        f"Mean label {mean:+.2f} ± {1.96 * s:.2f} points (push − fold, weighted by the "
        f"deal's candidate decisions); mean standard error of one position's label "
        f"{float(se.mean()):.2f} over {int(d['samples'][0])} play-out pairs. "
        f"Fold better (label < 0) in {100 * better:.1f} % of positions; clearly better "
        f"(label < −1.96 × its standard error) in {100 * clearly:.1f} %. The chosen tile "
        f"`t` truly deals in at {100 * tin:.2f} % of the decisions (mean calibrated danger "
        f"{100 * float(np.sum(w * d['d_t']) / w.sum()):.2f} %), the safest tile `s` at "
        f"{100 * sin:.2f} % (mean calibrated danger "
        f"{100 * float(np.sum(w * d['d_s']) / w.sum()):.2f} %). The last column of each "
        f"table is the share of decisions where `t` truly deals in."
    )
    out += truth_table(d)
    out += exact_section(d)
    out.append("\n### The labels by what the seat sees\n")
    m_in = d["true_t"] > 0
    for name, m in [("`t` truly deals in", m_in), ("`t` is truly safe", ~m_in)]:
        mm, ss = wmean_se(y[m], w[m])
        out.append(f"- {name}: {int(m.sum()):,} positions, mean label {mm:+.2f} ± {1.96 * ss:.2f}")
    table(
        "By calibrated danger of the chosen tile `t`",
        bins_of(d["d_t"], DANGER_EDGES),
        bin_labels(DANGER_EDGES, lambda v: f"{100 * v:g} %"),
        y,
        se,
        w,
        real,
        out,
    )
    pts_edges = [9.0, 17.0, 33.0, 65.0]
    table(
        "By my hand's points on a discard win (8 = 3 faan, 16 = 4, 24–32 = 5–6, 48–64 = 7–8)",
        bins_of(d["my_points"], pts_edges),
        ["8", "16", "24–32", "48–64", "≥ 96"],
        y,
        se,
        w,
        real,
        out,
    )
    live_edges = [3.0, 5.0, 9.0]
    table(
        "By live winning tiles",
        bins_of(d["live"], live_edges),
        ["1–2", "3–4", "5–8", "9+"],
        y,
        se,
        w,
        real,
        out,
    )
    rounds_edges = [3.0, 6.0, 9.0, 12.0]
    table(
        "By rounds left",
        bins_of(d["rounds_left"], rounds_edges),
        ["0–2", "3–5", "6–8", "9–11", "12+"],
        y,
        se,
        w,
        real,
        out,
    )
    threat = np.maximum.reduce([d["threat_next"], d["threat_across"], d["threat_prev"]])
    threat_edges = [0.1, 0.3, 0.6]
    table(
        "By the largest opponent ready chance (the belief's `tenpai`)",
        bins_of(threat, threat_edges),
        ["< 10 %", "10–30 %", "30–60 %", "≥ 60 %"],
        y,
        se,
        w,
        real,
        out,
    )
    pay = np.maximum.reduce([d["pay_next"], d["pay_across"], d["pay_prev"]])
    pay_edges = [9.0, 17.0, 33.0, 65.0]
    table(
        "By the largest opponent's expected points per share if it wins on a discard",
        bins_of(pay, pay_edges),
        ["≤ 8", "9–16", "17–32", "33–64", "≥ 65"],
        y,
        se,
        w,
        real,
        out,
    )
    # Task 11's rule on the same rows (the fold side as -loss(s); when `s` also
    # leaves the hand ready the real rule used s's push value, approximated here).
    t11_fold = (1 - d["d_t"]) * d["v_measured"] - d["loss_t"] <= -d["loss_s"]
    m, s = wmean_se(y[t11_fold], w[t11_fold])
    share = float(np.sum(w * t11_fold) / w.sum())
    out.append(
        f"\nTask 11's rule (from the row's own figures) folds in {100 * share:.1f} % of "
        f"these positions; the mean label where it folds is {m:+.2f} ± {1.96 * s:.2f} "
        f"(a positive value means those folds lose points)."
    )
    return out


class Net(torch.nn.Module):
    def __init__(self, n_in: int, hidden: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(n_in, hidden)
        self.l2 = torch.nn.Linear(hidden, hidden)
        self.l3 = torch.nn.Linear(hidden, 1)
        self.skip = torch.nn.Linear(n_in, 1, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.l1(z))
        h = torch.relu(self.l2(h))
        return (self.l3(h) + self.skip(z)).squeeze(-1)


def fit_net(
    z: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    hidden: int,
    epochs: int,
    seed: int,
    weight_decay: float,
) -> tuple[Net, float]:
    torch.manual_seed(seed)
    net = Net(z.shape[1], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=weight_decay)
    zt, yt, wt = (torch.tensor(a, dtype=torch.float32) for a in (z, y, w))
    zv, yv, wv = (torch.tensor(a, dtype=torch.float32) for a in (z_val, y_val, w_val))
    best_state, best_val, since = None, math.inf, 0
    n = len(zt)
    batch = 256
    for _ in range(epochs):
        perm = torch.randperm(n)
        net.train()
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            pred = net(zt[idx])
            loss = (wt[idx] * (pred - yt[idx]) ** 2).sum() / wt[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(((wv * (net(zv) - yv) ** 2).sum() / wv.sum()).item())
        if val < best_val - 1e-6:
            best_val, since = val, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            since += 1
            if since >= 20:
                break
    assert best_state is not None
    net.load_state_dict(best_state)
    net.eval()
    return net, best_val


def ridge(z: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Weighted ridge regression on standardised inputs: (coefficients, intercept)."""
    zb = np.hstack([z, np.ones((len(z), 1))])
    a = zb.T @ (w[:, None] * zb) / w.sum()
    reg = lam * np.eye(zb.shape[1])
    reg[-1, -1] = 0.0
    b = zb.T @ (w * y) / w.sum()
    coef = np.linalg.solve(a + reg, b)
    return coef[:-1], float(coef[-1])


def fold_check(pred: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Held-out check: the positions the model says fold."""
    m = pred < 0
    share = float(np.sum(w * m) / w.sum())
    if m.any():
        mean, se = wmean_se(y[m], w[m])
    else:
        mean, se = math.nan, math.nan
    gain = float(-np.sum(w * y * m) / w.sum())
    mse = float(np.sum(w * (pred - y) ** 2) / w.sum())
    return {"share": share, "mean": mean, "ci": 1.96 * se, "gain": gain, "mse": mse}


def rust_floats(a: np.ndarray) -> str:
    return "[" + ", ".join(repr(float(v)) for v in a) + "]"


def write_tables(
    path: Path,
    mean: np.ndarray,
    sd: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    lin: np.ndarray,
    b3: float,
    check: list[tuple[np.ndarray, float]],
    note: str,
) -> None:
    hidden = len(b1)
    exact = "exact" in path.name
    read = "read" in path.name
    tables_arg = f" --tables {path.as_posix()}" if exact or read else ""
    if exact:
        what = "with the exact danger input ([`crate::pushfold::EXACT_MODEL`])"
    elif read:
        what = "on the danger read from Task 19's model ([`crate::pushfold::READ_MODEL`])"
    else:
        what = "([`crate::pushfold::TABLE_MODEL`])"
    lines = [
        f"//! Weights of the push-or-fold model {what}. Generated by",
        f"//! `experiments/stage/fit_pushfold.py{tables_arg}`; do not edit.",
        f"//! {note}",
        "",
        "use crate::pushfold::{FEATURES, HIDDEN};",
        "",
        "/// Units per hidden layer: `pushfold::HIDDEN`, the size this file was written with.",
        f"const _: () = assert!(HIDDEN == {hidden});",
        "/// Input means and standard deviations (inputs are standardised first).",
        "#[rustfmt::skip]",
        f"pub const MEAN: [f64; FEATURES] = {rust_floats(mean)};",
        "#[rustfmt::skip]",
        f"pub const SD: [f64; FEATURES] = {rust_floats(sd)};",
        "#[rustfmt::skip]",
        "pub const W1: [[f64; FEATURES]; HIDDEN] = [",
    ]
    lines += [f"    {rust_floats(row)}," for row in w1]
    lines += [
        "];",
        "#[rustfmt::skip]",
        f"pub const B1: [f64; HIDDEN] = {rust_floats(b1)};",
        "#[rustfmt::skip]",
        "pub const W2: [[f64; HIDDEN]; HIDDEN] = [",
    ]
    lines += [f"    {rust_floats(row)}," for row in w2]
    lines += [
        "];",
        "#[rustfmt::skip]",
        f"pub const B2: [f64; HIDDEN] = {rust_floats(b2)};",
        "#[rustfmt::skip]",
        f"pub const W3: [f64; HIDDEN] = {rust_floats(w3)};",
        "/// Linear skip from the standardised inputs.",
        "#[rustfmt::skip]",
        f"pub const LIN: [f64; FEATURES] = {rust_floats(lin)};",
        f"pub const B3: f64 = {b3!r};",
        "/// Held-out rows and the script's predictions, for the parity test.",
        "#[rustfmt::skip]",
        "pub const CHECK: &[([f64; FEATURES], f64)] = &[",
    ]
    lines += [f"    ({rust_floats(x)}, {y!r})," for x, y in check]
    lines += ["];", ""]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def forward_np(
    x: np.ndarray,
    mean: np.ndarray,
    sd: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    lin: np.ndarray,
    b3: float,
) -> np.ndarray:
    """The Rust forward pass in float64, for the parity rows."""
    z = (x - mean) / sd
    h1 = np.maximum(z @ w1.T + b1, 0.0)
    h2 = np.maximum(h1 @ w2.T + b2, 0.0)
    return h2 @ w3 + z @ lin + b3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--report", default="runs/stage/t12-step2.md")
    ap.add_argument(
        "--tables",
        default="crates/search/src/pushfold_tables.rs",
        help="Rust file to write; 'none' skips the write",
    )
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--holdout", type=int, default=5, help="deal %% N == 0 is held out")
    args = ap.parse_args()

    # A few threads are enough for these small fits (and the labeller may be running).
    torch.set_num_threads(4)
    d = load(args.csv)
    report = step2(d)

    x = np.stack([d[f] for f in FEATURES], axis=1)
    y, w = d["label"], d["n_eligible"]
    held = (d["deal"].astype(np.int64) % args.holdout) == 0
    train, val = ~held, held
    mean = (x[train] * w[train, None]).sum(0) / w[train].sum()
    sd = np.sqrt(((x[train] - mean) ** 2 * w[train, None]).sum(0) / w[train].sum())
    sd = np.where(sd < 1e-9, 1.0, sd)
    z = (x - mean) / sd

    report.append(
        f"\n## Step 3: the model\n\nTraining rows {int(train.sum()):,}, held-out rows "
        f"{int(val.sum()):,} (deals with number divisible by {args.holdout})."
    )
    results = {}
    coef, b0 = ridge(z[train], y[train], w[train], args.ridge)
    pred_lin = z[val] @ coef + b0
    results["ridge linear"] = fold_check(pred_lin, y[val], w[val])
    net, _ = fit_net(
        z[train],
        y[train],
        w[train],
        z[val],
        y[val],
        w[val],
        args.hidden,
        args.epochs,
        args.seed,
        args.weight_decay,
    )
    with torch.no_grad():
        pred_net = net(torch.tensor(z[val], dtype=torch.float32)).numpy().astype(np.float64)
    results[f"network {args.hidden}×{args.hidden} + linear skip"] = fold_check(
        pred_net, y[val], w[val]
    )
    # The oracle: the same network with the true deal-in of `t` and `s` added as
    # inputs (not available to the bot). Its distance from the network above says
    # how much of the error comes from the table's blunt danger figure.
    oracle_cols = np.stack([d["true_t"], d["paid_t"], d["true_s"], d["paid_s"]], axis=1)
    zo = np.hstack([z, oracle_cols / np.maximum(oracle_cols[train].std(0), 1e-9)])
    net_o, _ = fit_net(
        zo[train],
        y[train],
        w[train],
        zo[val],
        y[val],
        w[val],
        args.hidden,
        args.epochs,
        args.seed,
        args.weight_decay,
    )
    with torch.no_grad():
        pred_o = net_o(torch.tensor(zo[val], dtype=torch.float32)).numpy().astype(np.float64)
    results["network + true deal-in of `t` and `s` (oracle, not for the bot)"] = fold_check(
        pred_o, y[val], w[val]
    )
    always = fold_check(np.full(val.sum(), 1.0), y[val], w[val])
    results["always push (`stage`)"] = always
    t11 = (1 - d["d_t"]) * d["v_measured"] - d["loss_t"] <= -d["loss_s"]
    results["Task 11's rule (approximate)"] = fold_check(
        np.where(t11[val], -1.0, 1.0), y[val], w[val]
    )
    report.append(
        "\n| model | held-out MSE | says fold | mean label where it folds | 95 % range | "
        "gain per decision |"
    )
    report.append("|---|---|---|---|---|---|")
    for name, r in results.items():
        mean_s = "–" if math.isnan(r["mean"]) else f"{r['mean']:+.2f}"
        ci_s = "–" if math.isnan(r["ci"]) else f"±{r['ci']:.2f}"
        report.append(
            f"| {name} | {r['mse']:.1f} | {100 * r['share']:.1f} % | {mean_s} | {ci_s} | "
            f"{r['gain']:+.3f} |"
        )
    report.append(
        "\n`gain per decision`: points per candidate decision gained on the held-out rows by "
        "folding where the model says so (−label summed over those rows ÷ all rows' weight); "
        "the fold set must have a mean label below 0 for the model to pass."
    )

    net_r = results[f"network {args.hidden}×{args.hidden} + linear skip"]
    lin_r = results["ridge linear"]
    use_net = net_r["gain"] >= lin_r["gain"]
    chosen = "network" if use_net else "ridge linear"
    chosen_r = net_r if use_net else lin_r
    report.append(f"\nChosen for the bot: **{chosen}** (the larger held-out gain per decision).")
    print("\n".join(report))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")

    if args.tables == "none":
        return 0
    if use_net:
        sd_ = net.state_dict()
        w1 = sd_["l1.weight"].double().numpy()
        b1 = sd_["l1.bias"].double().numpy()
        w2 = sd_["l2.weight"].double().numpy()
        b2 = sd_["l2.bias"].double().numpy()
        w3 = sd_["l3.weight"].double().numpy()[0]
        lin = sd_["skip.weight"].double().numpy()[0]
        b3 = float(sd_["l3.bias"].double().numpy()[0])
    else:
        h = args.hidden  # zero units of the size the Rust side is built with
        w1 = np.zeros((h, len(FEATURES)))
        b1 = np.zeros(h)
        w2 = np.zeros((h, h))
        b2 = np.zeros(h)
        w3 = np.zeros(h)
        lin = coef
        b3 = b0
    idx = np.flatnonzero(val)[:6]
    check = [
        (x[i], float(forward_np(x[i : i + 1], mean, sd, w1, b1, w2, b2, w3, lin, b3)[0]))
        for i in idx
    ]
    note = (
        f"Fitted on {int(train.sum()):,} rows from {', '.join(args.csv)}; {chosen}; "
        f"held-out: says fold in {100 * chosen_r['share']:.1f} % of positions, mean label "
        f"there {chosen_r['mean']:+.2f} ± {chosen_r['ci']:.2f}, gain per decision "
        f"{chosen_r['gain']:+.3f}."
    )
    write_tables(Path(args.tables), mean, sd, w1, b1, w2, b2, w3, lin, b3, check, note)
    print(f"wrote {args.tables}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
