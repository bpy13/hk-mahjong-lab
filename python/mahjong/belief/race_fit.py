"""The race bot's fitted tables and calibration checks (plan 2026-11-09 Tasks 8-9,
``docs/race-bot.md`` §6, §8.1 step 3, §8.2).

From a race log (the ``race_log`` example's rows, labelled with full information):

* ``a`` and ``b`` per wall bucket: an unfolded opponent's chance per round (4 wall
  tiles) to move far → one away (``a``) and one away → ready (``b``), read from the true
  shanten of the same seat one round apart (``fit_steps``);
* ``q`` per mode: P(one discard in that mode completes a given ready opponent's hand),
  the discards' feeds over the ready opponents at that moment (``fit_q``);
* the ``state`` and ``mode`` heads' calibration per wall bucket on a hold-out log
  (``head_calibration``): observed share against mean predicted probability per class,
  with Wilson 95 % intervals;
* the hazard check of §8.2 (``hazard_calibration``): the race bot's own P(opponent wins
  within its horizon) against what happened.

::

    python -m mahjong.belief.race_fit --logs runs/race/g1/logs --holdout runs/race/g1/holdout \\
        --model runs/race/g1/race-belief.pt --out runs/race/g1/fit.json

The output's ``spec`` is the text to append to a ``race:`` spec (``a=…,b=…,q=…``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.recfunctions import repack_fields

from mahjong.belief.evaluate import PROGRESS_BUCKETS, Calibration, buckets
from mahjong.belief.logs import LogDir
from mahjong.belief.model import N_OPP, BeliefNet, batch_from_rows

#: v0's constants (``crates/search/src/race.rs``), kept where the log has too few cases.
DEFAULT_STEP = 0.15
DEFAULT_Q = (0.08, 0.04, 0.01)
STATE_NAMES = ("far", "one_away", "ready", "folded")
MODE_NAMES = ("push", "detour", "fold")
FOLDED = 3
NO_MODE = 255
#: A round: one draw by each player.
ROUND_TILES = 4

Rows = np.ndarray[Any, np.dtype[Any]]


def _rel_to_abs(observer: np.ndarray, r: int) -> np.ndarray:
    return (observer.astype(np.int64) + r) % 4


def fit_steps(
    rows: Rows,
    default_a: list[float] | None = None,
    default_b: list[float] | None = None,
    min_n: int = 200,
) -> dict[str, list[float] | list[int]]:
    """Per-bucket ``a`` (far → one away or better) and ``b`` (one away → ready) per round.

    Every row shows the true shanten of three seats. For each observation of an unfolded
    seat that is far (shanten ≥ 2) or one away (shanten 1, not ready), the same seat's
    first later observation in the hand with at least one round less wall
    (``ROUND_TILES`` tiles) says whether it moved. Buckets with fewer than ``min_n``
    observations keep the default.
    """
    da = list(default_a or [DEFAULT_STEP] * PROGRESS_BUCKETS)
    db = list(default_b or [DEFAULT_STEP] * PROGRESS_BUCKETS)
    hand = np.repeat(rows["hand"].astype(np.int64), N_OPP)
    step = np.repeat(rows["step"].astype(np.int64), N_OPP)
    wall = np.repeat(rows["wall_remaining"].astype(np.int64), N_OPP)
    seat = np.stack([_rel_to_abs(rows["observer"], r) for r in (1, 2, 3)], axis=1).ravel()
    shanten = rows["opp_shanten"].astype(np.int64).ravel()
    ready = rows["opp_tenpai"].astype(bool).ravel()
    folded = rows["opp_state"].astype(np.int64).ravel() == FOLDED
    order = np.lexsort((step, seat, hand))
    hand, seat, wall, shanten, ready, folded = (
        x[order] for x in (hand, seat, wall, shanten, ready, folded)
    )
    move_a = np.zeros(PROGRESS_BUCKETS)
    n_a = np.zeros(PROGRESS_BUCKETS, dtype=np.int64)
    move_b = np.zeros(PROGRESS_BUCKETS)
    n_b = np.zeros(PROGRESS_BUCKETS, dtype=np.int64)
    key = hand * 4 + seat
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    for s, e in zip(starts, ends, strict=True):
        w = wall[s:e]
        # Walls fall along the hand: the first later row with wall ≤ w_i − 4.
        j = np.searchsorted(-w, -(w - ROUND_TILES), side="left")
        ok = (j < len(w)) & ~folded[s:e]
        if not ok.any():
            continue
        jj = np.minimum(j, len(w) - 1)
        sh, rd = shanten[s:e], ready[s:e]
        b = buckets(w)
        far = ok & ~rd & (sh >= 2)
        one = ok & ~rd & (sh == 1)
        moved_far = rd[jj] | (sh[jj] <= 1)
        np.add.at(n_a, b[far], 1)
        np.add.at(move_a, b[far], moved_far[far] * 1.0)
        np.add.at(n_b, b[one], 1)
        np.add.at(move_b, b[one], rd[jj][one] * 1.0)
    a = [float(move_a[k] / n_a[k]) if n_a[k] >= min_n else da[k] for k in range(PROGRESS_BUCKETS)]
    bb = [float(move_b[k] / n_b[k]) if n_b[k] >= min_n else db[k] for k in range(PROGRESS_BUCKETS)]
    return {"a": a, "b": bb, "n_a": n_a.tolist(), "n_b": n_b.tolist()}


def fit_q(rows: Rows, default: tuple[float, ...] = DEFAULT_Q) -> dict[str, list[Any]]:
    """``q(m)``: over the discards labelled with mode ``m``, the feeds (ready opponents
    whose wait the discard was) per ready opponent. No ready opponent seen: the default."""
    mode = rows["my_mode"].astype(np.int64)
    feeds = rows["my_feeds"].astype(np.int64)
    ready = rows["ready_opps"].astype(np.int64)
    q: list[float] = []
    n: list[int] = []
    for m in range(len(MODE_NAMES)):
        sel = mode == m
        r = int(ready[sel].sum())
        q.append(float(feeds[sel].sum() / r) if r > 0 else float(default[m]))
        n.append(r)
    return {"q": q, "n_ready": n}


def label_counts(rows: Rows, agents: list[str]) -> dict[str, dict[str, dict[str, int]]]:
    """Each agent's own discard labels (``docs/race-bot.md`` §12): push, detour, fold and
    mistake (written as push in ``my_mode``, flagged by ``my_mistake``; here ``push``
    excludes them), over all its discards and over those when no opponent was ready
    (``calm``) or some was (``threat``). Keys are the agent specs of the log's meta."""
    sel = rows["my_mode"] != NO_MODE
    r = rows[sel]
    agent = r["agents"][np.arange(len(r)), r["observer"].astype(np.int64)].astype(np.int64)
    mode = r["my_mode"].astype(np.int64)
    if "my_mistake" in (r.dtype.names or ()):
        mistake = r["my_mistake"].astype(bool)
    else:  # a log from before §12's mistake flag
        mistake = np.zeros(len(r), dtype=bool)
    label = np.where(mistake, len(MODE_NAMES), mode)
    calm = r["ready_opps"] == 0
    names = (*MODE_NAMES, "mistake")
    out: dict[str, dict[str, dict[str, int]]] = {}
    for i, spec in enumerate(agents):
        mine = agent == i
        if not mine.any():
            continue
        out[spec] = {
            part: dict(
                zip(names, np.bincount(label[mine & m], minlength=len(names)).tolist(), strict=True)
            )
            for part, m in (("all", np.ones_like(calm)), ("calm", calm), ("threat", ~calm))
        }
    return out


def head_calibration(
    rows: Rows, probs_state: np.ndarray, probs_mode: np.ndarray
) -> dict[str, dict[str, list[dict[str, float | bool]]]]:
    """Per bucket and class: observed share of the label against the mean predicted
    probability (``probs_*`` shaped ``(n, 3, classes)``)."""
    bucket = np.repeat(buckets(rows["wall_remaining"]), N_OPP)
    out: dict[str, dict[str, list[dict[str, float | bool]]]] = {}
    for head, names, probs, field in (
        ("state", STATE_NAMES, probs_state, "opp_state"),
        ("mode", MODE_NAMES, probs_mode, "opp_mode"),
    ):
        label = rows[field].astype(np.int64).ravel()
        out[head] = {}
        for c, name in enumerate(names):
            cal = Calibration()
            cal.add(bucket, probs[:, :, c].ravel(), (label == c) * 1.0)
            out[head][name] = cal.table()
    return out


def hazard_calibration(rows: Rows) -> list[dict[str, float | bool]]:
    """§8.2: the race bot's P(opponent wins within its horizon) (its chosen candidate's
    ``p_win``, logged as ``race_pwin``) against whether that opponent won within that
    many rounds (the wall fell by at most ``4 × rounds`` tiles before the hand ended).
    Rows of other bots carry no prediction (``race_rounds = 0``) and are skipped. A
    bucket whose prediction misses the observed share by more than 20 % relative is
    flagged ``refit``."""
    sel = rows["race_rounds"] > 0
    r = rows[sel]
    if len(r) == 0:
        return []
    pred = np.asarray(r["race_pwin"][:, 1:4], dtype=np.float64)
    elapsed = r["wall_remaining"].astype(np.int64) - r["wall_at_end"].astype(np.int64)
    within = elapsed <= ROUND_TILES * r["race_rounds"].astype(np.int64)
    obs = (r["opp_won"] > 0) & within[:, None]
    cal = Calibration()
    cal.add(np.repeat(buckets(r["wall_remaining"]), N_OPP), pred.ravel(), obs.ravel() * 1.0)
    table = cal.table()
    for row in table:
        o, p = float(row["observed"]), float(row["predicted"])
        row["refit"] = bool(int(row["n"]) > 0 and abs(p - o) > 0.2 * max(o, 1e-9))
    return table


def _fmt(xs: list[float]) -> str:
    return "/".join(f"{x:.4g}" for x in xs)


def spec_options(steps: dict[str, Any], q: dict[str, Any]) -> str:
    """The ``race:`` spec options for the fitted tables."""
    return f"a={_fmt(steps['a'])},b={_fmt(steps['b'])},q={_fmt(q['q'])}"


#: The fields the fits read (a generation's log is ~20 GB; only these are loaded).
LABEL_FIELDS = [
    "hand",
    "step",
    "observer",
    "agents",
    "wall_remaining",
    "opp_tenpai",
    "opp_won",
    "opp_state",
    "opp_mode",
    "opp_shanten",
    "my_mode",
    "my_feeds",
    "ready_opps",
    "my_mistake",
    "wall_at_end",
    "race_rounds",
    "race_pwin",
]


def _label_rows(log: LogDir) -> Rows:
    """The label fields of every chunk, packed (no features). A field the log lacks
    (``my_mistake`` before §12) is left out."""
    return np.concatenate(
        [
            repack_fields(np.asarray(c[[f for f in LABEL_FIELDS if f in (c.dtype.names or ())]]))
            for c in log.chunks()
        ]
    )


def _predict(net: BeliefNet, log: LogDir, bs: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    states, modes = [], []
    net.eval()
    with torch.no_grad():
        for rows in log.chunks():
            for i in range(0, len(rows), bs):
                b = batch_from_rows(rows[i : i + bs])
                p = net.predict(net(net.inputs(b.features, b.discards)), b.unseen)
                states.append(p["state"].numpy())
                modes.append(p["mode"].numpy())
    return np.concatenate(states), np.concatenate(modes)


def fit(args: argparse.Namespace) -> dict[str, Any]:
    log = LogDir(args.logs)
    rows = _label_rows(log)
    steps = fit_steps(rows, min_n=args.min_n)
    q = fit_q(rows)
    result: dict[str, Any] = {
        "logs": str(args.logs),
        "rows": int(len(rows)),
        "steps": steps,
        "q": q,
        "spec": spec_options(steps, q),
        "labels": {
            "state": np.bincount(rows["opp_state"].ravel(), minlength=4).tolist(),
            "mode": np.bincount(rows["opp_mode"].ravel(), minlength=3).tolist(),
            "my_mode": np.bincount(
                rows["my_mode"][rows["my_mode"] != NO_MODE], minlength=3
            ).tolist(),
            "by_agent": label_counts(rows, list(log.meta.get("agents", []))),
        },
    }
    hold_log = LogDir(args.holdout) if args.holdout else LogDir(args.logs)
    hold = _label_rows(hold_log) if args.holdout else rows
    result["hazard"] = hazard_calibration(hold)
    if args.model:
        net = BeliefNet.load(Path(args.model))
        if net.cfg.race:
            ps, pm = _predict(net, hold_log)
            result["calibration"] = head_calibration(hold, ps, pm)
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--logs", required=True, help="race log the tables are fitted on")
    p.add_argument("--holdout", default=None, help="race log for the calibration checks")
    p.add_argument("--model", default=None, help="race model (.pt) for the heads' calibration")
    p.add_argument("--min-n", type=int, default=200, help="cases a bucket needs to be refitted")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    result = fit(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=1), encoding="utf-8")
    tmp.replace(out)
    print(f"{out}: {result['rows']} rows; spec {result['spec']}")
    for spec, parts in result["labels"]["by_agent"].items():
        print(f"  {spec}: " + "; ".join(f"{p} {c}" for p, c in parts.items()))


if __name__ == "__main__":
    main()
