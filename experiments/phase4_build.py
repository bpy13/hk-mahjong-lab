"""Phase 4 of the build run: distil the labels, then measure the network (plan Tasks 15-16).

Stages (each resumes: finished work is skipped, chunked matches skip finished chunks):

1. ``distill``   train the suit-shared CNN on Task 13's labels on the GPU
                 (:mod:`mahjong.rl.distill_cnn`); writes ``models/cnn-distill.pt``,
                 ``models/cnn-distill-fp32.mjnn``, ``models/cnn-distill.mjnn`` (int8) and
                 ``runs/distill/result.json``.
2. ``evaluate``  the bare int8 network vs three ``value`` bots, 1,000 walls x 4 seatings
                 (seeds 7000-7019; since the ChaCha8 change these are not Task 7's
                 walls): ``runs/matches/cnn-distill.json``
                 (AC7 bars 1 and 2).
3. ``netpimc``   the 24-seating match {bare CNN, CNN + PIMC-48 (full rollouts), CNN +
                 PIMC-48 (leaf value, depth 6), belief-PIMC-64 (Task 7's teacher)} on
                 ``--netpimc-walls`` walls: ``runs/matches/netpimc.json``, then
                 ``runs/matches/netpimc-verdict.json`` and ``.md``, which apply Brian's
                 rule (2026-10-30): a challenger replaces the search teacher only if it
                 beats it with a 95 % interval above 0 (the wall-paired bootstrap
                 interval of challenger minus teacher), and AC7 bar 3 (a search variant
                 beats the bare network with an interval excluding 0).

4. ``ppo-eval``  plan Task 18's AC8 matches, after the PPO runs (not part of ``all``):
                 ``runs/matches/ppo-eval.json`` and ``.md``; exports ``models/cnn-ppo.mjnn``.

    python experiments/phase4_build.py --stage distill|evaluate|netpimc|ppo-eval|all [--quick]

Progress: ``runs/phase4/progress.txt`` (one line, rewritten) and ``runs/phase4/run.log``.
Intervals of merged chunked matches: each chunk's standard error combined over chunks
(chunks are independent strata of walls), as in Task 7's job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

# A queued job runs a frozen copy of this file from runs/; it names the repo here.
REPO = Path(os.environ.get("HK_LAB_REPO") or Path(__file__).resolve().parents[1])
RUNS = REPO / "runs"
PHASE = RUNS / "phase4"
MATCHES = RUNS / "matches"
MODELS = REPO / "models"

BELIEF = "models/belief.mjnn"
TEACHER = f"pimc:samples=64,candidates=4,belief={BELIEF},weights=1"
CNN = "models/cnn-distill.mjnn"
BOOTSTRAP = 2000


def net_agents(cnn: str = CNN, samples: int = 48) -> list[tuple[str, str]]:
    """The four agents of the netpimc match, in agent-index order (name, spec)."""
    search = f"netpimc:path={cnn},samples={samples},candidates=4,belief={BELIEF},weights=1"
    return [
        ("cnn", f"net:path={cnn}"),
        ("cnn+pimc48", search),
        ("cnn+pimc48-leaf6", f"{search},leaf=value,depth=6"),
        ("teacher", TEACHER),
    ]


def no_power_throttling() -> None:
    # Windows throttles detached headless processes to ~0.15 core each (EcoQoS); opt out.
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [
            ("Version", wintypes.ULONG),
            ("ControlMask", wintypes.ULONG),
            ("StateMask", wintypes.ULONG),
        ]

    st = State(1, 1, 0)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))


def log(msg: str) -> None:
    PHASE.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%dT%H:%M:%S} {msg}"
    print(line, flush=True)
    with (PHASE / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def progress(msg: str) -> None:
    write_atomic(PHASE / "progress.txt", f"{msg} ({datetime.now():%Y-%m-%d %H:%M})\n")


# --- Chunked matches ------------------------------------------------------------------


def combine_se(parts: list[tuple[int, float]], total: int) -> float:
    """Standard error of a hands-weighted mean of independent chunk means."""
    return math.sqrt(sum((n / total) ** 2 * se**2 for n, se in parts))


def merge(
    name: str, specs: list[str], seatings: int, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    """One report from a match's chunks (means and counts exact; intervals combined)."""
    hands = sum(c["hands"] for c in chunks)
    agents = []
    for i in range(4):
        rows = [c["agents"][i] for c in chunks]
        n = sum(r["hands"] for r in rows)
        mean = sum(r["hands"] * r["mean"] for r in rows) / n
        se = combine_se([(r["hands"], r["ci95"] / 1.96) for r in rows], n)
        bse = combine_se(
            [(r["hands"], (r["boot_hi"] - r["boot_lo"]) / (2 * 1.96)) for r in rows], n
        )
        wins = sum(r["wins"] for r in rows)
        pattern_wins: dict[str, int] = {}
        for r in rows:
            for k, v in r["pattern_wins"].items():
                pattern_wins[k] = pattern_wins.get(k, 0) + v
        win_faan = sum(r["win_faan"] for r in rows)
        agents.append(
            {
                "agent": rows[0]["agent"],
                "hands": n,
                "mean": mean,
                "ci95": 1.96 * se,
                "boot_lo": mean - 1.96 * bse,
                "boot_hi": mean + 1.96 * bse,
                "wins": wins,
                "win_rate": wins / n,
                "self_drawn": sum(r["self_drawn"] for r in rows),
                "self_drawn_rate": sum(r["self_drawn"] for r in rows) / n,
                "deal_ins": sum(r["deal_ins"] for r in rows),
                "deal_in_rate": sum(r["deal_ins"] for r in rows) / n,
                "draws": sum(r["draws"] for r in rows),
                "draw_rate": sum(r["draws"] for r in rows) / n,
                "win_faan": win_faan,
                "mean_faan": win_faan / wins if wins else 0.0,
                "pattern_wins": pattern_wins,
                "pattern_shares": {k: v / wins for k, v in pattern_wins.items()} if wins else {},
            }
        )
    pairs = []
    for i, p0 in enumerate(chunks[0]["pairs"]):
        rows = [c["pairs"][i] for c in chunks]
        ns = [c["hands"] for c in chunks]
        mean = sum(n * r["mean"] for n, r in zip(ns, rows, strict=True)) / hands
        se = combine_se([(n, r["ci95"] / 1.96) for n, r in zip(ns, rows, strict=True)], hands)
        bse = combine_se(
            [
                (n, (r["boot_hi"] - r["boot_lo"]) / (2 * 1.96))
                for n, r in zip(ns, rows, strict=True)
            ],
            hands,
        )
        pairs.append(
            {
                "a": p0["a"],
                "b": p0["b"],
                "agent_a": p0["agent_a"],
                "agent_b": p0["agent_b"],
                "mean": mean,
                "ci95": 1.96 * se,
                "boot_lo": mean - 1.96 * bse,
                "boot_hi": mean + 1.96 * bse,
            }
        )
    draws = sum(c["draws"] for c in chunks)
    return {
        "format": "hk-tournament/1-merged",
        "match": name,
        "specs": specs,
        "rules": "default RuleSet (flowers, 3 faan minimum, cap 10, half spicy, all winners, "
        "liability on)",
        "deals": sum(c["deals"] for c in chunks),
        "seatings": seatings,
        "seeds": [c["seed"] for c in chunks],
        "bootstrap": BOOTSTRAP,
        "hands": hands,
        "draws": draws,
        "draw_rate": draws / hands,
        "agents": agents,
        "pairs": pairs,
        "intervals": "chunk-combined: ci95 and the wall-bootstrap interval from per-chunk "
        "standard errors",
    }


def run_match(
    name: str,
    specs: list[str],
    walls: int,
    per_chunk: int,
    seed_base: int,
    seatings: int,
    on_chunk: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Plays ``walls`` walls in chunks of ``per_chunk`` (seed ``seed_base + i``),
    skipping chunks already in ``runs/phase4/chunks/``; writes and returns the merge.
    ``on_chunk`` gets the merge of the chunks so far after each chunk."""
    from mahjong._core import tournament

    chunk_dir = PHASE / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n_chunks = max(1, walls // per_chunk)
    done: list[dict[str, Any]] = []
    for i in range(n_chunks):
        path = chunk_dir / f"{name}-{i:02d}.json"
        if not path.exists():
            t = time.time()
            res = tournament(specs, per_chunk, seed_base + i, seatings, BOOTSTRAP)
            write_atomic(path, json.dumps(res))
            means = " ".join(f"{a['mean']:+.2f}" for a in res["agents"])
            log(f"{name} chunk {i + 1}/{n_chunks} in {time.time() - t:.0f} s: {means}")
        done.append(json.loads(path.read_text(encoding="utf-8")))
        hands = sum(c["hands"] for c in done)
        so_far = [sum(c["hands"] * c["agents"][k]["mean"] for c in done) / hands for k in range(4)]
        progress(
            f"{name}: {len(done)}/{n_chunks} chunks, {hands} hands, means "
            + " ".join(f"{m:+.2f}" for m in so_far)
        )
        if on_chunk is not None:
            on_chunk(merge(name, specs, seatings, done))
    report = merge(name, specs, seatings, done)
    write_atomic(MATCHES / f"{name}.json", json.dumps(report, indent=1))
    return report


# --- Stages -------------------------------------------------------------------------------


def stage_distill(quick: bool) -> None:
    from mahjong.rl.distill_cnn import DistillConfig, distill

    result = RUNS / "distill" / "result.json"
    if result.exists() and (MODELS / "cnn-distill.mjnn").exists() and not quick:
        log("distill: done already (runs/distill/result.json)")
        return
    if quick:
        cfg = DistillConfig(
            out=str(RUNS / "distill-quick"),
            models=str(RUNS / "distill-quick" / "models"),
            train_chunks=[0],
            holdout_chunks=[1],
            epochs=1,
            minutes=2,
        )
    else:
        # ~70 s per epoch over 1 M rows on the RTX 4060 Ti (measured on a 540 k slice).
        # On that slice held-out agreement levelled off near 0.90 after ~7 epochs while
        # the held-out value error rose after epoch 2, so 12 epochs over 1 M rows.
        cfg = DistillConfig(out=str(RUNS / "distill"), models=str(MODELS), epochs=12, minutes=110)
    progress("distill: training")
    log(f"distill: start ({cfg.epochs} epochs, at most {cfg.minutes} min)")
    res = distill(cfg, resume=True)
    log(
        f"distill: held-out top-1 {res['held_out']['top1_agreement']:.3f}, "
        f"int8 vs fp32 {res['int8_top1_agreement_with_fp32']:.4f}"
    )


def stage_evaluate(quick: bool, cnn: str) -> dict[str, Any]:
    name = "cnn-distill-quick" if quick else "cnn-distill"
    walls, per = (8, 4) if quick else (1000, 50)
    log(f"evaluate: {name}, {walls} walls x 4 seatings vs value")
    rep = run_match(name, [f"net:path={cnn}", "value", "value", "value"], walls, per, 7000, 4)
    a = rep["agents"][0]
    log(
        f"evaluate DONE: {a['mean']:+.2f} +/- {a['ci95']:.2f} (bootstrap [{a['boot_lo']:+.2f}, "
        f"{a['boot_hi']:+.2f}]), win {a['win_rate']:.1%}, deal-in {a['deal_in_rate']:.1%}"
    )
    return rep


def verdict(
    rep: dict[str, Any], names: list[str], distill_result: dict[str, Any] | None
) -> dict[str, Any]:
    """Applies Brian's rule and AC7 bar 3 to the netpimc match."""

    def pair(i: int, j: int) -> dict[str, float]:
        """Agent ``i`` minus agent ``j`` (the report stores ``a < b``: ``a - b``)."""
        for p in rep["pairs"]:
            if (p["a"], p["b"]) == (i, j):
                return {k: p[k] for k in ("mean", "ci95", "boot_lo", "boot_hi")}
            if (p["a"], p["b"]) == (j, i):
                return {
                    "mean": -p["mean"],
                    "ci95": p["ci95"],
                    "boot_lo": -p["boot_hi"],
                    "boot_hi": -p["boot_lo"],
                }
        raise KeyError((i, j))

    teacher = names.index("teacher")
    challengers: list[dict[str, Any]] = []
    for i, n in enumerate(names):
        if i == teacher:
            continue
        d = pair(i, teacher)
        challengers.append(
            {"challenger": n, "minus_teacher": d, "replaces_teacher": d["boot_lo"] > 0}
        )
    bare = names.index("cnn")
    search_vs_bare = [
        {
            "search": names[i],
            "minus_bare": pair(i, bare),
            "beats_bare": pair(i, bare)["boot_lo"] > 0,
        }
        for i in range(len(names))
        if names[i].startswith("cnn+")
    ]
    winners = [c for c in challengers if c["replaces_teacher"]]
    best = max(winners, key=lambda c: c["minus_teacher"]["mean"]) if winners else None
    return {
        "rule": "use teacher unless distilled model wins significantly (Brian, 2026-10-30): a "
        "challenger replaces the search teacher only if challenger minus teacher has a 95 % "
        "interval (wall-paired bootstrap, chunk-combined) above 0",
        "hands": rep["hands"],
        "walls": rep["deals"],
        "agents": [
            {
                "name": n,
                "spec": a["agent"],
                "mean": a["mean"],
                "boot_lo": a["boot_lo"],
                "boot_hi": a["boot_hi"],
            }
            for n, a in zip(names, rep["agents"], strict=True)
        ],
        "challengers": challengers,
        "decision": best["challenger"] if best else "teacher",
        "ac7_bar3_search_beats_bare": search_vs_bare,
        "ac7_bar3_met": any(s["beats_bare"] for s in search_vs_bare),
        "distill": None
        if distill_result is None
        else {
            "held_out_top1": distill_result["held_out"]["top1_agreement"],
            "int8_top1_agreement_with_fp32": distill_result["int8_top1_agreement_with_fp32"],
            "parameters": distill_result["parameters"],
            "int8_bytes": distill_result["int8_bytes"],
        },
    }


def verdict_markdown(v: dict[str, Any], evaluate_rep: dict[str, Any] | None) -> str:
    def iv(d: dict[str, float]) -> str:
        return f"{d['mean']:+.2f} [{d['boot_lo']:+.2f}, {d['boot_hi']:+.2f}]"

    lines = [
        f"Netpimc match: {v['walls']} walls x 24 seatings ({v['hands']} hands).",
        "",
        "| Agent | Points / hand [bootstrap 95 %] |",
        "|---|---|",
    ]
    for a in v["agents"]:
        lines.append(
            f"| {a['name']} (`{a['spec']}`) | {a['mean']:+.2f} "
            f"[{a['boot_lo']:+.2f}, {a['boot_hi']:+.2f}] |"
        )
    lines += [
        "",
        "| Challenger minus teacher | Paired difference [95 %] | Replaces the teacher? |",
        "|---|---|---|",
    ]
    for c in v["challengers"]:
        lines.append(
            f"| {c['challenger']} | {iv(c['minus_teacher'])} | "
            f"{'yes' if c['replaces_teacher'] else 'no'} |"
        )
    lines += [
        "",
        "| Search minus bare network (AC7 bar 3) | Paired difference [95 %] | Beats it? |",
        "|---|---|---|",
    ]
    for s in v["ac7_bar3_search_beats_bare"]:
        lines.append(
            f"| {s['search']} | {iv(s['minus_bare'])} | {'yes' if s['beats_bare'] else 'no'} |"
        )
    lines += ["", f"Brian's rule: **{v['decision']}** stays or becomes the search teacher."]
    if evaluate_rep is not None:
        a = evaluate_rep["agents"][0]
        lines.append(
            f"Bare int8 network vs three `value` bots ({evaluate_rep['hands']} hands): "
            f"{a['mean']:+.2f} ± {a['ci95']:.2f} (AC7 bar 1 needs ≥ 0 with half-width ≤ 2)."
        )
    if v["distill"]:
        d = v["distill"]
        lines.append(
            f"Distillation: held-out top-1 {d['held_out_top1']:.3f}; int8 vs fp32 top-1 "
            f"{d['int8_top1_agreement_with_fp32']:.4f} (AC7 bar 2 needs ≥ 0.98); "
            f"{d['parameters']:,} parameters; int8 {d['int8_bytes'] / 1e6:.2f} MB."
        )
    return "\n".join(lines) + "\n"


def stage_netpimc(quick: bool, cnn: str, walls: int, per_chunk: int) -> dict[str, Any]:
    agents = net_agents(cnn, samples=4 if quick else 48)
    names = [n for n, _ in agents]
    specs = [s for _, s in agents]
    name = "netpimc-quick" if quick else "netpimc"
    if quick:
        walls, per_chunk = 2, 1
        specs[3] = f"pimc:samples=4,candidates=4,belief={BELIEF},weights=1"
    log(f"netpimc: {walls} walls x 24 seatings, chunks of {per_chunk}: {specs}")
    dres_path = (RUNS / "distill-quick" if quick else RUNS / "distill") / "result.json"
    dres = json.loads(dres_path.read_text(encoding="utf-8")) if dres_path.exists() else None
    ev_path = MATCHES / ("cnn-distill-quick.json" if quick else "cnn-distill.json")
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None

    def partial(rep: dict[str, Any]) -> None:
        # The verdict so far, so a reader can look before the match ends.
        md = verdict_markdown(verdict(rep, names, dres), ev)
        write_atomic(MATCHES / f"{name}-partial-verdict.md", "PARTIAL (match running)\n\n" + md)

    rep = run_match(name, specs, walls, per_chunk, 7100, 24, on_chunk=partial)
    v = verdict(rep, names, dres)
    write_atomic(MATCHES / f"{name}-verdict.json", json.dumps(v, indent=1))
    write_atomic(MATCHES / f"{name}-verdict.md", verdict_markdown(v, ev))
    log(f"netpimc DONE: decision {v['decision']}; AC7 bar 3 met: {v['ac7_bar3_met']}")
    return v


def ppo_network(run: str, quick: bool) -> str | None:
    """The int8 file of a PPO run's final network (exported here), as a spec path, or
    ``None`` if the run has not finished. ``--quick`` plays the distilled network."""
    if quick:
        return CNN
    from mahjong.rl.cnn import SuitCNN, export

    final = RUNS / "ppo" / run / "final.pt"
    if not final.exists():
        return None
    path = RUNS / "ppo" / run / "final-int8.mjnn"
    if not path.exists():
        export(SuitCNN.load(final), path, int8=True)
    if run == "main" and not (MODELS / "cnn-ppo.mjnn").exists():
        shutil.copyfile(path, MODELS / "cnn-ppo.mjnn")
        shutil.copyfile(final, MODELS / "cnn-ppo.pt")
    return f"runs/ppo/{run}/final-int8.mjnn"


def pooled(rep: dict[str, Any], side_a: list[int], side_b: list[int]) -> dict[str, float]:
    """Side a minus side b, the mean over all pairs of agents across the sides. Each
    half-width is the mean of the pairs' half-widths: the pairs share walls, so the
    average of the differences cannot have a wider interval than that."""
    means, halves = [], []
    for i in side_a:
        for j in side_b:
            p = next(p for p in rep["pairs"] if {p["a"], p["b"]} == {i, j})
            means.append(p["mean"] if p["a"] == i else -p["mean"])
            halves.append((p["boot_hi"] - p["boot_lo"]) / 2)
    mean, half = sum(means) / len(means), sum(halves) / len(halves)
    return {"mean": mean, "lo": mean - half, "hi": mean + half}


def side_row(rep: dict[str, Any], agents: list[int]) -> dict[str, Any]:
    """Points and rates of one side of a match, pooled over its agents."""
    rows = [rep["agents"][i] for i in agents]
    hands = sum(r["hands"] for r in rows)
    wins = sum(r["wins"] for r in rows)
    pattern_wins: dict[str, int] = {}
    for r in rows:
        for k, v in r["pattern_wins"].items():
            pattern_wins[k] = pattern_wins.get(k, 0) + v
    return {
        "mean": sum(r["hands"] * r["mean"] for r in rows) / hands,
        "win_rate": wins / hands,
        "deal_in_rate": sum(r["deal_ins"] for r in rows) / hands,
        "mean_faan": sum(r["win_faan"] for r in rows) / wins if wins else 0.0,
        "pattern_shares": {k: v / wins for k, v in pattern_wins.items()} if wins else {},
    }


def share_drift(a: dict[str, float], b: dict[str, float], min_share: float = 0.02) -> list[str]:
    """Patterns (at least ``min_share`` of ``b``'s wins) whose share in ``a`` differs from
    ``b``'s by more than 30 % (AC8)."""
    return sorted(k for k, v in b.items() if v >= min_share and abs(a.get(k, 0.0) / v - 1) > 0.3)


def ppo_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "Plan Task 18: the AC8 matches (bare int8 networks; missing rows are not played yet).",
        "",
    ]
    if "vs_value" in report:
        v = report["vs_value"]
        lines.append(
            f"- Main network vs three `value` bots ({v['hands']} hands): {v['mean']:+.2f} ± "
            f"{v['ci95']:.2f}, bootstrap [{v['boot_lo']:+.2f}, {v['boot_hi']:+.2f}], win "
            f"{v['win_rate']:.1%}, deal-in {v['deal_in_rate']:.1%}, draws {v['draw_rate']:.1%} "
            "(bar: ≥ +5, interval excluding 0)."
        )
    for key, what in (
        ("ablate_5a", "without the paired reward"),
        ("ablate_1a", "without the target head"),
    ):
        if key in report:
            v = report[key]
            lines.append(
                f"- Ablation {what}, vs three `value` bots ({v['hands']} hands): {v['mean']:+.2f} "
                f"± {v['ci95']:.2f}."
            )
    if "cross_2v2" in report:
        c = report["cross_2v2"]
        lines.append(
            f"- Seeds 2 + 2 (24 seatings): main minus seed 2 {c['mean']:+.2f} "
            f"[{c['lo']:+.2f}, {c['hi']:+.2f}]."
        )
    for key, what in (
        ("cross_1v3_seed2", "seed 2 alone vs three main"),
        ("cross_1v3_main", "main alone vs three seed 2"),
    ):
        if key in report:
            v = report[key]
            lines.append(f"- 1 + 3, {what}: {v['mean']:+.2f} ± {v['ci95']:.2f}.")
    if "vs_pimc128" in report:
        h = report["vs_pimc128"]
        lines.append(
            f"- Main minus belief-PIMC-128, head to head ({h['hands']} hands, {h['walls']} walls, "
            f"24 seatings): {h['mean']:+.2f} [{h['lo']:+.2f}, {h['hi']:+.2f}] (bar: above 0). "
            f"Win shares more than 30 % from PIMC's: {', '.join(h['drift']) or 'none'}."
        )
    return "\n".join(lines) + "\n"


def stage_ppo_eval(quick: bool, h2h_walls: int) -> dict[str, Any]:
    """The AC8 matches of plan Task 18. Network-only matches first (minutes); the search
    match last, because it takes days and reports after every chunk."""
    main, seed2 = ppo_network("main", quick), ppo_network("seed2", quick)
    a5, a1 = ppo_network("ablate-5a", quick), ppo_network("ablate-1a", quick)
    if main is None:
        raise SystemExit("ppo-eval: runs/ppo/main/final.pt is missing")
    walls, per = (2, 1) if quick else (1000, 50)
    tag = "-quick" if quick else ""
    report: dict[str, Any] = {}

    def save() -> None:
        write_atomic(MATCHES / f"ppo-eval{tag}.json", json.dumps(report, indent=1))
        write_atomic(MATCHES / f"ppo-eval{tag}.md", ppo_report_markdown(report))

    def one_seat(rep: dict[str, Any]) -> dict[str, Any]:
        keys = ("hands", "mean", "ci95", "boot_lo", "boot_hi", "win_rate", "deal_in_rate")
        return {
            **{k: rep["agents"][0][k] for k in keys},
            "draw_rate": rep["draw_rate"],
            "mean_faan": rep["agents"][0]["mean_faan"],
            "pattern_shares": rep["agents"][0]["pattern_shares"],
        }

    def bare(path: str) -> str:
        return f"net:path={path}"

    field = ["value", "value", "value"]
    rep = run_match(f"ppo-vs-value{tag}", [bare(main), *field], walls, per, 7300, 4)
    report["vs_value"] = one_seat(rep)
    save()
    # The ablations play the first 400 walls of the main network's match (same seeds).
    for key, net in (("ablate_5a", a5), ("ablate_1a", a1)):
        if net is not None:
            name = f"ppo-{key.replace('_', '-')}{tag}"
            rep = run_match(name, [bare(net), *field], walls * 2 // 5, per, 7300, 4)
            report[key] = one_seat(rep)
            save()
    if seed2 is not None:
        a, b = bare(main), bare(seed2)
        rep = run_match(f"ppo-cross-2v2{tag}", [a, b, a, b], walls, per, 7400, 24)
        report["cross_2v2"] = pooled(rep, [0, 2], [1, 3])
        rep = run_match(f"ppo-cross-seed2{tag}", [b, a, a, a], walls, per, 7500, 4)
        report["cross_1v3_seed2"] = one_seat(rep)
        rep = run_match(f"ppo-cross-main{tag}", [a, b, b, b], walls, per, 7600, 4)
        report["cross_1v3_main"] = one_seat(rep)
        save()
    samples = 4 if quick else 128
    pimc = f"pimc:samples={samples},candidates=4,belief={BELIEF},weights=1"
    specs = [bare(main), pimc, bare(main), pimc]

    def head_to_head(r: dict[str, Any]) -> None:
        net_side, search_side = side_row(r, [0, 2]), side_row(r, [1, 3])
        report["vs_pimc128"] = {
            **pooled(r, [0, 2], [1, 3]),
            "hands": r["hands"],
            "walls": r["deals"],
            "network": net_side,
            "pimc": search_side,
            "drift": share_drift(net_side["pattern_shares"], search_side["pattern_shares"]),
        }
        save()

    walls2 = 2 if quick else h2h_walls
    run_match(f"ppo-vs-pimc128{tag}", specs, walls2, per, 7700, 24, on_chunk=head_to_head)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--stage", choices=["distill", "evaluate", "netpimc", "ppo-eval", "all"], default="all"
    )
    ap.add_argument("--quick", action="store_true", help="minutes-long smoke run on 2 chunks")
    ap.add_argument(
        "--cnn", default=None, help="network for evaluate/netpimc (default models/cnn-distill.mjnn)"
    )
    # 400 walls x 24 seatings = 9,600 hands: about 6 s per hand with all four agents on
    # 8 threads (measured 2026-10-30), so about 16 h; the plan's 1,000 walls would take ~40 h.
    ap.add_argument("--netpimc-walls", type=int, default=400)
    ap.add_argument("--netpimc-chunk", type=int, default=20)
    # Plan: 1,000 walls. Chunks of 50; the report is rewritten after each chunk.
    ap.add_argument("--h2h-walls", type=int, default=1000)
    a = ap.parse_args(argv)
    no_power_throttling()
    os.chdir(REPO)
    cnn = a.cnn or (str(Path("runs/distill-quick/models/cnn-distill.mjnn")) if a.quick else CNN)
    cnn = cnn.replace("\\", "/")
    log(f"start: stage {a.stage}, RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}")
    if a.stage in ("distill", "all"):
        stage_distill(a.quick)
    if a.stage in ("evaluate", "all"):
        stage_evaluate(a.quick, cnn)
    if a.stage in ("netpimc", "all"):
        stage_netpimc(a.quick, cnn, a.netpimc_walls, a.netpimc_chunk)
    if a.stage == "ppo-eval":
        stage_ppo_eval(a.quick, a.h2h_walls)
    progress("all stages done")
    log("all stages done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
