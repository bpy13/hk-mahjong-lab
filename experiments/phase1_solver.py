"""Phase 1 study: bots and the PIMC solver under Hong Kong rules.

Every comparison is a duplicate tournament (each wall played four times with the
agents rotated through the seats). Results go to runs/phase1/results.json and are
summarized in docs/study-results/phase1-solver.md.

    python experiments/phase1_solver.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from mahjong._core import tournament
from mahjong.hk import Rules

OUT = Path("runs/phase1")


def run(agents: list[str], deals: int, seed: int, rules: Rules) -> dict[str, Any]:
    start = time.time()
    res = tournament(agents, deals, seed, **rules.kwargs())
    res["seconds"] = time.time() - start
    res["rules"] = rules.kwargs()
    res["deals"] = deals
    return res


def row(res: dict[str, Any], i: int) -> dict[str, float]:
    a = res["agents"][i]
    n = a["hands"]
    return {
        "agent": a["agent"],
        "mean": a["mean"],
        "ci95": a["ci95"],
        "win": a["wins"] / n,
        "self_drawn": a["self_drawn"] / n,
        "deal_in": a["deal_ins"] / n,
        "avg_faan": a["win_faan"] / max(a["wins"], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny run to check the script")
    args = parser.parse_args()
    scale = 0.05 if args.quick else 1.0
    deals = lambda n: max(2, int(n * scale))  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    default = Rules()

    # 1. Bot ladder: each candidate against three value bots.
    ladder = []
    for cand in ["random", "efficiency", "closed", "fast", "value", "defensive"]:
        res = run([cand, "value", "value", "value"], deals(400), 11, default)
        ladder.append(row(res, 0))
        print("ladder", ladder[-1], flush=True)
    results["ladder"] = ladder

    # 2. Solver strength against thinking time.
    solver = []
    for samples in [16, 32, 64, 128]:
        spec = f"pimc:samples={samples},candidates=4"
        res = run([spec, "value", "value", "value"], deals(80 if samples < 128 else 50), 12, default)
        r = row(res, 0)
        r["samples"] = samples
        r["seconds"] = res["seconds"]
        solver.append(r)
        print("solver", r, flush=True)
    results["solver"] = solver

    # 3. The faan minimum: efficiency vs value under different minimums.
    minimum = []
    for m in [0, 1, 3]:
        rules = Rules(min_faan=m)
        res = run(["efficiency", "value", "value", "value"], deals(300), 13, rules)
        selfplay = run(["value"] * 4, deals(150), 14, rules)
        minimum.append(
            {
                "min_faan": m,
                "efficiency": row(res, 0),
                "draw_rate": selfplay["draws"] / selfplay["hands"],
                "avg_faan": sum(a["win_faan"] for a in selfplay["agents"])
                / max(1, sum(a["wins"] for a in selfplay["agents"])),
            }
        )
        print("minimum", minimum[-1], flush=True)
    results["minimum"] = minimum

    # 4. Seats and flowers: value self-play.
    seats = []
    for flowers in [True, False]:
        res = run(["value"] * 4, deals(400), 15, Rules(flowers=flowers))
        n = res["hands"]
        seats.append(
            {
                "flowers": flowers,
                "seat_points": [p / n for p in res["seat_points"]],
                "draw_rate": res["draws"] / n,
                "win_rate": sum(a["wins"] for a in res["agents"]) / n,
                "avg_faan": sum(a["win_faan"] for a in res["agents"])
                / max(1, sum(a["wins"] for a in res["agents"])),
                "self_drawn_share": sum(a["self_drawn"] for a in res["agents"])
                / max(1, sum(a["wins"] for a in res["agents"])),
            }
        )
        print("seats", seats[-1], flush=True)
    results["seats"] = seats

    path = OUT / ("results-quick.json" if args.quick else "results.json")
    path.write_text(json.dumps(results, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
