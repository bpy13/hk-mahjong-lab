"""Plan 2026-11-09 Task 16: tabulate runs/matches/passed-*.json into passed-summary.json.

The tournament report already holds the wall-paired bootstrap interval of every pair of seats,
so the head-to-head files need no new bootstrap: their `pairs` row `full - other` is the answer.
Solo files give each variant's points per hand against three `value` bots. The variants there
ran on the same walls, but the report keeps no per-wall scores, so the solo difference is only
an unpaired normal estimate (square root of the summed squared 95 % half-widths); trust the
head-to-head rows for the verdict.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

MATCHES = Path(__file__).resolve().parents[2] / "runs" / "matches"


def load(name: str) -> dict | None:
    path = MATCHES / f"passed-{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def row(a: dict) -> dict:
    keys = ("mean", "ci95", "boot_lo", "boot_hi", "win_rate", "deal_in_rate", "mean_faan")
    return {k: a[k] for k in keys}


def main() -> None:
    out: dict = {"solo": {}, "solo_differences": {}, "head_to_head": {}, "teacher_sanity": {}}
    for v in ("full", "off", "nopassed", "norisk"):
        if m := load(f"solo-{v}"):
            out["solo"][v] = row(m["agents"][0])
    for v in ("nopassed", "norisk", "off"):
        if "full" in out["solo"] and v in out["solo"]:
            f, o = out["solo"]["full"], out["solo"][v]
            d, half = f["mean"] - o["mean"], math.hypot(f["ci95"], o["ci95"])
            out["solo_differences"][f"full - {v}"] = {
                "mean": d,
                "lo": d - half,
                "hi": d + half,
                "paired": False,
            }
    for v in ("off", "nopassed", "norisk"):
        if m := load(f"h2h-full-vs-{v}"):
            full, other = m["agents"][0]["agent"], m["agents"][2]["agent"]
            pair = next(p for p in m["pairs"] if {p["agent_a"], p["agent_b"]} == {full, other})
            sign = 1 if pair["agent_a"] == full else -1
            lo, hi = sorted((sign * pair["boot_lo"], sign * pair["boot_hi"]))
            out["head_to_head"][f"full - {v}"] = {
                "mean": sign * pair["mean"],
                "boot_lo": lo,
                "boot_hi": hi,
                "walls": m["deals"],
                "paired": "by wall, bootstrap",
            }
    for tag in ("on", "off"):
        if m := load(f"teacher-rule-{tag}"):
            out["teacher_sanity"][f"rule_{tag}"] = {**row(m["agents"][0]), "walls": m["deals"]}
    (MATCHES / "passed-summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
