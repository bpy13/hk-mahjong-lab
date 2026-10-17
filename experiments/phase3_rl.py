"""Phase 3 study: learning to play, compared with the bots and the solver.

Stages (each skipped if its model already exists, so the script can resume):

1. distill   imitate the `value` bot (no human logs exist for HK rules)
2. dmc       Deep Monte Carlo self-play from the distilled network
3. dmc0      the same from scratch (what is the warm start worth?)
4. ppo       PPO with oracle guiding from the distilled network
5. ppo0      PPO without oracle guiding (what is oracle guiding worth?)
6. eval      duplicate evaluation of every network, and network + search,
             against three `value` bots

    python experiments/phase3_rl.py [--quick] [--scale 1.0]

Models go to models/, logs and results to runs/phase3/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mahjong.rl.distill import distill
from mahjong.rl.dmc import dmc
from mahjong.rl.evaluate import evaluate
from mahjong.rl.model import Net
from mahjong.rl.ppo import ppo

MODELS = Path("models")
RUNS = Path("runs/phase3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="minutes-long smoke run")
    parser.add_argument("--scale", type=float, default=1.0, help="multiply all time budgets")
    args = parser.parse_args()
    torch.set_num_threads(4)
    q = 0.01 if args.quick else args.scale
    tag = "-quick" if args.quick else ""
    RUNS.mkdir(parents=True, exist_ok=True)
    m = lambda name: MODELS / f"{name}{tag}.pt"  # noqa: E731

    if not m("distill").exists():
        distill(
            m("distill"),
            hands=max(40, int(48_000 * q)),
            chunk=600 if not args.quick else 20,
            log=RUNS / f"distill{tag}.json",
        )
        Net.load(m("distill")).export(m("distill").with_suffix(".mjnn"), "pi")
    if not m("dmc").exists():
        dmc(m("dmc"), init=m("distill"), minutes=90 * q, log=RUNS / f"dmc{tag}.json")
    if not m("dmc0").exists():
        dmc(m("dmc0"), init=None, minutes=30 * q, seed=1, log=RUNS / f"dmc0{tag}.json")
    if not m("ppo").exists():
        ppo(m("ppo"), init=m("distill"), minutes=90 * q, log=RUNS / f"ppo{tag}.json")
    if not m("ppo0").exists():
        ppo(
            m("ppo0"),
            init=m("distill"),
            minutes=40 * q,
            oracle_fraction=0.0,
            seed=1,
            log=RUNS / f"ppo0{tag}.json",
        )

    deals = 6 if args.quick else 250
    results: dict[str, Any] = {}
    for name in ["distill", "dmc", "dmc0", "ppo", "ppo0"]:
        results[name] = evaluate(f"net:path={m(name).with_suffix('.mjnn')}", deals, seed=2026)
        print("eval", name, results[name], flush=True)
    best = max(["dmc", "ppo"], key=lambda k: results[k]["mean"])
    path = m(best).with_suffix(".mjnn")
    for rollout, samples, n in [("fast", 32, 60), ("net", 16, 40)]:
        key = f"{best}+pimc({rollout})"
        spec = f"netpimc:path={path},samples={samples},candidates=4,rollout={rollout}"
        results[key] = evaluate(spec, 2 if args.quick else n, seed=2026)
        print("eval", key, results[key], flush=True)
    (RUNS / f"results{tag}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
