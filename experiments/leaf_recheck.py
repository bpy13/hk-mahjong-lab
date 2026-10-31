"""Short re-check of `netpimc ... leaf=value,depth=6` after the leaf fix (plan Task 16).

One seat against three `value` bots, 4 seatings, on the same walls for the old and the
new rule. The old rule lives only in the frozen phase-4 extension, so each run picks its
extension through PYTHONPATH (`runs\\phase4\\pkg` for the old rule, `python` for the new).

    python experiments/leaf_recheck.py <label>     writes runs/leaf-recheck/<label>.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SPEC = (
    "netpimc:path=models/cnn-distill.mjnn,samples=48,candidates=4,"
    "belief=models/belief.mjnn,weights=1,leaf=value,depth=6"
)
WALLS, CHUNKS, SEED = 40, 5, 7200


def main() -> None:
    from phase4_build import no_power_throttling

    no_power_throttling()
    from mahjong._core import tournament

    out = REPO / "runs" / "leaf-recheck"
    out.mkdir(exist_ok=True)
    label = sys.argv[1]
    rows = []
    for i in range(CHUNKS):
        res = tournament([SPEC, "value", "value", "value"], WALLS, SEED + i, 4, 2000)
        a = res["agents"][0]
        rows.append(
            {k: a[k] for k in ("hands", "mean", "ci95", "win_rate", "deal_in_rate", "mean_faan")}
        )
        print(label, i, rows[-1], flush=True)
        (out / f"{label}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
