"""Plan 2026-11-09 Task 4 (AC3): the teacher spec with the old and the fixed rollout bot.

The teacher is belief-PIMC-64 with anchor `value` and rollout `fast` (Task 7's spec,
`runs/labels-job/teacher-used.txt`). Tasks 1-2 changed the `fast` bot that plays the
rollouts, so the spec string is the same and only the engine differs:

* `old`: the extension built from 75dae32 (before the fix), unpacked in
  `runs/review-fix/pkg-old/`;
* `new`: the extension built from the fix, in `runs/review-fix/pkg-new/`.

Each variant plays 1,000 walls x 4 seatings against three `value` bots, on Task 7's walls
(20 chunks of 50 walls, seeds 7000..7019), so the two are paired by chunk. One process per
variant (two `mahjong` packages cannot share a process):

    python run_matches.py --variant old     # PYTHONPATH=runs/review-fix/pkg-old
    python run_matches.py --variant new     # PYTHONPATH=runs/review-fix/pkg-new
    python run_matches.py --compare         # after both: runs/review-fix/compare.json

Finished chunks go to `runs/review-fix/chunks/<variant>-NN.json`; a rerun skips them.
Progress: `runs/review-fix/<variant>.progress`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "review-fix"
CHUNKS = OUT / "chunks"

TEACHER = "pimc:samples=64,candidates=4,belief=models/belief.mjnn,weights=1"
DEALS_PER_CHUNK = 50
N_CHUNKS = 20
SEED_BASE = 7000  # Task 7's walls
SEATINGS = 4
BOOTSTRAP = 2000
T_19 = 2.093  # two-sided 95 % t quantile, 19 degrees of freedom


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%dT%H:%M:%S} {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def no_power_throttling() -> None:
    # Windows throttles headless processes to ~0.15 core each (EcoQoS); opt out.
    import ctypes
    from ctypes import wintypes

    class State(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    st = State(1, 1, 0)
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))


def play(variant: str) -> int:
    no_power_throttling()
    CHUNKS.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)  # the spec's model path is relative to the repo
    import mahjong
    from mahjong._core import tournament

    pkg = (OUT / f"pkg-{variant}" / "mahjong").resolve()
    if Path(mahjong.__file__).resolve().parent != pkg:
        log(f"{variant}: refusing to run: mahjong imported from {mahjong.__file__}, not {pkg}")
        return 2
    log(f"{variant}: start; RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}")
    progress = OUT / f"{variant}.progress"
    done = []
    for i in range(N_CHUNKS):
        path = CHUNKS / f"{variant}-{i:02d}.json"
        if not path.exists():
            t = time.time()
            res = tournament([TEACHER, "value", "value", "value"], DEALS_PER_CHUNK, SEED_BASE + i, SEATINGS, BOOTSTRAP)
            write_atomic(path, json.dumps(res))
            log(f"{variant} chunk {i + 1}/{N_CHUNKS} in {time.time() - t:.0f} s: teacher {res['agents'][0]['mean']:+.2f}")
        done.append(json.loads(path.read_text(encoding="utf-8")))
        hands = sum(c["hands"] for c in done)
        mean = sum(c["hands"] * c["agents"][0]["mean"] for c in done) / hands
        write_atomic(progress, f"{len(done)}/{N_CHUNKS} chunks, {hands} hands, teacher {mean:+.2f} per hand ({datetime.now():%Y-%m-%d %H:%M})\n")
    log(f"{variant}: done")
    return 0


def summary(chunks: list[dict]) -> dict:
    """Hands-weighted totals over the chunks; the interval combines the chunks' own."""
    hands = sum(c["hands"] for c in chunks)
    a = [c["agents"][0] for c in chunks]
    mean = sum(c["hands"] * x["mean"] for c, x in zip(chunks, a)) / hands
    se = math.sqrt(sum((c["hands"] * x["ci95"] / 1.96) ** 2 for c, x in zip(chunks, a))) / hands
    rate = lambda key: sum(x[key] for x in a) / hands  # noqa: E731
    return {
        "hands": hands,
        "mean": mean,
        "ci95": 1.96 * se,
        "win_rate": rate("wins"),
        "self_drawn_rate": rate("self_drawn"),
        "deal_in_rate": rate("deal_ins"),
        "draw_rate": sum(c["draws"] for c in chunks) / hands,
        "mean_faan": sum(x["win_faan"] for x in a) / max(1, sum(x["wins"] for x in a)),
    }


def compare() -> int:
    loaded = {}
    for v in ("old", "new"):
        paths = [CHUNKS / f"{v}-{i:02d}.json" for i in range(N_CHUNKS)]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            log(f"compare: {v} is missing {len(missing)} chunks; run it first")
            return 1
        loaded[v] = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    diffs = [n["agents"][0]["mean"] - o["agents"][0]["mean"] for o, n in zip(loaded["old"], loaded["new"])]
    d = sum(diffs) / N_CHUNKS
    sd = math.sqrt(sum((x - d) ** 2 for x in diffs) / (N_CHUNKS - 1))
    half = T_19 * sd / math.sqrt(N_CHUNKS)
    out = {
        "teacher": TEACHER,
        "walls": f"{N_CHUNKS} chunks x {DEALS_PER_CHUNK} walls, seeds {SEED_BASE}..{SEED_BASE + N_CHUNKS - 1}, {SEATINGS} seatings",
        "opponents": "three value bots",
        "old": summary(loaded["old"]),
        "new": summary(loaded["new"]),
        "new_minus_old": {"mean": d, "ci95": half, "lo": d - half, "hi": d + half, "paired_by": "chunk (t, 19 df)"},
        "chunk_differences": diffs,
        "ac3_pass": d - half > -0.5,
    }
    write_atomic(OUT / "compare.json", json.dumps(out, indent=1))
    log(f"compare: new - old {d:+.2f} [{d - half:+.2f}, {d + half:+.2f}] per hand; AC3 {'pass' if out['ac3_pass'] else 'FAIL'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["old", "new"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.compare:
        return compare()
    if not args.variant:
        ap.error("--variant or --compare")
    return play(args.variant)


if __name__ == "__main__":
    sys.exit(main())
