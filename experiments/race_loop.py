"""The race bot's self-play loop (plan 2026-11-09 Task 9, ``docs/race-bot.md`` §8).

Generation ``g`` has a race bot ``R_g``: ``R_0`` is v0 on the plain belief model
(``race:belief=models/belief.mjnn``, v0's constants); ``R_g`` for ``g ≥ 1`` reads its
own race model (``state`` and ``mode`` heads) and the ``a``, ``b``, ``q`` tables fitted
from its logs. One generation, ``g = 1, 2, …``:

1. **play** (``race_log`` example, CPU): ``--hands`` hands from the league of §8.1 with
   ``R_{g-1}`` as ``--main`` and every older accepted generation as ``--old`` (shares
   60 / 20 / 20; the fixed field is value, defensive, fast, efficiency, closed), plus a
   hold-out log of ``--holdout-hands`` with another seed;
2. **train** (GPU): ``python -m mahjong.belief.train --race``, warm-started from
   ``R_{g-1}``'s model (``models/belief.pt`` for ``g = 1``), ``--epochs`` epochs;
3. **fit**: ``python -m mahjong.belief.race_fit``: the ``a``, ``b``, ``q`` tables, the
   heads' calibration per wall bucket and §8.2's hazard check, from the logs;
4. **gate** (CPU, duplicate matches on Task 7's walls, ``--chunks`` chunks of
   ``--walls`` walls, seeds 7000…, 4 seatings): ``R_g`` against three ``R_{g-1}`` (the
   **gain**), against three ``value``, and against the teacher (``R_g``, teacher, two
   ``value``);
5. **report**: ``<root>/g<g>/result.json``, and ``<root>/report.json`` / ``report.md``
   with one row per generation.

Stop rule (§8.3): after at least ``--min-gens`` generations, stop when the last two
gains are inside ±1.2 and the score against ``value`` has risen by at most 1.2 over the
last three generations; at ``--max-gens``; or when a generation regresses (gain below
−1.2 with its interval below 0) twice: the first regression drops that generation and
trains again from the same logs with another seed. The **best generation** is the one
with the highest score against ``value`` (ties within 1.2 go to the later one); its
model is copied to ``models/race-belief.mjnn``.

Everything is resumable: logs resume by chunk, training from its newest epoch
checkpoint, matches by chunk file, and finished steps are skipped. Progress:
``<root>/progress.txt`` (rewritten after every step and chunk) and ``<root>/run.log``.

::

    python experiments/race_loop.py --root runs/race --threads 8 --device cuda
    python experiments/race_loop.py --root runs/race-smoke/loop --smoke   # minutes, CPU
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("HK_LAB_REPO", Path(__file__).resolve().parents[1]))
TEACHER = "pimc:samples=64,candidates=4,belief=models/belief.mjnn,weights=1"
R0 = "race:belief=models/belief.mjnn"
BASE_PT = "models/belief.pt"
SEED_BASE = 7000  # Task 7's walls (docs/race-bot.md §8.1 step 4)
NOISE = 1.2  # points per hand: a 4,000-hand duplicate match's 95 % half-width (§8.3)
Z95 = 1.96

ROOT: Path = REPO / "runs" / "race"


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%dT%H:%M:%S} {msg}"
    print(line, flush=True)
    with (ROOT / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def progress(msg: str) -> None:
    write_atomic(ROOT / "progress.txt", f"{msg} ({datetime.now():%Y-%m-%d %H:%M})\n")


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def no_power_throttling() -> None:
    """Windows throttles headless processes to ~0.15 core each (EcoQoS); opt out."""
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


def run(cmd: list[str], what: str) -> None:
    """Runs a step's command; its output goes to ``run.out``."""
    log(f"{what}: {' '.join(cmd)}")
    with (ROOT / "run.out").open("a", encoding="utf-8") as out:
        code = subprocess.call(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT)
    if code != 0:
        raise SystemExit(f"{what} failed with exit code {code} (see {ROOT / 'run.out'})")


def rel(path: Path) -> str:
    """A path relative to the repo with forward slashes (spec strings split on ':')."""
    return path.resolve().relative_to(REPO.resolve()).as_posix()


# ----- One generation's steps. -----


class Gen:
    """Paths of generation ``k`` (attempt ``a``: 0, or 1 after a regression)."""

    def __init__(self, k: int, attempt: int = 0) -> None:
        self.k = k
        self.attempt = attempt
        self.dir = ROOT / f"g{k}"
        self.logs = self.dir / "logs"
        self.holdout = self.dir / "holdout"
        self.run = self.dir / f"a{attempt}"
        self.pt = self.run / "race-belief.pt"
        self.mjnn = self.run / "race-belief.mjnn"
        self.fit = self.run / "fit.json"
        self.result = self.run / "result.json"


def spec_of(gen: Gen) -> str:
    fit = read_json(gen.fit)
    return f"race:belief={rel(gen.mjnn)},{fit['spec']}"


def play(gen: Gen, main: str, old: list[str], cfg: dict[str, Any]) -> None:
    exe = cfg["race_log"]
    league = ["--main", main]
    for o in old:
        league += ["--old", o]
    for out, hands, seed in (
        (gen.logs, cfg["hands"], 1000 + gen.k),
        (gen.holdout, cfg["holdout_hands"], 50_000 + gen.k),
    ):
        meta = out / "meta.json"
        if meta.exists() and read_json(meta).get("complete"):
            continue
        progress(f"g{gen.k}: play {rel(out)} ({hands} hands)")
        run(
            [
                exe,
                "--hands",
                str(hands),
                "--out",
                str(out),
                "--seed",
                str(seed),
                "--threads",
                str(cfg["threads"]),
                "--chunk",
                str(cfg["chunk"]),
                *league,
            ],
            f"g{gen.k} play",
        )


def train(gen: Gen, init: Path, cfg: dict[str, Any]) -> None:
    done = gen.run / "train.done"
    if done.exists() and gen.mjnn.exists():
        return
    progress(f"g{gen.k}: train (attempt {gen.attempt})")
    gen.run.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "mahjong.belief.train",
        "--logs",
        str(gen.logs),
        "--holdout",
        str(gen.holdout),
        "--out",
        str(gen.pt),
        "--race",
        "--init",
        str(init),
        "--epochs",
        str(cfg["epochs"]),
        "--ckpt-dir",
        str(gen.run / "ckpt"),
        "--resume",
        "--device",
        cfg["device"],
        "--threads",
        str(cfg["threads"]),
        "--seed",
        str(gen.attempt),
    ]
    run(cmd, f"g{gen.k} train")
    write_atomic(done, "done\n")


def fit(gen: Gen, cfg: dict[str, Any]) -> None:
    if gen.fit.exists():
        return
    progress(f"g{gen.k}: fit tables and calibration")
    run(
        [
            sys.executable,
            "-m",
            "mahjong.belief.race_fit",
            "--logs",
            str(gen.logs),
            "--holdout",
            str(gen.holdout),
            "--model",
            str(gen.pt),
            "--min-n",
            str(cfg["min_n"]),
            "--out",
            str(gen.fit),
        ],
        f"g{gen.k} fit",
    )


def match(name: str, agents: list[str], out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """A duplicate match in chunks (resumable), summed: agent 0's points per hand with
    a 95 % interval (the chunks' wall-paired bootstrap intervals combined), rates, and
    agent 0 minus agent 1 (paired by wall)."""
    from mahjong._core import tournament

    chunks: list[dict[str, Any]] = []
    for i in range(cfg["chunks"]):
        path = out_dir / f"{name}-{i:02d}.json"
        if not path.exists():
            progress(f"{rel(out_dir)}: {name} chunk {i + 1}/{cfg['chunks']}")
            t = time.time()
            res = tournament(agents, cfg["walls"], SEED_BASE + i, 4, 2000)
            write_atomic(path, json.dumps(res))
            log(
                f"{rel(out_dir)} {name} chunk {i + 1}/{cfg['chunks']} in {time.time() - t:.0f} s:"
                f" {res['agents'][0]['mean']:+.2f}"
            )
        chunks.append(read_json(path))
    return summarize(name, agents, chunks)


def _se(lo: float, hi: float) -> float:
    return (hi - lo) / (2 * Z95)


def summarize(name: str, agents: list[str], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    hands = sum(c["hands"] for c in chunks)
    a = [c["agents"][0] for c in chunks]
    w = [c["hands"] / hands for c in chunks]
    mean = sum(wi * x["mean"] for wi, x in zip(w, a, strict=True))
    se = math.sqrt(
        sum((wi * _se(x["boot_lo"], x["boot_hi"])) ** 2 for wi, x in zip(w, a, strict=True))
    )
    pair = [next(p for p in c["pairs"] if p["a"] == 0 and p["b"] == 1) for c in chunks]
    d = sum(wi * p["mean"] for wi, p in zip(w, pair, strict=True))
    dse = math.sqrt(
        sum((wi * _se(p["boot_lo"], p["boot_hi"])) ** 2 for wi, p in zip(w, pair, strict=True))
    )
    wins = sum(x["wins"] for x in a)
    return {
        "match": name,
        "agents": agents,
        "hands": hands,
        "mean": mean,
        "ci95": Z95 * se,
        "lo": mean - Z95 * se,
        "hi": mean + Z95 * se,
        "win_rate": wins / hands,
        "self_drawn_rate": sum(x["self_drawn"] for x in a) / hands,
        "deal_in_rate": sum(x["deal_ins"] for x in a) / hands,
        "draw_rate": sum(c["draws"] for c in chunks) / hands,
        "mean_faan": sum(x["win_faan"] for x in a) / max(1, wins),
        "minus_agent1": {"mean": d, "ci95": Z95 * dse, "lo": d - Z95 * dse, "hi": d + Z95 * dse},
    }


def regressed(gain: dict[str, Any]) -> bool:
    """§8.3: loses to its predecessor by more than the noise, paired interval below 0."""
    return bool(gain["mean"] < -NOISE and gain["hi"] < 0)


# ----- The loop. -----


def baseline(cfg: dict[str, Any]) -> dict[str, Any]:
    """Generation 0: v0 on the plain belief model, measured against three ``value``."""
    path = ROOT / "g0" / "result.json"
    if path.exists():
        return dict(read_json(path))
    value = match("value", [R0, "value", "value", "value"], ROOT / "g0" / "matches", cfg)
    res = {"generation": 0, "spec": R0, "value": value}
    write_atomic(path, json.dumps(res, indent=1))
    return res


def stop_reason(accepted: list[dict[str, Any]], cfg: dict[str, Any]) -> str | None:
    """§8.3 on the accepted generations (``accepted[0]`` is generation 0)."""
    learned = accepted[1:]
    if len(learned) >= cfg["max_gens"]:
        return f"hard cap of {cfg['max_gens']} generations"
    if len(learned) < max(cfg["min_gens"], 2):
        return None
    gains_small = all(abs(g["gain"]["mean"]) < NOISE for g in learned[-2:])
    values = [g["value"]["mean"] for g in accepted]
    rise = values[-1] - values[-4] if len(values) >= 4 else values[-1] - values[0]
    if gains_small and rise <= NOISE:
        return (
            f"converged: last two gains inside ±{NOISE} and the score against value rose"
            f" {rise:+.2f} over the last three generations"
        )
    return None


def best_generation(accepted: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The learned generation with the highest score against ``value``; every generation
    within the noise of the top counts as tied, and the latest of those wins (§8.3)."""
    learned = accepted[1:]
    if not learned:
        return None
    top = max(g["value"]["mean"] for g in learned)
    return [g for g in learned if g["value"]["mean"] >= top - NOISE][-1]


def report(accepted: list[dict[str, Any]], dropped: list[dict[str, Any]], stop: str | None) -> None:
    best = best_generation(accepted)
    out = {
        "updated": f"{datetime.now():%Y-%m-%dT%H:%M:%S}",
        "stopped": stop,
        "best": best["generation"] if best else None,
        "generations": accepted,
        "dropped": dropped,
    }
    write_atomic(ROOT / "report.json", json.dumps(out, indent=1))

    def cell(m: dict[str, Any] | None) -> str:
        return "—" if not m else f"{m['mean']:+.2f} [{m['lo']:+.2f}, {m['hi']:+.2f}]"

    lines = [
        "# Race bot self-play loop (generated by experiments/race_loop.py)",
        "",
        f"Stopped: {stop or 'running'}. Best generation: {out['best']}.",
        "",
        "| Gen | Gain vs previous | vs 3 value | vs teacher (race − teacher) "
        "| Win | Deal-in | Draw | Mean faan |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g in accepted:
        v = g["value"]
        t = g.get("teacher")
        lines.append(
            f"| {g['generation']} | {cell(g.get('gain'))} | {cell(v)} | "
            f"{cell(t['minus_agent1']) if t else '—'} | {v['win_rate']:.3f} | "
            f"{v['deal_in_rate']:.3f} | {v['draw_rate']:.3f} | {v['mean_faan']:.2f} |"
        )
    for g in dropped:
        lines.append(f"| {g['generation']} (dropped) | {cell(g.get('gain'))} | — | — | | | | |")
    write_atomic(ROOT / "report.md", "\n".join(lines) + "\n")


def generation(
    k: int, accepted: list[dict[str, Any]], cfg: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Plays, trains, fits and gates generation ``k``; returns the accepted result (or
    None after two regressions) and the dropped attempts."""
    prev = accepted[-1]
    olds = [g["spec"] for g in accepted[:-1]]
    play(Gen(k), prev["spec"], olds, cfg)
    init = REPO / prev["pt"] if prev["generation"] > 0 else REPO / BASE_PT
    dropped = []
    for attempt in (0, 1):
        gen = Gen(k, attempt)
        if gen.result.exists():
            res = read_json(gen.result)
        else:
            train(gen, init, cfg)
            fit(gen, cfg)
            spec = spec_of(gen)
            matches = gen.run / "matches"
            res = {
                "generation": k,
                "attempt": attempt,
                "spec": spec,
                "pt": rel(gen.pt),
                "fit": read_json(gen.fit),
                "gain": match("gain", [spec] + [prev["spec"]] * 3, matches, cfg),
            }
            if not regressed(res["gain"]):
                res["value"] = match("value", [spec, "value", "value", "value"], matches, cfg)
                res["teacher"] = match("teacher", [spec, TEACHER, "value", "value"], matches, cfg)
            write_atomic(gen.result, json.dumps(res, indent=1))
        g = res["gain"]
        log(f"g{k} attempt {attempt}: gain {g['mean']:+.2f} [{g['lo']:+.2f}, {g['hi']:+.2f}]")
        if not regressed(g):
            return res, dropped
        dropped.append(res)
        log(
            f"g{k} attempt {attempt} regressed; "
            + ("retrying with seed 1" if attempt == 0 else "stopping")
        )
    return None, dropped


def main(argv: list[str] | None = None) -> int:
    global ROOT
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="runs/race")
    ap.add_argument("--hands", type=int, default=100_000)
    ap.add_argument("--holdout-hands", type=int, default=5_000)
    ap.add_argument("--chunk", type=int, default=500, help="hands per log chunk")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--chunks", type=int, default=20, help="match chunks (seeds 7000…)")
    ap.add_argument("--walls", type=int, default=50, help="walls per match chunk")
    ap.add_argument("--max-gens", type=int, default=8)
    ap.add_argument("--min-gens", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=200, help="cases a table bucket needs")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("HKQ_THREADS", "8")))
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--race-log", default=str(REPO / "target" / "release" / "examples" / "race_log.exe")
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny sizes on the CPU: 300 hands, 1 epoch, 1 chunk of 2 walls, 2 generations",
    )
    args = ap.parse_args(argv)
    if args.smoke:
        args.hands, args.holdout_hands, args.chunk, args.epochs = 300, 100, 100, 1
        args.chunks, args.walls, args.max_gens, args.min_gens = 1, 2, 2, 2
        args.device, args.min_n = "cpu", 1
    ROOT = (REPO / args.root).resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RAYON_NUM_THREADS", str(args.threads))
    os.chdir(REPO)  # spec paths are relative to the repo
    no_power_throttling()
    cfg = {
        "hands": args.hands,
        "holdout_hands": args.holdout_hands,
        "chunk": args.chunk,
        "epochs": args.epochs,
        "chunks": args.chunks,
        "walls": args.walls,
        "max_gens": args.max_gens,
        "min_gens": args.min_gens,
        "min_n": args.min_n,
    }
    saved = ROOT / "config.json"
    if saved.exists():
        old = read_json(saved)
        if {k: old.get(k) for k in cfg} != cfg:
            log(f"refusing: {saved} was written with other sizes: {old}")
            return 2
    else:
        write_atomic(saved, json.dumps(cfg, indent=1))
    cfg |= {"threads": args.threads, "device": args.device, "race_log": args.race_log}
    done = ROOT / "done.json"
    if done.exists():
        log(f"already finished: {read_json(done)['stopped']}")
        return 0
    log(f"start: threads {args.threads}, device {args.device}, {cfg}")

    accepted = [baseline(cfg)]
    dropped: list[dict[str, Any]] = []
    stop = None
    k = 1
    while stop is None:
        res, lost = generation(k, accepted, cfg)
        dropped += lost
        if res is None:
            stop = f"generation {k} regressed twice"
        else:
            accepted.append(res)
            stop = stop_reason(accepted, cfg)
        report(accepted, dropped, stop)
        k += 1
    best = best_generation(accepted)
    if best is not None:
        shutil.copyfile(REPO / best["pt"], ROOT / "best.pt")
        shutil.copyfile((REPO / best["pt"]).with_suffix(".mjnn"), ROOT / "best.mjnn")
        if not args.smoke:
            shutil.copyfile(
                (REPO / best["pt"]).with_suffix(".mjnn"), REPO / "models" / "race-belief.mjnn"
            )
    write_atomic(done, json.dumps({"stopped": stop, "best": best["generation"] if best else None}))
    progress(f"done: {stop}; best generation {best['generation'] if best else None}")
    log(f"done: {stop}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
