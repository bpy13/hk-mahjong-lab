"""PPO self-play for the suit-shared CNN with the proposal's additions (plan Task 17).

``train-core`` of the strategy proposal: clipped PPO on :class:`~mahjong.rl.cnn.SuitCNN`
with batched inference (GPU when there is one) for every game of a Rust
``HkVecEnv``, plus:

* **Duplicate-baseline reward (idea 5a)**, on by default: every wall is played twice
  with the same tiles (common random numbers): copy 0 with the learner in seat
  ``i % 4``, copy 1 with the baseline agent in that seat; the other seats are the
  learner's network in both copies (not trained on); until the copies first differ
  every move is shared, so those seats' sampled moves are common too. The learner's reward is the
  paired difference of its seat's points, ``sq(points_learner - points_baseline)``,
  so the luck of the wall cancels. The baseline is the frozen init network (played
  greedily on the same device) or ``ev`` (``--baseline ev``, in Rust). With
  ``--no-paired-reward`` the learner plays all four seats and the advantage is
  ``sq(points) - V(s)`` with the critic below.
* **Oracle four-seat critic (5b) = who-pays value head (2b)**: the value head also
  reads the hidden oracle block (opponents' hands, next wall tiles) and outputs the
  four relative seats' points, summing to zero. The policy never sees the oracle
  block (it enters only the value head).
* **Log-scaled payouts (5d)**: ``sq(x) = sign(x) * log(1 + |x| / 8)`` for rewards
  and value targets.
* **Minimum-faan curriculum (1b)**: new deals use ``min_faan`` 0 for the first 20 %
  of the run, 1 until 40 %, then 3 (the real rules).
* **Target head on the final pattern (1a)**: on every decision of a hand its seat
  went on to win, the target head is trained (cross-entropy) on the target class of
  the winning hand (``finished(detail=True)``). ``--no-target-head`` turns it off.
* **Per-pattern KL anchor (1c)**: ``KL(teacher || student)`` on the policy, toward the
  frozen init network, only on decisions where the teacher's and the student's
  target argmax differ, weighted by a factor per pattern group (plan kind x
  structure, 15 groups, of the teacher's target). A group's factor grows x1.25 per
  iteration (up to x16) while the learner's win share with that pattern is below
  80 % of the teacher's at the same ``min_faan``, and decays back to x1 otherwise.
  Teacher shares come from the baseline copies (paired mode) or from a short
  teacher self-play run per ``min_faan`` (unpaired). With ``--init none`` the
  anchor is off (weight 0) unless ``--anchor-weight`` is given.
* Entropy bonus as in ``ppo.py``.

Checkpoints every 30 minutes of training to ``<out>/ckpt-<iteration>.pt`` (network,
optimizer) with ``<out>/state.json`` (hands, iteration, elapsed time, curriculum
stage, anchor factors, pattern statistics, the checkpoint holding the optimizer);
``--resume`` continues from the newest checkpoint. Every ``--eval-minutes`` (default
120) the network is exported and plays ``--eval-walls`` walls (4 seatings) against
three ``value`` bots in Rust; points, rates, draw rate and pattern shares go to
``<out>/log.json`` together with one entry per iteration.

CLI::

    python -m mahjong.rl.ppo_cnn --init <pt or none> (--minutes M | --hours H)
        --out runs/ppo/<run> [--seed S] [--resume] [--no-paired-reward]
        [--no-target-head] [--envs N] [--device cuda|cpu] [--baseline init|ev]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F

from mahjong._core import HkVecEnv, tournament
from mahjong.rl.cnn import ORACLE_LEN, STATE_LEN, CnnConfig, SuitCNN, export

NUM_ACTIONS = 134
WIN_CODES = (34, 104)
NO_TARGET = 255
KINDS = ("mixed", "half_flush", "full_flush", "honours", "orphans")
STRUCTURES = ("any", "all_pungs", "all_chows")
#: Pattern groups for shares and the anchor: plan kind x structure (suits merged).
GROUPS = tuple(f"{k}/{s}" for k in KINDS for s in STRUCTURES)
NUM_GROUPS = len(GROUPS)
#: (fraction of the run, min_faan from then on).
CURRICULUM = ((0.0, 0), (0.2, 1), (0.4, 3))
ROLE_LEARNER, ROLE_BASELINE = 0, 1


def no_power_throttling() -> None:
    """Windows throttles detached headless processes to ~0.15 core each (EcoQoS); opt out.

    Copied from ``runs/matches/task7/run_matches.py``."""
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
    k32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined,unused-ignore]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))


def squash(x: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Idea 5d: ``sign(x) * log(1 + |x| / 8)``, element-wise."""
    a = np.asarray(x, dtype=np.float64)
    return (np.sign(a) * np.log1p(np.abs(a) / 8.0)).astype(np.float32)


def group_of(target_class: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Pattern group (``kind * 3 + structure``) of target classes."""
    c = np.asarray(target_class, dtype=np.int64)
    return (c // 9) * 3 + c % 3


def stage_of(frac: float, curriculum: bool) -> tuple[int, int]:
    """``(stage, min_faan)`` at a fraction of the run."""
    if not curriculum:
        return len(CURRICULUM) - 1, CURRICULUM[-1][1]
    stage = max(i for i, (start, _) in enumerate(CURRICULUM) if frac >= start)
    return stage, CURRICULUM[stage][1]


@dataclass
class PpoCnnConfig:
    out: Path
    init: Path | None = None
    seconds: float = 3600.0
    seed: int = 0
    paired: bool = True
    target_head: bool = True
    envs: int = 256
    device: str = "cuda"
    baseline: str = "init"
    hands_per_iter: int = 1024
    epochs: int = 3
    batch: int = 2048
    lr: float = 1e-4
    clip: float = 0.2
    entropy: float = 0.005
    value_coef: float = 0.5
    target_coef: float = 0.2
    #: Base weight of the per-pattern KL anchor; None = 0.05, or 0 with no init.
    anchor_weight: float | None = None
    curriculum: bool = True
    ckpt_minutes: float = 30.0
    eval_minutes: float = 120.0
    eval_walls: int = 200
    keep_checkpoints: int = 3
    teacher_hands: int = 256
    log_seconds: float = 60.0
    resume: bool = False
    torch_threads: int = 1
    #: Stop after this many iterations (tests); None = run until the time is up.
    max_iterations: int | None = None


@dataclass
class Stats:
    """Win counts per pattern group: the teacher's per min_faan (cumulative) and the
    learner's (decayed by 0.95 per iteration, reset when min_faan changes)."""

    teacher: dict[int, list[float]] = field(default_factory=dict)
    teacher_hands: dict[int, int] = field(default_factory=dict)
    student: list[float] = field(default_factory=lambda: [0.0] * NUM_GROUPS)
    student_min_faan: int = -1
    anchor_mult: list[float] = field(default_factory=lambda: [1.0] * NUM_GROUPS)

    def add_teacher(self, min_faan: int, classes: list[int], hands: int) -> None:
        t = self.teacher.setdefault(min_faan, [0.0] * NUM_GROUPS)
        for c in classes:
            t[int(group_of(c))] += 1
        self.teacher_hands[min_faan] = self.teacher_hands.get(min_faan, 0) + hands

    def add_student(self, classes: list[int]) -> None:
        for c in classes:
            self.student[int(group_of(c))] += 1

    def update_anchor(self, min_faan: int, min_wins: int = 20) -> None:
        """Raises the factor of every group whose learner share is below 80 % of the
        teacher's; lets the others decay to 1. Then decays the learner's counts."""
        t = np.array(self.teacher.get(min_faan, [0.0] * NUM_GROUPS))
        s = np.array(self.student)
        if t.sum() >= min_wins and s.sum() > 0:
            ts, ss = t / t.sum(), s / s.sum()
            for g in range(NUM_GROUPS):
                if t[g] >= 3 and ss[g] < 0.8 * ts[g]:
                    self.anchor_mult[g] = min(16.0, self.anchor_mult[g] * 1.25)
                else:
                    self.anchor_mult[g] = max(1.0, self.anchor_mult[g] * 0.9)
        self.student = [x * 0.95 for x in self.student]

    def shares(self, counts: list[float]) -> dict[str, float]:
        total = sum(counts)
        return {GROUPS[g]: c / total for g, c in enumerate(counts) if total > 0 and c > 0}


@dataclass
class Outcome:
    deltas: npt.NDArray[np.int64]
    paired: float
    win_class: npt.NDArray[np.int64]  # per absolute seat, NO_TARGET if no win


class Buffer:
    """Learner decisions of hands in progress and finished, stored per step as arrays
    (no per-row Python objects). Each hand of each game has a unique id; outcomes
    are matched to rows by it when a batch is assembled."""

    def __init__(self, games: int) -> None:
        self.uid = np.arange(games, dtype=np.int64)
        self.next_uid = games
        self.chunks: list[dict[str, np.ndarray]] = []
        self.outcomes: dict[int, Outcome] = {}

    def add(self, rows: dict[str, np.ndarray], env_ids: npt.NDArray[np.int64]) -> None:
        if len(env_ids):
            rows["uid"] = self.uid[env_ids]
            self.chunks.append(rows)

    def finish(self, env: int, outcome: Outcome) -> None:
        self.outcomes[int(self.uid[env])] = outcome
        self.uid[env] = self.next_uid
        self.next_uid += 1

    def take(self, paired: bool) -> dict[str, np.ndarray] | None:
        """All rows of finished hands with rewards attached; the rest stay."""
        if not self.chunks:
            return None
        d = {k: np.concatenate([c[k] for c in self.chunks]) for k in self.chunks[0]}
        done = np.array([u in self.outcomes for u in d["uid"].tolist()], dtype=bool)
        rest = {k: v[~done] for k, v in d.items()}
        self.chunks = [rest] if len(rest["uid"]) else []
        d = {k: v[done] for k, v in d.items()}
        if not len(d["uid"]):
            return None
        outs = [self.outcomes[u] for u in d["uid"].tolist()]
        for u in set(d["uid"].tolist()):
            del self.outcomes[u]
        seat = d["seat"].astype(np.int64)
        deltas = np.stack([o.deltas for o in outs])  # (n, 4) absolute seats
        rel = deltas[np.arange(len(seat))[:, None], (seat[:, None] + np.arange(4)) % 4]
        d["returns"] = squash(rel)  # (n, 4) relative seats, 0 = me
        if paired:
            d["reward"] = squash(np.array([o.paired for o in outs]))
        else:
            d["reward"] = d["returns"][:, 0].copy()
        d["target"] = np.array([o.win_class[s] for o, s in zip(outs, seat, strict=True)])
        return d


def load_init(path: Path | None, cfg: CnnConfig | None = None) -> SuitCNN:
    """The init network with the oracle critic block added (zero-initialised, so the
    critic starts as the init's value head)."""
    if path is None:
        return SuitCNN(CnnConfig(**{**asdict(cfg or CnnConfig()), "oracle": True}))
    ckpt = torch.load(path, map_location="cpu")
    net = SuitCNN(CnnConfig(**{**ckpt["cfg"], "oracle": True}))
    missing, unexpected = net.load_state_dict(ckpt["state"], strict=False)
    if unexpected or any(not m.startswith("oracle") for m in missing):
        raise ValueError(f"{path}: not a SuitCNN checkpoint ({missing=}, {unexpected=})")
    if missing:
        torch.nn.init.zeros_(net.oracle_value.weight)
        torch.nn.init.zeros_(net.oracle_value.bias)
    return net


def masked(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(mask == 0, -1e9)


def force_wins(actions: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every seat declares a legal win (as in ``selfplay.force_wins``); returns the
    actions and which rows were forced."""
    forced = np.zeros(len(actions), dtype=bool)
    for code in WIN_CODES:
        hit = masks[:, code] == 1
        actions = np.where(hit, code, actions)
        forced |= hit
    return actions, forced


class Trainer:
    def __init__(self, cfg: PpoCnnConfig) -> None:
        self.cfg = cfg
        self.out = cfg.out
        self.out.mkdir(parents=True, exist_ok=True)
        dev = cfg.device
        if dev == "cuda" and not torch.cuda.is_available():
            print("ppo_cnn: CUDA not available, using the CPU", flush=True)
            dev = "cpu"
        self.device = torch.device(dev)
        # Few PyTorch CPU threads: its OpenMP threads and the Rust environments'
        # rayon threads otherwise fight for the cores (a 16-row forward pass took
        # 0.2-0.4 s instead of 1-3 ms on this PC).
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.iteration = 0
        self.hands = 0  # finished games (learner copies)
        self.seat_hands = 0  # learner seat-hands trained on
        self.elapsed = 0.0  # training seconds over all sessions
        self.last_ckpt = 0.0
        self.next_eval = cfg.eval_minutes * 60
        self.stats = Stats()
        self.history: list[dict[str, Any]] = []
        self.stage = -1
        self.min_faan = -1

        teacher_path = self.out / "teacher.pt"
        state_path = self.out / "state.json"
        state: dict[str, Any] | None = None
        if cfg.resume and state_path.exists():
            state = json.loads(state_path.read_text())
        if state is None:
            if cfg.resume:
                print(f"ppo_cnn: no {state_path}, starting fresh", flush=True)
            self.net = load_init(cfg.init)
            # The teacher (KL anchor, baseline) is the frozen start: the distilled
            # network, or with --init none this very random initialisation.
            self.teacher = copy.deepcopy(self.net)
            self.teacher.save(teacher_path)
        else:
            self.teacher = SuitCNN.load(teacher_path)
            self.net = SuitCNN(self.teacher.cfg)
        self.net.to(self.device)
        self.teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        if cfg.anchor_weight is not None:
            self.anchor = cfg.anchor_weight
        else:
            self.anchor = 0.05 if cfg.init is not None else 0.0
        if state is not None:
            self._restore(state)
        if (self.out / "log.json").exists() and state is not None:
            self.history = json.loads((self.out / "log.json").read_text())

        env_seed = cfg.seed * 7919 + self.iteration
        if cfg.paired:
            base = None if cfg.baseline == "init" else cfg.baseline
            self.env = HkVecEnv(cfg.envs, env_seed, paired=True, baseline=base, min_faan=0)
        else:
            self.env = HkVecEnv(cfg.envs, env_seed, min_faan=0)
        self.buffer = Buffer(cfg.envs)

    # --- Checkpoints ------------------------------------------------------------

    def _restore(self, state: dict[str, Any]) -> None:
        ckpt = torch.load(self.out / state["checkpoint"], map_location="cpu")
        self.net.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.iteration = state["iteration"]
        self.hands = state["hands"]
        self.seat_hands = state["seat_hands"]
        self.elapsed = state["elapsed_s"]
        self.last_ckpt = self.elapsed
        self.next_eval = state["next_eval_s"]
        self.anchor = state["anchor_weight"]
        s = state["stats"]
        self.stats = Stats(
            teacher={int(k): v for k, v in s["teacher"].items()},
            teacher_hands={int(k): v for k, v in s["teacher_hands"].items()},
            student=s["student"],
            student_min_faan=s["student_min_faan"],
            anchor_mult=s["anchor_mult"],
        )
        print(
            f"ppo_cnn: resumed {state['checkpoint']} (iteration {self.iteration}, "
            f"{self.hands} hands, {self.elapsed / 60:.1f} min)",
            flush=True,
        )

    def checkpoint(self) -> Path:
        name = f"ckpt-{self.iteration:06d}.pt"
        path = self.out / name
        tmp = path.with_suffix(".tmp")
        torch.save(
            {
                "model": self.net.state_dict(),
                "cfg": asdict(self.net.cfg),
                "optimizer": self.opt.state_dict(),
                "iteration": self.iteration,
                "hands": self.hands,
            },
            tmp,
        )
        os.replace(tmp, path)
        state = {
            "checkpoint": name,
            "optimizer": f"{name} ('optimizer')",
            "iteration": self.iteration,
            "hands": self.hands,
            "seat_hands": self.seat_hands,
            "elapsed_s": self.elapsed,
            "budget_s": self.cfg.seconds,
            "stage": self.stage,
            "min_faan": self.min_faan,
            "next_eval_s": self.next_eval,
            "anchor_weight": self.anchor,
            "stats": asdict(self.stats),
            "config": {
                k: str(v) if isinstance(v, Path) else v for k, v in asdict(self.cfg).items()
            },
        }
        tmp_state = self.out / "state.json.tmp"
        tmp_state.write_text(json.dumps(state, indent=1))
        os.replace(tmp_state, self.out / "state.json")
        old = sorted(self.out.glob("ckpt-*.pt"))
        for p in old[: max(0, len(old) - self.cfg.keep_checkpoints)]:
            p.unlink()
        self.last_ckpt = self.elapsed
        return path

    def write_log(self) -> None:
        tmp = self.out / "log.json.tmp"
        tmp.write_text(json.dumps(self.history, indent=1))
        os.replace(tmp, self.out / "log.json")

    # --- Playing ----------------------------------------------------------------

    def _tensors(
        self, f: np.ndarray, o: np.ndarray, m: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = self.device
        return (
            torch.from_numpy(f).to(dev, non_blocking=True),
            torch.from_numpy(o).to(dev, non_blocking=True),
            torch.from_numpy(m).to(dev, non_blocking=True),
        )

    def _act(
        self, net: SuitCNN, f: np.ndarray, o: np.ndarray, m: np.ndarray, greedy: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Actions, their log-probabilities and the critic's values for rows."""
        x, orc, mask = self._tensors(f, o, m)
        with torch.no_grad():
            pi, _, v = net(x, orc)
            logits = masked(pi, mask)
            if greedy:
                a = logits.argmax(1)
                logp = torch.zeros_like(a, dtype=torch.float32)
            else:
                # By hand: torch.distributions.Categorical's argument checks cost
                # ~20 ms per call on 500 rows.
                logp_all = F.log_softmax(logits, 1)
                a = torch.multinomial(logp_all.exp(), 1)[:, 0]
                logp = logp_all.gather(1, a[:, None])[:, 0]
        return a.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()

    def collect(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Steps the games until ``hands_per_iter`` learner hands have finished."""
        cfg, env, buf = self.cfg, self.env, self.buffer
        self.net.eval()
        finished = 0
        paired_sum, draws, learner_wins = 0.0, 0, 0
        while finished < cfg.hands_per_iter:
            fb, ob, mb, ids, seats = env.observe()
            n = len(ids)
            f = np.frombuffer(fb, dtype=np.float32).reshape(n, STATE_LEN)
            o = np.frombuffer(ob, dtype=np.float32).reshape(n, ORACLE_LEN)
            m = np.frombuffer(mb, dtype=np.uint8).reshape(n, NUM_ACTIONS)
            if cfg.paired:
                roles = np.frombuffer(env.observed_roles()[1], dtype=np.uint8).astype(np.int64)
            else:
                roles = np.zeros(n, dtype=np.int64)
            actions = np.zeros(n, dtype=np.int64)
            logp = np.zeros(n, dtype=np.float32)
            values = np.zeros((n, 4), dtype=np.float32)
            base = roles == ROLE_BASELINE
            lrn = ~base
            if lrn.any():
                actions[lrn], logp[lrn], values[lrn] = self._act(
                    self.net, f[lrn], o[lrn], m[lrn], greedy=False
                )
            if base.any():
                actions[base], _, _ = self._act(self.teacher, f[base], o[base], m[base], True)
            actions, forced = force_wins(actions, m)
            keep = (roles == ROLE_LEARNER) & ~forced & (m.sum(1) > 1)
            env_ids = np.asarray(ids, dtype=np.int64)
            buf.add(
                {
                    "feats": f[keep].astype(np.float16),
                    "oracle": o[keep].astype(np.float16),
                    "masks": m[keep].copy(),
                    "actions": actions[keep],
                    "logp": logp[keep],
                    "value": values[keep],
                    # Rust hands Vec<u8> over as bytes.
                    "seat": np.frombuffer(bytes(seats), dtype=np.uint8)[keep],
                },
                env_ids[keep],
            )
            env.step(actions.tolist())
            for d in env.finished(detail=True):
                win_class = np.full(4, NO_TARGET, dtype=np.int64)
                for seat, cls, _faan, _bits in d["wins"]:
                    win_class[seat] = cls
                learner = d["learner"]
                buf.finish(
                    d["env"],
                    Outcome(np.asarray(d["deltas"]), float(d.get("paired", 0)), win_class),
                )
                finished += 1
                if not d["wins"]:
                    draws += 1
                if learner is None:  # the learner played all four seats
                    classes = [c for _s, c, _f, _b in d["wins"]]
                    self.seat_hands += 4
                else:
                    classes = [c for s, c, _f, _b in d["wins"] if s == learner]
                    self.seat_hands += 1
                    paired_sum += d["paired"]
                    teacher = [c for s, c, _f, _b in d["baseline_wins"] if s == learner]
                    self.stats.add_teacher(d["min_faan"], teacher, 1)
                learner_wins += len(classes)
                if d["min_faan"] == self.stats.student_min_faan:
                    self.stats.add_student(classes)
        self.hands += finished
        batch = buf.take(cfg.paired)
        assert batch is not None, "no learner decisions in a whole iteration"
        info = {
            "draw_rate": draws / finished,
            "learner_win_rate": learner_wins / finished / (1 if cfg.paired else 4),
            "paired_mean": paired_sum / finished if cfg.paired else None,
        }
        return batch, info

    def measure_teacher(self, min_faan: int) -> None:
        """Unpaired runs: the teacher's win counts per pattern group from a short
        self-play run (sampling) at ``min_faan``."""
        games = min(self.cfg.envs, 64)
        env = HkVecEnv(games, 4242 + min_faan, min_faan=min_faan)
        done = 0
        classes: list[int] = []
        while done < self.cfg.teacher_hands:
            fb, ob, mb, ids, _ = env.observe()
            n = len(ids)
            f = np.frombuffer(fb, dtype=np.float32).reshape(n, STATE_LEN)
            o = np.frombuffer(ob, dtype=np.float32).reshape(n, ORACLE_LEN)
            m = np.frombuffer(mb, dtype=np.uint8).reshape(n, NUM_ACTIONS)
            a, _, _ = self._act(self.teacher, f, o, m, greedy=False)
            a, _ = force_wins(a, m)
            env.step(a.tolist())
            for d in env.finished(detail=True):
                classes += [c for _s, c, _f, _b in d["wins"]]
                done += 1
        # Four seats per hand: count seat-hands like the learner's statistics.
        self.stats.add_teacher(min_faan, classes, 4 * done)

    # --- Learning ---------------------------------------------------------------

    def update(self, d: dict[str, np.ndarray]) -> dict[str, float]:
        cfg, dev = self.cfg, self.device
        n = len(d["actions"])
        reward = d["reward"]
        adv = reward if cfg.paired else reward - d["value"][:, 0]
        adv = ((adv - adv.mean()) / (adv.std() + 1e-6)).astype(np.float32)

        def t(a: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
            x = torch.from_numpy(np.ascontiguousarray(a)).to(dev)
            return x if dtype is None else x.to(dtype)

        feats, oracle = t(d["feats"], torch.float32), t(d["oracle"], torch.float32)
        masks, actions = t(d["masks"]), t(d["actions"], torch.int64)
        old_logp, adv_t = t(d["logp"]), t(adv)
        returns, target = t(d["returns"]), t(d["target"].astype(np.int64))
        mult = torch.tensor(self.stats.anchor_mult, dtype=torch.float32, device=dev)
        self.net.train()
        sums: dict[str, float] = {}
        steps = 0
        for _ in range(cfg.epochs):
            perm = torch.from_numpy(self.rng.permutation(n)).to(dev)
            for i in range(0, n, cfg.batch):
                idx = perm[i : i + cfg.batch]
                x, orc, mask = feats[idx], oracle[idx], masks[idx]
                pi, tg, v = self.net(x, orc)
                logits = masked(pi, mask)
                logp_all = F.log_softmax(logits, 1)
                logp = logp_all.gather(1, actions[idx][:, None])[:, 0]
                ratio = torch.exp(logp - old_logp[idx])
                ad = adv_t[idx]
                loss_pi = -torch.min(
                    ratio * ad, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad
                ).mean()
                loss_v = F.mse_loss(v, returns[idx])
                probs = logp_all.exp()
                ent = -(probs * logp_all.masked_fill(mask == 0, 0.0)).sum(1).mean()
                loss = loss_pi + cfg.value_coef * loss_v - cfg.entropy * ent
                lab = target[idx]
                has = lab != NO_TARGET
                loss_t = torch.zeros((), device=dev)
                if cfg.target_head and bool(has.any()):
                    loss_t = F.cross_entropy(tg[has], lab[has])
                    loss = loss + cfg.target_coef * loss_t
                loss_a = torch.zeros((), device=dev)
                if self.anchor > 0:
                    with torch.no_grad():
                        tpi, ttg, _ = self.teacher(x)
                        t_logp = F.log_softmax(masked(tpi, mask), 1)
                        t_target = ttg.argmax(1)
                        disagree = (t_target != tg.argmax(1)).float()
                        groups = (t_target // 9) * 3 + t_target % 3
                        w = self.anchor * mult[groups] * disagree
                    kl = (t_logp.exp() * (t_logp - logp_all)).masked_fill(mask == 0, 0.0).sum(1)
                    loss_a = (w * kl).mean()
                    loss = loss + loss_a
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                for k, val in (
                    ("policy_loss", loss_pi),
                    ("value_loss", loss_v),
                    ("target_loss", loss_t),
                    ("anchor_loss", loss_a),
                    ("entropy", ent),
                ):
                    sums[k] = sums.get(k, 0.0) + float(val.item())
                steps += 1
        out = {k: v / max(1, steps) for k, v in sums.items()}
        out["decisions"] = float(n)
        out["advantage_std"] = float(reward.std())
        return out

    # --- Evaluation -------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        path = self.out / "eval.mjnn"
        export(self.net, path)
        res = tournament(
            [f"net:path={path}", "value", "value", "value"], self.cfg.eval_walls, 99, seatings=4
        )
        a = res["agents"][0]
        return {
            "vs": "value",
            "walls": self.cfg.eval_walls,
            "hands": a["hands"],
            "mean": a["mean"],
            "ci95": a["ci95"],
            "win_rate": a["win_rate"],
            "deal_in_rate": a["deal_in_rate"],
            "draw_rate": res["draw_rate"],
            "mean_faan": a["mean_faan"],
            "pattern_shares": a["pattern_shares"],
        }

    # --- Main loop --------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        session_start = time.time()
        session_hands0 = self.hands
        start_elapsed = self.elapsed
        last_log = 0.0
        while self.elapsed < cfg.seconds:
            if cfg.max_iterations is not None and self.iteration >= cfg.max_iterations:
                break
            stage, min_faan = stage_of(self.elapsed / cfg.seconds, cfg.curriculum)
            if min_faan != self.min_faan:
                self.env.set_min_faan(min_faan)
                self.stage, self.min_faan = stage, min_faan
                if self.stats.student_min_faan != min_faan:
                    self.stats.student = [0.0] * NUM_GROUPS
                    self.stats.student_min_faan = min_faan
                if not cfg.paired and self.anchor > 0 and min_faan not in self.stats.teacher:
                    self.measure_teacher(min_faan)
            t0 = time.time()
            batch, info = self.collect()
            t1 = time.time()
            losses = self.update(batch)
            t2 = time.time()
            if self.anchor > 0:
                self.stats.update_anchor(min_faan)
            self.iteration += 1
            self.elapsed = start_elapsed + (time.time() - session_start)
            session_hours = max(1e-9, (time.time() - session_start) / 3600)
            entry: dict[str, Any] = {
                "iter": self.iteration,
                "hands": self.hands,
                "seat_hands": self.seat_hands,
                "minutes": round(self.elapsed / 60, 3),
                "hands_per_hour": round((self.hands - session_hands0) / session_hours),
                "stage": self.stage,
                "min_faan": self.min_faan,
                "collect_s": round(t1 - t0, 2),
                "update_s": round(t2 - t1, 2),
                **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in info.items()},
                **{k: round(v, 5) for k, v in losses.items()},
                "learner_shares": self.stats.shares(self.stats.student),
            }
            if self.anchor > 0:
                entry["anchor_mult"] = {
                    GROUPS[g]: round(x, 3)
                    for g, x in enumerate(self.stats.anchor_mult)
                    if x > 1.0001
                }
            if self.elapsed >= self.next_eval:
                entry["eval"] = self.evaluate()
                self.next_eval += cfg.eval_minutes * 60
            print("ppo_cnn", json.dumps(entry), flush=True)
            self.history.append(entry)
            if self.elapsed - self.last_ckpt >= cfg.ckpt_minutes * 60:
                self.checkpoint()
            if "eval" in entry or time.time() - last_log >= cfg.log_seconds:
                self.write_log()
                last_log = time.time()
        ckpt = self.checkpoint()
        self.write_log()
        self.net.save(self.out / "final.pt")
        export(self.net, self.out / "final.mjnn")
        session_hours = max(1e-9, (time.time() - session_start) / 3600)
        summary = {
            "iterations": self.iteration,
            "hands": self.hands,
            "hands_per_hour": (self.hands - session_hands0) / session_hours,
            "checkpoint": str(ckpt),
        }
        print(
            f"ppo_cnn: done, {self.iteration} iterations, {self.hands} hands, "
            f"{summary['hands_per_hour']:,.0f} hands per hour "
            f"({'paired: each wall played twice' if cfg.paired else 'unpaired'})",
            flush=True,
        )
        return summary


def train(cfg: PpoCnnConfig) -> dict[str, Any]:
    """Runs (or resumes) a training run; returns a summary."""
    return Trainer(cfg).run()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m mahjong.rl.ppo_cnn",
        description="PPO self-play for the suit-shared CNN (plan Task 17).",
    )
    ap.add_argument("--init", required=True, help="a SuitCNN .pt file, or 'none' (random)")
    length = ap.add_mutually_exclusive_group()
    length.add_argument("--minutes", type=float)
    length.add_argument("--hours", type=float)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-paired-reward", action="store_true")
    ap.add_argument("--no-target-head", action="store_true")
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--baseline", default="init", help="'init' (frozen init network) or 'ev'")
    ap.add_argument("--hands-per-iter", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--anchor-weight", type=float, default=None)
    ap.add_argument("--ckpt-minutes", type=float, default=30.0)
    ap.add_argument("--eval-minutes", type=float, default=120.0)
    ap.add_argument("--eval-walls", type=int, default=200)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args(argv)
    no_power_throttling()

    state_path = args.out / "state.json"
    if args.minutes is not None:
        seconds = args.minutes * 60
    elif args.hours is not None:
        seconds = args.hours * 3600
    elif args.resume and state_path.exists():
        seconds = json.loads(state_path.read_text())["budget_s"]
    else:
        ap.error("give --minutes or --hours (optional only with --resume)")
    if args.baseline not in ("init", "ev"):
        ap.error("--baseline must be 'init' or 'ev'")
    init = None if args.init.lower() == "none" else Path(args.init)
    cfg = PpoCnnConfig(
        out=args.out,
        init=init,
        seconds=seconds,
        seed=args.seed,
        paired=not args.no_paired_reward,
        target_head=not args.no_target_head,
        curriculum=not args.no_curriculum,
        envs=args.envs,
        device=args.device,
        baseline=args.baseline,
        hands_per_iter=args.hands_per_iter,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        entropy=args.entropy,
        anchor_weight=args.anchor_weight,
        ckpt_minutes=args.ckpt_minutes,
        eval_minutes=args.eval_minutes,
        eval_walls=args.eval_walls,
        resume=args.resume,
        max_iterations=args.max_iterations,
        torch_threads=args.torch_threads,
    )
    train(cfg)


if __name__ == "__main__":
    main()
