"""A Laplace approximation on the belief model's last layer (plan 2026-11-09, Task 18c).

No retraining. At the trained weights, the head's weights ``(w, b)`` get a Gaussian
posterior ``N(w*, H⁻¹)`` where ``H = scale · Σ_n GGN_n + λ I`` is the generalised
Gauss-Newton curvature of the training loss on a sample of training rows (scaled up
to the full log) plus a prior precision ``λ``. The torso is fixed, so a logit is linear
in the head's weights and its variance is ``φᵀ H⁻¹ φ`` with ``φ`` the torso output
(plus a 1 for the bias). Per guess:

* ``tenpai``: one logit per opponent, a logistic curvature ``p (1 − p) φ φᵀ``;
* ``waits``: one logit per opponent and tile, weighted by the opponent's tenpai (the
  loss counts tenpai opponents only);
* ``outcome``: three logits per opponent, the softmax curvature ``(diag p − p pᵀ) ⊗ φ φᵀ``.

A logit's ``(mean, variance)`` becomes a probability's mean (the probit moderation
``σ(μ / √(1 + π v / 8))``) and standard deviation (the delta method, ``p (1 − p) √v``;
for the softmax, the mean is the stated probability and the sd uses ``∇p`` against the logit
covariance). Deal-in is the model's
``1 − Π (1 − tenpai · wait)``; its variance sums the opponents' delta-method variances
(opponents independent, tenpai and wait independent).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mahjong.belief.logs import NUM_KINDS
from mahjong.belief.model import N_OPP, N_OUTCOME, Batch, BeliefNet


def weighted_gram(phi: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``out[j] = Σ_n weights[n, j] · φ[n] φ[n]ᵀ`` for ``phi (n, d)``, ``weights (n, J)``;
    result ``(J, d, d)``."""
    x = phi.unsqueeze(0) * weights.t().unsqueeze(2)  # (J, n, d)
    return torch.bmm(x.transpose(1, 2), phi.unsqueeze(0).expand(weights.shape[1], -1, -1))


def with_bias(phi: torch.Tensor) -> torch.Tensor:
    """Appends the constant 1 for the head's bias."""
    return torch.cat([phi, torch.ones_like(phi[:, :1])], dim=1)


@dataclass
class Laplace:
    """Posterior covariances of the head's weights, one block per logit group."""

    tenpai: torch.Tensor  # (3, d, d): inverse precision, per opponent
    waits: torch.Tensor  # (3 · 34, d, d), index opp · 34 + tile
    outcome: torch.Tensor  # (3, 3, d, 3, d): per opponent, (class, weight) pairs
    prior_precision: float
    rows_used: int
    rows_scale: float


def _torso(net: BeliefNet, b: Batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    b = b.to(device)
    x = net.inputs(b.features, b.discards, b.passed)
    phi = with_bias(net.torso(x))
    return phi, net.head(phi[:, :-1])


def fit(
    net: BeliefNet,
    batches: Iterable[Batch],
    total_rows: int,
    prior_precision: float = 1.0,
    device: str | torch.device = "cpu",
) -> Laplace:
    """The posterior from ``batches`` (a sample of the training rows; ``total_rows`` is
    the size of the log the sample stands for, so the curvature is scaled up)."""
    dev = torch.device(device)
    net = copy.deepcopy(net).to(dev).eval()
    d = net.cfg.hidden[-1] + 1
    h_ten = torch.zeros(N_OPP, d, d, dtype=torch.float64, device=dev)
    h_wait = torch.zeros(N_OPP * NUM_KINDS, d, d, dtype=torch.float64, device=dev)
    h_out = torch.zeros(N_OPP, N_OUTCOME * N_OUTCOME, d, d, dtype=torch.float64, device=dev)
    used = 0
    with torch.no_grad():
        for b in batches:
            phi, out = _torso(net, b, dev)
            parts = net.split(out)
            used += len(b)
            p_ten = torch.sigmoid(parts["tenpai"])  # (n, 3)
            for o in range(N_OPP):
                h_ten[o] += weighted_gram(phi, (p_ten[:, o] * (1 - p_ten[:, o]))[:, None])[0]
            tenpai = b.tenpai.to(dev)
            p_w = torch.sigmoid(parts["waits"])  # (n, 3, 34)
            w = (tenpai[:, :, None] * p_w * (1 - p_w)).reshape(len(b), -1)
            h_wait += weighted_gram(phi, w)
            p_o = torch.softmax(parts["outcome"], dim=-1)  # (n, 3, 3)
            ggn = torch.diag_embed(p_o) - p_o.unsqueeze(-1) * p_o.unsqueeze(-2)  # (n, 3, 3, 3)
            for o in range(N_OPP):
                h_out[o] += weighted_gram(phi, ggn[:, o].reshape(len(b), -1))
    scale = total_rows / max(used, 1)
    eye = torch.eye(d, dtype=torch.float64, device=dev)

    def inv(h: torch.Tensor) -> torch.Tensor:
        inverse: torch.Tensor = torch.linalg.inv(scale * h + prior_precision * eye)
        return inverse

    n_c = N_OUTCOME
    out_cov = torch.zeros(N_OPP, n_c, d, n_c, d, dtype=torch.float64, device=dev)
    for o in range(N_OPP):
        # H[(c, a), (e, b)] = Σ ggn[c, e] φ_a φ_b
        big = (
            (scale * h_out[o]).reshape(n_c, n_c, d, d).permute(0, 2, 1, 3).reshape(n_c * d, n_c * d)
        )
        big = big + prior_precision * torch.eye(n_c * d, dtype=torch.float64, device=dev)
        out_cov[o] = torch.linalg.inv(big).reshape(n_c, d, n_c, d)
    return Laplace(
        tenpai=inv(h_ten).float().cpu(),
        waits=inv(h_wait).float().cpu(),
        outcome=out_cov.float().cpu(),
        prior_precision=prior_precision,
        rows_used=used,
        rows_scale=scale,
    )


def _moderate(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(mu / torch.sqrt(1 + math.pi * var / 8))


def sigmoid_moments(mu: torch.Tensor, var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean (probit moderation) and standard deviation (delta method) of ``σ(logit)``
    for a Gaussian logit."""
    mean = _moderate(mu, var)
    p = torch.sigmoid(mu)
    return mean, p * (1 - p) * torch.sqrt(var)


def predict(
    lap: Laplace, net: BeliefNet, b: Batch, tiles: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Mean and standard deviation of the guesses for rows ``b``.

    Returns ``tenpai_p`` (the model's stated probability), ``tenpai_mean``,
    ``tenpai_sd`` (each ``(n, 3)``), ``outcome_p`` ``(n, 3, 3)`` with ``outcome_mean`` and
    ``outcome_sd`` (the deal-in class 1 and self-draw class 2 are of interest), and, when
    ``tiles`` ``(n,)`` names a discarded tile per row, ``deal_p``, ``deal_mean``, ``deal_sd``
    ``(n,)``.
    """
    net = net.eval()
    with torch.no_grad():
        phi, out = _torso(net, b, torch.device("cpu"))
        parts = net.split(out)
        mu = parts["tenpai"]
        var_t = torch.einsum("na,oab,nb->no", phi, lap.tenpai, phi).clamp(min=0)
        t_mean, t_sd = sigmoid_moments(mu, var_t)
        res: dict[str, torch.Tensor] = {
            "tenpai_p": torch.sigmoid(mu),
            "tenpai_mean": t_mean,
            "tenpai_sd": t_sd,
        }
        o_logit = parts["outcome"]  # (n, 3, 3)
        o_p = torch.softmax(o_logit, dim=-1)
        o_var = torch.zeros(len(b), N_OPP, N_OUTCOME)
        o_mean = torch.zeros_like(o_p)
        for o in range(N_OPP):
            v = lap.outcome[o]  # (c, a, e, b)
            cov = torch.einsum("na,caeb,nb->nce", phi, v, phi)  # (n, c, e)
            g = o_p[:, o].unsqueeze(-1) * (torch.eye(N_OUTCOME) - o_p[:, o].unsqueeze(1))  # (n,c,k)
            o_var[:, o] = torch.einsum("nck,nke,nce->nc", g, cov, g).clamp(min=0)
            # The softmax mean is the stated probability: with sds this small the
            # second-order correction is negligible (shrinking raw logits would pull
            # every class toward 1/3).
            o_mean[:, o] = o_p[:, o]
        res["outcome_p"] = o_p
        res["outcome_mean"] = o_mean
        res["outcome_sd"] = torch.sqrt(o_var)
        if tiles is not None:
            w_mu = parts["waits"][
                torch.arange(len(b))[:, None], torch.arange(N_OPP), tiles[:, None]
            ]
            w_var = torch.zeros(len(b), N_OPP)
            for t in sorted(set(tiles.tolist())):
                rows = (tiles == t).nonzero(as_tuple=True)[0]
                for o in range(N_OPP):
                    cov = lap.waits[o * NUM_KINDS + t]
                    w_var[rows, o] = torch.einsum("na,ab,nb->n", phi[rows], cov, phi[rows])
            w_mean, w_sd = sigmoid_moments(w_mu, w_var.clamp(min=0))
            p_win_map = torch.sigmoid(mu) * torch.sigmoid(w_mu)
            p_win = t_mean * w_mean
            var_win = (w_mean * t_sd) ** 2 + (t_mean * w_sd) ** 2 + (t_sd * w_sd) ** 2
            res["deal_p"] = 1 - torch.prod(1 - p_win_map, dim=1)
            res["deal_mean"] = 1 - torch.prod(1 - p_win, dim=1)
            other = torch.stack(
                [
                    torch.prod(1 - p_win[:, [k for k in range(N_OPP) if k != o]], dim=1)
                    for o in range(N_OPP)
                ],
                dim=1,
            )
            res["deal_sd"] = torch.sqrt((other**2 * var_win).sum(dim=1))
    return res


#: Edges of the stated-probability groups (log-spaced: the guesses span 1e-4 … 0.5).
GROUP_EDGES = np.array(
    [
        0.0,
        1e-4,
        3e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        3e-2,
        5e-2,
        7.5e-2,
        0.1,
        0.15,
        0.2,
        0.3,
        0.5,
        1.0,
    ]
)
#: A group needs this many guesses to be scored.
MIN_GROUP = 1000


class GroupTable:
    """Per game and group sums of (count, observed, mean, variance) of one guess type,
    for the table and for resampling whole games."""

    def __init__(self, edges: npt.NDArray[np.float64] = GROUP_EDGES) -> None:
        self.edges = edges
        self.sums: dict[int, npt.NDArray[np.float64]] = {}

    def add(
        self,
        game: npt.NDArray[np.integer[Any]],
        stated: npt.NDArray[np.floating[Any]],
        mean: npt.NDArray[np.floating[Any]],
        sd: npt.NDArray[np.floating[Any]],
        observed: npt.NDArray[np.floating[Any]],
    ) -> None:
        g = np.clip(np.searchsorted(self.edges, stated, side="right") - 1, 0, len(self.edges) - 2)
        k = len(self.edges) - 1
        for gm in np.unique(game):
            sel = game == gm
            arr = self.sums.setdefault(int(gm), np.zeros((k, 4)))
            for col, w in enumerate([np.ones(sel.sum()), observed[sel], mean[sel], sd[sel] ** 2]):
                arr[:, col] += np.bincount(g[sel], weights=w, minlength=k)

    def stack(self) -> npt.NDArray[np.float64]:
        """``(games, groups, 4)``."""
        return np.stack([self.sums[k] for k in sorted(self.sums)])

    @staticmethod
    def score(total: npt.NDArray[np.float64]) -> list[dict[str, float | bool | int]]:
        """Per group of ``total (groups, 4)``: observed rate, mean, sd and whether the
        observed rate lies in ``mean ± 1 sd`` (groups with under ``MIN_GROUP`` guesses are
        listed with ``scored`` false)."""
        rows: list[dict[str, float | bool | int]] = []
        for i, (n, obs, mean, var) in enumerate(total):
            if n <= 0:
                continue
            m, sd, o = mean / n, math.sqrt(var / n), obs / n
            rows.append(
                {
                    "group": i,
                    "n": int(n),
                    "observed": o,
                    "mean": m,
                    "sd": sd,
                    "inside": bool(abs(o - m) <= sd),
                    "scored": bool(n >= MIN_GROUP),
                }
            )
        return rows

    def table(self) -> list[dict[str, float | bool | int]]:
        rows = self.score(self.stack().sum(axis=0))
        for r in rows:
            r["low"], r["high"] = (
                float(self.edges[int(r["group"])]),
                float(self.edges[int(r["group"]) + 1]),
            )
        return rows


def inside_counts(
    stacked: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> tuple[int, int]:
    """(groups inside ±1 sd, groups scored) for games weighted by ``weights``."""
    rows = GroupTable.score(np.einsum("g,gkc->kc", weights, stacked))
    scored = [r for r in rows if r["scored"]]
    return sum(1 for r in scored if r["inside"]), len(scored)


def coverage(tables: dict[str, GroupTable], resamples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Share of groups inside mean ± 1 sd, per table and overall, with a 95 % interval
    that resamples whole games (the same games for every table in a resample)."""
    stacks = {k: t.stack() for k, t in tables.items()}
    games = {k: sorted(t.sums) for k, t in tables.items()}
    union = sorted(set().union(*games.values()))
    pos = {g: i for i, g in enumerate(union)}
    # Align every table to the union of games (a game without guesses adds zeros).
    aligned = {}
    for k, s in stacks.items():
        a = np.zeros((len(union),) + s.shape[1:])
        a[[pos[g] for g in games[k]]] = s
        aligned[k] = a
    rng = np.random.default_rng(seed)

    def shares(w: npt.NDArray[np.float64]) -> dict[str, float]:
        inside = total = 0
        out: dict[str, float] = {}
        for k, a in aligned.items():
            i, n = inside_counts(a, w)
            out[k] = i / n if n else float("nan")
            inside, total = inside + i, total + n
        out["overall"] = inside / total if total else float("nan")
        return out

    point = shares(np.ones(len(union)))
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(resamples):
        w = np.bincount(rng.integers(0, len(union), len(union)), minlength=len(union)).astype(float)
        for k, v in shares(w).items():
            draws[k].append(v)
    result: dict[str, Any] = {}
    for k, v in point.items():
        lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
        result[k] = {"share_inside": v, "ci_low": float(lo), "ci_high": float(hi)}
    result["games"] = len(union)
    return result
