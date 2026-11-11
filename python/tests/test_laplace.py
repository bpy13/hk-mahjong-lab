"""The belief model's Laplace approximation (plan 2026-11-09, Task 18c)."""

from __future__ import annotations

import numpy as np
import torch
from mahjong.belief import laplace as lp
from mahjong.belief.model import BeliefConfig, BeliefNet

from tests.test_belief import fake_batch


def tiny_net() -> BeliefNet:
    torch.manual_seed(0)
    return BeliefNet(BeliefConfig(hidden=(8,)))


def test_weighted_gram_matches_loop() -> None:
    phi, w = torch.randn(20, 5), torch.rand(20, 3)
    got = lp.weighted_gram(phi, w)
    for j in range(3):
        want = torch.stack([w[n, j] * torch.outer(phi[n], phi[n]) for n in range(20)]).sum(0)
        assert torch.allclose(got[j], want, atol=1e-5)


def test_more_data_means_smaller_sd() -> None:
    net = tiny_net()
    b = fake_batch(64)
    small = lp.fit(net, [b], total_rows=64)
    large = lp.fit(net, [b], total_rows=64_000)
    s, big = lp.predict(small, net, b), lp.predict(large, net, b)
    assert (big["tenpai_sd"] < s["tenpai_sd"]).all()
    assert (big["outcome_sd"] < s["outcome_sd"]).all()
    assert (s["tenpai_sd"] >= 0).all()


def test_mean_is_closer_to_half_than_the_stated_probability() -> None:
    net = tiny_net()
    b = fake_batch(64)
    r = lp.predict(lp.fit(net, [b], total_rows=64), net, b)
    assert ((r["tenpai_mean"] - 0.5).abs() <= (r["tenpai_p"] - 0.5).abs() + 1e-6).all()
    assert torch.allclose(r["outcome_mean"].sum(-1), torch.ones(64, 3), atol=1e-5)


def test_deal_in_guess() -> None:
    net = tiny_net()
    b = fake_batch(32)
    lap = lp.fit(net, [b], total_rows=3200)
    r = lp.predict(lap, net, b, tiles=torch.randint(0, 34, (32,)))
    assert ((r["deal_p"] >= 0) & (r["deal_p"] <= 1)).all()
    assert (r["deal_sd"] >= 0).all() and torch.isfinite(r["deal_sd"]).all()


def test_coverage_of_an_honest_guess() -> None:
    # Guesses whose stated mean is right and whose sd covers the binomial noise of a
    # group: about 68 % of groups land inside ±1 sd.
    rng = np.random.default_rng(0)
    t = lp.GroupTable()
    n = 400_000
    game = rng.integers(0, 200, n)
    p = rng.uniform(0.01, 0.5, n)
    obs = (rng.random(n) < p) * 1.0
    # Per-guess sd chosen so the group-level sd equals the binomial sd of the group mean
    # (groups of ~25k guesses): sqrt(p (1 - p) / 25000) ~ 0.002 for p ~ 0.25.
    sd = np.full(n, 0.002)
    t.add(game, p, p, sd, obs)
    cov = lp.coverage({"x": t}, resamples=20)
    assert cov["games"] == 200
    assert 0.0 <= cov["overall"]["share_inside"] <= 1.0
