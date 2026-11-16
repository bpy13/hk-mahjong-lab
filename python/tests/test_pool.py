"""Self-play against a pool of opponents (plan 2026-11-16 Task 5)."""

from __future__ import annotations

import numpy as np
from mahjong._core import HkState, HkVecEnv, encode_history
from mahjong.rl.cnn import HISTORY_LEN


def test_observe_history_matches_encode_history_of_each_position() -> None:
    # Random play in four games (Python plays every seat); each finished hand is
    # replayed from its seed and every observed history row must equal
    # `encode_history` of the same position.
    env = HkVecEnv(4, 11, min_faan=0)
    rng = np.random.default_rng(5)
    rows: dict[int, list[tuple[np.ndarray, int]]] = {i: [] for i in range(4)}
    checked = 0
    while checked < 8:
        _fb, _ob, mb, ids, _seats = env.observe()
        n = len(ids)
        h = np.frombuffer(env.observe_history(), dtype=np.float32).reshape(n, HISTORY_LEN)
        m = np.frombuffer(mb, dtype=np.uint8).reshape(n, 134)
        actions = [int(rng.choice(np.flatnonzero(m[k]))) for k in range(n)]
        for k, i in enumerate(ids):
            rows[i].append((h[k].copy(), actions[k]))
        env.step(actions)
        for d in env.finished(detail=True):
            s = HkState(seed=d["seed"], min_faan=0)
            for hist, a in rows[d["env"]]:
                assert np.array_equal(hist, np.asarray(encode_history(s), dtype=np.float32))
                s.apply(a)
            assert s.is_terminal()
            rows[d["env"]] = []
            checked += 1
