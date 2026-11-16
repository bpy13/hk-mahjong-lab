"""Task 2 (docs/plans/2026-11-16-self-play-network.md): the binding plays the `stage` bot."""

from __future__ import annotations

import numpy as np
from mahjong._core import HkState, HkVecEnv, agent_action, teacher_data


def test_stage_plays_200_hands_in_three_seats() -> None:
    """`stage` in three seats and `value` in one: every hand ends, moves are legal."""
    finished = 0
    for seed in range(200):
        s = HkState(seed)
        while not s.is_terminal():
            seat = s.current_player()
            assert seat is not None
            spec = "value" if seat == 0 else "stage"
            a = agent_action(s, spec, seed)
            assert a in s.legal_actions()
            s.apply(a)
        assert s.result() is not None
        finished += 1
    assert finished == 200


def test_teacher_data_logs_stage_decisions() -> None:
    """The logged teacher moves are `stage`'s: legal, reproducible, and the same as
    what `HkVecEnv.agent_actions("stage")` plays (legal on 1,000 decisions)."""
    feats, _o, masks, actions, _r, _h = teacher_data(40, teacher="stage", seed=5)
    n = len(actions)
    assert n >= 1000
    m = np.frombuffer(masks, dtype=np.uint8).reshape(n, 134)
    acts = np.asarray(actions)
    assert m[np.arange(n), acts].all()  # every logged move was legal
    again = teacher_data(40, teacher="stage", seed=5)
    assert list(again[3]) == list(actions) and again[0] == feats  # reproducible
    other = teacher_data(40, teacher="value", seed=5)
    assert list(other[3]) != list(actions)  # a different bot was not logged by mistake

    # The vectorised env asks the same bot through agent_actions: drive 1,000 decisions
    # with it; each must be legal.
    env = HkVecEnv(16, 7)
    checked = 0
    for _ in range(400):
        if checked >= 1000:
            break
        _f, _o2, mb, ids, _seats = env.observe()
        if not ids:
            break
        moves = env.agent_actions("stage", 3)
        mm = np.frombuffer(mb, dtype=np.uint8).reshape(len(ids), 134)
        assert mm[np.arange(len(ids)), moves].all()
        checked += len(ids)
        env.step(moves)
    assert checked >= 1000
