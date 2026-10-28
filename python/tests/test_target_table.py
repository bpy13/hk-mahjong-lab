"""The target value table read through pybind (plan Task 9)."""

import numpy as np
from mahjong._core import target_value_table


def test_table_shape_and_ranges() -> None:
    t = target_value_table()
    shape = tuple(t["shape"])
    assert t["kinds"] == ["mixed", "all_pungs", "half_flush", "full_flush", "limit"]
    assert shape == (5, 5, 6, 5)
    for name in ("p_win", "p_self_draw"):
        values = np.asarray(t[name])
        assert values.size == np.prod(shape)
        assert ((values >= 0.0) & (values <= 1.0)).all(), name
    assert (np.asarray(t["points"]) >= 0.0).all()


def test_win_chance_falls_with_faan_shanten() -> None:
    t = target_value_table()
    p_win = np.asarray(t["p_win"]).reshape(tuple(t["shape"]))
    # Axis 1 is faan-shanten: a hand closer to a legal win never wins less often.
    steps = np.diff(p_win, axis=1)
    assert (steps <= 1e-12).all()
    # And the table is fitted, not the all-zero placeholder.
    assert p_win[:, 0].mean() > p_win[:, 4].mean()
