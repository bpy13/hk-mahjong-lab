"""The Rust engine and the independent Python reference agree, step by step.

At every decision both must agree on who acts and on the full set of legal
actions; at the end, on every winner's faan and patterns and on the payments.
"""

import random

from mahjong._core import HkState, score_hand

from .reference.hk_reference import (
    CLAIM_KONG,
    PUNG,
    SELF_DRAWN_WIN,
    WIN,
    Ctx,
    Meld,
    RefGame,
    Rules,
    all_tiles,
    score,
)


def random_rules(rng: random.Random) -> Rules:
    return Rules(
        flowers=rng.random() < 0.5,
        min_faan=rng.randint(0, 3),
        faan_cap=rng.choice([10, 13]),
        payout=rng.choice(["half_spicy", "full_spicy"]),
        multiple_winners=rng.choice(["all", "head_bump"]),
        twelve_tile_liability=rng.random() < 0.5,
        prevailing_wind=rng.randint(0, 3),
    )


def eager(legal: list[int], rng: random.Random) -> int:
    """Mostly takes wins, kongs and claims so that rare rules get exercised."""

    def rank(a: int) -> int:
        if a in (WIN, SELF_DRAWN_WIN):
            return 0
        if 35 <= a <= 102 or a == CLAIM_KONG:
            return 1
        if a == PUNG or a >= 107:
            return 2
        return 3

    if rng.random() < 0.85:
        best = min(map(rank, legal))
        if best < 3:
            return rng.choice([a for a in legal if rank(a) == best])
    return rng.choice(legal)


def play_both(seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    rules = random_rules(rng)
    wall = all_tiles(rules.flowers)
    rng.shuffle(wall)
    kwargs = {k: getattr(rules, k) for k in rules.__dataclass_fields__}
    rust = HkState.from_wall(wall, **kwargs)
    ref = RefGame(wall, rules)
    steps = 0
    while True:
        assert rust.current_player() == ref.current_player(), (seed, steps)
        if ref.current_player() is None:
            break
        legal = sorted(rust.legal_actions())
        assert legal == sorted(ref.legal()), (seed, steps, legal, sorted(ref.legal()))
        a = eager(legal, rng)
        rust.apply(a)
        ref.apply(a)
        steps += 1
    result = rust.result()
    assert result is not None and ref.result is not None
    assert result["deltas"] == ref.result["deltas"], seed
    got = [
        (w["seat"], w["score"]["faan"], sorted(p for p, _ in w["score"]["patterns"]), w["paid_by"])
        for w in result["wins"]
    ]
    want = [(w["seat"], w["faan"], w["patterns"], w["paid_by"]) for w in ref.result["wins"]]
    assert got == want, seed
    return steps, len(result["wins"])


def test_engine_matches_reference() -> None:
    total_wins = 0
    for seed in range(3_000):
        _, wins = play_both(seed)
        total_wins += wins
    assert total_wins > 300


# ----- scoring on constructed hands -----


def random_complete_hand(rng: random.Random):
    """Four random melds and a pair (respecting 4 copies), biased toward the
    shapes that make rare patterns: one suit, honours, terminals, kongs."""
    theme = rng.choice(["any", "suit", "honours", "terminals", "orphans", "gates"])
    if theme == "orphans":
        orphans = [0, 8, 9, 17, 18, 26, *range(27, 34)]
        tiles = [*orphans, rng.choice(orphans)]
        return tiles, [], tiles[-1]
    if theme == "gates":
        s = 9 * rng.randrange(3)
        w = s + rng.randrange(9)
        tiles = [s, s, s, *range(s + 1, s + 8), s + 8, s + 8, s + 8, w]
        return tiles, [], w
    suit = 9 * rng.randrange(3)
    pool = {
        "any": list(range(34)),
        "suit": [*range(suit, suit + 9), *range(27, 34)]
        if rng.random() < 0.5
        else list(range(suit, suit + 9)),
        "honours": list(range(27, 34)) + ([0, 8, 9, 17, 18, 26] if rng.random() < 0.5 else []),
        "terminals": [0, 8, 9, 17, 18, 26] + (list(range(27, 34)) if rng.random() < 0.3 else []),
    }[theme]
    used = [0] * 34
    concealed: list[int] = []
    melds: list[tuple[str, int, bool]] = []
    while len(melds) + len(concealed) // 3 < 4:
        t = rng.choice(pool)
        kind = rng.choice(["pung", "pung", "chow", "kong"])
        if kind == "chow" and (t >= 27 or t % 9 > 6 or not {t + 1, t + 2} <= set(pool)):
            kind = "pung"
        group = [t, t + 1, t + 2] if kind == "chow" else [t] * (4 if kind == "kong" else 3)
        if any(used[x] + group.count(x) > 4 for x in set(group)):
            continue
        for x in group:
            used[x] += 1
        where = rng.random()
        if kind == "kong":
            melds.append(("kong", t, where < 0.5))
        elif where < 0.35:
            melds.append((kind, t, False))
        else:
            concealed += group
    pair = rng.choice([x for x in pool if used[x] <= 2] or [x for x in range(34) if used[x] <= 2])
    concealed += [pair, pair]
    return concealed, melds, rng.choice(concealed)


def test_scoring_matches_reference_on_constructed_hands() -> None:
    rng = random.Random(1)
    seen: set[int] = set()
    for _ in range(20_000):
        rules = random_rules(rng)
        concealed, melds, win_tile = random_complete_hand(rng)
        self_drawn = rng.random() < 0.5
        bonus = (
            sorted(rng.sample(range(34, 42), rng.choice([0, 0, 1, 2, 4, 5])))
            if rules.flowers
            else []
        )
        kw = dict(
            last_tile=rng.random() < 0.1,
            kong_chain=rng.choice([0, 0, 0, 1, 2]) if self_drawn else 0,
            robbing_kong=not self_drawn and rng.random() < 0.1,
            heavenly=self_drawn and rng.random() < 0.03,
            earthly=not self_drawn and rng.random() < 0.03,
        )
        seat = rng.randrange(4)
        rule_kw = {k: getattr(rules, k) for k in rules.__dataclass_fields__}
        got = score_hand(concealed, melds, seat, win_tile, self_drawn, bonus, **kw, **rule_kw)
        counts = [concealed.count(t) for t in range(34)]
        ref_melds = [Meld(k, t, concealed=c and k == "kong") for k, t, c in melds]
        ctx = Ctx(
            seat,
            win_tile,
            self_drawn,
            frozenset(bonus),
            kw["last_tile"],
            kw["kong_chain"],
            kw["robbing_kong"],
            kw["heavenly"],
            kw["earthly"],
        )
        want = score(counts, ref_melds, ctx, rules)
        assert (got is None) == (want is None), (concealed, melds)
        if got is None or want is None:
            continue
        got_t = (got["faan"], got["total"], got["limit"], sorted(got["patterns"]))
        assert got_t == (want[0], want[1], want[2], sorted(want[3])), (
            concealed,
            melds,
            win_tile,
            ctx,
        )
        seen |= {p for p, _ in got["patterns"]}
    # Every hand-based pattern is exercised (31 and 32 are not hand-based).
    assert set(range(1, 31)) <= seen, sorted(set(range(1, 31)) - seen)


# ----- play from clumped walls -----


def clumped_wall(rng: random.Random, flowers: bool) -> list[int]:
    """Copies of a tile lie together (in random kind order), then a few random
    swaps: hands get many pungs and kongs, so rare rules come up often."""
    kinds = list(range(34))
    rng.shuffle(kinds)
    wall = [t for t in kinds for _ in range(4)]
    if flowers:
        for b in range(34, 42):
            wall.insert(rng.randrange(len(wall) + 1), b)
    for _ in range(rng.randrange(10, 80)):
        i, j = rng.randrange(len(wall)), rng.randrange(len(wall))
        wall[i], wall[j] = wall[j], wall[i]
    return wall


def test_engine_matches_reference_on_clumped_walls() -> None:
    multi = 0
    for seed in range(1_500):
        rng = random.Random(10_000 + seed)
        rules = random_rules(rng)
        wall = clumped_wall(rng, rules.flowers)
        kwargs = {k: getattr(rules, k) for k in rules.__dataclass_fields__}
        rust = HkState.from_wall(wall, **kwargs)
        ref = RefGame(wall, rules)
        while ref.current_player() is not None:
            assert rust.current_player() == ref.current_player(), seed
            legal = sorted(rust.legal_actions())
            assert legal == sorted(ref.legal()), (seed, legal, sorted(ref.legal()))
            a = eager(legal, rng)
            rust.apply(a)
            ref.apply(a)
        assert rust.is_terminal()
        result = rust.result()
        assert result is not None and ref.result is not None
        assert result["deltas"] == ref.result["deltas"], seed
        multi += len(result["wins"]) > 1
    # Several winners on one discard come up here; robbing a kong is too rare for
    # random play and is covered by crates/hk/tests/scenarios.rs.
    assert multi > 0
