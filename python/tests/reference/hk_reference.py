"""An independent, deliberately simple pure-Python implementation of the rules in
docs/rules/hk.md, used as an oracle for the Rust engine (test_differential.py).

It shares no code with the engine. It is written for clarity rather than speed:
the win check is a plain recursive search, and the faan table is a list of rules
evaluated one by one. Action codes and the order in which seats are asked to
declare claims follow the engine's documented interface (mahjong._core).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ----- tiles (§1) -----
WINDS = range(27, 31)
DRAGONS = range(31, 34)
BONUS = range(34, 42)
ALL_BONUS = frozenset(BONUS)


def is_suited(t: int) -> bool:
    return t < 27


def is_honour(t: int) -> bool:
    return 27 <= t < 34


def is_terminal(t: int) -> bool:
    return t < 27 and t % 9 in (0, 8)


def suit_of(t: int) -> int:
    return t // 9


def all_tiles(flowers: bool) -> list[int]:
    tiles = [t for t in range(34) for _ in range(4)]
    if flowers:
        tiles += list(BONUS)
    return sorted(tiles)


# ----- action codes (engine interface) -----
SELF_DRAWN_WIN, PASS, WIN, PUNG, CLAIM_KONG = 34, 103, 104, 105, 106


def discard(t: int) -> int:
    return t


def concealed_kong(t: int) -> int:
    return 35 + t


def added_kong(t: int) -> int:
    return 69 + t


def chow(low: int) -> int:
    return 107 + low


def chow_ban(claimed: int, low: int) -> set[int]:
    """§4.1.4 食換 after chowing `claimed` into the chow from `low`: the claimed
    kind, and the kind that re-forms the chow with the two hand tiles."""
    ban = {claimed}
    if claimed == low and low % 9 <= 5:
        ban.add(low + 3)
    elif claimed == low + 2 and low % 9 >= 1:
        ban.add(low - 1)
    return ban


# ----- rules -----
@dataclass(frozen=True)
class Rules:
    flowers: bool = True
    min_faan: int = 3
    faan_cap: int = 10
    payout: str = "half_spicy"
    multiple_winners: str = "all"
    twelve_tile_liability: bool = True
    prevailing_wind: int = 0
    passed_win_lock: bool = True

    def points(self, faan: int) -> int:
        """§8.1."""
        if self.payout == "full_spicy":
            return 2**faan
        if faan <= 4:
            return 2**faan
        # Above 4 faan each step adds half of the previous doubling.
        value, step = 16, 8
        for f in range(5, faan + 1):
            value += step
            if f % 2 == 0:
                step *= 2
        return value


# ----- complete hands (§6) -----
def standard_splits(counts: list[int]) -> list[tuple[int, list[tuple[str, int]]]]:
    """All (pair, groups) splits of a concealed part into melds and one pair."""
    out: set[tuple[int, tuple[tuple[str, int], ...]]] = set()

    def go(c: list[int], pair: int | None, groups: list[tuple[str, int]]) -> None:
        first = next((t for t in range(34) if c[t]), None)
        if first is None:
            if pair is not None:
                out.add((pair, tuple(sorted(groups))))
            return
        if pair is None and c[first] >= 2:
            c[first] -= 2
            go(c, first, groups)
            c[first] += 2
        if c[first] >= 3:
            c[first] -= 3
            go(c, pair, [*groups, ("pung", first)])
            c[first] += 3
        if is_suited(first) and first % 9 <= 6 and c[first + 1] and c[first + 2]:
            for k in range(3):
                c[first + k] -= 1
            go(c, pair, [*groups, ("chow", first)])
            for k in range(3):
                c[first + k] += 1

    go(list(counts), None, [])
    return [(p, list(g)) for p, g in sorted(out)]


def thirteen_orphans(counts: list[int]) -> bool:
    orphans = [t for t in range(34) if is_terminal(t) or is_honour(t)]
    return (
        sum(counts) == 14
        and all(counts[t] >= 1 for t in orphans)
        and all(counts[t] == 0 for t in range(34) if t not in orphans)
    )


def complete(counts: list[int], n_melds: int) -> bool:
    if sum(counts) + 3 * n_melds != 14:
        return False
    if n_melds == 0 and thirteen_orphans(counts):
        return True
    return bool(standard_splits(counts))


# ----- faan (§7) -----
@dataclass
class Meld:
    kind: str  # "chow" | "pung" | "kong"
    tile: int
    concealed: bool = False
    source: int | None = None

    def tiles(self) -> list[int]:
        if self.kind == "chow":
            return [self.tile, self.tile + 1, self.tile + 2]
        return [self.tile] * (4 if self.kind == "kong" else 3)


@dataclass
class Ctx:
    seat: int
    win_tile: int | None
    self_drawn: bool
    bonus: frozenset[int] = frozenset()
    last_tile: bool = False
    kong_chain: int = 0
    robbing: bool = False
    heavenly: bool = False
    earthly: bool = False
    human: bool = False


def flower_faan(bonus: frozenset[int], seat: int, rules: Rules) -> list[tuple[int, int]]:
    if not rules.flowers:
        return []
    if not bonus:
        return [(3, 1)]
    out = []
    seat_count = 0
    for base in (34, 38):  # flowers, seasons
        group = {base + i for i in range(4)}
        if group <= bonus:
            out.append((5, 2))
        elif base + seat in bonus:
            seat_count += 1
    if seat_count:
        out.append((4, seat_count))
    return out


def qualifies(s: tuple[int, int, bool, list[tuple[int, int]]] | None, rules: Rules) -> bool:
    """§6: a limit hand, or the minimum reached without bonus-tile faan (#3-#5)."""
    if s is None:
        return False
    faan, _, limit, patterns = s
    return (
        limit
        or min(rules.faan_cap, sum(f for n, f in patterns if n not in (3, 4, 5))) >= rules.min_faan
    )


def _rank(s: tuple[int, int, bool, list[tuple[int, int]]], rules: Rules) -> tuple[int, int, int]:
    """Which reading of a hand counts: most faan toward the minimum, then most paid."""
    faan, total, limit, patterns = s
    q = (
        rules.faan_cap
        if limit
        else min(rules.faan_cap, sum(f for n, f in patterns if n not in (3, 4, 5)))
    )
    return (q, faan, total)


def score(
    concealed: list[int], melds: list[Meld], ctx: Ctx, rules: Rules
) -> tuple[int, int, bool, list[tuple[int, int]]] | None:
    """(faan, total, limit, patterns) of the best reading, or None if incomplete."""
    if sum(concealed) + 3 * len(melds) != 14:
        return None
    cap = rules.faan_cap
    specials = [n for n, on in ((29, ctx.heavenly), (30, ctx.earthly), (33, ctx.human)) if on]
    if not melds and thirteen_orphans(concealed):
        return (cap, cap, True, [(n, cap) for n in [*specials, 25]])
    best = None
    for pair, groups in standard_splits(concealed):
        result = score_split(concealed, melds, pair, groups, ctx, rules, specials)
        if best is None or _rank(result, rules) > _rank(best, rules):
            best = result
    return best


def score_split(concealed, melds, pair, groups, ctx, rules, specials):
    cap = rules.faan_cap
    # (is_pung_or_kong, is_kong, tile, concealed) for all four melds
    all_melds = [(m.kind != "chow", m.kind == "kong", m.tile, m.concealed) for m in melds]
    all_melds += [(kind == "pung", False, tile, True) for kind, tile in groups]
    used = list(concealed)
    for m in melds:
        for t in m.tiles():
            used[t] += 1
    kinds = [t for t in range(34) if used[t]]
    suits = {suit_of(t) for t in kinds if is_suited(t)}
    has_honour = any(is_honour(t) for t in kinds)
    has_terminal = any(is_terminal(t) for t in kinds)
    only_th = all(is_terminal(t) or is_honour(t) for t in kinds)
    pung_tiles = [tile for pung, _, tile, _ in all_melds if pung]
    dragon_pungs = sum(1 for t in pung_tiles if t in DRAGONS)
    wind_pungs = sum(1 for t in pung_tiles if t in WINDS)
    kongs = sum(1 for _, kong, _, _ in all_melds if kong)

    hidden_pungs = sum(1 for pung, _, _, conc in all_melds if pung and conc)
    if not ctx.self_drawn and ctx.win_tile is not None:
        w = ctx.win_tile
        in_chow = any(
            kind == "chow" and w - 2 <= tile <= w and suit_of(tile) == suit_of(w)
            for kind, tile in groups
            if is_suited(w)
        )
        if pair != w and not in_chow and ("pung", w) in groups:
            hidden_pungs -= 1

    limits = list(specials)
    if wind_pungs == 4:
        limits.append(22)
    elif wind_pungs == 3 and pair in WINDS:
        limits.append(21)
    if not suits:
        limits.append(23)
    if only_th and not has_honour:
        limits.append(24)
    if not melds and ctx.win_tile is not None and is_suited(ctx.win_tile):
        s = suit_of(ctx.win_tile)
        gates = [3, 1, 1, 1, 1, 1, 1, 1, 3]
        if ctx.self_drawn:
            # #26 on a self-draw: the final shape is enough (ADR 0006 point 15).
            row = concealed[9 * s : 9 * s + 9]
            covers = all(a >= b for a, b in zip(row, gates, strict=True))
            if sum(row) == 14 == sum(concealed) and covers:
                limits.append(26)
        else:
            before = list(concealed)
            before[ctx.win_tile] -= 1
            if (
                before[ctx.win_tile] >= 0
                and sum(before) == 13
                and before[9 * s : 9 * s + 9] == gates
            ):
                limits.append(26)
    if hidden_pungs == 4:
        limits.append(27)
    if kongs == 4:
        limits.append(28)
    if limits:
        return (cap, cap, True, [(n, cap) for n in limits])

    p: list[tuple[int, int]] = []
    if ctx.kong_chain >= 2:
        p.append((8, 10))  # §7.1 #8: 10 faan (ADR 0005)
    else:
        if ctx.self_drawn:
            p.append((1, 1))
        if ctx.kong_chain == 1:
            p.append((7, 1))
    if all(m.concealed for m in melds):
        p.append((2, 1))
    p += flower_faan(ctx.bonus, ctx.seat, rules)
    if ctx.last_tile:
        p.append((6, 1))
    if ctx.robbing:
        p.append((9, 1))
    hand_patterns: list[tuple[int, int]] = []
    if dragon_pungs == 3:
        hand_patterns.append((20, 8))
    elif dragon_pungs == 2 and pair in DRAGONS:
        hand_patterns.append((19, 5))
    elif dragon_pungs:
        hand_patterns.append((10, dragon_pungs))
    if 27 + ctx.seat in pung_tiles:
        hand_patterns.append((11, 1))
    if 27 + rules.prevailing_wind in pung_tiles:
        hand_patterns.append((12, 1))
    if all(not pung for pung, _, _, _ in all_melds):
        hand_patterns.append((14, 1))
    if all(pung for pung, _, _, _ in all_melds):
        hand_patterns.append((15, 3))
    if len(suits) == 1:
        hand_patterns.append((16, 3) if has_honour else (17, 7))
    if only_th and has_honour and has_terminal:
        hand_patterns.append((18, 1))
    p += hand_patterns or [(13, 0)]
    total = sum(f for _, f in p)
    return (min(total, cap), total, False, p)


# ----- the hand (§3–§9) -----
@dataclass
class Window:
    tile: int
    source: int
    concealed_kong: bool
    pending: set[int] = field(default_factory=set)
    declared: dict[int, int] = field(default_factory=dict)


class RefGame:
    """A hand of Hong Kong mahjong, dealt from a given wall."""

    def __init__(self, wall: list[int], rules: Rules) -> None:
        assert sorted(wall) == all_tiles(rules.flowers)
        self.rules = rules
        self.wall = list(wall)
        self.front, self.back = 0, len(wall)
        self.hands = [[0] * 34 for _ in range(4)]
        self.melds: list[list[Meld]] = [[] for _ in range(4)]
        self.bonus: list[set[int]] = [set() for _ in range(4)]
        self.phase = "turn"
        self.current = self.turn = 0
        self.drew = True
        self.last_drawn: int | None = None
        self.drew_last = False
        self.kong_chain = 0
        self.dealer_first = True
        self.discards = 0
        self.window: Window | None = None
        self.liable: list[int | None] = [None] * 4
        # §8.3.4: seats whose liability is 包大三元 rather than the 12-tile rule.
        self.dragon_liable: set[int] = set()
        # §4.1.4 食換: kinds the claimer may not discard this turn.
        self.ban: set[int] = set()
        # §7.5.1: while the flower choice is open, what to resume after a pass.
        self.flower_resume: tuple | None = None
        # §3.3: the deal's set-aside, seat by seat: (seat, replacements owed).
        self.deal_seat = 0
        self.deal_left = 0
        self.dealing = True
        # §4.4: tiles each seat passed a win on since its turn last started.
        self.locked: list[set[int]] = [set() for _ in range(4)]
        self.result: dict | None = None
        self.claimed_any = False
        self.thrown: list[set[int]] = [set() for _ in range(4)]
        dealt = []
        for seat in range(4):
            n = 14 if seat == 0 else 13
            dealt.append(self.wall[self.front : self.front + n])
            self.front += n
        for seat in range(4):
            for t in dealt[seat]:
                if t < 34:
                    self.hands[seat][t] += 1
        # §3.3: each seat sets aside all its bonus tiles at once (so all eight
        # can arrive together, #31), then draws the replacements.
        self.dealt = dealt
        if not self._set_aside(0):
            self._deal_on()

    def _set_aside(self, seat: int) -> bool:
        set_aside = [t for t in self.dealt[seat] if t >= 34]
        self.bonus[seat].update(set_aside)
        self.deal_seat, self.deal_left = seat, len(set_aside)
        return self._flowers(seat, drawn=False)

    def _deal_on(self) -> None:
        while True:
            if self.deal_left > 0:
                self.deal_left -= 1
                if not self._draw(self.deal_seat, from_back=True):
                    return
                continue
            if self.deal_seat == 3:
                break
            if self._set_aside(self.deal_seat + 1):
                return
        self.dealing = False
        self.last_drawn = None
        self.phase = "turn"
        self.current = self.turn = 0

    # -- draws and bonus tiles --
    def _place_bonus(self, seat: int, tile: int) -> bool:
        self.bonus[seat].add(tile)
        return self._flowers(seat, drawn=True)

    def _flower_score(self, seat: int, drawn: bool):
        """§7.5: 花糊 (#32, 3) with seven, 大花糊 (#31, 10) with eight, plus
        海底 (#6) and 槓上 (#7, or #8 at 10) for the moment (§7.5.2)."""
        n = len(self.bonus[seat])
        p = [(31, 10) if n == 8 else (32, 3)]
        if self.kong_chain == 1:
            p.append((7, 1))
        elif self.kong_chain >= 2:
            p.append((8, 10))
        if drawn and self.front == self.back:
            p.append((6, 1))
        total = sum(f for _, f in p)
        faan = min(total, self.rules.faan_cap)
        return (faan, total, False, p)

    def _flowers(self, seat: int, drawn: bool) -> bool:
        """§7.5: all eight bonus tiles win 大花糊 at once; the seventh opens the
        choice of 花糊 or play on (§7.5.1). Nobody wins on another seat's bonus
        tile. True if the hand ended or waits for the choice."""
        if not self.rules.flowers:
            return False
        n = len(self.bonus[seat])
        if n == 8:
            self._finish([(seat, self._flower_score(seat, drawn), True, None, None)], flower=True)
            return True
        if n == 7:
            self.phase = "flower"
            self.current = seat
            self.window = None
            self.flower_resume = (seat, drawn)
            return True
        return False

    def _draw(self, seat: int, from_back: bool) -> bool:
        while True:
            if self.front == self.back:
                self._end_draw()
                return False
            if from_back:
                self.back -= 1
                tile = self.wall[self.back]
            else:
                tile = self.wall[self.front]
                self.front += 1
            if tile >= 34:
                if self._place_bonus(seat, tile):
                    return False
                from_back = True
                continue
            self.hands[seat][tile] += 1
            self.last_drawn = tile
            self.drew_last = self.front == self.back
            return True

    def _end_draw(self) -> None:
        self.phase = "over"
        self.window = None
        self.result = {"wins": [], "deltas": [0, 0, 0, 0]}

    def remaining(self) -> int:
        return self.back - self.front

    # -- wins --
    def _ctx(self, seat: int, tile: int | None, self_drawn: bool) -> Ctx:
        w = self.window
        return Ctx(
            seat=seat,
            win_tile=tile,
            self_drawn=self_drawn,
            bonus=frozenset(self.bonus[seat]),
            last_tile=self_drawn and self.drew_last,
            kong_chain=self.kong_chain if self_drawn else 0,
            robbing=self.phase == "rob",
            heavenly=self_drawn and seat == 0 and self.dealer_first,
            earthly=(
                not self_drawn
                and self.phase == "claim"
                and self.discards == 1
                and w is not None
                and w.source == 0
            ),
            # #33 人糊: a non-dealer's self-draw on its first turn, nothing claimed yet.
            human=self_drawn and seat != 0 and self.discards == seat and not self.claimed_any,
        )

    def _claim_win(self, seat: int, tile: int):
        c = list(self.hands[seat])
        if c[tile] >= 4:
            return None
        c[tile] += 1
        if not complete(c, len(self.melds[seat])):
            return None
        # §5.1: only thirteen orphans may rob a concealed kong.
        robbing_concealed = self.phase == "rob" and self.window and self.window.concealed_kong
        if robbing_concealed and (self.melds[seat] or not thirteen_orphans(c)):
            return None
        s = score(c, self.melds[seat], self._ctx(seat, tile, False), self.rules)
        return s if qualifies(s, self.rules) else None

    def _self_win(self):
        if self.phase != "turn" or not self.drew:
            return None
        seat = self.current
        c = self.hands[seat]
        if not complete(c, len(self.melds[seat])):
            return None
        s = score(c, self.melds[seat], self._ctx(seat, self.last_drawn, True), self.rules)
        return s if qualifies(s, self.rules) else None

    def _finish(self, wins, flower: bool = False) -> None:
        deltas = [0, 0, 0, 0]
        records = []
        for seat, sc, self_drawn, source, _tile in wins:
            share = self.rules.points(sc[0])
            paid = [0, 0, 0, 0]
            # §8.3: the liable seat covers a self-drawn win only (包自摸); 包大三元
            # not a win on a kong's replacement tile (§8.3.4).
            liable = None if flower or not self_drawn else self.liable[seat]
            if seat in self.dragon_liable and self.kong_chain > 0:
                liable = None
            if liable is not None:
                paid[liable] = 3 * share
            elif self_drawn:
                paid = [0 if p == seat else share for p in range(4)]
            else:
                # §8.2 (ADR 0005): the discarder pays 2 shares; the maker of a
                # robbed kong pays the whole self-draw, 3.
                paid[source] = (3 if self.phase == "rob" else 2) * share
            for p in range(4):
                deltas[p] -= paid[p]
                deltas[seat] += paid[p]
            records.append(
                {
                    "seat": seat,
                    "faan": sc[0],
                    "patterns": sorted(n for n, _ in sc[3]),
                    "paid_by": paid,
                }
            )
        self.phase = "over"
        self.window = None
        self.result = {"wins": records, "deltas": deltas}

    # -- claims --
    def _options(self, seat: int) -> list[int]:
        w = self.window
        if w is None or seat == w.source:
            return []
        out = []
        if w.tile not in self.locked[seat] and self._claim_win(seat, w.tile):
            out.append(WIN)
        if self.phase == "rob" or self.remaining() == 0:
            return out
        have = self.hands[seat][w.tile]
        hand = self.hands[seat]

        def can_discard_after(used: list[int], banned: set[int]) -> bool:
            # §4.1.4: the claim is offered only if a legal discard remains.
            rest = list(hand)
            for x in used:
                rest[x] -= 1
            return any(rest[x] and x not in banned for x in range(34))

        if have >= 2 and can_discard_after([w.tile] * 2, {w.tile}):
            out.append(PUNG)
        if have >= 3:
            out.append(CLAIM_KONG)
        if seat == (w.source + 1) % 4 and is_suited(w.tile):
            t = w.tile
            for low in (t - 2, t - 1, t):
                if low < 0 or suit_of(low) != suit_of(t) or low % 9 > 6:
                    continue
                if all(x == t or hand[x] for x in (low, low + 1, low + 2)):
                    used = [x for x in (low, low + 1, low + 2) if x != t]
                    if can_discard_after(used, chow_ban(t, low)):
                        out.append(chow(low))
        return out

    def _order(self, source: int) -> list[int]:
        return [(source + k) % 4 for k in (1, 2, 3)]

    def _open(self, phase: str, tile: int, source: int, concealed: bool) -> bool:
        self.phase = phase
        self.window = Window(tile, source, concealed)
        pending = {s for s in self._order(source) if self._options(s)}
        if not pending:
            self.window = None
            return False
        self.window.pending = pending
        self.current = next(s for s in self._order(source) if s in pending)
        return True

    def _resolve(self) -> None:
        w = self.window
        assert w is not None
        order = self._order(w.source)
        winners = [s for s in order if w.declared.get(s) == WIN]
        if self.rules.multiple_winners == "head_bump":
            winners = winners[:1]
        if winners:
            wins = [(s, self._claim_win(s, w.tile), False, w.source, w.tile) for s in winners]
            if self.phase == "rob":
                self._undo_kong(w.source, w.tile, w.concealed_kong)
            self._finish(wins)
            return
        if self.rules.passed_win_lock:
            # §4.4: a seat offered the win that did not take it is locked on the tile.
            for s in order:
                if s in w.declared and WIN in self._options(s):
                    self.locked[s].add(w.tile)
        if self.phase == "rob":
            self._kong_draw(w.source)
            return
        for s in order:
            if w.declared.get(s) in (PUNG, CLAIM_KONG):
                self._claim(s, w.source, w.tile, w.declared[s])
                return
        for s in order:
            a = w.declared.get(s)
            if a is not None and a >= 107:
                self._claim(s, w.source, w.tile, a)
                return
        if self.remaining() == 0:
            self._end_draw()
        else:
            self._start_turn((w.source + 1) % 4)

    def _undo_kong(self, seat: int, tile: int, concealed: bool) -> None:
        for i, m in enumerate(self.melds[seat]):
            if m.tile == tile and m.kind == "kong":
                if concealed:
                    del self.melds[seat][i]
                    self.hands[seat][tile] += 3
                else:
                    m.kind = "pung"
                return

    def _claim(self, seat: int, source: int, tile: int, action: int) -> None:
        if action == PUNG:
            self.hands[seat][tile] -= 2
            meld = Meld("pung", tile, source=source)
        elif action == CLAIM_KONG:
            self.hands[seat][tile] -= 3
            meld = Meld("kong", tile, source=source)
        else:
            low = action - 107
            for x in (low, low + 1, low + 2):
                if x != tile:
                    self.hands[seat][x] -= 1
            meld = Meld("chow", low, source=source)
        self.locked[seat].clear()  # §4.4: the claimer's turn starts
        self.claimed_any = True
        exposed = sum(1 for m in self.melds[seat] if not m.concealed)
        # §8.3.3: no 12-tile liability for a fourth set that is a pung of a tile
        # the claimer discarded earlier. A later liability replaces an earlier.
        own = action == PUNG and tile in self.thrown[seat]
        if self.rules.twelve_tile_liability and exposed + 1 == 4 and not own:
            self.liable[seat] = source
            self.dragon_liable.discard(seat)
        dragons = sum(1 for m in self.melds[seat] if m.kind != "chow" and m.tile in DRAGONS)
        if action in (PUNG, CLAIM_KONG) and tile in DRAGONS and dragons == 2:
            self.liable[seat] = source  # §8.3.4 包大三元
            self.dragon_liable.add(seat)
        if action == PUNG:
            self.ban = {tile}
        elif action != CLAIM_KONG:
            self.ban = chow_ban(tile, action - 107)
        self.melds[seat].append(meld)
        if action == CLAIM_KONG:
            self.kong_chain = 0
            self._kong_draw(seat)
        else:
            self.turn = self.current = seat
            self.phase = "turn"
            self.window = None
            self.kong_chain = 0
            self.drew = False
            self.drew_last = False

    # -- turns --
    def _start_turn(self, seat: int) -> None:
        self.turn = self.current = seat
        self.phase = "turn"
        self.window = None
        self.kong_chain = 0
        self.drew = True
        self.drew_last = False
        self.ban = set()
        self.locked[seat].clear()  # §4.4: a new turn ends the lock
        self._draw(seat, from_back=False)

    def _kong_draw(self, seat: int) -> None:
        self.kong_chain += 1
        self.turn = self.current = seat
        self.phase = "turn"
        self.window = None
        self.drew = True
        self.ban = set()
        self._draw(seat, from_back=True)

    # -- interface --
    def current_player(self) -> int | None:
        return None if self.phase == "over" else self.current

    def legal(self) -> list[int]:
        if self.phase == "over":
            return []
        seat = self.current
        if self.phase == "flower":
            return [SELF_DRAWN_WIN, PASS]
        if self.phase in ("claim", "rob"):
            return [*self._options(seat), PASS]
        out = []
        if self._self_win():
            out.append(SELF_DRAWN_WIN)
        hand = self.hands[seat]
        if self.remaining() > 0:
            out += [concealed_kong(t) for t in range(34) if hand[t] == 4]
            out += [
                added_kong(m.tile) for m in self.melds[seat] if m.kind == "pung" and hand[m.tile]
            ]
        out += [discard(t) for t in range(34) if hand[t] and t not in self.ban]
        return out

    def apply(self, a: int) -> None:
        assert a in self.legal(), (a, self.legal())
        seat = self.current
        if self.phase == "flower":
            assert self.flower_resume is not None
            who, drawn = self.flower_resume
            self.flower_resume = None
            if a == SELF_DRAWN_WIN:
                self._finish([(who, self._flower_score(who, drawn), True, None, None)], flower=True)
                return
            self.phase = "turn"
            if drawn and not self._draw(who, from_back=True):
                return
            if self.dealing:
                self._deal_on()
            return
        if self.phase in ("claim", "rob"):
            w = self.window
            assert w is not None
            w.declared[seat] = a
            w.pending.discard(seat)
            if not w.pending:
                self._resolve()
            else:
                self.current = next(s for s in self._order(w.source) if s in w.pending)
            return
        if a == SELF_DRAWN_WIN:
            self._finish([(seat, self._self_win(), True, None, self.last_drawn)])
        elif 35 <= a <= 68:
            t = a - 35
            self.hands[seat][t] -= 4
            self.melds[seat].append(Meld("kong", t, concealed=True))
            self.dealer_first = False
            if not self._open("rob", t, seat, True):
                self._kong_draw(seat)
        elif 69 <= a <= 102:
            t = a - 69
            self.hands[seat][t] -= 1
            next(m for m in self.melds[seat] if m.kind == "pung" and m.tile == t).kind = "kong"
            self.dealer_first = False
            if not self._open("rob", t, seat, False):
                self._kong_draw(seat)
        else:
            self.hands[seat][a] -= 1
            self.thrown[seat].add(a)
            self.ban = set()
            self.discards += 1
            self.dealer_first = False
            self.kong_chain = 0
            self.drew = False
            if not self._open("claim", a, seat, False):
                if self.remaining() == 0:
                    self._end_draw()
                else:
                    self._start_turn((seat + 1) % 4)
