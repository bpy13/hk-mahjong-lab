"""Type stubs for the Rust extension module `mahjong._core`.

A tile is an ``int`` index (docs/rules/hk.md §1): ``1m``–``9m`` = 0–8,
``1p``–``9p`` = 9–17, ``1s``–``9s`` = 18–26, ``E S W N`` = 27–30, ``Wh G R`` = 31–33,
``F1``–``F4`` = 34–37, ``S1``–``S4`` = 38–41.
"""

from typing import Any

__version__: str

def parse_tiles(text: str) -> list[int]:
    """Parse tile text such as ``"123m 55p EEE Wh F2"``. Raises ``ValueError``."""

def format_tiles(tiles: list[int]) -> str:
    """Format tile indices in sorted, grouped notation (``"123m 55p EEE"``)."""

NUM_ACTION_CODES: int

def action_text(code: int) -> str:
    """Describe an action code in English (``"discard 5p"``)."""

def pattern_names() -> list[tuple[int, str, str]]:
    """Faan-table rows as ``(number, chinese, english)``."""

def score_hand(
    concealed: list[int],
    melds: list[tuple[str, int, bool]],
    seat: int,
    win_tile: int | None,
    self_drawn: bool,
    bonus: list[int] = ...,
    last_tile: bool = ...,
    kong_chain: int = ...,
    robbing_kong: bool = ...,
    heavenly: bool = ...,
    earthly: bool = ...,
    **rules: object,
) -> dict[str, Any] | None:
    """Score a complete hand (docs/rules/hk.md §7); ``None`` if incomplete."""

class HkState:
    """A hand of Hong Kong mahjong."""

    def __init__(self, seed: int | None = None, **rules: object) -> None: ...
    @staticmethod
    def from_wall(wall: list[int], **rules: object) -> HkState: ...
    @staticmethod
    def from_position(
        hands: list[list[int]],
        melds: list[list[tuple[str, int, bool, int | None]]],
        river: list[tuple[int, int, bool]],
        wall_remaining: int,
        to_move: int,
        phase: str = "turn",
        claim_tile: int | None = None,
        claim_from: int | None = None,
        last_drawn: int | None = None,
        bonus: list[list[int]] | None = None,
        liable: list[int | None] | None = None,
        kong_chain: int = 0,
        **rules: object,
    ) -> HkState:
        """Build a mid-hand state without a wall (``State::from_position``).

        ``melds`` are ``(kind, tile, concealed, from)`` per seat; ``river`` is
        ``(seat, tile, claimed)`` in discard order; ``phase`` is ``"turn"``,
        ``"claim"`` or ``"rob"`` (the last two name ``claim_tile`` and
        ``claim_from``). Raises ``ValueError`` on a malformed position.
        """
    def current_player(self) -> int | None: ...
    def phase(self) -> str: ...
    def is_terminal(self) -> bool: ...
    def legal_actions(self) -> list[int]: ...
    def apply(self, code: int) -> None: ...
    def copy(self) -> HkState: ...
    def determinize(self, observer: int, seed: int | None = None) -> HkState: ...
    def wall(self) -> list[int]: ...
    def hand(self, seat: int) -> list[int]: ...
    def observation(self, seat: int) -> dict[str, Any]: ...
    def events(self) -> list[dict[str, Any]]: ...
    def result(self) -> dict[str, Any] | None: ...
    def self_drawn_score(self) -> dict[str, Any] | None: ...
    def claim_score(self, seat: int) -> dict[str, Any] | None: ...

def analyze(
    state: HkState,
    samples: int = 400,
    candidates: int = 6,
    time_ms: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """PIMC analysis of the decision for the player to act."""

def agent_action(state: HkState, spec: str, seed: int | None = None) -> int:
    """The action an agent spec (``"value"``, ``"pimc:samples=64"``) chooses."""

def ev_terms(state: HkState, spec: str = "ev") -> dict[str, Any]:
    """The ``ev`` bot's decision for the player to act (``docs/ev-bot.md`` §2.3): ``choice``
    (action code), ``text``, ``candidates`` (best ``ev`` first, every term; per-seat lists of
    four) and ``reasons`` (``[{"key": "reason.ev....", "args": {...}}]``). ``x`` per candidate
    is each opponent's discard-win chance: the belief model's ``outcome`` head with
    ``spec="ev:belief=models/belief.mjnn"``, else ``r_disc × sd``. ``defensive:belief=<model>``
    is also an agent spec (folds on expected loss)."""

def danger(state: HkState, seat: int) -> dict[str, Any]:
    """Heuristic per-tile deal-in danger (``tiles``) and per-seat ``threat``."""

def hand_progress(state: HkState, seat: int) -> dict[str, Any]:
    """Plain and faan-aware shanten of ``seat``'s hand, with the target plan."""

def target_value_table() -> dict[str, Any]:
    """The fitted target value table: ``kinds``, ``shape`` (kind, faan-shanten, tiles-left
    bucket, melds) and flat ``p_win``, ``p_self_draw``, ``points`` lists in row-major order."""

def discard_table(state: HkState) -> list[dict[str, Any]]:
    """Every discard's shanten, acceptance and target, for a turn decision."""

def solo(
    state: HkState, draws: int | None = None, samples: int = 200, seed: int = 0
) -> list[dict[str, Any]]:
    """Offense-only win estimates per discard."""

def tournament(
    agents: list[str],
    deals: int,
    seed: int = 0,
    seatings: int = 4,
    bootstrap: int = 2000,
    **rules: object,
) -> dict[str, Any]:
    """Duplicate tournament between four agent specs, in 4 or 24 seatings per wall.

    Returns the ``hk-tournament/1`` report: per-agent points with independent and
    wall-paired bootstrap intervals, rates, pattern shares, pairwise differences.
    """

def feature_sizes() -> tuple[int, int, int]:
    """``(STATE_LEN, ORACLE_LEN, NUM_ACTION_CODES)``."""

def encode_state(state: HkState) -> tuple[list[float], list[float], list[int]]:
    """Features, oracle features and legal-action mask for the player to act."""

def teacher_data(
    hands: int, teacher: str = ..., seed: int = ..., **rules: object
) -> tuple[bytes, bytes, bytes, list[int], list[float], list[int]]:
    """Teacher self-play decisions: features, oracle, masks, actions, returns, hand index."""

class HkVecEnv:
    """Many hands stepped in lockstep for learning."""

    def __init__(
        self, n: int, seed: int = 0, opponents: str | None = None, **rules: object
    ) -> None: ...
    def observe(self) -> tuple[bytes, bytes, bytes, list[int], list[int]]: ...
    def step(self, actions: list[int]) -> None: ...
    def finished(self) -> list[tuple[int, list[int], int]]: ...
