//! The hand as a state machine (§3–§9).
//!
//! A [`State`] holds the shuffled wall, every hand, the melds and bonus tiles on
//! the table, and the public event log. Decisions come in three phases:
//!
//! * **Turn**: the player to move may win by self-draw, declare a kong, or discard.
//! * **Claim**: after a discard, each other seat that has a claim available
//!   declares one ([`Action::Pass`] included). Seats with nothing but *pass* are
//!   skipped. Declarations stay hidden until all are in (§4.1.1).
//! * **Rob**: after a kong is declared, seats that could win on its tile declare
//!   whether to rob it (§5.1–§5.2).
//!
//! Tile draws (and bonus-tile replacements) happen inside [`State::apply`].

use crate::faan::{score_win, Score, WinContext};
use crate::hand::{is_complete, is_thirteen_orphans};
use crate::meld::{Meld, MeldKind};
use crate::rules::{MultipleWinners, RuleSet};
use mahjong_games::{Game, Outcome, Player};
use mahjong_tiles::{Tile, TileCounts, NUM_KINDS};
use rand::seq::SliceRandom;
use rand::Rng;
use std::fmt;
use std::ops::Deref;

/// The dealer's seat (§2.2).
pub const DEALER: usize = 0;

/// A decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Action {
    /// Turn: discard a tile (§4.3).
    Discard(Tile),
    /// Turn: declare a self-drawn win (§4.2).
    SelfDrawnWin,
    /// Turn: declare a concealed kong of the tile (§5.1).
    ConcealedKong(Tile),
    /// Turn: add the tile to an exposed pung (§5.2).
    AddedKong(Tile),
    /// Claim or rob: decline.
    Pass,
    /// Claim or rob: win on the discard, or rob the kong.
    Win,
    /// Claim: pung the discard.
    Pung,
    /// Claim: kong the discard (holding three).
    ClaimKong,
    /// Claim (next seat only): chow, giving the chow's lowest tile.
    Chow(Tile),
}

/// Number of distinct action codes ([`Action::code`]).
pub const NUM_ACTION_CODES: usize = 134;

impl Action {
    /// A stable integer code, `0..134`, used by the Python bindings and as the
    /// fixed action space for learning:
    ///
    /// | Codes | Action |
    /// |---|---|
    /// | 0–33 | `Discard(tile)` |
    /// | 34 | `SelfDrawnWin` |
    /// | 35–68 | `ConcealedKong(tile)` |
    /// | 69–102 | `AddedKong(tile)` |
    /// | 103, 104, 105, 106 | `Pass`, `Win`, `Pung`, `ClaimKong` |
    /// | 107–133 | `Chow(lowest tile)` (suited tiles only) |
    pub fn code(self) -> u16 {
        let i = |t: Tile| t.index() as u16;
        match self {
            Action::Discard(t) => i(t),
            Action::SelfDrawnWin => 34,
            Action::ConcealedKong(t) => 35 + i(t),
            Action::AddedKong(t) => 69 + i(t),
            Action::Pass => 103,
            Action::Win => 104,
            Action::Pung => 105,
            Action::ClaimKong => 106,
            Action::Chow(t) => 107 + i(t),
        }
    }

    /// The action with the given code, if valid.
    pub fn from_code(code: u16) -> Option<Action> {
        let tile = |i: u16| Tile::from_index(i as u8);
        Some(match code {
            0..=33 => Action::Discard(tile(code)),
            34 => Action::SelfDrawnWin,
            35..=68 => Action::ConcealedKong(tile(code - 35)),
            69..=102 => Action::AddedKong(tile(code - 69)),
            103 => Action::Pass,
            104 => Action::Win,
            105 => Action::Pung,
            106 => Action::ClaimKong,
            107..=133 => Action::Chow(tile(code - 107)),
            _ => return None,
        })
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Discard(t) => write!(f, "discard {t}"),
            Action::SelfDrawnWin => f.write_str("self-drawn win"),
            Action::ConcealedKong(t) => write!(f, "concealed kong {t}"),
            Action::AddedKong(t) => write!(f, "added kong {t}"),
            Action::Pass => f.write_str("pass"),
            Action::Win => f.write_str("win"),
            Action::Pung => f.write_str("pung"),
            Action::ClaimKong => f.write_str("kong"),
            Action::Chow(t) => write!(f, "chow from {t}"),
        }
    }
}

/// A public event, in order. Draws do not reveal the tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// `seat` drew a tile (from the back if `replacement`).
    Draw { seat: u8, replacement: bool },
    /// `seat` set aside a bonus tile.
    Bonus { seat: u8, tile: Tile },
    /// `seat` discarded `tile`.
    Discard { seat: u8, tile: Tile },
    /// `seat` claimed `from`'s discard into an exposed meld.
    Claim { seat: u8, from: u8, meld: Meld },
    /// `seat` declared a concealed kong.
    ConcealedKong { seat: u8, tile: Tile },
    /// `seat` added a tile to a pung.
    AddedKong { seat: u8, tile: Tile },
    /// `seat` won.
    Win { seat: u8 },
    /// The wall ran out (§9).
    ExhaustiveDraw,
}

/// The melds one seat has on the table (at most 4).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeldList {
    items: [Meld; 4],
    len: u8,
}

impl MeldList {
    const EMPTY: MeldList = MeldList {
        items: [Meld::PLACEHOLDER; 4],
        len: 0,
    };

    fn push(&mut self, m: Meld) {
        self.items[self.len as usize] = m;
        self.len += 1;
    }

    /// Number of exposed (non-concealed) melds.
    pub fn exposed_count(&self) -> usize {
        self.iter().filter(|m| !m.concealed).count()
    }
}

impl Deref for MeldList {
    type Target = [Meld];
    fn deref(&self) -> &[Meld] {
        &self.items[..self.len as usize]
    }
}

impl fmt::Debug for MeldList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// One winner's record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WinRecord {
    pub seat: usize,
    pub score: Score,
    pub self_drawn: bool,
    /// Who discarded (or declared the robbed kong, or drew the 8th bonus tile).
    pub from: Option<usize>,
    /// The winning tile. A claimed winning tile is not moved into the winner's
    /// hand (several players may win on it): it stays in the river, or, when a
    /// kong was robbed, is held only here.
    pub win_tile: Option<Tile>,
    /// Who paid how much for this win (positive = paid).
    pub paid_by: [i32; 4],
    /// Set if a liable player paid for the whole table (§8.3).
    pub liable: Option<usize>,
    /// Won by robbing a kong (§5.1–§5.2).
    pub robbed: bool,
}

/// How the hand ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandResult {
    /// Empty after an exhaustive draw.
    pub wins: Vec<WinRecord>,
    /// Net points per seat; sums to zero.
    pub deltas: [i32; 4],
}

/// A win about to be settled.
struct NewWin {
    seat: usize,
    score: Score,
    self_drawn: bool,
    from: Option<usize>,
    win_tile: Option<Tile>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Turn,
    Claim,
    Rob,
    Over,
}

/// The tile being claimed or robbed, and the declarations so far.
#[derive(Clone, Copy, Debug)]
struct Window {
    tile: Tile,
    from: u8,
    /// Seats still to declare (bit mask).
    pending: u8,
    declared: [Option<Action>; 4],
    /// For the rob window: the kong was concealed (only thirteen orphans may rob).
    concealed_kong: bool,
}

/// A hand of Hong Kong mahjong in progress.
#[derive(Clone)]
pub struct State {
    rules: RuleSet,
    wall: Vec<Tile>,
    front: u8,
    back: u8,
    hands: [TileCounts; 4],
    melds: [MeldList; 4],
    bonus: [u8; 4],
    events: Vec<Event>,
    phase: Phase,
    /// The seat whose turn it is (Turn phase) or who is declaring (Claim/Rob).
    current: u8,
    /// The seat whose turn it is in the turn order.
    turn: u8,
    /// The current turn began with a draw, so a self-drawn win is possible.
    drew: bool,
    /// The tile drawn last in this turn.
    last_drawn: Option<Tile>,
    /// That draw emptied the wall (#6).
    drew_last_tile: bool,
    /// Kongs declared this turn (#7, #8).
    kong_chain: u8,
    /// The dealer's first turn, before any kong or discard (#29).
    dealer_first_turn: bool,
    /// Number of discards so far (#30 needs the first one).
    discards: u16,
    window: Option<Window>,
    /// `liable[p]` is the seat liable for `p` under the 12-tile rule (§8.3).
    liable: [Option<u8>; 4],
    result: Option<HandResult>,
}

fn bit(t: Tile) -> u8 {
    1 << (t.index() - NUM_KINDS as u8)
}

impl State {
    /// A freshly shuffled and dealt hand.
    pub fn new_deal<R: Rng + ?Sized>(rules: RuleSet, rng: &mut R) -> State {
        let mut wall = Self::all_tiles(&rules);
        wall.shuffle(rng);
        Self::from_wall(rules, wall)
    }

    /// The tiles in play (144, or 136 without flowers), in index order.
    pub fn all_tiles(rules: &RuleSet) -> Vec<Tile> {
        let mut v: Vec<Tile> = Tile::all_kinds().flat_map(|t| [t; 4]).collect();
        if rules.flowers {
            v.extend(Tile::all_bonus());
        }
        v
    }

    /// Deals from a wall in the given order: tiles are drawn from the front, and
    /// replacements from the back (§3).
    ///
    /// # Panics
    /// Panics if `wall` is not exactly the tile set of `rules`.
    pub fn from_wall(rules: RuleSet, wall: Vec<Tile>) -> State {
        let mut sorted = wall.clone();
        sorted.sort();
        assert_eq!(
            sorted,
            Self::all_tiles(&rules),
            "wall must be a full tile set"
        );
        let n = wall.len() as u8;
        let mut s = State {
            rules,
            wall,
            front: 0,
            back: n,
            hands: [TileCounts::new(); 4],
            melds: [MeldList::EMPTY; 4],
            bonus: [0; 4],
            events: Vec::with_capacity(256),
            phase: Phase::Turn,
            current: DEALER as u8,
            turn: DEALER as u8,
            drew: true,
            last_drawn: None,
            drew_last_tile: false,
            kong_chain: 0,
            dealer_first_turn: true,
            discards: 0,
            window: None,
            liable: [None; 4],
            result: None,
        };
        // §3.2: the dealer takes 14 tiles, the others 13. Bonus tiles stay in the
        // hand list for now and are replaced in §3.3.
        let mut dealt: [Vec<Tile>; 4] = Default::default();
        for (seat, d) in dealt.iter_mut().enumerate() {
            let take = if seat == DEALER { 14 } else { 13 };
            for _ in 0..take {
                d.push(s.wall[s.front as usize]);
                s.front += 1;
            }
        }
        for (seat, d) in dealt.iter().enumerate() {
            for &t in d.iter().filter(|t| !t.is_bonus()) {
                s.hands[seat].add(t);
            }
        }
        // §3.3: flower replacement in seat order, from the back.
        for (seat, d) in dealt.iter().enumerate() {
            for &t in d.iter().filter(|t| t.is_bonus()) {
                if s.place_bonus(seat, t) {
                    return s;
                }
                if !s.draw(seat, true) {
                    return s;
                }
            }
        }
        s.last_drawn = None;
        s
    }

    // ----- Accessors (what is public is stated on each). -----

    /// The rules in force.
    pub fn rules(&self) -> &RuleSet {
        &self.rules
    }

    /// `seat`'s concealed tiles. Private to that seat.
    pub fn hand(&self, seat: usize) -> &TileCounts {
        &self.hands[seat]
    }

    /// How many concealed tiles `seat` holds. Public.
    pub fn concealed_count(&self, seat: usize) -> usize {
        self.hands[seat].total()
    }

    /// `seat`'s melds on the table. Public (concealed kongs show their tile, §5.1).
    pub fn melds(&self, seat: usize) -> &MeldList {
        &self.melds[seat]
    }

    /// `seat`'s bonus tiles as a bit mask (bit `i` = tile `34 + i`). Public.
    pub fn bonus(&self, seat: usize) -> u8 {
        self.bonus[seat]
    }

    /// The public event log.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Tiles left in the wall. Public.
    pub fn wall_remaining(&self) -> usize {
        (self.back - self.front) as usize
    }

    /// The undrawn wall, in draw order from the front. Hidden from everyone.
    pub fn wall(&self) -> &[Tile] {
        &self.wall[self.front as usize..self.back as usize]
    }

    /// The seat whose turn it is in turn order (the discarder during a claim).
    pub fn turn(&self) -> usize {
        self.turn as usize
    }

    /// The tile being claimed or robbed, and who it came from.
    pub fn claim_tile(&self) -> Option<(Tile, usize)> {
        self.window.map(|w| (w.tile, w.from as usize))
    }

    /// Whether the current decision is in a claim window (after a discard).
    pub fn in_claim(&self) -> bool {
        self.phase == Phase::Claim
    }

    /// The decision phase: `"turn"`, `"claim"`, `"rob"` or `"over"`.
    pub fn phase_name(&self) -> &'static str {
        match self.phase {
            Phase::Turn => "turn",
            Phase::Claim => "claim",
            Phase::Rob => "rob",
            Phase::Over => "over",
        }
    }

    /// Whether the current decision is whether to rob a kong.
    pub fn in_rob(&self) -> bool {
        self.phase == Phase::Rob
    }

    /// The tile drawn last by the player to move, if this turn began with a draw.
    /// Private to that player.
    pub fn last_drawn(&self) -> Option<Tile> {
        if self.phase == Phase::Turn && self.drew {
            self.last_drawn
        } else {
            None
        }
    }

    /// Who is liable for whom (§8.3). Public.
    pub fn liable_for(&self, seat: usize) -> Option<usize> {
        self.liable[seat].map(usize::from)
    }

    /// How the hand ended.
    pub fn result(&self) -> Option<&HandResult> {
        self.result.as_ref()
    }

    /// Each seat's discards still lying in the river (claimed ones excluded), in
    /// order, with the discarding seat. Public.
    pub fn river(&self) -> Vec<(usize, Tile, bool)> {
        let mut out: Vec<(usize, Tile, bool)> = Vec::new();
        for e in &self.events {
            match *e {
                Event::Discard { seat, tile } => out.push((seat as usize, tile, false)),
                Event::Claim { .. } => {
                    if let Some(last) = out.last_mut() {
                        last.2 = true;
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// The playing tiles `observer` has not seen: not in their hand, not in any
    /// meld, not in the river. These are the other hands plus the wall.
    pub fn unseen(&self, observer: usize) -> TileCounts {
        let mut c = TileCounts::full_set();
        for t in self.hands[observer].tiles() {
            c.remove(t);
        }
        for melds in &self.melds {
            for m in melds.iter() {
                for t in m.tiles() {
                    c.remove(t);
                }
            }
        }
        for (_, t, claimed) in self.river() {
            if !claimed {
                c.remove(t);
            }
        }
        c
    }

    // ----- Rules helpers. -----

    fn exposed_melds_after_claim(&self, seat: usize) -> usize {
        self.melds[seat].exposed_count() + 1
    }

    fn win_context(&self, seat: usize, tile: Option<Tile>, self_drawn: bool) -> WinContext {
        WinContext {
            seat,
            win_tile: tile,
            self_drawn,
            bonus: self.bonus[seat],
            last_tile: self_drawn && self.drew_last_tile,
            kong_chain: if self_drawn { self.kong_chain } else { 0 },
            robbing_kong: self.phase == Phase::Rob,
            heavenly: self_drawn && seat == DEALER && self.dealer_first_turn,
            earthly: !self_drawn
                && self.phase == Phase::Claim
                && self.discards == 1
                && self.window.is_some_and(|w| w.from as usize == DEALER),
        }
    }

    /// The score `seat` would get by winning with `tile` from another player (on
    /// a discard or by robbing a kong), if it is a legal win.
    fn claim_win_score(&self, seat: usize, tile: Tile) -> Option<Score> {
        let mut c = self.hands[seat];
        if c[tile] >= 4 {
            return None;
        }
        c.add(tile);
        if !is_complete(&c, self.melds[seat].len()) {
            return None;
        }
        if self.phase == Phase::Rob && self.window.is_some_and(|w| w.concealed_kong) {
            // §5.1: only thirteen orphans may rob a concealed kong.
            if !(self.melds[seat].is_empty() && is_thirteen_orphans(&c)) {
                return None;
            }
        }
        let ctx = self.win_context(seat, Some(tile), false);
        score_win(&c, &self.melds[seat], &ctx, &self.rules)
            .filter(|s| s.faan >= self.rules.min_faan)
    }

    /// The score of a self-drawn win for the player to move, if legal now.
    pub fn self_drawn_score(&self) -> Option<Score> {
        if self.phase != Phase::Turn || !self.drew {
            return None;
        }
        let seat = self.current as usize;
        let c = &self.hands[seat];
        if !is_complete(c, self.melds[seat].len()) {
            return None;
        }
        let ctx = self.win_context(seat, self.last_drawn, true);
        score_win(c, &self.melds[seat], &ctx, &self.rules).filter(|s| s.faan >= self.rules.min_faan)
    }

    /// The score `seat` would get by winning on the tile currently being claimed
    /// or robbed.
    pub fn claim_score(&self, seat: usize) -> Option<Score> {
        let w = self.window?;
        self.claim_win_score(seat, w.tile)
    }

    /// The claim-window options for `seat` on the current window's tile, other
    /// than [`Action::Pass`].
    fn window_options(&self, seat: usize, out: &mut Vec<Action>) {
        let Some(w) = self.window else { return };
        if seat == w.from as usize {
            return;
        }
        if self.claim_win_score(seat, w.tile).is_some() {
            out.push(Action::Win);
        }
        // Robbing (§5.2) and the last discard (§4.1.7) allow only a win.
        if self.phase == Phase::Rob || self.wall_remaining() == 0 {
            return;
        }
        let have = self.hands[seat][w.tile];
        if have >= 2 {
            out.push(Action::Pung);
        }
        if have >= 3 {
            out.push(Action::ClaimKong);
        }
        if seat == (w.from as usize + 1) % 4 && w.tile.is_suited() {
            for start in -2i8..=0 {
                let tiles: Option<Vec<Tile>> = (0..3).map(|k| w.tile.offset(start + k)).collect();
                if let Some(tiles) = tiles {
                    if tiles
                        .iter()
                        .all(|&t| t == w.tile || self.hands[seat][t] > 0)
                    {
                        out.push(Action::Chow(tiles[0]));
                    }
                }
            }
        }
    }

    fn has_window_options(&self, seat: usize) -> bool {
        let mut v = Vec::new();
        self.window_options(seat, &mut v);
        !v.is_empty()
    }

    // ----- Transitions. -----

    /// Places a bonus tile for `seat` (§5.3) and checks #31 and #32. Returns `true`
    /// if that ended the hand.
    fn place_bonus(&mut self, seat: usize, tile: Tile) -> bool {
        self.bonus[seat] |= bit(tile);
        self.events.push(Event::Bonus {
            seat: seat as u8,
            tile,
        });
        if self.bonus[seat] == 0xFF {
            let score = Score::limit(vec![crate::faan::Pattern::EightImmortals], &self.rules);
            self.finish(vec![NewWin {
                seat,
                score,
                self_drawn: true,
                from: None,
                win_tile: None,
            }]);
            return true;
        }
        if let Some(holder) = (0..4).find(|&p| p != seat && self.bonus[p].count_ones() == 7) {
            if self.bonus[holder] | self.bonus[seat] == 0xFF {
                let score = Score::limit(vec![crate::faan::Pattern::SevenRobOne], &self.rules);
                self.finish(vec![NewWin {
                    seat: holder,
                    score,
                    self_drawn: false,
                    from: Some(seat),
                    win_tile: Some(tile),
                }]);
                return true;
            }
        }
        false
    }

    /// Draws a playing tile for `seat`, replacing bonus tiles. Returns `false` if
    /// the hand ended (wall exhausted, or a bonus-tile limit hand).
    fn draw(&mut self, seat: usize, from_back: bool) -> bool {
        let mut replacement = from_back;
        loop {
            if self.front == self.back {
                self.exhaustive_draw();
                return false;
            }
            let tile = if replacement {
                self.back -= 1;
                self.wall[self.back as usize]
            } else {
                self.front += 1;
                self.wall[self.front as usize - 1]
            };
            self.events.push(Event::Draw {
                seat: seat as u8,
                replacement,
            });
            if tile.is_bonus() {
                if self.place_bonus(seat, tile) {
                    return false;
                }
                replacement = true;
                continue;
            }
            self.hands[seat].add(tile);
            self.last_drawn = Some(tile);
            self.drew_last_tile = self.front == self.back;
            return true;
        }
    }

    fn exhaustive_draw(&mut self) {
        self.events.push(Event::ExhaustiveDraw);
        self.phase = Phase::Over;
        self.window = None;
        self.result = Some(HandResult {
            wins: Vec::new(),
            deltas: [0; 4],
        });
    }

    /// Starts `seat`'s turn with a draw from the front (§4.1).
    fn start_turn(&mut self, seat: usize) {
        self.turn = seat as u8;
        self.current = seat as u8;
        self.phase = Phase::Turn;
        self.window = None;
        self.kong_chain = 0;
        self.drew = true;
        self.drew_last_tile = false;
        self.draw(seat, false);
    }

    /// After a kong: the replacement draw, then the same player continues (§5).
    fn kong_replacement(&mut self, seat: usize) {
        self.kong_chain += 1;
        self.turn = seat as u8;
        self.current = seat as u8;
        self.phase = Phase::Turn;
        self.window = None;
        self.drew = true;
        self.draw(seat, true);
    }

    /// Opens a claim or rob window on `tile`, or returns `false` if no seat has
    /// anything to declare.
    fn open_window(&mut self, phase: Phase, tile: Tile, from: usize, concealed_kong: bool) -> bool {
        self.phase = phase;
        self.window = Some(Window {
            tile,
            from: from as u8,
            pending: 0,
            declared: [None; 4],
            concealed_kong,
        });
        let mut pending = 0u8;
        for k in 1..4 {
            let seat = (from + k) % 4;
            if self.has_window_options(seat) {
                pending |= 1 << seat;
            }
        }
        if pending == 0 {
            self.window = None;
            return false;
        }
        let w = self.window.as_mut().expect("window just opened");
        w.pending = pending;
        self.current = Self::first_pending(pending, from) as u8;
        true
    }

    /// The first seat in `pending`, in turn order after `from`.
    fn first_pending(pending: u8, from: usize) -> usize {
        (1..4)
            .map(|k| (from + k) % 4)
            .find(|&s| pending & (1 << s) != 0)
            .expect("non-empty")
    }

    fn resolve_window(&mut self) {
        let w = self.window.expect("window open");
        let from = w.from as usize;
        let order: Vec<usize> = (1..4).map(|k| (from + k) % 4).collect();
        let mut winners: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&s| w.declared[s] == Some(Action::Win))
            .collect();
        if self.rules.multiple_winners == MultipleWinners::HeadBump {
            winners.truncate(1);
        }
        if !winners.is_empty() {
            let wins = winners
                .into_iter()
                .map(|s| {
                    let score = self
                        .claim_win_score(s, w.tile)
                        .expect("declared win is legal");
                    NewWin {
                        seat: s,
                        score,
                        self_drawn: false,
                        from: Some(from),
                        win_tile: Some(w.tile),
                    }
                })
                .collect();
            if self.phase == Phase::Rob {
                self.cancel_kong(from, w.tile, w.concealed_kong);
            }
            self.finish(wins);
            return;
        }
        if self.phase == Phase::Rob {
            self.kong_replacement(from);
            return;
        }
        let claim = order
            .iter()
            .find_map(|&s| match w.declared[s] {
                Some(a @ (Action::Pung | Action::ClaimKong)) => Some((s, a)),
                _ => None,
            })
            .or_else(|| {
                order.iter().find_map(|&s| match w.declared[s] {
                    Some(a @ Action::Chow(_)) => Some((s, a)),
                    _ => None,
                })
            });
        match claim {
            Some((seat, action)) => self.make_claim(seat, from, w.tile, action),
            None if self.wall_remaining() == 0 => self.exhaustive_draw(),
            None => self.start_turn((from + 1) % 4),
        }
    }

    /// The kong declarer's tile went to the robber: undo the kong (§5.2).
    fn cancel_kong(&mut self, seat: usize, tile: Tile, concealed: bool) {
        let melds = &mut self.melds[seat];
        let i = melds
            .iter()
            .position(|m| m.tile == tile && m.kind == MeldKind::Kong);
        let i = i.expect("kong being robbed");
        if concealed {
            // Remove the kong; its other three tiles return to the hand.
            let mut kept = MeldList::EMPTY;
            for (j, m) in melds.iter().enumerate() {
                if j != i {
                    kept.push(*m);
                }
            }
            *melds = kept;
            for _ in 0..3 {
                self.hands[seat].add(tile);
            }
        } else {
            melds.items[i].kind = MeldKind::Pung;
        }
    }

    fn make_claim(&mut self, seat: usize, from: usize, tile: Tile, action: Action) {
        let meld = match action {
            Action::Pung => {
                self.hands[seat][tile] -= 2;
                Meld::exposed(MeldKind::Pung, tile, from)
            }
            Action::ClaimKong => {
                self.hands[seat][tile] -= 3;
                Meld::exposed(MeldKind::Kong, tile, from)
            }
            Action::Chow(low) => {
                for k in 0..3 {
                    let t = low.offset(k).expect("valid chow");
                    if t != tile {
                        self.hands[seat].remove(t);
                    }
                }
                Meld::exposed(MeldKind::Chow, low, from)
            }
            _ => unreachable!("not a meld claim"),
        };
        if self.rules.twelve_tile_liability && self.exposed_melds_after_claim(seat) == 4 {
            self.liable[seat] = Some(from as u8);
        }
        self.melds[seat].push(meld);
        self.events.push(Event::Claim {
            seat: seat as u8,
            from: from as u8,
            meld,
        });
        if action == Action::ClaimKong {
            self.kong_chain = 0;
            self.kong_replacement(seat);
        } else {
            self.turn = seat as u8;
            self.current = seat as u8;
            self.phase = Phase::Turn;
            self.window = None;
            self.kong_chain = 0;
            self.drew = false;
            self.drew_last_tile = false;
        }
    }

    /// Ends the hand with the given wins and settles payment (§8).
    fn finish(&mut self, wins: Vec<NewWin>) {
        let robbed = self.phase == Phase::Rob;
        let mut deltas = [0i32; 4];
        let mut records = Vec::new();
        for NewWin {
            seat,
            score,
            self_drawn,
            from,
            win_tile,
        } in wins
        {
            let share = self.rules.points(score.faan) as i32;
            let mut paid_by = [0i32; 4];
            let liable = self.liable[seat].map(usize::from);
            match (liable, self_drawn, from) {
                (Some(l), _, _) => paid_by[l] = 3 * share,
                (None, true, _) => {
                    for (p, paid) in paid_by.iter_mut().enumerate() {
                        if p != seat {
                            *paid = share;
                        }
                    }
                }
                (None, false, Some(f)) => paid_by[f] = share,
                (None, false, None) => unreachable!("a claimed win has a source"),
            }
            for p in 0..4 {
                deltas[p] -= paid_by[p];
                deltas[seat] += paid_by[p];
            }
            self.events.push(Event::Win { seat: seat as u8 });
            records.push(WinRecord {
                seat,
                score,
                self_drawn,
                from,
                win_tile,
                paid_by,
                liable,
                robbed,
            });
        }
        self.phase = Phase::Over;
        self.window = None;
        self.result = Some(HandResult {
            wins: records,
            deltas,
        });
    }

    fn apply_turn(&mut self, action: Action) {
        let seat = self.current as usize;
        match action {
            Action::SelfDrawnWin => {
                let score = self.self_drawn_score().expect("legal self-drawn win");
                let tile = self.last_drawn;
                self.finish(vec![NewWin {
                    seat,
                    score,
                    self_drawn: true,
                    from: None,
                    win_tile: tile,
                }]);
            }
            Action::ConcealedKong(t) => {
                self.hands[seat][t] -= 4;
                self.melds[seat].push(Meld::concealed_kong(t));
                self.events.push(Event::ConcealedKong {
                    seat: seat as u8,
                    tile: t,
                });
                self.dealer_first_turn = false;
                if !self.open_window(Phase::Rob, t, seat, true) {
                    self.kong_replacement(seat);
                }
            }
            Action::AddedKong(t) => {
                self.hands[seat].remove(t);
                let melds = &mut self.melds[seat];
                let i = melds
                    .iter()
                    .position(|m| m.kind == MeldKind::Pung && m.tile == t)
                    .expect("pung to add to");
                melds.items[i].kind = MeldKind::Kong;
                self.events.push(Event::AddedKong {
                    seat: seat as u8,
                    tile: t,
                });
                self.dealer_first_turn = false;
                if !self.open_window(Phase::Rob, t, seat, false) {
                    self.kong_replacement(seat);
                }
            }
            Action::Discard(t) => {
                self.hands[seat].remove(t);
                self.events.push(Event::Discard {
                    seat: seat as u8,
                    tile: t,
                });
                self.discards += 1;
                self.dealer_first_turn = false;
                self.kong_chain = 0;
                self.drew = false;
                if !self.open_window(Phase::Claim, t, seat, false) {
                    if self.wall_remaining() == 0 {
                        self.exhaustive_draw();
                    } else {
                        self.start_turn((seat + 1) % 4);
                    }
                }
            }
            _ => panic!("{action} is not a turn action"),
        }
    }

    fn apply_declaration(&mut self, action: Action) {
        let seat = self.current as usize;
        let w = self.window.as_mut().expect("window open");
        w.declared[seat] = Some(action);
        w.pending &= !(1 << seat);
        if w.pending == 0 {
            self.resolve_window();
        } else {
            self.current = Self::first_pending(w.pending, w.from as usize) as u8;
        }
    }
}

impl Game for State {
    type Action = Action;

    fn num_players(&self) -> usize {
        4
    }

    fn current_player(&self) -> Option<Player> {
        (self.phase != Phase::Over).then_some(self.current as usize)
    }

    fn legal_actions(&self, out: &mut Vec<Action>) {
        out.clear();
        let seat = self.current as usize;
        match self.phase {
            Phase::Over => {}
            Phase::Turn => {
                if self.self_drawn_score().is_some() {
                    out.push(Action::SelfDrawnWin);
                }
                let hand = &self.hands[seat];
                if self.wall_remaining() > 0 {
                    for t in hand.kinds() {
                        if hand[t] == 4 {
                            out.push(Action::ConcealedKong(t));
                        }
                    }
                    for m in self.melds[seat].iter() {
                        if m.kind == MeldKind::Pung && hand[m.tile] > 0 {
                            out.push(Action::AddedKong(m.tile));
                        }
                    }
                }
                out.extend(hand.kinds().map(Action::Discard));
            }
            Phase::Claim | Phase::Rob => {
                self.window_options(seat, out);
                out.push(Action::Pass);
            }
        }
    }

    fn apply(&mut self, action: Action) {
        debug_assert!(
            self.legal_actions_vec().contains(&action),
            "illegal {action}"
        );
        match self.phase {
            Phase::Turn => self.apply_turn(action),
            Phase::Claim | Phase::Rob => self.apply_declaration(action),
            Phase::Over => panic!("the hand is over"),
        }
    }

    fn outcome(&self) -> Option<Outcome> {
        let r = self.result.as_ref()?;
        Some(Outcome {
            scores: r.deltas.iter().map(|&d| d as f64).collect(),
            winners: r.wins.iter().map(|w| w.seat).collect(),
        })
    }

    fn max_abs_score(&self) -> f64 {
        self.rules.max_abs_score()
    }

    fn determinize<R: Rng + ?Sized>(&self, observer: Player, rng: &mut R) -> State {
        let mut s = self.clone();
        if s.phase == Phase::Over {
            return s;
        }
        // Pool everything hidden from the observer: the other concealed hands and
        // the undrawn wall (which may contain bonus tiles).
        let mut pool: Vec<Tile> = s.wall().to_vec();
        let mut sizes = [0usize; 4];
        for seat in (0..4).filter(|&p| p != observer) {
            sizes[seat] = s.hands[seat].total();
            pool.extend(s.hands[seat].tiles());
            s.hands[seat] = TileCounts::new();
        }
        pool.shuffle(rng);
        // Bonus tiles cannot be in a hand: deal playing tiles to the hands first.
        let (mut playing, bonus): (Vec<Tile>, Vec<Tile>) =
            pool.into_iter().partition(|t| !t.is_bonus());
        for seat in (0..4).filter(|&p| p != observer) {
            for _ in 0..sizes[seat] {
                s.hands[seat].add(playing.pop().expect("enough hidden tiles"));
            }
        }
        let mut wall_rest = playing;
        wall_rest.extend(bonus);
        wall_rest.shuffle(rng);
        let start = s.front as usize;
        s.wall[start..s.back as usize].copy_from_slice(&wall_rest);
        // The player to move's last draw is only known to them.
        if s.current as usize != observer {
            s.last_drawn = None;
        }
        // Other seats' pending declarations are hidden: forget them and ask again
        // (their options may differ with the re-dealt hands).
        if let Some(mut w) = s.window {
            let from = w.from as usize;
            let mut pending = 0u8;
            for seat in (0..4).filter(|&p| p != from) {
                if seat == observer && w.pending & (1 << seat) == 0 {
                    continue; // the observer's own declaration is kept
                }
                w.declared[seat] = None;
                s.window = Some(w);
                if s.has_window_options(seat) {
                    pending |= 1 << seat;
                }
            }
            w.pending = pending;
            s.window = Some(w);
            if pending == 0 {
                s.resolve_window();
            } else if pending & (1 << observer) != 0 {
                s.current = observer as u8;
            } else {
                s.current = Self::first_pending(pending, from) as u8;
            }
        }
        s
    }
}

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "State {{ phase: {:?}, current: {}, wall: {} }}",
            self.phase,
            self.current,
            self.wall_remaining()
        )?;
        for seat in 0..4 {
            writeln!(
                f,
                "  {seat}: {} {:?} bonus {:08b}",
                self.hands[seat], self.melds[seat], self.bonus[seat]
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_codes_round_trip() {
        for code in 0..NUM_ACTION_CODES as u16 {
            let a = Action::from_code(code).unwrap();
            assert_eq!(a.code(), code);
        }
        assert_eq!(Action::from_code(NUM_ACTION_CODES as u16), None);
    }
}
