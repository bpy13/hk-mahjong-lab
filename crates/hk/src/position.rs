//! A mid-hand position without a wall: what [`State::from_position`] builds a
//! [`State`] from.
//!
//! A position lists what is on the table (hands, melds, bonus tiles, the river)
//! and whose decision it is. The undrawn wall is not given, only its size: the
//! constructor fills it with the tiles not accounted for, in index order. That
//! order is invisible to every observer, since the samplers sort the hidden pool
//! ([`State::hidden_pool`]). Before playing a position on, determinize it
//! ([`mahjong_games::Game::determinize`]) so the draws are random.
//!
//! [`State::from_position`]: crate::State::from_position
//! [`State::hidden_pool`]: crate::State::hidden_pool

use crate::meld::{Meld, MeldKind};
use crate::rules::RuleSet;
use crate::state::{Event, State};
use mahjong_tiles::{Tile, TileCounts, NUM_KINDS};
use std::fmt;

/// Whose decision it is, and on what.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionPhase {
    /// The player to move is on their turn: they may discard, declare a kong,
    /// or win by self-draw.
    Turn,
    /// `from` has just discarded `tile` (the last entry of the river, not
    /// claimed); the player to move decides whether to claim it.
    Claim { tile: Tile, from: usize },
    /// `from` has just declared a kong of `tile` (one of `from`'s melds); the
    /// player to move decides whether to rob it.
    Rob { tile: Tile, from: usize },
}

/// A mid-hand position (see the module docs).
///
/// Hidden claim declarations of other seats are not part of a position: in a
/// claim or rob window every seat with an option is asked again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionSpec {
    /// Each seat's concealed tiles. The player to move holds 14 tiles minus 3
    /// per meld in the turn phase; every other hand holds 13 minus 3 per meld.
    pub hands: [TileCounts; 4],
    /// Each seat's melds, in the order they were made (the 12-tile rule reads
    /// the fourth exposed one). Exposed melds name the seat they were claimed
    /// from; a chow is claimed from the previous seat.
    pub melds: [Vec<Meld>; 4],
    /// Each seat's bonus tiles (flowers and seasons). Empty without flowers.
    pub bonus: [Vec<Tile>; 4],
    /// Every discard in order: `(seat, tile, claimed)`. A claimed discard is now
    /// part of an exposed meld, so it is counted there, not in the river.
    pub river: Vec<(usize, Tile, bool)>,
    /// Tiles left in the wall (bonus tiles included).
    pub wall_remaining: usize,
    /// The seat whose decision it is: the player on turn, or the seat
    /// declaring in a claim or rob window.
    pub to_move: usize,
    /// The decision.
    pub phase: PositionPhase,
    /// In the turn phase, the tile the player to move drew to start this turn
    /// (or as a kong replacement); `None` if the turn began with a claim.
    pub last_drawn: Option<Tile>,
    /// Kongs declared in this turn so far (faan #7 and #8 read it); usually 0.
    pub kong_chain: u8,
    /// `liable[p]` is the seat liable for `p` under the 12-tile rule (§8.3).
    pub liable: [Option<usize>; 4],
}

/// Why a [`PositionSpec`] does not describe a reachable position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PositionError {
    /// A seat number is not 0–3.
    BadSeat(usize),
    /// A bonus tile in a meld or the river, or a playing tile among the bonus
    /// tiles.
    WrongTileKind(Tile),
    /// A meld that cannot exist (the reason says why).
    BadMeld { seat: usize, reason: &'static str },
    /// Bonus tiles that are repeated, given without flowers, or that would
    /// already have ended the hand.
    BadBonus(&'static str),
    /// More than four copies of a tile across hands, melds and the river.
    TooManyCopies { tile: Tile, count: usize },
    /// A hand of the wrong size for its melds and the phase.
    HandSize {
        seat: usize,
        expected: usize,
        got: usize,
    },
    /// The claimed discards in the river and the exposed melds do not pair up.
    ClaimsDoNotMatchMelds,
    /// `wall_remaining` differs from the number of tiles not accounted for.
    WallDoesNotBalance { wall_remaining: usize, left: usize },
    /// The phase does not fit the rest of the position (the reason says why).
    BadPhase(&'static str),
    /// `last_drawn` is set outside the turn phase, or is not in the hand.
    BadLastDrawn,
    /// A liability entry that the 12-tile rule cannot produce.
    BadLiability(usize),
    /// The player to move has no decision to make here.
    NoDecision,
}

impl fmt::Display for PositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PositionError::BadSeat(s) => write!(f, "seat {s} out of range"),
            PositionError::WrongTileKind(t) => write!(f, "tile {t} is the wrong kind here"),
            PositionError::BadMeld { seat, reason } => write!(f, "seat {seat}: {reason}"),
            PositionError::BadBonus(why) => write!(f, "bonus tiles: {why}"),
            PositionError::TooManyCopies { tile, count } => {
                write!(f, "{count} copies of {tile} (at most 4)")
            }
            PositionError::HandSize {
                seat,
                expected,
                got,
            } => write!(f, "seat {seat} holds {got} tiles, expected {expected}"),
            PositionError::ClaimsDoNotMatchMelds => {
                f.write_str("claimed discards do not match the exposed melds")
            }
            PositionError::WallDoesNotBalance {
                wall_remaining,
                left,
            } => write!(
                f,
                "wall_remaining is {wall_remaining} but {left} tiles are unaccounted for"
            ),
            PositionError::BadPhase(why) => write!(f, "phase: {why}"),
            PositionError::BadLastDrawn => {
                f.write_str("last_drawn must be a tile in the mover's hand, in the turn phase")
            }
            PositionError::BadLiability(s) => write!(f, "liability for seat {s} is impossible"),
            PositionError::NoDecision => f.write_str("the player to move has no decision"),
        }
    }
}

impl std::error::Error for PositionError {}

/// What [`check`] derives from a valid spec for the constructor.
pub(crate) struct Checked {
    /// The undrawn wall, in index order.
    pub wall: Vec<Tile>,
    /// Bonus tiles per seat as bit masks (bit `i` = tile `34 + i`).
    pub bonus: [u8; 4],
    /// The public log: each discard, followed by its claim if it was claimed.
    pub events: Vec<Event>,
}

fn seat_ok(s: usize) -> Result<usize, PositionError> {
    if s < 4 {
        Ok(s)
    } else {
        Err(PositionError::BadSeat(s))
    }
}

fn check_meld(seat: usize, m: &Meld) -> Result<(), PositionError> {
    let bad = |reason| Err(PositionError::BadMeld { seat, reason });
    if m.tile.is_bonus() {
        return Err(PositionError::WrongTileKind(m.tile));
    }
    if m.kind == MeldKind::Chow && m.tile.offset(2).is_none() {
        return bad("a chow needs three suited tiles in a row");
    }
    if m.concealed {
        if m.kind != MeldKind::Kong || m.claimed_from.is_some() {
            return bad("only a kong can be concealed, and it is not claimed");
        }
        return Ok(());
    }
    let Some(from) = m.claimed_from else {
        return bad("an exposed meld names the seat it was claimed from");
    };
    let from = seat_ok(from as usize)?;
    if from == seat {
        return bad("a meld cannot be claimed from oneself");
    }
    if m.kind == MeldKind::Chow && seat != (from + 1) % 4 {
        return bad("a chow is claimed from the previous seat");
    }
    Ok(())
}

/// Pairs each claimed discard with a distinct exposed meld that could have
/// claimed it (same discarder, contains the tile), by augmenting paths. Returns
/// for each claimed discard the claiming seat and meld, or `None` when no
/// complete pairing exists.
fn match_claims(spec: &PositionSpec, claimed: &[usize]) -> Option<Vec<(usize, Meld)>> {
    let melds: Vec<(usize, Meld)> = (0..4)
        .flat_map(|seat| {
            spec.melds[seat]
                .iter()
                .filter(|m| !m.concealed)
                .map(move |&m| (seat, m))
        })
        .collect();
    if melds.len() != claimed.len() {
        return None;
    }
    let fits = |d: usize, j: usize| {
        let (from, tile, _) = spec.river[claimed[d]];
        let (_, m) = melds[j];
        m.claimed_from == Some(from as u8) && m.tiles().contains(&tile)
    };
    // owner[j] = the claimed discard paired with meld j.
    let mut owner: Vec<Option<usize>> = vec![None; melds.len()];
    fn augment(
        d: usize,
        seen: &mut [bool],
        owner: &mut [Option<usize>],
        fits: &dyn Fn(usize, usize) -> bool,
    ) -> bool {
        for j in 0..owner.len() {
            if seen[j] || !fits(d, j) {
                continue;
            }
            seen[j] = true;
            if owner[j].is_none_or(|other| augment(other, seen, owner, fits)) {
                owner[j] = Some(d);
                return true;
            }
        }
        false
    }
    for d in 0..claimed.len() {
        let mut seen = vec![false; melds.len()];
        if !augment(d, &mut seen, &mut owner, &fits) {
            return None;
        }
    }
    // Every meld has an owner here: as many melds as discards, all paired.
    let mut out: Vec<Option<(usize, Meld)>> = vec![None; claimed.len()];
    for (j, o) in owner.iter().enumerate() {
        out[(*o)?] = Some(melds[j]);
    }
    out.into_iter().collect()
}

/// Checks everything that does not need a built [`State`]; the constructor then
/// checks that the player to move has a decision.
pub(crate) fn check(rules: &RuleSet, spec: &PositionSpec) -> Result<Checked, PositionError> {
    // Seats.
    seat_ok(spec.to_move)?;
    for &(seat, _, _) in &spec.river {
        seat_ok(seat)?;
    }
    for l in spec.liable.iter().flatten() {
        seat_ok(*l)?;
    }
    // Melds.
    for seat in 0..4 {
        if spec.melds[seat].len() > 4 {
            return Err(PositionError::BadMeld {
                seat,
                reason: "at most four melds",
            });
        }
        for m in &spec.melds[seat] {
            check_meld(seat, m)?;
        }
    }
    // Bonus tiles.
    let mut bonus = [0u8; 4];
    for seat in 0..4 {
        for &t in &spec.bonus[seat] {
            if !t.is_bonus() {
                return Err(PositionError::WrongTileKind(t));
            }
            if !rules.flowers {
                return Err(PositionError::BadBonus("the rules play without flowers"));
            }
            let bit = 1u8 << (t.index() - NUM_KINDS as u8);
            if bonus.iter().any(|b| b & bit != 0) {
                return Err(PositionError::BadBonus("a bonus tile appears twice"));
            }
            bonus[seat] |= bit;
        }
    }
    let ended = bonus.contains(&0xFF)
        || (0..4).any(|p| {
            bonus[p].count_ones() == 7 && (0..4).any(|q| q != p && bonus[p] | bonus[q] == 0xFF)
        });
    if ended {
        return Err(PositionError::BadBonus(
            "these bonus tiles end the hand (#31, #32)",
        ));
    }
    // Copies of each playing kind: hands, melds and the unclaimed river.
    let mut used = [0usize; NUM_KINDS];
    for hand in &spec.hands {
        for (u, &c) in used.iter_mut().zip(hand.0.iter()) {
            *u += c as usize;
        }
    }
    for melds in &spec.melds {
        for m in melds {
            for t in m.tiles() {
                used[t.index() as usize] += 1;
            }
        }
    }
    for &(_, t, claimed) in &spec.river {
        if t.is_bonus() {
            return Err(PositionError::WrongTileKind(t));
        }
        if !claimed {
            used[t.index() as usize] += 1;
        }
    }
    for t in Tile::all_kinds() {
        let count = used[t.index() as usize];
        if count > 4 {
            return Err(PositionError::TooManyCopies { tile: t, count });
        }
    }
    // Hand sizes.
    for seat in 0..4 {
        let on_turn = spec.phase == PositionPhase::Turn && seat == spec.to_move;
        let expected = 13 + on_turn as usize - 3 * spec.melds[seat].len();
        let got = spec.hands[seat].total();
        if got != expected {
            return Err(PositionError::HandSize {
                seat,
                expected,
                got,
            });
        }
    }
    // Claimed discards pair up with the exposed melds; the log follows the river.
    let claimed: Vec<usize> = (0..spec.river.len()).filter(|&i| spec.river[i].2).collect();
    let claims = match_claims(spec, &claimed).ok_or(PositionError::ClaimsDoNotMatchMelds)?;
    let mut events = Vec::with_capacity(spec.river.len() + claims.len() + 16);
    let mut next_claim = claims.iter();
    for &(seat, tile, was_claimed) in &spec.river {
        events.push(Event::Discard {
            seat: seat as u8,
            tile,
        });
        if was_claimed {
            let &(claimer, meld) = next_claim.next().expect("one pairing per claim");
            events.push(Event::Claim {
                seat: claimer as u8,
                from: seat as u8,
                meld,
            });
        }
    }
    // The wall: whatever is left, in index order.
    let mut wall: Vec<Tile> = Tile::all_kinds()
        .flat_map(|t| std::iter::repeat_n(t, 4 - used[t.index() as usize]))
        .collect();
    if rules.flowers {
        let placed = bonus.iter().fold(0u8, |a, b| a | b);
        wall.extend(Tile::all_bonus().filter(|t| placed & (1 << (t.index() - 34)) == 0));
    }
    if wall.len() != spec.wall_remaining {
        return Err(PositionError::WallDoesNotBalance {
            wall_remaining: spec.wall_remaining,
            left: wall.len(),
        });
    }
    // The phase.
    match spec.phase {
        PositionPhase::Turn => {
            if let Some(t) = spec.last_drawn {
                if t.is_bonus() || spec.hands[spec.to_move][t] == 0 {
                    return Err(PositionError::BadLastDrawn);
                }
            } else if !spec.river.is_empty() {
                // A turn that did not begin with a draw began with a pung or
                // chow of the last discard (a claimed kong draws a replacement).
                let claimed_by_mover = spec.river.last().is_some_and(|&(from, tile, claimed)| {
                    claimed
                        && spec.melds[spec.to_move].iter().any(|m| {
                            m.claimed_from == Some(from as u8) && m.tiles().contains(&tile)
                        })
                });
                if !claimed_by_mover {
                    return Err(PositionError::BadPhase(
                        "a turn without last_drawn must follow the mover's claim",
                    ));
                }
            }
        }
        PositionPhase::Claim { tile, from } => {
            seat_ok(from)?;
            if spec.river.last() != Some(&(from, tile, false)) {
                return Err(PositionError::BadPhase(
                    "the claimed tile must be the last discard, unclaimed",
                ));
            }
        }
        PositionPhase::Rob { tile, from } => {
            seat_ok(from)?;
            if !spec.melds[from]
                .iter()
                .any(|m| m.kind == MeldKind::Kong && m.tile == tile)
            {
                return Err(PositionError::BadPhase(
                    "the robbed tile must be a kong of the declaring seat",
                ));
            }
        }
    }
    if let PositionPhase::Claim { from, .. } | PositionPhase::Rob { from, .. } = spec.phase {
        if spec.to_move == from {
            return Err(PositionError::BadPhase("the declaring seat cannot claim"));
        }
        if spec.last_drawn.is_some() {
            return Err(PositionError::BadLastDrawn);
        }
    }
    if spec.kong_chain > 4 {
        return Err(PositionError::BadPhase("at most four kongs in a turn"));
    }
    // Liability (§8.3): only with the rule on, for a seat with four exposed melds.
    for (seat, l) in spec.liable.iter().enumerate() {
        if let Some(l) = *l {
            let exposed = spec.melds[seat].iter().filter(|m| !m.concealed).count();
            if !rules.twelve_tile_liability || exposed != 4 || l == seat {
                return Err(PositionError::BadLiability(seat));
            }
        }
    }
    Ok(Checked {
        wall,
        bonus,
        events,
    })
}

impl State {
    /// The position this state is in: every hand plus the public table (see
    /// [`PositionSpec`]). It includes hidden information (all hands). `None`
    /// once the hand is over.
    pub fn to_position(&self) -> Option<PositionSpec> {
        let to_move = mahjong_games::Game::current_player(self)?;
        let phase = match (self.phase_name(), self.claim_tile()) {
            ("claim", Some((tile, from))) => PositionPhase::Claim { tile, from },
            ("rob", Some((tile, from))) => PositionPhase::Rob { tile, from },
            _ => PositionPhase::Turn,
        };
        Some(PositionSpec {
            hands: std::array::from_fn(|s| *self.hand(s)),
            melds: std::array::from_fn(|s| self.melds(s).to_vec()),
            bonus: std::array::from_fn(|s| {
                Tile::all_bonus()
                    .filter(|t| self.bonus(s) & (1 << (t.index() - 34)) != 0)
                    .collect()
            }),
            river: self.river(),
            wall_remaining: self.wall_remaining(),
            to_move,
            phase,
            last_drawn: self.last_drawn(),
            kong_chain: self.kong_chain(),
            liable: std::array::from_fn(|s| self.liable_for(s)),
        })
    }
}
