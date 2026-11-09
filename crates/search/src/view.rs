//! What one seat can see, gathered once per decision.

use mahjong_analysis::HandView;
use mahjong_games::Game;
use mahjong_hk::{Meld, RiverEntry, RuleSet, State};
use mahjong_tiles::{Tile, TileCounts};

/// The acting seat's private and public information, owned.
#[derive(Clone, Debug)]
pub struct SeatView {
    pub seat: usize,
    pub concealed: TileCounts,
    pub melds: Vec<Meld>,
    pub bonus: u8,
    pub unseen: TileCounts,
    pub rules: RuleSet,
    /// Every discard on the table, in order, with whether each came from the
    /// hand 入章 or was the tile just drawn 摸打 (rules §4.5). Public.
    pub river: Vec<RiverEntry>,
}

impl SeatView {
    /// The view of the player to act.
    pub fn of_current(state: &State) -> SeatView {
        Self::of(state, state.current_player().expect("game not over"))
    }

    /// The view of `seat`.
    pub fn of(state: &State, seat: usize) -> SeatView {
        SeatView {
            seat,
            concealed: *state.hand(seat),
            melds: state.melds(seat).to_vec(),
            bonus: state.bonus(seat),
            unseen: state.unseen(seat),
            rules: *state.rules(),
            river: state.river_entries(),
        }
    }

    /// Borrowed form for `mahjong_analysis`.
    pub fn hand_view(&self) -> HandView<'_> {
        HandView {
            concealed: &self.concealed,
            melds: &self.melds,
            seat: self.seat,
            bonus: self.bonus,
            unseen: &self.unseen,
        }
    }

    /// The same view with different concealed tiles and melds.
    pub fn with(&self, concealed: TileCounts, melds: Vec<Meld>) -> SeatView {
        SeatView {
            concealed,
            melds,
            ..self.clone()
        }
    }

    /// Tiles of `t` this seat holds.
    pub fn count(&self, t: Tile) -> u8 {
        self.concealed[t]
    }
}
