//! Plan 2026-11-16 Task 23: `stage-oracle`, a test-only bot that CHEATS.
//!
//! It plays exactly as the three-stage bot [`StageBot`] but, at a discard
//! decision, reads the opponents' true hands: when `stage`'s own pick would be
//! won on at once (by any opponent; the engine settles the claim window with
//! every possible winner declaring, liability included) and the hand holds a
//! tile that would not, it throws `stage`'s pick among the tiles that would
//! not. It measures the ceiling of "avoid the dealing tile" and is never a
//! default, never an agent spec, and never exported to the app.
//!
//! Ranges ([`Range`]): `far` (two or more from ready), `notready` (one or more
//! from ready), `all` (every decision), `far-big` (as `far`, avoiding only a
//! deal-in of 6 faan or more). "From ready" is the faan-aware distance of the
//! 14-tile hand after its best discard, the measure `stage` uses for its stages
//! (0 = ready with a legal win).

use crate::bots::{rank_discards, Metric};
use crate::stage::StageBot;
use crate::view::SeatView;
use mahjong_games::Game;
use mahjong_hk::{Action, State};
use mahjong_tiles::Tile;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Which decisions the oracle may change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Range {
    /// Two or more from ready.
    Far,
    /// One or more from ready (every hand that is not ready).
    NotReady,
    /// Every decision, ready hands included.
    All,
    /// As [`Range::Far`], but a tile counts as dealing only when it pays a win
    /// of [`BIG_FAAN`] faan or more.
    FarBig,
}

/// The faan from which [`Range::FarBig`] avoids a deal-in.
pub const BIG_FAAN: u8 = 6;

impl Range {
    /// All four, in the plan's order.
    pub const ALL: [Range; 4] = [Range::Far, Range::NotReady, Range::All, Range::FarBig];

    /// The plan's name of the variant.
    pub fn name(self) -> &'static str {
        match self {
            Range::Far => "far",
            Range::NotReady => "notready",
            Range::All => "all",
            Range::FarBig => "far-big",
        }
    }

    /// Whether a hand `dist` from ready is in this range.
    pub fn covers(self, dist: i8) -> bool {
        match self {
            Range::Far | Range::FarBig => dist >= 2,
            Range::NotReady => dist >= 1,
            Range::All => true,
        }
    }

    /// Whether a discard with this truth counts as dealing in for this range.
    pub fn deals(self, t: &DealIn) -> bool {
        match self {
            Range::FarBig => t.paid > 0 && t.faan >= BIG_FAAN,
            _ => t.paid > 0,
        }
    }
}

/// What a discard would cost at once, read from the true hands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DealIn {
    /// Points the discarder pays (0: nobody wins on it).
    pub paid: i32,
    /// The largest faan among the wins on it.
    pub faan: u8,
    /// The most open sets among the winners.
    pub melds: usize,
}

/// Seat `seat` (on turn) throws `t`: who wins on it at once, and the cost.
pub fn deal_in(state: &State, seat: usize, t: Tile) -> DealIn {
    let mut s = state.clone();
    s.apply(Action::Discard(t));
    while !s.is_terminal() && s.in_claim() {
        let legal = s.legal_actions_vec();
        let a = if legal.contains(&Action::Win) {
            Action::Win
        } else {
            Action::Pass
        };
        s.apply(a);
    }
    let Some(r) = s.result() else {
        return DealIn::default();
    };
    let wins: Vec<_> = r
        .wins
        .iter()
        .filter(|w| !w.self_drawn && w.from == Some(seat))
        .collect();
    if wins.is_empty() {
        return DealIn::default();
    }
    DealIn {
        paid: -r.deltas[seat],
        faan: wins.iter().map(|w| w.score.faan).max().unwrap_or(0),
        melds: wins
            .iter()
            .map(|w| state.melds(w.seat).len())
            .max()
            .unwrap_or(0),
    }
}

/// One discard decision of the oracle bot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    /// `stage`'s own pick.
    pub stage_pick: Tile,
    /// The tile thrown.
    pub pick: Tile,
    /// Distance from ready of the 14-tile hand (best discard).
    pub dist: i8,
    /// `stage`'s pick deals in (for the range's sense of dealing).
    pub stage_deals: bool,
    /// The truth of `stage`'s pick.
    pub truth: DealIn,
    /// The hand held a tile that does not deal in.
    pub safe_exists: bool,
}

impl Decision {
    /// The oracle changed the pick.
    pub fn changed(&self) -> bool {
        self.pick != self.stage_pick
    }
}

/// The cheating bot: `base` plays, the oracle changes dealing discards in
/// `range`.
#[derive(Clone, Copy, Debug)]
pub struct OracleBot {
    pub base: StageBot,
    pub range: Range,
}

/// Distance from ready of the 14-tile hand after its best discard among
/// `candidates` (faan-aware, as `stage` measures it).
pub fn distance(view: &SeatView, candidates: &[Tile]) -> i8 {
    rank_discards(view, candidates, Metric::Faan)[0].1 .0
}

/// The discard tiles in `legal`.
pub fn discards(legal: &[Action]) -> Vec<Tile> {
    legal
        .iter()
        .filter_map(|a| match a {
            Action::Discard(t) => Some(*t),
            _ => None,
        })
        .collect()
}

impl OracleBot {
    /// A plain `stage` with the oracle in `range`.
    pub fn new(range: Range) -> OracleBot {
        OracleBot {
            base: StageBot::default(),
            range,
        }
    }

    /// The action, and the decision record when `stage` chose a discard. The
    /// main `rng` stream is used exactly as `stage` uses it; the re-pick draws
    /// its tie-breaks from its own generator, so games stay in step with a
    /// plain `stage` until the first change.
    pub fn decide<R: Rng + ?Sized>(
        &self,
        state: &State,
        legal: &[Action],
        rng: &mut R,
    ) -> (Action, Option<Decision>) {
        let (action, _) = self.base.decide(state, legal, rng);
        let Action::Discard(stage_pick) = action else {
            return (action, None);
        };
        let seat = state.current_player().expect("on turn");
        let truth = deal_in(state, seat, stage_pick);
        let mut d = Decision {
            stage_pick,
            pick: stage_pick,
            dist: -1,
            stage_deals: self.range.deals(&truth),
            truth,
            safe_exists: false,
        };
        if !d.stage_deals {
            return (action, Some(d));
        }
        let view = SeatView::of(state, seat);
        let candidates = discards(legal);
        d.dist = distance(&view, &candidates);
        let safe: Vec<Tile> = candidates
            .iter()
            .copied()
            .filter(|&t| t != stage_pick && !self.range.deals(&deal_in(state, seat, t)))
            .collect();
        d.safe_exists = !safe.is_empty();
        if !d.safe_exists || !self.range.covers(d.dist) {
            return (action, Some(d));
        }
        let key = (state.wall_remaining() * 4 + seat) * 64 + stage_pick.index() as usize;
        let mut own = ChaCha8Rng::seed_from_u64(key as u64);
        let (pick, _) = self.base.pick_discard(&view, &safe, &mut own);
        d.pick = pick;
        (Action::Discard(pick), Some(d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahjong_games::Policy;
    use mahjong_hk::RuleSet;

    /// Plays `deals` games with the oracle in seat 0 and plain `stage`
    /// elsewhere; checks every discard of seat 0; returns the changes made.
    fn check(range: Range, deals: u64) -> usize {
        let stage = StageBot::default();
        let bot = OracleBot::new(range);
        let mut changes = 0;
        for deal in 0..deals {
            let mut state =
                State::new_deal(RuleSet::default(), &mut ChaCha8Rng::seed_from_u64(deal));
            let mut rng = ChaCha8Rng::seed_from_u64(1000 + deal);
            while !state.is_terminal() {
                let legal = state.legal_actions_vec();
                let p = state.current_player().expect("not over");
                let action = if p == 0 {
                    let (a, d) = bot.decide(&state, &legal, &mut rng);
                    // A dealing discard is allowed only out of range or when
                    // every tile deals in (checked from scratch, not from `d`).
                    if let Action::Discard(t) = a {
                        if range.deals(&deal_in(&state, 0, t)) {
                            let cands = discards(&legal);
                            let dist = distance(&SeatView::of(&state, 0), &cands);
                            let any_safe =
                                cands.iter().any(|&c| !range.deals(&deal_in(&state, 0, c)));
                            assert!(
                                !(range.covers(dist) && any_safe),
                                "{}: deal {deal} threw dealing {t} while holding a safe tile",
                                range.name()
                            );
                        }
                        let d = d.expect("a discard decision");
                        changes += d.changed() as usize;
                        if d.changed() {
                            assert!(range.covers(d.dist) && d.stage_deals);
                        }
                    }
                    a
                } else {
                    stage.choose(&state, &legal, &mut rng)
                };
                state.apply(action);
            }
        }
        changes
    }

    #[test]
    fn oracle_never_throws_a_dealing_tile_while_holding_a_safe_one() {
        let mut total = 0;
        for range in Range::ALL {
            total += check(range, 10);
        }
        assert!(total > 0, "the oracle never acted in 40 games");
    }

    #[test]
    fn ranges_cover_the_right_distances() {
        assert!(!Range::Far.covers(1) && Range::Far.covers(2));
        assert!(!Range::NotReady.covers(0) && Range::NotReady.covers(1));
        assert!(Range::All.covers(0));
        let small = DealIn {
            paid: 16,
            faan: 5,
            melds: 0,
        };
        let big = DealIn { faan: 6, ..small };
        assert!(Range::Far.deals(&small) && !Range::FarBig.deals(&small));
        assert!(Range::FarBig.deals(&big));
        assert!(!Range::All.deals(&DealIn::default()));
    }
}
