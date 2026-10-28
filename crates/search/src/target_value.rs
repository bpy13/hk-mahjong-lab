//! The target value table (hard-problems.md idea 1e, plan Task 9).
//!
//! For a 13-tile hand (after the player's discard), the table gives, from
//! self-play between the rule-based bots:
//!
//! * `p_win`: P(the player declares a legal win before the hand ends),
//! * `p_self_draw`: P(the win is self-drawn | a win),
//! * `points`: E[points per share | a win] (`RuleSet::points` of the paid faan),
//!
//! keyed by the hand's best [`Target`] kind ([`TargetKind`]), its faan-aware
//! shanten (0–3, and 4 for "4 or more" or unreachable), the tiles-left bucket
//! (0 = start of the wall … 5 = the last sixth, as [`crate::danger`]'s wall
//! progress) and the number of melds on the table (0–4).
//!
//! The values live in [`crate::target_tables`], written by
//! `cargo run --release -p mahjong-search --example fit_targets [hands] [holdout]`.

use crate::danger::PROGRESS_BUCKETS;
use crate::target_tables::{POINTS, P_SELF_DRAW, P_WIN};
use crate::view::SeatView;
use mahjong_analysis::{faan_shanten, Structure, SuitPlan, Target};
use mahjong_hk::{RuleSet, State};

/// Number of target kinds.
pub const TARGET_KINDS: usize = 5;
/// Faan-aware shanten buckets: 0, 1, 2, 3, and 4 for 4 or more.
pub const FAAN_SHANTEN_BUCKETS: usize = 5;
/// Tiles-left buckets (the same split of the wall as the danger tables).
pub const TILES_LEFT_BUCKETS: usize = PROGRESS_BUCKETS;
/// Melds on the table: 0–4.
pub const MELD_BUCKETS: usize = 5;
/// Size of each table.
pub const TARGET_CELLS: usize =
    TARGET_KINDS * FAAN_SHANTEN_BUCKETS * TILES_LEFT_BUCKETS * MELD_BUCKETS;

/// The kind of plan a hand is heading for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetKind {
    /// Any tiles, any or all-chow structure (value pungs, 平糊, 門前清 …).
    /// Also used when no target is reachable.
    Mixed = 0,
    /// Any tiles, all pungs (對對糊).
    AllPungs = 1,
    /// One suit plus honours (混一色).
    HalfFlush = 2,
    /// One suit only (清一色).
    FullFlush = 3,
    /// A limit hand (all honours, thirteen orphans).
    Limit = 4,
}

impl TargetKind {
    /// Every kind, in index order.
    pub const ALL: [TargetKind; TARGET_KINDS] = [
        TargetKind::Mixed,
        TargetKind::AllPungs,
        TargetKind::HalfFlush,
        TargetKind::FullFlush,
        TargetKind::Limit,
    ];

    /// The kind of a target plan.
    pub fn of(t: &Target) -> TargetKind {
        if t.limit {
            return TargetKind::Limit;
        }
        match t.plan {
            SuitPlan::HalfFlush(_) => TargetKind::HalfFlush,
            SuitPlan::FullFlush(_) => TargetKind::FullFlush,
            SuitPlan::Honours | SuitPlan::Orphans => TargetKind::Limit,
            SuitPlan::Mixed if t.structure == Structure::AllPungs => TargetKind::AllPungs,
            SuitPlan::Mixed => TargetKind::Mixed,
        }
    }

    /// A short English name (for tables and the Python binding; not shown to players).
    pub fn name(self) -> &'static str {
        match self {
            TargetKind::Mixed => "mixed",
            TargetKind::AllPungs => "all_pungs",
            TargetKind::HalfFlush => "half_flush",
            TargetKind::FullFlush => "full_flush",
            TargetKind::Limit => "limit",
        }
    }
}

/// One cell of the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TargetKey {
    pub kind: TargetKind,
    /// 0–4 (4 = four or more, or unreachable).
    pub faan_shanten: usize,
    /// 0–5 (0 = most of the wall left).
    pub tiles_left: usize,
    /// 0–4.
    pub melds: usize,
}

/// The wall-progress bucket for `wall_remaining` (as [`crate::danger::Opponent::of`]).
pub fn tiles_left_bucket(wall_remaining: usize, rules: &RuleSet) -> usize {
    let initial_wall = if rules.flowers { 91.0 } else { 83.0 };
    let progress = (1.0 - wall_remaining as f64 / initial_wall).clamp(0.0, 0.999);
    (progress * TILES_LEFT_BUCKETS as f64) as usize
}

impl TargetKey {
    /// The key of a hand: its best target and faan-aware shanten, with
    /// `wall_remaining` tiles still to draw. Meant for a 13-tile hand (after
    /// the discard); a 14-tile complete hand counts as faan-shanten 0.
    pub fn of_view(view: &SeatView, wall_remaining: usize) -> TargetKey {
        let fs = faan_shanten(&view.hand_view(), &view.rules);
        TargetKey {
            kind: fs.target.as_ref().map_or(TargetKind::Mixed, TargetKind::of),
            faan_shanten: fs.shanten.clamp(0, 4) as usize,
            tiles_left: tiles_left_bucket(wall_remaining, &view.rules),
            melds: view.melds.len().min(4),
        }
    }

    /// The key of `seat`'s current hand in `state`.
    pub fn of_seat(state: &State, seat: usize) -> TargetKey {
        Self::of_view(&SeatView::of(state, seat), state.wall_remaining())
    }

    /// The flat index into the tables.
    pub fn cell(&self) -> usize {
        debug_assert!(self.faan_shanten < FAAN_SHANTEN_BUCKETS);
        debug_assert!(self.tiles_left < TILES_LEFT_BUCKETS);
        debug_assert!(self.melds < MELD_BUCKETS);
        (((self.kind as usize) * FAAN_SHANTEN_BUCKETS + self.faan_shanten) * TILES_LEFT_BUCKETS
            + self.tiles_left)
            * MELD_BUCKETS
            + self.melds
    }

    /// The key at a flat index (inverse of [`TargetKey::cell`]).
    pub fn from_cell(cell: usize) -> TargetKey {
        let melds = cell % MELD_BUCKETS;
        let rest = cell / MELD_BUCKETS;
        let tiles_left = rest % TILES_LEFT_BUCKETS;
        let rest = rest / TILES_LEFT_BUCKETS;
        let faan_shanten = rest % FAAN_SHANTEN_BUCKETS;
        let kind = TargetKind::ALL[rest / FAAN_SHANTEN_BUCKETS];
        TargetKey {
            kind,
            faan_shanten,
            tiles_left,
            melds,
        }
    }
}

/// What the table says about one key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TargetValue {
    /// P(a legal win before the hand ends).
    pub p_win: f64,
    /// P(self-drawn | win).
    pub p_self_draw: f64,
    /// E[points per share | win].
    pub points: f64,
}

/// The table's value for `key`.
pub fn lookup(key: TargetKey) -> TargetValue {
    let i = key.cell();
    TargetValue {
        p_win: P_WIN[i],
        p_self_draw: P_SELF_DRAW[i],
        points: POINTS[i],
    }
}

/// The key and value of `seat`'s current hand.
pub fn of_seat(state: &State, seat: usize) -> (TargetKey, TargetValue) {
    let key = TargetKey::of_seat(state, seat);
    (key, lookup(key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn cells_round_trip() {
        for i in 0..TARGET_CELLS {
            assert_eq!(TargetKey::from_cell(i).cell(), i);
        }
    }

    #[test]
    fn every_value_is_a_probability_or_positive_points() {
        for i in 0..TARGET_CELLS {
            let v = lookup(TargetKey::from_cell(i));
            assert!((0.0..=1.0).contains(&v.p_win), "p_win {i}: {v:?}");
            assert!(
                (0.0..=1.0).contains(&v.p_self_draw),
                "p_self_draw {i}: {v:?}"
            );
            assert!(v.points.is_finite() && v.points >= 0.0, "points {i}: {v:?}");
        }
    }

    #[test]
    fn closer_hands_win_at_least_as_often() {
        for i in 0..TARGET_CELLS {
            let key = TargetKey::from_cell(i);
            if key.faan_shanten + 1 >= FAAN_SHANTEN_BUCKETS {
                continue;
            }
            let farther = TargetKey {
                faan_shanten: key.faan_shanten + 1,
                ..key
            };
            assert!(
                lookup(key).p_win >= lookup(farther).p_win,
                "{key:?}: {} < {}",
                lookup(key).p_win,
                lookup(farther).p_win
            );
        }
    }

    #[test]
    fn a_dealt_hand_has_a_key_in_range() {
        let mut rng = SmallRng::seed_from_u64(7);
        let s = State::new_deal(RuleSet::default(), &mut rng);
        for seat in 0..4 {
            let (key, v) = of_seat(&s, seat);
            assert!(key.cell() < TARGET_CELLS);
            assert_eq!(key.tiles_left, 0, "start of the wall");
            assert_eq!(key.melds, 0);
            assert!(v.p_win <= 1.0);
        }
    }
}
