//! Faan-aware shanten: how far a hand is from a complete hand that scores at
//! least the minimum (§6), not merely from any complete hand.
//!
//! A **target** fixes a plan for the finished hand: which tiles it may use (any,
//! a half or full flush of one suit, or honours only), which meld shapes
//! ([`Structure`]), and which scoring pungs it must contain (dragons, the seat
//! wind, the prevailing wind). Each target has a *guaranteed* faan: what the
//! plan scores before anything lucky happens (self-draw, last tile, …), including
//! 門前清 if the hand is still concealed and flowers already held. Only targets
//! whose guaranteed faan *without flowers* reaches the minimum count, since bonus
//! tiles do not count toward it (§6). The hand's faan-aware shanten
//! is the smallest shanten among them.
//!
//! This is a lower bound on value and an estimate of distance: it does not try
//! every pattern combination (for example, it never plans on 自摸), so a hand can
//! sometimes win sooner than it predicts.

use crate::shanten::{orphans_shanten, standard_shanten, Structure};
use mahjong_hk::faan::flower_patterns;
use mahjong_hk::{Meld, MeldKind, RuleSet};
use mahjong_tiles::{Suit, Tile, TileCounts};

/// What one player knows about their own hand.
#[derive(Clone, Copy, Debug)]
pub struct HandView<'a> {
    /// Concealed tiles.
    pub concealed: &'a TileCounts,
    /// Melds on the table.
    pub melds: &'a [Meld],
    /// The player's seat.
    pub seat: usize,
    /// Bonus tiles held (bit mask, see [`mahjong_hk::WinContext::bonus`]).
    pub bonus: u8,
    /// Tiles the player has not seen (still obtainable).
    pub unseen: &'a TileCounts,
}

/// Which tiles a target hand may contain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SuitPlan {
    /// Any tiles.
    Mixed,
    /// One suit plus honours (#16).
    HalfFlush(Suit),
    /// One suit only (#17).
    FullFlush(Suit),
    /// Honours only (#23, a limit hand).
    Honours,
    /// Thirteen orphans (#25, a limit hand).
    Orphans,
}

impl SuitPlan {
    fn allows(self, t: Tile) -> bool {
        match self {
            SuitPlan::Mixed => true,
            SuitPlan::HalfFlush(s) => t.is_honour() || t.suit() == Some(s),
            SuitPlan::FullFlush(s) => t.suit() == Some(s),
            SuitPlan::Honours => t.is_honour(),
            SuitPlan::Orphans => t.is_terminal_or_honour(),
        }
    }
}

/// A plan for the finished hand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub plan: SuitPlan,
    pub structure: Structure,
    /// Scoring pungs the plan must contain.
    pub pungs: Vec<Tile>,
    /// Faan the plan guarantees (capped).
    pub faan: u8,
    /// Whether the plan is a limit hand.
    pub limit: bool,
}

/// The result of [`faan_shanten`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaanShanten {
    /// Exchanges to a ready hand that meets the minimum (`-1` = complete).
    /// [`UNREACHABLE`] if no target is feasible.
    pub shanten: i8,
    /// The best target (smallest shanten, then highest faan).
    pub target: Option<Target>,
}

/// Shanten value used when no feasible target exists.
pub const UNREACHABLE: i8 = 20;

fn melds_fit(melds: &[Meld], plan: SuitPlan, structure: Structure) -> bool {
    melds.iter().all(|m| {
        m.tiles().iter().all(|&t| plan.allows(t))
            && match structure {
                Structure::Any => true,
                Structure::AllPungs => m.is_pung_like(),
                Structure::AllChows => m.kind == MeldKind::Chow,
            }
    })
}

/// Guaranteed faan of a plan for this player (capped; `None` if below minimum).
fn guaranteed_faan(
    view: &HandView,
    plan: SuitPlan,
    structure: Structure,
    pungs: &[Tile],
    rules: &RuleSet,
) -> Option<(u8, bool)> {
    if matches!(plan, SuitPlan::Honours | SuitPlan::Orphans) {
        return Some((rules.faan_cap, true));
    }
    // Bonus tiles add faan but do not count toward the minimum (§6).
    let flowers: u32 = flower_patterns(view.bonus, view.seat, rules)
        .iter()
        .map(|&(_, x)| x as u32)
        .sum();
    let mut f: u32 = 0;
    if view.melds.iter().all(|m| m.concealed) {
        f += 1;
    }
    f += match plan {
        SuitPlan::HalfFlush(_) => 3,
        SuitPlan::FullFlush(_) => 7,
        _ => 0,
    };
    f += match structure {
        Structure::AllPungs => 3,
        Structure::AllChows => 1,
        Structure::Any => 0,
    };
    let dragons = pungs.iter().filter(|t| t.is_dragon()).count();
    f += if dragons == 3 { 8 } else { dragons as u32 };
    for &t in pungs.iter().filter(|t| t.is_wind()) {
        f += (t == Tile::wind(view.seat)) as u32 + (t == Tile::wind(rules.prevailing_wind)) as u32;
    }
    (f >= rules.min_faan as u32).then(|| ((f + flowers).min(rules.faan_cap as u32) as u8, false))
}

/// Shanten toward one plan, or `None` if the plan is impossible.
fn plan_shanten(
    view: &HandView,
    plan: SuitPlan,
    structure: Structure,
    pungs: &[Tile],
) -> Option<i8> {
    if plan == SuitPlan::Orphans {
        return view
            .melds
            .is_empty()
            .then(|| orphans_shanten(view.concealed));
    }
    if !melds_fit(view.melds, plan, structure) {
        return None;
    }
    let mut hand = TileCounts::new();
    for t in view.concealed.kinds().filter(|&t| plan.allows(t)) {
        hand[t] = view.concealed[t];
    }
    let mut melds = view.melds.len();
    let mut extra = 0i8;
    for &t in pungs {
        if view.melds.iter().any(|m| m.is_pung_like() && m.tile == t) {
            continue;
        }
        let take = hand[t].min(3);
        let missing = 3 - take;
        if view.unseen[t] < missing {
            return None;
        }
        hand[t] -= take;
        melds += 1;
        extra += missing as i8;
    }
    if melds > 4 {
        return None;
    }
    Some(standard_shanten(&hand, melds, structure) + extra)
}

/// The scoring tiles worth planning a pung of: dragons and the two value winds
/// already held at least in pairs, or already pung'ed on the table.
fn value_pung_candidates(view: &HandView, rules: &RuleSet) -> (Vec<Tile>, Vec<Tile>) {
    let mut values = vec![Tile::WHITE, Tile::GREEN, Tile::RED, Tile::wind(view.seat)];
    if rules.prevailing_wind != view.seat {
        values.push(Tile::wind(rules.prevailing_wind));
    }
    let on_table: Vec<Tile> = values
        .iter()
        .copied()
        .filter(|&t| view.melds.iter().any(|m| m.is_pung_like() && m.tile == t))
        .collect();
    let optional = values
        .into_iter()
        .filter(|t| !on_table.contains(t) && view.concealed[*t] >= 2)
        .collect();
    (on_table, optional)
}

/// The plans worth considering for this hand.
pub fn targets(view: &HandView, rules: &RuleSet) -> Vec<Target> {
    let mut all = TileCounts::new();
    for t in view.concealed.tiles() {
        all.add(t);
    }
    for m in view.melds {
        m.add_to(&mut all);
    }
    let best_suit = Suit::ALL
        .into_iter()
        .max_by_key(|&s| all.suit_counts(s).iter().map(|&c| c as u32).sum::<u32>())
        .expect("three suits");
    let honours: u32 = (27..34).map(|i| all.0[i] as u32).sum();
    let mut plans = vec![
        SuitPlan::Mixed,
        SuitPlan::HalfFlush(best_suit),
        SuitPlan::FullFlush(best_suit),
    ];
    if honours >= 7 {
        plans.push(SuitPlan::Honours);
    }
    let (fixed, optional) = value_pung_candidates(view, rules);
    let mut out = Vec::new();
    for plan in plans {
        for structure in [Structure::Any, Structure::AllPungs, Structure::AllChows] {
            for mask in 0..(1u32 << optional.len()) {
                let mut pungs = fixed.clone();
                pungs.extend(
                    optional
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| mask & (1 << i) != 0)
                        .map(|(_, &t)| t),
                );
                if structure == Structure::AllChows && !pungs.is_empty() {
                    continue;
                }
                if pungs.iter().any(|&t| !plan.allows(t)) {
                    continue;
                }
                if let Some((faan, limit)) = guaranteed_faan(view, plan, structure, &pungs, rules) {
                    out.push(Target {
                        plan,
                        structure,
                        pungs,
                        faan,
                        limit,
                    });
                }
            }
        }
    }
    if view.melds.is_empty() {
        out.push(Target {
            plan: SuitPlan::Orphans,
            structure: Structure::Any,
            pungs: Vec::new(),
            faan: rules.faan_cap,
            limit: true,
        });
    }
    out
}

/// Faan-aware shanten of the hand (see the module docs).
pub fn faan_shanten(view: &HandView, rules: &RuleSet) -> FaanShanten {
    let mut best = FaanShanten {
        shanten: UNREACHABLE,
        target: None,
    };
    for target in targets(view, rules) {
        let Some(s) = plan_shanten(view, target.plan, target.structure, &target.pungs) else {
            continue;
        };
        let better = match &best.target {
            None => true,
            Some(b) => {
                (s, std::cmp::Reverse(target.faan)) < (best.shanten, std::cmp::Reverse(b.faan))
            }
        };
        if better {
            best = FaanShanten {
                shanten: s,
                target: Some(target),
            };
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahjong_tiles::parse_tiles;

    fn c(s: &str) -> TileCounts {
        TileCounts::from_tiles(&parse_tiles(s).unwrap())
    }

    fn fs(hand: &str, melds: &[Meld], rules: &RuleSet) -> FaanShanten {
        let concealed = c(hand);
        let mut unseen = TileCounts::full_set();
        for t in concealed.tiles() {
            unseen.remove(t);
        }
        let view = HandView {
            concealed: &concealed,
            melds,
            seat: 1,
            bonus: 0,
            unseen: &unseen,
        };
        faan_shanten(&view, rules)
    }

    fn no_flowers() -> RuleSet {
        RuleSet {
            flowers: false,
            ..RuleSet::default()
        }
    }

    #[test]
    fn a_ready_chicken_hand_is_not_ready_under_the_minimum() {
        // Ready by tile count (waiting on 9p), but worth only 門前清.
        let r = no_flowers();
        let s = fs("123m 456p 789s 234s 9p", &[], &r);
        assert!(s.shanten >= 1, "{s:?}");
        // With a dragon pung and the concealed bonus it reaches... still 2 faan.
        // A half-flush plan in one suit makes it worth 4 when concealed.
        let s = fs("123p 456p 789p RR 1s 5s", &[], &r);
        assert!(s.shanten <= 1, "{s:?}");
    }

    #[test]
    fn complete_valuable_hand_is_minus_one() {
        let s = fs("123p 456p 789p RRR NN", &[], &no_flowers());
        assert_eq!(s.shanten, -1);
    }

    #[test]
    fn thirteen_orphans_target() {
        let s = fs("19m 19p 19s ESWN WhGR", &[], &no_flowers());
        assert_eq!(s.shanten, 0);
        assert_eq!(s.target.unwrap().plan, SuitPlan::Orphans);
    }

    #[test]
    fn no_target_without_any_value() {
        // Min 13 faan with nothing special: unreachable except via limit hands.
        let r = RuleSet {
            min_faan: 13,
            faan_cap: 13,
            flowers: false,
            ..RuleSet::default()
        };
        let s = fs("123m 456p 789s 234s 9p", &[], &r);
        assert!(s.target.as_ref().is_some_and(|t| t.limit) || s.shanten == UNREACHABLE);
    }
}
