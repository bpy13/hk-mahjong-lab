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

#[cfg(test)]
use crate::shanten::standard_shanten;
use crate::shanten::{group_frontier, orphans_shanten, shanten_of, Frontier, Structure};
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
/// `flowers` is the faan of the bonus tiles held ([`flower_faan`]) and
/// `concealed` whether every meld is concealed; both are the same for every
/// target of a hand, so the caller works them out once.
fn guaranteed_faan(
    view: &HandView,
    plan: SuitPlan,
    structure: Structure,
    pungs: &[Tile],
    rules: &RuleSet,
    flowers: u32,
    concealed: bool,
) -> Option<(u8, bool)> {
    if matches!(plan, SuitPlan::Honours | SuitPlan::Orphans) {
        return Some((rules.faan_cap, true));
    }
    let mut f: u32 = 0;
    if concealed {
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

/// Faan of the bonus tiles held. They add faan but do not count toward the
/// minimum (§6).
fn flower_faan(view: &HandView, rules: &RuleSet) -> u32 {
    flower_patterns(view.bonus, view.seat, rules)
        .iter()
        .map(|&(_, x)| x as u32)
        .sum()
}

/// The most scoring pungs a target can plan: three dragons and two value winds.
const MAX_PUNGS: usize = 5;

/// A short list of tiles on the stack (the targets' pungs), so that
/// [`faan_shanten`], which runs hundreds of times per encoded position, does not
/// allocate.
#[derive(Clone, Copy)]
struct Tiles {
    tiles: [Tile; MAX_PUNGS],
    len: usize,
}

impl Tiles {
    fn new() -> Tiles {
        Tiles {
            tiles: [Tile::WHITE; MAX_PUNGS],
            len: 0,
        }
    }
    fn push(&mut self, t: Tile) {
        self.tiles[self.len] = t;
        self.len += 1;
    }
    fn as_slice(&self) -> &[Tile] {
        &self.tiles[..self.len]
    }
}

/// The work [`plan_shanten`] shares between the targets of one hand: the
/// frontier of each suit under each structure, filled in when first needed.
struct PlanShanten<'v, 'a> {
    view: &'v HandView<'a>,
    /// `[structure][suit]`.
    suits: [[Option<Frontier>; 3]; 3],
    /// `[structure]`: the three suits combined (the mixed plan).
    mixed: [Option<Frontier>; 3],
}

impl<'v, 'a> PlanShanten<'v, 'a> {
    fn new(view: &'v HandView<'a>) -> Self {
        PlanShanten {
            view,
            suits: [[None; 3]; 3],
            mixed: [None; 3],
        }
    }

    fn suit(&mut self, structure: Structure, suit: Suit) -> Frontier {
        let concealed = self.view.concealed;
        *self.suits[structure as usize][suit as usize]
            .get_or_insert_with(|| group_frontier(&concealed.suit_counts(suit), false, structure))
    }

    /// The combined frontier of the suit tiles the plan keeps.
    fn suits_of(&mut self, plan: SuitPlan, structure: Structure) -> Frontier {
        match plan {
            SuitPlan::Mixed => {
                if let Some(f) = self.mixed[structure as usize] {
                    return f;
                }
                let mut f = Frontier::ZERO;
                for s in Suit::ALL {
                    f = f.combine(&self.suit(structure, s));
                }
                self.mixed[structure as usize] = Some(f);
                f
            }
            SuitPlan::HalfFlush(s) | SuitPlan::FullFlush(s) => self.suit(structure, s),
            SuitPlan::Honours | SuitPlan::Orphans => Frontier::ZERO,
        }
    }

    /// Shanten toward one plan, or `None` if the plan is impossible.
    ///
    /// The same as [`standard_shanten`] on the hand with the tiles the plan
    /// forbids taken out and the planned pungs set aside (the pungs are all
    /// honours, so only the honour group changes from target to target).
    fn get(&mut self, plan: SuitPlan, structure: Structure, pungs: &[Tile]) -> Option<i8> {
        let view = self.view;
        if plan == SuitPlan::Orphans {
            return view
                .melds
                .is_empty()
                .then(|| orphans_shanten(view.concealed));
        }
        if !melds_fit(view.melds, plan, structure) {
            return None;
        }
        // Honours are allowed by every plan here except a full flush.
        let mut honours = [0u8; 7];
        if !matches!(plan, SuitPlan::FullFlush(_)) {
            honours.copy_from_slice(&view.concealed.0[27..34]);
        }
        let mut melds = view.melds.len();
        let mut extra = 0i8;
        for &t in pungs {
            if view.melds.iter().any(|m| m.is_pung_like() && m.tile == t) {
                continue;
            }
            debug_assert!(t.is_honour() && plan.allows(t), "scoring pungs are honours");
            let h = &mut honours[t.index() as usize - 27];
            let take = (*h).min(3);
            let missing = 3 - take;
            if view.unseen[t] < missing {
                return None;
            }
            *h -= take;
            melds += 1;
            extra += missing as i8;
        }
        if melds > 4 {
            return None;
        }
        let f = group_frontier(&honours, true, structure).combine(&self.suits_of(plan, structure));
        Some(shanten_of(&f, melds) + extra)
    }
}

/// Shanten toward one plan, or `None` if the plan is impossible (the reference
/// form of [`PlanShanten::get`], kept for its test).
#[cfg(test)]
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
/// already held at least in pairs (`optional`), or already pung'ed on the table
/// (`on_table`). Returns `(on_table, optional)`.
fn value_pung_candidates(view: &HandView, rules: &RuleSet) -> (Tiles, Tiles) {
    let mut values = Tiles::new();
    for t in [Tile::WHITE, Tile::GREEN, Tile::RED, Tile::wind(view.seat)] {
        values.push(t);
    }
    if rules.prevailing_wind != view.seat {
        values.push(Tile::wind(rules.prevailing_wind));
    }
    let mut on_table = Tiles::new();
    for &t in values.as_slice() {
        if view.melds.iter().any(|m| m.is_pung_like() && m.tile == t) {
            on_table.push(t);
        }
    }
    let mut optional = Tiles::new();
    for &t in values.as_slice() {
        if !on_table.as_slice().contains(&t) && view.concealed[t] >= 2 {
            optional.push(t);
        }
    }
    (on_table, optional)
}

/// Calls `visit(plan, structure, pungs, faan, limit)` for every plan worth
/// considering for this hand, in the order [`targets`] lists them.
fn for_each_target(
    view: &HandView,
    rules: &RuleSet,
    mut visit: impl FnMut(SuitPlan, Structure, &[Tile], u8, bool),
) {
    let mut all = *view.concealed;
    for m in view.melds {
        m.add_to(&mut all);
    }
    let best_suit = Suit::ALL
        .into_iter()
        .max_by_key(|&s| all.suit_counts(s).iter().map(|&c| c as u32).sum::<u32>())
        .expect("three suits");
    let honours: u32 = (27..34).map(|i| all.0[i] as u32).sum();
    let plans = [
        SuitPlan::Mixed,
        SuitPlan::HalfFlush(best_suit),
        SuitPlan::FullFlush(best_suit),
        SuitPlan::Honours,
    ];
    let plans = &plans[..if honours >= 7 { 4 } else { 3 }];
    let (fixed, optional) = value_pung_candidates(view, rules);
    let flowers = flower_faan(view, rules);
    let concealed = view.melds.iter().all(|m| m.concealed);
    for &plan in plans {
        for structure in [Structure::Any, Structure::AllPungs, Structure::AllChows] {
            for mask in 0..(1u32 << optional.len) {
                let mut pungs = fixed;
                for (i, &t) in optional.as_slice().iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        pungs.push(t);
                    }
                }
                let pungs = pungs.as_slice();
                if structure == Structure::AllChows && !pungs.is_empty() {
                    continue;
                }
                if pungs.iter().any(|&t| !plan.allows(t)) {
                    continue;
                }
                if let Some((faan, limit)) =
                    guaranteed_faan(view, plan, structure, pungs, rules, flowers, concealed)
                {
                    visit(plan, structure, pungs, faan, limit);
                }
            }
        }
    }
    if view.melds.is_empty() {
        visit(SuitPlan::Orphans, Structure::Any, &[], rules.faan_cap, true);
    }
}

/// The plans worth considering for this hand.
pub fn targets(view: &HandView, rules: &RuleSet) -> Vec<Target> {
    let mut out = Vec::new();
    for_each_target(view, rules, |plan, structure, pungs, faan, limit| {
        out.push(Target {
            plan,
            structure,
            pungs: pungs.to_vec(),
            faan,
            limit,
        });
    });
    out
}

/// Faan-aware shanten of the hand (see the module docs).
pub fn faan_shanten(view: &HandView, rules: &RuleSet) -> FaanShanten {
    let mut plans = PlanShanten::new(view);
    // The best target so far: shanten, plan, structure, pungs, faan, limit.
    let mut best: Option<(i8, SuitPlan, Structure, Tiles, u8, bool)> = None;
    for_each_target(view, rules, |plan, structure, pungs, faan, limit| {
        let Some(s) = plans.get(plan, structure, pungs) else {
            return;
        };
        let better = match &best {
            None => true,
            Some(b) => (s, std::cmp::Reverse(faan)) < (b.0, std::cmp::Reverse(b.4)),
        };
        if better {
            let mut kept = Tiles::new();
            for &t in pungs {
                kept.push(t);
            }
            best = Some((s, plan, structure, kept, faan, limit));
        }
    });
    match best {
        None => FaanShanten {
            shanten: UNREACHABLE,
            target: None,
        },
        Some((shanten, plan, structure, pungs, faan, limit)) => FaanShanten {
            shanten,
            target: Some(Target {
                plan,
                structure,
                pungs: pungs.as_slice().to_vec(),
                faan,
                limit,
            }),
        },
    }
}

/// [`faan_shanten`] as it was first written (one full [`standard_shanten`] per
/// target), for the test that the shared-work version agrees with it.
#[cfg(test)]
fn faan_shanten_reference(view: &HandView, rules: &RuleSet) -> FaanShanten {
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

    /// The shared-work `faan_shanten` gives exactly what the original (one
    /// full `standard_shanten` per target) gave, on random hands with random
    /// melds, seats, flowers and rules.
    #[test]
    fn faan_shanten_matches_the_reference() {
        use mahjong_hk::MeldKind;
        use rand::rngs::SmallRng;
        use rand::{Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(3);
        let mut seen_plans = std::collections::HashSet::new();
        let mut pungs_planned = 0u32;
        for case in 0..20_000 {
            let mut wall: Vec<Tile> = TileCounts::full_set().tiles();
            let mut melds = Vec::new();
            // Often keep to one suit and honours, so flush and honour plans occur.
            let suit = rng.gen_range(0..3u8);
            let narrow = rng.gen_bool(0.4);
            let pick = |rng: &mut SmallRng| -> Tile {
                let i = if narrow && rng.gen_bool(0.7) {
                    suit * 9 + rng.gen_range(0..9)
                } else if rng.gen_bool(0.3) {
                    rng.gen_range(27..34)
                } else {
                    rng.gen_range(0..34)
                };
                Tile::from_index(i)
            };
            for _ in 0..rng.gen_range(0..=3) {
                let t = pick(&mut rng);
                let m = match rng.gen_range(0..4) {
                    0 if !t.is_honour() && t.index() % 9 <= 6 => {
                        Meld::exposed(MeldKind::Chow, t, 0)
                    }
                    1 => Meld::concealed_kong(t),
                    2 => Meld::exposed(MeldKind::Kong, t, 0),
                    _ => Meld::exposed(MeldKind::Pung, t, 0),
                };
                let tiles = m.tiles();
                if !tiles.iter().all(|x| wall.contains(x))
                    || tiles.iter().any(|x| {
                        tiles.iter().filter(|&y| y == x).count() > {
                            wall.iter().filter(|&w| w == x).count()
                        }
                    })
                {
                    continue;
                }
                for x in tiles {
                    let i = wall.iter().position(|&w| w == x).unwrap();
                    wall.swap_remove(i);
                }
                melds.push(m);
            }
            let size = 13 + rng.gen_range(0..2) - 3 * melds.len();
            let mut concealed = TileCounts::new();
            while concealed.0.iter().map(|&c| c as usize).sum::<usize>() < size {
                let t = pick(&mut rng);
                if let Some(i) = wall.iter().position(|&w| w == t) {
                    concealed.add(wall.swap_remove(i));
                }
            }
            let mut unseen = TileCounts::full_set();
            for t in concealed.tiles() {
                unseen.remove(t);
            }
            for m in &melds {
                for t in m.tiles() {
                    unseen.remove(t);
                }
            }
            // Random extra tiles seen elsewhere.
            for _ in 0..rng.gen_range(0..30) {
                let t = Tile::from_index(rng.gen_range(0..34));
                if unseen[t] > 0 {
                    unseen.remove(t);
                }
            }
            let rules = RuleSet {
                min_faan: rng.gen_range(0..8),
                flowers: rng.gen_bool(0.5),
                prevailing_wind: rng.gen_range(0..4),
                ..RuleSet::default()
            };
            let view = HandView {
                concealed: &concealed,
                melds: &melds,
                seat: rng.gen_range(0..4),
                bonus: rng.gen(),
                unseen: &unseen,
            };
            let got = faan_shanten(&view, &rules);
            assert_eq!(
                got,
                faan_shanten_reference(&view, &rules),
                "case {case}: {concealed:?} {melds:?}"
            );
            if let Some(t) = &got.target {
                seen_plans.insert(std::mem::discriminant(&t.plan));
                pungs_planned += !t.pungs.is_empty() as u32;
            }
        }
        // The cases reached every kind of plan and planned scoring pungs.
        assert_eq!(seen_plans.len(), 5, "plans reached: {}", seen_plans.len());
        assert!(pungs_planned > 1000, "{pungs_planned}");
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
