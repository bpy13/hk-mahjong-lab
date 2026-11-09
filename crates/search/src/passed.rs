//! Passed-tile safety and the risk-taking read (plan 2026-11-09 Tasks 14–15,
//! `docs/race-bot.md` §13). Both read only public events.
//!
//! **Passed tiles.** When another seat discards `t` and seat `P` does not win on
//! it, `P` has "let `t` pass": either `P` was not waiting on `t`, or the passed-win
//! lock (rules §4.4) now stops `P` from winning on it until `P`'s next turn. Either
//! way `t` cannot deal in to `P` before `P`'s next turn, and afterwards only as far
//! as `P`'s hand has changed: a 摸打 (the drawn tile goes straight out) leaves the
//! hand as it was, a 入章 (a tile from the hand goes out) changes it. [`Passed`] keeps,
//! per seat and tile, "passed, with `N` 入章 since", and [`Passed::factor`] turns
//! that into a multiplier on the table's deal-in estimate: 0 while the lock holds,
//! then [`PASSED_FACTOR`]`[N]` rising toward 1. A 換章 (Brian's definition: a pushing
//! player throws a tile that was clearly useful to them, so the hand took a new
//! shape) resets every passed tile of that seat ([`reshaped`]).
//!
//! **Risk taking.** [`risk_scores`] sums, per seat, the public danger of the tiles it
//! threw (the danger tables as the thrower saw them at the time, against the other
//! three seats), discounted by [`RISK_DECAY`] per later discard. The race bot reads
//! an opponent's push probability, closeness and hand value from it
//! (`race::RiskRead`).
//!
//! The constants are fitted by `cargo run --release -p mahjong-search --example
//! fit_passed`; each says where its numbers came from.

use crate::danger::{calibrate, Opponent, PROGRESS_BUCKETS};
use crate::danger_tables::{THREAT, WAIT};
use mahjong_hk::{Event, Meld, State};
use mahjong_tiles::{Suit, Tile, TileCounts, NUM_KINDS};

/// `f(N)`: the deal-in estimate of a passed tile against a seat with `N` 入章 since
/// it passed (`N = 0`: the seat has had a turn but kept its hand), as a share of
/// the estimate for a tile that did not pass. Index 4 is `N ≥ 4`.
///
/// Fitted by `fit_passed` (2026-11-09, 8,000 hands, seed 1, mixed field, the
/// heuristic belief, 29 M tile-seat cases): the true rate at which the seat was
/// waiting on the tile, divided by the table's estimate, relative to the same
/// ratio over tiles that did not pass (0.957). After a 入章 the tile is at 0.55 of
/// the unconditioned rate; four 入章 later the record says nothing any more. See
/// `docs/race-bot.md` §13.
pub const PASSED_FACTOR: [f64; 5] = [0.0, 0.546, 0.618, 0.763, 0.973];

/// [`PASSED_FACTOR`] against `models/belief.mjnn`'s estimates (`fit_passed --belief
/// models/belief.mjnn`, same run): the model already rates passed tiles lower, so
/// less of the fade is left for the factor. Used by `race:belief=...`.
pub const PASSED_FACTOR_MODEL: [f64; 5] = [0.0, 0.722, 0.804, 1.0, 1.0];

/// Weight of a seat's earlier discards in its risk score, per later discard of
/// that seat (a half-life of about 2.4 discards).
pub const RISK_DECAY: f64 = 0.75;

/// The public facts about one discard that the 換章 test may read.
#[derive(Clone, Copy, Debug)]
pub struct DiscardContext {
    pub seat: usize,
    pub tile: Tile,
    /// 入章 (from the hand) rather than 摸打 (the tile just drawn).
    pub from_hand: bool,
    /// The seat's visible flush suit: some exposed meld, every suited meld in one
    /// suit.
    pub flush: Option<Suit>,
    /// The seat's exposed melds so far.
    pub exposed: usize,
    /// The kinds the seat discarded before this one.
    pub discarded: [bool; NUM_KINDS],
}

impl DiscardContext {
    /// The context of `seat` discarding `tile` from `state` now (before the
    /// discard is applied).
    pub fn now(state: &State, seat: usize, tile: Tile, from_hand: bool) -> DiscardContext {
        let mut discarded = [false; NUM_KINDS];
        for e in state.events() {
            if let Event::Discard { seat: s, tile, .. } = *e {
                if s as usize == seat {
                    discarded[tile.index() as usize] = true;
                }
            }
        }
        let melds = state.melds(seat);
        DiscardContext {
            seat,
            tile,
            from_hand,
            flush: flush_suit(melds),
            exposed: melds.exposed_count(),
            discarded,
        }
    }
}

/// A 換章 test: does this discard show the seat's hand took a new shape?
pub type Reshape = fn(&DiscardContext) -> bool;

/// The 換章 test the bots use (Brian's definition, public view): a tile from the
/// hand that was clearly useful to the thrower, read as a tile of the suit the
/// thrower's exposed melds show it collecting (every suited meld in one suit).
/// Chosen by `fit_passed` among the public tests it reports (2026-11-09, 8,000
/// hands): after such a discard the seat's waits or distance had changed in 55 %
/// of cases, against 39.5 % after any 入章 and 39 % after a middle tile 3–7 from
/// the hand, and the tiles it had let pass were back at or above the unconditioned
/// deal-in rate. See `docs/race-bot.md` §13.
pub fn reshaped(c: &DiscardContext) -> bool {
    c.from_hand && c.flush.is_some() && c.tile.suit() == c.flush
}

/// One seat's record of one tile kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cell {
    /// Another seat discarded the tile and this seat did not win on it.
    pub passed: bool,
    /// The seat has had a turn since (a discard, a claim or a kong).
    pub turned: bool,
    /// The seat's 入章 since (a claim or kong counts as one).
    pub changes: u8,
    /// Reset bits: bit 0 is [`reshaped`], bit `1 + i` the `i`th extra test of
    /// [`Passed::of_with`]. A set bit 0 ends the tile's safety.
    pub reset: u8,
}

/// Which tiles each seat let pass and how much its hand changed since.
#[derive(Clone, Debug)]
pub struct Passed {
    cells: [[Cell; NUM_KINDS]; 4],
}

impl Passed {
    /// Scans `state`'s public events with the bots' 換章 test.
    pub fn of(state: &State) -> Passed {
        Passed::of_with(state, &[])
    }

    /// As [`Passed::of`], also recording in each cell's `reset` bits whether the
    /// `extra` tests fired (for the fitter; they do not affect [`Passed::factor`]).
    pub fn of_with(state: &State, extra: &[Reshape]) -> Passed {
        let mut cells = [[Cell::default(); NUM_KINDS]; 4];
        let mut discarded = [[false; NUM_KINDS]; 4];
        let mut melds: [Vec<Meld>; 4] = Default::default();
        for e in state.events() {
            match *e {
                Event::Discard {
                    seat,
                    tile,
                    from_hand,
                } => {
                    let seat = seat as usize;
                    let i = tile.index() as usize;
                    for p in (0..4).filter(|&p| p != seat) {
                        cells[p][i] = Cell {
                            passed: true,
                            ..Cell::default()
                        };
                    }
                    let ctx = DiscardContext {
                        seat,
                        tile,
                        from_hand,
                        flush: flush_suit(&melds[seat]),
                        exposed: melds[seat].iter().filter(|m| !m.concealed).count(),
                        discarded: discarded[seat],
                    };
                    let mut bits = reshaped(&ctx) as u8;
                    for (k, test) in extra.iter().enumerate() {
                        bits |= (test(&ctx) as u8) << (k + 1);
                    }
                    for c in cells[seat].iter_mut().filter(|c| c.passed) {
                        c.turned = true;
                        c.changes = c.changes.saturating_add(from_hand as u8);
                        c.reset |= bits;
                    }
                    discarded[seat][i] = true;
                }
                Event::Claim { seat, meld, .. } => {
                    let seat = seat as usize;
                    melds[seat].push(meld);
                    for c in cells[seat].iter_mut().filter(|c| c.passed) {
                        c.turned = true;
                    }
                }
                Event::ConcealedKong { seat, .. } | Event::AddedKong { seat, .. } => {
                    for c in cells[seat as usize].iter_mut().filter(|c| c.passed) {
                        c.turned = true;
                        c.changes = c.changes.saturating_add(1);
                    }
                }
                _ => {}
            }
        }
        Passed { cells }
    }

    /// The record of `tile` for `seat`.
    pub fn cell(&self, seat: usize, tile: Tile) -> Cell {
        self.cells[seat][tile.index() as usize]
    }

    /// The multiplier on the deal-in estimate of `tile` against `seat`: 0 while the
    /// passed-win lock holds (no turn since it passed), [`PASSED_FACTOR`] by the
    /// 入章 since, 1 for a tile that did not pass or whose safety a 換章 reset.
    pub fn factor(&self, seat: usize, tile: Tile) -> f64 {
        self.factor_with(seat, tile, &PASSED_FACTOR)
    }

    /// [`Passed::factor`] with another fade table (`f(N)` by 入章 since; the last
    /// entry covers every larger `N`).
    pub fn factor_with(&self, seat: usize, tile: Tile, table: &[f64; 5]) -> f64 {
        let c = self.cells[seat][tile.index() as usize];
        if !c.passed || c.reset & 1 != 0 {
            1.0
        } else if !c.turned {
            0.0
        } else {
            table[(c.changes as usize).min(table.len() - 1)]
        }
    }
}

/// A seat's visible flush suit (`danger.rs`'s tag): some exposed meld, every
/// suited meld in one suit.
pub fn flush_suit(melds: &[Meld]) -> Option<Suit> {
    if !melds.iter().any(|m| !m.concealed) {
        return None;
    }
    let mut suits = melds.iter().filter_map(|m| m.tile.suit());
    let first = suits.next()?;
    suits.all(|s| s == first).then_some(first)
}

/// Each seat's risk score: for every tile it discarded, the public danger of that
/// tile as the danger tables rated it when thrown (the thrower's view: the other
/// three seats' threat by exposed melds and wall progress, the tile's wait rate by
/// class, copies still unseen on the table and whether the target seat had
/// discarded it) **less the mean danger of every kind still unseen then** (what a
/// tile picked without regard to safety would have scored), summed with
/// [`RISK_DECAY`] per later discard of the same seat. Positive: the seat throws
/// more dangerous tiles than the table offers; negative: it picks safe ones. The
/// plain danger was tried first and only measured how threatening the table was
/// (`fit_passed`, 2026-11-09). The thrower's own hand is unknown to us, so the
/// unseen copies count only what lay on the table. Wall progress at each discard is
/// read back from the discards and claims that followed (one draw per discard,
/// none after a claim).
pub fn risk_scores(state: &State) -> [f64; 4] {
    let rules = state.rules();
    let initial = if rules.flowers { 91.0 } else { 83.0 };
    let events = state.events();
    let (mut discards_left, mut claims_left) = (0usize, 0usize);
    for e in events {
        match e {
            Event::Discard { .. } => discards_left += 1,
            Event::Claim { .. } => claims_left += 1,
            _ => {}
        }
    }
    let mut risk = [0.0f64; 4];
    let mut discarded = [[false; NUM_KINDS]; 4];
    let mut melds: [Vec<Meld>; 4] = Default::default();
    let mut unseen = TileCounts::full_set();
    for e in events {
        match *e {
            Event::Discard { seat, tile, .. } => {
                let seat = seat as usize;
                let wall_at = state.wall_remaining() + discards_left - claims_left;
                let progress = (1.0 - wall_at as f64 / initial).clamp(0.0, 0.999);
                let bucket = (progress * PROGRESS_BUCKETS as f64) as usize;
                let others: Vec<(Opponent, f64)> = (0..4)
                    .filter(|&j| j != seat)
                    .map(|j| {
                        let opp = Opponent::from_public(rules, j, &melds[j], bucket, discarded[j]);
                        let threat = THREAT[opp.threat_cell()];
                        (opp, threat)
                    })
                    .collect();
                let danger = |k: Tile| -> f64 {
                    let mut safe = 1.0;
                    for (opp, threat) in &others {
                        if let Some(c) = opp.wait_cell(k, &unseen) {
                            safe *= 1.0 - threat * WAIT[c];
                        }
                    }
                    calibrate(1.0 - safe)
                };
                // The thrown tile's danger over the mean of every kind still unseen
                // on the table: what the seat chose against what it could have
                // chosen, as far as we can tell.
                let (mut sum, mut kinds) = (0.0, 0usize);
                for k in Tile::all_kinds().filter(|&k| unseen[k] > 0) {
                    sum += danger(k);
                    kinds += 1;
                }
                let mean = if kinds > 0 { sum / kinds as f64 } else { 0.0 };
                risk[seat] = RISK_DECAY * risk[seat] + (danger(tile) - mean);
                discarded[seat][tile.index() as usize] = true;
                if unseen[tile] > 0 {
                    unseen.remove(tile);
                }
                discards_left -= 1;
            }
            Event::Claim { seat, meld, .. } => {
                // The claimed tile left the river for the meld: the meld's other
                // tiles become visible.
                for t in meld.tiles().into_iter().skip(1) {
                    if unseen[t] > 0 {
                        unseen.remove(t);
                    }
                }
                melds[seat as usize].push(meld);
                claims_left -= 1;
            }
            Event::ConcealedKong { seat, tile } => {
                for _ in 0..4 {
                    if unseen[tile] > 0 {
                        unseen.remove(tile);
                    }
                }
                melds[seat as usize].push(Meld::concealed_kong(tile));
            }
            Event::AddedKong { tile, .. } if unseen[tile] > 0 => unseen.remove(tile),
            _ => {}
        }
    }
    risk
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use mahjong_games::{Game, Policy};
    use mahjong_hk::{Action, RuleSet};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    /// Over random hands: a tile another seat just discarded is locked (factor 0)
    /// for every other seat until that seat's next discard; after a 摸打 the count
    /// of changes stays, after a 入章 it rises; no factor leaves [0, 1].
    #[test]
    fn passed_cells_follow_the_discards() {
        let mut rng = SmallRng::seed_from_u64(7);
        let bot = Bot::by_name("fast").unwrap();
        let (mut locks, mut kept, mut changed) = (0, 0, 0);
        for _ in 0..30 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            let mut last: Option<(usize, Tile)> = None;
            while let Some(p) = s.current_player() {
                let before = Passed::of(&s);
                if let Some((from, t)) = last {
                    for o in (0..4).filter(|&o| o != from) {
                        let c = before.cell(o, t);
                        assert!(c.passed);
                        // Until o acts, the lock holds.
                        if !c.turned {
                            assert_eq!(before.factor(o, t), 0.0);
                            locks += 1;
                        }
                    }
                }
                for o in 0..4 {
                    for t in Tile::all_kinds() {
                        let f = before.factor(o, t);
                        assert!((0.0..=1.0).contains(&f));
                    }
                }
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                if let (Action::Discard(t), "turn") = (a, s.phase_name()) {
                    let from_hand = s.last_drawn() != Some(t);
                    s.apply(a);
                    let after = Passed::of(&s);
                    for (k, c) in Tile::all_kinds()
                        .map(|k| (k, after.cell(p, k)))
                        .filter(|(_, c)| c.passed)
                    {
                        let old = before.cell(p, k);
                        if old.passed && old.reset & 1 == 0 && c.reset & 1 == 0 {
                            assert!(c.turned);
                            assert_eq!(c.changes, old.changes + from_hand as u8);
                            if from_hand {
                                changed += 1
                            } else {
                                kept += 1
                            }
                        }
                    }
                    last = Some((p, t));
                    continue;
                }
                s.apply(a);
                if !matches!(a, Action::Discard(_)) {
                    last = None;
                }
            }
        }
        assert!(
            locks > 0 && kept > 0 && changed > 0,
            "{locks} {kept} {changed}"
        );
    }

    /// A 換章 (a tile of the seat's visible flush suit, from the hand) resets every
    /// passed tile of that seat.
    #[test]
    fn a_reshape_resets_the_seat() {
        let mut rng = SmallRng::seed_from_u64(11);
        let bot = Bot::by_name("value").unwrap();
        let mut resets = 0;
        for _ in 0..20 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(p) = s.current_player() {
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                if let (Action::Discard(t), "turn") = (a, s.phase_name()) {
                    let from_hand = s.last_drawn() != Some(t);
                    let fires =
                        from_hand && flush_suit(s.melds(p)).is_some_and(|f| t.suit() == Some(f));
                    let before = Passed::of(&s);
                    s.apply(a);
                    let after = Passed::of(&s);
                    for k in Tile::all_kinds() {
                        if before.cell(p, k).passed && fires {
                            assert_eq!(after.factor(p, k), 1.0, "{k} after {t}");
                            resets += 1;
                        }
                    }
                    continue;
                }
                s.apply(a);
            }
        }
        assert!(resets > 0);
    }

    /// Risk scores are finite, and a seat that has thrown nothing scores 0.
    #[test]
    fn risk_scores_are_sane() {
        let mut rng = SmallRng::seed_from_u64(5);
        let bot = Bot::by_name("fast").unwrap();
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        assert_eq!(risk_scores(&s), [0.0; 4]);
        let mut positive = false;
        while s.current_player().is_some() {
            let r = risk_scores(&s);
            for x in r {
                assert!(x.is_finite() && x.abs() < 2.0);
                positive |= x != 0.0;
            }
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
        }
        assert!(positive);
    }
}
