//! A deal-in danger model from public information, fitted to simulated games.
//!
//! For each opponent it estimates a **threat**: the chance they are already
//! waiting on a tile that would give them a legal win (complete *and* at least the
//! minimum faan). Then, for each tile, how likely a waiting opponent is to win on
//! it. Both are looked up in tables fitted by the `fit_danger` example (see
//! [`crate::danger_tables`]) from games between the rule-based bots:
//!
//! * the threat by the number of exposed melds, how far the wall has gone, and
//!   whether the exposed melds already carry faan or all lie in one suit (a flush);
//! * the per-tile rate by the kind of tile (honour, terminal, 2/8, middle), how
//!   many copies are unseen, whether the opponent has discarded it themselves
//!   (Hong Kong rules let them win on it anyway), and, against a flush, whether
//!   the tile is in the flush suit (or an honour) or another suit.
//!
//! A tile's danger is the chance that at least one opponent wins on it:
//! 1 − Π(1 − threat × rate), then mapped through a calibration curve fitted on the
//! tiles players really discarded (they are not a random sample of the unseen
//! tiles). The `danger_check` example measures the calibration.
//!
//! [`Danger::from_belief`] (plan Task 6) builds the same interface from a
//! belief about each opponent (the trained model, or the tables above through
//! [`crate::HeuristicBelief`]) and adds payments: [`Danger::expected_loss`] in
//! points, [`Danger::points_if_win`] and the 12-tile liability check
//! [`Danger::liability_tile`]. `danger_check --belief <model>` measures the
//! expected loss per wall-progress bucket. [`Danger::estimate`] is unchanged.

use crate::danger_tables::{CALIBRATION, THREAT, WAIT};
use crate::ev::{Belief, Beliefs, FIRST_FAAN_BUCKET};
use mahjong_hk::{Event, State};
use mahjong_tiles::{Tile, TileCounts, NUM_KINDS};

/// Number of wall-progress buckets in the threat table.
pub const PROGRESS_BUCKETS: usize = 6;
/// Size of the threat table: exposed melds (0–4) × progress × meld tag.
pub const THREAT_CELLS: usize = 5 * PROGRESS_BUCKETS * 3;
/// Size of the per-tile table: tile class × unseen copies (1–4) × discarded by
/// the opponent × flush relation.
pub const WAIT_CELLS: usize = 4 * 4 * 2 * 3;

/// What a seat's exposed melds suggest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tag {
    exposed: usize,
    /// 0 = nothing special, 1 = a meld already carries faan, 2 = one suit only.
    kind: usize,
    flush: Option<mahjong_tiles::Suit>,
}

fn tag(state: &State, o: usize) -> Tag {
    let rules = state.rules();
    let melds = state.melds(o);
    let exposed = melds.exposed_count().min(4);
    let mut suits: Vec<_> = melds.iter().filter_map(|m| m.tile.suit()).collect();
    suits.sort();
    suits.dedup();
    let flush = (exposed >= 1 && suits.len() == 1).then(|| suits[0]);
    let has_value = melds.iter().any(|m| {
        m.is_pung_like()
            && (m.tile.is_dragon()
                || m.tile == Tile::wind(o)
                || m.tile == Tile::wind(rules.prevailing_wind))
    });
    let kind = if flush.is_some() {
        2
    } else if has_value {
        1
    } else {
        0
    };
    Tag {
        exposed,
        kind,
        flush,
    }
}

/// What the observer can see about one opponent, for the table lookups.
#[derive(Clone, Debug)]
pub struct Opponent {
    tag: Tag,
    progress: usize,
    discarded: [bool; NUM_KINDS],
}

impl Opponent {
    /// Public information about opponent `o`.
    pub fn of(state: &State, o: usize) -> Opponent {
        let initial_wall = if state.rules().flowers { 91.0 } else { 83.0 };
        let progress = (1.0 - state.wall_remaining() as f64 / initial_wall).clamp(0.0, 0.999);
        let mut discarded = [false; NUM_KINDS];
        for e in state.events() {
            if let Event::Discard { seat, tile } = *e {
                if seat as usize == o {
                    discarded[tile.index() as usize] = true;
                }
            }
        }
        Opponent {
            tag: tag(state, o),
            progress: (progress * PROGRESS_BUCKETS as f64) as usize,
            discarded,
        }
    }

    /// The threat-table cell.
    pub fn threat_cell(&self) -> usize {
        (self.tag.exposed * PROGRESS_BUCKETS + self.progress) * 3 + self.tag.kind
    }

    /// The per-tile cell of tile `t`, given the copies of each kind unseen by the
    /// observer; `None` when no copy is unseen (the opponent cannot wait on it).
    pub fn wait_cell(&self, t: Tile, unseen: &TileCounts) -> Option<usize> {
        let left = unseen[t].min(4) as usize;
        if left == 0 {
            return None;
        }
        let class = if t.is_honour() {
            0
        } else {
            match t.number() {
                Some(1 | 9) => 1,
                Some(2 | 8) => 2,
                _ => 3,
            }
        };
        let discarded = self.discarded[t.index() as usize] as usize;
        let relation = match self.tag.flush {
            None => 0,
            Some(s) if t.suit().is_none_or(|x| x == s) => 1,
            Some(_) => 2,
        };
        Some(((class * 4 + (left - 1)) * 2 + discarded) * 3 + relation)
    }
}

/// Maps a raw estimate to the deal-in rate observed for it on real discards
/// (piecewise linear through [`CALIBRATION`]; proportional below the first point,
/// flat above the last).
fn calibrate(raw: f64) -> f64 {
    let first = CALIBRATION[0];
    if raw <= first.0 {
        return if first.0 > 0.0 {
            raw * first.1 / first.0
        } else {
            first.1
        };
    }
    for w in CALIBRATION.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if raw <= x1 {
            return y0 + (y1 - y0) * (raw - x0) / (x1 - x0);
        }
    }
    CALIBRATION[CALIBRATION.len() - 1].1
}

/// The payment-aware parts of a [`Danger`] built from a belief (plan Task 6).
#[derive(Clone, Debug)]
struct Payment {
    /// `P_o(t)`: P(seat `o` wins on `t` if I discard it now); 0 for me and for
    /// dead tiles.
    p: [[f64; NUM_KINDS]; 4],
    /// `Pay(o)`: expected points per share of `o`'s discard win now.
    pay: [f64; 4],
    /// Shares I pay when `o` wins on my discard: 1, 3 when I am liable for `o`,
    /// 0 when someone else is (they pay everything).
    shares: [f64; 4],
    /// Seats with three exposed melds and nobody liable for them yet.
    three_exposed: [bool; 4],
    /// `P_claim4_o(t)`: o can claim `t` as its fourth exposed meld (a pung from
    /// any seat, a chow if o is my next seat), from the belief's counts.
    claim4: [[f64; NUM_KINDS]; 4],
    /// Whether the unseen copies leave that claim possible at all.
    can_claim4: [[bool; NUM_KINDS]; 4],
}

/// Estimated danger of each tile for one seat.
#[derive(Clone, Debug)]
pub struct Danger {
    tiles: [f64; NUM_KINDS],
    threat: [f64; 4],
    /// Present only when built by [`Danger::from_belief`].
    payment: Option<Box<Payment>>,
}

impl Danger {
    /// Estimates danger for `me` from public information.
    pub fn estimate(state: &State, me: usize) -> Danger {
        let unseen = state.unseen(me);
        let mut safe = [1.0f64; NUM_KINDS];
        let mut threat = [0.0f64; 4];
        for o in (0..4).filter(|&o| o != me) {
            let opp = Opponent::of(state, o);
            let t_o = THREAT[opp.threat_cell()];
            threat[o] = t_o;
            for t in Tile::all_kinds() {
                if let Some(c) = opp.wait_cell(t, &unseen) {
                    safe[t.index() as usize] *= 1.0 - t_o * WAIT[c];
                }
            }
        }
        let mut tiles = [0.0f64; NUM_KINDS];
        for i in 0..NUM_KINDS {
            tiles[i] = calibrate(1.0 - safe[i]);
        }
        Danger {
            tiles,
            threat,
            payment: None,
        }
    }

    /// Payment-aware danger for `me` from a belief about each opponent (plan
    /// Task 6): the trained model ([`crate::belief::BeliefModel`]) or, as the
    /// no-model fallback, [`crate::HeuristicBelief`] (the fitted tables).
    ///
    /// `tile(t)` is `1 − Π_o (1 − tenpai_o × wait_o(t))` (no calibration curve:
    /// the model is calibrated per bucket, `docs/belief.md` §4), `threat(o)` is
    /// `tenpai_o`, and [`Danger::expected_loss`], [`Danger::points_if_win`] and
    /// [`Danger::liability_tile`] become available. A tile with no unseen copy is
    /// dead: danger 0, expected loss 0.
    pub fn from_belief(state: &State, me: usize, belief: &dyn Belief) -> Danger {
        Self::from_beliefs(state, me, &belief.estimate(state, me))
    }

    /// [`Danger::from_belief`] from beliefs already estimated.
    pub fn from_beliefs(state: &State, me: usize, beliefs: &Beliefs) -> Danger {
        let rules = state.rules();
        let unseen = state.unseen(me);
        let points = |f: u8| rules.points(f.min(rules.faan_cap)) as f64;
        let mut pay = Payment {
            p: [[0.0; NUM_KINDS]; 4],
            pay: [0.0; 4],
            shares: [0.0; 4],
            three_exposed: [false; 4],
            claim4: [[0.0; NUM_KINDS]; 4],
            can_claim4: [[false; NUM_KINDS]; 4],
        };
        let mut safe = [1.0f64; NUM_KINDS];
        let mut threat = [0.0f64; 4];
        for o in (0..4).filter(|&o| o != me) {
            let b = &beliefs.seats[o];
            threat[o] = b.tenpai;
            pay.pay[o] = b
                .faan
                .iter()
                .enumerate()
                .map(|(k, &pb)| pb * points(FIRST_FAAN_BUCKET + k as u8))
                .sum();
            pay.shares[o] = match state.liable_for(o) {
                None => 1.0,
                Some(l) if l == me => 3.0,
                Some(_) => 0.0,
            };
            pay.three_exposed[o] = rules.twelve_tile_liability
                && state.melds(o).exposed_count() == 3
                && state.liable_for(o).is_none();
            let next = o == (me + 1) % 4;
            for t in Tile::all_kinds() {
                let i = t.index() as usize;
                if unseen[t] == 0 {
                    continue;
                }
                let p = (b.tenpai * b.wait[i]).clamp(0.0, 1.0);
                pay.p[o][i] = p;
                safe[i] *= 1.0 - p;
                if pay.three_exposed[o] {
                    let mut chow = 0.0f64;
                    let mut chow_possible = false;
                    if next && t.suit().is_some() {
                        for (a, c) in [(-2i8, -1i8), (-1, 1), (1, 2)] {
                            if let (Some(u), Some(v)) = (t.offset(a), t.offset(c)) {
                                chow_possible |= unseen[u] >= 1 && unseen[v] >= 1;
                                chow = chow.max(
                                    b.has_one[u.index() as usize] * b.has_one[v.index() as usize],
                                );
                            }
                        }
                    }
                    pay.can_claim4[o][i] = unseen[t] >= 2 || chow_possible;
                    pay.claim4[o][i] = (b.has_two[i] + chow).min(1.0);
                }
            }
        }
        let mut tiles = [0.0f64; NUM_KINDS];
        for i in 0..NUM_KINDS {
            tiles[i] = 1.0 - safe[i];
        }
        Danger {
            tiles,
            threat,
            payment: Some(Box::new(pay)),
        }
    }

    /// Expected points I lose if I discard `t` now: `Σ_o P_o(t) × Pay(o) ×
    /// shares(o)` (every winner is paid under the default `multiple_winners`
    /// rule). Built by [`Danger::estimate`], which knows no hand values, it is
    /// `tile(t) × 8` (one share of the 3-faan minimum).
    pub fn expected_loss(&self, t: Tile) -> f64 {
        let i = t.index() as usize;
        match &self.payment {
            Some(p) => (0..4).map(|o| p.p[o][i] * p.pay[o] * p.shares[o]).sum(),
            None => self.tiles[i] * 8.0,
        }
    }

    /// Expected points per share if `seat` wins on a discard now (`Pay(seat)`);
    /// 8 (the 3-faan minimum) when built by [`Danger::estimate`].
    pub fn points_if_win(&self, seat: usize) -> f64 {
        match &self.payment {
            Some(p) => p.pay[seat],
            None => 8.0,
        }
    }

    /// The seat that could claim `t` as its **fourth exposed meld** (12-tile
    /// rule, rules §8.3), making me liable for it: a seat showing three exposed
    /// melds with nobody liable yet, when the unseen copies leave a pung (or, for
    /// my next seat, a chow) possible. With several, the one the belief thinks
    /// likelier to hold the tiles. `None` when built by [`Danger::estimate`].
    pub fn liability_tile(&self, t: Tile) -> Option<usize> {
        let p = self.payment.as_ref()?;
        let i = t.index() as usize;
        (0..4)
            .filter(|&o| p.three_exposed[o] && p.can_claim4[o][i])
            .max_by(|&a, &b| p.claim4[a][i].total_cmp(&p.claim4[b][i]).then(b.cmp(&a)))
    }

    /// The belief's chance that `seat` can claim `t` as its fourth exposed meld
    /// (0 when [`Danger::liability_tile`] does not name `seat` as possible).
    pub fn liability_chance(&self, seat: usize, t: Tile) -> f64 {
        let i = t.index() as usize;
        self.payment.as_ref().map_or(0.0, |p| {
            if p.three_exposed[seat] && p.can_claim4[seat][i] {
                p.claim4[seat][i]
            } else {
                0.0
            }
        })
    }

    /// Estimated probability that discarding `t` now lets an opponent win.
    pub fn tile(&self, t: Tile) -> f64 {
        self.tiles[t.index() as usize]
    }

    /// Estimated probability that `seat` is waiting with a legal win.
    pub fn threat(&self, seat: usize) -> f64 {
        self.threat[seat]
    }

    /// The largest opponent threat.
    pub fn max_threat(&self) -> f64 {
        self.threat.iter().copied().fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::BeliefModel;
    use crate::bots::Bot;
    use crate::ev::HeuristicBelief;
    use mahjong_games::{Game, Policy};
    use mahjong_hk::{Meld, MeldKind, PositionPhase, PositionSpec, RuleSet};
    use mahjong_tiles::parse_tiles;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use std::sync::Arc;

    #[test]
    fn estimates_are_probabilities_and_dead_tiles_are_safe() {
        let mut rng = SmallRng::seed_from_u64(3);
        let bot = Bot::by_name("fast").unwrap();
        for _ in 0..20 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(p) = s.current_player() {
                if s.phase_name() == "turn" {
                    let d = Danger::estimate(&s, p);
                    let unseen = s.unseen(p);
                    for t in Tile::all_kinds() {
                        let x = d.tile(t);
                        assert!((0.0..=1.0).contains(&x));
                        if unseen[t] == 0 {
                            assert_eq!(x, 0.0, "{t} has no unseen copy");
                        }
                    }
                }
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
            }
        }
    }

    /// The trained model when present, else the table fallback.
    fn belief() -> Arc<dyn Belief> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief.mjnn");
        if p.exists() {
            Arc::new(BeliefModel::load(&p).unwrap())
        } else {
            Arc::new(HeuristicBelief)
        }
    }

    #[test]
    fn expected_loss_is_zero_for_dead_tiles() {
        let mut rng = SmallRng::seed_from_u64(4);
        let bot = Bot::by_name("fast").unwrap();
        let beliefs: [Arc<dyn Belief>; 2] = [belief(), Arc::new(HeuristicBelief)];
        let mut dead = 0;
        for _ in 0..8 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(p) = s.current_player() {
                if s.phase_name() == "turn" {
                    let unseen = s.unseen(p);
                    for b in &beliefs {
                        let d = Danger::from_belief(&s, p, b.as_ref());
                        for t in Tile::all_kinds() {
                            let (x, loss) = (d.tile(t), d.expected_loss(t));
                            assert!((0.0..=1.0).contains(&x), "{t}: {x}");
                            assert!(loss.is_finite() && loss >= 0.0, "{t}: {loss}");
                            if unseen[t] == 0 {
                                assert_eq!((x, loss), (0.0, 0.0), "{t} has no unseen copy");
                                dead += 1;
                            }
                        }
                        for o in (0..4).filter(|&o| o != p) {
                            assert!(d.points_if_win(o) >= 8.0 - 1e-9);
                        }
                    }
                }
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
            }
        }
        assert!(dead > 0, "some dead tile was seen");
    }

    fn counts(s: &str) -> TileCounts {
        TileCounts::from_tiles(&parse_tiles(s).unwrap())
    }

    /// Seat 2 shows three exposed pungs (1s from seat 1, 9s and R from seat 3)
    /// and holds `7p 7p 5s 6s`; I (seat 0, on turn) hold one 7p and three Wh.
    fn three_melds_position() -> State {
        let melds: [Vec<Meld>; 4] = [
            vec![],
            vec![],
            vec![
                Meld::exposed(MeldKind::Pung, "1s".parse().unwrap(), 1),
                Meld::exposed(MeldKind::Pung, "9s".parse().unwrap(), 3),
                Meld::exposed(MeldKind::Pung, "R".parse().unwrap(), 3),
            ],
            vec![],
        ];
        let hands = [
            counts("123456789m 7p WhWhWh N"),
            counts("2p 3p 4p 5p 6p 8p 9p 2s 3s 4s 7s 8s E"),
            counts("7p 7p 5s 6s"),
            counts("1m 2m 3m 4m 5m 6m 7m 8m 9m S S W W"),
        ];
        let mut used = [0u8; NUM_KINDS];
        for h in &hands {
            for t in h.kinds() {
                used[t.index() as usize] += h[t];
            }
        }
        for m in &melds[2] {
            for t in m.tiles() {
                used[t.index() as usize] += 1;
            }
        }
        let mut river = vec![
            (1, "1s".parse().unwrap(), true),
            (3, "9s".parse().unwrap(), true),
            (3, "R".parse().unwrap(), true),
        ];
        // The river takes what is left except the wall; 7p stays unseen.
        let wall = 30;
        let mut rest: Vec<Tile> = Vec::new();
        for t in Tile::all_kinds() {
            for _ in used[t.index() as usize]..4 {
                rest.push(t);
            }
        }
        rest.sort_by_key(|t| (*t == "7p".parse::<Tile>().unwrap(), t.index()));
        // Every bonus tile is already shown, so the wall holds playing tiles only.
        let river_len = rest.len() - wall;
        for (k, &t) in rest.iter().take(river_len).enumerate() {
            river.push(((k + 1) % 4, t, false));
        }
        let spec = PositionSpec {
            hands,
            melds,
            bonus: [
                vec![],
                parse_tiles("F1 F2 F3").unwrap(),
                parse_tiles("F4 S1 S2").unwrap(),
                parse_tiles("S3 S4").unwrap(),
            ],
            river,
            wall_remaining: wall,
            to_move: 0,
            phase: PositionPhase::Turn,
            last_drawn: Some("N".parse().unwrap()),
            kong_chain: 0,
            liable: [None; 4],
        };
        State::from_position(RuleSet::default(), &spec).expect("a valid position")
    }

    #[test]
    fn liability_flag_fires_for_the_fourth_exposed_meld() {
        let s = three_melds_position();
        let seven: Tile = "7p".parse().unwrap();
        let white: Tile = "Wh".parse().unwrap();
        assert_eq!(s.melds(2).exposed_count(), 3);
        assert!(s.unseen(0)[seven] >= 2);
        for b in [belief(), Arc::new(HeuristicBelief)] {
            let d = Danger::from_belief(&s, 0, b.as_ref());
            assert_eq!(d.liability_tile(seven), Some(2), "{b:?}");
            // I hold three Wh: at most one copy is unseen, so no pung (and seat 2
            // is not my next seat, so no chow).
            assert!(s.unseen(0)[white] <= 1);
            assert_eq!(d.liability_tile(white), None);
            assert_eq!(d.liability_chance(2, white), 0.0);
            assert!(d.liability_chance(2, seven) > 0.0);
        }
        // The heuristic danger knows nothing about liability.
        assert_eq!(Danger::estimate(&s, 0).liability_tile(seven), None);
    }
}
