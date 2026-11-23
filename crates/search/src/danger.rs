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
//!
//! **Exact counting** (plan Task 13, [`crate::exact_wait`]): with the switch on
//! ([`with_exact`], off by default), an opponent holding 1 or 4 concealed tiles is
//! read by listing its possible hands instead of the tables. One concealed tile:
//! its per-tile chance is the exact share (1.0 when one kind is left, 0.0 for a
//! kind no possible hand wins on); four: the table's (or belief's) chance that it
//! is ready, spread over the possible ready hands. Its points per win follow the
//! same list. The table opponents still go through the calibration curve; the
//! exact ones do not. With the switch off every figure is as before.
//!
//! **Risk network** (plan Task 15, [`crate::risk_net`]): with a network set by
//! [`with_risk`] (none by default), the player to act reads every opponent's
//! per-tile chance and points from the network instead of the tables or the
//! belief, outside the calibration curve; where exact counting gives exactly 0
//! or 1 for a kind, that figure wins over the network. With no network set
//! every figure is as before.
//!
//! **Reading** (plan Task 20): [`with_reading`] sets a belief model for this
//! thread (none by default); [`Danger::of_reading`] then reads it instead of the
//! fitted tables ([`crate::HeuristicBelief`]). The give-up rule, the learned
//! push-or-fold choice and the defence bot of [`crate::defence`] read their danger
//! through it, so `stage-fold:belief=...` runs Task 11's rule on another reading.

use crate::belief::BeliefModel;
use crate::danger_tables::{CALIBRATION, THREAT, WAIT};
use crate::ev::{Belief, Beliefs, HeuristicBelief, FIRST_FAAN_BUCKET};
use crate::exact_wait::exact_wait;
use crate::passed::Passed;
use crate::risk_net::RiskNet;
use mahjong_hk::{Event, Meld, State, DISCARD_SHARES};
use mahjong_tiles::{Tile, TileCounts, NUM_KINDS};
use std::cell::{Cell, RefCell};
use std::sync::Arc;

/// Points a deal-in costs when [`Danger::estimate`] knows no hand values: the
/// discarder's 2 shares (rules §8.2) of the 3-faan minimum, 8 points a share.
const ESTIMATE_LOSS: f64 = DISCARD_SHARES as f64 * 8.0;

thread_local! {
    static EXACT: Cell<bool> = const { Cell::new(false) };
    static RISK: RefCell<Option<Arc<RiskNet>>> = const { RefCell::new(None) };
    static READING: RefCell<Option<Arc<BeliefModel>>> = const { RefCell::new(None) };
}

/// Belief models given to bots as their reading ([`register_reading`]).
static REGISTERED: std::sync::RwLock<Vec<Arc<BeliefModel>>> = std::sync::RwLock::new(Vec::new());

/// Keeps `model` for the life of the process and returns its id (1 and up), so
/// a `Copy` bot can name it ([`crate::stage::StageBot::reading`]). The same
/// `Arc` registered twice gets the same id.
pub fn register_reading(model: Arc<BeliefModel>) -> u16 {
    let mut list = REGISTERED.write().expect("reading registry");
    if let Some(i) = list.iter().position(|m| Arc::ptr_eq(m, &model)) {
        return (i + 1) as u16;
    }
    list.push(model);
    u16::try_from(list.len()).expect("fewer than 65,536 readings")
}

/// The model registered under `id` ([`register_reading`]); `None` for 0.
pub fn registered_reading(id: u16) -> Option<Arc<BeliefModel>> {
    let i = usize::from(id).checked_sub(1)?;
    REGISTERED.read().expect("reading registry").get(i).cloned()
}

/// The belief model set for this thread by [`with_reading`], if any.
pub fn reading() -> Option<Arc<BeliefModel>> {
    READING.with(|r| r.borrow().clone())
}

/// Runs `f` with `model` as this thread's reading (plan Task 20; `None` is the
/// fitted tables), restoring the previous setting afterwards (also on a panic).
pub fn with_reading<T>(model: Option<Arc<BeliefModel>>, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<Arc<BeliefModel>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            READING.with(|r| *r.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(READING.with(|r| r.replace(model)));
    f()
}

/// The risk network set for this thread by [`with_risk`], if any.
pub fn risk_net() -> Option<Arc<RiskNet>> {
    RISK.with(|r| r.borrow().clone())
}

/// Runs `f` with `net` as this thread's risk network (plan Task 15; `None`
/// switches it off), restoring the previous setting afterwards (also on a
/// panic).
pub fn with_risk<T>(net: Option<Arc<RiskNet>>, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<Arc<RiskNet>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            RISK.with(|r| *r.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(RISK.with(|r| r.replace(net)));
    f()
}

/// Whether exact counting is on for this thread ([`with_exact`]).
pub fn exact_on() -> bool {
    EXACT.with(Cell::get)
}

/// Runs `f` with exact counting switched `on` for this thread (plan Task 13),
/// restoring the previous setting afterwards (also on a panic).
pub fn with_exact<T>(on: bool, f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            EXACT.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(EXACT.with(|c| c.replace(on)));
    f()
}

/// [`Danger`]'s figures for one tile split into the table opponents and the
/// exactly counted ones ([`Danger::split`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Split {
    /// `1 - prod (1 - raw_o)` over the table opponents, before any calibration.
    pub table_tile: f64,
    /// `1 - prod (1 - p_o)` over the exactly counted opponents.
    pub exact_tile: f64,
    /// Expected loss from the table opponents (as [`Danger::expected_loss`]).
    pub table_loss: f64,
    /// Expected loss from the exactly counted opponents.
    pub exact_loss: f64,
}

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
    tag_of(state.rules(), o, state.melds(o))
}

fn tag_of(rules: &mahjong_hk::RuleSet, o: usize, melds: &[Meld]) -> Tag {
    let exposed = melds.iter().filter(|m| !m.concealed).count().min(4);
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
            if let Event::Discard { seat, tile, .. } = *e {
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

    /// The same lookups from public facts gathered elsewhere (`passed::risk_scores`
    /// rebuilds each seat's view at an earlier discard): the seat's melds so far,
    /// the wall-progress bucket (0–5) and the kinds it has discarded.
    pub fn from_public(
        rules: &mahjong_hk::RuleSet,
        o: usize,
        melds: &[Meld],
        progress: usize,
        discarded: [bool; NUM_KINDS],
    ) -> Opponent {
        Opponent {
            tag: tag_of(rules, o, melds),
            progress: progress.min(PROGRESS_BUCKETS - 1),
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
pub(crate) fn calibrate(raw: f64) -> f64 {
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
    /// Seats with three exposed melds and nobody liable for them yet.
    three_exposed: [bool; 4],
    /// `P_claim4_o(t)`: o can claim `t` as its fourth exposed meld (a pung from
    /// any seat, a chow if o is my next seat), from the belief's counts.
    claim4: [[f64; NUM_KINDS]; 4],
    /// Whether the unseen copies leave that claim possible at all.
    can_claim4: [[bool; NUM_KINDS]; 4],
    /// Points per share of `o`'s win on each tile, for exactly counted
    /// opponents (plan Task 13); `pay` is used for the others.
    tile_pay: [[f64; NUM_KINDS]; 4],
}

/// Estimated danger of each tile for one seat.
#[derive(Clone, Debug)]
pub struct Danger {
    tiles: [f64; NUM_KINDS],
    threat: [f64; 4],
    /// Per opponent, P(it wins on each tile) before combining: the table's
    /// `threat × rate` for [`Danger::estimate`], the belief's `tenpai × wait` for
    /// [`Danger::from_belief`]. [`Danger::apply_passed`] rescales it.
    raw: [[f64; NUM_KINDS]; 4],
    /// Whether `tiles` goes through the calibration curve (the tables do).
    calibrated: bool,
    /// Present only when built by [`Danger::from_belief`].
    payment: Option<Box<Payment>>,
    /// Opponents read by exact counting (plan Task 13, switch on only).
    exact: [bool; 4],
    /// Opponents read by the risk network (plan Task 15, network set only).
    risk: [bool; 4],
}

impl Danger {
    /// Estimates danger for `me` from public information.
    pub fn estimate(state: &State, me: usize) -> Danger {
        let unseen = state.unseen(me);
        let mut raw = [[0.0f64; NUM_KINDS]; 4];
        let mut threat = [0.0f64; 4];
        for o in (0..4).filter(|&o| o != me) {
            let opp = Opponent::of(state, o);
            let t_o = THREAT[opp.threat_cell()];
            threat[o] = t_o;
            for t in Tile::all_kinds() {
                if let Some(c) = opp.wait_cell(t, &unseen) {
                    raw[o][t.index() as usize] = t_o * WAIT[c];
                }
            }
        }
        let mut d = Danger {
            tiles: [0.0; NUM_KINDS],
            threat,
            raw,
            calibrated: true,
            payment: None,
            exact: [false; 4],
            risk: [false; 4],
        };
        if let Some(net) = risk_net() {
            d.apply_risk(state, me, &net);
        }
        if exact_on() {
            d.apply_exact(state, me);
        }
        d.combine();
        d
    }

    /// Replaces every opponent's figures by the risk network's (plan Task 15)
    /// when `me` is the player to act; the figures leave the calibration curve.
    /// Where exact counting ([`crate::exact_wait`]) gives exactly 0 or 1 for a
    /// kind, that figure wins over the network; the caller recombines.
    fn apply_risk(&mut self, state: &State, me: usize, net: &RiskNet) {
        // The network's `features` input holds the table danger plane, which is
        // built by `Danger::estimate`: run it with no network set, so the plane
        // stays the table's (as in the training data) and nothing recurses.
        let Some(out) = with_risk(None, || net.estimate(state, me)) else {
            return;
        };
        self.calibrated = false;
        for o in (0..4).filter(|&o| o != me) {
            self.risk[o] = true;
            let exact = exact_wait(state, me, o);
            let ready = exact.as_ref().and_then(|e| {
                if !e.any_ready() {
                    Some(0.0)
                } else if e.concealed == 1 {
                    Some(1.0)
                } else {
                    None
                }
            });
            for i in 0..NUM_KINDS {
                let mut p = out.p[o][i];
                let mut pts = out.points[o][i];
                if let Some(e) = &exact {
                    if e.win[i] == 0.0 {
                        p = 0.0;
                    } else if ready == Some(1.0) && e.win[i] == 1.0 {
                        p = 1.0;
                        pts = e.points[i];
                    }
                }
                self.raw[o][i] = p;
                if let Some(pay) = self.payment.as_mut() {
                    pay.p[o][i] = p;
                    pay.tile_pay[o][i] = pts;
                }
            }
            self.threat[o] = self.raw[o]
                .iter()
                .copied()
                .fold(0.0, f64::max)
                .max(if ready == Some(1.0) { 1.0 } else { 0.0 });
            if let Some(pay) = self.payment.as_mut() {
                let w: f64 = self.raw[o].iter().sum();
                if w > 0.0 {
                    pay.pay[o] = (0..NUM_KINDS)
                        .map(|i| self.raw[o][i] * pay.tile_pay[o][i])
                        .sum::<f64>()
                        / w;
                }
            }
        }
    }

    /// Whether opponent `seat` is read by the risk network (plan Task 15).
    pub fn is_risk(&self, seat: usize) -> bool {
        self.risk[seat]
    }

    /// Replaces the figures of each opponent holding 1 or 4 concealed tiles by
    /// exact counting ([`crate::exact_wait`]); the caller recombines.
    fn apply_exact(&mut self, state: &State, me: usize) {
        for o in (0..4).filter(|&o| o != me) {
            let Some(e) = exact_wait(state, me, o) else {
                continue;
            };
            self.exact[o] = true;
            let ready = if !e.any_ready() {
                0.0
            } else if e.concealed == 1 {
                1.0
            } else {
                self.threat[o]
            };
            self.threat[o] = ready;
            for i in 0..NUM_KINDS {
                self.raw[o][i] = ready * e.win[i];
            }
            if let Some(p) = self.payment.as_mut() {
                let mut w = 0.0;
                let mut paid = 0.0;
                for i in 0..NUM_KINDS {
                    p.p[o][i] = self.raw[o][i];
                    p.tile_pay[o][i] = e.points[i];
                    w += e.win[i];
                    paid += e.win[i] * e.points[i];
                }
                if w > 0.0 {
                    p.pay[o] = paid / w;
                }
            }
        }
    }

    /// Whether opponent `seat` is read by exact counting (plan Task 13).
    pub fn is_exact(&self, seat: usize) -> bool {
        self.exact[seat]
    }

    /// The figures of tile `t` split into table and exact opponents; `None` when
    /// no opponent is read exactly (the switch off, or nobody with 1 or 4
    /// concealed tiles).
    pub fn split(&self, t: Tile) -> Option<Split> {
        if !self.exact.iter().any(|&e| e) {
            return None;
        }
        let i = t.index() as usize;
        let mut out = Split {
            table_tile: 0.0,
            exact_tile: 0.0,
            table_loss: 0.0,
            exact_loss: 0.0,
        };
        let (mut safe_table, mut safe_exact) = (1.0, 1.0);
        for o in 0..4 {
            let loss = self.loss_from(o, i);
            if self.exact[o] {
                safe_exact *= 1.0 - self.raw[o][i];
                out.exact_loss += loss;
            } else {
                safe_table *= 1.0 - self.raw[o][i];
                out.table_loss += loss;
            }
        }
        out.table_tile = 1.0 - safe_table;
        out.exact_tile = 1.0 - safe_exact;
        if self.payment.is_none() {
            // Built by `estimate`: 2 shares of 8 points per deal-in (§8.2), on the
            // calibrated table part.
            out.table_loss = if self.calibrated {
                calibrate(out.table_tile)
            } else {
                out.table_tile
            } * ESTIMATE_LOSS;
            out.exact_loss = out.exact_tile * ESTIMATE_LOSS;
        }
        Some(out)
    }

    /// Expected points lost to seat `o` alone on tile index `i` (payment only).
    fn loss_from(&self, o: usize, i: usize) -> f64 {
        match &self.payment {
            Some(p) => {
                let pay = if self.exact[o] || self.risk[o] {
                    p.tile_pay[o][i]
                } else {
                    p.pay[o]
                };
                // A discard win costs me the discarder's 2 shares whoever is liable
                // (§8.2; §8.3: liability covers self-draws only).
                p.p[o][i] * pay * DISCARD_SHARES as f64
            }
            None => 0.0,
        }
    }

    /// `tiles` from `raw`: `1 − Π_o (1 − raw_o)`, through the calibration curve for
    /// the table estimate.
    ///
    /// Exactly counted opponents (plan Task 13) stay outside the curve:
    /// `1 - (1 - calibrate(table part)) * prod_exact (1 - raw_o)`.
    fn combine(&mut self) {
        for i in 0..NUM_KINDS {
            let safe: f64 = (0..4)
                .filter(|&o| !self.exact[o])
                .map(|o| 1.0 - self.raw[o][i])
                .product();
            let table = if self.calibrated {
                calibrate(1.0 - safe)
            } else {
                1.0 - safe
            };
            let exact: f64 = (0..4)
                .filter(|&o| self.exact[o])
                .map(|o| 1.0 - self.raw[o][i])
                .product();
            self.tiles[i] = 1.0 - (1.0 - table) * exact;
        }
    }

    /// Passed-tile safety (plan Task 14, `docs/race-bot.md` §13): scales each
    /// opponent's rate on each tile by [`Passed::factor`] (0 while that seat's
    /// passed-win lock holds on the tile, then rising with its 入章 since), and
    /// rebuilds [`Danger::tile`] and [`Danger::expected_loss`] from the result.
    ///
    /// Exactly counted opponents are left alone: counting already applies the
    /// passed-win lock (rules §4.4).
    pub fn apply_passed(&mut self, passed: &Passed) {
        for o in (0..4).filter(|&o| !self.exact[o]) {
            for t in Tile::all_kinds() {
                let i = t.index() as usize;
                if self.raw[o][i] > 0.0 {
                    let f = passed.factor(o, t);
                    self.raw[o][i] *= f;
                    if let Some(p) = self.payment.as_mut() {
                        p.p[o][i] *= f;
                    }
                }
            }
        }
        self.combine();
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

    /// [`Danger::from_belief`] with this thread's reading ([`with_reading`]), or
    /// the fitted tables ([`HeuristicBelief`]) when none is set.
    pub fn of_reading(state: &State, me: usize) -> Danger {
        match reading() {
            Some(m) => Self::from_belief(state, me, &*m),
            None => Self::from_belief(state, me, &HeuristicBelief),
        }
    }

    /// [`Danger::from_belief`] from beliefs already estimated.
    pub fn from_beliefs(state: &State, me: usize, beliefs: &Beliefs) -> Danger {
        let rules = state.rules();
        let unseen = state.unseen(me);
        let points = |f: u8| rules.points(f.min(rules.faan_cap)) as f64;
        let mut pay = Payment {
            p: [[0.0; NUM_KINDS]; 4],
            pay: [0.0; 4],
            three_exposed: [false; 4],
            claim4: [[0.0; NUM_KINDS]; 4],
            can_claim4: [[false; NUM_KINDS]; 4],
            tile_pay: [[0.0; NUM_KINDS]; 4],
        };
        let mut raw = [[0.0f64; NUM_KINDS]; 4];
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
                raw[o][i] = p;
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
        let mut d = Danger {
            tiles: [0.0; NUM_KINDS],
            threat,
            raw,
            calibrated: false,
            payment: Some(Box::new(pay)),
            exact: [false; 4],
            risk: [false; 4],
        };
        if let Some(net) = risk_net() {
            d.apply_risk(state, me, &net);
        }
        if exact_on() {
            d.apply_exact(state, me);
        }
        d.combine();
        d
    }

    /// Expected points I lose if I discard `t` now: `Σ_o P_o(t) × Pay(o) × 2`,
    /// the discarder's 2 shares (rules §8.2; liability covers only self-draws,
    /// §8.3; every winner is paid under the default `multiple_winners` rule).
    /// Built by [`Danger::estimate`], which knows no hand values, it is
    /// `tile(t) × 16` (2 shares of the 3-faan minimum). Until 2026-11-23 the
    /// discarder paid 1 share and these figures were half.
    pub fn expected_loss(&self, t: Tile) -> f64 {
        let i = t.index() as usize;
        match &self.payment {
            Some(_) => (0..4).map(|o| self.loss_from(o, i)).sum(),
            None => self.tiles[i] * ESTIMATE_LOSS,
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
            passed_wins: Default::default(),
            river_from_hand: Vec::new(),
        };
        State::from_position(RuleSet::default(), &spec).expect("a valid position")
    }

    /// Plan Task 15: with a risk network set, every figure follows the network
    /// (recombined as the module says) except where exact counting proves 0 or
    /// 1; with none set the figures are the tables', bit for bit.
    #[test]
    fn risk_network_feeds_the_figures_and_exact_counting_overrides_it() {
        use crate::exact_wait::exact_wait;
        use crate::risk_net::test_network;
        let net = Arc::new(test_network());
        let bot = Bot::by_name("stage").unwrap();
        let mut rng = SmallRng::seed_from_u64(15);
        let (mut decisions, mut overridden, mut exact_seen) = (0, 0, 0);
        while decisions < 400 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(me) = s.current_player() {
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                if let mahjong_hk::Action::Discard(_) = a {
                    decisions += 1;
                    let plain = Danger::from_belief(&s, me, &HeuristicBelief);
                    let again = with_risk(None, || Danger::from_belief(&s, me, &HeuristicBelief));
                    let on = with_risk(Some(Arc::clone(&net)), || {
                        Danger::from_belief(&s, me, &HeuristicBelief)
                    });
                    let est = with_risk(Some(Arc::clone(&net)), || Danger::estimate(&s, me));
                    let out = net.estimate(&s, me).unwrap();
                    let unseen = s.unseen(me);
                    for t in Tile::all_kinds() {
                        let i = t.index() as usize;
                        assert_eq!(plain.tile(t), again.tile(t));
                        assert_eq!(plain.expected_loss(t), again.expected_loss(t));
                        let mut safe = 1.0;
                        let mut loss = 0.0;
                        for o in (0..4).filter(|&o| o != me) {
                            assert!(on.is_risk(o) && !plain.is_risk(o));
                            let mut p = out.p[o][i];
                            let mut pts = out.points[o][i];
                            if let Some(e) = exact_wait(&s, me, o) {
                                exact_seen += 1;
                                if e.win[i] == 0.0 {
                                    p = 0.0;
                                    overridden += 1;
                                } else if e.concealed == 1 && e.any_ready() && e.win[i] == 1.0 {
                                    p = 1.0;
                                    pts = e.points[i];
                                    overridden += 1;
                                }
                            }
                            safe *= 1.0 - p;
                            // 2 shares on a discard win, liable or not (§8.2, §8.3).
                            loss += p * pts * 2.0;
                        }
                        let want = 1.0 - safe;
                        assert!(
                            (on.tile(t) - want).abs() < 1e-12,
                            "{t}: {} vs {want}",
                            on.tile(t)
                        );
                        assert!((on.expected_loss(t) - loss).abs() < 1e-9, "{t}");
                        // `estimate` leaves the calibration curve: the same chance, 2 shares of 8 points each.
                        assert!((est.tile(t) - want).abs() < 1e-12);
                        assert!((est.expected_loss(t) - want * 16.0).abs() < 1e-9);
                        if unseen[t] == 0 {
                            assert_eq!(on.tile(t), 0.0);
                        }
                    }
                }
                s.apply(a);
            }
        }
        assert!(
            exact_seen > 0 && overridden > 0,
            "{exact_seen} {overridden}"
        );
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
