//! The `ev` bot: Brian's explicit expected-value design (decision B15), built
//! exactly as `docs/ev-bot.md` specifies (plan Tasks 8 and 10).
//!
//! Every candidate action is valued in points for the acting seat:
//!
//! ```text
//! EV(a) = Offence(a) − Defence(a)
//! Offence(a) = cont(a) × κ × P_win × Pay(me) × Shares(me)
//! Defence(a) = DealInNow(a) + cont(a) × [SelfDraw + OtherFeeds + Later(a) + Liab(a)]
//! ```
//!
//! `P_win`, `P_sd` and `pts_T` come from the target value table
//! ([`crate::target_value`]); everything about the opponents comes from a
//! [`Belief`]. The trained belief model (Task 5) plugs in through that trait;
//! until it exists, [`HeuristicBelief`] reads the fitted danger tables
//! (`docs/ev-bot.md` §8). Every number the bot uses is kept in [`EvTerms`] so hints
//! can show it; the chosen action's reason keys are in [`EvDecision::reasons`].
//!
//! The search is plain arithmetic per candidate: no threads, no randomness.

use crate::bots::{after_claim, rank_discards, Metric};
use crate::danger::Opponent;
use crate::danger_tables::{THREAT, WAIT};
use crate::target_value::{lookup, tiles_left_bucket, TargetKey, TargetKind};
use crate::view::SeatView;
use mahjong_analysis::{faan_shanten, HandView, UNREACHABLE};
use mahjong_games::Game;
use mahjong_hk::{Action, Meld, MeldKind, RuleSet, State};
use mahjong_tiles::{Tile, NUM_KINDS};
use std::fmt;
use std::sync::Arc;

/// Faan buckets of the belief's hand-value output: 3, 4, …, 10.
pub const FAAN_BUCKETS: usize = 8;
/// The faan of bucket 0.
pub const FIRST_FAAN_BUCKET: u8 = 3;

// ----- Beliefs about the opponents. -----

/// What the acting seat believes about one opponent, from public information
/// (`docs/ev-bot.md` §1, "From the belief model").
#[derive(Clone, Debug, PartialEq)]
pub struct SeatBelief {
    /// `tenpai_o`: P(the seat is waiting with a legal win now).
    pub tenpai: f64,
    /// `wait_o(t)`: P(`t` is a winning tile for the seat | tenpai), per tile.
    pub wait: [f64; NUM_KINDS],
    /// `P_b(o)`: P(a discard win now pays faan `3 + b`), `b = 0..8`.
    pub faan: [f64; FAAN_BUCKETS],
    /// `sd_o`: P(the seat self-draws a win before the hand ends).
    pub self_draw: f64,
    /// `P1_o(t)`: P(the seat holds at least one copy of `t`).
    pub has_one: [f64; NUM_KINDS],
    /// `P2_o(t)`: P(the seat holds at least two copies of `t`).
    pub has_two: [f64; NUM_KINDS],
    /// `E[cnt_o(t)]`: expected copies of `t` in the seat's hand.
    pub mean_count: [f64; NUM_KINDS],
}

impl SeatBelief {
    /// A seat that is no threat: not waiting, no copies, a 3-faan hand.
    pub fn quiet() -> SeatBelief {
        let mut faan = [0.0; FAAN_BUCKETS];
        faan[0] = 1.0;
        SeatBelief {
            tenpai: 0.0,
            wait: [0.0; NUM_KINDS],
            faan,
            self_draw: 0.0,
            has_one: [0.0; NUM_KINDS],
            has_two: [0.0; NUM_KINDS],
            mean_count: [0.0; NUM_KINDS],
        }
    }
}

/// Beliefs about every seat; the acting seat's own entry is ignored.
#[derive(Clone, Debug, PartialEq)]
pub struct Beliefs {
    pub seats: [SeatBelief; 4],
}

/// The belief model's interface: run once per decision, from what `me` can see.
/// The trained model of plan Task 5 implements it; tests stub it.
pub trait Belief: Send + Sync + fmt::Debug {
    /// Beliefs about each opponent of `me` in `state`. Must read only what `me`
    /// can see (their own hand and the public table).
    fn estimate(&self, state: &State, me: usize) -> Beliefs;
}

/// The fallback without a trained model (`docs/ev-bot.md` §8): the fitted danger
/// tables for tenpai and waits, hand value from the exposed melds.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeuristicBelief;

/// Faan read from `o`'s melds (§8): 3, +1 per scoring pung, +3 for a flush tag,
/// +3 for at least two exposed melds that are all pungs; capped at 10.
fn heuristic_faan(state: &State, o: usize) -> u8 {
    let rules = state.rules();
    let melds = state.melds(o);
    let exposed: Vec<&Meld> = melds.iter().filter(|m| !m.concealed).collect();
    let scoring = melds
        .iter()
        .filter(|m| {
            m.is_pung_like()
                && (m.tile.is_dragon()
                    || m.tile == Tile::wind(o)
                    || m.tile == Tile::wind(rules.prevailing_wind))
        })
        .count() as u8;
    // The flush tag of `danger.rs`: some exposed meld, and every suited meld in one suit.
    let mut suits: Vec<_> = melds.iter().filter_map(|m| m.tile.suit()).collect();
    suits.sort();
    suits.dedup();
    let flush = !exposed.is_empty() && suits.len() == 1;
    let all_pungs = exposed.len() >= 2 && exposed.iter().all(|m| m.is_pung_like());
    let f = 3 + scoring + if flush { 3 } else { 0 } + if all_pungs { 3 } else { 0 };
    f.min(10).min(rules.faan_cap.max(FIRST_FAAN_BUCKET))
}

impl Belief for HeuristicBelief {
    fn estimate(&self, state: &State, me: usize) -> Beliefs {
        let unseen = state.unseen(me);
        let pool = unseen.total().max(1) as f64;
        let mut seats: [SeatBelief; 4] = std::array::from_fn(|_| SeatBelief::quiet());
        for (o, b) in seats.iter_mut().enumerate().filter(|(o, _)| *o != me) {
            let opp = Opponent::of(state, o);
            b.tenpai = THREAT[opp.threat_cell()];
            for t in Tile::all_kinds() {
                let i = t.index() as usize;
                b.wait[i] = opp.wait_cell(t, &unseen).map_or(0.0, |c| WAIT[c]);
                // §8 fixes P2; P1 and E[cnt] (not named in §8) treat every unseen
                // copy as equally likely to sit in any hidden slot.
                let q = state.concealed_count(o) as f64 / pool;
                let left = unseen[t] as i32;
                b.has_one[i] = 1.0 - (1.0 - q).powi(left);
                b.has_two[i] = if left >= 2 { 0.1 } else { 0.0 };
                b.mean_count[i] = q * left as f64;
            }
            b.faan = [0.0; FAAN_BUCKETS];
            b.faan[(heuristic_faan(state, o) - FIRST_FAAN_BUCKET) as usize] = 1.0;
            b.self_draw = 0.25 * b.tenpai;
        }
        Beliefs { seats }
    }
}

/// Loads a trained belief model for `ev:belief=<path>`.
///
/// The MJNN v2 belief format and its loader arrive with plan Tasks 5 and 6; until
/// then this returns an error (never panics) so a spec naming a model fails
/// clearly instead of silently using the heuristic.
pub fn load_belief(path: &str) -> Result<Arc<dyn Belief>, String> {
    if !std::path::Path::new(path).exists() {
        return Err(format!("ev: belief model {path:?} not found"));
    }
    // TODO(plan Task 5/6): read the MJNN v2 belief model here.
    Err(format!(
        "ev: loading belief model {path:?} is not built yet (todo: plan Task 5/6 adds the \
         MJNN v2 loader); omit belief= to use the heuristic fallback"
    ))
}

// ----- Hyper-parameters. -----

/// The `ev:` spec keys and their defaults (`docs/ev-bot.md` §4).
#[derive(Clone, Debug, PartialEq)]
pub struct EvConfig {
    /// `H`: my future turns valued.
    pub horizon: usize,
    /// `P_clip` for `P_o(t)` and `win_o`.
    pub clip: f64,
    /// Points below which two candidates tie (§2.2).
    pub tie_margin: f64,
    /// `x_o = r_disc × sd_o`.
    pub r_disc: f64,
    /// Win chance of a seat right after it claims its fourth exposed meld.
    pub four_meld_win: f64,
    /// Lower bound on the feed ratio `φ`.
    pub feed_lo: f64,
    /// Upper bound on the feed ratio `φ`.
    pub feed_hi: f64,
    /// `reason.ev.deal_in_now` is emitted when some `P_o(t(a*)) ≥ warn_p`.
    pub warn_p: f64,
    /// `Pay(me) = max(points(f_T), pts_T)` (§9 Q4, the default) when true;
    /// `Pay(me) = pts_T` when false (spec `floor=none`). Not in the doc's §4:
    /// added because the floor decides test positions 1 and 2 (see §10).
    pub pay_floor: bool,
}

impl Default for EvConfig {
    fn default() -> Self {
        EvConfig {
            horizon: 8,
            clip: 0.95,
            tie_margin: 0.5,
            r_disc: 2.0,
            four_meld_win: 0.5,
            feed_lo: 0.5,
            feed_hi: 1.5,
            warn_p: 0.10,
            pay_floor: true,
        }
    }
}

impl EvConfig {
    /// Sets one `k=v` spec option. `belief` is handled by the caller.
    pub fn set(&mut self, k: &str, v: &str) -> Result<(), String> {
        let num = || {
            v.parse::<f64>()
                .ok()
                .filter(|x| x.is_finite() && *x >= 0.0)
                .ok_or_else(|| format!("bad number {v:?} for ev option {k}"))
        };
        match k {
            "horizon" => {
                self.horizon = v
                    .parse::<usize>()
                    .ok()
                    .filter(|&h| h >= 1)
                    .ok_or_else(|| format!("bad horizon {v:?}"))?
            }
            "clip" => self.clip = num()?.min(1.0),
            "tie_margin" => self.tie_margin = num()?,
            "r_disc" => self.r_disc = num()?,
            "four_meld_win" => self.four_meld_win = num()?.min(1.0),
            "warn_p" => self.warn_p = num()?,
            "feed_clip" => {
                let (lo, hi) = v
                    .split_once('-')
                    .and_then(|(a, b)| Some((a.parse::<f64>().ok()?, b.parse::<f64>().ok()?)))
                    .filter(|(a, b)| 0.0 < *a && a <= b)
                    .ok_or_else(|| format!("feed_clip wants lo-hi, e.g. 0.5-1.5, not {v:?}"))?;
                self.feed_lo = lo;
                self.feed_hi = hi;
            }
            "floor" => {
                self.pay_floor = match v {
                    "target" => true,
                    "none" => false,
                    _ => return Err(format!("floor is target or none, not {v:?}")),
                }
            }
            "fold" if v == "safest" => {}
            "fold" => return Err(format!("unknown fold plan {v:?} (only \"safest\")")),
            _ => return Err(format!("unknown ev option {k:?}")),
        }
        Ok(())
    }
}

// ----- Output. -----

/// Every number the bot used for one candidate (`docs/ev-bot.md` §2.3).
/// Arrays are per seat; the acting seat's entries are 0.
#[derive(Clone, Debug, PartialEq)]
pub struct EvTerms {
    pub action: Action,
    /// `t(a)`: the tile discarded now (for a claim, the best discard after it);
    /// `None` for a pass or a kong.
    pub discard: Option<Tile>,
    /// `EV(a) = offence − defence`, points for the acting seat.
    pub ev: f64,
    pub offence: f64,
    pub defence: f64,
    /// From the target table for `h(a)` (0 when no target is reachable).
    pub p_win: f64,
    pub p_sd: f64,
    /// `Pay(me) = max(points(f_T), pts_T)`.
    pub pay_me: f64,
    pub shares_me: f64,
    /// The feed ratio `φ` (1 unless `h(a)` is waiting).
    pub phi: f64,
    /// Faan-aware shanten of `h(a)` (`UNREACHABLE` when no target).
    pub faan_shanten: i8,
    /// `f_T`, the target's guaranteed faan (0 when no target).
    pub target_faan: u8,
    /// The horizon factor `κ`.
    pub kappa: f64,
    /// `Π_o (1 − P_o(t(a)))`.
    pub cont: f64,
    /// `P_o(t(a))` per seat.
    pub p_now: [f64; 4],
    pub deal_in_now: [f64; 4],
    pub self_draw: [f64; 4],
    pub other_feeds: [f64; 4],
    pub later: f64,
    pub liability: f64,
    /// `Liab(a)` split by the seat with three exposed melds.
    pub liability_by_seat: [f64; 4],
    /// The fold plan's first `min(H, |h(a)|)` tiles, safest first.
    pub fold_plan: Vec<Tile>,
    /// Belief inputs used.
    pub tenpai: [f64; 4],
    pub sd: [f64; 4],
    pub pay: [f64; 4],
    pub pay_sd: [f64; 4],
}

impl EvTerms {
    /// `DealInNow(a)` summed over seats.
    pub fn deal_in_total(&self) -> f64 {
        self.deal_in_now.iter().sum()
    }
}

/// A reason key with its arguments (`docs/ev-bot.md` §7). Engines return keys,
/// never text (D42).
#[derive(Clone, Debug, PartialEq)]
pub struct EvReason {
    pub key: &'static str,
    /// Named numbers; `seat` and `shares` are whole numbers.
    pub args: Vec<(&'static str, f64)>,
}

/// The bot's decision with everything behind it.
#[derive(Clone, Debug, PartialEq)]
pub struct EvDecision {
    pub action: Action,
    /// Every candidate valued, best `ev` first (empty when a win was declared).
    pub candidates: Vec<(Action, EvTerms)>,
    /// Reason keys for `action` (empty for a win).
    pub reasons: Vec<EvReason>,
}

// ----- The bot. -----

/// The `ev` bot (see the module docs).
#[derive(Clone, Debug)]
pub struct EvBot {
    pub config: EvConfig,
    pub belief: Arc<dyn Belief>,
}

impl Default for EvBot {
    fn default() -> Self {
        EvBot::new(EvConfig::default(), Arc::new(HeuristicBelief))
    }
}

/// Everything that is the same for every candidate of one decision.
struct Table {
    me: usize,
    rules: RuleSet,
    wall: usize,
    /// `n`, draws left per seat.
    n: f64,
    /// `H`.
    h: usize,
    kappa: f64,
    /// `P_o(t)`, 0 for `me`.
    p: [[f64; NUM_KINDS]; 4],
    /// `Σ_o P_o(t)`: the fold plan's order.
    danger: [f64; NUM_KINDS],
    /// `Σ_o P_o(t) × Pay(o) × L_disc(o)`: one fold-plan turn's expected loss.
    cost: [f64; NUM_KINDS],
    tenpai: [f64; 4],
    sd: [f64; 4],
    x: [f64; 4],
    pay: [f64; 4],
    pay_sd: [f64; 4],
    l_disc: [f64; 4],
    l_sd: [f64; 4],
    self_draw: [f64; 4],
    other_feeds: [f64; 4],
    /// `Σ_o win_o`.
    sum_win: f64,
    /// Seats with three exposed melds and nobody liable (the 12-tile lookahead).
    liab_seat: [bool; 4],
    has_one: [[f64; NUM_KINDS]; 4],
    has_two: [[f64; NUM_KINDS]; 4],
    mean_count: [[f64; NUM_KINDS]; 4],
    liable_me: bool,
    feed_avg: f64,
}

impl Table {
    fn new(state: &State, me: usize, cfg: &EvConfig, beliefs: &Beliefs) -> Table {
        let rules = *state.rules();
        let wall = state.wall_remaining();
        let n = wall.div_ceil(4).max(1);
        let h = n.min(cfg.horizon);
        let kappa = (h as f64 / n as f64).min(1.0);
        let points = |f: u8| rules.points(f.min(rules.faan_cap)) as f64;
        let mut t = Table {
            me,
            rules,
            wall,
            n: n as f64,
            h,
            kappa,
            p: [[0.0; NUM_KINDS]; 4],
            danger: [0.0; NUM_KINDS],
            cost: [0.0; NUM_KINDS],
            tenpai: [0.0; 4],
            sd: [0.0; 4],
            x: [0.0; 4],
            pay: [0.0; 4],
            pay_sd: [0.0; 4],
            l_disc: [0.0; 4],
            l_sd: [0.0; 4],
            self_draw: [0.0; 4],
            other_feeds: [0.0; 4],
            sum_win: 0.0,
            liab_seat: [false; 4],
            has_one: [[0.0; NUM_KINDS]; 4],
            has_two: [[0.0; NUM_KINDS]; 4],
            mean_count: [[0.0; NUM_KINDS]; 4],
            liable_me: state.liable_for(me).is_some(),
            feed_avg: 0.0,
        };
        let mut hidden = 0usize;
        for o in (0..4).filter(|&o| o != me) {
            let b = &beliefs.seats[o];
            hidden += state.concealed_count(o);
            t.tenpai[o] = b.tenpai;
            t.sd[o] = b.self_draw;
            t.x[o] = cfg.r_disc * b.self_draw;
            t.sum_win += (b.self_draw + t.x[o]).min(cfg.clip);
            for (bucket, &pb) in b.faan.iter().enumerate() {
                let f = FIRST_FAAN_BUCKET + bucket as u8;
                t.pay[o] += pb * points(f);
                t.pay_sd[o] += pb * points((f + 1).min(10));
            }
            // Share multipliers (§1): who pays when o wins.
            let (l_disc, l_sd, l_other) = match state.liable_for(o) {
                None => (1.0, 1.0, 0.0),
                Some(l) if l == me => (3.0, 3.0, 3.0),
                Some(_) => (0.0, 0.0, 0.0),
            };
            t.l_disc[o] = l_disc;
            t.l_sd[o] = l_sd;
            t.self_draw[o] = kappa * t.sd[o] * t.pay_sd[o] * l_sd;
            t.other_feeds[o] = kappa * t.x[o] * t.pay[o] * l_other;
            t.liab_seat[o] = rules.twelve_tile_liability
                && state.melds(o).exposed_count() == 3
                && state.liable_for(o).is_none();
            for i in 0..NUM_KINDS {
                let p = (b.tenpai * b.wait[i]).min(cfg.clip);
                t.p[o][i] = p;
                t.danger[i] += p;
                t.cost[i] += p * t.pay[o] * l_disc;
            }
            t.has_one[o] = b.has_one;
            t.has_two[o] = b.has_two;
            t.mean_count[o] = b.mean_count;
        }
        t.feed_avg = hidden as f64 / (hidden + wall).max(1) as f64;
        t
    }

    fn opponents(&self) -> impl Iterator<Item = usize> + '_ {
        (0..4).filter(move |&o| o != self.me)
    }

    /// `P_claim4_o(t)`: o can claim `t` as its fourth exposed meld.
    fn p_claim4(&self, o: usize, t: Tile) -> f64 {
        let i = t.index() as usize;
        let mut chow = 0.0f64;
        if o == (self.me + 1) % 4 {
            for (a, b) in [(-2i8, -1i8), (-1, 1), (1, 2)] {
                if let (Some(u), Some(v)) = (t.offset(a), t.offset(b)) {
                    chow = chow.max(
                        self.has_one[o][u.index() as usize] * self.has_one[o][v.index() as usize],
                    );
                }
            }
        }
        (self.has_two[o][i] + chow).min(1.0)
    }

    /// The feed ratio `φ` for a waiting hand (A3).
    fn phi(&self, view: &SeatView, cfg: &EvConfig) -> f64 {
        let (mut feed, mut seen) = (0.0, 0.0);
        let hv = view.hand_view();
        for t in Tile::all_kinds() {
            let left = view.unseen[t];
            if left == 0 || view.concealed[t] >= 4 {
                continue;
            }
            let mut h = view.concealed;
            h.add(t);
            let fs = faan_shanten(
                &HandView {
                    concealed: &h,
                    ..hv
                },
                &self.rules,
            );
            if fs.shanten < 0 {
                let i = t.index() as usize;
                feed += self.opponents().map(|o| self.mean_count[o][i]).sum::<f64>();
                seen += left as f64;
            }
        }
        if seen == 0.0 || self.feed_avg <= 0.0 {
            // No live copy of any wait: no discard win to expect (§9 is silent).
            return cfg.feed_lo;
        }
        (feed / seen / self.feed_avg).clamp(cfg.feed_lo, cfg.feed_hi)
    }

    /// Values one candidate: `after` is my hand after the action (13 tiles less 3
    /// per meld), `tile` the tile discarded now, `four_exposed` whether the
    /// action gives me four exposed melds (the discarder becomes liable for me).
    fn value(
        &self,
        cfg: &EvConfig,
        action: Action,
        after: &SeatView,
        tile: Option<Tile>,
        four_exposed: bool,
    ) -> EvTerms {
        let rules = &self.rules;
        // Offence: the target table.
        let fs = faan_shanten(&after.hand_view(), rules);
        let reachable = fs.shanten < UNREACHABLE;
        let key = TargetKey {
            kind: fs.target.as_ref().map_or(TargetKind::Mixed, TargetKind::of),
            faan_shanten: fs.shanten.clamp(0, 4) as usize,
            tiles_left: tiles_left_bucket(self.wall, rules),
            melds: after.melds.len().min(4),
        };
        let tv = lookup(key);
        let target_faan = fs.target.as_ref().map_or(0, |t| t.faan);
        let p_win = if reachable { tv.p_win } else { 0.0 };
        let p_sd = tv.p_self_draw;
        let pay_me = if cfg.pay_floor {
            (rules.points(target_faan.min(rules.faan_cap)) as f64).max(tv.points)
        } else {
            tv.points
        };
        let phi = if fs.shanten == 0 {
            self.phi(after, cfg)
        } else {
            1.0
        };
        let shares_me = if self.liable_me || four_exposed {
            3.0
        } else {
            3.0 * p_sd + (1.0 - p_sd) * phi
        };

        // The tile discarded now.
        let mut p_now = [0.0; 4];
        let mut deal_in_now = [0.0; 4];
        let mut liability_by_seat = [0.0; 4];
        let mut cont = 1.0;
        if let Some(t) = tile {
            let i = t.index() as usize;
            for o in self.opponents() {
                p_now[o] = self.p[o][i];
                cont *= 1.0 - p_now[o];
                deal_in_now[o] = p_now[o] * self.pay[o] * self.l_disc[o];
                if self.liab_seat[o] {
                    let sd4 = cfg.four_meld_win / 3.0;
                    let x4 = 2.0 * cfg.four_meld_win / 3.0;
                    liability_by_seat[o] = self.p_claim4(o, t)
                        * (1.0 - p_now[o])
                        * self.kappa
                        * (sd4 * 2.0 * self.pay_sd[o] + x4 * 3.0 * self.pay[o]);
                }
            }
        }

        // Later: the fold plan over h(a), safest copy first.
        let e = (self.kappa * p_win / self.n + self.sum_win / self.n).clamp(0.0, 1.0);
        let mut kinds: Vec<Tile> = after.concealed.kinds().collect();
        kinds.sort_by(|a, b| {
            self.danger[a.index() as usize]
                .total_cmp(&self.danger[b.index() as usize])
                .then(a.cmp(b))
        });
        let fold_plan: Vec<Tile> = kinds
            .iter()
            .flat_map(|&t| std::iter::repeat_n(t, after.concealed[t] as usize))
            .take(self.h)
            .collect();
        let mut later = 0.0;
        let mut survive = 1.0;
        for &t in &fold_plan {
            later += survive * self.cost[t.index() as usize];
            survive *= 1.0 - e;
        }

        let liability: f64 = liability_by_seat.iter().sum();
        let offence = cont * self.kappa * p_win * pay_me * shares_me;
        let deal_in: f64 = deal_in_now.iter().sum();
        let defence = deal_in
            + cont
                * (self.self_draw.iter().sum::<f64>()
                    + self.other_feeds.iter().sum::<f64>()
                    + later
                    + liability);
        EvTerms {
            action,
            discard: tile,
            ev: offence - defence,
            offence,
            defence,
            p_win,
            p_sd,
            pay_me,
            shares_me,
            phi,
            faan_shanten: fs.shanten,
            target_faan,
            kappa: self.kappa,
            cont,
            p_now,
            deal_in_now,
            self_draw: self.self_draw,
            other_feeds: self.other_feeds,
            later,
            liability,
            liability_by_seat,
            fold_plan,
            tenpai: self.tenpai,
            sd: self.sd,
            pay: self.pay,
            pay_sd: self.pay_sd,
        }
    }

    /// The best discard from a hand that must discard (by `ev`, no tie rule).
    fn best_discard_after(
        &self,
        cfg: &EvConfig,
        action: Action,
        view: &SeatView,
        four_exposed: bool,
    ) -> Option<EvTerms> {
        view.concealed
            .kinds()
            .map(|d| {
                let mut h = view.concealed;
                h.remove(d);
                self.value(
                    cfg,
                    action,
                    &view.with(h, view.melds.clone()),
                    Some(d),
                    four_exposed,
                )
            })
            .max_by(|a, b| a.ev.total_cmp(&b.ev))
    }
}

/// My hand after a kong of `t` declared on my turn.
fn after_own_kong(view: &SeatView, a: Action) -> Option<SeatView> {
    let mut h = view.concealed;
    let mut m = view.melds.clone();
    match a {
        Action::ConcealedKong(t) => {
            h[t] = h[t].checked_sub(4)?;
            m.push(Meld::concealed_kong(t));
        }
        Action::AddedKong(t) => {
            h[t] = h[t].checked_sub(1)?;
            let i = m
                .iter()
                .position(|x| x.kind == MeldKind::Pung && x.tile == t)?;
            m[i].kind = MeldKind::Kong;
        }
        _ => return None,
    }
    Some(view.with(h, m))
}

fn exposed(melds: &[Meld]) -> usize {
    melds.iter().filter(|m| !m.concealed).count()
}

impl EvBot {
    /// An `ev` bot with the given hyper-parameters and belief model.
    pub fn new(config: EvConfig, belief: Arc<dyn Belief>) -> EvBot {
        EvBot { config, belief }
    }

    /// Every candidate with its terms, best `ev` first (§2.3). Empty when the
    /// hand is over or a win is the only sensible move.
    pub fn explain(&self, state: &State) -> Vec<(Action, EvTerms)> {
        self.decide(state).candidates
    }

    /// The chosen action, the candidates and the reason keys.
    ///
    /// # Panics
    /// Panics if the hand is over.
    pub fn decide(&self, state: &State) -> EvDecision {
        let legal = state.legal_actions_vec();
        self.decide_among(state, &legal)
    }

    fn decide_among(&self, state: &State, legal: &[Action]) -> EvDecision {
        if let Some(&w) = legal
            .iter()
            .find(|a| matches!(a, Action::Win | Action::SelfDrawnWin))
        {
            return EvDecision {
                action: w,
                candidates: Vec::new(),
                reasons: Vec::new(),
            };
        }
        let me = state.current_player().expect("the hand is not over");
        let cfg = &self.config;
        let beliefs = self.belief.estimate(state, me);
        let table = Table::new(state, me, cfg, &beliefs);
        let view = SeatView::of(state, me);
        let mut cands: Vec<EvTerms> = Vec::new();
        let action = if state.in_claim() || state.in_rob() {
            // Pass: the current hand, no tile now.
            let pass = table.value(cfg, Action::Pass, &view, None, false);
            let pass_ev = pass.ev;
            cands.push(pass);
            if let Some((tile, from)) = state.claim_tile().filter(|_| state.in_claim()) {
                for &a in legal {
                    let Some(after) = after_claim(&view, a, tile, from) else {
                        continue;
                    };
                    let four = state.rules().twelve_tile_liability && exposed(&after.melds) == 4;
                    let terms = if a == Action::ClaimKong {
                        // The replacement draw is unknown and adds nothing (A8):
                        // the hand after the kong is valued as it stands.
                        Some(table.value(cfg, a, &after, None, four))
                    } else {
                        table.best_discard_after(cfg, a, &after, four)
                    };
                    cands.extend(terms);
                }
            }
            let best_claim = cands
                .iter()
                .filter(|c| c.action != Action::Pass)
                .max_by(|a, b| a.ev.total_cmp(&b.ev));
            match best_claim {
                Some(c) if c.ev > pass_ev + cfg.tie_margin => c.action,
                _ => Action::Pass,
            }
        } else {
            let mut discards: Vec<Tile> = Vec::new();
            for &a in legal {
                match a {
                    Action::Discard(t) => {
                        let mut h = view.concealed;
                        h.remove(t);
                        cands.push(table.value(
                            cfg,
                            a,
                            &view.with(h, view.melds.clone()),
                            Some(t),
                            false,
                        ));
                        discards.push(t);
                    }
                    Action::ConcealedKong(_) | Action::AddedKong(_) => {
                        if let Some(after) = after_own_kong(&view, a) {
                            cands.push(table.value(cfg, a, &after, None, false));
                        }
                    }
                    _ => {}
                }
            }
            let ev_of = |t: Tile| {
                cands
                    .iter()
                    .find(|c| c.action == Action::Discard(t))
                    .map_or(f64::NEG_INFINITY, |c| c.ev)
            };
            let best_ev = discards
                .iter()
                .map(|&t| ev_of(t))
                .fold(f64::NEG_INFINITY, f64::max);
            // Ties (§2.2): within tie_margin of the best, value's order decides.
            let tied: Vec<Tile> = discards
                .iter()
                .copied()
                .filter(|&t| ev_of(t) >= best_ev - cfg.tie_margin)
                .collect();
            let mut chosen = if tied.is_empty() {
                legal[0]
            } else {
                Action::Discard(rank_discards(&view, &tied, Metric::Faan)[0].0)
            };
            let best_kong = cands
                .iter()
                .filter(|c| matches!(c.action, Action::ConcealedKong(_) | Action::AddedKong(_)))
                .max_by(|a, b| a.ev.total_cmp(&b.ev));
            if let Some(k) = best_kong {
                if k.ev > best_ev + cfg.tie_margin {
                    chosen = k.action;
                }
            }
            chosen
        };
        cands.sort_by(|a, b| b.ev.total_cmp(&a.ev).then(a.action.cmp(&b.action)));
        let reasons = reasons(&table, cfg, action, &cands);
        EvDecision {
            action,
            candidates: cands.into_iter().map(|c| (c.action, c)).collect(),
            reasons,
        }
    }
}

/// The reason keys for the chosen action (`docs/ev-bot.md` §7).
fn reasons(table: &Table, cfg: &EvConfig, chosen: Action, cands: &[EvTerms]) -> Vec<EvReason> {
    let Some(star) = cands.iter().find(|c| c.action == chosen) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    // a_safe: the candidate with the smallest DealInNow.
    let safe = cands
        .iter()
        .min_by(|a, b| a.deal_in_total().total_cmp(&b.deal_in_total()))
        .expect("the chosen action is a candidate");
    let stops = star.deal_in_total() - safe.deal_in_total() >= 1.0 && star.defence < safe.defence;
    if stops {
        let seat = table
            .opponents()
            .max_by(|&a, &b| {
                let threat = |o: usize| table.sd[o] * table.pay_sd[o] * table.l_sd[o];
                threat(a).total_cmp(&threat(b))
            })
            .expect("three opponents");
        out.push(EvReason {
            key: "reason.ev.stops_big_hand",
            args: vec![
                ("seat", seat as f64),
                ("p_self_draw", table.sd[seat]),
                ("points", table.pay_sd[seat]),
            ],
        });
    }
    if star.offence > 0.0 && !stops {
        out.push(EvReason {
            key: "reason.ev.offence",
            args: vec![
                ("p_win", star.cont * star.kappa * star.p_win),
                ("points", star.pay_me * star.shares_me),
            ],
        });
    }
    let mut warn: Vec<usize> = table
        .opponents()
        .filter(|&o| star.p_now[o] >= cfg.warn_p)
        .collect();
    warn.sort_by(|&a, &b| star.p_now[b].total_cmp(&star.p_now[a]));
    for o in warn {
        out.push(EvReason {
            key: "reason.ev.deal_in_now",
            args: vec![
                ("seat", o as f64),
                ("p", star.p_now[o]),
                ("points", table.pay[o]),
                ("shares", table.l_disc[o]),
            ],
        });
    }
    for o in table.opponents() {
        let multiplied = table.l_disc[o] == 3.0
            && (star.deal_in_now[o] != 0.0
                || star.self_draw[o] != 0.0
                || star.other_feeds[o] != 0.0);
        let avoided = star.liability_by_seat[o] < 1.0
            && cands
                .iter()
                .any(|c| c.action != chosen && c.liability_by_seat[o] >= 1.0);
        if multiplied || avoided {
            out.push(EvReason {
                key: "reason.ev.liability",
                args: vec![("seat", o as f64)],
            });
        }
    }
    // Margin against the best alternative outside the chosen action's tie
    // (§5 example 1); 0 when every alternative ties.
    let margin = cands
        .iter()
        .filter(|c| c.action != chosen && c.ev < star.ev - cfg.tie_margin)
        .map(|c| star.ev - c.ev)
        .fold(f64::INFINITY, f64::min);
    out.push(EvReason {
        key: "reason.ev.margin",
        args: vec![("points", if margin.is_finite() { margin } else { 0.0 })],
    });
    out
}

impl mahjong_games::Policy<State> for EvBot {
    fn choose<R: rand::Rng + ?Sized>(
        &self,
        state: &State,
        legal: &[Action],
        _rng: &mut R,
    ) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        self.decide_among(state, legal).action
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use mahjong_games::Policy;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn plays_legal_moves_and_terms_are_finite() {
        let bot = EvBot::default();
        let other = Bot::by_name("value").unwrap();
        let mut rng = SmallRng::seed_from_u64(11);
        for _ in 0..6 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(p) = s.current_player() {
                let legal = s.legal_actions_vec();
                let a = if p == 0 {
                    let d = bot.decide(&s);
                    assert!(legal.contains(&d.action), "{:?} not legal", d.action);
                    for (_, t) in &d.candidates {
                        assert!(t.ev.is_finite() && (0.0..=1.0).contains(&t.cont), "{t:?}");
                        assert!(t.fold_plan.len() <= 8);
                    }
                    for r in &d.reasons {
                        assert!(r.key.starts_with("reason.ev."));
                    }
                    d.action
                } else {
                    other.choose(&s, &legal, &mut rng)
                };
                s.apply(a);
            }
        }
    }

    #[test]
    fn heuristic_faan_reads_exposed_melds() {
        let mut rng = SmallRng::seed_from_u64(1);
        let s = State::new_deal(RuleSet::default(), &mut rng);
        for o in 1..4 {
            assert_eq!(heuristic_faan(&s, o), 3, "no melds: 3 faan");
        }
    }

    #[test]
    fn options_parse_and_reject() {
        let mut c = EvConfig::default();
        c.set("horizon", "4").unwrap();
        c.set("feed_clip", "0.4-2").unwrap();
        c.set("fold", "safest").unwrap();
        c.set("floor", "none").unwrap();
        assert_eq!((c.horizon, c.feed_lo, c.feed_hi), (4, 0.4, 2.0));
        assert!(!c.pay_floor);
        assert!(c.set("floor", "max").is_err());
        assert!(c.set("horizon", "0").is_err());
        assert!(c.set("fold", "greedy").is_err());
        assert!(c.set("nope", "1").is_err());
        assert!(load_belief("models/does-not-exist.mjnn").is_err());
    }
}
