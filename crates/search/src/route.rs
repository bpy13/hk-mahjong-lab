//! The route-first bots (Brian's design, 2026-11-15).
//!
//! The `value` bot heads for the *nearest* plan. These bots instead weigh every
//! **route** (a family of finished hands: all pungs, a half flush, a value-pung
//! hand, …) by the chance of finishing it before the wall runs out times the
//! points it pays, and follow the route with the best product.
//!
//! * [`evaluate`]: for a 13-tile hand, the best target of each feasible route:
//!   hits still needed, and the helping tiles weighted by where they can come
//!   from (own draw, any discard, or the left player's discard for a chow).
//! * [`chain`]: P(finishing within the rounds left) from those numbers.
//! * [`RouteBot`]: the `chain` bot (numbers from [`chain`]), the `chain2` bot
//!   ([`chain_measured`], helper means measured by `examples/fit_routes.rs`, file
//!   in the env var `ROUTE_HELPERS`) and the `route` bot (numbers only from the
//!   table fitted by `fit_routes`, file in the env var `ROUTE_TABLE`).

use crate::bots::{after_claim, rank_discards, Metric};
use crate::target_value::{tiles_left_bucket, TILES_LEFT_BUCKETS};
use crate::view::SeatView;
use mahjong_analysis::{
    for_each_route_target, target_shanten, HandView, Structure, SuitPlan, Target,
};
use mahjong_games::Policy;
use mahjong_hk::{Action, Meld, MeldKind, RuleSet, State};
use mahjong_tiles::{Tile, TileCounts};
use rand::Rng;
use std::cmp::{Ordering, Reverse};
use std::collections::HashMap;
use std::sync::Arc;

/// A family of finished hands the bot can head for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Route {
    /// Mixed tiles, all pungs (對對糊).
    AllPungs = 0,
    /// Mixed tiles, all chows, concealed, won by self-draw (平糊 + 門前清 + 自摸).
    /// This route never claims.
    AllChows = 1,
    /// Thirteen orphans or all honours.
    Limit = 2,
    /// One suit plus honours, any sets (混一色).
    HalfFlush = 3,
    /// One suit only, any sets (清一色).
    FullFlush = 4,
    /// Half flush with all pungs.
    HalfFlushPungs = 5,
    /// Full flush with all pungs.
    FullFlushPungs = 6,
    /// Mixed tiles, any sets, faan from dragon and wind pungs (and the rest).
    ValuePungs = 7,
}

/// Number of routes.
pub const ROUTES: usize = 8;

impl Route {
    /// Every route, in index order.
    pub const ALL: [Route; ROUTES] = [
        Route::AllPungs,
        Route::AllChows,
        Route::Limit,
        Route::HalfFlush,
        Route::FullFlush,
        Route::HalfFlushPungs,
        Route::FullFlushPungs,
        Route::ValuePungs,
    ];

    /// The route of a target plan.
    pub fn of(plan: SuitPlan, structure: Structure, limit: bool) -> Route {
        let pungs = structure == Structure::AllPungs;
        match plan {
            _ if limit => Route::Limit,
            SuitPlan::Honours | SuitPlan::Orphans => Route::Limit,
            SuitPlan::HalfFlush(_) if pungs => Route::HalfFlushPungs,
            SuitPlan::HalfFlush(_) => Route::HalfFlush,
            SuitPlan::FullFlush(_) if pungs => Route::FullFlushPungs,
            SuitPlan::FullFlush(_) => Route::FullFlush,
            SuitPlan::Mixed => match structure {
                Structure::AllPungs => Route::AllPungs,
                Structure::AllChows => Route::AllChows,
                Structure::Any => Route::ValuePungs,
            },
        }
    }

    /// The route of an analysis [`Target`].
    pub fn of_target(t: &Target) -> Route {
        Route::of(t.plan, t.structure, t.limit)
    }

    /// The one route a finished winning hand belongs to.
    pub fn of_win(score: &mahjong_hk::Score) -> Route {
        use mahjong_hk::Pattern as P;
        let has = |p: P| score.patterns.iter().any(|(q, _)| *q == p);
        let pungs = has(P::AllPungs);
        if has(P::FullFlush) || has(P::NineGates) {
            if pungs {
                Route::FullFlushPungs
            } else {
                Route::FullFlush
            }
        } else if score.limit {
            Route::Limit
        } else if has(P::HalfFlush) {
            if pungs {
                Route::HalfFlushPungs
            } else {
                Route::HalfFlush
            }
        } else if pungs {
            Route::AllPungs
        } else if has(P::AllChows) {
            Route::AllChows
        } else {
            Route::ValuePungs
        }
    }

    /// A short English name (tables and logs).
    pub fn name(self) -> &'static str {
        match self {
            Route::AllPungs => "all_pungs",
            Route::AllChows => "all_chows",
            Route::Limit => "limit",
            Route::HalfFlush => "half_flush",
            Route::FullFlush => "full_flush",
            Route::HalfFlushPungs => "half_flush_pungs",
            Route::FullFlushPungs => "full_flush_pungs",
            Route::ValuePungs => "value_pungs",
        }
    }

    /// Whether the finished hand may hold chows.
    fn chows(self) -> bool {
        matches!(
            self,
            Route::AllChows | Route::HalfFlush | Route::FullFlush | Route::ValuePungs
        )
    }

    /// How the helping tiles shrink with each hit ([`chain`]). Brian's starter
    /// guess: 0.75 for routes with chows, 0.65 for the rest (all pungs, and the
    /// limit hands, which have no chows either). A fitted table replaces it.
    fn decay(self) -> f64 {
        if self.chows() {
            0.75
        } else {
            0.65
        }
    }
}

/// Brian's starting assumptions for "where can the tile come from": the share
/// of an opponent's copies of a suited tile they keep (1/3) and of an honour
/// (2/3); the rest they throw sooner or later.
pub const SUITED_KEEP: f32 = 1.0 / 3.0;
/// See [`SUITED_KEEP`].
pub const HONOUR_KEEP: f32 = 2.0 / 3.0;

/// One feasible route of a 13-tile hand, at its best target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteEval {
    pub route: Route,
    /// Hits still needed to win (distance to ready + 1).
    pub needed: u8,
    /// Helping tiles for the next hit, weighted by where they can come from.
    pub helpers: f32,
    /// Helping tiles, unweighted (unseen copies).
    pub live: u32,
    /// Guaranteed faan of the target.
    pub faan: u8,
    /// Points per share for that faan.
    pub points: f64,
}

/// One target of a route: a plan, its structure and its scoring pungs.
#[derive(Clone, Debug)]
pub(crate) struct RouteTarget {
    pub(crate) plan: SuitPlan,
    pub(crate) structure: Structure,
    pub(crate) pungs: Vec<Tile>,
    pub(crate) faan: u8,
    pub(crate) distance: i8,
}

/// Per route, its best targets: the lowest distance, then the highest faan
/// (all of them when several tie).
fn best_targets(view: &HandView, rules: &RuleSet) -> [Vec<RouteTarget>; ROUTES] {
    let mut best: [Vec<RouteTarget>; ROUTES] = Default::default();
    for_each_route_target(
        view,
        rules,
        |plan, structure, pungs, faan, limit, distance| {
            let list = &mut best[Route::of(plan, structure, limit) as usize];
            let order = list.first().map(|b: &RouteTarget| {
                (distance, Reverse(faan)).cmp(&(b.distance, Reverse(b.faan)))
            });
            if order == Some(Ordering::Greater) {
                return;
            }
            if order == Some(Ordering::Less) {
                list.clear();
            }
            list.push(RouteTarget {
                plan,
                structure,
                pungs: pungs.to_vec(),
                faan,
                distance,
            });
        },
    );
    best
}

/// Whether `t` completes a chow with two tiles of `hand`.
fn completes_chow(hand: &TileCounts, t: Tile) -> bool {
    let held = |d: i8| t.offset(d).is_some_and(|u| hand[u] > 0);
    (held(-2) && held(-1)) || (held(-1) && held(1)) || (held(1) && held(2))
}

/// How many places a helping tile `t` can come from (see the module docs and
/// Brian's rule): own draw counts 1; a tile that wins or completes a held pair
/// can come from any of the three opponents' throws; a chow only from the left
/// player's.
fn places(route: Route, hand: &TileCounts, t: Tile, wins: bool) -> f32 {
    let keep = if t.is_honour() {
        HONOUR_KEEP
    } else {
        SUITED_KEEP
    };
    let throw = 1.0 - keep;
    if route == Route::AllChows {
        return 1.0; // self-draw only
    }
    if wins {
        return 1.0 + 3.0 * throw;
    }
    if route == Route::Limit {
        return 1.0;
    }
    if hand[t] == 2 {
        return 1.0 + 3.0 * throw;
    }
    if route.chows() && completes_chow(hand, t) {
        return 1.0 + throw;
    }
    1.0
}

/// The helping tiles of one target: (weighted, unseen copies) of the tile
/// kinds that lower its distance.
pub(crate) fn helpers_of(view: &HandView, route: Route, target: &RouteTarget) -> (f32, u32) {
    let (mut weighted, mut live) = (0f32, 0u32);
    let mut hand = *view.concealed;
    for t in Tile::all_kinds() {
        let unseen = view.unseen[t];
        if unseen == 0 || hand[t] >= 4 {
            continue;
        }
        hand.add(t);
        let after = target_shanten(
            &HandView {
                concealed: &hand,
                ..*view
            },
            target.plan,
            target.structure,
            &target.pungs,
        );
        hand.remove(t);
        if let Some(a) = after.filter(|&a| a < target.distance) {
            live += unseen as u32;
            weighted += unseen as f32 * places(route, &hand, t, a < 0);
        }
    }
    (weighted, live)
}

/// Every feasible route of a 13-tile hand (fewer concealed tiles with melds on
/// the table), each at its best target: the lowest distance, then the highest
/// faan, then the most weighted helpers (so a flush in one suit is not
/// credited with the helping tiles of another suit's flush).
pub fn evaluate(view: &HandView, rules: &RuleSet) -> Vec<RouteEval> {
    Route::ALL
        .iter()
        .zip(best_targets(view, rules))
        .filter_map(|(&route, targets)| {
            let (target, (helpers, live)) = targets
                .iter()
                .map(|t| (t, helpers_of(view, route, t)))
                .max_by(|a, b| a.1 .0.total_cmp(&b.1 .0))?;
            Some(RouteEval {
                route,
                needed: (target.distance + 1).max(0) as u8,
                helpers,
                live,
                faan: target.faan,
                points: rules.points(target.faan) as f64,
            })
        })
        .collect()
}

/// The result of [`chain`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChainOut {
    /// P(all the needed hits come within the rounds left).
    pub p_win: f64,
    /// The mean round of the win, given a win (1 = the next round).
    pub mean_round: f64,
}

/// P(a route finishes in time): a small table over (hits done, rounds used).
/// Each round brings the next hit with probability min(1, helpers_k /
/// `unseen_total`), where helpers_0 is the route's weighted helpers and later
/// hits have max(2, helpers_0 × d^k) ([`Route::decay`]). This is the recorded
/// "initial chain method" of the `chain` bot; keep it as it is.
pub fn chain(eval: &RouteEval, rounds_left: usize, unseen_total: u32) -> ChainOut {
    let h0 = eval.helpers as f64;
    let d = eval.route.decay();
    chain_with(eval, rounds_left, unseen_total, |k, _| {
        (h0 * d.powi(k as i32)).max(2.0)
    })
}

/// [`chain`] for the `chain2` bot: a later hit has the measured mean helpers
/// of the route with the hits then remaining ([`HelperMeans`]), capped at the
/// previous step's helpers. Where no mean was measured, [`chain`]'s decay
/// stands in (still capped).
pub fn chain_measured(
    eval: &RouteEval,
    rounds_left: usize,
    unseen_total: u32,
    means: &HelperMeans,
) -> ChainOut {
    let need = eval.needed as usize;
    let h0 = eval.helpers as f64;
    let d = eval.route.decay();
    chain_with(eval, rounds_left, unseen_total, |k, prev| {
        means
            .get(eval.route, need - k)
            .unwrap_or_else(|| (h0 * d.powi(k as i32)).max(2.0))
            .min(prev)
    })
}

/// The table over (hits done, rounds used) behind [`chain`] and
/// [`chain_measured`]; `later(k, helpers_{k-1})` gives the helpers for hit `k`
/// (k >= 1).
fn chain_with(
    eval: &RouteEval,
    rounds_left: usize,
    unseen_total: u32,
    later: impl Fn(usize, f64) -> f64,
) -> ChainOut {
    let need = eval.needed as usize;
    if need == 0 {
        return ChainOut {
            p_win: 1.0,
            mean_round: 0.0,
        };
    }
    let unseen = unseen_total.max(1) as f64;
    let mut helpers = vec![eval.helpers as f64];
    for k in 1..need {
        helpers.push(later(k, helpers[k - 1]));
    }
    let p: Vec<f64> = helpers.iter().map(|h| (h / unseen).min(1.0)).collect();
    // at[k]: P(k hits done and not yet won) after the rounds so far.
    let mut at = vec![0.0; need];
    at[0] = 1.0;
    let (mut p_win, mut rounds) = (0.0, 0.0);
    for round in 1..=rounds_left {
        let win = at[need - 1] * p[need - 1];
        for k in (1..need).rev() {
            at[k] = at[k] * (1.0 - p[k]) + at[k - 1] * p[k - 1];
        }
        at[0] *= 1.0 - p[0];
        p_win += win;
        rounds += win * round as f64;
    }
    ChainOut {
        p_win,
        mean_round: if p_win > 0.0 { rounds / p_win } else { 0.0 },
    }
}

/// Rounds left before the wall runs out: a round is four draws; a part round
/// counts (a guess: the next draws still come).
pub fn rounds_left(wall_remaining: usize) -> usize {
    wall_remaining.div_ceil(4)
}

/// Hits-needed buckets of the table: 0–9 (9 = nine or more).
pub const NEEDED_BANDS: usize = 10;
/// Weighted-helper buckets: 0–3, 4–7, 8–12, 13–19, 20–29, 30+.
pub const HELPER_BANDS: usize = 6;
/// Melds on the table: 0–4.
pub const MELD_BANDS: usize = 5;
/// Size of the route table.
pub const ROUTE_CELLS: usize =
    ROUTES * NEEDED_BANDS * HELPER_BANDS * TILES_LEFT_BUCKETS * MELD_BANDS;

/// The band of a weighted helper count.
pub fn helper_band(helpers: f32) -> usize {
    [4.0, 8.0, 13.0, 20.0, 30.0]
        .iter()
        .take_while(|&&edge| helpers >= edge)
        .count()
}

/// One cell of the route table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RouteKey {
    pub route: Route,
    /// 0–6.
    pub needed: usize,
    /// 0–5 ([`helper_band`]).
    pub helpers: usize,
    /// 0–5 (the tiles-left buckets of [`crate::target_value`]).
    pub wall: usize,
    /// 0–4.
    pub melds: usize,
}

impl RouteKey {
    /// The key of a route evaluation with `wall_remaining` tiles to draw and
    /// `melds` melds on the table.
    pub fn of(eval: &RouteEval, wall_remaining: usize, melds: usize, rules: &RuleSet) -> RouteKey {
        RouteKey {
            route: eval.route,
            needed: (eval.needed as usize).min(NEEDED_BANDS - 1),
            helpers: helper_band(eval.helpers),
            wall: tiles_left_bucket(wall_remaining, rules),
            melds: melds.min(MELD_BANDS - 1),
        }
    }

    /// The flat index.
    pub fn cell(&self) -> usize {
        (((self.route as usize * NEEDED_BANDS + self.needed) * HELPER_BANDS + self.helpers)
            * TILES_LEFT_BUCKETS
            + self.wall)
            * MELD_BANDS
            + self.melds
    }

    /// The key at a flat index (inverse of [`RouteKey::cell`]).
    pub fn from_cell(cell: usize) -> RouteKey {
        let melds = cell % MELD_BANDS;
        let rest = cell / MELD_BANDS;
        let wall = rest % TILES_LEFT_BUCKETS;
        let rest = rest / TILES_LEFT_BUCKETS;
        let helpers = rest % HELPER_BANDS;
        let rest = rest / HELPER_BANDS;
        RouteKey {
            route: Route::ALL[rest / NEEDED_BANDS],
            needed: rest % NEEDED_BANDS,
            helpers,
            wall,
            melds,
        }
    }
}

/// One entry of a fitted route table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TableEntry {
    /// Records behind the entry.
    pub n: f64,
    /// P(a win before the hand ends), shrunk toward coarser entries.
    pub p_win: f64,
    /// Mean points per share given a win.
    pub points: f64,
    /// Mean rounds from the record to the win, given a win.
    pub rounds: f64,
}

/// The number after `"name":` in one line of a file `fit_routes` writes (one
/// object per line, so no general JSON parser is needed).
fn field(line: &str, name: &str) -> Result<f64, String> {
    let pat = format!("\"{name}\":");
    let at = line
        .find(&pat)
        .ok_or_else(|| format!("no {name} in {line:?}"))?;
    let rest = line[at + pat.len()..].trim_start();
    let end = rest.find([',', '}']).unwrap_or(rest.len());
    rest[..end]
        .trim()
        .parse()
        .map_err(|_| format!("bad {name} in {line:?}"))
}

/// Mean weighted helpers per (route, hits needed), measured by `fit_routes`
/// (`helpers.json`), for the `chain2` bot.
#[derive(Clone, Debug, Default)]
pub struct HelperMeans {
    /// `[route][needed band]`.
    means: [[Option<f64>; NEEDED_BANDS]; ROUTES],
}

impl HelperMeans {
    /// Reads `helpers.json`.
    pub fn load(path: &str) -> Result<HelperMeans, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("helper means {path:?}: {e}"))?;
        HelperMeans::parse(&text)
    }

    /// Parses the text of `helpers.json`.
    pub fn parse(text: &str) -> Result<HelperMeans, String> {
        let mut out = HelperMeans::default();
        for line in text.lines().filter(|l| l.contains("\"route_index\":")) {
            let r = field(line, "route_index")? as usize;
            let k = field(line, "needed")? as usize;
            if r >= ROUTES || k >= NEEDED_BANDS {
                return Err(format!("route {r} or needed {k} out of range"));
            }
            out.means[r][k] = Some(field(line, "helpers")?);
        }
        Ok(out)
    }

    /// The mean for a route with `needed` hits to go (6 or more share a band).
    pub fn get(&self, route: Route, needed: usize) -> Option<f64> {
        self.means[route as usize][needed.min(NEEDED_BANDS - 1)]
    }

    /// Sets one mean (for tests and tools).
    pub fn set(&mut self, route: Route, needed: usize, mean: f64) {
        self.means[route as usize][needed.min(NEEDED_BANDS - 1)] = Some(mean);
    }
}

/// A route table written by `fit_routes` (`table.json`).
#[derive(Clone, Debug, Default)]
pub struct RouteTable {
    pub entries: HashMap<usize, TableEntry>,
}

impl RouteTable {
    /// Reads `table.json`.
    pub fn load(path: &str) -> Result<RouteTable, String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("route table {path:?}: {e}"))?;
        RouteTable::parse(&text)
    }

    /// Parses the text of `table.json`: one entry object per line, as
    /// `fit_routes` writes it.
    pub fn parse(text: &str) -> Result<RouteTable, String> {
        let mut entries = HashMap::new();
        for line in text.lines().filter(|l| l.contains("\"cell\":")) {
            let cell = field(line, "cell")? as usize;
            if cell >= ROUTE_CELLS {
                return Err(format!("cell {cell} out of range"));
            }
            entries.insert(
                cell,
                TableEntry {
                    n: field(line, "n")?,
                    p_win: field(line, "p_win")?,
                    points: field(line, "points")?,
                    rounds: field(line, "rounds")?,
                },
            );
        }
        Ok(RouteTable { entries })
    }
}

/// A route evaluation with the numbers the bot uses for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scored {
    pub eval: RouteEval,
    /// P(win) by [`chain`], [`chain_measured`] or the table.
    pub p_win: f64,
    /// Points per share: the target's, or the table's mean given a win.
    pub points: f64,
    /// Whether the numbers came from the table.
    pub from_table: bool,
}

impl Scored {
    /// What the bot maximises.
    pub fn product(&self) -> f64 {
        self.p_win * self.points
    }
}

/// The best (highest product) route of a list.
pub fn best_route(scored: &[Scored]) -> Option<Scored> {
    scored
        .iter()
        .copied()
        .max_by(|a, b| a.product().total_cmp(&b.product()))
}

/// The `chain`, `chain2` and `route` bots (see the module docs).
#[derive(Clone, Debug)]
pub struct RouteBot {
    /// The fitted table (`route`): every number comes from it.
    table: Option<Arc<RouteTable>>,
    /// Measured helper means (`chain2`); without them and a table, `chain`.
    helpers: Option<Arc<HelperMeans>>,
    /// P(following the second-best route at a discard), from `ROUTE_EXPLORE`.
    explore: f64,
    /// Whether a hand is worth the sum of its routes' products. The table
    /// counts a win under the one route of the finished hand, so the sum is
    /// the expected points. `ROUTE_VALUE=max` takes the best route only.
    sum: bool,
}

fn explore_from_env() -> f64 {
    std::env::var("ROUTE_EXPLORE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0)
}

impl RouteBot {
    /// The `chain` bot.
    pub fn chain() -> RouteBot {
        RouteBot {
            table: None,
            helpers: None,
            explore: explore_from_env(),
            sum: false,
        }
    }

    /// The `chain2` bot with the helper means at `path`.
    pub fn with_helpers(path: &str) -> Result<RouteBot, String> {
        Ok(RouteBot {
            helpers: Some(Arc::new(HelperMeans::load(path)?)),
            ..RouteBot::chain()
        })
    }

    /// The `chain2` bot with the helper means named by `ROUTE_HELPERS`.
    pub fn chain2_from_env() -> Result<RouteBot, String> {
        let path = std::env::var("ROUTE_HELPERS")
            .map_err(|_| "the chain2 bot needs ROUTE_HELPERS=<helpers.json>".to_string())?;
        RouteBot::with_helpers(&path)
    }

    /// The `route` bot with the table at `path`. The table must hold every
    /// cell (as `fit_routes` writes it), so the bot never needs another source.
    pub fn with_table(path: &str) -> Result<RouteBot, String> {
        let table = RouteTable::load(path)?;
        if table.entries.len() != ROUTE_CELLS {
            return Err(format!(
                "route table {path:?} has {} of {ROUTE_CELLS} cells: refit it with fit_routes",
                table.entries.len()
            ));
        }
        Ok(RouteBot {
            table: Some(Arc::new(table)),
            sum: std::env::var("ROUTE_VALUE").map_or(true, |v| v != "max"),
            ..RouteBot::chain()
        })
    }

    /// The `route` bot with the table named by `ROUTE_TABLE`.
    pub fn route_from_env() -> Result<RouteBot, String> {
        let path = std::env::var("ROUTE_TABLE")
            .map_err(|_| "the route bot needs ROUTE_TABLE=<table.json>".to_string())?;
        RouteBot::with_table(&path)
    }

    /// Every feasible route of a 13-tile view with its numbers.
    pub fn score(&self, view: &SeatView, wall_remaining: usize) -> Vec<Scored> {
        let unseen_total = view.unseen.total() as u32;
        let rounds = rounds_left(wall_remaining);
        evaluate(&view.hand_view(), &view.rules)
            .into_iter()
            .map(|eval| {
                let key = RouteKey::of(&eval, wall_remaining, view.melds.len(), &view.rules);
                if let Some(t) = &self.table {
                    let e = t.entries[&key.cell()]; // complete: checked at load
                    return Scored {
                        eval,
                        p_win: e.p_win,
                        points: e.points,
                        from_table: true,
                    };
                }
                let p_win = match &self.helpers {
                    Some(m) => chain_measured(&eval, rounds, unseen_total, m).p_win,
                    None => chain(&eval, rounds, unseen_total).p_win,
                };
                Scored {
                    eval,
                    p_win,
                    points: eval.points,
                    from_table: false,
                }
            })
            .collect()
    }

    /// The highest product of a 13-tile view (0 with no feasible route).
    fn value_of(&self, view: &SeatView, wall_remaining: usize) -> f64 {
        self.worth(&self.score(view, wall_remaining))
    }

    /// What a hand with these scored routes is worth to the bot.
    pub fn worth(&self, scored: &[Scored]) -> f64 {
        if self.sum {
            scored.iter().map(|s| s.product()).sum()
        } else {
            best_route(scored).map_or(0.0, |s| s.product())
        }
    }

    /// The scored routes after each distinct discard of a 14-tile view.
    pub fn discard_routes(
        &self,
        view: &SeatView,
        wall_remaining: usize,
    ) -> Vec<(Tile, Vec<Scored>)> {
        view.concealed
            .kinds()
            .map(|t| {
                let mut hand = view.concealed;
                hand.remove(t);
                (
                    t,
                    self.score(&view.with(hand, view.melds.clone()), wall_remaining),
                )
            })
            .collect()
    }

    /// The discard the bot throws from a 14-tile view, with the route it
    /// follows. `explore` follows the second-best route. `ban` is the
    /// [`State::discard_ban`] mask (rules §4.1.4 食換): those kinds are not
    /// thrown.
    pub fn pick_discard(
        &self,
        view: &SeatView,
        wall_remaining: usize,
        explore: bool,
        ban: u64,
    ) -> (Tile, Option<Scored>) {
        let mut options = self.discard_routes(view, wall_remaining);
        options.retain(|(t, _)| ban & (1u64 << t.index()) == 0);
        if explore {
            // The best product each route reaches over all discards.
            let mut by_route: Vec<(Scored, Tile)> = Vec::new();
            for (t, scored) in &options {
                for s in scored {
                    match by_route
                        .iter_mut()
                        .find(|(b, _)| b.eval.route == s.eval.route)
                    {
                        Some(b) if b.0.product() >= s.product() => {}
                        Some(b) => *b = (*s, *t),
                        None => by_route.push((*s, *t)),
                    }
                }
            }
            by_route.sort_by(|a, b| b.0.product().total_cmp(&a.0.product()));
            if let Some(&(s, t)) = by_route.get(1) {
                return (t, Some(s));
            }
        }
        let best: Vec<(Tile, Option<Scored>)> =
            options.iter().map(|(t, s)| (*t, best_route(s))).collect();
        let worths: Vec<f64> = options.iter().map(|(_, s)| self.worth(s)).collect();
        let top = worths.iter().copied().fold(f64::MIN, f64::max);
        let tied: Vec<Tile> = best
            .iter()
            .zip(&worths)
            .filter(|(_, w)| **w >= top - 1e-12 * top.abs())
            .map(|(&(t, _), _)| t)
            .collect();
        let pick = if tied.len() == 1 {
            tied[0]
        } else {
            rank_discards(view, &tied, Metric::Faan)[0].0
        };
        let route = best.iter().find(|(t, _)| *t == pick).and_then(|(_, s)| *s);
        (pick, route)
    }

    /// The bot's move, with the route it follows after a discard.
    pub fn decide<R: Rng + ?Sized>(
        &self,
        state: &State,
        legal: &[Action],
        rng: &mut R,
    ) -> (Action, Option<Scored>) {
        if let Some(&w) = legal
            .iter()
            .find(|a| matches!(a, Action::Win | Action::SelfDrawnWin))
        {
            return (w, None);
        }
        let view = SeatView::of_current(state);
        let wall = state.wall_remaining();
        if state.in_claim() || state.in_rob() {
            return (self.claim(&view, state, legal, wall), None);
        }
        // Draw only when exploring, so `ROUTE_EXPLORE=0` leaves the random stream alone.
        let explore = self.explore > 0.0 && rng.gen::<f64>() < self.explore;
        let (tile, route) = self.pick_discard(&view, wall, explore, state.discard_ban());
        let now = {
            let mut h = view.concealed;
            h.remove(tile);
            self.value_of(&view.with(h, view.melds.clone()), wall)
        };
        // A kong: declare it if the hand is worth at least as much afterwards.
        for &a in legal {
            let after = match a {
                Action::ConcealedKong(t) => {
                    let mut h = view.concealed;
                    h[t] -= 4;
                    let mut m = view.melds.clone();
                    m.push(Meld::concealed_kong(t));
                    view.with(h, m)
                }
                Action::AddedKong(t) => {
                    let mut h = view.concealed;
                    h.remove(t);
                    let mut m = view.melds.clone();
                    if let Some(x) = m
                        .iter_mut()
                        .find(|x| x.kind == MeldKind::Pung && x.tile == t)
                    {
                        x.kind = MeldKind::Kong;
                    }
                    view.with(h, m)
                }
                _ => continue,
            };
            if self.value_of(&after, wall) >= now {
                return (a, None);
            }
        }
        (Action::Discard(tile), route)
    }

    /// Claims a meld only if the hand is then worth more than when passing.
    fn claim(&self, view: &SeatView, state: &State, legal: &[Action], wall: usize) -> Action {
        let Some((tile, from)) = state.claim_tile() else {
            return Action::Pass;
        };
        let mut best = (self.value_of(view, wall), Action::Pass);
        for &a in legal {
            let Some(after) = after_claim(view, a, tile, from) else {
                continue;
            };
            let worth = if a == Action::ClaimKong {
                // A kong leaves a 13-tile hand (the replacement draw comes later).
                self.value_of(&after, wall)
            } else {
                self.discard_routes(&after, wall)
                    .iter()
                    .map(|(_, s)| self.worth(s))
                    .fold(0.0, f64::max)
            };
            if worth > best.0 {
                best = (worth, a);
            }
        }
        best.1
    }
}

impl Policy<State> for RouteBot {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        self.decide(state, legal, rng).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahjong_tiles::parse_tiles;

    fn c(s: &str) -> TileCounts {
        TileCounts::from_tiles(&parse_tiles(s).unwrap())
    }

    fn eval_of(hand: &str, seat: usize) -> Vec<RouteEval> {
        let concealed = c(hand);
        let mut unseen = TileCounts::full_set();
        for t in concealed.tiles() {
            unseen.remove(t);
        }
        let view = HandView {
            concealed: &concealed,
            melds: &[],
            seat,
            bonus: 0,
            unseen: &unseen,
        };
        let rules = RuleSet {
            flowers: false,
            ..RuleSet::default()
        };
        evaluate(&view, &rules)
    }

    fn get(evals: &[RouteEval], r: Route) -> RouteEval {
        *evals.iter().find(|e| e.route == r).expect("route feasible")
    }

    #[test]
    fn targets_map_onto_routes() {
        use mahjong_tiles::Suit;
        let s = Suit::Dots;
        assert_eq!(
            Route::of(SuitPlan::Mixed, Structure::Any, false),
            Route::ValuePungs
        );
        assert_eq!(
            Route::of(SuitPlan::Mixed, Structure::AllPungs, false),
            Route::AllPungs
        );
        assert_eq!(
            Route::of(SuitPlan::Mixed, Structure::AllChows, false),
            Route::AllChows
        );
        assert_eq!(
            Route::of(SuitPlan::HalfFlush(s), Structure::Any, false),
            Route::HalfFlush
        );
        assert_eq!(
            Route::of(SuitPlan::HalfFlush(s), Structure::AllChows, false),
            Route::HalfFlush
        );
        assert_eq!(
            Route::of(SuitPlan::HalfFlush(s), Structure::AllPungs, false),
            Route::HalfFlushPungs
        );
        assert_eq!(
            Route::of(SuitPlan::FullFlush(s), Structure::Any, false),
            Route::FullFlush
        );
        assert_eq!(
            Route::of(SuitPlan::FullFlush(s), Structure::AllPungs, false),
            Route::FullFlushPungs
        );
        assert_eq!(
            Route::of(SuitPlan::Honours, Structure::AllPungs, true),
            Route::Limit
        );
        assert_eq!(
            Route::of(SuitPlan::Orphans, Structure::Any, true),
            Route::Limit
        );
        for (i, r) in Route::ALL.iter().enumerate() {
            assert_eq!(*r as usize, i);
        }
    }

    #[test]
    fn flushes_are_tried_in_every_suit() {
        // Six pins and six bamboo (a tie goes to bamboo as the "longest suit"),
        // but the pins are nearer: two chows and a West (no value for South).
        let e = eval_of("123p 456p W 123s 5s 7s 9s", 1);
        let hf = get(&e, Route::HalfFlush);
        assert_eq!(hf.needed, 5, "{hf:?}"); // pins: distance 4; bamboo would be 5
                                            // Single dragons count toward value pungs (each needs two more copies):
                                            // Red + Green + concealed = 3 faan.
        let e = eval_of("123m 456p 789s R G 5m 9p", 1);
        let v = get(&e, Route::ValuePungs);
        assert_eq!(v.faan, 3, "{v:?}");
    }

    #[test]
    fn helper_weights_follow_where_a_tile_can_come_from() {
        let throw_s = 1.0 - SUITED_KEEP;
        let throw_h = 1.0 - HONOUR_KEEP;
        let hand = c("45m 77p");
        let m = |s: &str| s.parse::<Tile>().unwrap();
        // A winning tile can come from any discard, except on the self-draw route.
        assert_eq!(
            places(Route::ValuePungs, &hand, m("3m"), true),
            1.0 + 3.0 * throw_s
        );
        assert_eq!(places(Route::AllChows, &hand, m("3m"), true), 1.0);
        // A chow only from the left player, and only on a route with chows.
        assert_eq!(
            places(Route::ValuePungs, &hand, m("3m"), false),
            1.0 + throw_s
        );
        assert_eq!(places(Route::AllPungs, &hand, m("3m"), false), 1.0);
        // A pung of a held pair from any discard.
        assert_eq!(
            places(Route::AllPungs, &hand, m("7p"), false),
            1.0 + 3.0 * throw_s
        );
        assert_eq!(places(Route::AllChows, &hand, m("7p"), false), 1.0);
        assert_eq!(places(Route::Limit, &hand, m("7p"), false), 1.0);
        let hand = c("RR");
        assert_eq!(
            places(Route::HalfFlush, &hand, m("R"), false),
            1.0 + 3.0 * throw_h
        );
        // Otherwise own draw only.
        assert_eq!(places(Route::ValuePungs, &hand, m("9s"), false), 1.0);
    }

    #[test]
    fn all_chows_counts_self_draw_and_only_self_draw() {
        // Ready on 3-6s with a concealed all-chow hand: 3 faan with self-draw.
        let e = eval_of("123m 456m 789p 22p 45s", 1);
        let ch = get(&e, Route::AllChows);
        assert_eq!((ch.needed, ch.faan), (1, 3), "{ch:?}");
        // Eight live winning tiles, each from own draw only.
        assert_eq!(ch.live, 8);
        assert_eq!(ch.helpers, 8.0);
    }

    fn eval(needed: u8, helpers: f32, route: Route) -> RouteEval {
        RouteEval {
            route,
            needed,
            helpers,
            live: helpers as u32,
            faan: 3,
            points: 8.0,
        }
    }

    #[test]
    fn chain_by_hand() {
        // needed 2, helpers 4 then max(2, 4 × 0.75) = 3, of 10 unseen: per
        // round 0.4 then 0.3. Within 3 rounds: hits in rounds (1,2) 0.4·0.3,
        // (1,3) 0.4·0.7·0.3, (2,3) 0.6·0.4·0.3.
        let out = chain(&eval(2, 4.0, Route::ValuePungs), 3, 10);
        let (a, b, c) = (0.4 * 0.3, 0.4 * 0.7 * 0.3, 0.6 * 0.4 * 0.3);
        assert!((out.p_win - (a + b + c)).abs() < 1e-12, "{out:?}");
        let mean = (2.0 * a + 3.0 * (b + c)) / (a + b + c);
        assert!((out.mean_round - mean).abs() < 1e-12, "{out:?}");
        // All pungs decay faster: 4 × 0.65 = 2.6.
        let out = chain(&eval(2, 4.0, Route::AllPungs), 2, 10);
        assert!((out.p_win - 0.4 * 0.26).abs() < 1e-12, "{out:?}");
        // The floor of 2 helpers after the first hit.
        let out = chain(&eval(2, 2.0, Route::ValuePungs), 2, 10);
        assert!((out.p_win - 0.2 * 0.2).abs() < 1e-12, "{out:?}");
        // One needed hit: 1 − (1 − p)^rounds.
        let out = chain(&eval(1, 5.0, Route::ValuePungs), 3, 10);
        assert!((out.p_win - (1.0 - 0.5f64.powi(3))).abs() < 1e-12);
        // No helpers, no win; no rounds, no win.
        assert_eq!(chain(&eval(1, 0.0, Route::ValuePungs), 5, 10).p_win, 0.0);
        assert_eq!(chain(&eval(1, 5.0, Route::ValuePungs), 0, 10).p_win, 0.0);
    }

    #[test]
    fn chain_measured_uses_the_means_capped_at_the_previous_step() {
        let mut m = HelperMeans::default();
        m.set(Route::ValuePungs, 2, 6.0);
        m.set(Route::ValuePungs, 1, 8.0);
        // needed 3: helpers 10, then the mean at 2 to go (6), then the mean at
        // 1 to go (8) capped at 6. Three hits in three rounds: 0.1 * 0.06 * 0.06.
        let out = chain_measured(&eval(3, 10.0, Route::ValuePungs), 3, 100, &m);
        assert!((out.p_win - 0.1 * 0.06 * 0.06).abs() < 1e-12, "{out:?}");
        // Without a mean the decay stands in: needed 2, 4 then max(2, 3) = 3.
        let e = eval(2, 4.0, Route::AllChows);
        let empty = HelperMeans::default();
        assert_eq!(chain_measured(&e, 3, 10, &empty), chain(&e, 3, 10));
        let text = "{\"route_index\": 7, \"route\": \"value_pungs\", \"needed\": 2, \"n\": 5, \"helpers\": 6.5}";
        assert_eq!(
            HelperMeans::parse(text).unwrap().get(Route::ValuePungs, 2),
            Some(6.5)
        );
    }

    #[test]
    fn keys_round_trip_and_bands() {
        for i in 0..ROUTE_CELLS {
            assert_eq!(RouteKey::from_cell(i).cell(), i);
        }
        let bands: Vec<usize> = [
            0.0, 3.9, 4.0, 7.5, 8.0, 12.9, 13.0, 19.9, 20.0, 29.9, 30.0, 99.0,
        ]
        .iter()
        .map(|&h| helper_band(h))
        .collect();
        assert_eq!(bands, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    }

    #[test]
    fn table_text_round_trips() {
        let text = "{\"entries\": [\n\
            {\"cell\": 17, \"route\": \"all_pungs\", \"n\": 600, \"p_win\": 0.25, \"points\": 16.5, \"rounds\": 4.5},\n\
            {\"cell\": 18, \"route\": \"all_pungs\", \"n\": 10, \"p_win\": 0.1, \"points\": 8, \"rounds\": 3}\n\
            ]}";
        let t = RouteTable::parse(text).unwrap();
        assert_eq!(t.entries.len(), 2);
        assert_eq!(
            t.entries[&17],
            TableEntry {
                n: 600.0,
                p_win: 0.25,
                points: 16.5,
                rounds: 4.5
            }
        );
    }
}
