//! The three-stage bot `stage` (plan agreed with Brian, 2026-11-16). No table.
//!
//! Four lines of play: half flush (with full flush and its all-pungs forms),
//! all pungs, thirteen orphans, all chows. The line with the lowest distance
//! leads; ties go in that order; all chows leads only when it is 4 tiles closer
//! than every other line.
//!
//! * **Stage 1**: the hand holds tiles outside the leading line. Throw the tile
//!   least relevant to the lines, in their order. For a half flush the two best
//!   suits are both kept while the third suit still has tiles.
//! * **Stage 2**: every tile fits. Order: distance, faan, helping tiles,
//!   connection, copies already visible (more: thrown first), random.
//! * **Stage 3**: the hand is ready or one tile from ready (any legal win; two from ready with `ready: 2`). The
//!   1-step and 2-step count: winning tiles times their points, plus the draws
//!   that lead to a better ready hand, weighted by their chance.
//!
//! Claims in stage 1: always take a win; never kong; pung when all pungs leads,
//! or an honour or a lone pair of a kept suit when the flush leads; chow only in
//! the flush suit, with 8 or more flush tiles and paired honours, and only when
//! the flush is closer afterwards.

use crate::bots::{after_claim, live_winning_tiles, rank_discards, Key as RankKey, Metric};
use crate::ev::after_own_kong;
use crate::route::{helpers_of, Route, RouteTarget};
use crate::view::SeatView;
use mahjong_analysis::{faan_shanten, for_each_route_target, Structure, SuitPlan};
use mahjong_games::Policy;
use mahjong_hk::{Action, State};
use mahjong_tiles::{Suit, Tile, TileCounts};
use rand::Rng;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::HashMap;

/// A line of play, in tie order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Line {
    Flush = 0,
    Pungs = 1,
    Orphans = 2,
    Chows = 3,
}

const LINES: [Line; 4] = [Line::Flush, Line::Pungs, Line::Orphans, Line::Chows];
/// Distance of a line with no feasible target.
const FAR: i8 = 99;
/// All chows leads only when it is this many tiles closer than every other line
/// (Brian chose 4 on 2026-11-16 after the 40,000-deal matches; his first plan said 2; the env var `STAGE_CHOWS_LEAD` overrides it for test matches).
fn chows_lead() -> i8 {
    static LEAD: std::sync::OnceLock<i8> = std::sync::OnceLock::new();
    *LEAD.get_or_init(|| {
        std::env::var("STAGE_CHOWS_LEAD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    })
}
/// Flush tiles plus paired honours a hand needs before it may chow in stage 1.
const CHOW_COUNT: u32 = 8;

/// The best targets of each line for one 13-tile hand (lowest distance, then
/// highest faan; all of them when several tie). The flush is kept per suit.
#[derive(Default)]
struct Goals {
    flush: [Vec<RouteTarget>; 3],
    pungs: Vec<RouteTarget>,
    orphans: Vec<RouteTarget>,
    chows: Vec<RouteTarget>,
}

fn push(list: &mut Vec<RouteTarget>, t: RouteTarget) {
    let key = |t: &RouteTarget| (t.distance, Reverse(t.faan));
    match list.first().map(|b| key(&t).cmp(&key(b))) {
        Some(std::cmp::Ordering::Greater) => return,
        Some(std::cmp::Ordering::Less) => list.clear(),
        _ => {}
    }
    list.push(t);
}

fn dist(list: &[RouteTarget]) -> i8 {
    list.first().map_or(FAR, |t| t.distance)
}

impl Goals {
    fn of(view: &SeatView) -> Goals {
        let mut g = Goals::default();
        for_each_route_target(
            &view.hand_view(),
            &view.rules,
            |plan, structure, pungs, faan, _limit, distance| {
                let t = RouteTarget {
                    plan,
                    structure,
                    pungs: pungs.to_vec(),
                    faan,
                    distance,
                };
                match (plan, structure) {
                    (SuitPlan::HalfFlush(s) | SuitPlan::FullFlush(s), _) => {
                        push(&mut g.flush[s as usize], t)
                    }
                    // All honours needs no suit: it serves whichever suit is kept.
                    (SuitPlan::Honours, _) => {
                        for list in &mut g.flush {
                            push(list, t.clone());
                        }
                    }
                    (SuitPlan::Orphans, _) => push(&mut g.orphans, t),
                    (SuitPlan::Mixed, Structure::AllPungs) => push(&mut g.pungs, t),
                    (SuitPlan::Mixed, Structure::AllChows) => push(&mut g.chows, t),
                    // Mixed hands that live on value pungs: dropped as a line.
                    (SuitPlan::Mixed, Structure::Any) => {}
                }
            },
        );
        g
    }

    /// The best flush targets over the given suits.
    fn flush_in(&self, suits: &[Suit]) -> Vec<RouteTarget> {
        let mut out = Vec::new();
        for &s in suits {
            for t in &self.flush[s as usize] {
                push(&mut out, t.clone());
            }
        }
        out
    }

    fn list(&self, line: Line, suits: &[Suit]) -> Vec<RouteTarget> {
        match line {
            Line::Flush => self.flush_in(suits),
            Line::Pungs => self.pungs.clone(),
            Line::Orphans => self.orphans.clone(),
            Line::Chows => self.chows.clone(),
        }
    }

    /// Distance of every line, the flush over the given suits.
    fn dists(&self, suits: &[Suit]) -> [i8; 4] {
        [
            suits
                .iter()
                .map(|&s| dist(&self.flush[s as usize]))
                .min()
                .unwrap_or(FAR),
            dist(&self.pungs),
            dist(&self.orphans),
            dist(&self.chows),
        ]
    }
}

/// The lines, leader first: lowest distance, ties in the fixed order; all chows
/// first only with its 4-tile lead, otherwise ranked as if 4 tiles further.
fn order(d: [i8; 4]) -> [Line; 4] {
    let mut lines = LINES;
    let rest = d[..3].iter().copied().min().unwrap_or(FAR);
    if d[3] < FAR && d[3].saturating_add(chows_lead()) <= rest {
        lines.rotate_right(1);
        return lines;
    }
    lines.sort_by_key(|&l| {
        let far = if l == Line::Chows { chows_lead() } else { 0 };
        (d[l as usize].saturating_add(far), l as usize)
    });
    lines
}

/// The route whose helper weights a line uses.
fn route_of(line: Line) -> Route {
    match line {
        Line::Flush => Route::HalfFlush,
        Line::Pungs => Route::AllPungs,
        Line::Orphans => Route::Limit,
        Line::Chows => Route::AllChows,
    }
}

/// Pairs and neighbours of `t` in the hand (the value bot's measure).
fn connection(hand: &TileCounts, t: Tile) -> i32 {
    let mut n = hand[t] as i32 - 1;
    for d in [-2i8, -1, 1, 2] {
        if let Some(u) = t.offset(d) {
            n += hand[u] as i32 * if d.abs() == 1 { 2 } else { 1 };
        }
    }
    n
}

/// An edge pair (Brian, 2026-11-16: "lone-pair is é‚Šå¼µ, 1 or 9, 2 or 8 when 1 or 9
/// absent respectively"): a pair of 1s or 9s, or of 2s with no 1 in the hand, or of 8s
/// with no 9 in the hand. "Absent" is read as absent from the own hand (my reading).
fn edge_pair(hand: &TileCounts, t: Tile) -> bool {
    let without = |d: i8| t.offset(d).is_none_or(|u| hand[u] == 0);
    match t.number() {
        Some(1 | 9) => true,
        Some(2) => without(-1),
        Some(8) => without(1),
        _ => false,
    }
}

/// Version 1 of the lone pair: no other tile of the hand within two steps.
fn lone_v1(hand: &TileCounts, t: Tile) -> bool {
    [-2i8, -1, 1, 2]
        .iter()
        .all(|&d| t.offset(d).is_none_or(|u| hand[u] == 0))
}

/// Pairs and triplets of the concealed hand among `main`-suit tiles and honours.
fn flush_groups(hand: &TileCounts, main: Suit) -> usize {
    hand.kinds()
        .filter(|t| hand[*t] >= 2 && (t.suit() == Some(main) || t.is_honour()))
        .count()
}

/// Copies of `t` the seat sees outside its own concealed hand.
fn visible(view: &SeatView, t: Tile) -> i32 {
    4 - view.unseen[t] as i32 - view.concealed[t] as i32
}

/// The suits a flush could still use, best first: lowest flush distance, then
/// most tiles, then most pairs and neighbours, then fewest copies visible.
fn suit_order(view: &SeatView, flush_dist: [i8; 3]) -> Vec<Suit> {
    let mut all = view.concealed;
    for m in &view.melds {
        m.add_to(&mut all);
    }
    let mut suits: Vec<Suit> = Suit::ALL
        .into_iter()
        .filter(|&s| flush_dist[s as usize] < FAR)
        .collect();
    suits.sort_by_key(|&s| {
        let of_suit = |t: &Tile| t.suit() == Some(s);
        let tiles: i32 = all.kinds().filter(of_suit).map(|t| all[t] as i32).sum();
        let conn: i32 = view
            .concealed
            .kinds()
            .filter(of_suit)
            .map(|t| connection(&view.concealed, t))
            .sum();
        let seen: i32 = Tile::all_kinds()
            .filter(of_suit)
            .map(|t| visible(view, t))
            .sum();
        (flush_dist[s as usize], -tiles, -conn, seen, s as usize)
    });
    suits
}

/// The suits the flush keeps: the two best while a third suit still has tiles in
/// the hand, else the best one.
fn kept_suits(view: &SeatView, order: &[Suit]) -> Vec<Suit> {
    let held = |s: Suit| view.concealed.kinds().any(|t| t.suit() == Some(s));
    let others = Suit::ALL
        .into_iter()
        .filter(|s| !order.iter().take(2).any(|k| k == s))
        .any(held);
    let keep = if others && order.len() >= 2 && held(order[1]) {
        2
    } else {
        1
    };
    order.iter().copied().take(keep).collect()
}

/// Whether `t` fits a flush in `suits`.
fn fits(t: Tile, suits: &[Suit]) -> bool {
    t.suit().is_none_or(|s| suits.contains(&s))
}

fn discard(view: &SeatView, t: Tile) -> SeatView {
    let mut h = view.concealed;
    h.remove(t);
    view.with(h, view.melds.clone())
}

fn faan_dist(view: &SeatView) -> i8 {
    faan_shanten(&view.hand_view(), &view.rules).shanten
}

/// Winning tiles times their points, for a ready hand (unseen copies only).
fn outs(view: &SeatView) -> f64 {
    live_winning_tiles(view, &view.concealed)
        .iter()
        .map(|&(w, faan)| view.unseen[w] as f64 * view.rules.points(faan) as f64)
        .sum()
}

/// Values at one from ready already worked out in one decision, by (concealed
/// hand, unseen tiles). Valid only while the melds and rules stay the same.
pub type Memo = RefCell<HashMap<(TileCounts, TileCounts), f64>>;

/// Stage 3 value of a hand without a 14th tile. `reach` is the distance at which
/// stage 3 starts (1 or 2). At two from ready (only when `reach` is 2) the count
/// goes one step further: every draw that brings the hand to one from ready,
/// weighted by its chance, times the best one-from-ready value after a discard
/// that keeps it there. That reading of "stage 3 from two from ready" is the
/// builder's, not Brian's (Brian said only "2 from ready", 2026-11-16).
/// Every other hand gets [`near_value`] (unchanged from the one-from-ready bot).
pub fn stage3_value(view: &SeatView, reach: i8, memo: &Memo) -> f64 {
    if reach < 2 || faan_dist(view) != 2 {
        return near_value(view);
    }
    let total = view.unseen.total().max(1) as f64;
    let mut value = 0.0;
    for t in Tile::all_kinds() {
        let n = view.unseen[t];
        if n == 0 || view.concealed[t] >= 4 {
            continue;
        }
        let mut drawn = view.clone();
        drawn.concealed.add(t);
        drawn.unseen.remove(t);
        // Prune: only draws that bring the hand to one from ready.
        if faan_dist(&drawn) != 1 {
            continue;
        }
        let best = drawn
            .concealed
            .kinds()
            .filter(|&d| d != t)
            .map(|d| discard(&drawn, d))
            // Prune: only discards that keep it one from ready.
            .filter(|v| faan_dist(v) == 1)
            .map(|v| {
                let key = (v.concealed, v.unseen);
                if let Some(&x) = memo.borrow().get(&key) {
                    return x;
                }
                let x = near_value(&v);
                memo.borrow_mut().insert(key, x);
                x
            })
            .fold(0.0, f64::max);
        value += n as f64 * best / total;
    }
    value
}

/// The 1-step and 2-step count of a hand without a 14th tile.
/// Ready: its outs, plus every other draw that leads to a better ready hand,
/// weighted by the chance of that draw. One from ready: the draws that make it
/// ready, each weighted the same way. Further away: 0.
fn near_value(view: &SeatView) -> f64 {
    let s = faan_dist(view);
    if !(0..=1).contains(&s) {
        return 0.0;
    }
    let base = if s == 0 { outs(view) } else { 0.0 };
    let total = view.unseen.total().max(1) as f64;
    let mut value = base;
    for t in Tile::all_kinds() {
        let n = view.unseen[t];
        if n == 0 || view.concealed[t] >= 4 {
            continue;
        }
        let mut drawn = view.clone();
        drawn.concealed.add(t);
        drawn.unseen.remove(t);
        // Complete (already counted as an out), or not ready after a discard: skip.
        if faan_dist(&drawn) != 0 {
            continue;
        }
        let best = drawn
            .concealed
            .kinds()
            .filter(|&d| d != t)
            .map(|d| discard(&drawn, d))
            .filter(|v| faan_dist(v) == 0)
            .map(|v| outs(&v))
            .fold(0.0, f64::max);
        if best > base {
            value += n as f64 * (best - base) / total;
        }
    }
    value
}

/// What the bot decided and why (for logs and the position checker).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Note {
    /// 1, 2 or 3; 0 when no line is feasible (value-bot order).
    pub stage: u8,
    pub line: Option<Line>,
}

/// The three-stage bot. The number is the version of the stage 1 pung rule for a
/// suited tile while the half flush leads:
/// 1 = a pair with no tile of the hand within two steps (my first reading);
/// 2 = an edge pair (Brian's correction, 2026-11-16);
/// 3 = an edge pair, or a tile of the flush suit when the hand holds three pairs or
///     triplets of flush-suit tiles and honours (Brian, 2026-11-16: "there should be 3
///     pair/trip ready for the half-suit"). The claimed pair counts as one of the three
///     (my reading).
///
/// `ready` is the faan distance at which stage 3 starts: 1 = one from ready (the
/// first build; default), 2 = two from ready (Brian, 2026-11-16: "I mean 2 from
/// ready, not 2 from winning"; the scoring there is the builder's reading, see
/// [`stage3_value`]).
///
/// `gate` decides whether a hand within `ready` may enter stage 3 while the flush
/// leads and the hand still holds a tile outside the kept suits (the `off` check
/// of stage 2):
/// 0 = it may (the first build, measured at +2.54 against three value bots; bot
///     `stage-gate0`);
/// 1 = it may not, even when ready: it plays stage 1 (Brian, 2026-11-16: "make
///     phase 3 the extra requirement that is found in phase 2 as well, no
///     irrelevant tile for top priority route"; and "use strict version as
///     default": the default, set by [`StageBot::new`]);
/// 2 = like 1, but a ready hand stays in stage 3 (the watch session's variant, for
///     comparison; not Brian's words).
///
/// Leads other than the flush have no outside tile, so stage 3 starts there as
/// with gate 0 (the watch session's reading).
#[derive(Clone, Copy, Debug)]
pub struct StageBot {
    pub pung: u8,
    pub ready: i8,
    pub gate: u8,
}

impl Default for StageBot {
    fn default() -> StageBot {
        StageBot::new(3, 1)
    }
}

impl StageBot {
    /// A bot with the strict stage 3 gate (gate 1).
    pub const fn new(pung: u8, ready: i8) -> StageBot {
        StageBot {
            pung,
            ready,
            gate: 1,
        }
    }

    /// The same bot with another stage 3 gate (0, 1 or 2).
    pub const fn with_gate(self, gate: u8) -> StageBot {
        StageBot { gate, ..self }
    }

    /// Whether the gate lets a hand with an outside tile into stage 3 anyway:
    /// always with gate 0, when ready (`dist` 0) with gate 2, never with gate 1.
    fn gate_open(&self, dist: i8) -> bool {
        match self.gate {
            0 => true,
            2 => dist == 0,
            _ => false,
        }
    }

    /// The discard from a 14-tile view among `candidates`.
    pub fn pick_discard<R: Rng + ?Sized>(
        &self,
        view: &SeatView,
        candidates: &[Tile],
        rng: &mut R,
    ) -> (Tile, Note) {
        let ranked = rank_discards(view, candidates, Metric::Faan);
        // Stage 3: within `ready` steps of ready after the best discard (and,
        // with a gate, no tile outside the flush; checked below).
        let near = ranked[0].1 .0 <= self.ready;
        if near && self.gate == 0 {
            return self.stage3_discard(view, &ranked);
        }
        let after: Vec<(Tile, SeatView, Goals)> = candidates
            .iter()
            .map(|&t| {
                let v = discard(view, t);
                let g = Goals::of(&v);
                (t, v, g)
            })
            .collect();
        // The 14-tile hand's distance on a line is its best over the discards.
        let mut flush_dist = [FAR; 3];
        for (_, _, g) in &after {
            for (best, list) in flush_dist.iter_mut().zip(&g.flush) {
                *best = (*best).min(dist(list));
            }
        }
        let suits = suit_order(view, flush_dist);
        let best_of = |line: Line| {
            after
                .iter()
                .map(|(_, _, g)| g.dists(&suits)[line as usize])
                .min()
                .unwrap_or(FAR)
        };
        let lines = order(LINES.map(best_of));
        let lead = lines[0];
        let feasible = best_of(lead) < FAR;
        let kept = if lead == Line::Flush {
            kept_suits(view, &suits)
        } else {
            suits.clone()
        };
        let off: Vec<Tile> = candidates
            .iter()
            .copied()
            .filter(|&t| feasible && lead == Line::Flush && !fits(t, &kept))
            .collect();
        if near && (off.is_empty() || self.gate_open(ranked[0].1 .0)) {
            return self.stage3_discard(view, &ranked);
        }
        if !feasible {
            // No line is feasible: play like the value bot.
            let note = Note {
                stage: 0,
                line: None,
            };
            return (ranked[0].0, note);
        }
        self.line_discard(view, &after, &suits, &lines, &kept, &off, candidates, rng)
    }

    /// Stage 3 discard: the best 1-step and 2-step count among the discards
    /// within `ready` (`ranked` from [`rank_discards`], best first).
    fn stage3_discard(&self, view: &SeatView, ranked: &[(Tile, RankKey)]) -> (Tile, Note) {
        let memo = Memo::default();
        let best = ranked
            .iter()
            .enumerate()
            .filter(|(_, (_, k))| k.0 <= self.ready)
            .map(|(i, &(t, _))| {
                (
                    stage3_value(&discard(view, t), self.ready, &memo),
                    Reverse(i),
                    t,
                )
            })
            .max_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
            .expect("a candidate");
        let note = Note {
            stage: 3,
            line: None,
        };
        (best.2, note)
    }

    /// Stage 1 or 2 discard on the leading line `lines[0]`.
    #[allow(clippy::too_many_arguments)]
    fn line_discard<R: Rng + ?Sized>(
        &self,
        view: &SeatView,
        after: &[(Tile, SeatView, Goals)],
        suits: &[Suit],
        lines: &[Line; 4],
        kept: &[Suit],
        off: &[Tile],
        candidates: &[Tile],
        rng: &mut R,
    ) -> (Tile, Note) {
        let lead = lines[0];
        // Stage 1: tiles outside the flush go first. Every other line holds
        // spare tiles until it is ready, so it stays in stage 1.
        let stage1 = lead != Line::Flush || !off.is_empty();
        let pool = if off.is_empty() { candidates } else { off };
        // (distance, -faan, -helpers, other lines' distances, connection,
        // -visible, random): lowest wins.
        type Key = (i8, i32, i32, [i8; 3], i32, i32, u32);
        let mut keyed: Vec<(Tile, Key)> = after
            .iter()
            .filter(|(t, _, _)| pool.contains(t))
            .map(|(t, _, g)| {
                let targets = g.list(lead, kept);
                let faan = targets.first().map_or(0, |x| x.faan as i32);
                let all = g.dists(suits);
                let mut others = [0i8; 3];
                if stage1 {
                    for (o, l) in others.iter_mut().zip(&lines[1..]) {
                        *o = all[*l as usize];
                    }
                }
                let key = (
                    dist(&targets),
                    -faan,
                    0,
                    others,
                    connection(&view.concealed, *t),
                    -visible(view, *t),
                    rng.gen::<u32>(),
                );
                (*t, key)
            })
            .collect();
        // Helping tiles only for the discards tied on distance and faan.
        let top = keyed.iter().map(|k| (k.1 .0, k.1 .1)).min().expect("pool");
        for (t, key) in keyed.iter_mut().filter(|k| (k.1 .0, k.1 .1) == top) {
            let (_, v, g) = after.iter().find(|a| a.0 == *t).expect("candidate");
            let helpers = g
                .list(lead, kept)
                .iter()
                .map(|x| helpers_of(&v.hand_view(), route_of(lead), x).0)
                .fold(0f32, f32::max);
            key.2 = -(helpers * 100.0) as i32;
        }
        keyed.sort_by_key(|a| a.1);
        let note = Note {
            stage: if stage1 { 1 } else { 2 },
            line: Some(lead),
        };
        (keyed[0].0, note)
    }

    /// The answer to a discard `tile` from seat `from` (a 13-tile view), among
    /// `legal`. A legal win is taken before this is called.
    pub fn claim(&self, view: &SeatView, tile: Tile, from: usize, legal: &[Action]) -> Action {
        let melds = |a: &&Action| matches!(a, Action::Pung | Action::ClaimKong | Action::Chow(_));
        // The best of `f` over the discards of a 14-tile view.
        let best_discard = |v: &SeatView, f: &dyn Fn(&SeatView) -> f64| {
            v.concealed
                .kinds()
                .map(|d| f(&discard(v, d)))
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let goals = Goals::of(view);
        let flush_dist = [0, 1, 2].map(|s| dist(&goals.flush[s]));
        let suits = suit_order(view, flush_dist);
        let d = goals.dists(&suits);
        let lead = order(d)[0];
        let feasible = d[lead as usize] < FAR;
        let kept = if feasible && lead == Line::Flush {
            kept_suits(view, &suits)
        } else {
            Vec::new()
        };
        // A tile outside the leading flush (the stage 2 check below).
        let outside =
            feasible && lead == Line::Flush && !view.concealed.kinds().all(|t| fits(t, &kept));
        // Stage 3: claim when the 1-step and 2-step count is higher afterwards.
        let now = faan_dist(view);
        if now <= self.ready && (!outside || self.gate_open(now)) {
            let mut best = (
                stage3_value(view, self.ready, &Memo::default()),
                Action::Pass,
            );
            for &a in legal.iter().filter(melds) {
                let Some(after) = after_claim(view, a, tile, from) else {
                    continue;
                };
                // One memo per claim: the melds differ between claims.
                let memo = Memo::default();
                let worth = if a == Action::ClaimKong {
                    stage3_value(&after, self.ready, &memo)
                } else {
                    best_discard(&after, &|v| stage3_value(v, self.ready, &memo))
                };
                if worth > best.0 {
                    best = (worth, a);
                }
            }
            return best.1;
        }
        if !feasible {
            return Action::Pass;
        }
        let pung = legal.contains(&Action::Pung);
        match lead {
            // A meld ends both of these lines.
            Line::Chows | Line::Orphans => Action::Pass,
            Line::Pungs if pung => Action::Pung,
            Line::Pungs => Action::Pass,
            Line::Flush => {
                let main = suits[0];
                // Minus the flush distance in the main suit (higher is better).
                let closeness = |v: &SeatView| -(dist(&Goals::of(v).flush[main as usize]) as f64);
                let now = closeness(view);
                if !outside {
                    // Stage 2: claim when the flush is closer afterwards.
                    let mut best = (now, Action::Pass);
                    for &a in legal.iter().filter(melds) {
                        let Some(after) = after_claim(view, a, tile, from) else {
                            continue;
                        };
                        let then = if a == Action::ClaimKong {
                            closeness(&after)
                        } else {
                            best_discard(&after, &closeness)
                        };
                        let kong_over_pung =
                            then == best.0 && a == Action::ClaimKong && best.1 == Action::Pung;
                        if then > best.0 || kong_over_pung {
                            best = (then, a);
                        }
                    }
                    return best.1;
                }
                // Stage 1.
                let hand = &view.concealed;
                let lone = match self.pung {
                    1 => lone_v1(hand, tile),
                    _ => edge_pair(hand, tile),
                };
                let three =
                    self.pung >= 3 && tile.suit() == Some(main) && flush_groups(hand, main) >= 3;
                if pung && (tile.is_honour() || fits(tile, &kept) && lone || three) {
                    return Action::Pung;
                }
                if tile.suit() != Some(main) {
                    return Action::Pass;
                }
                let mut all = view.concealed;
                for m in &view.melds {
                    m.add_to(&mut all);
                }
                let count: u32 = all
                    .kinds()
                    .filter(|t| t.suit() == Some(main) || t.is_honour() && all[*t] >= 2)
                    .map(|t| all[t] as u32)
                    .sum();
                if count < CHOW_COUNT {
                    return Action::Pass;
                }
                let mut best = (now, Action::Pass);
                for &a in legal.iter().filter(|a| matches!(a, Action::Chow(_))) {
                    if let Some(after) = after_claim(view, a, tile, from) {
                        let then = best_discard(&after, &closeness);
                        if then > best.0 {
                            best = (then, a);
                        }
                    }
                }
                best.1
            }
        }
    }

    /// The action and the stage it came from.
    pub fn decide<R: Rng + ?Sized>(
        &self,
        state: &State,
        legal: &[Action],
        rng: &mut R,
    ) -> (Action, Option<Note>) {
        if let Some(&w) = legal
            .iter()
            .find(|a| matches!(a, Action::Win | Action::SelfDrawnWin))
        {
            return (w, None);
        }
        let view = SeatView::of_current(state);
        if state.in_claim() || state.in_rob() {
            let Some((tile, from)) = state.claim_tile() else {
                return (Action::Pass, None);
            };
            return (self.claim(&view, tile, from, legal), None);
        }
        let candidates: Vec<Tile> = legal
            .iter()
            .filter_map(|a| match a {
                Action::Discard(t) => Some(*t),
                _ => None,
            })
            .collect();
        let (tile, note) = self.pick_discard(&view, &candidates, rng);
        // Kongs: never in stage 1; later, when the hand is no further from a win
        // afterwards (the value bot's rule).
        if note.stage >= 2 {
            let now = faan_dist(&discard(&view, tile));
            for &a in legal {
                let kong = matches!(a, Action::ConcealedKong(_) | Action::AddedKong(_));
                if kong && after_own_kong(&view, a).is_some_and(|v| faan_dist(&v) <= now) {
                    return (a, Some(note));
                }
            }
        }
        (Action::Discard(tile), Some(note))
    }
}

impl Policy<State> for StageBot {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        self.decide(state, legal, rng).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahjong_hk::RuleSet;
    use mahjong_tiles::parse_tiles;
    use rand::SeedableRng;

    fn view(hand: &str) -> SeatView {
        let concealed = TileCounts::from_tiles(&parse_tiles(hand).unwrap());
        let mut unseen = TileCounts::full_set();
        for t in concealed.tiles() {
            unseen.remove(t);
        }
        SeatView {
            seat: 1,
            concealed,
            melds: Vec::new(),
            bonus: 0,
            unseen,
            rules: RuleSet {
                flowers: false,
                ..RuleSet::default()
            },
            river: Vec::new(),
        }
    }

    fn t(s: &str) -> Tile {
        parse_tiles(s).unwrap()[0]
    }

    fn pick(hand: &str) -> (Tile, Note) {
        let v = view(hand);
        let c: Vec<Tile> = v.concealed.kinds().collect();
        StageBot::default().with_gate(0).pick_discard(
            &v,
            &c,
            &mut rand::rngs::StdRng::seed_from_u64(1),
        )
    }

    #[test]
    fn all_chows_needs_a_four_tile_lead() {
        assert_eq!(order([5, 5, 9, 2])[0], Line::Flush);
        assert_eq!(order([5, 5, 9, 1])[0], Line::Chows);
        assert_eq!(order([5, 4, 9, 1])[0], Line::Pungs);
        assert_eq!(order([FAR, FAR, FAR, 6])[0], Line::Chows);
    }

    #[test]
    fn flush_throws_the_third_suit_and_keeps_honours() {
        // Brian's hand (2026-11-16): 8p goes before West.
        let (tile, note) = pick("678m8p12366sWNWhWhG");
        assert_eq!(note.line, Some(Line::Flush));
        assert_eq!(note.stage, 1);
        assert_eq!(tile, t("8p"));
    }

    #[test]
    fn pairs_lead_to_all_pungs() {
        let (tile, note) = pick("22m55p77s999s3m6pEW1s");
        assert_eq!(note.line, Some(Line::Pungs));
        assert!(!["2m", "5p", "7s", "9s"].contains(&tile.to_string().as_str()));
    }

    #[test]
    fn stage_one_claims_follow_the_rules() {
        // Bamboos lead; 22s stands alone, 778s is a chow shape.
        let v = view("22s7789sEEN2m5m9m3p");
        assert_eq!(v.concealed.total(), 13);
        let pung = [Action::Pass, Action::Pung];
        // 77s is no edge pair (version 2 passes); 22s, 77s and EE are three pairs (version 3 claims).
        assert_eq!(
            StageBot::new(2, 1)
                .with_gate(0)
                .claim(&v, t("7s"), 0, &pung),
            Action::Pass
        );
        assert_eq!(
            StageBot::new(3, 1)
                .with_gate(0)
                .claim(&v, t("7s"), 0, &pung),
            Action::Pung
        );
        assert_eq!(
            StageBot::default()
                .with_gate(0)
                .claim(&v, t("2s"), 0, &pung),
            Action::Pung
        );
        assert_eq!(
            StageBot::default().with_gate(0).claim(&v, t("E"), 0, &pung),
            Action::Pung
        );
        // A lone pair in the second kept suit is claimed; one in the third suit is not.
        assert_eq!(
            StageBot::default()
                .with_gate(0)
                .claim(&view("22s7789sEEN22m9m3p"), t("2m"), 0, &pung),
            Action::Pung
        );
        let v = view("22s789sEE2345m33p");
        assert_eq!(
            StageBot::default()
                .with_gate(0)
                .claim(&v, t("3p"), 0, &pung),
            Action::Pass
        );
        // Edge pairs only: 44s is no edge pair; 22s with a 1s is none; 99s always is.
        let three = view("44s99sEEN2m5m9m3p7p1p");
        assert_eq!(
            StageBot::new(2, 1)
                .with_gate(0)
                .claim(&three, t("4s"), 0, &pung),
            Action::Pass
        );
        // Version 3: 44s, 99s and EE are three pairs for the flush, so 4s is claimed.
        assert_eq!(
            StageBot::new(3, 1)
                .with_gate(0)
                .claim(&three, t("4s"), 0, &pung),
            Action::Pung
        );
        assert_eq!(
            StageBot::new(3, 1).with_gate(0).claim(
                &view("44s9sEENW2m5m9m3p7p1p"),
                t("4s"),
                0,
                &pung
            ),
            Action::Pass
        );
        assert_eq!(
            StageBot::default().with_gate(0).claim(
                &view("44s99sEEN2m5m9m3p7p1p"),
                t("9s"),
                0,
                &pung
            ),
            Action::Pung
        );
        assert_eq!(
            StageBot::new(2, 1).with_gate(0).claim(
                &view("122s99sEEN2m5m9m3p7p"),
                t("2s"),
                0,
                &pung
            ),
            Action::Pass
        );
    }

    #[test]
    fn chow_needs_eight_flush_tiles_and_a_closer_flush() {
        // 1357s with East and West pairs: 4 bamboos + 4 paired honours = 8.
        let v = view("1357sEEWW2m5m9m3p8p");
        let take = |tile: &str, lows: &[&str]| {
            let mut legal = vec![Action::Pass];
            legal.extend(lows.iter().map(|l| Action::Chow(t(l))));
            StageBot::default()
                .with_gate(0)
                .claim(&v, t(tile), 0, &legal)
        };
        // 4s: the chow 345s leaves 1s and 7s alone, so pass.
        assert_eq!(take("4s", &["3s"]), Action::Pass);
        // 6s: the chow 567s leaves 13s waiting for 2s.
        assert_eq!(take("6s", &["5s"]), Action::Chow(t("5s")));
        // One honour pair fewer: the count is 6, so pass.
        let v = view("1357sEEWN2m5m9m3p8p");
        let legal = [Action::Pass, Action::Chow(t("5s"))];
        assert_eq!(
            StageBot::default()
                .with_gate(0)
                .claim(&v, t("6s"), 0, &legal),
            Action::Pass
        );
    }

    #[test]
    fn ready_hand_counts_outs_times_points() {
        // Full flush in bamboos waiting on 3s and 6s (7 faan, 32 points... per rules).
        let v = view("1112345678999s");
        assert_eq!(v.concealed.total(), 13);
        assert!(stage3_value(&v, 1, &Memo::default()) > 0.0);
        let far = view("147m258p369sESWN");
        assert_eq!(stage3_value(&far, 1, &Memo::default()), 0.0);
    }

    /// Half flush in bamboos with 5p and North still to go: two from ready.
    const TWO_AWAY: &str = "1234567sEEES5pN";

    #[test]
    fn ready_switch_scores_two_from_ready() {
        let v = view(TWO_AWAY);
        assert_eq!(faan_dist(&v), 2);
        // One from ready (today): 0. Two from ready: a real number.
        assert_eq!(stage3_value(&v, 1, &Memo::default()), 0.0);
        let two = stage3_value(&v, 2, &Memo::default());
        assert!(two > 0.0, "{two}");
        // The memo gives the same number on a second use.
        let memo = Memo::default();
        assert_eq!(stage3_value(&v, 2, &memo), two);
        assert!(!memo.borrow().is_empty());
        assert_eq!(stage3_value(&v, 2, &memo), two);
        // Ready and one-from-ready hands score the same under both switches.
        for hand in ["1112345678999s", "1234567sEEESS5p"] {
            let v = view(hand);
            let one = stage3_value(&v, 1, &Memo::default());
            assert_eq!(stage3_value(&v, 2, &Memo::default()), one, "{hand}");
        }
        // Three or more from ready: 0 under both.
        assert_eq!(
            stage3_value(&view("147m258p369sESWN"), 2, &Memo::default()),
            0.0
        );
    }

    #[test]
    fn ready_switch_moves_the_start_of_stage_three() {
        let v = view(&format!("{TWO_AWAY}9m"));
        let c: Vec<Tile> = v.concealed.kinds().collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let (_, note) = StageBot::new(3, 1)
            .with_gate(0)
            .pick_discard(&v, &c, &mut rng);
        assert_ne!(note.stage, 3);
        let (tile, note) = StageBot::new(3, 2)
            .with_gate(0)
            .pick_discard(&v, &c, &mut rng);
        assert_eq!(note.stage, 3);
        assert!(tile.suit() != Some(Suit::Bamboo), "{tile}");
    }

    fn pick_with(bot: StageBot, hand: &str) -> (Tile, Note) {
        let v = view(hand);
        let c: Vec<Tile> = v.concealed.kinds().collect();
        bot.pick_discard(&v, &c, &mut rand::rngs::StdRng::seed_from_u64(1))
    }

    /// Brian's example (2026-11-16): one from ready while it still holds 1m and 5p.
    const OUTSIDE: &str = "11123456789s1m5p";

    #[test]
    fn gate_one_keeps_outside_tiles_out_of_stage_three() {
        assert_eq!(faan_dist(&view(OUTSIDE)), 1);
        // The example hand after drawing 9m: today it is in stage 3.
        let hand = format!("{OUTSIDE}9m");
        let (_, note) = pick_with(StageBot::new(3, 1).with_gate(0), &hand);
        assert_eq!(note.stage, 3);
        // Gate 1 (and gate 2, as it is not ready): stage 1, an outside tile goes.
        for gate in [1, 2] {
            let (tile, note) = pick_with(StageBot::new(3, 1).with_gate(gate), &hand);
            assert_eq!(
                (note.stage, note.line),
                (1, Some(Line::Flush)),
                "gate {gate}"
            );
            assert!(
                ["1m", "5p", "9m"].contains(&tile.to_string().as_str()),
                "{tile}"
            );
        }
    }

    #[test]
    fn gate_two_lets_a_ready_hand_stay_in_stage_three() {
        // Nine gates, ready, with a drawn 5p as its one outside tile.
        let hand = "1112345678999s5p";
        assert_eq!(faan_dist(&discard(&view(hand), t("5p"))), 0);
        let (_, note) = pick_with(StageBot::new(3, 1).with_gate(0), hand);
        assert_eq!(note.stage, 3);
        let (tile, note) = pick_with(StageBot::new(3, 1).with_gate(1), hand);
        assert_eq!((note.stage, tile), (1, t("5p")));
        let (tile, note) = pick_with(StageBot::new(3, 1).with_gate(2), hand);
        assert_eq!((note.stage, tile), (3, t("5p")));
        // A hand with every tile inside the flush enters stage 3 under every gate.
        for gate in 0..3 {
            let (_, note) = pick_with(StageBot::new(3, 1).with_gate(gate), "1112345678999s9s");
            assert_eq!(note.stage, 3, "gate {gate}");
        }
    }

    #[test]
    fn gate_applies_to_claims() {
        // The example hand (13 tiles, one from ready) is offered a 4s. Today stage 3
        // chows 456s to be ready; under gates 1 and 2 the hand plays stage 1, where
        // the chow does not bring the flush closer (1m and 5p are still to go).
        let v = view(OUTSIDE);
        let legal = [
            Action::Pass,
            Action::Chow(t("2s")),
            Action::Chow(t("3s")),
            Action::Chow(t("4s")),
        ];
        let claim = |gate| {
            StageBot::new(3, 1)
                .with_gate(gate)
                .claim(&v, t("4s"), 0, &legal)
        };
        assert_eq!(claim(0), Action::Chow(t("4s")));
        assert_eq!(claim(1), Action::Pass);
        assert_eq!(claim(2), Action::Pass);
        // A ready hand with no outside tile claims as today under every gate.
        let ready = view("1112345678999s");
        let pung = [Action::Pass, Action::Pung];
        for gate in 0..3 {
            assert_eq!(
                StageBot::new(3, 1)
                    .with_gate(gate)
                    .claim(&ready, t("9s"), 0, &pung),
                StageBot::new(3, 1)
                    .with_gate(0)
                    .claim(&ready, t("9s"), 0, &pung),
                "gate {gate}"
            );
        }
    }

    #[test]
    fn bot_names_set_both_switches() {
        let of = |name: &str| match crate::Bot::by_name(name) {
            Some(crate::Bot::Stage(b)) => (b.pung, b.ready, b.gate),
            _ => panic!("{name} is no three-stage bot"),
        };
        assert_eq!(of("stage"), (3, 1, 1));
        assert_eq!(of("stage-gate0"), (3, 1, 0));
        assert_eq!(of("stage1"), (1, 1, 0));
        assert_eq!(of("stage2"), (2, 1, 0));
        assert_eq!(of("stage-orig"), (1, 1, 0));
        assert_eq!(of("stage-pung"), (3, 1, 0));
        assert_eq!(of("stage-ready2"), (1, 2, 0));
        assert_eq!(of("stage-both"), (3, 2, 0));
        assert_eq!(of("stage-gate1"), (3, 1, 1));
        assert_eq!(of("stage-gate2"), (3, 1, 2));
    }

    #[test]
    fn show_line_distances() {
        for hand in ["8m1357p1123sSSWWhG", "35667p1336677sEW"] {
            let v = view(hand);
            let mut best = [FAR; 4];
            for d in v.concealed.kinds() {
                let g = Goals::of(&discard(&v, d));
                let all = g.dists(&Suit::ALL);
                for i in 0..4 {
                    best[i] = best[i].min(all[i]);
                }
            }
            println!(
                "{hand}: flush {} pungs {} orphans {} chows {}",
                best[0], best[1], best[2], best[3]
            );
        }
    }
}
