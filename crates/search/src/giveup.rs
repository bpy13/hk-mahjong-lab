//! A give-up rule for the three-stage bot's ready hand (plan
//! `docs/plans/2026-11-16-self-play-network.md`, Task 11; Brian, 2026-11-19: "if we
//! equip this to the stage bot, what happens? can you try").
//!
//! When the bot is in stage 3 and its chosen discard `t` leaves it ready, it
//! pushes when
//!
//! ```text
//! (1 - d) × V - loss(t) > fold
//! ```
//!
//! and otherwise throws `s`, the tile with the smallest expected loss. `d` is
//! [`Danger::tile`], `loss` is [`Danger::expected_loss`] (both from
//! [`Danger::from_belief`] with [`HeuristicBelief`], the fitted tables), and `fold`
//! is `-loss(s)` (the orchestrator's rough model).
//!
//! `V` is what staying ready is worth over the rest of the hand, in points,
//! summed over the rounds left: (chance I draw a winning tile × 3 shares of the
//! self-drawn points + chance an opponent throws one × 1 share − the chance my
//! later discard deals in × the points it would pay), each round weighted by the
//! chance the hand is still going. The chances per round come from [`Figures`],
//! measured from 40,000 deals of four `stage` bots (`examples/giveup_log.rs`,
//! `docs/study-results/stage3-giveup.md`).
//!
//! Corrections of the parts, not of the shape (the worker's, 2026-11-19):
//! * `d` and `loss` go through [`CALIBRATION`] and [`LOSS_FACTOR`]: on stage 3
//!   discards the raw chance was 1.1 to 3.6 times too low.
//! * The later-discard term counts only how much more often a ready hand's
//!   discard deals in than the fold's smallest-loss tile would: the fold also
//!   discards every round.
//! * The points a later deal-in costs are those of the chosen tile's deal-in
//!   (calibrated loss ÷ calibrated chance).
//! * When `s` also leaves the hand ready, `fold` is `(1 - d(s)) × V(s) -
//!   loss(s)`, the push value of `s`; when no tile is safer than `t`, `s` is `t`.
//! * The self-drawn points carry the 自摸 faan.
//!
//! After a give-up the hand keeps throwing its smallest-loss tile until a discard
//! leaves it ready again; then the rule above decides again. The bot itself has
//! no memory, so the give-up is remembered per thread, keyed by the seat and the
//! public event history up to that decision ([`remember`]).
//!
//! **On another reading** (plan Task 20, `stage-fold:belief=...`): with a belief
//! model set by [`crate::danger::with_reading`], the danger comes from that model
//! and the calibration, loss factor and figures are those refitted on it
//! ([`READ_CALIBRATION`], [`READ_LOSS_FACTOR`], [`READ_MEASURED`]; [`calibration`],
//! [`loss_factor`] and [`measured`] pick the set in force).

use crate::bots::live_winning_tiles;
use crate::danger;
use crate::danger::Danger;
use crate::view::SeatView;
use mahjong_analysis::faan_shanten;
use mahjong_hk::{Event, State};
use mahjong_tiles::Tile;
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Rounds-left buckets of [`Figures`]: 0–2, 3–5, 6–8, 9–11, 12 and more.
pub const BUCKETS: usize = 5;

/// The bucket of `rounds` rounds left.
pub fn bucket(rounds: usize) -> usize {
    (rounds / 3).min(BUCKETS - 1)
}

/// Chances per round for a ready hand, by rounds-left bucket ([`bucket`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Figures {
    /// Chance I win by my own draw = `own × live ÷ unseen`.
    pub own: [f64; BUCKETS],
    /// Chance I win on an opponent's discard = `theirs × live ÷ unseen`.
    pub theirs: [f64; BUCKETS],
    /// Chance another player ends the hand (a win that is not mine).
    pub other_ends: [f64; BUCKETS],
    /// How much more often my discard of that round deals in than the fold's
    /// smallest-loss tile would (measured rate on ready hands minus the
    /// calibrated danger of the safest tile at the same decisions; at least 0).
    /// The fold also discards every round, so only the difference counts against
    /// staying ready (a correction of the rough model, which counted all of it).
    pub later_deal_in: [f64; BUCKETS],
}

/// The orchestrator's rough figures: the draw at face value, opponents throwing
/// my tile at random (3 discards a round), 6 % a round that another player ends
/// the hand, 2 % danger per later discard.
pub const ROUGH: Figures = Figures {
    own: [1.0; BUCKETS],
    theirs: [3.0; BUCKETS],
    other_ends: [0.06; BUCKETS],
    later_deal_in: [0.02; BUCKETS],
};

/// Step 1's measured figures: 40,000 deals of four `stage` bots, seed 2, with
/// the danger calibration fitted on seed 1 (`runs/stage/t11-step1-seed2.txt`,
/// `docs/study-results/stage3-giveup.md`).
pub const MEASURED: Figures = Figures {
    own: [0.2477, 0.4991, 0.6101, 0.7276, 0.8158],
    theirs: [1.1686, 1.8535, 2.201, 2.5333, 2.9106],
    other_ends: [0.1265, 0.1243, 0.0964, 0.0635, 0.0315],
    later_deal_in: [0.0785, 0.064, 0.0429, 0.0234, 0.0091],
};

/// The calibration of [`Danger::tile`] for this rule: (mean predicted, observed
/// deal-in rate) per danger bin of the stage 3 discards in step 1's log (40,000
/// deals of four `stage` bots, seed 1; `runs/stage/t11-step1-pass1.txt`). The raw
/// danger was 1.1 to 3.6 times too low (more than the plan's factor 1.5 in 8 of
/// 10 bins), so it is rescaled. Piecewise linear between the points, proportional
/// below the first, the last ratio above the last (capped at 1).
pub const CALIBRATION: [(f64, f64); 10] = [
    (0.0012, 0.0041),
    (0.0071, 0.0079),
    (0.0148, 0.0198),
    (0.0236, 0.0344),
    (0.0390, 0.0587),
    (0.0595, 0.1048),
    (0.0853, 0.1584),
    (0.1196, 0.2550),
    (0.1807, 0.3725),
    (0.2763, 0.7308),
];

/// The calibration of [`Danger::expected_loss`]: (mean raw danger of the bin,
/// points paid ÷ predicted loss) over the same bins. The expected loss was close
/// (0.87 to 1.24) except in the safest bin (4.2) and the last (1.6, 78 discards):
/// the model's chance is too low but its points per deal-in too high.
pub const LOSS_FACTOR: [(f64, f64); 10] = [
    (0.0012, 4.19),
    (0.0071, 0.957),
    (0.0148, 0.868),
    (0.0236, 1.037),
    (0.0390, 0.963),
    (0.0595, 1.029),
    (0.0853, 1.049),
    (0.1196, 1.240),
    (0.1807, 1.137),
    (0.2763, 1.579),
];

/// Piecewise linear through `points`; `below(x)` left of the first, the last
/// value's rule right of the last.
fn through(
    points: &[(f64, f64)],
    x: f64,
    below: impl Fn(f64) -> f64,
    above: impl Fn(f64) -> f64,
) -> f64 {
    if x <= points[0].0 {
        return below(x);
    }
    for w in points.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if x <= x1 {
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
        }
    }
    above(x)
}

/// Step 1's figures measured with the danger read from Task 19's model and
/// [`READ_CALIBRATION`] in force (plan Task 20): 40,000 deals of four `stage`
/// bots, seed 2 (`BELIEF=models/belief-order.mjnn giveup_log log 40000 2`,
/// `runs/t20/giveup-read-seed2.txt`). Only `later_deal_in` depends on the reading.
pub const READ_MEASURED: Figures = Figures {
    own: [0.2477, 0.4999, 0.6098, 0.7279, 0.8159],
    theirs: [1.1688, 1.851, 2.2004, 2.5333, 2.9108],
    other_ends: [0.1264, 0.1244, 0.0964, 0.0636, 0.0316],
    later_deal_in: [0.0769, 0.0623, 0.0421, 0.0232, 0.0092],
};

/// [`CALIBRATION`] refitted on the danger read from Task 19's model (plan Task 20):
/// (mean predicted, observed deal-in rate) per danger bin of the stage 3 discards
/// of 40,000 deals of four `stage` bots, seed 1
/// (`BELIEF=models/belief-order.mjnn giveup_log log 40000 1`). The model is too sure
/// at both ends: 6.6 times too low in the safest bin, 0.6 times in the top one.
pub const READ_CALIBRATION: [(f64, f64); 10] = [
    (0.0008, 0.0054),
    (0.0072, 0.0237),
    (0.0143, 0.0388),
    (0.0246, 0.0593),
    (0.0389, 0.0742),
    (0.0592, 0.0917),
    (0.0837, 0.1171),
    (0.1223, 0.1432),
    (0.1931, 0.1752),
    (0.4596, 0.2781),
];

/// [`LOSS_FACTOR`] refitted on the danger read from Task 19's model (plan Task 20):
/// (mean raw danger of the bin, points paid ÷ predicted loss), same log.
pub const READ_LOSS_FACTOR: [(f64, f64); 10] = [
    (0.0008, 6.542),
    (0.0072, 3.187),
    (0.0143, 2.55),
    (0.0246, 2.326),
    (0.0389, 1.747),
    (0.0592, 1.489),
    (0.0837, 1.348),
    (0.1223, 1.078),
    (0.1931, 0.861),
    (0.4596, 0.54),
];

/// Whether a belief model is this thread's reading ([`crate::danger::with_reading`]).
fn on_reading() -> bool {
    danger::reading().is_some()
}

/// The figures in force: [`READ_MEASURED`] on a model's reading, else [`MEASURED`].
pub fn measured() -> Figures {
    if on_reading() {
        READ_MEASURED
    } else {
        MEASURED
    }
}

/// The calibration in force ([`READ_CALIBRATION`] or [`CALIBRATION`]).
pub fn calibration() -> &'static [(f64, f64); 10] {
    if on_reading() {
        &READ_CALIBRATION
    } else {
        &CALIBRATION
    }
}

/// The loss factor in force ([`READ_LOSS_FACTOR`] or [`LOSS_FACTOR`]).
pub fn loss_factor() -> &'static [(f64, f64); 10] {
    if on_reading() {
        &READ_LOSS_FACTOR
    } else {
        &LOSS_FACTOR
    }
}

/// [`Danger::tile`] mapped through the calibration in force ([`calibration`]).
pub fn calibrate(raw: f64) -> f64 {
    let c = calibration();
    let (first, last) = (c[0], c[c.len() - 1]);
    through(
        c,
        raw,
        |x| x * first.1 / first.0,
        |x| (x * last.1 / last.0).min(1.0),
    )
}

/// Calibrated danger and expected loss of `t` ([`CALIBRATION`], [`LOSS_FACTOR`]).
pub fn danger_of(danger: &Danger, t: Tile) -> (f64, f64) {
    let raw = danger.tile(t);
    let f = loss_factor();
    let factor = through(f, raw, |_| f[0].1, |_| f[f.len() - 1].1);
    (calibrate(raw), danger.expected_loss(t) * factor)
}

/// What a ready hand is worth to the formula: live winning tiles and the points
/// per share of a win.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outs {
    /// Unseen copies of the winning tiles.
    pub live: f64,
    /// Points per share of a self-drawn win (+1 faan for 自摸), averaged over the
    /// live tiles.
    pub own_points: f64,
    /// Points per share of a win on a discard.
    pub discard_points: f64,
}

/// The outs of a 13-tile view (0 live tiles when it is not ready).
pub fn outs(view: &SeatView) -> Outs {
    let rules = &view.rules;
    let cap = rules.faan_cap;
    let (mut live, mut own, mut disc) = (0.0, 0.0, 0.0);
    for (w, faan) in live_winning_tiles(view, &view.concealed) {
        let n = view.unseen[w] as f64;
        live += n;
        own += n * rules.points(faan.saturating_add(1).min(cap)) as f64;
        disc += n * rules.points(faan.min(cap)) as f64;
    }
    let per = |x: f64| if live > 0.0 { x / live } else { 0.0 };
    Outs {
        live,
        own_points: per(own),
        discard_points: per(disc),
    }
}

/// `V`: what staying ready is worth over `rounds` rounds left, in points.
/// `unseen` counts the tiles I cannot see (wall and opponents' hands), `pay` the
/// points one later deal-in would cost me. `scale` multiplies the result
/// (`stage-fold-half` 0.5, `stage-fold-double` 2).
pub fn push_value(
    fig: &Figures,
    rounds: usize,
    outs: &Outs,
    unseen: f64,
    pay: f64,
    scale: f64,
) -> f64 {
    if outs.live <= 0.0 || unseen <= 0.0 {
        return 0.0;
    }
    let share = outs.live / unseen;
    let mut alive = 1.0;
    let mut v = 0.0;
    for left in (1..=rounds).rev() {
        let b = bucket(left);
        let own = (fig.own[b] * share).min(1.0);
        let theirs = (fig.theirs[b] * share).min(1.0 - own);
        let q = fig.later_deal_in[b];
        v += alive * (own * 3.0 * outs.own_points + theirs * outs.discard_points - q * pay);
        alive *= (1.0 - own - theirs - fig.other_ends[b] - q).max(0.0);
    }
    v * scale
}

/// The break-even danger with one threat and nothing lost by folding:
/// `V ÷ (V + L)`.
pub fn break_even(v: f64, loss_if_hit: f64) -> f64 {
    if v <= 0.0 {
        0.0
    } else {
        v / (v + loss_if_hit)
    }
}

/// The rule's verdict for one discard decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Verdict {
    /// The tile to throw.
    pub tile: Tile,
    /// Whether the rule refused the chosen tile.
    pub gave_up: bool,
    /// `d`, `loss(t)` and `V` of the chosen tile (for logs and tests).
    pub danger: f64,
    pub loss: f64,
    pub value: f64,
}

/// Points one deal-in costs, averaged over the opponents by their threat.
fn pay_per_deal_in(danger: &Danger, me: usize) -> f64 {
    let (mut w, mut p) = (0.0, 0.0);
    for o in (0..4).filter(|&o| o != me) {
        w += danger.threat(o);
        p += danger.threat(o) * danger.points_if_win(o);
    }
    if w > 0.0 {
        p / w
    } else {
        8.0
    }
}

fn after(view: &SeatView, t: Tile) -> SeatView {
    let mut h = view.concealed;
    h.remove(t);
    view.with(h, view.melds.clone())
}

fn is_ready(view: &SeatView) -> bool {
    faan_shanten(&view.hand_view(), &view.rules).shanten == 0
}

/// The give-up rule for a 14-tile view on turn in `state`: `chosen` is the
/// three-stage bot's discard among `candidates`. Returns `None` when `chosen`
/// does not leave the hand ready (the rule does not apply).
pub fn judge(
    state: &State,
    view: &SeatView,
    candidates: &[Tile],
    chosen: Tile,
    fig: &Figures,
    scale: f64,
) -> Option<Verdict> {
    let a = after(view, chosen);
    if !is_ready(&a) {
        return None;
    }
    let danger = Danger::of_reading(state, view.seat);
    // Points a later deal-in costs: those of the chosen tile's deal-in (the
    // calibrated loss over the calibrated chance), else the threats' average.
    let pay = match danger_of(&danger, chosen) {
        (d, loss) if d > 1e-6 => loss / d,
        _ => pay_per_deal_in(&danger, view.seat),
    };
    let rounds = state.wall_remaining() / 4;
    let unseen = view.unseen.total() as f64;
    let push = |t: Tile, v: &SeatView| -> (f64, f64, f64) {
        let (d, loss) = danger_of(&danger, t);
        let value = push_value(fig, rounds, &outs(v), unseen, pay, scale);
        (d, loss, value)
    };
    let (d, loss, value) = push(chosen, &a);
    // `s`: the smallest-loss tile; the chosen tile itself when none is safer.
    let safest = match smallest_loss(&danger, candidates) {
        s if danger_of(&danger, s).1 < loss => s,
        _ => chosen,
    };
    let fold = {
        let s_view = after(view, safest);
        if safest != chosen && is_ready(&s_view) {
            let (ds, ls, vs) = push(safest, &s_view);
            (1.0 - ds) * vs - ls
        } else {
            -danger_of(&danger, safest).1
        }
    };
    let gave_up = safest != chosen && (1.0 - d) * value - loss <= fold;
    Some(Verdict {
        tile: if gave_up { safest } else { chosen },
        gave_up,
        danger: d,
        loss,
        value,
    })
}

/// The candidate with the smallest expected loss (ties: the first).
pub fn smallest_loss(danger: &Danger, candidates: &[Tile]) -> Tile {
    *candidates
        .iter()
        .min_by(|a, b| {
            danger_of(danger, **a)
                .1
                .total_cmp(&danger_of(danger, **b).1)
        })
        .expect("a candidate")
}

/// The smallest-loss tile for a folded hand on turn in `state`.
pub fn fold_tile(state: &State, seat: usize, candidates: &[Tile]) -> Tile {
    smallest_loss(&Danger::of_reading(state, seat), candidates)
}

fn mix(h: &mut DefaultHasher, e: &Event) {
    match *e {
        Event::Draw { seat, replacement } => (0u8, seat, replacement).hash(h),
        Event::Bonus { seat, tile } => (1u8, seat, tile.index()).hash(h),
        Event::Discard {
            seat,
            tile,
            from_hand,
        } => (2u8, seat, tile.index(), from_hand).hash(h),
        Event::Claim { seat, from, meld } => (3u8, seat, from, meld).hash(h),
        Event::ConcealedKong { seat, tile } => (4u8, seat, tile.index()).hash(h),
        Event::AddedKong { seat, tile } => (5u8, seat, tile.index()).hash(h),
        Event::Win { seat } => (6u8, seat).hash(h),
        Event::ExhaustiveDraw => 7u8.hash(h),
    }
}

/// Hashes of every prefix of the event list, seat first (index `n` = the first
/// `n` events).
fn prefix_keys(state: &State, seat: usize) -> Vec<u64> {
    let mut h = DefaultHasher::new();
    seat.hash(&mut h);
    let mut out = Vec::with_capacity(state.events().len() + 1);
    out.push(h.finish());
    for e in state.events() {
        mix(&mut h, e);
        out.push(h.finish());
    }
    out
}

thread_local! {
    /// Per seat and event history: `true` = gave up there, `false` = pushed
    /// again after a give-up (the give-up ends).
    static MARKS: RefCell<HashMap<u64, bool>> = RefCell::new(HashMap::new());
}

/// Remembers a give-up (`true`) or the end of one (`false`) at this decision.
pub fn remember(state: &State, seat: usize, gave_up: bool) {
    let key = *prefix_keys(state, seat).last().expect("a key");
    MARKS.with(|m| {
        let mut m = m.borrow_mut();
        if m.len() > 200_000 {
            m.clear();
        }
        m.insert(key, gave_up);
    });
}

/// Whether `seat` gave up earlier in this hand and has not pushed again since.
pub fn folded(state: &State, seat: usize) -> bool {
    MARKS.with(|m| {
        let m = m.borrow();
        if m.is_empty() {
            return false;
        }
        prefix_keys(state, seat)
            .iter()
            .rev()
            .find_map(|k| m.get(k).copied())
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage::StageBot;
    use mahjong_games::Game;
    use mahjong_hk::{Action, Meld, MeldKind, PositionPhase, PositionSpec, RuleSet};
    use mahjong_tiles::{parse_tiles, TileCounts};
    use rand::SeedableRng;

    fn tile(s: &str) -> Tile {
        s.parse().unwrap()
    }

    /// Seat 0 (me) on turn holds `hand` (13 tiles, ready) plus the drawn `extra`.
    /// Seat 2 shows exposed pungs of red and green (the heuristic reads 8 faan:
    /// two scoring pungs and all pungs). 24 tiles left in the wall (6 rounds).
    /// The tiles in `dead` and three Norths lie in the river; every other copy of
    /// those and of the tiles in `keep` stays out of it (in a hand or the wall).
    fn position(
        hand: &str,
        extra: &str,
        dead: &str,
        keep: &[&str],
        flower: bool,
        chow: Option<&str>,
    ) -> State {
        let mine = {
            let mut c = TileCounts::from_tiles(&parse_tiles(hand).unwrap());
            c.add(tile(extra));
            c
        };
        assert_eq!(mine.total() + 3 * chow.iter().count(), 14);
        let melds: [Vec<Meld>; 4] = [
            chow.map(|c| vec![Meld::exposed(MeldKind::Chow, tile(c), 3)])
                .unwrap_or_default(),
            vec![],
            vec![
                Meld::exposed(MeldKind::Pung, tile("R"), 1),
                Meld::exposed(MeldKind::Pung, tile("G"), 3),
            ],
            vec![],
        ];
        let mut pool = TileCounts::full_set();
        for t in mine.tiles() {
            pool.remove(t);
        }
        for m in melds.iter().flatten() {
            for t in m.tiles() {
                pool.remove(t);
            }
        }
        let mut river = vec![(1, tile("R"), true), (3, tile("G"), true)];
        if let Some(c) = chow {
            river.push((3, tile(c), true));
        }
        let mut forced = parse_tiles(dead).unwrap();
        forced.extend(parse_tiles("N N N").unwrap());
        forced.retain(|&t| {
            let ok = pool[t] > 0;
            if ok {
                pool.remove(t);
            }
            ok
        });
        // Tiles to hand out, kept ones last so they end in the wall.
        let kept: Vec<Tile> = keep
            .iter()
            .map(|s| tile(s))
            .chain(parse_tiles(dead).unwrap())
            .collect();
        let mut rest = pool.tiles();
        rest.sort_by_key(|t| (kept.contains(t), t.index()));
        let mut take = |n: usize| -> TileCounts {
            let c = TileCounts::from_tiles(&rest[..n]);
            rest.drain(..n);
            c
        };
        let seat2 = take(7);
        let seat1 = take(13);
        let seat3 = take(13);
        let wall = 24;
        let river_tiles: Vec<Tile> = forced
            .into_iter()
            .chain(rest[..rest.len() - wall].iter().copied())
            .collect();
        for (k, t) in river_tiles.into_iter().enumerate() {
            river.push(([0, 1, 3][k % 3], t, false));
        }
        let spec = PositionSpec {
            hands: [mine, seat1, seat2, seat3],
            melds,
            bonus: [
                if flower {
                    parse_tiles("F2").unwrap()
                } else {
                    vec![]
                },
                parse_tiles(if flower { "F1 F3" } else { "F1 F2 F3" }).unwrap(),
                parse_tiles("F4 S1 S2").unwrap(),
                parse_tiles("S3 S4").unwrap(),
            ],
            river,
            river_from_hand: Vec::new(),
            wall_remaining: wall,
            to_move: 0,
            phase: PositionPhase::Turn,
            last_drawn: Some(tile(extra)),
            kong_chain: 0,
            liable: [None; 4],
            passed_wins: Default::default(),
        };
        State::from_position(RuleSet::default(), &spec).expect("a valid position")
    }

    fn verdict(state: &State, chosen: &str) -> Verdict {
        let view = SeatView::of(state, 0);
        let c: Vec<Tile> = view.concealed.kinds().collect();
        judge(state, &view, &c, tile(chosen), &MEASURED, 1.0).expect("ready after it")
    }

    /// A cheap hand: half flush (3 faan; an exposed 123s chow and a flower that is
    /// not its own, so nothing else), waiting on West alone; with one West in the
    /// river (`CHEAP_DEAD`) two copies are live.
    const CHEAP: &str = "456s 789s 111s W";
    const CHEAP_DEAD: &str = "W";
    /// A full flush waiting on 2s and 5s: 8 faan (full flush, concealed; it holds
    /// a flower that is not its own, so no 無花).
    const FLUSH: &str = "111s 234s 567s 789s 5s";

    #[test]
    fn cheap_ready_hand_gives_up_a_dangerous_tile_and_pushes_a_safe_one() {
        let s = position(CHEAP, "7p", CHEAP_DEAD, &["7p"], true, Some("1s"));
        let view = SeatView::of(&s, 0);
        let o = outs(&after(&view, tile("7p")));
        assert_eq!(o.live, 2.0, "{o:?}");
        assert_eq!(o.discard_points, 8.0, "3 faan: {o:?}");
        let v = verdict(&s, "7p");
        let line = break_even(v.value, v.loss / v.danger);
        assert!(
            v.danger > line,
            "danger {} vs break-even {line}: {v:?}",
            v.danger
        );
        assert!(v.gave_up, "{v:?}");
        assert_ne!(v.tile, tile("7p"));
        // A dead North (three in the river, one in my hand) is pushed.
        let s = position(CHEAP, "N", CHEAP_DEAD, &["7p"], true, Some("1s"));
        let v = verdict(&s, "N");
        assert_eq!(v.danger, 0.0);
        assert!(!v.gave_up && v.tile == tile("N"), "{v:?}");
    }

    #[test]
    fn big_ready_hand_pushes_the_same_tile() {
        let s = position(FLUSH, "7p", "5s", &["7p", "2s"], true, None);
        let view = SeatView::of(&s, 0);
        let o = outs(&after(&view, tile("7p")));
        assert_eq!(o.discard_points, 64.0, "8 faan: {o:?}");
        let cheap = verdict(
            &position(CHEAP, "7p", CHEAP_DEAD, &["7p"], true, Some("1s")),
            "7p",
        );
        let v = verdict(&s, "7p");
        assert!((v.danger - cheap.danger).abs() < 0.02, "{v:?} {cheap:?}");
        let line = break_even(v.value, v.loss / v.danger);
        assert!(
            v.danger < line,
            "danger {} vs break-even {line}: {v:?}",
            v.danger
        );
        assert!(!v.gave_up && v.tile == tile("7p"), "{v:?}");
    }

    #[test]
    fn the_switch_decides_and_a_give_up_is_remembered() {
        // Gate 0 lets the hand with the drawn 7p (outside the flush) into stage 3.
        let bot = StageBot::default().with_gate(0);
        let s = position(CHEAP, "7p", CHEAP_DEAD, &["7p"], true, Some("1s"));
        let legal = s.legal_actions_vec();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let (off, note) = bot.decide(&s, &legal, &mut rng);
        assert_eq!(note.map(|n| n.stage), Some(3));
        let Action::Discard(pushed) = off else {
            panic!("{off:?}")
        };
        let v = verdict(&s, &pushed.to_string());
        assert!(
            v.gave_up,
            "the rule refuses the stage 3 pick {pushed}: {v:?}"
        );
        assert!(!folded(&s, 0));
        let (on, note) = bot.with_giveup(Some(1.0)).decide(&s, &legal, &mut rng);
        assert_eq!(on, Action::Discard(v.tile));
        assert!(note.is_some_and(|n| n.gave_up));
        assert!(folded(&s, 0), "the give-up is remembered for this hand");
        assert!(!folded(&s, 1), "only for this seat");
    }

    fn outs_of(live: f64, my_faan: u8) -> Outs {
        let r = mahjong_hk::RuleSet::default();
        Outs {
            live,
            own_points: r.points(my_faan + 1) as f64,
            discard_points: r.points(my_faan) as f64,
        }
    }

    #[test]
    fn value_grows_with_outs_and_rounds_and_scale() {
        let a = push_value(&ROUGH, 6, &outs_of(2.0, 3), 60.0, 24.0, 1.0);
        let b = push_value(&ROUGH, 6, &outs_of(4.0, 3), 60.0, 24.0, 1.0);
        let c = push_value(&ROUGH, 2, &outs_of(2.0, 3), 60.0, 24.0, 1.0);
        assert!(b > a && a > c && c > 0.0, "{a} {b} {c}");
        let half = push_value(&ROUGH, 6, &outs_of(2.0, 3), 60.0, 24.0, 0.5);
        assert!((half - a / 2.0).abs() < 1e-12);
        assert_eq!(
            push_value(&ROUGH, 6, &outs_of(0.0, 3), 60.0, 24.0, 1.0),
            0.0
        );
        assert_eq!(break_even(0.0, 10.0), 0.0);
        assert!((break_even(10.0, 30.0) - 0.25).abs() < 1e-12);
    }
}
