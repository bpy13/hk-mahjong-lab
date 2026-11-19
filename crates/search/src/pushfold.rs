//! The ready hand's push-or-fold choice learned from play-outs (plan
//! `docs/plans/2026-11-16-self-play-network.md`, Task 12).
//!
//! Task 11's rule ([`crate::giveup`]) compared a formula for the value of staying
//! ready with the expected loss of the chosen tile, and lost 0.63 ± 0.17 points a
//! hand. Brian, 2026-11-19: "you compared offensive EV to the deal-in EV.
//! realistically the opponent if ready also should have their own offensive EV
//! ... overestimating the deal-in EV as it should only be the difference between
//! the deal-in EV and the opponent offensive EV." Here nothing is a formula: at a
//! stage 3 decision where the chosen discard `t` leaves the hand ready and a
//! tile `s` with a smaller expected loss exists, both choices are played out on
//! the true hands with the wall reshuffled ([`label`]), and a small network fitted to the
//! mean difference in my points ([`predict`]; weights in `pushfold_tables.rs`,
//! written by `experiments/stage/fit_pushfold.py`) decides for the bot
//! `stage-fold-learned` ([`judge`]): fold when the predicted `push − fold` is
//! below 0. An opponent's self-draw, a third player's deal-in and a drawn hand
//! all land in the label by themselves.
//!
//! The play-outs start from the TRUE state of the logged deal: every seat keeps
//! the tiles it really holds, and only the order of the tiles still in the wall is
//! shuffled again ([`world`]), one shuffle per sample, the same shuffle and the
//! same random stream for the push and the fold branch, so the two differ only
//! in my discard (the orchestrator's correction, 2026-11-19: re-dealing the
//! opponents' hands would throw away the hands they built, so the label would not
//! be the real cost of the tile; averaged over positions that look alike from the
//! seat, labels from true hands are the expected result given what the seat sees.
//! Brian: "The probability you feed is not the probability of losing with that
//! tile"). All four seats play `stage` after the discard; the folded hand keeps
//! throwing its smallest-loss tile ([`crate::giveup::fold_tile`]) until a stage 3
//! discard leaves it ready again, then plays `stage` as before (Task 11's
//! after-fold behaviour). [`truth`] records, for every tile the seat could throw,
//! whether an opponent's real hand wins on it and what that costs.
//!
//! The model reads only what the seat can see ([`FEATURE_NAMES`]): the
//! calibrated danger and loss of `t` and `s` ([`crate::giveup::danger_of`], the
//! Task 11 calibration), my hand's points and live winning tiles after each,
//! rounds left, Task 11's formula value `V` and its pay figure, and per opponent
//! (next, across, previous) the belief's ready chance and expected points, melds
//! shown, discards made and whether `t` would make me liable for its fourth meld.
//!
//! The fold memory is Task 11's ([`crate::giveup::remember`],
//! [`crate::giveup::folded`]).
//!
//! **With exact danger** (plan Task 14, bot `stage-fold-learned-exact`): the
//! danger and loss inputs of `t` and `s` come from [`Risk`], the three layers
//! together: where Task 13's exact counting ([`crate::exact_wait`], an opponent
//! holding 1 or 4 concealed tiles) gives exactly 0 or 1 for the tile, that figure;
//! everywhere else the table (Task 15's risk network was not better than the
//! table on held-out deals, `docs/study-results/risk-network.md`, so it is not
//! used). The safest tile `s`, and the tile a folded hand throws, follow the same
//! figures. The model has the same inputs and shape, refitted on labels made with
//! these figures (weights in `pushfold_exact_tables.rs`).

use crate::danger::{self, Danger};
use crate::ev::HeuristicBelief;
use crate::giveup::{self, Outs, MEASURED};
use crate::stage::StageBot;
use crate::view::SeatView;
use crate::{pushfold_exact_tables as exact_tables, pushfold_tables as tables};
use mahjong_analysis::faan_shanten;
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, Event, State};
use mahjong_tiles::{Tile, TileCounts};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Number of inputs of the model ([`FEATURE_NAMES`]).
pub const FEATURES: usize = 31;

/// Names of the inputs, in order (the CSV header of `pushfold label`).
pub const FEATURE_NAMES: [&str; FEATURES] = [
    "d_t",
    "loss_t",
    "d_s",
    "loss_s",
    "my_points",
    "my_own_points",
    "live",
    "rounds_left",
    "unseen",
    "fold_ready",
    "fold_live",
    "fold_points",
    "v_measured",
    "pay",
    "t_is_drawn",
    "my_melds",
    "threat_next",
    "pay_next",
    "melds_next",
    "discards_next",
    "liable_next",
    "threat_across",
    "pay_across",
    "melds_across",
    "discards_across",
    "liable_across",
    "threat_prev",
    "pay_prev",
    "melds_prev",
    "discards_prev",
    "liable_prev",
];

/// The model's inputs for one decision.
pub type Features = [f64; FEATURES];

/// Units per hidden layer of the fitted model (both weight files).
pub const HIDDEN: usize = 32;

/// Where one tile's danger figure came from ([`Risk::of_tile`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// The table (through Task 11's calibration).
    Table,
    /// Exact counting says no exactly counted opponent wins on it: their part
    /// is 0, the table opponents' part stays.
    ExactZero,
    /// Exact counting says an opponent wins on it for certain: danger 1.
    ExactOne,
}

impl Source {
    /// `-1` for the table, else the exact figure (0 or 1), for the CSV.
    pub fn code(self) -> i8 {
        match self {
            Source::Table => -1,
            Source::ExactZero => 0,
            Source::ExactOne => 1,
        }
    }
}

/// The danger figures of one seat's discards (plan Task 14): the table alone,
/// or the table with Task 13's exact figure where that is exactly 0 or 1.
pub struct Risk {
    table: Danger,
    exact: Option<Danger>,
}

impl Risk {
    /// The figures for `me` on turn in `state`; `exact` switches the exact layer on.
    pub fn of(state: &State, me: usize, exact: bool) -> Risk {
        Risk {
            table: danger::with_exact(false, || Danger::from_belief(state, me, &HeuristicBelief)),
            exact: exact.then(|| {
                danger::with_exact(true, || Danger::from_belief(state, me, &HeuristicBelief))
            }),
        }
    }

    /// The table's danger (every opponent through the tables).
    pub fn table(&self) -> &Danger {
        &self.table
    }

    /// Calibrated danger and expected loss of `t`, and where they came from.
    /// Exactly 1 from the exact layer: danger 1 and the exact points; exactly 0:
    /// the table opponents' figures alone (Task 11's calibration on them); else
    /// the table's figures as [`crate::giveup::danger_of`] gives them.
    pub fn of_tile(&self, t: Tile) -> (f64, f64, Source) {
        if let Some(e) = &self.exact {
            if let Some(s) = e.split(t) {
                if s.exact_tile == 1.0 {
                    return (1.0, e.expected_loss(t), Source::ExactOne);
                }
                if s.exact_tile == 0.0 {
                    let (d, loss) = giveup::danger_of(e, t);
                    return (d, loss, Source::ExactZero);
                }
            }
        }
        let (d, loss) = giveup::danger_of(&self.table, t);
        (d, loss, Source::Table)
    }

    /// The candidate with the smallest expected loss (ties: the first).
    pub fn smallest_loss(&self, candidates: &[Tile]) -> Tile {
        *candidates
            .iter()
            .min_by(|a, b| self.of_tile(**a).1.total_cmp(&self.of_tile(**b).1))
            .expect("a candidate")
    }
}

/// The smallest-loss tile for a folded hand on turn in `state` under the chosen
/// figures (the table alone is [`crate::giveup::fold_tile`]).
pub fn fold_tile(state: &State, seat: usize, candidates: &[Tile], exact: bool) -> Tile {
    Risk::of(state, seat, exact).smallest_loss(candidates)
}

/// One push-or-fold decision: the stage 3 pick `push` leaves the hand ready and
/// `fold` has a smaller calibrated expected loss.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub seat: usize,
    pub push: Tile,
    pub fold: Tile,
    /// `fold` also leaves the hand ready (then the "fold" is another push).
    pub fold_ready: bool,
    pub x: Features,
    /// The figures used: the table alone, or with the exact layer (Task 14).
    pub exact: bool,
    /// Where `push`'s and `fold`'s danger came from.
    pub source: (Source, Source),
}

fn after(view: &SeatView, t: Tile) -> SeatView {
    let mut h = view.concealed;
    h.remove(t);
    view.with(h, view.melds.clone())
}

fn is_ready(view: &SeatView) -> bool {
    faan_shanten(&view.hand_view(), &view.rules).shanten == 0
}

fn discards_in(legal: &[Action]) -> Vec<Tile> {
    legal
        .iter()
        .filter_map(|a| match a {
            Action::Discard(x) => Some(*x),
            _ => None,
        })
        .collect()
}

/// The decision at a 14-tile view on turn in `state`, where `chosen` is the
/// three-stage bot's stage 3 discard among `candidates`: `None` when `chosen`
/// does not leave the hand ready or no candidate has a smaller expected loss.
/// `exact` switches the exact layer of the danger figures on ([`Risk`]).
pub fn decision(
    state: &State,
    view: &SeatView,
    candidates: &[Tile],
    chosen: Tile,
    exact: bool,
) -> Option<Decision> {
    let a = after(view, chosen);
    if !is_ready(&a) {
        return None;
    }
    let me = view.seat;
    let risk = Risk::of(state, me, exact);
    let danger = risk.table();
    let (d_t, loss_t, src_t) = risk.of_tile(chosen);
    let s = risk.smallest_loss(candidates);
    let (d_s, loss_s, src_s) = risk.of_tile(s);
    if s == chosen || loss_s >= loss_t {
        return None;
    }
    let outs: Outs = giveup::outs(&a);
    let s_view = after(view, s);
    let fold_ready = is_ready(&s_view);
    let fold_outs = if fold_ready {
        giveup::outs(&s_view)
    } else {
        Outs {
            live: 0.0,
            own_points: 0.0,
            discard_points: 0.0,
        }
    };
    let rounds = state.wall_remaining() / 4;
    let unseen = view.unseen.total() as f64;
    // Task 11's pay figure: the chosen tile's deal-in points, else the threats'
    // average.
    let pay = if d_t > 1e-6 {
        loss_t / d_t
    } else {
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
    };
    let v = giveup::push_value(&MEASURED, rounds, &outs, unseen, pay, 1.0);
    let mut discards = [0.0f64; 4];
    for e in state.events() {
        if let Event::Discard { seat, .. } = *e {
            discards[seat as usize] += 1.0;
        }
    }
    let liable = danger.liability_tile(chosen);
    let mut x = [0.0; FEATURES];
    x[0] = d_t;
    x[1] = loss_t;
    x[2] = d_s;
    x[3] = loss_s;
    x[4] = outs.discard_points;
    x[5] = outs.own_points;
    x[6] = outs.live;
    x[7] = rounds as f64;
    x[8] = unseen;
    x[9] = fold_ready as u8 as f64;
    x[10] = fold_outs.live;
    x[11] = fold_outs.discard_points;
    x[12] = v;
    x[13] = pay;
    x[14] = (state.last_drawn() == Some(chosen)) as u8 as f64;
    x[15] = view.melds.len() as f64;
    for (k, o) in (1..4).map(|r| (r - 1, (me + r) % 4)) {
        let base = 16 + 5 * k;
        x[base] = danger.threat(o);
        x[base + 1] = danger.points_if_win(o);
        x[base + 2] = state.melds(o).len() as f64;
        x[base + 3] = discards[o];
        x[base + 4] = (liable == Some(o)) as u8 as f64;
    }
    Some(Decision {
        seat: me,
        push: chosen,
        fold: s,
        fold_ready,
        x,
        exact,
        source: (src_t, src_s),
    })
}

/// Plays `state` (just after my discard) to the end: every seat plays `stage`;
/// while `folding`, I throw my smallest-loss tile (under the `exact` figures)
/// until a stage 3 discard leaves me ready again. Returns my points.
fn play_out(
    mut state: State,
    me: usize,
    mut folding: bool,
    exact: bool,
    rng: &mut ChaCha8Rng,
) -> i32 {
    let bot = StageBot::default();
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let seat = state.current_player().expect("not over");
        if seat != me {
            let a = bot.choose(&state, &legal, rng);
            state.apply(a);
            continue;
        }
        let (mut action, note) = bot.decide(&state, &legal, rng);
        let on_turn = !state.in_claim() && !state.in_rob();
        if folding && on_turn && !matches!(action, Action::Win | Action::SelfDrawnWin) {
            let candidates = discards_in(&legal);
            let pushes_again = match action {
                Action::Discard(t) => {
                    let view = SeatView::of_current(&state);
                    note.is_some_and(|n| n.stage == 3) && is_ready(&after(&view, t))
                }
                // A kong while folding: throw the safe tile instead.
                _ => false,
            };
            if pushes_again {
                folding = false;
            } else if !candidates.is_empty() {
                action = Action::Discard(fold_tile(&state, me, &candidates, exact));
            }
        }
        state.apply(action);
    }
    state.result().expect("over").deltas[me]
}

/// One play-out of each choice from `world` (a complete state with me on turn,
/// 14 tiles), both with the random stream `seed`: (my points after the push, my
/// points after the fold).
pub fn pair(world: &State, dec: &Decision, seed: u64) -> (i32, i32) {
    let mut push = world.clone();
    push.apply(Action::Discard(dec.push));
    let mut fold = world.clone();
    fold.apply(Action::Discard(dec.fold));
    let p = play_out(
        push,
        dec.seat,
        false,
        dec.exact,
        &mut ChaCha8Rng::seed_from_u64(seed),
    );
    let f = play_out(
        fold,
        dec.seat,
        true,
        dec.exact,
        &mut ChaCha8Rng::seed_from_u64(seed),
    );
    (p, f)
}

/// The true state with the undrawn wall reshuffled: sample `k` of the stream
/// `seed` (the same `(seed, k)` always gives the same order). Every seat keeps
/// the tiles it really holds.
pub fn world(state: &State, dec: &Decision, seed: u64, k: u64) -> State {
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ k);
    let mut wall = state.wall().to_vec();
    wall.shuffle(&mut rng);
    let hands: [TileCounts; 4] = std::array::from_fn(|o| *state.hand(o));
    state.redeal(dec.seat, &hands, wall)
}

/// What one tile of the seat's hand would really cost now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Truth {
    pub tile: Tile,
    /// Calibrated danger the bot sees ([`Risk::of_tile`] under `exact`).
    pub danger: f64,
    /// The table's calibrated danger ([`crate::giveup::danger_of`]).
    pub table_danger: f64,
    /// Where `danger` came from.
    pub source: Source,
    /// An opponent's real hand wins on it by discard (a legal win).
    pub deals_in: bool,
    /// Points the seat would pay (every winner declares; 0 when none).
    pub paid: i32,
}

/// For every candidate discard of the seat on turn: whether an opponent's real
/// hand wins on it, and the points it would cost (the engine settles the claim
/// window with every possible winner declaring).
pub fn truth(state: &State, view: &SeatView, candidates: &[Tile], exact: bool) -> Vec<Truth> {
    let risk = Risk::of(state, view.seat, exact);
    candidates
        .iter()
        .map(|&t| {
            let mut s = state.clone();
            s.apply(Action::Discard(t));
            while !s.is_terminal() && s.in_claim() {
                let legal = s.legal_actions_vec();
                let a = if legal.contains(&Action::Win) {
                    Action::Win
                } else {
                    Action::Pass
                };
                s.apply(a);
            }
            let paid = s
                .result()
                .filter(|r| r.wins.iter().any(|w| w.from == Some(view.seat)))
                .map_or(0, |r| -r.deltas[view.seat]);
            let (danger, _, source) = risk.of_tile(t);
            Truth {
                tile: t,
                danger,
                table_danger: giveup::danger_of(risk.table(), t).0,
                source,
                deals_in: paid > 0,
                paid,
            }
        })
        .collect()
}

/// The label of one decision over `samples` wall orders.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Label {
    /// Mean of (my points after the push − my points after the fold).
    pub mean: f64,
    /// Its standard error over the samples.
    pub se: f64,
    pub samples: usize,
    pub push_mean: f64,
    pub fold_mean: f64,
}

/// Plays both choices out `samples` times from `state` (me on turn with 14
/// tiles) with the true hands and a reshuffled wall, and returns the label.
pub fn label(state: &State, dec: &Decision, samples: usize, seed: u64) -> Label {
    let (mut sum, mut sq, mut p_sum, mut f_sum) = (0.0, 0.0, 0.0, 0.0);
    for k in 0..samples as u64 {
        let w = world(state, dec, seed, k);
        let (p, f) = pair(&w, dec, seed.wrapping_add(k));
        let d = (p - f) as f64;
        sum += d;
        sq += d * d;
        p_sum += p as f64;
        f_sum += f as f64;
    }
    let n = samples.max(1) as f64;
    let mean = sum / n;
    let var = (sq / n - mean * mean).max(0.0);
    Label {
        mean,
        se: if samples > 1 {
            (var / (n - 1.0)).sqrt()
        } else {
            0.0
        },
        samples,
        push_mean: p_sum / n,
        fold_mean: f_sum / n,
    }
}

/// The weights of one fitted model (a generated `*_tables.rs` file).
pub struct Weights {
    pub mean: &'static [f64; FEATURES],
    pub sd: &'static [f64; FEATURES],
    pub w1: &'static [[f64; FEATURES]; HIDDEN],
    pub b1: &'static [f64; HIDDEN],
    pub w2: &'static [[f64; HIDDEN]; HIDDEN],
    pub b2: &'static [f64; HIDDEN],
    pub w3: &'static [f64; HIDDEN],
    pub lin: &'static [f64; FEATURES],
    pub b3: f64,
    /// Held-out rows and the fit script's predictions (the parity test).
    pub check: &'static [(Features, f64)],
}

/// Task 12's model, fitted on labels made with the table's danger
/// (`pushfold_tables.rs`; bot `stage-fold-learned`).
pub const TABLE_MODEL: Weights = Weights {
    mean: &tables::MEAN,
    sd: &tables::SD,
    w1: &tables::W1,
    b1: &tables::B1,
    w2: &tables::W2,
    b2: &tables::B2,
    w3: &tables::W3,
    lin: &tables::LIN,
    b3: tables::B3,
    check: tables::CHECK,
};

/// Task 14's model, fitted on labels made with the exact figure where it is 0
/// or 1 (`pushfold_exact_tables.rs`; bot `stage-fold-learned-exact`).
pub const EXACT_MODEL: Weights = Weights {
    mean: &exact_tables::MEAN,
    sd: &exact_tables::SD,
    w1: &exact_tables::W1,
    b1: &exact_tables::B1,
    w2: &exact_tables::W2,
    b2: &exact_tables::B2,
    w3: &exact_tables::W3,
    lin: &exact_tables::LIN,
    b3: exact_tables::B3,
    check: exact_tables::CHECK,
};

/// The model for the chosen figures ([`TABLE_MODEL`] or [`EXACT_MODEL`]).
pub const fn model(exact: bool) -> &'static Weights {
    if exact {
        &EXACT_MODEL
    } else {
        &TABLE_MODEL
    }
}

/// Task 12's fitted model: predicted `push − fold` in points
/// ([`predict_with`] on [`TABLE_MODEL`]).
pub fn predict(x: &Features) -> f64 {
    predict_with(&TABLE_MODEL, x)
}

/// A fitted model's prediction of `push − fold` in points. A small network with
/// two hidden layers (ReLU) and a linear skip on standardised inputs.
pub fn predict_with(m: &Weights, x: &Features) -> f64 {
    let mut z = [0.0; FEATURES];
    for i in 0..FEATURES {
        z[i] = (x[i] - m.mean[i]) / m.sd[i];
    }
    let mut h1 = [0.0; HIDDEN];
    for (j, h) in h1.iter_mut().enumerate() {
        let s: f64 = (0..FEATURES).map(|i| m.w1[j][i] * z[i]).sum::<f64>() + m.b1[j];
        *h = s.max(0.0);
    }
    let mut h2 = [0.0; HIDDEN];
    for (j, h) in h2.iter_mut().enumerate() {
        let s: f64 = (0..HIDDEN).map(|i| m.w2[j][i] * h1[i]).sum::<f64>() + m.b2[j];
        *h = s.max(0.0);
    }
    let net: f64 = (0..HIDDEN).map(|i| m.w3[i] * h2[i]).sum();
    let lin: f64 = (0..FEATURES).map(|i| m.lin[i] * z[i]).sum();
    net + lin + m.b3
}

/// The model's verdict for one discard decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Verdict {
    /// The tile to throw.
    pub tile: Tile,
    /// Whether the model refused the chosen tile.
    pub gave_up: bool,
    /// Predicted `push − fold`; 0 when no tile is safer (nothing to decide).
    pub predicted: f64,
}

/// The learned rule for the bots `stage-fold-learned` (`exact` off) and
/// `stage-fold-learned-exact` (`exact` on): `None` when `chosen` does not leave
/// the hand ready (the rule does not apply); otherwise the fold tile when the
/// model predicts `push − fold` below 0, else `chosen`.
pub fn judge(
    state: &State,
    view: &SeatView,
    candidates: &[Tile],
    chosen: Tile,
    exact: bool,
) -> Option<Verdict> {
    if !is_ready(&after(view, chosen)) {
        return None;
    }
    Some(match decision(state, view, candidates, chosen, exact) {
        Some(dec) => {
            let predicted = predict_with(model(exact), &dec.x);
            let gave_up = predicted < 0.0;
            Verdict {
                tile: if gave_up { dec.fold } else { chosen },
                gave_up,
                predicted,
            }
        }
        None => Verdict {
            tile: chosen,
            gave_up: false,
            predicted: 0.0,
        },
    })
}

/// Whether the model for the chosen figures holds fitted weights (a placeholder
/// file has no check rows).
pub fn fitted(exact: bool) -> bool {
    !model(exact).check.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahjong_hk::{Meld, MeldKind, PositionPhase, PositionSpec, RuleSet};
    use mahjong_tiles::{parse_tiles, TileCounts};

    fn tile(s: &str) -> Tile {
        s.parse().unwrap()
    }

    /// Seat 0 (me) on turn holds `hand` (10 tiles) plus the drawn `extra`, with
    /// an exposed 123s chow; seat 2 shows pungs of red and green. The tiles in
    /// `dead` and three Norths lie in the river; `wall` tiles stay in the wall.
    /// Based on the Task 11 test position (`giveup.rs`).
    fn position(hand: &str, extra: &str, dead: &str, wall: usize) -> State {
        let mine = {
            let mut c = TileCounts::from_tiles(&parse_tiles(hand).unwrap());
            c.add(tile(extra));
            c
        };
        assert_eq!(mine.total(), 11);
        let melds: [Vec<Meld>; 4] = [
            vec![Meld::exposed(MeldKind::Chow, tile("1s"), 3)],
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
        let mut river = vec![
            (1, tile("R"), true),
            (3, tile("G"), true),
            (3, tile("1s"), true),
        ];
        let mut forced = parse_tiles(dead).unwrap();
        forced.extend(parse_tiles("N N N").unwrap());
        forced.retain(|&t| {
            let ok = pool[t] > 0;
            if ok {
                pool.remove(t);
            }
            ok
        });
        // The other copies of `dead` and of 7p stay out of the river (in a hand
        // or the wall), as in the Task 11 test, so 7p is a live danger.
        let mut kept: Vec<Tile> = parse_tiles(dead).unwrap();
        kept.push(tile("7p"));
        let mut rest = pool.tiles();
        rest.sort_by_key(|t| (kept.contains(t), t.index()));
        // With an empty wall the kept tiles would land in the river: seat 3 holds
        // them instead (with man tiles; it cannot win on 7p or on a sou tile).
        let last = (wall == 0).then(|| {
            let tail = rest.split_off(rest.len() - 13);
            TileCounts::from_tiles(&tail)
        });
        let mut take = |n: usize| -> TileCounts {
            let c = TileCounts::from_tiles(&rest[..n]);
            rest.drain(..n);
            c
        };
        let seat2 = take(7);
        let seat1 = take(13);
        let seat3 = last.unwrap_or_else(|| take(13));
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
                parse_tiles("F2").unwrap(),
                parse_tiles("F1 F3").unwrap(),
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

    /// Half flush waiting on West; with one West in the river two are live.
    const CHEAP: &str = "456s 789s 111s W";

    fn decide(state: &State, chosen: &str) -> Option<Decision> {
        let view = SeatView::of(state, 0);
        let c: Vec<Tile> = view.concealed.kinds().collect();
        decision(state, &view, &c, tile(chosen), false)
    }

    /// I (seat 0) am on turn with `mine` (14 tiles) and an empty wall; seat 2
    /// shows four pungs of bamboos, seats 1 and 3 four pungs of characters, each
    /// holding the one tile in `singles`. Every other tile lies in the river
    /// (the last one thrown by seat 2). Task 13's test position (`exact_wait.rs`).
    fn four_pungs(singles: [&str; 3], mine: &str) -> State {
        let pung = |s: &str, from: usize| Meld::exposed(MeldKind::Pung, tile(s), from);
        let hands: [TileCounts; 4] = [
            TileCounts::from_tiles(&parse_tiles(mine).unwrap()),
            TileCounts::from_tiles(&parse_tiles(singles[0]).unwrap()),
            TileCounts::from_tiles(&parse_tiles(singles[1]).unwrap()),
            TileCounts::from_tiles(&parse_tiles(singles[2]).unwrap()),
        ];
        let melds: [Vec<Meld>; 4] = [
            vec![],
            vec![pung("1m", 0), pung("2m", 2), pung("3m", 3), pung("4m", 0)],
            vec![pung("1s", 1), pung("3s", 3), pung("5s", 1), pung("7s", 0)],
            vec![pung("5m", 0), pung("6m", 1), pung("7m", 2), pung("8m", 0)],
        ];
        let mut pool = TileCounts::full_set();
        for h in &hands {
            for t in h.tiles() {
                pool.remove(t);
            }
        }
        let mut river: Vec<(usize, Tile, bool)> = Vec::new();
        for m in melds.iter().flatten() {
            for t in m.tiles() {
                pool.remove(t);
            }
            river.push((m.claimed_from.unwrap() as usize, m.tile, true));
        }
        let rest = pool.tiles();
        let n = rest.len();
        for (k, &x) in rest.iter().enumerate() {
            river.push(((k + 4 * n + 3 - n) % 4, x, false));
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
            river_from_hand: Vec::new(),
            wall_remaining: 0,
            to_move: 0,
            phase: PositionPhase::Turn,
            last_drawn: Some(parse_tiles(mine).unwrap()[0]),
            kong_chain: 0,
            liable: [None; 4],
            passed_wins: Default::default(),
        };
        State::from_position(RuleSet::default(), &spec).expect("a valid position")
    }

    #[test]
    fn exact_figures_override_the_table_where_they_are_0_or_1() {
        // Every opponent holds one tile; seat 2 can only wait on 9s (full flush,
        // all pungs), seats 1 and 3 on 9s too (all pungs).
        let s = four_pungs(["9s", "9s", "9s"], "9s 123456789p ESWN");
        let view = SeatView::of(&s, 0);
        let cands: Vec<Tile> = view.concealed.kinds().collect();
        let exact = Risk::of(&s, 0, true);
        let table = Risk::of(&s, 0, false);
        let (d, loss, src) = exact.of_tile(tile("9s"));
        assert_eq!((d, src), (1.0, Source::ExactOne));
        assert!(loss >= 128.0, "the exact points, not the table's: {loss}");
        let (d, loss, src) = exact.of_tile(tile("5p"));
        assert_eq!((d, loss, src), (0.0, 0.0, Source::ExactZero));
        // The table alone never gives 1 (nor 0 for a tile with an unseen copy), and
        // it is what the table source reads.
        for t in &cands {
            let (d, loss, src) = table.of_tile(*t);
            assert_eq!(src, Source::Table);
            assert_eq!((d, loss), giveup::danger_of(table.table(), *t), "{t}");
            assert!(d < 1.0, "{t}: {d}");
            assert!(d > 0.0 || view.unseen[*t] == 0, "{t}: {d}");
        }
        assert_ne!(exact.smallest_loss(&cands), tile("9s"));
        assert_ne!(fold_tile(&s, 0, &cands, true), tile("9s"));
        assert_eq!(
            fold_tile(&s, 0, &cands, false),
            giveup::fold_tile(&s, 0, &cands),
            "the table figures give Task 11's fold tile"
        );
        // The truth agrees: 9s deals in, 5p does not; the recorded sources match.
        let tr = truth(&s, &view, &cands, true);
        let nine = tr.iter().find(|x| x.tile == tile("9s")).unwrap();
        assert!(nine.deals_in && nine.paid >= 128 && nine.source == Source::ExactOne);
        assert!(nine.table_danger < 1.0);
        let five = tr.iter().find(|x| x.tile == tile("5p")).unwrap();
        assert!(!five.deals_in && five.source == Source::ExactZero && five.danger == 0.0);
        // The same decision inputs with the switch off come from the table.
        let tr = truth(&s, &view, &cands, false);
        assert!(tr
            .iter()
            .all(|x| x.source == Source::Table && x.danger == x.table_danger));
    }

    #[test]
    fn a_decision_needs_a_ready_push_and_a_safer_tile() {
        let s = position(CHEAP, "7p", "W", 24);
        let d = decide(&s, "7p").expect("7p leaves the hand ready and a safer tile exists");
        assert_eq!((d.seat, d.push), (0, tile("7p")));
        assert_ne!(d.fold, tile("7p"));
        assert!(d.x[1] > d.x[3], "loss(t) {} > loss(s) {}", d.x[1], d.x[3]);
        assert_eq!(d.x[6], 2.0, "two live Wests: {:?}", d.x);
        assert_eq!(d.x[4], 8.0, "3 faan on a discard");
        assert_eq!(d.x[7], 6.0, "6 rounds left");
        assert_eq!(d.x[14], 1.0, "7p was just drawn");
        assert_eq!(d.x[15], 1.0, "one meld of mine");
        // The opponent across (seat 2) shows two melds.
        assert_eq!(d.x[16 + 5 + 2], 2.0, "{:?}", d.x);
        // Throwing West breaks the wait: no decision.
        assert!(decide(&s, "W").is_none());
        // A dead North is the safest tile itself: nothing to decide.
        let s = position(CHEAP, "N", "W", 24);
        assert!(decide(&s, "N").is_none());
        assert_eq!(FEATURE_NAMES.len(), FEATURES);
    }

    #[test]
    fn push_and_fold_play_outs_use_the_same_hidden_tiles() {
        let s = position(CHEAP, "7p", "W", 24);
        let d = decide(&s, "7p").unwrap();
        let a = world(&s, &d, 7, 0);
        let b = world(&s, &d, 7, 0);
        assert_eq!(a.wall(), b.wall(), "the same seed gives the same wall");
        for o in 0..4 {
            assert_eq!(a.hand(o), s.hand(o), "every seat keeps its true hand");
        }
        let mut sorted = (a.wall().to_vec(), s.wall().to_vec());
        sorted.0.sort();
        sorted.1.sort();
        assert_eq!(sorted.0, sorted.1, "the same tiles stay in the wall");
        let c = world(&s, &d, 7, 1);
        assert_ne!(c.wall(), a.wall(), "another sample is another order");
        // The truth of every tile: 7p and the sou tiles win for nobody here (the
        // opponents hold man tiles and the dragon pungs).
        let view = SeatView::of(&s, 0);
        let cands: Vec<Tile> = view.concealed.kinds().collect();
        let t = truth(&s, &view, &cands, false);
        assert_eq!(t.len(), cands.len());
        assert!(t.iter().all(|x| !x.deals_in && x.paid == 0), "{t:?}");
        assert!(t.iter().any(|x| x.danger > 0.0));
        // `pair` plays both choices from one world: the first discards differ,
        // nothing else before them.
        let (p, f) = pair(&a, &d, 3);
        assert_eq!(
            (p, f),
            pair(&a, &d, 3),
            "the same seed replays the same pair"
        );
        let l = label(&s, &d, 4, 7);
        assert_eq!(l.samples, 4);
        assert!((l.mean - (l.push_mean - l.fold_mean)).abs() < 1e-9);
        assert!(l.se >= 0.0);
    }

    #[test]
    fn a_position_where_every_play_out_ends_the_same_gives_label_0() {
        // Empty wall: after my discard the hand ends in an exhaustive draw unless
        // someone wins on it; 7p and the safe tile win for nobody here (all three
        // opponents are far from ready), so both branches score 0.
        let s = position(CHEAP, "7p", "W", 0);
        assert_eq!(s.wall_remaining(), 0);
        let d = decide(&s, "7p").unwrap();
        let l = label(&s, &d, 6, 1);
        assert_eq!(l.mean, 0.0, "{l:?}");
        assert_eq!(l.se, 0.0);
        assert_eq!(l.push_mean, l.fold_mean);
    }

    #[test]
    fn predict_matches_the_fit_script_on_its_check_rows() {
        for (name, m) in [
            ("pushfold_tables.rs", &TABLE_MODEL),
            ("pushfold_exact_tables.rs", &EXACT_MODEL),
        ] {
            if m.check.is_empty() {
                eprintln!("{name} is the placeholder: no check rows");
                continue;
            }
            for (x, y) in m.check {
                let p = predict_with(m, x);
                assert!(
                    (p - y).abs() < 1e-6,
                    "{name}: predicted {p}, the script says {y}"
                );
            }
        }
        // Task 12's weights unchanged: `predict` is the table model.
        if let Some((x, y)) = TABLE_MODEL.check.first() {
            assert!((predict(x) - y).abs() < 1e-6);
        }
        assert_eq!(fitted(false), !TABLE_MODEL.check.is_empty());
        assert_eq!(fitted(true), !EXACT_MODEL.check.is_empty());
    }

    #[test]
    fn judge_applies_only_to_a_ready_push() {
        let s = position(CHEAP, "7p", "W", 24);
        let view = SeatView::of(&s, 0);
        let c: Vec<Tile> = view.concealed.kinds().collect();
        assert!(judge(&s, &view, &c, tile("W"), false).is_none());
        let v = judge(&s, &view, &c, tile("7p"), false).unwrap();
        if v.gave_up {
            assert!(v.predicted < 0.0);
            assert_ne!(v.tile, tile("7p"));
        } else {
            assert!(v.predicted >= 0.0);
            assert_eq!(v.tile, tile("7p"));
        }
        // Nothing safer than a dead North: kept, nothing predicted.
        let s = position(CHEAP, "N", "W", 24);
        let view = SeatView::of(&s, 0);
        let c: Vec<Tile> = view.concealed.kinds().collect();
        let v = judge(&s, &view, &c, tile("N"), false).unwrap();
        assert_eq!((v.tile, v.gave_up, v.predicted), (tile("N"), false, 0.0));
        // With the exact layer on, no opponent here holds 1 or 4 concealed tiles:
        // the same figures, the same verdict (the exact model may differ in sign
        // only if it is fitted; with the placeholder it predicts 0).
        let v = judge(&s, &view, &c, tile("N"), true).unwrap();
        assert_eq!((v.tile, v.gave_up, v.predicted), (tile("N"), false, 0.0));
        let s = position(CHEAP, "7p", "W", 24);
        let view = SeatView::of(&s, 0);
        let c: Vec<Tile> = view.concealed.kinds().collect();
        let d = decide(&s, "7p").unwrap();
        let e = decision(&s, &view, &c, tile("7p"), true).unwrap();
        assert_eq!((e.exact, e.source), (true, (Source::Table, Source::Table)));
        assert_eq!((d.exact, d.source), (false, (Source::Table, Source::Table)));
        assert_eq!(e.x, d.x, "no short hand: the exact layer changes nothing");
    }
}
