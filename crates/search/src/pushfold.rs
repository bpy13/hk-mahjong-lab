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

use crate::danger::Danger;
use crate::ev::HeuristicBelief;
use crate::giveup::{self, Outs, MEASURED};
use crate::pushfold_tables::{B1, B2, B3, CHECK, HIDDEN, LIN, MEAN, SD, W1, W2, W3};
use crate::stage::StageBot;
use crate::view::SeatView;
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
pub fn decision(
    state: &State,
    view: &SeatView,
    candidates: &[Tile],
    chosen: Tile,
) -> Option<Decision> {
    let a = after(view, chosen);
    if !is_ready(&a) {
        return None;
    }
    let me = view.seat;
    let danger = Danger::from_belief(state, me, &HeuristicBelief);
    let (d_t, loss_t) = giveup::danger_of(&danger, chosen);
    let s = giveup::smallest_loss(&danger, candidates);
    let (d_s, loss_s) = giveup::danger_of(&danger, s);
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
    })
}

/// Plays `state` (just after my discard) to the end: every seat plays `stage`;
/// while `folding`, I throw my smallest-loss tile until a stage 3 discard
/// leaves me ready again. Returns my points.
fn play_out(mut state: State, me: usize, mut folding: bool, rng: &mut ChaCha8Rng) -> i32 {
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
                action = Action::Discard(giveup::fold_tile(&state, me, &candidates));
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
    let p = play_out(push, dec.seat, false, &mut ChaCha8Rng::seed_from_u64(seed));
    let f = play_out(fold, dec.seat, true, &mut ChaCha8Rng::seed_from_u64(seed));
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
    /// Calibrated danger the bot sees ([`crate::giveup::danger_of`]).
    pub danger: f64,
    /// An opponent's real hand wins on it by discard (a legal win).
    pub deals_in: bool,
    /// Points the seat would pay (every winner declares; 0 when none).
    pub paid: i32,
}

/// For every candidate discard of the seat on turn: whether an opponent's real
/// hand wins on it, and the points it would cost (the engine settles the claim
/// window with every possible winner declaring).
pub fn truth(state: &State, view: &SeatView, candidates: &[Tile]) -> Vec<Truth> {
    let danger = Danger::from_belief(state, view.seat, &HeuristicBelief);
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
            Truth {
                tile: t,
                danger: giveup::danger_of(&danger, t).0,
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

/// The fitted model: predicted `push − fold` in points. A small network with
/// two hidden layers (ReLU) and a linear skip on standardised inputs, weights
/// from `pushfold_tables.rs`.
pub fn predict(x: &Features) -> f64 {
    let mut z = [0.0; FEATURES];
    for i in 0..FEATURES {
        z[i] = (x[i] - MEAN[i]) / SD[i];
    }
    let mut h1 = [0.0; HIDDEN];
    for (j, h) in h1.iter_mut().enumerate() {
        let s: f64 = (0..FEATURES).map(|i| W1[j][i] * z[i]).sum::<f64>() + B1[j];
        *h = s.max(0.0);
    }
    let mut h2 = [0.0; HIDDEN];
    for (j, h) in h2.iter_mut().enumerate() {
        let s: f64 = (0..HIDDEN).map(|i| W2[j][i] * h1[i]).sum::<f64>() + B2[j];
        *h = s.max(0.0);
    }
    let net: f64 = (0..HIDDEN).map(|i| W3[i] * h2[i]).sum();
    let lin: f64 = (0..FEATURES).map(|i| LIN[i] * z[i]).sum();
    net + lin + B3
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

/// The learned rule for the bot `stage-fold-learned`: `None` when `chosen` does
/// not leave the hand ready (the rule does not apply); otherwise the fold tile
/// when the model predicts `push − fold` below 0, else `chosen`.
pub fn judge(state: &State, view: &SeatView, candidates: &[Tile], chosen: Tile) -> Option<Verdict> {
    if !is_ready(&after(view, chosen)) {
        return None;
    }
    Some(match decision(state, view, candidates, chosen) {
        Some(dec) => {
            let predicted = predict(&dec.x);
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

/// Whether `pushfold_tables.rs` holds a fitted model (the placeholder has no
/// check rows).
pub fn fitted() -> bool {
    !CHECK.is_empty()
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
        decision(state, &view, &c, tile(chosen))
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
        let t = truth(&s, &view, &cands);
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
        if CHECK.is_empty() {
            eprintln!("pushfold_tables.rs is the placeholder: no check rows");
            return;
        }
        for (x, y) in CHECK {
            let p = predict(x);
            assert!((p - y).abs() < 1e-6, "predicted {p}, the script says {y}");
        }
    }

    #[test]
    fn judge_applies_only_to_a_ready_push() {
        let s = position(CHEAP, "7p", "W", 24);
        let view = SeatView::of(&s, 0);
        let c: Vec<Tile> = view.concealed.kinds().collect();
        assert!(judge(&s, &view, &c, tile("W")).is_none());
        let v = judge(&s, &view, &c, tile("7p")).unwrap();
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
        let v = judge(&s, &view, &c, tile("N")).unwrap();
        assert_eq!((v.tile, v.gave_up, v.predicted), (tile("N"), false, 0.0));
    }
}
