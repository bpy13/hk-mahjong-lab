//! The race bot's training labels (plan 2026-11-09 Task 8, `docs/race-bot.md` §7) on
//! hand-made positions: the true state (far / one away / ready / folded) and the mode
//! label of a discard (push / detour / fold), plus the folded-state bookkeeping over
//! real hands.

mod common;

use common::{build, counts, t, Setup};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::race::{discard_mode, true_state, LabelTracker, Mode, TrueState};
use mahjong_search::{Bot, SeatView};
use mahjong_tiles::{Tile, TileCounts};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Seat 0's view of a concealed hand with no melds; everything it does not hold is unseen.
fn view(hand: &str) -> SeatView {
    let concealed = counts(hand);
    let mut unseen = TileCounts::new();
    for k in Tile::all_kinds() {
        for _ in concealed[k]..4 {
            unseen.add(k);
        }
    }
    SeatView {
        seat: 0,
        concealed,
        melds: Vec::new(),
        bonus: 0,
        unseen,
        rules: RuleSet::default(),
    }
}

const NO_LOSS: fn(Tile) -> f64 = |_| 0.0;

/// A characters full flush, ready on 1m / 4m after throwing the East (faan-aware
/// shanten 0); throwing the 5m instead leaves it one away.
const FLUSH: &str = "123m 456m 789m 11m 23m E";

#[test]
fn the_best_discard_is_push() {
    let v = view(FLUSH);
    assert_eq!(discard_mode(&v, t("E"), Some(1), NO_LOSS), Mode::Push);
    // No earlier shanten known (the dealer's first discard): the same.
    assert_eq!(discard_mode(&v, t("E"), None, NO_LOSS), Mode::Push);
}

#[test]
fn one_step_worse_than_the_best_is_detour() {
    let v = view(FLUSH);
    assert_eq!(discard_mode(&v, t("5m"), Some(1), NO_LOSS), Mode::Detour);
    assert_eq!(discard_mode(&v, t("5m"), None, NO_LOSS), Mode::Detour);
}

#[test]
fn breaking_a_ready_hand_is_fold_even_by_one_step() {
    let v = view(FLUSH);
    // Ready before the draw (s_before = 0), one away after the 5m: fold.
    assert_eq!(discard_mode(&v, t("5m"), Some(0), NO_LOSS), Mode::Fold);
    // Staying ready is still push.
    assert_eq!(discard_mode(&v, t("E"), Some(0), NO_LOSS), Mode::Push);
}

/// Several discards keep `123m 456m 789m 11m 233m` ready (1m, 2m, 3m among them),
/// with different waits. A ready discard is a detour only when its waits differ from
/// `rank_discards`' first choice and the public expected loss says it is safer.
#[test]
fn a_safer_ready_discard_with_other_waits_is_detour() {
    let v = view("123m 456m 789m 11m 233m");
    // Equal danger everywhere: every discard that stays ready is push, the others
    // (one step worse) detour.
    let ready: Vec<Tile> = v
        .concealed
        .kinds()
        .filter(|&d| discard_mode(&v, d, Some(1), NO_LOSS) == Mode::Push)
        .collect();
    for d in ["1m", "2m", "3m"] {
        assert!(ready.contains(&t(d)), "{d} keeps the hand ready: {ready:?}");
    }
    // Only `safe` is safe: it is a detour unless it is the first choice itself (same
    // waits). Exactly one ready discard is the first choice; the 1m, 2m and 3m
    // discards leave three different wait sets, so at least two of them are detours.
    let labels: Vec<Mode> = ready
        .iter()
        .map(|&safe| {
            discard_mode(
                &v,
                safe,
                Some(0),
                move |u| if u == safe { 0.0 } else { 10.0 },
            )
        })
        .collect();
    assert!(
        labels.iter().filter(|&&m| m == Mode::Detour).count() >= 2,
        "{labels:?}"
    );
    assert!(labels.contains(&Mode::Push), "{labels:?}");
    // A ready discard is a detour only when it is safer: with the danger reversed
    // (the chosen tile the most dangerous) nothing is a detour.
    for &risky in &ready {
        let m = discard_mode(
            &v,
            risky,
            Some(0),
            move |u| if u == risky { 10.0 } else { 0.0 },
        );
        assert_eq!(m, Mode::Push, "{risky}");
    }
}

/// Seat 1 ready (a characters flush on 1m / 4m), seat 2 one away (a dots flush
/// short of a tile), seat 3 far (scattered tiles and honours).
fn three_states() -> State {
    build(Setup {
        hands: [
            Some("123456789s SS WW N"),
            Some("123m 456m 789m 11m 23m"),
            Some("123p 456p 789p 11p 2p E"),
            Some("159m 26s 37s E S W N R G"),
        ],
        melds: Default::default(),
        claimed: vec![],
        last_drawn: "N",
        wall: 40,
        liable: [None; 4],
        keep: "",
    })
}

#[test]
fn true_states_of_hand_made_hands() {
    let s = three_states();
    assert_eq!(true_state(&s, 1, false), TrueState::Ready);
    assert_eq!(true_state(&s, 2, false), TrueState::OneAway);
    assert_eq!(true_state(&s, 3, false), TrueState::Far);
    // Folded overrides every state.
    for seat in 1..4 {
        assert_eq!(true_state(&s, seat, true), TrueState::Folded);
    }
}

/// Over real hands: every discard gets a label; a seat is folded exactly from the
/// turn after a fold label until a push label (a detour keeps it), claims count as
/// push; `Ready` agrees with the seat's legal wins.
#[test]
fn the_tracker_follows_the_labels_over_real_hands() {
    let bot = Bot::by_name("fast").unwrap();
    let mut rng = SmallRng::seed_from_u64(11);
    let hands = if cfg!(debug_assertions) { 6 } else { 60 };
    let mut labels = [0usize; 3];
    for _ in 0..hands {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let mut tr = LabelTracker::new(&s);
        let mut folded = [false; 4];
        while let Some(p) = s.current_player() {
            for (seat, &is_folded) in folded.iter().enumerate() {
                let st = tr.state_of(&s, seat);
                if is_folded {
                    assert_eq!(st, TrueState::Folded);
                } else {
                    let ready = Tile::all_kinds().any(|k| s.wins_on(seat, k));
                    assert_eq!(st == TrueState::Ready, ready, "seat {seat}");
                }
            }
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, &mut rng);
            let is_discard = matches!(a, Action::Discard(_)) && s.phase_name() == "turn";
            let m = tr.observe(&s, a);
            assert_eq!(m.is_some(), is_discard, "{a}");
            match (a, m) {
                (_, Some(Mode::Fold)) => folded[p] = true,
                (_, Some(Mode::Push)) | (Action::Pung | Action::Chow(_), None) => folded[p] = false,
                _ => {}
            }
            if let Some(m) = m {
                labels[m as usize] += 1;
                assert_eq!(tr.mode_of(p), m);
            }
            s.apply(a);
        }
    }
    eprintln!(
        "labels over {hands} hands: push {}, detour {}, fold {}",
        labels[0], labels[1], labels[2]
    );
    assert!(labels[0] > labels[1]);
}
