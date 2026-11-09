//! The race bot's training labels (plan 2026-11-09 Task 8, `docs/race-bot.md` §7 and
//! §12's redefinition) on hand-made positions: the true state (far / one away / ready /
//! folded), the label of a discard (push / detour / fold, and the "mistake" that is
//! written as push), plus the folded-state bookkeeping over real hands.

mod common;

use common::{build, counts, t, Setup};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::race::{discard_label, true_state, Label, LabelTracker, Mode, TrueState};
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

/// Expected loss 0 for `safe`, 10 points for every other tile.
fn only_safe(safe: Tile) -> impl Fn(Tile) -> f64 {
    move |u| if u == safe { 0.0 } else { 10.0 }
}

/// A characters full flush, ready on 1m / 4m after throwing the East (faan-aware
/// shanten 0); throwing any other tile leaves it one away.
const FLUSH: &str = "123m 456m 789m 11m 23m E";

#[test]
fn the_best_discard_is_push_whatever_its_danger() {
    let v = view(FLUSH);
    assert_eq!(discard_label(&v, t("E"), NO_LOSS), Label::Push);
    // The East is the only dangerous tile: still push.
    let east_deadly = |u: Tile| if u == t("E") { 10.0 } else { 0.0 };
    assert_eq!(discard_label(&v, t("E"), east_deadly), Label::Push);
}

/// Several discards of the same distance count as best for the hand when their
/// acceptance is within the tolerance: in `123m 456m 78m 11m 2369m E` the 6m, 9m and
/// East are all push, even when one of them is the dangerous one.
#[test]
fn near_ties_for_the_hand_are_push() {
    let v = view("123m 456m 78m 11m 2369m E");
    for d in ["6m", "9m", "E"] {
        assert_eq!(discard_label(&v, t(d), NO_LOSS), Label::Push, "{d}");
        assert_eq!(discard_label(&v, t(d), only_safe(t(d))), Label::Push, "{d}");
    }
}

/// `123m 456m 789m 11m 233m`: throwing the 9m keeps the hand ready on the most
/// tiles; throwing the 1m keeps it ready on fewer (same distance, acceptance beyond
/// the tolerance). For a clearly safer 1m that is a detour.
#[test]
fn same_distance_fewer_tiles_for_a_safer_tile_is_detour() {
    let v = view("123m 456m 789m 11m 233m");
    assert_eq!(discard_label(&v, t("9m"), NO_LOSS), Label::Push);
    assert_eq!(
        discard_label(&v, t("1m"), only_safe(t("1m"))),
        Label::Detour
    );
    assert_eq!(Label::Detour.mode(), Mode::Detour);
}

#[test]
fn breaking_the_ready_hand_for_a_safer_tile_is_fold() {
    let v = view(FLUSH);
    // The 5m leaves the hand one away (distance up by one) and is the safe tile.
    assert_eq!(discard_label(&v, t("5m"), only_safe(t("5m"))), Label::Fold);
    assert_eq!(Label::Fold.mode(), Mode::Fold);
}

/// Worse for the hand and not clearly safer: a mistake, written as push.
#[test]
fn worse_for_the_hand_and_not_safer_is_a_mistake() {
    let v = view(FLUSH);
    // Equal danger everywhere.
    assert_eq!(discard_label(&v, t("5m"), NO_LOSS), Label::Mistake);
    // The 5m is the most dangerous tile.
    let five_deadly = |u: Tile| if u == t("5m") { 10.0 } else { 1.0 };
    assert_eq!(discard_label(&v, t("5m"), five_deadly), Label::Mistake);
    // Safer, but not clearly: 0.1 points against the East's 0.2 is half, yet less
    // than the 0.16-point margin lower.
    let slightly = |u: Tile| if u == t("5m") { 0.1 } else { 0.2 };
    assert_eq!(discard_label(&v, t("5m"), slightly), Label::Mistake);
    // A mistake carries no defensive intent.
    assert_eq!(Label::Mistake.mode(), Mode::Push);
    // The safer side of the same margin: 0.0 against 0.2 is a fold.
    let clearly = |u: Tile| if u == t("5m") { 0.0 } else { 0.2 };
    assert_eq!(discard_label(&v, t("5m"), clearly), Label::Fold);
}

/// A fold starts the folded state; it lasts through detours and mistakes and ends at
/// the next push discard (or a claim).
#[test]
fn the_folded_state_persists_until_the_next_push() {
    let s = three_states();
    let mut tr = LabelTracker::new(&s);
    let seat = 1;
    assert!(!tr.folded(seat));
    tr.record(seat, Label::Fold);
    assert!(tr.folded(seat));
    assert_eq!(tr.state_of(&s, seat), TrueState::Folded);
    assert_eq!(tr.mode_of(seat), Mode::Fold);
    for later in [Label::Detour, Label::Mistake, Label::Fold, Label::Mistake] {
        tr.record(seat, later);
        assert_eq!(tr.state_of(&s, seat), TrueState::Folded, "{later:?}");
    }
    // The last discard was a mistake: its mode is push, but the seat stays folded.
    assert_eq!(tr.mode_of(seat), Mode::Push);
    tr.record(seat, Label::Push);
    assert!(!tr.folded(seat));
    assert_eq!(tr.state_of(&s, seat), TrueState::Ready);
    // Other seats are untouched.
    assert!(!tr.folded(2));
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
/// turn after a fold label until a push label (a detour or a mistake keeps it), claims
/// count as push; `Ready` agrees with the seat's legal wins. The `fast` bot plays for
/// its own hand only, so almost all its discards are push.
#[test]
fn the_tracker_follows_the_labels_over_real_hands() {
    let bot = Bot::by_name("fast").unwrap();
    let mut rng = SmallRng::seed_from_u64(11);
    let hands = if cfg!(debug_assertions) { 6 } else { 60 };
    let mut labels = [0usize; 4];
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
                (_, Some(Label::Fold)) => folded[p] = true,
                (_, Some(Label::Push)) | (Action::Pung | Action::Chow(_), None) => {
                    folded[p] = false
                }
                _ => {}
            }
            if let Some(m) = m {
                labels[m as usize] += 1;
                assert_eq!(tr.mode_of(p), m.mode());
            }
            s.apply(a);
        }
    }
    eprintln!(
        "labels over {hands} hands: push {}, detour {}, fold {}, mistake {}",
        labels[0], labels[1], labels[2], labels[3]
    );
    let total: usize = labels.iter().sum();
    assert!(labels[0] * 10 > total * 8, "{labels:?}");
}
