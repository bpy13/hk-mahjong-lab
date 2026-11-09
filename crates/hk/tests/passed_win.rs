//! The passed-win rule 過水 (`docs/rules/hk.md` §4.4): a seat that passes a win
//! on a tile cannot win on that tile from another player until its next turn.

mod common;
use common::{deal, no_flowers, play, t};
use mahjong_games::{Game, Policy};
use mahjong_hk::{PositionError, RuleSet, State};
use mahjong_tiles::TileCounts;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use mahjong_hk::Action::*;

/// Seat 2 (`123p 456p 789p RR NN`) waits on R or N. The draws give seat 3 an R,
/// the dealer the last R, seat 1 a 3m (it then throws its N), seat 2 an E and
/// seat 3 the last N.
fn waiting_on_r_or_n(rules: RuleSet) -> State {
    let mut s = deal(
        rules,
        [
            "2s 9m 11m 33m 5m 7m 9s E SS 4p 8p",
            "1s 2s 3s 4s 5s 6s 7s 8s E 5m 6m W N",
            "123p 456p 789p RR NN",
            "9s 9s W W G G Wh Wh 1p 2p 3p 6s 6s",
        ],
        "E 5p R R 3m E N",
        "",
    );
    play(
        &mut s,
        &[
            (0, Discard(t("9m"))),
            (1, Discard(t("E"))),
            (2, Discard(t("5p"))),
            (3, Discard(t("R"))),
        ],
    );
    // Seat 3's R: seat 2 may win, and passes.
    assert_eq!(s.legal_actions_vec(), vec![Win, Pung, Pass]);
    play(&mut s, &[(2, Pass)]);
    s
}

#[test]
fn a_passed_win_locks_the_tile_from_another_discarder_4_4() {
    let mut s = waiting_on_r_or_n(no_flowers());
    assert_eq!(s.passed_win_lock(2), vec![t("R")]);
    assert!(s.passed_win_locked(2, t("R")));
    assert!((0..4)
        .filter(|&p| p != 2)
        .all(|p| s.passed_win_lock(p).is_empty()));
    // The dealer throws the last R: no win for seat 2, but the pung stays (§4.4.4).
    play(&mut s, &[(0, Discard(t("R")))]);
    assert_eq!(s.current_player(), Some(2));
    assert_eq!(s.legal_actions_vec(), vec![Pung, Pass]);
    assert!(s.wins_on(2, t("R")), "the hand still waits on R");
    assert_eq!(s.claim_score(2), None);
    play(&mut s, &[(2, Pass)]);
    // A different tile is still winnable (§4.4.3).
    play(&mut s, &[(1, Discard(t("N")))]);
    assert_eq!(s.legal_actions_vec(), vec![Win, Pung, Pass]);
    play(&mut s, &[(2, Pass)]);
    // Seat 2's own draw follows at once and ends the lock (§4.4.5).
    assert_eq!(s.current_player(), Some(2));
    assert!(s.passed_win_lock(2).is_empty());
    play(&mut s, &[(2, Discard(t("E"))), (3, Discard(t("N")))]);
    assert_eq!(s.legal_actions_vec(), vec![Win, Pung, Pass]);
    play(&mut s, &[(2, Win)]);
    assert_eq!(s.result().unwrap().wins[0].seat, 2);
}

#[test]
fn the_seats_own_claim_ends_the_lock_4_4_5() {
    let mut s = waiting_on_r_or_n(no_flowers());
    play(&mut s, &[(0, Discard(t("R"))), (2, Pung)]);
    assert_eq!(s.current_player(), Some(2));
    assert!(s.passed_win_lock(2).is_empty());
}

#[test]
fn without_the_rule_play_is_unchanged() {
    let rules = RuleSet {
        passed_win_lock: false,
        ..no_flowers()
    };
    let mut s = waiting_on_r_or_n(rules);
    assert!(s.passed_win_lock(2).is_empty());
    play(&mut s, &[(0, Discard(t("R")))]);
    assert_eq!(s.legal_actions_vec(), vec![Win, Pung, Pass]);
    play(&mut s, &[(2, Win)]);
    assert_eq!(s.result().unwrap().wins[0].seat, 2);
}

/// Seats 1 and 2 both win on seat 3's 5p (as in `scenarios.rs`).
fn double_win_window() -> State {
    let mut s = deal(
        no_flowers(),
        [
            "9s 111m 222m 333m 444m 5s",
            "11p 234p 678p 789p 46p",
            "5p RRR GGG 123s 456s",
            "5p E E E S S S W W W N N N",
        ],
        "9m Wh 1m",
        "",
    );
    play(
        &mut s,
        &[
            (0, Discard(t("9s"))),
            (1, Discard(t("9m"))),
            (2, Discard(t("Wh"))),
            (3, Discard(t("5p"))),
        ],
    );
    s
}

#[test]
fn every_seat_that_passes_is_locked_and_only_those_4_4_6() {
    let mut s = double_win_window();
    play(&mut s, &[(1, Pass), (2, Pass)]);
    assert_eq!(s.passed_win_lock(1), vec![t("5p")]);
    assert_eq!(s.passed_win_lock(2), vec![t("5p")]);
    assert!(s.passed_win_lock(0).is_empty() && s.passed_win_lock(3).is_empty());
}

#[test]
fn a_locked_seat_does_not_stop_another_from_winning_4_4_2() {
    // The same window, rebuilt as a position in which seat 1 already passed 5p.
    let s = double_win_window();
    let mut spec = s.to_position().unwrap();
    spec.passed_wins[1] = vec![t("5p")];
    spec.to_move = 1;
    assert_eq!(
        State::from_position(*s.rules(), &spec).unwrap_err(),
        PositionError::NoDecision,
        "a locked seat with nothing else to claim is not asked"
    );
    spec.to_move = 2;
    let mut s = State::from_position(*s.rules(), &spec).unwrap();
    assert_eq!(s.legal_actions_vec(), vec![Win, Pass]);
    play(&mut s, &[(2, Win)]);
    let r = s.result().unwrap();
    assert_eq!(r.wins.iter().map(|w| w.seat).collect::<Vec<_>>(), [2]);
}

#[test]
fn passing_a_robbing_win_locks_the_tile_4_4_1() {
    // `scenarios.rs`'s robbing hand: seat 3 may rob seat 1's added 5p kong.
    let mut s = deal(
        no_flowers(),
        [
            "Wh 111m 222m 333m 444m 9m",
            "55p 5m 6m 7m 8m 1p 2p 3p 7p 8p 9p E",
            "5p 666m 777m 888m 99m 3p",
            "46p RRR 123s 456s 99s",
        ],
        "N E 1p N W 5p S",
        "9p",
    );
    play(
        &mut s,
        &[
            (0, Discard(t("Wh"))),
            (1, Discard(t("N"))),
            (2, Discard(t("5p"))),
            (3, Pass),
            (1, Pung),
            (1, Discard(t("E"))),
            (2, Discard(t("1p"))),
            (3, Discard(t("N"))),
            (0, Discard(t("W"))),
            (1, AddedKong(t("5p"))),
        ],
    );
    assert_eq!(s.legal_actions_vec(), vec![Win, Pass]);
    play(&mut s, &[(3, Pass)]);
    assert_eq!(s.passed_win_lock(3), vec![t("5p")]);
    // Seat 1 draws its replacement and plays on; seat 3's draw ends the lock.
    play(&mut s, &[(1, Discard(t("9p"))), (2, Discard(t("S")))]);
    assert_eq!(s.current_player(), Some(3));
    assert!(s.passed_win_lock(3).is_empty());
}

#[test]
fn the_lock_round_trips_through_a_position() {
    let s = waiting_on_r_or_n(no_flowers());
    let spec = s.to_position().unwrap();
    assert_eq!(spec.passed_wins[2], vec![t("R")]);
    let rebuilt = State::from_position(*s.rules(), &spec).unwrap();
    assert_eq!(rebuilt.passed_win_lock(2), vec![t("R")]);
    assert_eq!(rebuilt.to_position(), Some(spec.clone()));
    // A lock without the rule, or on the player on turn, is an error.
    let off = RuleSet {
        passed_win_lock: false,
        ..*s.rules()
    };
    assert!(matches!(
        State::from_position(off, &spec),
        Err(PositionError::BadPassedWin { seat: 2, .. })
    ));
    let mut on_turn = spec.clone();
    on_turn.passed_wins[0] = vec![t("E")];
    assert!(matches!(
        State::from_position(*s.rules(), &on_turn),
        Err(PositionError::BadPassedWin { seat: 0, .. })
    ));
}

/// Plays random hands in which seats often pass wins; at every decision, a
/// re-deal with the true hidden tiles re-derives every seat's lock exactly,
/// and a random re-deal keeps the observer's own lock.
#[test]
fn a_redeal_re_derives_hidden_locks() {
    let mut locks_seen = 0;
    for seed in 0..1500 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let rules = RuleSet {
            min_faan: rng.gen_range(0..=1),
            ..RuleSet::default()
        };
        let mut s = State::new_deal(rules, &mut rng);
        let picker = mahjong_games::UniformRandom;
        while !s.is_terminal() {
            let me = s.current_player().unwrap();
            let hands: [TileCounts; 4] = std::array::from_fn(|p| *s.hand(p));
            let same = s.redeal(me, &hands, s.wall().to_vec());
            for p in 0..4 {
                assert_eq!(same.passed_win_lock(p), s.passed_win_lock(p), "seed {seed}");
            }
            let other = s.determinize(me, &mut rng);
            assert_eq!(other.passed_win_lock(me), s.passed_win_lock(me));
            locks_seen += (0..4).filter(|&p| !s.passed_win_lock(p).is_empty()).count();
            let legal = s.legal_actions_vec();
            // Pass a win two times in three, so locks come up often.
            let a = if legal.contains(&Win) && rng.gen_bool(0.33) {
                Win
            } else if legal.contains(&Pass) && legal.contains(&Win) {
                Pass
            } else {
                picker.choose(&s, &legal, &mut rng)
            };
            s.apply(a);
        }
    }
    assert!(locks_seen > 40, "only {locks_seen} locks seen");
}
