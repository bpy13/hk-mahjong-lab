//! Turn flow, claims, kongs, payment and liability (`docs/rules/hk.md` §3–§9),
//! each on a hand built to exercise one rule.

mod common;
use common::{deal, no_flowers, play, t};
use mahjong_games::Game;
use mahjong_hk::{Action, MultipleWinners, Pattern, PositionPhase, PositionSpec, RuleSet, State};
use mahjong_tiles::{parse_tiles, TileCounts};

use Action::*;

/// A dealer hand that holds an `R` and can claim nothing in these scenarios.
const DEALER_R: &str = "R 9m 11m 33m 5m 7m 9s E SS 4p 8p";

#[test]
fn discard_win_discarder_pays_one_share_8_2() {
    // Seat 2 waits on R or N; R gives 混一色 3 + 中 1 + 門前清 1 = 5 faan = 24.
    let mut s = deal(
        no_flowers(),
        [
            DEALER_R,
            "1s 2s 3s 4s 5s 6s 7s 8s E 5m 6m W N",
            "123p 456p 789p RR NN",
            "9s 9s W W G G Wh Wh 1p 2p 3p 6s 6s",
        ],
        "R",
        "",
    );
    // Not the dealer's first discard (that would be 地糊, #30).
    play(&mut s, &[(0, Discard(t("9m"))), (1, Discard(t("R")))]);
    // Seat 2 may win or pung; nobody else has an option.
    assert_eq!(s.current_player(), Some(2));
    assert_eq!(s.legal_actions_vec(), vec![Win, Pung, Pass]);
    play(&mut s, &[(2, Win)]);
    let r = s
        .result()
        .unwrap_or_else(|| panic!("{s:?} {:?}", s.current_player()));
    assert_eq!(r.wins.len(), 1);
    assert_eq!(r.wins[0].score.faan, 5);
    assert_eq!(r.deltas, [0, -24, 24, 0]);
}

#[test]
fn self_drawn_win_everyone_pays_8_2() {
    // Seat 2 draws R: 自摸 + 門前清 + 混一色 + 中 = 6 faan = 32 from each opponent.
    let mut s = deal(
        no_flowers(),
        [
            DEALER_R,
            "1s 2s 3s 4s 5s 6s 7s 8s E 5m 6m W N",
            "123p 456p 789p RR NN",
            "9s 9s W W G G Wh Wh 1p 2p 3p 6s 6s",
        ],
        "9m R",
        "",
    );
    play(&mut s, &[(0, Discard(t("9m"))), (1, Discard(t("9m")))]);
    assert_eq!(s.current_player(), Some(2));
    assert!(s.legal_actions_vec().contains(&SelfDrawnWin));
    play(&mut s, &[(2, SelfDrawnWin)]);
    assert_eq!(s.result().unwrap().deltas, [-32, -32, 96, -32]);
}

#[test]
fn minimum_faan_blocks_a_chicken_hand_6() {
    // Seat 1 is complete with 9p but would score only 門前清 (1 faan).
    let mut s = deal(
        no_flowers(),
        [
            "9m 9p 11m 44m 77m 11s 66p 33p",
            "123m 456p 789s 234s 9p",
            "5m 6m 7m 1p 2p 3p E E S S W W N",
            "5s 6s 7s 8s G G G R R R Wh Wh Wh",
        ],
        "2s E 9p 3s",
        "",
    );
    assert!(mahjong_hk::hand::waits(s.hand(1), 0).contains(&t("9p")));
    play(
        &mut s,
        &[
            (0, Discard(t("9m"))),
            (1, Discard(t("2s"))),
            (2, Discard(t("E"))),
        ],
    );
    play(&mut s, &[(3, Discard(t("9p")))]);
    // No claim window: the dealer has already drawn and is to move.
    assert_eq!(s.current_player(), Some(0));
    assert!(!s.in_claim());
    assert_eq!(s.concealed_count(0), 14);
}

#[test]
fn pung_beats_chow_and_claimer_discards_without_drawing_4_1() {
    let mut s = deal(
        no_flowers(),
        [
            "3m 111p 222p 333p 444p 9s",
            "4m 5m 1s 2s 3s 4s 5s 6s 7s 8s E E S",
            "3m 3m 5p 6p 7p 8p 9p W W N N G G",
            "Wh Wh Wh R R R 6m 7m 8m 9m 9m 9m 2m",
        ],
        "",
        "",
    );
    let wall = s.wall_remaining();
    play(&mut s, &[(0, Discard(t("3m")))]);
    assert_eq!(s.current_player(), Some(1));
    assert!(s.legal_actions_vec().contains(&Chow(t("3m"))));
    play(&mut s, &[(1, Chow(t("3m"))), (2, Pung)]);
    // Seat 2 got the pung, has not drawn, and must discard.
    assert_eq!(s.current_player(), Some(2));
    assert_eq!(s.wall_remaining(), wall);
    assert_eq!(s.melds(2).len(), 1);
    assert!(s
        .legal_actions_vec()
        .iter()
        .all(|a| matches!(a, Discard(_))));
    // Seat 3 draws next (seat 1's turn is skipped... it is seat 3's turn after seat 2).
    play(&mut s, &[(2, Discard(t("9p")))]);
    assert_eq!(s.current_player(), Some(3));
}

/// Seat 1 and seat 2 both win on seat 3's 5p.
fn double_win(rules: mahjong_hk::RuleSet) -> mahjong_hk::State {
    let mut s = deal(
        rules,
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
        ],
    );
    // Seat 1: 清一色 7 + 平糊 1 + 門前清 1 = 9 → 96. Seat 2: 中 + 發 + 門前清 = 3 → 8.
    play(&mut s, &[(3, Discard(t("5p"))), (1, Win), (2, Win)]);
    s
}

#[test]
fn several_winners_all_win_4_1_3() {
    let s = double_win(no_flowers());
    let r = s.result().unwrap();
    assert_eq!(r.wins.iter().map(|w| w.seat).collect::<Vec<_>>(), [1, 2]);
    assert_eq!(r.deltas, [0, 96, 8, -104]);
}

#[test]
fn head_bump_variant_4_1_3() {
    let s = double_win(RuleSet {
        multiple_winners: MultipleWinners::HeadBump,
        ..no_flowers()
    });
    assert_eq!(s.result().unwrap().deltas, [0, 96, 0, -96]);
}

#[test]
fn earthly_hand_on_the_dealers_first_discard_7_4() {
    let mut s = deal(
        no_flowers(),
        [
            "5p 111m 222m 333m 444m 9s",
            "11p 234p 678p 789p 46p",
            "1s 2s 3s 4s 5s 6s 7s 8s W W N N G",
            "E E E S S S 6m 7m 8m 9m 9m 9m 2m",
        ],
        "",
        "",
    );
    play(&mut s, &[(0, Discard(t("5p"))), (1, Win)]);
    let w = &s.result().unwrap().wins[0];
    assert!(w.score.has(Pattern::Earthly));
    assert_eq!(s.result().unwrap().deltas, [-128, 128, 0, 0]);
}

#[test]
fn kong_replacement_win_5_4() {
    // The dealer konges 1m and draws 9s from the back: 自摸 + 槓上開花 + 門前清 = 3.
    let mut s = deal(
        no_flowers(),
        [
            "1111m 234p 567p 888s 9s",
            "2m 3m 4m 5m 6m 7m 8m 1s 2s 3s 4s E E",
            "S S W W N N G G R R Wh Wh 1p",
            "9m 9m 9m 2p 3p 4p 5p 6p 7p 8p 9p 5s 6s",
        ],
        "",
        "9s",
    );
    assert!(s.legal_actions_vec().contains(&ConcealedKong(t("1m"))));
    play(&mut s, &[(0, ConcealedKong(t("1m"))), (0, SelfDrawnWin)]);
    let w = &s.result().unwrap().wins[0];
    assert!(w.score.has(Pattern::KongReplacement) && w.score.has(Pattern::SelfDrawn));
    assert!(
        !w.score.has(Pattern::Heavenly),
        "a kong ends the dealer's first turn"
    );
    assert_eq!(w.score.faan, 3);
    assert_eq!(s.result().unwrap().deltas, [24, -8, -8, -8]);
}

#[test]
fn double_kong_win_7_1_row_8() {
    let mut s = deal(
        no_flowers(),
        [
            "1111m 2222p 567p 888s",
            "2m 3m 4m 5m 6m 7m 8m 1s 2s 3s 4s E E",
            "S S W W N N G G R R Wh Wh 1p",
            "9m 9m 9m 3p 3p 4p 4p 8p 9p 5s 6s 7s 7s",
        ],
        "",
        "9s 9s",
    );
    play(
        &mut s,
        &[
            (0, ConcealedKong(t("1m"))),
            (0, ConcealedKong(t("2p"))),
            (0, SelfDrawnWin),
        ],
    );
    let w = &s.result().unwrap().wins[0];
    assert!(w.score.has(Pattern::DoubleKong) && !w.score.has(Pattern::SelfDrawn));
    assert_eq!(w.score.faan, 9); // 槓上槓 8 + 門前清 1
}

#[test]
fn robbing_an_added_kong_5_2() {
    // Seat 1 pungs 5p and later adds the fourth; seat 3 robs it. Seat 3's hand is
    // worth only 2 faan on a discard (中 + 門前清), so 搶槓 is what makes it legal.
    let mut s = deal(
        no_flowers(),
        [
            "Wh 111m 222m 333m 444m 9m",
            "55p 5m 6m 7m 8m 1p 2p 3p 7p 8p 9p E",
            "5p 666m 777m 888m 99m 3p",
            "46p RRR 123s 456s 99s",
        ],
        "N E 1p N W 5p",
        "",
    );
    play(
        &mut s,
        &[
            (0, Discard(t("Wh"))),
            (1, Discard(t("N"))),
            (2, Discard(t("5p"))),
        ],
    );
    // Seat 3 could chow; seat 1 can pung. Seat 3 cannot win on a plain discard.
    assert_eq!(s.current_player(), Some(3));
    assert!(!s.legal_actions_vec().contains(&Win));
    play(&mut s, &[(3, Pass), (1, Pung), (1, Discard(t("E")))]);
    play(
        &mut s,
        &[
            (2, Discard(t("1p"))),
            (3, Discard(t("N"))),
            (0, Discard(t("W"))),
        ],
    );
    play(&mut s, &[(1, AddedKong(t("5p")))]);
    assert!(s.in_rob());
    assert_eq!(s.legal_actions_vec(), vec![Win, Pass]);
    play(&mut s, &[(3, Win)]);
    let w = &s.result().unwrap().wins[0];
    assert!(w.score.has(Pattern::RobbingKong));
    assert_eq!(s.result().unwrap().deltas, [0, -8, 0, 8]);
    // The kong is undone: seat 1 is left with the pung.
    assert_eq!(s.melds(1)[0].kind, mahjong_hk::MeldKind::Pung);
}

#[test]
fn only_thirteen_orphans_robs_a_concealed_kong_5_1() {
    let mut s = deal(
        no_flowers(),
        [
            "9999s 2m 3m 4m 5m 6m 7m 2p 3p 4p 5p",
            "1m 9m 1p 9p 1s E S W N Wh G R R",
            "78s GGG 123m 456m 99m",
            "6p 6p 6p 7p 7p 7p 8p 8p 8p 2s 2s 2s 3s",
        ],
        "",
        "",
    );
    play(&mut s, &[(0, ConcealedKong(t("9s")))]);
    // Seat 2 waits on 9s too, but only seat 1's thirteen orphans may rob.
    assert!(s.in_rob());
    assert_eq!(s.current_player(), Some(1));
    play(&mut s, &[(1, Win)]);
    let w = &s.result().unwrap().wins[0];
    assert!(w.score.has(Pattern::ThirteenOrphans));
    assert_eq!(s.result().unwrap().deltas, [-128, 128, 0, 0]);
}

#[test]
fn twelve_tile_liability_8_3() {
    // Seat 1 pungs four discards; seat 2 fed the fourth, so seat 2 pays for the
    // whole table when seat 1 self-draws (對對糊 3 + 自摸 1 = 4 faan = 16 × 3).
    let mut s = deal(
        no_flowers(),
        [
            "1s 9p 9p 1p 2p 3p 4p 4m 4m 8m 8m 9s 9s Wh",
            "11s 22s 33s 44s 9m E S W N",
            "2s 3s 4s 5m 6m 7m 1p 2p 3p 5p 6p 7p R",
            "1m 2m 3m 4m 5m 6m 7m 8m 9m Wh Wh G G",
        ],
        "8p 8p 8p 1s 2s 3s 9m",
        "",
    );
    play(
        &mut s,
        &[(0, Discard(t("1s"))), (1, Pung), (1, Discard(t("E")))],
    );
    play(
        &mut s,
        &[(2, Discard(t("2s"))), (1, Pung), (1, Discard(t("S")))],
    );
    play(
        &mut s,
        &[(2, Discard(t("3s"))), (1, Pung), (1, Discard(t("W")))],
    );
    assert_eq!(s.liable_for(1), None);
    play(&mut s, &[(2, Discard(t("4s"))), (1, Pung)]);
    assert_eq!(s.liable_for(1), Some(2));
    play(
        &mut s,
        &[
            (1, Discard(t("N"))),
            (2, Discard(t("1s"))),
            (3, Discard(t("2s"))),
            (0, Discard(t("3s"))),
        ],
    );
    play(&mut s, &[(1, SelfDrawnWin)]);
    let r = s.result().unwrap();
    assert_eq!(r.wins[0].liable, Some(2));
    assert_eq!(r.deltas, [0, 48, -48, 0]);
}

#[test]
fn initial_flower_replacement_3_3() {
    let rules = RuleSet::default();
    let s = deal(
        rules,
        [
            "F1 111m 222m 333m 444m 9m",
            "1s 2s 3s 4s 5s 6s 7s 8s E 5m 6m W N",
            "123p 456p 789p RR NN",
            "9s 9s W W G G Wh Wh 1p 2p 3p 6s 6s",
        ],
        "",
        "R",
    );
    assert_eq!(s.bonus(0), 0b0000_0001);
    assert_eq!(s.concealed_count(0), 14);
    assert_eq!(s.hand(0)[t("R")], 1);
}

/// Seat 1's hand: six bonus tiles and seven playing tiles.
const SIX_FLOWERS: &str = "F1 F2 F3 F4 S1 S2 5s 6s 7s 8s E E W";

#[test]
fn seven_flowers_wins_at_once_paid_by_all_7_5_32() {
    // §7.5 #32 花糊, §8.2: seat 1 draws its seventh bonus tile in play and wins
    // at once, 3 faan (8 points) from each other seat.
    let hands = [
        "9s 111m 222m 333m 444m 9m",
        SIX_FLOWERS,
        "123p 456p 789p RR NN",
        "9s 9s W W G G Wh Wh 1p 2p 3p 6s 6s",
    ];
    let mut s = deal(RuleSet::default(), hands, "S3", "1s 2s 3s 4s 5m 6m");
    assert_eq!(s.bonus(1), 0b0011_1111);
    play(&mut s, &[(0, Discard(t("9m")))]);
    let r = s.result().expect("the seventh flower ends the hand");
    assert_eq!(r.wins.len(), 1);
    let w = &r.wins[0];
    assert_eq!(w.seat, 1);
    assert!(w.self_drawn && w.win_tile.is_none());
    assert_eq!(w.score.patterns, vec![(Pattern::SevenFlowers, 3)]);
    assert_eq!((w.score.faan, w.score.limit), (3, false));
    assert_eq!(r.deltas, [-8, 24, -8, -8]);
}

#[test]
fn seven_flowers_in_the_deal_wins_at_once_7_5_32() {
    // §3.3, #32: seven bonus tiles set aside in the deal end the hand there.
    let hands = [
        "9s 111m 222m 333m 444m 9m",
        "F1 F2 F3 F4 S1 S2 S3 5s 6s 7s 8s E E",
        "123p 456p 789p RR NN",
        "9s 9s W W G G Wh Wh 1p 2p 3p 6s 6s",
    ];
    let s = deal(RuleSet::default(), hands, "", "1s 2s 3s 4s 5m 6m 7m");
    let r = s.result().expect("ended in the deal");
    assert_eq!(r.wins[0].seat, 1);
    assert!(r.wins[0].score.has(Pattern::SevenFlowers));
    assert_eq!(r.deltas, [-8, 24, -8, -8]);
}

#[test]
fn all_eight_flowers_pays_8_faan_7_5_31() {
    // §3.3, #31 大花糊: all eight bonus tiles are set aside together in the deal:
    // 8 faan (64 points) from each other seat. Not a limit hand any more.
    let hands = [
        "9s 111m 222m 333m 444m 9m",
        "F1 F2 F3 F4 S1 S2 S3 S4 5s 6s 7s 8s E",
        "123p 456p 789p RR NN",
        "9s 9s W W G G Wh Wh 1p 2p 3p 6s 6s",
    ];
    let s = deal(RuleSet::default(), hands, "", "1s 2s 3s 4s 5m 6m 7m 8m");
    let r = s.result().expect("ended in the deal");
    let w = &r.wins[0];
    assert_eq!(w.seat, 1);
    assert_eq!(w.score.patterns, vec![(Pattern::EightImmortals, 8)]);
    assert_eq!((w.score.faan, w.score.limit), (8, false));
    assert_eq!(r.deltas, [-64, 192, -64, -64]);
}

#[test]
fn no_robbing_flower_win_7_5() {
    // §7.5: the old 七搶一 is gone. Seat 1 holds seven bonus tiles (a position,
    // since in play the seventh would have won at once); seat 2 declares a
    // concealed kong and draws the eighth, S4, as its replacement. Nobody wins.
    let rules = RuleSet::default();
    let c = |s: &str| TileCounts::from_tiles(&parse_tiles(s).unwrap());
    let spec = PositionSpec {
        hands: [
            c("123456789m 1234s"),
            c("56789s 1234s EEE S"),
            c("1111p 2345678p 9p WW"),
            c("NNN WhWhWh GGG 5678m"),
        ],
        melds: Default::default(),
        bonus: [
            vec![],
            parse_tiles("F1 F2 F3 F4 S1 S2 S3").unwrap(),
            vec![],
            vec![],
        ],
        river: vec![],
        river_from_hand: vec![],
        wall_remaining: 144 - 13 * 3 - 14 - 7,
        to_move: 2,
        phase: PositionPhase::Turn,
        last_drawn: Some(t("W")),
        kong_chain: 0,
        liable: [None; 4],
        passed_wins: Default::default(),
    };
    let mut s = State::from_position(rules, &spec).expect("valid position");
    play(&mut s, &[(2, ConcealedKong(t("1p")))]);
    assert!(s.result().is_none(), "{:?}", s.result());
    assert_eq!(s.bonus(2), 0b1000_0000, "seat 2 keeps the eighth");
    assert_eq!(s.current_player(), Some(2));
}
