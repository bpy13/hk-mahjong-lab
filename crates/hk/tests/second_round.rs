//! The rule points of the second and last rounds of the line-by-line check
//! (ADR 0006 and 0007, plan 2026-11-16 Task 32): 食換 (§4.1.4), passing a
//! seven-flower win (§7.5.1), flower wins adding 海底 and 槓上 (§7.5.2), 九子連環
//! by its final shape (#26), 包大三元 (§8.3.4), 人糊 (#33), the 12-tile pung
//! exception (§8.3.3), and the three kept rules (points 5, 7 and 20).

mod common;
use common::{counts, deal, meld, no_flowers, play, t};
use mahjong_games::Game;
use mahjong_hk::{
    score_win, Action, Liability, Meld, Pattern, PositionError, PositionPhase, PositionSpec,
    RuleSet, Score, State, WinContext,
};
use mahjong_tiles::{Tile, TileCounts};

use Action::*;

/// A dealer hand that holds an `R` and a `4p` and claims nothing here.
const DEALER_R: &str = "R 9m 11m 33m 5m 7m 9s E SS 4p 8p";

/// Seat 1's hand: six bonus tiles and seven playing tiles.
const SIX_FLOWERS: &str = "F1 F2 F3 F4 S1 S2 5s 6s 7s 8s E E W";

const OTHERS: [&str; 2] = ["123p 456p 789p RR NN", "9s 9s W W G G Wh Wh 1p 2p 3p 6s 6s"];

fn discards(s: &State) -> Vec<Tile> {
    s.legal_actions_vec()
        .into_iter()
        .filter_map(|a| match a {
            Discard(t) => Some(t),
            _ => None,
        })
        .collect()
}

// ----- §4.1.4 食換 (point 13) -----

#[test]
fn chow_bans_the_claimed_tile_and_the_re_forming_tile_4_1_4() {
    // Seat 1 may chow the dealer's 4p three ways. Claim 4 with 5-6: not 4p and
    // not 7p; with 2-3: not 4p and not 1p; with 3-5: only 4p is banned.
    let s = deal(
        no_flowers(),
        [
            DEALER_R,
            "1p 2p 3p 5p 6p 7p 9s 9s E E W W N",
            "1s 2s 3s 4s 5s 6s 7s 8s 9s G G Wh Wh",
            "2m 4m 6m 6m 8m 8m 3p 9p 9p R R G Wh",
        ],
        "",
        "",
    );
    let mut s0 = s.clone();
    play(&mut s0, &[(0, Discard(t("4p")))]);
    assert_eq!(
        s0.legal_actions_vec(),
        vec![Chow(t("2p")), Chow(t("3p")), Chow(t("4p")), Pass]
    );
    for (low, banned, allowed) in [("4p", "7p", "1p"), ("2p", "1p", "7p"), ("3p", "", "7p")] {
        let mut s = s0.clone();
        play(&mut s, &[(1, Chow(t(low)))]);
        let d = discards(&s);
        if !banned.is_empty() {
            assert!(!d.contains(&t(banned)), "chow {low}: {d:?}");
            assert!(s.discard_ban() & (1 << t(banned).index()) != 0);
        }
        assert!(d.contains(&t(allowed)), "chow {low}: {d:?}");
        assert!(s.discard_ban() & (1 << t("4p").index()) != 0);
        // The ban lasts only for this discard.
        let first = d[0];
        play(&mut s, &[(1, Discard(first))]);
        assert_eq!(s.discard_ban(), 0);
    }
}

#[test]
fn pung_bans_the_claimed_tile_4_1_4() {
    // Seat 2 holds three 4p and pungs the dealer's 4p: it may not throw its
    // fourth 4p at once.
    let mut s = deal(
        no_flowers(),
        [
            DEALER_R,
            "1s 2s 3s 4s 5s 6s 7s 8s E 5m 6m W N",
            "4p 4p 4p 2m 4m 6m 9s 9s G G Wh Wh N",
            "1p 2p 3p 5p 6p 7p 8p 8p W W S R R",
        ],
        "",
        "",
    );
    play(&mut s, &[(0, Discard(t("4p")))]);
    assert_eq!(s.legal_actions_vec(), vec![Pung, ClaimKong, Pass]);
    play(&mut s, &[(2, Pung)]);
    assert_eq!(s.hand(2)[t("4p")], 1);
    assert!(!discards(&s).contains(&t("4p")));
    assert!(discards(&s).contains(&t("2m")));
}

/// Tiles in a position's hands, melds, unclaimed river and bonus tiles.
fn wall_remaining(rules: &RuleSet, spec: &PositionSpec) -> usize {
    let total = if rules.flowers { 144 } else { 136 };
    let hands: usize = spec.hands.iter().map(TileCounts::total).sum();
    let melds: usize = spec.melds.iter().flatten().map(|m| m.tiles().len()).sum();
    let river = spec.river.iter().filter(|r| !r.2).count();
    let bonus: usize = spec.bonus.iter().map(Vec::len).sum();
    total - hands - melds - river - bonus
}

/// Seat 1 shows three pungs and holds four tiles; the dealer has just thrown
/// 4p; seat 2 holds a pair of 4p.
fn last_four(seat1: &str, to_move: usize) -> Result<State, PositionError> {
    let rules = no_flowers();
    let pungs: Vec<Meld> = ["pung 111m", "pung 999m", "pung EEE"]
        .iter()
        .map(|m| meld(m))
        .collect();
    let mut spec = PositionSpec {
        hands: [
            counts("1s 2s 3s 4s 5s 6s 7s 8s 9s S S W W"),
            counts(seat1),
            counts("4p 4p 2m 3m 5m 6m 8m 1p 2p 9p 9p G G"),
            counts("Wh Wh Wh R R R N N 3p 8p 2m 7m 7m"),
        ],
        melds: [vec![], pungs, vec![], vec![]],
        bonus: Default::default(),
        river: vec![
            (3, t("1m"), true),
            (3, t("9m"), true),
            (3, t("E"), true),
            (0, t("4p"), false),
        ],
        river_from_hand: vec![],
        wall_remaining: 0,
        to_move,
        phase: PositionPhase::Claim {
            tile: t("4p"),
            from: 0,
        },
        last_drawn: None,
        kong_chain: 0,
        liable: [None; 4],
        passed_wins: Default::default(),
    };
    spec.wall_remaining = wall_remaining(&rules, &spec);
    State::from_position(rules, &spec)
}

#[test]
fn a_claim_that_would_leave_no_legal_discard_is_not_offered_4_1_4() {
    // The orchestrator's call (2026-11-23): seat 1 holds 4p 5p 6p 7p with three
    // open sets. Chowing the 4p with 5-6 would leave 4p and 7p, both banned, so
    // the chow is not offered; seat 1 is not even asked.
    let mut s = last_four("4p 5p 6p 7p", 2).expect("seat 2 may pung");
    assert_eq!(s.legal_actions_vec(), vec![Pung, Pass]);
    assert!(matches!(
        last_four("4p 5p 6p 7p", 1),
        Err(PositionError::NoDecision)
    ));
    play(&mut s, &[(2, Pass)]);
    assert_eq!((s.current_player(), s.phase_name()), (Some(1), "turn"));
    // With 8p instead of 7p the 8p may be thrown after the chow: offered.
    let s = last_four("4p 5p 6p 8p", 1).expect("seat 1 may chow");
    assert_eq!(s.legal_actions_vec(), vec![Chow(t("4p")), Pass]);
}

// ----- §7.5.1 passing a seven-flower win (point 9) -----

#[test]
fn a_seven_flower_win_may_be_passed_7_5_1() {
    // Seat 1 draws its seventh bonus tile, passes 花糊 (code 103), replaces it
    // with 7m and plays on.
    let hands = [
        "9s 111m 222m 333m 444m 9m",
        SIX_FLOWERS,
        OTHERS[0],
        OTHERS[1],
    ];
    let mut s = deal(RuleSet::default(), hands, "S3", "1s 2s 3s 4s 5m 6m 7m");
    play(&mut s, &[(0, Discard(t("9m")))]);
    assert!(s.in_flower());
    assert_eq!(s.flower_score().unwrap().faan, 3);
    assert_eq!(Pass.code(), 103);
    assert_eq!(SelfDrawnWin.code(), 34);
    play(&mut s, &[(1, Pass)]);
    assert!(s.result().is_none());
    assert_eq!((s.current_player(), s.phase_name()), (Some(1), "turn"));
    assert_eq!(s.bonus(1).count_ones(), 7);
    assert_eq!(s.last_drawn(), Some(t("7m")));
    assert_eq!(s.concealed_count(1), 14);
    assert!(discards(&s).contains(&t("7m")));
}

#[test]
fn after_a_passed_seven_the_eighth_wins_at_once_7_5_31() {
    // As above, but the replacement of the seventh is S4, the eighth bonus
    // tile: 大花糊 at once, 10 faan, 128 from each other seat.
    let hands = [
        "9s 111m 222m 333m 444m 9m",
        SIX_FLOWERS,
        OTHERS[0],
        OTHERS[1],
    ];
    let mut s = deal(RuleSet::default(), hands, "S3", "1s 2s 3s 4s 5m 6m S4");
    play(&mut s, &[(0, Discard(t("9m"))), (1, Pass)]);
    let r = s.result().expect("the eighth bonus tile ends the hand");
    assert_eq!(
        r.wins[0].score.patterns,
        vec![(Pattern::EightImmortals, 10)]
    );
    assert_eq!(r.deltas, [-128, 384, -128, -128]);
}

#[test]
fn a_seven_flower_win_in_the_deal_may_be_passed_3_3() {
    // Seven bonus tiles set aside in the deal: seat 1 passes, draws its seven
    // replacements, and the deal goes on to the dealer's first turn.
    let hands = [
        "9s 111m 222m 333m 444m 9m",
        "F1 F2 F3 F4 S1 S2 S3 5s 6s 7s 8s E E",
        OTHERS[0],
        OTHERS[1],
    ];
    let mut s = deal(RuleSet::default(), hands, "", "1s 2s 3s 4s 5m 6m 7m");
    assert!(s.in_flower());
    assert_eq!(s.current_player(), Some(1));
    play(&mut s, &[(1, Pass)]);
    assert_eq!((s.current_player(), s.phase_name()), (Some(0), "turn"));
    assert_eq!(s.concealed_count(1), 13);
    assert_eq!(s.hand(1)[t("7m")], 1);
    assert_eq!(s.wall_remaining(), 144 - 53 - 7);
}

// ----- §7.5.2 flower wins add 海底 and 槓上 (point 10) -----

#[test]
fn a_flower_win_on_a_kong_replacement_adds_kong_faan_7_5_2() {
    // The dealer's kong replacement is its seventh bonus tile: 花糊 3 + 槓上 1 =
    // 4 faan, 16 from each other seat.
    let mut s = deal(
        RuleSet::default(),
        [
            "F1 F2 F3 F4 S1 S2 1111m 2p 3p 4p 5p",
            "2m 3m 4m 5m 6m 7m 8m 1s 2s 3s 4s E E",
            "S S W W N N G G R R Wh Wh 1p",
            "9m 9m 9m 2p 3p 4p 5p 6p 7p 8p 9p 5s 6s",
        ],
        "",
        "9s 9s 8s 8s 7s 7s S3",
    );
    assert_eq!(s.bonus(0).count_ones(), 6);
    play(&mut s, &[(0, ConcealedKong(t("1m")))]);
    assert!(s.in_flower());
    play(&mut s, &[(0, SelfDrawnWin)]);
    let w = &s.result().unwrap().wins[0];
    assert_eq!(
        w.score.patterns,
        vec![(Pattern::SevenFlowers, 3), (Pattern::KongReplacement, 1)]
    );
    assert_eq!(w.score.faan, 4);
    assert_eq!(s.result().unwrap().deltas, [48, -16, -16, -16]);
}

#[test]
fn flower_win_scores_with_last_tile_and_double_kong_7_5_2() {
    let r = RuleSet::default();
    let seven = 0b0111_1111;
    let s = Score::flower_win_in(seven, true, 0, &r).unwrap();
    assert_eq!(
        s.patterns,
        vec![(Pattern::SevenFlowers, 3), (Pattern::LastTile, 1)]
    );
    assert_eq!(s.faan, 4);
    let s = Score::flower_win_in(seven, false, 2, &r).unwrap();
    assert!(s.has(Pattern::DoubleKong));
    assert_eq!(s.faan, 10);
    // No 自摸 is added (HKMA 花糊牌例 1).
    assert!(!Score::flower_win_in(seven, true, 1, &r)
        .unwrap()
        .has(Pattern::SelfDrawn));
    assert_eq!(Score::flower_win(seven, &r).unwrap().faan, 3);
}

// ----- #26 九子連環 by its final shape on a self-draw (point 15) -----

#[test]
fn nine_gates_by_final_shape_on_a_self_draw_26() {
    let rules = no_flowers();
    let hand = counts("11123456789999m");
    let tsumo = WinContext {
        seat: 1,
        win_tile: Some(t("1m")),
        self_drawn: true,
        ..WinContext::default()
    };
    let s = score_win(&hand, &[], &tsumo, &rules).unwrap();
    assert!(s.has(Pattern::NineGates) && s.limit);
    // On a discard the 13 tiles before the win must be the nine-sided wait:
    // 1123456789999m is not, so it is a full flush, not 九子連環.
    let ron = WinContext {
        self_drawn: false,
        ..tsumo
    };
    let s = score_win(&hand, &[], &ron, &rules).unwrap();
    assert!(!s.has(Pattern::NineGates) && s.has(Pattern::FullFlush));
}

// ----- #33 人糊 (ADR 0007 point 19) -----

#[test]
fn hand_of_man_on_the_first_go_round_33() {
    // Seat 1 self-draws a 3-faan hand on its first draw, nothing claimed:
    // 人糊, a limit hand, 128 from each other seat.
    let mut s = deal(
        no_flowers(),
        [
            DEALER_R,
            "1p 2p 3p 4p 5p 6p 7p 8p 9p 1s 2s 3s 5s",
            "2m 4m 6m 8m 6s 7s 8s N N W G Wh S",
            "9s 9s W W G G Wh Wh 1p 2p 3p 6s 6s",
        ],
        "5s",
        "",
    );
    play(&mut s, &[(0, Discard(t("R"))), (1, SelfDrawnWin)]);
    let w = &s.result().unwrap().wins[0];
    assert!(w.score.has(Pattern::HumanHand) && w.score.limit);
    assert_eq!(s.result().unwrap().deltas, [-128, 384, -128, -128]);
    assert!(Pattern::HumanHand.is_limit());
    assert_eq!(Pattern::HumanHand.chinese(), "人糊");
}

#[test]
fn a_claim_ends_the_first_go_round_33() {
    // Seat 2 pungs seat 1's N; seat 3 then self-draws on its first turn (after
    // three discards), but a claim has ended the first go-round: no 人糊.
    let mut s = deal(
        no_flowers(),
        [
            DEALER_R,
            "1m 2m 4m 6m 8m 6s 7s 8s E W W N S",
            "N N 1s 4s 7s 9p 9p G G G Wh Wh Wh",
            "1p 2p 3p 4p 5p 6p 7p 8p 9p 1s 2s 3s 5s",
        ],
        "9s 5s",
        "",
    );
    play(
        &mut s,
        &[
            (0, Discard(t("R"))),
            (1, Discard(t("N"))),
            (2, Pung),
            (2, Discard(t("7s"))),
        ],
    );
    play(&mut s, &[(3, SelfDrawnWin)]);
    let w = &s.result().unwrap().wins[0];
    assert!(!w.score.has(Pattern::HumanHand));
    assert_eq!(w.score.faan, 3);
}

// ----- §8.3.4 包大三元 (ADR 0007 point 14) -----

const DRAGONS: [&str; 4] = [
    "R 9m 1p 3p 5p 7p 2m 4m 6m 8m 1s 3s 5s E",
    "R R G G Wh Wh 2s 2s 2s 9p 5m 6m 7m",
    "G Wh N W S 2p 4p 6p 8p 6s 8s 1m 9s",
    "3m 9m 4m 8m 1p 3p 5p 7p 1s 6p 5s 7s S",
];

/// Seat 1 pungs R from the dealer, then G and Wh from seat 2.
fn three_dragon_pungs(draws: &str, replacements: &str) -> State {
    let mut s = deal(no_flowers(), DRAGONS, draws, replacements);
    play(
        &mut s,
        &[
            (0, Discard(t("R"))),
            (1, Pung),
            (1, Discard(t("5m"))),
            (2, Discard(t("G"))),
            (1, Pung),
            (1, Discard(t("6m"))),
        ],
    );
    assert_eq!(s.liable_for(1), None);
    play(&mut s, &[(2, Discard(t("Wh"))), (1, Pung)]);
    assert_eq!(s.liable_for(1), Some(2));
    assert_eq!(s.liability_kind(1), Some(Liability::BigDragons));
    play(&mut s, &[(1, Discard(t("7m")))]);
    s
}

#[test]
fn big_dragons_feeder_pays_the_whole_self_draw_8_3_4() {
    // 大三元 8 + 對對糊 3 + 自摸 1 = 12, capped at 10: seat 2 pays 3 × 128.
    let mut s = three_dragon_pungs("3s 4s E W N 9p", "");
    play(
        &mut s,
        &[
            (2, Discard(t("E"))),
            (3, Discard(t("W"))),
            (0, Discard(t("N"))),
            (1, SelfDrawnWin),
        ],
    );
    let r = s.result().unwrap();
    assert!(r.wins[0].score.has(Pattern::BigDragons));
    assert_eq!(r.wins[0].liable, Some(2));
    assert_eq!(r.deltas, [0, 384, -384, 0]);
}

#[test]
fn big_dragons_no_liability_on_a_kong_replacement_win_8_3_4() {
    // Seat 1 draws the fourth 2s, declares the kong and wins on its replacement
    // 9p: 槓後自摸不用包, each other seat pays 128.
    let mut s = three_dragon_pungs("3s 4s E W N 2s", "9p");
    play(
        &mut s,
        &[
            (2, Discard(t("E"))),
            (3, Discard(t("W"))),
            (0, Discard(t("N"))),
            (1, ConcealedKong(t("2s"))),
            (1, SelfDrawnWin),
        ],
    );
    let r = s.result().unwrap();
    assert!(r.wins[0].score.has(Pattern::KongReplacement));
    assert_eq!(r.wins[0].liable, None);
    assert_eq!(r.deltas, [-128, 384, -128, -128]);
}

#[test]
fn a_later_liability_replaces_the_earlier_8_3() {
    // Seat 3 then feeds seat 1's fourth open set (a kong of 2s): the 12-tile
    // liability replaces 包大三元, and it holds on the kong's replacement win.
    let mut s = three_dragon_pungs("3s 4s E 2s", "9p");
    play(&mut s, &[(2, Discard(t("E"))), (3, Discard(t("2s")))]);
    // The dealer may chow 1s-2s-3s and is asked first.
    play(&mut s, &[(0, Pass), (1, ClaimKong)]);
    assert_eq!(s.liable_for(1), Some(3));
    assert_eq!(s.liability_kind(1), Some(Liability::TwelveTiles));
    play(&mut s, &[(1, SelfDrawnWin)]);
    let r = s.result().unwrap();
    assert_eq!(r.wins[0].liable, Some(3));
    assert_eq!(r.deltas, [0, 384, 0, -384]);
}

// ----- §8.3.3 the 12-tile pung exception (ADR 0007 point 19) -----

#[test]
fn no_twelve_tile_liability_for_a_pung_of_the_players_own_discard_8_3_3() {
    // Seat 1 threw a 4s earlier; seat 2's 4s then gives seat 1 its fourth open
    // set, a pung: seat 2 is not liable.
    let mut s = deal(
        no_flowers(),
        [
            "1s 9p 9p 1p 2p 3p 4p 4m 4m 8m 8m 9s 9s Wh",
            "11s 22s 33s 444s E S W N",
            "2s 3s 4s 5m 6m 7m 1p 2p 3p 5p 6p 7p R",
            "1m 2m 3m 4m 5m 6m 7m 8m 9m Wh Wh G G",
        ],
        "8p 8p 8p",
        "",
    );
    play(
        &mut s,
        &[
            (0, Discard(t("1s"))),
            (1, Pung),
            (1, Discard(t("4s"))),
            (2, Pass),
            (2, Discard(t("2s"))),
            (1, Pung),
            (1, Discard(t("E"))),
            (2, Discard(t("3s"))),
            (1, Pung),
            (1, Discard(t("S"))),
            (2, Discard(t("4s"))),
            (1, Pung),
        ],
    );
    assert_eq!(s.melds(1).exposed_count(), 4);
    assert_eq!(s.liable_for(1), None);
}

// ----- kept rules (points 5, 7 and 20) -----

#[test]
fn kong_replacement_still_counts_after_a_flower_replacement_5_4() {
    // Point 5: the kong's replacement is S3, its replacement 9s wins: 自摸 +
    // 槓上開花 + 門前清 = 3 (S3 is not East's flower), 8 from each other seat.
    let mut s = deal(
        RuleSet::default(),
        [
            "1111m 234p 567p 888s 9s",
            "2m 3m 4m 5m 6m 7m 8m 1s 2s 3s 4s E E",
            "S S W W N N G G R R Wh Wh 1p",
            "9m 9m 9m 2p 3p 4p 5p 6p 7p 8p 9p 5s 6s",
        ],
        "",
        "S3 9s",
    );
    play(&mut s, &[(0, ConcealedKong(t("1m"))), (0, SelfDrawnWin)]);
    let w = &s.result().unwrap().wins[0];
    assert!(w.score.has(Pattern::KongReplacement));
    assert_eq!(w.score.faan, 3);
    assert_eq!(s.result().unwrap().deltas, [24, -8, -8, -8]);
}

#[test]
fn small_four_winds_is_a_10_faan_limit_21() {
    // Point 7 (also in faan_spec.rs `rows_21_to_28_limit_hands`).
    let ctx = WinContext {
        seat: 1,
        win_tile: Some(t("1m")),
        ..WinContext::default()
    };
    let s = score_win(&counts("EEE SSS WWW NN 123m"), &[], &ctx, &no_flowers()).unwrap();
    assert!(s.has(Pattern::SmallWinds) && s.limit);
    assert_eq!(s.faan, 10);
}

#[test]
fn earthly_hand_still_counts_after_the_dealers_concealed_kong_30() {
    // Point 20 (Brian: "暗槓 cannot 搶"): the dealer konges 1m, then throws 5p.
    let mut s = deal(
        no_flowers(),
        [
            "5p 1111m 222m 333m 44m 9s",
            "11p 234p 678p 789p 46p",
            "1s 2s 3s 4s 5s 6s 7s 8s W W N N G",
            "E E E S S S 6m 7m 8m 9m 9m 9m 2m",
        ],
        "",
        "6s",
    );
    play(
        &mut s,
        &[(0, ConcealedKong(t("1m"))), (0, Discard(t("5p"))), (1, Win)],
    );
    let w = &s.result().unwrap().wins[0];
    assert!(w.score.has(Pattern::Earthly));
    assert_eq!(s.result().unwrap().deltas, [-256, 256, 0, 0]);
}
