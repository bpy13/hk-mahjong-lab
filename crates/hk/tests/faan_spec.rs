//! Every example in the faan table (`docs/rules/hk.md` §7), scored directly.

mod common;
use common::{counts, meld, t};
use mahjong_hk::{score_win, Meld, Pattern, RuleSet, Score, WinContext};

/// Scores with flowers off unless the test is about flowers.
fn score(concealed: &str, melds: &[&str], ctx: WinContext) -> Score {
    score_with(
        concealed,
        melds,
        ctx,
        &RuleSet {
            flowers: false,
            ..RuleSet::default()
        },
    )
}

fn score_with(concealed: &str, melds: &[&str], ctx: WinContext, rules: &RuleSet) -> Score {
    let melds: Vec<Meld> = melds.iter().map(|m| meld(m)).collect();
    score_win(&counts(concealed), &melds, &ctx, rules).expect("complete hand")
}

/// A discard win for `seat` on `tile`.
fn ron(seat: usize, tile: &str) -> WinContext {
    WinContext {
        seat,
        win_tile: Some(t(tile)),
        ..WinContext::default()
    }
}

/// A self-drawn win for `seat` on `tile`.
fn tsumo(seat: usize, tile: &str) -> WinContext {
    WinContext {
        self_drawn: true,
        ..ron(seat, tile)
    }
}

fn patterns(s: &Score) -> Vec<(u8, u8)> {
    s.patterns.iter().map(|&(p, f)| (p.number(), f)).collect()
}

#[test]
fn row_1_2_self_drawn_and_concealed() {
    let s = score("234m 567m 345p 678s 99s", &[], tsumo(1, "2m"));
    assert_eq!(patterns(&s), [(1, 1), (2, 1), (14, 1)]);
    assert_eq!(s.faan, 3);
    // A concealed kong keeps the hand concealed.
    let s = score("567m 345p 678s 99s", &["ckong 1111m"], ron(1, "5m"));
    assert!(s.has(Pattern::Concealed));
    // An exposed meld does not.
    let s = score("567m 345p 678s 99s", &["pung 111m"], ron(1, "5m"));
    assert!(!s.has(Pattern::Concealed));
}

#[test]
fn flowers_do_not_count_toward_the_minimum_6() {
    let r = RuleSet::default();
    // 平糊 + 門前清 = 2, plus two seat flowers (F2, S2) for South.
    let ctx = WinContext {
        bonus: 0b0010_0010,
        ..ron(1, "2m")
    };
    let s = score_with("234m 567m 345p 678s 99s", &[], ctx, &r);
    assert_eq!((s.faan, s.qualifying), (4, 2));
    assert!(!s.meets_minimum(&r));
    // 混一色 3 + 門前清 1 qualifies; the flowers then add to what is paid.
    let ctx = WinContext {
        bonus: 0b0010_0010,
        ..ron(1, "1p")
    };
    let s = score_with("123p 456p 789p NNN 55p", &[], ctx, &r);
    assert!(s.meets_minimum(&r));
    assert_eq!(s.faan, s.qualifying + 2);
}

#[test]
fn rows_3_to_5_flowers() {
    let rules = RuleSet::default();
    let hand = "234m 567m 345p 678s 99s";
    let ctx = |bonus| WinContext {
        bonus,
        ..ron(1, "2m")
    };
    assert!(score_with(hand, &[], ctx(0), &rules).has(Pattern::NoFlowers));
    let s = score_with(hand, &[], ctx(0b0000_0010), &rules); // F2, South's flower
    assert!(s.has(Pattern::SeatFlower) && !s.has(Pattern::NoFlowers));
    let s = score_with(hand, &[], ctx(0b0000_1111), &rules);
    assert_eq!(
        s.patterns.iter().find(|p| p.0 == Pattern::FullFlowerSet),
        Some(&(Pattern::FullFlowerSet, 2))
    );
    assert!(!s.has(Pattern::SeatFlower));
}

#[test]
fn row_6_last_tile() {
    let ctx = WinContext {
        last_tile: true,
        ..tsumo(1, "2m")
    };
    let s = score("234m 567m 345p 678s 99s", &[], ctx);
    assert!(s.has(Pattern::LastTile) && s.has(Pattern::SelfDrawn));
}

#[test]
fn rows_7_8_kong_replacement() {
    // #7 is +1 on top of 自摸: 2 together.
    let ctx = WinContext {
        kong_chain: 1,
        ..tsumo(1, "9s")
    };
    let s = score("234p 567p 888s 99s", &["ckong 1111m"], ctx);
    assert!(s.has(Pattern::SelfDrawn) && s.has(Pattern::KongReplacement));
    assert_eq!(s.faan, 1 + 1 + 1); // with 門前清
                                   // #8 is 8 in total and replaces #7 and #1.
    let ctx = WinContext {
        kong_chain: 2,
        ..tsumo(1, "9s")
    };
    let s = score("567p 888s 99s", &["ckong 1111m", "ckong 2222p"], ctx);
    assert_eq!(patterns(&s), [(8, 8), (2, 1), (13, 0)]); // no hand pattern: chicken
}

#[test]
fn row_9_robbing_the_kong() {
    let ctx = WinContext {
        robbing_kong: true,
        ..ron(1, "5p")
    };
    let s = score("456p RRR 123s 456s 99s", &[], ctx);
    assert!(s.has(Pattern::RobbingKong));
    assert_eq!(s.faan, 3); // 搶槓 + 中 + 門前清
}

#[test]
fn rows_10_to_12_meld_patterns() {
    let s = score("RRR 123m 456m 789p 55s", &[], ron(1, "1m"));
    assert!(s.patterns.contains(&(Pattern::DragonPung, 1)));
    // South seat, East round: SSS is the seat wind only.
    let s = score("SSS 123m 456m 789p 55s", &[], ron(1, "1m"));
    assert!(s.has(Pattern::SeatWind) && !s.has(Pattern::PrevailingWind));
    // East seat in the East round: EEE scores both.
    let s = score("EEE 123m 456m 789p 55s", &[], ron(0, "1m"));
    assert!(s.has(Pattern::SeatWind) && s.has(Pattern::PrevailingWind));
    // A pung of an off wind scores nothing.
    let s = score("WWW 123m 456m 789p 55s", &[], ron(1, "1m"));
    assert!(s.has(Pattern::Chicken));
}

#[test]
fn row_13_chicken_hand() {
    let s = score("555p 789s 234s 99p", &["chow 123m"], ron(1, "5p"));
    assert_eq!(patterns(&s), [(13, 0)]);
    assert_eq!(s.faan, 0);
}

#[test]
fn row_14_all_chows() {
    let s = score("567m 345p 678s 99s", &["chow 234m"], ron(1, "5m"));
    assert_eq!(patterns(&s), [(14, 1)]);
}

#[test]
fn row_15_all_pungs() {
    let s = score("222m 555p 888s EEE 99p", &[], ron(1, "2m"));
    assert!(s.has(Pattern::AllPungs));
    // Kongs count as pungs.
    let s = score("555p 888s EEE 99p", &["kong 2222m"], ron(1, "5p"));
    assert!(s.has(Pattern::AllPungs));
}

#[test]
fn rows_16_17_flushes() {
    let s = score("123p 456p 789p RRR NN", &[], ron(1, "1p"));
    assert!(s.has(Pattern::HalfFlush) && !s.has(Pattern::FullFlush));
    let s = score("111s 234s 567s 789s 55s", &[], ron(1, "2s"));
    assert!(s.has(Pattern::FullFlush) && !s.has(Pattern::HalfFlush));
    assert_eq!(s.faan, 7 + 1); // with 門前清
}

#[test]
fn row_18_terminals_and_honours() {
    let s = score("111m 999p EEE GGG 11s", &[], ron(1, "1m"));
    assert!(s.has(Pattern::TerminalsAndHonours) && s.has(Pattern::AllPungs));
}

#[test]
fn rows_19_20_dragons() {
    let s = score("RRR GGG WhWh 123m 456m", &[], ron(1, "1m"));
    assert!(s.patterns.contains(&(Pattern::SmallDragons, 5)));
    assert!(!s.has(Pattern::DragonPung));
    let s = score("RRR GGG WhWhWh 234p 55s", &[], ron(1, "2p"));
    assert!(s.patterns.contains(&(Pattern::BigDragons, 8)));
    assert!(!s.has(Pattern::DragonPung) && !s.has(Pattern::SmallDragons));
}

#[test]
fn rows_21_to_28_limit_hands() {
    let cases: [(&str, &[&str], WinContext, Pattern); 9] = [
        (
            "EEE SSS WWW NN 123m",
            &[],
            ron(1, "1m"),
            Pattern::SmallWinds,
        ),
        ("EEE SSS WWW NNN 55m", &[], ron(1, "5m"), Pattern::BigWinds),
        ("EEE SSS RRR GGG WW", &[], ron(1, "E"), Pattern::AllHonours),
        (
            "111m 999m 111p 999s 99p",
            &[],
            ron(1, "1m"),
            Pattern::AllTerminals,
        ),
        (
            "19m 19p 19s ESWN WhGRR",
            &[],
            ron(1, "R"),
            Pattern::ThirteenOrphans,
        ),
        ("11123455678999m", &[], ron(1, "5m"), Pattern::NineGates),
        (
            "222m 555p 888s EEE 99p",
            &[],
            tsumo(1, "2m"),
            Pattern::FourConcealedPungs,
        ),
        (
            "99p",
            &["kong 2222m", "kong 5555p", "kong 8888s", "ckong EEEE"],
            ron(1, "9p"),
            Pattern::FourKongs,
        ),
        (
            "222m 555p 888s EEE 99p",
            &[],
            ron(1, "9p"),
            Pattern::FourConcealedPungs,
        ),
    ];
    for (hand, melds, ctx, pattern) in cases {
        let s = score(hand, melds, ctx);
        assert!(s.limit && s.has(pattern), "{hand}: {:?}", s.patterns);
        assert_eq!(s.faan, 10);
    }
}

#[test]
fn row_27_claimed_pung_is_exposed() {
    // Winning on a discard that completes a pung: that pung is exposed (§7.4 #27).
    let s = score("222m 555p 888s EEE 99p", &[], ron(1, "2m"));
    assert!(!s.has(Pattern::FourConcealedPungs) && s.has(Pattern::AllPungs));
}

#[test]
fn rows_29_30_heavenly_and_earthly() {
    let ctx = WinContext {
        heavenly: true,
        ..tsumo(0, "2m")
    };
    let s = score("234m 567m 345p 678s 99s", &[], ctx);
    assert!(s.limit && s.has(Pattern::Heavenly));
    let ctx = WinContext {
        earthly: true,
        ..ron(2, "2m")
    };
    let s = score("234m 567m 345p 678s 99s", &[], ctx);
    assert!(s.limit && s.has(Pattern::Earthly));
}

#[test]
fn best_split_is_used() {
    // 111222333m reads as three pungs (all pungs, half flush) or three chows.
    let s = score("111222333m EEE 55m", &[], tsumo(1, "1m"));
    assert!(
        s.limit,
        "four concealed pungs beats the chow reading: {:?}",
        s.patterns
    );
}

#[test]
fn worked_examples_7_5() {
    let r = RuleSet::default();
    let s = score_with("123p 456p 789p RRR NN", &[], tsumo(1, "1p"), &r);
    assert_eq!(s.faan, 7);
    // 2 faan with a seat flower: below the minimum.
    let ctx = WinContext {
        bonus: 0b0000_0010,
        ..ron(1, "5m")
    };
    let s = score_with("567m 345p 678s 99s", &["chow 234m"], ctx, &r);
    assert_eq!(s.faan, 2);
    let s = score_with("EEE 111m 999p 11s", &["pung GGG"], ron(0, "1m"), &r);
    assert_eq!(s.faan, 8);
    let s = score_with("RRR GGG WhWhWh 111s 22s", &[], tsumo(1, "1s"), &r);
    assert_eq!((s.total, s.faan), (10, 10)); // a limit hand (four concealed pungs)
    let s = score_with("GGG WhWhWh 111s 22s", &["pung RRR"], tsumo(1, "1s"), &r);
    assert_eq!(s.total, 8 + 3 + 3 + 1 + 1); // 大三元 混一色 對對糊 自摸 無花
    assert_eq!(s.faan, 10);
}
