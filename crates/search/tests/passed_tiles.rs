//! Passed-tile safety and the risk-taking read (plan 2026-11-09 Tasks 14–15,
//! `docs/race-bot.md` §13): the race bot's deal-in estimate, the defending `fast`
//! bot's safe tile, and the read of a seat that throws dangerous tiles.

mod common;

use common::{counts, seat, t, Stub};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, Meld, MeldKind, PositionPhase, PositionSpec, RuleSet, State};
use mahjong_search::passed::{risk_scores, Passed, PASSED_FACTOR};
use mahjong_search::race::{RaceBot, RaceConfig, RaceTerms};
use mahjong_search::{Beliefs, Danger, Fast, SeatBelief};
use mahjong_tiles::{Tile, TileCounts, NUM_KINDS};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;

/// A turn position for seat 0 with an explicit end of the river.
struct Pos<'a> {
    /// Every seat's concealed tiles (seat 0 holds 14 less 3 per meld).
    hands: [&'a str; 4],
    melds: [Vec<Meld>; 4],
    /// `(discarder, tile)` for each claimed discard, matching the exposed melds.
    claimed: Vec<(usize, &'a str)>,
    /// The last discards in order: `(seat, tile, from_hand)`. The filler before
    /// them cycles through the seats and comes from the hand.
    tail: Vec<(usize, &'a str, bool)>,
    last_drawn: &'a str,
    wall: usize,
    /// Kinds that stay unseen (in the wall) as long as possible.
    keep: &'a str,
}

fn build(p: Pos) -> State {
    let mut used = [0u8; NUM_KINDS];
    let hands: [TileCounts; 4] = std::array::from_fn(|s| counts(p.hands[s]));
    for h in &hands {
        for k in h.kinds() {
            used[k.index() as usize] += h[k];
        }
    }
    for m in p.melds.iter().flatten() {
        for x in m.tiles() {
            used[x.index() as usize] += 1;
        }
    }
    for &(_, x, _) in &p.tail {
        used[t(x).index() as usize] += 1;
    }
    let keep: Vec<Tile> = mahjong_tiles::parse_tiles(p.keep).unwrap();
    let mut order: Vec<Tile> = Tile::all_kinds().filter(|k| !keep.contains(k)).collect();
    order.extend(Tile::all_kinds().filter(|k| keep.contains(k)));
    let in_play: usize = used.iter().map(|&c| c as usize).sum();
    let filler_len = 136 - in_play - p.wall;
    let mut filler = Vec::new();
    for &k in &order {
        while filler.len() < filler_len && used[k.index() as usize] < 4 {
            used[k.index() as usize] += 1;
            filler.push(k);
        }
    }
    assert_eq!(filler.len(), filler_len, "not enough tiles left");
    let mut river: Vec<(usize, Tile, bool)> = p
        .claimed
        .iter()
        .map(|&(from, x)| (from, t(x), true))
        .collect();
    let mut from_hand = vec![true; river.len()];
    for (i, x) in filler.into_iter().enumerate() {
        river.push(((i + 1) % 4, x, false));
        from_hand.push(true);
    }
    for &(s, x, fh) in &p.tail {
        river.push((s, t(x), false));
        from_hand.push(fh);
    }
    let spec = PositionSpec {
        hands,
        melds: p.melds,
        bonus: [
            vec![],
            mahjong_tiles::parse_tiles("F1 F2 F3").unwrap(),
            mahjong_tiles::parse_tiles("F4 S1 S2").unwrap(),
            mahjong_tiles::parse_tiles("S3 S4").unwrap(),
        ],
        river,
        river_from_hand: from_hand,
        wall_remaining: p.wall,
        to_move: 0,
        phase: PositionPhase::Turn,
        last_drawn: Some(t(p.last_drawn)),
        kong_chain: 0,
        liable: [None; 4],
        passed_wins: Default::default(),
    };
    State::from_position(RuleSet::default(), &spec).expect("a valid position")
}

/// Three opponents the stub thinks are half ready and waiting a little on
/// everything.
fn even_beliefs() -> Beliefs {
    Beliefs {
        seats: [
            SeatBelief::quiet(),
            seat(0.5, 0.3, &[], 0.1, 4),
            seat(0.5, 0.3, &[], 0.1, 4),
            seat(0.5, 0.3, &[], 0.1, 4),
        ],
    }
}

fn race(beliefs: Beliefs, passed: bool, risk: bool) -> RaceBot {
    let config = RaceConfig {
        passed,
        risk,
        ..RaceConfig::default()
    };
    RaceBot::new(config, Arc::new(Stub(beliefs)))
}

fn candidate(terms: &[RaceTerms], a: Action) -> &RaceTerms {
    terms
        .iter()
        .find(|c| c.action == a)
        .unwrap_or_else(|| panic!("{a} is a candidate"))
}

/// Seat 3 has just discarded W; seats 1 and 2 have not acted since. Then seat 0
/// discarded 1m, seat 1 threw its drawn 9s (摸打), seat 2 threw 9p from the hand
/// (入章, not a 換章), seat 3 threw N; seat 0 is on turn holding a W and an E.
fn passed_position(tail: Vec<(usize, &'static str, bool)>) -> State {
    build(Pos {
        hands: [
            "2m 3m 5m 7m 2p 4p 6p 8p 3s 5s 7s E W Wh",
            "1m 1m 2s 3s 4s 5p 6p 7p 8s 8s 8s N N",
            "4m 5m 6m 1p 2p 3p 6s 7s 8s R R G G",
            "7m 8m 9m 1s 1s 2p 2p 4p 4p 8m 8m S S",
        ],
        melds: Default::default(),
        claimed: vec![],
        tail,
        last_drawn: "Wh",
        wall: 30,
        keep: "W E 9s 9p N 1m",
    })
}

/// AC10: a tile a seat let pass cannot deal in to it before its next turn (the
/// lock: zero), and afterwards only at `PASSED_FACTOR[N]` of the table's estimate.
#[test]
fn race_deal_in_follows_the_passed_record() {
    let west = Action::Discard(t("W"));
    // The lock: seat 3 threw W last; seats 1 and 2 have not acted since.
    let s = build_locked();
    let passed = Passed::of(&s);
    for o in [1, 2] {
        let c = passed.cell(o, t("W"));
        assert!(c.passed && !c.turned, "{o}: {c:?}");
        assert_eq!(passed.factor(o, t("W")), 0.0);
    }
    let on = race(even_beliefs(), true, false).explain(&s);
    let off = race(even_beliefs(), false, false).explain(&s);
    let (on_w, off_w) = (candidate(&on, west), candidate(&off, west));
    assert_eq!(on_w.deal_in_now[1], 0.0);
    assert_eq!(on_w.deal_in_now[2], 0.0);
    assert!(off_w.deal_in_now[1] > 0.0 && off_w.deal_in_now[2] > 0.0);
    // Seat 3 threw it: no record, the same estimate either way.
    assert!((on_w.deal_in_now[3] - off_w.deal_in_now[3]).abs() < 1e-12);
    // E did not pass anyone: unchanged.
    let east = Action::Discard(t("E"));
    assert_eq!(
        candidate(&on, east).deal_in_now,
        candidate(&off, east).deal_in_now
    );

    // Later: seat 1 kept its hand (N = 0), seat 2 changed it once (N = 1).
    let s = passed_position(vec![
        (3, "W", true),
        (0, "1m", true),
        (1, "9s", false),
        (2, "9p", true),
        (3, "N", true),
    ]);
    let passed = Passed::of(&s);
    assert_eq!(passed.cell(1, t("W")).changes, 0);
    assert_eq!(passed.cell(2, t("W")).changes, 1);
    assert_eq!(passed.factor(1, t("W")), PASSED_FACTOR[0]);
    assert_eq!(passed.factor(2, t("W")), PASSED_FACTOR[1]);
    let on = race(even_beliefs(), true, false).explain(&s);
    let off = race(even_beliefs(), false, false).explain(&s);
    let (on_w, off_w) = (candidate(&on, west), candidate(&off, west));
    for o in [1, 2, 3] {
        let expect = off_w.deal_in_now[o] * passed.factor(o, t("W"));
        assert!(
            (on_w.deal_in_now[o] - expect).abs() < 1e-9,
            "seat {o}: {} vs {expect}",
            on_w.deal_in_now[o]
        );
    }
    assert!(off_w.deal_in_now[1] > 0.0);
    // The spec switch.
    let mut c = RaceConfig::default();
    c.set("passed", "off").unwrap();
    assert!(!c.passed);
    assert!(c.set("passed", "maybe").is_err());
}

fn build_locked() -> State {
    passed_position(vec![(3, "W", true)])
}

/// A 換章 by seat 2 (a tile of its visible flush suit, from the hand) ends W's
/// safety against seat 2 only; the same tile without the flush tag does not.
#[test]
fn a_reshape_ends_the_safety_for_that_seat() {
    let tail = || {
        vec![
            (3, "W", true),
            (0, "1m", true),
            (1, "9s", false),
            (2, "2s", true),
            (3, "N", true),
        ]
    };
    let mut melds: [Vec<Meld>; 4] = Default::default();
    melds[2] = vec![Meld::exposed(MeldKind::Pung, t("7s"), 1)];
    let flush = build(Pos {
        hands: [
            "2m 3m 5m 7m 2p 4p 6p 8p 3s 5s 7s E W Wh",
            "1m 1m 2s 3s 4s 5p 6p 7p 8s 8s 8s N N",
            "4m 5m 6m 1p 2p 3p R R G G",
            "7m 8m 9m 1s 1s 2p 2p 4p 4p 8m 8m S S",
        ],
        melds,
        claimed: vec![(1, "7s")],
        tail: tail(),
        last_drawn: "Wh",
        wall: 30,
        keep: "W E 9s 9p N 1m 2s",
    });
    let passed = Passed::of(&flush);
    assert_eq!(passed.factor(2, t("W")), 1.0);
    assert_eq!(passed.factor(1, t("W")), PASSED_FACTOR[0]);
    let plain = passed_position(tail());
    assert_eq!(Passed::of(&plain).factor(2, t("W")), PASSED_FACTOR[1]);
}

/// The defending `fast` bot folds to the tile the threatening seat let pass.
#[test]
fn fast_folds_to_the_passed_tile() {
    // Seat 2 shows four exposed melds in bamboo; I am far from ready and hold E
    // and W, and every kind of mine still has unseen copies. Seat 3 threw W last
    // and seats 1 and 2 have not acted since.
    let melds: [Vec<Meld>; 4] = [
        vec![],
        vec![],
        vec![
            Meld::exposed(MeldKind::Pung, t("1s"), 1),
            Meld::exposed(MeldKind::Chow, t("4s"), 1),
            Meld::exposed(MeldKind::Pung, t("8s"), 3),
            Meld::exposed(MeldKind::Pung, t("7s"), 3),
        ],
        vec![],
    ];
    let s = build(Pos {
        hands: [
            "1m 4m 7m 2p 5p 8p 3s 6s 9s E W Wh R G",
            "2m 2m 3m 3m 4m 4m 6p 6p 7p 7p 9p 9p N",
            "2s",
            "5m 6m 8m 9m 1p 3p 4p 8p 9p 5s 5s S S",
        ],
        melds,
        claimed: vec![(1, "1s"), (1, "4s"), (3, "8s"), (3, "7s")],
        tail: vec![(3, "W", true)],
        last_drawn: "Wh",
        wall: 45,
        keep: "1m 4m 7m 2p 5p 8p 3s 6s 9s E W Wh R G",
    });
    let mut danger = Danger::estimate(&s, 0);
    let before = danger.tile(t("W"));
    assert!(
        before > 0.0 && danger.threat(2) >= 0.3,
        "{before} {}",
        danger.threat(2)
    );
    let unseen = s.unseen(0);
    for k in s.hand(0).kinds() {
        assert!(unseen[k] > 0, "{k} is dead");
    }
    danger.apply_passed(&Passed::of(&s));
    assert!(danger.tile(t("W")) < before);
    let bot = Fast::new(1.0);
    let legal = s.legal_actions_vec();
    let mut rng = SmallRng::seed_from_u64(1);
    assert_eq!(bot.choose(&s, &legal, &mut rng), Action::Discard(t("W")));
}

/// AC11: a seat that threw dangerous tiles reads as pushing: its risk score
/// rises, its fold chance falls and its ready chance rises against the same seat
/// throwing safe tiles.
#[test]
fn two_dangerous_discards_move_a_seat_toward_push() {
    // Seats 1 and 3 show three exposed melds in one suit each; seat 2 throws
    // either two fresh honours (dangerous) or two honours seats 1 and 3 had
    // themselves just discarded (safe).
    let melds = || -> [Vec<Meld>; 4] {
        [
            vec![],
            vec![
                Meld::exposed(MeldKind::Pung, t("2m"), 0),
                Meld::exposed(MeldKind::Chow, t("4m"), 0),
                Meld::exposed(MeldKind::Pung, t("8m"), 2),
            ],
            vec![],
            vec![
                Meld::exposed(MeldKind::Pung, t("2p"), 2),
                Meld::exposed(MeldKind::Chow, t("4p"), 2),
                Meld::exposed(MeldKind::Pung, t("8p"), 0),
            ],
        ]
    };
    let position = |tail: Vec<(usize, &'static str, bool)>| {
        build(Pos {
            hands: [
                "1m 3m 7m 1p 3p 7p 1s 3s 7s 9s E W Wh R",
                "1m 9m 2s 3s",
                "4s 5s 6s 7s 7s 8s 8s 9s 9s 2m S N N",
                "9p 9p 5s 6s",
            ],
            melds: melds(),
            claimed: vec![
                (0, "2m"),
                (0, "4m"),
                (2, "8m"),
                (2, "2p"),
                (2, "4p"),
                (0, "8p"),
            ],
            tail,
            last_drawn: "R",
            wall: 60,
            keep: "5m 6m 5p 6p 1s 9m W E N S 2m 7p 3s",
        })
    };
    // Dangerous: honours with all four copies unseen (the tables' most dangerous
    // tile against a flush). Safe: honours both flush seats had just thrown.
    let dangerous = position(vec![
        (1, "W", true),
        (3, "W", true),
        (2, "N", true),
        (1, "E", true),
        (3, "E", true),
        (2, "S", true),
    ]);
    let safe = position(vec![
        (1, "W", true),
        (3, "W", true),
        (2, "W", true),
        (1, "E", true),
        (3, "E", true),
        (2, "E", true),
    ]);
    let (r_d, r_s) = (risk_scores(&dangerous), risk_scores(&safe));
    assert!(r_d[2] > 0.0 && r_s[2] < 0.0, "{r_d:?} vs {r_s:?}");
    let beliefs = || Beliefs {
        seats: [
            SeatBelief::quiet(),
            seat(0.6, 0.3, &[], 0.1, 6),
            seat(0.3, 0.2, &[], 0.1, 4),
            seat(0.6, 0.3, &[], 0.1, 6),
        ],
    };
    let a = Action::Discard(t("R"));
    let d = race(beliefs(), false, true).explain(&dangerous);
    let s = race(beliefs(), false, true).explain(&safe);
    let (d, s) = (candidate(&d, a), candidate(&s, a));
    assert_eq!(d.risk, r_d);
    assert!(
        d.fold[2] <= s.fold[2],
        "fold {} vs {}",
        d.fold[2],
        s.fold[2]
    );
    assert!(
        d.tenpai[2] >= s.tenpai[2],
        "ready {} vs {}",
        d.tenpai[2],
        s.tenpai[2]
    );
    assert!(
        d.fold[2] < s.fold[2] || d.tenpai[2] > s.tenpai[2],
        "the read moved nothing: fold {} / {}, ready {} / {}",
        d.fold[2],
        s.fold[2],
        d.tenpai[2],
        s.tenpai[2]
    );
    // With the read off, seat 2 reads the same in both positions (the fold still
    // moves a little: the honours seat 2 threw change how useful my waits are to
    // it, §6).
    let d = race(beliefs(), false, false).explain(&dangerous);
    let s = race(beliefs(), false, false).explain(&safe);
    let (d, s) = (candidate(&d, a), candidate(&s, a));
    assert_eq!(d.tenpai[2], s.tenpai[2]);
    assert!(
        (d.fold[2] - s.fold[2]).abs() < 0.05,
        "{} vs {}",
        d.fold[2],
        s.fold[2]
    );
    assert_eq!(d.risk, [0.0; 4]);
}
