//! `State::from_position` (plan Task 2, AC2): a state rebuilt from a position
//! (the public table plus the hands, no wall) behaves like the original for
//! every observer, and malformed positions are errors, never panics.

use mahjong_games::{Game, Policy};
use mahjong_hk::{
    Action, Meld, MeldKind, PositionError, PositionPhase, PositionSpec, RuleSet, State,
};
use mahjong_search::features::{encode, STATE_LEN};
use mahjong_search::{shortlist, Bot, Danger};
use mahjong_tiles::{parse_tiles, Tile, TileCounts};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// Every decision point of one seeded hand played by the `value` bot.
fn decisions(seed: u64) -> Vec<State> {
    let bot = Bot::by_name("value").unwrap();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    let mut out = Vec::new();
    while !s.is_terminal() {
        out.push(s.clone());
        let legal = s.legal_actions_vec();
        let a = bot.choose(&s, &legal, &mut rng);
        s.apply(a);
    }
    out
}

/// Four stop points of a hand: three spread over the hand and the claim or rob
/// window nearest its middle (a turn decision if the hand has none).
fn stop_points(all: &[State]) -> Vec<&State> {
    let n = all.len();
    let mut picks = vec![n / 5, 2 * n / 5, 4 * n / 5];
    let window = (0..n)
        .filter(|&i| all[i].phase_name() != "turn" && !picks.contains(&i))
        .min_by_key(|&i| i.abs_diff(n / 2));
    picks.push(window.unwrap_or(3 * n / 5));
    picks.into_iter().map(|i| &all[i]).collect()
}

fn danger_bits(s: &State, me: usize) -> Vec<u64> {
    let d = Danger::estimate(s, me);
    Tile::all_kinds()
        .map(|t| d.tile(t).to_bits())
        .chain((0..4).map(|o| d.threat(o).to_bits()))
        .collect()
}

fn features(s: &State) -> Vec<u32> {
    let mut out = vec![0.0f32; STATE_LEN];
    encode(s, &mut out);
    out.iter().map(|x| x.to_bits()).collect()
}

/// Asserts that `a` and `b` look the same to the engine, the danger model, the
/// features and the PIMC shortlist.
fn assert_same(a: &State, b: &State, what: &str) {
    assert_eq!(a.current_player(), b.current_player(), "{what}: player");
    assert_eq!(a.phase_name(), b.phase_name(), "{what}: phase");
    assert_eq!(
        a.legal_actions_vec(),
        b.legal_actions_vec(),
        "{what}: legal"
    );
    assert_eq!(
        a.self_drawn_score(),
        b.self_drawn_score(),
        "{what}: self-drawn"
    );
    for seat in 0..4 {
        assert_eq!(
            a.claim_score(seat),
            b.claim_score(seat),
            "{what}: claim {seat}"
        );
        assert_eq!(
            danger_bits(a, seat),
            danger_bits(b, seat),
            "{what}: danger {seat}"
        );
        assert_eq!(a.unseen(seat), b.unseen(seat), "{what}: unseen {seat}");
    }
    assert_eq!(a.river(), b.river(), "{what}: river");
    assert_eq!(a.wall_remaining(), b.wall_remaining(), "{what}: wall");
    assert_eq!(features(a), features(b), "{what}: features");
    assert_eq!(shortlist(a, 4), shortlist(b, 4), "{what}: shortlist");
}

#[test]
fn rebuilt_positions_match_the_originals() {
    let mut checked = 0;
    let mut windows = 0;
    for seed in 0..50 {
        let all = decisions(seed);
        assert!(all.len() >= 8, "seed {seed}: hand too short");
        for original in stop_points(&all) {
            let spec = original.to_position().expect("in progress");
            let rebuilt = State::from_position(*original.rules(), &spec)
                .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            let what = format!("seed {seed}, {}", original.phase_name());
            assert_same(original, &rebuilt, &what);
            assert_eq!(
                rebuilt.to_position().as_ref(),
                Some(&spec),
                "{what}: round trip"
            );
            windows += (original.phase_name() != "turn") as usize;
            checked += 1;
            // The rebuilt state plays on to the end.
            let mut s = rebuilt.determinize(spec.to_move, &mut SmallRng::seed_from_u64(seed));
            let bot = Bot::by_name("fast").unwrap();
            let mut rng = SmallRng::seed_from_u64(seed);
            while !s.is_terminal() {
                let legal = s.legal_actions_vec();
                s.apply(bot.choose(&s, &legal, &mut rng));
            }
        }
    }
    assert_eq!(checked, 200);
    assert!(windows >= 40, "only {windows} claim or rob windows");
}

fn tiles(text: &str) -> TileCounts {
    TileCounts::from_tiles(&parse_tiles(text).unwrap())
}

fn t(text: &str) -> Tile {
    text.parse().unwrap()
}

/// A valid early position without flowers: the dealer (seat 0) discarded 9p,
/// seat 1 punged it and discarded W, and seat 2 is on turn after drawing 5s.
/// 49 concealed tiles + 3 in the pung + W in the river leave 82 in the wall.
fn turn_position() -> (RuleSet, PositionSpec) {
    let rules = RuleSet {
        flowers: false,
        ..RuleSet::default()
    };
    let spec = PositionSpec {
        hands: [
            tiles("123m 456m 789m 1p 23s E"),
            tiles("234p 567p 345s N"),
            tiles("123s 456s 789s RR 66m 5s"),
            tiles("111m 22m 33m 44m 55m WhWh"),
        ],
        melds: [
            vec![],
            vec![Meld::exposed(MeldKind::Pung, t("9p"), 0)],
            vec![],
            vec![],
        ],
        bonus: Default::default(),
        river: vec![(0, t("9p"), true), (1, t("W"), false)],
        wall_remaining: 82,
        to_move: 2,
        phase: PositionPhase::Turn,
        last_drawn: Some(t("5s")),
        kong_chain: 0,
        liable: [None; 4],
    };
    (rules, spec)
}

/// The same table one step later: seat 2 discarded R, and seat 0 (holding two
/// R instead of 1p and 2s) may pung it. The wall still holds 82 tiles.
fn claim_position() -> (RuleSet, PositionSpec) {
    let (rules, mut spec) = turn_position();
    spec.hands[0] = tiles("123m 456m 789m RR 3s E");
    spec.hands[2] = tiles("123s 456s 789s R 66m 5s");
    spec.river.push((2, t("R"), false));
    spec.phase = PositionPhase::Claim {
        tile: t("R"),
        from: 2,
    };
    spec.to_move = 0;
    spec.last_drawn = None;
    (rules, spec)
}

#[test]
fn a_turn_position_builds_and_round_trips() {
    let (rules, spec) = turn_position();
    let s = State::from_position(rules, &spec).unwrap();
    assert_eq!(s.current_player(), Some(2));
    assert_eq!(s.wall_remaining(), 82);
    assert_eq!(s.river(), spec.river);
    assert!(s.legal_actions_vec().contains(&Action::Discard(t("5s"))));
    assert_eq!(s.to_position(), Some(spec));
}

#[test]
fn a_claim_window_asks_the_player_to_move() {
    let (rules, mut spec) = claim_position();
    let s = State::from_position(rules, &spec).unwrap();
    assert_eq!(s.phase_name(), "claim");
    assert_eq!(s.current_player(), Some(0));
    assert_eq!(s.legal_actions_vec(), vec![Action::Pung, Action::Pass]);
    // Seat 1 has no option on R, so it cannot be the seat to move.
    spec.to_move = 1;
    assert_eq!(
        State::from_position(rules, &spec).unwrap_err(),
        PositionError::NoDecision
    );
}

#[test]
fn malformed_positions_are_errors_not_panics() {
    let (rules, base) = turn_position();
    let mut cases: Vec<(&str, PositionSpec)> = Vec::new();

    // The plan's five cases.
    let mut p = base.clone();
    p.hands[3] = tiles("1111m 2m 33m 44m 55m WhWh"); // seat 0 holds the fifth 1m
    cases.push(("a fifth copy", p));
    let mut p = base.clone();
    p.hands[3] = tiles("111m 22m 33m 44m 55m");
    cases.push(("a wrong hand size", p));
    let mut p = base.clone();
    p.river[0].2 = false; // the punged 9p is also lying in the river
    cases.push(("a meld tile also in the river unclaimed", p));
    let mut p = base.clone();
    p.wall_remaining += 1;
    cases.push(("a wall that does not balance", p));
    let (_, mut p) = claim_position();
    p.phase = PositionPhase::Claim {
        tile: t("5s"),
        from: 2,
    };
    cases.push(("a claim tile that is not the last discard", p));

    // More: seats, bonus tiles, melds, liability, last_drawn, turn order.
    let mut p = base.clone();
    p.to_move = 4;
    cases.push(("a seat out of range", p));
    let mut p = base.clone();
    p.bonus[0] = vec![Tile::flower(1)];
    cases.push(("a bonus tile without flowers", p));
    let mut p = base.clone();
    p.river[1].1 = Tile::season(2);
    cases.push(("a bonus tile in the river", p));
    let mut p = base.clone();
    p.melds[1][0] = Meld {
        kind: MeldKind::Pung,
        tile: t("9p"),
        concealed: true,
        claimed_from: None,
    };
    cases.push(("a concealed pung", p));
    let mut p = base.clone();
    p.melds[1][0] = Meld::exposed(MeldKind::Chow, t("8p"), 0);
    cases.push(("a chow off the end of a suit", p));
    let mut p = base.clone();
    p.melds[1][0] = Meld::exposed(MeldKind::Chow, t("7p"), 2);
    cases.push(("a chow from the wrong seat", p));
    let mut p = base.clone();
    p.liable[1] = Some(0);
    cases.push(("liability with one exposed meld", p));
    let mut p = base.clone();
    p.last_drawn = Some(t("E"));
    cases.push(("last_drawn not in the hand", p));
    let mut p = base.clone();
    p.last_drawn = None; // on turn without a draw, but seat 2 claimed nothing
    cases.push(("a turn without a draw or a claim", p));
    let (_, mut p) = claim_position();
    p.to_move = 2;
    cases.push(("the discarder claims its own tile", p));

    for (name, spec) in cases {
        let r = std::panic::catch_unwind(|| State::from_position(rules, &spec));
        match r {
            Ok(Err(e)) => assert!(!e.to_string().is_empty(), "{name}"),
            Ok(Ok(_)) => panic!("{name}: accepted a malformed position"),
            Err(_) => panic!("{name}: panicked"),
        }
    }
}
