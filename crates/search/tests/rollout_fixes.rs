//! The review's rollout bot (`fast`, plan 2026-11-09 Tasks 1–2): it keeps waits that
//! still have live tiles, keeps a flush whose wait is live, and a defending seat that
//! is ready or one away stops feeding an obvious flush.
//!
//! Brian's hand (`data/brian-2026-11-09.json`, the app's kept-hand JSON) is replayed
//! from its wall and history codes; the decision at history index 85 is seat 0's pung
//! of 8p with `4588p` and three pin pungs, where all four 5p are visible.

mod common;

use common::{build, t, Setup};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, Meld, MeldKind, RuleSet, State};
use mahjong_search::Fast;
use mahjong_tiles::Tile;
use rand::rngs::SmallRng;
use rand::SeedableRng;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/brian-2026-11-09.json"
);

/// The quoted strings of the JSON array that follows `"key": [`.
fn array_after<'a>(json: &'a str, key: &str) -> Vec<&'a str> {
    let start = json.find(&format!("\"{key}\": [")).expect("key") + key.len() + 4;
    let body = &json[start..start + json[start..].find(']').expect("array end")];
    body.split('"').skip(1).step_by(2).collect()
}

/// Brian's hand replayed up to (not including) history index `at`. The fixture's rules
/// are the default rule set (faan cap 10, minimum 3, flowers, half spicy, east round).
fn brian_at(at: usize) -> State {
    let json = std::fs::read_to_string(FIXTURE).expect("fixture");
    let wall: Vec<Tile> = array_after(&json, "wall").iter().map(|s| t(s)).collect();
    let history = &json[json.find("\"history\"").unwrap()..json.find("\"human_seats\"").unwrap()];
    let codes: Vec<u16> = history
        .split("\"code\": ")
        .skip(1)
        .map(|s| s[..s.find(',').unwrap()].parse().unwrap())
        .collect();
    let mut s = State::from_wall(RuleSet::default(), wall);
    for &c in &codes[..at] {
        let a = Action::from_code(c).unwrap();
        assert!(
            s.legal_actions_vec().contains(&a),
            "history replays legally"
        );
        s.apply(a);
    }
    s
}

fn fast_choice(bot: &Fast, s: &State) -> Action {
    let legal = s.legal_actions_vec();
    bot.choose(s, &legal, &mut SmallRng::seed_from_u64(1))
}

#[test]
fn after_brians_pung_fast_discards_the_dead_5p() {
    let mut s = brian_at(85);
    assert_eq!(s.current_player(), Some(0));
    assert!(s.in_claim());
    assert_eq!(s.unseen(0)[t("5p")], 0, "all four 5p are visible to seat 0");
    s.apply(Action::Pung);
    assert_eq!(fast_choice(&Fast::default(), &s), Action::Discard(t("5p")));
    assert_eq!(fast_choice(&Fast::new(1.0), &s), Action::Discard(t("5p")));
}

/// Seat 0 on turn with four pin pungs and `hand` (the drawn tile included). `dead` is
/// a tile seat 1 has pung'ed from seat 2, so seat 0 sees all its other copies; kinds
/// in `keep` stay unseen.
fn ready_position(hand: &str, drawn: &str, keep: &str, dead: &'static str) -> State {
    let (melds, mut claimed) = pin_pungs();
    let mut m: [Vec<Meld>; 4] = Default::default();
    m[0] = melds;
    m[1] = vec![Meld::exposed(MeldKind::Pung, t(dead), 2)];
    claimed.push((2, dead));
    build(Setup {
        hands: [Some(hand), None, None, None],
        melds: m,
        claimed,
        last_drawn: drawn,
        wall: 30,
        liable: [None; 4],
        keep,
    })
}

fn pin_pungs() -> (Vec<Meld>, Vec<(usize, &'static str)>) {
    (
        vec![
            Meld::exposed(MeldKind::Pung, t("3p"), 1),
            Meld::exposed(MeldKind::Pung, t("1p"), 2),
            Meld::exposed(MeldKind::Pung, t("7p"), 1),
            Meld::exposed(MeldKind::Pung, t("8p"), 2),
        ],
        vec![(1, "3p"), (2, "1p"), (1, "7p"), (2, "8p")],
    )
}

#[test]
fn keeps_the_live_single_wait_when_the_other_is_dead() {
    // Four pin pungs and 2p 6p: both single waits are full flush + all pungs. 6p is
    // dead (seat 1 has pung'ed the other three); 2p has live copies. The bot must
    // keep 2p even though discarding 2p sorts first.
    let s = ready_position("2p 6p", "6p", "2p", "6p");
    assert_eq!(s.unseen(0)[t("6p")], 0, "setup: 6p is dead");
    assert!(s.unseen(0)[t("2p")] > 0, "setup: 2p is live");
    assert_eq!(fast_choice(&Fast::default(), &s), Action::Discard(t("6p")));
    assert_eq!(fast_choice(&Fast::new(1.0), &s), Action::Discard(t("6p")));
}

#[test]
fn keeps_a_live_flush_wait_over_an_honour_single() {
    // Four pin pungs, 4p (live) and a drawn Green: keeping 4p is a full flush (10 faan
    // with the cap), keeping G only a half flush, with as many live copies.
    let s = ready_position("4p G", "G", "4p G", "2p");
    assert!(s.unseen(0)[t("4p")] > 0);
    assert_eq!(fast_choice(&Fast::default(), &s), Action::Discard(t("G")));
}

#[test]
fn drops_a_dead_flush_wait_for_a_live_honour_single() {
    // The same with a dead 2p: the flush is worth more, but it cannot win.
    let s = ready_position("2p G", "G", "G", "2p");
    assert_eq!(s.unseen(0)[t("2p")], 0, "setup: 2p is dead");
    assert!(s.unseen(0)[t("G")] > 0);
    assert_eq!(fast_choice(&Fast::default(), &s), Action::Discard(t("2p")));
}
