//! The discard record 摸打 / 入章 (`docs/rules/hk.md` §4.5): every discard says
//! whether it was the tile just drawn or a tile from the hand.

mod common;
use common::{deal, no_flowers, play, t};
use mahjong_games::Game;
use mahjong_hk::{Action::*, Event};
use mahjong_search::SeatView;

#[test]
fn each_discard_records_drawn_or_from_hand_4_5() {
    let mut s = deal(
        no_flowers(),
        [
            "3m 111p 222p 333p 444p 9s",
            "4m 5m 1s 2s 3s 4s 5s 6s 7s 8s E E S",
            "3m 3m 5p 6p 7p 8p 9p W W N N G G",
            "Wh Wh Wh R R R 6m 7m 8m 9m 9m 9m 2m",
        ],
        "1s 2s",
        "",
    );
    // The dealer's first discard (no draw this turn), then seat 2 pungs it and
    // discards without drawing: both from the hand.
    play(
        &mut s,
        &[
            (0, Discard(t("3m"))),
            (1, Pass),
            (2, Pung),
            (2, Discard(t("9p"))),
        ],
    );
    // Seat 3 throws the 1s it just drew (摸打); the dealer keeps its 2s and
    // throws 9s from the hand (入章).
    assert_eq!(s.last_drawn(), Some(t("1s")));
    play(&mut s, &[(3, Discard(t("1s")))]);
    assert_eq!(s.last_drawn(), Some(t("2s")));
    play(&mut s, &[(0, Discard(t("9s")))]);

    let got: Vec<_> = s
        .river_entries()
        .iter()
        .map(|r| (r.seat, r.tile, r.claimed, r.from_hand))
        .collect();
    let want = vec![
        (0, t("3m"), true, true),
        (2, t("9p"), false, true),
        (3, t("1s"), false, false),
        (0, t("9s"), false, true),
    ];
    assert_eq!(got, want);
    let logged: Vec<bool> = s
        .events()
        .iter()
        .filter_map(|e| match *e {
            Event::Discard { from_hand, .. } => Some(from_hand),
            _ => None,
        })
        .collect();
    assert_eq!(logged, [true, true, false, true]);
    // Every seat sees the same record, and it survives a position round trip.
    for seat in 0..4 {
        assert_eq!(SeatView::of(&s, seat).river, s.river_entries());
    }
    let spec = s.to_position().unwrap();
    assert_eq!(spec.river_from_hand, [true, true, false, true]);
    let rebuilt = mahjong_hk::State::from_position(*s.rules(), &spec).unwrap();
    assert_eq!(rebuilt.river_entries(), s.river_entries());
    // A re-deal changes only hidden tiles: the record stays.
    let me = s.current_player().unwrap();
    let mut rng = <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(1);
    assert_eq!(
        s.determinize(me, &mut rng).river_entries(),
        s.river_entries()
    );
}
