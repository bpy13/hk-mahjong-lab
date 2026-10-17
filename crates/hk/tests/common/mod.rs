//! Helpers for building exact positions in tests.
#![allow(dead_code)]

use mahjong_games::Game;
use mahjong_hk::{Action, Meld, MeldKind, RuleSet, State};
use mahjong_tiles::{parse_tiles, Tile, TileCounts};

/// Parses tiles into counts.
pub fn counts(s: &str) -> TileCounts {
    TileCounts::from_tiles(&parse_tiles(s).unwrap())
}

/// Parses one tile.
pub fn t(s: &str) -> Tile {
    s.parse().unwrap()
}

/// A meld from text: `"chow 234m"`, `"pung GGG"`, `"kong 5555p"`, `"ckong 1111m"`.
/// Exposed melds are recorded as claimed from seat 3.
pub fn meld(s: &str) -> Meld {
    let (kind, tiles) = s.split_once(' ').unwrap();
    let first = parse_tiles(tiles).unwrap()[0];
    match kind {
        "chow" => Meld::exposed(MeldKind::Chow, first, 3),
        "pung" => Meld::exposed(MeldKind::Pung, first, 3),
        "kong" => Meld::exposed(MeldKind::Kong, first, 3),
        "ckong" => Meld::concealed_kong(first),
        _ => panic!("unknown meld kind {kind}"),
    }
}

/// Rules without bonus tiles (most scenarios do not want them).
pub fn no_flowers() -> RuleSet {
    RuleSet {
        flowers: false,
        ..RuleSet::default()
    }
}

/// Builds a wall so that the deal gives exactly `hands` (dealer first; 14, 13, 13,
/// 13 tiles), the following draws from the front are `draws` in order, and the
/// replacement draws from the back are `replacements` in order. Every other tile
/// sits in the middle of the wall, playing tiles before bonus tiles.
pub fn deal(rules: RuleSet, hands: [&str; 4], draws: &str, replacements: &str) -> State {
    let mut front: Vec<Tile> = Vec::new();
    for (seat, h) in hands.iter().enumerate() {
        let tiles = parse_tiles(h).unwrap();
        assert_eq!(
            tiles.len(),
            if seat == 0 { 14 } else { 13 },
            "seat {seat} hand size"
        );
        front.extend(tiles);
    }
    front.extend(parse_tiles(draws).unwrap());
    let back: Vec<Tile> = parse_tiles(replacements).unwrap();
    let mut rest = State::all_tiles(&rules);
    for tile in front.iter().chain(back.iter()) {
        let i = rest
            .iter()
            .position(|x| x == tile)
            .unwrap_or_else(|| panic!("too many {tile}"));
        rest.remove(i);
    }
    let mut wall = front;
    wall.extend(rest); // sorted: playing tiles, then bonus tiles
    wall.extend(back.into_iter().rev());
    State::from_wall(rules, wall)
}

/// Applies `(seat, action)` steps, checking each is the current player's and legal.
pub fn play(s: &mut State, steps: &[(usize, Action)]) {
    for &(seat, a) in steps {
        assert_eq!(
            s.current_player(),
            Some(seat),
            "expected seat {seat} to act before {a}\n{s:?}"
        );
        let legal = s.legal_actions_vec();
        assert!(
            legal.contains(&a),
            "{a} is not legal for seat {seat}; legal: {legal:?}\n{s:?}"
        );
        s.apply(a);
    }
}
