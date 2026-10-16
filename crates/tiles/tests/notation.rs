//! Property tests for tile notation (`docs/rules/hk.md` §1).

use mahjong_tiles::{format_tiles, parse_tiles, Tile, TileCounts, NUM_TILES_WITH_BONUS};
use proptest::prelude::*;

fn any_tiles() -> impl Strategy<Value = Vec<Tile>> {
    prop::collection::vec(
        (0..NUM_TILES_WITH_BONUS as u8).prop_map(Tile::from_index),
        0..20,
    )
}

proptest! {
    #[test]
    fn format_then_parse_gives_the_same_multiset(tiles in any_tiles()) {
        let mut sorted = tiles.clone();
        sorted.sort();
        prop_assert_eq!(parse_tiles(&format_tiles(&tiles)).unwrap(), sorted);
    }

    #[test]
    fn counts_round_trip(tiles in any_tiles()) {
        let playing: Vec<Tile> = tiles.into_iter().filter(|t| !t.is_bonus()).collect();
        let c = TileCounts::from_tiles(&playing);
        prop_assert_eq!(c.total(), playing.len());
        prop_assert_eq!(TileCounts::from_tiles(&c.tiles()), c);
    }
}
