//! Shanten agrees with the engine's win check (`mahjong_hk::hand`).

use mahjong_analysis::shanten;
use mahjong_hk::hand::{is_complete, waits};
use mahjong_tiles::{Tile, TileCounts};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// A random hand of `n` tiles drawn from a narrow pool (so near-complete hands
/// are common), or from all tiles.
fn random_hand(rng: &mut SmallRng, n: usize) -> TileCounts {
    let narrow = rng.gen_bool(0.7);
    let start = rng.gen_range(0..3) * 9;
    let mut c = TileCounts::new();
    while c.total() < n {
        let i = if narrow {
            if rng.gen_bool(0.85) {
                start + rng.gen_range(0..9)
            } else {
                27 + rng.gen_range(0..7)
            }
        } else {
            rng.gen_range(0..34)
        };
        let t = Tile::from_index(i as u8);
        if c[t] < 4 {
            c.add(t);
        }
    }
    c
}

#[test]
fn ready_means_has_a_winning_tile() {
    let mut rng = SmallRng::seed_from_u64(5);
    let mut ready = 0;
    for _ in 0..30_000 {
        let melds = rng.gen_range(0..=2);
        let h = random_hand(&mut rng, 13 - 3 * melds);
        let s = shanten(&h, melds);
        let w = waits(&h, melds);
        // Known quirk: a hand whose only wait is a fifth copy of a tile it holds
        // four of counts as ready (see the shanten module docs).
        let quirk = s == 0 && w.is_empty() && h.kinds().any(|t| h[t] == 4);
        assert!(
            quirk || (s == 0) == !w.is_empty(),
            "{h} melds {melds}: shanten {s}, waits {w:?}"
        );
        ready += (s == 0) as u32;
    }
    assert!(ready > 1_000);
}

#[test]
fn complete_means_minus_one() {
    let mut rng = SmallRng::seed_from_u64(6);
    for _ in 0..30_000 {
        let h = random_hand(&mut rng, 14);
        assert_eq!(shanten(&h, 0) == -1, is_complete(&h, 0), "{h}");
    }
}

#[test]
fn one_exchange_changes_shanten_by_at_most_one() {
    let mut rng = SmallRng::seed_from_u64(7);
    for _ in 0..3_000 {
        let h = random_hand(&mut rng, 13);
        let s = shanten(&h, 0);
        let mut best = i8::MAX;
        for add in Tile::all_kinds().filter(|&t| h[t] < 4) {
            let mut h14 = h;
            h14.add(add);
            for drop in h14.kinds().collect::<Vec<_>>() {
                let mut h13 = h14;
                h13.remove(drop);
                let s2 = shanten(&h13, 0);
                assert!(s2 >= s - 1, "{h} +{add} -{drop}: {s} -> {s2}");
                best = best.min(s2);
            }
        }
        if s > 0 {
            assert_eq!(best, s - 1, "{h}: some exchange improves shanten {s}");
        }
    }
}
