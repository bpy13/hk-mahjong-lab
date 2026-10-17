//! Faan-aware shanten is consistent with the engine's scorer.

use mahjong_analysis::{faan_shanten, HandView};
use mahjong_hk::{score_win, RuleSet, WinContext};
use mahjong_tiles::{Tile, TileCounts};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

/// A random complete concealed hand: four melds and a pair, from a pool that
/// favours flushes and honours.
fn complete_hand(rng: &mut SmallRng) -> TileCounts {
    let suit = rng.gen_range(0..3) * 9;
    let pool: Vec<usize> = match rng.gen_range(0..3) {
        0 => (0..34).collect(),
        1 => (suit..suit + 9).chain(27..34).collect(),
        _ => (suit..suit + 9).collect(),
    };
    loop {
        let mut c = TileCounts::new();
        let mut ok = true;
        for _ in 0..4 {
            let t = pool[rng.gen_range(0..pool.len())];
            let chow = t < 27 && t % 9 <= 6 && rng.gen_bool(0.5) && pool.contains(&(t + 2));
            let tiles: Vec<usize> = if chow {
                vec![t, t + 1, t + 2]
            } else {
                vec![t; 3]
            };
            for x in tiles {
                ok &= c.0[x] < 4;
                c.0[x] += 1;
            }
        }
        let p = pool[rng.gen_range(0..pool.len())];
        ok &= c.0[p] <= 2;
        c.0[p] += 2;
        if ok {
            return c;
        }
    }
}

#[test]
fn minus_one_means_a_legal_win() {
    let rules = RuleSet {
        flowers: false,
        ..RuleSet::default()
    };
    let mut rng = SmallRng::seed_from_u64(3);
    let mut minus_one = 0;
    for _ in 0..20_000 {
        let hand = complete_hand(&mut rng);
        let unseen = TileCounts::full_set();
        let seat = rng.gen_range(0..4);
        let view = HandView {
            concealed: &hand,
            melds: &[],
            seat,
            bonus: 0,
            unseen: &unseen,
        };
        let fs = faan_shanten(&view, &rules);
        if fs.shanten == -1 {
            minus_one += 1;
            let w = hand.kinds().next().unwrap();
            let ctx = WinContext {
                seat,
                win_tile: Some(w),
                ..WinContext::default()
            };
            let s = score_win(&hand, &[], &ctx, &rules).expect("complete");
            assert!(
                s.faan >= rules.min_faan,
                "{hand}: target {:?}, scores {:?}",
                fs.target,
                s
            );
        }
    }
    assert!(minus_one > 2_000, "{minus_one}");
}

#[test]
fn speed() {
    let rules = RuleSet::default();
    let mut rng = SmallRng::seed_from_u64(4);
    let hands: Vec<TileCounts> = (0..2_000)
        .map(|_| {
            let mut c = TileCounts::new();
            while c.total() < 13 {
                let t = Tile::from_index(rng.gen_range(0..34));
                if c[t] < 4 {
                    c.add(t);
                }
            }
            c
        })
        .collect();
    let unseen = TileCounts::full_set();
    let start = Instant::now();
    for h in &hands {
        let view = HandView {
            concealed: h,
            melds: &[],
            seat: 0,
            bonus: 0,
            unseen: &unseen,
        };
        std::hint::black_box(faan_shanten(&view, &rules));
    }
    let per = start.elapsed() / hands.len() as u32;
    eprintln!("faan_shanten: {per:?} per hand");
}
