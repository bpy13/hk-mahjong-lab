//! Sampling the hidden tiles for search: **structured determinization**.
//!
//! Re-dealing the hidden tiles uniformly at random (what
//! [`mahjong_games::Game::determinize`] does) gives opponents hands that are much
//! less developed than real ones: by mid-hand a real player has thrown away their
//! useless tiles and kept the useful ones. Measured with `value` bots
//! (`cargo run --release -p mahjong-search --example calibrate`), opponents' mean
//! shanten is
//!
//! | Tiles drawn from the wall | Real | Uniform | Best of K (K below) |
//! |---|---|---|---|
//! | 0–20 | 2.70 | 3.55 | 2.69 (K = 4) |
//! | 21–41 | 1.54 | 3.24 | 1.36–1.78 (K = 16–64) |
//! | 42–62 | 1.09 | 2.97 | 1.17 (K = 64) |
//! | 63–83 | 0.77 | 2.54 | 0.56–0.88 (K = 64–256) |
//!
//! With uniform hands a searcher believes nobody else is close to winning, so it
//! ignores defense; in tests it won 2.5% of hands against `value` bots while its
//! own rollouts predicted 50–70%. Here each opponent's hand is instead the best
//! (lowest shanten) of `K` random hands drawn from the hidden tiles, with
//! `K = 4 + used² / 40` (`used` = tiles drawn from the wall so far), which tracks
//! the real means above. It still uses only the observer's information.

use mahjong_analysis::shanten;
use mahjong_games::Game;
use mahjong_hk::State;
use mahjong_tiles::{Tile, TileCounts};
use rand::seq::SliceRandom;
use rand::Rng;

/// Tiles in the wall after the deal (before flower replacement).
fn initial_wall(state: &State) -> usize {
    if state.rules().flowers {
        144 - 53
    } else {
        136 - 53
    }
}

/// The number of candidate hands per opponent at this point of the hand.
pub fn candidates_per_opponent(state: &State) -> usize {
    let used = initial_wall(state).saturating_sub(state.wall_remaining()) as f64;
    (4.0 + used * used / 40.0).min(200.0) as usize
}

/// A structured determinization for `observer` (see the module docs).
pub fn sample_world<R: Rng + ?Sized>(state: &State, observer: usize, rng: &mut R) -> State {
    if state.is_terminal() {
        return state.clone();
    }
    let k = candidates_per_opponent(state);
    let (pool, sizes) = state.hidden_pool(observer);
    let (mut playing, mut bonus): (Vec<Tile>, Vec<Tile>) =
        pool.into_iter().partition(|t| !t.is_bonus());
    let mut hands = [TileCounts::new(); 4];
    for seat in (0..4).filter(|&p| p != observer) {
        let n = sizes[seat];
        let melds = state.melds(seat).len();
        let mut best: Option<(i8, Vec<Tile>)> = None;
        for _ in 0..k {
            let (hand, _) = playing.partial_shuffle(rng, n);
            let s = shanten(&TileCounts::from_tiles(hand), melds);
            if best.as_ref().is_none_or(|(b, _)| s < *b) {
                best = Some((s, hand.to_vec()));
            }
        }
        let (_, chosen) = best.expect("k >= 1");
        for &t in &chosen {
            let i = playing.iter().position(|&x| x == t).expect("tile in pool");
            playing.swap_remove(i);
        }
        hands[seat] = TileCounts::from_tiles(&chosen);
    }
    let mut wall = playing;
    wall.append(&mut bonus);
    wall.shuffle(rng);
    state.redeal(observer, &hands, wall)
}
