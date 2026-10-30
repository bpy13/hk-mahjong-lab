//! Offense alone: how often each discard reaches a legal win if the player just
//! keeps drawing, with no opponents claiming, winning or being dealt into.
//!
//! Each sample shuffles the tiles the player has not seen and draws one per turn
//! for the remaining turns, discarding with the faan-aware measure. It answers
//! "how good is this discard for my own hand?", which study mode shows beside the
//! full solver's numbers (that include defense and opponents).

use crate::bots::{best_discard, Metric};
use crate::view::SeatView;
use mahjong_analysis::shanten;
use mahjong_hk::{score_win, WinContext};
use mahjong_tiles::{Tile, TileCounts};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Results for one discard.
#[derive(Clone, Debug)]
pub struct SoloStats {
    pub tile: Tile,
    /// Fraction of samples that reached a legal self-drawn win.
    pub win_rate: f64,
    /// Mean faan of those wins.
    pub mean_faan: f64,
    /// Mean draws needed, over the samples that won.
    pub mean_draws: f64,
}

/// Simulates `samples` futures of `draws` own draws after each discard.
pub fn solo(view: &SeatView, draws: usize, samples: usize, seed: u64) -> Vec<SoloStats> {
    let tiles: Vec<Tile> = view.concealed.kinds().collect();
    tiles
        .par_iter()
        .map(|&d| {
            let mut wins = 0u32;
            let mut faan = 0u64;
            let mut used = 0u64;
            for i in 0..samples {
                let mut rng =
                    ChaCha8Rng::seed_from_u64(seed ^ ((i as u64) << 8) ^ d.index() as u64);
                if let Some((f, k)) = one_future(view, d, draws, &mut rng) {
                    wins += 1;
                    faan += f as u64;
                    used += k as u64;
                }
            }
            SoloStats {
                tile: d,
                win_rate: wins as f64 / samples.max(1) as f64,
                mean_faan: if wins > 0 {
                    faan as f64 / wins as f64
                } else {
                    0.0
                },
                mean_draws: if wins > 0 {
                    used as f64 / wins as f64
                } else {
                    0.0
                },
            }
        })
        .collect()
}

fn one_future(
    view: &SeatView,
    discard: Tile,
    draws: usize,
    rng: &mut ChaCha8Rng,
) -> Option<(u8, usize)> {
    let mut pool: Vec<Tile> = view.unseen.tiles();
    pool.shuffle(rng);
    let mut hand: TileCounts = view.concealed;
    hand.remove(discard);
    for k in 1..=draws.min(pool.len()) {
        let t = pool[k - 1];
        hand.add(t);
        let ctx = WinContext {
            seat: view.seat,
            win_tile: Some(t),
            self_drawn: true,
            bonus: view.bonus,
            ..WinContext::default()
        };
        if let Some(s) = score_win(&hand, &view.melds, &ctx, &view.rules) {
            if s.meets_minimum(&view.rules) {
                return Some((s.faan, k));
            }
        }
        // Discard: shortlist by plain shanten, then the faan-aware measure.
        let now = view.with(hand, view.melds.clone());
        let mut keyed: Vec<(i8, Tile)> = hand
            .kinds()
            .map(|x| {
                let mut h = hand;
                h.remove(x);
                (shanten(&h, view.melds.len()), x)
            })
            .collect();
        keyed.sort();
        let shortlist: Vec<Tile> = keyed.iter().take(3).map(|&(_, x)| x).collect();
        let (out, _) = best_discard(&now, &shortlist, Metric::FaanQuick);
        hand.remove(out);
    }
    None
}
