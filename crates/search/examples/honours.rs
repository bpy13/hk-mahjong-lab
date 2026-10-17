//! Where are the unseen honour tiles? For the player to move, each honour copy
//! they cannot see is either in an opponent's hand or in the wall. Compares the
//! real split with the one [`sample_world`] produces, by stage of the hand, and
//! how often a held honour is part of a pair or better.
//!
//! `cargo run --release -p mahjong-search --example honours [hands]`
use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::world::sample_world_with;
use mahjong_search::Bot;
use mahjong_tiles::Tile;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// (unseen honour copies, of them in opponents' hands, of those in pairs or more)
fn split(s: &State, me: usize) -> (f64, f64, f64) {
    let unseen = s.unseen(me);
    let (mut u, mut held, mut paired) = (0.0, 0.0, 0.0);
    for t in Tile::all_kinds().filter(|t| t.is_honour()) {
        u += unseen[t] as f64;
        for o in (0..4).filter(|&o| o != me) {
            let c = s.hand(o)[t] as f64;
            held += c;
            if c >= 2.0 {
                paired += c;
            }
        }
    }
    (u, held, paired)
}

fn main() {
    let hands: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(300);
    let weight: f64 = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(mahjong_search::world::HONOUR_WEIGHT);
    let bots = ["defensive", "value", "efficiency", "defensive"].map(|b| Bot::by_name(b).unwrap());
    let mut rng = SmallRng::seed_from_u64(5);
    // [stage] -> (real u, held, paired, sampled u, held, paired, n)
    let mut acc = [[0.0f64; 7]; 4];
    for h in 0..hands {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        while let Some(p) = s.current_player() {
            if s.phase_name() == "turn" {
                let stage = ((91 - s.wall_remaining().min(91)) / 23).min(3);
                let r = split(&s, p);
                let w = sample_world_with(&s, p, weight, &mut rng);
                let m = split(&w, p);
                let a = &mut acc[stage];
                for (k, v) in [r.0, r.1, r.2, m.0, m.1, m.2, 1.0].into_iter().enumerate() {
                    a[k] += v;
                }
            }
            let legal = s.legal_actions_vec();
            let a = bots[(p + h) % 4].choose(&s, &legal, &mut rng);
            s.apply(a);
        }
    }
    println!(
        "tiles drawn   unseen honours   in hands: real  sampled   of held, paired: real  sampled"
    );
    for (st, a) in acc.iter().enumerate() {
        println!(
            "{:>3}-{:<3}       {:>6.1}               {:>5.1}%   {:>5.1}%                 {:>5.1}%   {:>5.1}%",
            st * 23,
            st * 23 + 22,
            a[0] / a[6],
            100.0 * a[1] / a[0],
            100.0 * a[4] / a[3],
            100.0 * a[2] / a[1].max(1.0),
            100.0 * a[5] / a[4].max(1.0),
        );
    }
}
