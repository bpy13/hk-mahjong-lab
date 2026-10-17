//! How developed are real opponents' hands, versus hands sampled from the hidden
//! tiles (best of K random draws by shanten)? Prints mean shanten by stage.
use mahjong_analysis::shanten;
use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::Bot;
use mahjong_tiles::TileCounts;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

fn main() {
    let bot = Bot::by_name("value").unwrap();
    let ks = [1usize, 4, 16, 64, 256];
    // [stage][0 = real, 1.. = K]
    let mut sums = [[0.0f64; 6]; 4];
    let mut counts = [0u32; 4];
    let mut rng = SmallRng::seed_from_u64(7);
    for _ in 0..150 {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        while let Some(p) = s.current_player() {
            if p == 0 && s.phase_name() == "turn" {
                let stage = ((83 - s.wall_remaining().min(83)) / 21).min(3);
                let (pool, sizes) = s.hidden_pool(0);
                let playing: Vec<_> = pool.into_iter().filter(|t| !t.is_bonus()).collect();
                for (o, &size) in sizes.iter().enumerate().skip(1) {
                    let melds = s.melds(o).len();
                    sums[stage][0] += shanten(s.hand(o), melds) as f64;
                    for (j, &k) in ks.iter().enumerate() {
                        let mut best = i8::MAX;
                        for _ in 0..k {
                            let mut v = playing.clone();
                            let (h, _) = v.partial_shuffle(&mut rng, size);
                            best = best.min(shanten(&TileCounts::from_tiles(h), melds));
                        }
                        sums[stage][j + 1] += best as f64;
                    }
                    counts[stage] += 1;
                }
            }
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
        }
    }
    println!("stage(wall used)  real   K=1   K=4  K=16  K=64 K=256");
    for st in 0..4 {
        let n = counts[st].max(1) as f64;
        let row: Vec<String> = sums[st].iter().map(|x| format!("{:5.2}", x / n)).collect();
        println!(
            "{:>2}-{:<2} (n={:4})   {}",
            st * 21,
            st * 21 + 20,
            counts[st],
            row.join(" ")
        );
    }
}
