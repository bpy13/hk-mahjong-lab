//! Rough speeds: full hands per bot, and one PIMC decision.
use mahjong_games::{play_out, Game};
use mahjong_hk::{RuleSet, State};
use mahjong_search::{pimc, Bot, PimcConfig};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let mut rng = SmallRng::seed_from_u64(1);
    for name in ["random", "efficiency", "fast", "value"] {
        let bot = Bot::by_name(name).unwrap();
        let n = if name == "random" { 2000 } else { 100 };
        let start = Instant::now();
        let mut wins = 0;
        for _ in 0..n {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            wins += !play_out(&mut s, &bot, &mut rng).winners.is_empty() as u32;
        }
        println!(
            "{name:>10}: {:>8.2?} per hand, {:.0}% of hands won",
            start.elapsed() / n,
            100.0 * wins as f64 / n as f64
        );
    }
    let s = State::new_deal(RuleSet::default(), &mut rng);
    let start = Instant::now();
    let a = pimc(
        &s,
        &PimcConfig {
            samples: 200,
            ..PimcConfig::default()
        },
    );
    println!(
        "pimc 200 samples x {} candidates: {:?} (threads {})",
        a.moves.len(),
        start.elapsed(),
        rayon::current_num_threads()
    );
    for m in &a.moves {
        println!(
            "  {:<16} {:+7.2} ±{:.2}  win {:.0}% deal-in {:.0}%",
            m.action.to_string(),
            m.mean,
            m.ci95,
            100.0 * m.win_rate,
            100.0 * m.deal_in_rate
        );
    }
    let _ = s.current_player();
}
