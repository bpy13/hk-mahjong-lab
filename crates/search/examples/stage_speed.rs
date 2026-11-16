//! Time per decision of the three-stage bot, one thread.
//!
//! `stage_speed <deals> <bot name>...`. Each deal is played once with the named bot
//! in seat 0 and value bots in the other seats; only the named bot's decisions are timed.
use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::Bot;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let deals: u64 = args.first().and_then(|d| d.parse().ok()).unwrap_or(200);
    let value = Bot::by_name("value").expect("the value bot");
    for name in &args[1..] {
        let bot = Bot::by_name(name).unwrap_or_else(|| panic!("unknown bot {name}"));
        let mut times: Vec<Duration> = Vec::new();
        for d in 0..deals {
            let mut state = State::new_deal(RuleSet::default(), &mut ChaCha8Rng::seed_from_u64(d));
            let mut rng = ChaCha8Rng::seed_from_u64(d + 1_000_000);
            while !state.is_terminal() {
                let legal = state.legal_actions_vec();
                let seat = state.current_player().expect("not over");
                let action = if seat == 0 {
                    let start = Instant::now();
                    let a = bot.choose(&state, &legal, &mut rng);
                    times.push(start.elapsed());
                    a
                } else {
                    value.choose(&state, &legal, &mut rng)
                };
                state.apply(action);
            }
        }
        times.sort();
        let n = times.len().max(1);
        let total: Duration = times.iter().sum();
        let at = |q: f64| times[((n - 1) as f64 * q) as usize];
        println!(
            "{name:<14} {} decisions, mean {:?}, median {:?}, 99th {:?}, max {:?}, total {:?}",
            times.len(),
            total / n as u32,
            at(0.5),
            at(0.99),
            at(1.0),
            total
        );
    }
}
