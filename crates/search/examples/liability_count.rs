//! How often the old 12-tile rule moved points (plan Task 27, ADR 0004).
//!
//! `liability_count <deals> [seed] [bot]`: plays each deal once with four copies of
//! `bot` (default `stage`) under the default rules and counts the wins whose
//! winner had a liable seat (rules §8.3), split by self-draw and discard. On a
//! discard win the rule of 17 Oct charged the liable seat 3 shares and the
//! discarder nothing; the current rule charges the discarder 1 share and the
//! liable seat nothing. The report gives, for those wins, the points the old rule
//! moved: from the liable seat (3 shares it no longer pays) and to the discarder
//! (1 share it now pays).
use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::Bot;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

#[derive(Default, Clone, Copy)]
struct Count {
    hands: u64,
    wins: u64,
    discard_wins: u64,
    liable_self_draws: u64,
    liable_discard_wins: u64,
    liable_is_discarder: u64,
    /// Points the liable seats paid under the old rule on those discard wins.
    old_liable_points: i64,
    /// Points the discarders pay under the new rule on those discard wins.
    new_discarder_points: i64,
}

impl Count {
    fn add(mut self, o: Count) -> Count {
        self.hands += o.hands;
        self.wins += o.wins;
        self.discard_wins += o.discard_wins;
        self.liable_self_draws += o.liable_self_draws;
        self.liable_discard_wins += o.liable_discard_wins;
        self.liable_is_discarder += o.liable_is_discarder;
        self.old_liable_points += o.old_liable_points;
        self.new_discarder_points += o.new_discarder_points;
        self
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let deals: u64 = args.first().and_then(|d| d.parse().ok()).unwrap_or(400);
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(27);
    let name = args.get(2).map_or("stage", String::as_str);
    let bot = Bot::by_name(name).unwrap_or_else(|| panic!("unknown bot {name}"));
    let rules = RuleSet::default();
    let start = std::time::Instant::now();
    let c = (0..deals)
        .into_par_iter()
        .map(|d| {
            let deal = seed * 1_000_000 + d;
            let mut state = State::new_deal(rules, &mut ChaCha8Rng::seed_from_u64(deal));
            let mut rng = ChaCha8Rng::seed_from_u64(deal ^ 0x5eed);
            // Who was liable for whom just before the hand ended.
            let mut liable = [None; 4];
            while !state.is_terminal() {
                liable = std::array::from_fn(|s| state.liable_for(s));
                let legal = state.legal_actions_vec();
                let a = bot.choose(&state, &legal, &mut rng);
                state.apply(a);
            }
            let r = state.result().expect("a finished hand");
            let mut c = Count {
                hands: 1,
                ..Count::default()
            };
            for w in &r.wins {
                c.wins += 1;
                if !w.self_drawn {
                    c.discard_wins += 1;
                }
                let Some(l) = liable[w.seat] else { continue };
                if w.score.is_flower_win() {
                    continue;
                }
                if w.self_drawn {
                    c.liable_self_draws += 1;
                } else {
                    let share = rules.points(w.score.faan) as i64;
                    c.liable_discard_wins += 1;
                    c.liable_is_discarder += (w.from == Some(l)) as u64;
                    c.old_liable_points += 3 * share;
                    c.new_discarder_points += share;
                }
            }
            c
        })
        .reduce(Count::default, Count::add);
    let per = |x: f64| x / c.hands as f64;
    println!("liability_count: {deals} deals, seed {seed}, four `{name}`, default rules");
    println!(
        "hands {}, wins {}, discard wins {}",
        c.hands, c.wins, c.discard_wins
    );
    println!(
        "wins with a liable seat: self-drawn {} ({:.3} % of hands), on a discard {} ({:.3} % of hands; the liable seat itself discarded in {})",
        c.liable_self_draws,
        100.0 * per(c.liable_self_draws as f64),
        c.liable_discard_wins,
        100.0 * per(c.liable_discard_wins as f64),
        c.liable_is_discarder,
    );
    println!(
        "on those discard wins: old rule, liable seats paid {} points ({:.3} a hand); new rule, discarders pay {} ({:.3} a hand)",
        c.old_liable_points,
        per(c.old_liable_points as f64),
        c.new_discarder_points,
        per(c.new_discarder_points as f64),
    );
    println!("time {:.1} s", start.elapsed().as_secs_f64());
}
