//! Prints PIMC decisions that override the value bot.
use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::{pimc, Bot, PimcConfig};
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn main() {
    let value = Bot::by_name("value").unwrap();
    let mut rng = SmallRng::seed_from_u64(3);
    for hand in 0..3 {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        while let Some(p) = s.current_player() {
            let legal = s.legal_actions_vec();
            let a = if p == 0 && legal.len() > 1 {
                let an = pimc(
                    &s,
                    &PimcConfig {
                        samples: 64,
                        candidates: 4,
                        seed: 1,
                        ..PimcConfig::default()
                    },
                );
                let c = an.choose(1.0);
                if c != an.baseline {
                    println!(
                        "hand {hand} {} hand {} baseline {} -> {}",
                        s.phase_name(),
                        s.hand(0),
                        an.baseline,
                        c
                    );
                    for m in &an.moves {
                        println!(
                            "   {:<18} mean {:+6.2} diff {:+6.2} se {:5.2} win {:.2} deal {:.2}",
                            m.action.to_string(),
                            m.mean,
                            m.diff,
                            m.diff_se,
                            m.win_rate,
                            m.deal_in_rate
                        );
                    }
                }
                c
            } else {
                value.choose(&s, &legal, &mut rng)
            };
            s.apply(a);
        }
        println!("hand {hand} result {:?}", s.result().unwrap().deltas);
    }
}
