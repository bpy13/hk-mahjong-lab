//! The one-thread guard (plan Task 1, AC10). The app runs the search as
//! WebAssembly, where there is one thread and no thread pool. PIMC must give the
//! same answer there as on a many-thread pool for the same seed; every later
//! search change keeps this test green.

use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::{pimc, Analysis, Bot, PimcConfig};
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// A mid-hand position: `steps` decisions of the `fast` bot from a seeded deal.
fn position(seed: u64, steps: usize) -> State {
    let bot = Bot::by_name("fast").unwrap();
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    for _ in 0..steps {
        let legal = s.legal_actions_vec();
        let a = bot.choose(&s, &legal, &mut rng);
        s.apply(a);
        if s.is_terminal() {
            break;
        }
    }
    s
}

/// What must match: each candidate with its sample count, mean and paired
/// difference (as bits, so any change in the arithmetic shows).
fn key(a: &Analysis) -> Vec<(String, u32, u64, u64)> {
    a.moves
        .iter()
        .map(|m| {
            (
                m.action.to_string(),
                m.n,
                m.mean.to_bits(),
                m.diff.to_bits(),
            )
        })
        .collect()
}

#[test]
fn pimc_on_one_thread_matches_the_default_pool() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let config = PimcConfig {
        samples: 48,
        candidates: 3,
        seed: 7,
        ..PimcConfig::default()
    };
    let mut checked = 0;
    for (seed, steps) in [(1, 10), (2, 25), (3, 40)] {
        let s = position(seed, steps);
        if s.is_terminal() {
            continue;
        }
        let default_pool = pimc(&s, &config);
        let single = one.install(|| pimc(&s, &config));
        assert_eq!(one.current_num_threads(), 1);
        assert_eq!(
            key(&single),
            key(&default_pool),
            "seed {seed}, step {steps}"
        );
        assert_eq!(single.baseline, default_pool.baseline);
        assert_eq!(single.choose(config.z), default_pool.choose(config.z));
        checked += 1;
    }
    assert!(checked >= 2, "too few positions were still in progress");
}
