//! How much the payment rule of ADR 0005 moves points (plan Task 31).
//!
//! `payment_count <deals> [seed] [bot]`: plays each deal once with four copies of
//! `bot` (default `stage`) under the default rules and settles every win twice:
//! as the engine does now (rules §8.2 since 2026-11-23: the discarder pays 2
//! shares, a robbed kong's maker 3, 槓上槓 #8 and 大花糊 #31 are 10 faan) and as
//! the rule before (the discarder and a robbed kong's maker 1 share, #8 and #31
//! 8 faan; the 12-tile liability on a self-draw only in both). It prints how
//! many wins each change touched and the points a hand that moved between seats.
use mahjong_games::{Game, Policy};
use mahjong_hk::{Pattern, RuleSet, State};
use mahjong_search::Bot;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

#[derive(Default, Clone, Copy)]
struct Count {
    hands: u64,
    wins: u64,
    self_draws: u64,
    discard_wins: u64,
    robbed: u64,
    double_kong: u64,
    eight_flowers: u64,
    /// Σ over hands of Σ_seats |new delta − old delta| / 2: points moved.
    moved: i64,
    /// Σ over wins of the winner's points, new and old rule.
    won_new: i64,
    won_old: i64,
    /// Σ over discard wins (robbed kongs apart) of the discarder's payment.
    disc_new: i64,
    disc_old: i64,
}

impl Count {
    fn add(mut self, o: Count) -> Count {
        self.hands += o.hands;
        self.wins += o.wins;
        self.self_draws += o.self_draws;
        self.discard_wins += o.discard_wins;
        self.robbed += o.robbed;
        self.double_kong += o.double_kong;
        self.eight_flowers += o.eight_flowers;
        self.moved += o.moved;
        self.won_new += o.won_new;
        self.won_old += o.won_old;
        self.disc_new += o.disc_new;
        self.disc_old += o.disc_old;
        self
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let deals: u64 = args.first().and_then(|d| d.parse().ok()).unwrap_or(400);
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(31);
    let name = args.get(2).map_or("stage", String::as_str);
    let bot = Bot::by_name(name).unwrap_or_else(|| panic!("unknown bot {name}"));
    let rules = RuleSet::default();
    let cap = rules.faan_cap as u16;
    let start = std::time::Instant::now();
    let c = (0..deals)
        .into_par_iter()
        .map(|d| {
            let deal = seed * 1_000_000 + d;
            let mut state = State::new_deal(rules, &mut ChaCha8Rng::seed_from_u64(deal));
            let mut rng = ChaCha8Rng::seed_from_u64(deal ^ 0x5eed);
            while !state.is_terminal() {
                let legal = state.legal_actions_vec();
                let a = bot.choose(&state, &legal, &mut rng);
                state.apply(a);
            }
            let r = state.result().expect("a finished hand");
            let mut c = Count {
                hands: 1,
                ..Count::default()
            };
            let mut old = [0i64; 4];
            for w in &r.wins {
                c.wins += 1;
                // The faan before ADR 0005: #8 and #31 at 8.
                let old_faan = if w.score.has(Pattern::EightImmortals) {
                    c.eight_flowers += 1;
                    8u16.min(cap)
                } else if w.score.has(Pattern::DoubleKong) && !w.score.limit {
                    c.double_kong += 1;
                    (w.score.total - 2).min(cap)
                } else {
                    w.score.faan as u16
                };
                let share = rules.points(old_faan as u8) as i64;
                let mut paid = [0i64; 4];
                if w.self_drawn {
                    c.self_draws += 1;
                    match w.liable {
                        Some(l) => paid[l] = 3 * share,
                        None => {
                            for (p, x) in paid.iter_mut().enumerate() {
                                if p != w.seat {
                                    *x = share;
                                }
                            }
                        }
                    }
                } else {
                    let f = w.from.expect("a discard win has a source");
                    paid[f] = share;
                    if w.robbed {
                        c.robbed += 1;
                    } else {
                        c.discard_wins += 1;
                        c.disc_new += w.paid_by[f] as i64;
                        c.disc_old += share;
                    }
                }
                for p in 0..4 {
                    old[p] -= paid[p];
                    old[w.seat] += paid[p];
                }
                c.won_new += w.paid_by.iter().map(|&x| x as i64).sum::<i64>();
                c.won_old += paid.iter().sum::<i64>();
            }
            let diff: i64 = (0..4).map(|p| (r.deltas[p] as i64 - old[p]).abs()).sum();
            c.moved += diff / 2;
            c
        })
        .reduce(Count::default, Count::add);
    let per = |x: i64| x as f64 / c.hands as f64;
    let pct = |x: u64| 100.0 * x as f64 / c.hands as f64;
    println!("payment_count: {deals} deals, seed {seed}, four `{name}`, default rules");
    println!(
        "hands {}, wins {}: self-drawn {}, on a discard {} ({:.2} % of hands), robbed kongs {} ({:.3} %), 槓上槓 {} ({:.3} %), 大花糊 {} ({:.3} %)",
        c.hands,
        c.wins,
        c.self_draws,
        c.discard_wins,
        pct(c.discard_wins),
        c.robbed,
        pct(c.robbed),
        c.double_kong,
        pct(c.double_kong),
        c.eight_flowers,
        pct(c.eight_flowers),
    );
    println!(
        "discarders paid: new rule {} ({:.3} a hand), old rule {} ({:.3} a hand)",
        c.disc_new,
        per(c.disc_new),
        c.disc_old,
        per(c.disc_old)
    );
    println!(
        "winners received: new rule {:.3} a hand, old rule {:.3} a hand",
        per(c.won_new),
        per(c.won_old)
    );
    println!(
        "points moved between seats by the change: {} ({:.3} a hand, {:.3} a seat a hand)",
        c.moved,
        per(c.moved),
        per(c.moved) / 4.0 * 2.0
    );
    println!("time {:.1} s", start.elapsed().as_secs_f64());
}
