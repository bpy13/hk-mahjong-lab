//! Are the deal-in numbers shown in study mode calibrated?
//!
//! 1. The per-tile danger heuristic ([`Danger`], the coloured bar under each tile):
//!    for every discard in real games, the estimate before the discard against
//!    whether that discard was won on, in buckets.
//! 2. The solver's 出銃率 (PIMC `deal_in_rate`, the chance of dealing in at any
//!    point in the rest of the hand): the estimate for the move actually played
//!    against whether that seat later dealt in.
//!
//! Games: seat 0 plays `value`, the others `defensive` (the UI's 中級).
//! `cargo run --release -p mahjong-search --example danger_check [hands] [pimc_every]`
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::{pimc, Bot, Danger, PimcConfig};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

const EDGES: [f64; 7] = [0.0, 0.01, 0.02, 0.04, 0.07, 0.10, 0.20];

fn bucket(p: f64) -> usize {
    EDGES.iter().rposition(|&e| p >= e).unwrap_or(0)
}

fn dealt_in(s: &State, seat: usize) -> bool {
    s.result()
        .is_some_and(|r| r.wins.iter().any(|w| !w.self_drawn && w.from == Some(seat)))
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let hands = args.first().copied().unwrap_or(400);
    let pimc_every = args.get(1).copied().unwrap_or(6);
    let me_bot = Bot::by_name("value").unwrap();
    let opp = Bot::by_name("defensive").unwrap();
    let mut rng = SmallRng::seed_from_u64(11);
    // [bucket] -> (sum of estimates, deal-ins, count)
    let mut tile_buckets = [(0.0f64, 0u32, 0u32); EDGES.len()];
    // PIMC: per decision (predicted deal-in, predicted win, actual deal-in, actual win, stage)
    let mut pimc_rows: Vec<(f64, f64, bool, bool, usize)> = Vec::new();
    for h in 0..hands {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let mut pending: Vec<(f64, f64, usize)> = Vec::new();
        let mut turns = 0usize;
        while let Some(p) = s.current_player() {
            let legal = s.legal_actions_vec();
            let a = if p == 0 {
                me_bot.choose(&s, &legal, &mut rng)
            } else {
                opp.choose(&s, &legal, &mut rng)
            };
            if p == 0 && s.phase_name() == "turn" && legal.len() > 1 {
                turns += 1;
                if turns % pimc_every == 0 {
                    let an = pimc(
                        &s,
                        &PimcConfig {
                            samples: 96,
                            candidates: 4,
                            seed: (h * 1000 + turns) as u64,
                            ..PimcConfig::default()
                        },
                    );
                    if let Some(m) = an.moves.iter().find(|m| m.action == a) {
                        let stage = (s.wall_remaining() < 45) as usize;
                        pending.push((m.deal_in_rate, m.win_rate, stage));
                    }
                }
            }
            if let Action::Discard(t) = a {
                let d = Danger::estimate(&s, p).tile(t);
                let mut next = s.clone();
                next.apply(a);
                // Resolve the claim window on this discard.
                while next.in_claim() {
                    let q = next.current_player().unwrap();
                    let l = next.legal_actions_vec();
                    let c = if q == 0 {
                        me_bot.choose(&next, &l, &mut rng)
                    } else {
                        opp.choose(&next, &l, &mut rng)
                    };
                    next.apply(c);
                }
                let hit = dealt_in(&next, p);
                let b = &mut tile_buckets[bucket(d)];
                b.0 += d;
                b.1 += hit as u32;
                b.2 += 1;
                s = next;
                continue;
            }
            s.apply(a);
            let _ = rng.gen::<u8>();
        }
        let won = s.result().unwrap().wins.iter().any(|w| w.seat == 0);
        let di = dealt_in(&s, 0);
        for (pd, pw, st) in pending {
            pimc_rows.push((pd, pw, di, won, st));
        }
    }

    println!("1. Per-tile danger heuristic vs immediate deal-in ({hands} hands)");
    println!("   bucket        discards   mean estimate   actual deal-in");
    for (i, (sum, hit, n)) in tile_buckets.iter().enumerate() {
        if *n == 0 {
            continue;
        }
        let hi = EDGES
            .get(i + 1)
            .map_or("+".to_string(), |e| format!("{:.0}%", e * 100.0));
        println!(
            "   {:>3.0}%–{:<5} {:>9}   {:>12.1}%   {:>8.1}% ± {:.1}",
            EDGES[i] * 100.0,
            hi,
            n,
            100.0 * sum / *n as f64,
            100.0 * *hit as f64 / *n as f64,
            196.0
                * ((*hit as f64 / *n as f64) * (1.0 - *hit as f64 / *n as f64) / *n as f64).sqrt(),
        );
    }
    let (s_all, h_all, n_all) = tile_buckets
        .iter()
        .fold((0.0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    println!(
        "   all           {:>9}   {:>12.1}%   {:>8.1}%",
        n_all,
        100.0 * s_all / n_all as f64,
        100.0 * h_all as f64 / n_all as f64
    );

    println!("\n2. Solver 出銃率 / 勝率 (rest of the hand) vs what happened");
    for (st, name) in [(0, "early (wall ≥ 45)"), (1, "late (wall < 45)")] {
        let rows: Vec<_> = pimc_rows.iter().filter(|r| r.4 == st).collect();
        let n = rows.len().max(1) as f64;
        let pd: f64 = rows.iter().map(|r| r.0).sum::<f64>() / n;
        let ad = rows.iter().filter(|r| r.2).count() as f64 / n;
        let pw: f64 = rows.iter().map(|r| r.1).sum::<f64>() / n;
        let aw = rows.iter().filter(|r| r.3).count() as f64 / n;
        println!(
            "   {name:<18} n={:>4}  deal-in predicted {:>5.1}% actual {:>5.1}%   win predicted {:>5.1}% actual {:>5.1}%",
            rows.len(),
            100.0 * pd,
            100.0 * ad,
            100.0 * pw,
            100.0 * aw
        );
    }
}
