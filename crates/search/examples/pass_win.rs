//! Task 21: the race bot with and without passing a win, paired on the same hands.
//!
//! ```text
//! cargo run --release -p mahjong-search --example pass_win -- [walls=1000] [spec_on] [spec_off]
//! ```
//!
//! Each wall is played four times (the race seat rotates), once with passing (`race`) and
//! once without (`race:passwin=off`), against three `value` bots with the same random
//! stream. Hands where the two plays never differ in a pass are identical, so the paired
//! difference is the earnings of the passes. Prints the match (mean points per hand of
//! each, paired difference with a 95 % interval) and the passes (how often, what the win
//! was worth, what the hand paid afterwards).

use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::{Agent, Bot};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rayon::prelude::*;

#[derive(Default, Clone, Copy)]
struct Hand {
    on: f64,
    off: f64,
    passes: u32,
    /// Sum of the win values passed (points).
    passed_win: f64,
    /// Whether I won the hand in the passing play.
    won_on: bool,
    won_off: bool,
}

fn play(bot: &Agent, seat: usize, wall_seed: u64, count: bool) -> (f64, u32, f64, bool) {
    let value = Agent::parse("value").unwrap();
    let mut rng = SmallRng::seed_from_u64(wall_seed);
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    let (mut passes, mut passed_win) = (0u32, 0.0);
    let mut legal = Vec::new();
    while let Some(me) = s.current_player() {
        s.legal_actions(&mut legal);
        let a = if me == seat {
            match bot {
                Agent::Bot(Bot::Race(b)) if count => {
                    let d = b.decide(&s);
                    if let Some(w) = d.win_choice {
                        if !matches!(d.action, Action::Win | Action::SelfDrawnWin) {
                            passes += 1;
                            passed_win += w.win;
                        }
                    }
                    d.action
                }
                _ => bot.choose(&s, &legal, &mut rng),
            }
        } else {
            value.choose(&s, &legal, &mut rng)
        };
        s.apply(a);
    }
    let r = s.result().expect("hand over");
    let won = r.wins.iter().any(|w| w.seat == seat);
    (r.deltas[seat] as f64, passes, passed_win, won)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let walls: u64 = args.first().and_then(|a| a.parse().ok()).unwrap_or(1000);
    let on = Agent::parse(args.get(1).map_or("race", String::as_str)).unwrap();
    let off = Agent::parse(args.get(2).map_or("race:passwin=off", String::as_str)).unwrap();
    let hands: Vec<Hand> = (0..walls * 4)
        .into_par_iter()
        .map(|i| {
            let (seat, wall) = ((i % 4) as usize, 1000 + i / 4);
            let (p_on, passes, passed_win, won_on) = play(&on, seat, wall, true);
            let (p_off, _, _, won_off) = play(&off, seat, wall, false);
            Hand {
                on: p_on,
                off: p_off,
                passes,
                passed_win,
                won_on,
                won_off,
            }
        })
        .collect();
    let n = hands.len() as f64;
    let stat = |f: &dyn Fn(&Hand) -> f64, hs: &[&Hand]| -> (f64, f64) {
        let m = hs.len().max(1) as f64;
        let mean = hs.iter().map(|h| f(h)).sum::<f64>() / m;
        let var = hs.iter().map(|h| (f(h) - mean).powi(2)).sum::<f64>() / m;
        (mean, 1.96 * (var / m).sqrt())
    };
    let all: Vec<&Hand> = hands.iter().collect();
    let (m_on, c_on) = stat(&|h| h.on, &all);
    let (m_off, c_off) = stat(&|h| h.off, &all);
    let (m_d, c_d) = stat(&|h| h.on - h.off, &all);
    println!("hands {n}");
    println!("with passing    {m_on:+7.2} ± {c_on:.2} per hand");
    println!("without passing {m_off:+7.2} ± {c_off:.2} per hand");
    println!("paired on - off {m_d:+7.2} ± {c_d:.2} per hand");
    let passed: Vec<&Hand> = hands.iter().filter(|h| h.passes > 0).collect();
    println!(
        "hands with a pass: {} ({:.1} % of hands), {} passes",
        passed.len(),
        100.0 * passed.len() as f64 / n,
        passed.iter().map(|h| h.passes).sum::<u32>()
    );
    if !passed.is_empty() {
        let (win, _) = stat(&|h| h.passed_win, &passed);
        let (on_pts, c1) = stat(&|h| h.on, &passed);
        let (off_pts, c2) = stat(&|h| h.off, &passed);
        let (d, c3) = stat(&|h| h.on - h.off, &passed);
        let won_on = passed.iter().filter(|h| h.won_on).count();
        let won_off = passed.iter().filter(|h| h.won_off).count();
        println!("  win value passed (mean per hand): {win:.1}");
        println!("  those hands with passing:    {on_pts:+.1} ± {c1:.1}, I won {won_on}");
        println!("  those hands without passing: {off_pts:+.1} ± {c2:.1}, I won {won_off}");
        println!("  paired difference:           {d:+.1} ± {c3:.1}");
    }
}
