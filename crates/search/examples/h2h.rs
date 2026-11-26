//! Plan 2026-11-16 Task 34: two agents head to head, two seats each.
//!
//! `h2h <A spec> <B spec> <deals> [seed]`: every wall is played in both 2v2
//! seatings, each in its 4 rotations: alternating (`A B A B`) and side by side
//! (`A A B B`), 8 games a wall, the same walls for both (same seed). A's score on
//! a wall is its mean points a hand over its 16 seat-games there (both seatings
//! weigh the same); B's is the same for B (the game is zero-sum, so B = −A).
//! Prints A's and B's points a hand with a 95 % range over walls (normal, walls
//! independent; seat and wall luck cancel inside each wall), A − B with its range,
//! and each seating alone, so the two can be compared.
use mahjong_hk::RuleSet;
use mahjong_search::tournament::{tournament, Seatings, TournamentConfig};
use mahjong_search::Agent;

/// Mean and 95 % half-width of per-wall scores.
fn mean_ci(x: &[f64]) -> (f64, f64) {
    let n = x.len().max(1) as f64;
    let m = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, 1.96 * (var / n).sqrt())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: h2h <A spec> <B spec> <deals> [seed]");
        std::process::exit(2);
    }
    let (a, b) = (args[0].as_str(), args[1].as_str());
    let deals: usize = args[2].parse().expect("deals");
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let pa = Agent::parse(a).expect("A spec");
    let pb = Agent::parse(b).expect("B spec");
    let start = std::time::Instant::now();
    let config = TournamentConfig {
        deals,
        seatings: Seatings::Rotations,
        rules: RuleSet::default(),
        seed,
        bootstrap: 0,
    };
    // Agent slots: alternating A B A B, side by side A A B B; A's slots per layout.
    let alt = tournament(&[pa.clone(), pb.clone(), pa.clone(), pb.clone()], &config);
    let side = tournament(&[pa.clone(), pa, pb.clone(), pb], &config);
    let (wa, ws) = (alt.wall_scores(), side.wall_scores());
    let mut s_alt = Vec::with_capacity(deals);
    let mut s_side = Vec::with_capacity(deals);
    let mut s_a = Vec::with_capacity(deals);
    let mut s_b = Vec::with_capacity(deals);
    let mut s_diff = Vec::with_capacity(deals);
    for (x, y) in wa.iter().zip(&ws) {
        let alt_a = (x[0] + x[2]) / 2.0;
        let alt_b = (x[1] + x[3]) / 2.0;
        let side_a = (y[0] + y[1]) / 2.0;
        let side_b = (y[2] + y[3]) / 2.0;
        let sa = (alt_a + side_a) / 2.0;
        let sb = (alt_b + side_b) / 2.0;
        s_alt.push(alt_a);
        s_side.push(side_a);
        s_a.push(sa);
        s_b.push(sb);
        s_diff.push(sa - sb);
    }
    let draws = alt
        .hands
        .iter()
        .chain(&side.hands)
        .filter(|h| h.result.wins.is_empty())
        .count();
    println!("{deals} walls x 8 games (A B A B and A A B B, 4 rotations each), seed {seed}");
    for (name, x) in [
        (format!("A {a}"), &s_a),
        (format!("B {b}"), &s_b),
        ("A - B".to_string(), &s_diff),
        ("A, alternating seats only".to_string(), &s_alt),
        ("A, side by side only".to_string(), &s_side),
    ] {
        let (m, c) = mean_ci(x);
        println!("{name:<40} {m:+7.3} ±{c:6.3} points a hand");
    }
    println!(
        "draws {:.1}%  ({:?})",
        100.0 * draws as f64 / (8 * deals) as f64,
        start.elapsed()
    );
}
