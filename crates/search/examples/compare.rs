//! Quick duplicate comparison: `cargo run --release --example compare -- value efficiency efficiency efficiency 100`
use mahjong_hk::RuleSet;
use mahjong_search::{duplicate, Agent};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let specs: Vec<&str> = args.iter().take(4).map(String::as_str).collect();
    let deals: usize = args.get(4).and_then(|d| d.parse().ok()).unwrap_or(50);
    let agents: [Agent; 4] = std::array::from_fn(|i| Agent::parse(specs[i]).unwrap());
    let start = std::time::Instant::now();
    let (res, draws) = duplicate(&agents, deals, RuleSet::default(), 1);
    for (spec, r) in specs.iter().zip(&res) {
        println!(
            "{spec:<28} {:+7.2} ±{:5.2}/hand  win {:4.1}%  deal-in {:4.1}%  avg faan {:.1}",
            r.mean(),
            r.ci95(),
            100.0 * r.wins as f64 / r.hands as f64,
            100.0 * r.deal_ins as f64 / r.hands as f64,
            r.win_faan as f64 / r.wins.max(1) as f64
        );
    }
    println!(
        "draws {:.1}%  ({:?})",
        100.0 * draws as f64 / (4 * deals) as f64,
        start.elapsed()
    );
}
