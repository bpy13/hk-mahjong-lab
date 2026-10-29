//! Speed of one network forward pass and of a hand played by the network.
//!
//! `cargo run --release -p mahjong-search --example net_speed -- models/cnn-test.mjnn`
//! (any MJNN version; v3 is the suit-shared CNN of plan Task 14). The forward
//! pass runs on one thread (it never uses the thread pool).
use mahjong_games::{play_out, Game};
use mahjong_hk::{RuleSet, State};
use mahjong_search::features::{encode, STATE_LEN};
use mahjong_search::{Mlp, NetPolicy};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("path to an .mjnn file");
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let net = Arc::new(Mlp::load(&path).unwrap());
    println!(
        "{path}: MJNN v{}, {} parameters, {:.2} MB",
        net.version(),
        net.parameters(),
        bytes as f64 / 1e6
    );
    let mut rng = SmallRng::seed_from_u64(1);
    let s = State::new_deal(RuleSet::default(), &mut rng);
    let mut f = vec![0.0; STATE_LEN];
    let t = Instant::now();
    for _ in 0..200 {
        encode(&s, &mut f);
    }
    println!("encode: {:?}", t.elapsed() / 200);
    for _ in 0..20 {
        std::hint::black_box(net.forward(&f)); // warm-up
    }
    let t = Instant::now();
    for _ in 0..200 {
        std::hint::black_box(net.forward(&f));
    }
    println!("forward: {:?}", t.elapsed() / 200);
    let p = NetPolicy(net);
    let t = Instant::now();
    for _ in 0..20 {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        play_out(&mut s, &p, &mut rng);
        let _ = s.is_terminal();
    }
    println!("hand played by the network: {:?}", t.elapsed() / 20);
}
