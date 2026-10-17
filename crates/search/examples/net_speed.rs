//! Speed of one network forward pass and of a hand played by the network.
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
    let net = Arc::new(Mlp::load(&path).unwrap());
    let mut rng = SmallRng::seed_from_u64(1);
    let s = State::new_deal(RuleSet::default(), &mut rng);
    let mut f = vec![0.0; STATE_LEN];
    let t = Instant::now();
    for _ in 0..200 {
        encode(&s, &mut f);
    }
    println!("encode: {:?}", t.elapsed() / 200);
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
