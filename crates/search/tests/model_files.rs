//! Every model file in `models/` that loaded before the discard-history input
//! (plan 2026-11-16 Task 3, done-when (d)) still loads and gives the same output
//! on 100 positions. The hashes were recorded with the code before that change.
//!
//! The model files are not in git (copied from the main checkout's `models/`), so
//! a missing file is skipped with a note. If the encoding itself changes on
//! purpose (`features.rs`'s golden hash), these hashes are re-recorded with it.

use mahjong_games::Game;
use mahjong_hk::{RuleSet, State};
use mahjong_search::Mlp;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// 100 positions from random legal play (no bot, so bot changes leave them alone).
fn positions() -> Vec<State> {
    let mut rng = SmallRng::seed_from_u64(20261116);
    let mut out = Vec::new();
    while out.len() < 100 {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let stop = rng.gen_range(0..70);
        for _ in 0..stop {
            if s.is_terminal() {
                break;
            }
            let legal = s.legal_actions_vec();
            s.apply(legal[rng.gen_range(0..legal.len())]);
        }
        if !s.is_terminal() {
            out.push(s);
        }
    }
    out
}

/// FNV-1a over the bit patterns of the outputs of `net` on every position.
fn output_hash(net: &Mlp, states: &[State]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for s in states {
        for x in net.forward(&net.encode_input(s)) {
            for b in x.to_bits().to_le_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        }
    }
    hash
}

/// `(file, hash of the outputs)`, recorded at commit 18b1b09 (before Task 3) with
/// the files of the main checkout's `models/` on 2026-11-16 (`belief.mjnn` was
/// already here and kept). Debug and release agree.
const GOLDEN: &[(&str, u64)] = &[
    ("belief-v2.mjnn", 135666235059543213),
    ("belief.mjnn", 15002789305761949555),
    ("cnn-distill-fp32.mjnn", 12406418968358885966),
    ("cnn-distill-v2-fp32.mjnn", 7581823408504652425),
    ("cnn-distill-v2.mjnn", 8509921048145687795),
    ("cnn-distill.mjnn", 16199717062268701523),
    ("cnn-ppo.mjnn", 2368955780929135675),
    ("cnn-test-fp32.mjnn", 3843490569965124928),
    ("cnn-test.mjnn", 1193252509238167656),
    ("distill.mjnn", 8860315183837135040),
    ("dmc.mjnn", 6225219863490870678),
    ("dmc0.mjnn", 13817480079443775240),
    ("ppo.mjnn", 1955907386064093650),
    ("ppo0.mjnn", 6653140515121474166),
    ("race-belief.mjnn", 3511677253258093659),
];

#[test]
fn every_model_file_gives_the_same_output_as_before() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");
    let states = positions();
    let mut checked = 0;
    for &(file, want) in GOLDEN {
        let path = format!("{dir}/{file}");
        if !std::path::Path::new(&path).exists() {
            eprintln!("{file}: not present, skipped");
            continue;
        }
        let net = Mlp::load(&path).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(output_hash(&net, &states), want, "{file}");
        checked += 1;
    }
    eprintln!("{checked} of {} model files checked", GOLDEN.len());
}

/// Prints the table for [`GOLDEN`] (`cargo test -p mahjong-search --test
/// model_files record -- --ignored --nocapture`).
#[test]
#[ignore]
fn record() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");
    let states = positions();
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|f| f.ends_with(".mjnn"))
        .collect();
    files.sort();
    for f in files {
        match Mlp::load(format!("{dir}/{f}")) {
            Ok(net) => println!("    (\"{f}\", {}),", output_hash(&net, &states)),
            Err(e) => println!("    // {f}: does not load: {e}"),
        }
    }
}
