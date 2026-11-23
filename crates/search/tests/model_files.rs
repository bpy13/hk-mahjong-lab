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

/// `(file, hash of the outputs)`, first recorded at commit 18b1b09 (before Task
/// 3) with the files of the main checkout's `models/` on 2026-11-16.
///
/// 2026-11-23 (plan 2026-11-16 Task 32): recorded again on purpose. The rules
/// changed which moves are legal (the 食換 ban after a claim, §4.1.4; the flower
/// choice, §7.5.1) and 人糊 (#33) ends some hands, so random legal play reaches
/// other positions. The encoders and network code did not change in that task.
/// The files not kept in this checkout were read from the standalone
/// `repo\hk-mahjong-lab\models\`; the newer ones (`belief-order*`,
/// `stage-imitation*`) are added.
const GOLDEN: &[(&str, u64)] = &[
    ("belief-order-reg.mjnn", 5067023546923881921),
    ("belief-order.mjnn", 1314433444243534169),
    ("belief-v2.mjnn", 12947365755819123908),
    ("belief.mjnn", 15834856924182770216),
    ("cnn-distill-fp32.mjnn", 14720371192874532388),
    ("cnn-distill-v2-fp32.mjnn", 8980125196881684588),
    ("cnn-distill-v2.mjnn", 9573007323899507942),
    ("cnn-distill.mjnn", 7011650531028678589),
    ("cnn-ppo.mjnn", 490894805378347328),
    ("cnn-test-fp32.mjnn", 14832671212717586367),
    ("cnn-test.mjnn", 9751140746710265002),
    ("distill.mjnn", 18067365240658524269),
    ("dmc.mjnn", 1903802436992702992),
    ("dmc0.mjnn", 16214124082308366771),
    ("ppo.mjnn", 7578930600296541684),
    ("ppo0.mjnn", 2832328489306107864),
    ("race-belief.mjnn", 245013708503341248),
    ("stage-imitation-fp32.mjnn", 13820460691784066295),
    ("stage-imitation-gate0-fp32.mjnn", 11623025917313762764),
    ("stage-imitation-gate0.mjnn", 6432698534209856701),
    ("stage-imitation.mjnn", 343048139103821153),
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

/// Plan 2026-11-16 Task 32: every network in `models/` still loads and plays a
/// duplicate match under the new move set (食換 ban, flower choice) with no
/// illegal move (the tournament checks every move). A policy network plays as
/// `net:path=...`; a belief network as the reading of `stage-safe:t=1`.
#[test]
fn every_model_file_plays_a_match_with_no_illegal_move() {
    use mahjong_search::agent::Agent;
    use mahjong_search::tournament::duplicate;
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("no models/ folder, skipped");
        return;
    };
    let mut files: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path().to_string_lossy().into_owned())
        .filter(|p| p.ends_with(".mjnn"))
        .collect();
    files.sort();
    let stage = Agent::parse("stage").unwrap();
    for path in &files {
        let net = Mlp::load(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let policy = net
            .output("policy")
            .is_some_and(|r| r.len() == mahjong_hk::NUM_ACTION_CODES);
        let spec = if policy {
            format!("net:path={path}")
        } else {
            format!("stage-safe:t=1,belief={path}")
        };
        let agent = Agent::parse(&spec).unwrap_or_else(|e| panic!("{path}: {e}"));
        let agents = [agent, stage.clone(), stage.clone(), stage.clone()];
        let r = duplicate(&agents, 6, RuleSet::default(), 32);
        assert_eq!(r.agents[0].hands, 24, "{path}");
        eprintln!(
            "{path}: {:+.2} a hand over 24 hands",
            r.agents[0].points / 24.0
        );
    }
}
