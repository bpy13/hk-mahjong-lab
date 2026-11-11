//! The belief model's race heads (plan 2026-11-09 Task 8): the loader reads the old
//! five-head file and the new file with `state` and `mode`, and the race bot reads the
//! `mode` head through its fold hook and the `state` head for "one away".
//!
//! `tests/data/race-belief-tiny.mjnn` is a small network (hidden 16) from the smoke
//! training run (`python -m mahjong.belief.train --race --hidden 16 ...`); its numbers
//! mean nothing, only its layout matters here.

use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::race::{parse_spec, RaceBot};
use mahjong_search::{Agent, BeliefModel, Bot};
use rand::rngs::SmallRng;
use rand::SeedableRng;

const OLD: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/belief.mjnn");
const TINY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/race-belief-tiny.mjnn"
);

fn some_states(n: usize) -> Vec<State> {
    let bot = Bot::by_name("fast").unwrap();
    let mut rng = SmallRng::seed_from_u64(4);
    let mut out = Vec::new();
    while out.len() < n {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let mut i = 0;
        while s.current_player().is_some() {
            if i % 9 == 4 {
                out.push(s.clone());
            }
            i += 1;
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
        }
    }
    out.truncate(n);
    out
}

#[test]
fn the_old_file_loads_without_race_heads() {
    if !std::path::Path::new(OLD).exists() {
        eprintln!("models/belief.mjnn missing: skipped");
        return;
    }
    let m = BeliefModel::load(OLD).unwrap();
    assert!(!m.has_race_heads());
    let s = &some_states(1)[0];
    let out = m.run(s);
    for b in &out.opponents {
        assert!(b.state.is_none() && b.mode.is_none());
    }
    // The race bot built on it has no heads and keeps the v0 heuristics.
    let bot = parse_spec(&[("belief", OLD)]).unwrap();
    assert!(bot.heads.is_none());
}

#[test]
fn the_new_file_loads_with_state_and_mode_probabilities() {
    let m = BeliefModel::load(TINY).unwrap();
    assert!(m.has_race_heads());
    for s in some_states(20) {
        let out = m.run(&s);
        for b in &out.opponents {
            let st = b.state.expect("state head");
            let md = b.mode.expect("mode head");
            assert!((st.iter().sum::<f64>() - 1.0).abs() < 1e-9, "{st:?}");
            assert!((md.iter().sum::<f64>() - 1.0).abs() < 1e-9, "{md:?}");
            assert!(st.iter().chain(md.iter()).all(|p| (0.0..=1.0).contains(p)));
            // The five old heads are still there.
            assert!((b.outcome.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
    }
}

#[test]
fn the_race_bot_reads_the_mode_head_through_the_fold_hook() {
    let bot = match Agent::parse(&format!("race:belief={TINY}")) {
        Ok(Agent::Bot(Bot::Race(b))) => b,
        other => panic!("{other:?}"),
    };
    let heads = bot.heads.clone().expect("the tiny file has race heads");
    let mut checked = 0;
    for s in some_states(30) {
        let me = s.current_player().unwrap();
        let d = bot.decide(&s);
        assert!(s.legal_actions_vec().contains(&d.action));
        let out = heads.run(&s);
        for c in &d.candidates {
            for o in (0..4).filter(|&o| o != me) {
                let fold = out.seat(o).unwrap().mode.unwrap()[2];
                assert!(
                    (c.fold[o] - fold).abs() < 1e-9,
                    "seat {o}: {} vs {fold}",
                    c.fold[o]
                );
                let st = out.seat(o).unwrap().state.unwrap();
                let want = st[1].min(1.0 - c.tenpai[o]);
                assert!((c.one_away[o] - want).abs() < 1e-9);
            }
            checked += 1;
        }
    }
    assert!(checked > 20);
    // A fixed override (the test hook) still wins over the head.
    let mut fixed: RaceBot = bot.clone();
    fixed.fold_override = [Some(0.25); 4];
    let s = &some_states(3)[2];
    let me = s.current_player().unwrap();
    for c in &fixed.decide(s).candidates {
        for o in (0..4).filter(|&o| o != me) {
            assert!((c.fold[o] - 0.25).abs() < 1e-9);
        }
    }
}

#[test]
fn fitted_tables_parse_from_the_spec() {
    let b = parse_spec(&[
        ("a", "0.1/0.11/0.12/0.13/0.14/0.15"),
        ("b", "0.2/0.2/0.2/0.2/0.2/0.25"),
        ("q", "0.07/0.03/0.005"),
    ])
    .unwrap();
    assert_eq!(b.config.opp_a, [0.1, 0.11, 0.12, 0.13, 0.14, 0.15]);
    assert_eq!(b.config.opp_b[5], 0.25);
    assert_eq!(b.config.q, [0.07, 0.03, 0.005]);
    assert!(parse_spec(&[("a", "0.1/0.2")]).is_err());
    assert!(parse_spec(&[("q", "0.1/0.2/1.5")]).is_err());
    assert!(Agent::parse("race:q=0.08/0.04/0.01,a=0.15/0.15/0.15/0.15/0.15/0.15").is_ok());
    // v0's untuned constants stay in `RaceConfig::v0()`; since Task 20 the defaults are the fitted tables.
    let v0 = mahjong_search::RaceConfig::v0();
    assert_eq!(v0.opp_a, [0.15; 6]);
    assert_eq!(v0.q, [0.08, 0.04, 0.01]);
    assert_ne!(RaceBot::default().config.opp_a, v0.opp_a);
}
