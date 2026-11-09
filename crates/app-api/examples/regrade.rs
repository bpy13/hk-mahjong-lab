//! Rechecks one decision of a kept hand (the app's `hkgames.keptDeal.mahjong` JSON) with
//! searches far bigger than the phone's review, to see whether the review's verdict holds.
//!
//! `cargo run --release -p mahjong-app-api --example regrade -- <hand.json> <mode>`
//! (README "Regrade a kept hand"). Modes, where `<index>` is a history index (default:
//! the last move):
//! - `<index> 0`: the scene at that decision (every hand as it stood then, the discards,
//!   the legal moves) and the phone-sized grade (12 s per hand split over the player's
//!   decisions, 0.2–1.5 s each, as the app's review).
//! - `<index> <scale>`: the same, then three bigger searches with `scale` × the samples:
//!   1. the review's own method with 50x the samples;
//!   2. opponents' hands drawn from the trained belief model;
//!   3. the belief model plus the stronger `ev` bot playing the player's later moves.
//! - `all`: every decision of the human seat, graded as the phone's review grades it.
//! - `<index> <scale> trace`: `scale` × 1,000 sampled futures of every legal move, played
//!   with the review's rollout bot (`fast`) and with `ev` as the player's own policy:
//!   the player's first move after the decision, win / other-wins / draw rates, mean
//!   points, the most common winning hands.
//! - `<index> 1 count`: copies of each pin tile in the wall and the opponents' hands,
//!   real and in five sampled worlds (checks the sampler).
use mahjong_app_api::shapes::{legal_action, rules_from_json, tile_from_id, TurnJson};
use mahjong_app_api::Api;
use mahjong_games::Game;
use mahjong_hk::State;
use mahjong_search::{
    pimc_anchored, Analysis, BeliefModel, Bot, PimcConfig, Rollout, WorldSampler,
};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Instant;

fn show(name: &str, a: &Analysis, secs: f64) {
    println!("\n== {name}: {} samples in {secs:.1} s", a.samples);
    for m in &a.moves {
        println!(
            "  {:<14} mean {:+7.2}  vs baseline {:+6.2} ± {:.2}  win {:.3}  deal-in {:.3}  draw {:.3}",
            m.action.to_string(),
            m.mean,
            m.diff,
            1.96 * m.diff_se,
            m.win_rate,
            m.deal_in_rate,
            m.draw_rate
        );
    }
}

/// Plays `n` sampled futures of every legal move with the review's rollout policy and with
/// the `ev` bot as the player's own policy (same samples), and summarises what happens:
/// the player's first move after the decision, how the hand ends, and the final hand.
fn trace(s: &State, seat: usize, n: usize) {
    use mahjong_games::Policy;
    use rand::SeedableRng;
    use std::collections::BTreeMap;
    let policies = [
        ("fast", Rollout::for_seat(seat)),
        (
            "ev",
            Rollout::with(
                seat,
                Bot::by_name("ev").unwrap(),
                Rollout::default_opponents(),
            ),
        ),
    ];
    for &cand in &s.legal_actions_vec() {
        for (pname, pol) in &policies {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
            let (mut wins, mut draws, mut lost, mut pts) = (0, 0, 0, 0.0);
            let mut first: BTreeMap<String, (u32, u32)> = BTreeMap::new();
            let mut endings: BTreeMap<String, u32> = BTreeMap::new();
            for _ in 0..n * 1000 {
                let mut w = mahjong_search::sample_world(s, seat, &mut rng);
                w.apply(cand);
                let mut first_move: Option<String> = None;
                while let Some(p) = w.current_player() {
                    let legal = w.legal_actions_vec();
                    let a = pol.choose(&w, &legal, &mut rng);
                    if p == seat && first_move.is_none() && legal.len() > 1 {
                        first_move = Some(a.to_string());
                    }
                    w.apply(a);
                }
                let o = w.outcome().unwrap();
                pts += o.scores[seat];
                let won = o.winners.contains(&seat);
                if won {
                    wins += 1;
                } else if o.winners.is_empty() {
                    draws += 1;
                } else {
                    lost += 1;
                }
                let e = first.entry(first_move.unwrap_or("-".into())).or_default();
                e.0 += 1;
                e.1 += won as u32;
                if won {
                    *endings
                        .entry(format!("{} {:?}", w.hand(seat), w.melds(seat)))
                        .or_default() += 1;
                }
            }
            let t = (n * 1000) as f64;
            println!(
                "\n### {cand} then {pname}: win {:.3} other-wins {:.3} draw {:.3} mean {:+.1}",
                wins as f64 / t,
                lost as f64 / t,
                draws as f64 / t,
                pts / t
            );
            println!("  first own move -> (count, wins):");
            for (k, v) in &first {
                println!("    {k:<14} {:>6} {:>6}", v.0, v.1);
            }
            let mut top: Vec<_> = endings.into_iter().collect();
            top.sort_by_key(|e| std::cmp::Reverse(e.1));
            println!("  most common winning hands:");
            for (k, v) in top.iter().take(6) {
                println!("    {v:>6}  {k}");
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let kept: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let pos = &kept["position"];
    let history: Vec<TurnJson> = serde_json::from_value(pos["history"].clone()).unwrap();
    if args.get(2).map(String::as_str) == Some("all") {
        // Every decision of the human seat, graded as the phone's review grades it.
        let human = pos["human_seats"][0].as_u64().unwrap_or(0) as usize;
        let idx: Vec<usize> = (0..history.len())
            .filter(|&i| history[i].player == human)
            .collect();
        let step = (12000 / idx.len().max(1)).clamp(200, 1500) as u32;
        let api = Api::new();
        for i in idx {
            let mut prefix = pos.clone();
            prefix["history"] = json!(history[..i]);
            let g: Value = serde_json::from_str(&api.grade(
                &prefix.to_string(),
                &serde_json::to_string(&history[i].action).unwrap(),
                step,
            ))
            .unwrap();
            println!(
                "{i:>3} played {:<40} best {:<40} grade {:<11} loss {}",
                serde_json::to_string(&history[i].action).unwrap(),
                g["best"].to_string(),
                g["grade"].as_str().unwrap_or("?"),
                g["loss"]
            );
        }
        return;
    }
    let at: usize = args
        .get(2)
        .map(|s| s.parse().unwrap())
        .unwrap_or(history.len() - 1);
    let scale: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);
    let rules = rules_from_json(pos.get("rules")).unwrap();
    let wall = pos["wall"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| tile_from_id(t.as_str().unwrap()).unwrap())
        .collect();
    let mut s = State::from_wall(rules, wall);
    for t in &history[..at] {
        let a = legal_action(&s, &t.action).unwrap();
        s.apply(a);
    }
    let seat = s.current_player().unwrap();
    let played = legal_action(&s, &history[at].action).unwrap();

    println!(
        "decision at history index {at} of {} (seat {seat}, phase {})",
        history.len(),
        s.phase_name()
    );
    println!("tiles left in the wall: {}", s.wall_remaining());
    for p in 0..4 {
        println!(
            "seat {p}: concealed {}  melds {:?}  bonus {:#010b}",
            s.hand(p),
            s.melds(p),
            s.bonus(p)
        );
    }
    let river: Vec<String> = s
        .river()
        .iter()
        .map(|(p, t, c)| format!("{p}:{t}{}", if *c { "*" } else { "" }))
        .collect();
    println!(
        "discards in order (seat:tile, * = claimed): {}",
        river.join(" ")
    );
    println!(
        "legal: {:?}",
        s.legal_actions_vec()
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
    );
    println!("played: {played}");
    let mut after = s.clone();
    after.apply(played);
    println!("outcome if played: {:?}", after.outcome());

    // The phone's review gives each decision 12 s / decisions, clamped to 0.2–1.5 s.
    let decisions = history.iter().filter(|t| t.player == seat).count();
    let step = (12000 / decisions.max(1)).clamp(200, 1500) as u32;
    let mut prefix = pos.clone();
    prefix["history"] = json!(history[..at]);
    let t0 = Instant::now();
    let phone = Api::new().grade(
        &prefix.to_string(),
        &serde_json::to_string(&history[at].action).unwrap(),
        step,
    );
    println!(
        "\n== phone-sized grade ({step} ms per decision, native, {:.1} s): {phone}",
        t0.elapsed().as_secs_f64()
    );

    if scale == 0 {
        return; // scene and phone grade only
    }
    if args.get(4).map(String::as_str) == Some("count") {
        // Copies of each pin tile outside the player's own hand and melds: real vs sampled.
        use rand::SeedableRng;
        let count = |w: &State, t: &str| -> String {
            let tile = tile_from_id(t).unwrap();
            let wall = w.wall().iter().filter(|&&x| x == tile).count();
            let opp: usize = (0..4)
                .filter(|&p| p != seat)
                .map(|p| w.hand(p)[tile] as usize)
                .sum();
            format!("{t}: wall {wall} opp-hands {opp}")
        };
        let pins = ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p"];
        println!("real:    {}", pins.map(|t| count(&s, t)).join(" | "));
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        for _ in 0..5 {
            let w = mahjong_search::sample_world(&s, seat, &mut rng);
            println!("sampled: {}", pins.map(|t| count(&w, t)).join(" | "));
        }
        return;
    }
    if args.get(4).map(String::as_str) == Some("trace") {
        trace(&s, seat, scale);
        return;
    }
    let fast = Rollout::for_seat(seat);
    let value = Bot::by_name("value").unwrap();
    let cfg = |n: usize| PimcConfig {
        samples: n * scale,
        candidates: 8,
        seed: 7,
        ..PimcConfig::default()
    };
    let t = Instant::now();
    let a = pimc_anchored(&s, &cfg(15000), &WorldSampler::BestOfK, &value, &fast);
    show("1. review method, big", &a, t.elapsed().as_secs_f64());

    let model = Arc::new(
        BeliefModel::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/belief.mjnn"
        ))
        .unwrap(),
    );
    let belief = WorldSampler::belief(model, true);
    let t = Instant::now();
    let a = pimc_anchored(&s, &cfg(5000), &belief, &value, &fast);
    show("2. belief sampler", &a, t.elapsed().as_secs_f64());

    let ev = Bot::by_name("ev").unwrap();
    let strong = Rollout::with(seat, ev, Rollout::default_opponents());
    let t = Instant::now();
    let a = pimc_anchored(&s, &cfg(1500), &belief, &value, &strong);
    show(
        "3. belief sampler + ev rollouts",
        &a,
        t.elapsed().as_secs_f64(),
    );
}
