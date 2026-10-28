//! The 24-seating duplicate harness (plan Task 3).

use mahjong_hk::RuleSet;
use mahjong_search::{duplicate, tournament, Agent, Seatings, TournamentConfig};

fn agents(specs: [&str; 4]) -> [Agent; 4] {
    specs.map(|s| Agent::parse(s).unwrap())
}

#[test]
fn identical_value_bots_score_zero_within_the_bootstrap_interval() {
    let config = TournamentConfig {
        deals: 12,
        seatings: Seatings::All,
        rules: RuleSet::default(),
        seed: 5,
        bootstrap: 2000,
    };
    let t = tournament(&agents(["value"; 4]), &config);
    assert_eq!(t.hands.len(), 12 * 24);

    // Deltas sum to zero in every hand.
    for h in &t.hands {
        assert_eq!(h.result.deltas.iter().sum::<i32>(), 0);
    }
    // Every wall is played in all 24 seatings, each a permutation.
    for wall in 0..12 {
        let mut seatings: Vec<[u8; 4]> = t
            .hands
            .iter()
            .filter(|h| h.wall == wall)
            .map(|h| h.seating)
            .collect();
        seatings.sort();
        seatings.dedup();
        assert_eq!(seatings.len(), 24);
    }

    let s = t.summary();
    let total: f64 = s.agents.iter().map(|a| a.points).sum();
    assert_eq!(total, 0.0);
    for a in &s.agents {
        assert_eq!(a.hands, 12 * 24);
        assert!(
            a.bootstrap.contains(0.0),
            "mean {} outside its bootstrap interval {:?}",
            a.mean(),
            a.bootstrap
        );
        assert!(a.bootstrap.lo <= a.mean() && a.mean() <= a.bootstrap.hi);
    }
    // Paired differences: one per pair; each agent's wall scores sum to zero.
    assert_eq!(s.pairs.len(), 6);
    for w in t.wall_scores() {
        assert!(w.iter().sum::<f64>().abs() < 1e-9);
    }
    for p in &s.pairs {
        assert!(p.bootstrap.lo <= p.bootstrap.hi);
        let direct = s.agents[p.a].mean() - s.agents[p.b].mean();
        assert!((p.mean - direct).abs() < 1e-9);
    }
    // Deal-ins split by the winning agent add up to the deal-in count.
    for a in &s.agents {
        assert_eq!(a.deal_ins_to.iter().sum::<u32>(), a.deal_ins);
        let draws_per_agent = s.draws;
        assert_eq!(a.draws, draws_per_agent);
    }
}

#[test]
fn rotations_match_the_old_duplicate_wrapper_and_json_parses_back() {
    let a = agents(["fast", "efficiency", "value", "random"]);
    let rules = RuleSet::default();
    let old = duplicate(&a, 6, rules, 3);
    let t = tournament(
        &a,
        &TournamentConfig {
            deals: 6,
            seatings: Seatings::Rotations,
            rules,
            seed: 3,
            bootstrap: 200,
        },
    );
    let s = t.summary();
    for (o, n) in old.agents.iter().zip(&s.agents) {
        assert_eq!(o.points, n.points);
        assert_eq!(o.wins, n.wins);
        assert_eq!(o.deal_ins, n.deal_ins);
    }
    assert_eq!(old.draws, s.draws);

    // The same seed gives the same results.
    let again = tournament(&a, &t.config).summary();
    for (x, y) in s.agents.iter().zip(&again.agents) {
        assert_eq!(x.points, y.points);
    }

    let names = ["fast", "efficiency", "value", "random"].map(String::from);
    let json = s.to_json(&names);
    assert!(json.contains("\"format\": \"hk-tournament/1\""));
    assert!(json.contains("\"pattern_shares\""));
    // Balanced braces and brackets: a cheap well-formedness check (the Python
    // test parses it for real).
    let depth = json.chars().fold(0i32, |d, c| match c {
        '{' | '[' => d + 1,
        '}' | ']' => d - 1,
        _ => d,
    });
    assert_eq!(depth, 0);
}
