//! The race bot v0 (plan 2026-11-09 Task 6, `docs/race-bot.md`): legal play, Brian's
//! pung hand (§9), the `ev` bot's deliberate deal-in (`docs/ev-bot.md` §5 example 2),
//! the mode choice against two threats (§5.1, §5.3), the time budget and the spec.

mod common;

use common::{build, position_2, seat, t, Setup, Stub};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::race::{Mode, RaceBot, RaceConfig, RaceDecision};
use mahjong_search::{tournament, Agent, Beliefs, Bot, SeatBelief, Seatings, TournamentConfig};
use mahjong_tiles::Tile;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::Arc;
use std::time::Instant;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/brian-2026-11-09.json"
);

/// The quoted strings of the JSON array that follows `"key": [`.
fn array_after<'a>(json: &'a str, key: &str) -> Vec<&'a str> {
    let start = json.find(&format!("\"{key}\": [")).expect("key") + key.len() + 4;
    let body = &json[start..start + json[start..].find(']').expect("array end")];
    body.split('"').skip(1).step_by(2).collect()
}

/// Brian's hand replayed up to (not including) history index `at`, as in
/// `rollout_fixes.rs` (the fixture uses the default rules).
fn brian_at(at: usize) -> State {
    let json = std::fs::read_to_string(FIXTURE).expect("fixture");
    let wall: Vec<Tile> = array_after(&json, "wall").iter().map(|s| t(s)).collect();
    let history = &json[json.find("\"history\"").unwrap()..json.find("\"human_seats\"").unwrap()];
    let codes: Vec<u16> = history
        .split("\"code\": ")
        .skip(1)
        .map(|s| s[..s.find(',').unwrap()].parse().unwrap())
        .collect();
    let mut s = State::from_wall(RuleSet::default(), wall);
    for &c in &codes[..at] {
        let a = Action::from_code(c).unwrap();
        assert!(
            s.legal_actions_vec().contains(&a),
            "history replays legally"
        );
        s.apply(a);
    }
    s
}

fn equity(d: &RaceDecision, a: Action) -> f64 {
    d.candidates
        .iter()
        .find(|c| c.action == a)
        .unwrap_or_else(|| panic!("{a} is a candidate"))
        .equity
}

fn print(d: &RaceDecision) {
    for c in &d.candidates {
        eprintln!(
            "{:>14} E {:8.2}  gain {:7.2}  sd {:6.2}  feeds {:6.2}  now {:6.2}  P(me) {:.3}  P(o) {:.3?}  s {} pay {:.0}/{:.0} modes {:?}",
            c.action.to_string(),
            c.equity,
            c.gain,
            c.loss_self_draw,
            c.loss_my_feeds + c.loss_other_feeds,
            c.deal_in_now.iter().sum::<f64>(),
            c.p_win[c.me],
            c.p_win,
            c.faan_shanten,
            c.pay_me,
            c.pay_sd_me,
            c.modes,
        );
    }
    eprintln!("-> {}", d.action);
}

/// Legal moves and finite terms on 1,000 random hands (race in seats 0 and 2, value
/// in 1 and 3; 30 hands in a debug build).
#[test]
fn plays_legal_moves_on_random_hands() {
    let race = RaceBot::default();
    let value = Bot::by_name("value").unwrap();
    let hands = if cfg!(debug_assertions) { 30 } else { 1000 };
    let mut rng = SmallRng::seed_from_u64(6);
    let mut decisions = 0usize;
    for _ in 0..hands {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        while let Some(p) = s.current_player() {
            let legal = s.legal_actions_vec();
            let a = if p % 2 == 0 {
                let d = race.decide(&s);
                assert!(legal.contains(&d.action), "{} not legal", d.action);
                for c in &d.candidates {
                    assert!(c.equity.is_finite(), "{c:?}");
                    let total: f64 = c.p_win.iter().sum::<f64>() + c.p_draw;
                    assert!((total - 1.0).abs() < 1e-6, "{c:?}");
                }
                decisions += 1;
                d.action
            } else {
                value.choose(&s, &legal, &mut rng)
            };
            s.apply(a);
        }
    }
    eprintln!("{decisions} race decisions over {hands} hands");
}

/// §9: Brian's pung of 8p (history index 85) is worth at least the pass.
#[test]
fn brians_pung_hand_values_pung_at_least_pass() {
    let s = brian_at(85);
    assert_eq!(s.current_player(), Some(0));
    assert!(s.in_claim());
    let d = RaceBot::default().decide(&s);
    print(&d);
    let (pung, pass) = (equity(&d, Action::Pung), equity(&d, Action::Pass));
    eprintln!("E(pung) {pung:.1}  E(pass) {pass:.1}  (doc §9.3: +161.7, +38.4)");
    assert!(pung >= pass, "pung {pung:.2} < pass {pass:.2}");
    assert_eq!(d.action, Action::Pung);
    // After the pung the bot throws the dead 5p, keeping the live 4p wait.
    let c = d
        .candidates
        .iter()
        .find(|c| c.action == Action::Pung)
        .unwrap();
    assert_eq!(c.discard, Some(t("5p")));
    // The same with the trained belief (§6's v0 inputs), when present.
    let model = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/belief.mjnn");
    if let Ok(Agent::Bot(Bot::Race(b))) = Agent::parse(&format!("race:belief={model}")) {
        let d = b.decide(&s);
        print(&d);
        let (pung, pass) = (equity(&d, Action::Pung), equity(&d, Action::Pass));
        eprintln!("belief.mjnn: E(pung) {pung:.1}  E(pass) {pass:.1}");
        assert!(pung >= pass, "pung {pung:.2} < pass {pass:.2}");
    }
}

/// `docs/ev-bot.md` §5 example 2 with the same belief stub: dealing a cheap 2m into
/// seat 1 ends the hand before seat 2's 96-point self-draw.
#[test]
fn still_deals_into_the_small_hand_to_stop_the_big_self_draw() {
    let (state, beliefs) = position_2(false);
    let bot = RaceBot::new(RaceConfig::v0(), Arc::new(Stub(beliefs)));
    let d = bot.decide(&state);
    print(&d);
    assert_eq!(d.action, Action::Discard(t("2m")));
}

/// Seat 0, ready on 7m for a characters half flush with a Red pung after throwing
/// the useless 5s. Seat 1 (A) waits cheaply on 2s, which seat 0 does not hold;
/// seat 2 (B) waits on 5s for an 8-faan hand. `b_fold` replaces §5.2's heuristic
/// for B (what Phase 4's mode head will supply).
fn two_threats(b_fold: Option<f64>) -> RaceDecision {
    let state = build(Setup {
        hands: [Some("123m 456m 789m RRR 7m 5s"), None, None, None],
        melds: Default::default(),
        claimed: vec![],
        last_drawn: "5s",
        wall: 40,
        liable: [None; 4],
        keep: "7m 5s 2s",
    });
    let beliefs = Beliefs {
        seats: [
            SeatBelief::quiet(),
            seat(0.5, 0.9, &["2s"], 0.05, 3),
            seat(0.9, 0.9, &["5s"], 0.2, 8),
            seat(0.0, 0.0, &[], 0.0, 3),
        ],
    };
    let mut bot = RaceBot::new(RaceConfig::v0(), Arc::new(Stub(beliefs)));
    bot.fold_override[2] = b_fold;
    let d = bot.decide(&state);
    print(&d);
    d
}

#[test]
fn folds_against_the_pushing_threat_and_pushes_again_when_it_folds() {
    let five_s = Action::Discard(t("5s"));
    // B pushes: keep the 5s and fold against B; A's wait is not in my hand, so A
    // gets no fold.
    let d = two_threats(Some(0.0));
    assert_ne!(d.action, five_s);
    let chosen = d.candidates.iter().find(|c| c.action == d.action).unwrap();
    assert_eq!(chosen.modes[2], Some(Mode::Fold), "{:?}", chosen.modes);
    assert_ne!(chosen.modes[1], Some(Mode::Fold), "{:?}", chosen.modes);
    // B has folded: push again and throw the 5s.
    let d = two_threats(Some(1.0));
    assert_eq!(d.action, five_s);
    let chosen = d.candidates.iter().find(|c| c.action == d.action).unwrap();
    assert!(
        chosen
            .modes
            .iter()
            .all(|m| m.is_none_or(|m| m == Mode::Push)),
        "{:?}",
        chosen.modes
    );
}

/// AC5: under 5 ms per decision on one thread in a release build, on Brian's
/// position, on the deal-in position and over the decisions of real hands.
#[test]
fn a_decision_takes_under_5_ms_on_one_thread() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let bot = RaceBot::default();
    let budget = if cfg!(debug_assertions) { 200.0 } else { 5.0 };
    let calls = if cfg!(debug_assertions) { 20 } else { 500 };
    let (state, beliefs) = position_2(false);
    let stubbed = RaceBot::new(RaceConfig::v0(), Arc::new(Stub(beliefs)));
    let brian = brian_at(85);
    let mut rng = SmallRng::seed_from_u64(1);
    for (name, b, s) in [
        ("brian 85", &bot, &brian),
        ("ev example 2", &stubbed, &state),
    ] {
        let legal = s.legal_actions_vec();
        let ms = one.install(|| {
            let start = Instant::now();
            for _ in 0..calls {
                std::hint::black_box(b.choose(s, &legal, &mut rng));
            }
            start.elapsed().as_secs_f64() * 1000.0 / calls as f64
        });
        eprintln!("race, {name}: {ms:.3} ms per decision over {calls} calls");
        assert!(ms < budget, "{name}: {ms:.3} ms");
    }
    // Real hands: the heuristic belief, and the trained model when present (each
    // decision on a new state, so the belief's cache does not help).
    let mut bots = vec![("heuristic", bot.clone())];
    let model = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/belief.mjnn");
    if std::path::Path::new(model).exists() {
        match Agent::parse(&format!("race:belief={model}")) {
            Ok(Agent::Bot(Bot::Race(b))) => bots.push(("belief.mjnn", b)),
            other => panic!("{other:?}"),
        }
    }
    let value = Bot::by_name("value").unwrap();
    for (name, bot) in &bots {
        let (mut n, mut secs, mut worst) = (0usize, 0.0f64, 0.0f64);
        one.install(|| {
            for _ in 0..if cfg!(debug_assertions) { 2 } else { 20 } {
                let mut s = State::new_deal(RuleSet::default(), &mut rng);
                while let Some(p) = s.current_player() {
                    let legal = s.legal_actions_vec();
                    let a = if p == 0 && legal.len() > 1 {
                        let start = Instant::now();
                        let a = bot.choose(&s, &legal, &mut rng);
                        let e = start.elapsed().as_secs_f64();
                        secs += e;
                        worst = worst.max(e);
                        n += 1;
                        a
                    } else {
                        value.choose(&s, &legal, &mut rng)
                    };
                    s.apply(a);
                }
            }
        });
        let ms = 1000.0 * secs / n.max(1) as f64;
        eprintln!(
            "race ({name}), real hands: {ms:.3} ms per decision over {n} (worst {:.3} ms)",
            worst * 1000.0
        );
        assert!(ms < budget, "{name}: {ms:.3} ms per decision");
    }
}

/// `race[:...]` parses (the CLI's `mahjong hk tournament` goes through
/// `Agent::parse`) and plays a duplicate tournament.
#[test]
fn the_race_spec_parses_and_plays_a_tournament() {
    assert!(matches!(Agent::parse("race"), Ok(Agent::Bot(Bot::Race(_)))));
    match Agent::parse("race:horizon=6,tie_margin=1,threat=0.3") {
        Ok(Agent::Bot(Bot::Race(b))) => {
            assert_eq!(b.config.horizon, 6);
            assert_eq!(b.config.tie_margin, 1.0);
            assert_eq!(b.config.threat, 0.3);
        }
        other => panic!("{other:?}"),
    }
    assert!(Agent::parse("race:horizon=0").is_err());
    assert!(Agent::parse("race:nope=1").is_err());
    assert!(Agent::parse("race:belief=models/does-not-exist.mjnn").is_err());
    let model = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/belief.mjnn");
    if std::path::Path::new(model).exists() {
        assert!(Agent::parse(&format!("race:belief={model}")).is_ok());
    }
    let agents = ["race", "value", "value", "value"].map(|s| Agent::parse(s).unwrap());
    let config = TournamentConfig {
        deals: 2,
        seatings: Seatings::Rotations,
        rules: RuleSet::default(),
        seed: 3,
        bootstrap: 100,
    };
    let res = tournament(&agents, &config);
    assert_eq!(res.hands.len(), 8);
}
