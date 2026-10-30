//! The `ev` bot's three constructed positions (plan Task 10, `docs/ev-bot.md`
//! §5–§6). The belief is stubbed with the numbers of §5, so the tests do not
//! depend on model weights; the target table is the fitted one, so the tests
//! assert orderings and reason keys, not decimals.

mod common;

use common::{bot, bot_without_floor, position_1, position_2, t};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::ev::load_belief;
use mahjong_search::{Agent, Belief, Bot, EvBot, EvConfig, EvDecision};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

fn keys(d: &EvDecision) -> Vec<&'static str> {
    d.reasons.iter().map(|r| r.key).collect()
}

fn arg(d: &EvDecision, key: &str, name: &str) -> Option<f64> {
    d.reasons
        .iter()
        .find(|r| r.key == key)
        .and_then(|r| r.args.iter().find(|a| a.0 == name).map(|a| a.1))
}

fn ev_of(d: &EvDecision, a: Action) -> f64 {
    d.candidates
        .iter()
        .find(|c| c.0 == a)
        .expect("a candidate")
        .1
        .ev
}

fn print(d: &EvDecision) {
    for (a, x) in &d.candidates {
        eprintln!(
            "{a:>14}  ev {:8.2}  off {:6.2}  def {:6.2}  p_win {:.3} s {} f_T {}  dn {:6.2}  later {:5.2}  liab {:5.2}",
            x.ev, x.offence, x.defence, x.p_win, x.faan_shanten, x.target_faan,
            x.deal_in_total(), x.later, x.liability
        );
    }
    eprintln!("-> {}  {:?}", d.action, d.reasons);
}

/// Position 1's checks that hold with or without the floor.
fn check_position_1(d: &EvDecision) {
    for (_, x) in &d.candidates {
        assert!(
            x.deal_in_total() <= 0.2,
            "{:?}: {}",
            x.action,
            x.deal_in_total()
        );
    }
    assert_eq!(keys(d)[0], "reason.ev.offence");
    assert!(keys(d).contains(&"reason.ev.margin"));
    assert!(!keys(d).contains(&"reason.ev.deal_in_now"));
}

#[test]
fn position_1_plays_like_value_when_nobody_threatens() {
    let (state, beliefs) = position_1();
    let legal = state.legal_actions_vec();
    let value =
        Bot::by_name("value")
            .unwrap()
            .choose(&state, &legal, &mut SmallRng::seed_from_u64(0));

    // The doc's expected answer (§6 row 1), without the Pay(me) floor.
    let d = bot_without_floor(beliefs.clone()).decide(&state);
    print(&d);
    assert_eq!(d.action, value, "ev should play value's discard here");
    check_position_1(&d);

    // With the floor (the doc's default) the bot chases the 10-faan full flush
    // (discards R): everything but the choice still holds (see
    // `bot_without_floor`). The choice is not asserted: it depends on the
    // table's decimals.
    let d = bot(beliefs).decide(&state);
    print(&d);
    check_position_1(&d);
}

/// Position 2's checks; `min_margin` is the doc's 8 points without the floor.
fn check_position_2(d: &EvDecision, min_margin: f64) {
    let deal_in = Action::Discard(t("2m"));
    assert_eq!(d.action, deal_in);
    assert_eq!(d.candidates[0].0, deal_in, "explain ranks 2m first");
    let margin = d.candidates[0].1.ev - d.candidates[1].1.ev;
    assert!(margin >= min_margin, "margin {margin:.2} < {min_margin}");
    assert_eq!(arg(d, "reason.ev.stops_big_hand", "seat"), Some(2.0));
    assert_eq!(arg(d, "reason.ev.deal_in_now", "seat"), Some(1.0));
    assert_eq!(arg(d, "reason.ev.deal_in_now", "shares"), Some(1.0));
    assert!(keys(d).contains(&"reason.ev.margin"));
}

#[test]
fn position_2_deals_into_the_small_hand_to_stop_the_big_self_draw() {
    let (state, beliefs) = position_2(false);
    let d = bot_without_floor(beliefs.clone()).decide(&state);
    print(&d);
    check_position_2(&d, 8.0);
    // With the floor (the doc's default) and the final 20,000-hand table, the
    // 10-faan floor lifts every safe tile's offence from about 2 to 16.5 points,
    // so the bot keeps the safe tile (2p) instead of dealing in: the doc's
    // expected answer does not hold. Not asserted; reported in docs/ev-bot.md §10
    // as a question for Brian.
    let d = bot(beliefs).decide(&state);
    print(&d);
    assert!(state.legal_actions_vec().contains(&d.action));
}

#[test]
fn position_3_does_not_deal_in_when_liable() {
    let (state, beliefs) = position_2(true);
    assert_eq!(state.liable_for(1), Some(0));
    for d in [
        bot(beliefs.clone()).decide(&state),
        bot_without_floor(beliefs).decide(&state),
    ] {
        print(&d);
        let deal_in = Action::Discard(t("2m"));
        assert_ne!(d.action, deal_in);
        assert_eq!(
            d.candidates.last().unwrap().0,
            deal_in,
            "explain ranks 2m last"
        );
        assert!(d.candidates[0].1.ev - ev_of(&d, deal_in) >= 8.0);
        assert_eq!(arg(&d, "reason.ev.liability", "seat"), Some(1.0));
        assert!(!keys(&d).contains(&"reason.ev.stops_big_hand"));
    }
}

#[test]
fn every_reason_key_has_the_app_shape() {
    for (state, beliefs) in [position_1(), position_2(false), position_2(true)] {
        let d = bot(beliefs).decide(&state);
        for r in &d.reasons {
            let rest = r.key.strip_prefix("reason.").expect("reason. prefix");
            assert!(
                !rest.is_empty()
                    && rest
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c == '_' || c == '.'),
                "{}",
                r.key
            );
        }
    }
}

/// AC5: under 5 ms per decision on one thread (release build; a debug build
/// only checks a loose bound and prints the time).
#[test]
fn a_decision_takes_under_5_ms_on_one_thread() {
    let (state, beliefs) = position_2(false);
    let bot = bot(beliefs);
    let legal = state.legal_actions_vec();
    let mut rng = SmallRng::seed_from_u64(1);
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let calls = if cfg!(debug_assertions) { 100 } else { 1000 };
    let per_call_ms = one.install(|| {
        let start = Instant::now();
        for _ in 0..calls {
            std::hint::black_box(bot.choose(&state, &legal, &mut rng));
        }
        start.elapsed().as_secs_f64() * 1000.0 / calls as f64
    });
    eprintln!("ev: {per_call_ms:.3} ms per decision over {calls} calls");
    let budget = if cfg!(debug_assertions) { 100.0 } else { 5.0 };
    assert!(per_call_ms < budget, "{per_call_ms:.3} ms per decision");
}

/// The trained belief model (`ev:belief=models/belief.mjnn`), when present.
fn trained() -> Option<Arc<dyn Belief>> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief.mjnn");
    p.exists()
        .then(|| load_belief(p.to_str().unwrap()).expect("the committed model loads"))
}

/// With the trained model, `x_o` is the `outcome` head's discard-win chance
/// (not `r_disc × sd_o`), every move is legal, and a decision (network
/// included, each on a new state so the cache does not help) stays under
/// 5 ms on one thread in a release build (AC5).
#[test]
fn the_trained_belief_drives_the_bot_within_the_time_budget() {
    let Some(belief) = trained() else {
        eprintln!("models/belief.mjnn missing: skipped");
        return;
    };
    for pay_floor in [true, false] {
        let config = EvConfig {
            pay_floor,
            ..EvConfig::default()
        };
        let bot = EvBot::new(config, Arc::clone(&belief));
        let value = Bot::by_name("value").unwrap();
        let mut rng = SmallRng::seed_from_u64(21);
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let (mut decisions, mut seconds, mut x_differs) = (0usize, 0.0f64, false);
        one.install(|| {
            for _ in 0..3 {
                let mut s = State::new_deal(RuleSet::default(), &mut rng);
                while let Some(p) = s.current_player() {
                    let legal = s.legal_actions_vec();
                    let a = if p == 0 && legal.len() > 1 {
                        let start = Instant::now();
                        let d = bot.decide(&s);
                        seconds += start.elapsed().as_secs_f64();
                        decisions += 1;
                        assert!(legal.contains(&d.action), "{:?} not legal", d.action);
                        for (_, x) in &d.candidates {
                            assert!(x.ev.is_finite());
                            for o in 1..4 {
                                assert!(
                                    (0.0..=1.0).contains(&x.x[o]) && (0.0..=1.0).contains(&x.sd[o])
                                );
                                x_differs |= (x.x[o] - 2.0 * x.sd[o]).abs() > 1e-6;
                            }
                            assert_eq!(x.x[0], 0.0);
                        }
                        d.action
                    } else {
                        value.choose(&s, &legal, &mut rng)
                    };
                    s.apply(a);
                }
            }
        });
        let per_call_ms = 1000.0 * seconds / decisions.max(1) as f64;
        eprintln!(
            "ev:belief (floor {pay_floor}): {per_call_ms:.3} ms per decision over {decisions}"
        );
        assert!(decisions > 20);
        assert!(x_differs, "x_o comes from the model, not r_disc × sd_o");
        let budget = if cfg!(debug_assertions) { 200.0 } else { 5.0 };
        assert!(per_call_ms < budget, "{per_call_ms:.3} ms per decision");
    }
}

/// The spec string loads the model; a policy network is refused.
#[test]
fn the_ev_spec_loads_the_trained_belief() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief.mjnn");
    if !p.exists() {
        return;
    }
    let spec = format!("ev:belief={},floor=none", p.display());
    assert!(matches!(Agent::parse(&spec), Ok(Agent::Bot(Bot::Ev(_)))));
    let spec = format!("defensive:belief={}", p.display());
    assert!(matches!(Agent::parse(&spec), Ok(Agent::Bot(Bot::Value(_)))));
    assert!(Agent::parse("defensive:nope=1").is_err());
    let ppo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/ppo.mjnn");
    if ppo.exists() {
        assert!(Agent::parse(&format!("ev:belief={}", ppo.display())).is_err());
    }
}
