//! Reason keys for the two designs (plan Task 20): the `ev` bot's five
//! `reason.ev.*` keys on the three constructed positions of `docs/ev-bot.md`
//! §6, and the network path's `reason.plan.target` and
//! `reason.risk.expected_loss`. Keys only, never text (D42).

mod common;

use common::{bot, bot_without_floor, position_1, position_2, t};
use mahjong_hk::{Action, State};
use mahjong_search::cnn::{TARGET_LEN, UNUSED_TARGET};
use mahjong_search::reasons::{
    expected_loss, expected_loss_delta, plan_target, plan_target_class, RISK_MIN_POINTS,
};
use mahjong_search::{
    Beliefs, Danger, EvBot, EvDecision, EvReason, EvTerms, Reason, ReasonKey, SeatView,
};

/// `^reason\.[a-z_.]+$`, without a regex crate.
fn has_app_shape(key: &str) -> bool {
    key.strip_prefix("reason.").is_some_and(|rest| {
        !rest.is_empty()
            && rest
                .chars()
                .all(|c| c.is_ascii_lowercase() || c == '_' || c == '.')
    })
}

fn chosen(d: &EvDecision) -> &EvTerms {
    &d.candidates
        .iter()
        .find(|(a, _)| *a == d.action)
        .expect("the chosen action is a candidate")
        .1
}

fn keys(r: &[Reason]) -> Vec<ReasonKey> {
    r.iter().map(|x| x.key).collect()
}

fn find(r: &[Reason], k: ReasonKey) -> Option<&Reason> {
    r.iter().find(|x| x.key == k)
}

fn close(a: Option<f64>, b: f64) -> bool {
    a.is_some_and(|a| (a - b).abs() < 1e-9)
}

/// The decision's reasons are the chosen candidate's `EvTerms::reasons()`, and
/// every candidate's list ends with exactly one `reason.ev.margin`.
fn check_consistent(d: &EvDecision) {
    let mine: Vec<EvReason> = chosen(d).reasons().iter().map(EvReason::from).collect();
    assert_eq!(mine, d.reasons);
    for (a, x) in &d.candidates {
        let r = x.reasons();
        assert_eq!(r.last().map(|x| x.key), Some(ReasonKey::EvMargin), "{a}");
        assert_eq!(r.iter().filter(|x| x.key == ReasonKey::EvMargin).count(), 1);
        for x in &r {
            assert!(x.key_str().starts_with("reason.ev."), "{}", x.key_str());
        }
    }
}

fn decide(make: fn(Beliefs) -> EvBot, (state, beliefs): (State, Beliefs)) -> EvDecision {
    let d = make(beliefs).decide(&state);
    check_consistent(&d);
    d
}

#[test]
fn every_key_string_has_the_app_shape() {
    for k in ReasonKey::ALL {
        assert!(has_app_shape(k.key()), "{}", k.key());
    }
    let mut seen = std::collections::HashSet::new();
    for k in ReasonKey::ALL {
        assert!(seen.insert(k.key()), "duplicate {}", k.key());
    }
    // Everything the EV bot emits, for every candidate of every position.
    for make in [bot, bot_without_floor] {
        for pos in [position_1(), position_2(false), position_2(true)] {
            let d = decide(make, pos);
            for (_, x) in &d.candidates {
                for r in x.reasons() {
                    assert!(has_app_shape(r.key_str()), "{}", r.key_str());
                }
            }
        }
    }
    assert!(!has_app_shape("reason."));
    assert!(!has_app_shape("reason.EV.margin"));
    assert!(!has_app_shape("reason.ev-margin"));
}

/// §6 row 1: `reason.ev.offence` first, `reason.ev.margin` present, no
/// `reason.ev.deal_in_now` (every `P_o ≤ 0.005 < warn_p`); with or without the
/// `Pay(me)` floor.
#[test]
fn position_1_gives_offence_and_margin() {
    for make in [bot, bot_without_floor] {
        let d = decide(make, position_1());
        let r = chosen(&d).reasons();
        assert_eq!(r[0].key, ReasonKey::EvOffence, "{r:?}");
        let off = &r[0].args;
        assert!(off.p_win.is_some_and(|p| p > 0.0 && p <= 1.0));
        assert!(off.points.is_some_and(|p| p > 0.0));
        assert!(find(&r, ReasonKey::EvMargin).is_some());
        assert!(find(&r, ReasonKey::EvDealInNow).is_none());
        assert!(find(&r, ReasonKey::EvStopsBigHand).is_none());
        assert!(find(&r, ReasonKey::EvLiability).is_none());
    }
}

/// §6 row 2 (without the floor, as `ev_bot.rs` asserts the answer): the bot
/// deals into B's `2m` and says why: it stops A's big self-draw and deals in
/// to B now, with the stubbed numbers as args.
#[test]
fn position_2_gives_stops_big_hand_and_deal_in_now() {
    let d = decide(bot_without_floor, position_2(false));
    assert_eq!(d.action, Action::Discard(t("2m")));
    let r = chosen(&d).reasons();
    assert_eq!(
        keys(&r),
        vec![
            ReasonKey::EvStopsBigHand,
            ReasonKey::EvDealInNow,
            ReasonKey::EvMargin
        ]
    );
    let stops = &find(&r, ReasonKey::EvStopsBigHand).unwrap().args;
    assert_eq!(stops.seat, Some(2));
    assert!(close(stops.p_self_draw, 0.30));
    assert!(close(stops.points, 96.0), "Pay_sd(A) = points(9)");
    let deal = &find(&r, ReasonKey::EvDealInNow).unwrap().args;
    assert_eq!((deal.seat, deal.shares), (Some(1), Some(1)));
    assert!(close(deal.p, 0.95));
    assert!(close(deal.points, 16.0), "Pay(B) = points(4)");
    let margin = find(&r, ReasonKey::EvMargin).unwrap().args.points.unwrap();
    assert!(margin >= 8.0, "margin {margin}");

    // A safe candidate the bot did not choose: no deal-in, no stop, and a
    // negative margin (its loss to 2m).
    let (_, safe) = d
        .candidates
        .iter()
        .find(|(a, _)| *a == Action::Discard(t("Wh")))
        .unwrap();
    let r = safe.reasons();
    assert!(find(&r, ReasonKey::EvDealInNow).is_none());
    assert!(find(&r, ReasonKey::EvStopsBigHand).is_none());
    let m = find(&r, ReasonKey::EvMargin).unwrap().args.points.unwrap();
    assert!(m <= -8.0, "{m}");
}

/// §6 row 3: liable for B, the bot keeps the safe tile; `reason.ev.liability
/// {seat: 1}`, no `reason.ev.stops_big_hand`; with or without the floor.
#[test]
fn position_3_gives_liability_not_stops_big_hand() {
    for make in [bot, bot_without_floor] {
        let d = decide(make, position_2(true));
        assert_ne!(d.action, Action::Discard(t("2m")));
        let r = chosen(&d).reasons();
        assert_eq!(
            find(&r, ReasonKey::EvLiability).map(|x| x.args.seat),
            Some(Some(1))
        );
        assert!(find(&r, ReasonKey::EvStopsBigHand).is_none());
        assert!(find(&r, ReasonKey::EvDealInNow).is_none());
        let margin = find(&r, ReasonKey::EvMargin).unwrap().args.points.unwrap();
        assert!(margin > 0.0, "margin {margin}");
        // Dealing in to B would pay 3 shares.
        let (_, deal_in) = d
            .candidates
            .iter()
            .find(|(a, _)| *a == Action::Discard(t("2m")))
            .unwrap();
        let deal = find(&deal_in.reasons(), ReasonKey::EvDealInNow).map(|x| x.args);
        assert_eq!(deal.and_then(|a| a.shares), Some(3));
    }
}

/// The reasons are plain arithmetic: inside a one-thread pool (the
/// WebAssembly case) they come out the same.
#[test]
fn reasons_match_on_one_thread() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for pos in [position_1(), position_2(false), position_2(true)] {
        let (state, beliefs) = pos;
        let b = bot_without_floor(beliefs);
        let here = b.decide(&state);
        let there = one.install(|| b.decide(&state));
        assert_eq!(here.reasons, there.reasons);
        assert_eq!(chosen(&here).reasons(), chosen(&there).reasons());
    }
}

// ----- The network path. -----

#[test]
fn plan_target_reads_the_class_layout() {
    // kind * 9 + suit * 3 + structure (labels::target_class).
    let mixed = plan_target_class(0, 0.5).unwrap();
    assert_eq!(mixed.key, ReasonKey::PlanTarget);
    assert_eq!(mixed.key_str(), "reason.plan.target");
    let a = mixed.args;
    assert_eq!(
        (a.class, a.kind, a.suit, a.structure),
        (Some(0), Some(0), None, Some(0))
    );
    assert_eq!(a.p, Some(0.5));

    // Half flush in bamboo, all pungs.
    let a = plan_target_class(9 + 2 * 3 + 1, 1.0).unwrap().args;
    assert_eq!((a.kind, a.suit, a.structure), (Some(1), Some(2), Some(1)));
    // Full flush in dots, all chows.
    let a = plan_target_class(18 + 3 + 2, 1.0).unwrap().args;
    assert_eq!((a.kind, a.suit, a.structure), (Some(2), Some(1), Some(2)));
    // Thirteen orphans.
    let a = plan_target_class(36, 1.0).unwrap().args;
    assert_eq!((a.kind, a.suit, a.structure), (Some(4), None, Some(0)));

    // Unused slots (a suit-less plan with suit 1 or 2), "no target" and out of range.
    for c in [3, 8, 27 + 3, 44, 45, 255] {
        assert!(plan_target_class(c, 1.0).is_none(), "class {c}");
    }
    let used = (0..=255u8)
        .filter(|&c| plan_target_class(c, 1.0).is_some())
        .count();
    assert_eq!(used, 27, "45 slots less the 18 unused");
}

#[test]
fn plan_target_takes_the_head_argmax_with_its_probability() {
    // The CNN fills the unused slots with UNUSED_TARGET (-1e4).
    let mut logits = vec![0.0f32; TARGET_LEN];
    for (c, l) in logits.iter_mut().enumerate() {
        let (kind, suit) = (c / 9, (c % 9) / 3);
        if !(kind == 1 || kind == 2) && suit != 0 {
            *l = UNUSED_TARGET;
        }
    }
    // Full flush in dots, all pungs.
    logits[18 + 3 + 1] = 3.0;
    let r = plan_target(&logits).unwrap();
    let a = r.args;
    assert_eq!(
        (a.class, a.kind, a.suit, a.structure),
        (Some(22), Some(2), Some(1), Some(1))
    );
    let want = 3f64.exp() / (3f64.exp() + 26.0);
    assert!((a.p.unwrap() - want).abs() < 1e-6, "{:?} vs {want}", a.p);

    // A large value on an unused slot is not a plan.
    logits[4] = 50.0;
    assert_eq!(plan_target(&logits).unwrap().args.class, Some(22));
    // Wrong length.
    assert!(plan_target(&logits[..44]).is_none());
}

#[test]
fn expected_loss_names_the_extra_risk_in_points() {
    // Position 2: B (seat 1) waits on 2m for 4 faan (16 points), tenpai 0.95.
    let (state, beliefs) = position_2(false);
    let danger = Danger::from_beliefs(&state, 0, &beliefs);
    let hand: Vec<_> = SeatView::of(&state, 0).concealed.kinds().collect();
    let r = expected_loss(&danger, t("2m"), &hand).expect("2m is risky");
    assert_eq!(r.key_str(), "reason.risk.expected_loss");
    assert!(close(r.args.points, 0.95 * 16.0), "{:?}", r.args.points);
    assert!(close(r.args.p, 0.95));
    // A safe tile risks nothing beyond the safest alternative.
    assert!(expected_loss(&danger, t("Wh"), &hand).is_none());

    // Liable for B: every win by B costs 3 shares.
    let (state, beliefs) = position_2(true);
    let danger = Danger::from_beliefs(&state, 0, &beliefs);
    let r = expected_loss(&danger, t("2m"), &hand).unwrap();
    assert!(
        close(r.args.points, 0.95 * 16.0 * 3.0),
        "{:?}",
        r.args.points
    );

    // From a delta computed elsewhere.
    assert!(expected_loss_delta(RISK_MIN_POINTS - 0.01, 0.1).is_none());
    assert!(expected_loss_delta(f64::NAN, 0.1).is_none());
    let r = expected_loss_delta(12.5, 0.2).unwrap();
    assert_eq!(r.args.named(), vec![("p", 0.2), ("points", 12.5)]);
}
