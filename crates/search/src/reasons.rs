//! Reason keys for the two bot designs (plan Task 20), in the engine's
//! vocabulary: a plain enum of keys and a struct of named numbers. No display
//! text anywhere (D42): the app stream maps these keys onto its own reason
//! vocabulary (`crates/app-api/src/reasons.rs`) and its dictionaries.
//!
//! * The EV bot (B15, `docs/ev-bot.md` §7): the five `reason.ev.*` keys, built
//!   by [`crate::ev`] and read with [`crate::EvTerms::reasons`].
//! * The network path: [`plan_target`] names the CNN target head's plan
//!   (`reason.plan.target`), and [`expected_loss`] names the extra points a
//!   discard risks against the safest alternative (`reason.risk.expected_loss`).
//!
//! Every key matches `^reason\.[a-z_.]+$`. Everything here is plain arithmetic:
//! no threads, no randomness, so it runs the same as WebAssembly.

use crate::danger::Danger;
use crate::labels::NUM_TARGETS;
use mahjong_tiles::Tile;

/// One reason key. [`ReasonKey::key`] gives the string the app maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ReasonKey {
    /// `reason.ev.offence`: args `p_win` (`cont × κ × P_win`), `points`
    /// (`Pay(me) × Shares(me)`).
    EvOffence,
    /// `reason.ev.deal_in_now`: args `seat`, `p` (`P_o(t(a))`), `points`
    /// (`Pay(o)`), `shares` (`L_disc(o)`).
    EvDealInNow,
    /// `reason.ev.stops_big_hand`: args `seat`, `p_self_draw` (`sd_o`), `points`
    /// (`Pay_sd(o)`).
    EvStopsBigHand,
    /// `reason.ev.liability`: arg `seat`.
    EvLiability,
    /// `reason.ev.margin`: arg `points` (`EV(a) − EV` of the best alternative
    /// outside the tie; 0 when every alternative ties, negative for a move that
    /// is not the best).
    EvMargin,
    /// `reason.plan.target`: the network's target-head plan. Args `class`
    /// (0–44), `kind` (0 mixed, 1 half flush, 2 full flush, 3 all honours,
    /// 4 thirteen orphans), `suit` (0 characters, 1 dots, 2 bamboo; only for
    /// the two flush kinds), `structure` (0 any, 1 all pungs, 2 all chows) and
    /// `p` (the head's probability of that class).
    PlanTarget,
    /// `reason.risk.expected_loss`: args `points` (the discard's expected loss
    /// minus the safest alternative's) and `p` (the discard's deal-in chance).
    RiskExpectedLoss,
}

impl ReasonKey {
    /// Every key, in a fixed order.
    pub const ALL: [ReasonKey; 7] = [
        ReasonKey::EvOffence,
        ReasonKey::EvDealInNow,
        ReasonKey::EvStopsBigHand,
        ReasonKey::EvLiability,
        ReasonKey::EvMargin,
        ReasonKey::PlanTarget,
        ReasonKey::RiskExpectedLoss,
    ];

    /// The key string, e.g. `reason.ev.offence`.
    pub fn key(&self) -> &'static str {
        match self {
            ReasonKey::EvOffence => "reason.ev.offence",
            ReasonKey::EvDealInNow => "reason.ev.deal_in_now",
            ReasonKey::EvStopsBigHand => "reason.ev.stops_big_hand",
            ReasonKey::EvLiability => "reason.ev.liability",
            ReasonKey::EvMargin => "reason.ev.margin",
            ReasonKey::PlanTarget => "reason.plan.target",
            ReasonKey::RiskExpectedLoss => "reason.risk.expected_loss",
        }
    }

    /// The key for a string, if it is one of [`ReasonKey::ALL`].
    pub fn from_key(s: &str) -> Option<ReasonKey> {
        ReasonKey::ALL.into_iter().find(|k| k.key() == s)
    }
}

/// The named numbers of a reason; `None` means the reason does not carry that
/// argument. Seats are absolute seat numbers (0–3).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReasonArgs {
    pub seat: Option<u8>,
    pub p: Option<f64>,
    pub p_win: Option<f64>,
    pub p_self_draw: Option<f64>,
    pub points: Option<f64>,
    pub shares: Option<u8>,
    pub class: Option<u8>,
    pub kind: Option<u8>,
    pub suit: Option<u8>,
    pub structure: Option<u8>,
}

impl ReasonArgs {
    /// The present arguments as `(name, value)`, in field order. Whole-number
    /// arguments (`seat`, `shares`, `class`, `kind`, `suit`, `structure`) come
    /// out as whole `f64`s.
    pub fn named(&self) -> Vec<(&'static str, f64)> {
        let whole = |v: Option<u8>| v.map(f64::from);
        [
            ("seat", whole(self.seat)),
            ("p", self.p),
            ("p_win", self.p_win),
            ("p_self_draw", self.p_self_draw),
            ("points", self.points),
            ("shares", whole(self.shares)),
            ("class", whole(self.class)),
            ("kind", whole(self.kind)),
            ("suit", whole(self.suit)),
            ("structure", whole(self.structure)),
        ]
        .into_iter()
        .filter_map(|(name, v)| v.map(|v| (name, v)))
        .collect()
    }

    /// One argument by name.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.named()
            .into_iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v)
    }
}

/// A reason key with its arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reason {
    pub key: ReasonKey,
    pub args: ReasonArgs,
}

impl Reason {
    pub fn new(key: ReasonKey, args: ReasonArgs) -> Reason {
        Reason { key, args }
    }

    /// The key string.
    pub fn key_str(&self) -> &'static str {
        self.key.key()
    }
}

// ----- The network path. -----

/// Plan kinds of the target classes (`labels::target_class`).
pub const PLAN_KINDS: u8 = 5;
/// Kinds 1 (half flush) and 2 (full flush) carry a suit; the others use suit 0.
fn kind_has_suit(kind: u8) -> bool {
    kind == 1 || kind == 2
}

/// `reason.plan.target` for target class `class` (`kind * 9 + suit * 3 +
/// structure`, the layout of `labels::target_class` and the CNN's 45-wide
/// target head) with the head's probability `p`. `None` for a class outside
/// 0–44 (255 is "no target") or one of the 18 unused slots (a suit-less plan
/// with suit 1 or 2).
pub fn plan_target_class(class: u8, p: f64) -> Option<Reason> {
    if class as usize >= NUM_TARGETS {
        return None;
    }
    let (kind, suit, structure) = (class / 9, (class % 9) / 3, class % 3);
    debug_assert!(kind < PLAN_KINDS);
    if !kind_has_suit(kind) && suit != 0 {
        return None;
    }
    Some(Reason::new(
        ReasonKey::PlanTarget,
        ReasonArgs {
            class: Some(class),
            kind: Some(kind),
            suit: kind_has_suit(kind).then_some(suit),
            structure: Some(structure),
            p: Some(p),
            ..ReasonArgs::default()
        },
    ))
}

/// `reason.plan.target` from the CNN's target logits (45 values, the slice
/// `POLICY_LEN..POLICY_LEN + TARGET_LEN` of [`crate::cnn`]'s output): the
/// class with the largest logit among the used ones, with its softmax
/// probability over the used classes. `None` when the slice is not 45 long or
/// holds no finite used logit.
pub fn plan_target(logits: &[f32]) -> Option<Reason> {
    if logits.len() != NUM_TARGETS {
        return None;
    }
    let used = |c: usize| {
        let kind = (c / 9) as u8;
        (kind_has_suit(kind) || (c % 9) / 3 == 0) && logits[c].is_finite()
    };
    let best = (0..NUM_TARGETS)
        .filter(|&c| used(c))
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then(b.cmp(&a)))?;
    let top = logits[best] as f64;
    let z: f64 = (0..NUM_TARGETS)
        .filter(|&c| used(c))
        .map(|c| (logits[c] as f64 - top).exp())
        .sum();
    plan_target_class(best as u8, 1.0 / z)
}

/// Below this many points of extra expected loss, [`expected_loss`] names no
/// risk (the `ev` bot's tie margin).
pub const RISK_MIN_POINTS: f64 = 0.5;

/// `reason.risk.expected_loss` for discarding `chosen` when `alternatives`
/// (the other legal discards) were available: `points` is
/// `Danger::expected_loss(chosen) − min_t Danger::expected_loss(t)` over the
/// alternatives, `p` the deal-in chance of `chosen`. `None` when that delta is
/// below [`RISK_MIN_POINTS`] (the move risks nothing a safer discard would
/// avoid). Use a [`Danger`] built by [`Danger::from_belief`] to get payments
/// in points; one built by [`Danger::estimate`] counts every win as 8 points.
pub fn expected_loss(danger: &Danger, chosen: Tile, alternatives: &[Tile]) -> Option<Reason> {
    let mine = danger.expected_loss(chosen);
    let safest = alternatives
        .iter()
        .filter(|&&t| t != chosen)
        .map(|&t| danger.expected_loss(t))
        .fold(mine, f64::min);
    expected_loss_delta(mine - safest, danger.tile(chosen))
}

/// `reason.risk.expected_loss` from a delta already computed (for a network
/// that predicts expected loss directly): `points` = `delta`, `p` = the
/// deal-in chance. `None` when `delta < RISK_MIN_POINTS` or is not finite.
pub fn expected_loss_delta(delta: f64, p: f64) -> Option<Reason> {
    (delta.is_finite() && delta >= RISK_MIN_POINTS).then(|| {
        Reason::new(
            ReasonKey::RiskExpectedLoss,
            ReasonArgs {
                points: Some(delta),
                p: Some(p),
                ..ReasonArgs::default()
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip() {
        for k in ReasonKey::ALL {
            assert_eq!(ReasonKey::from_key(k.key()), Some(k));
        }
        assert_eq!(ReasonKey::from_key("reason.nope"), None);
    }

    #[test]
    fn named_skips_missing_args() {
        let a = ReasonArgs {
            seat: Some(2),
            points: Some(96.0),
            ..ReasonArgs::default()
        };
        assert_eq!(a.named(), vec![("seat", 2.0), ("points", 96.0)]);
        assert_eq!(a.get("points"), Some(96.0));
        assert_eq!(a.get("p"), None);
    }
}
