//! Plan 2026-11-16 Task 24: the paired false-alarm log (measurement code, never
//! used in play).
//!
//! A defence bot's game is played beside the all-`stage` game on the same wall
//! (the set-up of Task 23's `examples/defence_ceiling.rs`: one generator per
//! game, seeded by deal and seat). Until the bot's first decision that differs
//! from plain `stage`'s, the two games are the same; at each of the bot's
//! decisions plain `stage`'s choice is worked out on a copy of the generator,
//! and while the two agree the game goes on with plain's copy, so the stream
//! stays in step by construction.
//!
//! At the first change the log reads the TRUE hands (as Task 23's oracle does,
//! for the label only): a *true alarm* when `stage`'s pick would have been won
//! on at once ([`crate::oracle::deal_in`], every possible winner declaring), a
//! *false alarm* otherwise. A first change that is not a discard replacing a
//! discard (a claim, a kong) is kept apart.

use crate::oracle::{deal_in, DealIn};
use crate::stage::StageBot;
use mahjong_analysis::{faan_shanten, HandView};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, HandResult, RuleSet, State};
use mahjong_tiles::Tile;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// How a hand ended for one seat.
pub const OUTCOMES: [&str; 5] = [
    "won",
    "dealt in",
    "other self-drew",
    "other's deal-in",
    "draw",
];

/// The seat's outcome index in [`OUTCOMES`].
pub fn outcome(r: &HandResult, seat: usize) -> usize {
    if r.wins.iter().any(|w| w.seat == seat) {
        0
    } else if r.wins.iter().any(|w| !w.self_drawn && w.from == Some(seat)) {
        1
    } else if r.wins.iter().any(|w| w.self_drawn) {
        2
    } else if !r.wins.is_empty() {
        3
    } else {
        4
    }
}

/// The bot's first decision that differs from plain `stage`'s.
#[derive(Clone, Debug)]
pub struct Change {
    /// The position before the decision.
    pub state: State,
    /// Index of the decision among the seat's decisions (0-based).
    pub decision: u32,
    /// Plain `stage`'s action and the bot's.
    pub plain: Action,
    pub bot: Action,
    /// What `stage`'s pick would cost at once (true hands); default when the
    /// change is not a discard replacing a discard.
    pub truth: DealIn,
}

impl Change {
    /// Both actions are discards (the case the alarm label is about).
    pub fn is_discard(&self) -> bool {
        matches!(
            (self.plain, self.bot),
            (Action::Discard(_), Action::Discard(_))
        )
    }

    /// `stage`'s pick, when the change is a discard.
    pub fn stage_pick(&self) -> Option<Tile> {
        match self.plain {
            Action::Discard(t) if self.is_discard() => Some(t),
            _ => None,
        }
    }

    /// The tile the bot threw instead, when the change is a discard.
    pub fn thrown(&self) -> Option<Tile> {
        match self.bot {
            Action::Discard(t) if self.is_discard() => Some(t),
            _ => None,
        }
    }

    /// `stage`'s pick would have been won on at once.
    pub fn true_alarm(&self) -> bool {
        self.is_discard() && self.truth.paid > 0
    }
}

/// One game, seen from `seat`.
#[derive(Clone, Debug)]
pub struct Played {
    pub result: HandResult,
    /// The seat's points and outcome ([`OUTCOMES`]).
    pub points: i32,
    pub outcome: usize,
    /// The seat's decisions, and its last decision (index, action).
    pub decisions: u32,
    /// The seat's discard decisions.
    pub discards: u32,
    /// Bot games: the seat's decisions where the bot's action differs from
    /// plain `stage`'s in the same position (the first change and later ones),
    /// and those that are discards replacing discards.
    pub changed: u32,
    pub changed_discards: u32,
    pub last: Option<(u32, Action)>,
    /// The bot's first change (bot games only).
    pub change: Option<Change>,
}

/// The generator of the game of `deal` with the tested seat `seat`.
pub fn game_rng(deal: u64, seat: usize) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(deal.wrapping_mul(4).wrapping_add(seat as u64))
}

/// Plays `deal` with `bot` in `seat` and plain `stage` elsewhere (`bot` =
/// `None`: four plain `stage`).
pub fn play(rules: &RuleSet, bot: Option<&StageBot>, deal: u64, seat: usize) -> Played {
    let stage = StageBot::default();
    let mut state = State::new_deal(*rules, &mut ChaCha8Rng::seed_from_u64(deal));
    let mut rng = game_rng(deal, seat);
    let mut decisions = 0u32;
    let mut discards = 0u32;
    let (mut changed, mut changed_discards) = (0u32, 0u32);
    let mut last = None;
    let mut change: Option<Change> = None;
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let p = state.current_player().expect("not over");
        let action = match bot {
            Some(b) if p == seat => {
                if change.is_none() {
                    let mut plain_rng = rng.clone();
                    let plain = stage.choose(&state, &legal, &mut plain_rng);
                    let mine = b.choose(&state, &legal, &mut rng);
                    if mine == plain {
                        rng = plain_rng;
                    } else {
                        let truth = match (plain, mine) {
                            (Action::Discard(t), Action::Discard(_)) => deal_in(&state, seat, t),
                            _ => DealIn::default(),
                        };
                        change = Some(Change {
                            state: state.clone(),
                            decision: decisions,
                            plain,
                            bot: mine,
                            truth,
                        });
                    }
                    if mine != plain {
                        changed += 1;
                        changed_discards +=
                            matches!((plain, mine), (Action::Discard(_), Action::Discard(_)))
                                as u32;
                    }
                    mine
                } else {
                    let plain = stage.choose(&state, &legal, &mut rng.clone());
                    let mine = b.choose(&state, &legal, &mut rng);
                    if mine != plain {
                        changed += 1;
                        changed_discards +=
                            matches!((plain, mine), (Action::Discard(_), Action::Discard(_)))
                                as u32;
                    }
                    mine
                }
            }
            _ => stage.choose(&state, &legal, &mut rng),
        };
        if p == seat {
            last = Some((decisions, action));
            decisions += 1;
            discards += matches!(action, Action::Discard(_)) as u32;
        }
        state.apply(action);
    }
    let result = state.result().expect("over").clone();
    Played {
        points: result.deltas[seat],
        outcome: outcome(&result, seat),
        result,
        decisions,
        discards,
        changed,
        changed_discards,
        last,
        change,
    }
}

/// Seat `o`'s true faan-aware distance to a ready hand that meets the minimum
/// (0 = ready), read from its concealed hand (log only).
pub fn true_distance(state: &State, o: usize) -> i8 {
    let concealed = *state.hand(o);
    let unseen = state.unseen(o);
    let view = HandView {
        concealed: &concealed,
        melds: &state.melds(o)[..],
        seat: o,
        bonus: state.bonus(o),
        unseen: &unseen,
    };
    faan_shanten(&view, state.rules()).shanten
}

/// The engine's own settlement of the plain game at the bot's first change:
/// whether plain `stage` dealt in with the discard it made at that decision
/// (the plain game ended on it with a win off `seat`). TEST-ONLY check of
/// [`Change::true_alarm`].
pub fn plain_dealt_in_at(plain: &Played, change: &Change) -> bool {
    plain.outcome == 1
        && plain.last == Some((change.decision, change.plain))
        && matches!(change.plain, Action::Discard(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::BeliefModel;
    use std::path::Path;
    use std::sync::Arc;

    /// On 10 deals a bot (each seat), the false-alarm label agrees with the
    /// plain game's own settlement of the claim window: `stage`'s pick was won
    /// on at once exactly when the label says true alarm; hands with no change
    /// end as the plain game.
    #[test]
    fn false_alarm_label_agrees_with_the_engine() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief-order.mjnn");
        if !p.exists() {
            eprintln!("models/belief-order.mjnn missing: skipped");
            return;
        }
        let model = Arc::new(BeliefModel::load(p.to_str().expect("utf-8")).expect("model"));
        let reading = Some(model);
        let bots = [
            StageBot::default().with_safe(1),
            StageBot::default().with_safe(5),
            StageBot::default().with_defence(true),
            StageBot::default().with_giveup(Some(1.0)),
        ]
        .map(|b| b.with_reading(reading.clone()));
        let rules = RuleSet::default();
        let mut total_trues = 0;
        for (k, bot) in bots.iter().enumerate() {
            let (mut changes, mut trues) = (0, 0);
            for deal in 0..10u64 {
                for seat in 0..4 {
                    let plain = play(&rules, None, 7000 + deal, seat);
                    let mine = play(&rules, Some(bot), 7000 + deal, seat);
                    let Some(c) = &mine.change else {
                        assert_eq!(mine.points, plain.points, "bot {k} deal {deal}: drift");
                        assert_eq!(mine.decisions, plain.decisions);
                        continue;
                    };
                    changes += 1;
                    if c.is_discard() {
                        assert_eq!(
                            c.true_alarm(),
                            plain_dealt_in_at(&plain, c),
                            "bot {k} deal {deal} seat {seat}: label against the engine"
                        );
                        trues += c.true_alarm() as usize;
                    }
                }
            }
            eprintln!("bot {k}: {changes} first changes, {trues} true alarms");
            total_trues += trues;
        }
        assert!(total_trues > 0, "no true alarm to check in 160 games");
    }

    #[test]
    fn a_bot_with_the_switch_off_never_changes() {
        let rules = RuleSet::default();
        for deal in 0..3u64 {
            let p = play(&rules, Some(&StageBot::default()), deal, 0);
            assert!(p.change.is_none());
            assert_eq!(p.points, play(&rules, None, deal, 0).points);
        }
    }
}
