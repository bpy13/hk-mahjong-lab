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

use crate::belief::BeliefModel;
use crate::oracle::{deal_in, DealIn};
use crate::stage::{read_dealin, safe_threshold, StageBot};
use crate::view::SeatView;
use mahjong_analysis::{faan_shanten, HandView};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, HandResult, RuleSet, State};
use mahjong_tiles::Tile;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};

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
    /// Index of the change in the whole deal's action list ([`Played::actions`]):
    /// the number of actions applied before it.
    pub index: u32,
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
    /// Every action of the deal (all seats, claim passes included), from the
    /// first draw to the end, in order.
    pub actions: Vec<Action>,
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
    /// Plan Task 26 (only from [`play_with`] given a reading): for each action of
    /// [`Played::actions`] the estimates of the seat that took it when it was a
    /// discard ([`estimate_json`]), else null; empty without a reading.
    pub estimates: Vec<Value>,
}

/// The generator of the game of `deal` with the tested seat `seat`.
pub fn game_rng(deal: u64, seat: usize) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(deal.wrapping_mul(4).wrapping_add(seat as u64))
}

/// The shuffled wall of `deal`, in draw order from the front (the wall
/// [`play`] deals from, so a replay can start from it).
pub fn wall_of(rules: &RuleSet, deal: u64) -> Vec<Tile> {
    let mut wall = State::all_tiles(rules);
    wall.shuffle(&mut ChaCha8Rng::seed_from_u64(deal));
    wall
}

/// Replays `actions` from `wall` through the engine and returns the ending
/// (None if the actions stop before the hand ends). Panics on an illegal action.
pub fn replay(rules: &RuleSet, wall: Vec<Tile>, actions: &[Action]) -> Option<HandResult> {
    let mut state = State::from_wall(*rules, wall);
    for (i, &a) in actions.iter().enumerate() {
        assert!(!state.is_terminal(), "action {i} after the end");
        assert!(
            state.legal_actions_vec().contains(&a),
            "action {i} ({a}) is not legal"
        );
        state.apply(a);
    }
    state.result().cloned()
}

/// The bot of the other seats and of the plain game: plain `stage` unless
/// [`set_base`] named another (plan Task 30 writes the worked example with the
/// early-game switches on, `examples-json ... base=stage-early`).
static BASE: std::sync::OnceLock<StageBot> = std::sync::OnceLock::new();

/// Plan Task 30: plays every seat but the tested one, and the plain game, with
/// `base` instead of plain `stage`, for the rest of the process (set once; a
/// second call is ignored). Measurement tools only.
pub fn set_base(base: StageBot) {
    let _ = BASE.set(base);
}

fn base() -> StageBot {
    BASE.get().copied().unwrap_or_default()
}

/// Plays `deal` with `bot` in `seat` and plain `stage` elsewhere (`bot` =
/// `None`: four plain `stage`; see [`set_base`]).
pub fn play(rules: &RuleSet, bot: Option<&StageBot>, deal: u64, seat: usize) -> Played {
    play_with(rules, bot, deal, seat, None)
}

/// [`play`], and with a reading `model` it also records every discard
/// decision's estimates ([`Played::estimates`]). Recording reads copies of the
/// position and of the generator and changes no action.
pub fn play_with(
    rules: &RuleSet,
    bot: Option<&StageBot>,
    deal: u64,
    seat: usize,
    model: Option<&BeliefModel>,
) -> Played {
    let stage = base();
    let mut state = State::new_deal(*rules, &mut ChaCha8Rng::seed_from_u64(deal));
    let mut rng = game_rng(deal, seat);
    let mut decisions = 0u32;
    let mut discards = 0u32;
    let (mut changed, mut changed_discards) = (0u32, 0u32);
    let mut last = None;
    let mut change: Option<Change> = None;
    let mut actions: Vec<Action> = Vec::new();
    let mut estimates: Vec<Value> = Vec::new();
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let p = state.current_player().expect("not over");
        let mut est = match model {
            Some(m) if !state.in_claim() && !state.in_rob() => {
                let thr = bot
                    .filter(|_| p == seat)
                    .and_then(|b| safe_threshold(b.safe));
                estimate_json(&state, &legal, &stage, &rng, m, thr)
            }
            _ => None,
        };
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
                            index: actions.len() as u32,
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
        if model.is_some() {
            estimates.push(match (est.as_mut(), action) {
                (Some(e), Action::Discard(t)) => {
                    mark_thrown(e, t);
                    std::mem::take(e)
                }
                _ => Value::Null,
            });
        }
        actions.push(action);
        state.apply(action);
    }
    let result = state.result().expect("over").clone();
    Played {
        actions,
        points: result.deltas[seat],
        outcome: outcome(&result, seat),
        result,
        decisions,
        discards,
        changed,
        changed_discards,
        last,
        change,
        estimates,
    }
}

fn mark_thrown(e: &mut Value, t: Tile) {
    let name = t.to_string();
    for x in e["tiles"].as_array_mut().expect("tiles") {
        x["thrown"] = json!(x["tile"] == json!(name));
    }
}

/// Plan Task 26: what the acting seat's `stage` computes from its own view at a
/// discard decision (`None` when the action list has no discard to make). `rng`
/// is the decision's generator (only cloned). `threshold` is the defence
/// threshold in force for this seat, if any. The reading `model` is run on the
/// seat's view for every seat, plain ones too, for the record only (plain
/// `stage` does not use it to choose). Format: `docs/study-results/false-alarm-hand-format.md`.
pub fn estimate_json(
    state: &State,
    legal: &[Action],
    stage: &StageBot,
    rng: &ChaCha8Rng,
    model: &BeliefModel,
    threshold: Option<f64>,
) -> Option<Value> {
    let tiles: Vec<Tile> = legal
        .iter()
        .filter_map(|a| match a {
            Action::Discard(t) => Some(*t),
            _ => None,
        })
        .collect();
    if tiles.is_empty()
        || legal
            .iter()
            .any(|a| matches!(a, Action::Win | Action::SelfDrawnWin))
    {
        return None;
    }
    let seat = state.current_player()?;
    let view = SeatView::of_current(state);
    let (pick, note) = match stage.decide(state, legal, &mut rng.clone()) {
        (Action::Discard(t), Some(n)) => (Some(t), Some(n)),
        (_, n) => (None, n),
    };
    let figures = stage.discard_figures(&view, &tiles);
    let own_distance = figures[0].distance_after;
    let b = model.run_for(state, seat);
    let rows: Vec<Value> = figures
        .iter()
        .map(|f| {
            let i = f.tile.index() as usize;
            let combined = read_dealin(&b, i);
            let dealin = [0, 1, 2].map(|j| b.opponents[j].dealin[i]);
            json!({
                "tile": f.tile.to_string(),
                "stage_value": {
                    "distance_after": f.distance_after,
                    "acceptance": f.acceptance,
                    "target_faan": f.target_faan,
                    "connectedness": f.connectedness,
                    "stage3_value": f.stage3_value,
                },
                "dealin": dealin,
                "dealin_combined": combined,
                "above_threshold": threshold.map(|t| combined > t),
                "stage_pick": pick == Some(f.tile),
                "thrown": false,
            })
        })
        .collect();
    let opponents: Vec<Value> = (0..3)
        .map(|j| {
            let o = (seat + 1 + j) % 4;
            let ob = &b.opponents[j];
            json!({
                "seat": o,
                "role": ROLES[j + 1],
                "p_ready": ob.state.map_or(ob.tenpai, |s| s[2]),
                "p_tenpai": ob.tenpai,
                "p_state": ob.state,
                "true_distance": true_distance(state, o),
            })
        })
        .collect();
    Some(json!({
        "seat": seat,
        "own_distance": own_distance,
        "stage": note.map(|n| n.stage),
        "line": note.and_then(|n| n.line).map(|l| format!("{l:?}")),
        "threshold": threshold,
        "defence_active": threshold.is_some() && own_distance >= 1,
        "tiles": rows,
        "opponents": opponents,
    }))
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

/// What the reading says at a first change (log only; see
/// `docs/study-results/false-alarm-hand-format.md`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Reading {
    /// The bot's distance from ready (14 tiles, best discard).
    pub own_dist: i8,
    /// Distance after the tile thrown minus after `stage`'s pick (13 tiles).
    pub cost: i8,
    /// The reading's deal-in figure for `stage`'s pick on the three opponents
    /// (next, opposite, previous, counted from the bot's seat).
    pub dealin: [f64; 3],
    /// Wall third (0..3) at the change.
    pub wall_third: usize,
}

const WINDS: [&str; 4] = ["E", "S", "W", "N"];
const ROLES: [&str; 4] = ["bot", "next", "opposite", "previous"];

/// An action as the app's engine names it (`action.*`, with the tile where it
/// has one, and the engine's action code).
pub fn action_json(a: Action) -> Value {
    let t = |t: Tile| t.to_string();
    let mut v = match a {
        Action::Discard(x) => json!({"type": "action.discard", "tile": t(x)}),
        Action::SelfDrawnWin => json!({"type": "action.self_drawn_win"}),
        Action::ConcealedKong(x) => json!({"type": "action.concealed_kong", "tile": t(x)}),
        Action::AddedKong(x) => json!({"type": "action.added_kong", "tile": t(x)}),
        Action::Pass => json!({"type": "action.pass"}),
        Action::Win => json!({"type": "action.win"}),
        Action::Pung => json!({"type": "action.pung"}),
        Action::ClaimKong => json!({"type": "action.kong"}),
        Action::Chow(x) => {
            let run: Vec<String> = (0..3).map(|k| t(Tile::from_index(x.index() + k))).collect();
            json!({"type": "action.chow", "tile": t(x), "tiles": run})
        }
    };
    v["code"] = json!(a.code());
    v
}

/// A hand's ending: the wins and the points of every seat.
pub fn result_json(r: &HandResult) -> Value {
    let wins: Vec<Value> = r
        .wins
        .iter()
        .map(|w| {
            json!({
                "winner": w.seat,
                "from": w.from,
                "self_drawn": w.self_drawn,
                "faan": w.score.faan,
                "win_tile": w.win_tile.map(|t| t.to_string()),
                "paid_by": w.paid_by,
                "liable": w.liable,
                "robbed": w.robbed,
            })
        })
        .collect();
    json!({"draw": r.wins.is_empty(), "wins": wins, "points": r.deltas})
}

/// The file of one example hand: everything needed to show the deal and step
/// through it. `plain` and `mine` are the plain game and the bot's game of the
/// same deal ([`play`]); `mine` must have a change. `ending_kind` is the
/// example report's name for how the hand ended beside the plain game.
#[allow(clippy::too_many_arguments)]
pub fn example_json(
    rules: &RuleSet,
    deal: u64,
    seat: usize,
    bot: &str,
    ending_kind: &str,
    plain: &Played,
    mine: &Played,
    rd: &Reading,
) -> Value {
    let c = mine.change.as_ref().expect("a change");
    let s = &c.state;
    let river = s.river_entries();
    let seats: Vec<Value> = (0..4)
        .map(|r| {
            let o = (seat + r) % 4;
            let melds: Vec<String> = s.melds(o).iter().map(|m| m.to_string()).collect();
            let discards: Vec<Value> = river
                .iter()
                .filter(|e| e.seat == o)
                .map(|e| json!({"tile": e.tile.to_string(), "claimed": e.claimed}))
                .collect();
            json!({
                "seat": o,
                "role": ROLES[r],
                "wind": WINDS[o],
                "hand": s.hand(o).to_string(),
                "open_sets": melds,
                "discards": discards,
                "true_distance": if r == 0 { rd.own_dist } else { true_distance(s, o) },
                "read_dealin_on_pick": if r == 0 { Value::Null } else { json!(rd.dealin[r - 1]) },
            })
        })
        .collect();
    json!({
        "format": 2,
        "deal": deal,
        "bot": bot,
        "seat": seat,
        "dealer": 0,
        "seat_winds": WINDS,
        "prevailing_wind": WINDS[rules.prevailing_wind],
        "wall": wall_of(rules, deal).iter().map(|t| t.to_string()).collect::<Vec<_>>(),
        "plain_actions": mine_actions(&plain.actions),
        "bot_actions": mine_actions(&mine.actions),
        "plain_estimates": estimates_or_null(plain),
        "bot_estimates": estimates_or_null(mine),
        "first_change_index": c.index,
        "label": if c.true_alarm() { "true alarm" } else { "false alarm" },
        "ending_kind": ending_kind,
        "change": {
            "stage_pick": c.stage_pick().map(|t| t.to_string()),
            "thrown_instead": c.thrown().map(|t| t.to_string()),
            "progress_steps_cost": rd.cost,
            "own_distance": rd.own_dist,
            "wall_remaining": s.wall_remaining(),
            "wall_third": rd.wall_third + 1,
            "paid_at_once_if_picked": c.truth.paid,
            "seats": seats,
        },
        "plain_ending": result_json(&plain.result),
        "bot_ending": result_json(&mine.result),
        "bot_points": mine.points,
        "plain_points": plain.points,
    })
}

fn estimates_or_null(p: &Played) -> Vec<Value> {
    if p.estimates.is_empty() {
        vec![Value::Null; p.actions.len()]
    } else {
        p.estimates.clone()
    }
}

fn mine_actions(a: &[Action]) -> Vec<Value> {
    a.iter().map(|&x| action_json(x)).collect()
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

    /// Replays a written file (through its text, as the app reads it) from its
    /// wall and both action lists, and checks the endings it states.
    fn check_file(text: &str, rules: &RuleSet) {
        let v: Value = serde_json::from_str(text).expect("json");
        let wall: Vec<Tile> = v["wall"]
            .as_array()
            .expect("wall")
            .iter()
            .map(|t| t.as_str().expect("tile").parse().expect("tile"))
            .collect();
        let actions = |key: &str| -> Vec<Action> {
            v[key]
                .as_array()
                .expect("actions")
                .iter()
                .map(|a| Action::from_code(a["code"].as_u64().expect("code") as u16).expect("code"))
                .collect()
        };
        let (plain, bot) = (actions("plain_actions"), actions("bot_actions"));
        let at = v["first_change_index"].as_u64().expect("index") as usize;
        for (list, key) in [(&plain, "plain_estimates"), (&bot, "bot_estimates")] {
            let est = v[key].as_array().expect("estimates");
            assert_eq!(est.len(), list.len(), "{key}: one entry per action");
            for (a, e) in list.iter().zip(est) {
                // An entry for every discard (and only those), with exactly one tile thrown.
                match a {
                    Action::Discard(t) => {
                        let thrown: Vec<&Value> = e["tiles"]
                            .as_array()
                            .expect("an entry for a discard")
                            .iter()
                            .filter(|x| x["thrown"] == json!(true))
                            .collect();
                        assert_eq!(thrown.len(), 1);
                        assert_eq!(thrown[0]["tile"], json!(t.to_string()));
                    }
                    _ => assert!(e.is_null(), "{key}: entry for a non-discard"),
                }
            }
        }
        let e = &v["bot_estimates"][at];
        let c = &v["change"];
        assert_eq!(e["seat"], v["seat"]);
        let pick = e["tiles"]
            .as_array()
            .expect("tiles")
            .iter()
            .find(|x| x["stage_pick"] == json!(true))
            .expect("a stage pick");
        assert_eq!(pick["tile"], c["stage_pick"], "stage pick at the change");
        let thrown = e["tiles"]
            .as_array()
            .expect("tiles")
            .iter()
            .find(|x| x["thrown"] == json!(true))
            .expect("a thrown tile");
        assert_eq!(thrown["tile"], c["thrown_instead"], "thrown at the change");
        for r in 0..3 {
            assert_eq!(
                pick["dealin"][r],
                c["seats"][r + 1]["read_dealin_on_pick"],
                "the reading at the change"
            );
        }
        assert_eq!(plain[..at], bot[..at], "the games agree up to the change");
        assert_ne!(plain[at], bot[at], "and differ at it");
        for (list, key, pts) in [
            (&plain, "plain_ending", "plain_points"),
            (&bot, "bot_ending", "bot_points"),
        ] {
            let r = replay(rules, wall.clone(), list).expect("the actions end the hand");
            assert_eq!(result_json(&r), v[key], "{key}");
            let seat = v["seat"].as_u64().expect("seat") as usize;
            assert_eq!(json!(r.deltas[seat]), v[pts], "{pts}");
        }
    }

    /// For 10 deals a bot (each seat), every written example file replays
    /// through the engine to the endings it states.
    #[test]
    fn example_files_replay_to_their_endings() {
        let rules = RuleSet::default();
        let bots = [
            ("safe1", StageBot::default().with_safe(1)),
            ("safe5", StageBot::default().with_safe(5)),
            ("fold", StageBot::default().with_giveup(Some(1.0))),
        ];
        let mut files = 0;
        for (name, bot) in &bots {
            // Without the reading these bots never change; use the model if present.
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../models/belief-order.mjnn");
            if !p.exists() {
                eprintln!("models/belief-order.mjnn missing: skipped");
                return;
            }
            let model = Arc::new(
                crate::belief::BeliefModel::load(p.to_str().expect("utf-8")).expect("model"),
            );
            let bot = bot.with_reading(Some(model.clone()));
            for deal in 0..10u64 {
                for seat in 0..4 {
                    let mine = play_with(&rules, Some(&bot), 7000 + deal, seat, Some(&model));
                    if mine.change.is_none() {
                        continue;
                    }
                    let bare = play(&rules, Some(&bot), 7000 + deal, seat);
                    assert_eq!(bare.actions, mine.actions, "recording changes no action");
                    assert_eq!(bare.points, mine.points, "recording changes no ending");
                    let plain = play_with(&rules, None, 7000 + deal, seat, Some(&model));
                    let v = example_json(
                        &rules,
                        7000 + deal,
                        seat,
                        name,
                        "test",
                        &plain,
                        &mine,
                        &reading_of(&model, &mine),
                    );
                    check_file(&serde_json::to_string_pretty(&v).expect("text"), &rules);
                    files += 1;
                }
            }
        }
        assert!(files > 0, "no changed hand to write");
        eprintln!("{files} example files replayed");
    }

    /// The change's reading, as the example binary builds it.
    fn reading_of(model: &BeliefModel, mine: &Played) -> Reading {
        let c = mine.change.as_ref().expect("a change");
        let b = model.run_for(&c.state, c.state.current_player().expect("seat"));
        let i = c.stage_pick().expect("discard").index() as usize;
        Reading {
            dealin: [0, 1, 2].map(|j| b.opponents[j].dealin[i]),
            ..Reading::default()
        }
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
