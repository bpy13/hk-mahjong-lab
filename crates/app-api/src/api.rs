//! The calls of `docs/engine-interface.md` §3, over the Hong Kong mahjong engine.
//!
//! [`Api`] holds the running hands by `game_id`. Every call takes and returns
//! JSON strings; see `README.md` for the shapes. Errors come back as
//! `{"api": 0, "error": {"code": "error.*", "detail": "..."}}`.

use crate::reasons::{pair_se, position_reasons, solver_reasons, Study, SIGNIFICANCE};
use crate::shapes::{
    action_to_json, bonus_to_json, event_to_json, faan_meta, hand_to_json, is_declaration,
    legal_action, meld_to_json, result_to_json, round2, round3, rules_from_json, rules_to_json,
    tile_from_id, tile_id, tile_keys_meta, tiles_to_json, variants_meta, ActionJson, ApiError,
    ApiResult, GameConfig, PositionJson, Reason, TurnJson, ACTION_KEYS, API_VERSION,
};
use crate::solver::{
    analysis, grade_for, search_budget, EXPERT_CANDIDATES, EXPERT_SAMPLES, EXPERT_Z,
    HINT_CANDIDATES, HINT_SAMPLES, REVIEW_CANDIDATES, REVIEW_SAMPLES,
};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::{ActionStats, Analysis, Bot};
use mahjong_tiles::Tile;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use web_time::Instant;

/// Bot levels, weakest first (`GameMeta.bot_levels`).
pub const BOT_LEVELS: [&str; 5] = ["random", "easy", "medium", "hard", "expert"];

/// The engine bot behind each rule-based level (see `README.md`, "Bot levels").
fn bot_name(level: &str) -> Option<&'static str> {
    match level {
        "random" => Some("random"),
        "easy" => Some("efficiency"),
        "medium" => Some("defensive"),
        "hard" => Some("value"),
        _ => None,
    }
}

/// How many hint moves are returned.
const HINT_MOVES: usize = 3;

/// One hand: the wall it was dealt from, its current state, the actions so far
/// and the settings the app gave it.
#[derive(Clone, Debug)]
struct GameEntry {
    rules: RuleSet,
    /// The full tile set in deal order (`State::from_wall`).
    wall: Vec<Tile>,
    state: State,
    seed: u64,
    human_seats: Vec<usize>,
    history: Vec<(usize, Action)>,
}

impl GameEntry {
    fn from_wall(
        rules: RuleSet,
        wall: Vec<Tile>,
        seed: u64,
        human_seats: Vec<usize>,
    ) -> ApiResult<GameEntry> {
        let mut sorted = wall.clone();
        sorted.sort();
        if sorted != State::all_tiles(&rules) {
            return Err(ApiError::new(
                "error.bad_position",
                format!(
                    "the wall must hold exactly the {} tiles in play under these rules",
                    State::all_tiles(&rules).len()
                ),
            ));
        }
        Ok(GameEntry {
            rules,
            state: State::from_wall(rules, wall.clone()),
            wall,
            seed,
            human_seats,
            history: Vec::new(),
        })
    }

    /// Applies a legal action and records it.
    fn apply(&mut self, action: Action) {
        let player = self.state.current_player().expect("checked by the caller");
        self.state.apply(action);
        self.history.push((player, action));
    }

    /// Replays `history` from the deal (each turn is validated).
    fn replay(&mut self, history: &[TurnJson]) -> ApiResult<()> {
        for (i, turn) in history.iter().enumerate() {
            let Some(player) = self.state.current_player() else {
                return Err(ApiError::new(
                    "error.bad_position",
                    format!("history continues after the hand ended (turn {i})"),
                ));
            };
            if turn.player != player {
                return Err(ApiError::new(
                    "error.bad_position",
                    format!(
                        "turn {i}: player {} acted but player {player} was to move",
                        turn.player
                    ),
                ));
            }
            let action = legal_action(&self.state, &turn.action).map_err(|e| {
                ApiError::new("error.bad_position", format!("turn {i}: {}", e.detail))
            })?;
            self.apply(action);
        }
        Ok(())
    }

    /// The states before each decision of `seat`, with the history index of each.
    fn decisions_of(&self, seat: usize) -> Vec<(usize, State, Action)> {
        let mut s = State::from_wall(self.rules, self.wall.clone());
        let mut out = Vec::new();
        for (i, &(player, action)) in self.history.iter().enumerate() {
            if player == seat {
                out.push((i, s.clone(), action));
            }
            s.apply(action);
        }
        out
    }

    /// How many trailing history entries are declarations in a claim or rob
    /// window that is still open. They stay hidden from the other seats (§4.1.1).
    fn open_declarations(&self) -> usize {
        if self.state.claim_tile().is_none() {
            return 0;
        }
        self.history
            .iter()
            .rev()
            .take_while(|&&(_, a)| is_declaration(a))
            .count()
    }

    /// The history as `viewer` may see it: another seat's declaration in the
    /// open window is `{"player": p, "action": null}`.
    fn history_json(&self, viewer: Option<usize>) -> Vec<Value> {
        let hidden_from = self.history.len() - self.open_declarations();
        self.history
            .iter()
            .enumerate()
            .map(|(i, &(player, action))| {
                let visible = viewer.is_none_or(|v| v == player) || i < hidden_from;
                json!({
                    "player": player,
                    "action": visible.then(|| action_to_json(action)),
                })
            })
            .collect()
    }

    fn history_turns(&self) -> Vec<TurnJson> {
        self.history
            .iter()
            .map(|&(player, action)| TurnJson {
                player,
                action: action_to_json(action),
            })
            .collect()
    }

    /// A seeded generator for the decision at this point of the hand: the same
    /// position always draws the same numbers, so a restored hand continues
    /// exactly as the original would have.
    fn rng(&self, salt: u64) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(self.decision_seed(salt))
    }

    fn decision_seed(&self, salt: u64) -> u64 {
        self.seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(self.history.len() as u64)
            .wrapping_mul(0xD1B5_4A32_D192_ED03)
            .wrapping_add(salt)
    }
}

/// The engine as the app sees it: hands by id, JSON in and out.
#[derive(Default, Debug)]
pub struct Api {
    games: HashMap<String, GameEntry>,
    next_id: u64,
}

/// Serializes a successful result with `"api": 0`, or the error.
fn respond(result: ApiResult<Value>) -> String {
    let mut v = match result {
        Ok(v) => v,
        Err(e) => e.to_json(),
    };
    if let Value::Object(m) = &mut v {
        m.insert("api".into(), json!(API_VERSION));
    }
    v.to_string()
}

fn parse_json<T: serde::de::DeserializeOwned>(what: &str, text: &str) -> ApiResult<T> {
    serde_json::from_str(text).map_err(|e| ApiError::new("error.bad_json", format!("{what}: {e}")))
}

fn check_seat(seat: usize) -> ApiResult<()> {
    if seat < 4 {
        Ok(())
    } else {
        Err(ApiError::new("error.bad_seat", format!("no seat {seat}")))
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    start.elapsed().as_millis() as u64
}

fn to_value<T: Serialize>(t: &T) -> Value {
    serde_json::to_value(t).expect("our shapes always serialize")
}

fn must_be_to_move(state: &State, seat: usize) -> ApiResult<()> {
    match state.current_player() {
        None => Err(ApiError::new("error.game_over", "the hand is already over")),
        Some(p) if p == seat => Ok(()),
        Some(p) => Err(ApiError::new(
            "error.not_your_turn",
            format!("seat {seat} asked but seat {p} is to move"),
        )),
    }
}

impl Api {
    pub fn new() -> Api {
        Api::default()
    }

    // -- the calls ------------------------------------------------------------

    /// `GameMeta` (§4), plus the key lists the app's dictionaries need.
    pub fn meta(&self) -> String {
        respond(Ok(json!({
            "game": "mahjong",
            "seats": [4, 4],
            "variants": variants_meta(),
            "bot_levels": BOT_LEVELS,
            "has_hint": true,
            "has_review": true,
            "has_grade": true,
            "tiles": tile_keys_meta(),
            "faan": faan_meta(),
            "actions": ACTION_KEYS,
        })))
    }

    /// Deals a new hand from `GameConfig` JSON; returns `{game_id}`.
    pub fn new_game(&mut self, config: &str) -> String {
        respond(self.new_game_inner(config))
    }

    fn new_game_inner(&mut self, config: &str) -> ApiResult<Value> {
        let cfg: GameConfig = parse_json("config", config)?;
        if cfg.seats != 4 {
            return Err(ApiError::new(
                "error.bad_config",
                format!("mahjong is played by 4 seats, not {}", cfg.seats),
            ));
        }
        for &s in &cfg.human_seats {
            check_seat(s)?;
        }
        let rules = rules_from_json(cfg.variants.as_ref())?;
        let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
        let state = State::new_deal(rules, &mut rng);
        // Re-derive the wall the engine shuffled, so a snapshot can carry it.
        let mut wall = State::all_tiles(&rules);
        {
            use rand::seq::SliceRandom;
            let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
            wall.shuffle(&mut rng);
        }
        let entry = GameEntry {
            rules,
            wall,
            state,
            seed: cfg.seed,
            human_seats: cfg.human_seats,
            history: Vec::new(),
        };
        let id = self.register(entry);
        Ok(json!({ "game_id": id }))
    }

    fn register(&mut self, entry: GameEntry) -> String {
        self.next_id += 1;
        let id = format!("g{}", self.next_id);
        self.games.insert(id.clone(), entry);
        id
    }

    fn game(&self, game_id: &str) -> ApiResult<&GameEntry> {
        self.games
            .get(game_id)
            .ok_or_else(|| ApiError::new("error.unknown_game", format!("no game {game_id:?}")))
    }

    fn game_mut(&mut self, game_id: &str) -> ApiResult<&mut GameEntry> {
        self.games
            .get_mut(game_id)
            .ok_or_else(|| ApiError::new("error.unknown_game", format!("no game {game_id:?}")))
    }

    /// `SeatView`: what `seat` may see.
    pub fn view(&self, game_id: &str, seat: usize) -> String {
        respond((|| {
            check_seat(seat)?;
            Ok(seat_view(game_id, self.game(game_id)?, seat))
        })())
    }

    /// `{actions: [Action], seat, phase}` for the seat to act (empty once the hand is over).
    pub fn legal_actions(&self, game_id: &str) -> String {
        respond((|| {
            let g = self.game(game_id)?;
            let actions: Vec<ActionJson> = g
                .state
                .legal_actions_vec()
                .into_iter()
                .map(action_to_json)
                .collect();
            Ok(json!({
                "actions": actions,
                "seat": g.state.current_player(),
                "phase": g.state.phase_name(),
            }))
        })())
    }

    /// Applies an `Action` for the seat to act; returns `{events, view}` where
    /// `view` is for the seat that acted.
    pub fn apply(&mut self, game_id: &str, action: &str) -> String {
        respond(self.apply_inner(game_id, action))
    }

    fn apply_inner(&mut self, game_id: &str, action: &str) -> ApiResult<Value> {
        let a: ActionJson = parse_json("action", action)?;
        let g = self.game_mut(game_id)?;
        let seat = g
            .state
            .current_player()
            .ok_or_else(|| ApiError::new("error.game_over", "the hand is already over"))?;
        let act = legal_action(&g.state, &a)?;
        let events = apply_with_events(g, act);
        Ok(json!({ "events": events, "view": seat_view(game_id, g, seat) }))
    }

    /// The bot's move for the seat to act, at `level`, within `budget_ms`.
    /// The move is not applied; call `apply` with it.
    pub fn bot_action(&self, game_id: &str, level: &str, budget_ms: u32) -> String {
        respond(self.bot_action_inner(game_id, level, budget_ms))
    }

    fn bot_action_inner(&self, game_id: &str, level: &str, budget_ms: u32) -> ApiResult<Value> {
        let g = self.game(game_id)?;
        let legal = g.state.legal_actions_vec();
        if legal.is_empty() {
            return Err(ApiError::new("error.game_over", "the hand is already over"));
        }
        let level_index = BOT_LEVELS.iter().position(|l| *l == level).ok_or_else(|| {
            ApiError::new(
                "error.unknown_level",
                format!("unknown bot level {level:?}"),
            )
        })? as u64;
        let start = Instant::now();
        let action = if legal.len() == 1 {
            legal[0]
        } else if let Some(name) = bot_name(level) {
            let bot = Bot::by_name(name).expect("a known engine bot");
            bot.choose(&g.state, &legal, &mut g.rng(level_index))
        } else {
            analysis(
                &g.state,
                EXPERT_SAMPLES,
                EXPERT_CANDIDATES,
                search_budget(budget_ms),
                g.decision_seed(level_index),
                None,
            )
            .choose(EXPERT_Z)
        };
        let mut v = to_value(&action_to_json(action));
        v["budget_used_ms"] = json!(elapsed_ms(start));
        Ok(v)
    }

    /// `Hint` (§5) for `seat`, who must be the player to act.
    pub fn hint(&self, game_id: &str, seat: usize, budget_ms: u32) -> String {
        respond(self.hint_inner(game_id, seat, budget_ms, HINT_SAMPLES))
    }

    /// [`Api::hint`] with an explicit sample count (for measurements).
    pub fn hint_with_samples(
        &self,
        game_id: &str,
        seat: usize,
        budget_ms: u32,
        samples: usize,
    ) -> String {
        respond(self.hint_inner(game_id, seat, budget_ms, samples))
    }

    fn hint_inner(
        &self,
        game_id: &str,
        seat: usize,
        budget_ms: u32,
        samples: usize,
    ) -> ApiResult<Value> {
        check_seat(seat)?;
        let g = self.game(game_id)?;
        must_be_to_move(&g.state, seat)?;
        let start = Instant::now();
        let legal = g.state.legal_actions_vec();
        let study = Study::of(&g.state, seat);
        let (moves, baseline, used) = if legal.len() == 1 {
            let reasons = vec![Reason::new("reason.forced")];
            let m = json!({
                "action": action_to_json(legal[0]),
                "score": Value::Null,
                "win_rate": Value::Null,
                "numbers": {},
                "reasons": reasons,
            });
            (vec![m], action_to_json(legal[0]), 0)
        } else {
            let a = analysis(
                &g.state,
                samples,
                HINT_CANDIDATES,
                search_budget(budget_ms),
                g.decision_seed(0xA1),
                None,
            );
            let moves = a
                .moves
                .iter()
                .take(HINT_MOVES)
                .map(|m| move_json(&g.state, &study, &legal, m, &a))
                .collect();
            (moves, action_to_json(a.baseline), a.samples)
        };
        Ok(json!({
            "seat": seat,
            "phase": g.state.phase_name(),
            "moves": moves,
            "baseline": baseline,
            "samples": used,
            "progress": study.progress_json(),
            "danger": study.danger_json(),
            "discards": study.discards_json(),
            "budget_used_ms": elapsed_ms(start),
        }))
    }

    /// `{items: [ReviewItem]}` (§5): one item per decision `seat` has made so far,
    /// forced moves included (they carry `reason.forced` and no search).
    pub fn review(&self, game_id: &str, seat: usize, budget_ms: u32) -> String {
        respond(self.review_inner(game_id, seat, budget_ms))
    }

    fn review_inner(&self, game_id: &str, seat: usize, budget_ms: u32) -> ApiResult<Value> {
        check_seat(seat)?;
        let g = self.game(game_id)?;
        let start = Instant::now();
        let deadline = search_budget(budget_ms).map(|b| start + b);
        let decisions = g.decisions_of(seat);
        let mut open = decisions
            .iter()
            .filter(|(_, s, _)| s.legal_actions_vec().len() > 1)
            .count() as u32;
        let mut items = Vec::with_capacity(decisions.len());
        // PIMC checks the clock only between batches, so a short budget is also
        // enforced through the sample cap, from the speed of the previous item.
        let mut ms_per_sample: Option<f64> = None;
        for (turn, state, played) in &decisions {
            let legal = state.legal_actions_vec();
            let played_json = to_value(&action_to_json(*played));
            if legal.len() == 1 {
                items.push(json!({
                    "turn": turn,
                    "player": seat,
                    "phase": state.phase_name(),
                    "played": played_json,
                    "best": played_json,
                    "loss": 0.0,
                    "grade": "best",
                    "reasons": [Reason::new("reason.forced")],
                    "numbers": {},
                }));
                continue;
            }
            // Share the remaining time equally among the decisions still to review.
            let item_limit =
                deadline.map(|d| d.saturating_duration_since(Instant::now()) / open.max(1));
            open -= 1;
            let samples = match (item_limit, ms_per_sample) {
                (Some(limit), Some(per)) if per > 0.0 => {
                    ((limit.as_secs_f64() * 1000.0 / per) as usize).clamp(1, REVIEW_SAMPLES)
                }
                _ => REVIEW_SAMPLES,
            };
            let a = analysis(
                state,
                samples,
                REVIEW_CANDIDATES,
                item_limit,
                g.seed.wrapping_add(*turn as u64),
                Some(*played),
            );
            if a.samples > 0 {
                ms_per_sample = Some(a.elapsed.as_secs_f64() * 1000.0 / a.samples as f64);
            }
            let study = Study::of(state, seat);
            let (mut item, best) = compare(state, &study, &legal, &a, *played);
            item["turn"] = json!(turn);
            item["player"] = json!(seat);
            item["phase"] = json!(state.phase_name());
            item["played"] = played_json;
            item["best"] = to_value(&action_to_json(best));
            items.push(item);
        }
        Ok(json!({
            "seat": seat,
            "items": items,
            "budget_used_ms": elapsed_ms(start),
        }))
    }

    /// `Grade` (§6): how `action` compares with the solver's best move in
    /// `position`, for the player to act there.
    pub fn grade(&self, position: &str, action: &str, budget_ms: u32) -> String {
        respond(grade_inner(position, action, budget_ms))
    }

    /// `Position`: the whole hidden state, enough to `restore` an identical hand.
    pub fn snapshot(&self, game_id: &str) -> String {
        respond(self.game(game_id).map(|g| to_value(&position_json(g))))
    }

    /// Rebuilds a hand from a `Position`; returns `{game_id}`.
    pub fn restore(&mut self, position: &str) -> String {
        respond((|| {
            let entry = entry_from_position(position)?;
            let id = self.register(entry);
            Ok(json!({ "game_id": id }))
        })())
    }

    /// Forgets a hand. Dropping an unknown id is not an error.
    pub fn drop_game(&mut self, game_id: &str) -> String {
        self.games.remove(game_id);
        respond(Ok(json!({})))
    }

    /// Number of hands held (for tests).
    pub fn len(&self) -> usize {
        self.games.len()
    }

    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }
}

/// The solver numbers of one candidate.
fn stats_json(m: &ActionStats, a: &Analysis) -> Value {
    json!({
        "deal_in_rate": round3(m.deal_in_rate),
        "draw_rate": round3(m.draw_rate),
        "score_ci95": round2(m.ci95),
        "samples": m.n,
        "diff": round2(m.diff),
        "diff_se": round2(m.diff_se),
        "baseline": m.action == a.baseline,
    })
}

/// One hint move with its numbers and reasons.
fn move_json(
    state: &State,
    study: &Study,
    legal: &[Action],
    m: &ActionStats,
    a: &Analysis,
) -> Value {
    let mut reasons = position_reasons(state, study, m.action, legal);
    reasons.extend(solver_reasons(m, a));
    json!({
        "action": action_to_json(m.action),
        "score": round2(m.mean),
        "win_rate": round3(m.win_rate),
        "numbers": stats_json(m, a),
        "reasons": reasons,
    })
}

/// Compares the move `played` with the solver's best move: the shared part of a
/// review item and a grade. Returns the item (without turn and player) and the
/// best action.
fn compare(
    state: &State,
    study: &Study,
    legal: &[Action],
    a: &Analysis,
    played: Action,
) -> (Value, Action) {
    let best = &a.moves[0];
    let mine = a
        .moves
        .iter()
        .find(|m| m.action == played)
        .expect("the played move is a candidate");
    // Both moves were played on the same samples, so the paired difference is
    // the precise measure of the loss. The grade uses the part of the loss that
    // is more than sampling noise, so a short search cannot call a move a
    // blunder; `loss_se` is reported with it.
    let loss = (best.diff - mine.diff).max(0.0);
    let loss_se = pair_se(mine, best);
    let is_best = best.action == played;
    let graded_loss = (loss - SIGNIFICANCE * loss_se).max(0.0);
    let grade = if !is_best && graded_loss <= 0.0 {
        "good"
    } else {
        grade_for(is_best, graded_loss)
    };
    let mut reasons = position_reasons(state, study, played, legal);
    reasons.extend(solver_reasons(mine, a));
    let item = json!({
        "loss": round2(loss),
        "grade": grade,
        "reasons": reasons,
        "best_reasons": position_reasons(state, study, best.action, legal),
        "numbers": {
            "played_score": round2(mine.mean),
            "best_score": round2(best.mean),
            "played_win_rate": round3(mine.win_rate),
            "best_win_rate": round3(best.win_rate),
            "played_deal_in_rate": round3(mine.deal_in_rate),
            "best_deal_in_rate": round3(best.deal_in_rate),
            "loss_se": round2(loss_se),
            "samples": a.samples,
        },
    });
    (item, best.action)
}

/// Applies `action` and reports what happened.
fn apply_with_events(g: &mut GameEntry, action: Action) -> Vec<Value> {
    let player = g.state.current_player().expect("checked by the caller");
    let before = g.state.events().len();
    g.apply(action);
    let mut events = Vec::new();
    if is_declaration(action) {
        // The engine logs no event for a declaration; the acting seat sees its own.
        events.push(
            json!({ "type": "event.declared", "player": player, "action": action_to_json(action) }),
        );
    }
    events.extend(g.state.events()[before..].iter().map(event_to_json));
    if let Some(r) = g.state.result() {
        events.push(json!({ "type": "event.hand_over", "result": result_to_json(r, &g.rules) }));
    }
    events
}

/// What `seat` may see of the hand.
fn seat_view(game_id: &str, g: &GameEntry, seat: usize) -> Value {
    let s = &g.state;
    let over = s.is_terminal();
    let your_turn = s.current_player() == Some(seat);
    let legal: Vec<ActionJson> = if your_turn {
        s.legal_actions_vec()
            .into_iter()
            .map(action_to_json)
            .collect()
    } else {
        Vec::new()
    };
    json!({
        "game_id": game_id,
        "seat": seat,
        "rules": rules_to_json(&g.rules),
        "seed": g.seed,
        "human_seats": g.human_seats,
        "phase": s.phase_name(),
        "current_player": s.current_player(),
        "your_turn": your_turn,
        "turn": s.turn(),
        "wall_remaining": s.wall_remaining(),
        "hand": hand_to_json(s.hand(seat)),
        "last_drawn": if your_turn { s.last_drawn().map(tile_id) } else { None },
        "melds": (0..4).map(|p| s.melds(p).iter().map(meld_to_json).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "bonus": (0..4).map(|p| bonus_to_json(s.bonus(p))).collect::<Vec<_>>(),
        "concealed_counts": (0..4).map(|p| s.concealed_count(p)).collect::<Vec<_>>(),
        "river": s.river().into_iter().map(|(p, t, claimed)| json!({ "player": p, "tile": tile_id(t), "claimed": claimed })).collect::<Vec<_>>(),
        "claim": s.claim_tile().map(|(t, from)| json!({ "tile": tile_id(t), "from": from })),
        "liable": (0..4).map(|p| s.liable_for(p)).collect::<Vec<_>>(),
        "legal": legal,
        "history": g.history_json(Some(seat)),
        "over": over,
        "result": s.result().map(|r| result_to_json(r, &g.rules)),
        "final_hands": over.then(|| (0..4).map(|p| hand_to_json(s.hand(p))).collect::<Vec<_>>()),
    })
}

fn position_json(g: &GameEntry) -> PositionJson {
    let s = &g.state;
    PositionJson {
        api: API_VERSION,
        game: "mahjong".into(),
        rules: Some(rules_to_json(&g.rules)),
        seed: g.seed,
        human_seats: g.human_seats.clone(),
        wall: tiles_to_json(&g.wall),
        history: g.history_turns(),
        phase: Some(s.phase_name().to_string()),
        current_player: s.current_player(),
        hands: (0..4).map(|p| hand_to_json(s.hand(p))).collect(),
        wall_remaining: Some(s.wall_remaining()),
        over: s.is_terminal(),
        result: s.result().map(|r| result_to_json(r, &g.rules)),
    }
}

fn entry_from_position(position: &str) -> ApiResult<GameEntry> {
    let p: PositionJson = parse_json("position", position)?;
    if p.game != "mahjong" {
        return Err(ApiError::new(
            "error.bad_position",
            format!("this engine plays mahjong, not {:?}", p.game),
        ));
    }
    for &s in &p.human_seats {
        check_seat(s)?;
    }
    let rules = rules_from_json(p.rules.as_ref())?;
    let wall = p
        .wall
        .iter()
        .map(|s| tile_from_id(s))
        .collect::<ApiResult<Vec<_>>>()
        .map_err(|e| ApiError::new("error.bad_position", format!("wall: {}", e.detail)))?;
    let mut entry = GameEntry::from_wall(rules, wall, p.seed, p.human_seats)?;
    entry.replay(&p.history)?;
    Ok(entry)
}

fn grade_inner(position: &str, action: &str, budget_ms: u32) -> ApiResult<Value> {
    let g = entry_from_position(position)?;
    let a: ActionJson = parse_json("action", action)?;
    let seat = g
        .state
        .current_player()
        .ok_or_else(|| ApiError::new("error.game_over", "the position is a finished hand"))?;
    let played = legal_action(&g.state, &a)?;
    let start = Instant::now();
    let legal = g.state.legal_actions_vec();
    if legal.len() == 1 {
        return Ok(json!({
            "seat": seat,
            "grade": "best",
            "loss": 0.0,
            "best": action_to_json(played),
            "reasons": [Reason::new("reason.forced")],
            "numbers": {},
            "budget_used_ms": elapsed_ms(start),
        }));
    }
    let a = analysis(
        &g.state,
        HINT_SAMPLES,
        HINT_CANDIDATES,
        search_budget(budget_ms),
        g.decision_seed(0xA1),
        Some(played),
    );
    let study = Study::of(&g.state, seat);
    let (mut item, best) = compare(&g.state, &study, &legal, &a, played);
    item["seat"] = json!(seat);
    item["best"] = to_value(&action_to_json(best));
    item["budget_used_ms"] = json!(elapsed_ms(start));
    Ok(item)
}
