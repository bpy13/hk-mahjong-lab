//! The JSON shapes that cross the app boundary, and their conversions to and
//! from engine types. Every string the app sees is a tile id (`"5p"`) or a key
//! (`tile.5_dots`, `action.pung`, `faan.full_flush`, `error.illegal_action`);
//! never display text (D42).

use mahjong_games::Game;
use mahjong_hk::{
    Action, Event, HandResult, Meld, MeldKind, MultipleWinners, Pattern, Payout, RuleSet, Score,
    State,
};
use mahjong_tiles::{Suit, Tile, TileCounts, NUM_TILES_WITH_BONUS};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Version of `docs/engine-interface.md` this crate implements.
pub const API_VERSION: u32 = 0;

/// An error response: `{"error": {"code": "error.*", "detail": "..."}}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    pub code: &'static str,
    pub detail: String,
}

impl ApiError {
    pub fn new(code: &'static str, detail: impl Into<String>) -> ApiError {
        ApiError {
            code,
            detail: detail.into(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "api": API_VERSION,
            "error": { "code": self.code, "detail": self.detail },
        })
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

// -- tiles --------------------------------------------------------------------

/// The tile id: the engine's text notation (`"5p"`, `"Wh"`, `"F2"`).
pub fn tile_id(t: Tile) -> String {
    t.to_string()
}

/// The `tile.*` key of a tile (`tile.5_dots`, `tile.white`, `tile.flower_2`).
pub fn tile_key(t: Tile) -> String {
    if let Some(suit) = t.suit() {
        return format!("tile.{}_{}", t.number().expect("suited"), suit_name(suit));
    }
    match t.index() {
        27 => "tile.east".into(),
        28 => "tile.south".into(),
        29 => "tile.west".into(),
        30 => "tile.north".into(),
        31 => "tile.white".into(),
        32 => "tile.green".into(),
        33 => "tile.red".into(),
        i if t.is_flower() => format!("tile.flower_{}", i - 33),
        i => format!("tile.season_{}", i - 37),
    }
}

/// `characters`, `dots` or `bamboo`.
pub fn suit_name(suit: Suit) -> &'static str {
    match suit {
        Suit::Characters => "characters",
        Suit::Dots => "dots",
        Suit::Bamboo => "bamboo",
    }
}

/// `{id: key}` for all 42 tiles (`GameMeta.tiles`).
pub fn tile_keys_meta() -> Value {
    let mut m = Map::new();
    for i in 0..NUM_TILES_WITH_BONUS as u8 {
        let t = Tile::from_index(i);
        m.insert(tile_id(t), json!(tile_key(t)));
    }
    Value::Object(m)
}

/// Parses one tile id; anything else is `error.bad_tile`.
pub fn tile_from_id(s: &str) -> ApiResult<Tile> {
    s.parse::<Tile>()
        .map_err(|e| ApiError::new("error.bad_tile", e.to_string()))
}

/// A hand as sorted tile ids (index order, repeats kept).
pub fn hand_to_json(hand: &TileCounts) -> Vec<String> {
    hand.tiles().into_iter().map(tile_id).collect()
}

pub fn tiles_to_json(tiles: &[Tile]) -> Vec<String> {
    tiles.iter().map(|&t| tile_id(t)).collect()
}

/// Bonus tiles held, from the engine's bit mask (bit `i` = tile `34 + i`).
pub fn bonus_to_json(mask: u8) -> Vec<String> {
    (0..8u8)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| tile_id(Tile::from_index(34 + i)))
        .collect()
}

// -- actions ------------------------------------------------------------------

pub const ACTION_DISCARD: &str = "action.discard";
pub const ACTION_SELF_DRAWN_WIN: &str = "action.self_drawn_win";
pub const ACTION_CONCEALED_KONG: &str = "action.concealed_kong";
pub const ACTION_ADDED_KONG: &str = "action.added_kong";
pub const ACTION_PASS: &str = "action.pass";
pub const ACTION_WIN: &str = "action.win";
pub const ACTION_PUNG: &str = "action.pung";
pub const ACTION_KONG: &str = "action.kong";
pub const ACTION_CHOW: &str = "action.chow";

/// A move: `{"type": "action.discard", "tile": "5p"}`, `{"type": "action.pung"}`,
/// `{"type": "action.chow", "tile": "3p"}` (the chow's lowest tile) and so on.
/// On output `code` (the engine's 0–133 action code) and, for a chow, `tiles`
/// are added. On input `code` alone is accepted when `type` is left out.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ActionJson {
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tiles: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<u16>,
}

/// The `action.*` key of an action.
pub fn action_key(action: Action) -> &'static str {
    match action {
        Action::Discard(_) => ACTION_DISCARD,
        Action::SelfDrawnWin => ACTION_SELF_DRAWN_WIN,
        Action::ConcealedKong(_) => ACTION_CONCEALED_KONG,
        Action::AddedKong(_) => ACTION_ADDED_KONG,
        Action::Pass => ACTION_PASS,
        Action::Win => ACTION_WIN,
        Action::Pung => ACTION_PUNG,
        Action::ClaimKong => ACTION_KONG,
        Action::Chow(_) => ACTION_CHOW,
    }
}

/// Every action key (`GameMeta.actions`).
pub const ACTION_KEYS: [&str; 9] = [
    ACTION_DISCARD,
    ACTION_SELF_DRAWN_WIN,
    ACTION_CONCEALED_KONG,
    ACTION_ADDED_KONG,
    ACTION_PASS,
    ACTION_WIN,
    ACTION_PUNG,
    ACTION_KONG,
    ACTION_CHOW,
];

pub fn action_to_json(action: Action) -> ActionJson {
    let tile = match action {
        Action::Discard(t) | Action::ConcealedKong(t) | Action::AddedKong(t) | Action::Chow(t) => {
            Some(tile_id(t))
        }
        _ => None,
    };
    // Action codes allow a chow from any suited tile; only one that fits in the
    // suit (lowest tile 1–7) has three tiles.
    let tiles = match action {
        Action::Chow(low) if low.offset(2).is_some() => Some(
            (0..3)
                .map(|k| tile_id(low.offset(k).expect("checked")))
                .collect(),
        ),
        _ => None,
    };
    ActionJson {
        kind: action_key(action).to_string(),
        tile,
        tiles,
        code: Some(action.code()),
    }
}

/// Reads an action's shape (not its legality).
pub fn action_from_json(a: &ActionJson) -> ApiResult<Action> {
    if a.kind.is_empty() {
        return match a.code {
            Some(code) => Action::from_code(code).ok_or_else(|| {
                ApiError::new("error.bad_action", format!("invalid action code {code}"))
            }),
            None => Err(ApiError::new(
                "error.bad_action",
                "an action needs a type (or a code)",
            )),
        };
    }
    let tile = || -> ApiResult<Tile> {
        let id = a
            .tile
            .as_deref()
            .ok_or_else(|| ApiError::new("error.bad_action", format!("{} needs a tile", a.kind)))?;
        tile_from_id(id)
    };
    Ok(match a.kind.as_str() {
        ACTION_DISCARD => Action::Discard(tile()?),
        ACTION_SELF_DRAWN_WIN => Action::SelfDrawnWin,
        ACTION_CONCEALED_KONG => Action::ConcealedKong(tile()?),
        ACTION_ADDED_KONG => Action::AddedKong(tile()?),
        ACTION_PASS => Action::Pass,
        ACTION_WIN => Action::Win,
        ACTION_PUNG => Action::Pung,
        ACTION_KONG => Action::ClaimKong,
        ACTION_CHOW => {
            // The chow may be given by its lowest tile or by its three tiles.
            match (&a.tile, &a.tiles) {
                (Some(_), _) => Action::Chow(tile()?),
                (None, Some(ts)) if !ts.is_empty() => {
                    let mut parsed = ts
                        .iter()
                        .map(|s| tile_from_id(s))
                        .collect::<ApiResult<Vec<_>>>()?;
                    parsed.sort();
                    Action::Chow(parsed[0])
                }
                _ => {
                    return Err(ApiError::new(
                        "error.bad_action",
                        "action.chow needs a tile",
                    ))
                }
            }
        }
        other => {
            return Err(ApiError::new(
                "error.bad_action",
                format!("unknown action type {other:?}"),
            ))
        }
    })
}

/// Reads an action and checks it is legal for the seat to act in `state`.
pub fn legal_action(state: &State, a: &ActionJson) -> ApiResult<Action> {
    let action = action_from_json(a)?;
    if state.is_terminal() {
        return Err(ApiError::new("error.game_over", "the hand is already over"));
    }
    if !state.legal_actions_vec().contains(&action) {
        return Err(ApiError::new(
            "error.illegal_action",
            format!(
                "{action} is not legal for seat {} in the {} phase",
                state.current_player().expect("not over"),
                state.phase_name()
            ),
        ));
    }
    Ok(action)
}

/// Whether an action is a claim-window or rob-window declaration (hidden from
/// the other seats until the window closes).
pub fn is_declaration(action: Action) -> bool {
    matches!(
        action,
        Action::Pass | Action::Win | Action::Pung | Action::ClaimKong | Action::Chow(_)
    )
}

/// One entry of the history: who acted and what they did.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct TurnJson {
    pub player: usize,
    pub action: ActionJson,
}

// -- rules --------------------------------------------------------------------

pub const WINDS: [&str; 4] = ["east", "south", "west", "north"];

fn payout_name(p: Payout) -> &'static str {
    match p {
        Payout::HalfSpicy => "half_spicy",
        Payout::FullSpicy => "full_spicy",
    }
}

fn multiple_winners_name(m: MultipleWinners) -> &'static str {
    match m {
        MultipleWinners::All => "all",
        MultipleWinners::HeadBump => "head_bump",
    }
}

fn on_off(b: bool) -> &'static str {
    if b {
        "on"
    } else {
        "off"
    }
}

/// Rule variants as `{"flowers": "on", "min_faan": 3, ..., "prevailing_wind": "east"}`.
pub fn rules_to_json(r: &RuleSet) -> Value {
    json!({
        "flowers": on_off(r.flowers),
        "min_faan": r.min_faan,
        "faan_cap": r.faan_cap,
        "payout": payout_name(r.payout),
        "multiple_winners": multiple_winners_name(r.multiple_winners),
        "twelve_tile_liability": on_off(r.twelve_tile_liability),
        "prevailing_wind": WINDS[r.prevailing_wind],
    })
}

/// Every variant and its options, default first (`GameMeta.variants`). The
/// numeric variants list the usual choices; any value in range is accepted.
pub fn variants_meta() -> Value {
    json!({
        "flowers": ["on", "off"],
        "min_faan": ["3", "0", "1", "2", "4", "5"],
        "faan_cap": ["10", "13", "8"],
        "payout": ["half_spicy", "full_spicy"],
        "multiple_winners": ["all", "head_bump"],
        "twelve_tile_liability": ["on", "off"],
        "prevailing_wind": WINDS,
    })
}

fn variant_err(key: &str, what: impl std::fmt::Display) -> ApiError {
    ApiError::new("error.unknown_variant", format!("{key}: {what}"))
}

fn parse_switch(key: &str, v: &Value) -> ApiResult<bool> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::String(s) if s == "on" => Ok(true),
        Value::String(s) if s == "off" => Ok(false),
        other => Err(variant_err(key, format!("{other} (use \"on\" or \"off\")"))),
    }
}

fn parse_number(key: &str, v: &Value) -> ApiResult<u8> {
    let n = match v {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse::<u64>().ok(),
        _ => None,
    };
    n.filter(|n| *n <= 13)
        .map(|n| n as u8)
        .ok_or_else(|| variant_err(key, format!("{v} (expected a number 0-13)")))
}

/// Reads rule variants; keys left out keep the default. Unknown keys or values
/// are `error.unknown_variant`.
pub fn rules_from_json(v: Option<&Value>) -> ApiResult<RuleSet> {
    let mut r = RuleSet::default();
    let Some(v) = v else { return Ok(r) };
    if v.is_null() {
        return Ok(r);
    }
    let map = v
        .as_object()
        .ok_or_else(|| ApiError::new("error.unknown_variant", "variants must be an object"))?;
    for (key, value) in map {
        match key.as_str() {
            "flowers" => r.flowers = parse_switch(key, value)?,
            "twelve_tile_liability" => r.twelve_tile_liability = parse_switch(key, value)?,
            "min_faan" => r.min_faan = parse_number(key, value)?,
            "faan_cap" => r.faan_cap = parse_number(key, value)?,
            "payout" => {
                let s = value
                    .as_str()
                    .ok_or_else(|| variant_err(key, "expected a string"))?;
                r.payout = s.parse().map_err(|e| variant_err(key, e))?;
            }
            "multiple_winners" => {
                let s = value
                    .as_str()
                    .ok_or_else(|| variant_err(key, "expected a string"))?;
                r.multiple_winners = s.parse().map_err(|e| variant_err(key, e))?;
            }
            "prevailing_wind" => {
                r.prevailing_wind = match value {
                    Value::Number(n) if n.as_u64().is_some_and(|w| w < 4) => {
                        n.as_u64().expect("checked") as usize
                    }
                    Value::String(s) => WINDS.iter().position(|w| w == s).ok_or_else(|| {
                        variant_err(key, format!("{s:?} (east, south, west or north)"))
                    })?,
                    other => {
                        return Err(variant_err(
                            key,
                            format!("{other} (east, south, west or north)"),
                        ))
                    }
                }
            }
            other => {
                return Err(ApiError::new(
                    "error.unknown_variant",
                    format!("unknown rule variant {other:?}"),
                ))
            }
        }
    }
    if r.min_faan > r.faan_cap {
        return Err(ApiError::new(
            "error.unknown_variant",
            format!("min_faan {} is above faan_cap {}", r.min_faan, r.faan_cap),
        ));
    }
    Ok(r)
}

// -- game setup ---------------------------------------------------------------

/// `GameConfig` of the contract (§4).
#[derive(Deserialize, Clone, Debug)]
pub struct GameConfig {
    #[serde(default = "four")]
    pub seats: usize,
    #[serde(default)]
    pub variants: Option<Value>,
    pub seed: u64,
    #[serde(default)]
    pub human_seats: Vec<usize>,
}

fn four() -> usize {
    4
}

// -- melds, scores, results ---------------------------------------------------

pub fn meld_kind_key(kind: MeldKind) -> &'static str {
    match kind {
        MeldKind::Chow => "meld.chow",
        MeldKind::Pung => "meld.pung",
        MeldKind::Kong => "meld.kong",
    }
}

/// `{kind, tile, tiles, concealed, from}`.
pub fn meld_to_json(m: &Meld) -> Value {
    json!({
        "kind": meld_kind_key(m.kind),
        "tile": tile_id(m.tile),
        "tiles": tiles_to_json(&m.tiles()),
        "concealed": m.concealed,
        "from": m.claimed_from.map(usize::from),
    })
}

/// The `faan.*` key of a faan-table row: the English name in snake case
/// (`faan.full_flush`, `faan.robbing_the_kong`).
pub fn pattern_key(p: Pattern) -> String {
    let mut key = String::from("faan.");
    let mut last_underscore = false;
    for c in p.english().chars() {
        if c.is_ascii_alphanumeric() {
            key.push(c.to_ascii_lowercase());
            last_underscore = false;
        } else if !last_underscore {
            key.push('_');
            last_underscore = true;
        }
    }
    key
}

/// Every faan-table row: `{key, number, limit}` (`GameMeta.faan`).
pub fn faan_meta() -> Vec<Value> {
    Pattern::ALL
        .iter()
        .map(|&p| json!({ "key": pattern_key(p), "number": p.number(), "limit": p.is_limit() }))
        .collect()
}

/// `{faan, total, qualifying, limit, patterns: [{key, faan}]}`.
pub fn score_to_json(s: &Score) -> Value {
    json!({
        "faan": s.faan,
        "total": s.total,
        "qualifying": s.qualifying,
        "limit": s.limit,
        "patterns": s.patterns.iter().map(|&(p, f)| json!({ "key": pattern_key(p), "faan": f })).collect::<Vec<_>>(),
    })
}

/// How the hand ended: `{wins: [...], deltas: [4]}`; `wins` is empty after an
/// exhaustive draw.
pub fn result_to_json(r: &HandResult, rules: &RuleSet) -> Value {
    json!({
        "wins": r.wins.iter().map(|w| json!({
            "seat": w.seat,
            "score": score_to_json(&w.score),
            "points_per_share": rules.points(w.score.faan),
            "self_drawn": w.self_drawn,
            "from": w.from,
            "win_tile": w.win_tile.map(tile_id),
            "paid_by": w.paid_by,
            "liable": w.liable,
            "robbed": w.robbed,
        })).collect::<Vec<_>>(),
        "deltas": r.deltas,
    })
}

// -- events -------------------------------------------------------------------

/// An engine event as `{"type": "event.*", ...}`. Draws do not reveal the tile.
pub fn event_to_json(e: &Event) -> Value {
    match *e {
        Event::Draw { seat, replacement } => {
            json!({ "type": "event.draw", "player": seat, "replacement": replacement })
        }
        Event::Bonus { seat, tile } => {
            json!({ "type": "event.bonus", "player": seat, "tile": tile_id(tile) })
        }
        Event::Discard { seat, tile } => {
            json!({ "type": "event.discard", "player": seat, "tile": tile_id(tile) })
        }
        Event::Claim { seat, from, meld } => {
            json!({ "type": "event.claim", "player": seat, "from": from, "meld": meld_to_json(&meld) })
        }
        Event::ConcealedKong { seat, tile } => {
            json!({ "type": "event.concealed_kong", "player": seat, "tile": tile_id(tile) })
        }
        Event::AddedKong { seat, tile } => {
            json!({ "type": "event.added_kong", "player": seat, "tile": tile_id(tile) })
        }
        Event::Win { seat } => json!({ "type": "event.win", "player": seat }),
        Event::ExhaustiveDraw => json!({ "type": "event.exhaustive_draw" }),
    }
}

// -- positions ----------------------------------------------------------------

/// `Position` of the contract: the whole hidden state. `wall` (the full tile set
/// in deal order) plus `history` rebuilds the hand; the other fields describe
/// the resulting position and are ignored on restore.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PositionJson {
    #[serde(default = "api_version")]
    pub api: u32,
    #[serde(default = "mahjong")]
    pub game: String,
    #[serde(default)]
    pub rules: Option<Value>,
    pub seed: u64,
    #[serde(default)]
    pub human_seats: Vec<usize>,
    pub wall: Vec<String>,
    #[serde(default)]
    pub history: Vec<TurnJson>,
    // Derived, informational:
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub current_player: Option<usize>,
    #[serde(default)]
    pub hands: Vec<Vec<String>>,
    #[serde(default)]
    pub wall_remaining: Option<usize>,
    #[serde(default)]
    pub over: bool,
    #[serde(default)]
    pub result: Option<Value>,
}

fn api_version() -> u32 {
    API_VERSION
}

fn mahjong() -> String {
    "mahjong".to_string()
}

// -- reasons ------------------------------------------------------------------

/// A reason key with its arguments: `{"key": "reason.safe_tile", "args": {"danger": 0.004}}`.
#[derive(Serialize, Clone, Debug)]
pub struct Reason {
    pub key: &'static str,
    pub args: Map<String, Value>,
}

impl Reason {
    pub fn new(key: &'static str) -> Reason {
        Reason {
            key,
            args: Map::new(),
        }
    }

    pub fn arg(mut self, name: &str, value: impl Into<Value>) -> Reason {
        self.args.insert(name.to_string(), value.into());
        self
    }
}

/// Rounds a solver number for the app (two decimals).
pub fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Three decimals, for rates.
pub fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_keys_follow_the_scheme() {
        let t = |s: &str| s.parse::<Tile>().unwrap();
        assert_eq!(tile_key(t("5p")), "tile.5_dots");
        assert_eq!(tile_key(t("1m")), "tile.1_characters");
        assert_eq!(tile_key(t("9s")), "tile.9_bamboo");
        assert_eq!(tile_key(t("E")), "tile.east");
        assert_eq!(tile_key(t("Wh")), "tile.white");
        assert_eq!(tile_key(t("R")), "tile.red");
        assert_eq!(tile_key(t("F2")), "tile.flower_2");
        assert_eq!(tile_key(t("S4")), "tile.season_4");
        assert_eq!(tile_keys_meta().as_object().unwrap().len(), 42);
    }

    #[test]
    fn pattern_keys_are_stable() {
        let expected = [
            "faan.self_drawn",
            "faan.concealed_hand",
            "faan.no_flowers",
            "faan.seat_flower",
            "faan.full_flower_set",
            "faan.last_tile",
            "faan.kong_replacement",
            "faan.double_kong",
            "faan.robbing_the_kong",
            "faan.dragon_pung",
            "faan.seat_wind",
            "faan.prevailing_wind",
            "faan.chicken_hand",
            "faan.all_chows",
            "faan.all_pungs",
            "faan.half_flush",
            "faan.full_flush",
            "faan.terminals_and_honours",
            "faan.small_three_dragons",
            "faan.big_three_dragons",
            "faan.small_four_winds",
            "faan.big_four_winds",
            "faan.all_honours",
            "faan.all_terminals",
            "faan.thirteen_orphans",
            "faan.nine_gates",
            "faan.four_concealed_pungs",
            "faan.four_kongs",
            "faan.heavenly_hand",
            "faan.earthly_hand",
            "faan.eight_immortals",
            "faan.seven_rob_one",
        ];
        let got: Vec<String> = Pattern::ALL.iter().map(|&p| pattern_key(p)).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn actions_round_trip() {
        for code in 0..mahjong_hk::NUM_ACTION_CODES as u16 {
            let a = Action::from_code(code).unwrap();
            let j = action_to_json(a);
            assert!(j.kind.starts_with("action."));
            assert_eq!(action_from_json(&j).unwrap(), a);
            // By code alone.
            let by_code = ActionJson {
                kind: String::new(),
                tile: None,
                tiles: None,
                code: Some(code),
            };
            assert_eq!(action_from_json(&by_code).unwrap(), a);
        }
        let chow = action_to_json(Action::Chow("3p".parse().unwrap()));
        assert_eq!(
            chow.tiles,
            Some(vec!["3p".into(), "4p".into(), "5p".into()])
        );
    }
}
