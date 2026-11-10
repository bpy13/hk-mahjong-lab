//! The stateless `waits` call: which tiles complete a concealed hand of 4, 7,
//! 10 or 13 tiles (hand checker, hk-games D69). Exposed sets are assumed to make
//! up the rest of the 13 (`(13 - n) / 3` sets); flowers and minimum faan are ignored.
//!
//! Tiles are the engine's tile ids, as in the JSON views (`"5p"`, `"E"`, `"Wh"`).

use crate::shapes::{tile_id, tile_key};
use mahjong_hk::hand;
use mahjong_tiles::{Tile, TileCounts};
use serde_json::{json, Value};

/// `{"ok":true,"waits":[ids]}` or `{"ok":false,"error":"error.…"}`.
pub fn waits(tiles_json: &str) -> String {
    match compute(tiles_json) {
        Ok(w) => json!({ "ok": true, "waits": w }),
        Err(e) => json!({ "ok": false, "error": e }),
    }
    .to_string()
}

pub(crate) fn parse_tile(v: &Value) -> Option<Tile> {
    let s = v.as_str()?;
    let t = s.parse::<Tile>().ok().or_else(|| {
        // Also accept the `tile.*` display keys.
        Tile::all_kinds().find(|&t| tile_key(t) == s)
    })?;
    (!t.is_bonus()).then_some(t)
}

fn compute(tiles_json: &str) -> Result<Vec<String>, &'static str> {
    let Ok(Value::Array(items)) = serde_json::from_str::<Value>(tiles_json) else {
        return Err("error.badTile");
    };
    if !matches!(items.len(), 4 | 7 | 10 | 13) {
        return Err("error.handSize");
    }
    let mut hand_counts = TileCounts::new();
    for item in &items {
        let t = parse_tile(item).ok_or("error.badTile")?;
        if hand_counts[t] >= 4 {
            return Err("error.tooMany");
        }
        hand_counts.add(t);
    }
    let melds = (13 - items.len()) / 3;
    // `hand::waits` already skips tiles whose four copies are all in the hand.
    Ok(hand::waits(&hand_counts, melds)
        .into_iter()
        .map(tile_id)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(tiles: &str) -> Value {
        let list: Vec<&str> = tiles.split_whitespace().collect();
        serde_json::from_str(&waits(&serde_json::to_string(&list).unwrap())).unwrap()
    }

    fn wait_list(tiles: &str) -> Vec<String> {
        let v = run(tiles);
        assert_eq!(v["ok"], true, "{v}");
        v["waits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect()
    }

    fn err(tiles: &str) -> String {
        let v = run(tiles);
        assert_eq!(v["ok"], false);
        v["error"].as_str().unwrap().to_string()
    }

    #[test]
    fn nine_gates_has_nine_waits() {
        let w = wait_list("1m 1m 1m 2m 3m 4m 5m 6m 7m 8m 9m 9m 9m");
        assert_eq!(w, ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"]);
    }

    #[test]
    fn thirteen_orphans_has_thirteen_waits() {
        let w = wait_list("1m 9m 1p 9p 1s 9s E S W N Wh G R");
        assert_eq!(w.len(), 13);
    }

    #[test]
    fn shanpon_has_two_waits() {
        assert_eq!(wait_list("3p 3p 7s 7s"), ["3p", "7s"]);
    }

    #[test]
    fn no_wait() {
        assert!(wait_list("1m 4m 7m 2p 5p 8p 3s").is_empty());
    }

    #[test]
    fn four_held_is_not_a_wait() {
        // 5m would make 234 + 555 + 55, but all four 5m are already in the hand.
        assert_eq!(wait_list("2m 3m 4m 5m 5m 5m 5m"), ["2m"]);
    }

    #[test]
    fn errors() {
        assert_eq!(err("1m 2m 3m"), "error.handSize");
        assert_eq!(err("1m 2m 3m 4m 5m"), "error.handSize");
        assert_eq!(err("1m 2m 3m F1"), "error.badTile");
        assert_eq!(err("1m 2m 3m xx"), "error.badTile");
        assert_eq!(err("1m 1m 1m 1m 1m 2m 3m"), "error.tooMany");
        assert_eq!(
            serde_json::from_str::<Value>(&waits("nope")).unwrap()["error"],
            "error.badTile"
        );
    }
}
