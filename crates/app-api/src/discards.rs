//! The stateless `best_discards` call: after a draw (5, 8, 11 or 14 concealed tiles), rank
//! each possible discard by chance of winning (hand checker, hk-games D85). Exposed sets are
//! assumed to make up the rest of the 13, as in [`crate::waits`]; flowers and minimum faan
//! are ignored.
//!
//! Per discard: shanten after it (0 = ready), the tiles that lower the shanten with the
//! copies of each still unseen, and their total. Best first: fewest steps, then most unseen
//! useful copies. Unseen copies are 4 minus the copies in the hand and in the seen list
//! (the player's own exposed sets belong in that list). Keys only (D42).

use crate::shapes::tile_id;
use crate::waits::parse_tile;
use mahjong_analysis::shanten;
use mahjong_tiles::{Tile, TileCounts};
use serde_json::{json, Value};

/// `{"ok":true,"discards":[{"discard","shanten","useful":[{"tile","left"}],"total"}]}`
/// or `{"ok":false,"error":"error.…"}`. `seen_json` may be empty.
pub fn best_discards(tiles_json: &str, seen_json: &str) -> String {
    match compute(tiles_json, seen_json) {
        Ok(d) => json!({ "ok": true, "discards": d }),
        Err(e) => json!({ "ok": false, "error": e }),
    }
    .to_string()
}

fn parse_list(json: &str) -> Result<Vec<Tile>, &'static str> {
    let Ok(Value::Array(items)) = serde_json::from_str::<Value>(json) else {
        return Err("error.badTile");
    };
    items
        .iter()
        .map(|v| parse_tile(v).ok_or("error.badTile"))
        .collect()
}

fn compute(tiles_json: &str, seen_json: &str) -> Result<Vec<Value>, &'static str> {
    let tiles = parse_list(tiles_json)?;
    if !matches!(tiles.len(), 5 | 8 | 11 | 14) {
        return Err("error.handSize");
    }
    let seen = if seen_json.trim().is_empty() {
        Vec::new()
    } else {
        parse_list(seen_json)?
    };
    let hand = TileCounts::from_tiles(&tiles);
    let mut taken = hand;
    for &t in &seen {
        taken.add(t);
    }
    if Tile::all_kinds().any(|t| taken[t] > 4) {
        return Err("error.tooMany");
    }
    // After the discard the hand has 13 - 3 * melds tiles.
    let melds = (13 - (tiles.len() - 1)) / 3;

    let mut rows = Vec::new();
    for discard in Tile::all_kinds().filter(|&t| hand[t] > 0) {
        let mut c = hand;
        c.remove(discard);
        let steps = shanten(&c, melds);
        let mut useful = Vec::new();
        let mut total = 0i64;
        for t in Tile::all_kinds() {
            let left = 4 - taken[t] as i64;
            if left == 0 {
                continue;
            }
            c.add(t);
            if shanten(&c, melds) < steps {
                useful.push(json!({ "tile": tile_id(t), "left": left }));
                total += left;
            }
            c.remove(t);
        }
        rows.push((steps, total, discard, useful));
    }
    rows.sort_by_key(|(steps, total, discard, _)| (*steps, -*total, discard.index()));
    Ok(rows
        .into_iter()
        .map(|(steps, total, discard, useful)| {
            json!({ "discard": tile_id(discard), "shanten": steps, "useful": useful, "total": total })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(tiles: &str, seen: &str) -> Value {
        let t: Vec<&str> = tiles.split_whitespace().collect();
        let s: Vec<&str> = seen.split_whitespace().collect();
        serde_json::from_str(&best_discards(
            &serde_json::to_string(&t).unwrap(),
            &serde_json::to_string(&s).unwrap(),
        ))
        .unwrap()
    }

    fn rows(tiles: &str, seen: &str) -> Vec<Value> {
        let v = run(tiles, seen);
        assert_eq!(v["ok"], true, "{v}");
        v["discards"].as_array().unwrap().clone()
    }

    fn useful(row: &Value) -> Vec<(String, i64)> {
        row["useful"]
            .as_array()
            .unwrap()
            .iter()
            .map(|u| {
                (
                    u["tile"].as_str().unwrap().into(),
                    u["left"].as_i64().unwrap(),
                )
            })
            .collect()
    }

    fn pair(t: &str, n: i64) -> (String, i64) {
        (t.to_string(), n)
    }

    #[test]
    fn five_tiles() {
        // 3 melds on the table; the stray 9m goes, leaving a double-pair wait.
        let r = rows("3p 3p 7s 7s 9m", "");
        assert_eq!(r[0]["discard"], "9m");
        assert_eq!(r[0]["shanten"], 0);
        assert_eq!(useful(&r[0]), [pair("3p", 2), pair("7s", 2)]);
        assert_eq!(r[0]["total"], 4);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn eight_tiles() {
        // 2 melds: 2m 3m 4m + 6m 6m + 8p 9p + a stray East; discard East.
        let r = rows("2m 3m 4m 6m 6m 8p 9p E", "");
        assert_eq!(r[0]["discard"], "E");
        assert_eq!(r[0]["shanten"], 0);
        assert_eq!(useful(&r[0]), [pair("7p", 4)]);
    }

    #[test]
    fn eleven_tiles() {
        // 1 meld: 1m 2m 3m, 4p 5p 6p, 7s 8s, 9s... plus a stray: 10 after discard.
        let r = rows("1m 2m 3m 4p 5p 6p 7s 8s 5s 5s E", "");
        assert_eq!(r[0]["discard"], "E");
        assert_eq!(r[0]["shanten"], 0);
        assert_eq!(useful(&r[0]), [pair("6s", 4), pair("9s", 4)]);
    }

    #[test]
    fn fourteen_tiles() {
        // Ready after discarding the East: waits on 1p / 4p.
        let r = rows("1m 2m 3m 4m 5m 6m 7s 8s 9s 2p 3p 5p 5p E", "");
        assert_eq!(r[0]["discard"], "E");
        assert_eq!(r[0]["shanten"], 0);
        assert_eq!(useful(&r[0]), [pair("1p", 4), pair("4p", 4)]);
        assert_eq!(
            r.len(),
            Tile::all_kinds()
                .filter(|t| [
                    "1m", "2m", "3m", "4m", "5m", "6m", "7s", "8s", "9s", "2p", "3p", "5p", "E"
                ]
                .contains(&t.to_string().as_str()))
                .count()
        );
    }

    #[test]
    fn seen_tiles_change_the_best_discard() {
        // After 3p 3p 7s 7s + a stray: dropping 9m or a pair-tile... use two pairs and a
        // two-sided pair: 3p 3p 7s 7s 5m 6m (8 tiles needs 2 melds): keep 2 melds.
        let hand = "3p 3p 7s 7s 5m 6m 9p 9p";
        let open = rows(hand, "");
        let open_best = open[0]["discard"].as_str().unwrap().to_string();
        // Hide every 4m and 7m: the 5m 6m wait is dead, so another discard must now lead.
        let seen = "4m 4m 4m 4m 7m 7m 7m 7m";
        let hidden = rows(hand, seen);
        assert_ne!(hidden[0]["discard"].as_str().unwrap(), open_best);
        assert!(useful(&hidden[0])
            .iter()
            .all(|(t, _)| t != "4m" && t != "7m"));
    }

    #[test]
    fn tie_on_steps_breaks_on_unseen_copies() {
        // 5 tiles: 3p 3p 7s 7s + 9m. Discard 3p or 7s both leave one pair-wait... both
        // are ready (shanten 0) after dropping 9m only; dropping a pair tile is worse.
        // Hide two 3p: dropping 7s... compare two ready discards instead:
        // 1p 2p 3p 4p 4p 5p 6p 7p (8 tiles, 2 melds), seen 4p x2 narrows 4p.
        let r = rows("1p 2p 3p 4p 4p 5p 6p 7p", "");
        let ready: Vec<_> = r.iter().filter(|x| x["shanten"] == 0).collect();
        assert!(ready.len() >= 2, "{r:?}");
        assert!(ready[0]["total"].as_i64() >= ready[1]["total"].as_i64());
        let seen = rows("1p 2p 3p 4p 4p 5p 6p 7p", "1p 1p 1p 4p 4p");
        let order: Vec<i64> = seen
            .iter()
            .filter(|x| x["shanten"] == 0)
            .map(|x| x["total"].as_i64().unwrap())
            .collect();
        assert!(order.windows(2).all(|w| w[0] >= w[1]), "{order:?}");
    }

    #[test]
    fn errors() {
        let e = |t: &str, s: &str| run(t, s)["error"].as_str().unwrap().to_string();
        assert_eq!(e("1m 2m 3m 4m", ""), "error.handSize");
        assert_eq!(e("1m 2m 3m F1 5m", ""), "error.badTile");
        assert_eq!(e("1m 1m 1m 1m 1m", ""), "error.tooMany");
        assert_eq!(e("1m 2m 3m 4m 5m", "5m 5m 5m 5m"), "error.tooMany");
        assert_eq!(
            serde_json::from_str::<Value>(&best_discards("nope", "")).unwrap()["error"],
            "error.badTile"
        );
    }
}
