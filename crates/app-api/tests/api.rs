//! End-to-end tests of the JSON calls, as the app will use them.

use mahjong_app_api::Api;
use serde_json::{json, Value};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Tests that check `budget_used_ms` hold this lock exclusively, so they never share the CPU
/// with other tests of this file; under contention a search overruns its wall-clock budget.
static CPU: RwLock<()> = RwLock::new(());

fn shared() -> (RwLockReadGuard<'static, ()>, Api) {
    (CPU.read().unwrap_or_else(|e| e.into_inner()), Api::new())
}

fn exclusive() -> (RwLockWriteGuard<'static, ()>, Api) {
    (CPU.write().unwrap_or_else(|e| e.into_inner()), Api::new())
}

const GRADES: [&str; 5] = ["best", "good", "inaccuracy", "mistake", "blunder"];

fn parse(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|e| panic!("bad JSON {e}: {s}"))
}

fn ok(s: &str) -> Value {
    let v = parse(s);
    assert!(v.get("error").is_none(), "unexpected error: {v}");
    assert_eq!(v["api"], 0, "every response carries api 0: {v}");
    v
}

fn error_code(s: &str) -> String {
    let v = parse(s);
    assert_eq!(v["api"], 0);
    v["error"]["code"]
        .as_str()
        .unwrap_or_else(|| panic!("expected an error, got {v}"))
        .to_string()
}

fn new_game(api: &mut Api, seed: u64) -> String {
    let v = ok(&api.new_game(&json!({ "seats": 4, "seed": seed, "human_seats": [0] }).to_string()));
    v["game_id"].as_str().unwrap().to_string()
}

/// Plays the hand to the end with `level` bots in every seat; returns the
/// number of actions and the last event list.
fn play_out(api: &mut Api, id: &str, level: &str) -> (usize, Vec<Value>) {
    let mut n = 0;
    let mut last = Vec::new();
    loop {
        let view = ok(&api.view(id, 0));
        if view["over"].as_bool().unwrap() {
            return (n, last);
        }
        let action = ok(&api.bot_action(id, level, 200));
        let r = ok(&api.apply(id, &action.to_string()));
        last = r["events"].as_array().unwrap().clone();
        n += 1;
        assert!(n < 1000, "the hand never ends");
    }
}

/// Wall-clock check for the two tests that time whole games: asserted in release
/// builds only, because a loaded machine runs a debug build too slowly to judge.
fn timing_ok(budget_ms: u64, used_ms: u64) -> bool {
    cfg!(debug_assertions) || within_budget(budget_ms, used_ms)
}
/// Whether a call kept to its budget: the contract allows 20% over, plus a
/// little fixed slack. A debug build is many times slower than the release
/// build the app ships, and the solver always runs at least one batch of
/// samples, so debug builds get more room.
fn within_budget(budget_ms: u64, used_ms: u64) -> bool {
    let allowed = budget_ms * 12 / 10 + 200;
    if cfg!(debug_assertions) {
        used_ms < allowed * 8
    } else {
        used_ms < allowed
    }
}

fn hand_over(events: &[Value]) -> &Value {
    let e = events.last().unwrap();
    assert_eq!(e["type"], "event.hand_over", "{events:?}");
    e
}

#[test]
fn meta_lists_variants_levels_and_keys() {
    let (_cpu, api) = shared();
    let m = ok(&api.meta());
    assert_eq!(m["game"], "mahjong");
    assert_eq!(m["seats"], json!([4, 4]));
    assert_eq!(m["bot_levels"][0], "random");
    assert_eq!(m["bot_levels"].as_array().unwrap().len(), 5);
    assert_eq!(m["variants"]["payout"][0], "half_spicy");
    assert_eq!(m["variants"]["flowers"], json!(["on", "off"]));
    assert_eq!(m["has_hint"], true);
    assert_eq!(m["has_review"], true);
    assert_eq!(m["tiles"]["5p"], "tile.5_dots");
    assert_eq!(m["tiles"].as_object().unwrap().len(), 42);
    let faan = m["faan"].as_array().unwrap();
    assert_eq!(faan.len(), 32);
    assert_eq!(faan[16]["key"], "faan.full_flush");
    assert_eq!(faan[16]["number"], 17);
    assert!(m["actions"]
        .as_array()
        .unwrap()
        .iter()
        .all(|a| a.as_str().unwrap().starts_with("action.")));
}

#[test]
fn four_random_bots_finish_hands() {
    let (_cpu, mut api) = shared();
    let mut wins = 0;
    let mut draws = 0;
    for seed in 1..=6 {
        let id = new_game(&mut api, seed);
        let (n, events) = play_out(&mut api, &id, "random");
        assert!(n > 20, "a hand takes many actions");
        let over = hand_over(&events);
        let deltas: Vec<i64> = over["result"]["deltas"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_i64().unwrap())
            .collect();
        assert_eq!(deltas.iter().sum::<i64>(), 0, "zero-sum: {deltas:?}");
        if over["result"]["wins"].as_array().unwrap().is_empty() {
            draws += 1;
            assert!(events.iter().any(|e| e["type"] == "event.exhaustive_draw"));
        } else {
            wins += 1;
            let w = &over["result"]["wins"][0];
            assert!(w["score"]["faan"].as_u64().unwrap() >= 3 || w["score"]["limit"] == true);
            assert!(w["score"]["patterns"][0]["key"]
                .as_str()
                .unwrap()
                .starts_with("faan."));
            assert!(events.iter().any(|e| e["type"] == "event.win"));
        }
        let view = ok(&api.view(&id, 2));
        assert_eq!(view["over"], true);
        assert_eq!(view["phase"], "over");
        assert_eq!(view["final_hands"].as_array().unwrap().len(), 4);
        assert_eq!(
            error_code(&api.bot_action(&id, "random", 10)),
            "error.game_over"
        );
        assert_eq!(
            error_code(&api.apply(&id, r#"{"type":"action.pass"}"#)),
            "error.game_over"
        );
    }
    assert_eq!(wins + draws, 6);
}

#[test]
fn hard_bots_win_a_hand() {
    let (_cpu, mut api) = shared();
    for seed in 20..40 {
        let id = new_game(&mut api, seed);
        let (_, events) = play_out(&mut api, &id, "hard");
        let over = hand_over(&events);
        let wins = over["result"]["wins"].as_array().unwrap();
        if wins.is_empty() {
            continue;
        }
        let w = &wins[0];
        let seat = w["seat"].as_u64().unwrap() as usize;
        let deltas = over["result"]["deltas"].as_array().unwrap();
        assert!(deltas[seat].as_i64().unwrap() > 0);
        assert_eq!(
            w["paid_by"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_i64().unwrap())
                .sum::<i64>(),
            deltas[seat].as_i64().unwrap()
        );
        assert!(w["points_per_share"].as_u64().unwrap() >= 8);
        let view = ok(&api.view(&id, (seat + 1) % 4));
        assert_eq!(view["result"]["wins"][0]["seat"], seat);
        assert!(
            view["history"]
                .as_array()
                .unwrap()
                .iter()
                .all(|t| !t["action"].is_null()),
            "nothing hidden once over"
        );
        return;
    }
    panic!("twenty hands of value bots and no win");
}

#[test]
fn four_easy_bots_finish_a_hand() {
    let (_cpu, mut api) = shared();
    let id = new_game(&mut api, 2);
    let (_, events) = play_out(&mut api, &id, "easy");
    hand_over(&events);
    let view = ok(&api.view(&id, 0));
    assert_eq!(view["result"]["deltas"].as_array().unwrap().len(), 4);
}

#[test]
fn rule_based_bots_and_expert_answer_timing_ok() {
    let (_cpu, mut api) = exclusive();
    let id = new_game(&mut api, 3);
    for level in ["easy", "medium", "hard"] {
        let a = ok(&api.bot_action(&id, level, 100));
        assert!(a["type"].as_str().unwrap().starts_with("action."));
        assert!(
            timing_ok(100, a["budget_used_ms"].as_u64().unwrap()),
            "{level}: {a}"
        );
    }
    let a = ok(&api.bot_action(&id, "expert", 300));
    assert!(a["type"].as_str().unwrap().starts_with("action."));
    assert!(
        timing_ok(300, a["budget_used_ms"].as_u64().unwrap()),
        "expert: {a}"
    );
    assert_eq!(
        error_code(&api.bot_action(&id, "grandmaster", 10)),
        "error.unknown_level"
    );
}

#[test]
fn opening_view_hides_other_hands_and_lists_legal_moves() {
    let (_cpu, mut api) = shared();
    let id = new_game(&mut api, 4);
    let view = ok(&api.view(&id, 0));
    // The dealer drew the 14th tile; the others hold 13.
    assert_eq!(view["hand"].as_array().unwrap().len(), 14);
    assert_eq!(view["concealed_counts"], json!([14, 13, 13, 13]));
    assert_eq!(view["current_player"], 0);
    assert_eq!(view["your_turn"], true);
    assert_eq!(view["phase"], "turn");
    assert!(view["final_hands"].is_null());
    assert!(view["result"].is_null());
    assert_eq!(
        view["wall_remaining"].as_u64().unwrap()
            + 53
            + view["bonus"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_array().unwrap().len() as u64)
                .sum::<u64>(),
        144
    );
    let legal = ok(&api.legal_actions(&id));
    assert_eq!(legal["seat"], 0);
    let actions = legal["actions"].as_array().unwrap();
    assert!(actions.iter().all(
        |a| a["type"] == "action.discard" || a["type"].as_str().unwrap().starts_with("action.")
    ));
    assert!(
        actions
            .iter()
            .filter(|a| a["type"] == "action.discard")
            .count()
            >= 5
    );
    assert_eq!(view["legal"], legal["actions"]);
    let other = ok(&api.view(&id, 1));
    assert_eq!(other["hand"].as_array().unwrap().len(), 13);
    assert_eq!(other["legal"], json!([]));
    assert!(other["last_drawn"].is_null());
    assert_eq!(error_code(&api.view(&id, 4)), "error.bad_seat");
    assert_eq!(error_code(&api.view("nope", 0)), "error.unknown_game");
}

#[test]
fn illegal_actions_are_rejected_with_a_code() {
    let (_cpu, mut api) = shared();
    let id = new_game(&mut api, 5);
    let view = ok(&api.view(&id, 0));
    let hand: Vec<&str> = view["hand"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    let foreign = ["1m", "5p", "9s", "E", "R"]
        .iter()
        .find(|t| !hand.contains(t))
        .unwrap();
    let a = json!({ "type": "action.discard", "tile": foreign }).to_string();
    assert_eq!(error_code(&api.apply(&id, &a)), "error.illegal_action");
    // Claim actions outside a claim window.
    assert_eq!(
        error_code(&api.apply(&id, r#"{"type":"action.pung"}"#)),
        "error.illegal_action"
    );
    assert_eq!(
        error_code(&api.apply(&id, r#"{"type":"action.pass"}"#)),
        "error.illegal_action"
    );
    assert_eq!(
        error_code(&api.apply(&id, r#"{"type":"action.discard","tile":"5x"}"#)),
        "error.bad_tile"
    );
    assert_eq!(
        error_code(&api.apply(&id, r#"{"type":"action.discard"}"#)),
        "error.bad_action"
    );
    assert_eq!(
        error_code(&api.apply(&id, r#"{"type":"action.fold"}"#)),
        "error.bad_action"
    );
    assert_eq!(
        error_code(&api.apply(&id, r#"{"code": 999}"#)),
        "error.bad_action"
    );
    assert_eq!(error_code(&api.apply(&id, "not json")), "error.bad_json");
    // The hand has not moved.
    assert_eq!(ok(&api.view(&id, 0))["history"], json!([]));
}

/// Plays random bots until a claim window where `wanted` is legal for the
/// declaring seat; returns that seat. Tries several seeds.
fn reach_claim(api: &mut Api, id: &mut String, wanted: &str) -> usize {
    for seed in 100..160 {
        *id = new_game(api, seed);
        loop {
            let legal = ok(&api.legal_actions(id));
            if legal["seat"].is_null() {
                break;
            }
            let actions = legal["actions"].as_array().unwrap();
            if legal["phase"] == "claim" && actions.iter().any(|a| a["type"] == wanted) {
                return legal["seat"].as_u64().unwrap() as usize;
            }
            let a = ok(&api.bot_action(id, "random", 10));
            ok(&api.apply(id, &a.to_string()));
        }
    }
    panic!("no claim window offering {wanted} in 60 seeds");
}

#[test]
fn claims_pung_chow_kong_pass_and_win() {
    let (_cpu, mut api) = shared();
    let mut id = String::new();
    for (wanted, kind) in [
        ("action.pung", "meld.pung"),
        ("action.chow", "meld.chow"),
        ("action.kong", "meld.kong"),
    ] {
        let seat = reach_claim(&mut api, &mut id, wanted);
        let before = ok(&api.view(&id, seat));
        assert_eq!(before["phase"], "claim");
        let claim = &before["claim"];
        assert!(claim["tile"].is_string());
        let from = claim["from"].as_u64().unwrap() as usize;
        assert_ne!(from, seat);
        let action = before["legal"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["type"] == wanted)
            .unwrap()
            .clone();
        if wanted == "action.chow" {
            assert_eq!(action["tiles"].as_array().unwrap().len(), 3);
            assert_eq!(seat, (from + 1) % 4, "only the next seat may chow");
        }
        // Everyone else in the window passes first, if they are ahead of us.
        loop {
            let legal = ok(&api.legal_actions(&id));
            if legal["seat"].as_u64().unwrap() as usize == seat {
                break;
            }
            ok(&api.apply(&id, r#"{"type":"action.pass"}"#));
        }
        let r = ok(&api.apply(&id, &action.to_string()));
        let events = r["events"].as_array().unwrap();
        assert_eq!(events[0]["type"], "event.declared");
        assert_eq!(events[0]["action"]["type"], wanted);
        // Let later declarers pass; then the claim resolves into a meld.
        let mut events = events.clone();
        while ok(&api.view(&id, seat))["phase"] == "claim" {
            let r = ok(&api.apply(&id, r#"{"type":"action.pass"}"#));
            events = r["events"].as_array().unwrap().clone();
        }
        let claim_event = events
            .iter()
            .find(|e| e["type"] == "event.claim")
            .unwrap_or_else(|| panic!("no claim event: {events:?}"));
        assert_eq!(claim_event["player"], seat);
        assert_eq!(claim_event["from"], from);
        assert_eq!(claim_event["meld"]["kind"], kind);
        let after = ok(&api.view(&id, seat));
        assert_eq!(
            after["melds"][seat].as_array().unwrap().last().unwrap()["kind"],
            kind
        );
        assert_eq!(after["current_player"], seat, "the claimer continues");
        assert_eq!(
            after["river"].as_array().unwrap().last().unwrap()["claimed"],
            true
        );
    }
    // A pass in a window, with the other seats' declarations hidden.
    let seat = reach_claim(&mut api, &mut id, "action.pung");
    let view = ok(&api.view(&id, seat));
    let history_len = view["history"].as_array().unwrap().len();
    let r = ok(&api.apply(&id, r#"{"type":"action.pass"}"#));
    assert_eq!(r["events"][0]["type"], "event.declared");
    assert_eq!(r["events"][0]["action"]["type"], "action.pass");
    assert_eq!(
        r["view"]["history"][history_len]["action"]["type"],
        "action.pass"
    );
    if ok(&api.view(&id, seat))["phase"] == "claim" {
        let other = ok(&api.view(&id, (seat + 1) % 4));
        assert!(
            other["history"][history_len]["action"].is_null(),
            "hidden until the window closes"
        );
    }
    // A win on a discard.
    let seat = reach_claim(&mut api, &mut id, "action.win");
    while ok(&api.legal_actions(&id))["seat"].as_u64().unwrap() as usize != seat {
        ok(&api.apply(&id, r#"{"type":"action.pass"}"#));
    }
    let mut events = ok(&api.apply(&id, r#"{"type":"action.win"}"#))["events"]
        .as_array()
        .unwrap()
        .clone();
    while ok(&api.view(&id, seat))["phase"] == "claim" {
        events = ok(&api.apply(&id, r#"{"type":"action.pass"}"#))["events"]
            .as_array()
            .unwrap()
            .clone();
    }
    let over = hand_over(&events);
    let win = over["result"]["wins"]
        .as_array()
        .unwrap()
        .iter()
        .find(|w| w["seat"] == seat)
        .unwrap();
    assert_eq!(win["self_drawn"], false);
    assert!(win["from"].is_number());
    assert!(
        win["paid_by"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_i64().unwrap())
            .sum::<i64>()
            > 0
    );
}

#[test]
fn snapshot_and_restore_give_identical_views() {
    let (_cpu, mut api) = shared();
    let id = new_game(&mut api, 7);
    for _ in 0..30 {
        let a = ok(&api.bot_action(&id, "easy", 50));
        ok(&api.apply(&id, &a.to_string()));
    }
    let snap = ok(&api.snapshot(&id));
    assert_eq!(snap["history"].as_array().unwrap().len(), 30);
    assert_eq!(snap["wall"].as_array().unwrap().len(), 144);
    assert_eq!(snap["game"], "mahjong");
    let restored = ok(&api.restore(&snap.to_string()));
    let id2 = restored["game_id"].as_str().unwrap();
    assert_ne!(id2, id);
    let strip = |mut v: Value| {
        let m = v.as_object_mut().unwrap();
        m.remove("game_id");
        m.remove("budget_used_ms");
        v
    };
    for seat in 0..4 {
        assert_eq!(
            strip(ok(&api.view(&id, seat))),
            strip(ok(&api.view(id2, seat)))
        );
    }
    assert_eq!(strip(ok(&api.snapshot(&id))), strip(ok(&api.snapshot(id2))));
    assert_eq!(ok(&api.legal_actions(&id)), ok(&api.legal_actions(id2)));
    // The two hands now continue identically: seeded bots draw the same moves.
    for level in ["random", "hard"] {
        let a1 = ok(&api.bot_action(&id, level, 10));
        let a2 = ok(&api.bot_action(id2, level, 10));
        assert_eq!(strip(a1), strip(a2));
    }
    // Drop and the id is gone.
    ok(&api.drop_game(&id));
    assert_eq!(error_code(&api.view(&id, 0)), "error.unknown_game");
    assert_eq!(api.len(), 1);
}

#[test]
fn restore_rejects_bad_positions() {
    let (_cpu, mut api) = shared();
    assert_eq!(error_code(&api.restore(r#"{"seed": 1}"#)), "error.bad_json");
    let bad_game = json!({ "game": "big2", "seed": 1, "wall": [] });
    assert_eq!(
        error_code(&api.restore(&bad_game.to_string())),
        "error.bad_position"
    );
    let short_wall = json!({ "game": "mahjong", "seed": 1, "wall": ["1m", "2m"] });
    assert_eq!(
        error_code(&api.restore(&short_wall.to_string())),
        "error.bad_position"
    );
    let id = new_game(&mut api, 8);
    let mut snap = ok(&api.snapshot(&id));
    snap["history"] = json!([{ "player": 2, "action": { "type": "action.pass" } }]);
    assert_eq!(
        error_code(&api.restore(&snap.to_string())),
        "error.bad_position"
    );
    // A different rule set with the same wall still restores (flowers stay in play).
    let mut snap = ok(&api.snapshot(&id));
    snap["rules"]["payout"] = json!("full_spicy");
    let v = ok(&api.restore(&snap.to_string()));
    assert_eq!(
        ok(&api.view(v["game_id"].as_str().unwrap(), 0))["rules"]["payout"],
        "full_spicy"
    );
}

#[test]
fn hint_gives_moves_with_numbers_reasons_and_study_extras() {
    let (_cpu, mut api) = exclusive();
    let id = new_game(&mut api, 9);
    assert_eq!(error_code(&api.hint(&id, 1, 100)), "error.not_your_turn");
    let h = ok(&api.hint(&id, 0, 600));
    assert_eq!(h["phase"], "turn");
    let moves = h["moves"].as_array().unwrap();
    assert!(!moves.is_empty() && moves.len() <= 3);
    for m in moves {
        assert!(m["action"]["type"].as_str().unwrap().starts_with("action."));
        assert!(m["score"].is_number());
        assert!(m["win_rate"].is_number());
        assert!(m["numbers"]["deal_in_rate"].is_number());
        assert!(m["numbers"]["samples"].as_u64().unwrap() > 0);
        let reasons = m["reasons"].as_array().unwrap();
        assert!(!reasons.is_empty(), "every hint move has a reason: {m}");
        assert!(reasons
            .iter()
            .all(|r| r["key"].as_str().unwrap().starts_with("reason.")));
    }
    let keys: Vec<&str> = moves[0]["reasons"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["key"].as_str().unwrap())
        .collect();
    assert!(
        keys.iter().any(|k| [
            "reason.keeps_faan_path",
            "reason.best_shanten",
            "reason.breaks_faan_path",
            "reason.no_faan_path",
            "reason.loses_shanten",
            "reason.ready_for_faan"
        ]
        .contains(k)),
        "a discard explains its effect on the hand: {keys:?}"
    );
    assert!(
        within_budget(600, h["budget_used_ms"].as_u64().unwrap()),
        "{}",
        h["budget_used_ms"]
    );
    assert!(h["samples"].as_u64().unwrap() > 0);
    assert!(h["baseline"]["type"].is_string());
    // Study extras: progress, danger and the discard table.
    assert!(h["progress"]["shanten"].is_number());
    assert!(h["progress"].get("faan_shanten").is_some());
    let danger = &h["danger"];
    assert_eq!(danger["threat"].as_array().unwrap().len(), 4);
    assert_eq!(danger["threat"][0], 0.0);
    assert!(!danger["tiles"].as_object().unwrap().is_empty());
    let discards = h["discards"].as_array().unwrap();
    assert!(discards.len() >= 5);
    for d in discards {
        assert!(d["tile"].is_string());
        assert!(d["shanten"].is_number());
        assert!(d["waits"].is_number());
        assert!(d["danger"].is_number());
    }
    // A hint during a claim window has no discard table.
    let mut cid = String::new();
    let seat = reach_claim(&mut api, &mut cid, "action.pung");
    let h = ok(&api.hint(&cid, seat, 300));
    assert_eq!(h["phase"], "claim");
    assert_eq!(h["discards"], json!([]));
    let kinds: Vec<&str> = h["moves"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["action"]["type"].as_str().unwrap())
        .collect();
    assert!(
        kinds.contains(&"action.pung") || kinds.contains(&"action.pass"),
        "{kinds:?}"
    );
}

#[test]
fn review_has_one_item_per_decision_of_the_seat() {
    let (_cpu, mut api) = exclusive();
    let id = new_game(&mut api, 10);
    play_out(&mut api, &id, "easy");
    let seat = 1;
    let view = ok(&api.view(&id, seat));
    let history = view["history"].as_array().unwrap();
    let mine = history.iter().filter(|t| t["player"] == seat).count();
    assert!(mine > 0);
    let r = ok(&api.review(&id, seat, 4000));
    let items = r["items"].as_array().unwrap();
    assert_eq!(items.len(), mine);
    assert!(
        timing_ok(4000, r["budget_used_ms"].as_u64().unwrap()),
        "{}",
        r["budget_used_ms"]
    );
    let mut searched = 0;
    for item in items {
        let turn = item["turn"].as_u64().unwrap() as usize;
        assert_eq!(history[turn]["player"], seat);
        assert_eq!(item["played"], history[turn]["action"]);
        assert!(GRADES.contains(&item["grade"].as_str().unwrap()), "{item}");
        assert!(item["loss"].as_f64().unwrap() >= 0.0);
        assert!(!item["reasons"].as_array().unwrap().is_empty());
        if item["numbers"].as_object().unwrap().is_empty() {
            assert_eq!(item["reasons"][0]["key"], "reason.forced");
            assert_eq!(item["grade"], "best");
        } else {
            searched += 1;
            assert!(item["numbers"]["samples"].as_u64().unwrap() > 0);
            assert!(item["best"]["type"].is_string());
        }
    }
    assert!(searched > 0, "some decisions had a choice");
    // A review before the hand ends is allowed too, and covers the decisions so far.
    let id2 = new_game(&mut api, 11);
    let a = ok(&api.bot_action(&id2, "easy", 10));
    ok(&api.apply(&id2, &a.to_string()));
    let early = ok(&api.review(&id2, 0, 300));
    assert_eq!(early["items"].as_array().unwrap().len(), 1);
    assert_eq!(ok(&api.review(&id2, 2, 10))["items"], json!([]));
}

#[test]
fn grade_scores_an_action_against_the_best() {
    let (_cpu, mut api) = shared();
    let id = new_game(&mut api, 12);
    let snap = ok(&api.snapshot(&id)).to_string();
    let legal = ok(&api.legal_actions(&id));
    let discards: Vec<&Value> = legal["actions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|a| a["type"] == "action.discard")
        .collect();
    // The dealer's lowest tile is often not the best discard.
    let g = ok(&api.grade(&snap, &discards[0].to_string(), 500));
    assert!(GRADES.contains(&g["grade"].as_str().unwrap()), "{g}");
    assert!(g["best"]["type"].as_str().unwrap().starts_with("action."));
    assert!(!g["reasons"].as_array().unwrap().is_empty());
    assert!(g["numbers"]["played_score"].is_number());
    assert_eq!(
        error_code(&api.grade(&snap, r#"{"type":"action.pung"}"#, 50)),
        "error.illegal_action"
    );
    // Grading a position does not register a game.
    assert_eq!(api.len(), 1);
}

#[test]
fn variants_round_trip_through_config_and_snapshot() {
    let (_cpu, mut api) = shared();
    let cfg = json!({
        "seats": 4, "seed": 13,
        "variants": { "flowers": "off", "min_faan": "1", "payout": "full_spicy", "multiple_winners": "head_bump",
                      "twelve_tile_liability": false, "prevailing_wind": "south" }
    });
    let id = ok(&api.new_game(&cfg.to_string()))["game_id"]
        .as_str()
        .unwrap()
        .to_string();
    let view = ok(&api.view(&id, 0));
    assert_eq!(view["rules"]["flowers"], "off");
    assert_eq!(view["rules"]["min_faan"], 1);
    assert_eq!(view["rules"]["faan_cap"], 10);
    assert_eq!(view["rules"]["payout"], "full_spicy");
    assert_eq!(view["rules"]["multiple_winners"], "head_bump");
    assert_eq!(view["rules"]["twelve_tile_liability"], "off");
    assert_eq!(view["rules"]["prevailing_wind"], "south");
    assert_eq!(
        view["wall_remaining"].as_u64().unwrap() + 53,
        136,
        "no flowers: 136 tiles"
    );
    let snap = ok(&api.snapshot(&id));
    assert_eq!(snap["wall"].as_array().unwrap().len(), 136);
    let id2 = ok(&api.restore(&snap.to_string()))["game_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(ok(&api.view(&id2, 0))["rules"], view["rules"]);
    for bad in [
        json!({ "flowers": "maybe" }),
        json!({ "min_faan": 14 }),
        json!({ "min_faan": 5, "faan_cap": 4 }),
        json!({ "payout": "spicy" }),
        json!({ "prevailing_wind": "up" }),
        json!({ "jokers": "two" }),
    ] {
        let cfg = json!({ "seats": 4, "seed": 1, "variants": bad });
        assert_eq!(
            error_code(&api.new_game(&cfg.to_string())),
            "error.unknown_variant",
            "{bad}"
        );
    }
    assert_eq!(
        error_code(&api.new_game(r#"{"seats": 3, "seed": 1}"#)),
        "error.bad_config"
    );
    assert_eq!(
        error_code(&api.new_game(r#"{"seats": 4, "seed": 1, "human_seats": [4]}"#)),
        "error.bad_seat"
    );
}

/// Timings for the README: `cargo test -p mahjong-app-api --release -- --ignored --nocapture timing`.
#[test]
#[ignore]
fn timing_of_hint_bot_and_review() {
    let (_cpu, mut api) = exclusive();
    let id = new_game(&mut api, 100);
    for budget in [300u32, 1000, 3000] {
        let h = ok(&api.hint(&id, 0, budget));
        println!(
            "hint budget {budget} ms: {} ms, {} samples",
            h["budget_used_ms"], h["samples"]
        );
    }
    let h = ok(&api.hint_with_samples(&id, 0, 0, 400));
    println!("hint 400 samples, no budget: {} ms", h["budget_used_ms"]);
    for level in ["hard", "expert"] {
        let a = ok(&api.bot_action(&id, level, 0));
        println!("{level} bot, no budget: {} ms", a["budget_used_ms"]);
    }
    let a = ok(&api.bot_action(&id, "expert", 500));
    println!("expert bot, 500 ms budget: {} ms", a["budget_used_ms"]);
    play_out(&mut api, &id, "hard");
    for budget in [3000u32, 0] {
        let r = ok(&api.review(&id, 0, budget));
        println!(
            "review of seat 0, {} items, budget {budget} ms: {} ms",
            r["items"].as_array().unwrap().len(),
            r["budget_used_ms"]
        );
    }
}

/// Plays up to `max_moves` of a hand from `seed` with `level` bots at `budget_ms`; returns every
/// response (deal view, each action, each result) so two runs compare byte for byte.
fn transcript(seed: u64, level: &str, budget_ms: u32, max_moves: usize) -> Vec<String> {
    let mut api = Api::new();
    let id = new_game(&mut api, seed);
    let mut out = Vec::new();
    for seat in 0..4 {
        out.push(api.view(&id, seat));
    }
    for _ in 0..max_moves {
        if ok(&api.view(&id, 0))["over"].as_bool().unwrap() {
            break;
        }
        let action = api.bot_action(&id, level, budget_ms);
        let mut v = ok(&action);
        // The wall-clock time the call took is the one field that may differ.
        v.as_object_mut().unwrap().remove("budget_used_ms");
        let applied = api.apply(&id, &v.to_string());
        out.push(v.to_string());
        out.push(applied);
    }
    out.push(api.snapshot(&id));
    out
}

/// D28 (server replay): the deal depends only on the seed, and a bot's move depends only
/// on the position and the level. Rule-based levels ignore the budget; the expert is
/// deterministic at `budget_ms = 0` (a fixed number of samples, no clock).
#[test]
fn same_seed_replays_identically_through_the_json_api() {
    let (_cpu, _) = shared();
    // The expert searches, so only its first moves are compared (a full hand is slow in debug).
    for (level, budget, moves) in [
        ("random", 200, 1000),
        ("easy", 200, 1000),
        ("hard", 200, 1000),
        ("expert", 0, 3),
    ] {
        let a = transcript(20261030, level, budget, moves);
        let b = transcript(20261030, level, budget, moves);
        assert_eq!(a, b, "level {level} replayed differently");
        assert!(a.len() > 10, "level {level} played no hand");
    }
    let other = transcript(7, "random", 200, 1000);
    assert_ne!(
        transcript(20261030, "random", 200, 1000)[0],
        other[0],
        "different seeds deal different hands"
    );
}
