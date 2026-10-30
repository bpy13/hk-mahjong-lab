//! The `ev` bot's three constructed positions (`docs/ev-bot.md` §5–§6) and a
//! stub belief, shared by `ev_bot.rs` and `reasons.rs`.
#![allow(dead_code)]

use mahjong_hk::{Meld, MeldKind, PositionPhase, PositionSpec, RuleSet, State};
use mahjong_search::ev::FAAN_BUCKETS;
use mahjong_search::{Belief, Beliefs, EvBot, EvConfig, SeatBelief};
use mahjong_tiles::{parse_tiles, Tile, TileCounts, NUM_KINDS};
use std::sync::Arc;

pub fn t(s: &str) -> Tile {
    s.parse().unwrap()
}

pub fn counts(s: &str) -> TileCounts {
    TileCounts::from_tiles(&parse_tiles(s).unwrap())
}

/// A belief stub that returns fixed numbers.
#[derive(Debug)]
pub struct Stub(Beliefs);

impl Belief for Stub {
    fn estimate(&self, _state: &State, _me: usize) -> Beliefs {
        self.0.clone()
    }
}

/// One opponent's stubbed belief: `tenpai`, the same `wait` on `waits` (or on
/// every tile when `waits` is empty), `sd`, and all weight on `faan`.
pub fn seat(tenpai: f64, wait: f64, waits: &[&str], sd: f64, faan: u8) -> SeatBelief {
    let mut b = SeatBelief::quiet();
    b.tenpai = tenpai;
    if waits.is_empty() {
        b.wait = [wait; NUM_KINDS];
    } else {
        for w in waits {
            b.wait[t(w).index() as usize] = wait;
        }
    }
    b.self_draw = sd;
    b.faan = [0.0; FAAN_BUCKETS];
    b.faan[(faan - 3) as usize] = 1.0;
    b
}

/// The bot with the doc's defaults (§4), including the `Pay(me)` floor (§9 Q4).
pub fn bot(beliefs: Beliefs) -> EvBot {
    EvBot::new(EvConfig::default(), Arc::new(Stub(beliefs)))
}

/// The same bot without the `Pay(me)` floor (`ev:floor=none`).
///
/// With the floor, `faan_shanten` finds a full flush with all chows (清一色 7 +
/// 平糊 1 + 門前清 1 + 無花 1 = 10 faan guaranteed) at faan-shanten 3 in
/// position 1 and 6 in position 2, so `Pay(me) = points(10) = 128` against a
/// table mean. With the final table (Task 9, 7e18101), in position 1 that makes
/// discarding `R` (chasing the flush) beat `value`'s `3p`; in position 2 it
/// raises the safe tiles' offence from about 2 to 16.5 points, so the bot no
/// longer deals in. The doc's expected answers hold without the floor; which
/// one Brian wants is an open question (docs/ev-bot.md §10).
pub fn bot_without_floor(beliefs: Beliefs) -> EvBot {
    let config = EvConfig {
        pay_floor: false,
        ..EvConfig::default()
    };
    EvBot::new(config, Arc::new(Stub(beliefs)))
}

/// Builds a turn position for seat 0 from the hands and melds given; seats
/// whose hand is `None` get 13 tiles less 3 per meld from what is left, the
/// eight bonus tiles go to seats 1–3, and the river takes the rest except
/// `wall` tiles. Kinds in `keep` stay unseen as long as possible.
pub struct Setup<'a> {
    pub hands: [Option<&'a str>; 4],
    pub melds: [Vec<Meld>; 4],
    /// `(discarder, tile)` for each claimed discard, matching the exposed melds.
    pub claimed: Vec<(usize, &'a str)>,
    pub last_drawn: &'a str,
    pub wall: usize,
    pub liable: [Option<usize>; 4],
    pub keep: &'a str,
}

pub fn build(s: Setup) -> State {
    let mut used = [0u8; NUM_KINDS];
    let mut hands = [TileCounts::new(); 4];
    for (p, hand) in hands.iter_mut().enumerate() {
        if let Some(h) = s.hands[p] {
            *hand = counts(h);
        }
        for m in &s.melds[p] {
            for x in m.tiles() {
                used[x.index() as usize] += 1;
            }
        }
    }
    for h in s.hands.iter().flatten() {
        for x in parse_tiles(h).unwrap() {
            used[x.index() as usize] += 1;
        }
    }
    let keep: Vec<Tile> = parse_tiles(s.keep).unwrap();
    // Kinds in fill order: the ones not kept first.
    let mut order: Vec<Tile> = Tile::all_kinds().filter(|k| !keep.contains(k)).collect();
    order.extend(Tile::all_kinds().filter(|k| keep.contains(k)));
    let take = |n: usize, used: &mut [u8; NUM_KINDS]| -> Vec<Tile> {
        let mut out = Vec::new();
        for &k in &order {
            while out.len() < n && used[k.index() as usize] < 4 {
                used[k.index() as usize] += 1;
                out.push(k);
            }
        }
        assert_eq!(out.len(), n, "not enough tiles left");
        out
    };
    for (p, hand) in hands.iter_mut().enumerate() {
        if s.hands[p].is_none() {
            let n = 13 - 3 * s.melds[p].len();
            *hand = TileCounts::from_tiles(&take(n, &mut used));
        }
    }
    let in_play: usize = used.iter().map(|&c| c as usize).sum();
    let river_len = 136 - in_play - s.wall;
    let mut river: Vec<(usize, Tile, bool)> = s
        .claimed
        .iter()
        .map(|&(from, tile)| (from, t(tile), true))
        .collect();
    for (i, x) in take(river_len, &mut used).into_iter().enumerate() {
        river.push(((i + 1) % 4, x, false));
    }
    let bonus = [
        vec![],
        parse_tiles("F1 F2 F3").unwrap(),
        parse_tiles("F4 S1 S2").unwrap(),
        parse_tiles("S3 S4").unwrap(),
    ];
    let spec = PositionSpec {
        hands,
        melds: s.melds,
        bonus,
        river,
        wall_remaining: s.wall,
        to_move: 0,
        phase: PositionPhase::Turn,
        last_drawn: Some(t(s.last_drawn)),
        kong_chain: 0,
        liable: s.liable,
    };
    State::from_position(RuleSet::default(), &spec).expect("a valid position")
}

/// §5 example 1: nobody threatens.
pub fn position_1() -> (State, Beliefs) {
    let state = build(Setup {
        hands: [Some("12345678m 8m RR 9p E 3p"), None, None, None],
        melds: Default::default(),
        claimed: vec![],
        last_drawn: "3p",
        wall: 32,
        liable: [None; 4],
        keep: "123456789m R E 3p 9p",
    });
    let quiet = seat(0.10, 0.05, &[], 0.04, 3);
    let beliefs = Beliefs {
        seats: [SeatBelief::quiet(), quiet.clone(), quiet.clone(), quiet],
    };
    (state, beliefs)
}

/// §5 examples 2 and 3: A (seat 2) threatens a 96-point self-draw, B (seat 1)
/// waits on 2m for a cheap hand. With `liable`, B's fourth meld (`EEE`) was
/// claimed from me and I am liable for B.
pub fn position_2(liable: bool) -> (State, Beliefs) {
    let mut melds: [Vec<Meld>; 4] = Default::default();
    melds[2] = vec![
        Meld::exposed(MeldKind::Pung, t("1s"), 1),
        Meld::exposed(MeldKind::Pung, t("9s"), 3),
        Meld::exposed(MeldKind::Pung, t("R"), 3),
    ];
    melds[1] = vec![
        Meld::exposed(MeldKind::Chow, t("3m"), 0),
        Meld::exposed(MeldKind::Chow, t("5p"), 0),
        Meld::exposed(MeldKind::Chow, t("2s"), 0),
    ];
    let mut claimed = vec![
        (1, "1s"),
        (3, "9s"),
        (3, "R"),
        (0, "4m"),
        (0, "6p"),
        (0, "3s"),
    ];
    let b_hand = if liable {
        melds[1].push(Meld::exposed(MeldKind::Pung, t("E"), 0));
        claimed.push((0, "E"));
        "2m"
    } else {
        "EEE 2m"
    };
    let state = build(Setup {
        hands: [
            Some("1m 2m 4m 6m 9m 2p 3p 5p 7p 8p S W N Wh"),
            Some(b_hand),
            Some("55s 66s"),
            None,
        ],
        melds,
        claimed,
        last_drawn: "Wh",
        wall: 30,
        liable: [None, liable.then_some(0), None, None],
        keep: "2m 5s 6s E R",
    });
    let a = seat(0.90, 1.0, &["5s", "6s"], 0.30, 8);
    let beliefs = Beliefs {
        seats: [
            SeatBelief::quiet(),
            seat(0.95, 1.0, &["2m"], 0.05, 4),
            a,
            seat(0.0, 0.0, &[], 0.0, 3),
        ],
    };
    (state, beliefs)
}
