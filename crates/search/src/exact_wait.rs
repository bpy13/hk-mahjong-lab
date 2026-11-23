//! Exact waits by counting, for an opponent with few concealed tiles (plan
//! `docs/plans/2026-11-16-self-play-network.md`, Task 13; Brian, 2026-11-19: "it
//! should be possible to reach 100%. let's say full flush all pung 12 exposed, you
//! can count the available tiles and sometimes it just boil down to a single
//! option").
//!
//! Seen from seat `me`, opponent `o` holding 1 or 4 concealed tiles can hold only
//! few hands. A concealed hand `H` is **possible** when
//!
//! * (a) its tiles are among those `me` cannot see ([`State::unseen`]: my hand,
//!   every river tile and every meld counted out; a concealed kong shows its tile,
//!   rules §5.1);
//! * (b) with `o`'s melds it is ready;
//! * (c) the win on tile `k` by discard reaches the faan minimum under the
//!   engine's scoring ([`score_win`], with `o`'s seat, flowers and the prevailing
//!   wind);
//! * (d) `k` is not a tile `o` was offered since its turn last started (a discard
//!   by another seat, or a tile added to a kong it could rob): under the passed-win
//!   rule 過水 (rules §4.4, `RuleSet::passed_win_lock`, default on) a hand that
//!   won on that tile then either won (the hand would be over) or passed and is
//!   locked on that kind until its next draw or claim. The lock covers the tile
//!   kind from every player, and only a win on a discard or a robbed kong; a
//!   self-draw is not refused.
//!
//! A hand is **ready now** when at least one tile passes (c) and (d). Each ready
//! hand weighs the number of ways to hold it from the unseen copies,
//! `Π_t C(unseen_t, H_t)`. [`ExactWait::win`] is, for each kind `k`, the share of
//! the ready weight whose hand wins on `k`; [`ExactWait::points`] the
//! weight-averaged points per share of that win.
//!
//! * One concealed tile: the opponent is ready whenever any hand is possible, so
//!   `win[k]` is the danger itself: one kind left gives exactly 1.0, a kind that
//!   fails exactly 0.0.
//! * Four concealed tiles: [`crate::Danger`] keeps the table's chance that `o`
//!   is ready and spreads it by `win`; a tile no possible hand wins on gets 0.0.
//! * Any other count: `None` (the tables as before).

use mahjong_hk::{score_win, Event, State, WinContext};
use mahjong_tiles::{Tile, TileCounts, NUM_KINDS};
use std::collections::HashMap;
use std::sync::OnceLock;

/// What counting says about one opponent.
#[derive(Clone, Debug, PartialEq)]
pub struct ExactWait {
    /// Concealed tiles the opponent holds (1 or 4).
    pub concealed: usize,
    /// Distinct concealed hands that are ready now.
    pub hands: usize,
    /// For each kind, the share of the ready hands' weight that wins on it by
    /// discard now (0 when no hand is ready).
    pub win: [f64; NUM_KINDS],
    /// For each kind, the weight-averaged points per share of that win (0 where
    /// `win` is 0).
    pub points: [f64; NUM_KINDS],
}

impl ExactWait {
    /// Whether any concealed hand is ready now.
    pub fn any_ready(&self) -> bool {
        self.hands > 0
    }
}

/// One concealed shape: a hand and the kinds that complete it.
struct Shape {
    hand: TileCounts,
    wins: Vec<u8>,
}

/// Every concealed hand of `n` tiles (1 or 4) that one tile completes into a pair
/// plus `(n - 1) / 3` sets, with the completing kinds. Built once.
fn shapes(n: usize) -> &'static [Shape] {
    static ONE: OnceLock<Vec<Shape>> = OnceLock::new();
    static FOUR: OnceLock<Vec<Shape>> = OnceLock::new();
    let build = |sets: usize| -> Vec<Shape> {
        let mut forms: Vec<TileCounts> = Vec::new();
        for p in 0..NUM_KINDS {
            let mut c = TileCounts::new();
            c.0[p] += 2;
            if sets == 0 {
                forms.push(c);
                continue;
            }
            for s in 0..NUM_KINDS {
                let mut pung = c;
                pung.0[s] += 3;
                forms.push(pung);
                if s < 27 && s % 9 <= 6 {
                    let mut chow = c;
                    for d in 0..3 {
                        chow.0[s + d] += 1;
                    }
                    forms.push(chow);
                }
            }
        }
        let mut by_hand: HashMap<TileCounts, Vec<u8>> = HashMap::new();
        for c in forms.into_iter().filter(|c| c.0.iter().all(|&x| x <= 4)) {
            for k in 0..NUM_KINDS {
                if c.0[k] > 0 {
                    let mut h = c;
                    h.0[k] -= 1;
                    let wins = by_hand.entry(h).or_default();
                    if !wins.contains(&(k as u8)) {
                        wins.push(k as u8);
                    }
                }
            }
        }
        let mut out: Vec<Shape> = by_hand
            .into_iter()
            .map(|(hand, mut wins)| {
                wins.sort();
                Shape { hand, wins }
            })
            .collect();
        out.sort_by_key(|s| s.hand.0);
        out
    };
    match n {
        1 => ONE.get_or_init(|| build(0)),
        4 => FOUR.get_or_init(|| build(1)),
        _ => &[],
    }
}

fn choose(n: u8, k: u8) -> f64 {
    match (n, k) {
        (_, 0) => 1.0,
        (n, k) if k > n => 0.0,
        (n, 1) => n as f64,
        (n, 2) => (n as f64) * (n as f64 - 1.0) / 2.0,
        (n, 3) => (n as f64) * (n as f64 - 1.0) * (n as f64 - 2.0) / 6.0,
        _ => 1.0, // C(4, 4)
    }
}

/// The kinds `o` may not win on by discard now under the passed-win rule (rules
/// §4.4): every tile another seat discarded, or added to a kong, since `o`'s turn
/// last started (its draw or its own claim; while another seat is on turn that is
/// the same as since `o`'s own last action). Empty with the rule off.
pub fn refused(state: &State, o: usize) -> [bool; NUM_KINDS] {
    let mut out = [false; NUM_KINDS];
    if !state.rules().passed_win_lock {
        return out;
    }
    for e in state.events().iter().rev() {
        match *e {
            // Any action of `o`'s own ends the walk: its turn started at or
            // before it. (Positions built by `State::from_position` record no
            // draws; `o`'s last discard still bounds the window there.)
            Event::Draw { seat, .. }
            | Event::Claim { seat, .. }
            | Event::Discard { seat, .. }
            | Event::ConcealedKong { seat, .. }
            | Event::AddedKong { seat, .. }
                if seat as usize == o =>
            {
                break
            }
            Event::Discard { tile, .. } | Event::AddedKong { tile, .. } => {
                out[tile.index() as usize] = true;
            }
            _ => {}
        }
    }
    out
}

/// Exact waits of opponent `o` seen from `me`, when `o` holds 1 or 4 concealed
/// tiles; `None` otherwise. Reads only what `me` may see.
pub fn exact_wait(state: &State, me: usize, o: usize) -> Option<ExactWait> {
    let concealed = state.concealed_count(o);
    if o == me || !matches!(concealed, 1 | 4) {
        return None;
    }
    let unseen = state.unseen(me);
    let rules = state.rules();
    let melds = state.melds(o);
    let lock = refused(state, o);
    let ctx = |k: Tile| WinContext {
        seat: o,
        win_tile: Some(k),
        self_drawn: false,
        bonus: state.bonus(o),
        last_tile: false,
        kong_chain: 0,
        robbing_kong: false,
        heavenly: false,
        earthly: false,
        human: false,
    };
    let mut total = 0.0f64;
    let mut hands = 0usize;
    let mut weight = [0.0f64; NUM_KINDS];
    let mut paid = [0.0f64; NUM_KINDS];
    for shape in shapes(concealed) {
        if (0..NUM_KINDS).any(|t| shape.hand.0[t] > unseen.0[t]) {
            continue;
        }
        let w: f64 = (0..NUM_KINDS)
            .map(|t| choose(unseen.0[t], shape.hand.0[t]))
            .product();
        let mut ready = false;
        for &k in &shape.wins {
            let ki = k as usize;
            if lock[ki] {
                continue;
            }
            let tile = Tile::from_index(k);
            let mut full = shape.hand;
            full.add(tile);
            let Some(score) = score_win(&full, melds, &ctx(tile), rules) else {
                continue;
            };
            if !score.meets_minimum(rules) {
                continue;
            }
            ready = true;
            let faan = score.total.min(rules.faan_cap as u16) as u8;
            weight[ki] += w;
            paid[ki] += w * rules.points(faan) as f64;
        }
        if ready {
            total += w;
            hands += 1;
        }
    }
    let mut win = [0.0; NUM_KINDS];
    let mut points = [0.0; NUM_KINDS];
    if total > 0.0 {
        for k in 0..NUM_KINDS {
            if weight[k] > 0.0 {
                win[k] = (weight[k] / total).min(1.0);
                points[k] = paid[k] / weight[k];
            }
        }
    }
    Some(ExactWait {
        concealed,
        hands,
        win,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::danger::{with_exact, Danger};
    use crate::ev::HeuristicBelief;
    use mahjong_hk::{Meld, MeldKind, PositionPhase, PositionSpec, RuleSet};
    use mahjong_tiles::parse_tiles;

    fn t(s: &str) -> Tile {
        s.parse().unwrap()
    }

    fn counts(s: &str) -> TileCounts {
        TileCounts::from_tiles(&parse_tiles(s).unwrap())
    }

    fn pung(s: &str, from: usize) -> Meld {
        Meld::exposed(MeldKind::Pung, t(s), from)
    }

    /// I (seat 0) am on turn with `mine` (14 tiles) and an empty wall; seats 1-3
    /// hold `hands` and `melds`. Every other tile lies in the river, the last
    /// entry by seat 2 (so nothing is refused to seat 2), then `tail`. All eight
    /// bonus tiles are shown.
    fn position(
        mine: &str,
        hands: [&str; 3],
        melds: [Vec<Meld>; 3],
        tail: &[(usize, &str)],
    ) -> State {
        let hands = [
            counts(mine),
            counts(hands[0]),
            counts(hands[1]),
            counts(hands[2]),
        ];
        let melds = [vec![], melds[0].clone(), melds[1].clone(), melds[2].clone()];
        let mut used = [0u8; NUM_KINDS];
        for h in &hands {
            for (u, c) in used.iter_mut().zip(h.0.iter()) {
                *u += c;
            }
        }
        let mut river: Vec<(usize, Tile, bool)> = Vec::new();
        for list in &melds {
            for m in list {
                for x in m.tiles() {
                    used[x.index() as usize] += 1;
                }
                river.push((m.claimed_from.unwrap() as usize, m.tile, true));
            }
        }
        for &(_, x) in tail {
            used[t(x).index() as usize] += 1;
        }
        let rest: Vec<Tile> = Tile::all_kinds()
            .flat_map(|x| std::iter::repeat_n(x, 4 - used[x.index() as usize] as usize))
            .collect();
        let n = rest.len();
        for (k, &x) in rest.iter().enumerate() {
            // The last one (k = n - 1) is seat 2's.
            river.push(((k + 4 * n + 3 - n) % 4, x, false));
        }
        for &(seat, x) in tail {
            river.push((seat, t(x), false));
        }
        let spec = PositionSpec {
            hands,
            melds,
            bonus: [
                vec![],
                parse_tiles("F1 F2 F3").unwrap(),
                parse_tiles("F4 S1 S2").unwrap(),
                parse_tiles("S3 S4").unwrap(),
            ],
            river,
            wall_remaining: 0,
            to_move: 0,
            phase: PositionPhase::Turn,
            last_drawn: Some(parse_tiles(mine).unwrap()[0]),
            kong_chain: 0,
            liable: [None; 4],
            passed_wins: Default::default(),
            river_from_hand: Vec::new(),
        };
        State::from_position(RuleSet::default(), &spec).expect("a valid position")
    }

    /// Seat 2 shows four pungs of bamboos; seats 1 and 3 four pungs of
    /// characters. Each holds one tile.
    fn four_pungs(singles: [&str; 3], mine: &str, tail: &[(usize, &str)]) -> State {
        position(
            mine,
            singles,
            [
                vec![pung("1m", 0), pung("2m", 2), pung("3m", 3), pung("4m", 0)],
                vec![pung("1s", 1), pung("3s", 3), pung("5s", 1), pung("7s", 0)],
                vec![pung("5m", 0), pung("6m", 1), pung("7m", 2), pung("8m", 0)],
            ],
            tail,
        )
    }

    /// The switch off gives what the builders gave before (no split, same tiles).
    fn check_switch_off_is_unchanged(s: &State) {
        let off = Danger::from_belief(s, 0, &HeuristicBelief);
        assert!(off.split(t("1m")).is_none());
        assert!(!(0..4).any(|o| off.is_exact(o)));
        let again = with_exact(false, || Danger::from_belief(s, 0, &HeuristicBelief));
        for x in Tile::all_kinds() {
            assert_eq!(off.tile(x), again.tile(x));
        }
    }

    #[test]
    fn shapes_cover_every_ready_hand() {
        // One tile: 34 single tiles, each waiting on itself.
        assert_eq!(shapes(1).len(), 34);
        for s in shapes(1) {
            let own = s.hand.0.iter().position(|&c| c == 1).unwrap() as u8;
            assert_eq!(s.wins, vec![own]);
        }
        // Four tiles: each shape's waits agree with the engine's, and every
        // 4-tile hand the engine calls ready is a shape.
        for s in shapes(4) {
            assert_eq!(s.hand.total(), 4);
            let want: Vec<u8> = mahjong_hk::hand::waits(&s.hand, 3)
                .into_iter()
                .map(|t| t.index())
                .collect();
            assert_eq!(s.wins, want, "{:?}", s.hand);
        }
        fn walk(h: &mut TileCounts, from: usize, left: usize, ready: &mut usize) {
            if left == 0 {
                if !mahjong_hk::hand::waits(h, 3).is_empty() {
                    *ready += 1;
                }
                return;
            }
            for k in from..NUM_KINDS {
                if h.0[k] < 4 {
                    h.0[k] += 1;
                    walk(h, k, left - 1, ready);
                    h.0[k] -= 1;
                }
            }
        }
        let mut ready = 0;
        walk(&mut TileCounts::new(), 0, 4, &mut ready);
        assert_eq!(ready, shapes(4).len());
    }

    /// Brian's case (2026-11-19): four exposed pungs of one suit, and the visible
    /// tiles leave one kind (9s): 9s is 1.0, every other tile 0.0 (a kind with
    /// every other copy visible included).
    #[test]
    fn one_kind_left_is_certain() {
        let s = four_pungs(["9s", "9s", "9s"], "9s 123456789p ESWN", &[]);
        let e = exact_wait(&s, 0, 2).unwrap();
        assert_eq!(e.concealed, 1);
        assert_eq!(e.hands, 1);
        assert_eq!(e.win[t("9s").index() as usize], 1.0);
        let d = with_exact(true, || Danger::from_belief(&s, 0, &HeuristicBelief));
        let est = with_exact(true, || Danger::estimate(&s, 0));
        for x in Tile::all_kinds() {
            let want = if x == t("9s") { 1.0 } else { 0.0 };
            assert_eq!(d.tile(x), want, "{x}");
            assert_eq!(est.tile(x), want, "{x}");
        }
        // Full flush + all pungs: the 10-faan cap, 128 points a share; seats 1
        // and 3 (all pungs and a seat flower) win on it too.
        assert_eq!(d.points_if_win(2), 128.0);
        let others = d.points_if_win(1) + d.points_if_win(3);
        assert!(others >= 16.0);
        // The discarder pays each winner 2 shares (rules §8.2).
        assert_eq!(d.expected_loss(t("9s")), 2.0 * (128.0 + others));
        assert!(d.is_exact(2) && d.split(t("9s")).is_some());
        check_switch_off_is_unchanged(&s);
    }

    /// Two kinds left with 2 and 1 unseen copies: 2/3 and 1/3.
    #[test]
    fn two_kinds_left_split_by_copies() {
        let s = four_pungs(["9s", "9s", "8s"], "9s 8s 12345678p ESWN", &[]);
        let e = exact_wait(&s, 0, 2).unwrap();
        assert!((e.win[t("9s").index() as usize] - 2.0 / 3.0).abs() < 1e-12);
        assert!((e.win[t("8s").index() as usize] - 1.0 / 3.0).abs() < 1e-12);
        let d = with_exact(true, || Danger::from_belief(&s, 0, &HeuristicBelief));
        assert_eq!(d.tile(t("5p")), 0.0);
        assert!(d.tile(t("8s")) > 0.0 && d.tile(t("8s")) < 1.0);
    }

    /// The passed-win lock (rules §4.4): seat 1 threw an 8s after seat 2's last
    /// discard, so seat 2 cannot win on 8s now; 9s is the only legal wait.
    #[test]
    fn a_passed_tile_is_refused() {
        let s = four_pungs(["9s", "9s", "8s"], "9s 12345678p ESWN 9p", &[(1, "8s")]);
        assert!(refused(&s, 2)[t("8s").index() as usize]);
        let e = exact_wait(&s, 0, 2).unwrap();
        assert_eq!(e.win[t("9s").index() as usize], 1.0);
        assert_eq!(e.win[t("8s").index() as usize], 0.0);
    }

    /// A hand that reaches the minimum only as a flush: one chow and three
    /// pungs of bamboos. 9s (full flush) and E (half flush) win, 2m does not.
    #[test]
    fn flush_only_hand_refuses_other_suits() {
        let s = position(
            "1p 2p 3p 4p 5p 6p 7p 8p 9p S W N Wh G",
            ["E", "9s", "2m"],
            [
                vec![pung("1m", 0), pung("3m", 2), pung("4m", 3), pung("5m", 0)],
                vec![
                    Meld::exposed(MeldKind::Chow, t("1s"), 1),
                    pung("4s", 3),
                    pung("6s", 1),
                    pung("8s", 0),
                ],
                vec![pung("6m", 0), pung("7m", 1), pung("8m", 2), pung("9m", 0)],
            ],
            &[],
        );
        let e = exact_wait(&s, 0, 2).unwrap();
        assert_eq!(e.win[t("9s").index() as usize], 0.5);
        assert_eq!(e.win[t("E").index() as usize], 0.5);
        assert_eq!(e.win[t("2m").index() as usize], 0.0);
        // Full flush 7 faan = 48 points; half flush 3 faan = 8.
        assert_eq!(e.points[t("9s").index() as usize], 48.0);
        assert_eq!(e.points[t("E").index() as usize], 8.0);
        let d = with_exact(true, || Danger::from_belief(&s, 0, &HeuristicBelief));
        assert_eq!(d.tile(t("5p")), 0.0);
    }

    /// Three exposed sets (two dragon pungs and the seat wind, 3 faan) and four
    /// concealed tiles from the unseen 2m 3m 9p 9p E N: the only ready hand
    /// waits on 1m and 4m. Those keep the table's chance that seat 2 is ready;
    /// every other tile is exactly 0.
    #[test]
    fn four_concealed_tiles_zero_where_no_hand_wins() {
        let s = position(
            "1m 4m 1p 2p 3p 4p 5p 6p 7p 8p S Wh 5s 6s",
            ["E", "2m 3m 9p 9p", "N"],
            [
                vec![pung("1s", 0), pung("2s", 2), pung("3s", 3), pung("4s", 0)],
                vec![pung("R", 1), pung("G", 3), pung("W", 1)],
                vec![pung("7s", 0), pung("8s", 1), pung("9s", 2), pung("6m", 0)],
            ],
            &[],
        );
        let e = exact_wait(&s, 0, 2).unwrap();
        assert_eq!(e.concealed, 4);
        assert_eq!(e.hands, 1);
        for x in Tile::all_kinds() {
            let want = if x == t("1m") || x == t("4m") {
                1.0
            } else {
                0.0
            };
            assert_eq!(e.win[x.index() as usize], want, "{x}");
        }
        let builders: [fn(&State) -> Danger; 2] = [
            |s| Danger::from_belief(s, 0, &HeuristicBelief),
            |s| Danger::estimate(s, 0),
        ];
        for build in builders {
            let off = build(&s);
            let on = with_exact(true, || build(&s));
            assert_eq!(on.threat(2), off.threat(2));
            assert!(off.threat(2) > 0.0);
            // Seats 1 and 3 hold one tile each: any unseen kind but 1m, 4m.
            assert!((on.tile(t("1m")) - off.threat(2)).abs() < 1e-12);
            assert_eq!(on.tile(t("5p")), 0.0);
            assert_eq!(on.tile(t("Wh")), 0.0);
        }
        check_switch_off_is_unchanged(&s);
    }

    /// With the switch on but nobody holding 1 or 4 concealed tiles, every
    /// figure is the same as with it off.
    #[test]
    fn switch_changes_nothing_without_a_short_hand() {
        use crate::bots::Bot;
        use mahjong_games::{Game, Policy};
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        let mut rng = SmallRng::seed_from_u64(5);
        let bot = Bot::by_name("value").unwrap();
        let mut checked = 0;
        for _ in 0..6 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(p) = s.current_player() {
                if s.phase_name() == "turn"
                    && (0..4).all(|o| o == p || !matches!(s.concealed_count(o), 1 | 4))
                {
                    let a = Danger::from_belief(&s, p, &HeuristicBelief);
                    let b = with_exact(true, || Danger::from_belief(&s, p, &HeuristicBelief));
                    let c = Danger::estimate(&s, p);
                    let d = with_exact(true, || Danger::estimate(&s, p));
                    for x in Tile::all_kinds() {
                        assert_eq!(a.tile(x), b.tile(x));
                        assert_eq!(a.expected_loss(x), b.expected_loss(x));
                        assert_eq!(c.tile(x), d.tile(x));
                    }
                    checked += 1;
                }
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
            }
        }
        assert!(checked > 0);
    }
}
