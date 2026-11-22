//! Plan Task 30: early-game throws of plain `stage`, deal by deal.
//!
//! `early_scan <deals> [seed]`: plays deals `seed * 1_000_000 + i` with four
//! plain `stage` (the deal and random streams of `false_alarm::play`, tested
//! seat 0), and at every stage 1 discard prints a line when the pick looks
//! wrong by one of these checks, with the deal id and the action index:
//!
//! * `suit`: the pick is a tile of the best flush suit while a suited single
//!   outside it ties (the variant `stage-early-suit` throws another tile);
//! * `far`: the variant `stage-early-far` throws another tile (a far line such
//!   as thirteen orphans decided the tie);
//! * `keepvalue`: the variant `stage-early-value` throws another tile;
//! * `value`: the pick is a single dragon or a single wind of the seat (seat 0
//!   is East, the dealer) or the round while a single honour worth no faan is held;
//! * `pair`: the pick breaks a pair or a triplet while the hand holds a tile
//!   with no tile within two steps of it and no copy (an isolated single);
//! * `middle`: the pick has a neighbour within two steps while an isolated
//!   suited single or honour single is held.
//!
//! Each line shows the hand before the throw, the leading line, the pick, the
//! variants' picks (`stage-early-suit`, `-far`, `-value`, `-ab`, `stage-early`), and
//! the throw's index among the seat's discards. A summary counts the checks and
//! how often each variant changes a stage 1 pick.
use mahjong_games::Game;
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::false_alarm::game_rng;
use mahjong_search::stage::{flush_suit, StageBot, EARLY_FAR, EARLY_SUIT, EARLY_VALUE};
use mahjong_search::SeatView;
use mahjong_tiles::{Tile, TileCounts};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn isolated(hand: &TileCounts, t: Tile) -> bool {
    hand[t] == 1
        && [-2i8, -1, 1, 2]
            .iter()
            .all(|&d| t.offset(d).is_none_or(|u| hand[u] == 0))
}

fn text(hand: &TileCounts) -> String {
    hand.tiles()
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let deals: u64 = args.first().and_then(|a| a.parse().ok()).unwrap_or(40);
    let seed: u64 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(30);
    let rules = RuleSet::default();
    let plain = StageBot::default();
    let variants = [
        ("suit", plain.with_early(EARLY_SUIT)),
        ("far", plain.with_early(EARLY_FAR)),
        ("value", plain.with_early(EARLY_VALUE)),
        ("ab", plain.with_early(EARLY_SUIT | EARLY_FAR)),
        (
            "all",
            plain.with_early(EARLY_SUIT | EARLY_FAR | EARLY_VALUE),
        ),
    ];
    let mut counts = std::collections::BTreeMap::<&str, u32>::new();
    let mut stage1 = 0u32;
    let mut changed = [0u32; 5];
    for i in 0..deals {
        let deal = seed * 1_000_000 + i;
        let mut state = State::new_deal(rules, &mut ChaCha8Rng::seed_from_u64(deal));
        let mut rng = game_rng(deal, 0);
        let mut index = 0usize;
        let mut thrown = [0u32; 4];
        while !state.is_terminal() {
            let legal = state.legal_actions_vec();
            let p = state.current_player().expect("not over");
            let discard_turn = !state.in_claim() && !state.in_rob();
            let before = rng.clone();
            let (action, note) = plain.decide(&state, &legal, &mut rng);
            if let (true, Some(note), Action::Discard(tile)) = (discard_turn, note, action) {
                thrown[p] += 1;
                if note.stage == 1 {
                    stage1 += 1;
                    let view = SeatView::of_current(&state);
                    let hand = view.concealed;
                    let picks: Vec<Action> = variants
                        .iter()
                        .map(|(_, b)| b.decide(&state, &legal, &mut before.clone()).0)
                        .collect();
                    for (k, a) in picks.iter().enumerate() {
                        if *a != action {
                            changed[k] += 1;
                        }
                    }
                    let mut flags: Vec<&str> = Vec::new();
                    if picks[0] != action {
                        flags.push("suit");
                    }
                    if picks[1] != action {
                        flags.push("far");
                    }
                    if picks[2] != action {
                        flags.push("keepvalue");
                    }
                    let value = |t: Tile| {
                        t.is_honour()
                            && (t.index() >= 31
                                || t.index() as usize == 27 + p
                                || t.index() as usize == 27 + rules.prevailing_wind)
                    };
                    if hand[tile] == 1
                        && value(tile)
                        && hand
                            .kinds()
                            .any(|u| u.is_honour() && hand[u] == 1 && !value(u))
                    {
                        flags.push("value");
                    }
                    let lone = hand.kinds().any(|u| isolated(&hand, u));
                    if hand[tile] >= 2 && lone {
                        flags.push("pair");
                    }
                    if hand[tile] == 1 && !isolated(&hand, tile) && lone {
                        flags.push("middle");
                    }
                    for f in &flags {
                        *counts.entry(f).or_default() += 1;
                    }
                    if !flags.is_empty() {
                        let name = |a: &Action| match a {
                            Action::Discard(t) => t.to_string(),
                            a => format!("{a:?}"),
                        };
                        println!(
                            "deal {deal} action {index} seat {p} throw {} [{}] hand {} | line {:?} flush {:?} | pick {} | suit {} far {} value {} ab {} all {}",
                            thrown[p],
                            flags.join(","),
                            text(&hand),
                            note.line,
                            flush_suit(&view_13(&view, tile)),
                            tile,
                            name(&picks[0]),
                            name(&picks[1]),
                            name(&picks[2]),
                            name(&picks[3]),
                            name(&picks[4]),
                        );
                    }
                }
            }
            state.apply(action);
            index += 1;
        }
    }
    println!("deals {deals} seed {seed}: stage 1 discards {stage1}");
    for (k, (n, _)) in variants.iter().enumerate() {
        println!("  variant {n} changes {} picks", changed[k]);
    }
    for (f, n) in &counts {
        println!("  check {f}: {n}");
    }
}

/// The 13-tile view after throwing `t`.
fn view_13(view: &SeatView, t: Tile) -> SeatView {
    let mut h = view.concealed;
    h.remove(t);
    view.with(h, view.melds.clone())
}
