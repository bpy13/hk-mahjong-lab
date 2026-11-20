//! Plan Task 20: the defence trials on Task 19's reading, paired matches.
//!
//! `defence_match match <bot> <old bot> <deals> [seed]`: each deal is played with
//! `<bot>` in each of the four seats against three plain `stage` bots, then the
//! same with `<old bot>` (the same bot on the old reading; `-` for none), and once
//! with four plain `stage` bots. Bots are agent specs (`stage-fold:belief=...`,
//! `race:belief=...`, `stage-ev:belief=...`). Prints per bot the points per hand
//! with a 95 % range over deals (the four seatings of a deal are one sample), win
//! and deal-in rates, and the share of hands where the bot's defence changed a
//! discard (stage bots); then the paired differences on the same deals and seats:
//! `bot − old bot`, `bot − plain stage` and `old bot − plain stage` (plain stage
//! is the seat's score in the all-`stage` game of the same deal).
//!
//! `defence_match scenario <bot spec with belief=>`: Brian's scenario (one
//! opponent on a 萬子 full flush, two on 筒子, my hand hopeless): every candidate
//! discard's terms from [`mahjong_search::defence::evaluate`].
use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::{Agent, Bot};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// One seat's result in one game.
#[derive(Clone, Copy, Default)]
struct Hand {
    points: f64,
    win: f64,
    deal_in: f64,
    changed: f64,
}

/// Plays one deal with `me` in `seat` and plain `stage` elsewhere (`me` = `None`:
/// four plain `stage`).
fn play(rules: &RuleSet, me: Option<&Agent>, stage: &Bot, deal: u64, seat: usize) -> Hand {
    let mut state = State::new_deal(*rules, &mut ChaCha8Rng::seed_from_u64(deal));
    let mut rng = ChaCha8Rng::seed_from_u64(deal.wrapping_mul(4).wrapping_add(seat as u64));
    let mut changed = false;
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let p = state.current_player().expect("not over");
        let action = match me {
            Some(Agent::Bot(Bot::Stage(b))) if p == seat => {
                let (a, note) = b.decide(&state, &legal, &mut rng);
                changed |= note.is_some_and(|n| n.gave_up);
                a
            }
            Some(a) if p == seat => a.choose(&state, &legal, &mut rng),
            _ => stage.choose(&state, &legal, &mut rng),
        };
        state.apply(action);
    }
    let r = state.result().expect("over");
    Hand {
        points: r.deltas[seat] as f64,
        win: r.wins.iter().any(|w| w.seat == seat) as u8 as f64,
        deal_in: r.wins.iter().any(|w| !w.self_drawn && w.from == Some(seat)) as u8 as f64,
        changed: changed as u8 as f64,
    }
}

/// Mean and 95 % range of per-deal samples.
fn mean_ci(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let v = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, 1.96 * (v / n).sqrt())
}

fn run_match(spec: &str, old: &str, deals: u64, seed: u64) {
    let rules = RuleSet::default();
    let stage = Bot::by_name("stage").expect("stage");
    let new = Agent::parse(spec).unwrap_or_else(|e| panic!("{spec}: {e}"));
    let old_agent =
        (old != "-").then(|| Agent::parse(old).unwrap_or_else(|e| panic!("{old}: {e}")));
    let start = std::time::Instant::now();
    // Per deal: [new, old, plain] sums over the four seats.
    let rows: Vec<[Hand; 3]> = (0..deals)
        .into_par_iter()
        .map(|i| {
            let deal = seed.wrapping_mul(1_000_003).wrapping_add(i);
            let mut sum = [Hand::default(); 3];
            let plain: Vec<Hand> = (0..4)
                .map(|s| play(&rules, None, &stage, deal, s))
                .collect();
            for (seat, &plain_hand) in plain.iter().enumerate() {
                let mut hands = [
                    play(&rules, Some(&new), &stage, deal, seat),
                    Hand::default(),
                    plain_hand,
                ];
                if let Some(o) = &old_agent {
                    hands[1] = play(&rules, Some(o), &stage, deal, seat);
                }
                for (s, h) in sum.iter_mut().zip(hands) {
                    s.points += h.points / 4.0;
                    s.win += h.win;
                    s.deal_in += h.deal_in;
                    s.changed += h.changed;
                }
            }
            sum
        })
        .collect();
    println!(
        "{spec} (old: {old}) vs three stage, {deals} deals x 4 seats, seed {seed} ({:?})",
        start.elapsed()
    );
    let hands = 4.0 * deals as f64;
    let col = |k: usize| -> Vec<f64> { rows.iter().map(|r| r[k].points).collect() };
    let names = [spec, old, "stage (plain)"];
    for k in 0..3 {
        if k == 1 && old_agent.is_none() {
            continue;
        }
        let (m, ci) = mean_ci(&col(k));
        let sum = |f: fn(&Hand) -> f64| rows.iter().map(|r| f(&r[k])).sum::<f64>();
        println!(
            "  {:<48} {:+.2} ± {:.2}/hand  win {:.1}%  deal-in {:.1}%  defence changed a discard in {:.1}% of hands",
            names[k],
            m,
            ci,
            100.0 * sum(|h| h.win) / hands,
            100.0 * sum(|h| h.deal_in) / hands,
            100.0 * sum(|h| h.changed) / hands,
        );
    }
    let diff = |a: usize, b: usize| -> (f64, f64) {
        mean_ci(
            &rows
                .iter()
                .map(|r| r[a].points - r[b].points)
                .collect::<Vec<_>>(),
        )
    };
    if old_agent.is_some() {
        let (m, ci) = diff(0, 1);
        println!("  paired new − old reading:  {m:+.3} ± {ci:.3}/hand");
    }
    let (m, ci) = diff(0, 2);
    println!("  paired new − plain stage:  {m:+.3} ± {ci:.3}/hand");
    if old_agent.is_some() {
        let (m, ci) = diff(1, 2);
        println!("  paired old − plain stage:  {m:+.3} ± {ci:.3}/hand");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize, d: u64| args.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
    match args.first().map(String::as_str) {
        Some("match") => run_match(&args[1], &args[2], num(3, 100), num(4, 1)),
        Some("scenario") => scenario::run(&args[1]),
        _ => {
            eprintln!("defence_match match <bot> <old bot|-> <deals> [seed] | scenario <bot spec>")
        }
    }
}

mod scenario {
    use mahjong_hk::{Meld, MeldKind, PositionPhase, PositionSpec, RuleSet, State};
    use mahjong_search::defence;
    use mahjong_search::{Agent, Bot, SeatView};
    use mahjong_tiles::{parse_tiles, Tile, TileCounts};

    fn t(s: &str) -> Tile {
        s.parse().unwrap()
    }

    fn counts(s: &str) -> TileCounts {
        TileCounts::from_tiles(&parse_tiles(s).unwrap())
    }

    /// Brian's scenario (2026-11-20): next seat (1) on a 萬子 full flush with
    /// three exposed 萬子 melds and ready; across (2) and previous (3) collecting
    /// 筒子 with two melds each; me (seat 0) on turn with a scattered hand. No
    /// 萬子 in seat 1's discards, no 筒子 in seats 2 and 3's.
    pub fn position() -> State {
        let melds: [Vec<Meld>; 4] = [
            vec![],
            vec![
                Meld::exposed(MeldKind::Pung, t("2m"), 3),
                Meld::exposed(MeldKind::Pung, t("7m"), 0),
                Meld::exposed(MeldKind::Chow, t("3m"), 0),
            ],
            vec![
                Meld::exposed(MeldKind::Pung, t("5p"), 1),
                Meld::exposed(MeldKind::Pung, t("Wh"), 3),
            ],
            vec![
                Meld::exposed(MeldKind::Pung, t("9p"), 2),
                Meld::exposed(MeldKind::Chow, t("1p"), 2),
            ],
        ];
        let hands = [
            counts("1s 2s 3s 5s 7s 9s 1p 4p 7p E W N G R"),
            counts("1m 1m 8m 9m"),
            counts("2p 3p 4p 4p 6p 7p 8p"),
            counts("6p 6p 7p 7p 8p S S"),
        ];
        let discards = [
            parse_tiles("8s G 2p S 3s N 6s 3p 4s").unwrap(),
            parse_tiles("E S 9s 1p W 2s 8p N 5s").unwrap(),
            parse_tiles("1m N 8s 3m 9m 2s 6m E 7s").unwrap(),
            parse_tiles("W 9s 1m 4s 8m 5m 3s Wh 6s").unwrap(),
        ];
        let claims: [(usize, usize, &str); 7] = [
            (0, 3, "2m"),
            (0, 0, "7m"),
            (1, 0, "4m"),
            (1, 1, "5p"),
            (2, 3, "Wh"),
            (2, 2, "9p"),
            (3, 2, "2p"),
        ];
        let mut river = Vec::new();
        for round in 0..9 {
            for (seat, d) in discards.iter().enumerate() {
                river.push((seat, d[round], false));
            }
            for &(_, seat, tile) in claims.iter().filter(|c| c.0 == round) {
                river.push((seat, t(tile), true));
            }
        }
        let used: usize = hands.iter().map(|h| h.total()).sum::<usize>()
            + melds.iter().map(|m| 3 * m.len()).sum::<usize>()
            + 8
            + 36;
        let spec = PositionSpec {
            hands,
            melds,
            bonus: [
                parse_tiles("F1 S1").unwrap(),
                parse_tiles("F2 S2").unwrap(),
                parse_tiles("F3 S3").unwrap(),
                parse_tiles("F4 S4").unwrap(),
            ],
            river,
            wall_remaining: 144 - used,
            to_move: 0,
            phase: PositionPhase::Turn,
            last_drawn: Some(t("R")),
            kong_chain: 0,
            liable: [None; 4],
            passed_wins: Default::default(),
            river_from_hand: Vec::new(),
        };
        State::from_position(RuleSet::default(), &spec).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn run(spec: &str) {
        let Ok(Agent::Bot(Bot::Stage(bot))) = Agent::parse(spec) else {
            panic!("{spec}: not a stage bot spec");
        };
        let model = bot.reading_model().expect("belief=... needed");
        let state = position();
        let view = SeatView::of(&state, 0);
        let candidates: Vec<Tile> = view.concealed.tiles();
        let out = model.run_for(&state, 0);
        println!(
            "Brian's scenario: wall {} tiles, my hand {}",
            state.wall_remaining(),
            view.concealed
        );
        for o in 1..4 {
            let b = out.seat(o).unwrap();
            println!("  seat {o}: P(ready) {:.1} %", 100.0 * b.tenpai);
        }
        if let Some(w) = out.win_chances() {
            println!(
                "  win chances now: me {:.1} %, next {:.1} %, across {:.1} %, previous {:.1} %, draw {:.1} %",
                100.0 * w[0], 100.0 * w[1], 100.0 * w[2], 100.0 * w[3], 100.0 * w[4]
            );
        }
        let terms = defence::evaluate(&state, &view, &candidates, &model).expect("a Task 19 file");
        println!("  tile     EV    win  deal-in  self-draw  liab  | my win  next/across/prev win after | claim next/across/prev");
        for x in &terms {
            println!(
                "  {:<4} {:+7.2} {:6.2} {:7.2} {:9.2} {:5.2} | {:5.1}%  {:5.1} {:5.1} {:5.1} % | {:4.1} {:4.1} {:4.1} %",
                x.tile.to_string(),
                x.ev,
                x.win,
                x.deal_in,
                x.self_draw,
                x.liability,
                100.0 * x.my_win,
                100.0 * x.their_win[0],
                100.0 * x.their_win[1],
                100.0 * x.their_win[2],
                100.0 * x.claim[0],
                100.0 * x.claim[1],
                100.0 * x.claim[2],
            );
        }
    }
}
