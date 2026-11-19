//! The ready hand's push-or-fold choice learned from play-outs (plan Task 12,
//! [`mahjong_search::pushfold`]).
//!
//! `pushfold label <deals> <samples> [seed] [out.csv]`: step 1. Four `stage` bots
//! play each deal once. Every stage 3 discard that leaves the hand ready while a
//! tile with a smaller expected loss exists is a candidate decision; one per deal
//! is drawn at random (the deal's count of candidates is in the row, as a weight)
//! and labelled: `samples` re-dealt worlds, each played out once with the chosen
//! tile thrown and once with the safest tile thrown, on the same hidden tiles and
//! random stream. The row holds the features ([`FEATURE_NAMES`]), the label
//! (mean push − fold), its standard error, the two means and, as a check of the
//! re-dealing, the same pair played on the deal's real hidden tiles. Rows are
//! appended to the CSV in blocks, with a progress line per block.
//!
//! `pushfold match <bot> <opponent> <deals> [seed]`: step 5's detail columns, as
//! Task 11's `giveup_log match`: each deal is played four times with `<bot>` in
//! each seat and `<opponent>` in the others; points per hand with a 95 % range
//! over deals (the four seatings of a deal are one sample), win and deal-in
//! rates, points paid per deal-in, the share of ready hands that gave up at least
//! once, and deal-ins per 100 hands from stage 3 discards.
use mahjong_analysis::faan_shanten;
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::pushfold::{self, Decision, FEATURE_NAMES};
use mahjong_search::stage::StageBot;
use mahjong_search::{Bot, SeatView};
use mahjong_tiles::Tile;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;

fn after(view: &SeatView, t: Tile) -> SeatView {
    let mut h = view.concealed;
    h.remove(t);
    view.with(h, view.melds.clone())
}

fn ready(view: &SeatView) -> bool {
    faan_shanten(&view.hand_view(), &view.rules).shanten == 0
}

fn discards_in(legal: &[Action]) -> Vec<Tile> {
    legal
        .iter()
        .filter_map(|a| match a {
            Action::Discard(x) => Some(*x),
            _ => None,
        })
        .collect()
}

/// One deal's labelled row, or `None` when no decision qualified.
fn label_deal(rules: &RuleSet, deal_seed: u64, samples: usize) -> Option<String> {
    let bot = StageBot::default();
    let mut state = State::new_deal(*rules, &mut ChaCha8Rng::seed_from_u64(deal_seed));
    let mut rng = ChaCha8Rng::seed_from_u64(deal_seed ^ 0x9e37_79b9);
    let mut found: Vec<(State, Decision)> = Vec::new();
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let (action, note) = bot.decide(&state, &legal, &mut rng);
        if let Action::Discard(t) = action {
            if note.is_some_and(|n| n.stage == 3) {
                let view = SeatView::of_current(&state);
                let candidates = discards_in(&legal);
                if let Some(d) = pushfold::decision(&state, &view, &candidates, t) {
                    found.push((state.clone(), d));
                }
            }
        }
        state.apply(action);
    }
    if found.is_empty() {
        return None;
    }
    let mut pick = ChaCha8Rng::seed_from_u64(deal_seed.wrapping_mul(31) ^ 0x51ed);
    let n = found.len();
    let (pos, dec) = found.swap_remove(pick.gen_range(0..n as u32) as usize);
    let l = pushfold::label(&pos, &dec, samples, deal_seed);
    let (rp, rf) = pushfold::pair(&pos, &dec, deal_seed ^ 0xabcd);
    let mut row = format!("{deal_seed},{},{},{},{n}", dec.seat, dec.push, dec.fold);
    for x in dec.x {
        row.push_str(&format!(",{x}"));
    }
    row.push_str(&format!(
        ",{},{},{},{},{},{rp},{rf}",
        l.mean, l.se, l.samples, l.push_mean, l.fold_mean
    ));
    Some(row)
}

fn run_label(deals: u64, samples: usize, seed: u64, out: &str) {
    let rules = RuleSet::default();
    let start = std::time::Instant::now();
    let fresh = !std::path::Path::new(out).exists();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .unwrap_or_else(|e| panic!("{out}: {e}"));
    if fresh {
        let mut header = String::from("deal,seat,push,fold,n_eligible");
        for name in FEATURE_NAMES {
            header.push(',');
            header.push_str(name);
        }
        header.push_str(",label,se,samples,push_mean,fold_mean,real_push,real_fold");
        writeln!(file, "{header}").expect("write");
    }
    let block = 500u64;
    let mut rows = 0usize;
    let mut done = 0u64;
    while done < deals {
        let end = (done + block).min(deals);
        let out_rows: Vec<String> = (done..end)
            .into_par_iter()
            .filter_map(|i| {
                label_deal(
                    &rules,
                    seed.wrapping_mul(1_000_003).wrapping_add(i),
                    samples,
                )
            })
            .collect();
        for r in &out_rows {
            writeln!(file, "{r}").expect("write");
        }
        file.flush().expect("flush");
        rows += out_rows.len();
        done = end;
        println!(
            "{done}/{deals} deals, {rows} rows, {:.0} s",
            start.elapsed().as_secs_f64()
        );
    }
    println!(
        "labels: {deals} deals, {samples} samples each, seed {seed}, {rows} rows -> {out} ({:?})",
        start.elapsed()
    );
}

#[derive(Clone, Copy, Default)]
struct Match {
    hands: f64,
    deal_sum: f64,
    deal_sq: f64,
    deals: f64,
    wins: f64,
    deal_ins: f64,
    paid: f64,
    ready_hands: f64,
    gave_up_hands: f64,
    stage3_deal_ins: f64,
}

impl Match {
    fn add(mut self, o: Match) -> Match {
        self.hands += o.hands;
        self.deal_sum += o.deal_sum;
        self.deal_sq += o.deal_sq;
        self.deals += o.deals;
        self.wins += o.wins;
        self.deal_ins += o.deal_ins;
        self.paid += o.paid;
        self.ready_hands += o.ready_hands;
        self.gave_up_hands += o.gave_up_hands;
        self.stage3_deal_ins += o.stage3_deal_ins;
        self
    }
}

fn play_match(rules: &RuleSet, me: &Bot, opp: &Bot, deal_seed: u64, seat_me: usize) -> Match {
    let mut state = State::new_deal(*rules, &mut ChaCha8Rng::seed_from_u64(deal_seed));
    let mut rng = ChaCha8Rng::seed_from_u64(deal_seed.wrapping_mul(4).wrapping_add(seat_me as u64));
    let mut m = Match {
        hands: 1.0,
        ..Match::default()
    };
    let (mut was_ready, mut gave_up) = (false, false);
    let mut last3: Option<Tile> = None;
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let seat = state.current_player().expect("not over");
        if seat != seat_me {
            let a = opp.choose(&state, &legal, &mut rng);
            state.apply(a);
            continue;
        }
        let (action, note) = match me {
            Bot::Stage(b) => b.decide(&state, &legal, &mut rng),
            other => (other.choose(&state, &legal, &mut rng), None),
        };
        if let Action::Discard(t) = action {
            last3 = None;
            if let Some(n) = note {
                gave_up |= n.gave_up;
                if n.stage == 3 {
                    last3 = Some(t);
                    if !was_ready {
                        let view = SeatView::of(&state, seat);
                        was_ready = view.concealed.kinds().any(|d| ready(&after(&view, d)));
                    }
                }
            }
        }
        state.apply(action);
    }
    let r = state.result().expect("over");
    let pts = r.deltas[seat_me] as f64;
    m.deal_sum = pts;
    m.wins = r.wins.iter().any(|w| w.seat == seat_me) as u8 as f64;
    let fed: Vec<_> = r
        .wins
        .iter()
        .filter(|w| !w.self_drawn && w.from == Some(seat_me))
        .collect();
    if !fed.is_empty() {
        m.deal_ins = 1.0;
        m.paid = fed.iter().map(|w| w.paid_by[seat_me] as f64).sum();
        if let Some(t) = last3 {
            if fed.iter().any(|w| !w.robbed && w.win_tile == Some(t)) {
                m.stage3_deal_ins = 1.0;
            }
        }
    }
    m.ready_hands = (was_ready || gave_up) as u8 as f64;
    m.gave_up_hands = gave_up as u8 as f64;
    m
}

fn run_match(me: &str, opp: &str, deals: u64, seed: u64) {
    let rules = RuleSet::default();
    let (bot, other) = (
        Bot::by_name(me).unwrap_or_else(|| panic!("unknown bot {me}")),
        Bot::by_name(opp).unwrap_or_else(|| panic!("unknown bot {opp}")),
    );
    let start = std::time::Instant::now();
    let m = (0..deals)
        .into_par_iter()
        .map(|i| {
            let deal = seed.wrapping_mul(1_000_003).wrapping_add(i);
            let mut sum = Match::default();
            for seat in 0..4 {
                sum = sum.add(play_match(&rules, &bot, &other, deal, seat));
            }
            // One sample per deal: the mean over its four seatings.
            let x = sum.deal_sum / 4.0;
            sum.deals = 1.0;
            sum.deal_sq = x * x;
            sum
        })
        .reduce(Match::default, Match::add);
    let mean_deal = m.deal_sum / 4.0 / m.deals;
    let var = (m.deal_sq / m.deals - mean_deal * mean_deal).max(0.0);
    let ci = 1.96 * (var / m.deals).sqrt();
    println!(
        "{me} vs three {opp}, {deals} deals x 4 seats, seed {seed} ({:?}; model fitted: {})",
        start.elapsed(),
        pushfold::fitted()
    );
    println!(
        "  points/hand {:+.2} ± {:.2}  win {:.1}%  deal-in {:.1}%  paid per deal-in {:.1}  ready hands that gave up {:.1}%  stage 3 deal-ins per 100 hands {:.2}",
        mean_deal,
        ci,
        100.0 * m.wins / m.hands,
        100.0 * m.deal_ins / m.hands,
        m.paid / m.deal_ins.max(1.0),
        100.0 * m.gave_up_hands / m.ready_hands.max(1.0),
        100.0 * m.stage3_deal_ins / m.hands,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize, d: u64| args.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
    match args.first().map(String::as_str) {
        Some("label") => {
            let seed = num(3, 1);
            let default = format!("runs/stage/t12-labels-seed{seed}.csv");
            let out = args.get(4).cloned().unwrap_or(default);
            run_label(num(1, 100), num(2, 16) as usize, seed, &out);
        }
        Some("match") => run_match(&args[1], &args[2], num(3, 100), num(4, 1)),
        _ => eprintln!(
            "pushfold label <deals> <samples> [seed] [out.csv] | match <bot> <opponent> <deals> [seed]"
        ),
    }
}
