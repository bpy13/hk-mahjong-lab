//! The ready hand's push-or-fold choice learned from play-outs (plan Task 12,
//! [`mahjong_search::pushfold`]; Task 14 adds the exact danger input).
//!
//! `pushfold label <deals> <samples> [seed] [out.csv]`: step 1. Four `stage` bots
//! play each deal once. Every stage 3 discard that leaves the hand ready while a
//! tile with a smaller expected loss exists is a candidate decision; one per deal
//! is drawn at random (the deal's count of candidates is in the row, as a weight)
//! and labelled: `samples` reshuffles of the undrawn wall (every seat keeps its
//! true hand), each played out once with the chosen tile thrown and once with the
//! safest tile thrown, on the same wall order and random stream. The row holds
//! the features ([`FEATURE_NAMES`]), the label (mean push − fold), its standard
//! error, the two means, whether the chosen and the safest tile truly deal in
//! (an opponent's real hand wins on it) and what that costs, the source of each
//! danger figure (`exact_t`, `exact_s`: -1 the table, 0 or 1 the exact figure),
//! the table-only decision at the same position (`table_eligible`, `s_table`
//! and its figures) and the same for every tile in hand
//! (`tile:danger:points:table danger`). Rows are appended to the CSV in blocks,
//! with a progress line per block.
//!
//! `pushfold label-exact <deals> <samples> [seed] [out.csv]`: the same with the
//! exact layer on (plan Task 14, [`mahjong_search::pushfold::Risk`]): the danger
//! and loss of `t` and `s`, the choice of `s` and the folded hand's discards use
//! Task 13's exact figure where it is exactly 0 or 1, else the table.
//!
//! With the environment variable `BELIEF=<model file>` (plan Task 20) `label`
//! reads the danger from that model ([`mahjong_search::Danger::of_reading`]): the
//! choice of `s`, the features and the folded hand's discards.
//!
//! `pushfold match <bot> <opponent> <deals> [seed]`: step 5's detail columns, as
//! Task 11's `giveup_log match`: each deal is played four times with `<bot>` in
//! each seat and `<opponent>` in the others; points per hand with a 95 % range
//! over deals (the four seatings of a deal are one sample), win and deal-in
//! rates, points paid per deal-in, the share of ready hands that gave up at least
//! once, and deal-ins per 100 hands from stage 3 discards.
//!
//! `pushfold diff <bot a> <bot b> <opponent> <deals> [seed]`: `match` for both
//! bots on the same deals and seats, and the paired difference `a − b` in points
//! per hand with its 95 % range over deals (plan Task 14's measure).
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
fn label_deal(rules: &RuleSet, deal_seed: u64, samples: usize, exact: bool) -> Option<String> {
    let bot = StageBot::default();
    let mut state = State::new_deal(*rules, &mut ChaCha8Rng::seed_from_u64(deal_seed));
    let mut rng = ChaCha8Rng::seed_from_u64(deal_seed ^ 0x9e37_79b9);
    let mut found: Vec<(State, Decision)> = Vec::new();
    let mut n_table = 0usize;
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let (action, note) = bot.decide(&state, &legal, &mut rng);
        if let Action::Discard(t) = action {
            if note.is_some_and(|n| n.stage == 3) {
                let view = SeatView::of_current(&state);
                let candidates = discards_in(&legal);
                if let Some(d) = pushfold::decision(&state, &view, &candidates, t, exact) {
                    found.push((state.clone(), d));
                }
                if exact && pushfold::decision(&state, &view, &candidates, t, false).is_some() {
                    n_table += 1;
                }
            }
        }
        state.apply(action);
    }
    if found.is_empty() {
        return None;
    }
    if !exact {
        n_table = found.len();
    }
    let mut pick = ChaCha8Rng::seed_from_u64(deal_seed.wrapping_mul(31) ^ 0x51ed);
    let n = found.len();
    let (pos, dec) = found.swap_remove(pick.gen_range(0..n as u32) as usize);
    let l = pushfold::label(&pos, &dec, samples, deal_seed);
    let view = SeatView::of(&pos, dec.seat);
    let candidates = discards_in(&pos.legal_actions_vec());
    let truth = pushfold::truth(&pos, &view, &candidates, exact);
    let of = |t: Tile| truth.iter().find(|x| x.tile == t).expect("a candidate");
    let (tt, ts) = (of(dec.push), of(dec.fold));
    let mut row = format!("{deal_seed},{},{},{},{n}", dec.seat, dec.push, dec.fold);
    for x in dec.x {
        row.push_str(&format!(",{x}"));
    }
    row.push_str(&format!(
        ",{},{},{},{},{},{},{},{},{}",
        l.mean,
        l.se,
        l.samples,
        l.push_mean,
        l.fold_mean,
        tt.deals_in as u8,
        tt.paid,
        ts.deals_in as u8,
        ts.paid
    ));
    // The sources of the two figures, and the table-only decision at this position.
    let table = pushfold::decision(&pos, &view, &candidates, dec.push, false);
    let (s_table, xt) = match &table {
        Some(d) => (d.fold.to_string(), [d.x[0], d.x[1], d.x[2], d.x[3]]),
        None => ("-".to_string(), [0.0; 4]),
    };
    row.push_str(&format!(
        ",{},{},{},{s_table},{},{},{},{},{n_table},",
        dec.source.0.code(),
        dec.source.1.code(),
        table.is_some() as u8,
        xt[0],
        xt[1],
        xt[2],
        xt[3]
    ));
    // Every tile in hand: `tile:danger:points paid (0 = truly safe):table danger`.
    let tiles: Vec<String> = truth
        .iter()
        .map(|x| {
            format!(
                "{}:{:.5}:{}:{:.5}",
                x.tile, x.danger, x.paid, x.table_danger
            )
        })
        .collect();
    row.push_str(&tiles.join(";"));
    Some(row)
}

/// Plan Task 20: the reading named by the environment variable `BELIEF` (a belief
/// model file), set for each deal by [`danger::with_reading`]; none: the tables.
fn env_reading() -> Option<std::sync::Arc<mahjong_search::BeliefModel>> {
    let path = std::env::var("BELIEF").ok()?;
    let m = mahjong_search::BeliefModel::load(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    eprintln!("reading: {path}");
    Some(std::sync::Arc::new(m))
}

fn run_label(deals: u64, samples: usize, seed: u64, out: &str, exact: bool) {
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
        header.push_str(
            ",label,se,samples,push_mean,fold_mean,true_t,paid_t,true_s,paid_s,exact_t,exact_s,\
             table_eligible,s_table,d_t_table,loss_t_table,d_s_table,loss_s_table,n_eligible_table,tiles",
        );
        writeln!(file, "{header}").expect("write");
    }
    let block = 500u64;
    let mut rows = 0usize;
    let mut done = 0u64;
    let reading = env_reading();
    while done < deals {
        let end = (done + block).min(deals);
        let out_rows: Vec<String> = (done..end)
            .into_par_iter()
            .filter_map(|i| {
                mahjong_search::danger::with_reading(reading.clone(), || {
                    label_deal(
                        &rules,
                        seed.wrapping_mul(1_000_003).wrapping_add(i),
                        samples,
                        exact,
                    )
                })
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
        "labels{}: {deals} deals, {samples} samples each, seed {seed}, {rows} rows -> {out} ({:?})",
        if exact { " (exact layer on)" } else { "" },
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

    /// Points per hand (mean over deals of the deal's four-seat mean) and its
    /// 95 % range over deals.
    fn mean_ci(&self) -> (f64, f64) {
        let mean = self.deal_sum / 4.0 / self.deals;
        let var = (self.deal_sq / self.deals - mean * mean).max(0.0);
        (mean, 1.96 * (var / self.deals).sqrt())
    }

    fn line(&self) -> String {
        let (mean, ci) = self.mean_ci();
        format!(
            "points/hand {:+.2} ± {:.2}  win {:.1}%  deal-in {:.1}%  paid per deal-in {:.1}  ready hands that gave up {:.1}%  stage 3 deal-ins per 100 hands {:.2}",
            mean,
            ci,
            100.0 * self.wins / self.hands,
            100.0 * self.deal_ins / self.hands,
            self.paid / self.deal_ins.max(1.0),
            100.0 * self.gave_up_hands / self.ready_hands.max(1.0),
            100.0 * self.stage3_deal_ins / self.hands,
        )
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

/// One deal played four times with `bot` in each seat: one sample (the mean of
/// its four seatings).
fn play_deal(rules: &RuleSet, bot: &Bot, other: &Bot, deal: u64) -> Match {
    let mut sum = Match::default();
    for seat in 0..4 {
        sum = sum.add(play_match(rules, bot, other, deal, seat));
    }
    let x = sum.deal_sum / 4.0;
    sum.deals = 1.0;
    sum.deal_sq = x * x;
    sum
}

fn by_name(name: &str) -> Bot {
    Bot::by_name(name).unwrap_or_else(|| panic!("unknown bot {name}"))
}

fn deal_seed(seed: u64, i: u64) -> u64 {
    seed.wrapping_mul(1_000_003).wrapping_add(i)
}

fn models_line() -> String {
    format!(
        "models fitted: table {}, exact {}",
        pushfold::fitted(false),
        pushfold::fitted(true)
    )
}

fn run_match(me: &str, opp: &str, deals: u64, seed: u64) {
    let rules = RuleSet::default();
    let (bot, other) = (by_name(me), by_name(opp));
    let start = std::time::Instant::now();
    let m = (0..deals)
        .into_par_iter()
        .map(|i| play_deal(&rules, &bot, &other, deal_seed(seed, i)))
        .reduce(Match::default, Match::add);
    println!(
        "{me} vs three {opp}, {deals} deals x 4 seats, seed {seed} ({:?}; {})",
        start.elapsed(),
        models_line()
    );
    println!("  {}", m.line());
}

fn run_diff(a: &str, b: &str, opp: &str, deals: u64, seed: u64) {
    let rules = RuleSet::default();
    let (bot_a, bot_b, other) = (by_name(a), by_name(b), by_name(opp));
    let start = std::time::Instant::now();
    // (a's stats, b's stats, Σ diff, Σ diff²) over deals.
    let (ma, mb, d_sum, d_sq) = (0..deals)
        .into_par_iter()
        .map(|i| {
            let deal = deal_seed(seed, i);
            let x = play_deal(&rules, &bot_a, &other, deal);
            let y = play_deal(&rules, &bot_b, &other, deal);
            let d = (x.deal_sum - y.deal_sum) / 4.0;
            (x, y, d, d * d)
        })
        .reduce(
            || (Match::default(), Match::default(), 0.0, 0.0),
            |p, q| (p.0.add(q.0), p.1.add(q.1), p.2 + q.2, p.3 + q.3),
        );
    let n = deals as f64;
    let mean = d_sum / n;
    let var = (d_sq / n - mean * mean).max(0.0);
    println!(
        "{a} and {b} vs three {opp}, {deals} deals x 4 seats each, seed {seed} ({:?}; {})",
        start.elapsed(),
        models_line()
    );
    println!("  {a}: {}", ma.line());
    println!("  {b}: {}", mb.line());
    println!(
        "  paired difference {a} − {b}: {:+.3} ± {:.3} points per hand (95 % range over deals)",
        mean,
        1.96 * (var / n).sqrt()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize, d: u64| args.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
    match args.first().map(String::as_str) {
        Some(mode @ ("label" | "label-exact")) => {
            let exact = mode == "label-exact";
            let seed = num(3, 1);
            let task = if exact { 14 } else { 12 };
            let default = format!("runs/stage/t{task}-labels-seed{seed}.csv");
            let out = args.get(4).cloned().unwrap_or(default);
            run_label(num(1, 100), num(2, 16) as usize, seed, &out, exact);
        }
        Some("match") => run_match(&args[1], &args[2], num(3, 100), num(4, 1)),
        Some("diff") => run_diff(&args[1], &args[2], &args[3], num(4, 100), num(5, 1)),
        _ => eprintln!(
            "pushfold label|label-exact <deals> <samples> [seed] [out.csv] | match <bot> <opponent> <deals> [seed] | diff <bot a> <bot b> <opponent> <deals> [seed]"
        ),
    }
}
