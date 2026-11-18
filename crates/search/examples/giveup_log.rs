//! The give-up rule for the three-stage bot's ready hand (plan Task 11).
//!
//! `giveup_log log <deals> [seed]`: step 1. Four `stage` bots play each deal once.
//! For every stage 3 discard that leaves the hand ready, the round that follows
//! (up to the seat's next discard) is followed to its end: I win by my own draw,
//! I win on a discard, my discard deals in, another player ends the hand, the
//! wall runs out, or the round passes. Per rounds-left bucket it prints the
//! measured [`Figures`] (own and theirs as multiples of live ÷ unseen), the
//! danger check of [`Danger::tile`] (from [`HeuristicBelief`]) on every stage 3
//! discard in ten bins, and the break-even table with the rough and the measured
//! figures.
//!
//! `giveup_log match <bot> <opponent> <deals> [seed]`: step 3. Each deal is played
//! four times with `<bot>` in each seat and `<opponent>` in the others. Prints the
//! bot's points per hand with a 95 % range over deals (the four seatings of a
//! deal are one sample), win and deal-in rates, points paid per deal-in, the
//! share of ready hands that gave up at least once, and deal-ins per 100 hands
//! from stage 3 discards.
use mahjong_analysis::faan_shanten;
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::ev::HeuristicBelief;
use mahjong_search::giveup::{self, Figures, Outs, BUCKETS, MEASURED, ROUGH};
use mahjong_search::stage::StageBot;
use mahjong_search::{Bot, Danger, SeatView};
use mahjong_tiles::Tile;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const BUCKET_NAMES: [&str; BUCKETS] = ["0-2", "3-5", "6-8", "9-11", "12+"];
/// Danger bins (upper edges, in %).
const EDGES: [f64; 9] = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 25.0];
const BINS: usize = EDGES.len() + 1;

fn bin(d: f64) -> usize {
    EDGES
        .iter()
        .position(|&e| 100.0 * d < e)
        .unwrap_or(EDGES.len())
}

fn after(view: &SeatView, t: Tile) -> SeatView {
    let mut h = view.concealed;
    h.remove(t);
    view.with(h, view.melds.clone())
}

fn ready(view: &SeatView) -> bool {
    faan_shanten(&view.hand_view(), &view.rules).shanten == 0
}

/// One ready round's sums, per rounds-left bucket.
#[derive(Clone, Copy, Default)]
struct Round {
    n: f64,
    share: f64,
    own: f64,
    theirs: f64,
    deal_in: f64,
    other: f64,
    wall_out: f64,
    /// Model's expected loss of the discard that opened the round, and points paid.
    pred_loss: f64,
    paid: f64,
    /// Calibrated danger of the smallest-loss tile at the opening decision.
    safe_d: f64,
}

#[derive(Clone, Copy, Default)]
struct Bin {
    n: f64,
    d: f64,
    est: f64,
    loss: f64,
    hits: f64,
    paid: f64,
}

#[derive(Clone, Default)]
struct Log {
    hands: f64,
    rounds: [Round; BUCKETS],
    bins: [Bin; BINS],
    /// The same check with the calibrated danger (`giveup::danger_of`).
    cal: [Bin; BINS],
    stage3_deal_ins: f64,
    deal_ins: f64,
}

impl Log {
    fn add(mut self, o: Log) -> Log {
        self.hands += o.hands;
        self.stage3_deal_ins += o.stage3_deal_ins;
        self.deal_ins += o.deal_ins;
        for (a, b) in self.rounds.iter_mut().zip(&o.rounds) {
            a.n += b.n;
            a.share += b.share;
            a.own += b.own;
            a.theirs += b.theirs;
            a.deal_in += b.deal_in;
            a.other += b.other;
            a.wall_out += b.wall_out;
            a.pred_loss += b.pred_loss;
            a.paid += b.paid;
            a.safe_d += b.safe_d;
        }
        let cells = self.bins.iter_mut().zip(&o.bins);
        for (a, b) in cells.chain(self.cal.iter_mut().zip(&o.cal)) {
            a.n += b.n;
            a.d += b.d;
            a.est += b.est;
            a.loss += b.loss;
            a.hits += b.hits;
            a.paid += b.paid;
        }
        self
    }

    fn figures(&self) -> Figures {
        let mut f = ROUGH;
        for (b, r) in self.rounds.iter().enumerate() {
            if r.n == 0.0 {
                continue;
            }
            f.own[b] = r.own / r.share.max(1e-9);
            f.theirs[b] = r.theirs / r.share.max(1e-9);
            f.other_ends[b] = r.other / r.n;
            f.later_deal_in[b] = ((r.deal_in - r.safe_d) / r.n).max(0.0);
        }
        f
    }
}

#[derive(Clone, Copy)]
struct Open {
    bucket: usize,
    share: f64,
    loss: f64,
    safe_d: f64,
}

#[derive(Clone, Copy)]
struct Stage3 {
    tile: Tile,
    bin: usize,
    cal: usize,
}

fn close(r: &mut Round, o: &Open) {
    r.n += 1.0;
    r.share += o.share;
    r.pred_loss += o.loss;
    r.safe_d += o.safe_d;
}

fn play_log(rules: &RuleSet, deal_seed: u64) -> Log {
    let bot = StageBot::default();
    let mut state = State::new_deal(*rules, &mut ChaCha8Rng::seed_from_u64(deal_seed));
    let mut rng = ChaCha8Rng::seed_from_u64(deal_seed ^ 0x9e37_79b9);
    let mut log = Log {
        hands: 4.0,
        ..Log::default()
    };
    let mut open: [Option<Open>; 4] = [None; 4];
    let mut last3: [Option<Stage3>; 4] = [None; 4];
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let seat = state.current_player().expect("not over");
        let (action, note) = bot.decide(&state, &legal, &mut rng);
        if let Action::Discard(t) = action {
            if let Some(o) = open[seat].take() {
                close(&mut log.rounds[o.bucket], &o);
            }
            last3[seat] = None;
            if note.is_some_and(|n| n.stage == 3) {
                let danger = Danger::from_belief(&state, seat, &HeuristicBelief);
                let (d, loss) = (danger.tile(t), danger.expected_loss(t));
                let est = Danger::estimate(&state, seat).tile(t);
                let (dc, lc) = giveup::danger_of(&danger, t);
                let (b, c) = (bin(d), bin(dc));
                for (cell, d, loss) in [(&mut log.bins[b], d, loss), (&mut log.cal[c], dc, lc)] {
                    cell.n += 1.0;
                    cell.d += d;
                    cell.loss += loss;
                    cell.est += est;
                }
                last3[seat] = Some(Stage3 {
                    tile: t,
                    bin: b,
                    cal: c,
                });
                let view = SeatView::of(&state, seat);
                let a = after(&view, t);
                if ready(&a) {
                    let outs: Outs = giveup::outs(&a);
                    let candidates: Vec<Tile> = legal
                        .iter()
                        .filter_map(|a| match a {
                            Action::Discard(x) => Some(*x),
                            _ => None,
                        })
                        .collect();
                    let safest = giveup::smallest_loss(&danger, &candidates);
                    open[seat] = Some(Open {
                        bucket: giveup::bucket(state.wall_remaining() / 4),
                        share: outs.live / view.unseen.total().max(1) as f64,
                        loss: lc,
                        safe_d: giveup::danger_of(&danger, safest).0,
                    });
                }
            }
        }
        state.apply(action);
    }
    let result = state.result().expect("over");
    for seat in 0..4 {
        let won = result.wins.iter().find(|w| w.seat == seat);
        let fed = result
            .wins
            .iter()
            .filter(|w| !w.self_drawn && w.from == Some(seat))
            .collect::<Vec<_>>();
        if !fed.is_empty() {
            log.deal_ins += 1.0;
        }
        if let Some(o) = open[seat] {
            let r = &mut log.rounds[o.bucket];
            close(r, &o);
            if result.wins.is_empty() {
                r.wall_out += 1.0;
            } else if let Some(w) = won {
                if w.self_drawn {
                    r.own += 1.0;
                } else {
                    r.theirs += 1.0;
                }
            } else if !fed.is_empty() {
                r.deal_in += 1.0;
                r.paid += fed.iter().map(|w| w.paid_by[seat] as f64).sum::<f64>();
            } else {
                r.other += 1.0;
            }
        }
        if let Some(s) = last3[seat] {
            let hit: Vec<_> = fed
                .iter()
                .filter(|w| !w.robbed && w.win_tile == Some(s.tile))
                .collect();
            if !hit.is_empty() {
                let paid = hit.iter().map(|w| w.paid_by[seat] as f64).sum::<f64>();
                log.stage3_deal_ins += 1.0;
                for cell in [&mut log.bins[s.bin], &mut log.cal[s.cal]] {
                    cell.hits += 1.0;
                    cell.paid += paid;
                }
            }
        }
    }
    log
}

fn print_figures(name: &str, f: &Figures) {
    println!("{name}:");
    println!("  rounds left  own×  theirs×  other ends  later deal-in");
    for (b, label) in BUCKET_NAMES.iter().enumerate() {
        println!(
            "  {:>11}  {:>5.2}  {:>7.2}  {:>9.1}%  {:>12.2}%",
            label,
            f.own[b],
            f.theirs[b],
            100.0 * f.other_ends[b],
            100.0 * f.later_deal_in[b]
        );
    }
    println!(
        "  as Rust: own: {:?}, theirs: {:?}, other_ends: {:?}, later_deal_in: {:?}",
        f.own.map(r4),
        f.theirs.map(r4),
        f.other_ends.map(r4),
        f.later_deal_in.map(r4)
    );
}

fn r4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// Break-even danger for one threatening opponent worth `their` faan: my hand
/// `mine` faan with `tiles` live winning tiles, `rounds` rounds left. Unseen
/// tiles: the wall (4 a round) plus three hands of 13.
fn break_even(
    fig: &Figures,
    rules: &RuleSet,
    mine: u8,
    tiles: f64,
    their: u8,
    rounds: usize,
) -> f64 {
    let outs = Outs {
        live: tiles,
        own_points: rules.points((mine + 1).min(rules.faan_cap)) as f64,
        discard_points: rules.points(mine) as f64,
    };
    let unseen = (4 * rounds + 39) as f64;
    let pay = rules.points(their) as f64;
    let v = giveup::push_value(fig, rounds, &outs, unseen, pay, 1.0);
    giveup::break_even(v, pay)
}

fn print_break_even(fig: &Figures, rules: &RuleSet) {
    for rounds in [6, 2] {
        println!("  {rounds} rounds left (unseen {}):", 4 * rounds + 39);
        println!("    mine  tiles |  vs 3     vs 5     vs 8     vs 10 faan");
        for mine in [3u8, 5, 8] {
            for tiles in [2.0, 4.0, 8.0] {
                let row: Vec<String> = [3u8, 5, 8, 10]
                    .iter()
                    .map(|&th| {
                        format!(
                            "{:>6.1}%",
                            100.0 * break_even(fig, rules, mine, tiles, th, rounds)
                        )
                    })
                    .collect();
                println!("    {mine:>4}  {tiles:>5} | {}", row.join("  "));
            }
        }
    }
}

fn print_bins(title: &str, bins: &[Bin; BINS]) {
    println!("\ndanger check: every stage 3 discard, {title}, in bins");
    println!("  bin          discards  predicted  observed  95% range          obs/pred | table estimate | pred loss  paid (per discard)");
    let mut lo = 0.0;
    let mut points = Vec::new();
    for (i, c) in bins.iter().enumerate() {
        let hi = EDGES.get(i).copied();
        let n = c.n.max(1.0);
        let p = c.hits / n;
        let half = 1.96 * (p * (1.0 - p) / n).sqrt();
        println!(
            "  {:>5.1}-{:<5}  {:>8}  {:>8.2}%  {:>7.2}%  [{:>5.2}, {:>5.2}]%  {:>8.2} | {:>13.2}% | {:>9.3}  {:>6.3}",
            lo,
            hi.map_or("+".into(), |h| format!("{h}%")),
            c.n,
            100.0 * c.d / n,
            100.0 * p,
            100.0 * (p - half).max(0.0),
            100.0 * (p + half),
            p / (c.d / n).max(1e-9),
            100.0 * c.est / n,
            c.loss / n,
            c.paid / n,
        );
        if c.n > 0.0 {
            points.push((r4(c.d / n), r4(p)));
        }
        lo = hi.unwrap_or(0.0);
    }
    println!("  points (mean predicted, observed): {points:?}");
}

fn run_log(deals: u64, seed: u64) {
    let rules = RuleSet::default();
    let start = std::time::Instant::now();
    let log = (0..deals)
        .into_par_iter()
        .map(|i| play_log(&rules, seed.wrapping_mul(1_000_003).wrapping_add(i)))
        .reduce(Log::default, Log::add);
    println!(
        "step 1: {deals} deals of four `stage` bots, seed {seed} ({:?})",
        start.elapsed()
    );
    println!(
        "deal-ins per 100 hands: all {:.2}, from stage 3 discards {:.2}",
        100.0 * log.deal_ins / log.hands,
        100.0 * log.stage3_deal_ins / log.hands
    );
    println!("\nready rounds (a stage 3 discard that leaves the hand ready, to the seat's next discard):");
    println!("  rounds left   rounds  live/unseen  own win  their win  my deal-in  other ends  wall out | pred loss  paid (points per round) | safest tile danger");
    for (b, r) in log.rounds.iter().enumerate() {
        let n = r.n.max(1.0);
        println!(
            "  {:>11}  {:>7}  {:>10.4}  {:>6.2}%  {:>8.2}%  {:>9.2}%  {:>9.2}%  {:>7.2}% | {:>9.3}  {:>6.3} | {:>6.2}%",
            BUCKET_NAMES[b],
            r.n,
            r.share / n,
            100.0 * r.own / n,
            100.0 * r.theirs / n,
            100.0 * r.deal_in / n,
            100.0 * r.other / n,
            100.0 * r.wall_out / n,
            r.pred_loss / n,
            r.paid / n,
            100.0 * r.safe_d / n,
        );
    }
    print_bins("Danger::tile (from_belief, heuristic), raw", &log.bins);
    print_bins("the same, calibrated (giveup::danger_of)", &log.cal);
    let fig = log.figures();
    println!();
    print_figures("measured figures", &fig);
    print_figures("rough figures", &ROUGH);
    println!("\nbreak-even danger (give up above), measured figures:");
    print_break_even(&fig, &rules);
    println!("\nbreak-even danger, rough figures:");
    print_break_even(&ROUGH, &rules);
    if fig != MEASURED {
        println!("\nnote: these figures differ from giveup::MEASURED in the build");
    }
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
        "{me} vs three {opp}, {deals} deals x 4 seats, seed {seed} ({:?})",
        start.elapsed()
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
        Some("log") => run_log(num(1, 400), num(2, 1)),
        Some("match") => run_match(&args[1], &args[2], num(3, 100), num(4, 1)),
        Some("table") => {
            let rules = RuleSet::default();
            println!("break-even danger, measured figures (giveup::MEASURED):");
            print_break_even(&MEASURED, &rules);
            println!("\nbreak-even danger, rough figures:");
            print_break_even(&ROUGH, &rules);
        }
        _ => eprintln!(
            "giveup_log log <deals> [seed] | match <bot> <opponent> <deals> [seed] | table"
        ),
    }
}
