//! The flush suit chosen by the neighbours' suits (plan Task 21,
//! [`mahjong_search::stage::StageBot::with_neighbour`]).
//!
//! `flush_neighbours pair <deals> [seed]`: for `stage-neighbour` and
//! `stage-neighbour-tiles` in turn, two of the test bot and two plain `stage`,
//! seated new, new, old, old in playing order, the table rotated through the
//! four winds on each deal (Brian, 2026-11-21: "if 2v2 and adjacent config, the
//! original version that has the new version as previous seat would have a lower
//! EV than it's copy at its next seat"; "Can just use the second one?"). Each
//! rotation is also played by four plain `stage` on the same wall and random
//! stream (the twin), so every seat has a plain counterpart on the same hand.
//!
//! Per seat role (new after plain, new after new, plain after new, plain after
//! plain): points a hand with a 95 % range over deals (the four rotations of a
//! deal are one sample), win and deal-in rates. The gain figure: the two new
//! bots' mean minus the two plain bots' mean, paired over deals. The paired
//! differences between the two plain bots (上家 new − 上家 plain) and between
//! the two new bots (上家 new − 上家 plain). For the new bots: the share of
//! hands where the rule changed the suit at least once (at a discard or a
//! claim); how often the neighbour reading is right (at each of their stage 1
//! flush discards, every neighbour with a known reading against that
//! neighbour's true leading line, [`mahjong_search::stage::flush_suit`]); and
//! for the changed hands, beside the same seat in the twin: points, win rate,
//! chows claimed from 上家, melds 下家 made with the seat's discards, and the
//! share that ended in a flush (every suited tile of the final hand, concealed
//! and melded, in one suit).
use mahjong_games::{Game, Policy};
use mahjong_hk::{MeldKind, RuleSet, State};
use mahjong_search::stage::{flush_suit, Line, Neighbours, StageBot};
use mahjong_search::{Bot, SeatView};
use mahjong_tiles::TileCounts;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

fn deal_seed(seed: u64, i: u64) -> u64 {
    seed.wrapping_mul(1_000_003).wrapping_add(i)
}

fn stage_of(name: &str) -> StageBot {
    match Bot::by_name(name) {
        Some(Bot::Stage(b)) => b,
        _ => panic!("{name} is no three-stage bot"),
    }
}

/// What happened to one seat in one game.
#[derive(Clone, Copy, Default)]
struct Seat {
    points: f64,
    win: bool,
    deal_in: bool,
    /// The neighbour rule changed the suit at least once.
    changed: bool,
    chows_from_before: u32,
    after_claims: u32,
    flush: bool,
    /// Neighbour readings at stage 1 flush discards: known, right, the
    /// neighbour not on a flush line.
    reads: u32,
    right: u32,
    not_flush: u32,
}

/// Plays one deal with `bots[s]` in seat `s`.
fn play(rules: &RuleSet, bots: &[StageBot; 4], deal: u64, rot: u64) -> [Seat; 4] {
    let mut state = State::new_deal(*rules, &mut ChaCha8Rng::seed_from_u64(deal));
    let mut rng = ChaCha8Rng::seed_from_u64(deal.wrapping_mul(4).wrapping_add(rot));
    let mut out = [Seat::default(); 4];
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let s = state.current_player().expect("not over");
        let bot = &bots[s];
        if bot.neighbour == 0 {
            let a = bot.choose(&state, &legal, &mut rng);
            state.apply(a);
            continue;
        }
        let g = &mut out[s];
        let action = if state.in_claim() || state.in_rob() {
            let (a, swapped) = bot.claim_traced(&state, &legal);
            g.changed |= swapped;
            a
        } else {
            let (a, note) = bot.decide(&state, &legal, &mut rng);
            if let Some(n) = note {
                g.changed |= n.neighbour;
                if n.stage == 1 && n.line == Some(Line::Flush) {
                    let nb = Neighbours::of(&state, s);
                    for (read, other) in [(nb.before, (s + 3) % 4), (nb.after, (s + 1) % 4)] {
                        if let Some(r) = read {
                            g.reads += 1;
                            match flush_suit(&SeatView::of(&state, other)) {
                                Some(t) if t == r => g.right += 1,
                                Some(_) => {}
                                None => g.not_flush += 1,
                            }
                        }
                    }
                }
            }
            a
        };
        state.apply(action);
    }
    let r = state.result().expect("over");
    for (seat, g) in out.iter_mut().enumerate() {
        g.points = r.deltas[seat] as f64;
        g.win = r.wins.iter().any(|w| w.seat == seat);
        g.deal_in = r.wins.iter().any(|w| !w.self_drawn && w.from == Some(seat));
        let (before, after) = ((seat + 3) % 4, (seat + 1) % 4);
        let melds = state.melds(seat);
        g.chows_from_before = melds
            .iter()
            .filter(|m| m.kind == MeldKind::Chow && m.claimed_from == Some(before as u8))
            .count() as u32;
        g.after_claims = state
            .melds(after)
            .iter()
            .filter(|m| m.claimed_from == Some(seat as u8))
            .count() as u32;
        let mut all: TileCounts = *state.hand(seat);
        for m in melds.iter() {
            m.add_to(&mut all);
        }
        let mut suits = all.kinds().filter_map(|t| t.suit());
        g.flush = suits.next().is_some_and(|first| suits.all(|s| s == first));
    }
    out
}

/// Σ x and Σ x² over deals of one per-deal figure.
#[derive(Clone, Copy, Default)]
struct Acc {
    sum: f64,
    sq: f64,
}

impl Acc {
    fn of(x: f64) -> Acc {
        Acc { sum: x, sq: x * x }
    }
    fn add(self, o: Acc) -> Acc {
        Acc {
            sum: self.sum + o.sum,
            sq: self.sq + o.sq,
        }
    }
    /// Mean and 95 % range over `n` samples.
    fn mean_ci(&self, n: f64) -> (f64, f64) {
        let mean = self.sum / n;
        let var = (self.sq / n - mean * mean).max(0.0);
        (mean, 1.96 * (var / n).sqrt())
    }
}

/// Sums for one test bot over deals.
#[derive(Clone, Copy, Default)]
struct Sum {
    deals: f64,
    /// Per role: points (per-deal mean over rotations), and against the twin.
    role: [Acc; 4],
    role_vs_twin: [Acc; 4],
    wins: [f64; 4],
    deal_ins: [f64; 4],
    /// New mean − plain mean; plain 上家 new − 上家 plain; new likewise.
    gain: Acc,
    d_old: Acc,
    d_new: Acc,
    /// New-bot hands, changed hands, readings.
    hands: f64,
    changed: f64,
    reads: f64,
    right: f64,
    not_flush: f64,
    /// Changed hands: this seat / the twin's same seat: points, win, chows from
    /// 上家, 下家's claims, flush; and Σ of the squared points difference.
    ch_new: [f64; 5],
    ch_old: [f64; 5],
    ch_diff_sq: f64,
}

impl Sum {
    fn add(mut self, o: Sum) -> Sum {
        self.deals += o.deals;
        for r in 0..4 {
            self.role[r] = self.role[r].add(o.role[r]);
            self.role_vs_twin[r] = self.role_vs_twin[r].add(o.role_vs_twin[r]);
            self.wins[r] += o.wins[r];
            self.deal_ins[r] += o.deal_ins[r];
        }
        self.gain = self.gain.add(o.gain);
        self.d_old = self.d_old.add(o.d_old);
        self.d_new = self.d_new.add(o.d_new);
        self.hands += o.hands;
        self.changed += o.changed;
        self.reads += o.reads;
        self.right += o.right;
        self.not_flush += o.not_flush;
        for k in 0..5 {
            self.ch_new[k] += o.ch_new[k];
            self.ch_old[k] += o.ch_old[k];
        }
        self.ch_diff_sq += o.ch_diff_sq;
        self
    }
}

fn stats(g: &Seat) -> [f64; 5] {
    [
        g.points,
        g.win as u8 as f64,
        g.chows_from_before as f64,
        g.after_claims as f64,
        g.flush as u8 as f64,
    ]
}

/// One deal for one test bot, given the twin games of its four rotations.
fn deal_sum(
    rules: &RuleSet,
    new: StageBot,
    old: StageBot,
    deal: u64,
    twins: &[[Seat; 4]; 4],
) -> Sum {
    let mut s = Sum {
        deals: 1.0,
        ..Sum::default()
    };
    let (mut role, mut vs) = ([0.0f64; 4], [0.0f64; 4]);
    for k in 0..4usize {
        // Seats k and k+1 new, k+2 and k+3 old (playing order s → s+1).
        let mut bots = [old; 4];
        bots[k] = new;
        bots[(k + 1) % 4] = new;
        let g = play(rules, &bots, deal, k as u64);
        for r in 0..4 {
            let seat = (k + r) % 4;
            let (x, y) = (&g[seat], &twins[k][seat]);
            role[r] += x.points / 4.0;
            vs[r] += (x.points - y.points) / 4.0;
            s.wins[r] += x.win as u8 as f64;
            s.deal_ins[r] += x.deal_in as u8 as f64;
            if r < 2 {
                s.hands += 1.0;
                s.reads += x.reads as f64;
                s.right += x.right as f64;
                s.not_flush += x.not_flush as f64;
                if x.changed {
                    s.changed += 1.0;
                    let (a, b) = (stats(x), stats(y));
                    for i in 0..5 {
                        s.ch_new[i] += a[i];
                        s.ch_old[i] += b[i];
                    }
                    s.ch_diff_sq += (a[0] - b[0]).powi(2);
                }
            }
        }
    }
    for r in 0..4 {
        s.role[r] = Acc::of(role[r]);
        s.role_vs_twin[r] = Acc::of(vs[r]);
    }
    s.gain = Acc::of((role[0] + role[1] - role[2] - role[3]) / 2.0);
    s.d_old = Acc::of(role[2] - role[3]);
    s.d_new = Acc::of(role[1] - role[0]);
    s
}

fn report(name: &str, s: &Sum) {
    let n = s.deals;
    println!("  {name}:");
    let labels = [
        "new, 上家 plain",
        "new, 上家 new",
        "plain, 上家 new",
        "plain, 上家 plain",
    ];
    for (r, label) in labels.iter().enumerate() {
        let (m, ci) = s.role[r].mean_ci(n);
        let (d, dci) = s.role_vs_twin[r].mean_ci(n);
        println!(
            "    {:<18} points/hand {m:+.3} ± {ci:.3}  win {:.2}%  deal-in {:.2}%  vs the same seat in the all-plain twin {d:+.3} ± {dci:.3}",
            label,
            100.0 * s.wins[r] / (4.0 * n),
            100.0 * s.deal_ins[r] / (4.0 * n),
        );
    }
    let (g, gci) = s.gain.mean_ci(n);
    println!("    gain: new bots' mean − plain bots' mean {g:+.3} ± {gci:.3} points a hand (paired over deals)");
    let (d, ci) = s.d_old.mean_ci(n);
    println!("    plain bots, 上家 new − 上家 plain: {d:+.3} ± {ci:.3}");
    let (d, ci) = s.d_new.mean_ci(n);
    println!("    new bots, 上家 new − 上家 plain: {d:+.3} ± {ci:.3}");
    println!(
        "    new-bot hands where the rule changed the suit: {:.0} of {:.0} ({:.2}%)",
        s.changed,
        s.hands,
        100.0 * s.changed / s.hands
    );
    println!(
        "    neighbour readings at stage 1 flush discards: {:.0}, right {:.1}%, neighbour not on a flush line {:.1}%, other suit {:.1}%",
        s.reads,
        100.0 * s.right / s.reads.max(1.0),
        100.0 * s.not_flush / s.reads.max(1.0),
        100.0 * (s.reads - s.right - s.not_flush) / s.reads.max(1.0)
    );
    let c = s.changed.max(1.0);
    let dmean = (s.ch_new[0] - s.ch_old[0]) / c;
    let dvar = (s.ch_diff_sq / c - dmean * dmean).max(0.0);
    println!(
        "    changed hands ({:.0}), this seat / the same seat in the twin: points {:+.2} / {:+.2} (diff {:+.2} ± {:.2}), win {:.1}% / {:.1}%, chows from 上家 {:.3} / {:.3}, melds 下家 made with its discards {:.3} / {:.3}, ended in a flush {:.1}% / {:.1}%",
        s.changed,
        s.ch_new[0] / c,
        s.ch_old[0] / c,
        dmean,
        1.96 * (dvar / c).sqrt(),
        100.0 * s.ch_new[1] / c,
        100.0 * s.ch_old[1] / c,
        s.ch_new[2] / c,
        s.ch_old[2] / c,
        s.ch_new[3] / c,
        s.ch_old[3] / c,
        100.0 * s.ch_new[4] / c,
        100.0 * s.ch_old[4] / c,
    );
}

fn run_pair(deals: u64, seed: u64) {
    let rules = RuleSet::default();
    let names = ["stage-neighbour", "stage-neighbour-tiles"];
    let old = stage_of("stage");
    let news = names.map(stage_of);
    let start = std::time::Instant::now();
    let sums = (0..deals)
        .into_par_iter()
        .map(|i| {
            let deal = deal_seed(seed, i);
            let twins: [[Seat; 4]; 4] =
                std::array::from_fn(|k| play(&rules, &[old; 4], deal, k as u64));
            news.map(|new| deal_sum(&rules, new, old, deal, &twins))
        })
        .reduce(
            || [Sum::default(); 2],
            |a, b| [a[0].add(b[0]), a[1].add(b[1])],
        );
    println!(
        "pair: two new bots and two plain stage seated new, new, old, old in playing order, rotated through the winds; {deals} deals x 4 rotations, seed {seed} ({:?})",
        start.elapsed()
    );
    for (name, s) in names.iter().zip(&sums) {
        report(name, s);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize, d: u64| args.get(i).and_then(|x| x.parse().ok()).unwrap_or(d);
    match args.first().map(String::as_str) {
        Some("pair") => run_pair(num(1, 100), num(2, 21)),
        _ => eprintln!("flush_neighbours pair <deals> [seed]"),
    }
}
