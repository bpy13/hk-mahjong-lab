//! Fits the passed-tile safety and risk-taking constants (plan 2026-11-09 Tasks 14–15,
//! `docs/race-bot.md` §13) from full-information self-play, and prints them as Rust.
//!
//! ```text
//! cargo run --release -p mahjong-search --example fit_passed -- \
//!     --hands 1500 --seed 1 [--threads 1] [--belief models/belief.mjnn]
//! ```
//!
//! Each hand seats four bots drawn uniformly from `race`, `value`, `defensive`,
//! `fast`, `closed` and `efficiency` (a mix of pushers and folders). At every
//! turn-phase decision the player to act is the observer; its view is public plus
//! its own hand, the truth is read from the hidden hands:
//!
//! * **f(N)**: for each opponent `o` and each kind `t` the observer has not seen out,
//!   the belief's `P_o(t) = min(tenpai × wait, 0.95)` against whether `o` really waits
//!   on `t` (`State::wins_on`), bucketed by `Passed`'s record of `t` for `o`: did not
//!   pass; passed and locked (no turn since); passed with `N` 入章 since. The ratio
//!   truth / estimate in each bucket, divided by the same ratio for tiles that did not
//!   pass, is `PASSED_FACTOR[N]`. The table is printed for every 換章 test (the
//!   module's and the alternatives below), with the bucket of tiles the test reset.
//! * **換章 tests**: for every 入章 discard, whether the seat's waits or faan-aware
//!   shanten changed against its hand after its previous discard, and whether it now
//!   waits on a tile it had let pass; per test, the rate among discards where the test
//!   fires, against all 入章 and all 摸打.
//! * **Risk read**: per bin of the opponent's risk score, the true fold rate, the true
//!   ready and one-away rates against the belief's `tenpai` and `min(1.5 tenpai, 1 −
//!   tenpai)`, and the discard-win points of ready opponents against the belief's
//!   `Pay(o)`; the multipliers are each bin's ratio over the all-bins ratio.
//!
//! With `--belief`, the estimates come from that model (for `RISK_MODEL`); without,
//! from the heuristic tables (`PASSED_FACTOR`, `RISK_HEURISTIC`).

use mahjong_games::{Game, Policy};
use mahjong_hk::{score_win, Action, RuleSet, State, WinContext};
use mahjong_search::danger::PROGRESS_BUCKETS;
use mahjong_search::passed::{reshaped, risk_scores, DiscardContext, Passed, Reshape};
use mahjong_search::race::{hand_shanten, LabelTracker, TrueState, RISK_HEURISTIC};
use mahjong_search::{Belief, BeliefModel, Bot, HeuristicBelief, SeatView};
use mahjong_tiles::Tile;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const FIELD: &[&str] = &["race", "value", "defensive", "fast", "closed", "efficiency"];
const P_CLIP: f64 = 0.95;
/// `N` buckets: 0, 1, 2, 3, 4+.
const N_BUCKETS: usize = 5;
/// A risk-bin multiplier needs this many positive cases (folds, ready seats, ...)
/// behind it, else the bin keeps 1.0.
const MIN_CASES: f64 = 30.0;

/// The alternative 換章 tests (bits 1.. of a cell's `reset`); bit 0 is the module's.
const EXTRA: &[(&str, Reshape)] = &[
    ("inner 2-8 from hand", |c| {
        c.from_hand && c.tile.number().is_some_and(|n| (2..=8).contains(&n))
    }),
    ("middle 3-7 from hand", |c| {
        c.from_hand && c.tile.number().is_some_and(|n| (3..=7).contains(&n))
    }),
    (
        "fresh middle 3-7 (kind and neighbours never discarded)",
        |c| {
            c.from_hand
                && c.tile.number().is_some_and(|n| (3..=7).contains(&n))
                && !c.discarded[c.tile.index() as usize]
                && [-1i8, 1].iter().all(|&d| {
                    c.tile
                        .offset(d)
                        .is_none_or(|u| !c.discarded[u.index() as usize])
                })
        },
    ),
    ("any tile from hand", |c| c.from_hand),
    ("terminal or honour from hand", |c| {
        c.from_hand && c.tile.is_terminal_or_honour()
    }),
];

#[derive(Clone, Copy, Default)]
struct Bucket {
    n: f64,
    pred: f64,
    truth: f64,
    legal: f64,
}

impl Bucket {
    fn add(&mut self, pred: f64, truth: bool, legal: bool) {
        self.n += 1.0;
        self.pred += pred;
        self.truth += truth as u8 as f64;
        self.legal += legal as u8 as f64;
    }
    fn merge(&mut self, o: &Bucket) {
        self.n += o.n;
        self.pred += o.pred;
        self.truth += o.truth;
        self.legal += o.legal;
    }
    fn ratio(&self) -> f64 {
        if self.pred > 0.0 {
            self.truth / self.pred
        } else {
            f64::NAN
        }
    }
}

/// One 換章 test's tally over the discards where it fired.
#[derive(Clone, Copy, Default)]
struct Reshaped {
    n: f64,
    changed: f64,
    closer: f64,
    waits_passed: f64,
}

impl Reshaped {
    fn merge(&mut self, o: &Reshaped) {
        self.n += o.n;
        self.changed += o.changed;
        self.closer += o.closer;
        self.waits_passed += o.waits_passed;
    }
}

#[derive(Clone, Copy, Default)]
struct RiskBin {
    n: f64,
    fold: f64,
    ready: f64,
    one: f64,
    tenpai_pred: f64,
    one_pred: f64,
    pay_n: f64,
    pay_pred: f64,
    pay_true: f64,
}

impl RiskBin {
    fn merge(&mut self, o: &RiskBin) {
        self.n += o.n;
        self.fold += o.fold;
        self.ready += o.ready;
        self.one += o.one;
        self.tenpai_pred += o.tenpai_pred;
        self.one_pred += o.one_pred;
        self.pay_n += o.pay_n;
        self.pay_pred += o.pay_pred;
        self.pay_true += o.pay_true;
    }
}

/// Per 換章 test variant (module's test first, then `EXTRA`, then "never reset"):
/// the unpassed, locked, `N` and reset buckets.
#[derive(Clone)]
struct Stats {
    unpassed: Bucket,
    locked: Bucket,
    by_n: Vec<[Bucket; N_BUCKETS]>,
    reset: Vec<Bucket>,
    reshaped: Vec<Reshaped>,
    from_hand: Reshaped,
    drawn_out: Reshaped,
    /// By wall-progress bucket, then risk bin: the risk read's truth. Folds and
    /// readiness both rise late in the hand, as the score's spread does, so the
    /// multipliers compare bins within a wall bucket.
    risk: [[RiskBin; 5]; PROGRESS_BUCKETS],
    /// Risk scores seen, in 0.01 steps from −0.2 (index 0) to 0.2 (index 40 holds
    /// everything above), to choose the bin edges.
    histogram: [f64; 41],
    hands: usize,
    decisions: usize,
}

impl Stats {
    fn new() -> Stats {
        let v = 1 + EXTRA.len();
        Stats {
            unpassed: Bucket::default(),
            locked: Bucket::default(),
            by_n: vec![[Bucket::default(); N_BUCKETS]; v + 1],
            reset: vec![Bucket::default(); v],
            reshaped: vec![Reshaped::default(); v],
            from_hand: Reshaped::default(),
            drawn_out: Reshaped::default(),
            risk: [[RiskBin::default(); 5]; PROGRESS_BUCKETS],
            histogram: [0.0; 41],
            hands: 0,
            decisions: 0,
        }
    }

    fn merge(&mut self, o: &Stats) {
        self.unpassed.merge(&o.unpassed);
        self.locked.merge(&o.locked);
        for (a, b) in self.by_n.iter_mut().zip(&o.by_n) {
            for (x, y) in a.iter_mut().zip(b) {
                x.merge(y);
            }
        }
        for (a, b) in self.reset.iter_mut().zip(&o.reset) {
            a.merge(b);
        }
        for (a, b) in self.reshaped.iter_mut().zip(&o.reshaped) {
            a.merge(b);
        }
        self.from_hand.merge(&o.from_hand);
        self.drawn_out.merge(&o.drawn_out);
        for (a, b) in self.risk.iter_mut().zip(&o.risk) {
            for (x, y) in a.iter_mut().zip(b) {
                x.merge(y);
            }
        }
        for (a, b) in self.histogram.iter_mut().zip(&o.histogram) {
            *a += b;
        }
        self.hands += o.hands;
        self.decisions += o.decisions;
    }
}

/// Points per share of `seat`'s best discard win now (0 if not ready).
fn best_discard_points(state: &State, seat: usize) -> f64 {
    let rules = state.rules();
    let mut best = 0.0f64;
    for k in Tile::all_kinds().filter(|&k| state.wins_on(seat, k)) {
        let mut c = *state.hand(seat);
        c.add(k);
        let ctx = WinContext {
            seat,
            win_tile: Some(k),
            self_drawn: false,
            bonus: state.bonus(seat),
            last_tile: false,
            kong_chain: 0,
            robbing_kong: false,
            heavenly: false,
            earthly: false,
        };
        if let Some(s) = score_win(&c, state.melds(seat), &ctx, rules) {
            best = best.max(rules.points(s.faan.min(rules.faan_cap)) as f64);
        }
    }
    best
}

/// `danger.rs`'s wall-progress bucket (0 = early … 5 = late), as `race.rs`.
fn wall_bucket(state: &State) -> usize {
    let initial = if state.rules().flowers { 91.0 } else { 83.0 };
    let progress = (1.0 - state.wall_remaining() as f64 / initial).clamp(0.0, 0.999);
    (progress * PROGRESS_BUCKETS as f64) as usize
}

/// The 34-bit set of tiles `seat` wins on, and its faan-aware shanten.
fn waits_of(state: &State, seat: usize) -> (u64, i8) {
    let mut m = 0u64;
    for k in Tile::all_kinds() {
        if state.wins_on(seat, k) {
            m |= 1 << k.index();
        }
    }
    (m, hand_shanten(&SeatView::of(state, seat)))
}

fn measure(state: &State, me: usize, tracker: &LabelTracker, belief: &dyn Belief, st: &mut Stats) {
    let extra: Vec<Reshape> = EXTRA.iter().map(|x| x.1).collect();
    let passed = Passed::of_with(state, &extra);
    let risk = risk_scores(state);
    let beliefs = belief.estimate(state, me);
    let unseen = state.unseen(me);
    let rules = state.rules();
    let variants = 1 + EXTRA.len();
    st.decisions += 1;
    for o in (0..4).filter(|&o| o != me) {
        let b = &beliefs.seats[o];
        let tenpai = b.tenpai.clamp(0.0, 1.0);
        // The risk read.
        let bin = RISK_HEURISTIC.bin(risk[o]);
        st.histogram[(((risk[o] + 0.2) / 0.01).floor().clamp(0.0, 40.0)) as usize] += 1.0;
        let r = &mut st.risk[wall_bucket(state)][bin];
        let truth = tracker.state_of(state, o);
        r.n += 1.0;
        r.fold += tracker.folded(o) as u8 as f64;
        r.ready += (truth == TrueState::Ready) as u8 as f64;
        r.one += (truth == TrueState::OneAway) as u8 as f64;
        r.tenpai_pred += tenpai;
        r.one_pred += (1.5 * tenpai).min(1.0 - tenpai);
        if truth == TrueState::Ready {
            let pay: f64 = b
                .faan
                .iter()
                .enumerate()
                .map(|(k, &pb)| pb * rules.points((3 + k as u8).min(rules.faan_cap)) as f64)
                .sum();
            r.pay_n += 1.0;
            r.pay_pred += pay;
            r.pay_true += best_discard_points(state, o);
        }
        // Passed tiles.
        for k in Tile::all_kinds() {
            if unseen[k] == 0 {
                continue;
            }
            let i = k.index() as usize;
            let pred = (tenpai * b.wait[i]).min(P_CLIP);
            let waits = state.wins_on(o, k);
            let legal = waits && !state.passed_win_locked(o, k);
            let c = passed.cell(o, k);
            if !c.passed {
                st.unpassed.add(pred, waits, legal);
                continue;
            }
            if !c.turned {
                st.locked.add(pred, waits, legal);
                continue;
            }
            let n = (c.changes as usize).min(N_BUCKETS - 1);
            for v in 0..variants {
                if c.reset & (1 << v) != 0 {
                    st.reset[v].add(pred, waits, legal);
                } else {
                    st.by_n[v][n].add(pred, waits, legal);
                }
            }
            st.by_n[variants][n].add(pred, waits, legal);
        }
    }
}

fn play(hands: usize, seed: u64, bots: &[Bot], belief: &dyn Belief) -> Stats {
    let mut st = Stats::new();
    let mut rng = SmallRng::seed_from_u64(seed);
    let rules = RuleSet::default();
    for _ in 0..hands {
        let seats: Vec<usize> = (0..4)
            .map(|_| rng.gen_range(0..bots.len() as u32) as usize)
            .collect();
        let mut s = State::new_deal(rules, &mut rng);
        let mut tracker = LabelTracker::new(&s);
        let mut last: [Option<(u64, i8)>; 4] = [None; 4];
        while let Some(p) = s.current_player() {
            let legal = s.legal_actions_vec();
            let turn = s.phase_name() == "turn";
            if turn {
                measure(&s, p, &tracker, belief, &mut st);
            }
            let a = bots[seats[p]].choose(&s, &legal, &mut rng);
            tracker.observe(&s, a);
            let reshape = match (a, turn) {
                (Action::Discard(t), true) => {
                    let from_hand = s.last_drawn() != Some(t);
                    let ctx = DiscardContext::now(&s, p, t, from_hand);
                    let fired: Vec<bool> = std::iter::once(reshaped(&ctx))
                        .chain(EXTRA.iter().map(|x| (x.1)(&ctx)))
                        .collect();
                    let before = Passed::of(&s);
                    let passed_mask: u64 = Tile::all_kinds()
                        .filter(|&k| {
                            let c = before.cell(p, k);
                            c.passed && c.reset & 1 == 0
                        })
                        .fold(0, |m, k| m | 1 << k.index());
                    Some((from_hand, fired, passed_mask))
                }
                _ => None,
            };
            s.apply(a);
            if let Some((from_hand, fired, passed_mask)) = reshape {
                let now = waits_of(&s, p);
                if let Some(old) = last[p] {
                    let changed = now != old;
                    let closer = now.1 < old.1;
                    let waits_passed = now.0 & passed_mask != 0;
                    let tally = |r: &mut Reshaped| {
                        r.n += 1.0;
                        r.changed += changed as u8 as f64;
                        r.closer += closer as u8 as f64;
                        r.waits_passed += waits_passed as u8 as f64;
                    };
                    if from_hand {
                        tally(&mut st.from_hand);
                    } else {
                        tally(&mut st.drawn_out);
                    }
                    for (v, &f) in fired.iter().enumerate() {
                        if f {
                            tally(&mut st.reshaped[v]);
                        }
                    }
                }
                last[p] = Some(now);
            }
        }
        st.hands += 1;
    }
    st
}

fn pct(x: f64, n: f64) -> String {
    if n > 0.0 {
        format!("{:5.1}%", 100.0 * x / n)
    } else {
        "    -".into()
    }
}

fn main() {
    let mut hands = 1500usize;
    let mut seed = 1u64;
    let mut threads = 1usize;
    let mut belief_path: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_else(|| panic!("{k} needs a value"));
        match k.as_str() {
            "--hands" => hands = v.parse().expect("hands"),
            "--seed" => seed = v.parse().expect("seed"),
            "--threads" => threads = v.parse::<usize>().expect("threads").max(1),
            "--belief" => belief_path = Some(v),
            _ => panic!("unknown option {k}"),
        }
    }
    let belief: Arc<dyn Belief> = match &belief_path {
        Some(p) => Arc::new(BeliefModel::load(p).expect("belief model")),
        None => Arc::new(HeuristicBelief),
    };
    let bots: Vec<Bot> = FIELD.iter().map(|n| Bot::by_name(n).expect(n)).collect();
    let start = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let per = hands.div_ceil(threads);
    let stats = pool.install(|| {
        (0..threads)
            .into_par_iter()
            .map(|i| {
                let n = per.min(hands.saturating_sub(i * per));
                play(n, seed * 1000 + i as u64, &bots, belief.as_ref())
            })
            .reduce(Stats::new, |mut a, b| {
                a.merge(&b);
                a
            })
    });
    let secs = start.elapsed().as_secs_f64();
    println!(
        "fit_passed: {} hands, {} decisions, seed {seed}, belief {}, {secs:.0} s on {threads} thread(s)",
        stats.hands,
        stats.decisions,
        belief_path.as_deref().unwrap_or("heuristic")
    );

    // f(N).
    let base = stats.unpassed.ratio();
    println!(
        "\n== Passed tiles: truth / estimate per bucket (relative to tiles that did not pass) =="
    );
    println!(
        "{:<44} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "bucket", "n", "P(wait)", "est", "ratio", "f"
    );
    let row = |name: &str, b: &Bucket| {
        println!(
            "{:<44} {:>10.0} {:>8.4} {:>8.4} {:>8.3} {:>8.3}",
            name,
            b.n,
            if b.n > 0.0 { b.truth / b.n } else { f64::NAN },
            if b.n > 0.0 { b.pred / b.n } else { f64::NAN },
            b.ratio(),
            b.ratio() / base
        );
    };
    row("did not pass", &stats.unpassed);
    row("passed, locked (no turn since)", &stats.locked);
    let names: Vec<String> = std::iter::once("module test: flush-suit tile from hand".to_string())
        .chain(EXTRA.iter().map(|x| x.0.to_string()))
        .chain(std::iter::once("never reset".to_string()))
        .collect();
    for (v, name) in names.iter().enumerate() {
        println!("-- 換章 test: {name}");
        for (n, b) in stats.by_n[v].iter().enumerate() {
            let label = if n + 1 == N_BUCKETS {
                format!("  passed, N >= {n} 入章 since")
            } else {
                format!("  passed, N = {n} 入章 since")
            };
            row(&label, b);
        }
        if v < stats.reset.len() {
            row("  reset by the test", &stats.reset[v]);
        }
    }
    let f: Vec<f64> = stats.by_n[0]
        .iter()
        .map(|b| (b.ratio() / base).clamp(0.0, 1.0))
        .collect();
    println!(
        "\npub const PASSED_FACTOR: [f64; 5] = [{}];",
        f.iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "(locked bucket legal-win rate: {:.5}, waiting rate {:.5}; the lock gives 0 by rule)",
        if stats.locked.n > 0.0 {
            stats.locked.legal / stats.locked.n
        } else {
            f64::NAN
        },
        if stats.locked.n > 0.0 {
            stats.locked.truth / stats.locked.n
        } else {
            f64::NAN
        }
    );

    // 換章 accuracy.
    println!("\n== 換章 tests: what the discard really meant (rates among discards where the test fires) ==");
    println!(
        "{:<56} {:>8} {:>9} {:>9} {:>14}",
        "test", "n", "changed", "closer", "waits a passed"
    );
    let rrow = |name: &str, r: &Reshaped| {
        println!(
            "{:<56} {:>8.0} {:>9} {:>9} {:>14}",
            name,
            r.n,
            pct(r.changed, r.n),
            pct(r.closer, r.n),
            pct(r.waits_passed, r.n)
        );
    };
    rrow("all 摸打 (drawn tile out)", &stats.drawn_out);
    rrow("all 入章 (from hand)", &stats.from_hand);
    for (v, r) in stats.reshaped.iter().enumerate() {
        rrow(&names[v], r);
    }

    // The risk read.
    println!("\n== Risk scores seen (share per 0.01 step; the last holds everything above 0.2) ==");
    let total: f64 = stats.histogram.iter().sum();
    let mut cum = 0.0;
    for (i, h) in stats.histogram.iter().enumerate() {
        cum += h;
        if *h > 0.0 {
            println!(
                "{:>6.2}: {:>6.2}%  cumulative {:>6.2}%",
                -0.2 + 0.01 * i as f64,
                100.0 * h / total,
                100.0 * cum / total
            );
        }
    }
    println!("\n== Risk read: truth by risk-score bin (all wall buckets together) ==");
    let rate = |x: f64, n: f64| if n > 0.0 { x / n } else { f64::NAN };
    // Per risk bin over every wall bucket, and per wall bucket over every bin.
    let by_bin: Vec<RiskBin> = (0..5)
        .map(|b| {
            stats.risk.iter().fold(RiskBin::default(), |mut a, w| {
                a.merge(&w[b]);
                a
            })
        })
        .collect();
    let by_wall: Vec<RiskBin> = stats
        .risk
        .iter()
        .map(|w| {
            w.iter().fold(RiskBin::default(), |mut a, b| {
                a.merge(b);
                a
            })
        })
        .collect();
    println!(
        "{:<14} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "bin", "n", "fold", "ready", "tenpai", "one", "one_est", "pay", "pay_est", "pay_n"
    );
    let label = |i: usize| match i {
        0 => format!("< {}", RISK_HEURISTIC.edges[0]),
        4 => format!(">= {}", RISK_HEURISTIC.edges[3]),
        _ => format!(
            "{}-{}",
            RISK_HEURISTIC.edges[i - 1],
            RISK_HEURISTIC.edges[i]
        ),
    };
    for (i, r) in by_bin.iter().enumerate() {
        println!(
            "{:<14} {:>9.0} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.1} {:>8.1} {:>8.0}",
            label(i),
            r.n,
            rate(r.fold, r.n),
            rate(r.ready, r.n),
            rate(r.tenpai_pred, r.n),
            rate(r.one, r.n),
            rate(r.one_pred, r.n),
            rate(r.pay_true, r.pay_n),
            rate(r.pay_pred, r.pay_n),
            r.pay_n
        );
    }
    // The multipliers, controlled for wall progress: in each bin, the truth
    // observed over what the wall bucket's own truth / estimate ratio would
    // predict for the same cases (indirect standardisation).
    println!(
        "\n== Risk read: multipliers within wall buckets (truth over the bucket's own rate) =="
    );
    println!(
        "{:<14} {:>8} {:>8} {:>8} {:>8}",
        "bin", "fold", "tenpai", "one", "pay"
    );
    let lift = |truth: fn(&RiskBin) -> f64, base: fn(&RiskBin) -> f64, b: usize| -> (f64, f64) {
        let (mut seen, mut expected) = (0.0, 0.0);
        for (w, row) in stats.risk.iter().enumerate() {
            let cell = &row[b];
            seen += truth(cell);
            expected += base(cell) * rate(truth(&by_wall[w]), base(&by_wall[w])).max(0.0);
        }
        (
            seen,
            if expected > 0.0 {
                seen / expected
            } else {
                f64::NAN
            },
        )
    };
    // A bin with few cases keeps 1.0; a multiplier stays within [0.25, 4].
    let guard = |(cases, x): (f64, f64)| {
        if cases >= MIN_CASES && x.is_finite() {
            x.clamp(0.25, 4.0)
        } else {
            1.0
        }
    };
    let (mut mf, mut mt, mut mo, mut mp) = (vec![], vec![], vec![], vec![]);
    for b in 0..5 {
        let f = guard(lift(|r| r.fold, |r| r.n, b));
        let t = guard(lift(|r| r.ready, |r| r.tenpai_pred, b));
        let o = guard(lift(|r| r.one, |r| r.one_pred, b));
        let p = guard(lift(|r| r.pay_true, |r| r.pay_pred, b));
        println!(
            "{:<14} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            label(b),
            f,
            t,
            o,
            p
        );
        mf.push(f);
        mt.push(t);
        mo.push(o);
        mp.push(p);
    }
    let fmt = |v: &[f64]| {
        v.iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!(
        "\npub const RISK_{}: RiskRead = RiskRead {{\n    edges: [{}],\n    fold: [{}],\n    tenpai: [{}],\n    one_away: [{}],\n    pay: [{}],\n}};",
        if belief_path.is_some() { "MODEL" } else { "HEURISTIC" },
        RISK_HEURISTIC
            .edges
            .iter()
            .map(|x| format!("{x}"))
            .collect::<Vec<_>>()
            .join(", "),
        fmt(&mf),
        fmt(&mt),
        fmt(&mo),
        fmt(&mp)
    );
}
