//! Plan 2026-11-16 Task 24: where the defence bots lose points to false alarms,
//! hand by hand ([`mahjong_search::false_alarm`]).
//!
//! `defence_false_alarms <deals> [seed] [model] [bot ...]`: each deal is played
//! with four plain `stage` bots (one game per seat, the set-up of
//! `defence_ceiling.rs`), then with each bot in each seat against three plain
//! `stage`. Bots (all reading `model`, default `models/belief-order.mjnn`):
//! `safe1` (`stage-safe:t=1`), `safe5` (`stage-safe:t=5`), `ev` (`stage-ev`),
//! `fold` (`stage-fold`, Task 11's give-up rule); default all four.
//!
//! At each bot hand's first change the true hands label it a true alarm
//! (`stage`'s pick would have been won on at once) or a false alarm. Prints per
//! bot: the paired points a hand against plain `stage` in the same seat on the
//! same wall (95 % range over deals; the four seats of a deal are one sample),
//! the share of hands and decisions changed, the true / false split of the gain
//! with its sum, the false-alarm loss split by the bot's distance from ready,
//! wall third, the flagged opponent, the reading's figure, the progress the
//! replacement cost and the ending beside the plain game; then the costliest
//! false alarms across the main kinds, written out in tiles.
use mahjong_games::Game;
use mahjong_hk::{HandResult, RuleSet, State};
use mahjong_search::belief::BeliefModel;
use mahjong_search::false_alarm::{
    example_json, play, play_with, true_distance, Played, Reading, OUTCOMES,
};
use mahjong_search::oracle::{discards, distance};
use mahjong_search::stage::{read_dealin, StageBot};
use mahjong_search::{Agent, Bot, SeatView};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// The bots of the plan: (short name, agent spec without the model).
const BOTS: [(&str, &str); 4] = [
    ("safe1", "stage-safe:t=1"),
    ("safe5", "stage-safe:t=5"),
    ("ev", "stage-ev"),
    ("fold", "stage-fold"),
];

/// Ending of a false-alarm hand beside the plain game.
const ENDINGS: [&str; 6] = [
    "liable for a win",
    "win lost",
    "later deal-in",
    "other's self-draw",
    "same ending",
    "better ending",
];

/// The ending kind of a bot hand beside its plain hand (liability first: the
/// bot paid for another seat's win, §8.3, and the plain game did not).
fn ending(r: &Rec) -> usize {
    let (plain, bot) = (r.plain_out, r.out);
    if r.liable && !r.plain_liable {
        0
    } else if plain == 0 && bot != 0 {
        1
    } else if bot == 1 && plain != 1 {
        2
    } else if bot == 2 && plain != 2 {
        3
    } else if bot == plain {
        4
    } else {
        5
    }
}

/// `seat` paid for a whole win as the liable player.
fn liable(r: &HandResult, seat: usize) -> bool {
    r.wins.iter().any(|w| w.liable == Some(seat))
}

/// What the log reads at a first change.
#[derive(Clone, Copy, Default)]
struct Info {
    /// 0 true alarm, 1 false alarm, 2 not a discard.
    kind: usize,
    /// The bot's distance from ready (14 tiles, best discard).
    dist: i8,
    /// Wall third (0..3).
    third: usize,
    /// The reading's figure for `stage`'s pick.
    fig: f64,
    /// The flagged opponent (highest read deal-in on the pick): its open sets
    /// and its true distance from ready.
    opp_melds: usize,
    opp_dist: i8,
    /// Distance after the tile thrown minus after `stage`'s pick (13 tiles).
    cost: i8,
    /// `stage`'s pick paid this at once (true alarms).
    paid: i32,
}

/// One bot hand beside its plain hand.
#[derive(Clone, Copy)]
struct Rec {
    deal: u64,
    seat: usize,
    plain_points: i32,
    plain_out: usize,
    points: i32,
    out: usize,
    liable: bool,
    plain_liable: bool,
    discards: u32,
    changed: u32,
    changed_discards: u32,
    info: Option<Info>,
}

impl Rec {
    fn delta(&self) -> f64 {
        (self.points - self.plain_points) as f64
    }
}

/// One deal's records (per seat, per bot) and its plain hands (points,
/// outcome, discards) per seat.
type DealRows = (Vec<Vec<Rec>>, Vec<(i32, usize, u32)>);

fn deal_seed(seed: u64, i: u64) -> u64 {
    seed.wrapping_mul(1_000_003).wrapping_add(i)
}

/// The log's reading of a bot hand's first change.
fn info_of(p: &Played, seat: usize, wall0: usize, model: &BeliefModel) -> Option<Info> {
    let c = p.change.as_ref()?;
    let state = &c.state;
    let w = state.wall_remaining();
    let mut info = Info {
        kind: if !c.is_discard() {
            2
        } else if c.true_alarm() {
            0
        } else {
            1
        },
        dist: -1,
        third: ((wall0 - w.min(wall0)) * 3 / wall0.max(1)).min(2),
        paid: c.truth.paid,
        ..Info::default()
    };
    let (Some(pick), Some(thrown)) = (c.stage_pick(), c.thrown()) else {
        return Some(info);
    };
    let legal = state.legal_actions_vec();
    let view = SeatView::of(state, seat);
    info.dist = distance(&view, &discards(&legal));
    info.cost = distance(&view, &[thrown]) - distance(&view, &[pick]);
    let b = model.run_for(state, seat);
    let i = pick.index() as usize;
    info.fig = read_dealin(&b, i);
    let j = (0..3)
        .max_by(|&a, &c| b.opponents[a].dealin[i].total_cmp(&b.opponents[c].dealin[i]))
        .expect("three opponents");
    let o = (seat + 1 + j) % 4;
    info.opp_melds = state.melds(o).len();
    info.opp_dist = true_distance(state, o);
    Some(info)
}

/// Mean and 95 % range of per-deal samples.
fn mean_ci(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    let v = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, 1.96 * (v / n).sqrt())
}

/// A split table: per bin, hands, share of the group, points a hand (over all
/// hands) and mean per hand of the group.
fn split(title: &str, names: &[String], recs: &[&Rec], bin: impl Fn(&Rec) -> usize, n_hands: f64) {
    let mut cells = vec![(0usize, 0.0f64); names.len()];
    for r in recs {
        let c = &mut cells[bin(r)];
        c.0 += 1;
        c.1 += r.delta();
    }
    let total = recs.len().max(1) as f64;
    let loss: f64 = recs.iter().map(|r| r.delta()).sum();
    println!("    by {title}: bin | hands | share | points a hand | share of the sum | mean each");
    for (name, &(n, s)) in names.iter().zip(&cells) {
        if n == 0 {
            continue;
        }
        println!(
            "      {name:<22} {n:>7}  {:>5.1}%  {:+.4}  {:>6.1}%  {:+.2}",
            100.0 * n as f64 / total,
            s / n_hands,
            100.0 * s / if loss == 0.0 { 1.0 } else { loss },
            s / n as f64
        );
    }
}

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

const REL: [&str; 4] = [
    "bot",
    "next (ä¸‹å®¶)",
    "opposite (å°å®¶)",
    "previous (ä¸Šå®¶)",
];

/// One hand's ending in words, seats named relative to `seat`.
fn ending_text(r: &HandResult, seat: usize) -> String {
    let rel = |s: usize| REL[(s + 4 - seat) % 4];
    if r.wins.is_empty() {
        return "draw (wall ran out)".into();
    }
    let wins: Vec<String> = r
        .wins
        .iter()
        .map(|w| {
            if w.self_drawn {
                format!("{} self-drew {} faan", rel(w.seat), w.score.faan)
            } else {
                format!(
                    "{} won off {} with {} faan",
                    rel(w.seat),
                    w.from.map_or("?", rel),
                    w.score.faan
                )
            }
        })
        .collect();
    format!("{} (bot {:+})", wins.join("; "), r.deltas[seat])
}

/// The position at a first change, written for Brian.
fn example_text(
    k: usize,
    rec: &Rec,
    info: &Info,
    plain: &Played,
    mine: &Played,
    model: &BeliefModel,
) -> String {
    let c = mine.change.as_ref().expect("a change");
    let s: &State = &c.state;
    let seat = rec.seat;
    let pick = c.stage_pick().expect("discard");
    let thrown = c.thrown().expect("discard");
    let b = model.run_for(s, seat);
    let i = pick.index() as usize;
    let mut out = format!(
        "#### Example {k}: {}, {:+} points (deal {}, seat {seat}; {} tiles left in the wall, wall third {})\n\n",
        ENDINGS[ending(rec)],
        rec.delta(),
        rec.deal,
        s.wall_remaining(),
        info.third + 1
    );
    let river = s.river();
    for (r, rel) in REL.iter().enumerate() {
        let o = (seat + r) % 4;
        let melds: Vec<String> = s.melds(o).iter().map(|m| m.to_string()).collect();
        let thrown_by: Vec<String> = river
            .iter()
            .filter(|e| e.0 == o)
            .map(|e| {
                if e.2 {
                    format!("({})", e.1)
                } else {
                    e.1.to_string()
                }
            })
            .collect();
        let extra = if r == 0 {
            format!("{} from ready", info.dist)
        } else {
            format!(
                "truly {} from ready; read deal-in on {pick}: {:.3}",
                true_distance(s, o),
                b.opponents[r - 1].dealin[i]
            )
        };
        out += &format!(
            "* {} (seat {o}): `{}`{}; {extra}\n  discards: {}\n",
            rel,
            s.hand(o),
            if melds.is_empty() {
                String::new()
            } else {
                format!(", open `{}`", melds.join(" "))
            },
            if thrown_by.is_empty() {
                "none".into()
            } else {
                thrown_by.join(" ")
            }
        );
    }
    out += &format!(
        "\n`stage`'s pick {pick}, read {:.3} (nobody wins on it); thrown instead {thrown} (costs {} step{} of progress).\n",
        info.fig,
        info.cost,
        if info.cost == 1 { "" } else { "s" }
    );
    out += &format!(
        "Plain game: {}. Bot game: {}.\n",
        ending_text(&plain.result, seat),
        ending_text(&mine.result, seat)
    );
    out
}

fn report(
    name: &str,
    spec: &str,
    bot: &StageBot,
    rows: &[Vec<Rec>],
    k: usize,
    rules: &RuleSet,
    model: &BeliefModel,
) {
    let recs: Vec<&Rec> = rows.iter().map(|d| &d[k]).collect();
    let n_hands = recs.len() as f64;
    let deals = recs.len() / 4;
    let paired: Vec<f64> = (0..deals)
        .map(|d| (0..4).map(|s| recs[d * 4 + s].delta()).sum::<f64>() / 4.0)
        .collect();
    let (m, ci) = mean_ci(&paired);
    let changed: Vec<&Rec> = recs.iter().copied().filter(|r| r.info.is_some()).collect();
    let drift = recs
        .iter()
        .filter(|r| r.info.is_none() && r.points != r.plain_points)
        .count();
    let rate = |o: usize, plain: bool| {
        100.0
            * recs
                .iter()
                .filter(|r| if plain { r.plain_out } else { r.out } == o)
                .count() as f64
            / n_hands
    };
    let discards: u32 = recs.iter().map(|r| r.discards).sum();
    let ch: u32 = recs.iter().map(|r| r.changed).sum();
    let ch_d: u32 = recs.iter().map(|r| r.changed_discards).sum();
    println!();
    println!("== {name} ({spec}) ==");
    println!("  paired {name} âˆ’ plain stage: {m:+.3} Â± {ci:.3}/hand");
    println!(
        "  {name}: win {:.2}%  deal-in {:.2}%  other self-drew {:.2}%  other's deal-in {:.2}%  draw {:.2}%",
        rate(0, false),
        rate(1, false),
        rate(2, false),
        rate(3, false),
        rate(4, false)
    );
    println!(
        "  hands changed: {} of {} ({:.2}%); decisions changed {ch} ({ch_d} discards for discards) of {discards} discard decisions ({:.3}% per discard, {:.3} per hand); unchanged hands that differ from plain: {drift}",
        changed.len(),
        recs.len(),
        100.0 * changed.len() as f64 / n_hands,
        100.0 * ch_d as f64 / discards.max(1) as f64,
        ch as f64 / n_hands
    );
    let group = |kind: usize| -> Vec<&Rec> {
        changed
            .iter()
            .copied()
            .filter(|r| r.info.expect("info").kind == kind)
            .collect()
    };
    let (tr, fa, ot) = (group(0), group(1), group(2));
    let sum = |g: &[&Rec]| g.iter().map(|r| r.delta()).sum::<f64>();
    let all_sum: f64 = recs.iter().map(|r| r.delta()).sum();
    println!(
        "  first changes: true alarms {} ({:.1}%), false alarms {} ({:.1}%), not a discard {} ({:.1}%)",
        tr.len(),
        100.0 * tr.len() as f64 / changed.len().max(1) as f64,
        fa.len(),
        100.0 * fa.len() as f64 / changed.len().max(1) as f64,
        ot.len(),
        100.0 * ot.len() as f64 / changed.len().max(1) as f64
    );
    println!(
        "  points a hand: true alarms {:+.4}, false alarms {:+.4}, not a discard {:+.4}; sum {:+.4}; paired gain {:+.4} (unchanged hands {:+.4})",
        sum(&tr) / n_hands,
        sum(&fa) / n_hands,
        sum(&ot) / n_hands,
        (sum(&tr) + sum(&fa) + sum(&ot)) / n_hands,
        all_sum / n_hands,
        (all_sum - sum(&tr) - sum(&fa) - sum(&ot)) / n_hands
    );
    let mean = |g: &[&Rec]| sum(g) / g.len().max(1) as f64;
    let caught: f64 = tr
        .iter()
        .map(|r| r.info.expect("info").paid as f64)
        .sum::<f64>()
        / tr.len().max(1) as f64;
    println!(
        "  mean per hand: one false alarm {:+.2}, one true alarm caught {:+.2} (its pick would have paid {:.2} at once)",
        mean(&fa),
        mean(&tr),
        caught
    );
    println!(
        "  per hand / per discard: false alarms {:.4} / {:.5}, true alarms {:.4} / {:.5}",
        fa.len() as f64 / n_hands,
        fa.len() as f64 / discards.max(1) as f64,
        tr.len() as f64 / n_hands,
        tr.len() as f64 / discards.max(1) as f64
    );
    if fa.is_empty() {
        return;
    }
    println!("  false-alarm loss split (first changes that are false alarms):");
    let inf = |r: &Rec| r.info.expect("info");
    split(
        "the bot's distance from ready",
        &names(&["0 (ready)", "1", "2", "3", "4+"]),
        &fa,
        |r| inf(r).dist.clamp(0, 4) as usize,
        n_hands,
    );
    split(
        "wall third",
        &names(&["1", "2", "3"]),
        &fa,
        |r| inf(r).third,
        n_hands,
    );
    split(
        "the flagged opponent's open sets",
        &names(&["0", "1", "2", "3", "4"]),
        &fa,
        |r| inf(r).opp_melds.min(4),
        n_hands,
    );
    split(
        "the flagged opponent's true state",
        &names(&["ready", "one away", "far (2+)"]),
        &fa,
        |r| inf(r).opp_dist.clamp(0, 2) as usize,
        n_hands,
    );
    const FIG: [f64; 7] = [0.01, 0.022, 0.05, 0.1347, 0.25, 0.5, f64::INFINITY];
    split(
        "the reading's figure for the pick",
        &names(&[
            "< 0.010",
            "0.010â€“0.022",
            "0.022â€“0.050",
            "0.050â€“0.135",
            "0.135â€“0.250",
            "0.250â€“0.500",
            ">= 0.500",
        ]),
        &fa,
        |r| FIG.iter().position(|&x| inf(r).fig < x).expect("bins"),
        n_hands,
    );
    split(
        "progress the replacement cost (steps from ready)",
        &names(&["none or better", "1 step", "2 steps", "3+ steps"]),
        &fa,
        |r| inf(r).cost.clamp(0, 3) as usize,
        n_hands,
    );
    split(
        "ending beside the plain game",
        &names(&ENDINGS),
        &fa,
        ending,
        n_hands,
    );
    println!("    ending pairs (plain -> bot): hands, points a hand");
    let mut cells: HashMap<(usize, usize), (usize, f64)> = HashMap::new();
    for r in &fa {
        let c = cells.entry((r.plain_out, r.out)).or_default();
        c.0 += 1;
        c.1 += r.delta();
    }
    let mut keys: Vec<_> = cells.keys().copied().collect();
    keys.sort();
    for key in keys {
        let (n, s) = cells[&key];
        println!(
            "      {:<16} -> {:<16} {n:>7}  {:+.4}",
            OUTCOMES[key.0],
            OUTCOMES[key.1],
            s / n_hands
        );
    }
    // A two-way table of the two strongest-looking splits: distance Ã— figure.
    println!("    distance from ready Ã— the reading's figure (hands, points a hand):");
    let fig_bin = |r: &Rec| {
        if inf(r).fig < 0.05 {
            0
        } else if inf(r).fig < 0.1347 {
            1
        } else {
            2
        }
    };
    for d in 0..5 {
        let mut line = format!(
            "      dist {}",
            if d == 4 { "4+".into() } else { d.to_string() }
        );
        for f in 0..3 {
            let g: Vec<&&Rec> = fa
                .iter()
                .filter(|r| inf(r).dist.clamp(0, 4) as usize == d && fig_bin(r) == f)
                .collect();
            let s: f64 = g.iter().map(|r| r.delta()).sum();
            line += &format!(
                "  [{}] {:>6} {:+.4}",
                ["<0.05", "0.05-0.135", ">=0.135"][f],
                g.len(),
                s / n_hands
            );
        }
        println!("{line}");
    }
    let picks = pick_examples(&fa);
    println!();
    println!("  examples (the costliest false alarms across the endings):");
    println!();
    for (n, r) in picks.iter().enumerate() {
        let plain = play(rules, None, r.deal, r.seat);
        let mine = play(rules, Some(bot), r.deal, r.seat);
        assert_eq!(mine.points, r.points, "replay of deal {}", r.deal);
        let info = r.info.expect("info");
        println!("{}", example_text(n + 1, r, &info, &plain, &mine, model));
    }
}

/// Up to 20 examples across the endings, by their share of the loss (at least 2
/// for each ending that loses), the costliest in each, filled up with the
/// costliest left.
fn pick_examples<'a>(fa: &[&'a Rec]) -> Vec<&'a Rec> {
    let mut by_end: Vec<Vec<&Rec>> = vec![Vec::new(); ENDINGS.len()];
    for &r in fa {
        by_end[ending(r)].push(r);
    }
    let losses: Vec<f64> = by_end
        .iter()
        .map(|g| -g.iter().map(|r| r.delta()).sum::<f64>().min(0.0))
        .collect();
    let total_loss: f64 = losses.iter().sum::<f64>().max(1e-9);
    let mut quota: Vec<usize> = losses
        .iter()
        .map(|&l| {
            if l > 0.0 {
                2 + (12.0 * l / total_loss).round() as usize
            } else {
                0
            }
        })
        .collect();
    while quota.iter().sum::<usize>() > 20 {
        let i = (0..quota.len()).max_by_key(|&i| quota[i]).expect("bins");
        quota[i] -= 1;
    }
    let mut picks: Vec<&Rec> = Vec::new();
    for (g, &q) in by_end.iter_mut().zip(&quota) {
        g.sort_by(|a, b| a.delta().total_cmp(&b.delta()));
        picks.extend(g.iter().filter(|r| r.delta() < 0.0).take(q).copied());
    }
    // Fill to 20 with the costliest left.
    if picks.len() < 20 {
        let mut rest: Vec<&Rec> = fa
            .iter()
            .copied()
            .filter(|r| r.delta() < 0.0 && !picks.iter().any(|p| std::ptr::eq(*p, *r)))
            .collect();
        rest.sort_by(|a, b| a.delta().total_cmp(&b.delta()));
        let need = 20 - picks.len();
        picks.extend(rest.into_iter().take(need));
    }
    picks
}

/// `examples-json`: one file per example the report prints (the same choice),
/// `<bot>-<deal>-seat<n>.json`, in `dir`.
fn write_example_files(
    dir: &str,
    only: Option<(u64, usize)>,
    chosen: &[&(&str, &str)],
    bots: &[(String, StageBot)],
    flat: &[Vec<Rec>],
    rules: &RuleSet,
    model: &BeliefModel,
) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("{dir}: {e}"));
    for (k, ((name, _), (_, bot))) in chosen.iter().zip(bots).enumerate() {
        let fa: Vec<&Rec> = flat
            .iter()
            .map(|d| &d[k])
            .filter(|r| r.info.is_some_and(|i| i.kind == 1))
            .collect();
        // `only=`: that hand whatever its ending and whatever the kind of its
        // first change (plan Task 30: with `base=` the hand may hold no false
        // alarm).
        let picks: Vec<&Rec> = match only {
            Some((deal, seat)) => flat
                .iter()
                .map(|d| &d[k])
                .filter(|r| r.deal == deal && r.seat == seat && r.info.is_some())
                .collect(),
            None => pick_examples(&fa),
        };
        for r in picks {
            let plain = play_with(rules, None, r.deal, r.seat, Some(model));
            let mine = play_with(rules, Some(bot), r.deal, r.seat, Some(model));
            assert_eq!(mine.points, r.points, "replay of deal {}", r.deal);
            let info = r.info.expect("info");
            let c = mine.change.as_ref().expect("a change");
            let b = model.run_for(&c.state, r.seat);
            let i = c.stage_pick().expect("discard").index() as usize;
            let rd = Reading {
                own_dist: info.dist,
                cost: info.cost,
                dealin: [0, 1, 2].map(|j| b.opponents[j].dealin[i]),
                wall_third: info.third,
            };
            let v = example_json(
                rules,
                r.deal,
                r.seat,
                name,
                ENDINGS[ending(r)],
                &plain,
                &mine,
                &rd,
            );
            let path = format!("{dir}/{name}-{}-seat{}.json", r.deal, r.seat);
            std::fs::write(
                &path,
                serde_json::to_string_pretty(&v).expect("json") + "\n",
            )
            .unwrap_or_else(|e| panic!("{path}: {e}"));
            println!("wrote {path}");
        }
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `only=<deal>:<seat>` (examples-json): write that hand whatever its ending (plan
    // Task 27 rewrites the worked example under the amended 12-tile rule).
    let only: Option<(u64, usize)> = args.iter().position(|a| a.starts_with("only=")).map(|i| {
        let a = args.remove(i);
        let (d, s) = a[5..].split_once(':').expect("only=<deal>:<seat>");
        (d.parse().expect("deal"), s.parse().expect("seat"))
    });
    // `base=<stage bot>` (plan Task 30): the other seats and the plain game play
    // that bot, and the tested bot gets its early-game switches.
    let base: Option<StageBot> = args.iter().position(|a| a.starts_with("base=")).map(|i| {
        let a = args.remove(i);
        match Bot::by_name(&a[5..]) {
            Some(Bot::Stage(b)) => b,
            _ => panic!("{a}: not a stage bot"),
        }
    });
    if let Some(b) = base {
        mahjong_search::false_alarm::set_base(b);
    }
    let json_dir = if args.first().map(String::as_str) == Some("examples-json") {
        args.remove(0);
        assert!(
            !args.is_empty(),
            "examples-json <out dir> <deals> [seed] [model] [bot ...] [only=<deal>:<seat>] [base=<stage bot>]"
        );
        Some(args.remove(0))
    } else {
        None
    };
    let deals: u64 = args.first().and_then(|x| x.parse().ok()).unwrap_or(100);
    let seed: u64 = args.get(1).and_then(|x| x.parse().ok()).unwrap_or(24);
    let path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "models/belief-order.mjnn".into());
    let chosen: Vec<&(&str, &str)> = if args.len() > 3 {
        BOTS.iter()
            .filter(|(n, _)| args[3..].iter().any(|a| a == n))
            .collect()
    } else {
        BOTS.iter().collect()
    };
    let model = Arc::new(BeliefModel::load(&path).unwrap_or_else(|e| panic!("{path}: {e}")));
    let bots: Vec<(String, StageBot)> = chosen
        .iter()
        .map(|(_, spec)| {
            let full = format!(
                "{spec}{}belief={path}",
                if spec.contains(':') { "," } else { ":" }
            );
            match Agent::parse(&full).unwrap_or_else(|e| panic!("{full}: {e}")) {
                Agent::Bot(Bot::Stage(b)) => (full, b.with_early(base.map_or(0, |x| x.early))),
                _ => panic!("{full}: not a stage bot"),
            }
        })
        .collect();
    let rules = RuleSet::default();
    let start = std::time::Instant::now();
    // Per deal: per seat, per bot, a record; the plain hands beside them.
    let rows: Vec<DealRows> = (0..deals)
        .into_par_iter()
        .map(|i| {
            let deal = deal_seed(seed, i);
            let wall0 = State::new_deal(rules, &mut ChaCha8Rng::seed_from_u64(deal))
                .wall_remaining()
                .max(1);
            let mut per_seat = Vec::new();
            let mut plain_hands = Vec::new();
            for seat in 0..4 {
                let plain = play(&rules, None, deal, seat);
                plain_hands.push((plain.points, plain.outcome, plain.discards));
                let recs: Vec<Rec> = bots
                    .iter()
                    .map(|(_, b)| {
                        let mine = play(&rules, Some(b), deal, seat);
                        Rec {
                            deal,
                            seat,
                            plain_points: plain.points,
                            plain_out: plain.outcome,
                            points: mine.points,
                            out: mine.outcome,
                            liable: liable(&mine.result, seat),
                            plain_liable: liable(&plain.result, seat),
                            discards: mine.discards,
                            changed: mine.changed,
                            changed_discards: mine.changed_discards,
                            info: info_of(&mine, seat, wall0, &model),
                        }
                    })
                    .collect();
                per_seat.push(recs);
            }
            (per_seat, plain_hands)
        })
        .collect();
    let names: Vec<&str> = chosen.iter().map(|(n, _)| *n).collect();
    println!(
        "false alarms: {} vs three stage, {deals} deals x 4 seats, seed {seed}, model {path} ({:.0} s)",
        names.join(", "),
        start.elapsed().as_secs_f64()
    );
    let flat: Vec<Vec<Rec>> = rows.iter().flat_map(|r| r.0.clone()).collect();
    if let Some(dir) = json_dir {
        write_example_files(&dir, only, &chosen, &bots, &flat, &rules, &model);
        return;
    }
    // Plain stage per hand and per discard (Brian: "dangerous moments are not rare").
    let plain: Vec<(i32, usize, u32)> = rows.iter().flat_map(|r| r.1.clone()).collect();
    let n = plain.len() as f64;
    let dealt = plain.iter().filter(|p| p.1 == 1).count() as f64;
    let disc: u32 = plain.iter().map(|p| p.2).sum();
    println!(
        "  plain stage: deal-in in {:.2}% of hands, {:.3}% of its discards ({} discards a hand)",
        100.0 * dealt / n,
        100.0 * dealt / disc.max(1) as f64,
        disc as f64 / n
    );
    for (k, ((name, _), (spec, bot))) in chosen.iter().zip(&bots).enumerate() {
        report(name, spec, bot, &flat, k, &rules, &model);
    }
}
