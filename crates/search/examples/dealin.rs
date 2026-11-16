//! Where deal-ins come from: the three-stage bot against three value bots.
//!
//! `dealin <deals> [seed]`. Every deal is played four times with the stage bot in
//! each seat. Every discard is counted in buckets (own discard number, tile class,
//! own distance, unseen copies, opponents' exposed melds); a deal-in is charged to
//! the bucket of the discard that was won on. Rates are deal-ins per 1,000 discards.
use mahjong_analysis::faan_shanten;
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::stage::StageBot;
use mahjong_search::{Bot, SeatView};
use mahjong_tiles::Tile;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const TURN: [&str; 6] = ["1-4", "5-8", "9-12", "13-16", "17-20", "21+"];
const CLASS: [&str; 4] = ["honour", "1 or 9", "2 or 8", "3 to 7"];
const DIST: [&str; 4] = ["ready", "1 away", "2 away", "3+ away"];
const UNSEEN: [&str; 4] = ["0 unseen", "1 unseen", "2 unseen", "3 unseen"];
const MELDS: [&str; 4] = ["0 melds", "1 meld", "2 melds", "3+ melds"];
const STAGE: [&str; 4] = ["no line", "stage 1", "stage 2", "stage 3"];
const LINE: [&str; 5] = ["flush", "pungs", "orphans", "chows", "none"];
/// The bucket groups, in the order of [`Rec::keys`].
const GROUPS: [(&str, &[&str]); 7] = [
    ("own discard number", &TURN),
    ("tile thrown", &CLASS),
    ("own distance from a win", &DIST),
    ("copies of the tile not yet seen", &UNSEEN),
    ("most exposed melds of one opponent", &MELDS),
    ("stage (three-stage bot only)", &STAGE),
    ("leading line (three-stage bot only)", &LINE),
];

/// One discard, as bucket indices in the order of [`GROUPS`].
#[derive(Clone, Copy, Default)]
struct Rec {
    keys: [usize; 7],
    tile: Option<Tile>,
}

#[derive(Clone, Default)]
struct Agg {
    hands: u64,
    wins: u64,
    win_faan: u64,
    deal_ins: u64,
    /// Points paid on deal-ins.
    paid: i64,
    /// Deal-ins by a robbed kong (no discard to charge).
    robbed: u64,
    /// `[group][bucket] = (discards, deal-ins)`.
    cells: [[(u64, u64); 6]; 7],
}

impl Agg {
    fn add(&mut self, o: &Agg) {
        self.hands += o.hands;
        self.wins += o.wins;
        self.win_faan += o.win_faan;
        self.deal_ins += o.deal_ins;
        self.paid += o.paid;
        self.robbed += o.robbed;
        for g in 0..7 {
            for b in 0..6 {
                self.cells[g][b].0 += o.cells[g][b].0;
                self.cells[g][b].1 += o.cells[g][b].1;
            }
        }
    }
}

fn class(t: Tile) -> usize {
    match t.number() {
        None => 0,
        Some(1 | 9) => 1,
        Some(2 | 8) => 2,
        Some(_) => 3,
    }
}

/// Plays one hand; `stage_seat` holds the three-stage bot. Returns `[stage, value]`.
fn play(rules: &RuleSet, deal_seed: u64, stage_seat: usize) -> [Agg; 2] {
    let mut state = State::new_deal(rules.clone(), &mut ChaCha8Rng::seed_from_u64(deal_seed));
    let mut rng = ChaCha8Rng::seed_from_u64(deal_seed.wrapping_mul(4).wrapping_add(stage_seat as u64));
    let value = Bot::by_name("value").expect("the value bot");
    let mut out = [Agg::default(), Agg::default()];
    let mut last = [Rec::default(); 4];
    let mut count = [0usize; 4];
    while !state.is_terminal() {
        let legal = state.legal_actions_vec();
        let seat = state.current_player().expect("not over");
        let kind = usize::from(seat != stage_seat);
        let (action, note) = if kind == 0 {
            StageBot::default().decide(&state, &legal, &mut rng)
        } else {
            (value.choose(&state, &legal, &mut rng), None)
        };
        if let Action::Discard(t) = action {
            let view = SeatView::of(&state, seat);
            count[seat] += 1;
            let dist = faan_shanten(&view.hand_view(), &view.rules).shanten;
            let melds = (0..4)
                .filter(|s| *s != seat)
                .map(|s| state.melds(s).exposed_count())
                .max()
                .unwrap_or(0);
            let rec = Rec {
                keys: [
                    ((count[seat] - 1) / 4).min(5),
                    class(t),
                    (dist.max(0) as usize).min(3),
                    (view.unseen[t] as usize).min(3),
                    melds.min(3),
                    note.map_or(0, |n| n.stage as usize),
                    note.and_then(|n| n.line).map_or(4, |l| l as usize),
                ],
                tile: Some(t),
            };
            let groups = if kind == 0 { 7 } else { 5 };
            for g in 0..groups {
                out[kind].cells[g][rec.keys[g]].0 += 1;
            }
            last[seat] = rec;
        }
        state.apply(action);
    }
    let result = state.result().expect("the hand is over");
    for seat in 0..4 {
        out[usize::from(seat != stage_seat)].hands += 1;
    }
    for w in &result.wins {
        let k = usize::from(w.seat != stage_seat);
        out[k].wins += 1;
        out[k].win_faan += u64::from(w.score.faan);
        let Some(from) = w.from.filter(|_| !w.self_drawn) else {
            continue;
        };
        let k = usize::from(from != stage_seat);
        out[k].deal_ins += 1;
        out[k].paid += i64::from(w.paid_by[from]);
        if w.robbed || last[from].tile != w.win_tile {
            out[k].robbed += 1;
            continue;
        }
        let groups = if k == 0 { 7 } else { 5 };
        for g in 0..groups {
            out[k].cells[g][last[from].keys[g]].1 += 1;
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let deals: u64 = args.first().and_then(|d| d.parse().ok()).unwrap_or(200);
    let seed: u64 = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(1);
    let rules = RuleSet::default();
    let total = (0..deals * 4)
        .into_par_iter()
        .map(|i| play(&rules, seed.wrapping_mul(1_000_003).wrapping_add(i / 4), (i % 4) as usize))
        .reduce(
            || [Agg::default(), Agg::default()],
            |mut a, b| {
                a[0].add(&b[0]);
                a[1].add(&b[1]);
                a
            },
        );
    let names = ["stage", "value"];
    for (name, a) in names.iter().zip(&total) {
        let h = a.hands as f64;
        let discards: u64 = a.cells[0].iter().map(|c| c.0).sum();
        println!(
            "{name}: hands {}  win {:.2}%  avg faan {:.2}  deal-in {:.2}%  paid per deal-in {:.1}  discards per hand {:.2}  deal-ins per 1000 discards {:.2}  robbed-kong deal-ins {}",
            a.hands,
            100.0 * a.wins as f64 / h,
            a.win_faan as f64 / a.wins.max(1) as f64,
            100.0 * a.deal_ins as f64 / h,
            a.paid as f64 / a.deal_ins.max(1) as f64,
            discards as f64 / h,
            1000.0 * a.deal_ins as f64 / discards.max(1) as f64,
            a.robbed,
        );
    }
    for (g, (title, labels)) in GROUPS.iter().enumerate() {
        println!("\n== by {title}");
        println!(
            "{:<10} | stage: discards/hand  per-1000  deal-in%/hand | value: discards/hand  per-1000  deal-in%/hand",
            ""
        );
        for (b, label) in labels.iter().enumerate() {
            let mut line = format!("{label:<10} |");
            for a in &total {
                let (d, n) = a.cells[g][b];
                let h = a.hands as f64;
                line += &format!(
                    "        {:6.3}        {:6.2}       {:6.3}        |",
                    d as f64 / h,
                    1000.0 * n as f64 / d.max(1) as f64,
                    100.0 * n as f64 / h
                );
            }
            println!("{line}");
        }
    }
}
