//! Does the belief network read a ready hand from its discards? (plan
//! 2026-11-16 Task 18, `docs/study-results/belief-ready-read.md`)
//!
//! ```text
//! cargo run --release -p mahjong-search --example belief_ready -- \
//!     --hand C:\Users\brian\.orchestrator\notes\mahjong-flush-game-2860.json [--from 56]
//! cargo run --release -p mahjong-search --example belief_ready -- \
//!     --deals 20000 --seed 18 [--field stage] [--belief models/belief.mjnn]
//! ```
//!
//! `--hand`: replays a saved hand (`seed`, `actions` as `[seat, code]`) and, at
//! every turn decision from step `--from` on, prints for each opponent of the
//! player to act the true ready state, the network's P(ready) (`tenpai` head),
//! the race bot's risk-adjusted reading of it (`race::RISK_MODEL`) and the
//! danger table's threat.
//!
//! `--deals`: plays deals of four `--field` bots (`stage` by default; hand seeds from `--seed`, none
//! used to fit the network, which was fitted on a `value`/`defensive` field) and,
//! at every discard decision of every seat, logs each opponent's true ready
//! state (`State::wins_on` on any kind, the network's own training target), the
//! network's P(ready) and the table's. Prints markdown: ten bins by each
//! estimate, and the slices of the task (last third of the wall; a seat whose
//! discards were nearly all outside one suit and has since thrown that suit; a
//! seat that just threw a tile of the suit another seat shows three or four
//! open sets of; both; runs of discards that were the drawn tile thrown back),
//! each with count, true rate ± 95 %, mean network and mean table estimate.
//! Threads: `RAYON_NUM_THREADS`.
//!
//! Task 18a adds, for `--field stage`: (1) every discard the three-stage bot
//! makes in its stage 3 while its hand is a flush (all concealed suited tiles and
//! suited sets in one suit before the discard) and the tile is of that suit: ready
//! after the discard, ready or one away (faan-aware), and the other three seats'
//! readings of it at their next decisions; (2) the strict suit collector seen from
//! the discards (at least N discards before its first tile of suit x, tiles of
//! both other suits among them, every suited open set in x, x thrown since), the
//! swing at its first and second x discard, and honour discards of a seat that
//! still has thrown no x after 8 or more discards. No 摸打/入章 input is used.

use mahjong_analysis::{faan_shanten, HandView};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, Event, RuleSet, State};
use mahjong_search::passed::risk_scores;
use mahjong_search::race::RISK_MODEL;
use mahjong_search::{BeliefModel, Bot, Danger};
use mahjong_tiles::{Suit, Tile};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Tiles in the wall at the deal (flowers on).
const INITIAL_WALL: usize = 91;
/// Discards a seat must have made before its suit pattern is read.
const MIN_DISCARDS: usize = 8;
/// The first discards that must all lie outside the suspected suit.
const OUTSIDE_FIRST: usize = 6;

/// One discard of one seat.
#[derive(Clone, Copy)]
struct Thrown {
    tile: Tile,
    from_hand: bool,
}

fn discards_of(state: &State) -> [Vec<Thrown>; 4] {
    let mut out: [Vec<Thrown>; 4] = Default::default();
    for e in state.events() {
        if let Event::Discard {
            seat,
            tile,
            from_hand,
        } = *e
        {
            out[seat as usize].push(Thrown { tile, from_hand });
        }
    }
    out
}

/// Whether seat `o` is truly ready: some kind would win for it now (the
/// network's training target, `log::truth`).
fn truly_ready(state: &State, o: usize) -> bool {
    Tile::all_kinds().any(|k| state.wins_on(o, k))
}

/// The suit seat `o` seems to collect: its first [`OUTSIDE_FIRST`] discards hold
/// no tile of it, a later discard does, and every suited exposed meld of `o` is
/// in it. Returns the suit and whether `o`'s latest discard is in it.
fn flush_suspect(state: &State, o: usize, thrown: &[Thrown]) -> Option<(Suit, bool)> {
    if thrown.len() < MIN_DISCARDS {
        return None;
    }
    let last = thrown.last()?.tile.suit();
    for x in Suit::ALL {
        let early_clean = thrown[..OUTSIDE_FIRST]
            .iter()
            .all(|d| d.tile.suit() != Some(x));
        let since = thrown[OUTSIDE_FIRST..]
            .iter()
            .any(|d| d.tile.suit() == Some(x));
        let melds_fit = state
            .melds(o)
            .iter()
            .filter(|m| !m.concealed)
            .all(|m| m.tile.suit().is_none_or(|s| s == x));
        if early_clean && since && melds_fit {
            return Some((x, last == Some(x)));
        }
    }
    None
}

/// The suit that another seat than `o` shows three or four open sets of (all of
/// its exposed melds in that one suit), if any.
fn flush_shown_against(state: &State, o: usize) -> Option<Suit> {
    (0..4).filter(|&q| q != o).find_map(|q| {
        let melds = state.melds(q);
        let exposed: Vec<_> = melds.iter().filter(|m| !m.concealed).collect();
        if exposed.len() < 3 {
            return None;
        }
        let s = exposed[0].tile.suit()?;
        exposed
            .iter()
            .all(|m| m.tile.suit() == Some(s))
            .then_some(s)
    })
}

/// Whether `o`'s latest discard (and the one before, for `two`) is a tile of a
/// suit another seat shows three or four open sets of. Returns (feeds, feeds
/// from hand 入章, feeds twice in a row).
fn feeds_flush(state: &State, o: usize, thrown: &[Thrown]) -> (bool, bool, bool) {
    let Some(y) = flush_shown_against(state, o) else {
        return (false, false, false);
    };
    let Some(last) = thrown.last() else {
        return (false, false, false);
    };
    let feeds = last.tile.suit() == Some(y);
    let two = feeds && thrown.len() >= 2 && thrown[thrown.len() - 2].tile.suit() == Some(y);
    (feeds, feeds && last.from_hand, two)
}

/// One opponent at one discard decision.
#[derive(Clone, Copy)]
struct Row {
    ready: bool,
    net: f64,
    table: f64,
    race: f64,
    wall: usize,
    exposed: usize,
    suspect: bool,
    suspect_last: bool,
    feeds: bool,
    feeds_hand: bool,
    feeds_two: bool,
    /// Latest discards in a row that were the drawn tile thrown back (摸打).
    streak: usize,
}

/// Sum statistics of a set of rows.
#[derive(Clone, Copy, Default)]
struct Cell {
    n: u64,
    ready: u64,
    net: f64,
    table: f64,
    race: f64,
    /// Summed log loss of the true ready state against the network's and the
    /// table's P(ready) (each clipped to [1e-4, 1 − 1e-4]; plan Task 19's check).
    ll_net: f64,
    ll_table: f64,
}

/// `−ln P(observed)` with `p` clipped away from 0 and 1.
fn log_loss(p: f64, ready: bool) -> f64 {
    let p = p.clamp(1e-4, 1.0 - 1e-4);
    -if ready { p.ln() } else { (1.0 - p).ln() }
}

impl Cell {
    fn add(&mut self, r: &Row) {
        self.n += 1;
        self.ready += r.ready as u64;
        self.net += r.net;
        self.table += r.table;
        self.race += r.race;
        self.ll_net += log_loss(r.net, r.ready);
        self.ll_table += log_loss(r.table, r.ready);
    }
    fn merge(&mut self, o: &Cell) {
        self.n += o.n;
        self.ready += o.ready;
        self.net += o.net;
        self.table += o.table;
        self.race += o.race;
        self.ll_net += o.ll_net;
        self.ll_table += o.ll_table;
    }
    fn line(&self, name: &str) -> String {
        if self.n == 0 {
            return format!("| {name} | 0 | – | – | – | – |");
        }
        let n = self.n as f64;
        let p = self.ready as f64 / n;
        let pm = 1.96 * (p * (1.0 - p) / n).sqrt();
        format!(
            "| {name} | {} | {:.2} ± {:.2} % | {:.2} % | {:.2} % | {:.2} % |",
            self.n,
            100.0 * p,
            100.0 * pm,
            100.0 * self.net / n,
            100.0 * self.table / n,
            100.0 * self.race / n
        )
    }
}

const SLICES: &[&str] = &[
    "all",
    "last third of the wall (≤ 30 tiles left)",
    "suit-suspect: first 6 discards outside one suit, that suit thrown since",
    "suit-suspect, latest discard in that suit",
    "feeds a shown flush: latest discard in the suit another seat shows 3–4 open sets of",
    "feeds a shown flush, thrown from hand (入章)",
    "feeds a shown flush, drawn and thrown (摸打)",
    "feeds a shown flush twice in a row",
    "suit-suspect and feeds a shown flush",
    "suit-suspect (latest in suit) and feeds a shown flush",
    "last third and suit-suspect",
    "last third and feeds a shown flush",
    "last third, suit-suspect and feeds a shown flush",
    "opponent shows 0 exposed sets",
    "opponent shows 1 exposed set",
    "opponent shows 2 exposed sets",
    "opponent shows 3–4 exposed sets",
    "1 exposed set, suit-suspect",
    "1 exposed set, feeds a shown flush",
    "1 exposed set, suit-suspect and feeds a shown flush",
    "1 exposed set, last third, suit-suspect and feeds a shown flush",
    "last 3 discards all drawn and thrown (摸打)",
    "last 5 discards all drawn and thrown",
    "last 8 discards all drawn and thrown",
    "1 exposed set, last 5 drawn and thrown",
    "suit-suspect, last 5 drawn and thrown",
    "1 exposed set, suit-suspect, last 5 drawn and thrown",
    "1 exposed set, suit-suspect, feeds a shown flush, last 5 drawn and thrown",
];

fn slice_flags(r: &Row) -> [bool; 28] {
    let late = r.wall <= INITIAL_WALL / 3;
    let one = r.exposed == 1;
    let s5 = r.streak >= 5;
    [
        true,
        late,
        r.suspect,
        r.suspect_last,
        r.feeds,
        r.feeds_hand,
        r.feeds && !r.feeds_hand,
        r.feeds_two,
        r.suspect && r.feeds,
        r.suspect_last && r.feeds,
        late && r.suspect,
        late && r.feeds,
        late && r.suspect && r.feeds,
        r.exposed == 0,
        one,
        r.exposed == 2,
        r.exposed >= 3,
        one && r.suspect,
        one && r.feeds,
        one && r.suspect && r.feeds,
        one && late && r.suspect && r.feeds,
        r.streak >= 3,
        s5,
        r.streak >= 8,
        one && s5,
        r.suspect && s5,
        one && r.suspect && s5,
        one && r.suspect && r.feeds && s5,
    ]
}

#[derive(Clone)]
struct Stats {
    slices: Vec<Cell>,
    by_net: [Cell; 10],
    by_table: [Cell; 10],
    /// Ready rate against the table by the network's own reading: rows where the
    /// network is below the table and rows where it is above.
    net_vs_table: [Cell; 2],
    decisions: u64,
    flush: Vec<FlushCell>,
    strict: Strict,
    /// Wall time of one `BeliefModel::run` call in microseconds (at most
    /// `TIME_SAMPLES` kept; exact per call only with `RAYON_NUM_THREADS=1`).
    times: Vec<u32>,
}

const TIME_SAMPLES: usize = 200_000;

impl Stats {
    fn new() -> Stats {
        Stats {
            slices: vec![Cell::default(); SLICES.len()],
            by_net: [Cell::default(); 10],
            by_table: [Cell::default(); 10],
            net_vs_table: [Cell::default(); 2],
            decisions: 0,
            flush: vec![FlushCell::default(); 2 + 2 * FLUSH_BLOCK],
            strict: Strict::new(),
            times: Vec::new(),
        }
    }
    fn add(&mut self, r: &Row) {
        for (c, on) in self.slices.iter_mut().zip(slice_flags(r)) {
            if on {
                c.add(r);
            }
        }
        self.by_net[bin(r.net)].add(r);
        self.by_table[bin(r.table)].add(r);
        self.net_vs_table[(r.net > r.table) as usize].add(r);
    }
    fn merge(&mut self, o: &Stats) {
        for (a, b) in self.slices.iter_mut().zip(&o.slices) {
            a.merge(b);
        }
        for (a, b) in self.by_net.iter_mut().zip(&o.by_net) {
            a.merge(b);
        }
        for (a, b) in self.by_table.iter_mut().zip(&o.by_table) {
            a.merge(b);
        }
        for (a, b) in self.net_vs_table.iter_mut().zip(&o.net_vs_table) {
            a.merge(b);
        }
        self.decisions += o.decisions;
        for (a, b) in self.flush.iter_mut().zip(&o.flush) {
            a.merge(b);
        }
        self.strict.merge(&o.strict);
        let room = TIME_SAMPLES.saturating_sub(self.times.len());
        self.times.extend(o.times.iter().take(room));
    }
}

fn bin(p: f64) -> usize {
    ((p * 10.0) as usize).min(9)
}

/// The three readings of opponent `o` for the player to act `me`.
fn readings(state: &State, me: usize, o: usize, model: &BeliefModel) -> (f64, f64, f64, u32) {
    let t0 = std::time::Instant::now();
    let out = model.run(state);
    let micros = t0.elapsed().as_micros().min(u32::MAX as u128) as u32;
    let net = out.seat(o).map_or(f64::NAN, |b| b.tenpai);
    let table = Danger::estimate(state, me).threat(o);
    let risk = risk_scores(state);
    let race = (net * RISK_MODEL.tenpai[RISK_MODEL.bin(risk[o])]).min(1.0);
    (net, table, race, micros)
}

/// Logs every opponent at this discard decision; returns the network's and the
/// table's reading of each opponent.
fn log_decision(
    state: &State,
    me: usize,
    model: &BeliefModel,
    stats: &mut Stats,
    snaps: &mut [Vec<Snap>; 4],
) -> [(f64, f64); 4] {
    let thrown = discards_of(state);
    stats.decisions += 1;
    let mut reads = [(f64::NAN, f64::NAN); 4];
    for o in (0..4).filter(|&o| o != me) {
        let (net, table, race, micros) = readings(state, me, o, model);
        if stats.times.len() < TIME_SAMPLES {
            stats.times.push(micros);
        }
        reads[o] = (net, table);
        let ready = truly_ready(state, o);
        if let Some(st) = story(state, o, &thrown[o]).filter(|s| s.fit && s.pre >= NS[0]) {
            let snap = Snap {
                d: thrown[o].len(),
                story: st,
                last_honour: thrown[o].last().is_some_and(|d| d.tile.is_honour()),
                wall: state.wall_remaining(),
                ready,
                near: ready || faan_dist_of(state, o) <= 1,
                clean: state
                    .hand(o)
                    .kinds()
                    .all(|t| t.suit().is_none_or(|s| s == st.x)),
                net,
                table,
            };
            stats.strict.add_row(&snap, state.melds(o).exposed_count());
            snaps[o].push(snap);
        }
        let suspect = flush_suspect(state, o, &thrown[o]);
        let (feeds, feeds_hand, feeds_two) = feeds_flush(state, o, &thrown[o]);
        stats.add(&Row {
            ready,
            net,
            table,
            race,
            wall: state.wall_remaining(),
            exposed: state.melds(o).exposed_count(),
            suspect: suspect.is_some(),
            suspect_last: suspect.is_some_and(|(_, last)| last),
            feeds,
            feeds_hand,
            feeds_two,
            streak: thrown[o].iter().rev().take_while(|d| !d.from_hand).count(),
        });
    }
    reads
}

fn play_deal(deal_seed: u64, bot: &Bot, model: &BeliefModel, stats: &mut Stats) {
    let mut state = State::new_deal(
        RuleSet::default(),
        &mut ChaCha8Rng::seed_from_u64(deal_seed),
    );
    let mut rng = ChaCha8Rng::seed_from_u64(deal_seed ^ 0x9e37_79b9);
    let mut snaps: [Vec<Snap>; 4] = Default::default();
    let mut watches: Vec<Watch> = Vec::new();
    while let Some(me) = state.current_player() {
        let legal = state.legal_actions_vec();
        // The same move as `bot.choose` (`Bot::Stage` calls `decide`), with the stage.
        let (action, note) = match bot {
            Bot::Stage(sb) => sb.decide(&state, &legal, &mut rng),
            _ => (bot.choose(&state, &legal, &mut rng), None),
        };
        let Action::Discard(tile) = action else {
            state.apply(action);
            continue;
        };
        let reads = log_decision(&state, me, model, stats, &mut snaps);
        // Watches on `me` end (it acts again); the others take `me`'s reading.
        watches.retain(|w| w.o != me);
        for w in &mut watches {
            if !w.read[me] {
                w.read[me] = true;
                let (net, table) = reads[w.o];
                for &c in &w.cells {
                    stats.flush[c].read(net, table);
                }
            }
        }
        watches.retain(|w| (0..4).any(|q| q != w.o && !w.read[q]));
        // Stage 3, the hand's flush before the discard, own suit, first of the hand.
        let event = note.filter(|n| n.stage == 3).map(|_| {
            let flush = hand_flush(&state, me);
            let own = flush.is_some_and(|(x, _)| tile.suit() == Some(x));
            let first = own
                && discards_of(&state)[me]
                    .iter()
                    .all(|d| d.tile.suit() != tile.suit());
            (flush, own, first)
        });
        let wall = state.wall_remaining();
        let exposed = state.melds(me).exposed_count();
        state.apply(action);
        if let Some((flush, own, first)) = event {
            let cells = flush_cells(flush.map(|(_, full)| full), own, first, exposed, wall);
            let ready = truly_ready(&state, me);
            let near = ready || faan_dist_of(&state, me) <= 1;
            for &c in &cells {
                stats.flush[c].add(ready, near);
            }
            watches.push(Watch {
                o: me,
                cells,
                read: [false; 4],
            });
        }
    }
    stats.strict.close_deal(&snaps);
}

// ----- Task 18a: the stage 3 flush count -----

/// Wall thirds by tiles left: 0 = first (61–91), 1 = middle (31–60), 2 = last (≤ 30).
fn wall_third(wall: usize) -> usize {
    if wall > 2 * (INITIAL_WALL / 3) {
        0
    } else if wall > INITIAL_WALL / 3 {
        1
    } else {
        2
    }
}

const THIRDS: [&str; 3] = [
    "first third (61–91 left)",
    "middle third (31–60 left)",
    "last third (≤ 30 left)",
];
const OPEN: [&str; 4] = ["0 open sets", "1 open set", "2 open sets", "3–4 open sets"];

/// Faan-aware distance of seat `o` to a ready hand that meets the minimum.
fn faan_dist_of(state: &State, o: usize) -> i8 {
    let concealed = *state.hand(o);
    let unseen = state.unseen(o);
    let view = HandView {
        concealed: &concealed,
        melds: &state.melds(o)[..],
        seat: o,
        bonus: state.bonus(o),
        unseen: &unseen,
    };
    faan_shanten(&view, state.rules()).shanten
}

/// The flush of `o`'s hand: every concealed suited tile and every suited set
/// (open or concealed) in one suit. Returns the suit and whether it is a full
/// flush (no honour anywhere).
fn hand_flush(state: &State, o: usize) -> Option<(Suit, bool)> {
    let mut suit: Option<Suit> = None;
    let mut honours = false;
    let tiles = state
        .hand(o)
        .kinds()
        .chain(state.melds(o).iter().map(|m| m.tile));
    for t in tiles {
        match t.suit() {
            None => honours = true,
            Some(s) if suit.is_none_or(|x| x == s) => suit = Some(s),
            Some(_) => return None,
        }
    }
    Some((suit?, !honours))
}

/// Rows of the stage 3 flush table: every stage 3 discard; flush hand, any
/// tile; then twice (every own-suit discard, the first own-suit discard of the
/// hand): all, half, full, four open-set bins, three wall thirds.
fn flush_labels() -> Vec<String> {
    let mut out = vec![
        "every stage 3 discard".to_string(),
        "stage 3, flush hand, any tile".to_string(),
    ];
    for head in ["own-suit discard", "first own-suit discard of the hand"] {
        out.push(format!("{head}: all"));
        out.push(format!("{head}: half flush"));
        out.push(format!("{head}: full flush"));
        for o in OPEN {
            out.push(format!("{head}: {o}"));
        }
        for t in THIRDS {
            out.push(format!("{head}: {t}"));
        }
    }
    out
}

const FLUSH_BLOCK: usize = 10;

fn flush_cells(
    full: Option<bool>,
    own: bool,
    first: bool,
    exposed: usize,
    wall: usize,
) -> Vec<usize> {
    let mut cells = vec![0];
    let Some(full) = full else {
        return cells;
    };
    cells.push(1);
    if !own {
        return cells;
    }
    let mut block = |base: usize| {
        cells.push(base);
        cells.push(base + 1 + full as usize);
        cells.push(base + 3 + exposed.min(3));
        cells.push(base + 7 + wall_third(wall));
    };
    block(2);
    if first {
        block(2 + FLUSH_BLOCK);
    }
    cells
}

/// One discard counted in the flush table, waiting for the other seats' readings.
struct Watch {
    o: usize,
    cells: Vec<usize>,
    read: [bool; 4],
}

#[derive(Clone, Copy, Default)]
struct FlushCell {
    n: u64,
    ready: u64,
    near: u64,
    reads: u64,
    net: f64,
    table: f64,
}

impl FlushCell {
    fn add(&mut self, ready: bool, near: bool) {
        self.n += 1;
        self.ready += ready as u64;
        self.near += near as u64;
    }
    fn read(&mut self, net: f64, table: f64) {
        self.reads += 1;
        self.net += net;
        self.table += table;
    }
    fn merge(&mut self, o: &FlushCell) {
        self.n += o.n;
        self.ready += o.ready;
        self.near += o.near;
        self.reads += o.reads;
        self.net += o.net;
        self.table += o.table;
    }
    fn line(&self, name: &str) -> String {
        let r = self.reads.max(1) as f64;
        format!(
            "| {name} | {} | {} | {} | {} | {:.1} % | {:.1} % |",
            self.n,
            pct(self.ready, self.n),
            pct(self.near, self.n),
            self.reads,
            100.0 * self.net / r,
            100.0 * self.table / r
        )
    }
}

/// `k` of `n` as a percentage with its 95 % range.
fn pct(k: u64, n: u64) -> String {
    if n == 0 {
        return "–".to_string();
    }
    let n = n as f64;
    let p = k as f64 / n;
    format!(
        "{:.1} ± {:.1} %",
        100.0 * p,
        100.0 * 1.96 * (p * (1.0 - p) / n).sqrt()
    )
}

// ----- Task 18a: the strict suit collector, from the discards -----

/// Seat `o`'s suit story: `x` is the suit whose first discard came last (or the
/// one suit never thrown, once the other two have been), `pre` the discards
/// before the first `x` (all of them while none), `n_x` the `x` tiles thrown,
/// `fit` whether every suited open set of `o` is in `x`. `None` until two suits
/// have been thrown (so both other suits are always among the `pre` discards).
#[derive(Clone, Copy, Debug)]
struct Story {
    x: Suit,
    pre: usize,
    n_x: usize,
    fit: bool,
}

fn story(state: &State, o: usize, thrown: &[Thrown]) -> Option<Story> {
    let mut first: [Option<usize>; 3] = [None; 3];
    let mut count = [0usize; 3];
    for (i, d) in thrown.iter().enumerate() {
        if let Some(s) = d.tile.suit() {
            count[s as usize] += 1;
            first[s as usize].get_or_insert(i);
        }
    }
    let seen = first.iter().filter(|f| f.is_some()).count();
    let k = match seen {
        2 => first.iter().position(Option::is_none)?,
        3 => (0..3).max_by_key(|&k| first[k])?,
        _ => return None,
    };
    let x = Suit::ALL[k];
    let fit = state
        .melds(o)
        .iter()
        .filter(|m| !m.concealed)
        .all(|m| m.tile.suit().is_none_or(|s| s == x));
    Some(Story {
        x,
        pre: first[k].unwrap_or(thrown.len()),
        n_x: count[k],
        fit,
    })
}

/// One opponent row kept for the swing and honour tables.
#[derive(Clone, Copy)]
struct Snap {
    /// Discards `o` had made.
    d: usize,
    story: Story,
    last_honour: bool,
    wall: usize,
    ready: bool,
    near: bool,
    clean: bool,
    net: f64,
    table: f64,
}

const NS: [usize; 3] = [6, 8, 10];
/// Splits of the strict table: all, four open-set bins, three wall thirds.
const SPLITS: usize = 8;

#[derive(Clone, Copy, Default)]
struct StrictCell {
    n: u64,
    ready: u64,
    near: u64,
    clean: u64,
    net: f64,
    net2: f64,
    table: f64,
    table2: f64,
}

impl StrictCell {
    fn add(&mut self, s: &Snap) {
        self.n += 1;
        self.ready += s.ready as u64;
        self.near += s.near as u64;
        self.clean += s.clean as u64;
        self.net += s.net;
        self.net2 += s.net * s.net;
        self.table += s.table;
        self.table2 += s.table * s.table;
    }
    fn merge(&mut self, o: &StrictCell) {
        self.n += o.n;
        self.ready += o.ready;
        self.near += o.near;
        self.clean += o.clean;
        self.net += o.net;
        self.net2 += o.net2;
        self.table += o.table;
        self.table2 += o.table2;
    }
    fn mean(&self, sum: f64, sum2: f64) -> String {
        if self.n == 0 {
            return "–".to_string();
        }
        let n = self.n as f64;
        let m = sum / n;
        let sd = (sum2 / n - m * m).max(0.0).sqrt();
        format!("{:.1} ± {:.1} %", 100.0 * m, 100.0 * 1.96 * sd / n.sqrt())
    }
    fn line(&self, name: &str) -> String {
        format!(
            "| {name} | {} | {} | {} | {} | {} | {} |",
            self.n,
            pct(self.ready, self.n),
            pct(self.near, self.n),
            pct(self.clean, self.n),
            self.mean(self.net, self.net2),
            self.mean(self.table, self.table2)
        )
    }
}

#[derive(Clone)]
struct Strict {
    /// Per N: rows of a strict collector, by split.
    rows: [[StrictCell; SPLITS]; 3],
    /// Per N: before the first `x` discard, after it, after the second.
    swing: [[StrictCell; 3]; 3],
    /// Honour discard of a seat with no `x` in 8+ discards: [before the last
    /// third, last third, all] × [before, after].
    honour: [[StrictCell; 2]; 3],
}

impl Strict {
    fn new() -> Strict {
        Strict {
            rows: [[StrictCell::default(); SPLITS]; 3],
            swing: [[StrictCell::default(); 3]; 3],
            honour: [[StrictCell::default(); 2]; 3],
        }
    }
    fn merge(&mut self, o: &Strict) {
        let pairs = self
            .rows
            .iter_mut()
            .flatten()
            .zip(o.rows.iter().flatten())
            .chain(
                self.swing
                    .iter_mut()
                    .flatten()
                    .zip(o.swing.iter().flatten()),
            )
            .chain(
                self.honour
                    .iter_mut()
                    .flatten()
                    .zip(o.honour.iter().flatten()),
            );
        for (a, b) in pairs {
            a.merge(b);
        }
    }
    /// A row of the main table, if the seat is a strict collector.
    fn add_row(&mut self, s: &Snap, exposed: usize) {
        for (k, &n) in NS.iter().enumerate() {
            if s.story.n_x >= 1 && s.story.pre >= n && s.story.fit {
                self.rows[k][0].add(s);
                self.rows[k][1 + exposed.min(3)].add(s);
                self.rows[k][5 + wall_third(s.wall)].add(s);
            }
        }
    }
    /// The swing and honour tables from one deal's kept rows.
    fn close_deal(&mut self, snaps: &[Vec<Snap>; 4]) {
        for seat in snaps {
            for (k, &n) in NS.iter().enumerate() {
                let Some(i1) = seat
                    .iter()
                    .position(|s| s.story.n_x == 1 && s.story.pre >= n && s.story.fit)
                else {
                    continue;
                };
                let a = seat[i1];
                let Some(before) = seat[..i1]
                    .iter()
                    .rev()
                    .find(|s| s.story.n_x == 0 && s.story.x == a.story.x && s.d == a.story.pre)
                else {
                    continue;
                };
                self.swing[k][0].add(before);
                self.swing[k][1].add(&a);
                if let Some(b) = seat[i1..]
                    .iter()
                    .find(|s| s.story.n_x == 2 && s.story.x == a.story.x)
                {
                    self.swing[k][2].add(b);
                }
            }
            for (i, a) in seat.iter().enumerate() {
                let new_d = i == 0 || seat[i - 1].d != a.d;
                // The honour is discard number a.d; 8 or more came before it.
                if !(new_d && a.last_honour && a.story.n_x == 0 && a.d > 8) {
                    continue;
                }
                let Some(before) = seat[..i].iter().rev().find(|s| {
                    s.d + 1 == a.d && s.story.n_x == 0 && s.story.fit && s.story.x == a.story.x
                }) else {
                    continue;
                };
                let late = (wall_third(a.wall) == 2) as usize;
                for g in [late, 2] {
                    self.honour[g][0].add(before);
                    self.honour[g][1].add(a);
                }
            }
        }
    }
}

fn header() -> &'static str {
    "| Slice | Opponent-decisions | True ready | Network mean | Table mean | Race-adjusted mean |\n|---|---|---|---|---|---|"
}

fn run_deals(deals: u32, seed: u64, field: &str, belief: &Path) {
    let chunk = 500u32;
    let chunks: Vec<u32> = (0..deals).step_by(chunk as usize).collect();
    let start = std::time::Instant::now();
    let total = chunks
        .into_par_iter()
        .map(|first| {
            let model = BeliefModel::load(belief).expect("belief model");
            let bot = Bot::by_name(field).unwrap_or_else(|| panic!("no bot {field:?}"));
            let mut stats = Stats::new();
            for d in first..(first + chunk).min(deals) {
                let deal_seed = seed.wrapping_mul(1_000_003).wrapping_add(d as u64);
                play_deal(deal_seed, &bot, &model, &mut stats);
            }
            stats
        })
        .reduce(Stats::new, |mut a, b| {
            a.merge(&b);
            a
        });
    println!(
        "{deals} deals of four `{field}` bots, seed {seed}, model {}: {} discard decisions, {} opponent rows, {:.0} s.\n",
        belief.display(),
        total.decisions,
        total.slices[0].n,
        start.elapsed().as_secs_f64()
    );
    let all = &total.slices[0];
    println!(
        "Log loss on ready over all {} opponent rows: network {:.5}, table {:.5}.\n",
        all.n,
        all.ll_net / all.n.max(1) as f64,
        all.ll_table / all.n.max(1) as f64
    );
    let mut times = total.times.clone();
    times.sort_unstable();
    if !times.is_empty() {
        let q = |f: f64| times[((times.len() - 1) as f64 * f) as usize];
        println!(
            "One `BeliefModel::run` call ({} samples, {} rayon threads): p50 {:.2} ms, p90 {:.2} ms, p99 {:.2} ms.\n",
            times.len(),
            rayon::current_num_threads(),
            q(0.5) as f64 / 1000.0,
            q(0.9) as f64 / 1000.0,
            q(0.99) as f64 / 1000.0
        );
    }
    println!("## Ten bins by the network's P(ready)\n\n{}", header());
    for (i, c) in total.by_net.iter().enumerate() {
        println!(
            "{}",
            c.line(&format!("network {}–{} %", i * 10, i * 10 + 10))
        );
    }
    println!("\n## Ten bins by the table's P(ready)\n\n{}", header());
    for (i, c) in total.by_table.iter().enumerate() {
        println!("{}", c.line(&format!("table {}–{} %", i * 10, i * 10 + 10)));
    }
    println!("\n## Where the two disagree\n\n{}", header());
    println!("{}", total.net_vs_table[0].line("network ≤ table"));
    println!("{}", total.net_vs_table[1].line("network > table"));
    println!("\n## Slices\n\n{}", header());
    for (name, c) in SLICES.iter().zip(&total.slices) {
        println!("{}", c.line(name));
    }
    print_task18a(&total);
}

fn print_task18a(total: &Stats) {
    println!("\n## Stage 3 discards of a flush hand (the bot's own stage)\n");
    println!(
        "| Discards | Count | Ready after it | Ready or one away after it | Other seats' readings | Network mean | Table mean |\n|---|---|---|---|---|---|---|"
    );
    for (name, c) in flush_labels().iter().zip(&total.flush) {
        println!("{}", c.line(name));
    }
    let head = "| Rows | Count | True ready | Ready or one away | Concealed tiles all x or honours | Network mean | Table mean |\n|---|---|---|---|---|---|---|";
    println!("\n## The strict suit collector (rows: opponent at a discard decision)\n\n{head}");
    let splits: Vec<String> = std::iter::once("all".to_string())
        .chain(OPEN.iter().map(|s| s.to_string()))
        .chain(THIRDS.iter().map(|s| s.to_string()))
        .collect();
    for (k, n) in NS.iter().enumerate() {
        for (name, c) in splits.iter().zip(&total.strict.rows[k]) {
            println!("{}", c.line(&format!("N = {n}, {name}")));
        }
    }
    println!("\n## The swing at the first own-suit discard (one row per seat and hand)\n\n{head}");
    let points = [
        "last decision before the first x",
        "first decision after the first x",
        "first decision after the second x",
    ];
    for (k, n) in NS.iter().enumerate() {
        for (name, c) in points.iter().zip(&total.strict.swing[k]) {
            println!("{}", c.line(&format!("N = {n}, {name}")));
        }
    }
    println!("\n## An honour from a seat with no x in 8 or more discards\n\n{head}");
    let groups = ["before the last third", "in the last third", "all"];
    for (g, name) in groups.iter().enumerate() {
        for (w, c) in ["before the honour", "after the honour"]
            .iter()
            .zip(&total.strict.honour[g])
        {
            println!("{}", c.line(&format!("{name}, {w}")));
        }
    }
}

/// Reads `"seed": N` and `"actions": [[seat, code], ...]` from the saved hand.
fn read_hand(path: &Path) -> (u64, Vec<(usize, u16)>) {
    let text = std::fs::read_to_string(path).expect("hand file");
    let after = |key: &str| -> &str {
        let i = text.find(key).unwrap_or_else(|| panic!("no {key}"));
        &text[i + key.len()..]
    };
    let seed: u64 = after("\"seed\":")
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse()
        .expect("seed");
    let acts = after("\"actions\":");
    let end = acts.find("]]").expect("actions end");
    let actions = acts[..end]
        .split("],")
        .map(|pair| {
            let nums: Vec<u64> = pair
                .chars()
                .map(|c| if c.is_ascii_digit() { c } else { ' ' })
                .collect::<String>()
                .split_whitespace()
                .map(|n| n.parse().unwrap())
                .collect();
            (nums[0] as usize, nums[1] as u16)
        })
        .collect();
    (seed, actions)
}

fn hand_text(state: &State, seat: usize) -> String {
    let melds: Vec<String> = state
        .melds(seat)
        .iter()
        .map(|m| {
            m.tiles()
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("")
        })
        .collect();
    format!("{} | sets {}", state.hand(seat), melds.join(" "))
}

fn replay(path: &Path, from: usize, belief: &Path) {
    let model = BeliefModel::load(belief).expect("belief model");
    let (seed, actions) = read_hand(path);
    // The same deal as `mahjong._core.HkState(seed)` (pybind `rng_from`: SmallRng).
    let mut state = State::new_deal(RuleSet::default(), &mut SmallRng::seed_from_u64(seed));
    println!(
        "Hand {} (seed {seed}, {} actions), model {}\n",
        path.display(),
        actions.len(),
        belief.display()
    );
    let seat_name = ["East", "South", "West", "North"];
    for (i, &(seat, code)) in actions.iter().enumerate() {
        let me = state.current_player().expect("hand in progress");
        assert_eq!(me, seat, "step {i}: seat {seat} in the file, {me} to act");
        let action = Action::from_code(code).expect("action code");
        if i >= from && state.phase_name() == "turn" {
            let thrown = discards_of(&state);
            println!(
                "step {i}, wall {}, {} to act ({action}); hand {}",
                state.wall_remaining(),
                seat_name[me],
                hand_text(&state, me)
            );
            for o in (0..4).filter(|&o| o != me) {
                let (net, table, race, _) = readings(&state, me, o, &model);
                let last = thrown[o].last().map_or("-".to_string(), |d| {
                    format!("{}{}", d.tile, if d.from_hand { " 入章" } else { " 摸打" })
                });
                let suspect = flush_suspect(&state, o, &thrown[o]);
                let (feeds, _, two) = feeds_flush(&state, o, &thrown[o]);
                let st = story(&state, o, &thrown[o]).map_or("-".to_string(), |s| {
                    format!(
                        "x {} pre {} n_x {} fit {} dist {}",
                        s.x.to_char(),
                        s.pre,
                        s.n_x,
                        s.fit,
                        faan_dist_of(&state, o)
                    )
                });
                println!(
                    "    {:<5} ready {:<5} network {:5.1} %  race-adjusted {:5.1} %  table {:5.1} %  last discard {last}  suspect {:?} feeds {feeds} twice {two}  story [{st}]  hand {}",
                    seat_name[o],
                    truly_ready(&state, o),
                    100.0 * net,
                    100.0 * race,
                    100.0 * table,
                    suspect,
                    hand_text(&state, o)
                );
            }
        }
        state.apply(action);
    }
    if let Some(r) = state.result() {
        println!("\nresult: {r:?}");
    }
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut hand: Option<PathBuf> = None;
    let mut from = 0usize;
    let mut deals = 0u32;
    let mut seed = 18u64;
    let mut field = "stage".to_string();
    let mut belief = PathBuf::from("models/belief.mjnn");
    let mut i = 0;
    while i < raw.len() {
        let v = raw.get(i + 1).cloned().unwrap_or_default();
        match raw[i].as_str() {
            "--hand" => hand = Some(PathBuf::from(&v)),
            "--from" => from = v.parse().expect("--from"),
            "--deals" => deals = v.parse().expect("--deals"),
            "--seed" => seed = v.parse().expect("--seed"),
            "--field" => field = v.clone(),
            "--belief" => belief = PathBuf::from(&v),
            other => panic!("unknown option {other}"),
        }
        i += 2;
    }
    if let Some(h) = hand {
        replay(&h, from, &belief);
    } else if deals > 0 {
        run_deals(deals, seed, &field, &belief);
    } else {
        eprintln!(
            "usage: --hand <file> [--from N] | --deals N [--seed S] [--field bot] [--belief file]"
        );
    }
}
