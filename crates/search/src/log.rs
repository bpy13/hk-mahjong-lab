//! Self-play logs with hidden hands, the training data of the belief model.
//!
//! Every decision of a self-play hand writes one **row**, seen from the player to
//! act (the observer): the observer's public features ([`features::encode`]),
//! its passed-tile record ([`features::encode_passed`], since layout 2), each seat's last [`DISCARD_HISTORY`] discards, and the **hidden truth** about
//! the three opponents (relative seats 1 = next, 2 = across, 3 = previous): their
//! concealed tiles, whether they are waiting with a legal win, on which tiles, how
//! much a win on a discard would pay them now, and how the hand really ended for
//! them. The belief model is queried at the observer's own decisions, so rows are
//! taken exactly there (turn, claim and rob decisions alike).
//!
//! Layout 3 (plan 2026-11-16 Task 19) appends the discard and claim order
//! without the 摸打 / 入章 flag ([`pack_order`], the bytes of
//! [`features::encode_order`]), each opponent's true state (far / one away /
//! ready), who won the hand and how (or a draw), per opponent and tile kind
//! whether it would claim that kind into a meld if the observer threw it now
//! (the three-stage bot's own claim rule on the opponent's true hand), and the
//! row's `view`: besides the observer's own decisions ([`capture`]), layout 3
//! logs rows from a seat that is **not** to act ([`capture_for`]): right after
//! its discard, after the claim window on it closed without a claim, and after
//! an opponent claimed it, so the network can be read there (Brian, 2026-11-20:
//! "the solver needs to be able to re-evaluate the win-rate after hitting the
//! tile, whether it leads to no action or it gets claimed").
//!
//! Rows have a fixed size and layout ([`FIELDS`], little-endian, packed). The
//! log writer ([`LogWriter`]) writes them in chunk files `chunk-NNNNN.bin` plus a
//! `meta.json` that repeats the layout, so readers (the Python package's
//! `mahjong.belief.logs`) build their record type from the file, not from code.

use crate::features::{self, DiscardRecord, HISTORY_DEPTH, HISTORY_PLANES, PASSED_LEN, STATE_LEN};
use crate::race::true_state;
use crate::stage::StageBot;
use crate::view::SeatView;
use mahjong_games::Game;
use mahjong_hk::{score_win, Action, Event, HandResult, State, WinContext};
use mahjong_tiles::{Tile, NUM_KINDS};
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Discards kept per seat in a row (most recent first).
pub const DISCARD_HISTORY: usize = 8;
/// Marks an empty slot in the discard history and "nobody" in seat fields.
pub const NONE: u8 = 255;
/// Layout version written to `meta.json`; bump when [`FIELDS`] changes.
pub const LAYOUT_VERSION: u32 = 3;
/// Bytes of the packed discard order ([`pack_order`]): one per seat and slot,
/// then one overflow count per seat.
pub const ORDER_BYTES: usize = HISTORY_PLANES + 4;
/// The `winner` field's value for an exhaustive draw (else `2 × relative seat`,
/// plus 1 when the win was off a discard).
pub const DRAW: u8 = 8;

/// Whose moment a row describes (the `view` field).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum View {
    /// The observer is to act (every layout).
    Mover = 0,
    /// Right after the observer's discard: a claim window on it, or (nobody
    /// could claim) the next seat's turn.
    AfterDiscard = 1,
    /// The claim window on the observer's discard closed without a claim: the
    /// next seat's turn.
    AfterPass = 2,
    /// An opponent claimed the observer's discard and is to discard.
    AfterClaim = 3,
}

/// One field of a row: a name, a NumPy-style little-endian type and a shape.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub name: &'static str,
    /// `u1`, `u2`, `u4`, `i4` or `f2` (half-precision float).
    pub dtype: &'static str,
    pub shape: &'static [usize],
}

impl Field {
    /// Bytes this field takes in a row.
    pub fn bytes(&self) -> usize {
        let item = match self.dtype {
            "u1" => 1,
            "u2" | "f2" => 2,
            _ => 4,
        };
        item * self.shape.iter().product::<usize>()
    }
}

const fn f(name: &'static str, dtype: &'static str, shape: &'static [usize]) -> Field {
    Field { name, dtype, shape }
}

/// The row layout, in order. Seats inside a row are relative to the observer
/// (0 = observer). Opponent fields have three entries: relative seats 1, 2, 3.
pub const FIELDS: &[Field] = &[
    // Global hand index (the hand's seed derives from it) and decision number.
    f("hand", "u4", &[]),
    f("step", "u2", &[]),
    // Absolute seat of the observer (0 = dealer).
    f("observer", "u1", &[]),
    // 0 = turn, 1 = claim, 2 = rob.
    f("phase", "u1", &[]),
    f("wall_remaining", "u1", &[]),
    // Index into meta.json's `agents` for each relative seat.
    f("agents", "u1", &[4]),
    // The observer's public features (features::encode), as float16.
    f("features", "f2", &[STATE_LEN]),
    // The passed-tile record and 摸打 / 入章 (features::encode_passed), as float16
    // (layout version 2, plan 2026-11-09 Task 18).
    f("passed", "f2", &[PASSED_LEN]),
    // Tile index of each relative seat's discards, most recent first; 255 = none.
    f("discards", "u1", &[4, DISCARD_HISTORY]),
    // Hidden truth per opponent: concealed tile counts.
    f("opp_counts", "u1", &[3, NUM_KINDS]),
    // 1 if a discard of some tile would give the opponent a legal win now.
    f("opp_tenpai", "u1", &[3]),
    // 1 for each tile the opponent would legally win on now.
    f("opp_waits", "u1", &[3, NUM_KINDS]),
    // Points per share and faan of the best legal discard win now (0 if none).
    f("opp_points_now", "u2", &[3]),
    f("opp_faan_now", "u1", &[3]),
    // How the hand ended for the opponent: 0 = no win, 1 = won on a discard (or
    // robbed a kong), 2 = self-drawn win.
    f("opp_won", "u1", &[3]),
    // Points the opponent collected for that win, its faan, and who paid
    // (relative seat of the discarder; 255 for a self-draw or no win).
    f("opp_win_points", "i4", &[3]),
    f("opp_win_faan", "u1", &[3]),
    f("opp_win_from", "u1", &[3]),
    // Final net points of each relative seat (sum to zero).
    f("deltas", "i4", &[4]),
    // Layout 3 (plan 2026-11-16 Task 19). The discard and claim order of every
    // seat, packed (pack_order): the bytes of features::encode_order.
    f("order", "u1", &[ORDER_BYTES]),
    // The opponent's true state now: 0 far, 1 one away, 2 ready (race::true_state,
    // never folded here).
    f("opp_state", "u1", &[3]),
    // Who won the hand and how: 2 × relative seat (0-3; the first win when two
    // seats win on one discard) plus 1 for a win off a discard, 8 for a draw.
    f("winner", "u1", &[]),
    // 1 if the opponent would claim a discard of that kind into a meld (pung,
    // kong or, for the observer's next seat, chow) were the observer to throw it
    // now, by the three-stage bot's claim rule on the opponent's true hand; 0 when
    // no such claim is legal or the bot would pass. A win is not a claim here
    // (`opp_waits` holds it).
    f("opp_claim", "u1", &[3, NUM_KINDS]),
    // Whose moment the row is (`View`): 0 the observer's decision, 1 right after
    // its discard, 2 after the claim window on it closed without a claim, 3 after
    // an opponent claimed it.
    f("view", "u1", &[]),
];

/// Packs the discard order of the player to act (what [`features::encode_order`]
/// encodes) into [`ORDER_BYTES`] bytes. Byte `r * 24 + k` (relative seat `r`, its
/// `k`-th discard) is 0 for an empty slot, else `1 + tile + 34 * claimed` (at most
/// 68); byte `96 + r` is the number of seat `r`'s discards beyond the 24th (at
/// most 255). Nothing in it tells a drawn tile from a hand tile.
///
/// # Panics
/// Panics if the hand is over.
pub fn pack_order(state: &State) -> [u8; ORDER_BYTES] {
    let me = state.current_player().expect("hand in progress");
    pack_order_of(&features::discard_records(state), me)
}

/// [`pack_order`] from a list of discards (in the order thrown) for the absolute
/// seat `me`.
pub fn pack_order_of(records: &[DiscardRecord], me: usize) -> [u8; ORDER_BYTES] {
    let mut out = [0u8; ORDER_BYTES];
    let mut count = [0usize; 4];
    for d in records {
        let r = (d.seat as usize + 4 - me) % 4;
        let k = count[r];
        count[r] += 1;
        if k < HISTORY_DEPTH {
            out[r * HISTORY_DEPTH + k] = 1 + d.tile.index() + 34 * d.claimed as u8;
        }
    }
    for (r, &n) in count.iter().enumerate() {
        out[HISTORY_PLANES + r] = n.saturating_sub(HISTORY_DEPTH).min(255) as u8;
    }
    out
}

/// Unpacks [`pack_order`]'s bytes into [`features::encode_order`]'s block (`out`
/// of length [`features::ORDER_LEN`]).
///
/// # Panics
/// Panics if a length is wrong.
pub fn unpack_order(packed: &[u8], out: &mut [f32]) {
    assert_eq!(packed.len(), ORDER_BYTES);
    assert_eq!(out.len(), features::ORDER_LEN);
    out.fill(0.0);
    let (planes, s) = out.split_at_mut(HISTORY_PLANES * NUM_KINDS);
    for (slot, &b) in packed[..HISTORY_PLANES].iter().enumerate() {
        if b == 0 {
            continue;
        }
        let v = (b - 1) as usize;
        planes[slot * NUM_KINDS + v % NUM_KINDS] = 1.0;
        s[2 * slot] = 1.0;
        s[2 * slot + 1] = (v / NUM_KINDS) as f32;
    }
    for r in 0..4 {
        s[2 * HISTORY_PLANES + r] = (packed[HISTORY_PLANES + r] as f32 / 4.0).min(1.0);
    }
}

/// [`features::encode_suits`]'s block (`out` of length [`features::SUITS_LEN`])
/// from [`pack_order`]'s bytes: the same numbers the live encoder gives, so a
/// network can be fitted on logs of layout 3 without a new field.
///
/// # Panics
/// Panics if a length is wrong.
pub fn suits_from_packed(packed: &[u8], out: &mut [f32]) {
    assert_eq!(packed.len(), ORDER_BYTES);
    let mut kinds: [Vec<u8>; 4] = Default::default();
    let mut total = [0usize; 4];
    for r in 0..4 {
        for &b in &packed[r * HISTORY_DEPTH..(r + 1) * HISTORY_DEPTH] {
            if b == 0 {
                break;
            }
            kinds[r].push((b - 1) % NUM_KINDS as u8);
        }
        total[r] = kinds[r].len() + packed[HISTORY_PLANES + r] as usize;
    }
    features::suits_summary(&kinds, total, out);
}

/// The meld claims `seat` could legally declare on a discard of `tile` by `from`
/// now (as `State::window_options` without the win): pung with two copies in
/// hand, kong with three, chow for `from`'s next seat; none when the wall is
/// empty (rules §4.1.7). `out` ends with `Pass`.
fn claim_options(state: &State, seat: usize, from: usize, tile: Tile, out: &mut Vec<Action>) {
    out.clear();
    if state.wall_remaining() == 0 {
        return;
    }
    let hand = state.hand(seat);
    let have = hand[tile];
    if have >= 2 {
        out.push(Action::Pung);
    }
    if have >= 3 {
        out.push(Action::ClaimKong);
    }
    if seat == (from + 1) % 4 && tile.is_suited() {
        for start in -2i8..=0 {
            let tiles: Option<Vec<Tile>> = (0..3).map(|k| tile.offset(start + k)).collect();
            if let Some(tiles) = tiles {
                if tiles.iter().all(|&t| t == tile || hand[t] > 0) {
                    out.push(Action::Chow(tiles[0]));
                }
            }
        }
    }
    out.push(Action::Pass);
}

/// Per kind, whether `seat` (the three-stage bot's claim rule on its true hand)
/// would claim a discard of it by `from` into a meld now.
fn would_claim(state: &State, seat: usize, from: usize, bot: &StageBot) -> [bool; NUM_KINDS] {
    let mut out = [false; NUM_KINDS];
    let view = SeatView::of(state, seat);
    let mut legal = Vec::new();
    for k in Tile::all_kinds() {
        claim_options(state, seat, from, k, &mut legal);
        if legal.len() > 1 {
            out[k.index() as usize] = bot.claim(&view, k, from, &legal) != Action::Pass;
        }
    }
    out
}

/// Bytes per row.
pub fn row_bytes() -> usize {
    layout_bytes(FIELDS)
}

/// Bytes per row of any packed layout (the labels of [`crate::labels`] use
/// their own).
pub fn layout_bytes(fields: &[Field]) -> usize {
    fields.iter().map(Field::bytes).sum()
}

/// Converts an `f32` to IEEE half-precision bits, rounding to nearest even.
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // Subnormal half (or zero): value = m * 2^-24.
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let mut r = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    // A carry out of the mantissa correctly bumps the exponent.
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

/// The hidden truth about one opponent at a decision.
#[derive(Clone, Copy, Debug)]
struct Truth {
    counts: [u8; NUM_KINDS],
    waits: [bool; NUM_KINDS],
    points_now: u16,
    faan_now: u8,
    /// `race::true_state` without a fold: 0 far, 1 one away, 2 ready.
    state: u8,
    /// Per kind: the seat would claim it into a meld if the observer threw it now.
    claim: [bool; NUM_KINDS],
}

const NO_TRUTH: Truth = Truth {
    counts: [0; NUM_KINDS],
    waits: [false; NUM_KINDS],
    points_now: 0,
    faan_now: 0,
    state: 0,
    claim: [false; NUM_KINDS],
};

fn truth(state: &State, seat: usize, me: usize, bot: &StageBot) -> Truth {
    let hand = state.hand(seat);
    let mut t = NO_TRUTH;
    t.state = true_state(state, seat, false) as u8;
    t.claim = would_claim(state, seat, me, bot);
    for k in Tile::all_kinds() {
        t.counts[k.index() as usize] = hand[k];
    }
    for k in Tile::all_kinds() {
        // `wins_on` reads the seat's hidden tiles; fine here: this is the log.
        if state.wins_on(seat, k) {
            t.waits[k.index() as usize] = true;
        }
    }
    // `wins_on` only answers yes or no; score each wait as a plain discard win.
    let (faan, points) = best_wait_score(state, seat, &t.waits);
    t.faan_now = faan;
    t.points_now = points;
    t
}

/// Faan and points per share of the best legal discard win for `seat` now.
fn best_wait_score(state: &State, seat: usize, waits: &[bool; NUM_KINDS]) -> (u8, u16) {
    let rules = state.rules();
    let mut best = (0u8, 0u16);
    for k in Tile::all_kinds().filter(|k| waits[k.index() as usize]) {
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
            human: false,
        };
        if let Some(s) = score_win(&c, state.melds(seat), &ctx, rules) {
            let p = rules.points(s.faan).min(u16::MAX as u32) as u16;
            if p > best.1 || (p == best.1 && s.faan > best.0) {
                best = (s.faan, p);
            }
        }
    }
    best
}

/// A row before the hand's outcome is known.
#[derive(Clone, Debug)]
pub struct PendingRow {
    hand: u32,
    step: u16,
    observer: u8,
    phase: u8,
    wall_remaining: u8,
    agents: [u8; 4],
    /// The `features` and `passed` fields, in that order.
    features: Vec<f32>,
    discards: [[u8; DISCARD_HISTORY]; 4],
    opp: [Truth; 3],
    order: [u8; ORDER_BYTES],
    view: View,
}

/// Captures the row for the player to act in `state`.
///
/// `agent_ids` holds the agent index of each absolute seat.
///
/// # Panics
/// Panics if the hand is over.
pub fn capture(state: &State, hand: u32, step: u16, agent_ids: [u8; 4]) -> PendingRow {
    let me = state.current_player().expect("hand in progress");
    capture_for(state, me, View::Mover, hand, step, agent_ids)
}

/// Captures the row for seat `me` in `state`, whoever is to act (`view` says
/// which moment it is; plan 2026-11-16 Task 19).
///
/// # Panics
/// Panics if the hand is over.
pub fn capture_for(
    state: &State,
    me: usize,
    view: View,
    hand: u32,
    step: u16,
    agent_ids: [u8; 4],
) -> PendingRow {
    assert!(state.current_player().is_some(), "hand in progress");
    let rel = |seat: usize| (seat + 4 - me) % 4;
    // `features` then `passed`, adjacent in the row.
    let mut feats = vec![0.0f32; STATE_LEN + PASSED_LEN];
    let (f, p) = feats.split_at_mut(STATE_LEN);
    features::encode_for(state, me, f);
    features::encode_passed_for(state, me, p);
    let mut history: [Vec<Tile>; 4] = Default::default();
    for e in state.events() {
        if let Event::Discard { seat, tile, .. } = *e {
            history[rel(seat as usize)].push(tile);
        }
    }
    let mut discards = [[NONE; DISCARD_HISTORY]; 4];
    for (r, tiles) in history.iter().enumerate() {
        for (slot, t) in discards[r].iter_mut().zip(tiles.iter().rev()) {
            *slot = t.index();
        }
    }
    let mut agents = [0u8; 4];
    let mut opp = [NO_TRUTH; 3];
    let bot = StageBot::default();
    for seat in 0..4 {
        agents[rel(seat)] = agent_ids[seat];
        if seat != me {
            opp[rel(seat) - 1] = truth(state, seat, me, &bot);
        }
    }
    let phase = match state.phase_name() {
        "turn" => 0,
        "claim" => 1,
        _ => 2,
    };
    PendingRow {
        hand,
        step,
        observer: me as u8,
        phase,
        wall_remaining: state.wall_remaining().min(255) as u8,
        agents,
        features: feats,
        discards,
        opp,
        order: pack_order_of(&features::discard_records(state), me),
        view,
    }
}

/// Appends `row`, completed with the hand's `result`, to `out` in the
/// [`FIELDS`] layout.
pub fn finish(row: &PendingRow, result: &HandResult, out: &mut Vec<u8>) {
    let start = out.len();
    let me = row.observer as usize;
    let rel = |seat: usize| (seat + 4 - me) % 4;
    out.extend_from_slice(&row.hand.to_le_bytes());
    out.extend_from_slice(&row.step.to_le_bytes());
    out.push(row.observer);
    out.push(row.phase);
    out.push(row.wall_remaining);
    out.extend_from_slice(&row.agents);
    for &x in &row.features {
        out.extend_from_slice(&f16_bits(x).to_le_bytes());
    }
    for d in &row.discards {
        out.extend_from_slice(d);
    }
    for o in &row.opp {
        out.extend_from_slice(&o.counts);
    }
    for o in &row.opp {
        out.push(o.waits.iter().any(|&w| w) as u8);
    }
    for o in &row.opp {
        out.extend(o.waits.iter().map(|&w| w as u8));
    }
    for o in &row.opp {
        out.extend_from_slice(&o.points_now.to_le_bytes());
    }
    for o in &row.opp {
        out.push(o.faan_now);
    }
    // Outcome per opponent (the first win of that seat; a seat wins at most once).
    let mut won = [0u8; 3];
    let mut points = [0i32; 3];
    let mut faan = [0u8; 3];
    let mut from = [NONE; 3];
    for w in &result.wins {
        let r = rel(w.seat);
        if r == 0 {
            continue;
        }
        let i = r - 1;
        won[i] = if w.self_drawn { 2 } else { 1 };
        points[i] = w.paid_by.iter().sum();
        faan[i] = w.score.faan;
        from[i] = match (w.self_drawn, w.from) {
            (false, Some(s)) => rel(s) as u8,
            _ => NONE,
        };
    }
    out.extend_from_slice(&won);
    for p in points {
        out.extend_from_slice(&p.to_le_bytes());
    }
    out.extend_from_slice(&faan);
    out.extend_from_slice(&from);
    for r in 0..4 {
        let seat = (me + r) % 4;
        out.extend_from_slice(&result.deltas[seat].to_le_bytes());
    }
    out.extend_from_slice(&row.order);
    for o in &row.opp {
        out.push(o.state);
    }
    out.push(
        result
            .wins
            .first()
            .map_or(DRAW, |w| 2 * rel(w.seat) as u8 + (!w.self_drawn) as u8),
    );
    for o in &row.opp {
        out.extend(o.claim.iter().map(|&c| c as u8));
    }
    out.push(row.view as u8);
    debug_assert_eq!(out.len() - start, row_bytes());
}

/// Name of chunk file `i`.
pub fn chunk_name(i: usize) -> String {
    format!("chunk-{i:05}.bin")
}

/// Writes chunk files and keeps `meta.json` up to date; resumes from the chunks
/// already on disk.
pub struct LogWriter {
    dir: PathBuf,
    /// Rows in each finished chunk, in order.
    chunks: Vec<u64>,
    meta_head: String,
    fields: &'static [Field],
    layout_version: u32,
}

impl LogWriter {
    /// Opens `dir` (created if missing). Complete chunks already there are kept:
    /// a chunk is written to a `.tmp` file and renamed only when complete, so
    /// every `chunk-NNNNN.bin` is whole. `meta_head` is extra JSON members
    /// (without braces) describing the run, written at the top of `meta.json`.
    ///
    /// # Errors
    /// Fails on I/O errors, or when an existing chunk's size is not a whole
    /// number of rows (a log written with a different layout).
    pub fn open(dir: &Path, meta_head: String) -> io::Result<LogWriter> {
        Self::open_layout(dir, meta_head, FIELDS, LAYOUT_VERSION)
    }

    /// [`LogWriter::open`] for rows of another layout (`fields`, written to
    /// `meta.json` with `layout_version`).
    ///
    /// # Errors
    /// As [`LogWriter::open`].
    pub fn open_layout(
        dir: &Path,
        meta_head: String,
        fields: &'static [Field],
        layout_version: u32,
    ) -> io::Result<LogWriter> {
        fs::create_dir_all(dir)?;
        let row = layout_bytes(fields) as u64;
        let mut chunks = Vec::new();
        loop {
            let p = dir.join(chunk_name(chunks.len()));
            let Ok(m) = fs::metadata(&p) else { break };
            let size = m.len();
            if size % row != 0 {
                return Err(io::Error::other(format!(
                    "{} is not a whole number of {row}-byte rows",
                    p.display(),
                )));
            }
            chunks.push(size / row);
        }
        Ok(LogWriter {
            dir: dir.to_path_buf(),
            chunks,
            meta_head,
            fields,
            layout_version,
        })
    }

    /// Rows in each complete chunk, in order.
    pub fn chunk_rows(&self) -> &[u64] {
        &self.chunks
    }

    /// Number of complete chunks.
    pub fn chunks_done(&self) -> usize {
        self.chunks.len()
    }

    /// Total rows in complete chunks.
    pub fn rows(&self) -> u64 {
        self.chunks.iter().sum()
    }

    /// Writes the next chunk (encoded rows) and refreshes `meta.json`.
    ///
    /// # Errors
    /// Fails on I/O errors.
    pub fn write_chunk(&mut self, bytes: &[u8], status: &str) -> io::Result<()> {
        let name = chunk_name(self.chunks.len());
        let tmp = self.dir.join(format!("{name}.tmp"));
        fs::write(&tmp, bytes)?;
        fs::rename(&tmp, self.dir.join(&name))?;
        self.chunks
            .push((bytes.len() / layout_bytes(self.fields)) as u64);
        self.write_meta(status)
    }

    /// Rewrites `meta.json` with the layout, the chunks and `status`
    /// (extra JSON members, without braces).
    ///
    /// # Errors
    /// Fails on I/O errors.
    pub fn write_meta(&self, status: &str) -> io::Result<()> {
        let mut s = String::from("{\n");
        let _ = writeln!(s, "  \"layout_version\": {},", self.layout_version);
        let _ = writeln!(s, "  \"row_bytes\": {},", layout_bytes(self.fields));
        if !self.meta_head.is_empty() {
            let _ = writeln!(s, "  {},", self.meta_head);
        }
        if !status.is_empty() {
            let _ = writeln!(s, "  {status},");
        }
        let _ = writeln!(s, "  \"rows\": {},", self.rows());
        s.push_str("  \"fields\": [\n");
        for (i, fd) in self.fields.iter().enumerate() {
            let shape: Vec<String> = fd.shape.iter().map(|x| x.to_string()).collect();
            let _ = write!(
                s,
                "    {{\"name\": \"{}\", \"dtype\": \"<{}\", \"shape\": [{}]}}",
                fd.name,
                fd.dtype,
                shape.join(", ")
            );
            s.push_str(if i + 1 < self.fields.len() {
                ",\n"
            } else {
                "\n"
            });
        }
        s.push_str("  ],\n  \"chunks\": [\n");
        for (i, rows) in self.chunks.iter().enumerate() {
            let _ = write!(
                s,
                "    {{\"file\": \"{}\", \"rows\": {rows}}}",
                chunk_name(i)
            );
            s.push_str(if i + 1 < self.chunks.len() {
                ",\n"
            } else {
                "\n"
            });
        }
        s.push_str("  ]\n}\n");
        let tmp = self.dir.join("meta.json.tmp");
        fs::write(&tmp, s)?;
        fs::rename(tmp, self.dir.join("meta.json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use mahjong_games::Policy;
    use mahjong_hk::RuleSet;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn half_precision_matches_known_values() {
        assert_eq!(f16_bits(0.0), 0x0000);
        assert_eq!(f16_bits(-0.0), 0x8000);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(0.25), 0x3400);
        assert_eq!(f16_bits(1.0 / 3.0), 0x3555);
        assert_eq!(f16_bits(2.0 / 3.0), 0x3955);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(1e6), 0x7c00);
        assert_eq!(f16_bits(5.960_464_5e-8), 0x0001);
        assert_eq!(f16_bits(f32::NAN) & 0x7c00, 0x7c00);
    }

    #[test]
    fn rows_have_the_layout_size_and_truth_is_consistent() {
        let rules = RuleSet::default();
        let bot = Bot::by_name("value").unwrap();
        let mut rng = SmallRng::seed_from_u64(7);
        for hand in 0..4 {
            let mut s = State::new_deal(rules, &mut rng);
            let mut pending = Vec::new();
            let mut step = 0u16;
            let mut legal = Vec::new();
            while let Some(p) = s.current_player() {
                let row = capture(&s, hand, step, [0, 1, 2, 3]);
                // Opponents' tiles plus what the observer sees never exceed 4.
                let unseen = s.unseen(p);
                for k in Tile::all_kinds() {
                    let held: u8 = row.opp.iter().map(|o| o.counts[k.index() as usize]).sum();
                    assert!(held <= unseen[k], "{k}: {held} > unseen {}", unseen[k]);
                }
                for o in &row.opp {
                    assert_eq!(o.waits.iter().any(|&w| w), o.points_now > 0);
                }
                pending.push(row);
                s.legal_actions(&mut legal);
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
                step += 1;
            }
            let result = s.result().unwrap().clone();
            let mut out = Vec::new();
            for r in &pending {
                finish(r, &result, &mut out);
            }
            assert_eq!(out.len(), pending.len() * row_bytes());
            // The last row's opponents' outcomes agree with the result.
            let winners = result.wins.len();
            assert!(winners <= 3);
            // Layout 3: the winner byte names the winner's relative seat or a draw.
            let last = &out[out.len() - row_bytes()..];
            let winner = last[row_bytes() - 3 * NUM_KINDS - 2];
            let me = pending.last().unwrap().observer as usize;
            match result.wins.first() {
                Some(w) => assert_eq!(
                    winner as usize,
                    2 * ((w.seat + 4 - me) % 4) + (!w.self_drawn) as usize
                ),
                None => assert_eq!(winner, DRAW),
            }
            assert_eq!(last[row_bytes() - 1], View::Mover as u8);
        }
    }

    /// Layout 3: the packed order unpacks to `encode_order`, the state agrees
    /// with the waits, and a claim is marked only where one is legal.
    #[test]
    fn order_and_claims_in_the_row() {
        let rules = RuleSet::default();
        let bot = Bot::by_name("stage").unwrap();
        let mut rng = SmallRng::seed_from_u64(19);
        let (mut claims, mut checked) = (0usize, 0usize);
        let mut want = vec![0.0; features::ORDER_LEN];
        let mut got = vec![0.0; features::ORDER_LEN];
        let mut want_suits = vec![0.0; features::SUITS_LEN];
        let mut got_suits = vec![0.0; features::SUITS_LEN];
        for hand in 0..6 {
            let mut s = State::new_deal(rules, &mut rng);
            let mut legal = Vec::new();
            while let Some(me) = s.current_player() {
                let row = capture(&s, hand, 0, [0; 4]);
                features::encode_order(&s, &mut want);
                unpack_order(&row.order, &mut got);
                assert_eq!(want, got);
                features::encode_suits(&s, &mut want_suits);
                suits_from_packed(&row.order, &mut got_suits);
                assert_eq!(want_suits, got_suits);
                for (r, o) in row.opp.iter().enumerate() {
                    let seat = (me + r + 1) % 4;
                    assert!(o.state <= 2);
                    assert_eq!(o.state == 2, o.waits.iter().any(|&w| w));
                    for k in Tile::all_kinds() {
                        let i = k.index() as usize;
                        if o.claim[i] {
                            claims += 1;
                            let hand = s.hand(seat);
                            let chow_seat = seat == (me + 1) % 4;
                            assert!(hand[k] >= 2 || (chow_seat && k.is_suited()), "{k}");
                            assert!(s.wall_remaining() > 0);
                        }
                    }
                }
                checked += 1;
                s.legal_actions(&mut legal);
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
            }
        }
        assert!(
            checked > 200 && claims > 0,
            "{checked} rows, {claims} claims"
        );
    }

    /// Layout 3: a row from a seat that is not to act carries that seat's view
    /// (its own hand, relative seats, the order from its side) and the view byte.
    #[test]
    fn spectator_rows_take_the_observers_view() {
        let bot = Bot::by_name("stage").unwrap();
        let mut rng = SmallRng::seed_from_u64(31);
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let mut legal = Vec::new();
        let mut checked = 0;
        while let Some(p) = s.current_player() {
            s.legal_actions(&mut legal);
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
            if s.current_player().is_none() || s.current_player() == Some(p) {
                continue;
            }
            let row = capture_for(&s, p, View::AfterDiscard, 1, 2, [0; 4]);
            assert_eq!(row.observer as usize, p);
            assert_eq!(row.view, View::AfterDiscard);
            let mut want = vec![0.0; STATE_LEN];
            features::encode_for(&s, p, &mut want);
            assert_eq!(&row.features[..STATE_LEN], &want[..]);
            let mut order = vec![0.0; features::ORDER_LEN];
            unpack_order(&row.order, &mut order);
            let mut want_order = vec![0.0; features::ORDER_LEN];
            features::encode_order_for(&s, p, &mut want_order);
            assert_eq!(order, want_order);
            let mut out = Vec::new();
            finish(
                &row,
                &HandResult {
                    wins: Vec::new(),
                    deltas: [0; 4],
                },
                &mut out,
            );
            assert_eq!(out.len(), row_bytes());
            assert_eq!(out[row_bytes() - 1], View::AfterDiscard as u8);
            checked += 1;
        }
        assert!(checked > 20);
    }
}
