//! Self-play logs with hidden hands, the training data of the belief model.
//!
//! Every decision of a self-play hand writes one **row**, seen from the player to
//! act (the observer): the observer's public features ([`features::encode`]),
//! each seat's last [`DISCARD_HISTORY`] discards, and the **hidden truth** about
//! the three opponents (relative seats 1 = next, 2 = across, 3 = previous): their
//! concealed tiles, whether they are waiting with a legal win, on which tiles, how
//! much a win on a discard would pay them now, and how the hand really ended for
//! them. The belief model is queried at the observer's own decisions, so rows are
//! taken exactly there (turn, claim and rob decisions alike).
//!
//! Rows have a fixed size and layout ([`FIELDS`], little-endian, packed). The
//! log writer ([`LogWriter`]) writes them in chunk files `chunk-NNNNN.bin` plus a
//! `meta.json` that repeats the layout, so readers (the Python package's
//! `mahjong.belief.logs`) build their record type from the file, not from code.

use crate::features::{self, STATE_LEN};
use mahjong_games::Game;
use mahjong_hk::{score_win, Event, HandResult, State, WinContext};
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
pub const LAYOUT_VERSION: u32 = 1;

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
];

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
}

const NO_TRUTH: Truth = Truth {
    counts: [0; NUM_KINDS],
    waits: [false; NUM_KINDS],
    points_now: 0,
    faan_now: 0,
};

fn truth(state: &State, seat: usize) -> Truth {
    let hand = state.hand(seat);
    let mut t = NO_TRUTH;
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
    features: Vec<f32>,
    discards: [[u8; DISCARD_HISTORY]; 4],
    opp: [Truth; 3],
}

/// Captures the row for the player to act in `state`.
///
/// `agent_ids` holds the agent index of each absolute seat.
///
/// # Panics
/// Panics if the hand is over.
pub fn capture(state: &State, hand: u32, step: u16, agent_ids: [u8; 4]) -> PendingRow {
    let me = state.current_player().expect("hand in progress");
    let rel = |seat: usize| (seat + 4 - me) % 4;
    let mut feats = vec![0.0f32; STATE_LEN];
    features::encode(state, &mut feats);
    let mut history: [Vec<Tile>; 4] = Default::default();
    for e in state.events() {
        if let Event::Discard { seat, tile } = *e {
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
    for seat in 0..4 {
        agents[rel(seat)] = agent_ids[seat];
        if seat != me {
            opp[rel(seat) - 1] = truth(state, seat);
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
        }
    }
}
