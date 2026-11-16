//! Search-labelled decisions for distilling a network (plan Task 13).
//!
//! The labeller plays single hands. In each hand one seat (the **labelled
//! seat**, hand index mod 4) is played by the **teacher**, a search agent such as
//! belief-PIMC-64; the other three seats play a cheaper agent (`ev` by default).
//! At every teacher decision with more than one legal action it writes one row:
//! the acting seat's public features and the hidden oracle block, the legal
//! mask, the search's per-candidate numbers (means, paired differences with the
//! baseline and their standard errors), the move played, the target plan of the
//! hand after that move ([`target_class`]), and, once the hand is over, the four
//! final point deltas and how the labelled seat's hand ended.
//!
//! Rows use the packed-row machinery of [`crate::log`] (chunk files plus a
//! `meta.json` listing [`FIELDS`]); the Python reader is
//! `mahjong.rl.labels`. Seats in a row are relative to the acting seat
//! (0 = actor, 1 = next, 2 = across, 3 = previous).
//!
//! Layout 3 (plan 2026-11-16 Task 4) adds the full discard history
//! (`features::encode_history`) in a packed form of [`HISTORY_BYTES`] bytes
//! per row ([`pack_history`]); [`unpack_history`] and the Python reader's
//! `mahjong.rl.labels.history` turn it back into the 3,460-float block.
//!
//! Hands are seeded from `(seed, chunk, hand)` alone, so a chunk's rows do not
//! depend on the number of threads or on earlier chunks, and a rerun resumes
//! after the last complete chunk with the same rows a single run would write.

use crate::agent::Agent;
use crate::features::{
    self, DiscardRecord, HISTORY_DEPTH, HISTORY_LEN, HISTORY_PLANES, ORACLE_LEN, PASSED_LEN,
    STATE_LEN,
};
use crate::log::{f16_bits, layout_bytes, Field};
use crate::view::SeatView;
use mahjong_analysis::{faan_shanten, HandView, Structure, SuitPlan, Target};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, HandResult, Pattern, RuleSet, State, NUM_ACTION_CODES};
use mahjong_tiles::{Suit, Tile};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;

/// Layout version written to `meta.json`; bump when [`FIELDS`] changes.
pub const LABEL_LAYOUT_VERSION: u32 = 3;
/// Candidate slots per row (PIMC-64 with 4 discard candidates evaluates at most
/// 4 discards, the special actions and the anchor's move).
pub const MAX_CANDIDATES: usize = 12;
/// Number of target classes: plan kind (5) × suit (3) × structure (3).
pub const NUM_TARGETS: usize = 45;
/// "No feasible target" and empty candidate slots.
pub const NO_TARGET: u8 = 255;
pub const NO_ACTION: u16 = u16::MAX;

const fn f(name: &'static str, dtype: &'static str, shape: &'static [usize]) -> Field {
    Field { name, dtype, shape }
}

/// The row layout, in order (little-endian, packed).
pub const FIELDS: &[Field] = &[
    // Global hand index inside its chunk, decision number in the hand.
    f("hand", "u4", &[]),
    f("chunk", "u2", &[]),
    f("step", "u2", &[]),
    // Absolute seat of the actor (0 = dealer), 0 = turn / 1 = claim / 2 = rob.
    f("seat", "u1", &[]),
    f("phase", "u1", &[]),
    f("wall_remaining", "u1", &[]),
    // The actor's public features (features::encode) and the hidden oracle
    // block (features::encode_oracle), as float16.
    f("features", "f2", &[STATE_LEN]),
    f("oracle", "f2", &[ORACLE_LEN]),
    // The passed-tile record and 摸打 / 入章 (features::encode_passed), as float16
    // (layout version 2, plan 2026-11-09 Task 18).
    f("passed", "f2", &[PASSED_LEN]),
    // The full discard history, packed (pack_history; layout version 3, plan
    // 2026-11-16 Task 4).
    f("history", "u1", &[HISTORY_BYTES]),
    // 1 for each legal action code.
    f("legal", "u1", &[NUM_ACTION_CODES]),
    // The search: candidates (action code, 65535 = empty slot), samples per
    // candidate, mean points, 95 % half-width, mean paired difference with the
    // baseline and its standard error, win and deal-in rate in the rollouts.
    f("n_candidates", "u1", &[]),
    f("cand_code", "u2", &[MAX_CANDIDATES]),
    f("cand_n", "u2", &[MAX_CANDIDATES]),
    f("cand_mean", "f4", &[MAX_CANDIDATES]),
    f("cand_ci95", "f4", &[MAX_CANDIDATES]),
    f("cand_diff", "f4", &[MAX_CANDIDATES]),
    f("cand_diff_se", "f4", &[MAX_CANDIDATES]),
    f("cand_win", "f4", &[MAX_CANDIDATES]),
    f("cand_deal_in", "f4", &[MAX_CANDIDATES]),
    // The search's baseline (anchor) move, the move played, the z it used, and
    // the determinizations sampled.
    f("baseline", "u2", &[]),
    f("chosen", "u2", &[]),
    f("z", "f4", &[]),
    f("samples", "u2", &[]),
    // Target class of the hand after the chosen move (target_class; 255 = none)
    // and the faan that target guarantees.
    f("target", "u1", &[]),
    f("target_faan", "u1", &[]),
    // Final net points per relative seat (sum to zero).
    f("deltas", "i4", &[4]),
    // How the actor's hand ended: 0 = no win, 1 = discard win (or robbed kong),
    // 2 = self-drawn; its faan; the faan patterns of the win as bits
    // (bit i = Pattern::ALL[i]).
    f("won", "u1", &[]),
    f("win_faan", "u1", &[]),
    f("win_patterns", "u4", &[]),
];

/// Bytes of the packed discard history: one per seat and slot, then one
/// overflow count per seat.
pub const HISTORY_BYTES: usize = HISTORY_PLANES + 4;

/// Packs the discard history of the player to act (what
/// `features::encode_history` encodes) into [`HISTORY_BYTES`] bytes. Byte
/// `r * 24 + k` (relative seat `r`, its `k`-th discard) is 0 for an empty
/// slot, else `1 + tile + 34 * from_hand + 68 * claimed` (at most 136); byte
/// `96 + r` is the number of seat `r`'s discards beyond the 24th (at most 255).
///
/// # Panics
/// Panics if the hand is over.
pub fn pack_history(state: &State) -> [u8; HISTORY_BYTES] {
    let me = state.current_player().expect("hand in progress");
    pack_history_of(&features::discard_records(state), me)
}

/// [`pack_history`] from a list of discards (in the order thrown) for the
/// absolute seat `me`.
pub fn pack_history_of(records: &[DiscardRecord], me: usize) -> [u8; HISTORY_BYTES] {
    let mut out = [0u8; HISTORY_BYTES];
    let mut count = [0usize; 4];
    for d in records {
        let r = (d.seat as usize + 4 - me) % 4;
        let k = count[r];
        count[r] += 1;
        if k < HISTORY_DEPTH {
            out[r * HISTORY_DEPTH + k] =
                1 + d.tile.index() + 34 * d.from_hand as u8 + 68 * d.claimed as u8;
        }
    }
    for (r, &n) in count.iter().enumerate() {
        out[HISTORY_PLANES + r] = n.saturating_sub(HISTORY_DEPTH).min(255) as u8;
    }
    out
}

/// Unpacks [`pack_history`]'s bytes into `features::encode_history`'s block
/// (`out` of length `HISTORY_LEN`).
///
/// # Panics
/// Panics if a length is wrong.
pub fn unpack_history(packed: &[u8], out: &mut [f32]) {
    assert_eq!(packed.len(), HISTORY_BYTES);
    assert_eq!(out.len(), HISTORY_LEN);
    out.fill(0.0);
    let (planes, s) = out.split_at_mut(HISTORY_PLANES * 34);
    for (slot, &b) in packed[..HISTORY_PLANES].iter().enumerate() {
        if b == 0 {
            continue;
        }
        let v = (b - 1) as usize;
        planes[slot * 34 + v % 34] = 1.0;
        s[2 * slot] = if (v / 34) % 2 == 1 { 1.0 } else { 0.5 };
        s[2 * slot + 1] = (v / 68) as f32;
    }
    for r in 0..4 {
        s[2 * HISTORY_PLANES + r] = (packed[HISTORY_PLANES + r] as f32 / 4.0).min(1.0);
    }
}

/// Bytes per row.
pub fn label_row_bytes() -> usize {
    layout_bytes(FIELDS)
}

/// The class of a target plan: `kind * 9 + suit * 3 + structure`, with kind
/// 0 = mixed, 1 = half flush, 2 = full flush, 3 = honours, 4 = thirteen
/// orphans; suit 0 = characters, 1 = dots, 2 = bamboo (0 for plans without a
/// suit); structure 0 = any, 1 = all pungs, 2 = all chows. A suit swap of the
/// position permutes the flush classes' suit digit (the network's suit-swap
/// augmentation relies on this).
pub fn target_class(t: &Target) -> u8 {
    let suit = |s: Suit| s as u8;
    let (kind, s) = match t.plan {
        SuitPlan::Mixed => (0, 0),
        SuitPlan::HalfFlush(s) => (1, suit(s)),
        SuitPlan::FullFlush(s) => (2, suit(s)),
        SuitPlan::Honours => (3, 0),
        SuitPlan::Orphans => (4, 0),
    };
    let st = match t.structure {
        Structure::Any => 0,
        Structure::AllPungs => 1,
        Structure::AllChows => 2,
    };
    kind * 9 + s * 3 + st
}

/// The target class and guaranteed faan of the actor's hand after `action`:
/// for a discard, the hand without that tile; for every other move, the hand as
/// it is now. `(NO_TARGET, 0)` when no target reaches the minimum.
pub fn target_after(state: &State, action: Action) -> (u8, u8) {
    let view = SeatView::of_current(state);
    let mut concealed = view.concealed;
    if let Action::Discard(t) = action {
        concealed.remove(t);
    }
    let fs = faan_shanten(
        &HandView {
            concealed: &concealed,
            ..view.hand_view()
        },
        state.rules(),
    );
    fs.target
        .map_or((NO_TARGET, 0), |t| (target_class(&t), t.faan))
}

/// One labelled decision before the hand's outcome is known.
#[derive(Clone, Debug)]
pub struct LabelRow {
    pub hand: u32,
    pub chunk: u16,
    pub step: u16,
    pub seat: u8,
    pub phase: u8,
    pub wall_remaining: u8,
    pub features: Vec<f32>,
    pub oracle: Vec<f32>,
    pub passed: Vec<f32>,
    /// The packed discard history ([`pack_history`]).
    pub history: [u8; HISTORY_BYTES],
    pub legal: Vec<u8>,
    /// (code, n, mean, ci95, diff, diff_se, win rate, deal-in rate).
    pub candidates: Vec<(u16, u16, [f32; 6])>,
    pub baseline: u16,
    pub chosen: u16,
    pub z: f32,
    pub samples: u16,
    pub target: u8,
    pub target_faan: u8,
}

/// The outcome of a hand seen from `seat`: relative deltas, how it ended, faan
/// and pattern bits.
fn outcome(result: &HandResult, seat: usize) -> ([i32; 4], u8, u8, u32) {
    let deltas = std::array::from_fn(|r| result.deltas[(seat + r) % 4]);
    match result.wins.iter().find(|w| w.seat == seat) {
        None => (deltas, 0, 0, 0),
        Some(w) => {
            let bits = w
                .score
                .patterns
                .iter()
                .filter_map(|(p, _)| Pattern::ALL.iter().position(|q| q == p))
                .fold(0u32, |acc, i| acc | 1 << i);
            (deltas, if w.self_drawn { 2 } else { 1 }, w.score.faan, bits)
        }
    }
}

impl LabelRow {
    /// Appends the row, completed with the hand's `result`, in the [`FIELDS`]
    /// layout.
    pub fn write(&self, result: &HandResult, out: &mut Vec<u8>) {
        let start = out.len();
        out.extend_from_slice(&self.hand.to_le_bytes());
        out.extend_from_slice(&self.chunk.to_le_bytes());
        out.extend_from_slice(&self.step.to_le_bytes());
        out.push(self.seat);
        out.push(self.phase);
        out.push(self.wall_remaining);
        for &x in self.features.iter().chain(&self.oracle).chain(&self.passed) {
            out.extend_from_slice(&f16_bits(x).to_le_bytes());
        }
        out.extend_from_slice(&self.history);
        out.extend_from_slice(&self.legal);
        out.push(self.candidates.len() as u8);
        let slot = |i: usize| self.candidates.get(i);
        for i in 0..MAX_CANDIDATES {
            out.extend_from_slice(&slot(i).map_or(NO_ACTION, |c| c.0).to_le_bytes());
        }
        for i in 0..MAX_CANDIDATES {
            out.extend_from_slice(&slot(i).map_or(0, |c| c.1).to_le_bytes());
        }
        for k in 0..6 {
            for i in 0..MAX_CANDIDATES {
                out.extend_from_slice(&slot(i).map_or(0.0, |c| c.2[k]).to_le_bytes());
            }
        }
        out.extend_from_slice(&self.baseline.to_le_bytes());
        out.extend_from_slice(&self.chosen.to_le_bytes());
        out.extend_from_slice(&self.z.to_le_bytes());
        out.extend_from_slice(&self.samples.to_le_bytes());
        out.push(self.target);
        out.push(self.target_faan);
        let (deltas, won, faan, bits) = outcome(result, self.seat as usize);
        for d in deltas {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.push(won);
        out.push(faan);
        out.extend_from_slice(&bits.to_le_bytes());
        debug_assert_eq!(out.len() - start, label_row_bytes());
    }
}

/// Who plays a labelling hand.
#[derive(Clone, Debug)]
pub struct Labeller {
    /// The labelled seat's agent (a search agent, or any agent for tests: a
    /// bot teacher writes rows with no candidates).
    pub teacher: Agent,
    /// The other three seats' agent.
    pub others: Agent,
    pub rules: RuleSet,
    pub seed: u64,
}

/// The seed of hand `hand` of chunk `chunk`.
pub fn hand_seed(seed: u64, chunk: u32, hand: u32) -> u64 {
    let x = ((chunk as u64) << 32) | hand as u64;
    seed ^ x.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

fn phase_code(state: &State) -> u8 {
    match state.phase_name() {
        "turn" => 0,
        "claim" => 1,
        _ => 2,
    }
}

impl Labeller {
    /// Plays hand `hand` of chunk `chunk` and returns its labelled rows
    /// (encoded, outcome included) and the number of rows.
    ///
    /// The labelled seat is `hand % 4`.
    pub fn label_hand(&self, chunk: u32, hand: u32) -> (Vec<u8>, usize) {
        self.label_hand_with(chunk, hand, |_, _| {})
    }

    /// [`Labeller::label_hand`] that also shows `seen` each labelled position
    /// and its row (before the move is applied), for tests.
    pub fn label_hand_with(
        &self,
        chunk: u32,
        hand: u32,
        mut seen: impl FnMut(&State, &LabelRow),
    ) -> (Vec<u8>, usize) {
        let mut rng = SmallRng::seed_from_u64(hand_seed(self.seed, chunk, hand));
        let mut wall = State::all_tiles(&self.rules);
        wall.shuffle(&mut rng);
        let mut s = State::from_wall(self.rules, wall);
        let me = (hand % 4) as usize;
        let mut rows = Vec::new();
        let mut legal = Vec::new();
        let mut step = 0u16;
        while let Some(p) = s.current_player() {
            s.legal_actions(&mut legal);
            let a = if p == me && legal.len() > 1 {
                let (row, a) = self.label_decision(&s, &legal, chunk, hand, step, &mut rng);
                seen(&s, &row);
                rows.push(row);
                a
            } else {
                let agent = if p == me { &self.teacher } else { &self.others };
                agent.choose(&s, &legal, &mut rng)
            };
            s.apply(a);
            step = step.saturating_add(1);
        }
        let result = s.result().expect("finished");
        let mut out = Vec::with_capacity(rows.len() * label_row_bytes());
        for r in &rows {
            r.write(result, &mut out);
        }
        (out, rows.len())
    }

    fn label_decision(
        &self,
        s: &State,
        legal: &[Action],
        chunk: u32,
        hand: u32,
        step: u16,
        rng: &mut SmallRng,
    ) -> (LabelRow, Action) {
        let mut feats = vec![0.0f32; STATE_LEN];
        features::encode(s, &mut feats);
        let mut oracle = vec![0.0f32; ORACLE_LEN];
        features::encode_oracle(s, &mut oracle);
        let mut passed = vec![0.0f32; PASSED_LEN];
        features::encode_passed(s, &mut passed);
        let mut mask = vec![0u8; NUM_ACTION_CODES];
        for a in legal {
            mask[a.code() as usize] = 1;
        }
        let (candidates, baseline, z, samples, chosen) = match self.teacher.search(s, legal, rng) {
            Some((an, z)) => {
                let chosen = an.choose(z);
                let cands = an
                    .moves
                    .iter()
                    .take(MAX_CANDIDATES)
                    .map(|m| {
                        (
                            m.action.code(),
                            m.n.min(u16::MAX as u32) as u16,
                            [
                                m.mean as f32,
                                m.ci95 as f32,
                                m.diff as f32,
                                m.diff_se as f32,
                                m.win_rate as f32,
                                m.deal_in_rate as f32,
                            ],
                        )
                    })
                    .collect();
                let n = an.samples.min(u16::MAX as usize) as u16;
                (cands, an.baseline.code(), z as f32, n, chosen)
            }
            None => {
                let a = self.teacher.choose(s, legal, rng);
                (Vec::new(), a.code(), 0.0, 0, a)
            }
        };
        let (target, target_faan) = target_after(s, chosen);
        let row = LabelRow {
            hand,
            chunk: chunk.min(u16::MAX as u32) as u16,
            step,
            seat: s.current_player().expect("to act") as u8,
            phase: phase_code(s),
            wall_remaining: s.wall_remaining().min(255) as u8,
            features: feats,
            oracle,
            passed,
            history: pack_history(s),
            legal: mask,
            candidates,
            baseline,
            chosen: chosen.code(),
            z,
            samples,
            target,
            target_faan,
        };
        (row, chosen)
    }

    /// Labels chunk `chunk`: plays its hands in order, `batch` at a time in
    /// parallel, until at least `decisions` rows exist, and keeps the first
    /// `decisions` rows. Returns the encoded rows and the hands played.
    /// `progress(rows so far, hands so far)` is called after each batch.
    pub fn label_chunk(
        &self,
        chunk: u32,
        decisions: usize,
        batch: usize,
        mut progress: impl FnMut(usize, u32),
    ) -> (Vec<u8>, u32) {
        let row = label_row_bytes();
        let mut out = Vec::with_capacity(decisions * row);
        let mut rows = 0usize;
        let mut next = 0u32;
        while rows < decisions {
            let hands: Vec<u32> = (next..next + batch.max(1) as u32).collect();
            let done: Vec<(Vec<u8>, usize)> = hands
                .par_iter()
                .map(|&h| self.label_hand(chunk, h))
                .collect();
            for (bytes, n) in done {
                if rows >= decisions {
                    break;
                }
                let keep = n.min(decisions - rows);
                out.extend_from_slice(&bytes[..keep * row]);
                rows += keep;
                next += 1;
            }
            progress(rows, next);
        }
        (out, next)
    }
}

/// Every [`Tile`] kind's index under a suit permutation `perm` (new suit of each
/// old suit): what the network's suit-swap augmentation applies to tile planes
/// and tile action codes. Honours keep their index.
pub fn suit_permuted_index(tile: u8, perm: [u8; 3]) -> u8 {
    let t = Tile::from_index(tile);
    match t.suit() {
        Some(s) => perm[s as usize] * 9 + tile % 9,
        None => tile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labeller(teacher: &str) -> Labeller {
        Labeller {
            teacher: Agent::parse(teacher).unwrap(),
            others: Agent::parse("value").unwrap(),
            rules: RuleSet::default(),
            seed: 3,
        }
    }

    #[test]
    fn target_classes_cover_the_45_slots() {
        let mut seen = std::collections::HashSet::new();
        for plan in [
            SuitPlan::Mixed,
            SuitPlan::HalfFlush(Suit::Characters),
            SuitPlan::HalfFlush(Suit::Dots),
            SuitPlan::HalfFlush(Suit::Bamboo),
            SuitPlan::FullFlush(Suit::Bamboo),
            SuitPlan::Honours,
            SuitPlan::Orphans,
        ] {
            for structure in [Structure::Any, Structure::AllPungs, Structure::AllChows] {
                let c = target_class(&Target {
                    plan,
                    structure,
                    pungs: vec![],
                    faan: 3,
                    limit: false,
                });
                assert!((c as usize) < NUM_TARGETS);
                assert!(seen.insert(c), "{plan:?} {structure:?} collides");
            }
        }
        assert_eq!(
            target_class(&Target {
                plan: SuitPlan::FullFlush(Suit::Dots),
                structure: Structure::AllPungs,
                pungs: vec![],
                faan: 7,
                limit: false,
            }),
            2 * 9 + 3 + 1
        );
    }

    #[test]
    fn rows_have_the_layout_size_and_zero_sum_deltas() {
        let l = labeller("value");
        let (bytes, n) = l.label_hand(0, 1);
        assert!(n > 0);
        assert_eq!(bytes.len(), n * label_row_bytes());
        // The same hand twice gives the same bytes (seeded from chunk and hand).
        assert_eq!(l.label_hand(0, 1).0, bytes);
        assert_ne!(l.label_hand(1, 1).0, bytes);
    }

    #[test]
    fn pimc_teacher_rows_carry_the_search() {
        let l = labeller("pimc:samples=4,candidates=2");
        let (out, hands) = l.label_chunk(0, 6, 2, |_, _| {});
        assert_eq!(out.len(), 6 * label_row_bytes());
        assert!(hands >= 1);
        // n_candidates sits right after the legal mask.
        let off = 4
            + 2
            + 2
            + 3
            + 2 * (STATE_LEN + ORACLE_LEN + PASSED_LEN)
            + HISTORY_BYTES
            + NUM_ACTION_CODES;
        for r in 0..6 {
            let n = out[r * label_row_bytes() + off];
            assert!(n >= 1 && n as usize <= MAX_CANDIDATES, "row {r}: {n}");
        }
    }

    /// Plan 2026-11-16 Task 4: on 1,000 decisions of the `stage` teacher, the
    /// history bytes in the written row unpack to `encode_history` of the same
    /// position, and every packed slot kind occurs.
    #[test]
    fn history_in_the_row_equals_encode_history() {
        let l = Labeller {
            teacher: Agent::parse("stage").unwrap(),
            others: Agent::parse("stage").unwrap(),
            rules: RuleSet::default(),
            seed: 11,
        };
        let row_bytes = label_row_bytes();
        let off = 4 + 2 + 2 + 3 + 2 * (STATE_LEN + ORACLE_LEN + PASSED_LEN);
        let (mut checked, mut drawn, mut from_hand, mut claimed) = (0, 0, 0, 0);
        let mut want = vec![0.0f32; HISTORY_LEN];
        let mut got = vec![0.0f32; HISTORY_LEN];
        let mut hand = 0;
        while checked < 1000 {
            let mut expected = Vec::new();
            let (bytes, n) = l.label_hand_with(0, hand, |s, row| {
                features::encode_history(s, &mut want);
                expected.push((want.clone(), row.history));
            });
            assert_eq!(n, expected.len());
            for (i, (w, packed)) in expected.iter().enumerate() {
                let stored = &bytes[i * row_bytes + off..i * row_bytes + off + HISTORY_BYTES];
                assert_eq!(stored, packed, "hand {hand} row {i}");
                unpack_history(stored, &mut got);
                assert_eq!(&got, w, "hand {hand} row {i}");
                for &b in &stored[..HISTORY_PLANES] {
                    match b {
                        0 => {}
                        1..=34 => drawn += 1,
                        35..=68 => from_hand += 1,
                        _ => claimed += 1,
                    }
                }
                checked += 1;
            }
            hand += 1;
        }
        assert!(drawn > 0 && from_hand > 0 && claimed > 0);
    }

    #[test]
    fn packed_history_counts_discards_beyond_24() {
        let recs: Vec<DiscardRecord> = (0..30)
            .map(|i| DiscardRecord {
                seat: 2,
                tile: Tile::from_index((i % 34) as u8),
                from_hand: i % 2 == 0,
                claimed: i == 3,
            })
            .collect();
        let packed = pack_history_of(&recs, 1);
        // Seat 2 is seat 1's next seat (relative 1).
        assert_eq!(packed[HISTORY_DEPTH], 1 + 34);
        assert_eq!(packed[HISTORY_DEPTH + 3], 1 + 3 + 68);
        assert_eq!(packed[HISTORY_PLANES + 1], 6);
        let mut want = vec![0.0f32; HISTORY_LEN];
        let mut got = vec![0.0f32; HISTORY_LEN];
        features::encode_history_of(&recs, 1, &mut want);
        unpack_history(&packed, &mut got);
        assert_eq!(got, want);
    }

    #[test]
    fn suit_permutation_maps_suits_and_keeps_honours() {
        let perm = [2, 0, 1];
        assert_eq!(suit_permuted_index(0, perm), 18);
        assert_eq!(suit_permuted_index(9 + 4, perm), 4);
        assert_eq!(suit_permuted_index(26, perm), 9 + 8);
        assert_eq!(suit_permuted_index(30, perm), 30);
    }
}
