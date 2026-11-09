//! The observation encoding for learning (Phase 3).
//!
//! [`encode`] describes a decision from the acting seat's point of view as
//! [`STATE_LEN`] floats. Seats are relative: 0 = me, 1 = next (下家), 2 = across,
//! 3 = previous (上家). Everything in it is public or the actor's own.
//!
//! | Block | Size | Contents |
//! |---|---|---|
//! | tile planes | 26 × 34 | own hand (≥1, ≥2, ≥3, 4 copies); melds per seat (count / 4); discards per seat (count / 4); each seat's last three discards (1, ⅔, ⅓); unseen copies / 4; the tile being claimed; the tile just drawn; scoring tiles (dragons, own and prevailing wind); heuristic danger; and, for each tile that could be discarded: plain and faan-aware shanten after it (scaled, 1 = not in hand), plain acceptance after it (/ 60), faan-aware acceptance after it (/ 60, only for the discards at the best faan-aware shanten, as the `value` bot counts it), and the faan its target guarantees (/ 10) |
//! | scalars | 49 | phase; who discarded the claimed tile; wall left; own and prevailing wind; bonus counts per seat; own bonus tiles; exposed melds and hand sizes per seat; liability; each opponent's threat; own shanten, faan-aware shanten, target faan, concealed; minimum faan and flowers on/off |
//!
//! [`encode_oracle`] adds what is hidden ([`ORACLE_LEN`] floats: each opponent's
//! concealed tiles and the next 16 wall tiles). It is used only in training, for
//! Suphx-style oracle guiding, and is zero at play time.
//!
//! [`encode_passed`] is a separate public block ([`PASSED_LEN`] floats, plan
//! 2026-11-09 Task 18): the passed-tile record ([`crate::passed::Passed`]), which
//! discards were 摸打 or 入章, and the actor's own passed-win lock. It is a block of
//! its own, not more planes in [`encode`], so that networks trained before the
//! passed-win rule still load; a network reads it through a `passed` input slice
//! (`crate::net`).
//!
//! A test checks that two positions that differ only in hidden tiles encode
//! identically.

use crate::danger::Danger;
use crate::passed::Passed;
use crate::view::SeatView;
use mahjong_analysis::{faan_shanten, shanten, HandView, UNREACHABLE};
use mahjong_games::Game;
use mahjong_hk::{Event, State};
use mahjong_tiles::{Suit, Tile, TileCounts, NUM_KINDS};

/// Number of 34-wide tile planes.
pub const PLANES: usize = 26;
/// Number of scalar features.
pub const SCALARS: usize = 49;
/// Length of [`encode`]'s output.
pub const STATE_LEN: usize = PLANES * NUM_KINDS + SCALARS;
/// Length of [`encode_oracle`]'s output.
pub const ORACLE_LEN: usize = 4 * NUM_KINDS;

const INITIAL_WALL: f32 = 91.0;

fn plane(out: &mut [f32], p: usize) -> &mut [f32] {
    &mut out[p * NUM_KINDS..(p + 1) * NUM_KINDS]
}

/// Whether a copy of `t` added to `hand` would sit alone: no copy of `t` held
/// and, for a suited tile, no tile within two places of it in its suit. Such a
/// tile joins no meld, pair or partial meld, so no group's shape counts change.
fn isolated(hand: &TileCounts, t: Tile) -> bool {
    if hand[t] > 0 {
        return false;
    }
    let i = t.index() as usize;
    if t.is_honour() {
        return true;
    }
    let (start, pos) = (i / 9 * 9, i % 9);
    (pos.saturating_sub(2)..=(pos + 2).min(8)).all(|p| hand.0[start + p] == 0)
}

/// The suit with the most tiles, as `analysis::targets` picks its flush suit
/// (`max_by_key`: the last of equals).
fn best_suit(all: &TileCounts) -> Suit {
    Suit::ALL
        .into_iter()
        .max_by_key(|&s| all.suit_counts(s).iter().map(|&c| c as u32).sum::<u32>())
        .expect("three suits")
}

/// What [`acceptance_given`] may skip for a tile.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shortcut {
    /// Evaluate the measure.
    Evaluate,
    /// Adding the tile provably does not lower the measure.
    Unchanged,
    /// Adding the tile changes the measure exactly as adding any other tile of
    /// the same class (0–3) does: evaluate the first, reuse its answer.
    SameAs(u8),
}

/// [`mahjong_analysis::acceptance`] for a hand whose measure is already known
/// (`base = metric(hand)`), skipping the evaluations `shortcut(t)` makes
/// unnecessary. Same count, fewer evaluations: the faan-aware acceptance was
/// 85 % of [`encode`]'s time. Returns the count and the tiles that lower the
/// measure (bit = tile index).
fn acceptance_given(
    hand: &TileCounts,
    unseen: &TileCounts,
    base: i8,
    metric: impl Fn(&TileCounts) -> i8,
    shortcut: impl Fn(Tile) -> Shortcut,
) -> (u32, u64) {
    debug_assert_eq!(base, metric(hand));
    let mut c = *hand;
    let mut n = 0;
    let mut lowering = 0u64;
    let mut class_lowers: [Option<bool>; 4] = [None; 4];
    for t in Tile::all_kinds() {
        if unseen[t] == 0 || c[t] >= 4 {
            continue;
        }
        let mut eval = || {
            c.add(t);
            let lower = metric(&c) < base;
            c.remove(t);
            lower
        };
        let lower = match shortcut(t) {
            Shortcut::Unchanged => false,
            Shortcut::Evaluate => eval(),
            Shortcut::SameAs(k) => *class_lowers[k as usize].get_or_insert_with(eval),
        };
        if lower {
            n += unseen[t] as u32;
            lowering |= 1 << t.index();
        }
    }
    (n, lowering)
}

/// Encodes the decision of the player to act into `out` (length [`STATE_LEN`]).
///
/// # Panics
/// Panics if the hand is over or `out` has the wrong length.
pub fn encode(state: &State, out: &mut [f32]) {
    assert_eq!(out.len(), STATE_LEN);
    out.fill(0.0);
    let me = state.current_player().expect("hand in progress");
    let rel = |seat: usize| (seat + 4 - me) % 4;
    let view = SeatView::of(state, me);
    let rules = state.rules();

    // Own hand.
    for t in view.concealed.kinds() {
        let c = view.concealed[t] as usize;
        for k in 0..c.min(4) {
            plane(out, k)[t.index() as usize] = 1.0;
        }
    }
    // Melds per seat.
    for seat in 0..4 {
        let p = plane(out, 4 + rel(seat));
        for m in state.melds(seat).iter() {
            for t in m.tiles() {
                p[t.index() as usize] += 0.25;
            }
        }
    }
    // Discards per seat and recency.
    let mut recent: [Vec<Tile>; 4] = Default::default();
    for e in state.events() {
        if let Event::Discard { seat, tile, .. } = *e {
            let r = rel(seat as usize);
            plane(out, 8 + r)[tile.index() as usize] += 0.25;
            recent[r].push(tile);
        }
    }
    for (r, tiles) in recent.iter().enumerate() {
        let p = plane(out, 12 + r);
        for (k, t) in tiles.iter().rev().take(3).enumerate() {
            let x = &mut p[t.index() as usize];
            *x = x.max(1.0 - k as f32 / 3.0);
        }
    }
    // Unseen.
    {
        let p = plane(out, 16);
        for t in Tile::all_kinds() {
            p[t.index() as usize] = view.unseen[t] as f32 / 4.0;
        }
    }
    let claim = state.claim_tile();
    if let Some((t, _)) = claim {
        plane(out, 17)[t.index() as usize] = 1.0;
    }
    if let Some(t) = state.last_drawn() {
        plane(out, 18)[t.index() as usize] = 1.0;
    }
    {
        let p = plane(out, 19);
        for t in [
            Tile::WHITE,
            Tile::GREEN,
            Tile::RED,
            Tile::wind(me),
            Tile::wind(rules.prevailing_wind),
        ] {
            p[t.index() as usize] = 1.0;
        }
    }
    let danger = Danger::estimate(state, me);
    {
        let p = plane(out, 20);
        for t in Tile::all_kinds() {
            p[t.index() as usize] = danger.tile(t) as f32;
        }
    }
    // Shanten, acceptance and target faan after each discard (turn decisions).
    // The acceptance counts skip the tiles that provably leave the measure
    // unchanged (see `acceptance_given`); the golden-hash tests check that the
    // output is the same as with `analysis::acceptance`.
    let n_melds = view.melds.len();
    let scale = |s: i8| ((s as f32 + 1.0) / 9.0).clamp(0.0, 1.0);
    let in_turn = state.phase_name() == "turn";
    let mut faan_after = [i8::MAX; NUM_KINDS];
    // Per discard: plain shanten after it and the tiles that would lower it.
    let mut plain_after = [i8::MAX; NUM_KINDS];
    let mut plain_lowering = [0u64; NUM_KINDS];
    for t in Tile::all_kinds() {
        let i = t.index() as usize;
        if !in_turn || view.concealed[t] == 0 {
            plane(out, 21)[i] = 1.0;
            plane(out, 22)[i] = 1.0;
            continue;
        }
        let mut h = view.concealed;
        h.remove(t);
        let base = shanten(&h, n_melds);
        plane(out, 21)[i] = scale(base);
        let (acc, lowering) = acceptance_given(
            &h,
            &view.unseen,
            base,
            |x| shanten(x, n_melds),
            |x| {
                if isolated(&h, x) && !(n_melds == 0 && x.is_terminal_or_honour()) {
                    Shortcut::Unchanged
                } else {
                    Shortcut::Evaluate
                }
            },
        );
        plane(out, 23)[i] = (acc as f32 / 60.0).min(1.0);
        plain_after[i] = base;
        plain_lowering[i] = lowering;
        let fs = faan_shanten(
            &HandView {
                concealed: &h,
                ..view.hand_view()
            },
            rules,
        );
        plane(out, 22)[i] = if fs.shanten >= UNREACHABLE {
            1.0
        } else {
            scale(fs.shanten)
        };
        plane(out, 25)[i] = fs.target.map_or(0.0, |x| x.faan as f32 / 10.0);
        faan_after[i] = fs.shanten;
    }
    let best = faan_after.iter().copied().min().unwrap_or(i8::MAX);
    if in_turn && best < UNREACHABLE {
        // What the faan-aware measure's list of plans reads besides the shapes:
        // the suit with the most tiles (melds included) and the honour count.
        let mut melded = TileCounts::new();
        for m in &view.melds {
            m.add_to(&mut melded);
        }
        for t in Tile::all_kinds().filter(|t| faan_after[t.index() as usize] == best) {
            let mut h = view.concealed;
            h.remove(t);
            let hv = HandView {
                concealed: &h,
                ..view.hand_view()
            };
            let mut all = h;
            for k in Tile::all_kinds() {
                all.0[k.index() as usize] += melded[k];
            }
            let suit_now = best_suit(&all);
            let honours_now: u32 = (27..34).map(|i| all.0[i] as u32).sum();
            // The faan-aware shanten is never below the plain one (each plan
            // only drops tiles, restricts shapes or fixes pungs, and each tile
            // short of a planned pung costs one exchange). So when the two are
            // equal now, only a tile that lowers the plain shanten can lower
            // the faan-aware one.
            let i = t.index() as usize;
            let only = if best == plain_after[i] {
                plain_lowering[i]
            } else {
                u64::MAX
            };
            let (acc, _) = acceptance_given(
                &h,
                &view.unseen,
                best,
                |x| faan_shanten(&HandView { concealed: x, ..hv }, rules).shanten,
                |x| {
                    if only & (1 << x.index()) == 0 {
                        return Shortcut::Unchanged;
                    }
                    // An isolated tile changes no group's shapes; it must also
                    // leave the plans unchanged: thirteen orphans (no melds), the
                    // flush suit, and the honours plan (from 7 honours). A value
                    // honour going from 0 to 1 copy adds no pung plan (needs 2).
                    let same_plans = isolated(&h, x)
                        && !(n_melds == 0 && x.is_terminal_or_honour())
                        && !(x.is_honour() && honours_now == 6)
                        && (x.is_honour() || {
                            let mut more = all;
                            more.add(x);
                            best_suit(&more) == suit_now
                        });
                    // It can still lower the measure: a plan whose kept tiles all
                    // sit in melds and partials gains the leftover tile its eyes
                    // need (`analysis::shanten_of`). Which plans keep it depends
                    // only on its suit (or honour), so one tile per class decides.
                    match (same_plans, x.suit()) {
                        (false, _) => Shortcut::Evaluate,
                        (true, Some(s)) => Shortcut::SameAs(s as u8),
                        (true, None) => Shortcut::SameAs(3),
                    }
                },
            );
            plane(out, 24)[t.index() as usize] = (acc as f32 / 60.0).min(1.0);
        }
    }

    // Scalars.
    let s = &mut out[PLANES * NUM_KINDS..];
    let mut k = 0;
    let mut push = |x: f32| {
        s[k] = x;
        k += 1;
    };
    for ph in ["turn", "claim", "rob"] {
        push((state.phase_name() == ph) as u8 as f32);
    }
    for r in 0..4 {
        push(claim.is_some_and(|(_, from)| rel(from) == r) as u8 as f32);
    }
    push(state.wall_remaining() as f32 / INITIAL_WALL);
    for w in 0..4 {
        push((me == w) as u8 as f32);
    }
    for w in 0..4 {
        push((rules.prevailing_wind == w) as u8 as f32);
    }
    let mut by_rel = [0usize; 4];
    for (seat, r) in (0..4).map(|seat| (seat, rel(seat))) {
        by_rel[r] = seat;
    }
    for &seat in &by_rel {
        push(state.bonus(seat).count_ones() as f32 / 8.0);
    }
    for b in 0..8 {
        push((state.bonus(me) >> b & 1) as f32);
    }
    for &seat in &by_rel {
        push(state.melds(seat).exposed_count() as f32 / 4.0);
    }
    for &seat in &by_rel {
        push(state.concealed_count(seat) as f32 / 14.0);
    }
    for &seat in &by_rel {
        push(state.liable_for(seat).is_some() as u8 as f32);
    }
    for &seat in &by_rel[1..] {
        push(danger.threat(seat) as f32);
    }
    let fs = faan_shanten(&view.hand_view(), rules);
    push(scale(shanten(&view.concealed, n_melds)));
    push(if fs.shanten >= UNREACHABLE {
        1.0
    } else {
        scale(fs.shanten)
    });
    push(fs.target.map_or(0.0, |t| t.faan as f32 / 10.0));
    push(view.melds.iter().all(|m| m.concealed) as u8 as f32);
    push(rules.min_faan as f32 / 10.0);
    push(rules.flowers as u8 as f32);
    debug_assert_eq!(k, SCALARS);
}

/// Tiles per suit (positions of the suit-shared convolution).
pub const SUIT_LEN: usize = 9;
/// Honour kinds (winds, then dragons).
pub const HONOUR_KINDS: usize = 7;
/// Length of [`encode_planes`]'s suit block: 3 suits × 9 positions × 26 planes.
pub const SUIT_BLOCK_LEN: usize = 3 * SUIT_LEN * PLANES;
/// Length of [`encode_planes`]'s rest block: honours (26 planes × 7) then the
/// scalars.
pub const REST_LEN: usize = PLANES * HONOUR_KINDS + SCALARS;

/// The suit-shared CNN's view of [`encode`]'s output (plan Task 14).
///
/// `suits` gets, for suit `s` (characters, dots, bamboo) and position `i`
/// (tile `i + 1`), the 26 plane values of that tile at
/// `(s * 9 + i) * 26 + plane` (suit-major, then position, then plane, so each
/// position's planes are contiguous for the convolution). `rest` gets the
/// honour columns plane-major (`plane * 7 + honour`) followed by the
/// [`SCALARS`] scalars. The PyTorch model (`mahjong.rl.cnn`) reads the same
/// layout.
///
/// # Panics
/// Panics if a slice has the wrong length.
pub fn encode_planes(features: &[f32], suits: &mut [f32], rest: &mut [f32]) {
    assert_eq!(features.len(), STATE_LEN);
    assert_eq!(suits.len(), SUIT_BLOCK_LEN);
    assert_eq!(rest.len(), REST_LEN);
    for p in 0..PLANES {
        let plane = &features[p * NUM_KINDS..(p + 1) * NUM_KINDS];
        for (k, &v) in plane[..3 * SUIT_LEN].iter().enumerate() {
            suits[k * PLANES + p] = v;
        }
        rest[p * HONOUR_KINDS..(p + 1) * HONOUR_KINDS].copy_from_slice(&plane[3 * SUIT_LEN..]);
    }
    rest[PLANES * HONOUR_KINDS..].copy_from_slice(&features[PLANES * NUM_KINDS..]);
}

/// Tile planes of [`encode_passed`].
pub const PASSED_PLANES: usize = 14;
/// Scalars of [`encode_passed`]: the 摸打 / 入章 flags of each seat's last three
/// discards.
pub const PASSED_SCALARS: usize = 12;
/// Length of [`encode_passed`]'s output.
pub const PASSED_LEN: usize = PASSED_PLANES * NUM_KINDS + PASSED_SCALARS;

/// Encodes the passed-tile record of the player to act into `out` (length
/// [`PASSED_LEN`]). Seats are relative as in [`encode`]; all of it is public or
/// the actor's own.
///
/// | Planes | Contents |
/// |---|---|
/// | 0–2 | per opponent (next, across, previous): tiles it let pass and has not had a turn since (the passed-win lock may hold) |
/// | 3–5 | per opponent: tiles it let pass, has had a turn since, and no 換章 reset |
/// | 6–8 | per opponent: its 入章 since it let the tile pass (/ 4, at most 1), for the tiles of planes 0–5 |
/// | 9–12 | per seat (me first): its 入章 discards (count / 4) |
/// | 13 | the tiles the actor may not win on now (its own passed-win lock, rules §4.4) |
///
/// Scalars: for each seat (me first), its last three discards, most recent first:
/// 1 for 入章 (a tile from the hand), ½ for 摸打 (the tile just drawn), 0 for none.
///
/// # Panics
/// Panics if the hand is over or `out` has the wrong length.
pub fn encode_passed(state: &State, out: &mut [f32]) {
    assert_eq!(out.len(), PASSED_LEN);
    out.fill(0.0);
    let me = state.current_player().expect("hand in progress");
    let rel = |seat: usize| (seat + 4 - me) % 4;
    let passed = Passed::of(state);
    for r in 1..4 {
        let seat = (me + r) % 4;
        for t in Tile::all_kinds() {
            let c = passed.cell(seat, t);
            if !c.passed || c.reset & 1 != 0 {
                continue;
            }
            let i = t.index() as usize;
            let p = if c.turned { 3 + r - 1 } else { r - 1 };
            plane(out, p)[i] = 1.0;
            plane(out, 6 + r - 1)[i] = (c.changes as f32 / 4.0).min(1.0);
        }
    }
    let mut recent: [Vec<bool>; 4] = Default::default();
    for e in state.events() {
        if let Event::Discard {
            seat,
            tile,
            from_hand,
        } = *e
        {
            let r = rel(seat as usize);
            if from_hand {
                plane(out, 9 + r)[tile.index() as usize] += 0.25;
            }
            recent[r].push(from_hand);
        }
    }
    for t in state.passed_win_lock(me) {
        plane(out, 13)[t.index() as usize] = 1.0;
    }
    let s = &mut out[PASSED_PLANES * NUM_KINDS..];
    for (r, flags) in recent.iter().enumerate() {
        for (k, &from_hand) in flags.iter().rev().take(3).enumerate() {
            s[r * 3 + k] = if from_hand { 1.0 } else { 0.5 };
        }
    }
}

/// Encodes what is hidden from the player to act into `out` (length
/// [`ORACLE_LEN`]): each opponent's concealed tiles (relative seats 1–3, count
/// / 4) and the next 16 tiles of the wall (count / 4).
pub fn encode_oracle(state: &State, out: &mut [f32]) {
    assert_eq!(out.len(), ORACLE_LEN);
    out.fill(0.0);
    let me = state.current_player().expect("hand in progress");
    for r in 1..4 {
        let seat = (me + r) % 4;
        let h: &TileCounts = state.hand(seat);
        for t in h.kinds() {
            out[(r - 1) * NUM_KINDS + t.index() as usize] = h[t] as f32 / 4.0;
        }
    }
    for t in state.wall().iter().take(16).filter(|t| !t.is_bonus()) {
        out[3 * NUM_KINDS + t.index() as usize] += 0.25;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use mahjong_games::Policy;
    use mahjong_hk::RuleSet;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    /// FNV-1a over the bit patterns of [`encode`], [`encode_planes`] and
    /// [`encode_oracle`] at every decision of `hands` hands (varied rules, a
    /// mix of the `fast` bot and random moves). Returns `(positions, hash)`.
    fn golden_hash(hands: usize, seed: u64) -> (usize, u64) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let bot = Bot::by_name("fast").unwrap();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |xs: &[f32]| {
            for x in xs {
                for b in x.to_bits().to_le_bytes() {
                    hash ^= b as u64;
                    hash = hash.wrapping_mul(0x100_0000_01b3);
                }
            }
        };
        let (mut f, mut o) = (vec![0.0; STATE_LEN], vec![0.0; ORACLE_LEN]);
        let (mut suits, mut rest) = (vec![0.0; SUIT_BLOCK_LEN], vec![0.0; REST_LEN]);
        let mut passed = vec![0.0; PASSED_LEN];
        let mut positions = 0;
        for h in 0..hands {
            let rules = RuleSet {
                min_faan: [0, 1, 3, 3, 5][h % 5],
                flowers: h % 3 != 0,
                prevailing_wind: (h / 5) % 4,
                ..RuleSet::default()
            };
            let mut s = State::new_deal(rules, &mut rng);
            while !s.is_terminal() {
                encode(&s, &mut f);
                encode_planes(&f, &mut suits, &mut rest);
                encode_oracle(&s, &mut o);
                encode_passed(&s, &mut passed);
                eat(&f);
                eat(&suits);
                eat(&rest);
                eat(&o);
                eat(&passed);
                positions += 1;
                let legal = s.legal_actions_vec();
                let a = if rng.gen_range(0..5) == 0 {
                    legal[rng.gen_range(0..legal.len())]
                } else {
                    bot.choose(&s, &legal, &mut rng)
                };
                s.apply(a);
            }
        }
        (positions, hash)
    }

    /// The encoding must not change when it is made faster: these hashes come
    /// from the original code (commit 86e303b). Change them only on purpose,
    /// together with every model trained on the old encoding.
    ///
    /// 2026-11-09: new hashes because the `fast` bot that plays the games now
    /// keeps live waits (plan 2026-11-09 Task 1), so the positions differ. The
    /// encoding itself did not change (`features.rs` and `mahjong-analysis`
    /// untouched); the models stay valid.
    ///
    /// 2026-11-10 (plan 2026-11-09 Task 18): new hashes on purpose. The
    /// `faan_shanten` fix changes planes 22, 24, 25 and the faan scalars (and the
    /// `fast` bot's play, so the positions), and the hash now also covers
    /// [`encode_passed`]. Models trained before this read a slightly different
    /// input; they are retrained in Task 18.
    #[test]
    fn encoding_matches_the_golden_hash() {
        assert_eq!(golden_hash(12, 7), (981, 1486345915991748437));
    }

    /// The same on many more positions (release build: `cargo test --release -p
    /// mahjong-search golden -- --ignored`).
    #[test]
    #[ignore]
    fn encoding_matches_the_golden_hash_on_many_positions() {
        assert_eq!(golden_hash(400, 11), (36235, 16421221364795889110));
    }

    /// Checks that the fast acceptance counts (planes 23 and 24) equal the plain
    /// `analysis::acceptance` count at every turn decision of `hands` hands. The
    /// golden hashes used to prove this; since the `faan_shanten` fix (plan
    /// 2026-11-09 Task 18) changed the measure, these tests check the shortcut
    /// directly. Returns the faan-aware counts checked.
    fn check_acceptance(hands: usize, seed: u64) -> usize {
        let mut rng = SmallRng::seed_from_u64(seed);
        let bot = Bot::by_name("fast").unwrap();
        let mut f = vec![0.0; STATE_LEN];
        let mut checked = 0;
        for h in 0..hands {
            let rules = RuleSet {
                min_faan: [0, 1, 3, 3, 5][h % 5],
                ..RuleSet::default()
            };
            let mut s = State::new_deal(rules, &mut rng);
            while let Some(me) = s.current_player() {
                if s.phase_name() == "turn" {
                    encode(&s, &mut f);
                    let view = SeatView::of(&s, me);
                    let n = view.melds.len();
                    let faan = |c: &TileCounts| {
                        faan_shanten(
                            &HandView {
                                concealed: c,
                                ..view.hand_view()
                            },
                            &rules,
                        )
                        .shanten
                    };
                    let kinds: Vec<Tile> = view.concealed.kinds().collect();
                    let after = |t: Tile| {
                        let mut h = view.concealed;
                        h.remove(t);
                        h
                    };
                    let best = kinds.iter().map(|&t| faan(&after(t))).min().unwrap();
                    for &t in &kinds {
                        let h = after(t);
                        let i = t.index() as usize;
                        let plain =
                            mahjong_analysis::acceptance(&h, &view.unseen, |x| shanten(x, n));
                        assert_eq!(f[23 * NUM_KINDS + i], (plain as f32 / 60.0).min(1.0));
                        if best < UNREACHABLE && faan(&h) == best {
                            let fa = mahjong_analysis::acceptance(&h, &view.unseen, faan);
                            assert_eq!(
                                f[24 * NUM_KINDS + i],
                                (fa as f32 / 60.0).min(1.0),
                                "{t} out of {}",
                                view.concealed
                            );
                            checked += 1;
                        }
                    }
                }
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
            }
        }
        checked
    }

    #[test]
    fn fast_acceptance_matches_the_reference() {
        let checked = check_acceptance(24, 19);
        assert!(checked > 500, "{checked}");
    }

    /// The same on many more hands (release build: `cargo test --release -p
    /// mahjong-search reference -- --ignored`).
    #[test]
    #[ignore]
    fn fast_acceptance_matches_the_reference_on_many_hands() {
        let checked = check_acceptance(600, 23);
        assert!(checked > 10_000, "{checked}");
    }

    /// Right after a discard: the discarder's newest 摸打 / 入章 flag is set, the
    /// tile is "passed, no turn since" for every other opponent of the actor, and
    /// plane 13 is the actor's own passed-win lock.
    #[test]
    fn passed_block_follows_the_last_discard() {
        let mut rng = SmallRng::seed_from_u64(3);
        let bot = Bot::by_name("fast").unwrap();
        let mut out = vec![0.0; PASSED_LEN];
        let (mut checked, mut from_hand_seen, mut drawn_seen) = (0, 0, 0);
        for _ in 0..20 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(me) = s.current_player() {
                encode_passed(&s, &mut out);
                let lock = &out[13 * NUM_KINDS..14 * NUM_KINDS];
                for t in Tile::all_kinds() {
                    let want = s.passed_win_locked(me, t) as u8 as f32;
                    assert_eq!(lock[t.index() as usize], want);
                }
                if let Some(&Event::Discard {
                    seat,
                    tile,
                    from_hand,
                }) = s.events().last()
                {
                    let q = (seat as usize + 4 - me) % 4;
                    let flag = out[PASSED_PLANES * NUM_KINDS + q * 3];
                    assert_eq!(flag, if from_hand { 1.0 } else { 0.5 });
                    for r in (1..4).filter(|&r| r != q) {
                        assert_eq!(plane(&mut out, r - 1)[tile.index() as usize], 1.0);
                    }
                    checked += 1;
                    if from_hand {
                        from_hand_seen += 1;
                    } else {
                        drawn_seen += 1;
                    }
                }
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
            }
        }
        assert!(checked > 100 && from_hand_seen > 0 && drawn_seen > 0);
    }

    #[test]
    fn hidden_tiles_do_not_change_the_encoding() {
        let mut rng = SmallRng::seed_from_u64(1);
        let bot = Bot::by_name("fast").unwrap();
        for _ in 0..30 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            let steps = rng.gen_range(0..80);
            for _ in 0..steps {
                if s.is_terminal() {
                    break;
                }
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
            }
            if s.is_terminal() {
                continue;
            }
            let other = s.determinize(s.current_player().unwrap(), &mut rng);
            let mut pa = vec![0.0; PASSED_LEN];
            let mut pb = vec![0.0; PASSED_LEN];
            encode_passed(&s, &mut pa);
            encode_passed(&other, &mut pb);
            assert_eq!(pa, pb, "the passed block depends on hidden tiles");
            assert!(pa.iter().all(|x| (0.0..=1.0).contains(x)));
            let mut a = vec![0.0; STATE_LEN];
            let mut b = vec![0.0; STATE_LEN];
            encode(&s, &mut a);
            encode(&other, &mut b);
            assert_eq!(a, b, "the encoding depends on hidden tiles");
            // The CNN view is a permutation of the flat encoding.
            let mut suits = vec![0.0; SUIT_BLOCK_LEN];
            let mut rest = vec![0.0; REST_LEN];
            encode_planes(&a, &mut suits, &mut rest);
            assert_eq!(SUIT_BLOCK_LEN + REST_LEN, STATE_LEN);
            let t = 9 + 4; // 5p
            for p in 0..PLANES {
                assert_eq!(suits[t * PLANES + p], a[p * NUM_KINDS + t]);
                assert_eq!(rest[p * HONOUR_KINDS + 2], a[p * NUM_KINDS + 29]);
            }
            assert_eq!(rest[REST_LEN - 1], a[STATE_LEN - 1]);
            assert!(a
                .iter()
                .all(|x| x.is_finite() && (-0.01..=1.01).contains(x)));
        }
    }
}
