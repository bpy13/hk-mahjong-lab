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
//! A test checks that two positions that differ only in hidden tiles encode
//! identically.

use crate::danger::Danger;
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

/// [`mahjong_analysis::acceptance`] for a hand whose measure is already known
/// (`base = metric(hand)`), skipping the tiles for which `unchanged(t)` proves
/// that adding `t` does not lower the measure below `base`. Same count, fewer
/// evaluations: the faan-aware acceptance was 85 % of [`encode`]'s time.
/// Returns the count and the tiles that lower the measure (bit = tile index).
fn acceptance_given(
    hand: &TileCounts,
    unseen: &TileCounts,
    base: i8,
    metric: impl Fn(&TileCounts) -> i8,
    unchanged: impl Fn(Tile) -> bool,
) -> (u32, u64) {
    debug_assert_eq!(base, metric(hand));
    let mut c = *hand;
    let mut n = 0;
    let mut lowering = 0u64;
    for t in Tile::all_kinds() {
        if unseen[t] == 0 || c[t] >= 4 || unchanged(t) {
            continue;
        }
        c.add(t);
        if metric(&c) < base {
            n += unseen[t] as u32;
            lowering |= 1 << t.index();
        }
        c.remove(t);
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
        if let Event::Discard { seat, tile } = *e {
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
            |x| isolated(&h, x) && !(n_melds == 0 && x.is_terminal_or_honour()),
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
                        return true;
                    }
                    // An isolated tile changes no group's shapes; it must also
                    // leave the plans unchanged: thirteen orphans (no melds), the
                    // flush suit, and the honours plan (from 7 honours). A value
                    // honour going from 0 to 1 copy adds no pung plan (needs 2).
                    isolated(&h, x)
                        && !(n_melds == 0 && x.is_terminal_or_honour())
                        && !(x.is_honour() && honours_now == 6)
                        && (x.is_honour() || {
                            let mut more = all;
                            more.add(x);
                            best_suit(&more) == suit_now
                        })
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
                eat(&f);
                eat(&suits);
                eat(&rest);
                eat(&o);
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
    #[test]
    fn encoding_matches_the_golden_hash() {
        assert_eq!(golden_hash(12, 7), (1003, 12087291700939618118));
    }

    /// The same on many more positions (release build: `cargo test --release -p
    /// mahjong-search golden -- --ignored`).
    #[test]
    #[ignore]
    fn encoding_matches_the_golden_hash_on_many_positions() {
        assert_eq!(golden_hash(400, 11), (36213, 15051755790659174273));
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
