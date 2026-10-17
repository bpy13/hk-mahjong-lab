//! The observation encoding for learning (Phase 3).
//!
//! [`encode`] describes a decision from the acting seat's point of view as
//! [`STATE_LEN`] floats. Seats are relative: 0 = me, 1 = next (下家), 2 = across,
//! 3 = previous (上家). Everything in it is public or the actor's own.
//!
//! | Block | Size | Contents |
//! |---|---|---|
//! | tile planes | 23 × 34 | own hand (≥1, ≥2, ≥3, 4 copies); melds per seat (count / 4); discards per seat (count / 4); each seat's last three discards (1, ⅔, ⅓); unseen copies / 4; the tile being claimed; the tile just drawn; scoring tiles (dragons, own and prevailing wind); heuristic danger; plain and faan-aware shanten after discarding each tile (scaled, 1 = not in hand) |
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
use mahjong_tiles::{Tile, TileCounts, NUM_KINDS};

/// Number of 34-wide tile planes.
pub const PLANES: usize = 23;
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
    // Shanten after each discard (turn decisions only).
    let n_melds = view.melds.len();
    let scale = |s: i8| ((s as f32 + 1.0) / 9.0).clamp(0.0, 1.0);
    let in_turn = state.phase_name() == "turn";
    for t in Tile::all_kinds() {
        let i = t.index() as usize;
        if !in_turn || view.concealed[t] == 0 {
            plane(out, 21)[i] = 1.0;
            plane(out, 22)[i] = 1.0;
            continue;
        }
        let mut h = view.concealed;
        h.remove(t);
        plane(out, 21)[i] = scale(shanten(&h, n_melds));
        let fs = faan_shanten(
            &HandView {
                concealed: &h,
                ..view.hand_view()
            },
            rules,
        )
        .shanten;
        plane(out, 22)[i] = if fs >= UNREACHABLE { 1.0 } else { scale(fs) };
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
            assert!(a
                .iter()
                .all(|x| x.is_finite() && (-0.01..=1.01).contains(x)));
        }
    }
}
