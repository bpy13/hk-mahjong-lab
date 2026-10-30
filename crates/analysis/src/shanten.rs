//! Shanten: how many tile exchanges a hand is from being ready (one tile from
//! complete). `-1` means complete, `0` ready, `1` one exchange from ready, …
//!
//! The standard form uses the classic count: with `m` melds, `t` partial melds
//! (pairs, two-sided or gapped pairs of a suit) and `p` = 1 if a pair is set aside
//! as the eyes, a hand needing 4 melds is `8 − 2m − t − p` exchanges from complete,
//! where at most `4 − m` partials count. Each suit is searched once per count
//! vector and cached; honours are counted directly.
//!
//! Like the usual formula, this ignores how many copies remain: a hand whose
//! only wait would be a fifth copy of a tile it already holds four of counts as
//! ready. Tile-acceptance counts ([`crate::acceptance`]) do use availability.

use mahjong_tiles::{Suit, Tile, TileCounts};
use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Which meld shapes a target hand may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Structure {
    /// Chows and pungs.
    Any,
    /// Pungs (and kongs) only (#15).
    AllPungs,
    /// Chows only (#14).
    AllChows,
}

/// For a group of tiles (one suit, or all honours): for each pair count `p`
/// (0 or 1) and meld count `m` (0–4), the most partial melds `t` that can be
/// formed alongside, or -1 if that `(p, m)` is impossible. For fixed `p` and
/// `m`, more partials are never worse, so this frontier is all that matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Frontier([[i8; 5]; 2]);

impl Frontier {
    const EMPTY: Frontier = Frontier([[-1; 5]; 2]);
    /// The frontier of no tiles at all (the identity of [`Frontier::combine`]).
    pub(crate) const ZERO: Frontier = Frontier([[0, -1, -1, -1, -1], [-1; 5]]);

    fn insert(&mut self, m: u8, t: u8, p: u8) {
        let cell = &mut self.0[p as usize][m.min(4) as usize];
        *cell = (*cell).max(t.min(4) as i8);
    }

    /// Every way to take one option from each group. Associative and
    /// commutative (sums capped at 4, then the maximum), so groups can be
    /// combined in any order.
    pub(crate) fn combine(&self, other: &Frontier) -> Frontier {
        let mut out = Frontier::EMPTY;
        for p1 in 0..2 {
            for m1 in 0..5 {
                let t1 = self.0[p1][m1];
                if t1 < 0 {
                    continue;
                }
                for p2 in 0..2 - p1 {
                    for m2 in 0..5 {
                        let t2 = other.0[p2][m2];
                        if t2 >= 0 {
                            out.insert((m1 + m2) as u8, (t1 + t2) as u8, (p1 + p2) as u8);
                        }
                    }
                }
            }
        }
        out
    }
}

/// A cheap hasher for the cache's packed `u32` keys. The default SipHash took
/// most of the time of a cache hit, and the faan-aware measures look up the
/// cache hundreds of times per encoded position.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u32(&mut self, k: u32) {
        self.write_u64(k as u64);
    }
    fn write_u64(&mut self, k: u64) {
        // Fibonacci hashing; hashbrown takes its bucket from the top bits and its
        // tag from the top 7, so mix them well.
        self.0 = (self.0 ^ k).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

thread_local! {
    /// Key: the base-5 count code × 6 + honours × 3 + structure.
    static CACHE: RefCell<HashMap<u32, Frontier, BuildHasherDefault<KeyHasher>>> =
        RefCell::new(HashMap::default());
}

/// The frontier for one group: a suit (`honours = false`, 9 counts) or the seven
/// honours (`honours = true`, which cannot form chows).
pub(crate) fn group_frontier(counts: &[u8], honours: bool, structure: Structure) -> Frontier {
    let code = counts.iter().rev().fold(0u32, |k, &c| k * 5 + c as u32);
    if code == 0 {
        return Frontier::ZERO;
    }
    let key = code * 6 + honours as u32 * 3 + structure as u32;
    if let Some(f) = CACHE.with(|c| c.borrow().get(&key).copied()) {
        return f;
    }
    let mut out = Frontier::EMPTY;
    let mut c = [0u8; 9];
    c[..counts.len()].copy_from_slice(counts);
    let chows = !honours && structure != Structure::AllPungs;
    search(
        &mut c,
        counts.len(),
        0,
        (0, 0, 0),
        chows,
        structure != Structure::AllChows,
        &mut out,
    );
    CACHE.with(|cache| cache.borrow_mut().insert(key, out));
    out
}

#[allow(clippy::too_many_arguments)]
fn search(
    c: &mut [u8; 9],
    len: usize,
    from: usize,
    (m, t, p): (u8, u8, u8),
    chows: bool,
    pungs: bool,
    out: &mut Frontier,
) {
    let Some(i) = (from..len).find(|&i| c[i] > 0) else {
        out.insert(m, t, p);
        return;
    };
    if pungs && c[i] >= 3 {
        c[i] -= 3;
        search(c, len, i, (m + 1, t, p), chows, pungs, out);
        c[i] += 3;
    }
    if chows && i + 2 < len && c[i + 1] > 0 && c[i + 2] > 0 {
        c[i] -= 1;
        c[i + 1] -= 1;
        c[i + 2] -= 1;
        search(c, len, i, (m + 1, t, p), chows, pungs, out);
        c[i] += 1;
        c[i + 1] += 1;
        c[i + 2] += 1;
    }
    if c[i] >= 2 {
        c[i] -= 2;
        if p == 0 {
            search(c, len, i, (m, t, 1), chows, pungs, out);
        }
        if pungs {
            search(c, len, i, (m, t + 1, p), chows, pungs, out);
        }
        c[i] += 2;
    }
    if chows {
        for gap in [1, 2] {
            if i + gap < len && c[i + gap] > 0 {
                c[i] -= 1;
                c[i + gap] -= 1;
                search(c, len, i, (m, t + 1, p), chows, pungs, out);
                c[i] += 1;
                c[i + gap] += 1;
            }
        }
    }
    // Leave one copy of tile i unused.
    c[i] -= 1;
    search(c, len, i, (m, t, p), chows, pungs, out);
    c[i] += 1;
}

/// Standard-form shanten of `concealed` with `melds` melds already on the table,
/// using only meld shapes allowed by `structure`. Works for 13- and 14-tile hands.
pub fn standard_shanten(concealed: &TileCounts, melds: usize, structure: Structure) -> i8 {
    let mut f = group_frontier(&concealed.0[27..34], true, structure);
    for suit in Suit::ALL {
        f = f.combine(&group_frontier(
            &concealed.suit_counts(suit),
            false,
            structure,
        ));
    }
    shanten_of(&f, melds)
}

/// Standard-form shanten from the combined frontier of all four groups.
pub(crate) fn shanten_of(f: &Frontier, melds: usize) -> i8 {
    let need = 4 - melds.min(4) as i8;
    let mut best = i8::MAX;
    for p in 0..2 {
        for m in 0..5 {
            let t = f.0[p][m];
            if t < 0 {
                continue;
            }
            let m = (m as i8).min(need);
            let t = t.min(need - m);
            best = best.min(2 * need - 2 * m - t - p as i8);
        }
    }
    best
}

/// Thirteen-orphans shanten (only meaningful with no melds).
pub fn orphans_shanten(concealed: &TileCounts) -> i8 {
    let orphans = Tile::all_kinds().filter(|t| t.is_terminal_or_honour());
    let mut kinds = 0;
    let mut pair = false;
    for t in orphans {
        if concealed[t] > 0 {
            kinds += 1;
        }
        if concealed[t] >= 2 {
            pair = true;
        }
    }
    13 - kinds - pair as i8
}

/// The smaller of standard and thirteen-orphans shanten.
pub fn shanten(concealed: &TileCounts, melds: usize) -> i8 {
    let s = standard_shanten(concealed, melds, Structure::Any);
    if melds == 0 {
        s.min(orphans_shanten(concealed))
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahjong_tiles::parse_tiles;

    fn c(s: &str) -> TileCounts {
        TileCounts::from_tiles(&parse_tiles(s).unwrap())
    }

    #[test]
    fn known_values() {
        assert_eq!(shanten(&c("123m 456m 789p EEE 55s"), 0), -1);
        assert_eq!(shanten(&c("123m 456m 789p EEE 5s"), 0), 0);
        assert_eq!(shanten(&c("123m 456m 789p EE 56s"), 0), 0);
        assert_eq!(shanten(&c("123m 456m 789p E 5s 7s N"), 0), 1);
        assert_eq!(shanten(&c("19m 19p 19s ESWN WhG 5m"), 0), 1);
        assert_eq!(shanten(&c("19m 19p 19s ESWN WhGR"), 0), 0);
        // With a meld on the table, 10 concealed tiles.
        assert_eq!(shanten(&c("456m 789p EE 56s"), 1), 0);
        // Scattered: no partials, so 8 by the standard form, 7 toward orphans.
        let scattered = c("147m 258p 369s ESWN");
        assert_eq!(standard_shanten(&scattered, 0, Structure::Any), 8);
        assert_eq!(shanten(&scattered, 0), 7);
    }

    #[test]
    fn structures() {
        let h = c("123m 456m 789p EEE 5s");
        assert_eq!(standard_shanten(&h, 0, Structure::AllPungs), 6);
        assert_eq!(
            standard_shanten(&c("222m 555p 88s EE 9s 1p R"), 0, Structure::AllPungs),
            2
        );
        assert_eq!(
            standard_shanten(&c("234m 567m 345p 678s 9s"), 0, Structure::AllChows),
            0
        );
        assert_eq!(
            standard_shanten(&c("234m 567m 345p RRR 9s"), 0, Structure::AllChows),
            1
        );
    }
}
