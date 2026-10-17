//! Complete hands (§6): the fast win check and the full list of ways to split a
//! hand into melds and a pair (which faan scoring needs, §6 last paragraph).

use mahjong_tiles::{Suit, Tile, TileCounts};
use std::sync::OnceLock;

/// A group in the concealed part of a hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// A chow starting at the tile.
    Chow(Tile),
    /// A pung of the tile.
    Pung(Tile),
}

/// One way to split the concealed tiles of a complete standard hand.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Decomposition {
    /// The pair 眼.
    pub pair: Tile,
    /// The concealed melds (sorted).
    pub groups: Vec<Group>,
}

const SUIT_KEYS: usize = 1_953_125; // 5^9
const MELDS_ONLY: u8 = 1;
const MELDS_AND_PAIR: u8 = 2;

/// For each per-suit count vector (as a base-5 key, see
/// [`TileCounts::suit_key`]): can it be split into melds only, and/or into melds
/// plus one pair? Built once, by enumerating meld combinations (16 kinds of suit
/// meld, at most 4 of them, plus an optional pair).
fn suit_table() -> &'static [u8] {
    static TABLE: OnceLock<Vec<u8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = vec![0u8; SUIT_KEYS];
        let mut counts = [0u8; 9];
        fill(&mut table, &mut counts, 0, 0);
        table
    })
}

/// Recursively adds melds with index ≥ `from` (0–6 chows, 7–15 pungs).
fn fill(table: &mut [u8], counts: &mut [u8; 9], from: usize, melds: usize) {
    table[key(counts)] |= MELDS_ONLY;
    for p in 0..9 {
        if counts[p] + 2 <= 4 {
            counts[p] += 2;
            table[key(counts)] |= MELDS_AND_PAIR;
            counts[p] -= 2;
        }
    }
    if melds == 4 {
        return;
    }
    for m in from..16 {
        let tiles: &[usize] = if m < 7 {
            &[m, m + 1, m + 2]
        } else {
            &[m - 7; 3]
        };
        for &t in tiles {
            counts[t] += 1;
        }
        if tiles.iter().all(|&t| counts[t] <= 4) {
            fill(table, counts, m, melds + 1);
        }
        for &t in tiles {
            counts[t] -= 1;
        }
    }
}

fn key(counts: &[u8; 9]) -> usize {
    counts.iter().rev().fold(0, |k, &c| k * 5 + c as usize)
}

/// Is `concealed` a complete standard hand: melds and exactly one pair, using
/// every tile? (Thirteen orphans is checked by [`is_thirteen_orphans`].)
pub fn is_standard_complete(concealed: &TileCounts) -> bool {
    if concealed.total() % 3 != 2 {
        return false;
    }
    let table = suit_table();
    let mut pairs = 0;
    for suit in Suit::ALL {
        let n: u8 = concealed.suit_counts(suit).iter().sum();
        let flags = table[concealed.suit_key(suit) as usize];
        match n % 3 {
            0 if flags & MELDS_ONLY != 0 => {}
            2 if flags & MELDS_AND_PAIR != 0 => pairs += 1,
            _ => return false,
        }
    }
    for t in 27..34 {
        match concealed.0[t] {
            0 | 3 => {}
            2 => pairs += 1,
            _ => return false,
        }
    }
    pairs == 1
}

/// Is `concealed` thirteen orphans (§6.2)? Only possible with no melds.
pub fn is_thirteen_orphans(concealed: &TileCounts) -> bool {
    concealed.total() == 14
        && Tile::all_kinds().all(|t| {
            let c = concealed[t];
            if t.is_terminal_or_honour() {
                (1..=2).contains(&c)
            } else {
                c == 0
            }
        })
}

/// Is `concealed` (with `melds` melds already on the table) a complete hand of
/// either form?
pub fn is_complete(concealed: &TileCounts, melds: usize) -> bool {
    if concealed.total() + 3 * melds != 14 {
        return false;
    }
    is_standard_complete(concealed) || (melds == 0 && is_thirteen_orphans(concealed))
}

/// The tiles that would complete a hand of 13 (`concealed` plus `melds` melds),
/// ignoring how many copies are still available.
pub fn waits(concealed: &TileCounts, melds: usize) -> Vec<Tile> {
    let mut out = Vec::new();
    let mut c = *concealed;
    for t in Tile::all_kinds() {
        if c[t] >= 4 {
            continue;
        }
        c.add(t);
        if is_complete(&c, melds) {
            out.push(t);
        }
        c.remove(t);
    }
    out
}

/// Every distinct way to split a complete standard concealed part into melds and
/// one pair. Empty if it is not a complete standard hand.
pub fn decompositions(concealed: &TileCounts) -> Vec<Decomposition> {
    let mut out = Vec::new();
    if !is_standard_complete(concealed) {
        return out;
    }
    let mut c = *concealed;
    let mut groups = Vec::new();
    split(&mut c, None, &mut groups, &mut out);
    out.sort_by(|a, b| (a.pair, &a.groups).cmp(&(b.pair, &b.groups)));
    out.dedup();
    out
}

fn split(
    c: &mut TileCounts,
    pair: Option<Tile>,
    groups: &mut Vec<Group>,
    out: &mut Vec<Decomposition>,
) {
    let Some(t) = c.kinds().next() else {
        if let Some(pair) = pair {
            let mut g = groups.clone();
            g.sort();
            out.push(Decomposition { pair, groups: g });
        }
        return;
    };
    if pair.is_none() && c[t] >= 2 {
        c[t] -= 2;
        split(c, Some(t), groups, out);
        c[t] += 2;
    }
    if c[t] >= 3 {
        c[t] -= 3;
        groups.push(Group::Pung(t));
        split(c, pair, groups, out);
        groups.pop();
        c[t] += 3;
    }
    if let (Some(t1), Some(t2)) = (t.offset(1), t.offset(2)) {
        if c[t1] > 0 && c[t2] > 0 {
            c.remove(t);
            c.remove(t1);
            c.remove(t2);
            groups.push(Group::Chow(t));
            split(c, pair, groups, out);
            groups.pop();
            c.add(t);
            c.add(t1);
            c.add(t2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahjong_tiles::parse_tiles;

    fn counts(s: &str) -> TileCounts {
        TileCounts::from_tiles(&parse_tiles(s).unwrap())
    }

    #[test]
    fn standard_hands_from_spec_6_1() {
        assert!(is_complete(&counts("123m 456m 789p EEE 55s"), 0));
        assert!(is_complete(&counts("123p 456p 789p RRR NN"), 0));
        assert!(!is_complete(&counts("123m 456m 789p EEE 56s"), 0));
        // A meld on the table leaves 11 concealed tiles.
        assert!(is_complete(&counts("456m 789p EEE 55s"), 1));
        assert!(!is_complete(&counts("456m 789p EEE 55s"), 0));
    }

    #[test]
    fn thirteen_orphans_spec_6_2() {
        let h = counts("19m 19p 19s ESWN WhGRR");
        assert!(is_thirteen_orphans(&h) && is_complete(&h, 0));
        assert!(!is_thirteen_orphans(&counts("19m 19p 19s ESWN WhGR 5m")));
        assert_eq!(waits(&counts("19m 19p 19s ESWN WhGR"), 0).len(), 13);
    }

    #[test]
    fn four_identical_honours_cannot_be_split() {
        assert!(!is_complete(&counts("EEEE 123m 456m 789p 1s"), 0));
        assert!(is_complete(&counts("EEE 123m 456m 789p 11s"), 0));
    }

    #[test]
    fn ambiguous_hand_has_several_splits() {
        // 111222333m can be three pungs or three identical chows.
        let d = decompositions(&counts("111222333m 456p 99s"));
        assert_eq!(d.len(), 2);
        assert_eq!(decompositions(&counts("11123455678999m")).len(), 1);
    }

    #[test]
    fn nine_gates_waits_on_every_tile_of_the_suit() {
        let w = waits(&counts("1112345678999m"), 0);
        assert_eq!(w.len(), 9);
    }

    #[test]
    fn table_agrees_with_brute_force_split() {
        // Every complete hand from the recursive splitter is accepted by the table
        // check and vice versa, over many random suit-heavy hands.
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(7);
        // A small pool (one suit, three dots, three winds) makes complete hands common.
        let pool: Vec<u8> = (0..9).chain(9..12).chain(27..30).collect();
        for _ in 0..20_000 {
            let mut c = TileCounts::new();
            while c.total() < 14 {
                let t = Tile::from_index(pool[rng.gen_range(0..pool.len())]);
                if c[t] < 4 {
                    c.add(t);
                }
            }
            let mut g = Vec::new();
            let mut out = Vec::new();
            let mut cc = c;
            split(&mut cc, None, &mut g, &mut out);
            assert_eq!(is_standard_complete(&c), !out.is_empty(), "{c}");
        }
    }
}
