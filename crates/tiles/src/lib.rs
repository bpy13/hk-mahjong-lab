//! Mahjong tile vocabulary (rules spec: `docs/rules/hk.md` §1).
//!
//! A [`Tile`] is one of 42 kinds: the 34 playing kinds (four copies of each in a
//! set) and the 8 bonus tiles (one copy each). Its index is
//!
//! | Tiles | Text | Index |
//! |---|---|---|
//! | characters 萬 1–9 | `1m` … `9m` | 0–8 |
//! | dots 筒 1–9 | `1p` … `9p` | 9–17 |
//! | bamboo 索 1–9 | `1s` … `9s` | 18–26 |
//! | winds East South West North | `E S W N` | 27–30 |
//! | dragons White Green Red | `Wh G R` | 31–33 |
//! | flowers 梅 蘭 菊 竹 | `F1` … `F4` | 34–37 |
//! | seasons 春 夏 秋 冬 | `S1` … `S4` | 38–41 |
//!
//! A hand is held as [`TileCounts`]: how many of each playing kind it contains.
//! Bonus tiles are never part of a hand, so they are kept elsewhere.

use std::fmt;
use std::ops::{Index, IndexMut};
use std::str::FromStr;

/// Number of playing tile kinds (suited tiles and honours).
pub const NUM_KINDS: usize = 34;
/// Number of tile kinds including the 8 bonus tiles.
pub const NUM_TILES_WITH_BONUS: usize = 42;
/// Number of physical tiles in a full set, bonus tiles included.
pub const SET_SIZE: usize = 144;

/// A suit of numbered tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Suit {
    /// Characters 萬, text `m`.
    Characters = 0,
    /// Dots 筒, text `p`.
    Dots = 1,
    /// Bamboo 索, text `s`.
    Bamboo = 2,
}

impl Suit {
    /// All suits, in index order.
    pub const ALL: [Suit; 3] = [Suit::Characters, Suit::Dots, Suit::Bamboo];

    /// The suit letter: `m`, `p` or `s`.
    pub fn to_char(self) -> char {
        b"mps"[self as usize] as char
    }

    /// Parses a suit letter (lower case only, so `S` stays South).
    pub fn from_char(c: char) -> Option<Suit> {
        Some(Suit::ALL["mps".find(c)?])
    }

    /// The tile of this suit with number `n` (1–9).
    ///
    /// # Panics
    /// Panics if `n` is not in `1..=9`.
    pub fn tile(self, n: u8) -> Tile {
        assert!((1..=9).contains(&n), "suit tile number {n} out of range");
        Tile(self as u8 * 9 + n - 1)
    }
}

/// One tile kind. See the [crate docs](crate) for the index layout.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tile(u8);

const HONOUR_TEXT: [&str; 7] = ["E", "S", "W", "N", "Wh", "G", "R"];

impl Tile {
    pub const EAST: Tile = Tile(27);
    pub const SOUTH: Tile = Tile(28);
    pub const WEST: Tile = Tile(29);
    pub const NORTH: Tile = Tile(30);
    pub const WHITE: Tile = Tile(31);
    pub const GREEN: Tile = Tile(32);
    pub const RED: Tile = Tile(33);

    /// The tile with index `i`.
    ///
    /// # Panics
    /// Panics if `i >= 42`.
    pub fn from_index(i: u8) -> Tile {
        assert!(
            (i as usize) < NUM_TILES_WITH_BONUS,
            "tile index {i} out of range"
        );
        Tile(i)
    }

    /// The tile index, `0..42`.
    pub fn index(self) -> u8 {
        self.0
    }

    /// The wind for seat `seat` (0 = East … 3 = North).
    pub fn wind(seat: usize) -> Tile {
        assert!(seat < 4, "seat {seat} out of range");
        Tile(27 + seat as u8)
    }

    /// Flower `n` (1–4: 梅 蘭 菊 竹).
    pub fn flower(n: u8) -> Tile {
        assert!((1..=4).contains(&n), "flower number {n} out of range");
        Tile(33 + n)
    }

    /// Season `n` (1–4: 春 夏 秋 冬).
    pub fn season(n: u8) -> Tile {
        assert!((1..=4).contains(&n), "season number {n} out of range");
        Tile(37 + n)
    }

    /// The 34 playing kinds, in index order.
    pub fn all_kinds() -> impl Iterator<Item = Tile> {
        (0..NUM_KINDS as u8).map(Tile)
    }

    /// The 8 bonus tiles: flowers 1–4, then seasons 1–4.
    pub fn all_bonus() -> impl Iterator<Item = Tile> {
        (NUM_KINDS as u8..NUM_TILES_WITH_BONUS as u8).map(Tile)
    }

    /// The suit, for a suited tile.
    pub fn suit(self) -> Option<Suit> {
        (self.0 < 27).then(|| Suit::ALL[(self.0 / 9) as usize])
    }

    /// The number (1–9) of a suited tile, or the bonus number (1–4) of a flower or
    /// season. `None` for honours.
    pub fn number(self) -> Option<u8> {
        match self.0 {
            0..=26 => Some(self.0 % 9 + 1),
            27..=33 => None,
            _ => Some((self.0 - 34) % 4 + 1),
        }
    }

    /// Characters, dots or bamboo.
    pub fn is_suited(self) -> bool {
        self.0 < 27
    }

    /// A wind or a dragon.
    pub fn is_honour(self) -> bool {
        (27..34).contains(&self.0)
    }

    /// East, South, West or North.
    pub fn is_wind(self) -> bool {
        (27..31).contains(&self.0)
    }

    /// White, Green or Red dragon.
    pub fn is_dragon(self) -> bool {
        (31..34).contains(&self.0)
    }

    /// A 1 or 9 of a suit.
    pub fn is_terminal(self) -> bool {
        matches!(self.number(), Some(1 | 9)) && self.is_suited()
    }

    /// A terminal or an honour (the kinds in thirteen orphans).
    pub fn is_terminal_or_honour(self) -> bool {
        self.is_terminal() || self.is_honour()
    }

    /// A flower or a season.
    pub fn is_bonus(self) -> bool {
        self.0 >= NUM_KINDS as u8
    }

    /// A flower (as opposed to a season). Only meaningful for bonus tiles.
    pub fn is_flower(self) -> bool {
        (34..38).contains(&self.0)
    }

    /// How many physical copies a set has: 4 for playing kinds, 1 for bonus tiles.
    pub fn copies(self) -> u8 {
        if self.is_bonus() {
            1
        } else {
            4
        }
    }

    /// The suited tile `delta` steps along the same suit, if it exists
    /// (`3m.offset(2) == 5m`, `8m.offset(2) == None`).
    pub fn offset(self, delta: i8) -> Option<Tile> {
        let n = self.number()? as i8 + delta;
        (self.is_suited() && (1..=9).contains(&n)).then(|| Tile((self.0 as i8 + delta) as u8))
    }

    /// The Unicode Mahjong Tiles glyph (`🀇` for 1m, `🀄` for Red).
    pub fn to_unicode(self) -> char {
        let cp = match self.0 {
            0..=8 => 0x1F007 + self.0 as u32,
            9..=17 => 0x1F019 + (self.0 - 9) as u32,
            18..=26 => 0x1F010 + (self.0 - 18) as u32,
            27..=30 => 0x1F000 + (self.0 - 27) as u32,
            31 => 0x1F006,
            32 => 0x1F005,
            33 => 0x1F004,
            // Unicode orders the flowers plum, orchid, bamboo, chrysanthemum;
            // Hong Kong numbers chrysanthemum 3 and bamboo 4.
            34 => 0x1F022,
            35 => 0x1F023,
            36 => 0x1F025,
            37 => 0x1F024,
            _ => 0x1F026 + (self.0 - 38) as u32,
        };
        char::from_u32(cp).expect("valid code point")
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            0..=26 => write!(
                f,
                "{}{}",
                self.0 % 9 + 1,
                Suit::ALL[(self.0 / 9) as usize].to_char()
            ),
            27..=33 => f.write_str(HONOUR_TEXT[(self.0 - 27) as usize]),
            34..=37 => write!(f, "F{}", self.0 - 33),
            _ => write!(f, "S{}", self.0 - 37),
        }
    }
}

impl fmt::Debug for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Error from parsing tile text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseTilesError(pub String);

impl fmt::Display for ParseTilesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseTilesError {}

impl FromStr for Tile {
    type Err = ParseTilesError;

    /// Parses exactly one tile, e.g. `"5p"`, `"Wh"`, `"F2"`.
    fn from_str(s: &str) -> Result<Tile, ParseTilesError> {
        match parse_tiles(s)?.as_slice() {
            [t] => Ok(*t),
            _ => Err(ParseTilesError(format!("expected one tile, got {s:?}"))),
        }
    }
}

/// Parses a list of tiles.
///
/// Suited tiles may be grouped: digits followed by a suit letter, so `"123m"` is
/// `1m 2m 3m`. Honours are written by letter, without separators if desired:
/// `"EEE"`, `"WhWh"`, `"RRGG"`. Bonus tiles are `F1`–`F4` and `S1`–`S4`; an `F` or
/// `S` directly followed by a digit is always read as a bonus tile, so put a space
/// between South and a group of digits (`"S 123m"`). Whitespace is ignored.
///
/// ```
/// use mahjong_tiles::parse_tiles;
/// let t = parse_tiles("123m 55p EEE Wh F2").unwrap();
/// assert_eq!(t.len(), 10);
/// ```
pub fn parse_tiles(s: &str) -> Result<Vec<Tile>, ParseTilesError> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut digits: Vec<u8> = Vec::new();
    let mut i = 0;
    let err = |msg: String| Err(ParseTilesError(format!("{msg} in {s:?}")));
    while i < chars.len() {
        let c = chars[i];
        if let Some(d) = c.to_digit(10) {
            if d == 0 {
                return err("tile number 0".into());
            }
            digits.push(d as u8);
            i += 1;
            continue;
        }
        if let Some(suit) = Suit::from_char(c) {
            if digits.is_empty() {
                return err(format!("suit letter {c:?} with no numbers"));
            }
            out.extend(digits.drain(..).map(|n| suit.tile(n)));
            i += 1;
            continue;
        }
        if !digits.is_empty() {
            return err(format!("numbers {digits:?} with no suit letter"));
        }
        let next = chars.get(i + 1).copied();
        let bonus_number = next
            .and_then(|n| n.to_digit(10))
            .filter(|n| (1..=4).contains(n));
        match (c, bonus_number) {
            (_, _) if c.is_whitespace() => {}
            ('F', Some(n)) => {
                out.push(Tile::flower(n as u8));
                i += 1;
            }
            ('S', Some(n)) => {
                out.push(Tile::season(n as u8));
                i += 1;
            }
            ('W', _) if next == Some('h') => {
                out.push(Tile::WHITE);
                i += 1;
            }
            ('E', _) => out.push(Tile::EAST),
            ('S', _) => out.push(Tile::SOUTH),
            ('W', _) => out.push(Tile::WEST),
            ('N', _) => out.push(Tile::NORTH),
            ('G', _) => out.push(Tile::GREEN),
            ('R', _) => out.push(Tile::RED),
            _ => return err(format!("unexpected character {c:?}")),
        }
        i += 1;
    }
    if !digits.is_empty() {
        return err(format!("numbers {digits:?} with no suit letter"));
    }
    Ok(out)
}

/// Formats tiles in sorted, grouped notation: one group per suit (`"1235m"`), one
/// group per honour kind (`"EEE"`), then bonus tiles, separated by spaces.
/// [`parse_tiles`] reads the result back.
pub fn format_tiles(tiles: &[Tile]) -> String {
    let mut sorted = tiles.to_vec();
    sorted.sort();
    let mut groups: Vec<String> = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let t = sorted[i];
        if let Some(suit) = t.suit() {
            let mut g = String::new();
            while i < sorted.len() && sorted[i].suit() == Some(suit) {
                g.push((b'0' + sorted[i].number().unwrap()) as char);
                i += 1;
            }
            g.push(suit.to_char());
            groups.push(g);
        } else if t.is_honour() {
            let mut g = String::new();
            while i < sorted.len() && sorted[i] == t {
                g.push_str(&t.to_string());
                i += 1;
            }
            groups.push(g);
        } else {
            groups.push(t.to_string());
            i += 1;
        }
    }
    groups.join(" ")
}

/// How many of each of the 34 playing kinds a hand (or any multiset of playing
/// tiles) contains.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCounts(pub [u8; NUM_KINDS]);

impl Default for TileCounts {
    fn default() -> Self {
        Self([0; NUM_KINDS])
    }
}

impl TileCounts {
    /// No tiles.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every playing tile of a full set: four of each kind.
    pub fn full_set() -> Self {
        Self([4; NUM_KINDS])
    }

    /// Counts the given tiles.
    ///
    /// # Panics
    /// Panics on a bonus tile: bonus tiles are never part of a hand.
    pub fn from_tiles(tiles: &[Tile]) -> Self {
        let mut c = Self::new();
        for &t in tiles {
            c.add(t);
        }
        c
    }

    /// Adds one tile.
    ///
    /// # Panics
    /// Panics on a bonus tile.
    pub fn add(&mut self, t: Tile) {
        assert!(!t.is_bonus(), "bonus tile {t} cannot be counted in a hand");
        self.0[t.index() as usize] += 1;
    }

    /// Removes one tile.
    ///
    /// # Panics
    /// Panics if the tile is not there.
    pub fn remove(&mut self, t: Tile) {
        let c = &mut self.0[t.index() as usize];
        assert!(*c > 0, "no {t} to remove");
        *c -= 1;
    }

    /// Total number of tiles.
    pub fn total(&self) -> usize {
        self.0.iter().map(|&c| c as usize).sum()
    }

    /// The tiles, in index order, each repeated by its count.
    pub fn tiles(&self) -> Vec<Tile> {
        Tile::all_kinds()
            .flat_map(|t| std::iter::repeat_n(t, self[t] as usize))
            .collect()
    }

    /// The distinct kinds present, in index order.
    pub fn kinds(&self) -> impl Iterator<Item = Tile> + '_ {
        Tile::all_kinds().filter(|&t| self[t] > 0)
    }

    /// The counts of one suit (`[count of 1, …, count of 9]`).
    pub fn suit_counts(&self, suit: Suit) -> [u8; 9] {
        let start = suit as usize * 9;
        self.0[start..start + 9]
            .try_into()
            .expect("nine tiles per suit")
    }

    /// The counts of one suit packed into a base-5 number (each count is 0–4):
    /// `count(1) + 5·count(2) + … + 5⁸·count(9)`, so `< 5⁹ = 1,953,125`.
    /// Precomputed per-suit tables are indexed by this key.
    pub fn suit_key(&self, suit: Suit) -> u32 {
        self.suit_counts(suit)
            .iter()
            .rev()
            .fold(0, |k, &c| k * 5 + c as u32)
    }
}

impl Index<Tile> for TileCounts {
    type Output = u8;
    fn index(&self, t: Tile) -> &u8 {
        &self.0[t.index() as usize]
    }
}

impl IndexMut<Tile> for TileCounts {
    fn index_mut(&mut self, t: Tile) -> &mut u8 {
        &mut self.0[t.index() as usize]
    }
}

impl fmt::Display for TileCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_tiles(&self.tiles()))
    }
}

impl fmt::Debug for TileCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TileCounts({self})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> Tile {
        s.parse().unwrap()
    }

    #[test]
    fn index_layout_matches_spec_section_1() {
        assert_eq!(t("1m").index(), 0);
        assert_eq!(t("9m").index(), 8);
        assert_eq!(t("1p").index(), 9);
        assert_eq!(t("1s").index(), 18);
        assert_eq!(t("9s").index(), 26);
        let honours = ["E", "S", "W", "N", "Wh", "G", "R"];
        for (i, h) in honours.iter().enumerate() {
            assert_eq!(t(h).index(), 27 + i as u8);
        }
        assert_eq!(t("F1").index(), 34);
        assert_eq!(t("F4").index(), 37);
        assert_eq!(t("S1").index(), 38);
        assert_eq!(t("S4").index(), 41);
    }

    #[test]
    fn every_tile_round_trips_through_text() {
        for i in 0..NUM_TILES_WITH_BONUS as u8 {
            let tile = Tile::from_index(i);
            assert_eq!(t(&tile.to_string()), tile);
        }
    }

    #[test]
    fn a_set_has_144_tiles() {
        let n: usize = (0..NUM_TILES_WITH_BONUS as u8)
            .map(|i| Tile::from_index(i).copies() as usize)
            .sum();
        assert_eq!(n, SET_SIZE);
        assert_eq!(TileCounts::full_set().total(), 136);
    }

    #[test]
    fn classification() {
        assert!(t("1m").is_terminal() && t("9s").is_terminal());
        assert!(!t("5p").is_terminal() && !t("E").is_terminal());
        assert!(t("E").is_wind() && t("N").is_wind() && !t("Wh").is_wind());
        assert!(t("Wh").is_dragon() && t("R").is_dragon());
        assert!(t("G").is_honour() && !t("9m").is_honour());
        assert!(t("F3").is_bonus() && t("F3").is_flower() && !t("S3").is_flower());
        assert_eq!(t("7p").suit(), Some(Suit::Dots));
        assert_eq!(t("7p").number(), Some(7));
        assert_eq!(t("S2").number(), Some(2));
        assert_eq!(t("R").number(), None);
        let orphans = Tile::all_kinds()
            .filter(|k| k.is_terminal_or_honour())
            .count();
        assert_eq!(orphans, 13);
    }

    #[test]
    fn offset_stays_in_suit() {
        assert_eq!(t("3m").offset(2), Some(t("5m")));
        assert_eq!(t("8m").offset(2), None);
        assert_eq!(t("1p").offset(-1), None);
        assert_eq!(t("E").offset(1), None);
    }

    #[test]
    fn parses_grouped_notation() {
        let tiles = parse_tiles("123m 456p 789s EEE WhWh").unwrap();
        assert_eq!(format_tiles(&tiles), "123m 456p 789s EEE WhWh");
        assert_eq!(parse_tiles("WhW").unwrap(), vec![Tile::WHITE, Tile::WEST]);
        assert_eq!(parse_tiles("S 1m").unwrap(), vec![Tile::SOUTH, t("1m")]);
        assert_eq!(parse_tiles("S1").unwrap(), vec![Tile::season(1)]);
        assert!(parse_tiles("S5").unwrap_err().0.contains("no suit"));
    }

    #[test]
    fn rejects_bad_text() {
        for bad in ["12", "m", "0m", "X", "1z"] {
            assert!(parse_tiles(bad).is_err(), "{bad:?} should not parse");
        }
        assert!("1m2m".parse::<Tile>().is_err());
    }

    #[test]
    fn format_sorts_and_groups() {
        let tiles = parse_tiles("R 9m F2 1m E 5m R").unwrap();
        assert_eq!(format_tiles(&tiles), "159m E RR F2");
    }

    #[test]
    fn unicode_glyphs() {
        assert_eq!(t("1m").to_unicode(), '🀇');
        assert_eq!(t("9p").to_unicode(), '🀡');
        assert_eq!(t("1s").to_unicode(), '🀐');
        assert_eq!(t("E").to_unicode(), '🀀');
        assert_eq!(t("R").to_unicode(), '🀄');
        assert_eq!(t("Wh").to_unicode(), '🀆');
        assert_eq!(t("F3").to_unicode(), '🀥'); // chrysanthemum 菊
        assert_eq!(t("S4").to_unicode(), '🀩'); // winter 冬
    }

    #[test]
    fn counts_and_suit_key() {
        let c = TileCounts::from_tiles(&parse_tiles("1129m 5p").unwrap());
        assert_eq!(c.total(), 5);
        assert_eq!(c[t("1m")], 2);
        assert_eq!(c.suit_counts(Suit::Characters), [2, 1, 0, 0, 0, 0, 0, 0, 1]);
        assert_eq!(c.suit_key(Suit::Characters), 2 + 5 + 5u32.pow(8));
        assert_eq!(c.suit_key(Suit::Bamboo), 0);
        assert_eq!(TileCounts::full_set().suit_key(Suit::Dots), 5u32.pow(9) - 1);
        assert_eq!(c.to_string(), "1129m 5p");
    }

    #[test]
    #[should_panic(expected = "bonus tile")]
    fn bonus_tiles_are_not_counted() {
        TileCounts::from_tiles(&[Tile::flower(1)]);
    }
}
