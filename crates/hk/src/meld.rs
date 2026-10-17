//! Melds: chows, pungs and kongs (§1).

use mahjong_tiles::{Tile, TileCounts};
use std::fmt;

/// The shape of a meld.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MeldKind {
    /// Three consecutive tiles of one suit; `tile` is the lowest.
    Chow,
    /// Three identical tiles.
    Pung,
    /// Four identical tiles.
    Kong,
}

/// A meld laid down on the table: exposed (made with a claimed discard, or an
/// added kong) or a concealed kong.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Meld {
    pub kind: MeldKind,
    /// The meld's tile (the lowest tile for a chow).
    pub tile: Tile,
    /// `true` only for a concealed kong (§5.1).
    pub concealed: bool,
    /// The seat whose discard was claimed, for exposed melds.
    pub claimed_from: Option<u8>,
}

impl Meld {
    /// A filler value for fixed-size arrays; never part of a hand.
    pub(crate) const PLACEHOLDER: Meld = Meld {
        kind: MeldKind::Kong,
        tile: Tile::EAST,
        concealed: true,
        claimed_from: None,
    };

    /// An exposed meld made from `from`'s discard.
    pub fn exposed(kind: MeldKind, tile: Tile, from: usize) -> Meld {
        Meld {
            kind,
            tile,
            concealed: false,
            claimed_from: Some(from as u8),
        }
    }

    /// A concealed kong.
    pub fn concealed_kong(tile: Tile) -> Meld {
        Meld {
            kind: MeldKind::Kong,
            tile,
            concealed: true,
            claimed_from: None,
        }
    }

    /// A pung or kong (counts as a pung for patterns, §1).
    pub fn is_pung_like(&self) -> bool {
        matches!(self.kind, MeldKind::Pung | MeldKind::Kong)
    }

    /// The tiles in the meld (4 for a kong).
    pub fn tiles(&self) -> Vec<Tile> {
        match self.kind {
            MeldKind::Chow => (0..3)
                .map(|d| self.tile.offset(d).expect("valid chow"))
                .collect(),
            MeldKind::Pung => vec![self.tile; 3],
            MeldKind::Kong => vec![self.tile; 4],
        }
    }

    /// Adds the meld's tiles to `counts`.
    pub fn add_to(&self, counts: &mut TileCounts) {
        for t in self.tiles() {
            counts.add(t);
        }
    }
}

impl fmt::Display for Meld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = mahjong_tiles::format_tiles(&self.tiles());
        if self.concealed {
            write!(f, "[{text}]")
        } else {
            f.write_str(&text)
        }
    }
}

impl fmt::Debug for Meld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
