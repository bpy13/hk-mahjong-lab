//! Hand mathematics for Hong Kong mahjong.
//!
//! * [`shanten`]: distance to a ready hand, for any hand shape.
//! * [`target`]: the same distance toward hands that meet the faan minimum.
//! * [`discards`]: both measures and tile acceptance for every discard.

pub mod shanten;
pub mod target;

pub use shanten::{orphans_shanten, shanten, standard_shanten, Structure};
pub use target::{faan_shanten, targets, FaanShanten, HandView, SuitPlan, Target, UNREACHABLE};

use mahjong_hk::RuleSet;
use mahjong_tiles::{Tile, TileCounts};

/// Both distance measures and their tile acceptance for one discard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscardInfo {
    pub tile: Tile,
    /// Shanten after the discard (any complete hand).
    pub shanten: i8,
    /// Unseen tiles that would lower `shanten`.
    pub ukeire: u32,
    /// Faan-aware shanten after the discard.
    pub faan_shanten: i8,
    /// Unseen tiles that would lower `faan_shanten`.
    pub faan_ukeire: u32,
    /// The plan the faan-aware measure follows.
    pub target: Option<Target>,
}

/// Unseen tiles (with multiplicity) that would lower `metric` for a 13-tile hand.
pub fn acceptance(
    concealed: &TileCounts,
    unseen: &TileCounts,
    metric: impl Fn(&TileCounts) -> i8,
) -> u32 {
    let base = metric(concealed);
    let mut c = *concealed;
    let mut n = 0;
    for t in Tile::all_kinds() {
        if unseen[t] == 0 || c[t] >= 4 {
            continue;
        }
        c.add(t);
        if metric(&c) < base {
            n += unseen[t] as u32;
        }
        c.remove(t);
    }
    n
}

/// Evaluates every possible discard from a 14-tile hand (or 14 counting melds).
pub fn discards(view: &HandView, rules: &RuleSet) -> Vec<DiscardInfo> {
    let melds = view.melds.len();
    view.concealed
        .kinds()
        .map(|tile| {
            let mut hand = *view.concealed;
            hand.remove(tile);
            let sub = HandView {
                concealed: &hand,
                ..*view
            };
            let fs = faan_shanten(&sub, rules);
            let faan_metric = |h: &TileCounts| {
                faan_shanten(
                    &HandView {
                        concealed: h,
                        ..*view
                    },
                    rules,
                )
                .shanten
            };
            DiscardInfo {
                tile,
                shanten: shanten(&hand, melds),
                ukeire: acceptance(&hand, view.unseen, |h| shanten(h, melds)),
                faan_shanten: fs.shanten,
                faan_ukeire: if fs.shanten < UNREACHABLE {
                    acceptance(&hand, view.unseen, faan_metric)
                } else {
                    0
                },
                target: fs.target,
            }
        })
        .collect()
}
