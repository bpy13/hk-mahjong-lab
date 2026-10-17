//! Rule variants (`docs/rules/hk.md`). Every field's default is the spec's default.

use std::fmt;
use std::str::FromStr;

/// How faan convert into points (§8.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Payout {
    /// 半辣上: doubling up to 4 faan, then half-steps (3 → 8 … 10 → 128).
    #[default]
    HalfSpicy,
    /// 全辣: doubling at every faan (3 → 8 … 10 → 1024).
    FullSpicy,
}

/// What happens when several players win on the same discard (§4.1.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum MultipleWinners {
    /// All of them win; the discarder pays each.
    #[default]
    All,
    /// 截糊: only the first winner after the discarder in turn order wins.
    HeadBump,
}

/// A complete set of rule choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuleSet {
    /// Whether the 8 bonus tiles are in play (§3.5).
    pub flowers: bool,
    /// Minimum faan to declare a win (§6).
    pub min_faan: u8,
    /// Faan cap; limit hands score this (§7).
    pub faan_cap: u8,
    /// Faan → points (§8.1).
    pub payout: Payout,
    /// Several winners on one discard (§4.1.3).
    pub multiple_winners: MultipleWinners,
    /// The 12-tile liability rule (§8.3).
    pub twelve_tile_liability: bool,
    /// The prevailing wind as a seat index (0 = East … 3 = North, §2.3).
    pub prevailing_wind: usize,
}

impl Default for RuleSet {
    fn default() -> Self {
        RuleSet {
            flowers: true,
            min_faan: 3,
            faan_cap: 10,
            payout: Payout::HalfSpicy,
            multiple_winners: MultipleWinners::All,
            twelve_tile_liability: true,
            prevailing_wind: 0,
        }
    }
}

const HALF_SPICY: [u32; 14] = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384];

impl RuleSet {
    /// Points for one share of a win worth `faan` (already capped), §8.1.
    pub fn points(&self, faan: u8) -> u32 {
        let f = faan.min(13) as usize;
        match self.payout {
            Payout::HalfSpicy => HALF_SPICY[f],
            Payout::FullSpicy => 1 << f,
        }
    }

    /// The largest number of points one player can win or lose in a hand: three
    /// shares of a limit hand.
    pub fn max_abs_score(&self) -> f64 {
        3.0 * self.points(self.faan_cap) as f64
    }
}

/// Error from parsing a variant name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseVariantError(pub String);

impl fmt::Display for ParseVariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParseVariantError {}

impl FromStr for Payout {
    type Err = ParseVariantError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "half_spicy" => Ok(Payout::HalfSpicy),
            "full_spicy" => Ok(Payout::FullSpicy),
            _ => Err(ParseVariantError(format!(
                "unknown payout {s:?} (expected half_spicy or full_spicy)"
            ))),
        }
    }
}

impl FromStr for MultipleWinners {
    type Err = ParseVariantError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "all" => Ok(MultipleWinners::All),
            "head_bump" => Ok(MultipleWinners::HeadBump),
            _ => Err(ParseVariantError(format!(
                "unknown multiple_winners {s:?} (expected all or head_bump)"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_spicy_table_matches_spec_8_1() {
        let r = RuleSet::default();
        let got: Vec<u32> = (3..=10).map(|f| r.points(f)).collect();
        assert_eq!(got, [8, 16, 24, 32, 48, 64, 96, 128]);
    }

    #[test]
    fn full_spicy_doubles() {
        let r = RuleSet {
            payout: Payout::FullSpicy,
            ..RuleSet::default()
        };
        assert_eq!(r.points(3), 8);
        assert_eq!(r.points(10), 1024);
    }
}
