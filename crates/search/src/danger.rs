//! A deal-in danger model from public information, fitted to simulated games.
//!
//! For each opponent it estimates a **threat**: the chance they are already
//! waiting on a tile that would give them a legal win (complete *and* at least the
//! minimum faan). Then, for each tile, how likely a waiting opponent is to win on
//! it. Both are looked up in tables fitted by the `fit_danger` example (see
//! [`crate::danger_tables`]) from games between the rule-based bots:
//!
//! * the threat by the number of exposed melds, how far the wall has gone, and
//!   whether the exposed melds already carry faan or all lie in one suit (a flush);
//! * the per-tile rate by the kind of tile (honour, terminal, 2/8, middle), how
//!   many copies are unseen, whether the opponent has discarded it themselves
//!   (Hong Kong rules let them win on it anyway), and, against a flush, whether
//!   the tile is in the flush suit (or an honour) or another suit.
//!
//! A tile's danger is the chance that at least one opponent wins on it:
//! 1 − Π(1 − threat × rate), then mapped through a calibration curve fitted on the
//! tiles players really discarded (they are not a random sample of the unseen
//! tiles). The `danger_check` example measures the calibration.

use crate::danger_tables::{CALIBRATION, THREAT, WAIT};
use mahjong_hk::{Event, State};
use mahjong_tiles::{Tile, TileCounts, NUM_KINDS};

/// Number of wall-progress buckets in the threat table.
pub const PROGRESS_BUCKETS: usize = 6;
/// Size of the threat table: exposed melds (0–4) × progress × meld tag.
pub const THREAT_CELLS: usize = 5 * PROGRESS_BUCKETS * 3;
/// Size of the per-tile table: tile class × unseen copies (1–4) × discarded by
/// the opponent × flush relation.
pub const WAIT_CELLS: usize = 4 * 4 * 2 * 3;

/// What a seat's exposed melds suggest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Tag {
    exposed: usize,
    /// 0 = nothing special, 1 = a meld already carries faan, 2 = one suit only.
    kind: usize,
    flush: Option<mahjong_tiles::Suit>,
}

fn tag(state: &State, o: usize) -> Tag {
    let rules = state.rules();
    let melds = state.melds(o);
    let exposed = melds.exposed_count().min(4);
    let mut suits: Vec<_> = melds.iter().filter_map(|m| m.tile.suit()).collect();
    suits.sort();
    suits.dedup();
    let flush = (exposed >= 1 && suits.len() == 1).then(|| suits[0]);
    let has_value = melds.iter().any(|m| {
        m.is_pung_like()
            && (m.tile.is_dragon()
                || m.tile == Tile::wind(o)
                || m.tile == Tile::wind(rules.prevailing_wind))
    });
    let kind = if flush.is_some() {
        2
    } else if has_value {
        1
    } else {
        0
    };
    Tag {
        exposed,
        kind,
        flush,
    }
}

/// What the observer can see about one opponent, for the table lookups.
#[derive(Clone, Debug)]
pub struct Opponent {
    tag: Tag,
    progress: usize,
    discarded: [bool; NUM_KINDS],
}

impl Opponent {
    /// Public information about opponent `o`.
    pub fn of(state: &State, o: usize) -> Opponent {
        let initial_wall = if state.rules().flowers { 91.0 } else { 83.0 };
        let progress = (1.0 - state.wall_remaining() as f64 / initial_wall).clamp(0.0, 0.999);
        let mut discarded = [false; NUM_KINDS];
        for e in state.events() {
            if let Event::Discard { seat, tile } = *e {
                if seat as usize == o {
                    discarded[tile.index() as usize] = true;
                }
            }
        }
        Opponent {
            tag: tag(state, o),
            progress: (progress * PROGRESS_BUCKETS as f64) as usize,
            discarded,
        }
    }

    /// The threat-table cell.
    pub fn threat_cell(&self) -> usize {
        (self.tag.exposed * PROGRESS_BUCKETS + self.progress) * 3 + self.tag.kind
    }

    /// The per-tile cell of tile `t`, given the copies of each kind unseen by the
    /// observer; `None` when no copy is unseen (the opponent cannot wait on it).
    pub fn wait_cell(&self, t: Tile, unseen: &TileCounts) -> Option<usize> {
        let left = unseen[t].min(4) as usize;
        if left == 0 {
            return None;
        }
        let class = if t.is_honour() {
            0
        } else {
            match t.number() {
                Some(1 | 9) => 1,
                Some(2 | 8) => 2,
                _ => 3,
            }
        };
        let discarded = self.discarded[t.index() as usize] as usize;
        let relation = match self.tag.flush {
            None => 0,
            Some(s) if t.suit().is_none_or(|x| x == s) => 1,
            Some(_) => 2,
        };
        Some(((class * 4 + (left - 1)) * 2 + discarded) * 3 + relation)
    }
}

/// Maps a raw estimate to the deal-in rate observed for it on real discards
/// (piecewise linear through [`CALIBRATION`]; proportional below the first point,
/// flat above the last).
fn calibrate(raw: f64) -> f64 {
    let first = CALIBRATION[0];
    if raw <= first.0 {
        return if first.0 > 0.0 {
            raw * first.1 / first.0
        } else {
            first.1
        };
    }
    for w in CALIBRATION.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if raw <= x1 {
            return y0 + (y1 - y0) * (raw - x0) / (x1 - x0);
        }
    }
    CALIBRATION[CALIBRATION.len() - 1].1
}

/// Estimated danger of each tile for one seat.
#[derive(Clone, Debug)]
pub struct Danger {
    tiles: [f64; NUM_KINDS],
    threat: [f64; 4],
}

impl Danger {
    /// Estimates danger for `me` from public information.
    pub fn estimate(state: &State, me: usize) -> Danger {
        let unseen = state.unseen(me);
        let mut safe = [1.0f64; NUM_KINDS];
        let mut threat = [0.0f64; 4];
        for o in (0..4).filter(|&o| o != me) {
            let opp = Opponent::of(state, o);
            let t_o = THREAT[opp.threat_cell()];
            threat[o] = t_o;
            for t in Tile::all_kinds() {
                if let Some(c) = opp.wait_cell(t, &unseen) {
                    safe[t.index() as usize] *= 1.0 - t_o * WAIT[c];
                }
            }
        }
        let mut tiles = [0.0f64; NUM_KINDS];
        for i in 0..NUM_KINDS {
            tiles[i] = calibrate(1.0 - safe[i]);
        }
        Danger { tiles, threat }
    }

    /// Estimated probability that discarding `t` now lets an opponent win.
    pub fn tile(&self, t: Tile) -> f64 {
        self.tiles[t.index() as usize]
    }

    /// Estimated probability that `seat` is waiting with a legal win.
    pub fn threat(&self, seat: usize) -> f64 {
        self.threat[seat]
    }

    /// The largest opponent threat.
    pub fn max_threat(&self) -> f64 {
        self.threat.iter().copied().fold(0.0, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use mahjong_games::{Game, Policy};
    use mahjong_hk::RuleSet;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn estimates_are_probabilities_and_dead_tiles_are_safe() {
        let mut rng = SmallRng::seed_from_u64(3);
        let bot = Bot::by_name("fast").unwrap();
        for _ in 0..20 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(p) = s.current_player() {
                if s.phase_name() == "turn" {
                    let d = Danger::estimate(&s, p);
                    let unseen = s.unseen(p);
                    for t in Tile::all_kinds() {
                        let x = d.tile(t);
                        assert!((0.0..=1.0).contains(&x));
                        if unseen[t] == 0 {
                            assert_eq!(x, 0.0, "{t} has no unseen copy");
                        }
                    }
                }
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                s.apply(a);
            }
        }
    }
}
