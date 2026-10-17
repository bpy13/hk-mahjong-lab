//! A heuristic deal-in danger model from public information.
//!
//! For each opponent it estimates a **threat**: the chance they are already
//! waiting on a winning tile, from how many melds they have exposed, whether
//! those melds already carry faan, and how far the hand has progressed. Then,
//! for each tile, how likely a waiting opponent is to want it:
//!
//! * no unseen copies left → the opponent cannot get it, so it is safe;
//! * an honour with at most one unseen copy can only be a single-tile wait;
//! * middle suited tiles are wanted by more shapes than terminals;
//! * a tile the opponent has discarded themselves is less likely (Hong Kong
//!   rules let them win on it anyway, so it is not fully safe);
//! * if every exposed meld of the opponent is in one suit (plus honours), they
//!   are probably making a flush: that suit and honours are dangerous, the
//!   other suits nearly safe.
//!
//! These are rough, documented guesses for bots and for study-mode hints. The
//! solver measures deal-in rates directly by simulation, and Phase 3 replaces this
//! model with a learned one.

use mahjong_hk::{MeldKind, State};
use mahjong_tiles::{Suit, Tile, NUM_KINDS};

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
        let river = state.river();
        let rules = state.rules();
        let initial_wall = if rules.flowers { 91.0 } else { 83.0 };
        let progress = (1.0 - state.wall_remaining() as f64 / initial_wall).clamp(0.0, 1.0);
        let mut tiles = [0.0f64; NUM_KINDS];
        let mut safe = [1.0f64; NUM_KINDS];
        let mut threat = [0.0f64; 4];
        for o in (0..4).filter(|&o| o != me) {
            let melds = state.melds(o);
            let exposed = melds.exposed_count().min(4);
            let from_melds = [0.04, 0.15, 0.35, 0.6, 0.85][exposed];
            let from_time = 0.55 * progress * progress;
            let mut t_o = 1.0 - (1.0 - from_melds) * (1.0 - from_time);
            // Exposed melds without any value mean the rest of the hand must find
            // the faan: less likely to be ready to win.
            let has_value = melds.iter().any(|m| {
                m.is_pung_like()
                    && (m.tile.is_dragon()
                        || m.tile == Tile::wind(o)
                        || m.tile == Tile::wind(rules.prevailing_wind))
            });
            let suits: Vec<Suit> = {
                let mut v: Vec<Suit> = melds.iter().filter_map(|m| m.tile.suit()).collect();
                v.sort();
                v.dedup();
                v
            };
            let flush = (exposed >= 1 && suits.len() == 1).then(|| suits[0]);
            if exposed > 0 && !has_value && flush.is_none() {
                t_o *= 0.6;
            }
            threat[o] = t_o;
            let discarded: Vec<Tile> = river.iter().filter(|r| r.0 == o).map(|r| r.1).collect();
            let mut weights = [0.0f64; NUM_KINDS];
            for t in Tile::all_kinds() {
                let left = unseen[t];
                let mut w = if left == 0 {
                    0.0
                } else if t.is_honour() {
                    if left <= 1 {
                        0.3
                    } else {
                        0.8
                    }
                } else {
                    match t.number() {
                        Some(1 | 9) => 0.7,
                        Some(2 | 8) => 0.9,
                        _ => 1.2,
                    }
                };
                if discarded.contains(&t) {
                    w *= 0.35;
                }
                if let Some(s) = flush {
                    w *= match t.suit() {
                        Some(x) if x == s => 2.0,
                        Some(_) => 0.15,
                        None => 1.3,
                    };
                }
                // Chows on the table do not change which tiles they wait on, but
                // pungs of a tile mean no one else holds many of it.
                if melds
                    .iter()
                    .any(|m| m.kind != MeldKind::Chow && m.tile == t)
                {
                    w *= 0.5;
                }
                weights[t.index() as usize] = w;
            }
            let total: f64 = weights.iter().sum();
            if total <= 0.0 {
                continue;
            }
            for i in 0..NUM_KINDS {
                // A waiting hand has about 2.5 winning tile kinds on average.
                let p = t_o * (2.5 * weights[i] / total).min(1.0);
                safe[i] *= 1.0 - p;
            }
        }
        for i in 0..NUM_KINDS {
            tiles[i] = 1.0 - safe[i];
        }
        Danger { tiles, threat }
    }

    /// Estimated probability that discarding `t` lets an opponent win.
    pub fn tile(&self, t: Tile) -> f64 {
        self.tiles[t.index() as usize]
    }

    /// Estimated probability that `seat` is waiting.
    pub fn threat(&self, seat: usize) -> f64 {
        self.threat[seat]
    }

    /// The largest opponent threat.
    pub fn max_threat(&self) -> f64 {
        self.threat.iter().copied().fold(0.0, f64::max)
    }
}
