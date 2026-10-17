//! Rule-based players. All of them look only at what their seat can see.
//!
//! * [`Efficiency`] plays for *any* complete hand as fast as possible (fewest
//!   exchanges, then most accepting tiles). Under the 3-faan minimum it often
//!   completes hands it cannot declare: it is the "Riichi-style efficiency"
//!   baseline that HK strategy has to beat.
//! * [`Value`] plays for complete hands that meet the minimum (faan-aware
//!   shanten, then acceptance, then the target's faan). It claims only when that
//!   brings it closer. With `defense > 0` it also folds: when its hand is far from
//!   ready and an opponent looks dangerous, it discards the safest tile.
//! * [`Fast`] is a cheaper `Value` for Monte Carlo rollouts: it shortlists
//!   discards by plain shanten before applying the faan-aware measure.
//!
//! Every bot declares a legal win whenever it can.

use crate::danger::Danger;
use crate::view::SeatView;
use mahjong_analysis::{acceptance, faan_shanten, shanten, HandView, UNREACHABLE};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, Meld, MeldKind, State};
use mahjong_tiles::{Tile, TileCounts};
use rand::Rng;

/// How a bot measures progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Metric {
    /// Plain shanten, then tile acceptance.
    Plain,
    /// Faan-aware shanten, then acceptance, then the target's faan.
    Faan,
    /// Faan-aware shanten, then the target's faan, then how isolated the tile
    /// is (no acceptance counting: for fast rollouts).
    FaanQuick,
}

fn distance(view: &SeatView, metric: Metric) -> i8 {
    match metric {
        Metric::Plain => shanten(&view.concealed, view.melds.len()),
        Metric::Faan | Metric::FaanQuick => faan_shanten(&view.hand_view(), &view.rules).shanten,
    }
}

/// Lower is better: (distance, -acceptance, -target faan, connectedness).
pub(crate) type Key = (i8, i32, i32, i32);

/// How many tiles of the hand are near `t` (same tile, or within two in the same
/// suit): isolated tiles are the natural discards.
fn connectedness(hand: &TileCounts, t: Tile) -> i32 {
    let mut n = hand[t] as i32 - 1;
    for d in [-2i8, -1, 1, 2] {
        if let Some(u) = t.offset(d) {
            n += hand[u] as i32 * if d.abs() == 1 { 2 } else { 1 };
        }
    }
    n
}

/// The cheap part of the key for the hand left after discarding `tile`.
fn distance_after(view: &SeatView, tile: Tile, metric: Metric) -> (i8, i32) {
    let mut hand = view.concealed;
    hand.remove(tile);
    match metric {
        Metric::Plain => (shanten(&hand, view.melds.len()), 0),
        Metric::Faan | Metric::FaanQuick => {
            let fs = faan_shanten(
                &HandView {
                    concealed: &hand,
                    ..view.hand_view()
                },
                &view.rules,
            );
            if fs.shanten >= UNREACHABLE {
                // No plan meets the minimum yet: fall back to plain efficiency.
                (UNREACHABLE + shanten(&hand, view.melds.len()), 0)
            } else {
                (fs.shanten, -(fs.target.map_or(0, |t| t.faan) as i32))
            }
        }
    }
}

/// Ranks discards, best first. Acceptance is counted only for the candidates
/// tied at the best distance, which keeps this fast.
pub(crate) fn rank_discards(
    view: &SeatView,
    candidates: &[Tile],
    metric: Metric,
) -> Vec<(Tile, Key)> {
    let mut keyed: Vec<(Tile, Key)> = candidates
        .iter()
        .map(|&t| {
            let (d, faan) = distance_after(view, t, metric);
            (t, (d, 0, faan, connectedness(&view.concealed, t)))
        })
        .collect();
    let best = keyed
        .iter()
        .map(|k| k.1 .0)
        .min()
        .expect("at least one discard");
    if metric != Metric::FaanQuick {
        for (t, key) in keyed.iter_mut().filter(|k| k.1 .0 == best) {
            let mut hand = view.concealed;
            hand.remove(*t);
            let melds = view.melds.len();
            key.1 = -(match metric {
                Metric::Plain => acceptance(&hand, &view.unseen, |h| shanten(h, melds)),
                _ => {
                    let hv = HandView {
                        concealed: &hand,
                        ..view.hand_view()
                    };
                    acceptance(&hand, &view.unseen, |h| {
                        faan_shanten(&HandView { concealed: h, ..hv }, &view.rules).shanten
                    })
                }
            } as i32);
        }
    }
    keyed.sort_by_key(|&(t, k)| (k, t));
    keyed
}

/// The best discard and its key.
pub(crate) fn best_discard(view: &SeatView, candidates: &[Tile], metric: Metric) -> (Tile, Key) {
    rank_discards(view, candidates, metric)[0]
}

/// The hand and melds after claiming `action` on `tile` from `from`.
fn after_claim(view: &SeatView, action: Action, tile: Tile, from: usize) -> Option<SeatView> {
    let mut hand = view.concealed;
    let mut melds = view.melds.clone();
    match action {
        Action::Pung => {
            hand[tile] -= 2;
            melds.push(Meld::exposed(MeldKind::Pung, tile, from));
        }
        Action::ClaimKong => {
            hand[tile] -= 3;
            melds.push(Meld::exposed(MeldKind::Kong, tile, from));
        }
        Action::Chow(low) => {
            for k in 0..3 {
                let t = low.offset(k)?;
                if t != tile {
                    hand.remove(t);
                }
            }
            melds.push(Meld::exposed(MeldKind::Chow, low, from));
        }
        _ => return None,
    }
    Some(view.with(hand, melds))
}

/// Shared decision logic for [`Efficiency`], [`Value`] and [`Fast`].
#[derive(Clone, Debug)]
struct Brain {
    metric: Metric,
    /// Shortlist this many discards by plain shanten first (0 = no shortlist).
    shortlist: usize,
    /// 0 = never fold; higher = folds more readily.
    defense: f64,
}

impl Brain {
    fn discard_candidates(&self, view: &SeatView) -> Vec<Tile> {
        let all: Vec<Tile> = view.concealed.kinds().collect();
        if self.shortlist == 0 || all.len() <= self.shortlist {
            return all;
        }
        let mut keyed: Vec<(i8, Tile)> = all
            .iter()
            .map(|&t| {
                let mut h = view.concealed;
                h.remove(t);
                (shanten(&h, view.melds.len()), t)
            })
            .collect();
        keyed.sort();
        keyed
            .into_iter()
            .take(self.shortlist)
            .map(|(_, t)| t)
            .collect()
    }

    fn turn(&self, state: &State, legal: &[Action]) -> Action {
        if legal.contains(&Action::SelfDrawnWin) {
            return Action::SelfDrawnWin;
        }
        let view = SeatView::of_current(state);
        let discards: Vec<Tile> = legal
            .iter()
            .filter_map(|a| {
                if let Action::Discard(t) = a {
                    Some(*t)
                } else {
                    None
                }
            })
            .collect();
        let candidates = self.discard_candidates(&view);
        let (mut best, key) = best_discard(&view, &candidates, self.metric);
        // Kongs: declare one if the hand is no further from its goal afterwards.
        for &a in legal {
            let (hand, melds) = match a {
                Action::ConcealedKong(t) => {
                    let mut h = view.concealed;
                    h[t] -= 4;
                    let mut m = view.melds.clone();
                    m.push(Meld::concealed_kong(t));
                    (h, m)
                }
                Action::AddedKong(t) => {
                    let mut h = view.concealed;
                    h.remove(t);
                    let mut m = view.melds.clone();
                    let i = m
                        .iter()
                        .position(|x| x.kind == MeldKind::Pung && x.tile == t)
                        .expect("pung");
                    m[i].kind = MeldKind::Kong;
                    (h, m)
                }
                _ => continue,
            };
            if distance(&view.with(hand, melds), self.metric) <= key.0 {
                return a;
            }
        }
        if self.defense > 0.0 && key.0 >= 1 {
            let danger = Danger::estimate(state, view.seat);
            let threat = danger.max_threat();
            // Fold when far from ready and someone looks close to winning.
            let fold = (key.0 >= 2 && threat * self.defense >= 0.3)
                || (key.0 >= 1 && threat * self.defense >= 0.6);
            if fold {
                best = *discards
                    .iter()
                    .min_by(|a, b| danger.tile(**a).total_cmp(&danger.tile(**b)))
                    .expect("a discard");
            } else if threat > 0.0 {
                // Among equally good discards, prefer the safer one.
                let near: Vec<Tile> = candidates
                    .iter()
                    .copied()
                    .filter(|&t| distance_after(&view, t, self.metric).0 == key.0)
                    .collect();
                if let Some(&safer) = near
                    .iter()
                    .min_by(|a, b| danger.tile(**a).total_cmp(&danger.tile(**b)))
                {
                    if danger.tile(safer) + 0.02 < danger.tile(best) {
                        best = safer;
                    }
                }
            }
        }
        Action::Discard(best)
    }

    fn claim(&self, state: &State, legal: &[Action]) -> Action {
        if legal.contains(&Action::Win) {
            return Action::Win;
        }
        let Some((tile, from)) = state.claim_tile() else {
            return Action::Pass;
        };
        let view = SeatView::of_current(state);
        let now = distance(&view, self.metric);
        let mut best = (now, Action::Pass);
        for &a in legal {
            if let Some(after) = after_claim(&view, a, tile, from) {
                let cands = self.discard_candidates(&after);
                if cands.is_empty() {
                    continue;
                }
                let (_, key) = best_discard(&after, &cands, self.metric);
                // A kong also draws a replacement: treat it like the pung.
                if key.0 < best.0
                    || (key.0 == best.0 && a == Action::ClaimKong && best.1 == Action::Pung)
                {
                    best = (key.0, a);
                }
            }
        }
        best.1
    }

    fn choose(&self, state: &State, legal: &[Action]) -> Action {
        if state.in_claim() || state.in_rob() {
            self.claim(state, legal)
        } else {
            self.turn(state, legal)
        }
    }
}

macro_rules! bot {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Clone, Debug)]
        pub struct $name(Brain);

        impl Policy<State> for $name {
            fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], _rng: &mut R) -> Action {
                self.0.choose(state, legal)
            }
        }
    };
}

bot!(
    /// Plays for any complete hand as fast as possible (see the module docs).
    Efficiency
);
bot!(
    /// Plays for complete hands that meet the minimum (see the module docs).
    Value
);
bot!(
    /// A cheaper [`Value`] for rollouts.
    Fast
);

impl Default for Efficiency {
    fn default() -> Self {
        Efficiency(Brain {
            metric: Metric::Plain,
            shortlist: 0,
            defense: 0.0,
        })
    }
}

impl Value {
    /// A value bot that folds with the given strength (0 = never).
    pub fn new(defense: f64) -> Self {
        Value(Brain {
            metric: Metric::Faan,
            shortlist: 0,
            defense,
        })
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::new(0.0)
    }
}

impl Default for Fast {
    fn default() -> Self {
        Fast(Brain {
            metric: Metric::FaanQuick,
            shortlist: 5,
            defense: 0.0,
        })
    }
}

/// Plays uniformly at random, except that it always declares a legal win.
#[derive(Clone, Copy, Debug, Default)]
pub struct RandomBot;

impl Policy<State> for RandomBot {
    fn choose<R: Rng + ?Sized>(&self, _state: &State, legal: &[Action], rng: &mut R) -> Action {
        if let Some(&w) = legal
            .iter()
            .find(|a| matches!(a, Action::Win | Action::SelfDrawnWin))
        {
            return w;
        }
        legal[rng.gen_range(0..legal.len())]
    }
}

/// Any of the bots, chosen at run time.
#[derive(Clone, Debug)]
pub enum Bot {
    Random(RandomBot),
    Efficiency(Efficiency),
    Value(Value),
    Fast(Fast),
}

impl Bot {
    /// Builds a bot by name: `random`, `efficiency`, `value`, `defensive`, `fast`.
    pub fn by_name(name: &str) -> Option<Bot> {
        Some(match name {
            "random" => Bot::Random(RandomBot),
            "efficiency" => Bot::Efficiency(Efficiency::default()),
            "value" => Bot::Value(Value::default()),
            "defensive" => Bot::Value(Value::new(1.0)),
            "fast" => Bot::Fast(Fast::default()),
            _ => return None,
        })
    }
}

impl Policy<State> for Bot {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        match self {
            Bot::Random(b) => b.choose(state, legal, rng),
            Bot::Efficiency(b) => b.choose(state, legal, rng),
            Bot::Value(b) => b.choose(state, legal, rng),
            Bot::Fast(b) => b.choose(state, legal, rng),
        }
    }
}

/// Convenience: is the player to act in a position to win right now?
pub fn can_win(state: &State) -> bool {
    state
        .legal_actions_vec()
        .iter()
        .any(|a| matches!(a, Action::Win | Action::SelfDrawnWin))
}
