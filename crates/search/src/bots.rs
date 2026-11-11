//! Rule-based players. All of them look only at what their seat can see.
//!
//! * [`Efficiency`] plays for *any* complete hand as fast as possible (fewest
//!   exchanges, then most accepting tiles). Under the 3-faan minimum it often
//!   completes hands it cannot declare: it is the "Riichi-style efficiency"
//!   baseline that HK strategy has to beat.
//! * [`Value`] plays for complete hands that meet the minimum (faan-aware
//!   shanten, then acceptance, then the target's faan). It claims only when that
//!   brings it closer (or, as `closed`, never claims except to win). With
//!   `defense > 0` it also folds: when its hand is far from
//!   ready and an opponent looks dangerous, it discards the safest tile.
//! * [`Fast`] is a cheaper `Value` for Monte Carlo rollouts: it shortlists
//!   discards by plain shanten before applying the faan-aware measure.
//!
//! Every bot declares a legal win whenever it can.

use crate::danger::{Danger, Opponent};
use crate::danger_tables::WAIT;
use crate::ev::{Belief, EvBot};
use crate::passed::Passed;
use crate::race::RaceBot;
use crate::view::SeatView;
use mahjong_analysis::{acceptance, faan_shanten, shanten, HandView, UNREACHABLE};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, Meld, MeldKind, State};
use mahjong_tiles::{Suit, Tile, TileCounts};
use rand::Rng;
use std::sync::Arc;

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

/// Lower is better: (distance, -acceptance, -target faan, connectedness). For
/// [`Metric::FaanQuick`] on a ready hand: (0, dead wait, -best winning faan,
/// -live winning tiles).
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
    if metric == Metric::FaanQuick && best == 0 && keyed.iter().filter(|k| k.1 .0 == 0).count() > 1
    {
        // Ready: look at the actual winning tiles of each tied discard. Without
        // this the rollout bot broke the tie by tile order and kept a wait whose
        // four copies were all visible (Brian's pung of 8p, plan 2026-11-09
        // Task 1). A dead wait ranks last; then the most faan any live winning
        // tile gives; then more live tiles. The faan comes from the completed
        // hands, not from `distance_after`: `faan_shanten` calls four pin pungs
        // and a Green "ready for a full flush" (it sets the Green aside and finds
        // no tile left to pair), which made the bot swap a live flush wait for
        // an honour single.
        for (t, key) in keyed.iter_mut().filter(|k| k.1 .0 == 0) {
            let mut hand = view.concealed;
            hand.remove(*t);
            let wins = live_winning_tiles(view, &hand);
            let live: i32 = wins.iter().map(|&(w, _)| view.unseen[w] as i32).sum();
            let faan = wins.iter().map(|&(_, f)| f as i32).max().unwrap_or(0);
            *key = (0, (live == 0) as i32, -faan, -live);
        }
    } else if metric != Metric::FaanQuick {
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

/// The live winning tiles of a waiting hand (`hand` is the concealed part, 13
/// tiles less 3 per meld of `view`): each kind with an unseen copy that
/// completes a legal win, with the faan of that win (`Target::faan`, no win
/// way). Dead waits (no unseen copy) are left out. Shared by the rollout bot's
/// ready tie-break and the race bot (`crate::race`).
pub(crate) fn live_winning_tiles(view: &SeatView, hand: &TileCounts) -> Vec<(Tile, u8)> {
    let mut hand = *hand;
    let mut out = Vec::new();
    for w in Tile::all_kinds() {
        // A winning tile is a held tile or within two of one in its suit
        // (any tile that can finish thirteen orphans, with no melds).
        let near = hand[w] > 0
            || view.melds.is_empty() && w.is_terminal_or_honour()
            || [-2i8, -1, 1, 2]
                .iter()
                .any(|&d| w.offset(d).is_some_and(|u| hand[u] > 0));
        if !near || view.unseen[w] == 0 || hand[w] >= 4 {
            continue;
        }
        hand.add(w);
        let fs = faan_shanten(
            &HandView {
                concealed: &hand,
                ..view.hand_view()
            },
            &view.rules,
        );
        hand.remove(w);
        if fs.shanten < 0 {
            out.push((w, fs.target.map_or(0, |t| t.faan)));
        }
    }
    out
}

/// The best discard and its key.
pub(crate) fn best_discard(view: &SeatView, candidates: &[Tile], metric: Metric) -> (Tile, Key) {
    rank_discards(view, candidates, metric)[0]
}

/// The hand and melds after claiming `action` on `tile` from `from`.
pub(crate) fn after_claim(
    view: &SeatView,
    action: Action,
    tile: Tile,
    from: usize,
) -> Option<SeatView> {
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

/// An opponent of `me` showing an obvious flush: three or more exposed melds whose
/// suited ones all lie in one suit. Returns that seat and suit.
fn flush_seat(state: &State, me: usize) -> Option<(usize, Suit)> {
    (0..4).filter(|&o| o != me).find_map(|o| {
        let melds = state.melds(o);
        if melds.exposed_count() < 3 {
            return None;
        }
        let mut suits = melds.iter().filter_map(|m| m.tile.suit());
        let first = suits.next()?;
        suits.all(|s| s == first).then_some((o, first))
    })
}

/// Shared decision logic for [`Efficiency`], [`Value`] and [`Fast`].
#[derive(Clone, Debug)]
struct Brain {
    metric: Metric,
    /// Shortlist this many discards by plain shanten first (0 = no shortlist).
    shortlist: usize,
    /// 0 = never fold; higher = folds more readily.
    defense: f64,
    /// Whether to claim discards for melds (wins are always claimed).
    claims: bool,
    /// With a belief model (plan Task 6), danger comes from
    /// [`Danger::from_belief`] and the fold picks the tile with the smallest
    /// expected loss in points; without one, the fitted tables (unchanged).
    belief: Option<Arc<dyn Belief>>,
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
        // Against an obvious flush (plan 2026-11-09 Task 2). After Brian's pass on
        // 8p the defending rollout opponents fed his three exposed pin pungs:
        // a seat that was ready or one away never reached the fold below, and the
        // tables call a tile safe when the observer holds its last unseen copy,
        // though that copy is exactly what a waiting flush needs. For the
        // defending `fast` bot only (the other bots' play is unchanged): a tile of
        // the flush suit counts as if one copy were still unseen, and a seat
        // ready or one away gives way to the safest discard outside the suit that
        // keeps its distance or costs one step.
        let flush = (self.defense > 0.0 && self.metric == Metric::FaanQuick)
            .then(|| flush_seat(state, view.seat))
            .flatten();
        if let Some((o, suit)) = flush.filter(|_| key.0 <= 1) {
            if best.suit() == Some(suit) {
                let danger = Danger::estimate(state, view.seat);
                if danger.threat(o) * self.defense >= 0.25 {
                    let safer = discards
                        .iter()
                        .filter(|t| t.suit() != Some(suit))
                        .map(|&t| (distance_after(&view, t, self.metric).0, t))
                        .filter(|&(d, _)| d <= key.0 + 1)
                        .min_by(|a, b| {
                            a.0.cmp(&b.0)
                                .then(danger.tile(a.1).total_cmp(&danger.tile(b.1)))
                        });
                    if let Some((_, t)) = safer {
                        return Action::Discard(t);
                    }
                }
            }
        }
        if self.defense > 0.0 && key.0 >= 1 {
            let (mut danger, risk_of): (Danger, fn(&Danger, Tile) -> f64) = match &self.belief {
                Some(b) => (
                    Danger::from_belief(state, view.seat, b.as_ref()),
                    Danger::expected_loss,
                ),
                None => (Danger::estimate(state, view.seat), Danger::tile),
            };
            // Passed-tile safety (plan Task 14, `docs/race-bot.md` §13), for the
            // defending rollout bot only: a tile a seat let pass is safe against
            // it until that seat's next turn, and less safe with each 入章 since.
            if self.metric == Metric::FaanQuick {
                danger.apply_passed(&Passed::of(state));
            }
            // The "clearly safer" margin: 0.3 % deal-in, or that much of one
            // 3-faan share (8 points) in expected loss.
            let margin = if self.belief.is_some() { 0.024 } else { 0.003 };
            let flush_risk =
                flush.map(|(o, suit)| (suit, Opponent::of(state, o), danger.threat(o)));
            let risk = |t: Tile| {
                let r = risk_of(&danger, t);
                match &flush_risk {
                    Some((suit, opp, threat)) if t.suit() == Some(*suit) => {
                        let mut one = view.unseen;
                        one[t] = one[t].max(1);
                        let cell = opp.wait_cell(t, &one).expect("one copy unseen");
                        r.max(threat * WAIT[cell])
                    }
                    _ => r,
                }
            };
            let threat = danger.max_threat();
            // Fold when far from ready and someone looks close to winning.
            let fold = (key.0 >= 2 && threat * self.defense >= 0.3)
                || (key.0 >= 1 && threat * self.defense >= 0.6);
            if fold {
                best = *discards
                    .iter()
                    .min_by(|a, b| risk(**a).total_cmp(&risk(**b)))
                    .expect("a discard");
            } else if threat > 0.0 {
                // Among equally good discards, prefer the safer one.
                let near: Vec<Tile> = candidates
                    .iter()
                    .copied()
                    .filter(|&t| distance_after(&view, t, self.metric).0 == key.0)
                    .collect();
                if let Some(&safer) = near.iter().min_by(|a, b| risk(**a).total_cmp(&risk(**b))) {
                    if risk(safer) + margin < risk(best) {
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
        if !self.claims {
            return Action::Pass;
        }
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
            claims: true,
            belief: None,
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
            claims: true,
            belief: None,
        })
    }

    /// The `defensive` bot (fold strength 1) reading danger from a belief
    /// model: it folds to the tile with the smallest expected loss in points
    /// (`defensive:belief=models/belief.mjnn`, plan Task 6).
    pub fn defensive_with(belief: Arc<dyn Belief>) -> Self {
        let mut v = Value::new(1.0);
        v.0.belief = Some(belief);
        v
    }

    /// A value bot that never claims a discard except to win (stays concealed).
    pub fn closed() -> Self {
        Value(Brain {
            metric: Metric::Faan,
            shortlist: 0,
            defense: 0.0,
            claims: false,
            belief: None,
        })
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::new(0.0)
    }
}

impl Fast {
    /// A rollout bot that folds with the given strength (0 = never), like
    /// [`Value::new`].
    pub fn new(defense: f64) -> Self {
        Fast(Brain {
            metric: Metric::FaanQuick,
            shortlist: 5,
            defense,
            claims: true,
            belief: None,
        })
    }
}

impl Default for Fast {
    fn default() -> Self {
        Fast::new(0.0)
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
        // u32, not usize: the same draw on wasm32 and 64-bit (hk-games D28).
        legal[rng.gen_range(0..legal.len() as u32) as usize]
    }
}

/// Any of the bots, chosen at run time.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Bot {
    Random(RandomBot),
    Efficiency(Efficiency),
    Value(Value),
    Fast(Fast),
    /// Brian's explicit EV design ([`crate::ev`]).
    Ev(EvBot),
    /// The race over all four players' chances (B28, [`crate::race`]).
    Race(RaceBot),
}

impl Bot {
    /// Builds a bot by name: `random`, `efficiency`, `value`, `defensive`,
    /// `closed` (value, never claims melds), `fast`, `ev` and `race` (with the
    /// heuristic belief; `ev:` and `race:` spec options are parsed by
    /// [`crate::Agent`]).
    pub fn by_name(name: &str) -> Option<Bot> {
        Some(match name {
            "ev" => Bot::Ev(EvBot::default()),
            "race" => Bot::Race(RaceBot::default()),
            "random" => Bot::Random(RandomBot),
            "efficiency" => Bot::Efficiency(Efficiency::default()),
            "value" => Bot::Value(Value::default()),
            "defensive" => Bot::Value(Value::new(1.0)),
            "closed" => Bot::Value(Value::closed()),
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
            Bot::Ev(b) => b.choose(state, legal, rng),
            Bot::Race(b) => b.choose(state, legal, rng),
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
