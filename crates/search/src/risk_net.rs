//! The risk network: a trained network in place of the danger tables (plan
//! `docs/plans/2026-11-16-self-play-network.md`, Task 15; Brian, 2026-11-19: "I
//! think the network should be a network, not a lookup table. given all the
//! conditions (the targeted tile you are trying to hit + its unseen copies +
//! opponent discards and their relation to his flush)").
//!
//! For the player to act it gives, for each opponent and each of the 34 tile
//! kinds, the chance that throwing the kind now lets that opponent win on it by
//! discard (with the faan minimum and the passed-win lock, rules §4.4) and the
//! points per share such a win would pay. [`crate::Danger`] combines them as it
//! combines the tables (`1 − Π (1 − p_o)`, `Σ p_o × points_o × shares_o`) when
//! [`crate::danger::with_risk`] is on; off by default.
//!
//! **Input.** The lab's `features` block ([`crate::features::encode`], 933
//! floats: own hand, melds, discards, unseen copies, …), the `history` block
//! ([`crate::features::encode_history`], every discard in order) and the `risk`
//! block of this module ([`encode_risk`], [`RISK_LEN`] floats), which spells out
//! per opponent what Brian named: each kind's relation to the opponent's exposed
//! sets (same suit as a one-suit shape or an honour, other suit), whether the
//! opponent itself discarded the kind, whether the kind is three apart from one
//! of its discards (筋), the passed-win lock on the kind
//! ([`crate::exact_wait::refused`]), and its concealed-tile count (1, 4, 7, 10,
//! 13), exposed melds and shape.
//!
//! **Output.** MJNN v2 ([`crate::net`]) with the output slices `win` (3 × 34
//! logits, relative seats 1 = next, 2 = across, 3 = previous, then kind) and
//! `pay` (3 × 34 values of `log2(points ÷ 8)`, 0 = the 3-faan minimum, 4 = the
//! 10-faan cap). Trained by `python/mahjong/risk/` on the rows the `risk_log`
//! example writes.
//!
//! **Labels.** [`labels`] is the true answer from the opponents' real hands,
//! for the training data and its test: a kind wins exactly when the engine's own
//! win check accepts it (complete, the faan minimum, not locked).

use crate::exact_wait::refused;
use crate::net::Mlp;
use mahjong_games::Game;
use mahjong_hk::{score_win, Event, State, WinContext};
use mahjong_tiles::{Tile, NUM_KINDS};
use std::fmt;
use std::ops::Range;
use std::path::Path;

/// Opponents per decision (relative seats 1, 2, 3).
pub const OPPONENTS: usize = 3;
/// Tile planes per opponent in the `risk` block: one-suit shape and the kind in
/// that suit or an honour; one-suit shape and the kind in another suit; the
/// opponent itself discarded the kind; the kind is three apart from one of its
/// discards (筋); the passed-win lock on the kind.
pub const RISK_PLANES: usize = 5;
/// Scalars per opponent: concealed count one-hot (1, 4, 7, 10, 13), exposed
/// melds / 4, one-suit shape, a meld that already carries faan.
pub const RISK_SCALARS: usize = 8;
/// Length of [`encode_risk`]'s output.
pub const RISK_LEN: usize = OPPONENTS * (RISK_PLANES * NUM_KINDS + RISK_SCALARS);
/// Values per output head: opponents × kinds.
pub const HEAD_LEN: usize = OPPONENTS * NUM_KINDS;
/// The 3-faan minimum's points per share, the `pay` head's zero.
pub const BASE_POINTS: f64 = 8.0;

/// The `risk` input block for the player to act (length [`RISK_LEN`]).
///
/// # Panics
/// Panics if the hand is over or `out` has the wrong length.
pub fn encode_risk(state: &State, out: &mut [f32]) {
    let me = state.current_player().expect("hand in progress");
    encode_risk_for(state, me, out);
}

/// [`encode_risk`] from seat `me`'s point of view.
pub fn encode_risk_for(state: &State, me: usize, out: &mut [f32]) {
    assert_eq!(out.len(), RISK_LEN);
    out.fill(0.0);
    let rules = state.rules();
    let mut discarded = [[false; NUM_KINDS]; 4];
    for e in state.events() {
        if let Event::Discard { seat, tile, .. } = *e {
            discarded[seat as usize][tile.index() as usize] = true;
        }
    }
    for r in 1..4 {
        let o = (me + r) % 4;
        let melds = state.melds(o);
        let exposed = melds.iter().filter(|m| !m.concealed).count().min(4);
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
        let lock = refused(state, o);
        let base = (r - 1) * (RISK_PLANES * NUM_KINDS + RISK_SCALARS);
        let planes = &mut out[base..base + RISK_PLANES * NUM_KINDS];
        for t in Tile::all_kinds() {
            let i = t.index() as usize;
            if let Some(s) = flush {
                if t.suit().is_none_or(|x| x == s) {
                    planes[i] = 1.0;
                } else {
                    planes[NUM_KINDS + i] = 1.0;
                }
            }
            if discarded[o][i] {
                planes[2 * NUM_KINDS + i] = 1.0;
            }
            if t.suit().is_some()
                && [-3i8, 3].iter().any(|&d| {
                    t.offset(d)
                        .is_some_and(|u| discarded[o][u.index() as usize])
                })
            {
                planes[3 * NUM_KINDS + i] = 1.0;
            }
            if lock[i] {
                planes[4 * NUM_KINDS + i] = 1.0;
            }
        }
        let s =
            &mut out[base + RISK_PLANES * NUM_KINDS..base + RISK_PLANES * NUM_KINDS + RISK_SCALARS];
        let concealed = state.concealed_count(o);
        if let Some(k) = [1usize, 4, 7, 10, 13].iter().position(|&c| c == concealed) {
            s[k] = 1.0;
        }
        s[5] = exposed as f32 / 4.0;
        s[6] = flush.is_some() as u8 as f32;
        s[7] = has_value as u8 as f32;
    }
}

/// The true answer for one decision, from the opponents' real hands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Labels {
    /// `win[o][k]`: opponent `o` (absolute seat) wins on kind `k` by discard
    /// now: complete with its melds, the faan minimum, not locked (§4.4).
    /// Always false for `me`.
    pub win: [[bool; NUM_KINDS]; 4],
    /// Points per share of that win (0 where `win` is false).
    pub points: [[u8; NUM_KINDS]; 4],
}

/// What each opponent would really win on if `me` discarded each kind now
/// (reads the opponents' hands: training data and tests only). The same
/// check as the engine's claim window except the dealer's first discard (#30
/// 地糊, a limit hand either way: the yes or no is the same, the points may
/// differ). A kind of which `me` sees all four copies cannot be held, so it
/// is false too.
pub fn labels(state: &State, me: usize) -> Labels {
    let rules = state.rules();
    let mut out = Labels {
        win: [[false; NUM_KINDS]; 4],
        points: [[0; NUM_KINDS]; 4],
    };
    for o in (0..4).filter(|&o| o != me) {
        let hand = state.hand(o);
        let melds = state.melds(o);
        for k in Tile::all_kinds() {
            let i = k.index() as usize;
            if hand[k] >= 4 || state.passed_win_locked(o, k) {
                continue;
            }
            let mut full = *hand;
            full.add(k);
            let ctx = WinContext {
                seat: o,
                win_tile: Some(k),
                self_drawn: false,
                bonus: state.bonus(o),
                last_tile: false,
                kong_chain: 0,
                robbing_kong: false,
                heavenly: false,
                earthly: false,
            };
            if let Some(score) = score_win(&full, melds, &ctx, rules) {
                if score.meets_minimum(rules) {
                    out.win[o][i] = true;
                    out.points[o][i] = rules.points(score.faan).min(255) as u8;
                }
            }
        }
    }
    out
}

/// What the network says for one decision.
#[derive(Clone, Debug, PartialEq)]
pub struct RiskOutput {
    /// `p[o][k]`: P(absolute seat `o` wins on kind `k` if I discard it now);
    /// 0 for me and for kinds with no unseen copy.
    pub p: [[f64; NUM_KINDS]; 4],
    /// Points per share of that win (`BASE_POINTS × 2^pay`, within the rules'
    /// range).
    pub points: [[f64; NUM_KINDS]; 4],
}

/// A loaded risk network (see the module docs).
pub struct RiskNet {
    net: Mlp,
    win: Range<usize>,
    pay: Range<usize>,
}

impl fmt::Debug for RiskNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RiskNet")
            .field("inputs", &self.net.input_len())
            .field("parameters", &self.net.parameters())
            .finish()
    }
}

impl RiskNet {
    /// Loads an MJNN v2 risk network.
    pub fn load(path: impl AsRef<Path>) -> Result<RiskNet, String> {
        let path = path.as_ref();
        Self::from_mlp(Mlp::load(path)?).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// Wraps a loaded network; it must read the `risk` input and name the heads
    /// `win` and `pay`, each of [`HEAD_LEN`] values.
    pub fn from_mlp(net: Mlp) -> Result<RiskNet, String> {
        if !net.inputs().iter().any(|s| s.name == "risk") {
            return Err("not a risk network: no `risk` input slice".into());
        }
        let head = |name: &str| match net.output(name) {
            Some(r) if r.len() == HEAD_LEN => Ok(r),
            Some(r) => Err(format!(
                "output slice {name:?} has {} values, expected {HEAD_LEN}",
                r.len()
            )),
            None => Err(format!("not a risk network: no {name:?} output slice")),
        };
        Ok(RiskNet {
            win: head("win")?,
            pay: head("pay")?,
            net,
        })
    }

    /// The network itself.
    pub fn mlp(&self) -> &Mlp {
        &self.net
    }

    /// The network's answer for `me`, who must be the player to act (the input
    /// blocks describe the player to act); `None` otherwise.
    pub fn estimate(&self, state: &State, me: usize) -> Option<RiskOutput> {
        if state.current_player() != Some(me) {
            return None;
        }
        let y = self.net.forward_state(state);
        let unseen = state.unseen(me);
        let rules = state.rules();
        let max_points = rules.points(rules.faan_cap) as f64;
        let mut out = RiskOutput {
            p: [[0.0; NUM_KINDS]; 4],
            points: [[0.0; NUM_KINDS]; 4],
        };
        for r in 1..4 {
            let o = (me + r) % 4;
            for t in Tile::all_kinds() {
                let i = t.index() as usize;
                if unseen[t] == 0 {
                    continue;
                }
                let at = (r - 1) * NUM_KINDS + i;
                out.p[o][i] = sigmoid(y[self.win.start + at]);
                let pay = (y[self.pay.start + at] as f64).clamp(0.0, 4.0);
                out.points[o][i] = (BASE_POINTS * pay.exp2()).clamp(BASE_POINTS, max_points);
            }
        }
        Some(out)
    }
}

fn sigmoid(x: f32) -> f64 {
    1.0 / (1.0 + (-(x as f64)).exp())
}

/// A small risk network with fixed weights, for tests here and in `danger.rs`.
#[cfg(test)]
pub(crate) fn test_network() -> RiskNet {
    use crate::features::{HISTORY_LEN, STATE_LEN};
    let mut b = b"MJNN".to_vec();
    b.extend(2u32.to_le_bytes());
    let slices = |b: &mut Vec<u8>, s: &[(&str, usize, usize)]| {
        b.extend((s.len() as u32).to_le_bytes());
        for &(name, off, len) in s {
            b.extend((name.len() as u32).to_le_bytes());
            b.extend(name.as_bytes());
            b.extend((off as u32).to_le_bytes());
            b.extend((len as u32).to_le_bytes());
        }
    };
    let inp = STATE_LEN + HISTORY_LEN + RISK_LEN;
    slices(
        &mut b,
        &[
            ("features", 0, STATE_LEN),
            ("history", STATE_LEN, HISTORY_LEN),
            ("risk", STATE_LEN + HISTORY_LEN, RISK_LEN),
        ],
    );
    slices(&mut b, &[("win", 0, HEAD_LEN), ("pay", HEAD_LEN, HEAD_LEN)]);
    b.extend(2u32.to_le_bytes());
    for &(out, i) in &[(8usize, inp), (2 * HEAD_LEN, 8usize)] {
        b.extend((out as u32).to_le_bytes());
        b.extend((i as u32).to_le_bytes());
        for k in 0..out * i {
            b.extend((((k % 5) as f32 - 2.0) * 0.01).to_le_bytes());
        }
        for k in 0..out {
            b.extend((k as f32 * 0.01).to_le_bytes());
        }
    }
    RiskNet::from_mlp(Mlp::from_bytes(&b).unwrap()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::Bot;
    use mahjong_games::Policy;
    use mahjong_hk::{Action, RuleSet};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    /// Plays hands of four `stage` bots and calls `f` at every discard
    /// decision (state, seat, the tile thrown) until it has seen `n` of them.
    fn each_discard(seed: u64, n: usize, mut f: impl FnMut(&State, usize, Tile)) {
        let bot = Bot::by_name("stage").unwrap();
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut seen = 0;
        while seen < n {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while let Some(p) = s.current_player() {
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                if let Action::Discard(t) = a {
                    f(&s, p, t);
                    seen += 1;
                }
                s.apply(a);
            }
        }
    }

    /// The labeller says yes exactly when the engine lets that opponent declare
    /// a win on the tile really thrown (claim window), and a hand below the
    /// faan minimum gives no.
    #[test]
    fn labels_match_the_engines_win_check() {
        let (mut wins, mut checked, mut below) = (0, 0, 0);
        each_discard(21, 3000, |s, me, t| {
            let l = labels(s, me);
            let mut next = s.clone();
            next.apply(Action::Discard(t));
            // Every opponent's legal actions in the claim window, in turn.
            let mut engine = [false; 4];
            let mut probe = next.clone();
            while probe.in_claim() {
                let q = probe.current_player().unwrap();
                let legal = probe.legal_actions_vec();
                engine[q] = legal.contains(&Action::Win);
                probe.apply(Action::Pass);
            }
            for o in (0..4).filter(|&o| o != me) {
                let i = t.index() as usize;
                assert_eq!(l.win[o][i], engine[o], "seat {me} throws {t}: opponent {o}");
                checked += 1;
                wins += l.win[o][i] as usize;
                if l.win[o][i] {
                    assert!(l.points[o][i] >= 8, "{}", l.points[o][i]);
                } else {
                    assert_eq!(l.points[o][i], 0);
                    // A complete hand that does not reach the minimum is a no.
                    let mut full = *s.hand(o);
                    full.add(t);
                    if full.total() + 3 * s.melds(o).len() == 14 && !s.passed_win_locked(o, t) {
                        below += 1;
                    }
                }
            }
            assert!(l.win[me].iter().all(|&w| !w));
        });
        assert!(checked >= 9000, "{checked}");
        assert!(wins > 20, "{wins} wins seen");
        let _ = below;
    }

    /// The `risk` block is public information, the right length, and marks the
    /// passed-win lock and the flush relation as the facts say.
    #[test]
    fn risk_block_reads_public_facts() {
        let mut rng = SmallRng::seed_from_u64(5);
        let mut locked = 0;
        let mut flush = 0;
        each_discard(7, 400, |s, me, _| {
            let mut x = vec![0.0; RISK_LEN];
            encode_risk(s, &mut x);
            let other = s.determinize(me, &mut rng);
            let mut y = vec![0.0; RISK_LEN];
            encode_risk(&other, &mut y);
            assert_eq!(x, y, "hidden tiles changed the block");
            for r in 1..4 {
                let o = (me + r) % 4;
                let base = (r - 1) * (RISK_PLANES * NUM_KINDS + RISK_SCALARS);
                let lock = refused(s, o);
                for i in 0..NUM_KINDS {
                    assert_eq!(x[base + 4 * NUM_KINDS + i] == 1.0, lock[i]);
                    locked += lock[i] as usize;
                    flush += (x[base + i] == 1.0) as usize;
                    assert!(!(x[base + i] == 1.0 && x[base + NUM_KINDS + i] == 1.0));
                }
                let sc = &x[base + RISK_PLANES * NUM_KINDS..][..RISK_SCALARS];
                assert_eq!(sc[..5].iter().sum::<f32>(), 1.0, "concealed count one-hot");
                assert_eq!(sc[5], s.melds(o).exposed_count() as f32 / 4.0);
            }
        });
        assert!(locked > 0 && flush > 0);
    }

    /// A tiny v2 file with `features`, `history` and `risk` inputs and the two
    /// heads loads, and its answer is 0 on dead kinds and within the points range.
    #[test]
    fn a_small_network_loads_and_answers() {
        let net = test_network();
        let mut dead = 0;
        each_discard(3, 50, |s, me, _| {
            let out = net.estimate(s, me).unwrap();
            let unseen = s.unseen(me);
            let cap = s.rules().points(s.rules().faan_cap) as f64;
            for o in 0..4 {
                for t in Tile::all_kinds() {
                    let i = t.index() as usize;
                    let (p, pts) = (out.p[o][i], out.points[o][i]);
                    if o == me || unseen[t] == 0 {
                        assert_eq!((p, pts), (0.0, 0.0));
                        dead += 1;
                    } else {
                        assert!((0.0..=1.0).contains(&p));
                        assert!((BASE_POINTS..=cap).contains(&pts));
                    }
                }
            }
            assert_eq!(net.estimate(s, (me + 1) % 4), None);
        });
        assert!(dead > 50);
        // A belief model is not a risk network.
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief.mjnn");
        if p.exists() {
            assert!(RiskNet::load(&p)
                .unwrap_err()
                .contains("not a risk network"));
        }
    }
}
