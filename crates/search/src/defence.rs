//! The defence bot Brian described for plan Task 20 (`docs/plans/2026-11-16-self-play-network.md`):
//! the three-stage bot whose discard, in stage 3 and wherever it has no line
//! (stage 0), is the one with the best expected points read from Task 19's
//! belief model (`stage-ev:belief=models/belief-order.mjnn`).
//!
//! Brian, 2026-11-20: "realistically the sole purpose of the defensive strategy is
//! to figure out the correct move that gives maximum EV. So the solver needs to be
//! able to re-evaluate the win-rate after hitting the tile, whether it leads to no
//! action or it gets claimed"; "realistically your loss EV is your deal in rate +
//! their self-draw rate". For each legal discard `t`:
//!
//! ```text
//! EV(t) = P(I self-draw | after t) × 3 × my self-drawn points
//!       + P(I win off a discard | after t) × 2 × my discard points
//!       − Σ_o deal-in_o(t) × Pay(o) × 2
//!       − Σ_o P(o self-draws | after t) × o's self-drawn points × my share
//!       − Σ_o P(t gives o its fourth exposed meld) × P(o self-draws | after t) × o's self-drawn points × 2
//! ```
//!
//! * `deal-in_o(t)` is the model's `dealin` head read **before** the discard (the
//!   chance `o` wins on `t` now); `Pay(o)` its `faan` head's expected points a
//!   share, and the discarder pays 2 shares whoever is liable (rules §8.2, 1 until
//!   2026-11-23; §8.3 covers self-draws only; [`Danger::expected_loss`]),
//!   times the loss factor refitted on this reading for stage 3 discards
//!   ([`crate::giveup::danger_of`], [`crate::giveup::READ_LOSS_FACTOR`]: the model's
//!   deal-in chance is 6.6 times too low on the safest tiles and 0.6 times on the
//!   most dangerous ones; the worker's addition to Brian's sum).
//! * The "after" chances are the `winner` head read from my seat on the position
//!   **right after** the discard, before the claim window resolves
//!   ([`BeliefModel::run_for`]). Task 19 logged such rows with the hand's real
//!   outcome as the label, so the reading already averages over no action and
//!   each opponent's claim by how often they happen (the worker's call, agreed
//!   by the orchestrator 2026-11-20: the position after a claim cannot be built
//!   without the claimer's hidden hand). The claim chances are kept in
//!   [`Terms::claim`] for the report.
//! * My points: when the hand is ready after `t`, the live winning tiles' mean
//!   points ([`crate::giveup::outs`]; the self-draw with its 自摸 faan); else the
//!   minimum hand's points (my win chance is small there).
//! * An opponent's self-drawn points use its `faan` head shifted by one faan
//!   (自摸); my share is 1, 3 when I am liable for it, 0 when another seat is.
//! * Liability (12-tile rule): when `t` could be the fourth exposed meld of `o`
//!   ([`Danger::liability_chance`]), I would pay the two other shares of its
//!   self-drawn win (§8.3: the liability covers self-draws only).
//!
//! Later deal-ins of mine are left out, as in Brian's sum. The tile with the
//! largest `EV` is played; the stage pick wins ties within [`TIE`].

use crate::belief::BeliefModel;
use crate::danger::Danger;
use crate::ev::FIRST_FAAN_BUCKET;
use crate::giveup;
use crate::view::SeatView;
use mahjong_games::Game;
use mahjong_hk::{Action, State, DISCARD_SHARES};
use mahjong_tiles::Tile;

/// Points within which the stage bot's own pick is kept.
pub const TIE: f64 = 0.05;

/// The terms of one candidate discard (points for me, per hand).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Terms {
    pub tile: Tile,
    /// My win term.
    pub win: f64,
    /// Expected loss from a deal-in on `tile` now.
    pub deal_in: f64,
    /// Expected loss from the opponents' self-drawn wins after it.
    pub self_draw: f64,
    /// Expected extra loss from the 12-tile liability.
    pub liability: f64,
    /// `win − deal_in − self_draw − liability`.
    pub ev: f64,
    /// P(I win) after the discard (self-drawn plus off a discard).
    pub my_win: f64,
    /// Per opponent (next, across, previous): P(it claims `tile` into a meld).
    pub claim: [f64; 3],
    /// Per opponent (next, across, previous): P(it wins the hand) after the discard.
    pub their_win: [f64; 3],
}

fn ready(view: &SeatView) -> bool {
    mahjong_analysis::faan_shanten(&view.hand_view(), &view.rules).shanten == 0
}

/// Every distinct candidate's terms, best `ev` first. `None` when the model has
/// no `winner` head (not a Task 19 file).
pub fn evaluate(
    state: &State,
    view: &SeatView,
    candidates: &[Tile],
    model: &BeliefModel,
) -> Option<Vec<Terms>> {
    let me = view.seat;
    let rules = state.rules();
    let pts = |f: u8| rules.points(f.min(rules.faan_cap)) as f64;
    let pre = model.run_for(state, me);
    pre.winner?;
    let danger = Danger::from_belief(state, me, model);
    // Expected self-drawn points per share of each opponent (its faan head + 1).
    let mut sd_pay = [0.0; 4];
    let mut sd_share = [0.0; 4];
    for o in (0..4).filter(|&o| o != me) {
        let b = pre.seat(o).expect("an opponent");
        sd_pay[o] = b
            .faan
            .iter()
            .enumerate()
            .map(|(k, &p)| p * pts(FIRST_FAAN_BUCKET + k as u8 + 1))
            .sum();
        sd_share[o] = match state.liable_for(o) {
            None => 1.0,
            Some(l) if l == me => 3.0,
            Some(_) => 0.0,
        };
    }
    let mut kinds: Vec<Tile> = candidates.to_vec();
    kinds.sort();
    kinds.dedup();
    let mut out = Vec::with_capacity(kinds.len());
    for t in kinds {
        let mut h = view.concealed;
        h.remove(t);
        let after_view = view.with(h, view.melds.clone());
        let (own_pts, disc_pts) = if ready(&after_view) {
            let o = giveup::outs(&after_view);
            (o.own_points, o.discard_points)
        } else {
            let m = rules.min_faan;
            (pts(m + 1), pts(m))
        };
        let mut next = state.clone();
        next.apply(Action::Discard(t));
        let deal_in = giveup::danger_of(&danger, t).1;
        let mut claim = [0.0; 3];
        for (r, c) in claim.iter_mut().enumerate() {
            *c = pre.opponents[r]
                .claim
                .map_or(0.0, |c| c[t.index() as usize]);
        }
        if next.is_terminal() {
            out.push(Terms {
                tile: t,
                win: 0.0,
                deal_in,
                self_draw: 0.0,
                liability: 0.0,
                ev: -deal_in,
                my_win: 0.0,
                claim,
                their_win: [0.0; 3],
            });
            continue;
        }
        let w = model.run_for(&next, me).winner?;
        // A discard win brings 2 shares from the discarder (rules §8.2).
        let win = w[0] * 3.0 * own_pts + w[1] * DISCARD_SHARES as f64 * disc_pts;
        let (mut self_draw, mut liability) = (0.0, 0.0);
        let mut their_win = [0.0; 3];
        for r in 1..4 {
            let o = (me + r) % 4;
            self_draw += w[2 * r] * sd_pay[o] * sd_share[o];
            their_win[r - 1] = w[2 * r] + w[2 * r + 1];
            liability += danger.liability_chance(o, t) * w[2 * r] * sd_pay[o] * 2.0;
        }
        let ev = win - deal_in - self_draw - liability;
        out.push(Terms {
            tile: t,
            win,
            deal_in,
            self_draw,
            liability,
            ev,
            my_win: w[0] + w[1],
            claim,
            their_win,
        });
    }
    out.sort_by(|a, b| b.ev.total_cmp(&a.ev));
    Some(out)
}

/// The defence bot's discard: the best-`EV` tile, or `chosen` (the stage bot's
/// pick) when it is within [`TIE`] of the best. `None` keeps `chosen` (no
/// `winner` head).
pub fn choose(
    state: &State,
    view: &SeatView,
    candidates: &[Tile],
    chosen: Tile,
    model: &BeliefModel,
) -> Option<Tile> {
    let terms = evaluate(state, view, candidates, model)?;
    let best = terms.first()?;
    let mine = terms.iter().find(|x| x.tile == chosen)?;
    Some(if best.ev > mine.ev + TIE {
        best.tile
    } else {
        chosen
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Agent, Bot};
    use mahjong_games::Policy;
    use mahjong_hk::RuleSet;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use std::path::Path;

    fn model_path() -> Option<String> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief-order.mjnn");
        p.exists().then(|| p.to_string_lossy().into_owned())
    }

    /// Plan Task 20: the terms add up, come best first, and `stage-ev` with a
    /// reading plays legal moves through whole hands; without a Task 19 file
    /// (`models/belief.mjnn` has no `winner` head) nothing is evaluated.
    #[test]
    fn terms_add_up_and_the_bot_plays_legal_hands() {
        let Some(path) = model_path() else {
            eprintln!("models/belief-order.mjnn missing: skipped");
            return;
        };
        let Ok(Agent::Bot(Bot::Stage(bot))) = Agent::parse(&format!("stage-ev:belief={path}"))
        else {
            panic!("stage-ev:belief= parses to a stage bot");
        };
        assert!(bot.defence && bot.reading != 0);
        let model = bot.reading_model().unwrap();
        let mut rng = SmallRng::seed_from_u64(20);
        let mut checked = 0;
        for _ in 0..3 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while !s.is_terminal() {
                let legal = s.legal_actions_vec();
                if !s.in_claim() && !s.in_rob() && checked < 20 {
                    let view = SeatView::of_current(&s);
                    let c: Vec<Tile> = legal
                        .iter()
                        .filter_map(|a| match a {
                            Action::Discard(t) => Some(*t),
                            _ => None,
                        })
                        .collect();
                    if !c.is_empty() {
                        let terms = evaluate(&s, &view, &c, &model).expect("a Task 19 file");
                        for w in terms.windows(2) {
                            assert!(w[0].ev >= w[1].ev);
                        }
                        for x in &terms {
                            let sum = x.win - x.deal_in - x.self_draw - x.liability;
                            assert!((x.ev - sum).abs() < 1e-9);
                            assert!((0.0..=1.0).contains(&x.my_win));
                        }
                        checked += 1;
                    }
                }
                let a = bot.choose(&s, &legal, &mut rng);
                assert!(legal.contains(&a));
                s.apply(a);
            }
        }
        assert!(checked > 0);
        let old = crate::BeliefModel::load(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/belief.mjnn"),
        );
        if let Ok(old) = old {
            let s = State::new_deal(RuleSet::default(), &mut rng);
            let view = SeatView::of_current(&s);
            let c: Vec<Tile> = view.concealed.kinds().collect();
            assert!(evaluate(&s, &view, &c, &old).is_none());
        }
    }

    /// Plan Task 20: `belief=` gives the stage bots a reading; under it the give-up
    /// rule and the learned choice use the tables refitted on it, and without it
    /// everything is as before.
    #[test]
    fn a_reading_switches_the_refitted_tables() {
        assert!(Agent::parse("stage-fold:belief=no/such/file.mjnn").is_err());
        assert!(Agent::parse("stage-fold:depth=2").is_err());
        assert!(Agent::parse("value:belief=x").is_err());
        assert!(crate::danger::reading().is_none());
        assert_eq!(giveup::calibration(), &giveup::CALIBRATION);
        assert_eq!(giveup::measured(), giveup::MEASURED);
        let Some(path) = model_path() else {
            eprintln!("models/belief-order.mjnn missing: skipped");
            return;
        };
        for name in ["stage-fold", "stage-fold-learned", "stage-ev"] {
            let Ok(Agent::Bot(Bot::Stage(b))) = Agent::parse(&format!("{name}:belief={path}"))
            else {
                panic!("{name}:belief= parses to a stage bot");
            };
            assert!(b.reading_model().is_some());
            assert_eq!(b.defence, name == "stage-ev");
        }
        let m = std::sync::Arc::new(BeliefModel::load(&path).unwrap());
        crate::danger::with_reading(Some(m), || {
            assert_eq!(giveup::calibration(), &giveup::READ_CALIBRATION);
            assert_eq!(giveup::loss_factor(), &giveup::READ_LOSS_FACTOR);
            assert_eq!(giveup::measured(), giveup::READ_MEASURED);
            let m = crate::pushfold::model(false);
            assert!(
                m.b3 == crate::pushfold::READ_MODEL.b3
                    && m.mean == crate::pushfold::READ_MODEL.mean
            );
        });
        assert!(crate::danger::reading().is_none());
        let m = crate::pushfold::model(false);
        assert!(
            m.b3 == crate::pushfold::TABLE_MODEL.b3 && m.mean == crate::pushfold::TABLE_MODEL.mean
        );
    }
}
