//! Reasons for a move, derived from the position (hand mathematics and the
//! danger model) and the solver's numbers. Every key here is a fact the code
//! checks; nothing is guessed. The keys and what each one means are listed in
//! `README.md` ("Reason keys").
//!
//! [`Study`] gathers the study-mode figures the lab UI shows (shanten, faan-aware
//! shanten and its target, tile danger, the discard table) once per position;
//! the hint returns them and the reasons read them.

use crate::shapes::{round2, round3, tile_id, tiles_to_json, Reason};
use mahjong_analysis::{
    discards, faan_shanten, shanten, DiscardInfo, FaanShanten, Structure, SuitPlan, Target,
    UNREACHABLE,
};
use mahjong_games::Game;
use mahjong_hk::{Action, Meld, MeldKind, State};
use mahjong_search::{ActionStats, Analysis, Danger, SeatView};
use mahjong_tiles::Tile;
use serde_json::{json, Map, Value};

/// A tile this safe (deal-in chance) is called safe when someone is a threat.
pub const SAFE_DANGER: f64 = 0.01;
/// A tile this dangerous is called dangerous.
pub const DANGEROUS: f64 = 0.05;
/// An opponent this likely to be waiting is named as a threat.
pub const THREAT: f64 = 0.4;
/// A deal-in rate this high in the solver's samples is named.
pub const HIGH_DEAL_IN: f64 = 0.15;

/// The study figures for one seat in one position.
pub struct Study {
    pub seat: usize,
    pub view: SeatView,
    pub danger: Danger,
    /// Plain shanten of the seat's hand as it stands.
    pub shanten: i8,
    /// Faan-aware shanten of the hand as it stands, with its target.
    pub faan: FaanShanten,
    /// The discard table: every discard with both measures and acceptance.
    /// Filled only when `seat` is to move in the turn phase (14 tiles).
    pub rows: Vec<DiscardInfo>,
}

impl Study {
    pub fn of(state: &State, seat: usize) -> Study {
        let view = SeatView::of(state, seat);
        let hv = view.hand_view();
        let turn_to_move = state.current_player() == Some(seat) && state.phase_name() == "turn";
        let rows = if turn_to_move {
            discards(&hv, &view.rules)
        } else {
            Vec::new()
        };
        Study {
            seat,
            danger: Danger::estimate(state, seat),
            shanten: shanten(&view.concealed, view.melds.len()),
            faan: faan_shanten(&hv, &view.rules),
            rows,
            view,
        }
    }

    fn row(&self, tile: Tile) -> Option<&DiscardInfo> {
        self.rows.iter().find(|r| r.tile == tile)
    }

    /// `{shanten, faan_shanten, target}` for the hint.
    pub fn progress_json(&self) -> Value {
        json!({
            "shanten": self.shanten,
            "faan_shanten": faan_shanten_json(self.faan.shanten),
            "target": self.faan.target.as_ref().map(target_json),
        })
    }

    /// `{tiles: {id: danger}, threat: [4]}`: deal-in chance of each tile in the
    /// hand and each seat's chance of waiting (0 for the seat itself).
    pub fn danger_json(&self) -> Value {
        let mut tiles = Map::new();
        for t in self.view.concealed.kinds() {
            tiles.insert(tile_id(t), json!(round3(self.danger.tile(t))));
        }
        json!({
            "tiles": tiles,
            "threat": (0..4).map(|p| if p == self.seat { 0.0 } else { round3(self.danger.threat(p)) }).collect::<Vec<_>>(),
        })
    }

    /// The discard table (empty unless the seat is to move in the turn phase).
    pub fn discards_json(&self) -> Vec<Value> {
        self.rows
            .iter()
            .map(|r| {
                json!({
                    "tile": tile_id(r.tile),
                    "shanten": r.shanten,
                    "waits": r.ukeire,
                    "faan_shanten": faan_shanten_json(r.faan_shanten),
                    "faan_waits": r.faan_ukeire,
                    "target": r.target.as_ref().map(target_json),
                    "danger": round3(self.danger.tile(r.tile)),
                })
            })
            .collect()
    }

    /// The faan-aware shanten after claiming the window's tile with `action`
    /// and then making the best discard; `None` if the claim is not a meld.
    fn faan_shanten_after_claim(&self, state: &State, action: Action) -> Option<i8> {
        let (tile, from) = state.claim_tile()?;
        let mut hand = self.view.concealed;
        let meld = match action {
            Action::Pung => {
                hand[tile] -= 2;
                Meld::exposed(MeldKind::Pung, tile, from)
            }
            Action::ClaimKong => {
                hand[tile] -= 3;
                Meld::exposed(MeldKind::Kong, tile, from)
            }
            Action::Chow(low) => {
                for k in 0..3 {
                    let t = low.offset(k).expect("valid chow");
                    if t != tile {
                        hand.remove(t);
                    }
                }
                Meld::exposed(MeldKind::Chow, low, from)
            }
            _ => return None,
        };
        let mut melds = self.view.melds.clone();
        melds.push(meld);
        if action == Action::ClaimKong {
            // A kong draws a replacement tile first: the hand keeps its count.
            let after = self.view.with(hand, melds);
            return Some(faan_shanten(&after.hand_view(), &after.rules).shanten);
        }
        let after = self.view.with(hand, melds);
        discards(&after.hand_view(), &after.rules)
            .iter()
            .map(|r| r.faan_shanten)
            .min()
    }
}

/// `UNREACHABLE` becomes `null`: no plan reaches the minimum.
fn faan_shanten_json(s: i8) -> Value {
    if s >= UNREACHABLE {
        Value::Null
    } else {
        json!(s)
    }
}

/// `{plan: {kind, suit?}, structure, pungs, faan, limit}`.
pub fn target_json(t: &Target) -> Value {
    let (kind, suit) = match t.plan {
        SuitPlan::Mixed => ("plan.mixed", None),
        SuitPlan::HalfFlush(s) => ("plan.half_flush", Some(s)),
        SuitPlan::FullFlush(s) => ("plan.full_flush", Some(s)),
        SuitPlan::Honours => ("plan.honours", None),
        SuitPlan::Orphans => ("plan.orphans", None),
    };
    let structure = match t.structure {
        Structure::Any => "structure.any",
        Structure::AllPungs => "structure.all_pungs",
        Structure::AllChows => "structure.all_chows",
    };
    json!({
        "plan": { "kind": kind, "suit": suit.map(|s| format!("suit.{}", crate::shapes::suit_name(s))) },
        "structure": structure,
        "pungs": tiles_to_json(&t.pungs),
        "faan": t.faan,
        "limit": t.limit,
    })
}

/// Reasons that follow from the position alone (before the move is made).
/// `legal` is the full legal-action list, so reasons can contrast the move with
/// its alternatives.
pub fn position_reasons(
    state: &State,
    study: &Study,
    action: Action,
    legal: &[Action],
) -> Vec<Reason> {
    let mut out = Vec::new();
    if legal.len() == 1 {
        out.push(Reason::new("reason.forced"));
        return out;
    }
    let seat = study.seat;
    let rules = state.rules();
    match action {
        Action::SelfDrawnWin => {
            if let Some(s) = state.self_drawn_score() {
                out.push(
                    Reason::new("reason.wins")
                        .arg("faan", s.faan)
                        .arg("points", 3 * rules.points(s.faan)),
                );
            }
        }
        Action::Win => {
            if let Some(s) = state.claim_score(seat) {
                out.push(
                    Reason::new("reason.wins")
                        .arg("faan", s.faan)
                        .arg("points", rules.points(s.faan)),
                );
            }
        }
        Action::Discard(t) => discard_reasons(study, t, &mut out),
        Action::Pass => {
            if legal.contains(&Action::Win) {
                out.push(Reason::new("reason.declines_win"));
            }
            // Would any claim bring the hand closer to a declarable win?
            let now = study.faan.shanten;
            let best_claim = legal
                .iter()
                .filter_map(|&a| study.faan_shanten_after_claim(state, a).map(|s| (s, a)))
                .min_by_key(|&(s, _)| s);
            match best_claim {
                Some((after, _)) if after < now => {
                    out.push(
                        Reason::new("reason.pass_misses_claim")
                            .arg("faan_shanten_now", faan_shanten_json(now))
                            .arg("faan_shanten_after", after),
                    );
                }
                Some(_) => out.push(Reason::new("reason.pass_keeps_hand")),
                None => {}
            }
            if study.view.melds.iter().all(|m| m.concealed) && state.in_claim() {
                out.push(Reason::new("reason.stays_concealed"));
            }
        }
        Action::Pung | Action::ClaimKong | Action::Chow(_) => {
            let now = study.faan.shanten;
            if let Some(after) = study.faan_shanten_after_claim(state, action) {
                if after < now {
                    out.push(
                        Reason::new("reason.claim_advances")
                            .arg("faan_shanten_now", faan_shanten_json(now))
                            .arg("faan_shanten_after", after),
                    );
                } else {
                    out.push(
                        Reason::new("reason.claim_no_gain")
                            .arg("faan_shanten_now", faan_shanten_json(now))
                            .arg("faan_shanten_after", faan_shanten_json(after)),
                    );
                }
            }
            if study.view.melds.iter().all(|m| m.concealed) {
                out.push(Reason::new("reason.opens_hand"));
            }
            if let (Action::Pung | Action::ClaimKong, Some((tile, _))) =
                (action, state.claim_tile())
            {
                scoring_pung(tile, seat, rules.prevailing_wind, &mut out);
            }
            if action == Action::ClaimKong {
                out.push(Reason::new("reason.kong_extra_draw"));
            }
        }
        Action::ConcealedKong(t) | Action::AddedKong(t) => {
            out.push(Reason::new("reason.kong_extra_draw"));
            scoring_pung(t, seat, rules.prevailing_wind, &mut out);
        }
    }
    out
}

/// A pung of a dragon, the seat wind or the prevailing wind scores faan (#10–#12).
fn scoring_pung(tile: Tile, seat: usize, prevailing: usize, out: &mut Vec<Reason>) {
    let mut faan = 0;
    if tile.is_dragon() {
        faan += 1;
    }
    if tile == Tile::wind(seat) {
        faan += 1;
    }
    if tile == Tile::wind(prevailing) {
        faan += 1;
    }
    if faan > 0 {
        out.push(Reason::new("reason.scoring_pung").arg("faan", faan));
    }
}

fn discard_reasons(study: &Study, tile: Tile, out: &mut Vec<Reason>) {
    let Some(row) = study.row(tile) else { return };
    let rows = &study.rows;
    let best_shanten = rows.iter().map(|r| r.shanten).min().unwrap_or(row.shanten);
    let best_faan = rows
        .iter()
        .map(|r| r.faan_shanten)
        .min()
        .unwrap_or(row.faan_shanten);

    if row.faan_shanten >= UNREACHABLE {
        out.push(Reason::new("reason.no_faan_path"));
    } else if row.faan_shanten == 0 {
        out.push(
            Reason::new("reason.ready_for_faan")
                .arg("faan", row.target.as_ref().map_or(0, |t| t.faan))
                .arg("waits", row.faan_ukeire),
        );
    } else if row.faan_shanten == best_faan {
        let mut r = Reason::new("reason.keeps_faan_path").arg("faan_shanten", row.faan_shanten);
        if let Some(t) = &row.target {
            r = r.arg("faan", t.faan).arg("target", target_json(t));
        }
        out.push(r);
    } else {
        out.push(
            Reason::new("reason.breaks_faan_path")
                .arg("faan_shanten", row.faan_shanten)
                .arg("best", best_faan),
        );
    }

    if row.shanten == 0 && row.faan_shanten != 0 {
        // Ready by tile count, but the hand cannot be declared yet.
        out.push(Reason::new("reason.ready_short_of_faan").arg("waits", row.ukeire));
    }
    if row.shanten == best_shanten {
        out.push(Reason::new("reason.best_shanten").arg("shanten", row.shanten));
    } else {
        out.push(
            Reason::new("reason.loses_shanten")
                .arg("shanten", row.shanten)
                .arg("best", best_shanten),
        );
    }

    // Among the discards that keep the same shanten, which accepts most tiles?
    let peers: Vec<&DiscardInfo> = rows.iter().filter(|r| r.shanten == row.shanten).collect();
    if peers.len() > 1 {
        let max_waits = peers.iter().map(|r| r.ukeire).max().unwrap_or(0);
        if row.ukeire == max_waits {
            out.push(Reason::new("reason.more_waits").arg("waits", row.ukeire));
        } else {
            out.push(
                Reason::new("reason.fewer_waits")
                    .arg("waits", row.ukeire)
                    .arg("best", max_waits),
            );
        }
    }

    let danger = study.danger.tile(tile);
    let threat = study.danger.max_threat();
    if danger >= DANGEROUS {
        out.push(Reason::new("reason.dangerous_tile").arg("danger", round3(danger)));
    } else if danger <= SAFE_DANGER && threat >= THREAT {
        out.push(Reason::new("reason.safe_tile").arg("danger", round3(danger)));
    }
    if let Some(p) = (0..4)
        .filter(|&p| p != study.seat && study.danger.threat(p) >= THREAT)
        .max_by(|&a, &b| study.danger.threat(a).total_cmp(&study.danger.threat(b)))
    {
        out.push(
            Reason::new("reason.opponent_threat")
                .arg("player", p)
                .arg("threat", round3(study.danger.threat(p))),
        );
    }
}

/// Standard errors a difference must reach to count as more than sampling noise.
pub const SIGNIFICANCE: f64 = 1.96;

/// Standard error of the difference between two candidates' means. Each
/// candidate's `diff_se` is against the baseline; against each other the two
/// add in quadrature (exact when one of them is the baseline, otherwise an
/// upper bound, since all candidates share the same samples).
pub fn pair_se(a: &ActionStats, b: &ActionStats) -> f64 {
    (a.diff_se * a.diff_se + b.diff_se * b.diff_se).sqrt()
}

/// Reasons from the solver's numbers: the margin of the best move over the
/// runner-up, how much a move loses against the best (each only when it is
/// more than sampling noise, else `reason.close_call`), and a high deal-in rate.
pub fn solver_reasons(this: &ActionStats, analysis: &Analysis) -> Vec<Reason> {
    let mut out = Vec::new();
    let best = &analysis.moves[0];
    // The best move is measured against the runner-up; any other move against the best.
    let other = if this.action == best.action {
        analysis.moves.get(1)
    } else {
        Some(best)
    };
    if let Some(other) = other {
        let gap = (best.mean - other.mean.min(this.mean)).max(0.0);
        let se = pair_se(this, other);
        let key = if gap <= SIGNIFICANCE * se {
            "reason.close_call"
        } else if this.action == best.action {
            "reason.score_margin"
        } else {
            "reason.score_loss"
        };
        out.push(
            Reason::new(key)
                .arg("points", round2(gap))
                .arg("se", round2(se)),
        );
    }
    if this.deal_in_rate >= HIGH_DEAL_IN {
        out.push(Reason::new("reason.high_deal_in").arg("rate", round3(this.deal_in_rate)));
    }
    out
}
