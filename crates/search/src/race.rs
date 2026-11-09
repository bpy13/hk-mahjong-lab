//! The race bot v0 (Brian's decisions B28), built as `docs/race-bot.md` specifies
//! (plan 2026-11-09 Task 6).
//!
//! Every candidate action is valued by a race over the rest of the hand: each
//! player's state chain (far, one away, ready, folded) and per-round win hazard,
//! competing risks `S(k + 1) = S(k) × (1 − Σ_i h_i(k))`, and my equity
//!
//! ```text
//! E(a) = Σ_k S(k) × [ my wins × what they pay me − Σ_o o's wins × what I pay o ]
//! ```
//!
//! (§3.3), maximised over my mode (push, detour, fold) against each of at most two
//! threats (§5.3). Opponents' estimates come from a [`Belief`] (the trained model with
//! `race:belief=models/belief.mjnn`, else [`HeuristicBelief`]); their states beyond
//! "ready" and their modes are the v0 heuristics of §2.1 and §5.2. My own hand is
//! exact: `faan_shanten`, acceptance and live winning tiles.
//!
//! Every number behind a candidate is kept in [`RaceTerms`] ([`RaceBot::explain`]).
//! The search is plain arithmetic: no threads, no randomness.
//!
//! Phase 4 (plan Task 8): a belief file with the race heads (`state`, `mode`;
//! `models/race-belief.mjnn`) replaces §2.1's one-away estimate by the `state` head and
//! §5.2's fold heuristic by the `mode` head's P(fold), through the fold hook
//! ([`RaceBot::fold_override`] still wins when set). The per-round steps `a`, `b` (per
//! wall bucket) and the feed rates `q(m)` are spec options, fitted from the logs by
//! `python -m mahjong.belief.race_fit`. The training labels of §7 are built here too:
//! [`true_state`], [`discard_label`] and [`LabelTracker`] (used by the `race_log` example;
//! the label tolerances are the `LABEL_*` constants, §12).

use crate::belief::BeliefModel;
use crate::bots::{after_claim, live_winning_tiles, rank_discards, Metric};
use crate::danger::{Danger, PROGRESS_BUCKETS};
use crate::ev::{after_own_kong, exposed, Belief, Beliefs, HeuristicBelief};
use crate::ev::{FAAN_BUCKETS, FIRST_FAAN_BUCKET};
use crate::view::SeatView;
use mahjong_analysis::{faan_shanten, HandView, UNREACHABLE};
use mahjong_games::Game;
use mahjong_hk::{Action, Event, RuleSet, State};
use mahjong_tiles::{Suit, Tile, TileCounts, NUM_KINDS};
use std::collections::HashMap;
use std::sync::Arc;

// ----- The v0 estimates (`docs/race-bot.md` §2, §5.2, §6). Phase 4 replaces them. -----

/// `P_clip` for `P_o(t)` (as `ev-bot.md`).
const P_CLIP: f64 = 0.95;
/// `a_o = b_o`: an opponent's chance per round to move F → O or O → R (§2.1); the
/// default of [`RaceConfig::opp_a`] and [`RaceConfig::opp_b`] in every wall bucket.
const OPP_STEP: f64 = 0.15;
/// `q(m)`: P(one discard by a player in mode `m` completes a given ready
/// opponent's hand), push / detour / fold (§2.3); the default of [`RaceConfig::q`].
const Q: [f64; 3] = [0.08, 0.04, 0.01];
/// §5.2: `π_fold = min(0.9, cost / 64) × {F: 1.0, O: 0.6, R: 0.2}`.
const FOLD_CAP: f64 = 0.9;
const FOLD_SCALE: f64 = 64.0;
const FOLD_BY_STATE: [f64; 3] = [1.0, 0.6, 0.2];
/// §5.2: `π_detour = 0.5 × (1 − π_fold)` in states O and R.
const DETOUR_SHARE: f64 = 0.5;
/// §5.2: `P̂(i wins) = tenpai_i × 0.6` for an opponent.
const OPP_WIN_GIVEN_READY: f64 = 0.6;
/// §6: a seat discards a tile its visible plan can use at 0.3 of the rate of one it
/// cannot use.
const USEFUL_FEED: f64 = 0.3;
/// §5.1: tiles allowed against a threat in detour and fold.
const DETOUR_SAFE: f64 = 0.05;
const FOLD_SAFE: f64 = 0.02;
/// Deviation (docs/race-bot.md §11): my hazard in detour, as a share of push,
/// instead of recomputing `b_me` and `L_me` on a detour hand.
const DETOUR_HAZARD: f64 = 0.5;
/// Deviation (§11): live winning copies once a far hand (faan-aware shanten ≥ 2)
/// becomes ready.
const FAR_LIVE: f64 = 4.0;

/// My mode against one threat (§5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Push,
    Detour,
    Fold,
}

const MODES: [Mode; 3] = [Mode::Push, Mode::Detour, Mode::Fold];

// ----- Hyper-parameters. -----

/// The `race:` spec keys and their defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct RaceConfig {
    /// Rounds valued (§5.3: `n` capped at 8).
    pub horizon: usize,
    /// Points below which two candidates tie; ties go to `rank_discards`, as in
    /// the `ev` bot (`ev-bot.md` §2.2).
    pub tie_margin: f64,
    /// `τ_threat`: an unfolded opponent with `p_O + p_R` above it gets a mode (§5.1).
    pub threat: f64,
    /// `a_o`: an opponent's chance per round to move far → one away, per wall bucket
    /// (`danger.rs`'s six progress buckets). Spec `a=x0/x1/x2/x3/x4/x5`.
    pub opp_a: [f64; PROGRESS_BUCKETS],
    /// `b_o`: one away → ready, per wall bucket. Spec `b=...`.
    pub opp_b: [f64; PROGRESS_BUCKETS],
    /// `q(m)` for push, detour, fold. Spec `q=push/detour/fold`.
    pub q: [f64; 3],
}

impl Default for RaceConfig {
    fn default() -> Self {
        RaceConfig {
            horizon: 8,
            tie_margin: 0.5,
            threat: 0.2,
            opp_a: [OPP_STEP; PROGRESS_BUCKETS],
            opp_b: [OPP_STEP; PROGRESS_BUCKETS],
            q: Q,
        }
    }
}

/// `n` probabilities separated by `/`.
fn probabilities<const N: usize>(k: &str, v: &str) -> Result<[f64; N], String> {
    let xs: Vec<f64> = v
        .split('/')
        .map(|x| x.parse::<f64>().ok().filter(|p| (0.0..=1.0).contains(p)))
        .collect::<Option<_>>()
        .ok_or_else(|| format!("race option {k}: {v:?} must be probabilities split by '/'"))?;
    xs.try_into()
        .map_err(|_| format!("race option {k} needs {N} values, got {v:?}"))
}

impl RaceConfig {
    /// Sets one `k=v` spec option. `belief` is handled by the caller.
    pub fn set(&mut self, k: &str, v: &str) -> Result<(), String> {
        let num = || {
            v.parse::<f64>()
                .ok()
                .filter(|x| x.is_finite() && *x >= 0.0)
                .ok_or_else(|| format!("bad number {v:?} for race option {k}"))
        };
        match k {
            "a" => return probabilities(k, v).map(|x| self.opp_a = x),
            "b" => return probabilities(k, v).map(|x| self.opp_b = x),
            "q" => return probabilities(k, v).map(|x| self.q = x),
            _ => {}
        }
        match k {
            "horizon" => {
                self.horizon = v
                    .parse::<usize>()
                    .ok()
                    .filter(|&h| h >= 1)
                    .ok_or_else(|| format!("bad horizon {v:?}"))?
            }
            "tie_margin" => self.tie_margin = num()?,
            "threat" => self.threat = num()?.min(1.0),
            _ => return Err(format!("unknown race option {k:?}")),
        }
        Ok(())
    }
}

// ----- Output. -----

/// Every number the bot used for one candidate. Arrays are per seat; the acting
/// seat's opponent entries are 0.
#[derive(Clone, Debug, PartialEq)]
pub struct RaceTerms {
    pub action: Action,
    /// The tile discarded now (for a claim, the best discard after it); `None` for
    /// a pass or a kong.
    pub discard: Option<Tile>,
    /// `E(a)`: points for the acting seat, under the best mode vector.
    pub equity: f64,
    /// The acting seat.
    pub me: usize,
    /// My mode against each threat under the best vector; `None` for seats that
    /// are not threats (and for me).
    pub modes: [Option<Mode>; 4],
    /// `E(a, M)` for every mode vector tried, in enumeration order (push first).
    pub by_modes: Vec<([Option<Mode>; 4], f64)>,
    /// Rounds valued (`n`, capped by the horizon).
    pub rounds: usize,
    // My hand after the action (§4.3).
    /// Faan-aware shanten (`UNREACHABLE` without a target).
    pub faan_shanten: i8,
    /// `f_T`, the target's guaranteed faan.
    pub target_faan: u8,
    /// Points per share of my discard win and of my self-draw.
    pub pay_me: f64,
    pub pay_sd_me: f64,
    /// Shares paid to me on a discard win and on a self-draw (3 and 3 when a seat
    /// is liable for me).
    pub shares_disc: f64,
    pub shares_sd: f64,
    /// My live winning tiles (ready hands only).
    pub waits: Vec<Tile>,
    /// `L_me`: live copies of my waits among the unseen tiles.
    pub live: f64,
    /// `W_me`: of those, the expected copies in the wall.
    pub wall_live: f64,
    /// `H_{me,j}`: the expected copies in seat `j`'s hand.
    pub held: [f64; 4],
    /// Unseen copies that lower my faan-aware shanten (not ready).
    pub acceptance: f64,
    /// Expected live copies once I become ready (not ready).
    pub live_when_ready: f64,
    /// My discards planned for the later rounds before the per-round rates take
    /// over (§4.4), under the best mode vector.
    pub plan: Vec<Tile>,
    // The opponents (§2, §5.2).
    pub tenpai: [f64; 4],
    pub one_away: [f64; 4],
    pub pay: [f64; 4],
    pub pay_sd: [f64; 4],
    /// `d_o`: an opponent's self-draw chance per draw when ready.
    pub draw_hazard: [f64; 4],
    /// `π_fold(o)` and `π_detour(o)`.
    pub fold: [f64; 4],
    pub detour: [f64; 4],
    /// `q_j`: seat `j`'s feed rate to a ready opponent, averaged over its modes.
    pub q: [f64; 4],
    /// `φ_{j→me}`: P(seat `j` discards a live tile of mine it holds or draws).
    pub phi: [f64; 4],
    /// `c_o(t)`: the chance the tile now lets `o` claim (§4.2).
    pub claim: [f64; 4],
    /// My discard now: `P_o(t) × (1 − fold_R,o) × Pay(o) × L_disc(o)` per seat.
    pub deal_in_now: [f64; 4],
    // The race (§3), under the best mode vector.
    /// What I collect: `Σ_k S(k) × my hazard × payment`.
    pub gain: f64,
    /// What I pay for opponents' self-draws, wins on my discards (now included)
    /// and wins on others' discards (only when I am liable).
    pub loss_self_draw: f64,
    pub loss_my_feeds: f64,
    pub loss_other_feeds: f64,
    /// P(each seat wins within the horizon), me included.
    pub p_win: [f64; 4],
    /// P(nobody wins within the horizon).
    pub p_draw: f64,
}

/// The bot's decision with every candidate.
#[derive(Clone, Debug, PartialEq)]
pub struct RaceDecision {
    pub action: Action,
    /// Every candidate valued, best `equity` first (empty when a win was declared).
    pub candidates: Vec<RaceTerms>,
}

// ----- The bot. -----

/// The race bot (see the module docs).
#[derive(Clone, Debug)]
pub struct RaceBot {
    pub config: RaceConfig,
    pub belief: Arc<dyn Belief>,
    /// A fixed fold probability per seat that replaces §5.2's heuristic (in every
    /// state). Tests set it; when it is `None` for a seat and [`RaceBot::heads`] is
    /// present, the `mode` head's P(fold) for that seat fills it at each decision.
    pub fold_override: [Option<f64>; 4],
    /// The belief file's race heads (`state`, `mode`), when it has them: the same
    /// model as `belief`, so one network pass (cached) serves both.
    pub heads: Option<Arc<BeliefModel>>,
}

impl Default for RaceBot {
    fn default() -> Self {
        RaceBot::new(RaceConfig::default(), Arc::new(HeuristicBelief))
    }
}

/// Builds the bot of a `race:` spec's options (`belief=` loads a model; a file with
/// the race heads also feeds [`RaceBot::heads`]).
pub fn parse_spec(opts: &[(&str, &str)]) -> Result<RaceBot, String> {
    let mut config = RaceConfig::default();
    let mut model: Option<Arc<BeliefModel>> = None;
    for &(k, v) in opts {
        if k == "belief" {
            if !std::path::Path::new(v).exists() {
                return Err(format!("belief model {v:?} not found"));
            }
            model = Some(Arc::new(BeliefModel::load(v)?));
        } else {
            config.set(k, v)?;
        }
    }
    Ok(match model {
        Some(m) => {
            let heads = m.has_race_heads().then(|| Arc::clone(&m));
            RaceBot {
                heads,
                ..RaceBot::new(config, m)
            }
        }
        None => RaceBot::new(config, Arc::new(HeuristicBelief)),
    })
}

/// Who draws in which round (§1, `n_i`): rounds end with my draw, so in round 1
/// only the seats between the next drawer and me act before my first draw.
struct Schedule {
    /// `acts[k][i]`: seat `i` draws (and then discards) in round `k` (0-based).
    acts: Vec<[bool; 4]>,
    /// The first round each seat acts in.
    first: [Option<usize>; 4],
}

impl Schedule {
    fn new(me: usize, next: usize, wall: usize, horizon: usize) -> Schedule {
        let mut acts = vec![[false; 4]; horizon];
        let mut first = [None; 4];
        let (mut seat, mut round) = (next, 0usize);
        for _ in 0..wall {
            acts[round][seat] = true;
            first[seat].get_or_insert(round);
            if seat == me {
                round += 1;
                if round == horizon {
                    break;
                }
            }
            seat = (seat + 1) % 4;
        }
        acts.retain(|r| r.iter().any(|&a| a));
        Schedule { acts, first }
    }
}

/// Everything that is the same for every candidate of one decision.
struct Table {
    me: usize,
    rules: RuleSet,
    wall: f64,
    /// `U`: tiles I have not seen.
    pool: f64,
    /// For each seat: P(ready), P(one away), P(far) (§2.1).
    states: [[f64; 3]; 4],
    pay: [f64; 4],
    pay_sd: [f64; 4],
    /// `wait_o(t)` and `P_o(t) = min(tenpai × wait, 0.95)`.
    wait: [[f64; NUM_KINDS]; 4],
    p: [[f64; NUM_KINDS]; 4],
    mean_count: [[f64; NUM_KINDS]; 4],
    has_one: [[f64; NUM_KINDS]; 4],
    has_two: [[f64; NUM_KINDS]; 4],
    /// `d_o = W_o / wall`.
    draw_hazard: [f64; 4],
    /// Share multipliers before any new liability (`ev-bot.md` §1).
    l_disc: [f64; 4],
    l_sd: [f64; 4],
    l_other: [f64; 4],
    /// Seats with three exposed melds and nobody liable (a claim makes me liable).
    liab_seat: [bool; 4],
    /// The seat liable for me, if any.
    liable_me: Option<usize>,
    /// Who is liable for each opponent (for `cost_o`).
    liable_for: [Option<usize>; 4],
    /// A seat's visible flush suit (some exposed meld, every suited meld in one suit).
    flush: [Option<Suit>; 4],
    /// Honours each seat has discarded.
    discarded: [[bool; NUM_KINDS]; 4],
    horizon: usize,
    threat: f64,
    fold_override: [Option<f64>; 4],
    /// `a_o`, `b_o` in this decision's wall bucket, and `q(m)`.
    step_far: f64,
    step_one: f64,
    q: [f64; 3],
}

/// `danger.rs`'s wall-progress bucket (0 = early … 5 = late).
fn wall_bucket(state: &State) -> usize {
    let initial = if state.rules().flowers { 91.0 } else { 83.0 };
    let progress = (1.0 - state.wall_remaining() as f64 / initial).clamp(0.0, 0.999);
    (progress * PROGRESS_BUCKETS as f64) as usize
}

fn points(rules: &RuleSet, faan: u8) -> f64 {
    rules.points(faan.min(rules.faan_cap)) as f64
}

impl Table {
    /// `fold` replaces §5.2's heuristic per seat and `one_away` §2.1's estimate (the
    /// race heads, [`RaceBot::head_estimates`]).
    fn new(
        state: &State,
        me: usize,
        beliefs: &Beliefs,
        bot: &RaceBot,
        fold: [Option<f64>; 4],
        one_away: [Option<f64>; 4],
    ) -> Table {
        let rules = *state.rules();
        let unseen = state.unseen(me);
        let wall = state.wall_remaining() as f64;
        let bucket = wall_bucket(state);
        let mut t = Table {
            me,
            rules,
            wall,
            pool: unseen.total().max(1) as f64,
            states: [[0.0; 3]; 4],
            pay: [0.0; 4],
            pay_sd: [0.0; 4],
            wait: [[0.0; NUM_KINDS]; 4],
            p: [[0.0; NUM_KINDS]; 4],
            mean_count: [[0.0; NUM_KINDS]; 4],
            has_one: [[0.0; NUM_KINDS]; 4],
            has_two: [[0.0; NUM_KINDS]; 4],
            draw_hazard: [0.0; 4],
            l_disc: [0.0; 4],
            l_sd: [0.0; 4],
            l_other: [0.0; 4],
            liab_seat: [false; 4],
            liable_me: state.liable_for(me),
            liable_for: std::array::from_fn(|o| state.liable_for(o)),
            flush: [None; 4],
            discarded: [[false; NUM_KINDS]; 4],
            horizon: bot.config.horizon,
            threat: bot.config.threat,
            fold_override: fold,
            step_far: bot.config.opp_a[bucket],
            step_one: bot.config.opp_b[bucket],
            q: bot.config.q,
        };
        for e in state.events() {
            if let Event::Discard { seat, tile } = *e {
                t.discarded[seat as usize][tile.index() as usize] = true;
            }
        }
        for o in (0..4).filter(|&o| o != me) {
            let b = &beliefs.seats[o];
            let ready = b.tenpai.clamp(0.0, 1.0);
            let one_away = one_away[o].unwrap_or(1.5 * ready).min(1.0 - ready);
            t.states[o] = [ready, one_away, 1.0 - ready - one_away];
            for (bucket, &pb) in b.faan.iter().enumerate().take(FAAN_BUCKETS) {
                let f = FIRST_FAAN_BUCKET + bucket as u8;
                t.pay[o] += pb * points(&rules, f);
                t.pay_sd[o] += pb * points(&rules, (f + 1).min(10));
            }
            (t.l_disc[o], t.l_sd[o], t.l_other[o]) = match state.liable_for(o) {
                None => (1.0, 1.0, 0.0),
                Some(l) if l == me => (3.0, 3.0, 3.0),
                Some(_) => (0.0, 0.0, 0.0),
            };
            let melds = state.melds(o);
            t.liab_seat[o] = rules.twelve_tile_liability
                && melds.exposed_count() == 3
                && state.liable_for(o).is_none();
            let mut suits = melds.iter().filter_map(|m| m.tile.suit());
            if melds.exposed_count() > 0 {
                if let Some(first) = suits.next() {
                    t.flush[o] = suits.all(|s| s == first).then_some(first);
                }
            }
            t.wait[o] = b.wait;
            t.mean_count[o] = b.mean_count;
            t.has_one[o] = b.has_one;
            t.has_two[o] = b.has_two;
            for i in 0..NUM_KINDS {
                t.p[o][i] = (b.tenpai * b.wait[i]).min(P_CLIP);
            }
        }
        // `W_o`: o's winning copies expected in the wall (§2.2), the unseen copies
        // less those the belief puts in any hidden hand (o's own included).
        for o in (0..4).filter(|&o| o != me) {
            let mut w = 0.0;
            for k in Tile::all_kinds() {
                let i = k.index() as usize;
                let in_hands: f64 = (0..4)
                    .filter(|&j| j != me)
                    .map(|j| t.mean_count[j][i])
                    .sum();
                w += t.wait[o][i] * (unseen[k] as f64 - in_hands).max(0.0);
            }
            t.draw_hazard[o] = if wall > 0.0 { (w / wall).min(0.5) } else { 0.0 };
        }
        t
    }

    fn opponents(&self) -> impl Iterator<Item = usize> + '_ {
        (0..4).filter(move |&o| o != self.me)
    }

    /// `P_claim_o(t)`: o can pung `t`, or chow it when o is my next seat (§4.2,
    /// the same as `ev-bot.md`'s `P_claim4_o`).
    fn p_claim(&self, o: usize, t: Tile) -> f64 {
        let mut chow = 0.0f64;
        if o == (self.me + 1) % 4 {
            for (a, b) in [(-2i8, -1i8), (-1, 1), (1, 2)] {
                if let (Some(u), Some(v)) = (t.offset(a), t.offset(b)) {
                    chow = chow.max(
                        self.has_one[o][u.index() as usize] * self.has_one[o][v.index() as usize],
                    );
                }
            }
        }
        (self.has_two[o][t.index() as usize] + chow).min(1.0)
    }

    /// Whether seat `j`'s visible plan cannot use `w` (§6): another suit than its
    /// flush, or an honour it has discarded.
    fn useless_to(&self, j: usize, w: Tile) -> bool {
        match (w.suit(), self.flush[j]) {
            (Some(s), Some(f)) => s != f,
            (None, _) => self.discarded[j][w.index() as usize],
            _ => false,
        }
    }
}

/// My hand after a candidate action (§4.3).
struct Mine {
    view: SeatView,
    shanten: i8,
    target_faan: u8,
    pay: f64,
    pay_sd: f64,
    shares_disc: f64,
    shares_sd: f64,
    /// The seat that pays everything when I win (liable for me), if any.
    payer: Option<usize>,
    waits: Vec<Tile>,
    live: f64,
    held: [f64; 4],
    wall_live: f64,
    acceptance: f64,
    live_when_ready: f64,
    /// The kinds that would finish (ready) or advance (not ready) my hand,
    /// weighted by unseen copies: for `φ`'s usefulness.
    targets: Vec<(Tile, f64)>,
    /// One copy of each kind I could throw later without losing ground (§4.4's
    /// push plan); empty after a pass or a kong.
    useless: Vec<Tile>,
}

/// One candidate's inputs before the modes.
struct Candidate {
    action: Action,
    tile: Option<Tile>,
    next: usize,
    mine: Mine,
}

/// Per decision cache: the live copies of a ready 13-tile hand (none when not
/// ready), for `live_when_ready`.
type ReadyCache = HashMap<TileCounts, Option<f64>>;

fn fs_of(view: &SeatView, hand: &TileCounts) -> mahjong_analysis::FaanShanten {
    faan_shanten(
        &HandView {
            concealed: hand,
            ..view.hand_view()
        },
        &view.rules,
    )
}

/// The live copies of `hand` if it is ready (faan-aware shanten 0).
fn ready_live(view: &SeatView, hand: &TileCounts, cache: &mut ReadyCache) -> Option<f64> {
    *cache.entry(*hand).or_insert_with(|| {
        (fs_of(view, hand).shanten == 0).then(|| {
            live_winning_tiles(view, hand)
                .iter()
                .map(|&(w, _)| view.unseen[w] as f64)
                .sum()
        })
    })
}

impl Mine {
    /// `after` is my hand after the action (13 tiles less 3 per meld);
    /// `useless` the later-discard candidates.
    fn new(
        t: &Table,
        after: SeatView,
        payer: Option<usize>,
        useless: Vec<Tile>,
        cache: &mut ReadyCache,
    ) -> Mine {
        let rules = &t.rules;
        let fs = fs_of(&after, &after.concealed);
        let target_faan = fs.target.as_ref().map_or(0, |x| x.faan);
        let (shares_disc, shares_sd) = if payer.is_some() {
            (3.0, 3.0)
        } else {
            (1.0, 3.0)
        };
        let mut m = Mine {
            shanten: fs.shanten,
            target_faan,
            pay: points(rules, target_faan.max(rules.min_faan)),
            pay_sd: points(rules, target_faan.max(rules.min_faan) + 1),
            shares_disc,
            shares_sd,
            payer,
            waits: Vec::new(),
            live: 0.0,
            held: [0.0; 4],
            wall_live: 0.0,
            acceptance: 0.0,
            live_when_ready: 0.0,
            targets: Vec::new(),
            useless,
            view: after,
        };
        let view = &m.view;
        if fs.shanten == 0 {
            let wins = live_winning_tiles(view, &view.concealed);
            let (mut pay, mut pay_sd) = (0.0, 0.0);
            for &(w, f) in &wins {
                let n = view.unseen[w] as f64;
                m.live += n;
                pay += n * points(rules, f);
                pay_sd += n * points(rules, f + 1);
                m.targets.push((w, n));
                for j in t.opponents() {
                    m.held[j] += t.mean_count[j][w.index() as usize];
                }
            }
            m.waits = wins.iter().map(|&(w, _)| w).collect();
            if m.live > 0.0 {
                m.pay = pay / m.live;
                m.pay_sd = pay_sd / m.live;
            }
            let in_hands: f64 = m.held.iter().sum();
            if in_hands > m.live {
                m.held.iter_mut().for_each(|h| *h *= m.live / in_hands);
            }
            m.wall_live = (m.live - m.held.iter().sum::<f64>()).max(0.0);
        } else if fs.shanten < UNREACHABLE {
            // Acceptance, the faan the accepting draws reach, and (one away) the live
            // copies of the waits each accepting draw leaves.
            let (mut pay, mut pay_sd, mut live) = (0.0, 0.0, 0.0);
            let mut h = view.concealed;
            for k in Tile::all_kinds() {
                if view.unseen[k] == 0 || h[k] >= 4 {
                    continue;
                }
                h.add(k);
                let next = fs_of(view, &h);
                if next.shanten < fs.shanten {
                    let n = view.unseen[k] as f64;
                    let f = next
                        .target
                        .as_ref()
                        .map_or(target_faan, |x| x.faan)
                        .max(rules.min_faan);
                    m.acceptance += n;
                    pay += n * points(rules, f);
                    pay_sd += n * points(rules, f + 1);
                    m.targets.push((k, n));
                    if fs.shanten == 1 {
                        let best = h
                            .kinds()
                            .filter_map(|d| {
                                let mut r = h;
                                r.remove(d);
                                ready_live(view, &r, cache)
                            })
                            .fold(0.0, f64::max);
                        live += n * best;
                    }
                }
                h.remove(k);
            }
            if m.acceptance > 0.0 {
                m.pay = pay / m.acceptance;
                m.pay_sd = pay_sd / m.acceptance;
                m.live_when_ready = if fs.shanten == 1 {
                    live / m.acceptance
                } else {
                    FAR_LIVE
                };
            }
        }
        m
    }
}

/// The modes of the opponents for one candidate (§5.2) and the feed rates that
/// follow from them.
struct Opponents {
    /// Fold probability by state (R, O, F) and overall.
    fold_by_state: [[f64; 3]; 4],
    fold: [f64; 4],
    detour: [f64; 4],
    q: [f64; 4],
    phi: [f64; 4],
}

/// One race (§3) for a candidate under a mode vector.
#[derive(Default)]
struct Outcome {
    equity: f64,
    gain: f64,
    loss_sd: f64,
    loss_mine: f64,
    loss_other: f64,
    p_win: [f64; 4],
    p_draw: f64,
    deal_in_now: [f64; 4],
    plan: Vec<Tile>,
}

impl RaceBot {
    /// A race bot with the given hyper-parameters and belief model.
    pub fn new(config: RaceConfig, belief: Arc<dyn Belief>) -> RaceBot {
        RaceBot {
            config,
            belief,
            fold_override: [None; 4],
            heads: None,
        }
    }

    /// Per seat: the fold probability (the fixed override, else the `mode` head's
    /// P(fold)) and the `state` head's P(one away), for the player to act.
    fn head_estimates(&self, state: &State, me: usize) -> ([Option<f64>; 4], [Option<f64>; 4]) {
        let mut fold = self.fold_override;
        let mut one_away = [None; 4];
        if let Some(m) = self
            .heads
            .as_ref()
            .filter(|_| state.current_player() == Some(me))
        {
            let out = m.run(state);
            for o in (0..4).filter(|&o| o != me) {
                let Some(b) = out.seat(o) else { continue };
                if fold[o].is_none() {
                    fold[o] = b.mode.map(|p| p[2]);
                }
                one_away[o] = b.state.map(|p| p[1]);
            }
        }
        (fold, one_away)
    }

    /// Every candidate with its terms, best equity first. Empty when a win is legal
    /// (it is always taken).
    pub fn explain(&self, state: &State) -> Vec<RaceTerms> {
        self.decide(state).candidates
    }

    /// The chosen action and every candidate.
    ///
    /// # Panics
    /// Panics if the hand is over.
    pub fn decide(&self, state: &State) -> RaceDecision {
        let legal = state.legal_actions_vec();
        self.decide_among(state, &legal)
    }

    fn decide_among(&self, state: &State, legal: &[Action]) -> RaceDecision {
        if let Some(&w) = legal
            .iter()
            .find(|a| matches!(a, Action::Win | Action::SelfDrawnWin))
        {
            return RaceDecision {
                action: w,
                candidates: Vec::new(),
            };
        }
        let me = state.current_player().expect("the hand is not over");
        let beliefs = self.belief.estimate(state, me);
        let (fold, one_away) = self.head_estimates(state, me);
        let table = Table::new(state, me, &beliefs, self, fold, one_away);
        let view = SeatView::of(state, me);
        let wall = state.wall_remaining();
        let mut cache = ReadyCache::new();
        let mut cands: Vec<Candidate> = Vec::new();
        let liability_rule = state.rules().twelve_tile_liability;

        if state.in_claim() || state.in_rob() {
            let claim = state.claim_tile();
            let next = claim.map_or((me + 1) % 4, |(_, from)| (from + 1) % 4);
            let mine = Mine::new(
                &table,
                view.clone(),
                table.liable_me,
                Vec::new(),
                &mut cache,
            );
            cands.push(Candidate {
                action: Action::Pass,
                tile: None,
                next,
                mine,
            });
            if let Some((tile, from)) = claim.filter(|_| state.in_claim()) {
                for &a in legal {
                    let Some(after) = after_claim(&view, a, tile, from) else {
                        continue;
                    };
                    let four = liability_rule && exposed(&after.melds) == 4;
                    let payer = if four { Some(from) } else { table.liable_me };
                    if a == Action::ClaimKong {
                        // The replacement draw is unknown and adds nothing (§4.5).
                        let mine = Mine::new(&table, after, payer, Vec::new(), &mut cache);
                        cands.push(Candidate {
                            action: a,
                            tile: None,
                            next: (me + 1) % 4,
                            mine,
                        });
                    } else {
                        for (d, useless) in discards_of(&after) {
                            let mut h = after.concealed;
                            h.remove(d);
                            let after_d = after.with(h, after.melds.clone());
                            let mine = Mine::new(&table, after_d, payer, useless, &mut cache);
                            cands.push(Candidate {
                                action: a,
                                tile: Some(d),
                                next: (me + 1) % 4,
                                mine,
                            });
                        }
                    }
                }
            }
        } else {
            let disc = discards_of(&view);
            for &a in legal {
                match a {
                    Action::Discard(t) => {
                        let useless = disc
                            .iter()
                            .find(|(d, _)| *d == t)
                            .map_or_else(Vec::new, |(_, u)| u.clone());
                        let mut h = view.concealed;
                        h.remove(t);
                        let after = view.with(h, view.melds.clone());
                        let mine = Mine::new(&table, after, table.liable_me, useless, &mut cache);
                        cands.push(Candidate {
                            action: a,
                            tile: Some(t),
                            next: (me + 1) % 4,
                            mine,
                        });
                    }
                    Action::ConcealedKong(_) | Action::AddedKong(_) => {
                        if let Some(after) = after_own_kong(&view, a) {
                            let mine =
                                Mine::new(&table, after, table.liable_me, Vec::new(), &mut cache);
                            cands.push(Candidate {
                                action: a,
                                tile: None,
                                next: (me + 1) % 4,
                                mine,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut terms: Vec<RaceTerms> = cands
            .into_iter()
            .map(|c| self.value(&table, c, wall))
            .collect();
        // A claim is one candidate: its best discard.
        terms.sort_by(|a, b| b.equity.total_cmp(&a.equity).then(a.action.cmp(&b.action)));
        let mut seen: Vec<Action> = Vec::new();
        terms.retain(|c| {
            let first = !seen.contains(&c.action);
            seen.push(c.action);
            first
        });

        let margin = self.config.tie_margin;
        let eq = |a: Action| {
            terms
                .iter()
                .find(|c| c.action == a)
                .map_or(f64::NEG_INFINITY, |c| c.equity)
        };
        let action = if state.in_claim() || state.in_rob() {
            let pass = eq(Action::Pass);
            match terms.iter().find(|c| c.action != Action::Pass) {
                Some(c) if c.equity > pass + margin => c.action,
                _ => Action::Pass,
            }
        } else {
            let discards: Vec<Tile> = terms
                .iter()
                .filter_map(|c| match c.action {
                    Action::Discard(t) => Some(t),
                    _ => None,
                })
                .collect();
            let best = discards
                .iter()
                .map(|&t| eq(Action::Discard(t)))
                .fold(f64::NEG_INFINITY, f64::max);
            let tied: Vec<Tile> = discards
                .iter()
                .copied()
                .filter(|&t| eq(Action::Discard(t)) >= best - margin)
                .collect();
            let mut chosen = if tied.is_empty() {
                legal[0]
            } else {
                Action::Discard(rank_discards(&view, &tied, Metric::Faan)[0].0)
            };
            if let Some(k) = terms
                .iter()
                .find(|c| matches!(c.action, Action::ConcealedKong(_) | Action::AddedKong(_)))
            {
                if k.equity > best + margin {
                    chosen = k.action;
                }
            }
            chosen
        };
        RaceDecision {
            action,
            candidates: terms,
        }
    }

    /// Values one candidate: the opponents' modes, then the race under every mode
    /// vector over my threats; keeps the best.
    fn value(&self, t: &Table, c: Candidate, wall: usize) -> RaceTerms {
        let me = t.me;
        let sched = Schedule::new(me, c.next, wall, t.horizon);
        let mine = &c.mine;
        // `c_o(t)`: the tile now feeds o a claim (§4.2).
        let mut claim = [0.0; 4];
        if let Some(x) = c.tile {
            for o in t.opponents() {
                claim[o] = self.claim_chance(t, o, x);
            }
        }
        let opp = self.opponents(t, mine, &sched);
        let threats = {
            let mut v: Vec<usize> = t
                .opponents()
                .filter(|&o| {
                    let s = t.states[o];
                    s[0] + s[1] > t.threat && opp.fold[o] < 0.5
                })
                .collect();
            v.sort_by(|&a, &b| {
                let s = |o: usize| t.states[o][0] + t.states[o][1];
                s(b).total_cmp(&s(a)).then(a.cmp(&b))
            });
            v.truncate(2);
            v
        };
        let mut by_modes = Vec::new();
        let mut best: Option<(Outcome, [Option<Mode>; 4])> = None;
        for n in 0..3usize.pow(threats.len() as u32) {
            let mut modes = [None; 4];
            let mut rest = n;
            for &o in &threats {
                modes[o] = Some(MODES[rest % 3]);
                rest /= 3;
            }
            let out = self.race(t, &c, &sched, &opp, &claim, &modes);
            by_modes.push((modes, out.equity));
            // Equal equities go to the less defensive vector (fewer folds, then
            // fewer detours): a fold is chosen only where it gains something.
            let caution = |m: &[Option<Mode>; 4]| {
                let n = |x: Mode| m.iter().filter(|&&y| y == Some(x)).count();
                (n(Mode::Fold), n(Mode::Detour))
            };
            let better = best.as_ref().is_none_or(|(b, bm)| {
                out.equity > b.equity + 1e-9
                    || (out.equity > b.equity - 1e-9 && caution(&modes) < caution(bm))
            });
            if better {
                best = Some((out, modes));
            }
        }
        let (out, modes) = best.expect("at least one mode vector");
        RaceTerms {
            action: c.action,
            discard: c.tile,
            equity: out.equity,
            me,
            modes,
            by_modes,
            rounds: sched.acts.len(),
            faan_shanten: mine.shanten,
            target_faan: mine.target_faan,
            pay_me: mine.pay,
            pay_sd_me: mine.pay_sd,
            shares_disc: mine.shares_disc,
            shares_sd: mine.shares_sd,
            waits: mine.waits.clone(),
            live: mine.live,
            wall_live: mine.wall_live,
            held: mine.held,
            acceptance: mine.acceptance,
            live_when_ready: mine.live_when_ready,
            plan: out.plan,
            tenpai: std::array::from_fn(|o| t.states[o][0]),
            one_away: std::array::from_fn(|o| t.states[o][1]),
            pay: t.pay,
            pay_sd: t.pay_sd,
            draw_hazard: t.draw_hazard,
            fold: opp.fold,
            detour: opp.detour,
            q: opp.q,
            phi: opp.phi,
            claim,
            deal_in_now: out.deal_in_now,
            gain: out.gain,
            loss_self_draw: out.loss_sd,
            loss_my_feeds: out.loss_mine,
            loss_other_feeds: out.loss_other,
            p_win: out.p_win,
            p_draw: out.p_draw,
        }
    }

    /// `c_o(t) = P_claim_o(t) × (1 − P_o(t))` when o can claim the tile at all.
    fn claim_chance(&self, t: &Table, o: usize, x: Tile) -> f64 {
        t.p_claim(o, x) * (1.0 - t.p[o][x.index() as usize])
    }

    /// §5.2 and §6: each opponent's fold and detour chances given what the others
    /// (me with this candidate's hand included) would cost it, and the feed rates.
    fn opponents(&self, t: &Table, mine: &Mine, sched: &Schedule) -> Opponents {
        // φ before folds: how useful my targets are to each seat.
        let mut usefulness = [0.0; 4];
        let weight: f64 = mine.targets.iter().map(|x| x.1).sum();
        for j in t.opponents() {
            usefulness[j] = if weight > 0.0 {
                mine.targets
                    .iter()
                    .map(|&(w, n)| n * if t.useless_to(j, w) { 1.0 } else { USEFUL_FEED })
                    .sum::<f64>()
                    / weight
            } else {
                USEFUL_FEED
            };
        }
        // My one-pass win chance (`P̂(me)`), split by who feeds me.
        let (sd_me, fed_by) = my_win_estimate(t, mine, sched, &usefulness);
        let mut out = Opponents {
            fold_by_state: [[0.0; 3]; 4],
            fold: [0.0; 4],
            detour: [0.0; 4],
            q: [0.0; 4],
            phi: [0.0; 4],
        };
        for o in t.opponents() {
            let fold_by_state = match t.fold_override[o] {
                Some(f) => [f.clamp(0.0, 1.0); 3],
                None => {
                    let mut cost = 0.0;
                    for i in t.opponents().filter(|&i| i != o) {
                        let share = match t.liable_for[i] {
                            None => 1.0,
                            Some(l) if l == o => 3.0,
                            Some(_) => 0.0,
                        };
                        cost += t.states[i][0] * OPP_WIN_GIVEN_READY * t.pay[i] * share;
                    }
                    // What my win would cost o.
                    cost += match mine.payer {
                        Some(p) if p == o => {
                            3.0 * (sd_me * mine.pay_sd + fed_by.iter().sum::<f64>() * mine.pay)
                        }
                        Some(_) => 0.0,
                        None => sd_me * mine.pay_sd + fed_by[o] * mine.pay,
                    };
                    let f = (cost / FOLD_SCALE).min(FOLD_CAP);
                    // R, O, F order in `states`; FOLD_BY_STATE is F, O, R.
                    [
                        f * FOLD_BY_STATE[2],
                        f * FOLD_BY_STATE[1],
                        f * FOLD_BY_STATE[0],
                    ]
                }
            };
            let s = t.states[o];
            let fold: f64 = (0..3).map(|i| s[i] * fold_by_state[i]).sum();
            let detour =
                DETOUR_SHARE * (s[0] * (1.0 - fold_by_state[0]) + s[1] * (1.0 - fold_by_state[1]));
            out.fold_by_state[o] = fold_by_state;
            out.fold[o] = fold;
            out.detour[o] = detour;
            out.q[o] = (1.0 - fold - detour) * t.q[0] + detour * t.q[1] + fold * t.q[2];
            out.phi[o] = (1.0 - fold) * usefulness[o];
        }
        out
    }

    /// The race of §3 for candidate `c` under my mode vector `modes`.
    fn race(
        &self,
        t: &Table,
        c: &Candidate,
        sched: &Schedule,
        opp: &Opponents,
        claim: &[f64; 4],
        modes: &[Option<Mode>; 4],
    ) -> Outcome {
        let me = t.me;
        let mine = &c.mine;
        let mut out = Outcome::default();
        let folding = modes.contains(&Some(Mode::Fold));
        let detouring = modes.contains(&Some(Mode::Detour));
        // My later discards (§4.4). Push: the useless tiles, the most dangerous
        // first (cheaper now than when the opponents are closer). Fold against some
        // threats: the tiles allowed against every threat (§5.1, the intersection),
        // the useless ones first in the push order (a seat I push against is
        // ignored, so its danger does not stop them), then the rest safest first.
        let danger = |u: Tile| -> f64 {
            t.opponents()
                .map(|o| t.p[o][u.index() as usize] * t.pay[o] * t.l_disc[o])
                .sum()
        };
        let mut useless = mine.useless.clone();
        useless.sort_by(|&a, &b| danger(b).total_cmp(&danger(a)).then(a.cmp(&b)));
        let plan: Vec<Tile> = if folding {
            let allowed = |u: Tile| {
                let p = |o: usize| t.p[o][u.index() as usize];
                t.opponents().all(|o| match modes[o] {
                    Some(Mode::Fold) => p(o) <= FOLD_SAFE,
                    Some(Mode::Detour) => p(o) <= DETOUR_SAFE,
                    _ => true,
                })
            };
            let fold_danger = |u: Tile| -> f64 {
                t.opponents()
                    .filter(|&o| modes[o] == Some(Mode::Fold))
                    .map(|o| t.p[o][u.index() as usize] * t.pay[o])
                    .sum()
            };
            let mut left = mine.view.concealed;
            let mut plan: Vec<Tile> = Vec::new();
            for &u in useless.iter().filter(|&&u| allowed(u)) {
                left.remove(u);
                plan.push(u);
            }
            let mut rest: Vec<Tile> = left
                .kinds()
                .flat_map(|u| std::iter::repeat_n(u, left[u] as usize))
                .filter(|&u| allowed(u))
                .collect();
            rest.sort_by(|&a, &b| fold_danger(a).total_cmp(&fold_danger(b)).then(a.cmp(&b)));
            plan.extend(rest);
            plan
        } else {
            useless
        };
        // After the plan, the per-discard rate of my mode toward each opponent.
        let rate = |o: usize| -> f64 {
            match modes[o] {
                Some(Mode::Fold) => t.q[0], // the reserve is used up: push-level danger
                Some(m) => t.q[m as usize],
                None => t.q[0],
            }
        };

        // My chain: (far, one away, ready), exact at the start.
        let pool = t.pool;
        let wall = t.wall;
        let reachable = mine.shanten < UNREACHABLE;
        let mut pm = match mine.shanten {
            0 => [0.0, 0.0, 1.0],
            1 => [0.0, 1.0, 0.0],
            _ if reachable => [1.0, 0.0, 0.0],
            _ => [0.0, 0.0, 0.0],
        };
        let accept = (mine.acceptance / pool).min(1.0);
        let steps_to_one_away = (mine.shanten.max(2) - 1) as f64;
        let rate_far = accept / steps_to_one_away;
        let ready_rate = mine.live_when_ready / pool;

        // Opponents' unfolded chains (ready, one away, far) and multipliers.
        let mut po: [[f64; 3]; 4] = [[0.0; 3]; 4];
        for o in t.opponents() {
            po[o] = std::array::from_fn(|i| t.states[o][i] * (1.0 - opp.fold_by_state[o][i]));
        }
        let new_liab = |o: usize, k: usize| -> (f64, f64, f64) {
            let base = (t.l_disc[o], t.l_sd[o], t.l_other[o]);
            if k == 0 || !t.liab_seat[o] || claim[o] == 0.0 {
                return base;
            }
            let c = claim[o];
            (
                (1.0 - c) * base.0 + c * 3.0,
                (1.0 - c) * base.1 + c * 3.0,
                (1.0 - c) * base.2 + c * 3.0,
            )
        };

        // §4.1: the deal-in now, from the belief, before anyone else acts (a seat
        // that has folded does not win). Several winners are all paid.
        let mut s = 1.0;
        if let Some(x) = c.tile {
            let mut p = [0.0; 4];
            for o in t.opponents() {
                p[o] = t.p[o][x.index() as usize] * (1.0 - opp.fold_by_state[o][0]);
                out.deal_in_now[o] = p[o] * t.pay[o] * t.l_disc[o];
                out.loss_mine += out.deal_in_now[o];
            }
            let total: f64 = p.iter().sum();
            let scale = if total > 1.0 { 1.0 / total } else { 1.0 };
            for o in t.opponents() {
                out.p_win[o] += p[o] * scale;
            }
            s -= total * scale;
        }
        let mut plan_at = 0usize;
        let mut drew_last = false;
        for (k, acts) in sched.acts.iter().enumerate() {
            // My discard that opponents see in this round: the tile now in round 1,
            // then the plan or my mode's rate after each of my draws.
            let my_feed: [f64; 4] = if k == 0 {
                [0.0; 4]
            } else if drew_last {
                let u = mine_plan_tile(&plan, &mut plan_at);
                std::array::from_fn(|o| {
                    if o == me {
                        0.0
                    } else {
                        u.map_or_else(|| rate(o), |u| t.wait[o][u.index() as usize])
                    }
                })
            } else {
                [0.0; 4]
            };
            // Opponents' hazards.
            let mut h = [0.0; 4];
            let mut parts = [[0.0; 3]; 4]; // self-draw, my feed, others' feeds
            for o in t.opponents() {
                let ready = po[o][0];
                let sd = if acts[o] { t.draw_hazard[o] } else { 0.0 };
                let others: f64 = t
                    .opponents()
                    .filter(|&j| j != o && acts[j])
                    .map(|j| opp.q[j])
                    .sum();
                let mine_part = ready * my_feed[o];
                parts[o] = [ready * sd, mine_part, ready * others];
                h[o] = parts[o].iter().sum();
            }
            // My hazard: fed by the seats that act this round, then my draw.
            let factor = if k == 0 {
                1.0
            } else if folding {
                0.0
            } else if detouring {
                DETOUR_HAZARD
            } else {
                1.0
            };
            let i_draw = acts[me];
            let (mut g, mut d) = (0.0, 0.0);
            if pm[2] > 0.0 {
                if mine.shanten == 0 {
                    let per_draw = if wall > 0.0 {
                        mine.wall_live / wall
                    } else {
                        0.0
                    };
                    for j in t.opponents().filter(|&j| acts[j]) {
                        let held = if sched.first[j] == Some(k) {
                            mine.held[j]
                        } else {
                            0.0
                        };
                        g += opp.phi[j] * (held + per_draw);
                    }
                    d = if i_draw { per_draw } else { 0.0 };
                } else {
                    for j in t.opponents().filter(|&j| acts[j]) {
                        g += opp.phi[j] * ready_rate;
                    }
                    d = if i_draw { ready_rate } else { 0.0 };
                }
            }
            let g = g.min(P_CLIP);
            let win_disc = pm[2] * factor * g;
            let win_sd = pm[2] * factor * (1.0 - g) * d.min(P_CLIP);
            let mut total = win_disc + win_sd + h.iter().sum::<f64>();
            // Hazards are exclusive within a round; keep their sum a probability.
            let scale = if total > 1.0 { 1.0 / total } else { 1.0 };
            total *= scale;
            out.gain += s
                * scale
                * (win_disc * mine.pay * mine.shares_disc + win_sd * mine.pay_sd * mine.shares_sd);
            out.p_win[me] += s * scale * (win_disc + win_sd);
            for o in t.opponents() {
                let (l_disc, l_sd, l_other) = new_liab(o, k);
                out.loss_sd += s * scale * parts[o][0] * t.pay_sd[o] * l_sd;
                out.loss_mine += s * scale * parts[o][1] * t.pay[o] * l_disc;
                out.loss_other += s * scale * parts[o][2] * t.pay[o] * l_other;
                out.p_win[o] += s * scale * h[o];
            }
            s *= 1.0 - total;
            // Progress after the round: my draw moves my chain (unless I fold);
            // each acting opponent's unfolded share moves (faster in round 1 if the
            // tile now feeds it a claim).
            if i_draw && reachable && !(folding && k > 0) {
                let to_ready = pm[1] * accept.min(1.0);
                let to_one = pm[0] * rate_far.min(1.0);
                pm = [pm[0] - to_one, pm[1] + to_one - to_ready, pm[2] + to_ready];
            }
            for o in t.opponents().filter(|&o| acts[o]) {
                let bonus = if k == 0 { claim[o] } else { 0.0 };
                let [r, one, far] = po[o];
                let up_far = far * (t.step_far + bonus).min(1.0);
                let up_one = one * (t.step_one + bonus).min(1.0);
                po[o] = [r + up_one, one - up_one + up_far, far - up_far];
            }
            drew_last = i_draw;
        }
        out.p_draw = s;
        out.equity = out.gain - out.loss_sd - out.loss_mine - out.loss_other;
        out.plan = plan;
        out
    }
}

fn mine_plan_tile(plan: &[Tile], at: &mut usize) -> Option<Tile> {
    let u = plan.get(*at).copied();
    *at += 1;
    u
}

/// `P̂(me wins)` for §5.2: my hazard over the rounds with nobody else racing,
/// split into self-draws and the wins each seat feeds (with `φ` before folds).
fn my_win_estimate(
    t: &Table,
    mine: &Mine,
    sched: &Schedule,
    usefulness: &[f64; 4],
) -> (f64, [f64; 4]) {
    let me = t.me;
    let (mut sd, mut fed) = (0.0, [0.0; 4]);
    if mine.shanten >= UNREACHABLE {
        return (sd, fed);
    }
    let accept = (mine.acceptance / t.pool).min(1.0);
    let rate_far = accept / (mine.shanten.max(2) - 1) as f64;
    let mut pm = match mine.shanten {
        0 => [0.0, 0.0, 1.0],
        1 => [0.0, 1.0, 0.0],
        _ => [1.0, 0.0, 0.0],
    };
    let mut alive = 1.0;
    for (k, acts) in sched.acts.iter().enumerate() {
        let (per_draw, held) = if mine.shanten == 0 {
            let p = if t.wall > 0.0 {
                mine.wall_live / t.wall
            } else {
                0.0
            };
            (p, mine.held)
        } else {
            (mine.live_when_ready / t.pool, [0.0; 4])
        };
        let mut g = 0.0;
        for j in t.opponents().filter(|&j| acts[j]) {
            let first = if sched.first[j] == Some(k) {
                held[j]
            } else {
                0.0
            };
            let x = alive * pm[2] * usefulness[j] * (first + per_draw);
            fed[j] += x;
            g += x;
        }
        let d = if acts[me] { per_draw } else { 0.0 };
        let x = (alive * pm[2] - g).max(0.0) * d;
        sd += x;
        alive = (alive - g - x).max(0.0);
        if acts[me] {
            let to_ready = pm[1] * accept;
            let to_one = pm[0] * rate_far;
            pm = [pm[0] - to_one, pm[1] + to_one - to_ready, pm[2] + to_ready];
        }
    }
    (sd, fed)
}

/// Each discard kind of a hand that must discard, with the kinds I could throw
/// later from the hand left without losing ground: one copy of every other kind
/// whose discard now reaches the same faan-aware shanten or better (§4.4).
fn discards_of(view: &SeatView) -> Vec<(Tile, Vec<Tile>)> {
    let kinds: Vec<Tile> = view.concealed.kinds().collect();
    let s: Vec<i8> = kinds
        .iter()
        .map(|&d| {
            let mut h = view.concealed;
            h.remove(d);
            fs_of(view, &h).shanten
        })
        .collect();
    kinds
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            let useless = kinds
                .iter()
                .enumerate()
                .filter(|&(j, &u)| s[j] <= s[i] && (u != d || view.concealed[d] > 1))
                .map(|(_, &u)| u)
                .collect();
            (d, useless)
        })
        .collect()
}

// ----- Training labels (§7, plan Task 8), from the true hands. -----

/// A player's true state (§2.1, §7), as written to the race log.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TrueState {
    /// Faan-aware shanten ≥ 2 (or no target meets the minimum).
    Far = 0,
    /// Faan-aware shanten 1 (or 0 without a legal win).
    OneAway = 1,
    /// A legal win is available on some tile now.
    Ready = 2,
    /// Folded: from the turn after a fold-labelled discard until a push label.
    Folded = 3,
}

/// Faan-aware shanten of `view`'s concealed hand; for a hand that must discard
/// (14-tile shape), the best over its discards.
pub fn hand_shanten(view: &SeatView) -> i8 {
    let tiles = view.concealed.total();
    if tiles % 3 == 2 {
        view.concealed
            .kinds()
            .map(|d| {
                let mut h = view.concealed;
                h.remove(d);
                fs_of(view, &h).shanten
            })
            .min()
            .unwrap_or(UNREACHABLE)
    } else {
        fs_of(view, &view.concealed).shanten
    }
}

/// `seat`'s true state in `state` (reads its hidden hand: labels only). `folded`
/// overrides the rest.
pub fn true_state(state: &State, seat: usize, folded: bool) -> TrueState {
    if folded {
        TrueState::Folded
    } else if Tile::all_kinds().any(|k| state.wins_on(seat, k)) {
        TrueState::Ready
    } else if hand_shanten(&SeatView::of(state, seat)) <= 1 {
        TrueState::OneAway
    } else {
        TrueState::Far
    }
}

// The label tolerances (docs/race-bot.md §12, "the mode label redefined"). All of
// them in one place: the label of a discard compares it with the discards that are
// best for the player's own hand.

/// A discard at the best distance is still "best for the hand" when its acceptance
/// (unseen tiles that bring the hand closer; the live winning tiles once ready) is
/// short of the best by at most this many tiles...
pub const LABEL_ACCEPT_TILES: i32 = 2;
/// ...or by at most this share of the best's acceptance, whichever is larger.
pub const LABEL_ACCEPT_SHARE: f64 = 0.15;
/// ...and its target's faan is at most this much below the best's.
pub const LABEL_FAAN: i32 = 1;
/// Keeping the best distance but less than this share of the best acceptance gives
/// up most of the hand's speed: it counts as a fold, not a detour.
pub const LABEL_FOLD_ACCEPT_SHARE: f64 = 0.5;
/// "Clearly safer": the discard's expected loss (`Danger::expected_loss`, points) is
/// at most this share of the safest best-for-hand discard's...
pub const LABEL_SAFER_SHARE: f64 = 0.5;
/// ...and lower by at least this many points (2 percentage points of deal-in at
/// `Danger::estimate`'s 8 points per deal-in).
pub const LABEL_SAFER_POINTS: f64 = 0.16;

/// The label of one discard (§12's redefinition of §7). [`Label::mode`] is what the
/// log writes as the mode; a [`Label::Mistake`] is written as push and counted
/// apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Label {
    /// Among the best discards for the player's own hand, whatever its danger.
    Push,
    /// Keeps the best distance but gives up acceptance or value, for a clearly
    /// safer tile.
    Detour,
    /// Gives up distance (or most of the acceptance), for a clearly safer tile.
    Fold,
    /// Worse for the hand and not clearly safer: no defensive intent (mode push).
    Mistake,
}

impl Label {
    /// The mode the log writes: a mistake is push.
    pub fn mode(self) -> Mode {
        match self {
            Label::Detour => Mode::Detour,
            Label::Fold => Mode::Fold,
            Label::Push | Label::Mistake => Mode::Push,
        }
    }
}

/// The label of discarding `discard` from `view`'s 14-tile-shape hand (§12):
///
/// * the **best-for-hand** discards are those at `rank_discards`' best faan-aware
///   distance whose acceptance is within [`LABEL_ACCEPT_TILES`] /
///   [`LABEL_ACCEPT_SHARE`] of the best and whose target faan is within
///   [`LABEL_FAAN`]; `discard` among them is **push**;
/// * otherwise it is clearly safer when `loss(discard)` is at most
///   [`LABEL_SAFER_SHARE`] of the lowest `loss` among the best-for-hand discards and
///   lower by at least [`LABEL_SAFER_POINTS`];
/// * a clearly safer discard is **fold** if it raises the distance or keeps less than
///   [`LABEL_FOLD_ACCEPT_SHARE`] of the best acceptance, else **detour**;
/// * a discard neither best for the hand nor clearly safer is a **mistake**.
///
/// `loss` is the public expected loss of a discard (`Danger::expected_loss`).
pub fn discard_label(view: &SeatView, discard: Tile, loss: impl Fn(Tile) -> f64) -> Label {
    let kinds: Vec<Tile> = view.concealed.kinds().collect();
    // Key: (distance, −acceptance, −target faan, connectedness); acceptance is filled
    // for the discards at the best distance only, which are the only ones compared.
    let ranked = rank_discards(view, &kinds, Metric::Faan);
    let (best_d, best_acc, best_faan) = {
        let k = ranked[0].1;
        (k.0, -k.1, -k.2)
    };
    let tolerance = f64::from(LABEL_ACCEPT_TILES).max(LABEL_ACCEPT_SHARE * f64::from(best_acc));
    let best_for_hand = |k: &crate::bots::Key| {
        k.0 == best_d && f64::from(best_acc + k.1) <= tolerance && best_faan + k.2 <= LABEL_FAAN
    };
    let Some(&(_, key)) = ranked.iter().find(|(t, _)| *t == discard) else {
        return Label::Mistake; // not a held tile: cannot happen for a legal discard
    };
    if best_for_hand(&key) {
        return Label::Push;
    }
    let safest = ranked
        .iter()
        .filter(|(_, k)| best_for_hand(k))
        .map(|&(t, _)| loss(t))
        .fold(f64::INFINITY, f64::min);
    let mine = loss(discard);
    if !(mine <= LABEL_SAFER_SHARE * safest && safest - mine >= LABEL_SAFER_POINTS) {
        return Label::Mistake;
    }
    if key.0 > best_d || f64::from(-key.1) < LABEL_FOLD_ACCEPT_SHARE * f64::from(best_acc) {
        Label::Fold
    } else {
        Label::Detour
    }
}

/// The per-seat bookkeeping behind the labels over one hand: the mode of each seat's
/// last discard and whether it is folded (§7: from a fold label until the seat's next
/// push label or claim; a detour or a mistake keeps the fold).
#[derive(Clone, Debug)]
pub struct LabelTracker {
    last: [Mode; 4],
    folded: [bool; 4],
}

impl LabelTracker {
    /// At the start of a hand.
    pub fn new(_state: &State) -> LabelTracker {
        LabelTracker {
            last: [Mode::Push; 4],
            folded: [false; 4],
        }
    }

    /// Records `seat`'s discard labelled `label`: its mode becomes the seat's last
    /// mode; a fold starts the folded state, a push ends it, a detour or a mistake
    /// leaves it as it was.
    pub fn record(&mut self, seat: usize, label: Label) {
        self.last[seat] = label.mode();
        match label {
            Label::Fold => self.folded[seat] = true,
            Label::Push => self.folded[seat] = false,
            Label::Detour | Label::Mistake => {}
        }
    }

    /// `seat`'s true state now.
    pub fn state_of(&self, state: &State, seat: usize) -> TrueState {
        true_state(state, seat, self.folded[seat])
    }

    /// The mode of `seat`'s last discard (push before its first).
    pub fn mode_of(&self, seat: usize) -> Mode {
        self.last[seat]
    }

    /// Whether `seat` is folded.
    pub fn folded(&self, seat: usize) -> bool {
        self.folded[seat]
    }

    /// Records the player to act's `action` **before** it is applied. Returns the
    /// label of a discard ([`discard_label`] with the public danger estimate); a pung
    /// or chow counts as push (it ends a fold) and returns `None`, as does everything
    /// else.
    ///
    /// # Panics
    /// Panics if the hand is over.
    pub fn observe(&mut self, state: &State, action: Action) -> Option<Label> {
        let seat = state.current_player().expect("hand in progress");
        match action {
            Action::Discard(t) if !state.in_claim() && !state.in_rob() => {
                let view = SeatView::of(state, seat);
                let danger = Danger::estimate(state, seat);
                let label = discard_label(&view, t, |u| danger.expected_loss(u));
                self.record(seat, label);
                Some(label)
            }
            Action::Pung | Action::Chow(_) => {
                self.last[seat] = Mode::Push;
                self.folded[seat] = false;
                None
            }
            _ => None,
        }
    }
}

impl mahjong_games::Policy<State> for RaceBot {
    fn choose<R: rand::Rng + ?Sized>(
        &self,
        state: &State,
        legal: &[Action],
        _rng: &mut R,
    ) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        self.decide_among(state, legal).action
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_parse_and_reject() {
        let mut c = RaceConfig::default();
        c.set("horizon", "4").unwrap();
        c.set("threat", "0.3").unwrap();
        assert_eq!((c.horizon, c.threat), (4, 0.3));
        assert!(c.set("horizon", "0").is_err());
        assert!(c.set("threat", "-1").is_err());
        assert!(c.set("nope", "1").is_err());
        assert!(parse_spec(&[("belief", "models/does-not-exist.mjnn")]).is_err());
    }

    #[test]
    fn rounds_end_with_my_draw() {
        // Pass on seat 2's discard: seat 3 then me in round 1, eight draws for me.
        let s = Schedule::new(0, 3, 30, 8);
        assert_eq!(s.acts[0], [true, false, false, true]);
        assert_eq!(s.acts.iter().filter(|r| r[0]).count(), 8);
        assert_eq!((s.first[1], s.first[2]), (Some(1), Some(1)));
        // After my discard: seats 1, 2, 3 then me; seven draws for me.
        let s = Schedule::new(0, 1, 30, 8);
        assert_eq!(s.acts[0], [true; 4]);
        assert_eq!(s.acts.iter().filter(|r| r[0]).count(), 7);
    }
}
