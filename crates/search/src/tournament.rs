//! Duplicate tournaments: the same walls for every agent, seats permuted.
//!
//! [`tournament`] plays each wall once per **seating**: either the 4 rotations
//! (every agent plays every seat once per wall) or all 24 permutations of the
//! four agents (every agent plays every seat 6 times per wall, against every
//! arrangement of the others). It keeps every hand ([`HandRecord`]) and
//! [`Tournament::summary`] reduces them to the numbers every match in this repo
//! reports, written as one JSON shape by [`Summary::to_json`]:
//!
//! * per agent: points per hand with (a) a 95% interval treating hands as
//!   independent and (b) a **wall-paired bootstrap** interval (resample walls with
//!   replacement; an agent's score on a wall is its mean over that wall's
//!   seatings, so seat and wall luck cancel inside each wall); win, self-drawn,
//!   deal-in and draw rates; mean faan of wins; win share per faan pattern (keyed
//!   by `Pattern::english`); deal-ins split by the agent that won; points lost as
//!   the liable player (§8.3);
//! * per pair of agents: the paired difference of their wall scores with a normal
//!   and a bootstrap interval.
//!
//! Results depend only on the agents, rules, `seed` and `deals` (each wall has its
//! own generator; walls run in parallel).

use crate::agent::Agent;
use mahjong_games::{Game, Policy};
use mahjong_hk::{HandResult, RuleSet, State};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Which seatings each wall is played in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seatings {
    /// The 4 rotations: agent `(seat + r) % 4` sits in `seat` in rotation `r`.
    Rotations,
    /// All 24 permutations of the four agents.
    All,
}

impl Seatings {
    /// From the count: 4 or 24.
    pub fn from_count(n: usize) -> Option<Seatings> {
        match n {
            4 => Some(Seatings::Rotations),
            24 => Some(Seatings::All),
            _ => None,
        }
    }

    /// The seatings, each as the agent index in seats 0–3.
    pub fn list(self) -> Vec<[u8; 4]> {
        match self {
            Seatings::Rotations => (0..4u8)
                .map(|r| std::array::from_fn(|seat| ((seat as u8) + r) % 4))
                .collect(),
            Seatings::All => {
                let mut out = Vec::with_capacity(24);
                for a in 0..4u8 {
                    for b in (0..4).filter(|&b| b != a) {
                        for c in (0..4).filter(|&c| c != a && c != b) {
                            let d = 6 - a - b - c;
                            out.push([a, b, c, d]);
                        }
                    }
                }
                out
            }
        }
    }
}

/// Tournament settings.
#[derive(Clone, Copy, Debug)]
pub struct TournamentConfig {
    pub deals: usize,
    pub seatings: Seatings,
    pub rules: RuleSet,
    pub seed: u64,
    /// Bootstrap resamples for the wall-paired intervals.
    pub bootstrap: usize,
}

impl Default for TournamentConfig {
    fn default() -> Self {
        TournamentConfig {
            deals: 100,
            seatings: Seatings::Rotations,
            rules: RuleSet::default(),
            seed: 0,
            bootstrap: 2000,
        }
    }
}

/// One hand of a tournament.
#[derive(Clone, Debug)]
pub struct HandRecord {
    /// Wall index (0-based).
    pub wall: u32,
    /// Agent index in each seat.
    pub seating: [u8; 4],
    pub result: HandResult,
}

/// Every hand of a tournament.
#[derive(Clone, Debug)]
pub struct Tournament {
    pub config: TournamentConfig,
    pub hands: Vec<HandRecord>,
}

/// Plays a duplicate tournament (see the module docs).
pub fn tournament(agents: &[Agent; 4], config: &TournamentConfig) -> Tournament {
    let seatings = config.seatings.list();
    let rules = config.rules;
    let seed = config.seed;
    let per_wall: Vec<Vec<HandRecord>> = (0..config.deals)
        .into_par_iter()
        .map(|d| {
            // One generator per wall, used for the shuffle and then by the agents
            // through the seatings in order (as `duplicate` always did).
            let mut rng =
                ChaCha8Rng::seed_from_u64(seed ^ (d as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut wall = State::all_tiles(&rules);
            wall.shuffle(&mut rng);
            let mut out = Vec::with_capacity(seatings.len());
            let mut legal = Vec::new();
            for seating in &seatings {
                let mut s = State::from_wall(rules, wall.clone());
                while let Some(p) = s.current_player() {
                    s.legal_actions(&mut legal);
                    let a = agents[seating[p] as usize].choose(&s, &legal, &mut rng);
                    s.apply(a);
                }
                out.push(HandRecord {
                    wall: d as u32,
                    seating: *seating,
                    result: s.result().expect("finished").clone(),
                });
            }
            out
        })
        .collect();
    Tournament {
        config: *config,
        hands: per_wall.into_iter().flatten().collect(),
    }
}

/// A 95% interval.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    /// Whether `x` lies inside.
    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }
}

/// One agent's numbers over a tournament.
#[derive(Clone, Debug, Default)]
pub struct AgentSummary {
    pub hands: u32,
    pub points: f64,
    pub points_sq: f64,
    pub wins: u32,
    pub self_drawn: u32,
    pub deal_ins: u32,
    pub draws: u32,
    pub win_faan: u64,
    /// Deal-ins by the agent that won (agent index).
    pub deal_ins_to: [u32; 4],
    /// Points paid as the liable player (§8.3), counted positive.
    pub liability_points: f64,
    /// Wins containing each pattern, keyed by `Pattern::english`.
    pub pattern_wins: BTreeMap<&'static str, u32>,
    /// Wall-paired bootstrap interval of the mean.
    pub bootstrap: Interval,
}

impl AgentSummary {
    /// Mean points per hand.
    pub fn mean(&self) -> f64 {
        self.points / self.hands.max(1) as f64
    }

    /// Half-width of the 95% interval treating hands as independent.
    pub fn ci95(&self) -> f64 {
        let n = self.hands.max(1) as f64;
        let m = self.mean();
        1.96 * ((self.points_sq / n - m * m).max(0.0) / n).sqrt()
    }

    fn rate(&self, x: u32) -> f64 {
        x as f64 / self.hands.max(1) as f64
    }
}

/// The paired difference between two agents (`a` minus `b`).
#[derive(Clone, Debug, Default)]
pub struct PairSummary {
    pub a: usize,
    pub b: usize,
    /// Mean over walls of (a's wall score − b's wall score).
    pub mean: f64,
    /// Normal 95% half-width over walls.
    pub ci95: f64,
    pub bootstrap: Interval,
}

/// A tournament reduced to its report.
#[derive(Clone, Debug)]
pub struct Summary {
    pub deals: usize,
    pub seatings: usize,
    pub seed: u64,
    pub bootstrap_draws: usize,
    /// Hands in total (deals × seatings).
    pub hands: usize,
    /// Hands that ended in an exhaustive draw.
    pub draws: u32,
    /// Points won by each seat in total (all agents together).
    pub seat_points: [f64; 4],
    pub agents: Vec<AgentSummary>,
    pub pairs: Vec<PairSummary>,
}

/// Percentile interval of `draws` (sorted in place).
fn percentile(draws: &mut [f64]) -> Interval {
    if draws.is_empty() {
        return Interval::default();
    }
    draws.sort_by(f64::total_cmp);
    let n = draws.len();
    let lo = ((0.025 * n as f64).floor() as usize).min(n - 1);
    let hi = (((0.975 * n as f64).ceil() as usize).max(1) - 1).min(n - 1);
    Interval {
        lo: draws[lo],
        hi: draws[hi],
    }
}

impl Tournament {
    /// Each agent's mean points per hand on each wall: `[wall][agent]`.
    pub fn wall_scores(&self) -> Vec<[f64; 4]> {
        let mut sum = vec![[0.0f64; 4]; self.config.deals];
        let mut n = vec![[0u32; 4]; self.config.deals];
        for h in &self.hands {
            for seat in 0..4 {
                let a = h.seating[seat] as usize;
                sum[h.wall as usize][a] += h.result.deltas[seat] as f64;
                n[h.wall as usize][a] += 1;
            }
        }
        sum.iter()
            .zip(&n)
            .map(|(s, c)| std::array::from_fn(|a| s[a] / c[a].max(1) as f64))
            .collect()
    }

    /// Reduces the hands to the report.
    pub fn summary(&self) -> Summary {
        let mut agents = vec![AgentSummary::default(); 4];
        let mut seat_points = [0.0f64; 4];
        let mut draws = 0;
        for h in &self.hands {
            let r = &h.result;
            draws += r.wins.is_empty() as u32;
            for seat in 0..4 {
                let me = &mut agents[h.seating[seat] as usize];
                let x = r.deltas[seat] as f64;
                seat_points[seat] += x;
                me.hands += 1;
                me.points += x;
                me.points_sq += x * x;
                me.draws += r.wins.is_empty() as u32;
                for w in &r.wins {
                    if w.seat == seat {
                        me.wins += 1;
                        me.self_drawn += w.self_drawn as u32;
                        me.win_faan += w.score.faan as u64;
                        for (p, _) in &w.score.patterns {
                            *me.pattern_wins.entry(p.english()).or_default() += 1;
                        }
                    }
                    if !w.self_drawn && w.from == Some(seat) {
                        me.deal_ins += 1;
                        me.deal_ins_to[h.seating[w.seat] as usize] += 1;
                    }
                    if w.liable == Some(seat) {
                        me.liability_points += w.paid_by[seat] as f64;
                    }
                }
            }
        }

        // Wall-paired bootstrap: resample walls, average the wall scores.
        let walls = self.wall_scores();
        let w = walls.len();
        let b = self.config.bootstrap;
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.seed ^ 0xB007_57A9_0000_0001);
        let mut agent_draws: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(b)).collect();
        let pairs_idx: Vec<(usize, usize)> = (0..4)
            .flat_map(|a| (a + 1..4).map(move |c| (a, c)))
            .collect();
        let mut pair_draws: Vec<Vec<f64>> =
            pairs_idx.iter().map(|_| Vec::with_capacity(b)).collect();
        if w > 0 {
            for _ in 0..b {
                let mut m = [0.0f64; 4];
                for _ in 0..w {
                    let s = &walls[rng.gen_range(0..w as u32) as usize];
                    for a in 0..4 {
                        m[a] += s[a];
                    }
                }
                for a in 0..4 {
                    agent_draws[a].push(m[a] / w as f64);
                }
                for (k, &(a, c)) in pairs_idx.iter().enumerate() {
                    pair_draws[k].push((m[a] - m[c]) / w as f64);
                }
            }
        }
        for (s, d) in agents.iter_mut().zip(agent_draws.iter_mut()) {
            s.bootstrap = percentile(d);
        }
        let pairs = pairs_idx
            .iter()
            .zip(pair_draws.iter_mut())
            .map(|(&(a, c), d)| {
                let diffs: Vec<f64> = walls.iter().map(|s| s[a] - s[c]).collect();
                let n = diffs.len().max(1) as f64;
                let mean = diffs.iter().sum::<f64>() / n;
                let var =
                    diffs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
                PairSummary {
                    a,
                    b: c,
                    mean,
                    ci95: 1.96 * (var / n).sqrt(),
                    bootstrap: percentile(d),
                }
            })
            .collect();
        Summary {
            deals: self.config.deals,
            seatings: self.config.seatings.list().len(),
            seed: self.config.seed,
            bootstrap_draws: b,
            hands: self.hands.len(),
            draws,
            seat_points,
            agents,
            pairs,
        }
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A finite number as JSON (`null` for NaN or infinity).
fn json_num(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".into()
    }
}

impl Summary {
    /// The report as JSON, `names` being the agents' spec strings. Every match
    /// in this repo writes this shape (`format`: `hk-tournament/1`); the Python
    /// CLI adds the rules.
    pub fn to_json(&self, names: &[String; 4]) -> String {
        let mut s = String::from("{\n");
        let _ = writeln!(s, "  \"format\": \"hk-tournament/1\",");
        let _ = writeln!(s, "  \"deals\": {},", self.deals);
        let _ = writeln!(s, "  \"seatings\": {},", self.seatings);
        let _ = writeln!(s, "  \"seed\": {},", self.seed);
        let _ = writeln!(s, "  \"bootstrap\": {},", self.bootstrap_draws);
        let _ = writeln!(s, "  \"hands\": {},", self.hands);
        let _ = writeln!(s, "  \"draws\": {},", self.draws);
        let _ = writeln!(
            s,
            "  \"draw_rate\": {},",
            json_num(self.draws as f64 / self.hands.max(1) as f64)
        );
        let seats: Vec<String> = self.seat_points.iter().map(|&x| json_num(x)).collect();
        let _ = writeln!(s, "  \"seat_points\": [{}],", seats.join(", "));
        s.push_str("  \"agents\": [\n");
        for (i, (a, name)) in self.agents.iter().zip(names).enumerate() {
            let patterns: Vec<String> = a
                .pattern_wins
                .iter()
                .map(|(k, v)| format!("{}: {v}", json_str(k)))
                .collect();
            let shares: Vec<String> = a
                .pattern_wins
                .iter()
                .map(|(k, &v)| {
                    format!(
                        "{}: {}",
                        json_str(k),
                        json_num(v as f64 / a.wins.max(1) as f64)
                    )
                })
                .collect();
            let to: Vec<String> = a.deal_ins_to.iter().map(|x| x.to_string()).collect();
            let _ = write!(
                s,
                "    {{\"agent\": {}, \"hands\": {}, \"mean\": {}, \"ci95\": {}, \
                 \"boot_lo\": {}, \"boot_hi\": {}, \"wins\": {}, \"win_rate\": {}, \
                 \"self_drawn\": {}, \"self_drawn_rate\": {}, \"deal_ins\": {}, \
                 \"deal_in_rate\": {}, \"draws\": {}, \"draw_rate\": {}, \"win_faan\": {}, \
                 \"mean_faan\": {}, \"liability_points\": {}, \"deal_ins_to\": [{}], \
                 \"pattern_wins\": {{{}}}, \"pattern_shares\": {{{}}}}}",
                json_str(name),
                a.hands,
                json_num(a.mean()),
                json_num(a.ci95()),
                json_num(a.bootstrap.lo),
                json_num(a.bootstrap.hi),
                a.wins,
                json_num(a.rate(a.wins)),
                a.self_drawn,
                json_num(a.rate(a.self_drawn)),
                a.deal_ins,
                json_num(a.rate(a.deal_ins)),
                a.draws,
                json_num(a.rate(a.draws)),
                a.win_faan,
                json_num(a.win_faan as f64 / a.wins.max(1) as f64),
                json_num(a.liability_points),
                to.join(", "),
                patterns.join(", "),
                shares.join(", ")
            );
            s.push_str(if i + 1 < self.agents.len() {
                ",\n"
            } else {
                "\n"
            });
        }
        s.push_str("  ],\n  \"pairs\": [\n");
        for (i, p) in self.pairs.iter().enumerate() {
            let _ = write!(
                s,
                "    {{\"a\": {}, \"b\": {}, \"agent_a\": {}, \"agent_b\": {}, \"mean\": {}, \
                 \"ci95\": {}, \"boot_lo\": {}, \"boot_hi\": {}}}",
                p.a,
                p.b,
                json_str(&names[p.a]),
                json_str(&names[p.b]),
                json_num(p.mean),
                json_num(p.ci95),
                json_num(p.bootstrap.lo),
                json_num(p.bootstrap.hi)
            );
            s.push_str(if i + 1 < self.pairs.len() {
                ",\n"
            } else {
                "\n"
            });
        }
        s.push_str("  ]\n}\n");
        s
    }
}

/// One agent's totals over a tournament (the older, smaller report).
#[derive(Clone, Debug, Default)]
pub struct AgentResult {
    pub hands: u32,
    pub points: f64,
    pub points_sq: f64,
    pub wins: u32,
    pub deal_ins: u32,
    pub self_drawn: u32,
    pub win_faan: u64,
}

impl AgentResult {
    /// Mean points per hand.
    pub fn mean(&self) -> f64 {
        self.points / self.hands.max(1) as f64
    }

    /// Half-width of the 95% confidence interval of the mean (treating hands as
    /// independent; duplicate seating makes the true interval narrower).
    pub fn ci95(&self) -> f64 {
        let n = self.hands.max(1) as f64;
        let m = self.mean();
        1.96 * ((self.points_sq / n - m * m).max(0.0) / n).sqrt()
    }
}

/// Totals of a duplicate tournament (the older, smaller report).
#[derive(Clone, Debug, Default)]
pub struct TournamentResult {
    /// One result per agent, in input order.
    pub agents: Vec<AgentResult>,
    /// Points won by each seat (all agents together); shows seat effects such as
    /// the dealer's extra tile.
    pub seat_points: [f64; 4],
    /// Hands that ended in an exhaustive draw.
    pub draws: u32,
}

/// Plays `deals` walls, each four times with the agents rotated through the
/// seats, so every agent plays every seat on every wall (duplicate format).
/// A wrapper over [`tournament`] with [`Seatings::Rotations`].
pub fn duplicate(agents: &[Agent; 4], deals: usize, rules: RuleSet, seed: u64) -> TournamentResult {
    let t = tournament(
        agents,
        &TournamentConfig {
            deals,
            seatings: Seatings::Rotations,
            rules,
            seed,
            bootstrap: 0,
        },
    );
    let s = t.summary();
    TournamentResult {
        agents: s
            .agents
            .iter()
            .map(|a| AgentResult {
                hands: a.hands,
                points: a.points,
                points_sq: a.points_sq,
                wins: a.wins,
                deal_ins: a.deal_ins,
                self_drawn: a.self_drawn,
                win_faan: a.win_faan,
            })
            .collect(),
        seat_points: s.seat_points,
        draws: s.draws,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seatings_are_distinct_permutations() {
        for (s, n) in [(Seatings::Rotations, 4), (Seatings::All, 24)] {
            let l = s.list();
            assert_eq!(l.len(), n);
            let mut seen = l.clone();
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), n);
            for p in &l {
                let mut q = *p;
                q.sort();
                assert_eq!(q, [0, 1, 2, 3]);
            }
            // Every agent sits in every seat equally often.
            for seat in 0..4 {
                for a in 0..4u8 {
                    let k = l.iter().filter(|p| p[seat] == a).count();
                    assert_eq!(k, n / 4);
                }
            }
        }
    }

    #[test]
    fn percentile_of_a_uniform_grid() {
        let mut d: Vec<f64> = (0..1000).map(f64::from).collect();
        let i = percentile(&mut d);
        assert_eq!((i.lo, i.hi), (25.0, 974.0));
    }

    #[test]
    fn json_escapes_strings() {
        assert_eq!(json_str("a\"b\\c"), "\"a\\\"b\\\\c\"");
        assert_eq!(json_num(f64::NAN), "null");
    }
}
