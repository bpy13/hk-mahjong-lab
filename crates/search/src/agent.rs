//! Agents chosen by a spec string, and duplicate tournaments between them.
//!
//! Spec strings: a bot name (`random`, `efficiency`, `value`, `defensive`,
//! `fast`), or `pimc` with options, e.g. `pimc:samples=64,candidates=4,ms=300,z=1.0`
//! (see [`crate::pimc::Analysis::choose`] for `z`).

use crate::bots::Bot;
use crate::pimc::{pimc, PimcConfig};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Duration;

/// A player: a rule-based bot or the PIMC solver.
#[derive(Clone, Debug)]
pub enum Agent {
    Bot(Bot),
    Pimc(PimcConfig),
}

impl Agent {
    /// Parses a spec string (see the module docs).
    pub fn parse(spec: &str) -> Result<Agent, String> {
        let (name, opts) = spec.split_once(':').unwrap_or((spec, ""));
        if name == "pimc" {
            let mut c = PimcConfig {
                samples: 64,
                candidates: 4,
                ..PimcConfig::default()
            };
            for kv in opts.split(',').filter(|s| !s.is_empty()) {
                let (k, v) = kv
                    .split_once('=')
                    .ok_or_else(|| format!("bad option {kv:?}"))?;
                if k == "z" {
                    c.z = v.parse().map_err(|_| format!("bad number in {kv:?}"))?;
                    continue;
                }
                let n: u64 = v.parse().map_err(|_| format!("bad number in {kv:?}"))?;
                match k {
                    "samples" => c.samples = n as usize,
                    "candidates" => c.candidates = n as usize,
                    "ms" => c.time_limit = Some(Duration::from_millis(n)),
                    "seed" => c.seed = n,
                    _ => return Err(format!("unknown pimc option {k:?}")),
                }
            }
            return Ok(Agent::Pimc(c));
        }
        if !opts.is_empty() {
            return Err(format!("{name} takes no options"));
        }
        Bot::by_name(name)
            .map(Agent::Bot)
            .ok_or_else(|| format!("unknown agent {name:?}"))
    }
}

impl Policy<State> for Agent {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        match self {
            Agent::Bot(b) => b.choose(state, legal, rng),
            Agent::Pimc(c) => {
                if legal.len() == 1 {
                    return legal[0];
                }
                let config = PimcConfig {
                    seed: rng.gen(),
                    ..c.clone()
                };
                pimc(state, &config).choose(config.z)
            }
        }
    }
}

/// One agent's totals over a tournament.
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

/// Totals of a duplicate tournament.
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
pub fn duplicate(agents: &[Agent; 4], deals: usize, rules: RuleSet, seed: u64) -> TournamentResult {
    let per_deal: Vec<(Vec<AgentResult>, [f64; 4], u32)> = (0..deals)
        .into_par_iter()
        .map(|d| {
            let mut rng =
                SmallRng::seed_from_u64(seed ^ (d as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut wall = State::all_tiles(&rules);
            wall.shuffle(&mut rng);
            let mut out = vec![AgentResult::default(); 4];
            let mut seats = [0.0f64; 4];
            let mut draws = 0;
            for rot in 0..4 {
                let mut s = State::from_wall(rules, wall.clone());
                let seat_agent = |seat: usize| (seat + rot) % 4;
                let mut legal = Vec::new();
                while let Some(p) = s.current_player() {
                    s.legal_actions(&mut legal);
                    let a = agents[seat_agent(p)].choose(&s, &legal, &mut rng);
                    s.apply(a);
                }
                let r = s.result().expect("finished");
                draws += r.wins.is_empty() as u32;
                for seat in 0..4 {
                    let res = &mut out[seat_agent(seat)];
                    let x = r.deltas[seat] as f64;
                    seats[seat] += x;
                    res.hands += 1;
                    res.points += x;
                    res.points_sq += x * x;
                    for w in &r.wins {
                        if w.seat == seat {
                            res.wins += 1;
                            res.self_drawn += w.self_drawn as u32;
                            res.win_faan += w.score.faan as u64;
                        }
                        if !w.self_drawn && w.from == Some(seat) {
                            res.deal_ins += 1;
                        }
                    }
                }
            }
            (out, seats, draws)
        })
        .collect();
    let mut total = vec![AgentResult::default(); 4];
    let mut seat_points = [0.0f64; 4];
    let mut draws = 0;
    for (r, seats, d) in per_deal {
        draws += d;
        for (t, x) in seat_points.iter_mut().zip(seats) {
            *t += x;
        }
        for (t, x) in total.iter_mut().zip(r) {
            t.hands += x.hands;
            t.points += x.points;
            t.points_sq += x.points_sq;
            t.wins += x.wins;
            t.deal_ins += x.deal_ins;
            t.self_drawn += x.self_drawn;
            t.win_faan += x.win_faan;
        }
    }
    TournamentResult {
        agents: total,
        seat_points,
        draws,
    }
}
