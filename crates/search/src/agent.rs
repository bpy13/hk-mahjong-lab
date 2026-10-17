//! Agents chosen by a spec string, and duplicate tournaments between them.
//!
//! Spec strings:
//!
//! * a bot name: `random`, `efficiency`, `value`, `defensive`, `closed`, `fast`;
//! * `pimc` with options, e.g. `pimc:samples=64,candidates=4,ms=300,z=2`
//!   (see [`crate::pimc::Analysis::choose`] for `z`);
//! * `net:path=models/x.mjnn`: a trained network ([`crate::net`]);
//! * `netpimc:path=models/x.mjnn,samples=32,candidates=4,rollout=fast`: network-
//!   guided search. The network proposes the candidates and is the baseline that
//!   PIMC must beat; rollouts use the `fast` bot or (`rollout=net`) the network
//!   in every seat.

use crate::bots::{Bot, Fast};
use crate::net::{top_actions, Mlp, NetPolicy};
use crate::pimc::{pimc, pimc_with, PimcConfig};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Duration;

/// A player: a rule-based bot, the PIMC solver, a network, or network + search.
#[derive(Clone, Debug)]
pub enum Agent {
    Bot(Bot),
    Pimc(PimcConfig),
    Net(NetPolicy),
    NetPimc {
        net: NetPolicy,
        config: PimcConfig,
        net_rollouts: bool,
    },
}

/// Parses `k=v` options; `apply` handles each pair.
fn options(
    opts: &str,
    mut apply: impl FnMut(&str, &str) -> Result<(), String>,
) -> Result<(), String> {
    for kv in opts.split(',').filter(|s| !s.is_empty()) {
        let (k, v) = kv
            .split_once('=')
            .ok_or_else(|| format!("bad option {kv:?}"))?;
        apply(k, v)?;
    }
    Ok(())
}

fn pimc_option(c: &mut PimcConfig, k: &str, v: &str) -> Result<bool, String> {
    let num = || {
        v.parse::<u64>()
            .map_err(|_| format!("bad number {v:?} for {k}"))
    };
    match k {
        "samples" => c.samples = num()? as usize,
        "candidates" => c.candidates = num()? as usize,
        "ms" => c.time_limit = Some(Duration::from_millis(num()?)),
        "seed" => c.seed = num()?,
        "z" => c.z = v.parse().map_err(|_| format!("bad number {v:?} for z"))?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn load_net(path: Option<String>) -> Result<NetPolicy, String> {
    let path = path.ok_or("network agents need path=...")?;
    Ok(NetPolicy(Arc::new(Mlp::load(path)?)))
}

impl Agent {
    /// Parses a spec string (see the module docs).
    pub fn parse(spec: &str) -> Result<Agent, String> {
        let (name, opts) = spec.split_once(':').unwrap_or((spec, ""));
        match name {
            "pimc" => {
                let mut c = PimcConfig {
                    samples: 64,
                    candidates: 4,
                    ..PimcConfig::default()
                };
                options(opts, |k, v| {
                    if pimc_option(&mut c, k, v)? {
                        Ok(())
                    } else {
                        Err(format!("unknown pimc option {k:?}"))
                    }
                })?;
                Ok(Agent::Pimc(c))
            }
            "net" => {
                let mut path = None;
                options(opts, |k, v| match k {
                    "path" => {
                        path = Some(v.to_string());
                        Ok(())
                    }
                    _ => Err(format!("unknown net option {k:?}")),
                })?;
                Ok(Agent::Net(load_net(path)?))
            }
            "netpimc" => {
                let mut c = PimcConfig {
                    samples: 32,
                    candidates: 4,
                    ..PimcConfig::default()
                };
                let mut path = None;
                let mut net_rollouts = false;
                options(opts, |k, v| {
                    if pimc_option(&mut c, k, v)? {
                        return Ok(());
                    }
                    match (k, v) {
                        ("path", _) => path = Some(v.to_string()),
                        ("rollout", "net") => net_rollouts = true,
                        ("rollout", "fast") => net_rollouts = false,
                        _ => return Err(format!("unknown netpimc option {k}={v}")),
                    }
                    Ok(())
                })?;
                Ok(Agent::NetPimc {
                    net: load_net(path)?,
                    config: c,
                    net_rollouts,
                })
            }
            _ if opts.is_empty() => Bot::by_name(name)
                .map(Agent::Bot)
                .ok_or_else(|| format!("unknown agent {name:?}")),
            _ => Err(format!("{name} takes no options")),
        }
    }
}

impl Policy<State> for Agent {
    fn choose<R: Rng + ?Sized>(&self, state: &State, legal: &[Action], rng: &mut R) -> Action {
        if legal.len() == 1 {
            return legal[0];
        }
        match self {
            Agent::Bot(b) => b.choose(state, legal, rng),
            Agent::Net(n) => n.choose(state, legal, rng),
            Agent::Pimc(c) => {
                let config = PimcConfig {
                    seed: rng.gen(),
                    ..c.clone()
                };
                pimc(state, &config).choose(config.z)
            }
            Agent::NetPimc {
                net,
                config,
                net_rollouts,
            } => {
                if let Some(&w) = legal
                    .iter()
                    .find(|a| matches!(a, Action::Win | Action::SelfDrawnWin))
                {
                    return w;
                }
                let config = PimcConfig {
                    seed: rng.gen(),
                    ..config.clone()
                };
                let candidates = top_actions(&net.0, state, config.candidates.max(1));
                let baseline = candidates[0];
                let a = if *net_rollouts {
                    pimc_with(state, &config, candidates, baseline, net)
                } else {
                    pimc_with(state, &config, candidates, baseline, &Fast::default())
                };
                a.choose(config.z)
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
