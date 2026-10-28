//! Agents chosen by a spec string. Duplicate tournaments between them live in
//! [`crate::tournament`]; `duplicate` and its result types are re-exported here
//! for older callers.
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

use crate::bots::Bot;
use crate::net::{top_actions, Mlp, NetPolicy};
use crate::pimc::Rollout;
use crate::pimc::{pimc, pimc_with, PimcConfig};
pub use crate::tournament::{duplicate, AgentResult, TournamentResult};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, State};
use rand::Rng;
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
                    pimc_with(
                        state,
                        &config,
                        candidates,
                        baseline,
                        &Rollout::for_seat(state.current_player().expect("to act")),
                    )
                };
                a.choose(config.z)
            }
        }
    }
}
