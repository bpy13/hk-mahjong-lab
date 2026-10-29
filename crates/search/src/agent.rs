//! Agents chosen by a spec string. Duplicate tournaments between them live in
//! [`crate::tournament`]; `duplicate` and its result types are re-exported here
//! for older callers.
//!
//! Spec strings:
//!
//! * a bot name: `random`, `efficiency`, `value`, `defensive`, `closed`, `fast`, `ev`;
//! * `ev` with options, e.g. `ev:horizon=8,clip=0.95,tie_margin=0.5,r_disc=2,
//!   four_meld_win=0.5,feed_clip=0.5-1.5,warn_p=0.1,fold=safest,floor=target`
//!   ([`crate::ev`],
//!   `docs/ev-bot.md` §4). Without `belief=` it uses the heuristic belief;
//!   `belief=models/belief.mjnn` reads the trained model ([`crate::belief`]);
//! * `defensive:belief=models/belief.mjnn`: `defensive` folding on expected loss
//!   from the belief model ([`crate::Danger::from_belief`]);
//! * `pimc` with options, e.g. `pimc:samples=64,candidates=4,ms=300,z=2`
//!   (see [`crate::pimc::Analysis::choose`] for `z`); `belief=models/belief.mjnn`
//!   draws determinizations from the belief model ([`crate::world::WorldSampler`]),
//!   with importance weights unless `weights=0` (`weights=1` is the default);
//!   `draws=` and `tilt=` override the candidate hands per opponent (16) and the
//!   shanten tilt of its resampling step ([`crate::world::default_tilt`]);
//!   `netpimc` takes the same options;
//!   `anchor=value|ev` picks the baseline bot the search must beat (default
//!   `value`) and `rollout=fast|ev` the bot that plays the searcher's own later
//!   moves in the rollouts (default `fast`; opponents always play the defending
//!   `fast`). The `ev` bot there reads the `belief=` model when one is given
//!   (else the heuristic belief) and runs with `floor=none`, the stronger
//!   setting in Task 10's matches (`docs/ev-bot.md` §10.2); `ev_<key>=<value>`
//!   passes any `ev:` option to it, e.g. `ev_floor=target,ev_horizon=6`
//!   ([`crate::pimc::pimc_anchored`]);
//! * `net:path=models/x.mjnn`: a trained network ([`crate::net`]);
//! * `netpimc:path=models/x.mjnn,samples=32,candidates=4,rollout=fast`: network-
//!   guided search. The network proposes the candidates and is the baseline that
//!   PIMC must beat; rollouts use the `fast` bot or (`rollout=net`) the network
//!   in every seat.

use crate::belief::BeliefModel;
use crate::bots::{Bot, Fast, Value};
use crate::ev::{load_belief, Belief, EvBot, EvConfig, HeuristicBelief};
use crate::net::{top_actions, Mlp, NetPolicy};
use crate::pimc::Rollout;
use crate::pimc::{pimc, pimc_anchored, pimc_sampled, pimc_with_sampler, PimcConfig};
pub use crate::tournament::{duplicate, AgentResult, TournamentResult};
use crate::world::{default_tilt, WorldSampler, BELIEF_DRAWS};
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, State};
use rand::Rng;
use std::sync::Arc;
use std::time::Duration;

/// A player: a rule-based bot, the PIMC solver, a network, or network + search.
#[derive(Clone, Debug)]
pub enum Agent {
    Bot(Bot),
    /// PIMC with best-of-K determinizations.
    Pimc(PimcConfig),
    /// PIMC with another world sampler (`pimc:...,belief=...`).
    PimcSampled {
        config: PimcConfig,
        sampler: WorldSampler,
    },
    /// PIMC with another baseline bot or own-rollout bot
    /// (`pimc:...,anchor=ev,rollout=ev`). Specs without `anchor=` and
    /// `rollout=` keep the two variants above, so their play is unchanged.
    PimcAnchored {
        config: PimcConfig,
        sampler: WorldSampler,
        anchor: Box<Bot>,
        /// The rollout policies; its seat is replaced by the player to act.
        rollout: Box<Rollout>,
    },
    Net(NetPolicy),
    NetPimc {
        net: NetPolicy,
        config: PimcConfig,
        net_rollouts: bool,
        sampler: WorldSampler,
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

/// PIMC options gathered from a spec; `belief=` and `weights=` may come in
/// either order, so the sampler is built at the end ([`PimcSpec::finish`]).
struct PimcSpec {
    config: PimcConfig,
    belief: Option<String>,
    weights: Option<bool>,
    draws: Option<usize>,
    tilt: Option<f64>,
}

impl PimcSpec {
    fn new(samples: usize, candidates: usize) -> PimcSpec {
        PimcSpec {
            config: PimcConfig {
                samples,
                candidates,
                ..PimcConfig::default()
            },
            belief: None,
            weights: None,
            draws: None,
            tilt: None,
        }
    }

    /// Handles one option; `Ok(false)` if `k` is not a PIMC option.
    fn option(&mut self, k: &str, v: &str) -> Result<bool, String> {
        let c = &mut self.config;
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
            "belief" => self.belief = Some(v.to_string()),
            "draws" => self.draws = Some((num()? as usize).max(1)),
            "tilt" => {
                self.tilt = Some(
                    v.parse()
                        .map_err(|_| format!("bad number {v:?} for tilt"))?,
                )
            }
            "weights" => {
                self.weights = Some(match v {
                    "1" => true,
                    "0" => false,
                    _ => return Err(format!("weights takes 0 or 1, not {v:?}")),
                })
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// The finished config and sampler: with `belief=`, the belief sampler
    /// (importance weights on unless `weights=0`), else best-of-K.
    fn finish(self) -> Result<(PimcConfig, WorldSampler), String> {
        let sampler = match (self.belief, self.weights) {
            (Some(path), weights) => {
                if !std::path::Path::new(&path).exists() {
                    return Err(format!("belief model {path:?} not found"));
                }
                let weights = weights.unwrap_or(true);
                WorldSampler::Belief {
                    model: Arc::new(BeliefModel::load(&path)?),
                    weights,
                    draws: self.draws.unwrap_or(BELIEF_DRAWS),
                    tilt: self.tilt.unwrap_or(default_tilt(weights)),
                }
            }
            (None, weights) if weights.is_some() || self.draws.is_some() || self.tilt.is_some() => {
                return Err("weights=, draws= and tilt= need belief=...".to_string())
            }
            (None, _) => WorldSampler::BestOfK,
        };
        Ok((self.config, sampler))
    }
}

/// The `pimc` options `anchor=`, `rollout=` and `ev_<key>=` (see the module
/// docs). Kept apart from [`PimcSpec`] because `netpimc` has its own
/// `rollout=`.
struct AnchorSpec {
    anchor: Option<String>,
    rollout: Option<String>,
    ev: EvConfig,
    ev_options: bool,
}

impl AnchorSpec {
    fn new() -> AnchorSpec {
        AnchorSpec {
            anchor: None,
            rollout: None,
            // floor=none: the stronger `ev` in Task 10's matches.
            ev: EvConfig {
                pay_floor: false,
                ..EvConfig::default()
            },
            ev_options: false,
        }
    }

    /// Handles one option; `Ok(false)` if `k` is not one of these.
    fn option(&mut self, k: &str, v: &str) -> Result<bool, String> {
        match k {
            "anchor" if matches!(v, "value" | "ev") => self.anchor = Some(v.to_string()),
            "anchor" => return Err(format!("anchor is value or ev, not {v:?}")),
            "rollout" if matches!(v, "fast" | "ev") => self.rollout = Some(v.to_string()),
            "rollout" => return Err(format!("rollout is fast or ev, not {v:?}")),
            _ => match k.strip_prefix("ev_") {
                Some(key) if key != "belief" => {
                    self.ev.set(key, v)?;
                    self.ev_options = true;
                }
                _ => return Ok(false),
            },
        }
        Ok(true)
    }

    /// The anchor and rollout policies, or `None` when both are the defaults
    /// (`value`, `fast`), in which case the plain PIMC agents are used.
    fn finish(self, sampler: &WorldSampler) -> Result<Option<(Bot, Rollout)>, String> {
        let anchor_ev = self.anchor.as_deref() == Some("ev");
        let rollout_ev = self.rollout.as_deref() == Some("ev");
        if !anchor_ev && !rollout_ev {
            if self.ev_options {
                return Err("ev_ options need anchor=ev or rollout=ev".to_string());
            }
            return Ok(None);
        }
        let belief: Arc<dyn Belief> = match sampler {
            WorldSampler::Belief { model, .. } => Arc::clone(model) as Arc<dyn Belief>,
            WorldSampler::BestOfK => Arc::new(HeuristicBelief),
        };
        let ev = Bot::Ev(EvBot::new(self.ev, belief));
        let anchor = if anchor_ev {
            ev.clone()
        } else {
            Bot::Value(Value::default())
        };
        let own = if rollout_ev {
            ev
        } else {
            Bot::Fast(Fast::default())
        };
        Ok(Some((
            anchor,
            Rollout::with(0, own, Rollout::default_opponents()),
        )))
    }
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
                let mut c = PimcSpec::new(64, 4);
                let mut a = AnchorSpec::new();
                options(opts, |k, v| {
                    if c.option(k, v)? || a.option(k, v)? {
                        Ok(())
                    } else {
                        Err(format!("unknown pimc option {k:?}"))
                    }
                })?;
                let (config, sampler) = c.finish()?;
                Ok(match (a.finish(&sampler)?, sampler) {
                    (Some((anchor, rollout)), sampler) => Agent::PimcAnchored {
                        config,
                        sampler,
                        anchor: Box::new(anchor),
                        rollout: Box::new(rollout),
                    },
                    (None, WorldSampler::BestOfK) => Agent::Pimc(config),
                    (None, sampler) => Agent::PimcSampled { config, sampler },
                })
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
                let mut c = PimcSpec::new(32, 4);
                let mut path = None;
                let mut net_rollouts = false;
                options(opts, |k, v| {
                    if c.option(k, v)? {
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
                let (config, sampler) = c.finish()?;
                Ok(Agent::NetPimc {
                    net: load_net(path)?,
                    config,
                    net_rollouts,
                    sampler,
                })
            }
            "ev" => {
                let mut config = EvConfig::default();
                let mut belief: Option<String> = None;
                options(opts, |k, v| {
                    if k == "belief" {
                        belief = Some(v.to_string());
                        Ok(())
                    } else {
                        config.set(k, v)
                    }
                })?;
                let belief: Arc<dyn Belief> = match belief {
                    Some(path) => load_belief(&path)?,
                    None => Arc::new(HeuristicBelief),
                };
                Ok(Agent::Bot(Bot::Ev(EvBot::new(config, belief))))
            }
            "defensive" if !opts.is_empty() => {
                let mut belief = None;
                options(opts, |k, v| match k {
                    "belief" => {
                        belief = Some(v.to_string());
                        Ok(())
                    }
                    _ => Err(format!("unknown defensive option {k:?}")),
                })?;
                let belief = load_belief(&belief.ok_or("defensive takes only belief=...")?)?;
                Ok(Agent::Bot(Bot::Value(Value::defensive_with(belief))))
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
            Agent::PimcSampled { config, sampler } => {
                let config = PimcConfig {
                    seed: rng.gen(),
                    ..config.clone()
                };
                pimc_sampled(state, &config, sampler).choose(config.z)
            }
            Agent::PimcAnchored {
                config,
                sampler,
                anchor,
                rollout,
            } => {
                let config = PimcConfig {
                    seed: rng.gen(),
                    ..config.clone()
                };
                pimc_anchored(state, &config, sampler, anchor, rollout).choose(config.z)
            }
            Agent::NetPimc {
                net,
                config,
                net_rollouts,
                sampler,
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
                    pimc_with_sampler(state, &config, sampler, candidates, baseline, net)
                } else {
                    pimc_with_sampler(
                        state,
                        &config,
                        sampler,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pimc_belief_options_parse_in_either_order() {
        assert!(matches!(
            Agent::parse("pimc:samples=8"),
            Ok(Agent::Pimc(PimcConfig { samples: 8, .. }))
        ));
        assert!(Agent::parse("pimc:weights=1").is_err());
        assert!(Agent::parse("pimc:belief=no/such/file.mjnn").is_err());
        let model = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/belief.mjnn");
        if !std::path::Path::new(model).exists() {
            return;
        }
        for (spec, want) in [
            (format!("pimc:weights=0,belief={model}"), false),
            (format!("pimc:belief={model}"), true),
            (format!("pimc:samples=64,belief={model},weights=1"), true),
        ] {
            match Agent::parse(&spec) {
                Ok(Agent::PimcSampled { sampler, .. }) => {
                    assert_eq!(sampler.weighted(), want, "{spec}")
                }
                other => panic!("{spec}: {other:?}"),
            }
        }
        assert!(Agent::parse(&format!("pimc:belief={model},weights=2")).is_err());
    }

    #[test]
    fn pimc_anchor_and_rollout_options() {
        // The defaults keep the plain agents (and so their play).
        assert!(matches!(
            Agent::parse("pimc:samples=8,anchor=value,rollout=fast"),
            Ok(Agent::Pimc(_))
        ));
        assert!(Agent::parse("pimc:anchor=net").is_err());
        assert!(Agent::parse("pimc:rollout=net").is_err());
        assert!(Agent::parse("pimc:ev_floor=none").is_err());
        assert!(Agent::parse("pimc:anchor=ev,ev_floor=max").is_err());
        assert!(Agent::parse("pimc:anchor=ev,ev_nope=1").is_err());
        // netpimc keeps its own rollout= values.
        assert!(Agent::parse("netpimc:rollout=ev").is_err());
        let ev_of = |b: &Bot| match b {
            Bot::Ev(e) => Some(e.config.clone()),
            _ => None,
        };
        match Agent::parse("pimc:anchor=ev,samples=8") {
            Ok(Agent::PimcAnchored {
                anchor, rollout, ..
            }) => {
                let c = ev_of(&anchor).expect("ev anchor");
                assert!(!c.pay_floor, "floor=none by default");
                assert!(format!("{rollout:?}").contains("Fast"));
            }
            other => panic!("{other:?}"),
        }
        match Agent::parse("pimc:rollout=ev,ev_floor=target,ev_horizon=6") {
            Ok(Agent::PimcAnchored {
                anchor, rollout, ..
            }) => {
                assert!(matches!(*anchor, Bot::Value(_)));
                assert!(format!("{rollout:?}").contains("horizon: 6"));
                assert!(format!("{rollout:?}").contains("pay_floor: true"));
            }
            other => panic!("{other:?}"),
        }
        let model = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/belief.mjnn");
        if !std::path::Path::new(model).exists() {
            return;
        }
        let spec = format!("pimc:samples=64,candidates=4,belief={model},anchor=ev,rollout=ev");
        match Agent::parse(&spec) {
            Ok(Agent::PimcAnchored {
                sampler, anchor, ..
            }) => {
                assert!(sampler.weighted());
                assert!(format!("{anchor:?}").contains("BeliefModel"));
            }
            other => panic!("{spec}: {other:?}"),
        }
    }
}
