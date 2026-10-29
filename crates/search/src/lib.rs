//! Players and solvers for Hong Kong mahjong.
//!
//! * [`agent`]: agents by spec string.
//! * [`tournament`]: duplicate tournaments (4 rotations or all 24 seatings).
//! * [`belief`]: the trained belief model (MJNN v2), cached per state.
//! * [`bots`]: rule-based players (random, efficiency, value, defensive, fast).
//! * [`cnn`]: the suit-shared 1-D CNN (MJNN v3, plan Task 14).
//! * [`danger`]: a deal-in danger model fitted to simulated games ([`danger_tables`]).
//! * [`ev`]: Brian's explicit EV bot (B15, `docs/ev-bot.md`) and the [`ev::Belief`]
//!   interface for opponent models.
//! * [`features`]: the observation encoding for learning.
//! * [`labels`]: search-labelled decisions for distillation (plan Task 13).
//! * [`log`]: self-play logs with hidden hands (belief-model training data).
//! * [`net`]: running trained networks.
//! * [`pimc`]: the Monte Carlo solver used for recommendations and study mode.
//! * [`solo`]: offense-only estimates for each discard.
//! * [`target_value`]: P(win), P(self-draw | win) and points by target plan
//!   ([`target_tables`]).
//! * [`view`]: what one seat can see.
//! * [`world`]: structured sampling of hidden tiles for search.

pub mod agent;
pub mod belief;
pub mod bots;
pub mod cnn;
pub mod danger;
pub mod danger_tables;
pub mod ev;
pub mod features;
pub mod labels;
pub mod log;
pub mod net;
pub mod pimc;
pub mod solo;
pub mod target_tables;
pub mod target_value;
pub mod tournament;
pub mod view;
pub mod world;

pub use agent::Agent;
pub use belief::{BeliefModel, BeliefOutput, OpponentBelief};
pub use bots::{Bot, Efficiency, Fast, RandomBot, Value};
pub use danger::Danger;
pub use ev::{
    Belief, Beliefs, EvBot, EvConfig, EvDecision, EvReason, EvTerms, HeuristicBelief, SeatBelief,
};
pub use net::{Mlp, NetPolicy};
pub use pimc::{
    pimc, pimc_anchored, pimc_with, shortlist, shortlist_anchored, ActionStats, Analysis,
    PimcConfig, Rollout,
};
pub use solo::{solo, SoloStats};
pub use tournament::{
    duplicate, tournament, AgentResult, Seatings, Summary, Tournament, TournamentConfig,
    TournamentResult,
};
pub use view::SeatView;
pub use world::{sample_world, WorldSampler};
