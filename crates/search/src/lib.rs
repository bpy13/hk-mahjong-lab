//! Players and solvers for Hong Kong mahjong.
//!
//! * [`agent`]: agents by spec string, and duplicate tournaments.
//! * [`bots`]: rule-based players (random, efficiency, value, defensive, fast).
//! * [`danger`]: a heuristic deal-in danger model.
//! * [`pimc`]: the Monte Carlo solver used for recommendations and study mode.
//! * [`solo`]: offense-only estimates for each discard.
//! * [`view`]: what one seat can see.
//! * [`world`]: structured sampling of hidden tiles for search.

pub mod agent;
pub mod bots;
pub mod danger;
pub mod pimc;
pub mod solo;
pub mod view;
pub mod world;

pub use agent::{duplicate, Agent, AgentResult, TournamentResult};
pub use bots::{Bot, Efficiency, Fast, RandomBot, Value};
pub use danger::Danger;
pub use pimc::{pimc, shortlist, ActionStats, Analysis, PimcConfig};
pub use solo::{solo, SoloStats};
pub use view::SeatView;
pub use world::sample_world;
