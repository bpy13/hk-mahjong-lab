//! Hong Kong mahjong rules engine. The rules are specified in `docs/rules/hk.md`;
//! section numbers in the docs and tests (`§4.1.3`, `#17`) refer to it.
//!
//! * [`rules`]: rule variants ([`RuleSet`]) and the payout table.
//! * [`meld`]: chows, pungs and kongs on the table.
//! * [`hand`]: the win check and the ways to split a hand into melds.
//! * [`faan`]: the faan table and [`score_win`].
//! * [`state`]: the hand as a state machine implementing [`mahjong_games::Game`].

pub mod faan;
pub mod hand;
pub mod meld;
pub mod rules;
pub mod state;

pub use faan::{score_win, Pattern, Score, WinContext};
pub use meld::{Meld, MeldKind};
pub use rules::{MultipleWinners, Payout, RuleSet};
pub use state::{Action, Event, HandResult, State, WinRecord, DEALER};
