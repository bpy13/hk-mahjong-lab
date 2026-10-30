//! Interfaces between a game engine and the solvers in `mahjong-search`.
//!
//! The central trait is [`Game`]: a mutable state that knows whose decision it is,
//! which actions are legal, how to apply one and how the game was scored.
//!
//! Mahjong has two features this interface makes explicit:
//!
//! * **Chance is inside the state.** The shuffled wall is part of the state, so
//!   applying an action may also draw tiles. Solvers never see the wall directly:
//!   they call [`Game::determinize`] to get a state in which everything hidden from
//!   one player has been re-dealt at random.
//! * **Simultaneous claims are sequenced but hidden.** After a discard, several
//!   players declare claims at once (rules §4.1). The engine asks them one at a time
//!   (so [`Game::current_player`] is always a single seat), but a seat's declaration
//!   is private until the claim window closes: it never appears in another seat's
//!   observation, and `determinize` forgets other seats' pending declarations.

use rand::Rng;
use std::fmt::Debug;
use std::hash::Hash;

/// Index of a seat at the table, `0..num_players()`.
pub type Player = usize;

/// The result of a finished game.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    /// Points per player. Sums to zero for zero-sum games.
    pub scores: Vec<f64>,
    /// The players who won (mahjong: possibly several on one discard, or none
    /// after an exhaustive draw).
    pub winners: Vec<Player>,
}

/// A game state that can be stepped forward.
///
/// States are cheap to clone; search algorithms clone them constantly.
pub trait Game: Clone + Send + Sync + Debug {
    /// A decision a player can make.
    type Action: Copy + Eq + Hash + Debug + Send + Sync;

    /// Number of seats.
    fn num_players(&self) -> usize;

    /// Whose decision it is, or `None` if the game is over.
    fn current_player(&self) -> Option<Player>;

    /// Writes the legal actions for the current player into `out` (cleared first).
    /// Leaves `out` empty when the game is over.
    fn legal_actions(&self, out: &mut Vec<Self::Action>);

    /// Applies a legal action for the current player, including any chance events
    /// (tile draws) that follow from it.
    ///
    /// Implementations may panic on illegal actions; callers that accept actions
    /// from users should validate them against [`Game::legal_actions`] first.
    fn apply(&mut self, action: Self::Action);

    /// The result, or `None` while the game is still running.
    fn outcome(&self) -> Option<Outcome>;

    /// The largest absolute score any player can get. Used to scale scores for
    /// search.
    fn max_abs_score(&self) -> f64;

    /// Samples a complete state that `observer` cannot tell apart from this one:
    /// the observer's private information and all public information are kept;
    /// everything hidden from the observer (other hands, the wall order, other
    /// seats' pending claim declarations) is re-dealt at random.
    fn determinize<R: Rng + ?Sized>(&self, observer: Player, rng: &mut R) -> Self;

    /// `true` once the game is over.
    fn is_terminal(&self) -> bool {
        self.current_player().is_none()
    }

    /// Convenience wrapper around [`Game::legal_actions`].
    fn legal_actions_vec(&self) -> Vec<Self::Action> {
        let mut v = Vec::new();
        self.legal_actions(&mut v);
        v
    }
}

/// Something that picks an action in a state: a bot, a rollout policy inside a
/// search, or a learned network.
///
/// A policy must only use what the current player can see. Implementations get
/// the full state for speed; the engine's observation helpers say which parts
/// are public.
pub trait Policy<G: Game>: Send + Sync {
    /// Picks one of the `legal` actions (never empty) for the current player.
    fn choose<R: Rng + ?Sized>(&self, state: &G, legal: &[G::Action], rng: &mut R) -> G::Action;
}

/// Picks uniformly at random among legal actions.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformRandom;

impl<G: Game> Policy<G> for UniformRandom {
    fn choose<R: Rng + ?Sized>(&self, _state: &G, legal: &[G::Action], rng: &mut R) -> G::Action {
        // A u32 draw: rand 0.8 reads 32 bits for a usize range on wasm32 and 64 on
        // native, so a usize draw would pick different moves in the app (hk-games D28).
        legal[rng.gen_range(0..legal.len() as u32) as usize]
    }
}

/// Plays `state` to the end with `policy` for every seat and returns the outcome.
pub fn play_out<G: Game, P: Policy<G>, R: Rng + ?Sized>(
    state: &mut G,
    policy: &P,
    rng: &mut R,
) -> Outcome {
    let mut legal = Vec::new();
    while !state.is_terminal() {
        state.legal_actions(&mut legal);
        let a = policy.choose(state, &legal, rng);
        state.apply(a);
    }
    state.outcome().expect("terminal state has an outcome")
}
