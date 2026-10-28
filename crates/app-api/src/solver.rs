//! Thin wrappers over the engine's PIMC solver, for hints, the expert bot, the
//! review and drill grading. Every search runs on one thread (the app runs in
//! WebAssembly, where no thread pool exists) and within a wall-clock budget.
//!
//! Grade thresholds live here too (`grade_for`), so hint, review and grade
//! agree on what a "mistake" is.

use mahjong_games::Game;
use mahjong_hk::{Action, State};
use mahjong_search::{pimc_with, shortlist, Analysis, PimcConfig, Rollout};
use std::time::Duration;

/// Determinizations of a hint when the budget allows (the lab UI's default).
pub const HINT_SAMPLES: usize = 400;
/// Discards the hint shortlists (special actions are always included).
pub const HINT_CANDIDATES: usize = 6;
/// Determinizations per reviewed decision when the budget allows (as the lab UI).
pub const REVIEW_SAMPLES: usize = 300;
pub const REVIEW_CANDIDATES: usize = 8;
/// The expert bot: the lab UI's hard level (`pimc:samples=96,candidates=4`).
pub const EXPERT_SAMPLES: usize = 96;
pub const EXPERT_CANDIDATES: usize = 4;
/// How many standard errors the expert bot needs before it leaves the heuristic
/// baseline ([`Analysis::choose`]).
pub const EXPERT_Z: f64 = 2.0;

/// The share of `budget_ms` handed to the search. PIMC checks the clock only
/// between batches of samples, so the rest covers the last batch and the study
/// extras; the contract allows `budget + 20%`.
const BUDGET_SHARE: f64 = 0.8;

/// The time limit to give PIMC for a call budget (`None` when `budget_ms == 0`).
pub fn search_budget(budget_ms: u32) -> Option<Duration> {
    (budget_ms > 0).then(|| Duration::from_millis((f64::from(budget_ms) * BUDGET_SHARE) as u64))
}

/// Runs `f` with rayon limited to one thread. Natively this is a one-thread pool
/// built once; in the browser rayon's global pool already runs on the calling
/// thread (rayon-core falls back when threads cannot be spawned).
#[cfg(not(target_arch = "wasm32"))]
pub fn single_thread<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    use std::sync::OnceLock;
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    let pool = POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("a one-thread pool")
    });
    pool.install(f)
}

#[cfg(target_arch = "wasm32")]
pub fn single_thread<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    f()
}

/// PIMC for the player to move: the engine's shortlist (every special action,
/// the best `candidates` discards and the heuristic baseline) plus `extra`
/// (the move actually played, so a review can score it on the same samples),
/// up to `samples` determinizations within `time_limit`.
pub fn analysis(
    state: &State,
    samples: usize,
    candidates: usize,
    time_limit: Option<Duration>,
    seed: u64,
    extra: Option<Action>,
) -> Analysis {
    let seat = state.current_player().expect("the hand is not over");
    let (mut list, baseline) = shortlist(state, candidates.max(1));
    if let Some(a) = extra {
        if !list.contains(&a) {
            list.push(a);
        }
    }
    let config = PimcConfig {
        samples,
        candidates,
        time_limit,
        seed,
        z: EXPERT_Z,
    };
    single_thread(|| pimc_with(state, &config, list, baseline, &Rollout::for_seat(seat)))
}

/// Grade buckets in expected points (the solver's `mean`): `good` under 1 point,
/// `inaccuracy` under 3, `mistake` under 8 (one share of a 3-faan win under
/// half-spicy payout), `blunder` from 8. `best` is the solver's own first choice.
pub const GRADE_THRESHOLDS: [(&str, f64); 3] =
    [("good", 1.0), ("inaccuracy", 3.0), ("mistake", 8.0)];

/// The grade of a move that loses `loss` expected points against the best move.
pub fn grade_for(is_best: bool, loss: f64) -> &'static str {
    if is_best || loss <= 0.0 {
        return "best";
    }
    for (grade, limit) in GRADE_THRESHOLDS {
        if loss < limit {
            return grade;
        }
    }
    "blunder"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grades_follow_the_thresholds() {
        assert_eq!(grade_for(true, 5.0), "best");
        assert_eq!(grade_for(false, 0.0), "best");
        assert_eq!(grade_for(false, 0.5), "good");
        assert_eq!(grade_for(false, 2.0), "inaccuracy");
        assert_eq!(grade_for(false, 5.0), "mistake");
        assert_eq!(grade_for(false, 8.0), "blunder");
    }

    #[test]
    fn budget_leaves_slack() {
        assert_eq!(search_budget(0), None);
        assert_eq!(search_budget(1000), Some(Duration::from_millis(800)));
    }
}
