//! How well the value head predicts rollouts at `netpimc`'s leaf, for two leaf
//! rules (plan Task 16, `docs/study-results/phase4-build.md` §3):
//!
//! * `exact`: stop after exactly `depth` discards, whoever is to act (the first
//!   version of `leaf=value`), reading the value from that player's view;
//! * `own turn`: stop at the searching seat's first turn after `depth` discards
//!   ([`LeafValue::run_to_leaf`], the fixed rule).
//!
//! For positions from hands played by `value`, it runs to the leaf on the true
//! world, reads the value for the seat that was to act, and compares it with
//! the mean of 32 `fast` playouts from that leaf. Prints how often the leaf is
//! the searching seat's own turn, the correlation, the regression slope of the
//! rollouts on the value, and both means.
//!
//! `cargo run --release -p mahjong-search --example leaf_check -- models/cnn-distill.mjnn 2000`
use mahjong_games::{play_out, Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::{Bot, LeafValue, Mlp, Rollout};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::sync::Arc;

const DEPTH: usize = 6;
const PLAYOUTS: u64 = 32;

/// A live decision with a choice, from a hand played by `value`.
fn position(seed: u64) -> Option<State> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let bot = Bot::by_name("value").unwrap();
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    for _ in 0..rng.gen_range(5..40) {
        if s.is_terminal() {
            return None;
        }
        let legal = s.legal_actions_vec();
        let a = bot.choose(&s, &legal, &mut rng);
        s.apply(a);
    }
    (!s.is_terminal() && s.legal_actions_vec().len() > 1).then_some(s)
}

/// The old rule: stops after exactly `depth` discards; `false` if the hand ended.
fn run_exact(s: &mut State, rollout: &Rollout, rng: &mut SmallRng, depth: usize) -> bool {
    let mut discards = 0;
    while !s.is_terminal() {
        if discards >= depth {
            return true;
        }
        let legal = s.legal_actions_vec();
        let a = rollout.choose(s, &legal, rng);
        discards += usize::from(matches!(a, Action::Discard(_)));
        s.apply(a);
    }
    false
}

/// Mean of `fast` playouts from `leaf`, for `seat`.
fn rollout_mean(leaf: &State, rollout: &Rollout, seat: usize, seed: u64) -> f64 {
    let mut total = 0.0;
    for i in 0..PLAYOUTS {
        let mut rng = SmallRng::seed_from_u64(seed ^ (i + 1).wrapping_mul(0x9E37_79B9));
        let mut w = leaf.clone();
        play_out(&mut w, rollout, &mut rng);
        total += w.result().expect("hand over").deltas[seat] as f64;
    }
    total / PLAYOUTS as f64
}

#[derive(Default)]
struct Rule {
    own_turn: usize,
    /// (value head's points, mean of playouts).
    pairs: Vec<(f64, f64)>,
}

impl Rule {
    fn report(&self, name: &str) {
        let n = self.pairs.len().max(1) as f64;
        let (mx, my) = self
            .pairs
            .iter()
            .fold((0.0, 0.0), |(a, b), p| (a + p.0 / n, b + p.1 / n));
        let (mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0);
        for (x, y) in &self.pairs {
            xx += (x - mx) * (x - mx);
            yy += (y - my) * (y - my);
            xy += (x - mx) * (y - my);
        }
        println!(
            "{name}: {} leaves, own turn {:.1} %, correlation {:.3}, slope {:.2}, \
             value sd {:.2}, rollouts sd {:.2}, mean value {:+.2}, mean rollouts {:+.2}",
            self.pairs.len(),
            100.0 * self.own_turn as f64 / n,
            xy / (xx * yy).sqrt(),
            xy / xx,
            (xx / n).sqrt(),
            (yy / n).sqrt(),
            mx,
            my
        );
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("path to an MJNN v3 file");
    let n: u64 = std::env::args()
        .nth(2)
        .map_or(2000, |s| s.parse().expect("number of seeds"));
    let net = Arc::new(Mlp::load(&path).unwrap());
    let leaf = LeafValue::new(net, DEPTH).unwrap();
    let rows: Vec<_> = (0..n)
        .into_par_iter()
        .filter_map(|seed| {
            let s = position(seed)?;
            let seat = s.current_player()?;
            let rollout = Rollout::for_seat(seat);
            let mut rng = SmallRng::seed_from_u64(seed ^ 0xABCD);
            let mut old = s.clone();
            let a = run_exact(&mut old, &rollout, &mut rng, DEPTH).then(|| {
                (
                    old.current_player() == Some(seat),
                    leaf.points(&old, seat),
                    rollout_mean(&old, &rollout, seat, seed),
                )
            });
            let mut new = s;
            let b = leaf
                .run_to_leaf(&mut new, &rollout, &mut rng, seat)
                .then(|| {
                    (
                        new.current_player() == Some(seat),
                        leaf.points(&new, seat),
                        rollout_mean(&new, &rollout, seat, seed),
                    )
                });
            Some((a, b))
        })
        .collect();
    let (mut exact, mut own) = (Rule::default(), Rule::default());
    for (a, b) in rows {
        for (rule, r) in [(&mut exact, a), (&mut own, b)] {
            if let Some((turn, v, m)) = r {
                rule.own_turn += usize::from(turn);
                rule.pairs.push((v, m));
            }
        }
    }
    exact.report("exact");
    own.report("own turn");
}
