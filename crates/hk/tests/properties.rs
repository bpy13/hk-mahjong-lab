//! Invariants over many random hands under random rule variants.

use mahjong_games::Game;
use mahjong_hk::{Action, MultipleWinners, Payout, RuleSet, State};
use mahjong_tiles::Tile;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn random_rules(rng: &mut SmallRng) -> RuleSet {
    RuleSet {
        flowers: rng.gen_bool(0.5),
        min_faan: rng.gen_range(0..=3),
        faan_cap: if rng.gen_bool(0.5) { 10 } else { 13 },
        payout: if rng.gen_bool(0.5) {
            Payout::HalfSpicy
        } else {
            Payout::FullSpicy
        },
        multiple_winners: if rng.gen_bool(0.5) {
            MultipleWinners::All
        } else {
            MultipleWinners::HeadBump
        },
        twelve_tile_liability: rng.gen_bool(0.5),
        prevailing_wind: rng.gen_range(0..4),
    }
}

/// Picks randomly, but takes wins and claims most of the time so that hands
/// with many melds, kongs and wins are common.
fn eager(legal: &[Action], rng: &mut SmallRng) -> Action {
    let pref = |a: &Action| match a {
        Action::Win | Action::SelfDrawnWin => 0,
        Action::ConcealedKong(_) | Action::AddedKong(_) | Action::ClaimKong => 1,
        Action::Pung | Action::Chow(_) => 2,
        _ => 3,
    };
    if rng.gen_bool(0.85) {
        let best = legal.iter().map(pref).min().unwrap();
        if best < 3 {
            let picks: Vec<_> = legal.iter().filter(|a| pref(a) == best).collect();
            return *picks[rng.gen_range(0..picks.len())];
        }
    }
    legal[rng.gen_range(0..legal.len())]
}

/// Every tile is somewhere: hands, melds, the river, the wall, bonus tiles, or (a
/// robbed kong's tile) only in the win record.
fn assert_conserved(s: &State) {
    let mut all: Vec<Tile> = Vec::new();
    for seat in 0..4 {
        all.extend(s.hand(seat).tiles());
        for m in s.melds(seat).iter() {
            all.extend(m.tiles());
        }
        all.extend(Tile::all_bonus().filter(|b| s.bonus(seat) & (1 << (b.index() - 34)) != 0));
    }
    all.extend(s.river().into_iter().filter(|r| !r.2).map(|r| r.1));
    all.extend_from_slice(s.wall());
    // A robbed kong's tile is held only in the win record.
    if let Some(w) = s.result().and_then(|r| r.wins.iter().find(|w| w.robbed)) {
        all.push(w.win_tile.unwrap());
    }
    all.sort();
    assert_eq!(
        all,
        State::all_tiles(s.rules()),
        "tiles not conserved\n{s:?}"
    );
}

#[test]
fn random_hands_respect_the_invariants() {
    let mut rng = SmallRng::seed_from_u64(2026);
    let mut wins = 0;
    let mut draws = 0;
    let mut liability = 0;
    for _ in 0..3_000 {
        let rules = random_rules(&mut rng);
        let mut s = State::new_deal(rules, &mut rng);
        let mut steps = 0;
        while !s.is_terminal() {
            let legal = s.legal_actions_vec();
            assert!(!legal.is_empty());
            if s.in_claim() && s.wall_remaining() == 0 {
                // §4.1.7: the last discard can only be claimed for a win.
                assert!(legal
                    .iter()
                    .all(|a| matches!(a, Action::Win | Action::Pass)));
            }
            if s.in_rob() {
                assert!(legal
                    .iter()
                    .all(|a| matches!(a, Action::Win | Action::Pass)));
            }
            let a = eager(&legal, &mut rng);
            s.apply(a);
            steps += 1;
            assert!(steps < 1_000, "hand does not terminate");
            if steps % 7 == 0 {
                assert_conserved(&s);
            }
        }
        assert_conserved(&s);
        let r = s.result().unwrap();
        assert_eq!(r.deltas.iter().sum::<i32>(), 0, "not zero-sum");
        if r.wins.is_empty() {
            draws += 1;
            assert_eq!(r.deltas, [0; 4]);
        }
        for w in &r.wins {
            wins += 1;
            assert!(w.score.meets_minimum(&rules));
            assert!(w.score.faan >= w.score.qualifying);
            assert!(w.score.faan <= rules.faan_cap);
            if let Some(l) = w.liable {
                liability += 1;
                assert_eq!(w.paid_by.iter().filter(|&&p| p != 0).count(), 1);
                assert_eq!(w.paid_by[l], 3 * rules.points(w.score.faan) as i32);
            }
        }
    }
    // The eager policy should reach every kind of ending.
    assert!(
        wins > 100 && draws > 100 && liability > 5,
        "wins {wins} draws {draws} liability {liability}"
    );
}

#[test]
fn determinize_keeps_the_observers_view() {
    let mut rng = SmallRng::seed_from_u64(99);
    for _ in 0..300 {
        let rules = random_rules(&mut rng);
        let mut s = State::new_deal(rules, &mut rng);
        let stop = rng.gen_range(0..80);
        for _ in 0..stop {
            if s.is_terminal() {
                break;
            }
            let legal = s.legal_actions_vec();
            s.apply(eager(&legal, &mut rng));
        }
        if s.is_terminal() {
            continue;
        }
        let me = s.current_player().unwrap();
        let d = s.determinize(me, &mut rng);
        assert_eq!(d.current_player(), Some(me));
        assert_eq!(d.hand(me), s.hand(me));
        assert_eq!(d.legal_actions_vec(), s.legal_actions_vec());
        assert_eq!(d.wall_remaining(), s.wall_remaining());
        assert_eq!(d.events(), s.events());
        for seat in 0..4 {
            assert_eq!(
                d.melds(seat).iter().collect::<Vec<_>>(),
                s.melds(seat).iter().collect::<Vec<_>>()
            );
            assert_eq!(d.bonus(seat), s.bonus(seat));
            assert_eq!(d.concealed_count(seat), s.concealed_count(seat));
        }
        assert_eq!(d.unseen(me), s.unseen(me));
        assert_conserved(&d);
        // The re-dealt state can be played to the end.
        let mut d = d;
        while !d.is_terminal() {
            let legal = d.legal_actions_vec();
            d.apply(eager(&legal, &mut rng));
        }
    }
}
