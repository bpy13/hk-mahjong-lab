//! Bots and the solver: legality, no peeking at hidden tiles, obvious choices.

use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::{pimc, Agent, Bot, Danger, PimcConfig};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn advance(rng: &mut SmallRng, steps: usize) -> State {
    let bot = Bot::by_name("fast").unwrap();
    loop {
        let mut s = State::new_deal(RuleSet::default(), rng);
        for _ in 0..steps {
            if s.is_terminal() {
                break;
            }
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, rng);
            s.apply(a);
        }
        if !s.is_terminal() {
            return s;
        }
    }
}

#[test]
fn bots_play_legal_moves_and_take_wins() {
    let mut rng = SmallRng::seed_from_u64(1);
    for name in ["random", "efficiency", "value", "defensive", "fast"] {
        let bot = Bot::by_name(name).unwrap();
        for _ in 0..20 {
            let mut s = State::new_deal(RuleSet::default(), &mut rng);
            while !s.is_terminal() {
                let legal = s.legal_actions_vec();
                let a = bot.choose(&s, &legal, &mut rng);
                assert!(legal.contains(&a), "{name} chose illegal {a}");
                if legal
                    .iter()
                    .any(|x| matches!(x, Action::Win | Action::SelfDrawnWin))
                {
                    assert!(
                        matches!(a, Action::Win | Action::SelfDrawnWin),
                        "{name} declined a win"
                    );
                }
                s.apply(a);
            }
        }
    }
}

#[test]
fn decisions_do_not_depend_on_hidden_tiles() {
    let mut rng = SmallRng::seed_from_u64(2);
    for _ in 0..12 {
        let steps = rng.gen_range(0..60);
        let s = advance(&mut rng, steps);
        let me = s.current_player().unwrap();
        let other = s.determinize(me, &mut rng);
        let legal = s.legal_actions_vec();
        for name in ["efficiency", "value", "defensive", "fast"] {
            let bot = Bot::by_name(name).unwrap();
            let a = bot.choose(&s, &legal, &mut SmallRng::seed_from_u64(0));
            let b = bot.choose(&other, &legal, &mut SmallRng::seed_from_u64(0));
            assert_eq!(a, b, "{name} peeked");
        }
        let config = PimcConfig {
            samples: 24,
            candidates: 3,
            seed: 9,
            ..PimcConfig::default()
        };
        let x = pimc(&s, &config);
        let y = pimc(&other, &config);
        let key = |a: &mahjong_search::Analysis| {
            a.moves
                .iter()
                .map(|m| (m.action, m.mean.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(key(&x), key(&y), "the solver peeked");
    }
}

#[test]
fn danger_is_zero_for_tiles_with_no_copies_left() {
    let mut rng = SmallRng::seed_from_u64(3);
    let s = advance(&mut rng, 80);
    let me = s.current_player().unwrap();
    let d = Danger::estimate(&s, me);
    let unseen = s.unseen(me);
    for t in mahjong_tiles::Tile::all_kinds() {
        if unseen[t] == 0 {
            assert_eq!(d.tile(t), 0.0);
        }
        assert!((0.0..=1.0).contains(&d.tile(t)));
    }
}

#[test]
fn agent_specs_parse() {
    assert!(matches!(Agent::parse("value"), Ok(Agent::Bot(_))));
    assert!(matches!(
        Agent::parse("pimc:samples=10,candidates=2,ms=50"),
        Ok(Agent::Pimc(_))
    ));
    assert!(Agent::parse("pimc:depth=3").is_err());
    assert!(Agent::parse("nobody").is_err());
}
