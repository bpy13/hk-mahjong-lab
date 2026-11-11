//! Task 20: how often does the race bot pick the pure-offence discard (`rank_discards`)
//! in early positions where every opponent is most likely two or more tiles from
//! winning (P(ready) + P(one away) < 0.5 for each)? Brian's bar: at least 95 %.
//!
//! ```text
//! race_offence --spec "race:belief=models/belief.mjnn" --positions 1000 --seed 5
//! ```
use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::race::offence_discard;
use mahjong_search::{Agent, Bot, SeatView};
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn main() {
    let mut spec = "race:belief=models/belief.mjnn".to_string();
    let mut want = 1000usize;
    let mut seed = 5u64;
    let mut min_wall = 70usize;
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let v = it.next().expect("value");
        match k.as_str() {
            "--spec" => spec = v,
            "--positions" => want = v.parse().unwrap(),
            "--seed" => seed = v.parse().unwrap(),
            "--min-wall" => min_wall = v.parse().unwrap(),
            _ => panic!("unknown {k}"),
        }
    }
    let race = Agent::parse(&spec).expect("spec");
    let Agent::Bot(Bot::Race(bot)) = &race else {
        panic!("--spec must be a race bot")
    };
    let value = Agent::parse("value").unwrap();
    let (mut n, mut same, mut hand) = (0usize, 0usize, 0u64);
    let mut legal = Vec::new();
    let mut shown = 0;
    while n < want {
        let mut rng = SmallRng::seed_from_u64(seed.wrapping_mul(1_000_003).wrapping_add(hand));
        hand += 1;
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let seat = (hand % 4) as usize;
        while let Some(me) = s.current_player() {
            s.legal_actions(&mut legal);
            let a = if me == seat && legal.len() > 1 {
                let d = bot.decide(&s);
                if !s.in_claim()
                    && !s.in_rob()
                    && s.wall_remaining() >= min_wall
                    && n < want
                    && d.candidates.first().is_some_and(|c| {
                        (1..4).all(|r| {
                            let o = (me + r) % 4;
                            c.tenpai[o] + c.one_away[o] < 0.5
                        })
                    })
                {
                    if let (Some(off), Action::Discard(t)) =
                        (offence_discard(&SeatView::of(&s, me), &legal), d.action)
                    {
                        n += 1;
                        if off == t {
                            same += 1;
                        } else if shown < 8 {
                            shown += 1;
                            println!(
                                "diff: race {t:?} offence {off:?} wall {}",
                                s.wall_remaining()
                            );
                            for c in d
                                .candidates
                                .iter()
                                .filter(|c| c.discard == Some(t) || c.discard == Some(off))
                            {
                                println!("   {:?} eq {:.3} gain {:.3} sd {:.3} mine {:.3} other {:.3} dealnow {:?} pwin {:?} sh {} acc {:.0} tenpai {:?} one {:?}", c.discard, c.equity, c.gain, c.loss_self_draw, c.loss_my_feeds, c.loss_other_feeds, c.deal_in_now, c.p_win, c.faan_shanten, c.acceptance, c.tenpai, c.one_away);
                            }
                        }
                    }
                }
                d.action
            } else {
                value.choose(&s, &legal, &mut rng)
            };
            s.apply(a);
        }
    }
    println!(
        "positions {n}, same discard {same} ({:.1} %)",
        100.0 * same as f64 / n as f64
    );
}
