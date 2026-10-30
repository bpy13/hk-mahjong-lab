//! Speed of the learning encoding (`features::encode`) on one thread, and the
//! cost of its expensive parts computed the straightforward way (every tile
//! tried in each acceptance count; `encode` itself skips the tiles that cannot
//! count), on positions from self-play with varied rules.
//!
//! 2026-10-30, under load from other jobs: `encode` 653 µs before the speed-up,
//! 76 µs after (shared work in `faan_shanten`, a cheap cache hasher, skipped
//! tiles), with identical output (`features::tests::encoding_matches_the_golden_hash*`).
//!
//! `cargo run --release -p mahjong-search --example encode_speed`
use mahjong_analysis::{acceptance, faan_shanten, shanten, HandView, UNREACHABLE};
use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::features::{encode, encode_oracle, ORACLE_LEN, STATE_LEN};
use mahjong_search::view::SeatView;
use mahjong_search::{Bot, Danger};
use mahjong_tiles::{Tile, NUM_KINDS};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::Instant;

fn positions(hands: usize) -> Vec<State> {
    let mut rng = SmallRng::seed_from_u64(5);
    let bot = Bot::by_name("fast").unwrap();
    let mut out = Vec::new();
    for h in 0..hands {
        let rules = RuleSet {
            min_faan: [0, 1, 3][h % 3],
            ..RuleSet::default()
        };
        let mut s = State::new_deal(rules, &mut rng);
        while !s.is_terminal() {
            out.push(s.clone());
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
            let _ = rng.gen::<u8>();
        }
    }
    out
}

fn time(label: &str, n: usize, f: impl FnOnce()) {
    let start = Instant::now();
    f();
    let per = start.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!("{label:<40} {per:>8.1} us per position");
}

fn main() {
    let states = positions(300);
    let n = states.len();
    let turns = states.iter().filter(|s| s.phase_name() == "turn").count();
    println!("{n} positions ({turns} turn decisions)");
    let mut f = vec![0.0; STATE_LEN];
    let mut o = vec![0.0; ORACLE_LEN];
    time("encode", n, || {
        for s in &states {
            encode(s, &mut f);
            black_box(&f);
        }
    });
    time("encode_oracle", n, || {
        for s in &states {
            encode_oracle(s, &mut o);
            black_box(&o);
        }
    });
    time("legal actions", n, || {
        for s in &states {
            black_box(s.legal_actions_vec());
        }
    });
    time("  Danger::estimate", n, || {
        for s in &states {
            black_box(Danger::estimate(s, s.current_player().unwrap()));
        }
    });
    time("  SeatView::of", n, || {
        for s in &states {
            black_box(SeatView::of(s, s.current_player().unwrap()));
        }
    });
    let views: Vec<(SeatView, bool)> = states
        .iter()
        .map(|s| {
            (
                SeatView::of(s, s.current_player().unwrap()),
                s.phase_name() == "turn",
            )
        })
        .collect();
    time("  per discard: shanten + acceptance", n, || {
        for (v, turn) in &views {
            if !turn {
                continue;
            }
            for t in v.concealed.kinds() {
                let mut h = v.concealed;
                h.remove(t);
                black_box(shanten(&h, v.melds.len()));
                black_box(acceptance(&h, &v.unseen, |x| shanten(x, v.melds.len())));
            }
        }
    });
    time("  per discard: faan_shanten", n, || {
        for (v, turn) in &views {
            if !turn {
                continue;
            }
            for t in v.concealed.kinds() {
                let mut h = v.concealed;
                h.remove(t);
                black_box(faan_shanten(
                    &HandView {
                        concealed: &h,
                        ..v.hand_view()
                    },
                    &v.rules,
                ));
            }
        }
    });
    let mut best_count = 0usize;
    time("  best discards: faan acceptance", n, || {
        for (v, turn) in &views {
            if !turn {
                continue;
            }
            let mut after = [i8::MAX; NUM_KINDS];
            for t in v.concealed.kinds() {
                let mut h = v.concealed;
                h.remove(t);
                after[t.index() as usize] = faan_shanten(
                    &HandView {
                        concealed: &h,
                        ..v.hand_view()
                    },
                    &v.rules,
                )
                .shanten;
            }
            let best = *after.iter().min().unwrap();
            if best >= UNREACHABLE {
                continue;
            }
            for t in Tile::all_kinds().filter(|t| after[t.index() as usize] == best) {
                best_count += 1;
                let mut h = v.concealed;
                h.remove(t);
                let hv = HandView {
                    concealed: &h,
                    ..v.hand_view()
                };
                black_box(acceptance(&h, &v.unseen, |x| {
                    faan_shanten(&HandView { concealed: x, ..hv }, &v.rules).shanten
                }));
            }
        }
    });
    println!(
        "  ({:.1} best discards per turn decision; the line above includes their faan_shanten)",
        best_count as f64 / turns as f64
    );
    time("  whole-hand faan_shanten", n, || {
        for (v, _) in &views {
            black_box(faan_shanten(&v.hand_view(), &v.rules));
        }
    });
}
