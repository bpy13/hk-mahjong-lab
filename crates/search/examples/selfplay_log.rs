//! Self-play logs with hidden hands for the belief model (plan Task 4).
//!
//! ```text
//! cargo run --release -p mahjong-search --example selfplay_log -- \
//!     --hands 25000 --out runs/logs/train --seed 1 [--field train|holdout] \
//!     [--threads 14] [--chunk 500]
//! ```
//!
//! Plays `--hands` single hands and writes one row per decision (see
//! `mahjong_search::log`) in chunks of `--chunk` hands. Rerunning the same
//! command resumes after the last complete chunk; `meta.json` says how far it got.
//!
//! Fields (per hand, seats drawn with the hand's own generator):
//!
//! * `train`: each seat from {value ×3, defensive ×2, fast, efficiency}; every
//!   tenth hand one random seat plays `pimc:samples=16,candidates=4` instead;
//! * `holdout`: one each of value, closed, the PPO network
//!   (`net:path=models/ppo.mjnn`) and defensive, in a random seat order. Use a
//!   different `--seed`: hand seeds derive from `--seed` and the hand index.
//!
//! `--bestof K` (plan Task 5's best-of-K baseline) also draws `K` structured
//! determinizations (`world::sample_world`) at every decision and writes, beside
//! each chunk, `bestof-NNNNN.bin`: per row `[3][34][5]` bytes, how many of the
//! `K` samples gave relative opponent `o` 0–4 copies of tile `t`. The sampler
//! uses its own generator, so the hands played are the same as without it.

use mahjong_games::{Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::log::{capture, finish, row_bytes, LogWriter};
use mahjong_search::{sample_world, Agent};
use mahjong_tiles::{Tile, NUM_KINDS};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const TRAIN_AGENTS: &[&str] = &[
    "value",
    "defensive",
    "fast",
    "efficiency",
    "pimc:samples=16,candidates=4",
];
/// Draw weights of the non-search train agents (value ×3, defensive ×2, fast, efficiency).
const TRAIN_POOL: &[u8] = &[0, 0, 0, 1, 1, 2, 3];
const TRAIN_PIMC: u8 = 4;
const HOLDOUT_AGENTS: &[&str] = &["value", "closed", "net:path=models/ppo.mjnn", "defensive"];

struct Args {
    hands: usize,
    out: PathBuf,
    seed: u64,
    holdout: bool,
    threads: usize,
    chunk: usize,
    bestof: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        hands: 25_000,
        out: PathBuf::from("runs/logs/train"),
        seed: 1,
        holdout: false,
        threads: 14,
        chunk: 500,
        bestof: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or(format!("{k} needs a value"));
        let num = |s: String| s.parse::<u64>().map_err(|_| format!("bad number {s:?}"));
        match k.as_str() {
            "--hands" => a.hands = num(v()?)? as usize,
            "--out" => a.out = PathBuf::from(v()?),
            "--seed" => a.seed = num(v()?)?,
            "--threads" => a.threads = num(v()?)?.max(1) as usize,
            "--chunk" => a.chunk = num(v()?)?.max(1) as usize,
            "--bestof" => a.bestof = num(v()?)?.min(255) as usize,
            "--field" => {
                a.holdout = match v()?.as_str() {
                    "train" => false,
                    "holdout" => true,
                    f => return Err(format!("unknown field {f:?} (train or holdout)")),
                }
            }
            _ => return Err(format!("unknown argument {k:?}")),
        }
    }
    Ok(a)
}

/// SplitMix64: independent-looking hand seeds from (run seed, hand index).
fn hand_seed(seed: u64, hand: u64) -> u64 {
    let mut z = seed
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
        .wrapping_add(hand.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Agent index (into the field's agent list) for each seat of `hand`.
fn seat_agents(holdout: bool, hand: u64, rng: &mut SmallRng) -> [u8; 4] {
    if holdout {
        let mut ids = [0u8, 1, 2, 3];
        ids.shuffle(rng);
        return ids;
    }
    let mut ids = [0u8; 4];
    for id in &mut ids {
        *id = *TRAIN_POOL.choose(rng).expect("pool");
    }
    if hand.is_multiple_of(10) {
        ids[rng.gen_range(0..4)] = TRAIN_PIMC;
    }
    ids
}

/// Bytes per row of the best-of-K sidecar: 3 opponents × 34 kinds × 5 counts.
const BESTOF_ROW: usize = 3 * NUM_KINDS * 5;

/// A hand's encoded rows and, with `--bestof`, the sidecar bytes.
struct HandOut {
    rows: Vec<u8>,
    bestof: Vec<u8>,
}

/// Plays one hand and returns its encoded rows.
///
/// A hand with a search seat runs in its own one-thread pool. In the shared
/// pool, PIMC's inner parallel loop lets a waiting thread steal whole hands, so
/// one search decision waited on chains of stolen hands and a 500-hand chunk
/// ended with two busy cores for most of its time (5× slower overall).
fn play_hand(agents: &[Agent], holdout: bool, seed: u64, hand: u64, bestof: usize) -> HandOut {
    let mut rng = SmallRng::seed_from_u64(hand_seed(seed, hand));
    let ids = seat_agents(holdout, hand, &mut rng);
    if ids
        .iter()
        .any(|&i| matches!(agents[i as usize], Agent::Pimc(_)))
    {
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread pool");
        return one.install(|| play_seated(agents, ids, hand, &mut rng, bestof));
    }
    play_seated(agents, ids, hand, &mut rng, bestof)
}

/// Appends the best-of-`k` count histogram of this decision to `out` (its own
/// generator, seeded from the hand and the step, so play is unchanged).
fn bestof_row(s: &State, me: usize, hand: u64, step: u16, k: usize, out: &mut Vec<u8>) {
    let mut rng = SmallRng::seed_from_u64(hand_seed(hand ^ 0x5eed_b0cf, u64::from(step)));
    let mut hist = [0u8; BESTOF_ROW];
    for _ in 0..k {
        let w = sample_world(s, me, &mut rng);
        for r in 1..4 {
            let h = w.hand((me + r) % 4);
            for t in Tile::all_kinds() {
                let c = (h[t] as usize).min(4);
                hist[((r - 1) * NUM_KINDS + t.index() as usize) * 5 + c] += 1;
            }
        }
    }
    out.extend_from_slice(&hist);
}

fn play_seated(
    agents: &[Agent],
    ids: [u8; 4],
    hand: u64,
    rng: &mut SmallRng,
    bestof: usize,
) -> HandOut {
    let mut s = State::new_deal(RuleSet::default(), rng);
    let mut pending = Vec::new();
    let mut legal = Vec::new();
    let mut step = 0u16;
    let mut sidecar = Vec::new();
    while let Some(p) = s.current_player() {
        pending.push(capture(&s, hand as u32, step, ids));
        if bestof > 0 {
            bestof_row(&s, p, hand, step, bestof, &mut sidecar);
        }
        s.legal_actions(&mut legal);
        let a = agents[ids[p] as usize].choose(&s, &legal, rng);
        s.apply(a);
        step = step.saturating_add(1);
    }
    let result = s.result().expect("finished").clone();
    let mut out = Vec::with_capacity(pending.len() * row_bytes());
    for r in &pending {
        finish(r, &result, &mut out);
    }
    HandOut {
        rows: out,
        bestof: sidecar,
    }
}

fn json_list(xs: &[&str]) -> String {
    let q: Vec<String> = xs.iter().map(|x| format!("\"{x}\"")).collect();
    format!("[{}]", q.join(", "))
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("selfplay_log: {e}");
            std::process::exit(2);
        }
    };
    let names = if args.holdout {
        HOLDOUT_AGENTS
    } else {
        TRAIN_AGENTS
    };
    let agents: Vec<Agent> = names
        .iter()
        .map(|n| Agent::parse(n).unwrap_or_else(|e| panic!("agent {n}: {e}")))
        .collect();
    let meta_head = format!(
        "\"field\": \"{}\", \"seed\": {}, \"hands_target\": {}, \"chunk_hands\": {}, \"threads\": {}, \"bestof\": {}, \"agents\": {}",
        if args.holdout { "holdout" } else { "train" },
        args.seed,
        args.hands,
        args.chunk,
        args.threads,
        args.bestof,
        json_list(names)
    );
    // A resumed run must keep the chunk size, or hand indices would shift.
    if let Ok(old) = std::fs::read_to_string(args.out.join("meta.json")) {
        let want = format!("\"chunk_hands\": {},", args.chunk);
        let seed = format!("\"seed\": {},", args.seed);
        if !old.contains(&want) || !old.contains(&seed) {
            eprintln!(
                "selfplay_log: {} holds a log with another --chunk or --seed",
                args.out.display()
            );
            std::process::exit(2);
        }
    }
    let mut writer = LogWriter::open(&args.out, meta_head).expect("open the log directory");
    let n_chunks = args.hands.div_ceil(args.chunk);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .expect("thread pool");
    let started = Instant::now();
    let first = writer.chunks_done();
    if first > 0 {
        println!(
            "resuming after {first} complete chunks ({} rows)",
            writer.rows()
        );
    }
    for c in first..n_chunks {
        let lo = c * args.chunk;
        let hi = ((c + 1) * args.chunk).min(args.hands);
        let hands: Vec<HandOut> = pool.install(|| {
            (lo..hi)
                .into_par_iter()
                .map(|h| play_hand(&agents, args.holdout, args.seed, h as u64, args.bestof))
                .collect()
        });
        let bytes: Vec<u8> = hands.iter().flat_map(|h| h.rows.iter().copied()).collect();
        if args.bestof > 0 {
            let side: Vec<u8> = hands
                .iter()
                .flat_map(|h| h.bestof.iter().copied())
                .collect();
            debug_assert_eq!(side.len() / BESTOF_ROW, bytes.len() / row_bytes());
            std::fs::write(args.out.join(format!("bestof-{c:05}.bin")), side)
                .expect("write the best-of-K sidecar");
        }
        let status = format!(
            "\"hands_done\": {hi}, \"complete\": {}, \"updated_unix\": {}, \"session_seconds\": {}",
            hi == args.hands,
            now_unix(),
            started.elapsed().as_secs()
        );
        writer
            .write_chunk(&bytes, &status)
            .expect("write the chunk");
        println!(
            "chunk {}/{n_chunks}: hands {hi}/{}, rows {}, {:.0} s",
            c + 1,
            args.hands,
            writer.rows(),
            started.elapsed().as_secs_f64()
        );
    }
    if first == n_chunks {
        println!("already complete: {} rows", writer.rows());
    }
}
