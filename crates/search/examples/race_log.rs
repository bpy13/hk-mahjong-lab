//! Full-information self-play logs for the race bot (plan 2026-11-09 Task 8,
//! `docs/race-bot.md` §7–8).
//!
//! ```text
//! cargo run --release -p mahjong-search --example race_log -- \
//!     --hands 100000 --out runs/race/g1/logs --seed 1 --threads 8 \
//!     --main "race:belief=models/race-belief-g0.mjnn" \
//!     [--old "race:belief=models/race-belief-g-1.mjnn"]... \
//!     [--fixed value --fixed defensive ...] [--shares 60,20,20] [--chunk 500]
//! ```
//!
//! Each hand is one of the §8.1 league tables, drawn with the hand's own generator:
//! with probability `shares[0]` four copies of `--main` (`R_g`); `shares[1]` one `--main`
//! and three seats drawn uniformly from the `--old` specs (older generations; without
//! any, the hand is four `--main`); `shares[2]` one `--main` and three seats from the
//! `--fixed` field (default value, defensive, fast, efficiency, closed). The `--main`
//! seat is random.
//!
//! Rows are the belief log's rows (`mahjong_search::log`, so the belief heads train on
//! them unchanged) followed by the race labels of §7 ([`RACE_FIELDS`]), all from the
//! true hands. Opponent fields are relative seats 1, 2, 3:
//!
//! * `opp_state`: 0 far, 1 one away, 2 ready, 3 folded ([`race::true_state`]);
//! * `opp_mode`: the mode of the opponent's last discard, 0 push, 1 detour, 2 fold
//!   ([`race::discard_label`], `docs/race-bot.md` §12; push before its first discard);
//! * `opp_shanten`: true faan-aware shanten (255 when no target meets the minimum), for
//!   the `a`, `b` tables;
//! * `opp_points_sd`: points per share of the best self-drawn win now (the discard win
//!   is the belief log's `opp_points_now`);
//! * `my_mode`, `my_feeds`, `ready_opps`: the observer's own action at this row when it
//!   is a discard (its mode, how many ready opponents could win on it, how many
//!   opponents were ready; `my_mode` 255 otherwise), for `q(m)`;
//! * `my_mistake`: 1 when that discard is a mistake (worse for the hand and not clearly
//!   safer, §12; its `my_mode` is push), else 0;
//! * `league`: 0 self-play, 1 against old generations, 2 against the fixed field;
//! * `wall_at_end`: wall tiles left when the hand ended (rounds to the outcome);
//! * `race_rounds`, `race_pwin`: when the observer is a race bot, the rounds its race
//!   valued and P(each relative seat wins within them) for the chosen action (§8.2's
//!   hazard check); 0 and NaN otherwise.
//!
//! Rerunning the same command resumes after the last complete chunk.

use mahjong_games::{Game, Policy};
use mahjong_hk::{score_win, Action, RuleSet, State, WinContext};
use mahjong_search::log::{capture, f16_bits, finish, row_bytes, Field, LogWriter, FIELDS};
use mahjong_search::race::{Label, LabelTracker, TrueState};
use mahjong_search::{Agent, Bot, SeatView};
use mahjong_tiles::Tile;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const fn field(name: &'static str, dtype: &'static str, shape: &'static [usize]) -> Field {
    Field { name, dtype, shape }
}

/// The race labels appended to every belief-log row (see the module docs).
const RACE_FIELDS: &[Field] = &[
    field("opp_state", "u1", &[3]),
    field("opp_mode", "u1", &[3]),
    field("opp_shanten", "u1", &[3]),
    field("opp_points_sd", "u2", &[3]),
    field("my_mode", "u1", &[]),
    field("my_feeds", "u1", &[]),
    field("ready_opps", "u1", &[]),
    field("my_mistake", "u1", &[]),
    field("league", "u1", &[]),
    field("wall_at_end", "u1", &[]),
    field("race_rounds", "u1", &[]),
    field("race_pwin", "f2", &[4]),
];
/// Layout version of a race log: 103 = the belief log's layout 2 (with `passed`)
/// followed by the race labels (102 was over layout 1).
const RACE_LAYOUT_VERSION: u32 = 103;
const NONE: u8 = 255;
const DEFAULT_FIXED: &[&str] = &["value", "defensive", "fast", "efficiency", "closed"];

struct Args {
    hands: usize,
    out: PathBuf,
    seed: u64,
    threads: usize,
    chunk: usize,
    main: String,
    old: Vec<String>,
    fixed: Vec<String>,
    shares: [u32; 3],
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        hands: 1000,
        out: PathBuf::from("runs/race/logs"),
        seed: 1,
        threads: 8,
        chunk: 500,
        main: "race".into(),
        old: Vec::new(),
        fixed: Vec::new(),
        shares: [60, 20, 20],
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
            "--main" => a.main = v()?,
            "--old" => a.old.push(v()?),
            "--fixed" => a.fixed.push(v()?),
            "--shares" => {
                let s = v()?;
                let xs: Vec<u32> = s
                    .split(',')
                    .map(|x| x.trim().parse::<u32>())
                    .collect::<Result<_, _>>()
                    .map_err(|_| format!("bad --shares {s:?}"))?;
                a.shares = xs
                    .try_into()
                    .map_err(|_| format!("--shares needs three numbers, got {s:?}"))?;
                if a.shares.iter().sum::<u32>() == 0 {
                    return Err("--shares sum to zero".into());
                }
            }
            _ => return Err(format!("unknown argument {k:?}")),
        }
    }
    if a.fixed.is_empty() {
        a.fixed = DEFAULT_FIXED.iter().map(|s| s.to_string()).collect();
    }
    Ok(a)
}

/// SplitMix64, as `selfplay_log`: hand seeds from (run seed, hand index).
fn hand_seed(seed: u64, hand: u64) -> u64 {
    let mut z = seed
        .wrapping_mul(0xD1B5_4A32_D192_ED03)
        .wrapping_add(hand.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The agents in index order: `--main`, then `--old`, then `--fixed`.
struct League {
    specs: Vec<String>,
    agents: Vec<Agent>,
    n_old: usize,
    n_fixed: usize,
    shares: [u32; 3],
}

impl League {
    /// The league table of a hand (0 self, 1 old, 2 fixed) and the agent index per seat.
    fn seat(&self, rng: &mut SmallRng) -> (u8, [u8; 4]) {
        let total: u32 = self.shares.iter().sum();
        let r = rng.gen_range(0..total);
        let mut kind = if r < self.shares[0] {
            0
        } else if r < self.shares[0] + self.shares[1] {
            1
        } else {
            2
        };
        if kind == 1 && self.n_old == 0 {
            kind = 0;
        }
        let mut ids = [0u8; 4];
        if kind > 0 {
            let (lo, n) = if kind == 1 {
                (1, self.n_old)
            } else {
                (1 + self.n_old, self.n_fixed)
            };
            let main_seat = rng.gen_range(0..4);
            for (seat, id) in ids.iter_mut().enumerate() {
                if seat != main_seat {
                    *id = (lo + rng.gen_range(0..n)) as u8;
                }
            }
        }
        (kind, ids)
    }
}

/// Points per share of the best self-drawn win for `seat` now (0 if not ready).
fn best_self_draw(state: &State, seat: usize) -> u16 {
    let rules = state.rules();
    let mut best = 0u16;
    for k in Tile::all_kinds().filter(|&k| state.wins_on(seat, k)) {
        let mut c = *state.hand(seat);
        c.add(k);
        let ctx = WinContext {
            seat,
            win_tile: Some(k),
            self_drawn: true,
            bonus: state.bonus(seat),
            last_tile: false,
            kong_chain: 0,
            robbing_kong: false,
            heavenly: false,
            earthly: false,
        };
        if let Some(s) = score_win(&c, state.melds(seat), &ctx, rules) {
            best = best.max(rules.points(s.faan).min(u16::MAX as u32) as u16);
        }
    }
    best
}

/// The race labels of one row, before the hand's end is known.
#[derive(Clone, Debug)]
struct RaceRow {
    state: [u8; 3],
    mode: [u8; 3],
    shanten: [u8; 3],
    points_sd: [u16; 3],
    my_mode: u8,
    my_feeds: u8,
    ready_opps: u8,
    mistake: u8,
    rounds: u8,
    pwin: [f32; 4],
}

impl RaceRow {
    fn encode(&self, league: u8, wall_at_end: u8, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.state);
        out.extend_from_slice(&self.mode);
        out.extend_from_slice(&self.shanten);
        for p in self.points_sd {
            out.extend_from_slice(&p.to_le_bytes());
        }
        out.extend_from_slice(&[self.my_mode, self.my_feeds, self.ready_opps, self.mistake]);
        out.extend_from_slice(&[league, wall_at_end, self.rounds]);
        for p in self.pwin {
            out.extend_from_slice(&f16_bits(p).to_le_bytes());
        }
    }
}

fn play_hand(league: &League, seed: u64, hand: u64) -> Vec<u8> {
    let mut rng = SmallRng::seed_from_u64(hand_seed(seed, hand));
    let (kind, ids) = league.seat(&mut rng);
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    let mut tracker = LabelTracker::new(&s);
    let mut pending = Vec::new();
    let mut labels: Vec<RaceRow> = Vec::new();
    let mut legal = Vec::new();
    let mut step = 0u16;
    while let Some(me) = s.current_player() {
        pending.push(capture(&s, hand as u32, step, ids));
        let mut row = RaceRow {
            state: [0; 3],
            mode: [0; 3],
            shanten: [0; 3],
            points_sd: [0; 3],
            my_mode: NONE,
            my_feeds: 0,
            ready_opps: 0,
            mistake: 0,
            rounds: 0,
            pwin: [f32::NAN; 4],
        };
        for r in 1..4 {
            let seat = (me + r) % 4;
            let st = tracker.state_of(&s, seat);
            row.state[r - 1] = st as u8;
            row.mode[r - 1] = tracker.mode_of(seat) as u8;
            let sh = mahjong_search::race::hand_shanten(&SeatView::of(&s, seat));
            row.shanten[r - 1] = u8::try_from(sh).ok().filter(|&x| x < 20).unwrap_or(NONE);
            let ready = st == TrueState::Ready
                || (st == TrueState::Folded && Tile::all_kinds().any(|k| s.wins_on(seat, k)));
            if ready {
                row.points_sd[r - 1] = best_self_draw(&s, seat);
                row.ready_opps += 1;
            }
        }
        s.legal_actions(&mut legal);
        let action = match &league.agents[ids[me] as usize] {
            Agent::Bot(Bot::Race(b)) if legal.len() > 1 => {
                let d = b.decide(&s);
                if let Some(c) = d.candidates.iter().find(|c| c.action == d.action) {
                    row.rounds = c.rounds.min(254) as u8;
                    for r in 0..4 {
                        row.pwin[r] = c.p_win[(me + r) % 4] as f32;
                    }
                }
                d.action
            }
            agent => agent.choose(&s, &legal, &mut rng),
        };
        if let Some(label) = tracker.observe(&s, action) {
            row.my_mode = label.mode() as u8;
            row.mistake = u8::from(label == Label::Mistake);
            if let Action::Discard(t) = action {
                row.my_feeds = (1..4).filter(|r| s.wins_on((me + r) % 4, t)).count() as u8;
            }
        }
        labels.push(row);
        s.apply(action);
        step = step.saturating_add(1);
    }
    let result = s.result().expect("finished").clone();
    let wall_at_end = s.wall_remaining().min(255) as u8;
    let mut out = Vec::with_capacity(pending.len() * (row_bytes() + race_bytes()));
    for (p, r) in pending.iter().zip(&labels) {
        finish(p, &result, &mut out);
        r.encode(kind, wall_at_end, &mut out);
    }
    out
}

fn race_bytes() -> usize {
    mahjong_search::log::layout_bytes(RACE_FIELDS)
}

fn json_list(xs: &[String]) -> String {
    let q: Vec<String> = xs
        .iter()
        .map(|x| format!("\"{}\"", x.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect();
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
            eprintln!("race_log: {e}");
            std::process::exit(2);
        }
    };
    let mut specs = vec![args.main.clone()];
    specs.extend(args.old.iter().cloned());
    specs.extend(args.fixed.iter().cloned());
    if specs.len() > 250 {
        eprintln!("race_log: too many agents");
        std::process::exit(2);
    }
    let agents: Vec<Agent> = specs
        .iter()
        .map(|n| {
            Agent::parse(n).unwrap_or_else(|e| {
                eprintln!("race_log: agent {n}: {e}");
                std::process::exit(2)
            })
        })
        .collect();
    let league = League {
        agents,
        n_old: args.old.len(),
        n_fixed: args.fixed.len(),
        shares: args.shares,
        specs,
    };
    let meta_head = format!(
        "\"layout\": \"race\", \"field\": \"race-league\", \"seed\": {}, \"hands_target\": {}, \"chunk_hands\": {}, \"threads\": {}, \"shares\": [{}, {}, {}], \"agents\": {}",
        args.seed,
        args.hands,
        args.chunk,
        args.threads,
        args.shares[0],
        args.shares[1],
        args.shares[2],
        json_list(&league.specs)
    );
    // A resumed run must keep the seed, chunk size and agents, or rows would mix.
    if let Ok(old) = std::fs::read_to_string(args.out.join("meta.json")) {
        let agents = format!("\"agents\": {}", json_list(&league.specs));
        let same = old.contains(&format!("\"chunk_hands\": {},", args.chunk))
            && old.contains(&format!("\"seed\": {},", args.seed))
            && old.contains(&agents);
        if !same {
            eprintln!(
                "race_log: {} holds a log with another --chunk, --seed or league",
                args.out.display()
            );
            std::process::exit(2);
        }
    }
    let mut fields = FIELDS.to_vec();
    fields.extend_from_slice(RACE_FIELDS);
    let fields: &'static [Field] = Box::leak(fields.into_boxed_slice());
    let mut writer = LogWriter::open_layout(&args.out, meta_head, fields, RACE_LAYOUT_VERSION)
        .expect("open the log directory");
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
        let hands: Vec<Vec<u8>> = pool.install(|| {
            (lo..hi)
                .into_par_iter()
                .map(|h| play_hand(&league, args.seed, h as u64))
                .collect()
        });
        let bytes: Vec<u8> = hands.concat();
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
