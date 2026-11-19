//! Training and check data for the risk network (plan 2026-11-16 Task 15), and
//! its timing.
//!
//! ```text
//! cargo run --release -p mahjong-search --example risk_log -- \
//!     --deals 20000 --chunk 1000 --seed 15 --out runs/risk/data \
//!     [--belief models/belief.mjnn]
//! cargo run --release -p mahjong-search --example risk_log -- --time models/risk.mjnn [deals]
//! ```
//!
//! Deals of four `stage` bots. At every discard decision of every seat one row
//! holds the network input (`features` as float16, the packed `history`, the
//! `risk` block as bytes), the true answer for every kind and opponent
//! ([`mahjong_search::risk_net::labels`]: wins by discard now, points per
//! share) and, for the checks, what the three existing sources say for every
//! kind: the table (Task 11's calibration, `giveup::danger_of`), exact
//! counting (Task 13, the switch on) and the belief model (`--belief`). Rows
//! are fixed-size records, `chunk-NNN.bin` holding `--chunk` whole deals each;
//! `meta.json` names the fields in order (`python/mahjong/risk/data.py` reads
//! them). Rerunning resumes after the last complete chunk. Threads:
//! `RAYON_NUM_THREADS`.
//!
//! Row layout 2 (Task 16; `"layout": 2` in `meta.json`): the six per-tile
//! check columns are float16 and the `risk` block is packed two values to a
//! byte (value `2i` in the low half of byte `i`, `2i + 1` in the high half),
//! 3,613 bytes a row against layout 1's 4,288. The Python reader loads both.
//!
//! `--time`: on the discard decisions of `deals` deals (default 200), one
//! thread, the time per decision of the network alone, of
//! `Danger::from_belief` with the network set, and of the same without it
//! (p50 and p99).

use mahjong_games::{Game, Policy};
use mahjong_hk::{Action, RuleSet, State};
use mahjong_search::danger::{with_exact, with_risk};
use mahjong_search::ev::HeuristicBelief;
use mahjong_search::exact_wait::exact_wait;
use mahjong_search::features::{encode, encode_history, HISTORY_LEN, STATE_LEN};
use mahjong_search::labels::{pack_history, HISTORY_BYTES};
use mahjong_search::risk_net::{encode_risk, labels, RiskNet, RISK_LEN};
use mahjong_search::{giveup, Belief, BeliefModel, Bot, Danger};
use mahjong_tiles::{Tile, NUM_KINDS};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Row layout version written (`meta.json` `layout`).
const LAYOUT: u32 = 2;

/// Bytes of the packed `risk` block: two values (0 to 4) per byte.
const RISK_BYTES: usize = RISK_LEN.div_ceil(2);

/// The fields of one row, in order: name, numpy dtype, shape.
const FIELDS: &[(&str, &str, &[usize])] = &[
    ("deal", "<u4", &[]),
    ("seat", "u1", &[]),
    ("wall", "u1", &[]),
    ("thrown", "u1", &[]),
    ("hit", "u1", &[]),
    ("concealed", "u1", &[3]),
    ("shares", "u1", &[3]),
    ("is_exact", "u1", &[3]),
    ("pad", "u1", &[3]),
    ("hand", "u1", &[NUM_KINDS]),
    ("features", "<f2", &[STATE_LEN]),
    ("history", "u1", &[HISTORY_BYTES]),
    ("risk", "u1", &[RISK_BYTES]),
    ("win", "u1", &[3, NUM_KINDS]),
    ("points", "u1", &[3, NUM_KINDS]),
    ("table_tile", "<f2", &[NUM_KINDS]),
    ("exact_tile", "<f2", &[NUM_KINDS]),
    ("belief_tile", "<f2", &[NUM_KINDS]),
    ("table_loss", "<f2", &[NUM_KINDS]),
    ("exact_loss", "<f2", &[NUM_KINDS]),
    ("belief_loss", "<f2", &[NUM_KINDS]),
    ("table_p", "<f2", &[3, NUM_KINDS]),
    ("belief_p", "<f2", &[3, NUM_KINDS]),
    ("exact_p", "<f2", &[3, NUM_KINDS]),
    ("exact_points", "u1", &[3, NUM_KINDS]),
];

fn dtype_bytes(d: &str) -> usize {
    match d {
        "u1" => 1,
        "<f2" => 2,
        "<u4" | "<f4" => 4,
        _ => panic!("dtype {d}"),
    }
}

fn row_bytes() -> usize {
    FIELDS
        .iter()
        .map(|(_, d, shape)| dtype_bytes(d) * shape.iter().product::<usize>().max(1))
        .sum()
}

/// IEEE half precision (round to nearest even), little-endian bytes.
fn f16(x: f32) -> [u8; 2] {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    let h = if exp == 0xff {
        0x7c00 | ((mant != 0) as u16) << 9
    } else {
        let e = exp - 127 + 15;
        if e >= 0x1f {
            0x7c00
        } else if e <= 0 {
            if e < -10 {
                0
            } else {
                let m = mant | 0x80_0000;
                let shift = (14 - e) as u32;
                let half = m >> shift;
                let rem = m & ((1 << shift) - 1);
                let mid = 1u32 << (shift - 1);
                (half + ((rem > mid || (rem == mid && half & 1 == 1)) as u32)) as u16
            }
        } else {
            let half = ((e as u32) << 10) | (mant >> 13);
            let rem = mant & 0x1fff;
            (half + ((rem > 0x1000 || (rem == 0x1000 && half & 1 == 1)) as u32)) as u16
        }
    };
    (sign | h).to_le_bytes()
}

struct Writer<'a> {
    buf: &'a mut Vec<u8>,
}

impl Writer<'_> {
    fn u8s(&mut self, xs: &[u8]) {
        self.buf.extend_from_slice(xs);
    }
    fn f16s(&mut self, xs: &[f32]) {
        for &x in xs {
            self.buf.extend_from_slice(&f16(x));
        }
    }
    fn f16s_of(&mut self, xs: &[f64]) {
        for &x in xs {
            self.buf.extend_from_slice(&f16(x as f32));
        }
    }
    /// Two values (0 to 4) per byte: `xs[2i]` low, `xs[2i + 1]` high.
    fn nibbles(&mut self, xs: &[u8]) {
        for pair in xs.chunks(2) {
            debug_assert!(pair.iter().all(|&v| v < 16));
            self.buf
                .push(pair[0] | (pair.get(1).copied().unwrap_or(0) << 4));
        }
    }
}

fn dealt_in_to(s: &State, seat: usize) -> Vec<usize> {
    s.result().map_or(Vec::new(), |r| {
        r.wins
            .iter()
            .filter(|w| !w.self_drawn && !w.robbed && w.from == Some(seat))
            .map(|w| w.seat)
            .collect()
    })
}

/// Plays one deal and appends its rows to `out`; returns the row count.
fn log_deal(deal: u32, deal_seed: u64, belief: Option<&BeliefModel>, out: &mut Vec<u8>) -> usize {
    let rules = RuleSet::default();
    let bot = Bot::by_name("stage").unwrap();
    let mut state = State::new_deal(rules, &mut ChaCha8Rng::seed_from_u64(deal_seed));
    let mut rng = ChaCha8Rng::seed_from_u64(deal_seed ^ 0x9e37_79b9);
    let mut rows = 0;
    let mut features = vec![0.0f32; STATE_LEN];
    let mut history = vec![0.0f32; HISTORY_LEN];
    let mut risk = vec![0.0f32; RISK_LEN];
    while let Some(me) = state.current_player() {
        let legal = state.legal_actions_vec();
        let action = bot.choose(&state, &legal, &mut rng);
        let Action::Discard(t) = action else {
            state.apply(action);
            continue;
        };
        // The row: everything before the discard.
        let start = out.len();
        let mut w = Writer { buf: out };
        let truth = labels(&state, me);
        let off = Danger::from_belief(&state, me, &HeuristicBelief);
        let on = with_exact(true, || Danger::from_belief(&state, me, &HeuristicBelief));
        let heur = HeuristicBelief.estimate(&state, me);
        let model = belief.map(|m| (Danger::from_belief(&state, me, m), m.estimate(&state, me)));
        let unseen = state.unseen(me);
        let exact: Vec<_> = (1..4)
            .map(|r| exact_wait(&state, me, (me + r) % 4))
            .collect();
        let mut next = state.clone();
        next.apply(action);
        while next.in_claim() {
            let l = next.legal_actions_vec();
            let c = bot.choose(&next, &l, &mut rng);
            next.apply(c);
        }
        let hit = !dealt_in_to(&next, me).is_empty();

        w.u8s(&deal.to_le_bytes());
        w.u8s(&[
            me as u8,
            state.wall_remaining().min(255) as u8,
            t.index(),
            hit as u8,
        ]);
        let opp = |r: usize| (me + r) % 4;
        w.u8s(&[1, 2, 3].map(|r| state.concealed_count(opp(r)).min(255) as u8));
        w.u8s(&[1, 2, 3].map(|r| match state.liable_for(opp(r)) {
            None => 1,
            Some(l) if l == me => 3,
            Some(_) => 0,
        }));
        w.u8s(&[1, 2, 3].map(|r| on.is_exact(opp(r)) as u8));
        w.u8s(&[0, 0, 0]);
        w.u8s(&state.hand(me).0);
        encode(&state, &mut features);
        w.f16s(&features);
        w.u8s(&pack_history(&state));
        encode_history(&state, &mut history);
        debug_assert!(history.iter().all(|v| v.is_finite()));
        encode_risk(&state, &mut risk);
        w.nibbles(
            &risk
                .iter()
                .map(|&v| (v * 4.0).round() as u8)
                .collect::<Vec<_>>(),
        );
        for r in 1..4 {
            w.u8s(&truth.win[opp(r)].map(|b| b as u8));
        }
        for r in 1..4 {
            w.u8s(&truth.points[opp(r)]);
        }
        let kinds: Vec<Tile> = Tile::all_kinds().collect();
        let table: Vec<(f64, f64)> = kinds.iter().map(|&k| giveup::danger_of(&off, k)).collect();
        let exact_fig: Vec<(f64, f64)> = kinds
            .iter()
            .map(|&k| {
                let loss = giveup::danger_of(&on, k).1;
                match on.split(k) {
                    Some(sp) => (
                        1.0 - (1.0 - giveup::calibrate(sp.table_tile)) * (1.0 - sp.exact_tile),
                        loss,
                    ),
                    None => (giveup::calibrate(on.tile(k)), loss),
                }
            })
            .collect();
        let bel: Vec<(f64, f64)> = kinds
            .iter()
            .map(|&k| match &model {
                Some((d, _)) => (d.tile(k), d.expected_loss(k)),
                None => (f64::NAN, f64::NAN),
            })
            .collect();
        w.f16s_of(&table.iter().map(|x| x.0).collect::<Vec<_>>());
        w.f16s_of(&exact_fig.iter().map(|x| x.0).collect::<Vec<_>>());
        w.f16s_of(&bel.iter().map(|x| x.0).collect::<Vec<_>>());
        w.f16s_of(&table.iter().map(|x| x.1).collect::<Vec<_>>());
        w.f16s_of(&exact_fig.iter().map(|x| x.1).collect::<Vec<_>>());
        w.f16s_of(&bel.iter().map(|x| x.1).collect::<Vec<_>>());
        let per_opp = |b: &mahjong_search::Beliefs, r: usize| -> Vec<f32> {
            let s = &b.seats[opp(r)];
            kinds
                .iter()
                .map(|&k| {
                    if unseen[k] == 0 {
                        0.0
                    } else {
                        (s.tenpai * s.wait[k.index() as usize]).clamp(0.0, 1.0) as f32
                    }
                })
                .collect()
        };
        for r in 1..4 {
            w.f16s(&per_opp(&heur, r));
        }
        for r in 1..4 {
            match &model {
                Some((_, b)) => w.f16s(&per_opp(b, r)),
                None => w.f16s(&[f32::NAN; NUM_KINDS]),
            }
        }
        for r in 1..4 {
            match &exact[r - 1] {
                Some(e) => {
                    let ready = on.threat(opp(r)) as f32;
                    w.f16s(&e.win.map(|x| ready * x as f32));
                }
                None => w.f16s(&[f32::NAN; NUM_KINDS]),
            }
        }
        for r in 1..4 {
            match &exact[r - 1] {
                Some(e) => w.u8s(&e.points.map(|x| x.min(255.0) as u8)),
                None => w.u8s(&[0; NUM_KINDS]),
            }
        }
        assert_eq!(w.buf.len() - start, row_bytes(), "row layout");
        rows += 1;
        state = next;
    }
    rows
}

struct Args {
    deals: u32,
    chunk: u32,
    seed: u64,
    out: PathBuf,
    belief: Option<PathBuf>,
}

fn write_meta(a: &Args, chunks: &[(u32, usize)]) {
    let fields: Vec<String> = FIELDS
        .iter()
        .map(|(n, d, s)| format!("[\"{n}\", \"{d}\", {s:?}]"))
        .collect();
    let chunk_list: Vec<String> = chunks
        .iter()
        .map(|(c, rows)| format!("{{\"chunk\": {c}, \"rows\": {rows}}}"))
        .collect();
    let meta = format!(
        "{{\n  \"layout\": {LAYOUT},\n  \"row_bytes\": {},\n  \"deals\": {},\n  \"chunk_deals\": {},\n  \"seed\": {},\n  \"belief\": {},\n  \"bots\": \"stage\",\n  \"fields\": [\n    {}\n  ],\n  \"chunks\": [\n    {}\n  ]\n}}\n",
        row_bytes(),
        a.deals,
        a.chunk,
        a.seed,
        a.belief
            .as_ref()
            .map_or("null".to_string(), |p| format!("\"{}\"", p.display()).replace('\\', "/")),
        fields.join(",\n    "),
        chunk_list.join(",\n    "),
    );
    std::fs::write(a.out.join("meta.json"), meta).expect("meta.json");
}

fn run_log(a: &Args) {
    std::fs::create_dir_all(&a.out).expect("out dir");
    let belief = a
        .belief
        .as_ref()
        .map(|p| BeliefModel::load(p).unwrap_or_else(|e| panic!("{e}")));
    let n_chunks = a.deals.div_ceil(a.chunk);
    let mut chunks: Vec<(u32, usize)> = Vec::new();
    let start = Instant::now();
    for c in 0..n_chunks {
        let path = a.out.join(format!("chunk-{c:03}.bin"));
        if path.exists() {
            let rows = std::fs::metadata(&path).unwrap().len() as usize / row_bytes();
            chunks.push((c, rows));
            continue;
        }
        let first = c * a.chunk;
        let last = (first + a.chunk).min(a.deals);
        let t0 = Instant::now();
        let parts: Vec<(Vec<u8>, usize)> = (first..last)
            .into_par_iter()
            .map(|d| {
                let mut buf = Vec::with_capacity(70 * row_bytes());
                let seed = a.seed.wrapping_mul(1_000_003).wrapping_add(d as u64);
                let rows = log_deal(d, seed, belief.as_ref(), &mut buf);
                (buf, rows)
            })
            .collect();
        let rows: usize = parts.iter().map(|p| p.1).sum();
        let tmp = a.out.join(format!("chunk-{c:03}.tmp"));
        {
            let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp).unwrap());
            for (buf, _) in &parts {
                f.write_all(buf).unwrap();
            }
            f.flush().unwrap();
        }
        std::fs::rename(&tmp, &path).unwrap();
        chunks.push((c, rows));
        println!(
            "chunk {c}: deals {first}..{last}, {rows} rows, {:.1} s (total {:.0} s)",
            t0.elapsed().as_secs_f64(),
            start.elapsed().as_secs_f64()
        );
        write_meta(a, &chunks);
    }
    write_meta(a, &chunks);
    println!(
        "done: {} chunks, {} rows, {} bytes per row",
        chunks.len(),
        chunks.iter().map(|c| c.1).sum::<usize>(),
        row_bytes()
    );
}

fn percentile(xs: &mut [f64], q: f64) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[((xs.len() - 1) as f64 * q).round() as usize]
}

fn run_time(path: &Path, deals: u32) {
    let net = Arc::new(RiskNet::load(path).unwrap_or_else(|e| panic!("{e}")));
    println!("{net:?}");
    let bot = Bot::by_name("stage").unwrap();
    let (mut t_net, mut t_on, mut t_off, mut t_exact) = (vec![], vec![], vec![], vec![]);
    let mut moved = 0usize;
    let mut decisions = 0usize;
    for d in 0..deals {
        let seed = 77u64.wrapping_mul(1_000_003).wrapping_add(d as u64);
        let mut state = State::new_deal(RuleSet::default(), &mut ChaCha8Rng::seed_from_u64(seed));
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x9e37_79b9);
        while let Some(me) = state.current_player() {
            let legal = state.legal_actions_vec();
            let action = bot.choose(&state, &legal, &mut rng);
            if let Action::Discard(t) = action {
                decisions += 1;
                let t0 = Instant::now();
                let out = net.estimate(&state, me).unwrap();
                t_net.push(t0.elapsed().as_secs_f64() * 1e3);
                let t0 = Instant::now();
                let on = with_risk(Some(Arc::clone(&net)), || {
                    Danger::from_belief(&state, me, &HeuristicBelief)
                });
                t_on.push(t0.elapsed().as_secs_f64() * 1e3);
                let t0 = Instant::now();
                let both = with_risk(Some(Arc::clone(&net)), || {
                    with_exact(true, || Danger::from_belief(&state, me, &HeuristicBelief))
                });
                t_exact.push(t0.elapsed().as_secs_f64() * 1e3);
                let t0 = Instant::now();
                let off = Danger::from_belief(&state, me, &HeuristicBelief);
                t_off.push(t0.elapsed().as_secs_f64() * 1e3);
                let _ = (out, both);
                if (on.tile(t) - off.tile(t)).abs() > 0.01 {
                    moved += 1;
                }
            }
            state.apply(action);
        }
    }
    println!("{decisions} discard decisions of {deals} deals, one thread");
    for (name, xs) in [
        ("network alone", &mut t_net),
        ("Danger::from_belief, network set", &mut t_on),
        ("the same with exact counting on too", &mut t_exact),
        ("Danger::from_belief, tables (today)", &mut t_off),
    ] {
        println!(
            "{name:<40} p50 {:.3} ms  p99 {:.3} ms  mean {:.3} ms",
            percentile(xs, 0.5),
            percentile(xs, 0.99),
            xs.iter().sum::<f64>() / xs.len() as f64
        );
    }
    println!(
        "discarded tile's danger moved by more than 0.01 against the table: {moved} of {decisions}"
    );
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().map(String::as_str) == Some("--time") {
        let path = raw.get(1).expect("--time needs a model path");
        let deals = raw.get(2).and_then(|x| x.parse().ok()).unwrap_or(200);
        run_time(Path::new(path), deals);
        return;
    }
    let mut a = Args {
        deals: 20_000,
        chunk: 1_000,
        seed: 15,
        out: PathBuf::from("runs/risk/data"),
        belief: None,
    };
    let mut it = raw.into_iter();
    while let Some(k) = it.next() {
        let v = it.next().unwrap_or_else(|| panic!("{k} needs a value"));
        match k.as_str() {
            "--deals" => a.deals = v.parse().expect("--deals"),
            "--chunk" => a.chunk = v.parse().expect("--chunk"),
            "--seed" => a.seed = v.parse().expect("--seed"),
            "--out" => a.out = PathBuf::from(v),
            "--belief" => a.belief = Some(PathBuf::from(v)),
            other => panic!("unknown option {other}"),
        }
    }
    run_log(&a);
}
