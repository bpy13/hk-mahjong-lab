//! Labels decisions with a search teacher for distillation (plan Task 13).
//!
//! ```text
//! cargo run --release -p mahjong-search --example label -- \
//!     --decisions 1000000 --out runs/labels --chunk 20000 --seed 3 \
//!     --teacher "pimc:samples=64,candidates=4,belief=models/belief.mjnn,weights=1" \
//!     [--others "ev:belief=models/belief.mjnn,floor=none"] [--batch 32] \
//!     [--stop-after-chunks N]
//! ```
//!
//! Writes `chunk-NNNNN.bin` files of `--chunk` rows each (the last may be
//! shorter) and `meta.json` (layout, teacher, chunks) in `--out`; rows are
//! described in `mahjong_search::labels`. Rerunning the same command resumes
//! after the last complete chunk. A different `--teacher`, `--others` or
//! `--seed` than the existing chunks' is refused, so one directory never mixes
//! teachers. `--stop-after-chunks N` exits after writing N new chunks (used by
//! the resume test).
//!
//! Progress: `progress.txt` in `--out` (one line, rewritten after every batch)
//! and `label.log` (one line per chunk). Threads: `RAYON_NUM_THREADS`.

use mahjong_hk::RuleSet;
use mahjong_search::labels::{label_row_bytes, Labeller, FIELDS, LABEL_LAYOUT_VERSION};
use mahjong_search::log::{chunk_name, LogWriter};
use mahjong_search::Agent;
use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

struct Args {
    decisions: usize,
    out: PathBuf,
    chunk: usize,
    seed: u64,
    teacher: String,
    others: String,
    batch: usize,
    stop_after: Option<usize>,
}

fn parse_args() -> Args {
    let mut a = Args {
        decisions: 1_000_000,
        out: PathBuf::from("runs/labels"),
        chunk: 20_000,
        seed: 3,
        teacher: "pimc:samples=64,candidates=4,belief=models/belief.mjnn,weights=1".into(),
        others: "ev:belief=models/belief.mjnn,floor=none".into(),
        batch: 32,
        stop_after: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("{k} needs a value"));
        match k.as_str() {
            "--decisions" => a.decisions = v().parse().expect("--decisions"),
            "--out" => a.out = PathBuf::from(v()),
            "--chunk" => a.chunk = v().parse().expect("--chunk"),
            "--seed" => a.seed = v().parse().expect("--seed"),
            "--teacher" => a.teacher = v(),
            "--others" => a.others = v(),
            "--batch" => a.batch = v().parse().expect("--batch"),
            "--stop-after-chunks" => a.stop_after = Some(v().parse().expect("--stop-after-chunks")),
            _ => panic!("unknown argument {k}"),
        }
    }
    assert!(
        a.chunk > 0 && a.decisions > 0,
        "--chunk and --decisions must be positive"
    );
    a
}

/// JSON string literal.
fn js(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Reads a string member from an earlier `meta.json` (written by this program,
/// so one member per line).
fn meta_member(meta: &str, key: &str) -> Option<String> {
    let prefix = format!("{}: ", js(key));
    meta.lines().find_map(|l| {
        let l = l.trim().trim_end_matches(',');
        l.strip_prefix(&prefix).map(str::to_string)
    })
}

/// The time now as `YYYY-MM-DDTHH:MM:SSZ` (UTC; no date crate in this workspace).
fn now() -> String {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    let (days, secs) = (t.div_euclid(86_400), t.rem_euclid(86_400));
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        secs / 3600,
        secs / 60 % 60,
        secs % 60
    )
}

fn append(path: &Path, line: &str) {
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open the log");
    let _ = writeln!(f, "{line}");
}

fn write_atomic(path: &Path, text: &str) {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).expect("write");
    std::fs::rename(&tmp, path).expect("rename");
}

/// Windows throttles detached headless processes (EcoQoS) to a fraction of a
/// core; opt this process out (`no_power_throttling` in the match runners).
#[cfg(windows)]
fn no_power_throttling() {
    use std::ffi::c_void;
    #[repr(C)]
    struct PowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn SetProcessInformation(
            process: *mut c_void,
            class: i32,
            info: *mut c_void,
            size: u32,
        ) -> i32;
    }
    // ProcessPowerThrottling = 4; control EXECUTION_SPEED (1), state 0 = off.
    let mut st = PowerThrottlingState {
        version: 1,
        control_mask: 1,
        state_mask: 0,
    };
    // SAFETY: a documented Win32 call with a correctly sized, live struct.
    unsafe {
        SetProcessInformation(
            GetCurrentProcess(),
            4,
            &mut st as *mut PowerThrottlingState as *mut c_void,
            std::mem::size_of::<PowerThrottlingState>() as u32,
        );
    }
}

#[cfg(not(windows))]
fn no_power_throttling() {}

fn main() {
    no_power_throttling();
    let args = parse_args();
    let n_chunks = args.decisions.div_ceil(args.chunk);
    std::fs::create_dir_all(&args.out).expect("create --out");
    let log = args.out.join("label.log");
    let progress = args.out.join("progress.txt");

    // One directory, one teacher: refuse to resume with other settings.
    let meta_path = args.out.join("meta.json");
    if args.out.join(chunk_name(0)).exists() {
        let old = std::fs::read_to_string(&meta_path).unwrap_or_default();
        for (key, want) in [
            ("teacher", js(&args.teacher)),
            ("others", js(&args.others)),
            ("seed", args.seed.to_string()),
            ("chunk_rows", args.chunk.to_string()),
        ] {
            let have = meta_member(&old, key);
            if have.as_deref() != Some(want.as_str()) {
                eprintln!(
                    "{}: existing chunks have {key} = {}, this run {want}; use another --out",
                    args.out.display(),
                    have.unwrap_or_else(|| "?".into())
                );
                std::process::exit(2);
            }
        }
    }

    let labeller = Labeller {
        teacher: Agent::parse(&args.teacher).expect("--teacher"),
        others: Agent::parse(&args.others).expect("--others"),
        rules: RuleSet::default(),
        seed: args.seed,
    };
    let meta_head = format!(
        "\"teacher\": {},\n  \"others\": {},\n  \"seed\": {},\n  \"chunk_rows\": {},\n  \"decisions_target\": {},\n  \"labelled_seat\": \"hand index mod 4\"",
        js(&args.teacher),
        js(&args.others),
        args.seed,
        args.chunk,
        args.decisions
    );
    let mut writer = LogWriter::open_layout(&args.out, meta_head, FIELDS, LABEL_LAYOUT_VERSION)
        .expect("open the label directory");
    let threads = rayon::current_num_threads();
    let done_rows: u64 = writer.rows();
    append(
        &log,
        &format!(
            "{} start: {} of {n_chunks} chunks done ({done_rows} rows); {threads} threads; teacher {}",
            now(),
            writer.chunks_done(),
            args.teacher
        ),
    );
    let start = Instant::now();
    let mut new_rows = 0usize;
    let first = writer.chunks_done();
    for (new_chunks, c) in (first..n_chunks).enumerate() {
        if args.stop_after.is_some_and(|n| new_chunks >= n) {
            break;
        }
        let want = args.chunk.min(args.decisions - c * args.chunk);
        let t = Instant::now();
        let (bytes, hands) = labeller.label_chunk(c as u32, want, args.batch, |rows, hands| {
            let total = new_rows + rows;
            let secs = start.elapsed().as_secs_f64().max(1e-9);
            let rate = total as f64 / secs;
            let left = args.decisions as f64 - done_rows as f64 - total as f64;
            let eta_h = if rate > 0.0 { left / rate / 3600.0 } else { f64::NAN };
            write_atomic(
                &progress,
                &format!(
                    "chunk {}/{n_chunks}: {rows}/{want} rows, {hands} hands; total {} of {}; {rate:.2} rows/s; about {eta_h:.1} h left ({})\n",
                    c + 1,
                    done_rows as usize + total,
                    args.decisions,
                    now()
                ),
            );
        });
        let rows = bytes.len() / label_row_bytes();
        writer
            .write_chunk(
                &bytes,
                &format!(
                    "\"last_chunk_hands\": {hands}, \"last_chunk_seconds\": {:.0}",
                    t.elapsed().as_secs_f64()
                ),
            )
            .expect("write the chunk");
        new_rows += rows;
        append(
            &log,
            &format!(
                "{} chunk {}/{n_chunks}: {rows} rows from {hands} hands in {:.0} s",
                now(),
                c + 1,
                t.elapsed().as_secs_f64()
            ),
        );
    }
    let total = writer.rows();
    let status = if writer.chunks_done() >= n_chunks {
        "done"
    } else {
        "stopped"
    };
    append(
        &log,
        &format!(
            "{} {status}: {} chunks, {total} rows",
            now(),
            writer.chunks_done()
        ),
    );
    write_atomic(
        &progress,
        &format!(
            "{status}: {} of {n_chunks} chunks, {total} rows ({})\n",
            writer.chunks_done(),
            now()
        ),
    );
}
