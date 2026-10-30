//! MJNN v3, the suit-shared CNN (plan Task 14): loading, the forward pass's
//! invariants, int8 weights and malformed files. The PyTorch side checks the
//! numbers against its own forward pass (`python/tests/test_cnn.py`).

use mahjong_games::{play_out, Game, Policy};
use mahjong_hk::{RuleSet, State};
use mahjong_search::agent::DEFAULT_LEAF_DEPTH;
use mahjong_search::cnn::{OUTPUT_LEN, POLICY_LEN, TARGET_LEN, UNUSED_TARGET};
use mahjong_search::features::{encode, PLANES, STATE_LEN};
use mahjong_search::net::top_actions;
use mahjong_search::pimc::VALUE_LOG_SCALE;
use mahjong_search::{
    pimc_with_leaf, Agent, Analysis, Bot, LeafValue, Mlp, NetPolicy, PimcConfig, Rollout,
    WorldSampler,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

/// Tiny hyper-parameters: channels, blocks, kernel, honour_hidden, merge, value_hidden.
const C: usize = 6;
const BLOCKS: usize = 2;
const K: usize = 3;
const HH: usize = 10;
const M: usize = 12;
const VH: usize = 5;
const REST: usize = PLANES * 7 + 49;

fn name(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u32).to_le_bytes());
    b.extend(s.as_bytes());
}

fn tensor_list() -> Vec<(String, Vec<usize>)> {
    let mut t: Vec<(String, Vec<usize>)> = vec![
        ("honour1.weight".into(), vec![HH, REST]),
        ("honour1.bias".into(), vec![HH]),
        ("honour2.weight".into(), vec![HH, HH]),
        ("honour2.bias".into(), vec![HH]),
        ("stem.weight".into(), vec![C, PLANES, K]),
        ("stem.bias".into(), vec![C]),
    ];
    for i in 0..BLOCKS {
        t.push((format!("blocks.{i}.ctx.weight"), vec![C, 2 * C + HH]));
        t.push((format!("blocks.{i}.ctx.bias"), vec![C]));
        for conv in ["conv1", "conv2"] {
            t.push((format!("blocks.{i}.{conv}.weight"), vec![C, C, K]));
            t.push((format!("blocks.{i}.{conv}.bias"), vec![C]));
        }
    }
    for (n, d) in [
        ("merge", vec![M, 2 * C + HH]),
        ("zproj", vec![C, M]),
        ("suit_policy", vec![4, C, 1]),
        ("honour_policy", vec![26, M]),
        ("flush_target", vec![6, C + M]),
        ("other_target", vec![9, M]),
        ("value1", vec![VH, M]),
        ("value2", vec![4, VH]),
    ] {
        t.push((format!("{n}.bias"), vec![d[0]]));
        t.push((format!("{n}.weight"), d));
    }
    t
}

/// A v3 file with pseudo-random weights; `int8` stores matrices as int8.
/// `edit` may change the tensor list before writing (malformed-file tests).
fn file(int8: bool, arch: &str, edit: impl Fn(&mut Vec<(String, Vec<usize>)>)) -> Vec<u8> {
    let mut rng = SmallRng::seed_from_u64(5);
    let mut b = b"MJNN".to_vec();
    b.extend(3u32.to_le_bytes());
    for slices in [
        vec![("features", 0, STATE_LEN)],
        vec![
            ("policy", 0, POLICY_LEN),
            ("target", POLICY_LEN, TARGET_LEN),
            ("value", POLICY_LEN + TARGET_LEN, 4),
        ],
    ] {
        b.extend((slices.len() as u32).to_le_bytes());
        for (n, off, len) in slices {
            name(&mut b, n);
            b.extend((off as u32).to_le_bytes());
            b.extend((len as u32).to_le_bytes());
        }
    }
    name(&mut b, arch);
    let hp = [
        ("channels", C),
        ("blocks", BLOCKS),
        ("kernel", K),
        ("honour_hidden", HH),
        ("merge", M),
        ("value_hidden", VH),
    ];
    b.extend((hp.len() as u32).to_le_bytes());
    for (n, v) in hp {
        name(&mut b, n);
        b.extend((v as u32).to_le_bytes());
    }
    let mut tensors = tensor_list();
    edit(&mut tensors);
    b.extend((tensors.len() as u32).to_le_bytes());
    for (n, dims) in tensors {
        name(&mut b, &n);
        let len: usize = dims.iter().product();
        let q = int8 && dims.len() >= 2;
        b.extend((q as u32).to_le_bytes());
        b.extend((dims.len() as u32).to_le_bytes());
        for d in &dims {
            b.extend((*d as u32).to_le_bytes());
        }
        let scale = 0.3 / (len / dims[0]) as f32;
        if q {
            for _ in 0..dims[0] {
                b.extend((scale / 127.0).to_le_bytes());
            }
            for _ in 0..len {
                b.push(rng.gen_range(-127i8..=127) as u8);
            }
        } else {
            for _ in 0..len {
                b.extend((rng.gen_range(-1.0f32..1.0) * scale * 3.0).to_le_bytes());
            }
        }
    }
    b
}

fn positions(n: usize) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(9);
    let bot = Bot::by_name("value").unwrap();
    let mut out = Vec::new();
    while out.len() < n {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let stop = rng.gen_range(0..60);
        for _ in 0..stop {
            if s.is_terminal() {
                break;
            }
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
        }
        if !s.is_terminal() {
            let mut f = vec![0.0; STATE_LEN];
            encode(&s, &mut f);
            out.push(f);
        }
    }
    out
}

#[test]
fn loads_and_keeps_its_invariants() {
    let net = Mlp::from_bytes(&file(false, "suitcnn-1", |_| {})).unwrap();
    assert_eq!(net.version(), 3);
    assert_eq!(net.input_len(), STATE_LEN);
    assert_eq!(net.output_len(), OUTPUT_LEN);
    assert_eq!(net.output("policy"), Some(0..POLICY_LEN));
    assert_eq!(
        net.output("value"),
        Some(POLICY_LEN + TARGET_LEN..OUTPUT_LEN)
    );
    assert!(net.parameters() > 1000);
    for f in positions(20) {
        let y = net.forward(&f);
        assert_eq!(y.len(), OUTPUT_LEN);
        assert!(y.iter().all(|v| v.is_finite()));
        let v: f32 = y[POLICY_LEN + TARGET_LEN..].iter().sum();
        assert!(v.abs() < 1e-4, "values sum to {v}");
        // Suit-less plans (mixed, honours, orphans) have only suit digit 0.
        for kind in [0, 3, 4] {
            for s in 1..3 {
                for st in 0..3 {
                    assert_eq!(y[POLICY_LEN + kind * 9 + s * 3 + st], UNUSED_TARGET);
                }
            }
        }
    }
}

/// New index of an old tile kind when old suit `s` becomes `perm[s]`.
fn tile_map(t: usize, perm: [usize; 3]) -> usize {
    if t < 27 {
        perm[t / 9] * 9 + t % 9
    } else {
        t
    }
}

#[test]
fn swapping_suits_swaps_the_outputs() {
    let net = Mlp::from_bytes(&file(false, "suitcnn-1", |_| {})).unwrap();
    let perm = [2, 0, 1];
    for f in positions(10) {
        let mut g = f.clone();
        for p in 0..PLANES {
            for t in 0..34 {
                g[p * 34 + tile_map(t, perm)] = f[p * 34 + t];
            }
        }
        let (y, z) = (net.forward(&f), net.forward(&g));
        let close = |a: f32, b: f32| (a - b).abs() < 1e-4 * (1.0 + a.abs());
        for t in 0..34 {
            let u = tile_map(t, perm);
            for base in [0, 35, 69] {
                assert!(close(y[base + t], z[base + u]), "code {}", base + t);
            }
            if t < 27 {
                assert!(close(y[107 + t], z[107 + u]));
            }
        }
        for code in [34, 103, 104, 105, 106] {
            assert!(close(y[code], z[code]));
        }
        for kind in [1, 2] {
            for (s, &ps) in perm.iter().enumerate() {
                for st in 0..3 {
                    let old = POLICY_LEN + kind * 9 + s * 3 + st;
                    let new = POLICY_LEN + kind * 9 + ps * 3 + st;
                    assert!(close(y[old], z[new]));
                }
            }
        }
        for i in POLICY_LEN + TARGET_LEN..OUTPUT_LEN {
            assert!(close(y[i], z[i]));
        }
    }
}

#[test]
fn int8_files_load_and_play() {
    let net = Arc::new(Mlp::from_bytes(&file(true, "suitcnn-1", |_| {})).unwrap());
    assert_eq!(net.version(), 3);
    let fp32 = file(false, "suitcnn-1", |_| {});
    let int8 = file(true, "suitcnn-1", |_| {});
    // Names and scales weigh more in a tiny network; a real one shrinks ~4×.
    assert!(int8.len() * 2 < fp32.len());
    // A network agent plays whole hands with a v3 file.
    let p = NetPolicy(net);
    let mut rng = SmallRng::seed_from_u64(2);
    for _ in 0..3 {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        play_out(&mut s, &p, &mut rng);
        assert!(s.is_terminal());
    }
}

#[test]
fn malformed_files_are_errors() {
    let bad = [
        file(false, "suitcnn-9", |_| {}),
        file(false, "suitcnn-1", |t| {
            t.retain(|(n, _)| n != "merge.weight");
        }),
        file(false, "suitcnn-1", |t| {
            t[0].1 = vec![HH, REST + 1];
        }),
        file(false, "suitcnn-1", |t| {
            t.push(("mystery.weight".into(), vec![2, 2]));
        }),
        {
            let mut b = file(false, "suitcnn-1", |_| {});
            b.push(0);
            b
        },
        {
            let mut b = file(false, "suitcnn-1", |_| {});
            b.truncate(b.len() - 3);
            b
        },
    ];
    for (i, b) in bad.iter().enumerate() {
        assert!(Mlp::from_bytes(b).is_err(), "case {i} loaded");
    }
    // The oracle critic's tensors are allowed (training only, ignored).
    let ok = file(false, "suitcnn-1", |t| {
        t.push(("oracle_value.weight".into(), vec![VH, 136]));
    });
    assert!(Mlp::from_bytes(&ok).is_ok());
}

// --- Plan Task 16: the network's value head at the leaves of PIMC -------------

/// Live mid-hand states: `value` plays a seeded deal for a random number of steps.
fn live_states(n: usize, seed: u64) -> Vec<State> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let bot = Bot::by_name("value").unwrap();
    let mut out = Vec::new();
    while out.len() < n {
        let mut s = State::new_deal(RuleSet::default(), &mut rng);
        let stop = rng.gen_range(5..40);
        for _ in 0..stop {
            if s.is_terminal() {
                break;
            }
            let legal = s.legal_actions_vec();
            let a = bot.choose(&s, &legal, &mut rng);
            s.apply(a);
        }
        if !s.is_terminal() && s.legal_actions_vec().len() > 1 {
            out.push(s);
        }
    }
    out
}

/// What must match between two searches: candidates, sample counts, means and
/// paired differences, as bits.
fn search_key(a: &Analysis) -> Vec<(String, u32, u64, u64)> {
    a.moves
        .iter()
        .map(|m| {
            (
                m.action.to_string(),
                m.n,
                m.mean.to_bits(),
                m.diff.to_bits(),
            )
        })
        .collect()
}

fn leaf_search(s: &State, net: &Arc<Mlp>, leaf: Option<&LeafValue>) -> Analysis {
    let config = PimcConfig {
        samples: 24,
        candidates: 3,
        seed: 11,
        ..PimcConfig::default()
    };
    let candidates = top_actions(net, s, 3);
    let baseline = candidates[0];
    let rollout = Rollout::for_seat(s.current_player().unwrap());
    pimc_with_leaf(
        s,
        &config,
        &WorldSampler::BestOfK,
        candidates,
        baseline,
        &rollout,
        leaf,
    )
}

#[test]
fn leaf_value_undoes_the_log_scale_for_the_searching_seat() {
    let net = Arc::new(Mlp::from_bytes(&file(false, "suitcnn-1", |_| {})).unwrap());
    let leaf = LeafValue::new(Arc::clone(&net), 6).unwrap();
    for s in live_states(10, 4) {
        let me = s.current_player().unwrap();
        let y = net.forward_state(&s);
        let v = &y[POLICY_LEN + TARGET_LEN..OUTPUT_LEN];
        for seat in 0..4 {
            let x = v[(seat + 4 - me) % 4] as f64;
            let want = x.signum() * VALUE_LOG_SCALE * x.abs().exp_m1();
            assert!((leaf.points(&s, seat) - want).abs() < 1e-9);
        }
    }
    // A network without a 4-wide value output cannot be a leaf evaluator: a
    // v1 policy network (one 134-wide output) is refused.
    let mut v1 = b"MJNN".to_vec();
    v1.extend(1u32.to_le_bytes());
    v1.extend(1u32.to_le_bytes());
    v1.extend((POLICY_LEN as u32).to_le_bytes());
    v1.extend((STATE_LEN as u32).to_le_bytes());
    v1.extend(vec![0u8; 4 * POLICY_LEN * STATE_LEN + 4 * POLICY_LEN]);
    let policy_only = Arc::new(Mlp::from_bytes(&v1).unwrap());
    assert!(LeafValue::new(policy_only, 6).is_err());
}

#[test]
fn a_leaf_deeper_than_the_hand_is_a_full_rollout() {
    let net = Arc::new(Mlp::from_bytes(&file(false, "suitcnn-1", |_| {})).unwrap());
    let deep = LeafValue::new(Arc::clone(&net), 10_000).unwrap();
    let shallow = LeafValue::new(Arc::clone(&net), 2).unwrap();
    let mut differ = 0;
    for s in live_states(3, 5) {
        let full = leaf_search(&s, &net, None);
        assert_eq!(
            search_key(&leaf_search(&s, &net, Some(&deep))),
            search_key(&full)
        );
        if search_key(&leaf_search(&s, &net, Some(&shallow))) != search_key(&full) {
            differ += 1;
        }
    }
    assert!(differ > 0, "a 2-discard leaf never changed a number");
}

#[test]
fn leaf_search_on_one_thread_matches_the_default_pool() {
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let net = Arc::new(Mlp::from_bytes(&file(true, "suitcnn-1", |_| {})).unwrap());
    let leaf = LeafValue::new(Arc::clone(&net), 6).unwrap();
    for s in live_states(3, 6) {
        let pool = leaf_search(&s, &net, Some(&leaf));
        let single = one.install(|| leaf_search(&s, &net, Some(&leaf)));
        assert_eq!(search_key(&single), search_key(&pool));
        assert_eq!(single.choose(2.0), pool.choose(2.0));
    }
}

#[test]
fn netpimc_takes_leaf_and_depth_and_plays() {
    let path = std::env::temp_dir().join(format!("cnn-io-leaf-{}.mjnn", std::process::id()));
    std::fs::write(&path, file(true, "suitcnn-1", |_| {})).unwrap();
    let p = path.display();
    match Agent::parse(&format!("netpimc:path={p},samples=4,leaf=value,depth=4")) {
        Ok(Agent::NetPimc { leaf: Some(l), .. }) => assert_eq!(l.depth, 4),
        other => panic!("{other:?}"),
    }
    match Agent::parse(&format!("netpimc:path={p},leaf=value")) {
        Ok(Agent::NetPimc { leaf: Some(l), .. }) => assert_eq!(l.depth, DEFAULT_LEAF_DEPTH),
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        Agent::parse(&format!("netpimc:path={p}")),
        Ok(Agent::NetPimc { leaf: None, .. })
    ));
    assert!(Agent::parse(&format!("netpimc:path={p},depth=3")).is_err());
    assert!(Agent::parse(&format!("netpimc:path={p},leaf=maybe")).is_err());
    assert!(Agent::parse(&format!("netpimc:path={p},leaf=value,depth=x")).is_err());
    // The agent plays a whole hand in every seat.
    let agent = Agent::parse(&format!("netpimc:path={p},samples=4,leaf=value,depth=2")).unwrap();
    let mut rng = SmallRng::seed_from_u64(3);
    let mut s = State::new_deal(RuleSet::default(), &mut rng);
    play_out(&mut s, &agent, &mut rng);
    assert!(s.is_terminal());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn the_exported_test_network_loads_when_present() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/cnn-test.mjnn");
    if !std::path::Path::new(path).exists() {
        return;
    }
    let net = Mlp::load(path).unwrap();
    assert_eq!(net.version(), 3);
    let f = &positions(1)[0];
    assert!(net.forward(f).iter().all(|v| v.is_finite()));
}
