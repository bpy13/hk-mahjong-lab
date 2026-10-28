//! Bindings for the bots, the danger model, the solver and tournaments.

use crate::convert::rules_from;
use crate::{rng_from, value_error, PyState};
use mahjong_analysis::{discards, faan_shanten, shanten, Structure, SuitPlan, Target};
use mahjong_games::{Game, Policy};
use mahjong_hk::State;
use mahjong_search::tournament::tournament as run_tournament;
use mahjong_search::{
    pimc, solo as solo_sim, Agent, Danger, PimcConfig, SeatView, Seatings, TournamentConfig,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::time::Duration;

fn target_dict<'py>(py: Python<'py>, t: &Target) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let plan = match t.plan {
        SuitPlan::Mixed => "mixed".to_string(),
        SuitPlan::HalfFlush(s) => format!("half_flush_{}", s.to_char()),
        SuitPlan::FullFlush(s) => format!("full_flush_{}", s.to_char()),
        SuitPlan::Honours => "honours".to_string(),
        SuitPlan::Orphans => "orphans".to_string(),
    };
    d.set_item("plan", plan)?;
    let structure = match t.structure {
        Structure::Any => "any",
        Structure::AllPungs => "all_pungs",
        Structure::AllChows => "all_chows",
    };
    d.set_item("structure", structure)?;
    d.set_item(
        "pungs",
        t.pungs.iter().map(|x| x.index() as u32).collect::<Vec<_>>(),
    )?;
    d.set_item("faan", t.faan)?;
    d.set_item("limit", t.limit)?;
    Ok(d)
}

fn require_live(state: &PyState) -> PyResult<usize> {
    state
        .inner
        .current_player()
        .ok_or_else(|| value_error("the hand is over"))
}

/// A copy of the state, taken without holding the borrow while the caller
/// computes with the GIL released (so Python may change the original meanwhile).
fn snapshot(state: &Bound<'_, PyState>) -> PyResult<State> {
    let s = state.borrow();
    require_live(&s)?;
    Ok(s.inner.clone())
}

/// Runs the PIMC solver for the player to act. Returns the candidates, best
/// first, with mean points, 95% interval, win, deal-in and draw rates.
#[pyfunction]
#[pyo3(signature = (state, samples=400, candidates=6, time_ms=None, seed=0))]
pub(crate) fn analyze<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyState>,
    samples: usize,
    candidates: usize,
    time_ms: Option<u64>,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let s = snapshot(state)?;
    let config = PimcConfig {
        samples,
        candidates,
        time_limit: time_ms.map(Duration::from_millis),
        seed,
        ..PimcConfig::default()
    };
    let a = py.detach(move || pimc(&s, &config));
    let d = PyDict::new(py);
    d.set_item("seat", a.seat)?;
    d.set_item("samples", a.samples)?;
    d.set_item("elapsed_ms", a.elapsed.as_secs_f64() * 1000.0)?;
    d.set_item("baseline", a.baseline.code())?;
    d.set_item("choice", a.choose(PimcConfig::default().z).code())?;
    let moves = PyList::empty(py);
    for m in &a.moves {
        let md = PyDict::new(py);
        md.set_item("action", m.action.code())?;
        md.set_item("text", m.action.to_string())?;
        md.set_item("n", m.n)?;
        md.set_item("mean", m.mean)?;
        md.set_item("ci95", m.ci95)?;
        md.set_item("win_rate", m.win_rate)?;
        md.set_item("deal_in_rate", m.deal_in_rate)?;
        md.set_item("draw_rate", m.draw_rate)?;
        md.set_item("diff", m.diff)?;
        md.set_item("diff_se", m.diff_se)?;
        moves.append(md)?;
    }
    d.set_item("moves", moves)?;
    Ok(d)
}

/// The action an agent (spec string, e.g. `"value"` or `"pimc:samples=64"`)
/// chooses for the player to act.
#[pyfunction]
#[pyo3(signature = (state, spec, seed=None))]
pub(crate) fn agent_action(
    py: Python<'_>,
    state: &Bound<'_, PyState>,
    spec: &str,
    seed: Option<u64>,
) -> PyResult<u16> {
    let s = snapshot(state)?;
    let agent = Agent::parse(spec).map_err(value_error)?;
    Ok(py.detach(move || {
        let legal = s.legal_actions_vec();
        agent.choose(&s, &legal, &mut rng_from(seed)).code()
    }))
}

/// The heuristic danger model for `seat`: per-tile deal-in estimates (34) and
/// each seat's estimated chance of waiting.
#[pyfunction]
pub(crate) fn danger<'py>(
    py: Python<'py>,
    state: &PyState,
    seat: usize,
) -> PyResult<Bound<'py, PyDict>> {
    crate::check_seat(seat)?;
    let d = Danger::estimate(&state.inner, seat);
    let out = PyDict::new(py);
    out.set_item(
        "tiles",
        mahjong_tiles::Tile::all_kinds()
            .map(|t| d.tile(t))
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "threat",
        (0..4)
            .map(|p| if p == seat { 0.0 } else { d.threat(p) })
            .collect::<Vec<_>>(),
    )?;
    Ok(out)
}

/// `seat`'s hand: plain shanten, faan-aware shanten and its target plan.
#[pyfunction]
pub(crate) fn hand_progress<'py>(
    py: Python<'py>,
    state: &PyState,
    seat: usize,
) -> PyResult<Bound<'py, PyDict>> {
    crate::check_seat(seat)?;
    let view = SeatView::of(&state.inner, seat);
    let fs = faan_shanten(&view.hand_view(), &view.rules);
    let d = PyDict::new(py);
    d.set_item("shanten", shanten(&view.concealed, view.melds.len()))?;
    d.set_item("faan_shanten", fs.shanten)?;
    d.set_item(
        "target",
        fs.target.as_ref().map(|t| target_dict(py, t)).transpose()?,
    )?;
    Ok(d)
}

/// For the player to act on their turn: every discard with shanten, acceptance,
/// faan-aware shanten, faan-aware acceptance and target plan.
#[pyfunction]
pub(crate) fn discard_table<'py>(py: Python<'py>, state: &PyState) -> PyResult<Bound<'py, PyList>> {
    let seat = require_live(state)?;
    if state.inner.phase_name() != "turn" {
        return Err(value_error("discard_table needs a turn decision"));
    }
    let view = SeatView::of(&state.inner, seat);
    let rows = discards(&view.hand_view(), &view.rules);
    let out = PyList::empty(py);
    for r in rows {
        let d = PyDict::new(py);
        d.set_item("tile", r.tile.index() as u32)?;
        d.set_item("shanten", r.shanten)?;
        d.set_item("ukeire", r.ukeire)?;
        d.set_item("faan_shanten", r.faan_shanten)?;
        d.set_item("faan_ukeire", r.faan_ukeire)?;
        d.set_item(
            "target",
            r.target.as_ref().map(|t| target_dict(py, t)).transpose()?,
        )?;
        out.append(d)?;
    }
    Ok(out)
}

/// Offense-only estimates for each discard of the player to act (no opponents):
/// P(reaching a legal self-drawn win within `draws` draws), mean faan, mean draws.
#[pyfunction]
#[pyo3(signature = (state, draws=None, samples=200, seed=0))]
pub(crate) fn solo<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyState>,
    draws: Option<usize>,
    samples: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyList>> {
    let s = snapshot(state)?;
    let seat = s.current_player().expect("live");
    if s.phase_name() != "turn" {
        return Err(value_error("solo needs a turn decision"));
    }
    let view = SeatView::of(&s, seat);
    let draws = draws.unwrap_or(s.wall_remaining() / 4);
    let rows = py.detach(move || solo_sim(&view, draws, samples, seed));
    let out = PyList::empty(py);
    for r in rows {
        let d = PyDict::new(py);
        d.set_item("tile", r.tile.index() as u32)?;
        d.set_item("win_rate", r.win_rate)?;
        d.set_item("mean_faan", r.mean_faan)?;
        d.set_item("mean_draws", r.mean_draws)?;
        out.append(d)?;
    }
    Ok(out)
}

/// A duplicate tournament: each wall is played in every seating (`seatings=4`:
/// the agents rotated; `seatings=24`: all permutations). Returns the report as a
/// dict in the `hk-tournament/1` JSON shape (`mahjong_search::tournament`):
/// per-agent points with independent and wall-paired bootstrap intervals, rates,
/// pattern shares, deal-ins by winner, liability points, and pairwise paired
/// differences. The older keys (`agents[i].mean`, `ci95`, `wins`, `draws`,
/// `seat_points`, `hands`) keep their meaning.
#[pyfunction]
#[pyo3(signature = (agents, deals, seed=0, seatings=4, bootstrap=2000, **rules))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tournament<'py>(
    py: Python<'py>,
    agents: Vec<String>,
    deals: usize,
    seed: u64,
    seatings: usize,
    bootstrap: usize,
    rules: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    if agents.len() != 4 {
        return Err(value_error("need exactly four agent specs"));
    }
    let seatings =
        Seatings::from_count(seatings).ok_or_else(|| value_error("seatings must be 4 or 24"))?;
    let rules = rules_from(rules)?;
    let parsed = agents
        .iter()
        .map(|s| Agent::parse(s).map_err(value_error))
        .collect::<PyResult<Vec<_>>>()?;
    let arr: [Agent; 4] = parsed.try_into().map_err(|_| value_error("four agents"))?;
    let names: [String; 4] = agents.try_into().map_err(|_| value_error("four agents"))?;
    let config = TournamentConfig {
        deals,
        seatings,
        rules,
        seed,
        bootstrap,
    };
    let json = py.detach(move || run_tournament(&arr, &config).summary().to_json(&names));
    py.import("json")?.call_method1("loads", (json,))
}
