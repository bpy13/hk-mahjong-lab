//! The `mahjong._core` Python extension module.
//!
//! The Python side sees only plain values:
//!
//! * a **tile** is an `int` index (`docs/rules/hk.md` §1: `1m`–`9m` = 0–8,
//!   `1p`–`9p` = 9–17, `1s`–`9s` = 18–26, `E S W N` = 27–30, `Wh G R` = 31–33,
//!   `F1`–`F4` = 34–37, `S1`–`S4` = 38–41);
//! * an **action** is an `int` code (see [`mahjong_hk::Action::code`]);
//! * rule variants are keyword arguments (`flowers=False`, `payout="full_spicy"`).
//!
//! The friendlier Python API is in `python/mahjong/hk.py`.

mod convert;
mod search;

use convert::{event_dict, result_dict, rules_from, score_dict, tile_from};
use mahjong_games::Game;
use mahjong_hk::{Action, State, NUM_ACTION_CODES};
use mahjong_tiles::{format_tiles as fmt_tiles, parse_tiles as parse, Tile};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::rngs::SmallRng;
use rand::SeedableRng;

pub(crate) fn value_error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

pub(crate) fn rng_from(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    }
}

/// Parses tile text such as `"123m 55p EEE Wh F2"` into tile indices.
#[pyfunction]
fn parse_tiles(text: &str) -> PyResult<Vec<u32>> {
    parse(text)
        .map(|v| v.into_iter().map(|t| t.index() as u32).collect())
        .map_err(value_error)
}

/// Formats tile indices in sorted, grouped notation (`"123m 55p EEE"`).
#[pyfunction]
fn format_tiles(tiles: Vec<u32>) -> PyResult<String> {
    let tiles = tiles
        .into_iter()
        .map(tile_from)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(fmt_tiles(&tiles))
}

/// Describes an action code in English (`"discard 5p"`, `"chow from 3s"`).
#[pyfunction]
fn action_text(code: u16) -> PyResult<String> {
    Action::from_code(code)
        .map(|a| a.to_string())
        .ok_or_else(|| value_error(format!("invalid action code {code}")))
}

/// A hand of Hong Kong mahjong.
#[pyclass(name = "HkState", module = "mahjong._core")]
#[derive(Clone)]
pub struct PyState {
    pub(crate) inner: State,
}

#[pymethods]
impl PyState {
    /// Shuffles and deals a new hand. Rule variants are keyword arguments.
    #[new]
    #[pyo3(signature = (seed=None, **rules))]
    fn new(seed: Option<u64>, rules: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let rules = rules_from(rules)?;
        Ok(PyState {
            inner: State::new_deal(rules, &mut rng_from(seed)),
        })
    }

    /// Deals from a wall given in draw order (front first; replacements are
    /// drawn from the end).
    #[staticmethod]
    #[pyo3(signature = (wall, **rules))]
    fn from_wall(wall: Vec<u32>, rules: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let rules = rules_from(rules)?;
        let wall = wall
            .into_iter()
            .map(tile_from)
            .collect::<PyResult<Vec<Tile>>>()?;
        let mut expected = State::all_tiles(&rules);
        let mut got = wall.clone();
        expected.sort();
        got.sort();
        if got != expected {
            return Err(value_error("wall must contain exactly the tiles in play"));
        }
        Ok(PyState {
            inner: State::from_wall(rules, wall),
        })
    }

    /// The seat to act, or `None` when the hand is over.
    fn current_player(&self) -> Option<usize> {
        self.inner.current_player()
    }

    /// `"turn"`, `"claim"`, `"rob"` or `"over"`.
    fn phase(&self) -> &'static str {
        self.inner.phase_name()
    }

    fn is_terminal(&self) -> bool {
        self.inner.is_terminal()
    }

    /// Legal action codes for the player to act.
    fn legal_actions(&self) -> Vec<u16> {
        self.inner
            .legal_actions_vec()
            .into_iter()
            .map(Action::code)
            .collect()
    }

    /// Applies a legal action code.
    fn apply(&mut self, code: u16) -> PyResult<()> {
        let a = Action::from_code(code)
            .ok_or_else(|| value_error(format!("invalid action code {code}")))?;
        if !self.inner.legal_actions_vec().contains(&a) {
            return Err(value_error(format!("illegal action: {a}")));
        }
        self.inner.apply(a);
        Ok(())
    }

    /// A copy of the state.
    fn copy(&self) -> Self {
        self.clone()
    }

    /// A state `observer` cannot tell apart from this one, with everything hidden
    /// from them re-dealt at random.
    #[pyo3(signature = (observer, seed=None))]
    fn determinize(&self, observer: usize, seed: Option<u64>) -> Self {
        PyState {
            inner: self.inner.determinize(observer, &mut rng_from(seed)),
        }
    }

    /// The undrawn wall in draw order. Hidden information.
    fn wall(&self) -> Vec<u32> {
        self.inner.wall().iter().map(|t| t.index() as u32).collect()
    }

    /// `seat`'s concealed tiles (sorted). Private to that seat.
    fn hand(&self, seat: usize) -> PyResult<Vec<u32>> {
        check_seat(seat)?;
        Ok(self
            .inner
            .hand(seat)
            .tiles()
            .iter()
            .map(|t| t.index() as u32)
            .collect())
    }

    /// What `seat` can see, as a dict (their own hand and all public information).
    fn observation<'py>(&self, py: Python<'py>, seat: usize) -> PyResult<Bound<'py, PyDict>> {
        check_seat(seat)?;
        convert::observation(py, &self.inner, seat)
    }

    /// The public event log, as a list of dicts.
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let items = self
            .inner
            .events()
            .iter()
            .map(|e| event_dict(py, e))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    /// The result once the hand is over: wins (with faan breakdown and payments)
    /// and net points per seat.
    fn result<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner.result().map(|r| result_dict(py, r)).transpose()
    }

    /// The score of a self-drawn win available now, if any.
    fn self_drawn_score<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .self_drawn_score()
            .map(|s| score_dict(py, &s))
            .transpose()
    }

    /// The score `seat` would get by winning on the tile being claimed or robbed.
    fn claim_score<'py>(
        &self,
        py: Python<'py>,
        seat: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        check_seat(seat)?;
        self.inner
            .claim_score(seat)
            .map(|s| score_dict(py, &s))
            .transpose()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

pub(crate) fn check_seat(seat: usize) -> PyResult<()> {
    if seat < 4 {
        Ok(())
    } else {
        Err(value_error(format!("seat {seat} out of range")))
    }
}

/// Scores a complete hand (§7). `concealed` includes the winning tile; `melds`
/// are `(kind, tile, concealed)` with kind `"chow"` (lowest tile), `"pung"` or
/// `"kong"`; `bonus` lists bonus tiles (34–41). Returns `None` if incomplete.
#[pyfunction]
#[pyo3(signature = (concealed, melds, seat, win_tile, self_drawn, bonus=Vec::new(), last_tile=false,
    kong_chain=0, robbing_kong=false, heavenly=false, earthly=false, **rules))]
#[allow(clippy::too_many_arguments)]
fn score_hand<'py>(
    py: Python<'py>,
    concealed: Vec<u32>,
    melds: Vec<(String, u32, bool)>,
    seat: usize,
    win_tile: Option<u32>,
    self_drawn: bool,
    bonus: Vec<u32>,
    last_tile: bool,
    kong_chain: u8,
    robbing_kong: bool,
    heavenly: bool,
    earthly: bool,
    rules: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    use mahjong_hk::{score_win, Meld, MeldKind, WinContext};
    check_seat(seat)?;
    let rules = rules_from(rules)?;
    let tiles = concealed
        .into_iter()
        .map(tile_from)
        .collect::<PyResult<Vec<_>>>()?;
    if tiles.iter().any(|t| t.is_bonus()) {
        return Err(value_error("bonus tiles are not part of a hand"));
    }
    let counts = mahjong_tiles::TileCounts::from_tiles(&tiles);
    let mut ms = Vec::new();
    for (kind, tile, concealed) in melds {
        let tile = tile_from(tile)?;
        let kind = match kind.as_str() {
            "chow" if tile.offset(2).is_some() => MeldKind::Chow,
            "pung" => MeldKind::Pung,
            "kong" => MeldKind::Kong,
            _ => return Err(value_error(format!("bad meld {kind} {tile}"))),
        };
        ms.push(if concealed && kind == MeldKind::Kong {
            Meld::concealed_kong(tile)
        } else {
            Meld::exposed(kind, tile, (seat + 1) % 4)
        });
    }
    let mut mask = 0u8;
    for b in bonus {
        let t = tile_from(b)?;
        if !t.is_bonus() {
            return Err(value_error(format!("{t} is not a bonus tile")));
        }
        mask |= 1 << (t.index() - 34);
    }
    let ctx = WinContext {
        seat,
        win_tile: win_tile.map(tile_from).transpose()?,
        self_drawn,
        bonus: mask,
        last_tile,
        kong_chain,
        robbing_kong,
        heavenly,
        earthly,
    };
    score_win(&counts, &ms, &ctx, &rules)
        .map(|s| score_dict(py, &s))
        .transpose()
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("NUM_ACTION_CODES", NUM_ACTION_CODES)?;
    m.add_function(wrap_pyfunction!(parse_tiles, m)?)?;
    m.add_function(wrap_pyfunction!(format_tiles, m)?)?;
    m.add_function(wrap_pyfunction!(action_text, m)?)?;
    m.add_function(wrap_pyfunction!(score_hand, m)?)?;
    m.add_function(wrap_pyfunction!(convert::pattern_names, m)?)?;
    m.add_function(wrap_pyfunction!(search::analyze, m)?)?;
    m.add_function(wrap_pyfunction!(search::agent_action, m)?)?;
    m.add_function(wrap_pyfunction!(search::danger, m)?)?;
    m.add_function(wrap_pyfunction!(search::hand_progress, m)?)?;
    m.add_function(wrap_pyfunction!(search::discard_table, m)?)?;
    m.add_function(wrap_pyfunction!(search::solo, m)?)?;
    m.add_function(wrap_pyfunction!(search::tournament, m)?)?;
    m.add_class::<PyState>()?;
    Ok(())
}
