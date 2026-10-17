//! Conversions between engine values and Python dicts.

use crate::value_error;
use mahjong_hk::{Event, HandResult, Meld, MeldKind, Pattern, RuleSet, Score, State};
use mahjong_tiles::{Tile, NUM_TILES_WITH_BONUS};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

pub(crate) fn tile_from(i: u32) -> PyResult<Tile> {
    if (i as usize) < NUM_TILES_WITH_BONUS {
        Ok(Tile::from_index(i as u8))
    } else {
        Err(value_error(format!("tile index {i} out of range")))
    }
}

fn idx(t: Tile) -> u32 {
    t.index() as u32
}

/// Builds a `RuleSet` from keyword arguments; unknown names are an error.
pub(crate) fn rules_from(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<RuleSet> {
    let mut r = RuleSet::default();
    let Some(kwargs) = kwargs else { return Ok(r) };
    for (k, v) in kwargs.iter() {
        let key: String = k.extract()?;
        match key.as_str() {
            "flowers" => r.flowers = v.extract()?,
            "min_faan" => r.min_faan = v.extract()?,
            "faan_cap" => r.faan_cap = v.extract()?,
            "payout" => r.payout = v.extract::<String>()?.parse().map_err(value_error)?,
            "multiple_winners" => {
                r.multiple_winners = v.extract::<String>()?.parse().map_err(value_error)?
            }
            "twelve_tile_liability" => r.twelve_tile_liability = v.extract()?,
            "prevailing_wind" => {
                let w: usize = v.extract()?;
                if w > 3 {
                    return Err(value_error("prevailing_wind must be 0-3"));
                }
                r.prevailing_wind = w;
            }
            _ => return Err(value_error(format!("unknown rule variant {key:?}"))),
        }
    }
    if r.faan_cap > 13 || r.min_faan > r.faan_cap {
        return Err(value_error("need min_faan <= faan_cap <= 13"));
    }
    Ok(r)
}

/// `{number: (chinese, english)}` for every faan-table row.
#[pyfunction]
pub(crate) fn pattern_names() -> Vec<(u8, &'static str, &'static str)> {
    Pattern::ALL
        .iter()
        .map(|p| (p.number(), p.chinese(), p.english()))
        .collect()
}

pub(crate) fn score_dict<'py>(py: Python<'py>, s: &Score) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("faan", s.faan)?;
    d.set_item("total", s.total)?;
    d.set_item("qualifying", s.qualifying)?;
    d.set_item("limit", s.limit)?;
    let patterns: Vec<(u8, u8)> = s.patterns.iter().map(|&(p, f)| (p.number(), f)).collect();
    d.set_item("patterns", patterns)?;
    Ok(d)
}

fn meld_dict<'py>(py: Python<'py>, m: &Meld) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let kind = match m.kind {
        MeldKind::Chow => "chow",
        MeldKind::Pung => "pung",
        MeldKind::Kong => "kong",
    };
    d.set_item("kind", kind)?;
    d.set_item("tile", idx(m.tile))?;
    d.set_item("tiles", m.tiles().into_iter().map(idx).collect::<Vec<_>>())?;
    d.set_item("concealed", m.concealed)?;
    d.set_item("from", m.claimed_from)?;
    Ok(d)
}

pub(crate) fn event_dict<'py>(py: Python<'py>, e: &Event) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match *e {
        Event::Draw { seat, replacement } => {
            d.set_item("type", "draw")?;
            d.set_item("seat", seat)?;
            d.set_item("replacement", replacement)?;
        }
        Event::Bonus { seat, tile } => {
            d.set_item("type", "bonus")?;
            d.set_item("seat", seat)?;
            d.set_item("tile", idx(tile))?;
        }
        Event::Discard { seat, tile } => {
            d.set_item("type", "discard")?;
            d.set_item("seat", seat)?;
            d.set_item("tile", idx(tile))?;
        }
        Event::Claim { seat, from, meld } => {
            d.set_item("type", "claim")?;
            d.set_item("seat", seat)?;
            d.set_item("from", from)?;
            d.set_item("meld", meld_dict(py, &meld)?)?;
        }
        Event::ConcealedKong { seat, tile } => {
            d.set_item("type", "concealed_kong")?;
            d.set_item("seat", seat)?;
            d.set_item("tile", idx(tile))?;
        }
        Event::AddedKong { seat, tile } => {
            d.set_item("type", "added_kong")?;
            d.set_item("seat", seat)?;
            d.set_item("tile", idx(tile))?;
        }
        Event::Win { seat } => {
            d.set_item("type", "win")?;
            d.set_item("seat", seat)?;
        }
        Event::ExhaustiveDraw => d.set_item("type", "exhaustive_draw")?,
    }
    Ok(d)
}

pub(crate) fn result_dict<'py>(py: Python<'py>, r: &HandResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let wins = PyList::empty(py);
    for w in &r.wins {
        let wd = PyDict::new(py);
        wd.set_item("seat", w.seat)?;
        wd.set_item("score", score_dict(py, &w.score)?)?;
        wd.set_item("self_drawn", w.self_drawn)?;
        wd.set_item("from", w.from)?;
        wd.set_item("win_tile", w.win_tile.map(idx))?;
        wd.set_item("paid_by", w.paid_by.to_vec())?;
        wd.set_item("liable", w.liable)?;
        wd.set_item("robbed", w.robbed)?;
        wins.append(wd)?;
    }
    d.set_item("wins", wins)?;
    d.set_item("deltas", r.deltas.to_vec())?;
    Ok(d)
}

fn bonus_tiles(mask: u8) -> Vec<u32> {
    (0..8)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| 34 + i)
        .collect()
}

/// Everything `seat` can see.
pub(crate) fn observation<'py>(
    py: Python<'py>,
    s: &State,
    seat: usize,
) -> PyResult<Bound<'py, PyDict>> {
    use mahjong_games::Game;
    let d = PyDict::new(py);
    d.set_item("seat", seat)?;
    d.set_item("phase", s.phase_name())?;
    d.set_item("current_player", s.current_player())?;
    d.set_item("turn", s.turn())?;
    d.set_item("wall_remaining", s.wall_remaining())?;
    d.set_item(
        "hand",
        s.hand(seat)
            .tiles()
            .into_iter()
            .map(idx)
            .collect::<Vec<_>>(),
    )?;
    let drawn = if s.current_player() == Some(seat) {
        s.last_drawn()
    } else {
        None
    };
    d.set_item("last_drawn", drawn.map(idx))?;
    let melds = PyList::empty(py);
    let bonus = PyList::empty(py);
    let counts = PyList::empty(py);
    for p in 0..4 {
        let ms = s
            .melds(p)
            .iter()
            .map(|m| meld_dict(py, m))
            .collect::<PyResult<Vec<_>>>()?;
        melds.append(PyList::new(py, ms)?)?;
        bonus.append(bonus_tiles(s.bonus(p)))?;
        counts.append(s.concealed_count(p))?;
    }
    d.set_item("melds", melds)?;
    d.set_item("bonus", bonus)?;
    d.set_item("concealed_counts", counts)?;
    let river: Vec<(usize, u32, bool)> = s
        .river()
        .into_iter()
        .map(|(p, t, c)| (p, idx(t), c))
        .collect();
    d.set_item("river", river)?;
    d.set_item("claim", s.claim_tile().map(|(t, from)| (idx(t), from)))?;
    d.set_item(
        "liable",
        (0..4).map(|p| s.liable_for(p)).collect::<Vec<_>>(),
    )?;
    let legal: Vec<u16> = if s.current_player() == Some(seat) {
        s.legal_actions_vec()
            .into_iter()
            .map(|a| a.code())
            .collect()
    } else {
        Vec::new()
    };
    d.set_item("legal", legal)?;
    d.set_item("unseen", s.unseen(seat).0.to_vec())?;
    Ok(d)
}
