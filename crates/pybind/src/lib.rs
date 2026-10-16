//! The `mahjong._core` Python extension module.
//!
//! The Python side sees only plain values: a tile is an `int` index
//! (`docs/rules/hk.md` §1: `1m`–`9m` = 0–8, `1p`–`9p` = 9–17, `1s`–`9s` = 18–26,
//! `E S W N` = 27–30, `Wh G R` = 31–33, `F1`–`F4` = 34–37, `S1`–`S4` = 38–41).

use mahjong_tiles::{format_tiles as fmt_tiles, parse_tiles as parse, Tile, NUM_TILES_WITH_BONUS};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Parses tile text such as `"123m 55p EEE Wh F2"` into tile indices.
#[pyfunction]
fn parse_tiles(text: &str) -> PyResult<Vec<u32>> {
    parse(text)
        .map(|v| v.into_iter().map(|t| t.index() as u32).collect())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Formats tile indices in sorted, grouped notation (`"123m 55p EEE"`).
#[pyfunction]
fn format_tiles(tiles: Vec<u32>) -> PyResult<String> {
    let tiles = tiles
        .into_iter()
        .map(|i| {
            if (i as usize) < NUM_TILES_WITH_BONUS {
                Ok(Tile::from_index(i as u8))
            } else {
                Err(PyValueError::new_err(format!(
                    "tile index {i} out of range"
                )))
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(fmt_tiles(&tiles))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(parse_tiles, m)?)?;
    m.add_function(wrap_pyfunction!(format_tiles, m)?)?;
    Ok(())
}
