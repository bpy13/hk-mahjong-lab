//! Snapshot of the faan keys (plan AC10). The app stream keys its faan text on
//! `Pattern::english`, so these strings must never change. If this test fails,
//! the engine change is wrong, not the test.

use mahjong_hk::Pattern;

/// Every faan-table row (§7) with its English key, in table order.
const EXPECTED: [(u8, &str); 32] = [
    (1, "Self-drawn"),
    (2, "Concealed hand"),
    (3, "No flowers"),
    (4, "Seat flower"),
    (5, "Full flower set"),
    (6, "Last tile"),
    (7, "Kong replacement"),
    (8, "Double kong"),
    (9, "Robbing the kong"),
    (10, "Dragon pung"),
    (11, "Seat wind"),
    (12, "Prevailing wind"),
    (13, "Chicken hand"),
    (14, "All chows"),
    (15, "All pungs"),
    (16, "Half flush"),
    (17, "Full flush"),
    (18, "Terminals and honours"),
    (19, "Small three dragons"),
    (20, "Big three dragons"),
    (21, "Small four winds"),
    (22, "Big four winds"),
    (23, "All honours"),
    (24, "All terminals"),
    (25, "Thirteen orphans"),
    (26, "Nine gates"),
    (27, "Four concealed pungs"),
    (28, "Four kongs"),
    (29, "Heavenly hand"),
    (30, "Earthly hand"),
    (31, "Eight immortals"),
    (32, "Seven rob one"),
];

#[test]
fn every_pattern_keeps_its_english_key() {
    assert_eq!(
        Pattern::ALL.len(),
        EXPECTED.len(),
        "a pattern was added or removed"
    );
    for (p, &(number, english)) in Pattern::ALL.iter().zip(EXPECTED.iter()) {
        assert_eq!(p.number(), number, "row number of {p:?}");
        assert_eq!(p.english(), english, "English key of row {number}");
        assert_eq!(Pattern::from_number(number), Some(*p));
    }
}
