//! Faan scoring (§7). [`score_win`] takes a complete hand and the situation it was
//! won in, and returns the patterns that apply and the capped faan.

use crate::hand::{decompositions, is_thirteen_orphans, Decomposition, Group};
use crate::meld::{Meld, MeldKind};
use crate::rules::RuleSet;
use mahjong_tiles::{Suit, Tile, TileCounts};

/// A row of the faan table (§7). The discriminant is the row number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Pattern {
    SelfDrawn = 1,
    Concealed = 2,
    NoFlowers = 3,
    SeatFlower = 4,
    FullFlowerSet = 5,
    LastTile = 6,
    KongReplacement = 7,
    DoubleKong = 8,
    RobbingKong = 9,
    DragonPung = 10,
    SeatWind = 11,
    PrevailingWind = 12,
    Chicken = 13,
    AllChows = 14,
    AllPungs = 15,
    HalfFlush = 16,
    FullFlush = 17,
    TerminalsAndHonours = 18,
    SmallDragons = 19,
    BigDragons = 20,
    SmallWinds = 21,
    BigWinds = 22,
    AllHonours = 23,
    AllTerminals = 24,
    ThirteenOrphans = 25,
    NineGates = 26,
    FourConcealedPungs = 27,
    FourKongs = 28,
    Heavenly = 29,
    Earthly = 30,
    EightImmortals = 31,
    SevenRobOne = 32,
}

impl Pattern {
    /// All patterns, in table order.
    pub const ALL: [Pattern; 32] = [
        Pattern::SelfDrawn,
        Pattern::Concealed,
        Pattern::NoFlowers,
        Pattern::SeatFlower,
        Pattern::FullFlowerSet,
        Pattern::LastTile,
        Pattern::KongReplacement,
        Pattern::DoubleKong,
        Pattern::RobbingKong,
        Pattern::DragonPung,
        Pattern::SeatWind,
        Pattern::PrevailingWind,
        Pattern::Chicken,
        Pattern::AllChows,
        Pattern::AllPungs,
        Pattern::HalfFlush,
        Pattern::FullFlush,
        Pattern::TerminalsAndHonours,
        Pattern::SmallDragons,
        Pattern::BigDragons,
        Pattern::SmallWinds,
        Pattern::BigWinds,
        Pattern::AllHonours,
        Pattern::AllTerminals,
        Pattern::ThirteenOrphans,
        Pattern::NineGates,
        Pattern::FourConcealedPungs,
        Pattern::FourKongs,
        Pattern::Heavenly,
        Pattern::Earthly,
        Pattern::EightImmortals,
        Pattern::SevenRobOne,
    ];

    /// The row number in the faan table.
    pub fn number(self) -> u8 {
        self as u8
    }

    /// The pattern with row number `n` (1–32).
    pub fn from_number(n: u8) -> Option<Pattern> {
        Pattern::ALL.get((n as usize).checked_sub(1)?).copied()
    }

    /// A limit hand (§7.4), scoring the cap.
    pub fn is_limit(self) -> bool {
        self.number() >= 21
    }

    /// The Chinese name, e.g. `清一色`.
    pub fn chinese(self) -> &'static str {
        [
            "自摸",
            "門前清",
            "無花",
            "正花",
            "一台花",
            "海底撈月",
            "槓上開花",
            "槓上槓",
            "搶槓",
            "三元牌",
            "門風",
            "圈風",
            "雞糊",
            "平糊",
            "對對糊",
            "混一色",
            "清一色",
            "混么九",
            "小三元",
            "大三元",
            "小四喜",
            "大四喜",
            "字一色",
            "清么九",
            "十三么",
            "九子連環",
            "坎坎糊",
            "十八羅漢",
            "天糊",
            "地糊",
            "八仙過海",
            "七搶一",
        ][self as usize - 1]
    }

    /// The English name, e.g. `Full flush`.
    pub fn english(self) -> &'static str {
        [
            "Self-drawn",
            "Concealed hand",
            "No flowers",
            "Seat flower",
            "Full flower set",
            "Last tile",
            "Kong replacement",
            "Double kong",
            "Robbing the kong",
            "Dragon pung",
            "Seat wind",
            "Prevailing wind",
            "Chicken hand",
            "All chows",
            "All pungs",
            "Half flush",
            "Full flush",
            "Terminals and honours",
            "Small three dragons",
            "Big three dragons",
            "Small four winds",
            "Big four winds",
            "All honours",
            "All terminals",
            "Thirteen orphans",
            "Nine gates",
            "Four concealed pungs",
            "Four kongs",
            "Heavenly hand",
            "Earthly hand",
            "Eight immortals",
            "Seven rob one",
        ][self as usize - 1]
    }
}

/// The situation a hand was won in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WinContext {
    /// The winner's seat (its seat wind and flower number follow from it).
    pub seat: usize,
    /// The tile that completed the hand.
    pub win_tile: Option<Tile>,
    /// Drawn from the wall (including replacement draws) rather than claimed.
    pub self_drawn: bool,
    /// The winner's bonus tiles, as a bit mask: bit `i` = bonus tile `34 + i`
    /// (flowers 1–4 are bits 0–3, seasons 1–4 bits 4–7).
    pub bonus: u8,
    /// Won on the last tile of the wall (#6).
    pub last_tile: bool,
    /// Number of kongs declared in this turn before the winning replacement draw
    /// (0 = not a replacement win; 1 = #7; 2 or more = #8).
    pub kong_chain: u8,
    /// Won by robbing a kong (#9).
    pub robbing_kong: bool,
    /// Dealer complete after the initial deal (#29).
    pub heavenly: bool,
    /// Non-dealer won on the dealer's first discard (#30).
    pub earthly: bool,
}

/// The scored value of a winning hand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Score {
    /// The patterns that count, with their faan (limit hands list 0 for each
    /// pattern; the hand scores the cap).
    pub patterns: Vec<(Pattern, u8)>,
    /// Faan before the cap (the cap itself for limit hands).
    pub total: u16,
    /// Faan after the cap: the number that is paid.
    pub faan: u8,
    /// Whether this is a limit hand.
    pub limit: bool,
}

impl Score {
    /// A limit hand made of the given patterns.
    pub fn limit(patterns: Vec<Pattern>, rules: &RuleSet) -> Score {
        Score {
            patterns: patterns.into_iter().map(|p| (p, rules.faan_cap)).collect(),
            total: rules.faan_cap as u16,
            faan: rules.faan_cap,
            limit: true,
        }
    }

    fn from_patterns(patterns: Vec<(Pattern, u8)>, rules: &RuleSet) -> Score {
        let total: u16 = patterns.iter().map(|&(_, f)| f as u16).sum();
        Score {
            patterns,
            total,
            faan: total.min(rules.faan_cap as u16) as u8,
            limit: false,
        }
    }

    /// Whether `pattern` is among the patterns that count.
    pub fn has(&self, pattern: Pattern) -> bool {
        self.patterns.iter().any(|&(p, _)| p == pattern)
    }

    fn key(&self) -> (u8, u16) {
        (self.faan, self.total)
    }
}

/// Faan from bonus tiles (#3–#5). Flowers are bits 0–3, seasons 4–7.
pub fn flower_patterns(bonus: u8, seat: usize, rules: &RuleSet) -> Vec<(Pattern, u8)> {
    let mut out = Vec::new();
    if !rules.flowers {
        return out;
    }
    if bonus == 0 {
        out.push((Pattern::NoFlowers, 1));
        return out;
    }
    let mut seat_flowers = 0;
    for set in [bonus & 0x0F, bonus >> 4] {
        if set == 0x0F {
            out.push((Pattern::FullFlowerSet, 2));
        } else if set & (1 << seat) != 0 {
            seat_flowers += 1;
        }
    }
    if seat_flowers > 0 {
        out.push((Pattern::SeatFlower, seat_flowers));
    }
    out
}

/// Scores a complete hand: `concealed` (including the winning tile) plus the
/// `melds` on the table. Returns `None` if the hand is not complete.
///
/// When the hand splits into melds in several ways, the highest-scoring split is
/// used (§6).
pub fn score_win(
    concealed: &TileCounts,
    melds: &[Meld],
    ctx: &WinContext,
    rules: &RuleSet,
) -> Option<Score> {
    if concealed.total() + 3 * melds.len() != 14 {
        return None;
    }
    let mut special = Vec::new();
    if ctx.heavenly {
        special.push(Pattern::Heavenly);
    }
    if ctx.earthly {
        special.push(Pattern::Earthly);
    }
    if melds.is_empty() && is_thirteen_orphans(concealed) {
        special.push(Pattern::ThirteenOrphans);
        return Some(Score::limit(special, rules));
    }
    let mut best: Option<Score> = None;
    for d in decompositions(concealed) {
        let s = score_decomposition(concealed, melds, &d, ctx, rules, &special);
        if best.as_ref().is_none_or(|b| s.key() > b.key()) {
            best = Some(s);
        }
    }
    best
}

/// A meld of the complete hand, from the table or from the concealed split.
#[derive(Clone, Copy)]
struct HandMeld {
    pung_like: bool,
    kong: bool,
    tile: Tile,
    concealed: bool,
}

fn score_decomposition(
    concealed: &TileCounts,
    melds: &[Meld],
    d: &Decomposition,
    ctx: &WinContext,
    rules: &RuleSet,
    special: &[Pattern],
) -> Score {
    let mut all: Vec<HandMeld> = melds
        .iter()
        .map(|m| HandMeld {
            pung_like: m.is_pung_like(),
            kong: m.kind == MeldKind::Kong,
            tile: m.tile,
            concealed: m.concealed,
        })
        .collect();
    for g in &d.groups {
        let (pung_like, tile) = match *g {
            Group::Pung(t) => (true, t),
            Group::Chow(t) => (false, t),
        };
        all.push(HandMeld {
            pung_like,
            kong: false,
            tile,
            concealed: true,
        });
    }

    // Which tiles the whole hand uses (for flushes and terminal patterns).
    let mut every = *concealed;
    for m in melds {
        m.add_to(&mut every);
    }
    let suits: Vec<Suit> = Suit::ALL
        .into_iter()
        .filter(|&s| every.suit_counts(s).iter().any(|&c| c > 0))
        .collect();
    let honours = every.kinds().any(|t| t.is_honour());
    let only_terminals_honours = every.kinds().all(|t| t.is_terminal_or_honour());
    let terminals = every.kinds().any(|t| t.is_terminal());

    let pungs_of =
        |pred: fn(Tile) -> bool| all.iter().filter(|m| m.pung_like && pred(m.tile)).count();
    let dragon_pungs = pungs_of(Tile::is_dragon);
    let wind_pungs = pungs_of(Tile::is_wind);
    let all_pungs = all.iter().all(|m| m.pung_like);
    let all_chows = all.iter().all(|m| !m.pung_like);

    // Four concealed pungs: on a discard win the claimed tile makes its pung
    // exposed, unless the winning tile can be read as the pair or part of a chow.
    let mut concealed_pungs = all.iter().filter(|m| m.pung_like && m.concealed).count();
    if !ctx.self_drawn {
        if let Some(w) = ctx.win_tile {
            let in_chow = d.groups.iter().any(|g| match *g {
                Group::Chow(t) => (0..3).any(|k| t.offset(k) == Some(w)),
                Group::Pung(_) => false,
            });
            let in_pung = d.groups.contains(&Group::Pung(w));
            if d.pair != w && !in_chow && in_pung {
                concealed_pungs -= 1;
            }
        }
    }

    // Limit hands (§7.4).
    let mut limits: Vec<Pattern> = special.to_vec();
    if wind_pungs == 4 {
        limits.push(Pattern::BigWinds);
    } else if wind_pungs == 3 && d.pair.is_wind() {
        limits.push(Pattern::SmallWinds);
    }
    if suits.is_empty() {
        limits.push(Pattern::AllHonours);
    }
    if !honours && only_terminals_honours {
        limits.push(Pattern::AllTerminals);
    }
    if melds.is_empty() && is_nine_gates(concealed, ctx.win_tile) {
        limits.push(Pattern::NineGates);
    }
    if concealed_pungs == 4 {
        limits.push(Pattern::FourConcealedPungs);
    }
    if all.iter().filter(|m| m.kong).count() == 4 {
        limits.push(Pattern::FourKongs);
    }
    if !limits.is_empty() {
        return Score::limit(limits, rules);
    }

    let mut p: Vec<(Pattern, u8)> = Vec::new();
    // Situational (§7.1).
    if ctx.kong_chain >= 2 {
        p.push((Pattern::DoubleKong, 8));
    } else {
        if ctx.self_drawn {
            p.push((Pattern::SelfDrawn, 1));
        }
        if ctx.kong_chain == 1 {
            p.push((Pattern::KongReplacement, 1));
        }
    }
    if melds.iter().all(|m| m.concealed) {
        p.push((Pattern::Concealed, 1));
    }
    p.extend(flower_patterns(ctx.bonus, ctx.seat, rules));
    if ctx.last_tile {
        p.push((Pattern::LastTile, 1));
    }
    if ctx.robbing_kong {
        p.push((Pattern::RobbingKong, 1));
    }

    // Meld patterns (§7.2) and dragons (#19, #20).
    let hand_start = p.len();
    if dragon_pungs == 3 {
        p.push((Pattern::BigDragons, 8));
    } else if dragon_pungs == 2 && d.pair.is_dragon() {
        p.push((Pattern::SmallDragons, 5));
    } else if dragon_pungs > 0 {
        p.push((Pattern::DragonPung, dragon_pungs as u8));
    }
    let has_pung = |t: Tile| all.iter().any(|m| m.pung_like && m.tile == t);
    if has_pung(Tile::wind(ctx.seat)) {
        p.push((Pattern::SeatWind, 1));
    }
    if has_pung(Tile::wind(rules.prevailing_wind)) {
        p.push((Pattern::PrevailingWind, 1));
    }

    // Whole-hand patterns (§7.3).
    if all_chows {
        p.push((Pattern::AllChows, 1));
    }
    if all_pungs {
        p.push((Pattern::AllPungs, 3));
    }
    if suits.len() == 1 {
        if honours {
            p.push((Pattern::HalfFlush, 3));
        } else {
            p.push((Pattern::FullFlush, 7));
        }
    }
    if only_terminals_honours && honours && terminals {
        p.push((Pattern::TerminalsAndHonours, 1));
    }
    if p.len() == hand_start {
        p.push((Pattern::Chicken, 0));
    }
    Score::from_patterns(p, rules)
}

/// Nine gates (#26): one suit, and the 13 tiles before the winning tile are
/// exactly `1112345678999`.
fn is_nine_gates(concealed: &TileCounts, win_tile: Option<Tile>) -> bool {
    let Some(w) = win_tile else { return false };
    let Some(suit) = w.suit() else { return false };
    let mut before = *concealed;
    if before[w] == 0 {
        return false;
    }
    before.remove(w);
    before.suit_counts(suit) == [3, 1, 1, 1, 1, 1, 1, 1, 3] && before.total() == 13
}

#[cfg(test)]
mod tests {
    use super::*;
    use mahjong_tiles::parse_tiles;

    fn counts(s: &str) -> TileCounts {
        TileCounts::from_tiles(&parse_tiles(s).unwrap())
    }

    fn t(s: &str) -> Tile {
        s.parse().unwrap()
    }

    #[test]
    fn pattern_numbers_and_names() {
        for (i, p) in Pattern::ALL.iter().enumerate() {
            assert_eq!(p.number() as usize, i + 1);
            assert_eq!(Pattern::from_number(p.number()), Some(*p));
        }
        assert_eq!(Pattern::FullFlush.chinese(), "清一色");
        assert!(Pattern::SmallWinds.is_limit() && !Pattern::BigDragons.is_limit());
    }

    #[test]
    fn flower_faan_spec_7_1() {
        let r = RuleSet::default();
        assert_eq!(flower_patterns(0, 1, &r), vec![(Pattern::NoFlowers, 1)]);
        // South (seat 1) holds F2 and S2: two seat flowers.
        assert_eq!(
            flower_patterns(0b0010_0010, 1, &r),
            vec![(Pattern::SeatFlower, 2)]
        );
        // F1–F4 complete, plus S2 for South.
        assert_eq!(
            flower_patterns(0b0010_1111, 1, &r),
            vec![(Pattern::FullFlowerSet, 2), (Pattern::SeatFlower, 1)]
        );
        // Off-seat flower only: nothing.
        assert!(flower_patterns(0b0000_0100, 1, &r).is_empty());
        let off = RuleSet {
            flowers: false,
            ..r
        };
        assert!(flower_patterns(0, 1, &off).is_empty());
    }

    #[test]
    fn nine_gates_needs_the_exact_13() {
        assert!(is_nine_gates(&counts("11123455678999m"), Some(t("5m"))));
        assert!(!is_nine_gates(&counts("11123455678999m"), Some(t("1m"))));
    }
}
