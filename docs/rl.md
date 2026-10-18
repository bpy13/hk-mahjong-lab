# Reinforcement learning

Phase 3 trains networks to play Hong Kong mahjong and compares them with the
Phase 1 bots and solver. **Status:** the full study ran on 2026-10-18;
results are in [study-results/phase3-rl.md](study-results/phase3-rl.md). Code:
`crates/search/src/features.rs` (observation encoding), `crates/search/src/net.rs`
(running networks in Rust), `crates/pybind/src/rl.rs` (batched environments,
teacher data) and `python/mahjong/rl/` (training).

## 1. Why not supervised learning first?

The strongest mahjong programs (Suphx, NAGA, Mortal, LuckyJ) are Japanese Riichi
programs that start by imitating millions of human games from Tenhou. No such
collection exists for Hong Kong rules, and Riichi games teach the wrong lessons
(no 3-faan minimum, no flowers, different scoring and claiming). So the
"human" stage is replaced by imitating the best rule-based bot (`value`), and
everything after that is self-play.

## 2. What the network sees

Features are computed in Rust from the acting seat's point of view (seats are
relative: me, next, across, previous), 933 floats:

* 26 tile planes of 34: own hand (one plane per copy count); melds, discards and
  the last three discards of each seat; unseen copies; the tile being claimed;
  the tile just drawn; scoring tiles (dragons and the two value winds); the
  heuristic danger of each tile; and, for each tile that could be discarded,
  plain and faan-aware shanten after it, plain and faan-aware acceptance after
  it, and the faan its target guarantees;
* 49 scalars: decision phase, who discarded the claimed tile, wall left, winds,
  bonus tiles, exposed melds and hand sizes, liability, each opponent's threat,
  own shanten, faan-aware shanten, target faan, whether still concealed, and the
  minimum faan.

The hand-made planes (danger, shanten and acceptance after each discard) give the network the
results of the Phase 1 analysis for free; it can learn to trust or ignore them.

**Oracle features** (136 floats: each opponent's concealed tiles and the next 16
wall tiles, as counts per kind) are appended for training only, and are zero at play time.

A test checks that two positions that differ only in hidden tiles produce
identical features, so the network cannot see what it should not.

## 3. Network

```
features (933) + oracle (136) → 512 → 512 → 256 ─┬─ q   (134)  points per action (DMC)
                                                 ├─ pi  (134)  policy logits (imitation, PPO)
                                                 └─ v   (1)    points (PPO baseline)
```

Every action is one of 134 codes (discard each tile, kongs, pass, win, pung,
chow by lowest tile), so one forward pass scores every legal action; illegal ones
are masked. Points are divided by 32 for training. After training, the torso and
one head are exported to a small binary format (`MJNN`) that Rust runs directly,
so networks play in the same duplicate tournaments as the bots and can drive
search. Every agent, networks included, always declares a legal win.

## 4. Methods

**Distillation** (`distill.py`). Hands are played by the `value` bot in all four
seats, generated in Rust in parallel, in chunks so they fit in memory. The policy
head learns the bot's action (cross-entropy), the Q head the final points after
the action taken, and the value head the final points. This network starts every
other run.

**Deep Monte Carlo** (`dmc.py`, DouZero, Zha et al. 2021). The Q head plays all
four seats, ε-greedily (ε = 0.05). Every decision of a finished hand is labelled
with the acting seat's final points, a Monte Carlo return with no bootstrapping,
and Q(state, action taken) is fitted to it by mean squared error over the last
three iterations' hands.

**PPO with oracle guiding** (`ppo.py`, Suphx, Li et al. 2020). The policy head
plays all four seats by sampling. Early in training each decision keeps the
oracle features with probability *keep*, which falls linearly from 1 to 0 over the
first half of training: the network first learns with perfect information (which
is far easier: it can see who is about to win and on which tile), then is weaned
off it. Advantages are final points minus the value estimate (rewards come only at
the end of a hand, so no discounting is needed). Clipped objective (ε = 0.2),
3 epochs per iteration, entropy bonus 0.005.

**Network + search** (`netpimc` agent). The network proposes the candidates
and its top choice is the baseline that PIMC must beat by two standard errors of
the paired difference (as in Phase 1). Rollouts are played either by the `fast`
bot or by the network itself in every seat.

## 5. Running it

```sh
python experiments/phase3_rl.py --quick     # a few minutes, checks everything runs
python experiments/phase3_rl.py             # the full study (about 5.3 hours on an M1)
```

Stages resume: a stage whose model already exists in `models/` is skipped.
