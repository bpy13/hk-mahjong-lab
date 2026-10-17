# What others have done, and what it means for us

A survey of published mahjong AI (October 2026), compared with this project's
solver (Phase 1) and learning pipeline (Phase 3), ending with a ranked list of
changes. Sources are linked at the end.

## 1. The landscape

| System | Variant | Method | Data / compute | Strength |
|---|---|---|---|---|
| Mizukami & Tsuruoka 2015 | Riichi | Monte Carlo simulation with three learned **opponent models** (waiting, winning tiles, score; logistic / linear regression) | 17–72 M positions of expert logs | Tenhou rating 1718 (above average human) |
| Kurita & Hoki 2021 | Riichi | Search over **abstracted MDPs** (hands a few exchanges from ready) | — | Tenhou, strong amateur |
| NAGA (Dwango) | Riichi | Supervised CNNs on top-player logs, **no RL** | Tenhou logs | Tenhou 10 dan |
| Suphx (MSRA 2020) | Riichi | 5 CNNs (discard, riichi, chow, pong, kong) + rule-based win; SL then RL (policy gradient), **global reward prediction**, **oracle guiding**, run-time adaptation (pMCPA) | 4–15 M SL samples per model; 1.5 M self-play games, distributed CPU+GPU | Tenhou stable 8.74 dan, above 99.99% of humans |
| Mortal (open source) | Riichi | Rust engine; ResNet (40 blocks × 192, channel attention), dueling **DQN with CQL** (offline RL from logs), GRU **rank predictor** for reward shaping | Tenhou logs | strong; widely used for review |
| LuckyJ (Tencent 2023) | Riichi | **ACH**: actor-critic minimizing weighted counterfactual regret (Hedge), a neural CFR | large-scale self-play | Tenhou 10 dan in 1,321 games (stable 10.68) |
| IJCAI 2020/22/23 (Botzone) | Chinese Official (MCR, 8-fan minimum) | 2020: heuristics → PPO + IMPALA from companies with hundreds of cores; 2022–23: **SL dominates** top 16 (ResNet/ResNeXt on tile images, suit-swap augmentation, shanten features) | human logs (2020), AI logs (2022–23) | top teams ≈ advanced humans |
| RVR (Tencent/CASIA 2022) | MCR | PPO from scratch with **relative value network** (critic sees hidden tiles, predicts all 4 seats' values, zero-sum) and **expected reward network** | 8 GPUs × 3 days | beat 62.5% of IJCAI top-16 |
| Tjong (2024) | MCR | Transformer, two-stage decision (action type, then tile), "fan backward" reward shaping | 0.5 M SL samples, 2 GPUs × 7 days, then 3 days RL | top 1% on Botzone |
| PerfectDou (NetEase 2022) | DouDizhu | PPO + GAE, **critic sees all cards, actor does not** ("perfect-training-imperfect-execution") | — | state of the art in DouDizhu |
| DouZero (Kuaishou 2021) | DouDizhu | Deep Monte Carlo (Q of state-action vs final reward) | 4 GPUs, days | strong |
| ReBeL, Student of Games | poker, etc. | search over public belief states | — | guarantees only for 2-player zero-sum |
| Mahjax (2026) | Riichi | JAX environment, 1–2 M steps/s on 8 A100s | — | infrastructure |

**Hong Kong rules: no published research found.** The public HK projects are faan
calculators, score trackers and hobby games with heuristic bots (hk-mahjong,
Faan Club, ClemL/mahjong, peteteaa/mahjong-game). The closest studied variant is
**MCR**: like HK it has a faan minimum (8 there, 3 here), so a ready hand is not
necessarily a legal win, and the strategy is a choice among target patterns.

## 2. Lessons from the literature

1. **Strong programs start from large human datasets.** Suphx, NAGA and Mortal
   learn from millions of Tenhou positions; most top IJCAI agents are supervised.
   There is no HK dataset. The one well-documented from-scratch success (RVR)
   used 8 GPUs for 3 days. We have an M1 for a few hours.
2. **Random opponents in simulation make the searcher over-aggressive.**
   Mizukami & Tsuruoka cite an earlier Monte Carlo player whose opponents were
   random: it "often steals a tile and tries to become waiting, since opponent
   players do not win in the simulation." Their fix was learned opponent models:
   each simulated opponent has a *waiting* flag (from a waiting predictor) and a
   *folding* flag, and wins on tiles drawn from a winning-tile predictor. This is
   exactly the failure found in our solver on 2026-10-18 (dragons thrown freely in
   rollouts, ready hands broken to chase them).
3. **Use hidden information in training, not in play.** Suphx feeds it to the
   actor and drops it out gradually (oracle guiding). PerfectDou and RVR feed it
   only to the **critic**, which is simpler and needs no schedule. RVR's ablation
   found the oracle critic (with a four-seat, zero-sum value output) was its
   most effective component. VLOG (ICLR 2022) is a variational version for
   value-based RL.
4. **Rewards are noisy; reduce the variance.** Suphx predicts the final game
   reward from round scores (matters for multi-round matches). RVR replaces the
   raw final reward by an *expected* reward predicted from the full state one
   step before the end. The first IJCAI winner clipped scores. Duplicate
   evaluation is standard (IJCAI: 24 games per deal, all seat permutations;
   Mizukami: duplicate "match" mode).
5. **Features and architecture.** Everyone encodes tiles as planes over the 34
   kinds (or a 4×9 image) and uses CNNs without pooling (Suphx), ResNets (Mortal,
   RVR, IJCAI teams) or transformers (Tjong). Suphx adds **look-ahead features**:
   a depth-first search over winning hands giving win probability and score per
   discard. That is close to our faan-aware shanten / acceptance / target-faan
   planes. The 2023 IJCAI winner reported that adding shanten features helped.
   Suit-permutation **data augmentation** is common.
6. **Multi-target hands are hard for RL.** IJCAI organisers: agents trained by RL
   from scratch "never choose some winning patterns", and even SL-pretrained
   models lose them during RL. HK's faan minimum makes this central for us.
7. **Validation accuracy does not measure strength.** Only games do (IJCAI 2023
   discussion). We already evaluate by duplicate games.
8. **Search at play time helps but is expensive.** Suphx's run-time adaptation
   won 66% against the plain policy, but needed 100 K simulations per round and
   was never deployed. Equilibrium search (ReBeL, Student of Games) covers only
   two players; ACH (LuckyJ) is the one regret-based method used at scale in
   mahjong.

## 3. How we compare

| Component | This project | State of the art | Gap |
|---|---|---|---|
| Rules engine | Rust, ~µs per decision, differential-tested against a Python oracle | Rust (Mortal), JAX (Mahjax) | none |
| Hand analysis | exact shanten, faan-aware shanten, acceptance, solo expectimax | Suphx look-ahead features | comparable |
| Danger model | tables fitted from 1.6 M bot positions, calibrated (0.9% vs 0.8%) | Mizukami: logistic regression on expert logs, waiting AUC 0.777, plus a score model | ours lacks discard-history features (order, from hand vs just drawn) and a score model |
| Determinization | best-of-K by shanten, honours weighted ×1.6 | Mizukami: no hands at all, abstract waiting/folding opponents; Suphx pMCPA: random | ours ignores what opponents' discards reveal |
| Rollout policy | rule-based `fast` bot, opponents fold like `defensive` | learned opponent models (Mizukami); learned policy (Suphx pMCPA) | calibration still off late (win 27% vs 22%, deal-in 10% vs 15%) |
| Decision rule | paired differences vs a heuristic baseline, z = 2 | (not found elsewhere) | ours is a strength |
| Imitation | distil the `value` bot | millions of human positions | teacher is weak; the search is stronger than its own teacher |
| RL algorithm | DMC, then PPO with oracle dropout on the actor | PPO + oracle critic (RVR, PerfectDou); DQN + CQL (Mortal); ACH (LuckyJ) | no oracle critic, no four-seat value |
| Network | MLP 933 → 512 → 512 → 256 | CNN / ResNet / transformer over tile planes | MLP ignores tile-axis structure |
| Data augmentation | none | suit permutations | free 6× data |
| Reward | final points / 32 | clipped, predicted, expected-reward | raw high variance (8–384 points) |
| Compute | M1, a few hours | 8 GPUs × days to hundreds of cores | 2–3 orders of magnitude |
| Evaluation | duplicate, 4 rotations, 95% CIs | duplicate with all 24 seat orders | fine for 1-vs-3; mixed fields need permutations |

## 4. What to change, ranked

1. **Learn a belief model for opponents and use it in search** (lesson 2).
   Self-play logs know every hidden hand, so we can train, from public
   information only: P(opponent ready with a legal win), the probability that
   each tile wins for them, the expected payment, and the expected count of each
   hidden tile kind in their hand. Use it three ways: (a) as the danger model,
   replacing the fitted tables; (b) to sample determinizations, drawing
   opponents' tiles from the predicted counts instead of best-of-K plus an honour
   weight; (c) in rollouts, Mizukami-style. Add the public features the
   literature finds informative: discard order, and whether each discard came
   from the hand or was the tile just drawn. This targets the solver's remaining
   miscalibration and the bad advice found in play.
2. **Expert iteration: distil the solver, not the heuristic.** PIMC with the
   paired z-test beats the `value` bot by about 3 points per hand, so imitate
   PIMC decisions (several hours of labelling at 0.4 s each, in parallel).
   Then use the network inside PIMC as the candidate generator and rollout
   policy (`netpimc` exists), and repeat. This is the cheapest way to get
   stronger with little compute, and it replaces the human-data stage the strong
   programs rely on.
3. **Oracle critic with a four-seat zero-sum value head** (lesson 3; RVR,
   PerfectDou). In `ppo.py`, give the value head the oracle features always and
   the policy head never. Predict all four seats' points, constrained to sum to
   zero. This is a small code change with the best-documented gain.
4. **A convolutional network over the 34-tile axis** (lesson 5). For example, 1-D
   convolutions over each suit's nine tiles plus a dense honour branch, a few
   residual blocks, within M1 limits. Add **suit-permutation augmentation** (the
   three suits are interchangeable under HK rules; the dragons are too).
5. **Keep the patterns during RL** (lesson 6). Add a KL penalty toward the
   distilled teacher that decays slowly. Report win frequency per faan pattern
   (flushes, all pungs, dragons) at every evaluation, so loss of a pattern is
   visible.
6. **Variance reduction for the reward** (lesson 4). Train an expected-final-
   points network on the full state one step before the end, and use it as the
   reward. Or at least compare clipped and log-scaled points against raw points.
7. **Evaluation hygiene.** Use all 24 seat permutations when more than two
   different agents share a table, and track the IJCAI-style secondary metrics
   (win rate, deal-in rate, average faan) next to points.

Not recommended for now: CFR-style equilibrium methods (ACH needs LuckyJ-scale
compute and its guarantees are for two players), and transformers (data-hungry
for our budget).

## Sources

* Suphx: Li et al., *Suphx: Mastering Mahjong with Deep Reinforcement Learning*, 2020 — https://arxiv.org/abs/2003.13590
* Mizukami & Tsuruoka, *Building a Computer Mahjong Player Based on Monte Carlo Simulation and Opponent Models*, CIG 2015 — https://www.logos.ic.i.u-tokyo.ac.jp/~tsuruoka/papers/cig2015mizukami.pdf
* Kurita & Hoki, *Method for Constructing AI Player with Abstraction to MDPs in Multiplayer Mahjong*, IEEE ToG 2021 — https://arxiv.org/abs/1904.07491
* Lu & Li, *Mahjong AI Competition: Exploring AI Application in Complex Real-World Games*, IJCAI 2024 — https://ijcai.org/Proceedings/2024/1020.pdf
* Li et al., *Speedup Training AI for Mahjong via Reward Variance Reduction*, CoG 2022 — https://ieee-cog.org/2022/assets/papers/paper_103.pdf
* Fu et al., *Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game* (ACH), ICLR 2022 — https://mlanthology.org/iclr/2022/fu2022iclr-actorcritic
* LuckyJ on Tenhou — https://technode.com/2023/07/12/tencents-mahjong-ai-sets-new-gaming-record-on-international-mahjong-platform/
* Han et al., *Variational Oracle Guiding for Reinforcement Learning*, ICLR 2022 — https://iclr.cc/virtual/2022/poster/6904
* Yang et al., *PerfectDou: Dominating DouDizhu with Perfect Information Distillation*, NeurIPS 2022 — https://arxiv.org/abs/2203.16406
* Li et al., *Tjong: A Transformer-based Mahjong AI*, CAAI TIT 2024 — https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12298
* Mortal — https://github.com/Equim-chan/Mortal ; model notes: https://huggingface.co/hhim8826/mortal3-b40c192
* NAGA — https://dmv.nico/en/articles/mahjong_ai_naga/
* Brown et al., *ReBeL*, NeurIPS 2020 — https://arxiv.org/abs/2007.13544
* Mahjax, 2026 — https://arxiv.org/abs/2605.20577
* HK hobby projects: https://github.com/HSKPeter/hk-mahjong-hand-calculation , https://github.com/leejustin/faan-club , https://github.com/ClemL/mahjong
