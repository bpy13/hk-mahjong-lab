# What makes Hong Kong mahjong hard for a bot, and ideas of our own

**Date:** 2026-10-28. Companion to [survey.md](survey.md). For each problem: what the rules
do, what is known (ours and published), and at least three ideas of our own, each marked
**(our idea)** with an expected cost and a test. Costs are estimates for one RTX 4060 Ti,
an i7-12700K and this repo's Rust engine; "dev-days" means days of building. Ideas that
serve teaching (reasons, grading) are in §8 and marked ⚑ elsewhere.

**Terms.** *Shanten*: how many tile exchanges a hand is from ready. *Faan-aware shanten*:
the same distance, but to a hand that also scores the 3-faan minimum (solver.md §1).
*Deal-in*: discarding the tile an opponent wins on. *PIMC*: search that re-deals the hidden
tiles many times and plays each candidate out. *Oracle critic*: a value network that sees
hidden tiles during training only. *Determinization*: one sampled fill-in of the hidden
tiles.

## 0. Why HK is not riichi with different scoring

Five rule facts drive everything below (rules §4–§8):

1. A complete hand is a legal win only at **≥ 3 faan without flowers**. Most patterns worth
   3+ faan need claimed melds (dragon pungs, half flush, all pungs), so claiming is worth
   about 7 points a hand and "speed only" loses 11 points a hand (Phase 1 §1–2).
2. **The discarder alone pays a discard win; everyone pays a self-draw.** So a deal-in is a
   private loss and a self-draw is a shared loss. Payment grows half-geometrically from 8
   points (3 faan) to 128 (10 faan).
3. **12-tile liability:** feed someone's fourth exposed meld and you pay three shares if
   they win, by any route.
4. **No furiten, no riichi declaration.** An opponent's readiness is never announced; only
   exposed melds, discards and wall progress tell.
5. **Four players, one hand as the unit, zero-sum in points.** There is no two-player
   equilibrium guarantee to lean on.

## 1. The faan minimum and multi-target hands

**The problem.** A hand must be planned toward a scoring target from the first discards.
Several targets compete (half flush vs all pungs vs dragons vs 平糊 + something), and the
right one changes with every draw and claim. Pure tile efficiency is worthless here.

**What is known.**
- Ours: faan-aware shanten with explicit targets (tiles allowed × meld shapes × scoring
  pungs) and guaranteed faan; `value` beats `efficiency` by 10.3 points a hand under
  min_faan 3 and by 0.1 under min_faan 0–1 (Phase 1 §2). Imitation at 77 % agreement loses
  5.6 points a hand: a few wrong discards make a hand undeclarable (Phase 3).
- Published: MCR has the same structure at 8 fan. The IJCAI organisers report that RL from
  scratch "never choose[s] some winning patterns" and SL-pretrained agents lose patterns
  during RL ([IJCAI 2024](https://ijcai.org/Proceedings/2024/1020.pdf)). Tjong adds "fan
  backward", crediting the actions that built the winning hand
  ([Tjong 2024](https://digitalcommons.njit.edu/fac_pubs/267), abstract-only) and splits
  decisions into action type then tile. Suphx's 100+ look-ahead features compute win
  probability and score per discard by depth-first search
  ([Suphx](https://ar5iv.labs.arxiv.org/html/2003.13590)).

**Ideas.**
- **1a. Target-conditioned policy with a plan head (our idea).** ⚑ The network outputs
  a distribution over targets (the same target space as the faan-aware measure) and a
  discard conditioned on the sampled target. Train the plan head by imitation of PIMC's
  best target, then by RL. The chosen target is itself a reason key
  (`reason.aiming_half_flush`). Cost: 3 dev-days, 1 GPU-day. Test: pattern-coverage audit
  (survey §8 bar 5) and points vs `value`; the plan head's accuracy against the pattern the
  hand finally won with.
- **1b. Minimum-faan curriculum (our idea).** The rules engine already has the
  `min_faan` variant. Train with min_faan 0 for the first fifth of RL, then 1, then 3.
  Early play gets dense wins (26 % win rate at 0 faan) so the policy learns tile shapes
  before it must learn targets. Cost: 0.5 dev-days, no extra compute. Test: a run with the
  curriculum vs one without, same wall-clock; measure final points and draw rate.
- **1c. Per-pattern KL anchor (our idea).** Keep the distilled teacher during RL and add a
  KL penalty only on decisions where the teacher's target differs from the student's, with
  a weight per pattern that rises when that pattern's win share falls below the solver's.
  This is related-work item 5 made pattern-specific. Cost: 1 dev-day. Test: win share per
  pattern over training; must stay within ±30 % of PIMC's.
- **1d. Faan-aware potential shaping (our idea).** Reward each step with the change in a
  potential Φ = −(faan-aware shanten) scaled by the stage of the hand. Potential-based
  shaping does not change the optimal policy, so it is safe. Cost: 0.5 dev-days. Test:
  DMC re-run (which collapsed to 92 % draws) with and without shaping; PPO sample
  efficiency at equal wall-clock.
- **1e. Target value table (our idea).** ⚑ Precompute, from self-play, P(declare a legal win
  | target, faan-aware shanten, tiles left, exposed melds) and E[points | win]. Feed the table
  as features and show it in hints ("half flush: 31 % to finish, 24 points when it does").
  Cost: 1 dev-day, hours of CPU. Test: calibration plot of the table against outcomes.

## 2. Four players and no equilibrium guarantee

**The problem.** With three opponents, a "correct" strategy does not exist in the
two-player sense (holdem multiway.md §1). A self-play policy can be strong against itself
and weak against a different style. Opponents' interests also differ by seat: the player
who feeds a cheap win to end the hand may be protecting against someone else's big hand.

**What is known.**
- Published: Pluribus accepted the lack of guarantee and measured empirically; CFR-family
  methods give coarse correlated equilibria at best for n players (multiway.md §1). ACH's
  guarantee is two-player ([ACH](https://mlanthology.org/iclr/2022/fu2022iclr-actorcritic)).
  RVR's oracle critic predicts all four seats' values under a zero-sum constraint and was the
  most effective component ([RVR](https://ieee-cog.org/2022/assets/papers/paper_103.pdf)).
  Cross-play between independent seeds is the standard check that a strategy does not depend
  on everyone agreeing (multiway.md §5).
- Ours: evaluations so far are 1-vs-3 against `value`; mixed fields need 24 seatings
  (related-work item 7).

**Ideas.**
- **2a. Exploiter ladder as a training signal (our idea).** Train a best-response seat
  against three frozen copies for 1 GPU-day; its gain is the robustness number (survey §8
  bar 3). Then add the exploiter to the opponent pool and continue; repeat. This is a
  cheap league. Cost: 1 dev-day; 1 GPU-day per cycle. Test: the exploiter's gain must fall
  cycle by cycle; the main agent's points vs `value` must not fall.
- **2b. Who-pays value head (our idea).** ⚑ Replace the scalar value with four payout
  heads: my gain and each opponent's expected payment to me or from me. The policy can
  then prefer a discard that risks a cheap deal-in to the player across over a tile that
  risks a self-draw by the dealer (everyone pays). Hints can show "if this deals in, the
  likely loss is 8 points to West". Cost: 1 dev-day. Test: ablation vs scalar value at
  equal compute; calibration of each head.
- **2c. Style pool for training and for bot levels (our idea).** Train three
  reward-reweighted agents (aggressive: win points × 1.5; defensive: deal-in cost × 2;
  concealed: 門前清 bonus) and a `value`-like heuristic, and sample opponents from the pool.
  MPPO showed that demonstrations from different-style bots preserve those styles
  ([MPPO 2025](https://arxiv.org/html/2506.16995v1)). The pool doubles as the easy/medium
  bot levels the app needs. Cost: 2 dev-days, 3 GPU-days. Test: cross-play matrix (each
  agent vs each pool) with 24 seatings; the final agent must be ≥ 0 against every column.
- **2d. Seed cross-play as acceptance (known, applied).** Train two seeds; seat them
  together; if results depend on the seed mix, the policy is population-specific
  (multiway.md §5.4). Cost: one extra training run.

## 3. Deal-in liability and the payment structure

**The problem.** Defence in HK is about *who* wins on *which* tile and *how much*. A
discard win costs the discarder alone; the 12-tile rule triples the cost; a self-draw
costs everyone. A danger number per tile is not enough; the bot needs expected loss per
tile and the shape of the three opponents' hands.

**What is known.**
- Ours: a danger model from public information, calibrated (0.9 % predicted vs 0.8 %
  observed per discard); `defensive` cuts deal-ins from 17.9 % to 14.8 % for −1.9 ± 1.5
  points (Phase 1 §1, solver.md §3). It lacks discard order and a score model
  (related-work §3).
- Published: Mizukami's three models (waiting, winning tiles, score) are exactly the
  decomposition we lack
  ([Mizukami 2015](https://www.logos.ic.i.u-tokyo.ac.jp/~tsuruoka/papers/cig2015mizukami.pdf)).
  Suphx and Mortal report deal-in rates of about 10 % in riichi; HK rates are higher because
  nothing is announced. Riichi safe-tile logic (furiten, genbutsu) does not exist here.

**Ideas.**
- **3a. Payment-aware danger (our idea).** ⚑ A two-head model per opponent from public
  information: P(waiting and the tile wins) and E[points | they win], the latter predicted
  from exposed melds and discards (a visible dragon pung or one-suit melds raise it). Danger
  becomes expected loss in points. Trained from self-play logs, which know every hidden
  hand. Cost: 1 dev-day, hours of CPU. Test: calibration by bucket on held-out games;
  `defensive` re-run with expected-loss folding vs probability folding.
- **3b. Liability lookahead (our idea).** ⚑ A feature and a reason key for "this tile
  would complete a fourth exposed meld for seat X" and for "seat X already has three melds
  and I hold their likely fourth". The rule is rare but costs three shares; a drill skill
  ("don't feed the twelfth tile") falls out directly. Cost: 0.5 dev-days. Test: count
  liability events per 1,000 hands before and after; drill grading on hand-written
  liability positions.
- **3c. Safe-tile budget as a fold planner (our idea).** ⚑ When folding, count the tiles in
  hand that are safe against each threat for the next k turns (the danger model gives the
  per-tile numbers; the count is the plan). Fold only if the budget covers the expected
  remaining turns; otherwise push. Shown in hints as "3 safe tiles for 5 turns: fold" or "1
  safe tile: push". Cost: 1 dev-day. Test: points of `defensive` with and without the
  budget rule.
- **3d. Discard-order features and from-hand vs just-drawn (known, from related-work).**
  Add them to the danger model. Cost: 0.5 dev-days. Test: calibration improves.

## 4. Hidden information and belief

**The problem.** Three hidden hands and the wall; opponents' discards are the only clue to
their shapes. Uniform re-dealing makes opponents look harmless (Phase 1 §4.1); our
best-of-K fix matches shanten statistics but ignores what opponents discarded. PIMC also
suffers strategy fusion (it plays each sample as if everyone could see it).

**What is known.**
- Ours: best-of-K determinization with K = 4 + used²/40; anchoring to `value` with a paired
  z-test; rollouts by `fast` (5.4 points worse than `value`), which under-plays every seat.
  Network-guided PIMC did not beat the bare network (Phase 3 §2).
- Published: Mizukami replaces hands with abstract waiting/folding opponents; Suphx's pMCPA
  samples the rest of the round with the policy itself; MeowCaTS applies importance
  weights in MCTS backpropagation and merges "solitary tile" branches
  ([MeowCaTS 2025](https://scholar.nycu.edu.tw/en/publications/applying-importance-sampling-to-mcts-for-mahjong/),
  abstract-only); Ataraxos trains a belief network and samples hidden states for search
  (holdem survey §6); ReBeL's public-belief-state framing (multiway.md §4.2).

**Ideas.**
- **4a. Learned per-tile count posterior for sampling (known, related-work item 1).** Train
  a network from self-play logs to predict each opponent's hidden counts per tile kind;
  sample determinizations from it. Cost: 2 dev-days, 1 GPU-day.
- **4b. Discard-history importance weights (our idea).** Keep best-of-K sampling, but
  weight each determinization by how likely the rollout policy would have produced the
  opponent's actual discards from the sampled hand (product of policy probabilities over
  the last few discards). Candidates are scored by weighted means. This brings MeowCaTS's
  idea into PIMC without a new sampler. Cost: 1 dev-day. Test: calibration of rollout win
  and deal-in rates vs real play (now 27 % vs 22 % and 10 % vs 15 %, solver.md); points of
  PIMC-64 vs `value`.
- **4c. Belief-summary leaf evaluator (our idea).** Train the value net on the public
  state plus the predicted count posterior (4a) rather than on sampled hands, then use it as
  the leaf value after a short rollout instead of playing to the end. This cuts strategy
  fusion (the leaf is evaluated under uncertainty) and rollout cost. Cost: 3 dev-days,
  2 GPU-days. Test: PIMC with leaf net vs full rollouts at equal time; points and speed.
- **4d. Rollout policy = distilled PPO net (our idea).** Replace `fast` in rollouts with a
  quantised small network distilled from the PPO or the final agent. Phase 3 found network
  rollouts dealt in less than `fast` rollouts (23.8 % vs 27.5 %). Cost: 1 dev-day. Test:
  rollout calibration and PIMC points.
- **4e. Reveal-at-the-end explanation (our idea).** ⚑ In review, show the actual hidden
  hands next to the bot's belief at the decision ("the bot thought West was 2 tiles from
  ready; West was ready on 5p"). Cost: 0.5 dev-days in the review pipeline. Test: users
  understand the folds; no compute.

## 5. Reward variance and null outcomes

**The problem.** A hand's payout ranges from −384 to +384 (three shares of 128) and most
hands end with the learner neither winning nor paying; 6 % of `value` self-play hands and
92 % of DMC self-play hands are draws. Deep Monte Carlo collapsed: zero labels shrank all
Q-values to zero (Phase 3 §3).

**What is known.**
- Published: Suphx predicts the final game reward and uses its change per round; RVR
  replaces the final reward by an expected reward predicted one step before the end, and its
  oracle critic cut value variance from 6.36 to 4.44; the 2020 IJCAI winner clipped scores
  ([RVR](https://ieee-cog.org/2022/assets/papers/paper_103.pdf),
  [IJCAI 2024](https://ijcai.org/Proceedings/2024/1020.pdf)). AIVAT reduces evaluation
  variance by 85 % in poker (multiway.md §5.3).
- Ours: PPO's value baseline survived; DMC did not.

**Ideas.**
- **5a. Duplicate baseline as the training reward (our idea).** Every self-play wall is
  also played by `value` in the same seat with the same hidden draws (common random
  numbers). The reward is the paired difference. This is AIVAT-style variance reduction
  moved from evaluation into training; our tournament code already does the pairing. Cost:
  1 dev-day; doubles simulation cost (cheap in Rust). Test: advantage variance and
  points-vs-`value` after equal wall-clock, against raw reward.
- **5b. Oracle expected-payout critic (known, RVR; applied).** Value head sees hidden
  tiles always, policy never; four outputs that sum to zero. Cost: 0.5 dev-days.
- **5c. Teacher seats to stop draw collapse (our idea).** In self-play, one or two seats
  are always the frozen teacher (`value` or the last good checkpoint), so wins keep
  happening and labels stay informative. Rotate which seats learn. Cost: 0.5 dev-days.
  Test: DMC re-run; draw rate must stay under 15 %.
- **5d. Log-scaled payouts with a sign-preserving map (our idea).** Train on
  sign(x) · log(1 + |x| / 8) and report in points. Keeps the 128-point limit hand from
  dominating gradients without clipping it away. Cost: hours. Test: PPO run vs raw and
  clipped rewards.

## 6. No human data

**The problem.** Every strong published agent begins with millions of human decisions.
There are no Hong Kong logs, and riichi logs teach wrong habits (no minimum, riichi,
furiten). Imitating our own heuristic caps at the heuristic.

**What is known.**
- Published: MPPO lifted PPO above the best Botzone bot with demonstrations from a
  weaker bot ([MPPO 2025](https://arxiv.org/html/2506.16995v1)); Mahjax cloned a rule-based
  agent then ran PPO with a KL penalty ([Mahjax 2026](https://arxiv.org/html/2605.20577v1));
  Xu distilled Mortal into a 4.9 M network at parity on one 16 GB GPU
  ([Xu 2026](https://arxiv.org/html/2610.34791)); behaviour cloning of bot logs tied or lost
  to the reference bot (IJCAI-2026 logs card).
- Ours: distillation of `value` reached 77 % agreement and −5.6 points; PPO closed the gap
  to 0 (Phase 3).

**Ideas.**
- **6a. Expert iteration from PIMC (known, related-work item 2).** Label with PIMC
  decisions, distil, put the net inside PIMC, repeat. Cost: hours of CPU per round.
- **6b. Riichi-shape pretraining (our idea).** Tile-shape knowledge (which 34-tile
  arrangements become melds fast) is rule-independent. Pretrain the suit-convolution
  encoder on a generated riichi or min_faan-0 dataset (our engine with min_faan 0 produces
  dense wins), then fine-tune the heads on HK rules. Cost: 1 dev-day, 1 GPU-day. Test:
  sample efficiency of HK PPO from the pretrained encoder vs from scratch at equal
  wall-clock.
- **6c. Brian's games as a calibration set, not a training set (our idea).** ⚑ Record
  Brian's and volunteers' hands in the app (D1 already plans review). Use them only to
  measure agreement (survey §8 bar 4) and to pick drill positions where humans and the bot
  disagree. Cost: app-side logging already planned; 0.5 dev-days for the agreement script.
- **6d. Synthetic "human-like" opponents from style rewards (our idea; shared with 2c).**
  The reweighted agents stand in for a human population during training and as the
  weaker bot levels, until real player data exists.

## 7. The phone budget

**The problem.** One WebAssembly thread, 1–2 s per hint, a review that must respect
`budget_ms`, no file I/O, bounded memory (engine-interface §1; bot-stream §4). PIMC today
needs 8 threads for 0.2 s at 32 samples; a full review takes 30–60 s.

**What is known.**
- Survey §9: a Mortal-size net is 30–120 ms on a phone (estimate); our MLP under 1 ms; a
  5–15 M parameter net is 5–15 MB in int8. Botzone imposes 1 s per move and the top MCR bots
  are pure networks ([Botzone wiki](https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong)).
  Suphx's pMCPA (100 K simulations) was never deployed for time reasons.

**Ideas.**
- **7a. Anytime PIMC with sequential stopping (our idea).** Run determinizations in
  batches of 8; stop as soon as the best candidate leads the anchor by more than two
  standard errors, or the budget ends. Most decisions are clear early; hard ones get the
  whole budget. Cost: 0.5 dev-days. Test: mean samples per decision and points vs fixed-64
  PIMC on a single thread.
- **7b. Suit-shared quantised 1-D CNN (our idea).** One convolution tower shared across the
  three suits (they are symmetric under HK rules), a dense honour branch, int8 weights.
  About a third of the parameters of three separate towers and the same accuracy, with
  suit-swap augmentation for free. Cost: 2 dev-days. Test: agreement with the fp32 model;
  latency in a WebAssembly build on a mid-range Android phone.
- **7c. Review during idle time (our idea).** ⚑ Compute each review item right after the
  human's decision, while the bots take their turns, so the post-hand review is already
  done when the hand ends. Uses the Web Worker the app already runs. Cost: 1 dev-day on
  the engine side (an incremental `review` call). Test: review latency at hand end
  under 1 s.
- **7d. Per-suit shanten table (our idea).** Replace the unbounded per-thread cache with a
  precomputed table keyed by a suit's count vector (there are 405,350 count vectors of up to
  14 tiles over 9 kinds with 4 copies, fewer in practice), stored as bytes. The 2024 TAAI
  "deficiency lookup table" paper did this for Taiwanese rules (survey §4.1, unread).
  Cost: 1 dev-day. Test: memory bounded; shanten speed on one thread.
- **7e. Phone/server split by call (known pattern, D13).** `bot_action` and `hint` on the
  phone with the net plus anytime PIMC; a deep review or a drill solve on a server with
  PIMC-512. Decide after measuring 7a on a phone.

## 8. Teaching: reasons and value-lost grading

**The problem.** D1 forbids "a solver that says what, never why". Every hint and review
item needs a reason key (D42), and grades need a loss in points that is honest about
noise. A neural policy gives neither by itself.

**What is known.**
- Published: Mxplainer fits a parametric search to an agent and explains step by step
  ([Mxplainer 2025](https://arxiv.org/abs/2506.14246), abstract-only); NeSy-Mahjong and Xu
  show rationale accuracy jumps (43 % → 93 %) when a program supplies the facts; NAGA's
  product sells "the 80 discards the AI hesitates over" as teaching content; Suphx's
  look-ahead features are themselves explanations.
- Ours: study mode already shows the top 3 moves with expected points, win and deal-in
  rates, tile danger and faan-aware progress with 95 % intervals (README; solver.md §4).

**Ideas.**
- **8a. Reasons from feature deltas (our idea).** For each candidate, compute the
  hand-made features after it (faan-aware shanten, acceptance, target faan, danger,
  liability flag). The reason keys are the features where the best move and the played move
  differ most: `reason.keeps_half_flush`, `reason.loses_two_acceptance`,
  `reason.danger_vs_west`. Deterministic, cheap, and tied to numbers the UI already shows.
  Cost: 2 dev-days. Test: on 200 reviewed positions, a strong player marks each reason as
  "right", "irrelevant" or "wrong"; target ≥ 80 % right.
- **8b. Value-lost grading with uncertainty (our idea).** Loss = paired difference between
  the played move and the best move on the same determinizations, with its standard error.
  Grade thresholds in points (best < 0.5, good < 2, inaccuracy < 5, mistake < 12, blunder
  beyond), scaled down late in the hand when little is at stake, and "unclear" when the
  interval straddles a threshold. Cost: 1 dev-day. Test: grade stability across repeated
  runs with different seeds (same position must get the same grade ≥ 90 % of the time).
- **8c. Counterfactual replay (our idea).** For a mistake, replay the same walls with the
  best move and show the outcome distribution side by side ("you won 12 % and dealt in 18 %;
  the fold wins 9 % and deals in 4 %"). The determinizations already exist. Cost: 1 dev-day.
- **8d. Drill mining and difficulty rating (our idea).** Mine positions from self-play
  where PIMC's best move beats the second by > 2 standard errors and at least one bot level
  picks the wrong move. Difficulty = the share of the bot ladder (random, efficiency, fast,
  value, PPO) that gets it wrong, mapped to the drill Elo scale; skill tag from the dominant
  reason key (8a). Cost: 2 dev-days; hours of CPU per thousand drills. Test: human
  solve rate per rating band must fall as the rating rises.
- **8e. Parametric-search explainer for the net (our idea, after Mxplainer).** Fit a small
  weighted-feature search (weights on shanten, acceptance, target faan, danger, liability)
  to the final agent's choices; use its weights to explain the net and to sanity-check it
  ("the bot weights danger 2.3× acceptance in the last third of the wall"). Cost:
  2 dev-days. Test: top-3 agreement with the net ≥ 85 %.

## 9. Which ideas to test first

Ranked by expected gain per cost, for the strategy proposal to weigh:

1. **5a duplicate-baseline reward** + **5b oracle critic**: variance is the known blocker
   for RL here; both are a day's work.
2. **3a payment-aware danger** and **4b discard-history weights**: fix the solver's two
   measured miscalibrations without a new sampler.
3. **1a target-conditioned policy** + **1c per-pattern anchor**: the HK-specific failure
   mode, and the source of the best reason keys.
4. **2a exploiter ladder**: the only robustness number we can have with four players.
5. **8a/8b reasons and graded loss**: what the app ships; cheap because the features exist.
6. **7a anytime PIMC** + **7b suit-shared CNN**: the phone path.
7. **6b riichi-shape pretraining** and **4c belief-summary leaf**: higher risk, test last.

## Open questions for Brian (numbered, each with a default)

1. **Human benchmark (survey §8 bar 4).** Can you and a few club players commit to about
   400 hands each in the app once bots exist, so we can measure human level? Default: yes,
   you alone first; club players later.
2. **Riichi proxy (bar 6).** Is a riichi port of the training recipe worth 1–3 weeks to get
   an external comparison with Mortal? Default: no for now; revisit if bars 1–5 pass.
3. **Teaching grade scale (8b).** Are the point thresholds (0.5 / 2 / 5 / 12) the right
   feel, given 8 points is the smallest win? Default: use them, tune from your review
   sessions.
4. **Style bots (2c).** Should the easy and medium levels be "weaker versions of the best
   bot" or "different human-like styles"? Default: styles, because weaker-by-noise bots
   teach bad habits.
5. **Matches (rules §10).** Does the app need full four-round matches with dealer
   retention for the first release, which would bring Suphx-style match-reward prediction
   into scope? Default: single hands first.
