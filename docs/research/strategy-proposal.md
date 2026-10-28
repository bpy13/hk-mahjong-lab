# Strategy proposal for the Hong Kong mahjong bot (Task 6)

**Date:** 2026-10-28. Written for the hk-games bot stream from this folder's reports
([survey.md](survey.md), [hard-problems.md](hard-problems.md), [related-work.md](related-work.md)),
the cross-game report
([teaching-and-deployment.md](../../../bot-stream/docs/research/teaching-and-deployment.md)),
this repo's measured results ([phase1-solver.md](../study-results/phase1-solver.md),
[phase3-rl.md](../study-results/phase3-rl.md)) and the hk-games contract
([decisions.md](../../../hk-games/docs/decisions.md),
[engine-interface.md](../../../hk-games/docs/engine-interface.md),
[bot-stream.md](../../../hk-games/docs/streams/bot-stream.md)). It follows the shape of
holdem's `docs/research/strategy-proposal.md`.

**How to read it.** Claims about published work link to the report section that cites the
paper. **(estimate)** marks a number I scaled from a published figure or from this repo's
measurements; nothing in this file has been run. **(judgment)** marks a choice of mine that
Brian can overturn. **(our idea)** marks an idea of ours; the report that first proposed it
is linked. Brian decides the questions in §10. Training details (sizes, learning rates,
schedules) are left to the build plan.

**Terms.** *PIMC*: a search that re-deals the hidden tiles many times and plays each
candidate move to the end of the hand. *Determinization*: one such re-deal. *Faan-aware
shanten*: how many tile exchanges a hand is from a hand that can legally win (3 faan).
*Expert iteration*: a loop where a search labels positions, a network copies the labels,
and the network then speeds up the search. *PPO*: a self-play learning method that moves
the policy in small, clipped steps. *Oracle critic*: a value network that sees the hidden
tiles during training only. *Duplicate*: every wall is played from every seat so the luck
of the deal cancels. *CI*: a 95 % confidence interval.

## 1. Summary (one screen)

**What we would build.** One brain in three parts. (1) A **search**: today's PIMC, given a
learned belief about the hidden tiles and a learned leaf value, so it stops guessing that
opponents are harmless. (2) A **network** of about 2–5 M parameters, trained first by
copying the search (expert iteration), then improved by PPO self-play with a four-seat
oracle critic, a duplicate-baseline reward and a target head that knows which faan pattern
the hand aims at. (3) The same network **inside** a small anytime search on the phone, which
gives the `expert` bot and every hint. All reasons come from computed feature differences
(faan-aware shanten, acceptance, target, danger, liability), never from the network and
never as text (D42).

**Why this order.** Every strong published mahjong program starts from millions of human
decisions ([survey §Summary](survey.md#summary)). Hong Kong rules have none. The one thing in
this repo that already beats the best heuristic is the search (+1.7 ± 3.2 points per hand,
[phase1 §4.2](../study-results/phase1-solver.md#42-fixes)), and the only learning method
that worked was PPO from a copied start ([phase3 §2](../study-results/phase3-rl.md#2-results)).
Published small-budget successes did the same: MPPO lifted PPO above the best Botzone bot
with demonstrations from a *weaker* bot, and Xu distilled a strong teacher into a 4.9 M
network at parity on one 16 GB GPU ([survey §2](survey.md#2-mcr-chinese-official-rules-and-the-botzone-competitions),
[survey §3](survey.md#3-llms-and-mahjong-20252026)). So: make the search a better teacher,
copy it, let self-play go past it, and put the result back inside the search.

**What the app gets first.** Phase A ships the four engine calls (`bot_action`, `hint`,
`review`, `grade`) in WebAssembly with reason keys, built from the bots and search we have
today. No training is needed for that. Strength work starts after the app has something to
call.

**Cost.** About 8 GPU-days in total across phases A–D **(estimate)**, in runs of at most
2 days; about 10 CPU-days of labelling and evaluation in the background; about 12 weeks of
calendar time. Phase E (the optional "state of the art" claim) adds 4.5–7.5 GPU-days.

**Measured by.** The six bars of [survey §8](survey.md#8-what-state-of-the-art-for-hk-mahjong-can-mean-and-six-measurable-bars):
an internal ladder with intervals, ported published recipes at equal compute, a
best-response exploiter, duplicate play against Brian and club players, pattern coverage,
and an optional riichi proxy against Mortal. §6 fixes the match format and sample sizes.

## 2. The recommended method: "search teaches the network, the network speeds the search"

### 2.1 The pieces

| Piece | What it is | Source |
|---|---|---|
| Belief model | A network trained on self-play logs (which know every hidden hand) that predicts, from public information only, each opponent's hidden tile counts, whether they can win, on which tiles, and for how many points | [related-work §4 item 1](related-work.md#4-what-to-change-ranked); [hard-problems 3a, 4a](hard-problems.md#3-deal-in-liability-and-the-payment-structure) |
| Search | PIMC with determinizations drawn from the belief model, weighted by how likely each sampled hand produced the opponent's real discards, rollouts by the network, and a leaf value instead of playing to the end | [hard-problems 4b, 4c, 4d](hard-problems.md#4-hidden-information-and-belief) |
| Network | A suit-shared 1-D convolution tower over the three suits plus a dense honour branch, int8 on the phone; heads: policy, target (which faan pattern), four-seat value | [hard-problems 7b, 1a, 2b](hard-problems.md#7-the-phone-budget) |
| Training | Expert iteration from the search; then PPO self-play with an oracle four-seat zero-sum critic, a duplicate-baseline reward, a minimum-faan curriculum and a per-pattern anchor to the teacher | [hard-problems §5, §1, §9](hard-problems.md#9-which-ideas-to-test-first); [survey §6](survey.md#6-what-transfers-and-what-does-not) |
| Phone | The int8 network alone for `easy`–`hard`; the network inside an anytime PIMC with sequential stopping for `expert`, `hint` and `grade`; `review` computed during the bots' turns | [hard-problems 7a, 7c](hard-problems.md#7-the-phone-budget); [teaching §5.2](../../../bot-stream/docs/research/teaching-and-deployment.md#52-recommendation) |
| Reasons and grades | Feature deltas between the best move and the runner-up as keys; paired loss with an interval and σ-scaled thresholds; counterfactual replay in review | [hard-problems §8](hard-problems.md#8-teaching-reasons-and-value-lost-grading); [teaching §1–2](../../../bot-stream/docs/research/teaching-and-deployment.md#1-reasons-from-solver-output-to-keys) |

### 2.2 Why each piece is there

- **The belief model fixes the search's measured flaw.** Uniform re-dealing made the
  solver ignore danger; best-of-K sampling fixed the shanten statistics but still ignores
  what opponents discarded ([phase1 §4.1](../study-results/phase1-solver.md#41-naive-pimc)).
  Mizukami's opponent models were the published fix for the same failure
  ([survey §1.1](survey.md#11-before-deep-learning-simulation-with-opponent-models)).
- **The search is the teacher because it is the strongest thing we have.** Copying `value`
  at 77 % agreement cost 5.6 points per hand; a few wrong discards make a hand undeclarable
  ([phase3 §2](../study-results/phase3-rl.md#2-results)). Copying a stronger teacher
  with more data is the Xu result: 1.71 M teacher decisions gave parity, 100 K did not.
- **PPO goes past the teacher.** Phase 3's PPO closed a 5.6-point gap in 90 minutes on a
  laptop. The published additions we lack are the oracle critic with four zero-sum outputs
  (RVR's most effective component) and variance reduction of the reward
  ([related-work §2 lessons 3–4](related-work.md#2-lessons-from-the-literature)).
- **The target head guards the patterns.** RL from scratch "never chooses some winning
  patterns" and SL-pretrained agents lose them during RL (the IJCAI organisers,
  [survey §2](survey.md#2-mcr-chinese-official-rules-and-the-botzone-competitions)). Our
  DMC run collapsed to 92 % draws for a related reason
  ([phase3 §3](../study-results/phase3-rl.md#3-why-deep-monte-carlo-collapsed)). The
  target head and the per-pattern anchor make the pattern an explicit, monitored choice.
- **The network goes back inside the search.** Phase 3's network-guided PIMC did not beat
  the bare network because rollouts used the weak `fast` bot
  ([phase3 §2](../study-results/phase3-rl.md#2-results)). With network rollouts and a leaf
  value, the search should add strength again; Phase C measures this.
- **Reasons from features, not from the network.** Rationale accuracy rose from 43 % to
  93 % when a program supplied checkable facts (Xu, [survey §3](survey.md#3-llms-and-mahjong-20252026)).
  The features already exist in `crates/search/src/features.rs`.

### 2.3 Our innovations in the recommended path (our idea, each)

Each was proposed in hard-problems.md or the teaching report; the link says where.

1. **Duplicate-baseline reward for training** ([5a](hard-problems.md#5-reward-variance-and-null-outcomes)):
   the reward is the paired difference from `value` playing the same wall in the same seat.
2. **Target-conditioned policy with a plan head** ([1a](hard-problems.md#1-the-faan-minimum-and-multi-target-hands)):
   the chosen pattern is both a policy input and the reason key `reason.plan.target`.
3. **Per-pattern KL anchor** ([1c](hard-problems.md#1-the-faan-minimum-and-multi-target-hands)):
   the penalty toward the teacher rises for any pattern whose win share falls.
4. **Minimum-faan curriculum** ([1b](hard-problems.md#1-the-faan-minimum-and-multi-target-hands)):
   train at min_faan 0, then 1, then 3, so tile shapes are learned before targets.
5. **Who-pays value head** ([2b](hard-problems.md#2-four-players-and-no-equilibrium-guarantee)):
   four payout outputs, so a cheap deal-in to one seat and a self-draw by the dealer
   (everyone pays) are told apart.
6. **Payment-aware danger** ([3a](hard-problems.md#3-deal-in-liability-and-the-payment-structure)):
   danger as expected loss in points, not a probability.
7. **Discard-history importance weights in PIMC** ([4b](hard-problems.md#4-hidden-information-and-belief)).
8. **Belief-summary leaf evaluator** ([4c](hard-problems.md#4-hidden-information-and-belief)):
   the leaf value is computed under uncertainty, which cuts strategy fusion.
9. **Anytime PIMC with sequential stopping** ([7a](hard-problems.md#7-the-phone-budget)) and
   **review during idle time** ([7c](hard-problems.md#7-the-phone-budget)).
10. **Suit-shared quantised 1-D CNN** ([7b](hard-problems.md#7-the-phone-budget)).
11. **Exploiter ladder as a training signal** ([2a](hard-problems.md#2-four-players-and-no-equilibrium-guarantee)).
12. **Reasons from feature deltas**, **value-lost grading with an "unclear" grade**,
    **counterfactual replay** and **drills by ladder disagreement**
    ([8a–8d](hard-problems.md#8-teaching-reasons-and-value-lost-grading)), plus the
    cross-game **luck ledger** and **reason truth checked by the ladder**
    ([teaching §2.4, §1.5](../../../bot-stream/docs/research/teaching-and-deployment.md#24-luck-versus-skill-the-luck-ledger-our-idea)).

Not in the path, kept in reserve: riichi-shape pretraining (6b), the safe-tile budget (3c),
the parametric-search explainer (8e, an audit once per release), and the style pool as a
training signal (2c, used only for bot levels in Phase D).

## 3. Alternatives considered, and why they lose

| Alternative | What it would be | Why not as the main line | Kept as |
|---|---|---|---|
| **A. Scale up Phase 3's PPO** from the distilled `value` net, no search teacher | The cheapest path: more compute on what already works | The start point is 5.6 points below `value`; PPO only reached `value` level (−0.1 ± 2.9) and never beat the search; without an oracle critic and variance reduction its reward is 8–384 points of noise ([phase3 §2](../study-results/phase3-rl.md#2-results), [related-work §3](related-work.md#3-how-we-compare)); no pattern guard, so the IJCAI collapse is a live risk | The PPO stage of Phase C, with the additions |
| **B. Supervised learning from logs** (the Suphx, NAGA, Mortal, IJCAI recipe) | Copy millions of human or bot decisions, then RL or nothing | No Hong Kong logs exist; riichi logs teach wrong habits (no minimum, riichi, furiten); the IJCAI 2026 logs are MCR bot logs, and imitation of a bot "tied or lost" to that bot ([survey §Summary, §2](survey.md#summary)) | Brian's and volunteers' games as a calibration set, never a training set ([6c](hard-problems.md#6-no-human-data)) |
| **C. Search only, no learning** (the Taiwanese Computer Olympiad tradition, Mizukami) | PIMC with a hand-fitted belief model and more samples | Strength plateaued: 32, 64 and 128 samples all score about +2 with ±5 intervals, "more samples barely help" ([phase1 §4.2](../study-results/phase1-solver.md#42-fixes)); one phone thread affords far fewer samples than the desktop (§7); the app needs cheap `easy`–`hard` levels and millisecond hints that only a network gives | The search stays as teacher and as the `expert` layer; MeowCaTS's importance weights are adopted ([4b](hard-problems.md#4-hidden-information-and-belief)) |
| **D. Offline RL on our own self-play logs** (Mortal-style DQN with CQL, or DMC) | Value learning from stored hands | DMC collapsed to a never-win policy with or without a warm start ([phase3 §3](../study-results/phase3-rl.md#3-why-deep-monte-carlo-collapsed)); MPPO found CQL, BPPO, GAIL and SAIL "performed poorly" on MCR ([survey §2](survey.md#2-mcr-chinese-official-rules-and-the-botzone-competitions)); offline RL caps near the policy that wrote the logs | Teacher seats to stop draw collapse ([5c](hard-problems.md#5-reward-variance-and-null-outcomes)) if PPO ever stalls |
| **E. Neural regret minimisation** (ACH, LuckyJ) | Actor-critic on counterfactual regret with Hedge | Convergence is for two players; the four-player system has no paper, no code and undisclosed compute; the 10.68-dan claim is press only ([survey §1.7, §10](survey.md#17-luckyj-tencent-ai-lab-20222023-and-ach)) | Nothing |
| **F. Transformer policy** (Tjong 15 M, Xu 4.9 M) instead of the CNN | A different network shape | Not a different strategy; Tjong needed 0.5 M SL samples and 7 GPU-days of SL; the CNN gives suit symmetry for free and a third of the parameters ([7b](hard-problems.md#7-the-phone-budget)); related-work.md called transformers data-hungry for our budget | One ablation in Phase C if the CNN plateaus (Q7) |

**(judgment)** The recommended path is what A and C become when each borrows the other's
strength: the search supplies the data that A lacks, and the network supplies the speed that
C lacks.

## 4. Conflicts between the reports, and what I chose

| Topic | The reports say | Chosen here | Why |
|---|---|---|---|
| Phone PIMC samples per hint | The bot-stream brief estimates 100–200 samples in 2 s on a phone ([bot-stream §4](../../../hk-games/docs/streams/bot-stream.md)); survey §9's arithmetic gives 32 samples in 1.6 s on one *desktop* thread, so about 15–30 samples in 2 s on a phone ([survey §9](survey.md#9-fit-for-our-hardware-and-for-a-phone)) | The survey's arithmetic as the planning number; the Phase A benchmark on a real phone replaces both | The survey's number rests on a measured 0.2 s at 8 threads; the brief's is unmeasured. The gap is why Phase B adds the leaf value, network rollouts and a cheaper sampler before the phone search can reach 100 samples |
| Network size on the phone | Survey §9 accepts 5–15 M parameters; the teaching report sets ≤ 5 M and ≤ 5 ms per forward pass ([teaching §5.2](../../../bot-stream/docs/research/teaching-and-deployment.md#52-recommendation)) | ≤ 5 M, target 2–5 M | The search runs on top of the network, and the budget is one thread |
| Oracle information | rl.md uses Suphx-style oracle guiding on the actor; related-work item 3 and hard-problems 5b use RVR's oracle critic | Oracle critic only; drop actor guiding | Phase 3 measured no gain from guiding at twice the compute ([phase3 §2](../study-results/phase3-rl.md#2-results)); RVR's ablation shows the critic's gain |
| What to do first | related-work ranks the belief model first; hard-problems §9 ranks variance reduction first; the bot-stream brief asks for WebAssembly readiness and reason keys first | App first (Phase A), then belief and teacher (B), then RL (C) | The app has nothing to call until Phase A ships; B and C both need the harness and keys A builds |
| Bot levels | hard-problems Q4 proposes style agents; the teaching report proposes checkpoints plus a named error model, with the style pool for `medium` ([teaching §4.2](../../../bot-stream/docs/research/teaching-and-deployment.md#42-stage-a-no-human-data-recommendation-for-the-first-release)) | The teaching table: `easy` = early checkpoint + error model, `medium` = style pool, `hard` = final network, `expert` = network + search | Every injected mistake is a reason key; a noise bot cannot name its mistakes |
| Grade thresholds | hard-problems 8b: 0.5 / 2 / 5 / 12 points; teaching §2.2: fractions of the hand-score spread σ | The σ rule, which gives the same four numbers at σ ≈ 20 | One rule for five games; same feel |
| Bar 1 target | Survey bar 1 asks for +5 points per hand over `value`; today's best is +1.7 ± 3.2 | +3 at the end of Phase B, +5 at the end of Phase C | A step a phase can fail honestly |
| Bar 2 cost | Survey proposes three ports at 3 GPU-days each (9 GPU-days) | Three ports at 1.5 GPU-days each, optional Phase E (Q8) | The GPU serves five games |

## 5. Phases with acceptance criteria

Phases are serial. Each phase ends with a 10-line status and a commit; the bars below are my
defaults **(judgment)**; Brian sets them.

### Phase A – The app's four calls, in WebAssembly, with reason keys (about 3 weeks, 0.1 GPU-day)

What the app needs first ([bot-stream §4](../../../hk-games/docs/streams/bot-stream.md)).
No new strength; everything is built from today's bots, search and features.

Delivers:
1. WebAssembly readiness: `web-time` for `Instant`, no file I/O, a single-thread path behind
   the `threads` setting, models from bytes, the per-suit shanten table with a memory cap
   ([7d](hard-problems.md#7-the-phone-budget)). The engine stops returning Chinese labels;
   every name becomes a key (D42).
2. `bot_action` at five levels from the existing ladder as placeholders: `random`;
   `easy` = `fast` plus the error model; `medium` = `defensive`; `hard` = `value`;
   `expert` = anytime PIMC anchored to `value` ([7a](hard-problems.md#7-the-phone-budget)).
   Phase D replaces `easy`–`hard`.
3. `hint`: anytime PIMC, top 3 moves with `score`, `win_rate`, `numbers` (deal-in rate,
   faan-aware shanten, target faan) and up to three reasons from feature deltas
   ([8a](hard-problems.md#8-teaching-reasons-and-value-lost-grading)) under the shared
   taxonomy ([teaching §1.2](../../../bot-stream/docs/research/teaching-and-deployment.md#12-shared-taxonomy-recommendation)).
   First keys: `reason.plan.target`, `reason.tempo.shanten`, `reason.tempo.acceptance`,
   `reason.risk.deal_in`, `reason.risk.liability`, `reason.belief.likely_waiting`,
   `reason.value.gap`, `reason.counterfactual.outcomes`.
4. `review`: one item per decision, computed during the bots' turns and chunked to
   `budget_ms` ([7c](hard-problems.md#7-the-phone-budget)); paired loss with its interval,
   σ-scaled grades, "unclear" when the interval straddles a line, stake scaling late in the
   hand ([teaching §2.2](../../../bot-stream/docs/research/teaching-and-deployment.md#22-recommendation-sigma-scaled-thresholds-paired-loss-an-unclear-grade)).
5. `grade`: as `hint` on a drill position. The ten hand-written drills (D6) are checked
   against it.
6. The evaluation harness: duplicate play over 4 seatings (1-vs-3) and all 24 seatings
   (mixed fields), paired differences, bootstrap intervals, win, deal-in, draw and
   per-pattern win shares in one report format.
7. A phone benchmark harness run on the named phone (Q5): p99 time per call, forward-pass
   time, peak memory, a 10-minute thermal run.
8. `bot-notes.md` entries: the keys, "mahjong `hint` live from commit X", and the two
   interface requests (`quality`, `interval`).

Done when (all):
- (a) Every call returns within `budget_ms` × 1.2 at p99 on the named phone, with
  `bot_action` for `easy`–`hard` under 30 ms and `hint` at 2,000 ms.
- (b) A test proves no engine output contains display text; a test proves two positions
  that differ only in hidden tiles give identical `hint` output.
- (c) The five levels order strictly in a 1,600-hand duplicate round-robin with intervals
  excluding 0.
- (d) Brian rates the reasons on 50 hint positions: ≥ 80 % "right", ≤ 10 % "wrong".
- (e) Grade repeatability: the same position gets the same grade on ≥ 90 % of re-runs
  with new seeds.
- (f) The phone benchmark reports how many PIMC samples fit 2 s; this number replaces the
  estimates in §7.

### Phase B – A stronger teacher: belief, calibrated search, first network (about 3 weeks, 0.5 GPU-day, ~4 CPU-days)

Delivers:
1. Self-play logs with hidden hands (2 M positions, hours of CPU); the belief model
   ([4a](hard-problems.md#4-hidden-information-and-belief)) and the payment-aware danger
   model ([3a](hard-problems.md#3-deal-in-liability-and-the-payment-structure)) with
   discard-order features ([3d](hard-problems.md#3-deal-in-liability-and-the-payment-structure)).
2. PIMC with belief-drawn determinizations weighted by discard history
   ([4b](hard-problems.md#4-hidden-information-and-belief)), the liability flag
   ([3b](hard-problems.md#3-deal-in-liability-and-the-payment-structure)), and the
   danger model inside `defensive` and the rollouts.
3. Expert iteration round 1: label about 1 M decisions with PIMC-64 (2–3 CPU-days in the
   background, [related-work §4 item 2](related-work.md#4-what-to-change-ranked)); train
   the suit-shared CNN ([7b](hard-problems.md#7-the-phone-budget)) with suit-swap
   augmentation; export int8; put it inside PIMC as candidate generator, rollout policy
   ([4d](hard-problems.md#4-hidden-information-and-belief)) and leaf value
   ([4c](hard-problems.md#4-hidden-information-and-belief)). Round 2 if round 1 moved the
   search.
4. The phone search re-benchmarked with the network rollouts and the leaf value.

Done when (all):
- (a) Rollout calibration: win rate and deal-in rate of the rollouts within 3 points of
  real play (today 27 % vs 22 % and 10 % vs 15 %, [solver.md §4](../solver.md#4-pimc-search-cratessearchsrcpimcrs)).
- (b) Belief model: per-tile count log-likelihood beats best-of-K sampling on held-out
  games; danger calibration within error bars in every bucket.
- (c) PIMC-64 scores ≥ +3 points per hand against three `value` with a CI excluding 0
  over 4,000 hands; its deal-in rate is at or below `value`'s 17.9 %.
- (d) The distilled network scores ≥ 0 against `value` with a CI half-width ≤ 2 (Phase 3's
  copy scored −5.6).
- (e) The network inside PIMC beats the bare network with a CI excluding 0 (Phase 3's
  result reversed).
- (f) int8 top-1 agreement with the fp32 network ≥ 98 %; ≥ 48 determinizations in
  1.5 s on the named phone with the leaf value **(estimate; Q5 phone)**.

### Phase C – Past the teacher: PPO self-play (about 3 weeks, 3.5 GPU-days)

Delivers:
1. `train-core` PPO with: oracle four-seat zero-sum critic ([5b](hard-problems.md#5-reward-variance-and-null-outcomes));
   duplicate-baseline reward ([5a](hard-problems.md#5-reward-variance-and-null-outcomes));
   log-scaled payouts ([5d](hard-problems.md#5-reward-variance-and-null-outcomes));
   minimum-faan curriculum ([1b](hard-problems.md#1-the-faan-minimum-and-multi-target-hands));
   target head ([1a](hard-problems.md#1-the-faan-minimum-and-multi-target-hands)) and
   per-pattern anchor ([1c](hard-problems.md#1-the-faan-minimum-and-multi-target-hands));
   who-pays value head ([2b](hard-problems.md#2-four-players-and-no-equilibrium-guarantee));
   per-pattern win shares logged at every evaluation.
2. One main run of 2 GPU-days from the Phase B network, with a throughput measurement in
   the first hour (§7) and a status note at 12 h. One second seed of 1 GPU-day for
   cross-play. Ablations of 5a and 1a at 0.25 GPU-day each.
3. Expert iteration round 3: the PPO network back inside the search; a fresh distillation
   if the search moved again.

Done when (all):
- (a) Bar 1: ≥ +5 points per hand against three `value` with a CI excluding 0 over 4,000
  hands, and beats PIMC-128 (Phase B version) head-to-head with a CI excluding 0 over
  24 seatings.
- (b) Bar 5: win share per faan pattern within ±30 % of the Phase B search's shares; draw
  rate 5–10 % in self-play.
- (c) Cross-play: two seeds seated together score within their intervals of each other
  ([2d](hard-problems.md#2-four-players-and-no-equilibrium-guarantee)).
- (d) The ablations report the gain of 5a and 1a with intervals; a piece that gains nothing
  is removed from the build plan.
- (e) The reason audit of Phase A is repeated on 50 positions with the new network as
  `expert`: ≥ 80 % right.

### Phase D – Robust, human-like, and teachable (about 3 weeks, 3.5 GPU-days)

Delivers:
1. Exploiter ladder ([2a](hard-problems.md#2-four-players-and-no-equilibrium-guarantee)):
   a best-response seat for 1 GPU-day against three frozen copies; add it to the pool;
   repeat once (2 GPU-days).
2. Bot levels per [teaching §4.2](../../../bot-stream/docs/research/teaching-and-deployment.md#42-stage-a-no-human-data-recommendation-for-the-first-release):
   `easy` = an early PPO checkpoint plus the named error model (first list: discards the
   danger tile when a safe one exists, breaks a scoring pung for speed, chases a hand that
   cannot reach 3 faan, feeds a fourth exposed meld); `medium` = the style pool, three
   agents fine-tuned from the main network at 0.5 GPU-day each
   ([2c](hard-problems.md#2-four-players-and-no-equilibrium-guarantee)); `hard` = the
   final network; `expert` = the network inside the anytime search.
3. Drill miner and seed ratings ([8d](hard-problems.md#8-teaching-reasons-and-value-lost-grading),
   [teaching §3](../../../bot-stream/docs/research/teaching-and-deployment.md#32-selecting-instructive-positions-recommendation));
   counterfactual replay in review ([8c](hard-problems.md#8-teaching-reasons-and-value-lost-grading));
   the luck ledger; reveal-at-the-end belief in review ([4e](hard-problems.md#4-hidden-information-and-belief)).
4. The human benchmark (bar 4) with Brian's 400 hands in the app.

Done when (all):
- (a) Bar 3: the second exploiter gains < 3 points per hand; the main agent's score
  against `value` does not fall between cycles.
- (b) Ladder order with gaps ≥ 1.5 points and intervals excluding 0 over 1,600 hands per
  pair; the `easy` bot's injected mistakes appear in review as keys.
- (c) Bar 4: the bot's paired margin on Brian's walls is ≥ 0 with CI ±4 or better; top-1
  agreement with Brian's choices ≥ 75 % on 200 labelled positions.
- (d) 1,000 mined drills with seed ratings; solve rate per rating band falls as the band
  rises over the first attempts (Spearman, sign test); tag precision ≥ 80 % on 50 drills
  audited by Brian.
- (e) The luck ledger's four terms sum to the outcome on every reviewed hand.

### Phase E – Claim "state of the art for HK" (optional, 2–3 weeks, 4.5–7.5 GPU-days)

Bar 2: port three published recipes to HK and train each for 1.5 GPU-days (RVR-style PPO
with oracle critic from scratch; Mortal-style ResNet with offline RL on our logs;
MPPO-style PPO with `value`/PIMC demonstration actors). Done when our agent beats each by a
CI-separated margin over 24 seatings. Bar 6 (riichi proxy against Mortal, 1–3 weeks and
3–6 GPU-days) only if Brian says so (Q9).

## 6. How strength is measured

| Measure | Format | Sample size and interval | Applies to | Source |
|---|---|---|---|---|
| Points per hand vs three `value` | duplicate, 4 seatings per wall | 1,600 hands → ±1.5 for bots; 4,000 hands → ±1.3 for search agents (240 hands gave ±5.1; the half-width falls as 1/√n) **(estimate)** | every agent | [phase1 §1](../study-results/phase1-solver.md#1-the-bot-ladder) |
| Head-to-head in mixed fields | duplicate, all 24 seatings per wall | 4,000 hands; paired by wall; bootstrap CI | bar 1 second half, bar 2, ladder gaps | [survey §8](survey.md#8-what-state-of-the-art-for-hk-mahjong-can-mean-and-six-measurable-bars) |
| Secondary numbers | logged with every match | win rate, deal-in rate, draw rate, mean faan of wins, win share per faan pattern | all | [related-work §4 item 7](related-work.md#4-what-to-change-ranked) |
| Best-response gap | one exploiter seat vs three frozen copies, 1 GPU-day, then 4,000 duplicate hands | gain in points per hand with CI; bar < 3 | bar 3 | [survey §8](survey.md#8-what-state-of-the-art-for-hk-mahjong-can-mean-and-six-measurable-bars), [2a](hard-problems.md#2-four-players-and-no-equilibrium-guarantee) |
| Cross-play of seeds | two seeds seated 2 + 2 and 1 + 3, 24 seatings | 4,000 hands | Phase C | [2d](hard-problems.md#2-four-players-and-no-equilibrium-guarantee) |
| Human benchmark | Brian plays ≥ 400 hands vs three bots; the bot then plays the same walls in the same seat vs the same bots | paired margin with CI (±4 expected at 400 hands **(estimate)**); agreement on 200 labelled positions | bar 4 | [survey §8](survey.md#8-what-state-of-the-art-for-hk-mahjong-can-mean-and-six-measurable-bars) |
| Pattern coverage | per-pattern win shares vs the search's | ±30 %; draws 5–10 % | bar 5, every evaluation | [survey §8](survey.md#8-what-state-of-the-art-for-hk-mahjong-can-mean-and-six-measurable-bars) |
| Ported recipes at equal compute | three ports, 1.5 GPU-days each | 24 seatings, 4,000 hands | bar 2 (Phase E) | [survey §8](survey.md#8-what-state-of-the-art-for-hk-mahjong-can-mean-and-six-measurable-bars) |
| Riichi proxy | our recipe on riichi rules vs three Mortal community checkpoints | 1,000 hanchan; average rank ≤ 2.55 | bar 6 (optional) | [survey §7](survey.md#7-open-agents-that-could-serve-as-benchmark-opponents) |
| Reason and grade quality | human audit; repeatability; ladder check | 50 positions per release; ≥ 90 % same grade | every phase | [teaching §1.5](../../../bot-stream/docs/research/teaching-and-deployment.md#15-how-to-check-a-reason-is-true) |

Cost of evaluation: a 4,000-hand match where one seat runs PIMC-128 takes about 8 hours on
20 threads **(estimate: 0.8 s per decision at 8 threads, about 20 search decisions per
hand)**; network-only matches take minutes. There is no external HK opponent, so every bar
except 6 is internal ([survey §7](survey.md#7-open-agents-that-could-serve-as-benchmark-opponents)).
We do not claim "superhuman" or "beats Suphx"; the survey explains why.

## 7. Compute on the local box

Hardware: RTX 4060 Ti 16 GB, i7-12700K (20 threads), 32 GB RAM. One run ≤ 3 days on the GPU;
runs over 12 h are announced; the GPU is shared with four other games, so every GPU figure
is in GPU-days. All numbers are **(estimate)**; the basis is in the last column; Phase A's
benchmark and Phase C's first hour replace them.

| Step | Time | GPU-days | RAM / VRAM | Basis |
|---|---|---|---|---|
| A: WebAssembly port, keys, harness | 3 weeks of building | 0 | — | code exists; the work is porting and keys |
| A: phone benchmark | hours | 0 | — | one named phone |
| B: self-play logs, 2 M positions | 1–2 h of CPU | 0 | < 4 GB | `fast` at 1.7 ms per hand for four seats ([solver.md §2](../solver.md#2-bots-cratessearchsrcbotsrs)) |
| B: belief and danger models | 2–4 GPU-hours | 0.15 | < 2 GB VRAM | small nets on 2 M rows |
| B: PIMC-64 labels, 1 M decisions | 2–3 days of CPU, background | 0 | < 8 GB | 0.4 s per decision at 8 threads ([related-work §4 item 2](related-work.md#4-what-to-change-ranked)) → ~0.2 s at 20 threads |
| B: distillation, 2–5 M CNN | 1–2 GPU-hours | 0.1 | < 4 GB VRAM | Xu: 1.71 M decisions in 28 min on a 16 GB GPU ([survey §3](survey.md#3-llms-and-mahjong-20252026)) |
| B: evaluations (bars 1, 5) | ~8 h of CPU each | 0 | — | §6 |
| C: PPO throughput | 300 K–1 M hands per hour | — | 8–12 GB VRAM, < 16 GB RAM | Phase 3: 316 K hands in 90 min on an M1 with a Python-bound MLP; the Rust environment and batched GPU inference should give 10–30×; Mortal reports 320 K hands/h on a 4090 with an 11 M ResNet ([survey §1.4](survey.md#14-mortal-equim-2021-open-source)) |
| C: main PPO run | 2 days → 15–50 M hands | 2 | as above | Suphx's RL stage used 1.5 M games (~12 M hands) per agent ([survey §1.2](survey.md#12-suphx-microsoft-research-asia-2020)); RVR 3 days on 8 GPUs; our run is 50–100× Phase 3's hands |
| C: second seed + two ablations | 1 + 0.5 days | 1.5 | — | shorter runs from the same start |
| D: exploiter, 2 cycles | 1 day each | 2 | — | [2a](hard-problems.md#2-four-players-and-no-equilibrium-guarantee) |
| D: style pool, 3 agents | 0.5 day each, fine-tuned from the main net | 1.5 | — | [2c](hard-problems.md#2-four-players-and-no-equilibrium-guarantee) at a third of its cost by fine-tuning |
| D: drill mining, 1,000 drills | hours of CPU | 0 | — | [8d](hard-problems.md#8-teaching-reasons-and-value-lost-grading) |
| E: three ports | 1.5 days each | 4.5 | — | survey bar 2, halved |
| E: riichi proxy | 1–3 weeks + 3–6 GPU-days | 3–6 | — | survey bar 6 |

Totals: Phases A–D about **8 GPU-days** in runs of ≤ 2 days, about 10 CPU-days in the
background, nothing over 16 GB of RAM. Phase E adds 4.5–7.5 GPU-days (Q8, Q9). The 3-day
cap is never reached; the 12-hour announcement applies to the Phase C main run, the second
seed, the exploiter cycles and the Phase E ports.

## 8. What runs on the phone and what on a server

Rule: the phone is the default (D12); a server tier is added only after a measured
shortfall, written in `bot-notes.md`
([teaching §5.3](../../../bot-stream/docs/research/teaching-and-deployment.md#53-what-runs-where-per-game)).
Default at first release: phone only (Q6).

| Call | Level or use | On the phone (one WebAssembly thread) | Budget (default) | Server (optional, later) |
|---|---|---|---|---|
| `bot_action` | `random` | random legal move | < 1 ms | — |
| `bot_action` | `easy`, `medium`, `hard` | int8 network forward pass plus the error model or style agent | < 30 ms | — |
| `bot_action` | `expert` | network inside anytime PIMC: batches of 8 determinizations, stop when the leader is two standard errors ahead or the budget ends | 1,500 ms | — |
| `hint` | all | the same search, top 3 moves, numbers and up to three reasons; `quality` = samples used | 2,000 ms | PIMC-512 deep hint if the phone grader's bias measures too high |
| `review` | all | one item per decision, computed during the bots' turns; N = 32 determinizations first, doubled while the interval straddles a grade line; chunked to `budget_ms` | 500 ms per chunk; whole hand under 1 s at hand end | PIMC-512 re-grade of a sample to measure the phone grader's bias ([teaching §2.3](../../../bot-stream/docs/research/teaching-and-deployment.md#23-variance-how-many-samples-a-grade-needs)) |
| `grade` | drills | as `hint` on the drill position | 2,000 ms | — |
| drill mining, training, evaluation | — | never | — | Brian's PC, offline; not a server |

Memory on the phone: the network ≤ 5 MB in int8; the per-suit shanten table a few MB; the
belief model ≤ 2 MB; peak under 64 MB **(estimate)**, measured in Phase A. A hint's
sampling noise is reported, so the app can show "unclear" rather than a confident wrong
answer when the budget was short.

## 9. Risks, and what would make us change course

| Risk | Signal | Change of course |
|---|---|---|
| The phone search is far slower than estimated (survey §9 vs the brief, §4) | Phase A benchmark: < 16 determinizations in 2 s | `expert` and `hint` become network-only with the belief gate (8 worlds; answer at once if the policy agrees), and the search runs only for `review` chunks; a server hint tier is proposed in `bot-notes.md` |
| The belief model does not move the search | Phase B (c) fails: PIMC-64 stays near +2 | Keep best-of-K sampling with the discard weights (4b) and the leaf value; spend the saved weeks on Phase C |
| Expert iteration caps at the search | Phase B (d) passes but Phase C (a) fails | Phase C is the remedy by design; if PPO also stalls, add teacher seats (5c) and the exploiter pool early |
| Pattern collapse during PPO (the IJCAI warning; our DMC) | per-pattern win shares drift > 30 %; draws rise above 10 % | Raise the per-pattern anchor; shorten the curriculum's min_faan 0 stage; fall back to the Phase B network inside the search |
| PPO throughput is 10× below the estimate | the first-hour measurement in Phase C | Smaller network for training, fewer determinizations in the duplicate baseline, or accept 5 M hands per run and report the shortfall |
| The network's choices disagree with the feature-delta reasons | reason audit < 80 % right in Phase C (e) | Run the parametric-search explainer (8e) to find which feature the network weights differently; add that feature as a key or fix the feature |
| The grader rewards greed (biased rollouts) | the server re-grade disagrees with the phone grade on > 10 % of decisions | Strongest network in rollouts; more determinizations in review; a server review tier |
| No human level can be measured | Brian cannot play 400 hands; club players unavailable | Report bars 1, 3 and 5 only; label the app's "expert" as "best bot", not "human level" |
| The 32 GB RAM or 16 GB VRAM ceiling | any step over 24 GB RAM or 14 GB VRAM | Stream labels from disk; halve the batch; smaller network |
| GPU contention with the other four games | a Phase C run cannot get 2 days | Split into two 1-day runs with a checkpoint; the plan file names the week |
| Abstract-only sources carry a number we rely on (Tjong, MeowCaTS, Mxplainer) | listed in [survey §10](survey.md#10-risks-and-where-the-claims-are-weak) | No phase bar depends on them; 4b and 8e are tested on our own numbers |

## 10. Questions for Brian

Each stands on its own: what it means, why it matters, and a default you can accept with
"ok".

1. **The overall path.** Accept "search teaches the network, the network speeds the search"
   (§2) with alternatives A–F (§3) held in reserve? This fixes the order of the build plan.
   *Default: yes.*

2. **App first.** Phase A ships the four calls from today's bots and search, with
   placeholder levels, before any strength work. The app can wire mahjong three weeks
   earlier; the placeholder `easy`–`hard` bots are rule-based until Phase D. *Default: yes,
   Phase A first.*

3. **Phase bars.** Phase B asks the search for +3 points per hand over `value`; Phase C asks
   the network for +5 and a win over the search. Today's best is +1.7 ± 3.2. A bar a phase
   can fail is the point. *Default: +3 and +5 as written.*

4. **Grade thresholds.** Best / good / inaccuracy / mistake lines at 0.5 / 2 / 5 / 12 points
   (0.025 / 0.1 / 0.25 / 0.6 of the hand-score spread), "unclear" when the interval straddles
   a line. The smallest win pays 8 points, so 0.5 is a fine line. *Default: use them; tune
   after your first 20 reviewed hands.*

5. **Benchmark phone.** Every phone number here is an estimate; Phase A measures on one
   named mid-range Android phone. Which one? *Default: your phone, plus one mid-range model
   you name.*

6. **Server tier.** Phone only at first release, with a `quality` number in every hint and
   review so the app can route later. A PIMC-512 server tier for deep review comes only
   after a measured shortfall. *Default: phone only.*

7. **Network shape.** A suit-shared 1-D CNN of 2–5 M parameters, int8 on the phone. A
   transformer (Tjong, Xu) is the published alternative. *Default: the CNN; one transformer
   ablation in Phase C only if the CNN plateaus.*

8. **Phase E ports (bar 2).** Porting three published recipes to HK at 1.5 GPU-days each
   (4.5 GPU-days) is the only way to say "a Suphx or RVR team with our compute would not
   have done better". *Default: not before Phases A–D pass; decide then.*

9. **Riichi proxy (bar 6).** Running our recipe on riichi rules against Mortal gives the
   only external comparison. It costs 1–3 weeks and 3–6 GPU-days. *Default: no.*

10. **Human benchmark (bar 4).** You play 400 hands against three bots in the app; the bot
    replays your walls; you also label 200 positions. About 20 hours of your time. *Default:
    yes, you alone first; club players later.*

11. **Bot levels.** `easy` = an early checkpoint plus a named error model; `medium` = three
    style agents (aggressive, defensive, concealed); `hard` = the final network; `expert` =
    the network inside search. The alternative is "weaker by noise", which teaches bad
    habits. *Default: as written.*

12. **Single hands first.** Full four-round matches with dealer retention ([rules §10](../rules/hk.md#10-matches-later))
    would bring Suphx-style match-reward prediction into scope. *Default: single hands for
    the first release.*

13. **Shared crates.** The harness, the teaching crate (reasons, grades, drills, luck
    ledger), the int8 inference crate and `train-core` are shared with the other games in a
    new private repo, pinned per game ([teaching §6](../../../bot-stream/docs/research/teaching-and-deployment.md#6-cross-game-reuse)).
    *Default: yes; mahjong's Phase A builds the first versions.*

14. **Logging human games.** With a consent toggle, the app's games calibrate the bot
    levels, seed drills and later train a skill-conditioned human-like policy. *Default:
    log with consent; calibration and drills first, training later.*
