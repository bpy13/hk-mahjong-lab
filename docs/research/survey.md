# Published mahjong AI, and what "state of the art" can mean for Hong Kong rules

**Date:** 2026-10-28. Written for the hk-games bot stream (Task 1 of the research plan).
This file builds on [related-work.md](related-work.md) (October 2026), which already
summarised Suphx, Mortal, LuckyJ, the IJCAI competitions, RVR, Tjong and PerfectDou and
ranked seven changes for this repo. This survey does not repeat that ranking. It adds the
2024–2026 work, the open agents and their licences, the fit for our hardware and a phone, and
the question "what bar must an HK bot clear to call itself state of the art?".

**How to read the citations.** Every claim about a published method links to its source.
Where I could read only the abstract, a model card or press, the entry says
*(abstract-only)* or *(press)*. Numbers are the authors' own, in their own setups. Nothing
was re-run here. Our own ideas are marked **(our idea)**.

**Terms used.** *Riichi* is the Japanese variant (no faan minimum; a "yaku" is needed).
*MCR* is Chinese Official rules (8-fan minimum). *Tenhou* and *Mahjong Soul* are online riichi
platforms; a *stable rank* on Tenhou is a rating estimated from placements over many games.
*Botzone* is Peking University's bot platform that hosts the IJCAI MCR competition.
*SL* means supervised learning from game logs. *RL* means reinforcement learning by self-play.
*Duplicate* play means the same walls are dealt to every seating so luck cancels.

## Summary

1. **Every strong mahjong program is a neural policy trained first on human logs.** Suphx,
   NAGA, Mortal, LuckyJ (riichi) and the 2022–2023 IJCAI winners (MCR) all start from
   millions of human decisions and then use RL, offline RL or nothing. Hong Kong rules have no
   such logs. This is still the largest gap between us and the published recipes.
2. **Nobody has published a Hong Kong mahjong agent.** Searches in English and Chinese, on
   arXiv, Semantic Scholar, GitHub and Hong Kong university project lists, found only faan
   calculators and hobby games (related-work.md §1). No Sichuan "bloody" agent paper exists
   either. The closest studied variants are MCR (faan minimum 8) and Taiwanese 16-tile
   mahjong (Computer Olympiad programs, search-based).
3. **The strongest open code is Mortal** (riichi, AGPL-3.0, ResNet 40 × 192, offline RL on
   Tenhou logs). The author withholds the weights; community checkpoints of 11.3 M parameters
   exist on Hugging Face under AGPL. We may run it as an opponent for a riichi proxy test and
   read it; we may not link or ship it (holdem D9, carried over).
4. **2024–2026 work is mostly infrastructure, explanation and LLMs, not strength.** Mahjax
   (2026) is a GPU simulator; Tjong (2024) and a 2026 hierarchical transformer improve discard
   accuracy; Mxplainer (2025) and NeSy-Mahjong (2026) explain decisions; three 2025–2026
   papers show that text-prompted LLMs lose to Mortal and to mid-tier Botzone bots. The one
   new strength result for a small budget is Xu (2026): a 4.9 M-parameter transformer
   distilled from Mortal on one 16 GB GPU reached parity with Mortal.
5. **Small-budget recipes that worked:** RVR (8 GPUs × 3 days, PPO with an oracle critic,
   4th among the IJCAI top 16); MPPO (2 × RTX 3090 × 7 days, PPO mixed with demonstrations
   from a weaker bot, beat the best Botzone bot); Xu 2026 (distillation, ~5 GPU-days). All
   three are within 1–3× of our box.
6. **"State of the art for HK" has no external yardstick, so we must build one.** §8
   proposes six measurable bars: an internal ladder with confidence intervals, ported
   versions of the published recipes trained with equal compute, a best-response exploiter,
   duplicate matches against human players, pattern coverage, and an optional riichi proxy
   against Mortal.
7. **Phone fit.** A Mortal-size network is about 0.6 GFLOP per decision; my estimate for
   one WebAssembly thread on a phone is 30–120 ms. Our current MLP is under 1 ms. Our PIMC
   search is the only component that strains the 1–2 s budget.

## 1. Riichi mahjong (Japanese rules)

### 1.1 Before deep learning: simulation with opponent models

- **Mizukami & Tsuruoka 2015** (CIG 2015,
  <https://www.logos.ic.i.u-tokyo.ac.jp/~tsuruoka/papers/cig2015mizukami.pdf>): Monte Carlo
  simulation with three supervised opponent models (is the opponent waiting, which tiles win
  for them, how much they score). Tenhou rating 1718. Their diagnosis of random-opponent
  simulation ("the player often steals a tile and tries to become waiting, since opponent
  players do not win in the simulation") is the failure our Phase 1 solver showed
  (related-work.md lesson 2). Read in full for related-work.md.
- **Gao, Okuya, Kawahara & Tsuruoka 2019** (arXiv:1906.02146,
  <https://ar5iv.arxiv.org/html/1906.02146>): supervised CNNs on 2015 Houou-table logs
  (2.1 M discard situations; 86 planes of 34 × 4; three 5 × 2 conv blocks of 100 filters).
  Discard agreement 70.44 %, pon 88.32 %, chi 90.62 %, riichi 75.85 %. Tenhou rating about
  1850 after 300 matches (an average intermediate player is about 1600). One Tesla K10 GPU.
  This is the cheapest documented "supervised CNN reaches above-average human" result.
- **Kurita & Hoki 2021** (IEEE ToG; arXiv:1904.07491,
  <https://arxiv.org/abs/1904.07491>) *(abstract-only)*: abstract mahjong into several Markov
  decision processes and search those; evaluated against "the strongest AI player at the
  time". Relevant because it is a search method, not a learned policy.

### 1.2 Suphx (Microsoft Research Asia, 2020)

Li et al., *Suphx: Mastering Mahjong with Deep Reinforcement Learning*, arXiv:2003.13590,
read in full at <https://ar5iv.labs.arxiv.org/html/2003.13590>.

- Five CNNs (discard, riichi, chow, pong, kong) with no pooling; input 34 × 838 for discard
  and 34 × 958 for the others; the win decision is a rule. "100+" look-ahead features, each
  a 34-vector, computed by depth-first search that ignores opponents (close to our
  faan-aware shanten and acceptance planes).
- Supervised stage: 15 M discard samples (76.7 % test agreement), 5 M riichi (85.7 %),
  10 M chow (95.0 %), 10 M pong (91.9 %), 4 M kong (94.0 %).
- RL stage: policy gradient with **global reward prediction** (a two-layer GRU predicts the
  final game reward from the round history; the reward for a round is the change in that
  prediction) and **oracle guiding** (the actor sees hidden tiles early; a Bernoulli mask
  removes them on a schedule). Each RL agent: 1.5 M self-play games, about two days on 44
  GPUs (4 Titan XP for the parameter server, 40 Tesla K80 for workers). Offline evaluation: 1 M
  games on 20 K80s for two days.
- Run-time **parametric Monte-Carlo policy adaptation**: 100 K simulated trajectories per
  round; the adapted agent won 66 % of games against itself. Not deployed on Tenhou (time).
- Tenhou: 5,760 games in the expert room, record 10 dan, stable rank 8.74. Comparisons in the
  same table: NAGA 6.64 (9,649 games), Bakuuchi 6.59 (30,516), top human players 7.46
  (8,031). Win rate 22.83 %, deal-in 10.06 %.

### 1.3 NAGA (Dwango Media Village, 2019–)

<https://dmv.nico/en/articles/mahjong_ai_naga/>: four CNNs (discard, call, riichi, kan)
trained by supervised learning on top-table Tenhou logs, a heuristic for final-round wins,
and confidence estimation during training. Reached 8 dan in 2018–2019; Suphx's table puts its
stable rank at 6.64. No RL, no search. It is sold as a paid review service and appears in
a 2025 KADOKAWA book of "80 discards the AI hesitates over" (Watanabe × NAGA), a nice model
for teaching content built from an AI's uncertainty. Closed source.

### 1.4 Mortal (Equim, 2021–; open source)

- Code: <https://github.com/Equim-chan/Mortal>, AGPL-3.0-or-later, last push 2025-09-28,
  1,595 stars. Rust engine (`libriichi`) with Python training; mjai protocol; backend of the
  public reviewer <https://mjai.ekyu.moe>. The documentation site
  (<https://mortal.ekyu.moe>) says "most pages are empty right now"; the Reinforcement
  Learning pages return 404, so architecture facts below come from a community model card.
- Architecture (model card of a community checkpoint,
  <https://huggingface.co/hhim8826/mortal3-b40c192>): ResNet, 40 blocks × 192 channels with
  channel attention, dueling DQN head; 11,310,592 parameters; trained one epoch on 191,132
  Tenhou Houou games (107,200 steps at batch 2048); 43.4 MB in fp32; an ONNX export is
  provided. related-work.md adds the CQL-style offline RL and GRU rank predictor from the
  Mortal source.
- Strength (Mortal's own page, <https://mortal.ekyu.moe/perf/strength.html>): one Mortal
  "3.1" in a seat against three akochan: average rank 2.477 vs 2.568, average rank points
  +1.89 vs −5.67 over 39,456 / 13,152 games. Mortal versions improve by 0.2–1.2 rank points
  per generation in 1 M-game self-matches. No Tenhou or Mahjong Soul statistics are
  published. Independent evidence of strength: human Houou experts agree with Mortal's first
  choice on 81.9 % of decisions (Xu 2026, §1.8).
- Weights: the author withholds them (Gist, August 2022: "currently I have no plan to release
  the trained model", mainly to avoid cheating on Mahjong Soul). Community-trained
  checkpoints are public (the HF card above; Xu 2026 used "VoidShine-298k").
- Speed: "up to 40 K hanchans per hour" of self-play on an RTX 4090 + Ryzen 9 7950X at
  batch 2,000.
- Fit for us: it is riichi, so it cannot play HK. Its value to us is (a) a design reference,
  (b) a benchmark opponent for a riichi proxy test of our *method* (§8, bar 6), and
  (c) a licence warning: AGPL means run as a separate process, never link.

### 1.5 akochan (critter-mj, C++)

<https://github.com/critter-mj/akochan>: "Artificial Intelligence for Japanese mahjong"; last
push 2022-07-05; licence "Other" (a custom Japanese terms-of-use file, not an OSI licence).
The README gives build steps only; commit names mention fold-expectation calculations
(`cal_betaori_exp`). Mortal's page above measures it 7.5 rank points per game weaker than
Mortal. Usable as a second, weaker riichi reference only after reading its terms.

### 1.6 Kanachan (Cryolite)

<https://github.com/Cryolite/kanachan>: a riichi framework aimed at Mahjong Soul rules,
"to train large scale models based on more powerfully expressive framework (e.g.,
transformer)"; behavioural cloning plus curriculum fine-tuning; no crawler, no data and no
trained models are provided; GitHub lists no licence (last push 2025-02-16, 346 stars). No
strength evaluation is published. Not usable as a benchmark.

### 1.7 LuckyJ (Tencent AI Lab, 2022–2023) and ACH

- The published algorithm is **ACH**, Fu et al., *Actor-Critic Policy Optimization in a
  Large-Scale Imperfect-Information Game*, ICLR 2022
  (<https://mlanthology.org/iclr/2022/fu2022iclr-actorcritic>) *(abstract-only; OpenReview
  blocks fetching)*: actor-critic that minimises a weighted cumulative counterfactual regret,
  approximated by a network, with Hedge for the policy; convergence to Nash "under certain
  conditions"; benchmark is **1-on-1 mahjong, a two-player zero-sum game**, where the agent
  "defeats a human champion". No compute figures on the page; no code.
- The four-player Tenhou system has no paper. Press (IT之家, 2023-07-11,
  <https://www.ithome.com/0/705/059.htm>; TechNode,
  <https://technode.com/2023/07/12/tencents-mahjong-ai-sets-new-gaming-record-on-international-mahjong-platform/>)
  *(press)*: stable rank 10.68 dan in the Tenhou 特上 room after "just over 1,000 games", a
  record; it also beat six professional MCR players in an offline invitational; Tencent
  describes "optimistic value estimation" for imperfect-information search on top of ACH.
  Treat the four-player claims as unverified.

### 1.8 Newer riichi work, 2022–2026

- **Meowjong** (Zhao & Holden, CoG 2022; arXiv:2202.12847; CoG text read via
  <https://ieee-cog.org/2022/assets/papers/paper_170.pdf>): three-player riichi. Five CNNs on
  50,000 Houou sanma rounds (2019); discard SL took about 30 hours on 4 × P100; RL for 400
  self-play episodes by Monte Carlo policy gradient. Discard agreement 65.81 %. Evaluated
  only against random bots (1st place 72.38 % vs 21.80 % for the SL agent). Code:
  <https://github.com/victorzxy/meowjong>. Weak evidence, but it is a one-student, four-GPU
  project, so it is a realistic cost reference.
- **Mjx** (Koyamada et al., CoG 2022; <https://github.com/mjx-project/mjx>, MIT): riichi
  simulator, "100× faster than Mjai", Tenhou-validated; ships only a random agent; README
  warns the build is broken and the API will change; no Windows support.
- **Mahjax** (Nishimori, Okano, Habara, Koyamada, Yu, Sugiyama, arXiv:2605.20577, May 2026,
  <https://arxiv.org/html/2605.20577v1>): riichi in JAX; up to 2 M steps/s on eight A100s;
  100 M training steps in 5.8 h on one GH200; transformer encoder with policy and value
  heads; behavioural cloning on 500 K samples from a rule-based agent, then PPO with a KL
  penalty toward the cloned policy. Strength measured only against its own cloned opponents
  (average rank below 2.5). Code at <https://github.com/nissymori/mahjax>; code licence not
  stated on the paper page. A possible riichi environment for a method-proxy test on our GPU.
- **Hierarchical Transformer discard prediction** (Yokoyama, Tahara & Sei, IPSJ TOD 19(1),
  January 2026, <https://ipsj.ixsq.nii.ac.jp/records/2006864>) *(abstract-only)*:
  transformers at several feature scales matched to mahjong units; supervised on Tenhou
  logs; "higher discard prediction accuracy than Suphx" with fewer inputs and parameters. No
  numbers on the page.
- **A Hierarchical and Risk-Aware Decision Framework for Riichi Mahjong AI** (2026,
  ResearchGate listing only, fetch blocked) *(unread)*: a two-level policy (strategy, then
  action) with two-stage self-play guided by expert knowledge. Listed for completeness.
- **LLM papers on riichi** are in §3.

## 2. MCR (Chinese Official rules) and the Botzone competitions

MCR is the variant closest to Hong Kong rules: a hand must reach 8 fan, so a ready hand is
not always a legal win, and play is a choice among target patterns. Our minimum is 3 faan
with a much shorter faan table, but the structural lesson carries.

- **IJCAI Mahjong AI Competition** (Lu & Li, IJCAI 2024,
  <https://ijcai.org/Proceedings/2024/1020.pdf>, read in full through a text renderer):
  2020, 2022, 2023 editions with 37, 25 and 39 qualifying teams. 2020: most teams used
  heuristics; the top three used RL (PPO, IMPALA, self-play), the winner clipped scores as
  reward shaping. 2022–2023: SL "making up most of the top 16"; tiles encoded as images into
  ResNet/ResNeXt; symmetric-tile swapping to enlarge data; the 2023 winner added shanten
  features. A 2023 team found DQN better than PPO "under limited compute" but below SL.
  The organisers' two warnings: "agents trained by RL from scratch never choose some
  winning patterns, and even pretrained SL models can lose the ability to choose them
  during RL training"; and validation accuracy "cannot reflect its level of play". Format:
  Swiss rounds in duplicate, 24 games per deal. No human-vs-AI matches.
- **Later editions** (Botzone pages, <https://botzone.org.cn/static/gamecontest2025a.html>,
  <https://botzone.org.cn/static/gamecontest2026a.html>): 2025 winner 超强小登队 (SeaMan);
  2026 (the "6th", with an added LLM track) winner kong (海中队), LLM-track winner
  KitagawaHoly. Per-move limit 1 s (<https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong>);
  duplicate games use 136 tiles with flowers removed. Winners publish no papers.
- **Logs of the 2026 finalists** are public: 44,400 match files from the top 16 and
  simulation rounds, CC-BY-4.0, with 27 imitation networks
  (<https://huggingface.co/datasets/Dannibal/ijcai-mahjong-round2>). The card reports that
  behaviour-cloned policies "tied or lost to a strong reference bot in large-sample tests",
  a useful reminder that imitation of a bot caps at the bot.
- **RVR** (Li et al., CoG 2022, <https://ieee-cog.org/2022/assets/papers/paper_103.pdf>,
  read in full): PPO from scratch; the value network sees hidden information and outputs all
  four seats' values under a zero-sum constraint; an expected-reward network at the
  second-to-last step replaces the raw final reward. 56-core Xeon + 8 Titan V for three days.
  Ranked 4th among the IJCAI top 16 (top three excluded), beating 62.5 %. Ablations on MCR:
  the relative value network alone +5.22 % average fan and +6.34 % win rate over plain PPO;
  value variance 6.36 → 4.44. Settings: batch 2048, λ 0.995, γ 0.95, lr 3e-4, clip 0.2.
- **Tjong** (Li et al., CAAI TIT 9(4), 2024, <https://digitalcommons.njit.edu/fac_pubs/267>)
  *(abstract-only; full text paywalled)*: transformer of about 15 M parameters; two-stage
  decision (action type, then tile); "fan backward" reward shaping that credits the actions
  that built the winning hand; 0.5 M SL samples for 7 days on 2 GPUs, then 3 days RL;
  discard accuracy 81.51 %; top 1 % on Botzone. No code.
- **MPPO, "Elevating Styled Mahjong Agents with Learning from Demonstration"** (Li, Lu, Wang
  & Li, PKU, arXiv:2506.16995, June 2025, <https://arxiv.org/html/2506.16995v1>): PPO on
  IMPALA where 10 of 90 actors replay demonstrations from a teacher bot; 31,300 trajectories
  per teacher; 90 CPU cores + 2 × RTX 3090 for 7 days. Students of a rank-17 teacher beat
  the top Botzone bot (Elo 1328.76) 52.98 % of the time, while plain PPO students got
  39.04 %; CQL, BPPO, GAIL and SAIL "performed poorly". Configs at
  <https://github.com/AMysteriousBeing/MPPO-MCRMahjong>. **This is the most relevant
  small-budget result for us:** demonstrations from a *weaker* bot still lifted PPO above
  the best bot, which is the position our `value` bot and PIMC solver are in.
- **Mxplainer** (Li, Lu, Wang, Zheng & Li, *Algorithms* 18(12):738, 2025; arXiv:2506.14246,
  <https://arxiv.org/abs/2506.14246>) *(abstract-only)*: a parameterised search algorithm
  rewritten as an equivalent neural network, fitted to imitate a black-box agent; top-3
  action prediction over 92 % on humans and 90 % on AI agents (decision tree: 34.8 %); gives
  strategy-level insight and step-by-step explanations. No code link. Directly relevant to
  our "reasons" requirement (D1, D42).
- **Adapting rules of Official International Mahjong for online players** (Wang, Li, Lu &
  Li, arXiv:2601.08211, January 2026) *(abstract-only)*: uses "a world champion AI" in
  self-play to measure first-mover advantage and propose compensations. A model for using
  our bot to study HK rule variants (dealer disadvantage, flowers; Phase 1 §3).

## 3. LLMs and mahjong (2025–2026)

- **Wang et al., *Can Large Language Models Master Complex Card Games?*** (NeurIPS 2025;
  arXiv:2509.01328, <https://arxiv.org/html/2509.01328v5>): riichi track built from 7 K
  Tenhou 2020 human games (1 M training instances); Qwen2.5-7B, Llama3.1-8B, GLM4-9B with
  LoRA. Reward score vs three Mortal over 50 games (3/2/1/0 for 1st–4th, so parity is 1.5):
  base models 0.04–0.08, GPT-4o 0.25, fine-tuned 1.33–1.44. The text calls this "comparable
  to a very strong Mahjong AI"; the numbers say "below parity after fine-tuning".
- **Xu, *Text Is the Bottleneck*** (U. Tokyo, arXiv:2610.34791, 26 October 2026,
  <https://arxiv.org/html/2610.34791>, read in full): replicates the above with models up to
  2 B on one 16 GB GPU in about five GPU-days. Against three Mortal (community checkpoint),
  LLMs placed 2.95–3.20 on average (random 3.90, parity 2.50); a 2.6 M-parameter CNN on the
  same 100 K decisions placed 2.85; a 4.9 M-parameter transformer distilled on 1.71 M Mortal
  decisions (28 minutes of training) placed 2.51 ± 0.03, "statistically equivalent to
  parity". Agreement with Mortal: LLMs 71–78 %, CNN 80.6 %, transformer 94.7 %, human experts
  81.9 %. When a program supplies checkable facts, LLM rationale accuracy rises from 43.1 %
  to 92.8 %. Weights withheld because of Tenhou's terms; code released. **Two lessons for
  us:** a small structured network distilled from a strong teacher matches the teacher on
  our class of GPU; and explanations should come from computed facts, not from a language
  model's guess.
- **BotzoneBench** (Li et al., PKU, arXiv:2602.13214, January 2026,
  <https://arxiv.org/html/2602.13214v1>): LLMs against four graded tiers of Botzone MCR bots;
  the best model (Gemini 3 Pro preview) reached tier 3 with about a 44.6 % win rate against
  it; GPT-5.2, DeepSeek-3.2 and Claude Sonnet 4.5 sat at tier 2. No model reached the top
  tier. The 2026 Botzone LLM track (60 s per move, 32 K context, ≤ 10 B parameters) exists
  as a separate competition from the algorithm track.
- **NeSy-Mahjong** (Fang et al., ICPR 2026, LNCS 16817,
  <https://link.springer.com/chapter/10.1007/978-3-032-31673-8_11>) *(abstract-only)*: a
  decision LoRA aligned to professional riichi discards plus a reasoning LoRA grounded by a
  symbolic calculator that injects shanten and acceptance into the prompt; sub-10 B models
  beat DeepSeek-V3 and Qwen-Plus on decision accuracy and explanation quality. Code cited
  at <https://github.com/DarwinFang/icpr2026-nesy-mahjong>. The "calculator in the loop"
  design is the right shape for our review text, if an LLM is ever used for wording.
- **Abstraction Agent** (Li & Huang, arXiv:2609.04303, September 2026) *(abstract-only)*:
  an LLM proposes strategic features to bucket private states; transfers to riichi with no
  quantitative mahjong result.

**Reading of all four:** LLMs are not a route to strength in mahjong. They may help write
drill prompts or wording, with computed facts supplied; D42 forbids engines returning text
anyway.

## 4. Other variants

### 4.1 Taiwanese 16-tile mahjong (search-based tradition)

National Yang Ming Chiao Tung University (I-Chen Wu's group) and National Taipei University
(Jr-Chang Chen) have fielded Computer Olympiad programs since 2010. All are search-based
with hand knowledge; none publishes code.

- LongCat (NCTU theses 2008–2010, <https://ir.lib.nycu.edu.tw/handle/11536/48444>,
  <https://ir.lib.nycu.edu.tw/handle/11536/40106>) *(abstract-only, Chinese)*: shanten
  reduction first, then Monte Carlo simulation mid-hand to avoid dealing in; TCGA 2011
  champion; Computer Olympiad 2013 winner.
- KF-TREE (ICGA Journal 2021, <https://content.iospress.com/articles/icga-journal/icg210179>)
  *(abstract-only)*: single-player search for winning paths, knowledge plus Monte Carlo for
  hidden information; silver medal 2019.
- VeryLongCat won the 2022 Olympiad (ICGA Journal 44(3), 2022,
  <https://doi.org/10.3233/ICG-230217>, five Taiwanese teams) *(abstract-only)*.
- **MeowCaTS** (Tang, Chen & Wu, IEEE ToG 17(3), 2025,
  <https://scholar.nycu.edu.tw/en/publications/applying-importance-sampling-to-mcts-for-mahjong/>)
  *(abstract-only)*: MCTS with importance sampling, a "merging solitary tile" tree that cuts
  branching, importance weights in backpropagation, and a multi-depth transposition table;
  won the 2023 Olympiad. Search with learned or weighted sampling is the live research line
  for a four-player game with no logs, which is our situation.
- Sparrow mahjong (O'Connor, Gezgin & Parker, AIIDE 2025, arXiv:2508.07522)
  *(abstract-only)*: a small variant; CMA-ES-evolved LSTM "comparable to PPO". Minor.

### 4.2 Sichuan / bloody-battle mahjong

Three searches (English and Chinese, plus Semantic Scholar) found no paper on an agent for
血战到底. The only items are a 2022 patent (CN115115995A, a self-learning decision method
from video data) and a conference abstract on reward design for 2v2 Shangrao mahjong
(<https://sciforum.net/paper/view/24368>, login-walled, unread). Nothing to build on.

### 4.3 Hong Kong mahjong

No paper, thesis, competition or open bot was found (searches: "Hong Kong mahjong" with AI,
RL, bot, thesis; 港式麻雀 人工智能/機械學習; HKU, CUHK, HKUST, PolyU project lists; GitHub
topics). Public HK projects are faan calculators and hobby games with simple heuristics
(related-work.md). **This repo's Phase 1 and Phase 3 results are, as far as I can find, the
only measured HK-rules agents.**

## 5. Comparison table

Compute is the authors'. "Fit" is my judgement for an i7-12700K (20 threads), RTX 4060 Ti
16 GB, 32 GB RAM, local only, and for a phone running one WebAssembly thread with a 1–2 s
budget per hint. *Phone fit* refers to running the trained agent, not training it.

| System (year) | Variant | Method | Compute | Evaluation | Strength evidence | Open code / weights / licence | Fit: our box | Fit: phone |
|---|---|---|---|---|---|---|---|---|
| Mizukami & Tsuruoka (2015) | Riichi | Monte Carlo sim + 3 logistic opponent models | CPU; 17–72 M expert positions | Tenhou | rating 1718 | none | yes (needs logs we lack; models could be fitted on self-play) | sim too slow for 1 s unless few samples |
| Gao et al. (2019) | Riichi | SL CNNs (3 blocks × 100 filters) | 1 Tesla K10; 2.1 M discards | Tenhou, 300 matches | rating ~1850 | none | yes | yes (tiny net) |
| NAGA (2019–) | Riichi | 4 SL CNNs + rules | undisclosed; top-table logs | Tenhou | 8 dan; stable 6.64 (Suphx table) | closed; paid service | n/a | n/a |
| Suphx (2020) | Riichi | 5 CNNs, SL → policy-gradient RL, GRP, oracle guiding, pMCPA | 44 GPUs × 2 days per agent; 15 M SL samples | Tenhou, 5,760 games | stable 8.74 dan; > 99.99 % of players | none | training no (≈ 100× over); recipe yes at small scale | policy yes; pMCPA (100 K sims) no |
| Mortal (2021–) | Riichi | ResNet 40 × 192, dueling DQN, offline RL on logs | 1 GPU-days scale (community: 107 K steps × 2048) | self-match, vs akochan | +1.89 vs −5.67 pt vs akochan; experts agree 81.9 % | code AGPL-3.0; author's weights withheld; community weights AGPL on HF | run as opponent only (AGPL, riichi) | ONNX export exists; ~0.6 GFLOP/decision, 30–120 ms est. |
| akochan (–2022) | Riichi | search, expected value (undocumented) | CPU | vs Mortal | weaker than Mortal by 7.5 pt | custom terms | reference only | — |
| Kanachan | Riichi | BC + curriculum, transformer aim | — | none published | none | no licence listed; no data, no models | no | no |
| LuckyJ / ACH (2022–23) | Riichi (and MCR) | ACH neural regret, Hedge; 4-player system unpublished | undisclosed, large | Tenhou ~1,000 games; 6 MCR pros | stable 10.68 dan (press) | none | ACH idea only; 2-player guarantee | — |
| Meowjong (2022) | 3-player riichi | 5 SL CNNs + MC policy gradient | 4 × P100, ~30 h SL | vs random bots | 1st place 72 % vs random | code on GitHub (licence not checked) | yes | yes |
| Mjx (2022) | Riichi | simulator | — | — | — | MIT; build broken | env only | — |
| Mahjax (2026) | Riichi | JAX sim; BC + PPO + KL | 8 A100 sim; 1 GH200 × 5.8 h for 100 M steps | vs own BC agent | rank < 2.5 vs BC | code on GitHub; licence unstated | env on our GPU, ~5–10× slower (est.) | transformer policy, small |
| Yokoyama et al. (2026) | Riichi | hierarchical transformer, SL | — | discard accuracy | > Suphx accuracy (no number) | none | idea only | — |
| IJCAI winners 2022–23 | MCR | SL ResNet/ResNeXt, suit-swap aug., shanten features | "hundreds of cores" (2020); AI logs | duplicate Swiss, 24 games/deal | top of 25–39 teams | sample bot only; 2026 finalist logs CC-BY-4.0 | yes (recipe) | yes |
| RVR (2022) | MCR | PPO + oracle 4-seat zero-sum critic + expected reward | 56 cores + 8 Titan V × 3 days | Botzone top-16 | 4th of 16, beats 62.5 % | none | **yes, ~3–8× longer** | yes |
| Tjong (2024) | MCR | 15 M transformer, 2-stage decision, fan backward | 2 GPUs × 7 days SL + 3 days RL | Botzone | top 1 % | none | yes | yes (15 M params) |
| MPPO (2025) | MCR | PPO + demonstration actors | 90 cores + 2 × 3090 × 7 days | Botzone Elo, win rate | 52.98 % vs best bot | configs on GitHub | **yes, ~2× longer** | yes |
| Mxplainer (2025) | MCR | parametric search as a network, fitted to agents | small | prediction accuracy | 92 % top-3 on humans | none | yes | yes (it is a search) |
| Wang et al. LLM (2025) | Riichi | 7–9 B LoRA on human logs | GPU-days | vs 3 Mortal, 50 games | below parity (1.33–1.44 of 1.5) | data from Tenhou; weights? | no (LLM) | no |
| Xu (2026) | Riichi | 4.9 M transformer distilled from Mortal | 1 × 16 GB GPU, 28 min | vs 3 Mortal, 1,000 games | 2.51 ± 0.03 = parity | code yes; weights withheld | **yes** | yes |
| BotzoneBench (2026) | MCR | LLMs vs graded bots | API calls | tiered win rates | best LLM below tier 4 | bench code | n/a | no |
| MeowCaTS (2025) | Taiwanese | MCTS + importance sampling | CPU | Computer Olympiad | 1st 2023 | none | yes (recipe) | depends on sim count |
| This repo (2026) | **Hong Kong** | faan-aware shanten, PIMC anchored to `value`; PPO from distilled `value` | M1, hours | duplicate vs `value`, 95 % CI | PIMC +1.7 ± 3.2; PPO −0.1 ± 2.9 | ours | — | PIMC 100–200 samples in 2 s (bot-stream §4 estimate) |

## 6. What transfers, and what does not

- **Transfers:** tile-plane encodings and look-ahead features (Suphx; we have them); an
  oracle critic that sees hidden tiles and outputs four zero-sum values (RVR; cheap, large
  effect); demonstrations mixed into PPO (MPPO; our `value` bot and PIMC are the teachers);
  distillation of a search or a strong net into a small network (Xu; AlphaZero-style);
  suit-swap augmentation (IJCAI; HK's three suits are symmetric, and so are the three
  dragons); duplicate evaluation with all seatings (IJCAI, Mortal); importance-weighted
  sampling in search (MeowCaTS); parametric-search explainers (Mxplainer).
- **Transfers with care:** global reward prediction (Suphx) matters for multi-round matches;
  our unit is a hand, so it only matters once matches (rules §10) exist. Riichi defensive
  knowledge (furiten, safe tiles after riichi) does not exist in HK; HK has no furiten and
  no riichi declaration, so threat detection must come from exposed melds and discard
  patterns alone.
- **Does not transfer:** any weights (riichi, and licence); human-log pipelines (no logs);
  ACH's convergence story (two-player only); LLM policies (lose to mid-tier bots).

## 7. Open agents that could serve as benchmark opponents

Rule reminder: open-source agents may be run as benchmarks and read, never linked or
shipped (holdem decisions, carried over in the plan).

| Agent | Rules | How we would use it | Licence / caveat |
|---|---|---|---|
| Mortal (community checkpoint) | Riichi | Opponent in a riichi proxy match for our *training recipe* (§8 bar 6), through the mjai protocol in a separate process | AGPL-3.0; community weights only; Tenhou terms cited by Xu 2026 as a reason to withhold weights |
| akochan | Riichi | Second riichi opponent, weaker | custom terms; read before use |
| Mahjax BC/PPO agent | Riichi | Weak sanity opponent inside Mahjax | licence unstated |
| Botzone sample bot; 2026 finalist logs + imitation nets (HF) | MCR | MCR proxy for the faan-minimum problem; imitation nets as opponents | CC-BY-4.0 logs; nets tied or lost to a reference bot |
| Mjx random agent | Riichi | floor only | MIT |
| RLCard mahjong | simplified | not useful (toy rules) | MIT |

**For Hong Kong rules there is no external opponent.** Every HK benchmark must be built
here: the `value`/`defensive` ladder, PIMC, and ported versions of the published recipes.

## 8. What "state of the art for HK mahjong" can mean, and six measurable bars

With no HK benchmark, "state of the art" can mean four different things. Each has a
measurable bar. I propose we claim the title only when bars 1–4 are met, and report 5 and 6
as supporting evidence.

| # | Meaning | Bar (measurable) | Cost (estimate) |
|---|---|---|---|
| 1 | **Best agent in this repo, with intervals** | Beats the `value` field by ≥ +5 points/hand with a 95 % CI that excludes 0, in a duplicate tournament of ≥ 4,000 hands (CI ≈ ±1.5 at 1,600 hands, Phase 1), and beats PIMC-128 head-to-head with a CI that excludes 0 (24 seatings when three agent types share a table) | hours of CPU |
| 2 | **Beats the published recipes ported to HK, at equal compute** | Three ports trained for the same 3 GPU-days each: (a) RVR-style PPO with oracle 4-seat critic; (b) Mortal-style ResNet with offline RL on our self-play logs; (c) MPPO-style PPO with `value`/PIMC demonstrations. Our final agent beats each by a CI-separated margin | ~2 weeks of build; 9 GPU-days |
| 3 | **Hard to exploit** | A best-response exploiter (one seat, trained 1 GPU-day against three frozen copies, oracle-free) gains < 3 points/hand (this is the multi-player stand-in for exploitability; see holdem multiway.md §5) | 1 GPU-day per check |
| 4 | **Human level, measured** | Duplicate-vs-human: Brian and invited club players each play ≥ 400 hands against three bots; the bot then plays the same walls in the same seat against the same three bots; the bot's paired margin is ≥ 0 with CI ±4 or better. Secondary: on a hand-labelled set of ≥ 200 positions from those games, the bot's top choice agrees with the strong player's choice ≥ 75 % (Xu 2026 reports 81.9 % expert–Mortal agreement in riichi as a reference point) | human time: ~20 hours of play |
| 5 | **Plays every pattern** (no collapse) | Win share per faan pattern within ±30 % of the PIMC solver's shares; draw rate 5–10 % in self-play (IJCAI warning; our DMC collapsed to 92 % draws) | free (logged at every evaluation) |
| 6 | **Method proxy on riichi** (optional) | Our training recipe, run unchanged on riichi rules (Mahjax or our engine with riichi rules), reaches average rank ≤ 2.55 over 1,000 hanchan against three Mortal community checkpoints, from no human data | riichi engine or Mahjax port: 1–3 weeks; 3–6 GPU-days |

Why these six: bar 1 is what the repo already measures. Bar 2 answers "would a Suphx/RVR/
Mortal team have done better with our compute?". Bar 3 is the only robustness measure that
exists for four players. Bar 4 is what the product promises. Bar 5 catches the failure the
IJCAI organisers and our own DMC run both showed. Bar 6 is the only way to compare our
*method* with an external strong agent, since none plays HK rules.

**What we cannot claim:** "superhuman" (no population of HK players to rank against), and
"state of the art versus Suphx/LuckyJ" (different game, 100× compute).

## 9. Fit for our hardware and for a phone

- **Training.** RVR (24 GPU-days on Titan V), MPPO (14 GPU-days on 3090 plus 90 CPU cores)
  and Xu (5 GPU-days on a 5070 Ti) are the published results nearest our budget. On one
  4060 Ti with 20 CPU threads, an RVR-size run is 3–8× longer than theirs (own estimate:
  Titan V and 4060 Ti have similar FP32 throughput; the gap is the CPU count for self-play,
  56 cores vs 20 threads). Our Rust engine at ~1.7 ms per full hand for four `fast` seats
  (solver.md) is faster than the Python environments those papers used, which narrows the
  gap. Mahjax's 100 M steps in 5.8 h on a GH200 would be roughly 1–2 days on a 4060 Ti (own
  estimate).
- **Inference on a phone (WebAssembly, one thread).** A Mortal-size ResNet (40 blocks ×
  192 channels over 34 tile positions) is about 300 M multiply-adds per decision (own
  arithmetic: 34 × 192 × 192 × 3 per conv, two convs per block). With WebAssembly SIMD on one
  phone core at 5–20 GFLOP/s, that is 30–120 ms, well inside 1–2 s; thermal throttling on
  phones can double latency (ONNX Runtime Web guidance, no measured mobile numbers found).
  Our current MLP (933 → 512 → 512 → 256) is under 1 M multiply-adds, under 1 ms. A 5–15 M
  parameter network (Xu, Tjong) is 20–60 MB in fp32 and 5–15 MB in int8; acceptable for an
  app download. ONNX Runtime Web or a hand-written Rust inference path (our `MJNN` format
  already runs in Rust) both compile to WebAssembly.
- **Search on a phone.** PIMC at 32 samples × 4 candidates takes 0.2 s on 8 threads
  (Phase 1 §4.3), so about 1.6 s on one desktop thread; a phone core is 2–3× slower. The
  bot-stream brief estimates 100–200 samples in 2 s. Search must therefore be anytime
  (stop when the paired z-test is decided) and candidate-pruned by the network; a full-hand
  review of 30–60 s must run in pieces inside `budget_ms` (bot-stream §4).
- **Memory.** Shanten caches are unbounded per thread today (bot-stream §4); they need a cap
  or a precomputed per-suit table before the WebAssembly build.

## 10. Risks and where the claims are weak

- **Press-only strength claims:** LuckyJ's 10.68 dan and its MCR match come from Tencent's
  announcement; the four-player system has no paper. NAGA's rank is self-reported.
- **Abstract-only reads:** ACH, Tjong, Kurita & Hoki, Mxplainer, Yokoyama 2026, MeowCaTS,
  KF-TREE, LongCat, NeSy-Mahjong, Abstraction Agent, the rules-adaptation paper and Sparrow
  mahjong. The risk-aware hierarchical framework is unread. Numbers from those entries should
  not be load-bearing in a build plan without a second look.
- **Mortal's documentation is incomplete;** its architecture facts here come from a
  community model card, and its only published strength number is against akochan.
- **"Parity with Mortal" is a ceiling for imitation,** not a strength benchmark (Xu 2026
  says so). The same holds for any agent we distil from PIMC: distillation alone caps at
  PIMC; RL or search must add the rest.
- **The IJCAI competition logs are bot logs,** not human logs, since 2022; "beats the top 16"
  is a claim about a field of student bots, not about humans.
- **Semantic Scholar rate-limited** the final sweep for 2024–2026 papers, so the list in
  §1.8–§3 comes from targeted searches and may miss a paper.
- **No HK population to measure against.** Bar 4 depends on volunteers; a small sample
  gives wide intervals, so its wording ("≥ 0 with CI ±4") is deliberately modest.

## Sources

- Mizukami, N., Tsuruoka, Y. *Building a computer mahjong player based on Monte Carlo
  simulation and opponent models.* CIG 2015.
  <https://www.logos.ic.i.u-tokyo.ac.jp/~tsuruoka/papers/cig2015mizukami.pdf>
- Gao, S., Okuya, F., Kawahara, Y., Tsuruoka, Y. *Building a computer mahjong player via
  deep convolutional neural networks.* 2019. <https://ar5iv.arxiv.org/html/1906.02146>
- Kurita, M., Hoki, K. *Method for constructing AI player with abstraction to MDPs in
  multiplayer game of mahjong.* IEEE ToG 2021. <https://arxiv.org/abs/1904.07491>
- Li, J. et al. *Suphx: Mastering Mahjong with deep reinforcement learning.* 2020.
  <https://ar5iv.labs.arxiv.org/html/2003.13590>
- NAGA. Dwango Media Village. <https://dmv.nico/en/articles/mahjong_ai_naga/>
- Mortal. <https://github.com/Equim-chan/Mortal>; strength page
  <https://mortal.ekyu.moe/perf/strength.html>; weights statement
  <https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56>; community model
  card <https://huggingface.co/hhim8826/mortal3-b40c192>
- akochan. <https://github.com/critter-mj/akochan>
- Kanachan. <https://github.com/Cryolite/kanachan>
- Fu, H. et al. *Actor-critic policy optimization in a large-scale imperfect-information
  game.* ICLR 2022. <https://mlanthology.org/iclr/2022/fu2022iclr-actorcritic>
- LuckyJ press: <https://www.ithome.com/0/705/059.htm>;
  <https://technode.com/2023/07/12/tencents-mahjong-ai-sets-new-gaming-record-on-international-mahjong-platform/>
- Zhao, X., Holden, S. *Building a 3-player mahjong AI using deep reinforcement learning.*
  CoG 2022. <https://ieee-cog.org/2022/assets/papers/paper_170.pdf>;
  <https://arxiv.org/abs/2202.12847>; code <https://github.com/victorzxy/meowjong>
- Koyamada, S. et al. *Mjx: A framework for mahjong AI research.* CoG 2022.
  <https://github.com/mjx-project/mjx>
- Nishimori, S. et al. *Mahjax: A GPU-accelerated mahjong simulator for reinforcement
  learning in JAX.* 2026. <https://arxiv.org/html/2605.20577v1>;
  <https://github.com/nissymori/mahjax>
- Yokoyama, H., Tahara, Y., Sei, Y. *Hierarchical transformer-based tile discard prediction
  in mahjong.* IPSJ TOD 19(1), 2026. <https://ipsj.ixsq.nii.ac.jp/records/2006864>
- *A hierarchical and risk-aware decision framework for riichi mahjong AI.* 2026.
  <https://www.researchgate.net/publication/407079265_A_Hierarchical_and_Risk-Aware_Decision_Framework_for_Riichi_Mahjong_AI> (unread)
- Lu, Y., Li, W. *Mahjong AI competition: Exploring AI application in complex real-world
  games.* IJCAI 2024. <https://ijcai.org/Proceedings/2024/1020.pdf>
- Botzone competition pages: <https://botzone.org.cn/static/gamecontest2025a.html>;
  <https://botzone.org.cn/static/gamecontest2026a.html>; rules wiki
  <https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong>; sample code
  <https://github.com/ailab-pku/Chinese-Standard-Mahjong>
- IJCAI-2026 finalist logs: <https://huggingface.co/datasets/Dannibal/ijcai-mahjong-round2>
- Li, Z. et al. *Speedup training AI for mahjong via reward variance reduction.* CoG 2022.
  <https://ieee-cog.org/2022/assets/papers/paper_103.pdf>
- Li, X. et al. *Tjong: A transformer-based mahjong AI via hierarchical decision-making and
  fan backward.* CAAI TIT 2024. <https://digitalcommons.njit.edu/fac_pubs/267>
- Li, L., Lu, Y., Wang, Y., Li, W. *Elevating styled mahjong agents with learning from
  demonstration.* 2025. <https://arxiv.org/html/2506.16995v1>;
  <https://github.com/AMysteriousBeing/MPPO-MCRMahjong>
- Li, L. et al. *Mxplainer: Explain and learn insights by imitating mahjong agents.*
  Algorithms 2025. <https://arxiv.org/abs/2506.14246>
- Wang, C., Li, L., Lu, Y., Li, W. *Adapting rules of Official International Mahjong for
  online players.* 2026. <https://arxiv.org/abs/2601.08211>
- Wang, W. et al. *Can large language models master complex card games?* NeurIPS 2025.
  <https://arxiv.org/html/2509.01328v5>
- Xu, M. *Text is the bottleneck: A small-scale replication of LLM game playing in riichi
  mahjong.* 2026. <https://arxiv.org/html/2610.34791>
- Li, L. et al. *BotzoneBench: Scalable LLM evaluation via graded AI anchors.* 2026.
  <https://arxiv.org/html/2602.13214v1>
- Fang, Z. et al. *Neuro-symbolic instruction tuning for explainable mahjong agents via
  two-stage dual-LoRA.* ICPR 2026. <https://link.springer.com/chapter/10.1007/978-3-032-31673-8_11>
- Li, B., Huang, L. *Abstraction Agent.* 2026. <https://arxiv.org/abs/2609.04303>
- Tang, S.-C., Chen, J.-C., Wu, I-C. *Applying importance sampling to MCTS for mahjong.*
  IEEE ToG 17(3), 2025.
  <https://scholar.nycu.edu.tw/en/publications/applying-importance-sampling-to-mcts-for-mahjong/>
- KF-TREE, ICGA Journal 2021. <https://content.iospress.com/articles/icga-journal/icg210179>
- VeryLongCat, ICGA Journal 44(3), 2022. <https://doi.org/10.3233/ICG-230217>
- LongCat theses (NCTU). <https://ir.lib.nycu.edu.tw/handle/11536/48444>;
  <https://ir.lib.nycu.edu.tw/handle/11536/40106>
- O'Connor, J., Gezgin, D., Parker, G. *Evolutionary optimization of deep learning agents for
  Sparrow mahjong.* AIIDE 2025. <https://arxiv.org/abs/2508.07522>
- Shangrao 2v2 reward design (abstract, login-walled): <https://sciforum.net/paper/view/24368>
- Sichuan patent CN115115995A: <https://patents.google.com/patent/CN115115995A/en>
- Wikipedia, *Mahjong and artificial intelligence* (checked; nothing after 2023):
  <https://en.wikipedia.org/wiki/Mahjong_and_artificial_intelligence>
- ONNX Runtime Web: <https://opensource.microsoft.com/blog/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/>
- This repo: [related-work.md](related-work.md), [../solver.md](../solver.md),
  [../rl.md](../rl.md), [../study-results/phase1-solver.md](../study-results/phase1-solver.md),
  [../study-results/phase3-rl.md](../study-results/phase3-rl.md); holdem
  `docs/research/multiway.md` §5 (best-response gap, cross-play).
