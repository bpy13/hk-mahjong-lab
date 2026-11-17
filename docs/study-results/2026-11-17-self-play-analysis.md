# Why self-play has not lifted our bots above a strong start (17 Nov 2026)

## Summary

1. **Did self-play ever improve on a baseline?** Yes, but only from weak starts, or against fixed
   opponents. Never from a strong start by playing itself. Big Two from scratch rose from -2.22 to
   -0.17 +- 0.82 against the search bot; mahjong rose from -5.6 to -0.1 +- 2.9; Big Two networks trained
   against one frozen bot gained +1.8 to +3.0. From a strong start, self-play ended worse three times
   (Dou Dizhu twice, Big Two once) and level once (mahjong).
2. **Is there a bug?** No wrong sign, wrong seat or wrong index in either trainer. There are setup
   faults that matter:
   - **Dou Dizhu: the training is mostly on a different game from the one we measure.** 75 % of the
     training positions are under the app's rules (bidding, doubling, bigger stakes). The student
     never saw those rules, its bidding layer is untrained, and those positions carry about 96 % of
     the weight in each update. The yardstick uses the research rules.
   - **Mahjong: the run could not have shown a gain.** A pull toward the start was on for the whole
     10 hours, and the stop rule needed a jump of 2 points a hand in one check.
3. **Most likely reason it fails.** The learning signal is far smaller than the noise in each update.
   Two new numbers from the existing logs show it. Mahjong: the training score itself (learner minus
   the frozen start, same tiles) stayed flat over 1.29 M hands: -0.70 to -0.82, trend -0.008 +- 0.031
   a hand per hour. Dou Dizhu: even in the run where 76 % of the games were against the fixed outside
   programs, the score in those training games fell from -1.15 to -1.48 (+- 0.05). So it is not
   mainly "the bot fools its own copies"; the update itself damages a good network.
4. **What to try next.** Three short Dou Dizhu runs (about 5 GPU-hours in all) that separate
   "wrong game", "update too rough" and "hidden bug"; one 4-hour mahjong run with less noise and no
   fading pull; and, as the cheaper route to a stronger bot, more imitation data.

Every claim below is marked **checked** (I read the code or computed the number from a log) or
**inferred** (my reasoning from checked facts). Paths are under
`C:\Users\brian\OneDrive\Documents\workspace\repo\.worktrees\`. Nothing was changed in any project.

Terms used: **self-play**: the bot learns by playing copies of itself. **PPO**: the learning method in
both trainers; it plays games, then nudges the network toward moves that turned out better than
expected. **Critic**: the part of the network that estimates how good a position is. **Advantage**:
how much better a move turned out than the critic expected; it is the signal the network learns from.
**Imitation network / student**: a network trained only to copy another bot's moves. **Greedy play**:
always the network's top move; **sampled play**: moves drawn at random according to the network's
probabilities (training uses this). **Learning rate**: the size of each nudge.

---

## Part A. Bug audit

### A1. Dou Dizhu trainer (`doudizhu-workers\python\ddzbot\train\`)

#### Faults found

**F1. Most of the training is on a game the student never learned, and that game dominates each
update. Severity: high. (checked, effect on the curves inferred)**

- The student was trained only on research-rule positions (fixed landlord, no bidding):
  `distill\train.py:46` builds every position with `DdzTable`, which uses the research rules
  (`crates\pysim\src\game.rs:145`). `distill\train.py:160` says the auction does not occur in the data.
  So the bidding and doubling output layer of the student has its random starting weights. (checked)
- Self-play plays 75 % of its games under the app's rules: `research_share: 0.10`, `league_share: 0.15`
  in all four 12e configs and in 2c (`train.py:187-199`). (checked)
- The network is told which rule set it plays by 12 numbers; 3 of them differ between the two rule
  sets. I fed 4,000 held-out positions to the student with the app-rule numbers instead of the
  research ones. Its top move changed in 10.8 % of decisions and agreement with PerfectDou fell from
  0.810 to 0.778. The critic's estimate jumped from 0.084 to 0.339 (in units of 8 points), so the
  critic is off by about 2 points a position under the app rules at the start. (checked, short CPU script)
- App-rule games have much larger stakes. From `runs\2c-diag\result-selfplay.json` (student playing
  itself, 400 games each): mean payout size 0.605 under the app rules against 0.223 under the research
  rules; spread of the advantage 0.545 against 0.179. (checked)
- The trainer puts all positions of an update on one common scale (`train.py:145`). A position's
  weight in the update grows with the square of its advantage. With 75 % of positions at 0.545 and
  25 % at 0.179, the app-rule positions carry about 96 % of the weight; in run D (half the games
  against the outside programs) still about 86 %. (inferred from the checked numbers; the outside
  games' own spread was not logged)
- What it means: the four 12e runs did not test "self-play in the game the student knows". They
  tested "learn the app's game, starting with random bidding, and see what happens to research-rule
  play". The yardstick (the tracker) is research rules only (`league.py:182-187`).
- How it would show in the curves: research-rule strength drifts down while nothing in the log
  looks wrong; more bombs (each doubles the stake under the app rules). Both are seen: bombs per game
  rose from 0.35 to 0.41-0.47 in all four runs. (checked in `results.jsonl`)
- Not proven as the cause: nobody has measured whether the trained networks got better under the app
  rules. See experiment 1.

**F2. The shaping reward is not neutral. Severity: medium for runs 2a and 2c; cleared as the cause of
the 12e fall. (checked)**

- Shaping is a small extra reward for getting closer to going out than the other side
  (`rollout.py:180-187`). Such a reward leaves the best strategy unchanged only if it is zero at the end
  of the game and the critic accounts for it (Ng, Harada, Russell 1999).
- Here the end value is not zero (`rollout.py:75-81`: it rewards winning by a wide margin), and the
  critic predicts the payout only, never the shaping (`critic.py:71-80`, targets at `rollout.py:189`).
  So the shaping acts as a real second goal: "shed cards faster than the others, soon".
- It is a fifth (app rules) to two fifths (research rules) of the reward size (2c-diag file). PerfectDou
  removes its shaping in a final stage; run 2c never reached its fade.
- Run 12e-C (shaping 0) fell the same as the control, so this is not what makes the student worse.

**F3. No gentle start when training begins from the student. Severity: medium. (checked, effect inferred)**

- `train.py:262` and `:274-278`: starting from the student loads the weights only; the optimizer (the
  routine that applies the nudges, here "Adam") starts fresh, at the full learning rate 1e-4, with no
  warm-up. Adam's first steps move every weight by about the learning rate whatever the signal.
- One update is 4 passes over about 100,000 positions in batches of 1,024, about 400 nudges
  (`train.py:149-176`). From the first update on, 11 % of the played moves had their probability changed
  by more than 20 % (`clip_frac` 0.11 in every 12e log). For a network that is already good, that is a
  lot of movement per update. (checked)
- Published fine-tuning from an imitation network uses much smaller steps: PIRLNav warms the policy's
  learning rate from 0 up to 1.5e-5; Suphx cut its learning rate to a tenth for continued training and
  reports that otherwise training "is not stable and does not lead to further improvements". (Part C)

**F4. The critic does not know the self-play game at the start. Severity: medium, but not the whole
story. (checked)**

- The student's critic was fitted only on positions where the played move was PerfectDou's move, under
  the research rules (`distill\train.py:230-232`). Its values for other moves were never trained.
- In 12e its error starts at 0.51 and is still falling at the end of 90 minutes (0.22-0.27). (checked)
- The advantage leans heavily on the critic: with the trainer's setting `lam: 0.9`
  (`advantage.py:27-34`), the real result of the game has only 0.9^19 = 13 % of the weight for a move
  20 steps before the end; the rest comes from the critic's own estimates. (checked arithmetic)
- Against it being the whole cause: in run 2c the critic had 57 M steps to learn (error 0.13), and
  the network still fell within 4 M steps of the imitation term reaching zero. (checked)

**F5. Smaller faults. Severity: low. (checked)**

- Games against the outside programs are counted wrongly. `league_share: 0.15` is the share of
  tables, but those tables finish games two to three times as fast, so they are 32 % of finished games
  (76 % in run D). Tables where the learner is the landlord finish about twice as fast as tables where
  it is both peasants, so the learner meets the outside programs about twice as often as landlord.
  (computed from `iter_games` and `iter_trajectories` in the logs)
- Each update stops when 2,048 games have ended and throws away the games still running
  (`rollout.py:118`, new tables every update at `train.py:326`). Long games are dropped more often, so
  the data lean toward short games, by roughly a tenth of the games.
- The critic is not told whether the opponents are the outside programs or copies of the learner, so
  it over-rates the learner's chances against the outside programs.

#### Checked and found correct

| Item | Verdict | Evidence |
|---|---|---|
| Sign and scale of the payout per seat | correct | `rollout.py:105-116` gives each seat its own payout / 8; the simulator asserts payouts sum to zero (`env.rs:321`); run 2a improved both roles from scratch |
| The two peasants' reward and value targets | correct | each peasant's steps get its own (equal) payout; one shared output layer (`model.py:217-218`) |
| Payout assigned to each seat's steps | correct | added on the seat's last step (`rollout.py:187`), carried back by the return formula |
| Return and advantage formula, discount 1.0, lambda 0.9 | correct as designed | `advantage.py:20-35`; unit test `test_advantage_matches_hand_computed_expected_sarsa_lambda` |
| Seat order of hands, plays-to-go and landlord | consistent | all relative to the seat to act (`game.rs:255-311`); end-of-game numbers by absolute seat, used that way (`rollout.py:75-81`) |
| Shaping sign per seat | correct | `rollout.py:60-72`; test `test_potential_favours_the_side_that_is_closer_to_going_out` |
| Illegal moves masked when sampling | correct | `model.py:222`, `rollout.py:137-139` |
| Illegal moves masked in the loss (log-probability, entropy) | correct | `train.py:156-161`; the masked moves contribute exactly 0 |
| Old-policy log-probabilities | stored at play time, same arithmetic as the update | `rollout.py:139`, `:134`; test `test_collect_and_update_use_the_same_autocast` |
| Advantage scaling | one scale over all seats and rule sets | `train.py:145`; correct in itself, but see F1 |
| Outside programs' seats give no training rows | correct | only learner seats are observed (`league.py:132-146`, `rollout.py:120-126`) |
| Learning rate, clip, passes, batch, gradient clipping | as configured: 1e-4, 0.2, 4, 1,024, 1.0 | `train.py:149-176`; policy and critic share one clip, harmless with Adam |
| Features in imitation data vs self-play | same code path | both use the simulator's `observe`; the difference is the rule numbers (F1) |
| Train mode vs play mode | no difference | no dropout in the network (`model.py:121`) |
| Tracker play | greedy, research rules, 1,000 paired deals | `league.py:201` |
| Start from the student | weights of policy and critic loaded; optimizer fresh | `train.py:274-278`; see F3. There is no input normaliser to load |
| Half-precision arithmetic (`bf16: true`) | small effect | measured on 3,000 positions: move log-probabilities change by 0.005 on average, top move changes in 0.4 % |

#### What the existing logs say (all checked)

- **Score in the training games against the outside programs** (learner's points per game, sampled
  play; computed from `mean_payout`, self-play games sum to zero):

  | Run | first sixth of the run | last sixth | accuracy |
  |---|---|---|---|
  | A control | -1.20 | -1.61 | +- 0.06 to 0.11 |
  | B entropy 0 | -1.06 | -1.28 | +- 0.05 to 0.08 |
  | C shaping 0 | -1.24 | -1.95 | +- 0.08 to 0.11 |
  | D half the tables outside | -1.15 | -1.48 | +- 0.05 |
  | 2c, while the imitation term held (10-54 M steps) | -0.78 to -0.81 | | +- 0.02 |
  | 2c, after it reached zero (67 M steps on) | -1.79 | -1.74 | +- 0.01 |

  This matters. Against fixed opponents there is no "fooling your own copies". A working learner
  cannot keep getting worse at the very score it is trained on, unless its updates are too rough,
  its signal is biased, or other positions drown these out (F1). The fall is in sampled play too, so
  it is not an artefact of measuring greedy play.
- **Critic quality**: the win-chance error falls from 0.44 to about 0.20 in 12e and to 0.085 in 2c.
  The critic learns; it does not explode. The explained share of the result was not logged.
- **Clip fraction** 0.09-0.12 throughout; **entropy** (how spread the move probabilities are) 0.32 in
  A, falling to 0.20 in B, rising to 0.41 in C. The fall in strength is the same in all three, so
  entropy is not the driver.
- **Self-play landlord win rate** rose 0.42 to 0.50-0.55 while the landlord's win rate against
  DouZero fell 0.34 to 0.24. Part of the rise is expected for a dull reason: under the app rules the
  bidding layer starts random and learns to bid on good hands (inferred). The 2c-diag file shows the
  rise under the research rules too (0.38 to 0.52), so peasant play also got worse.

#### Could not check without running code

- How far the policy moves per update (the "KL" figure) and how well the critic explains results are
  not logged. Add them, split by rule set and by seat.
- Whether the trained networks are better under the app rules (experiment 1).
- Whether the critic's values for rarely played moves are biased enough to matter.

#### Cheap decisive tests

- **Rigged-reward test (10 minutes, GPU).** Start from the student, pay +1 for every bomb played and
  nothing else, 5 updates. Bombs per game must rise clearly. If they do not, the PPO loop has a bug.
- **App-rules head-to-head (5 CPU minutes after a small script change).** Experiment 1 below.

### A2. Mahjong trainer (`hk-mahjong-lab-workers\python\mahjong\rl\`)

How it works, in short: every deal is played twice with the same tiles. In one copy the learner
(sampled play) sits in one seat; in the other the frozen starting network (greedy play) sits there.
The reward is the difference in points, squashed by a logarithm. Every decision of the hand gets that
same number as its signal.

#### Faults found

**M1. The run could not have shown a gain. Severity: high for the conclusion drawn, not a code bug.
(checked)**

- The pull toward the starting network was on for the whole run: weight 0.1 at the start, still
  0.058 at hour 10 (`ppo_cnn.py:740`, `log.json`). The distance from the start stayed small (0.07
  rising to 0.12). The network was held at its start, so "level" was the expected result whether or
  not learning works.
- The stop rule (`pool.py:157-167`) counts a gain only if one 2-hourly check beats the best earlier
  check by more than that check's range, +- 2.0 points a hand. No realistic gain is that large. With
  the first check as "best", four checks without such a jump end the run at hour 10. It did.
  The run folder's own review says the same.

**M2. The learning signal is buried in noise. Severity: high. (checked numbers, conclusion inferred)**

- In paired mode the advantage is the hand's final paired difference, the same for all of the
  learner's decisions in that hand; the critic is not used (`ppo_cnn.py:728-729`). There is no
  per-decision credit.
- The paired difference varies by about 53 points from hand to hand (from the spread of the
  per-update means in `log.json`). One update uses 1,024 hands, so its average is only good to
  +- 1.65 points. A real improvement of 0.5 points a hand is a third of that noise.
- **The training score never moved.** Mean paired difference by block (1.29 M hands in all):

  | Hours | learner minus frozen start, points a hand | 95 % range |
  |---|---|---|
  | 0-1 | -0.71 | +- 0.24 |
  | 1-2 | -0.58 | +- 0.27 |
  | 2-4 | -0.87 | +- 0.20 |
  | 4-6 | -0.55 | +- 0.20 |
  | 6-8 | -0.71 | +- 0.23 |
  | 8-10 | -0.82 | +- 0.22 |

  Trend: -0.008 +- 0.031 points a hand per hour. The learner starts 0.7 behind because it samples its
  moves and the frozen copy does not. It never closed even that gap. Meanwhile its entropy rose from
  0.19 to 0.25: it drifted toward more random play, the mark of noise, not of learning.
- This table is ten times more accurate than the 2-hourly checks (+- 0.2 against +- 2.0) and was in the
  log all along. It should be the run's main gauge.

**M3. The squashed reward is not points a hand. Severity: low to medium. (checked)**

- `ppo_cnn.py:175-178`: reward = sign x log(1 + |difference| / 8). This shrinks big wins and big
  losses. In a game where much of the score comes from rare large hands, the learner is trained
  toward many small gains. The yardstick counts raw points.

#### Checked and found correct

| Item | Verdict | Evidence |
|---|---|---|
| Paired deals | correct | same tiles, same random numbers until the two copies differ (`crates\pybind\src\rl.rs:295-304`, `:320-345`) |
| Baseline in the pair | does not depend on the learner's moves, so it is a fair yardstick | frozen start, greedy (`ppo_cnn.py:651`) |
| Learner seat vs pool seats | correct | only the learner's seat in copy 0 is trained on (`ppo_cnn.py:653`); pool seats drawn per hand, 50/30/20 (`pool.py:40-51`), same draw for both copies |
| Stage bot and older networks as opponents | play, are not trained on | `ppo_cnn.py:629-645` |
| Forced wins | every legal win is declared and not trained on | `ppo_cnn.py:383-391`, `:653` |
| Illegal moves masked in play, log-probability, entropy | correct | `ppo_cnn.py:379-380`, `:604-606`, `:751-760` |
| Value targets | squashed points of the four seats; the critic is trained but unused in paired mode | `ppo_cnn.py:330-331`, `:758` |
| Minimum-faan stages | off in this run; 3 faan throughout | `selfplay.ps1` passes `--no-curriculum`; `log.json` has min_faan 3 in all 1,262 entries |
| Pattern anchor | off | `--anchor-weight 0`; anchor loss 0 in the log |
| Pull | as planned: 0.1 x distance to the start, fading to zero at hour 24 | `pool.py:110-115` |
| Collapse rule | correct as written | `pool.py:148-156` |
| Stall rule | works as written, cannot work as intended | M1 |
| Sign of the learning signal | correct | the older Phase 3 run lifted a weak start by 5.5 points with the same family of code |

#### Could not check without running code

- What share of hands end with a paired difference of exactly zero (learner and frozen copy played
  the same moves). Those hands teach nothing.
- Whether the network would drift down once the pull reaches zero, as Dou Dizhu 2c did.

#### Cheap decisive test

- Already done above: the paired-difference table from `log.json`. For any future run, print it
  every hour.

---

## Part B. Every learning-by-playing run

Verdict is against the run's own starting point. Dou Dizhu and the November mahjong run: numbers
checked by me in the logs and result files. Big Two, Thirteen Cards and older mahjong: read by a
helper from the result files; I re-checked the Big Two rows marked *.

### Dou Dizhu (yardstick: points a deal against DouZero-ADP, research rules, greedy, 1,000 paired deals, about +- 0.055)

| Run | Start and its strength | Size | End | Verdict |
|---|---|---|---|---|
| 2a-smoke | random weights | 1 GPU-h, 3.9 M steps | -1.06 | start of 2a |
| 2a | random | 8 GPU-h, 16.5 M steps | -1.13 (mean of 8 checks) | flat |
| 2a-r2 | random | 142.7 GPU-h, 599 M steps | -0.72 | improved, slowly; far below 0 |
| Student (imitation only, not self-play) | copies PerfectDou, 1.1 M positions | 0.3 GPU-h | **-0.273 +- 0.053** | best bot |
| 2c | student, imitation term fading to 0 at 60 M steps | 42 GPU-h, 178 M steps | -0.26 to -0.37 while held, then -0.64 to -0.76 | worse |
| 12e A, B, C, D | student, no imitation term | 1.35 GPU-h each, 3.8-4.9 M steps | -0.53, -0.51, -0.51, -0.56; each loses to the student by 0.26 to 0.38 (+- 0.08 to 0.10) | worse |
| 12f (imitation only) | one network per seat | 0.5 GPU-h | -0.474 +- 0.054 | worse than shared |

### Hong Kong mahjong (yardstick: points a hand against three `value` bots; from 16 Nov against three `stage` bots)

| Run | Start and its strength | Size | End | Verdict |
|---|---|---|---|---|
| `ppo` (18 Oct) | imitation of `value`, -5.6 +- 1.8 | 90 min, 316k hands | -0.1 +- 2.9 | **improved** to level with its teacher |
| `ppo0` (18 Oct) | same | 40 min, 143k hands | -0.2 +- 2.6 | improved |
| `dmc` (18 Oct) | same | 90 min, 131k hands | -11.5 +- 1.5 | collapsed (92 % draws) |
| `dmc0` | random | 30 min | -11.5 +- 1.4 | never learned |
| PPO on the CNN (1 Nov) | imitation of the search bot, +0.77 +- 1.17 | 60 min, 374k hands | +0.34 +- 2.77 | level |
| 48 h PPO (queued 1 Nov) | same | | no result in the docs | unknown |
| Race-bot generations | rule bot -9.63 | 4 generations | -9.37 | level |
| Exploiter runs (3) | random, against frozen bots | 2 to 10 h, 12k-455k hands | -10.9 to -12.2 | never learned |
| Self-play network (17 Nov) | imitation of `stage`, -0.11 +- 0.71 vs three `stage` | 10 h, 1.29 M hands | **-0.09 +- 0.22** (40,000 deals) | level |

### Big Two (yardstick: points a game against the search bot ISMCTS-1000)

| Run | Start and its strength | Size | End | Verdict |
|---|---|---|---|---|
| DMC self-play, 90 min | random | 811k games | -2.22 | improved |
| DMC self-play, +10 h * | the 90-min network, -2.22 | 5.6 M games | **-0.17 +- 0.82** (later -0.17 +- 0.37) | **improved**; ties the search bot, does not beat it |
| PPO self-play, +10 h | -2.76 | 2.7 M games | -1.42 +- 0.69 | improved |
| Imitation network, then DMC self-play * | imitation of the search bot, -0.54 | 60 min, 647k games | -1.97; lost to its own start by 1.51 | **worse** |
| Training against three frozen copies of one bot * | imitation network; -1.40 vs ISMCTS-1000, -1.82 vs `expert` | 50k to 28 M games | +1.65 vs ISMCTS-1000 (gain +3.05 +- 1.03); +0.01 vs `expert` (gain +1.82 +- 1.01) | **improved from a strong start**, against fixed opponents |
| From scratch against `expert` | random | 10 h, 98k games | -2.57 | far below |

DMC (Deep Monte Carlo) is a simpler method than PPO: the network learns the average final score
after each move.

### Thirteen Cards

No network was trained. The shipped bot's tables were built by a self-play loop that plays against
the average of its own earlier versions. It improved from +1.54 to +0.012 on its weakness measure
(lower is better) in 1.08 billion hands, and the shipped `expert` beats the best hand-built field
(+0.019 [+0.013, +0.025] per opponent per hand). This is the one self-play success, and it uses
averaging, the method the research record says self-play needs in hidden-information games (Part C).

### The pattern

- Weak start, self-play: improved 5 times out of 5 with PPO or long DMC.
- Strong start, self-play: worse 3 times (Dou Dizhu 2c, 12e; Big Two fine-tune), level 2 times (mahjong).
- Strong start, fixed opponents: improved (Big Two, +1.8 to +3.0). This is against the opponent it
  trained on; it is not proof of general strength.
- No self-play network has beaten its project's strongest hand-built or search bot.

---

## Part C. What the published record says, and whether our runs show it

| # | Documented reason | Source | Do our runs show its mark? |
|---|---|---|---|
| 1 | **Critic not matched to the policy when fine-tuning starts.** A fresh or mismatched critic gives wrong signals and the copied policy falls at once. Fix: train the critic alone first, then warm the policy's learning rate up from zero. | PIRLNav (Ramrakhya et al., CVPR 2023): critic alone for 8 M steps, policy rate warmed 0 to 1.5e-5; naive fine-tuning drops early. | Partly. Dou Dizhu 12e falls from the first update with a critic error of 0.51 (checked). But 2c fell after 57 M steps of critic training, so it is not the whole cause. Mahjong: not applicable, the critic is unused. |
| 2 | **The policy forgets once nothing holds it to the start.** Fix: keep a pull to the copied policy, small and long-lasting, in place of the entropy bonus. | VPT (Baker et al., NeurIPS 2022): the pull replaces the entropy bonus; without it training collapses. AlphaStar (Vinyals et al., Nature 2019): a pull to human play throughout training. | Yes. Dou Dizhu 2c held while the term was on and fell within 4 M steps of zero (checked). A pull hides the problem; it does not show that learning works (mahjong M1). |
| 3 | **Steps too large for continued training.** | Suphx (Li et al., 2020): continued training needed the learning rate cut to a tenth and extreme samples rejected, or it was unstable and did not improve. PIRLNav: 1.5e-5. | Likely. Both trainers use 1e-4 with no warm-up; Dou Dizhu moves 11 % of move probabilities by more than 20 % per update from the first update (checked). Not tested at a lower rate. |
| 4 | **Too little data per update for a high-variance game.** | OpenAI Five (Berner et al., 2019): batches of the order of a million time steps. Suphx: 1.5 M games per agent on 44 GPUs. PerfectDou: 880 CPU cores, 8 GPUs, 2.5e9 steps. "Deep RL that Matters" (Henderson et al., AAAI 2018) on variance. | Yes for mahjong: noise 53 points a hand, 1,024 hands per update, training score flat over 1.29 M hands (checked). Likely for Dou Dizhu: about 5,100 seat-games per update. |
| 5 | **Plain self-play does not converge in hidden-information games.** Each side keeps adapting to the other's latest habits; play can cycle or settle on something exploitable. Fix: play against an average of past selves, or use regret-based methods or search. | NFSP (Heinrich and Silver, 2016); Deep CFR (Brown et al., ICML 2019); ReBeL (Brown et al., NeurIPS 2020); Student of Games (Schmid et al., Science Advances 2023); Pluribus (Brown and Sandholm, Science 2019). | Not the main cause here. Dou Dizhu got worse against fixed opponents inside training (checked), and lost to its own start head to head. Thirteen Cards, the one success, uses averaging. |
| 6 | **Forgetting and cycling without an opponent pool; beating weak copies of oneself.** | AlphaStar league and exploiters (Vinyals et al. 2019); OpenAI Five (20 % of games against past versions); "Open-ended Learning in Symmetric Zero-sum Games" (Balduzzi et al., ICML 2019). | Mahjong has a pool. Dou Dizhu has none, only the two outside programs. The rising landlord win rate in self-play fits this, but it has a dull explanation too (bidding improves) and run D speaks against it as the main cause. |
| 7 | **Reward shaping changes what is best.** | Ng, Harada and Russell, ICML 1999; PerfectDou removes its shaping in a final stage. | Present in the Dou Dizhu code (F2) and in mahjong's squashed reward (M3). Cleared as the cause of the 12e fall by run C. |
| 8 | **Entropy bonus set wrong.** Too small: training converges at once and self-play adds nothing. Too large: unstable. | Suphx adjusts it automatically toward a target. | Cleared for Dou Dizhu by run B (entropy 0 fell alike). Mahjong: entropy rose under a tiny bonus, so the rise is drift, not the bonus. |
| 9 | **Noisy evaluation hides real change.** | Henderson et al. 2018; "Real World Games Look Like Spinning Tops" (Czarnecki et al., NeurIPS 2020). | Yes for mahjong: +- 2.0 per check. No for Dou Dizhu: the falls are five times the range. |
| 10 | **Small code details decide whether PPO works.** | "Implementation Matters in Deep Policy Gradients" (Engstrom et al., ICLR 2020); "What Matters in On-Policy RL" (Andrychowicz et al., ICLR 2021); "The 37 Implementation Details of PPO" (Huang et al., ICLR Blog Track 2022). | Checked item by item in Part A. The one that bites is the common scale across two games (F1). |
| 11 | **Scale.** Published wins used 100 to 1,000 times our compute. | PerfectDou: 2.5e9 steps on 880 cores; its plain-PPO version scored -0.709 against the strong DouZero after 1e9 steps. DouZero (Zha et al., ICML 2021): 48 cores and 4 GPUs for 10 days. Suphx: 44 GPUs. | Yes. Our 2a-r2 (-0.72 at 6e8 steps) sits where PerfectDou's own plain-PPO line ended (-0.709). |
| 12 | **Cooperating seats are hard for PPO.** Fewer passes over the data and large batches help. | "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (Yu et al., NeurIPS 2022). | Fits: in 2c the peasants lost most (agreement 0.82 to 0.57). Each peasant's stake is half the landlord's, so its signal is a quarter of the weight per position. |

Links:
- PerfectDou: https://arxiv.org/abs/2203.16406
- DouZero: https://arxiv.org/abs/2106.06135
- Suphx: https://arxiv.org/abs/2003.13590
- AlphaStar: https://www.nature.com/articles/s41586-019-1724-z
- OpenAI Five: https://arxiv.org/abs/1912.06680
- NFSP: https://arxiv.org/abs/1603.01121
- Deep CFR: https://arxiv.org/abs/1811.00164
- ReBeL: https://arxiv.org/abs/2007.13544
- Student of Games: https://arxiv.org/abs/2112.03178
- Pluribus: https://www.science.org/doi/10.1126/science.aay2400
- VPT: https://arxiv.org/abs/2206.11795
- PIRLNav: https://arxiv.org/abs/2301.07302
- Ng, Harada, Russell (shaping): https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Implementation Matters: https://arxiv.org/abs/2005.12729
- What Matters in On-Policy RL: https://arxiv.org/abs/2006.05990
- 37 Implementation Details of PPO: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Deep RL that Matters: https://arxiv.org/abs/1709.06560
- Open-ended Learning in Symmetric Zero-sum Games: https://arxiv.org/abs/1901.08106
- Real World Games Look Like Spinning Tops: https://arxiv.org/abs/2004.09468
- PPO in Cooperative Multi-Agent Games: https://arxiv.org/abs/2103.01955

Read by me for this report: PerfectDou, Suphx and PIRLNav (numbers above are from the papers' text),
and search summaries for VPT, AlphaStar, DouZero and the PPO studies. The rest are cited from memory
of the papers; the OpenAI Five batch size is an order of magnitude, not an exact figure.

---

## Part D. Diagnosis and next steps

### Dou Dizhu: most likely causes

| Rank | Cause | Confidence | Would confirm | Would rule out |
|---|---|---|---|---|
| 1 | Updates too rough for a network that is already good: learning rate 1e-4, 400 nudges per update, no warm-up, noisy signal (F3, C3, C4) | about 40 % | a run at a tenth of the rate with one pass holds the student's level | it falls just as fast per update at the lower rate |
| 2 | Training mostly on the app's game, which swamps the research-rule game (F1) | about 30 % | the 12e networks beat the student under the app rules; a research-rules-only run does not fall | they lose under the app rules too |
| 3 | Critic not matched at the start (F4) | about 15 % | critic-only warm-up stops the early fall | it falls the same after the warm-up (2c already suggests so) |
| 4 | Self-play itself: beating weak copies, cycling (C5, C6) | about 10 % | a run against the outside programs only improves, self-play does not | a run against the outside programs only also falls |
| 5 | A code bug not found by reading | about 5 % | the rigged-reward test fails | it passes |

The first three can all be true together. All three say the same thing in plain words: the update is
not giving a good network a clean enough signal.

### Mahjong: most likely causes

| Rank | Cause | Confidence | Would confirm | Would rule out |
|---|---|---|---|---|
| 1 | Signal buried in noise: one number per hand for every decision, 53 points of noise, 1,024 hands per update (M2) | about 55 % | the hourly paired-difference table rises once noise is cut | still flat with 4 to 8 times the hands per update and a per-decision signal |
| 2 | The pull and the stop rule kept the run from showing anything (M1) | a fact; as the reason for "level", about 25 % | a run without the fading pull moves, up or down | |
| 3 | The start is near the best this network can do with what it sees | about 15 % | flat even with a clean signal and 10 M hands | |
| 4 | A code bug | about 5 % | | the Phase 3 run improved a weak start with this code family |

### Next experiments, ordered by information per GPU-hour

1. **App-rules head-to-head (no GPU, about 5 CPU minutes, small script change).** Play the end
   network of run 12e-A against the student under the app's rules, 400 paired deals, greedy, with each
   network doing its own bidding. `scripts\probe-12e-h2h.py` does this for the research rules today.
   - 12e-A wins clearly: PPO was learning, but the other game. Cause 2 is confirmed. Fix the game mix.
   - 12e-A loses: PPO made the network worse at everything. Causes 1, 3 or 5.
2. **Rigged-reward test (10 GPU minutes).** Described in A1. Passes: the loop is sound. Fails: a bug;
   stop and find it before any other run.
3. **Fixed opponents only, research rules only (two runs of 90 minutes, 3 GPU-hours).** From the
   student, every game against DouZero-ADP or PerfectDou, no app-rule games. Needs a small change:
   `make_slots` must allow zero self-play tables. Run P keeps run A's settings. Run Q adds: learning
   rate 1e-5, one pass per update, and the first 10 updates train the critic only. Read the score in
   the training games (accurate to +- 0.05 per sixth of the run) and the tracker.
   - P rises: the game mix was the cause (cause 2).
   - P falls, Q holds or rises: step size and critic (causes 1 and 3). Use Q's settings from then on.
   - Both fall: self-play is cleared for good and the trainer itself is suspect; go back to step 2
     with finer tests.
   - Either rises above -0.27 on the tracker: the first time learning by playing has lifted a strong
     start in this project. Then add self-play back, a share at a time.
4. **Mahjong with less noise (4 GPU-hours).** Same start; pull constant at 0.02, not fading; 4,096
   hands per update; one pass; learning rate 3e-5. Print the paired-difference table every hour
   (+- 0.15 per hour at this size). The start is -0.7.
   - Rises to -0.3 or better: the signal is there once noise is cut. Then build per-decision credit
     with the critic that is already in the network, and run long.
   - Flat: at this speed (129,000 hands an hour) self-play will not lift this network. Stop spending
     GPU on it.
5. **More imitation data for Dou Dizhu (about 1 CPU-hour and 2 GPU-hours).** The worker's own
   proposal: 4 M more PerfectDou positions, refit. The student is short of data: agreement 0.752, and
   the one-network-per-seat test (12f) lost because each seat had too few positions. This is the
   likeliest way to a better bot this week. It is not a test of self-play.

Also, for every future run: log the per-update policy movement (KL), the critic's explained share,
and the scores split by rule set and by seat. Each of these would have shortened this analysis.

### Can one PC beat a strong bot by self-play in these games?

- **Dou Dizhu: not above PerfectDou's level.** The student copies PerfectDou. PerfectDou itself
  gained only 0.116 points a deal between 1e9 and 2.5e9 steps on 880 CPU cores. Our PC does about
  1e8 steps a day. Closing the student's gap to its teacher (-0.30) is realistic, by imitation.
  Going past the teacher by self-play would take months of a recipe that first has to pass
  experiment 3.
- **Mahjong: possible in principle, not shown.** The `stage` bot is hand-built, so there is room
  above it, unlike a copy of a top program. But the signal is 53 points of noise against gains of
  tenths of a point. Suphx used 44 GPUs for two days per agent and gained modestly per stage.
  Experiment 4 tells whether it is worth more time.
- **Big Two: self-play from scratch reached a tie with the search bot in 10 hours.** That is the best
  self-play result we have. It never passed the search bot.

### Cheaper routes to a stronger bot

- **More imitation data** (experiment 5). Strength rose with data every time we tried it.
- **Search on top of the network.** In Big Two the strongest bot is search with an exact endgame, and
  the network inside a search beat the search bot it was trained on (+1.06 [+0.46, +1.67]). In mahjong
  the search bots are the strongest too.
- **Imitate a search that uses the network, then repeat** ("expert iteration"). Big Two's rounds
  raised agreement from 68.5 % to 76.3 %. Each round is supervised learning, which is stable on one GPU.
- **Train against fixed strong opponents.** It worked in Big Two (+1.8 to +3.0). Its risk: the bot
  learns that opponent's habits, not the game. Use several opponents and test against one it did not
  train on.
- **Averaging self-play for small table games.** It is what made Thirteen Cards work.
