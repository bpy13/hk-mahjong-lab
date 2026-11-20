# A risk network in place of the danger tables (plan Task 15, 2026-11-19)

Plan: `docs/plans/2026-11-16-self-play-network.md`, Task 15. Brian, 2026-11-19: "no I mean the
whole training seems to be wrong to me. The probability you feed is not the probability of
losing with that tile"; "I think the network should be a network, not a lookup table. given all
the conditions (the targeted tile you are trying to hit + its unseen copies + opponent discards
and their relation to his flush)".

Nothing becomes the default: `stage` plays exactly as before. Task 14 takes this network as the
risk input of the learned push-or-fold choice if the verdict line below says yes on both points.

## What was built

* `crates/search/src/risk_net.rs`: the `risk` input block ([`encode_risk`], 534 floats: per
  opponent and tile kind, the kind's relation to the opponent's exposed sets (in the suit of a
  one-suit shape or an honour; another suit), whether the opponent itself discarded the kind,
  whether the kind is three apart from one of its discards (筋), the passed-win lock on the kind
  (rules §4.4, `exact_wait::refused`); per opponent its concealed count as one-hot over 1, 4, 7,
  10, 13, exposed melds ÷ 4, one-suit shape, a meld that already carries faan). The true answer
  [`labels`] from the opponents' real hands: for each opponent and kind, does it win on the kind by
  discard now (complete with its melds, the faan minimum under the engine's own scoring, not
  locked), and the points per share. [`RiskNet`]: the loaded network (MJNN v2, output slices
  `win` and `pay`), answering for the player to act with the chance per opponent and kind and
  the points per share (`8 × 2^pay`, within the rules' range); kinds with no unseen copy get 0.
* `crates/search/src/net.rs`: the input-slice name `risk` (additive; every earlier `.mjnn` loads
  and gives the same output, pinned by the existing tests).
* `crates/search/src/danger.rs`: a third source behind `danger::with_risk(Some(net), …)`, off by
  default. With a network set, `Danger::estimate` and `Danger::from_belief` read every opponent's
  per-tile chance and points from it (outside the calibration curve) and combine them as before
  (`1 − Π (1 − p_o)`, `Σ p_o × points × shares`). Where Task 13's exact counting gives exactly 0
  (no possible hand wins on the kind) or exactly 1 (one concealed tile, one kind left) that
  figure wins over the network. The network's `features` input holds the table danger plane, so
  the network runs with no network set (no recursion; the plane is the table's, as in training).
  `Danger::is_risk(seat)` says which opponents it covers. `with_exact` on top replaces the exact
  opponents fully, as today. Every existing public function, `with_exact`, `split` and `is_exact`
  are unchanged; with no network set every figure is bit for bit as before (tests).
* `crates/search/examples/risk_log.rs`: the data (below) and `--time`.
* `python/mahjong/risk/`: `data.py` (the rows), `model.py` (the network, loss, export),
  `fit.py` (training), `check.py` (checks (a) to (d), the Markdown below); tests
  `python/tests/test_risk.py`.
* `experiments/stage/risk.ps1`: the three steps as a job script (data, fit, check).

Tests: the labeller says yes exactly when the engine lets that opponent declare a win on the tile
really thrown, over 9,000 opponent-decision pairs of `stage` play (`labels_match_the_engines_win_check`);
the `risk` block is public information (re-dealing the hidden tiles leaves it unchanged) and marks
the lock and the flush relation as the facts say; a small network loads, answers 0 on dead kinds
and within the points range, and a belief model is refused; with a network set the figures equal
the network's recombined by hand except where exact counting proves 0 or 1, and with none set
they equal the tables' (`risk_network_feeds_the_figures_and_exact_counting_overrides_it`); Rust
and Python give the same output on 1,000 real positions (largest difference under 1e-4,
`test_export_round_trips_and_matches_rust`).

## The existing belief model first

`models/belief.mjnn` (`docs/belief.md`) predicts per opponent `tenpai` (ready now) and `waits`
(P(a discard of the kind wins | ready), independent sigmoids), so `Danger::from_belief` gives
`tenpai × wait` per kind: the same quantity this network predicts, from the 933 `features` alone
(no discard order, no passed-tile block). It is scored below as source `belief`. I started fresh
rather than building on it: its heads are shaped for the EV bot (counts, faan buckets, outcome),
its input lacks the discard history and the lock, and a fresh fit on the per-kind true answer is
the plan's own definition of the target. Its figures are a useful yardstick: it is the only
existing source without a calibration curve on top.

## Data

`risk_log --deals 30000 --chunk 1000 --seed 15 --belief models/belief.mjnn` (8 threads, 19
minutes beside other jobs): 30,000 deals of four `stage` bots, 1,856,776 discard decisions, one
row each with the input, the true answer for all 34 kinds × 3 opponents, and the three sources'
figures for every kind. Chunks of 1,000 whole deals; chunks 27 to 29 (3,000 deals, 186,000
decisions) are held out, never seen in training. Rows are 4,288 bytes (7.5 GB), in
`runs/risk/data` of this checkout (not in git).

Base rates: 1.20 % of (decision, kind) pairs with an unseen copy would deal in to at least one
opponent; 0.40 % of (decision, opponent, kind) triples. The tile really thrown dealt in 1.1 % of
the time (the bots pick safe tiles).

## Model and fit

MLP 4,927 → 768 → ReLU → 384 → ReLU → 204 (4,158,540 parameters): input `features` (933) +
`history` (3,460) + `risk` (534); outputs `win` logits (3 × 34) and `pay` (3 × 34). Loss: binary
cross-entropy of `win` over the kinds with an unseen copy (mean per kind) + 0.25 × the squared
error of `pay` against `log2(points ÷ 8)` on the kinds that win. The output bias starts at the
base rate (logit −5.5) and at `pay` 0.7. Adam, learning rate 1e-3 with cosine decay, batch
1,024, chunks visited in a fresh random order each epoch, rows shuffled inside the chunk.

Fitted on the CPU directly (orchestrator's call, 2026-11-19: no gpu lane wait; 8 threads beside
slot 2's matches and gpu job 252), `python -m mahjong.risk.fit --data runs/risk/data --out
runs/risk/fit --holdout-chunks 27 28 29 --epochs 6 --minutes 100 --threads 8`. The first epoch took
30 minutes on the loaded PC, the later ones 7 to 9; all 6 epochs ran (69 minutes). The held-out
win loss was still falling at the end (0.02102 → 0.02005; the training loss 0.01955 is close to
it, so this is not overfitting but a fit that is not finished), while the `pay` head overfits
(train 0.14, held-out 0.76). Epoch 4 had the best held-out total and is the exported network. Held-out loss per epoch (`runs/risk/fit/train.log`):

| epoch | train win BCE | held-out win BCE | train pay SE | held-out pay SE | held-out total | minutes | |
|---|---|---|---|---|---|---|---|
| 0 | 0.02145 | 0.02102 | 0.564 | 0.773 | 0.21422 | 30.3 | kept as best
| 1 | 0.02042 | 0.02058 | 0.348 | 0.767 | 0.21244 | 8.8 | kept as best
| 2 | 0.02009 | 0.02035 | 0.242 | 0.776 | 0.21445 | 8.5 |
| 3 | 0.01982 | 0.02031 | 0.174 | 0.770 | 0.21277 | 7.1 |
| 4 | 0.01964 | 0.02012 | 0.137 | 0.758 | 0.20956 | 7.0 | kept as best
| 5 | 0.01955 | 0.02005 | 0.140 | 0.761 | 0.21037 | 7.2 |

Exported: `runs/risk/fit/risk.mjnn` = `models/risk.mjnn` in this checkout (fp32, 16.6 MB; not in
git, as every model).

## Checks on the held-out deals

`python -m mahjong.risk.check --data runs/risk/data --model runs/risk/fit/best.pt --chunks 27 28 29`
(`runs/risk/check.md`). Sources per tile, combined over the three opponents as `Danger::tile`
does: `table` = `Danger::from_belief` with the fitted tables through Task 11's calibration
(`giveup::danger_of`, as Task 13's check); `exact` = the same with Task 13's switch on (the exact
opponents outside the curve); `belief` = `Danger::from_belief` with `models/belief.mjnn` (no
curve); `network` = `1 − Π (1 − p_o)` from this network; `network+exact 0/1` = the network with
the exact 0 and 1 override, what `Danger` gives with the switch on. Observed = at least one
opponent really wins on the kind by discard now, from the real hands. Populations: `all` = every
kind with an unseen copy (the plan's "all 34 kinds are labelled"), `hand` = the kinds in the
thrower's hand (the real candidates), `thrown` = the kind really thrown (what the earlier danger
checks measured).

Held-out chunks [27, 28, 29]: 3,000 deals, 185,796 discard decisions, 5,843,360 (decision, alive kind) pairs, 1,657,949 in hand. Network `best.pt`, 4,158,540 parameters.

#### Population `all` (5,843,360 pairs, observed 1.214 %)

(a) Ten bins of predicted against observed (n / mean predicted / observed / observed ÷ predicted):

| predicted bin | table: n / pred / obs / obs÷pred | exact: n / pred / obs / obs÷pred | belief: n / pred / obs / obs÷pred | network: n / pred / obs / obs÷pred | network+exact 0/1: n / pred / obs / obs÷pred |
|---|---|---|---|---|---|
| 0–0.5 % | 3,520,019 / 0.18 % / 0.03 % / 0.18 | 3,544,565 / 0.17 % / 0.03 % / 0.19 | 4,451,623 / 0.09 % / 0.08 % / 0.97 | 3,963,190 / 0.07 % / 0.07 % / 1.09 | 3,984,917 / 0.07 % / 0.08 % / 1.10 |
| 0.5–1 % | 939,161 / 0.66 % / 0.40 % / 0.60 | 913,150 / 0.67 % / 0.43 % / 0.64 | 453,597 / 0.71 % / 1.00 % / 1.40 | 433,711 / 0.73 % / 0.82 % / 1.13 | 439,605 / 0.73 % / 0.83 % / 1.15 |
| 1–2 % | 464,981 / 1.44 % / 1.08 % / 0.75 | 498,221 / 1.44 % / 1.19 % / 0.83 | 367,558 / 1.42 % / 2.18 % / 1.54 | 485,218 / 1.45 % / 1.64 % / 1.13 | 484,285 / 1.45 % / 1.68 % / 1.16 |
| 2–3 % | 206,069 / 2.49 % / 2.14 % / 0.86 | 232,536 / 2.49 % / 2.39 % / 0.96 | 166,831 / 2.45 % / 3.92 % / 1.60 | 292,449 / 2.46 % / 2.91 % / 1.18 | 287,001 / 2.46 % / 3.03 % / 1.23 |
| 3–5 % | 275,288 / 3.95 % / 3.25 % / 0.82 | 302,256 / 3.93 % / 3.92 % / 1.00 | 161,679 / 3.86 % / 6.05 % / 1.57 | 321,778 / 3.87 % / 4.74 % / 1.22 | 312,503 / 3.87 % / 4.86 % / 1.25 |
| 5–7 % | 132,129 / 5.79 % / 5.27 % / 0.91 | 133,702 / 5.81 % / 6.82 % / 1.17 | 77,741 / 5.90 % / 8.66 % / 1.47 | 154,647 / 5.89 % / 7.07 % / 1.20 | 149,205 / 5.89 % / 7.22 % / 1.23 |
| 7–10 % | 99,254 / 8.42 % / 7.20 % / 0.85 | 89,973 / 8.30 % / 10.28 % / 1.24 | 61,711 / 8.32 % / 11.97 % / 1.44 | 104,101 / 8.27 % / 9.01 % / 1.09 | 100,753 / 8.27 % / 9.17 % / 1.11 |
| 10–15 % | 95,958 / 12.24 % / 10.82 % / 0.88 | 76,125 / 12.06 % / 13.95 % / 1.16 | 47,502 / 12.15 % / 16.15 % / 1.33 | 58,644 / 11.98 % / 12.25 % / 1.02 | 56,595 / 11.98 % / 12.55 % / 1.05 |
| 15–25 % | 64,016 / 19.17 % / 15.65 % / 0.82 | 39,725 / 19.03 % / 22.48 % / 1.18 | 34,296 / 19.02 % / 23.96 % / 1.26 | 24,484 / 18.42 % / 16.53 % / 0.90 | 23,534 / 18.42 % / 16.98 % / 0.92 |
| 25 % + | 46,485 / 36.99 % / 28.37 % / 0.77 | 13,107 / 31.29 % / 35.13 % / 1.12 | 20,822 / 35.91 % / 39.83 % / 1.11 | 5,138 / 31.96 % / 21.33 % / 0.67 | 4,962 / 31.94 % / 21.83 % / 0.68 |

(b) Telling single tiles apart (log loss, natural, lower is better; Brier beside it):

| source | n | mean predicted | observed | log loss | Brier |
|---|---|---|---|---|---|
| table | 5,843,360 | 1.579 % | 1.214 % | 0.04594 | 0.010878 |
| exact | 5,843,360 | 1.252 % | 1.214 % | 0.04620 | 0.010878 |
| belief | 5,843,360 | 0.892 % | 1.214 % | 0.04404 | 0.010615 |
| network | 5,843,360 | 1.086 % | 1.214 % | 0.04946 | 0.011375 |
| network+exact 0/1 | 5,843,360 | 1.060 % | 1.214 % | 0.04936 | 0.011363 |

(c) Confident calls:

| source | above 50 %: n / share / observed | below 1 %: n / share / observed |
|---|---|---|
| table | 6,616 / 0.11 % / 42.74 % | 4,459,180 / 76.31 % / 0.11 % |
| exact | 233 / 0.00 % / 48.07 % | 4,457,715 / 76.29 % / 0.11 % |
| belief | 2,177 / 0.04 % / 64.81 % | 4,905,220 / 83.95 % / 0.17 % |
| network | 147 / 0.00 % / 31.97 % | 4,396,901 / 75.25 % / 0.15 % |
| network+exact 0/1 | 142 / 0.00 % / 33.10 % | 4,424,522 / 75.72 % / 0.15 % |

#### Population `hand` (1,657,949 pairs, observed 1.275 %)

(a) Ten bins of predicted against observed (n / mean predicted / observed / observed ÷ predicted):

| predicted bin | table: n / pred / obs / obs÷pred | exact: n / pred / obs / obs÷pred | belief: n / pred / obs / obs÷pred | network: n / pred / obs / obs÷pred | network+exact 0/1: n / pred / obs / obs÷pred |
|---|---|---|---|---|---|
| 0–0.5 % | 1,089,242 / 0.14 % / 0.03 % / 0.23 | 1,098,075 / 0.14 % / 0.04 % / 0.25 | 1,280,027 / 0.08 % / 0.07 % / 0.96 | 1,193,909 / 0.06 % / 0.06 % / 0.98 | 1,198,778 / 0.06 % / 0.06 % / 0.99 |
| 0.5–1 % | 259,051 / 0.67 % / 0.49 % / 0.74 | 255,431 / 0.67 % / 0.53 % / 0.79 | 119,391 / 0.72 % / 0.97 % / 1.36 | 108,597 / 0.72 % / 0.84 % / 1.16 | 109,866 / 0.72 % / 0.85 % / 1.18 |
| 1–2 % | 88,558 / 1.45 % / 1.64 % / 1.13 | 95,555 / 1.45 % / 1.98 % / 1.36 | 98,998 / 1.42 % / 2.26 % / 1.59 | 117,254 / 1.45 % / 1.77 % / 1.23 | 116,753 / 1.45 % / 1.83 % / 1.27 |
| 2–3 % | 56,205 / 2.52 % / 2.71 % / 1.07 | 61,835 / 2.53 % / 2.93 % / 1.16 | 45,651 / 2.45 % / 4.32 % / 1.76 | 69,012 / 2.46 % / 3.25 % / 1.32 | 67,835 / 2.46 % / 3.39 % / 1.38 |
| 3–5 % | 65,075 / 3.79 % / 4.47 % / 1.18 | 68,968 / 3.77 % / 5.47 % / 1.45 | 45,330 / 3.86 % / 6.48 % / 1.68 | 78,286 / 3.88 % / 5.34 % / 1.37 | 76,251 / 3.88 % / 5.45 % / 1.40 |
| 5–7 % | 31,537 / 5.86 % / 7.14 % / 1.22 | 33,161 / 5.89 % / 8.70 % / 1.48 | 21,982 / 5.90 % / 9.41 % / 1.59 | 39,128 / 5.89 % / 8.37 % / 1.42 | 38,035 / 5.89 % / 8.47 % / 1.44 |
| 7–10 % | 23,692 / 8.45 % / 9.40 % / 1.11 | 20,801 / 8.32 % / 14.21 % / 1.71 | 17,709 / 8.32 % / 13.58 % / 1.63 | 27,242 / 8.27 % / 11.05 % / 1.34 | 26,572 / 8.27 % / 11.20 % / 1.35 |
| 10–15 % | 23,032 / 12.14 % / 14.83 % / 1.22 | 15,641 / 11.89 % / 20.63 % / 1.73 | 13,738 / 12.14 % / 17.59 % / 1.45 | 16,199 / 12.03 % / 16.64 % / 1.38 | 15,787 / 12.02 % / 16.89 % / 1.41 |
| 15–25 % | 14,572 / 18.95 % / 22.46 % / 1.19 | 7,490 / 18.40 % / 31.38 % / 1.71 | 9,499 / 18.96 % / 26.93 % / 1.42 | 6,828 / 18.39 % / 22.79 % / 1.24 | 6,623 / 18.39 % / 23.31 % / 1.27 |
| 25 % + | 6,985 / 32.59 % / 35.19 % / 1.08 | 992 / 30.50 % / 50.71 % / 1.66 | 5,624 / 36.46 % / 43.23 % / 1.19 | 1,494 / 31.92 % / 31.79 % / 1.00 | 1,449 / 32.00 % / 32.51 % / 1.02 |

(b) Telling single tiles apart (log loss, natural, lower is better; Brier beside it):

| source | n | mean predicted | observed | log loss | Brier |
|---|---|---|---|---|---|
| table | 1,657,949 | 1.214 % | 1.275 % | 0.04551 | 0.011145 |
| exact | 1,657,949 | 0.968 % | 1.275 % | 0.04677 | 0.011385 |
| belief | 1,657,949 | 0.869 % | 1.275 % | 0.04474 | 0.011041 |
| network | 1,657,949 | 0.976 % | 1.275 % | 0.04842 | 0.011660 |
| network+exact 0/1 | 1,657,949 | 0.957 % | 1.275 % | 0.04839 | 0.011654 |

(c) Confident calls:

| source | above 50 %: n / share / observed | below 1 %: n / share / observed |
|---|---|---|
| table | 338 / 0.02 % / 49.41 % | 1,348,293 / 81.32 % / 0.12 % |
| exact | 14 / 0.00 % / 78.57 % | 1,353,506 / 81.64 % / 0.13 % |
| belief | 699 / 0.04 % / 62.09 % | 1,399,418 / 84.41 % / 0.15 % |
| network | 48 / 0.00 % / 58.33 % | 1,302,506 / 78.56 % / 0.12 % |
| network+exact 0/1 | 48 / 0.00 % / 60.42 % | 1,308,644 / 78.93 % / 0.13 % |

#### Population `thrown` (185,796 pairs, observed 1.146 %)

(a) Ten bins of predicted against observed (n / mean predicted / observed / observed ÷ predicted):

| predicted bin | table: n / pred / obs / obs÷pred | exact: n / pred / obs / obs÷pred | belief: n / pred / obs / obs÷pred | network: n / pred / obs / obs÷pred | network+exact 0/1: n / pred / obs / obs÷pred |
|---|---|---|---|---|---|
| 0–0.5 % | 126,579 / 0.12 % / 0.17 % / 1.33 | 126,363 / 0.12 % / 0.11 % / 0.87 | 148,282 / 0.07 % / 0.18 % / 2.66 | 133,957 / 0.05 % / 0.18 % / 3.25 | 134,540 / 0.05 % / 0.18 % / 3.24 |
| 0.5–1 % | 27,250 / 0.66 % / 0.46 % / 0.70 | 26,535 / 0.66 % / 0.54 % / 0.81 | 11,956 / 0.72 % / 0.90 % / 1.26 | 11,691 / 0.72 % / 0.75 % / 1.04 | 11,783 / 0.72 % / 0.75 % / 1.03 |
| 1–2 % | 9,583 / 1.45 % / 1.37 % / 0.94 | 10,651 / 1.46 % / 1.50 % / 1.03 | 10,238 / 1.43 % / 2.44 % / 1.71 | 12,803 / 1.45 % / 1.63 % / 1.13 | 12,792 / 1.45 % / 1.70 % / 1.17 |
| 2–3 % | 5,781 / 2.51 % / 2.87 % / 1.14 | 6,593 / 2.51 % / 2.90 % / 1.15 | 4,495 / 2.45 % / 4.29 % / 1.75 | 8,042 / 2.46 % / 3.08 % / 1.25 | 7,902 / 2.46 % / 3.19 % / 1.30 |
| 3–5 % | 6,678 / 3.81 % / 4.25 % / 1.12 | 7,706 / 3.82 % / 5.39 % / 1.41 | 4,415 / 3.86 % / 6.05 % / 1.57 | 9,089 / 3.88 % / 4.49 % / 1.16 | 8,878 / 3.88 % / 4.61 % / 1.19 |
| 5–7 % | 3,354 / 5.88 % / 6.62 % / 1.13 | 3,447 / 5.90 % / 7.80 % / 1.32 | 2,155 / 5.93 % / 9.33 % / 1.57 | 4,462 / 5.89 % / 7.01 % / 1.19 | 4,315 / 5.89 % / 7.00 % / 1.19 |
| 7–10 % | 2,334 / 8.44 % / 8.83 % / 1.05 | 2,120 / 8.29 % / 13.07 % / 1.58 | 1,703 / 8.33 % / 13.98 % / 1.68 | 3,073 / 8.24 % / 8.33 % / 1.01 | 3,002 / 8.24 % / 8.53 % / 1.03 |
| 10–15 % | 2,188 / 12.19 % / 13.44 % / 1.10 | 1,572 / 11.88 % / 18.13 % / 1.53 | 1,269 / 12.15 % / 16.31 % / 1.34 | 1,793 / 11.96 % / 12.38 % / 1.03 | 1,730 / 11.96 % / 12.77 % / 1.07 |
| 15–25 % | 1,347 / 18.97 % / 18.49 % / 0.97 | 707 / 18.34 % / 27.16 % / 1.48 | 797 / 18.92 % / 25.09 % / 1.33 | 738 / 18.40 % / 16.26 % / 0.88 | 713 / 18.39 % / 16.41 % / 0.89 |
| 25 % + | 702 / 32.62 % / 34.62 % / 1.06 | 102 / 29.87 % / 57.84 % / 1.94 | 486 / 36.60 % / 41.56 % / 1.14 | 148 / 33.46 % / 20.27 % / 0.61 | 141 / 33.38 % / 20.57 % / 0.62 |

(b) Telling single tiles apart (log loss, natural, lower is better; Brier beside it):

| source | n | mean predicted | observed | log loss | Brier |
|---|---|---|---|---|---|
| table | 185,796 | 1.088 % | 1.146 % | 0.05134 | 0.010238 |
| exact | 185,796 | 0.901 % | 1.146 % | 0.04825 | 0.010369 |
| belief | 185,796 | 0.734 % | 1.146 % | 0.05053 | 0.010170 |
| network | 185,796 | 0.974 % | 1.146 % | 0.05496 | 0.010786 |
| network+exact 0/1 | 185,796 | 0.953 % | 1.146 % | 0.05490 | 0.010776 |

(c) Confident calls:

| source | above 50 %: n / share / observed | below 1 %: n / share / observed |
|---|---|---|
| table | 39 / 0.02 % / 58.97 % | 153,829 / 82.79 % / 0.22 % |
| exact | 0 / 0.00 % / nan % | 152,898 / 82.29 % / 0.18 % |
| belief | 61 / 0.03 % / 62.30 % | 160,238 / 86.24 % / 0.23 % |
| network | 7 / 0.00 % / 28.57 % | 145,648 / 78.39 % / 0.22 % |
| network+exact 0/1 | 7 / 0.00 % / 28.57 % | 146,323 / 78.75 % / 0.22 % |

#### Points it would cost (expected loss per discard, population `all`)

| source | mean predicted | mean true | ratio | mean abs error | corr |
|---|---|---|---|---|---|
| table | 0.4147 | 0.3220 | 1.29 | 0.6451 | 0.265 |
| exact | 0.3212 | 0.3220 | 1.00 | 0.5798 | 0.288 |
| network | 0.2490 | 0.3220 | 0.77 | 0.5343 | 0.155 |
| network+exact 0/1 | 0.2440 | 0.3220 | 0.76 | 0.5295 | 0.156 |
| belief | 0.2315 | 0.3220 | 0.72 | 0.4904 | 0.280 |

On the 75,370 (decision, opponent, kind) triples that really win: true points per share 25.9, network 21.0 (mean abs error 12.5).

#### (d) Per opponent, by its concealed tiles (every (decision, opponent, alive kind) triple)

| concealed | triples | observed | table (raw): pred / log loss / Brier | exact: pred / log loss / Brier | network: pred / log loss / Brier | network+exact 0/1: pred / log loss / Brier | belief (raw): pred / log loss / Brier |
|---|---|---|---|---|---|---|---|
| 1 | 72,823 | 3.896 % | 4.391 % / 0.12072 / 0.033141 | 3.916 % / 0.12447 / 0.033408 | 2.637 % / 0.14134 / 0.036037 | 2.441 % / 0.13919 / 0.035925 | 3.376 % / 0.10840 / 0.030483 |
| 4 | 733,755 | 3.197 % | 1.956 % / 0.11435 / 0.029286 | 1.582 % / 0.11632 / 0.029466 | 1.947 % / 0.13321 / 0.030440 | 1.751 % / 0.13124 / 0.030392 | 1.983 % / 0.11341 / 0.028627 |
| 7 | 2,139,606 | 1.356 % | 0.804 % / 0.05945 / 0.013001 | 0.804 % / 0.05945 / 0.013001 | 1.096 % / 0.06424 / 0.013174 | 1.096 % / 0.06424 / 0.013174 | 0.867 % / 0.05831 / 0.012863 |
| 10 | 4,051,584 | 0.340 % | 0.253 % / 0.01849 / 0.003332 | 0.253 % / 0.01849 / 0.003332 | 0.458 % / 0.01983 / 0.003378 | 0.458 % / 0.01983 / 0.003378 | 0.300 % / 0.01769 / 0.003313 |
| 13 | 10,532,312 | 0.036 % | 0.089 % / 0.00276 / 0.000362 | 0.089 % / 0.00276 / 0.000362 | 0.060 % / 0.00260 / 0.000363 | 0.060 % / 0.00260 / 0.000363 | 0.058 % / 0.00237 / 0.000360 |

#### Where exact counting proves 0 or 1, what the network says

Exactly 0: 274,478 triples (observed 0.000 %); network mean 0.575 %, median 0.291 %, max 57.14 %; share within 0.05 of the proof: 99.1 %.

Worst cases (deal, seat, wall left, opponent's concealed, kind index, network %):

- deal 29434, seat 3, wall 13, opponent +1 with 1 concealed, kind 33: 57.1 %
- deal 29434, seat 3, wall 18, opponent +1 with 1 concealed, kind 33: 55.5 %
- deal 29434, seat 2, wall 19, opponent +2 with 1 concealed, kind 33: 48.4 %
- deal 29434, seat 3, wall 22, opponent +1 with 1 concealed, kind 32: 43.5 %
- deal 28363, seat 1, wall 22, opponent +1 with 4 concealed, kind 8: 41.8 %
- deal 28614, seat 3, wall 24, opponent +1 with 4 concealed, kind 20: 40.8 %
- deal 28971, seat 2, wall 35, opponent +2 with 4 concealed, kind 26: 40.6 %
- deal 29434, seat 2, wall 23, opponent +2 with 1 concealed, kind 32: 40.0 %
Exactly 1: 3 triples (observed 100.000 %); network mean 7.865 %, median 7.293 %, min 4.56 %; share within 0.05 of the proof: 0.0 %.

Worst cases (deal, seat, wall left, opponent's concealed, kind index, network %):

- deal 28217, seat 0, wall 0, opponent +2 with 1 concealed, kind 29: 4.6 %
- deal 28217, seat 3, wall 1, opponent +3 with 1 concealed, kind 29: 7.3 %
- deal 28568, seat 1, wall 3, opponent +3 with 1 concealed, kind 33: 11.7 %

## Timing

`risk_log --time models/risk.mjnn 200` (one thread, 12008 discard decisions, beside
the other jobs):

| what | p50 | p99 | mean |
|---|---|---|---|
| network alone | 5.341 ms | 41.346 ms | 7.804 ms |
| Danger::from_belief, network set | 6.784 ms | 48.529 ms | 9.089 ms |
| the same with exact counting on too | 7.270 ms | 78.027 ms | 12.014 ms |
| Danger::from_belief, tables (today) | 0.010 ms | 0.046 ms | 0.067 ms |

The discarded tile's danger moved by more than 0.01 against the table on 1948 of 12008 decisions.

The network is a 4.2 M-parameter dense pass per decision (`features` + `history` + `risk`
encoding included). The app's hint runs the danger once per decision, so this adds about
5.341 ms at p50; the int8 export of `cnn.rs` does not apply to a dense MJNN v2 file, so a
smaller hidden layer is the lever if it must be quicker.

## Verdict

**Is the network right on average (every bin with 500 or more pairs within 0.8 to 1.25 of
observed, population `all`): no.** Nine of its ten bins are (0.90 to 1.22), the top bin is not:
where it says 32 % the observed rate is 21 % (5,138 pairs, ratio 0.67). For comparison the table
is within in six bins of ten (it reads 0.18 to 0.91 of observed: too high in every bin on this
population, which Task 11's curve was never fitted on), exact counting in eight (0.83 to 1.24
from 1 % up), the belief model in two (1.26 to 1.60 from 0.5 % to 25 %: too low).

**Does it tell single tiles apart better than the table (log loss, population `all`): no**
(0.04946 against 0.04594). **Better than the exact figure: no** (0.04620). The existing belief
model is the best source on every population and in every concealed-count group (0.04404 on
`all`), the network the worst; the Brier score orders them the same way. On the tile really
thrown the exact figure is best (0.04825), the network again last (0.05496).

So Task 14 takes the table as the risk input (with Task 13's exact figure where it is 0 or 1),
not this network, and says so.

What the network does get right: it is the only source whose figures are about right across the
whole range (ratios 0.90 to 1.22 up to 25 %; the table and the exact figure read too high on
`all` and too low on `hand` and `thrown`, by up to a factor 1.7); on the 274,478 triples that
exact counting proves 0 it says 0.58 % on average and is within 0.05 on 99.1 % of them, with the
worst cases at 40 to 57 % (all against a one-suit opponent with 1 or 4 concealed tiles on a
honour or terminal kind, late in the hand: deal 29434 of seed 15). On the 3 triples proved 1 it
says 5 to 12 %: it has not learnt the one-tile wait at all (3 cases in 186,000 decisions is too
few to learn from, which is why the override is there). Its expected loss runs 0.77 of the true
mean (the belief model 0.72, the table 1.29, exact 1.00) and its points per winning kind average
21.0 against 25.9 true.

Why it loses on log loss while being better calibrated: it ranks worse. Its 0 to 0.5 % bin holds
3.96 M pairs at an observed 0.07 % where the belief model's holds 4.45 M at 0.08 %: the network
leaves half a million more safe tiles in higher bins. The fit is short of converged (the
held-out loss fell in every epoch; one cosine cycle of 6 epochs at learning rate 1e-3) and the
4,927-float input with one-hot discards is a harder function to fit than the belief model's 933
public features. A longer fit is cheap on the graphics card: queued as gpu job 262 `risk-fit-gpu`
(30 epochs, learning rate 5e-4, after job 252) and cpu job 263 `risk-check-gpu` (the same checks,
output `runs/risk/fit-gpu/check.md` in this checkout); check with `jobq.ps1 status` and
`jobq.ps1 log 262`. If that fit beats the belief model's 0.04404, the verdict changes and the
orchestrator should say so before Task 14 refits. Until then the answer is no on both points.

Left out: the copy of `models/risk.mjnn` into the worker clone's `models/` folder was refused by
the permission check (a write outside this checkout); the file is `models/risk.mjnn` and
`runs/risk/fit/risk.mjnn` here. Timing suffered from the other jobs on the PC (slot 2's matches,
gpu job 252): the one-thread p99 of 41 ms is load, the p50 of 5.3 ms is the network's own cost.

No paid API calls.

## 300,000 deals (plan Task 16)

Brian, 2026-11-19, on the suggestion to train on far more deals: "first go ahead with the 300k
deals to see if it's effective". Task 15's network scored 0.04839 on held-out log loss after its
longer fit (job 262, 30 epochs at learning rate 5e-4; population `all`; 0.04946 after the
first fit above), behind the table (0.04594), the exact figure (0.04620) and the belief
model (0.04404), with 4.16 M parameters for 1.86 M decisions and a held-out loss still falling:
the question is whether ten times the data moves it toward or past the belief model.

### What was built

* `risk_log` writes row layout 2 (`"layout": 2` in `meta.json`): the six per-tile check
  columns as float16 and the `risk` block packed two values to a byte, 3,613 bytes a row
  against 4,288 (16 % smaller; about 67 GB for 300,000 deals instead of 80). The `features`
  block stays float16: two of its planes (the heuristic danger, the acceptance counts) are not
  on a coarse grid, so bytes would have changed the input the network sees. Checked on seed
  15's first 10 deals against Task 15's data: inputs, labels and the per-opponent figures are
  identical, the check columns within float16 rounding (largest relative difference 0.05 %).
  `python/mahjong/risk/data.py` reads both layouts (Task 15's 7.4 GB still load).
* `fit.py`: `--train-chunks lo-hi` (the data size), `--select win` (the best epoch by the
  held-out `win` loss alone, not `win + 0.25 × pay`: the `pay` head's noise picked epoch 8 of
  30 in job 262 while `win` kept falling to epoch 12), `--patience N` (stop after N epochs
  without a gain; the cosine schedule still spans `--epochs`); a resumed fit that had already
  stopped does nothing (the queue restarts a job under its old id after a Windows restart).
* `python -m mahjong.risk.summary`: the one table over several fits, streaming chunk by chunk
  (the memory is one chunk's), with Task 15's ten-bin verdict, log loss, Brier, the above-50 %
  and below-1 % calls, and a second table with each fit's size, epochs and minutes.
* `experiments/stage/risk300k.ps1`: the steps as queue jobs (data, fit, big, check, summary).

### Data

cpu job 265: `risk_log --deals 300000 --chunk 1000 --seed 16 --belief models/belief.mjnn`
(8 threads), 300 chunks of 1,000 whole deals under `runs/risk/data-300k` of the workers
clone (not in git); chunks 290 to 299 (10,000 deals) are held out and never seen in fitting.
C: had 1,150 GB free at the start. Wall time 38.0 min (7.6 s a chunk; Task 15's 30,000 deals
took 19 min, so the queue slot's 8 threads paid for ten times the deals in twice the time).
18,609,334 rows of 3,613 bytes: 67,235,535,142 bytes (62.6 GiB, 300 chunk files of about
220 MB each and `meta.json`). The folder stays; deleting it is Brian's call.

### Fits

Task 15's network and input unchanged (4,927 → 768 → 384 → 204; `features` + `history` +
`risk`), the same recipe three times on the gpu: Adam, learning rate 5e-4 with cosine decay over
12 epochs, batch 1,024, chunks in a fresh random order each epoch, the best epoch kept by the
held-out `win` loss (200,000 held-out rows, about 3,200 deals, for the per-epoch figure), stop
after 3 epochs without a gain. Gpu jobs 266 (chunks 0–29, 30,000 deals), 267 (0–99, 100,000),
268 (0–289, 290,000); job 269 fits a larger network (4,927 → 1,536 → 768 → 204, 8.9 M
parameters) on all 290,000 only if the three took under 180 gpu minutes together (they took
143.4, so it ran). Cpu jobs 270 to 273 run Task 15's checks (a) to (d) per fit on the 10,000
held-out deals (`check.md` in each fit folder), cpu job 274 the table below
(`runs/risk/300k/summary.md`).

Wall time per step (from each job's log; the PC did not restart, no job resumed):

| step | job | lane | wall time |
|---|---|---|---|
| data, 300,000 deals | 265 | cpu | 38.0 min |
| fit, 30,000 deals | 266 | gpu | 12.1 min |
| fit, 100,000 deals | 267 | gpu | 31.1 min |
| fit, 290,000 deals | 268 | gpu | 100.2 min |
| fit, larger network, 290,000 deals | 269 | gpu | 128.3 min |
| checks (a)–(d), four fits | 270–273 | cpu | 1.0, 0.9, 0.9, 1.2 min |
| summary table | 274 | cpu | 1.3 min |

Total: 38 cpu minutes for the data, 271.7 gpu minutes for the four fits, 5.3 cpu minutes for
the checks; about 5.3 hours end to end (23:45 to 05:00 on 2026-11-20). Disk: 62.6 GiB of data;
each fit folder (`best.pt`, `ckpt.pt`, `risk.mjnn`, `train.log`, `check.md`) about 80 MB for
the small network and 170 MB for the large.

No fit stopped on the held-out loss: all four ran the 12 epochs, patience 3 never fired. The
fit's own per-epoch held-out `win` loss (a mean over all 204 kinds, so a smaller scale than
the log loss over alive kinds below) fell every epoch for the 30,000- and 100,000-deal fits
(best epoch 11 of 0–11; the last gain 0.00002); the 290,000-deal fit was flat from epoch 9
(0.018932, then 0.018938, 0.018936) and the larger network from epoch 10 (0.018784, 0.018778,
0.018777). The cosine schedule reaches zero at epoch 12, so a flat end says the schedule ran
out as much as the data did. Final per-epoch figures: 0.019406 (30,000), 0.019354 (100,000),
0.018932 (290,000), 0.018777 (larger network).

### Results

Table from `runs/risk/300k/summary.md` (job 274). Same 10,000 held-out deals for every row;
the table, the exact figure and the belief model are scored on these deals, so their figures
differ a little from the Task 15 section (3,000 other deals). "Ten-bin verdict": the number
of the ten prediction bins whose observed-to-predicted ratio lies within 0.8–1.25, "yes" when
all ten do (Task 15's check (a)). "Above 50 %" and "below 1 %": how many pairs the source puts
there, their share, and the observed deal-in rate among them (check (c)).

Held-out chunks 290–299 of `runs/risk/data-300k`: 10,000 deals, 621,097 discard decisions, 19,524,872 (decision, alive kind) pairs, 5,528,409 in hand.

#### Population `all` (19,524,872 pairs, observed 1.210 %)

| source | log loss | Brier | ten-bin verdict (bins within 0.8–1.25) | above 50 %: n / share / observed | below 1 %: n / share / observed |
|---|---|---|---|---|---|
| table | 0.04607 | 0.010872 | no (6 of 10) | 20,750 / 0.11 % / 42.7 % | 14,883,121 / 76.23 % / 0.11 % |
| exact | 0.04631 | 0.010869 | no (8 of 10) | 908 / 0.00 % / 60.5 % | 14,875,064 / 76.19 % / 0.11 % |
| belief | 0.04427 | 0.010627 | no (3 of 10) | 7,057 / 0.04 % / 61.0 % | 16,357,600 / 83.78 % / 0.17 % |
| network, 30,000 deals | 0.04820 | 0.011238 | no (9 of 10) | 1,710 / 0.01 % / 41.3 % | 15,015,492 / 76.90 % / 0.14 % |
| network, 100,000 deals | 0.04822 | 0.011248 | no (9 of 10) | 1,279 / 0.01 % / 35.7 % | 14,882,782 / 76.22 % / 0.13 % |
| network, 290,000 deals | 0.04773 | 0.011191 | yes (10 of 10) | 1,267 / 0.01 % / 39.3 % | 14,854,306 / 76.08 % / 0.12 % |
| larger network, 290,000 deals | 0.04750 | 0.011167 | yes (10 of 10) | 1,037 / 0.01 % / 41.9 % | 15,066,171 / 77.16 % / 0.13 % |

#### Population `hand` (5,528,409 pairs, observed 1.244 %)

| source | log loss | Brier | ten-bin verdict (bins within 0.8–1.25) | above 50 %: n / share / observed | below 1 %: n / share / observed |
|---|---|---|---|---|---|
| table | 0.04498 | 0.010925 | no (8 of 10) | 1,114 / 0.02 % / 50.5 % | 4,498,873 / 81.38 % / 0.13 % |
| exact | 0.04601 | 0.011118 | no (2 of 10) | 79 / 0.00 % / 68.4 % | 4,514,388 / 81.66 % / 0.13 % |
| belief | 0.04386 | 0.010803 | no (2 of 10) | 2,330 / 0.04 % / 60.9 % | 4,658,452 / 84.26 % / 0.15 % |
| network, 30,000 deals | 0.04608 | 0.011236 | no (7 of 10) | 737 / 0.01 % / 49.8 % | 4,424,156 / 80.03 % / 0.11 % |
| network, 100,000 deals | 0.04610 | 0.011238 | yes (10 of 10) | 498 / 0.01 % / 45.2 % | 4,382,286 / 79.27 % / 0.11 % |
| network, 290,000 deals | 0.04561 | 0.011174 | no (8 of 10) | 492 / 0.01 % / 51.4 % | 4,382,684 / 79.28 % / 0.10 % |
| larger network, 290,000 deals | 0.04536 | 0.011134 | yes (10 of 10) | 435 / 0.01 % / 52.0 % | 4,422,732 / 80.00 % / 0.11 % |

#### The fits

| fit | hidden | parameters | deals | rows | epochs run | best epoch | minutes | stopped by |
|---|---|---|---|---|---|---|---|---|
| network, 30,000 deals | [768, 384] | 4,158,540 | 30,000 | 1,861,300 | 12 | 11 | 11.9 | the epoch count |
| network, 100,000 deals | [768, 384] | 4,158,540 | 100,000 | 6,202,195 | 12 | 11 | 30.9 | the epoch count |
| network, 290,000 deals | [768, 384] | 4,158,540 | 290,000 | 17,988,237 | 12 | 9 | 100.0 | the epoch count |
| larger network, 290,000 deals | [1536, 768] | 8,906,700 | 290,000 | 17,988,237 | 12 | 11 | 128.1 | the epoch count |

**Verdict on Brian's question: no.** Ten times the data does not move the network clearly
toward the belief model, and it does not pass it. On population `all` the log loss goes
0.04820 (30,000 deals) → 0.04822 (100,000) → 0.04773 (290,000): the move over one tenfold
step is 0.00047, with no move at all over the first threefold step, while the gap to the
belief model (0.04427) is 0.00346 and to the table (0.04607) 0.00166. At that rate the
network at 3 million deals would score about 0.0473, still behind the table and about
0.0030 behind the belief model; closing the gap to the belief model by tenfold steps would
need about seven more of them. The larger network (8.9 M parameters) on the same 290,000
deals gains another 0.00023 (0.04750), less than half a tenfold step of data, and is still
behind the table. On population `hand` (tiles the player holds, the ones the bot's choice
is about) the picture is the same: 0.04608 → 0.04610 → 0.04561, the larger network 0.04536,
against the table's 0.04498 and the belief model's 0.04386.

Where the data did help: calibration. The 290,000-deal fits pass the ten-bin check on `all`
(10 of 10, where the table has 6 and the belief model 3), and the larger network passes it on
`hand` too; the exact-0 and above-50 % calls are as sharp as Task 15's. The network is now the
best-calibrated source and still the worst-ranked one on log loss: the ordering, not the
level, is where it loses to the belief model, which sees the same `history` and reads the
opponents' waits far better. The 30,000-deal fit here (0.04820, 12 epochs) also beats Task
15's longer fit on the same amount of data (0.04839, 30 epochs), so the recipe (cosine over 12
epochs, best epoch by `win` alone) accounts for about half a tenfold step by itself.

Nothing becomes the default; `danger::with_risk` stays off. The 3-million-deal run would cost
about 6.3 cpu hours of data (630 GB at 3,613 bytes a row, more than C: should give it, so the
rows would first have to shrink again or the fit read them in parts) and about 17 gpu hours
for one fit of the small network at this rate, for a predicted 0.0005 gain; the numbers say it
is not worth it, and Brian decides.

Files: `runs/risk/300k/{fit-30k,fit-100k,fit-290k,fit-290k-big}/` (each with `risk.mjnn`,
`best.pt`, `ckpt.pt`, `train.log`, `check.md`) and `runs/risk/300k/summary.md`, in the
workers clone, not in git.
