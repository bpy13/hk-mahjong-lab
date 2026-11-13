# Board

## Workers

## Results
- Task 22 (slot 1, 2026-11-13): ticked, commit 321c384; `Api::set_belief`, hint/review/expert use the teacher sampler; hint 6.85 vs 4.34 ms per sample (1.6x); models/belief.mjnn loads; study-results/teacher-in-app.md.
- Task 21 (slot 2, 2026-11-11): ticked; race bot weighs a legal win against passing it (`passwin=on`); paired match on 4,000 hands -0.03 ± 0.16 per hand vs `passwin=off`, passes in 0.8 % of hands; tests for pass, take (opponent ready / short wall) and the passed-tile lock.
- Task 20 (slot 1, 2026-11-11): ticked; race bot -1.97 ± 1.16 per hand vs three value (was -9.4) after a calm-position offence margin and refitted tables; hazard check 5 of 6 buckets, offence agreement 1000/1000; loop re-queued (job 97, gpu lane); inbox checkpoint posted.
- Tasks 8 and 9 (slot 1, 2026-11-11): ticked; race loop (job 90) 4 generations, no measurable gain (gen 3 vs 2 +0.18 [-0.08, +0.44]); -9.37 vs three value; calibration table and hazard overestimate recorded in study-results/race-bot.md and open-items.
- Task 18 (slot 1, 2026-11-11): ticked on Brian's ruling; belief-v2 calibration per bucket recorded in open-items (tenpai within 1 of 6, deal-in 4 of 6).
- Task 18c (slot 2, 2026-11-11): Laplace confidence ranges built (`mahjong.belief.laplace`, `evaluate --laplace`); check FAILED, 26 % of groups inside ±1 sd (pass 60–76 %); inbox question posted for the five-copy ensemble in milestone 2 (docs/belief.md §8).
- Task 9 (slot 2, 2026-11-11): loop recorded in study-results/race-bot.md; 3 generations, converged, best gen 3 (-9.37 [-9.98, -8.76] vs three value, -10.67 vs teacher); gains vs previous all inside noise. Task 8 tick left to slot 1.
