# The teacher in the app (Task 22)

`Api::set_belief(bytes)` (wasm: `&[u8]`, Python: `bytes`) loads a belief model. While one is set,
`hint`, `review` and the `expert` bot deal their guesses with
`WorldSampler::Belief { weights: true, draws: BELIEF_DRAWS, tilt: default_tilt(true) }` (the
teacher's `weights=1`); the expert bot then uses 64 samples and 4 candidates. `grade` and every
call without a model stay on best-of-K. A bad blob returns `error.bad_belief`.

Model used: `models/belief.mjnn` (the teacher's measured model, 2026-10-29). It still loads with
today's feature encoding; `belief-v2.mjnn` was not needed.

## Speed

Native, release build, one thread, 20 mid-hand positions (seeds 100-119, after 25 actions of
`hard` bots), `hint_with_samples` with 32 samples, no time budget, 640 samples in total:

| Sampler | Total | Per sample |
|---|---|---|
| best-of-K (no model) | 2779 ms | 4.34 ms |
| belief model | 4387 ms | 6.85 ms |

The model costs about 1.6 times per sample. Rerun:
`cargo test -p mahjong-app-api --release --test api belief_timing -- --ignored --nocapture`
(needs `models/belief.mjnn`). The browser (wasm) speed was not measured here.
