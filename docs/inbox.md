# Inbox

Questions only Brian can answer. Format: a bold title that is the question itself, then What it is,
Example, Why it matters, Options with the Default marked. The orchestrator turns each into a card.

## Open

- **Should milestone 2 train five copies of the belief model to give honest confidence ranges?**
  - What it is: Task 18c's cheap confidence range (a last-layer Laplace approximation on
    `belief-v2`) failed its check: the observed rate fell inside mean ± 1 sd in 26 % of probability
    groups (95 % interval 13–30 %), where 60–76 % passes. The ranges are far too narrow because the
    model's bias, not its weight noise, is the main error. Details: `docs/belief.md` §8.
  - Example: a tenpai guess of 0.39 with sd 0.010 has an observed rate of 0.33; the range excludes it.
  - Why it matters: you asked for each guess to carry its own confidence, and "if fails add to
    Milestone 2". Nothing else waits on it.
  - Options: (a) add "train five copies of belief-v2 and use their spread" to milestone 2 (about 5 × 6 GPU
    minutes plus an evaluation); (b) drop the confidence ranges.
  - Default: (a), as you said.

## Answered
