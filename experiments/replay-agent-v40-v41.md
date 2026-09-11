# Replay Agent v40-v41

## Decision

V41 is the strongest candidate so far, but it does not replace v32 yet. Its
5,000-game comparison against the deployed v32 was positive but not
statistically decisive. The selected deployment remains
`models/replay-agent-v32.json`; the reproducible candidate configuration is
`models/replay-agent-v41-candidate.json`.

## V40 calibrated abstention

V40 trained on all 469 deterministic-agent validation states, including equal
outcome states, with ranking and binary cross-entropy losses. Its best epoch
improved offline selected outcome from 35.61% to 38.81%. Against the actual
minimum-risk runtime baseline, the ungated Policy gained only 5 validation
wins; a calibrated probability-advantage gate peaked at a 7.5 percentage-point
threshold with 12 beneficial and 6 harmful interventions.

Runtime now supports `policy_min_advantage`. It compares the sigmoid Policy
score of the Policy-preferred cell with the score of the minimum-risk cell and
abstains unless the difference reaches the configured threshold. The default
is zero, preserving all older deployments.

### V40 paired gameplay

| Reference | Games | V40 wins | Reference wins | V40 only | Reference only | Delta | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v34 | 5,000 | 3,362 | 3,351 | 28 | 17 | +0.22 pp | 0.13516 |
| deployed v32 | 5,000 | 3,379 | 3,362 | 64 | 47 | +0.34 pp | 0.12848 |

V40 was consistently positive but neither comparison was statistically
decisive. Against v32, intermediate boards were -3 wins while expert boards
were +20, motivating a difficulty-aware intervention rule.

## V41 expert-board gate

V41 reuses the V40 checkpoint and adds `policy_min_board_cells=480`. The
outcome Policy is therefore disabled on 16x16/40 boards and enabled on
30x16/99 boards. This costs no additional model space. The option defaults to
zero for backward compatibility and is available in deployment configs and
the paired checkpoint comparison CLI.

An independent 500-game screen scored 336 wins against v32's 332, with 8:4
discordant outcomes. A second independent 5,000-game formal set produced:

| Split | V41 wins | v32 wins | V41 only | v32 only | Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intermediate (2,500) | 2,173 | 2,170 | 12 | 9 | +0.12 pp |
| Expert (2,500) | 1,224 | 1,209 | 60 | 45 | +0.60 pp |
| Combined (5,000) | 3,397 | 3,379 | 72 | 54 | +0.36 pp |

The combined exact two-sided p-value is 0.12958. Across the two post-threshold
expert test batches, whose expert-board runtime behavior is identical, the
exploratory aggregate is 120:85 discordant outcomes (p=0.01735) and +0.70
percentage points over 5,000 expert games. Because the board-size subgroup was
chosen after inspecting the first batch, the second V41 set remains the proper
confirmatory result and is not significant by itself. V41 is retained as the
best candidate but is not promoted over v32.

## Next direction

V42 should make the difficulty gate principled rather than relying only on
board area. Candidate signals include mine density, frontier size, solver
exactness, and calibrated Policy uncertainty. Training should oversample
expert-board states and evaluate calibration separately by board class. A
future promotion should reproduce the expert-board gain on another locked
holdout and improve the combined paired result decisively.
