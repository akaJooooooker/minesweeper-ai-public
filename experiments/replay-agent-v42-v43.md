# Replay agent V42–V44

Date: 2026-09-04

## Benchmark contract

- Standard beginner: 9x9/10; intermediate: 16x16/40; expert: 30x16/99.
- A standard first click is chosen freely by the agent and is safe only. It is not forced to be a zero or to open an area.
- No-guess uses the prescribed first click from a real replay. That click must be a zero and open an area.
- No-guess beginner/intermediate/expert/evil are 9x9/10, 16x16/40, 30x16/99, and 20x30/130.
- 3BV is one click per zero opening plus one click per remaining isolated safe number.
- 3BV/s is completed-board 3BV divided by agent solve time. IOE is 3BV divided by reveal, flag, and chord clicks.

## Agent changes

V42 raised the exact constraint frontier from 24 to 40 cells. V43 keeps the 24-cell primary solver for normal standard positions and escalates to 40 cells only when the primary result is non-exact and has no proven-safe move. In no-guess mode V43 goes directly to the 40-cell proof solver. V44 adds a 64-cell proof tier only for no-guess mode: a non-exact 40-cell result is completed by the 64-cell tier before selecting a move. Standard mode deliberately remains on the faster V43 24→40 path.

The play loop also uses safe chords. A chord is allowed only when every adjacent flag was independently proven to be a mine; it selects the chord that opens the most covered cells.

## Independent real no-guess replay scorecard

Agent: `models/replay-agent-v44.json`; CPU; all available winning replays; no replay identity is retained.

| Mode | Boards | Wins | Win rate | Guesses | 3BV/s | IOE | Median solve |
|---|---:|---:|---:|---:|---:|---:|---:|
| Beginner | 124 | 124 | 100.00% | 0 | 6154.36 | 0.7359 | 2.27 ms |
| Intermediate | 1110 | 1110 | 100.00% | 0 | 2483.77 | 0.8815 | 27.63 ms |
| Expert | 307 | 307 | 100.00% | 0 | 1340.84 | 0.9107 | 128.91 ms |
| Evil | 155 | 155 | 100.00% | 0 | 770.15 | 0.8365 | 229.86 ms |

The intermediate no-guess acceptance target of at least 95% is exceeded on 1110 independent real boards. V44 solved all 1696 available real no-guess boards without guessing. This is a 100% corpus result, not a proof that every possible no-guess board is solved.

## Corrected standard random-board scorecard

Each row uses 500 safe-first-click random boards. All candidates use the same seeds per difficulty. Timing is CPU wall time and can vary between runs.

| Agent | Beginner | Intermediate | Expert |
|---|---:|---:|---:|
| V41, 24-cell solver | 410/500 (82.0%) | 336/500 (67.2%) | 162/500 (32.4%) |
| V42, 40-cell solver | 413/500 (82.6%) | 338/500 (67.6%) | 175/500 (35.0%) |
| V43, 24→40 escalation | 413/500 (82.6%) | 338/500 (67.6%) | 175/500 (35.0%) |

V43 preserves all 18 additional wins from V42 compared with V41 while recovering most of the speed lost by always running the 40-cell solver. Its measured standard 3BV/s was 4053.35 / 1939.83 / 1118.29 for beginner / intermediate / expert.

The promotion gate compared selected V44 with formal V32 on 1000 identical safe-first-click boards per mode (`seed=2026090401`):

| Mode | V32 wins | V44 wins | Delta | V32-only | V44-only | McNemar p | V44 3BV/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Beginner | 852 | 854 | +0.2 pp | 0 | 2 | 0.5000 | 3788.16 |
| Intermediate | 690 | 699 | +0.9 pp | 4 | 13 | 0.0490 | 1803.37 |
| Expert | 312 | 327 | +1.5 pp | 8 | 23 | 0.0107 | 1049.94 |

V44 is statistically better on this gate for intermediate and expert, neutral on beginner, and about 4–5% slower than V32. A trial that also enabled 64 cells in standard expert reached 330 wins but only 806.87 3BV/s, so that slower policy was rejected.

## Verification

- Full suite: 85 tests passed.
- Real no-guess openings rejected as invalid: 0.
- `models/replay-agent-v44.json` is the selected deployment. V32 is retained as the rollback reference.
