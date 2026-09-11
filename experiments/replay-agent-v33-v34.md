# Replay Agent v33-v34

## Decision

Neither candidate replaces v32.

- v33 added an outcome-conditioned Policy head, but both independent
  1,000-game comparisons regressed.
- v34 doubled model width from 64 to 128 and improved every Risk-oriented
  offline metric, but tied v32 exactly over 3,000 paired games.
- The selected deployment remains `models/replay-agent-v32.json`.

## v33 outcome policy

The v33 experiment trained only `standard_policy_head`; the parent v32
backbone, Risk head, and Value head were frozen. Training used geometry-
augmented v14 deployment gaps plus human deployment gaps. The loss increased
the demonstrated move probability for games that eventually won and decreased
it for games that eventually lost.

The validation loss improved from 2.109119 to 1.920113 over 60 epochs. The
winning-versus-losing chosen-probability separation increased from 0.022521 to
0.023497, but this did not translate to wins.

### Policy-blend screen (200 paired games, seed 20260934)

| Policy blend | v33 wins | v32 wins | v33 only | v32 only |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 131 | 131 | 0 | 0 |
| 0.10 | 132 | 131 | 1 | 0 |
| 0.20 | 132 | 131 | 1 | 0 |
| 0.30 | 131 | 131 | 1 | 1 |

### Independent validation (1,000 paired games, seed 20260935)

| Policy blend | v33 wins | v32 wins | v33 only | v32 only |
| ---: | ---: | ---: | ---: | ---: |
| 0.10 | 672 | 676 | 3 | 7 |
| 0.20 | 674 | 676 | 6 | 8 |

The game-level outcome is too noisy to label every earlier guess in a losing
trace as a bad action. Future outcome learning needs counterfactual rollouts or
per-decision returns rather than terminal imitation alone.

## v34 wide Risk model

v34 was trained from scratch with width 128, batch size 128, 12 epochs, learning
rate 5e-4, no batch delay, AMP, and a 60% PyTorch memory-fraction limit.

Training data:

- 61,485 original synthetic multi-head states.
- 38,208 human Risk-only states.
- 99,693 states total.

The Risk-only conversion explicitly sets Policy and Value weights to zero, so
human won-game selection does not bias those heads.

### Resource observations

- GPU utilization: about 66%.
- Total observed VRAM: about 12.2 / 16.4 GB.
- GPU temperature: 61 C.
- GPU power: about 152 W.
- Training Python RAM: about 7.39 GB.
- Free system RAM: about 34 GB.

The larger batch and wider model used the GPU substantially better than earlier
runs while keeping safe headroom.

### Offline evaluation

| Metric | v34 | v32 | Relative result |
| --- | ---: | ---: | ---: |
| General holdout Risk Brier | 0.00268041 | 0.00288591 | 7.1% better |
| Synthetic deployment regret | 0.0475272 | 0.0573814 | 17.2% better |
| Synthetic optimal rate | 39.10% | 36.84% | +2.26 pp |
| Human-final deployment regret | 0.0481267 | 0.0790926 | 39.2% better |
| Human-final optimal rate | 66.67% | 55.56% | +11.11 pp |

The human-final set contains only 18 states, so its large improvement is not a
reliable standalone promotion signal.

### Paired gameplay

A 200-game screen gave 141 wins for v34 and 142 for v32 at both alpha 0.50 and
0.80.

At alpha 0.65, matching the current deployment blend:

| Seed | Games | v34 wins | v32 wins | v34 only | v32 only |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20260937 | 1,000 | 674 | 669 | 14 | 9 |
| 20260938 | 500 | 337 | 339 | 3 | 5 |
| 20260939 | 500 | 341 | 341 | 7 | 7 |
| 20260940 | 500 | 341 | 340 | 6 | 5 |
| 20260941 | 500 | 352 | 356 | 2 | 6 |
| **Total** | **3,000** | **2,045** | **2,045** | **32** | **32** |

The aggregate McNemar exact p-value is 1.0. v34 made different decisions, but
those changes exchanged wins and losses exactly evenly. It also made 4,930
guesses versus 4,909 for v32.

## Resource decision after v34

Increasing training utilization was worthwhile: it produced a much stronger
offline Risk model in about 18 minutes without approaching thermal or memory
limits. Raising the GPU limit beyond 60% is not the next quality bottleneck,
because the better offline predictor still failed to improve gameplay.

Evaluation parallelism was worthwhile. Four two-thread shards used about 4 GB
of Python RAM in total, left about 38 GB free, and reduced 2,000-game wall time
to roughly three minutes. Future large paired evaluations can safely use four
workers, with six to eight workers considered only after another measurement.

The next quality step should generate counterfactual per-choice win labels (or
an equivalent rollout target) instead of scaling the same one-step Risk labels
again.

