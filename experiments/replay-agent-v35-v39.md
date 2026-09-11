# Replay Agent v35-v39

## Decision

No candidate replaces v32. V38 and V39 learned their offline rollout targets,
but neither improved paired gameplay against v34. The selected deployment
remains `models/replay-agent-v32.json`.

## Counterfactual pipeline

V35 added hidden-board cloning and per-cell counterfactual examples. At a
non-exact expert guess state, the generator selects the lowest-risk candidates,
clones the current game, reveals each candidate, and completes the branch. Only
the public observation is encoded for model training; hidden mines are used
solely by the simulator to obtain outcomes.

The standard reveal Policy channel is fine-tuned while the Risk head, Value
head, backbone, flag Policy channel, and chord Policy channel stay frozen.
Geometry augmentation supplies all eight rotations/reflections. Runtime
`policy_top_k` restricts Policy reranking to the solver/model's lowest-risk
cells.

V36 added common random numbers, so every candidate at a state receives the
same continuation seeds. It also filtered low-confidence outcome gaps. An
overfit diagnostic reached 95.8% selected outcome on a tiny split, confirming
that the head has enough capacity; sparse and noisy generalization was the
limitation.

## V37 stochastic CRN target

V37 used three candidates, four stochastic continuations per candidate, and
three maximum recorded states per expert game. The new dataset contained 359
train states and 114 validation states. Under the original unique-best filter,
78 train and 34 validation states had at least a 0.25 outcome gap. Eight-way
augmentation produced 624 training examples.

The best checkpoint improved validation selected win rate from 39.71% to
44.85%, but its Policy score was too weak to affect low-blend gameplay. Over
200 paired games, blends 0.05, 0.10, and 0.20 were identical to v34. At blend
0.60 it scored 132 wins against 133, and at 1.00 it scored 131 against 133.

## V38 scaled stochastic target

V38 increased CPU data generation from four to five stable workers and added
2,400 complete expert training games plus valid partial shards. Together with
V37 training data, this yielded 377 unique-best high-confidence train states.
The primary holdout had 42 states, and two secondary holdouts had 35 and 34.

Offline selected outcome improved consistently:

| Holdout | v34 | v38 |
| --- | ---: | ---: |
| Primary v38 validation-c | 40.48% | 45.24% |
| Secondary v38 validation-b | 45.00% | 47.14% |
| Historical v37 validation | 39.71% | 42.65% |

The rollout generator exposed a numerical bug when a non-finite score reached
`random.choices`. V38 added finite-score filtering and stable shifted
exponentials, plus a regression test. All already-written JSONL lines from
interrupted workers were valid and retained.

### V38 paired gameplay at Policy blend 0.60

| Games | v38 wins | v34 wins | v38 only | v34 only | Delta | Exact p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 3,355 | 3,373 | 26 | 44 | -0.36 pp | 0.04139 |

The stochastic target produced a statistically detectable regression and was
rejected.

## V39 deterministic agent target

V39 added `--continuation-mode agent`. After the candidate's first reveal, the
actual deterministic continuation agent finishes the game. This removes the
stochastic continuation mismatch and costs one rollout per candidate instead
of four.

The training filter was corrected from best-minus-second-best to
maximum-minus-minimum outcome. The ranking loss already supports tied winners,
so states such as `[win, win, loss]` now teach the model to avoid the loser.

Three complete 1,000-game train shards, one large valid partial shard, a smoke
shard, and one independent 1,000-game validation shard produced 274 informative
train states and 111 informative validation states after the corrected filter.
The best checkpoint improved validation selected outcome from 41.44% to 53.15%.

### V39 paired gameplay at Policy blend 0.60

| Games | v39 wins | v34 wins | v39 only | v34 only | Delta | Exact p |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 3,361 | 3,373 | 37 | 49 | -0.24 pp | 0.23538 |

V39 reduced the V38 regression and increased the number of changed outcomes,
but it still did not beat v34 and therefore did not proceed to a promotion test
against v32.

## Next direction

Scaling Policy labels alone is insufficient. V40 should add a learned or
calibrated intervention gate: Policy may rerank only when its advantage over
the minimum-risk baseline is large enough. The evaluation must include
coverage-versus-regret curves so the gate can trade decision frequency for
reliability. Any candidate still needs at least 5,000 unseen paired games and a
positive discordant result before replacing v32.
