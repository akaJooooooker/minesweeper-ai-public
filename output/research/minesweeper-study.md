# Exact Inference and Selective Learning for Minesweeper
## Paired Evaluation and Deployment Aligned Counterfactual Adaptation

AI Minesweeper Project | 7 September 2026

### Abstract
Minesweeper combines exact local constraints with sequential decisions under irreducible uncertainty. This study evaluates V44, a deployed hybrid agent that prioritizes constraint reasoning and selectively invokes a residual neural network for unresolved guesses. Adaptive component enumeration, risk blending, and a gated outcome policy are assessed using distinct evaluation protocols. A preserved standard-test report describes 6,000 paired boards with a safe, but not necessarily zero, first click. It reports 3,746 V44 wins against 3,697 for V32; intermediate and expert improvements are 0.85 and 1.65 percentage points, respectively. A separate deployment-aligned experiment collects 2,000 complete games under a guaranteed-zero opening and a no-flag policy, yielding 194 counterfactual decision states, only 34 of which distinguish candidate outcomes. Updating the reveal policy improves validation Brier score from 0.2581 to 0.2532, but a new 5,000-pair test yields only two additional wins (43.82% versus 43.78%; exact McNemar p = 0.7905). The candidate is therefore not promoted. An offline interface preserves decision provenance and supports reversible board replay. The results support selective learning backed by exact inference, while showing that more trajectories and lower offline prediction error do not by themselves establish stronger play. Claims are restricted to the tested board distributions and internal baselines.

Keywords: Minesweeper; constraint satisfaction; neural risk estimation; counterfactual supervision; paired evaluation; reproducibility

### 1 Introduction
A Minesweeper agent must distinguish what is logically certain from what is merely favorable. A safe move can follow directly from adjacent clues, whereas a guess can lose even when its estimated mine probability is exact and minimal. Consequently, a failed game is not a sequence of uniformly incorrect decisions. Likewise, a successful game can include a poor guess that happened to survive. This makes terminal win or loss an incomplete training signal for individual moves.

The practical objective is to improve whole-game success while preserving dependable deductions and usable decision latency. This study examines an existing hybrid implementation, V44, and a targeted attempt to adapt its outcome policy to the same decision engine used by a local graphical application. It addresses three questions: whether V44 improves on its frozen predecessor under matched conditions; whether fresh wins and losses contain useful learning opportunities; and whether a small deployment-aligned counterfactual update improves independent full-game outcomes.

The contribution is an empirical systems study, not a new proof procedure or a claim of state-of-the-art performance. The evidence combines a preserved report of a 6,000-board comparison, an available corpus of 1,696 winning no-guess replay boards, and a separate 2,000-game collection followed by a 5,000-board paired test. The latter experiment provides a negative result: a measurable change in an offline score does not translate into statistically supported gains in play. An accompanying replay interface makes recorded actions, risks, and terminal failures inspectable on the game board.

### 2 Related Work
Kaye established NP-completeness for the Minesweeper consistency problem [1]. This concerns deciding whether a partial arrangement admits a consistent placement of mines; it does not imply that each practical position requires exhaustive search or that minimum immediate risk maximizes eventual victory. The distinction motivates efficient propagation followed by bounded enumeration of connected frontier components.

Prior work has combined constraint satisfaction with learned decision mechanisms. Sinha, Malviya, and Nayak study CSP-based enumeration, learned heuristics, and deep Q-learning for Minesweeper [2]. Sajjad investigates neural approximators trained from large game collections and reports both their utility and computational limitations [3]. These establish relevant methodological precedents. Their reported accuracies are not used as direct numerical baselines here because identical board generation, opening rules, action semantics, and evaluation seeds have not been established.

Sequential learning also depends on the state distribution induced by a deployed policy. Ross, Gordon, and Bagnell formalize this difficulty and propose dataset aggregation with expert supervision [4]. The present collection follows the deployed agent and evaluates alternative actions through simulation. It is not an implementation of DAgger and does not inherit its guarantees: labels come from a fixed continuation policy on sampled hidden boards, rather than an expert oracle over all reachable states.

### 3 System and Learning Method
#### 3.1 Public observations and exact inference
The environment contains a rectangular board with a fixed mine count. An observation consists of revealed clues, unopened cells, and flags. The inference boundary receives this public observation and the mine count; it does not receive the random seed, hidden mine positions, or replay outcome. Ground truth is available only to the environment, audit, statistics, and offline label-generation layers.

For a revealed clue j, let U(j) be its unopened neighbors and f(j) the number of adjacent flags. With binary mine indicators x(i), each clue supplies a constraint:

[EQUATION] sum over i in U(j) of x(i) = clue(j) - f(j), with x(i) in {0, 1}.  (1)

The solver propagates forced safe and mined cells, derives subset constraints, partitions the frontier into connected components, and enumerates admissible assignments within its component limit. Component counts are combined with the global remaining-mine count; unconstrained cells contribute combinatorial multiplicities. When enumeration is complete, the posterior mine probability is the weighted fraction of globally consistent assignments containing a mine at that cell. Exactness is relative to the observation, flag assumptions, and board-generation model, not a guarantee of survival for a positive-risk guess.

In standard mode, V44 starts with a 24-cell component limit and escalates to 40 when the result is non-exact and no safe move is available. With guessing disabled, it begins at 40 and may escalate to 64. These are per-component bounds, not limits on total board area. Safe moves take precedence over learned choices. Contradictory observations stop action selection. In the command-line play loop, flags and chords are supported; a chord is executed only around independently proven mines.

[FIGURE pipeline.png]
Figure 1. Decision flow for the hybrid agent. Hidden board state remains outside inference. The 64-cell tier applies only when guessing is disabled. The graphical adapter preserves an eligible learned decision after prioritizing proven-safe actions.

#### 3.2 Neural risk and gated outcome selection
The network consumes 18 spatial feature channels: nine clue planes; unopened, flagged, frontier, and remaining-mine-ratio planes; two mode indicators; revealed-safe fraction; mine density; and a source indicator. A validity mask excludes padding. The selected checkpoint has width 128, a convolutional stem, and six residual blocks with first-convolution dilations 1, 2, 4, 8, 1, and 2. Masked group normalization and SiLU activations are used. A masked global pool supplies contextual features. Separate heads predict cell risk, standard and no-guess action logits, and a scalar value. Each policy head has reveal, flag, and chord channels.

The risk network is consulted only when no proven-safe action exists and the analysis remains non-exact. For eligible cells, the selected deployment blends learned risk p(N) with the solver estimate p(S):

[EQUATION] p(blend, i) = 0.65 p(N, i) + 0.35 p(S, i).  (2)

Outcome reranking is restricted to boards with at least 480 cells and considers at most the three lowest-risk candidates. Let q(i) be the sigmoid of the reveal-policy logit. The policy is admitted only if its preferred candidate exceeds the risk-preferred candidate by at least 0.075 in q. Accepted choices minimize a rank mixture with risk weight 0.4 and inverted outcome-policy weight 0.6. Otherwise the agent retains risk-based selection. Ties use an information heuristic and deterministic coordinates. Exact probability calculations bypass this learned reranking.

The selected V44 deployment uses the V40 calibrated-abstention checkpoint. Its recorded parent is the wider V34 network. The V40 training metadata contain 14,728 augmented training examples and 469 validation states, with 111 informative validation states. Training used ranking and binary-cross-entropy objectives, learning rate 0.0005, batch size 256, and patience 40; epoch 47 was retained after 87 epochs. These historical development results explain the lineage and are not an additional independent test of the final system.

#### 3.3 Counterfactual supervision aligned with deployment
Ordinary self-play was already present in development data: two prior on-policy files contain 2,000 expert games, including 937 wins and 1,063 losses. Four later expert counterfactual collections contain 1,784 decision states, with 339 informative states [7]. The new collection therefore addresses a narrower gap: decisions actually reached through the graphical decision engine, including 30 by 20 boards with 130 mines.

At up to two non-exact guess states per game, the generator selects three low-risk candidates, clones the same underlying board, reveals each candidate separately, and completes the branch with the same V44 continuation engine. A branch label y is one if the game is ultimately won and zero otherwise. The chosen-action branch is checked against the original complete trajectory. A state is informative for ranking when candidate outcomes differ. Keeping states where all branches agree retains calibration information, although such states provide no relative ranking signal.

These outcomes are conditional simulation samples. A zero does not establish that an action is generally inferior, and a one is not an oracle estimate of its win probability over all boards compatible with the observation. The continuation policy also constrains the labels: a branch may fail because of a later choice. Grouping by board before augmentation prevents related views from crossing training and validation partitions.

The new candidate freezes the backbone, risk head, value head, no-guess head, and non-reveal channels. Only the standard reveal channel is updated. Its loss combines class-balanced binary cross-entropy over candidate outcomes and a softmax ranking term over informative states. The ranking target distributes mass uniformly across the best observed candidate outcomes. Both loss weights equal one. Training uses AdamW with zero weight decay, learning rate 0.0001, batch size 64, eight geometric transforms, at most 40 epochs, and patience eight. Validation selects epoch 18; training stops at epoch 26. Transformations increase training examples, not the number of independent boards or decision states.

### 4 Evaluation Design
#### 4.1 Separate evaluation tracks
Opening rules materially change the distribution of positions. A guaranteed-safe first click and a guaranteed-zero first click must not be treated as the same benchmark. Likewise, a corpus of human-winning no-guess boards is a selected population. Table 1 separates the evidence tracks. All quoted new gameplay results use local simulation; no online service is queried to obtain additional games.

Table 1. Evaluation populations and action protocols. Dimensions are width by height.

| Track | Boards | Opening and actions | Purpose |
| Standard blind test | 6,000 pairs; 2,000 per mode | Safe only; freely chosen opening; proven flags and chords | V44 versus V32 |
| Recorded no-guess corpus | 1,696 winning replay boards | Prescribed zero opening; guessing disabled | Corpus-level solving coverage |
| New collection | 1,000 expert and 1,000 evil games | Guaranteed zero; no flags; graphical decision engine | Counterfactual training and validation |
| Candidate test | 2,500 expert and 2,500 evil pairs | Guaranteed zero; no flags; same corrected adapter | Candidate versus frozen V44 |

Standard modes are beginner 9 by 9 with 10 mines, intermediate 16 by 16 with 40 mines, and expert 30 by 16 with 99 mines. The new evil mode is 30 by 20 with 130 mines. Historical no-guess evil boards are reported as 20 by 30 with 130 mines. Orientation, prescribed openings, and population selection are retained as protocol distinctions.

The preserved September 4 report describes an internally pre-specified standard protocol with master seed 6739657020312379691 and hashes for deployment, checkpoint, solver, agent, loader, and evaluator [6]. Its stated gate required no per-mode regression beyond 0.5 percentage points, a positive intermediate-plus-expert win gain, at least 90% retained 3BV/s in every mode, and passing software tests. The available archive does not independently establish that the protocol was fixed before outcomes were observed; no external registration or contemporaneous Git history is available.

The new collection uses master seed 2636368083988871269. Before play, indices divisible by five are assigned to validation; other indices go to training. This yields 1,600 training games and 400 validation games. All 2,000 completed trajectories reconstruct to their recorded outcomes. No board-hash overlap is found between these partitions. The saved candidate protocol specifies a separate master seed, 175985001720591491, and describes the design as internally pre-specified. No independent pre-training timestamp is available. Neither model is tuned on its 5,000 pairs. The acceptance rule requires a positive combined gain with exact McNemar p below 0.05 and no mode worse by over 0.5 percentage points.

#### 4.2 Statistical and operational measures
Whole-game win rate is the primary effectiveness measure. In a paired comparison, b counts reference-only wins and c counts candidate-only wins. The reported two-sided exact conditional McNemar test conditions on d = b + c and uses a Binomial(d, 0.5) null distribution:

[EQUATION] p = min(1, 2 Pr[B <= min(b, c)]), B ~ Binomial(b + c, 0.5).  (3)

The test can be conservative [5], especially with few discordant pairs. Its selection is preserved from the experiment protocol rather than replaced after viewing outcomes. Descriptive 95% Wilson intervals accompany single-agent win rates in Figure 2; these are not confidence intervals for paired differences. Per-mode standard p-values are reported individually. As a sensitivity check, both positive standard-mode findings also pass a Bonferroni threshold of 0.05/3; this adjustment was not part of the historical acceptance gate.

For historical speed reporting, 3BV counts connected zero openings plus isolated safe numbered cells. For each agent and mode, 3BV/s is total 3BV divided by total solve time over that agent's won games only. IOE uses the same winning subset, dividing total 3BV by total reveal, flag, and chord actions. These are ratios of sums, not means of per-game ratios. Because the agents win different board subsets, these are operational throughput summaries, not paired speed estimates. Win rates and guess counts include all tested boards. The GUI instead shows completed 3BV progress and elapsed play time, excluding replay pauses.

### 5 Results
#### 5.1 Reported historical standard comparison
The preserved report records 3,746 V44 wins in 6,000 boards, compared with 3,697 for V32 (Table 2). The combined gain is 49 wins. Intermediate and expert together gain 50 wins, with one fewer beginner win. Both positive per-mode comparisons have small exact paired p-values. The reported results support improvement over the internal reference under the stated standard protocol; they do not rank V44 against external solvers.

Table 2. Preserved standard-test results. Each row reports 2,000 paired boards. Delta is in percentage points (pp); b and c are reference-only and V44-only wins.

| Mode | V32 wins | V44 wins | Delta (pp) | b / c | Exact p |
| Beginner | 1,669 (83.45%) | 1,668 (83.40%) | -0.05 | 1 / 0 | 1.000000 |
| Intermediate | 1,371 (68.55%) | 1,388 (69.40%) | +0.85 | 5 / 22 | 0.001514 |
| Expert | 657 (32.85%) | 690 (34.50%) | +1.65 | 22 / 55 | 0.000217 |

[FIGURE standard_rates.png]
Figure 2. Standard safe-first test: win rates and descriptive 95% Wilson intervals, with 2,000 boards per mode. The paired tests in Table 2 use discordant outcomes, not overlap of the marginal intervals.

Under the wins-only definition, the preserved report gives a minimum speed retention of 92.28%, exceeding its 90% gate. Reported V44 3BV/s is 4,070.76, 1,808.76, and 1,054.61 for beginner, intermediate, and expert; retention is 100.16%, 92.28%, and 95.55%. Total guesses decrease from 15,335 to 15,205. The report records that all six listed hashes matched after execution and that the historical test suite passed [6]. These are implementation-specific CPU summaries, not paired speed estimates or standardized cross-hardware measurements.

On the recorded no-guess corpus, V44 solves 124 beginner, 1,110 intermediate, 307 expert, and 155 evil boards, all without guessing: 1,696 of 1,696 [8]. This is a corpus-level coverage result. The source population already consists of winning no-guess replays, and a separate training-overlap audit for all historical replay identities is not established by the evidence assembled here. The result must not be interpreted as 100% success on arbitrary random boards or all possible no-guess instances.

#### 5.2 Yield of deployment-aligned game data
The new collection produces 870 wins and 1,130 losses in 255.13 seconds with four worker processes. Table 3 shows how total gameplay volume contracts into states relevant to the currently trainable policy. Of 6,101 guesses, 5,799 have exact probabilities and only 302 use neural estimation. The outcome policy is invoked in 98 guesses. The selected counterfactual states total 194, with 34 informative comparisons.

Table 3. New collection under guaranteed-zero openings and no flags. CF denotes counterfactual decision states; informative states have differing branch outcomes.

| Mode | Wins / losses | All guesses | Neural guesses | CF states | Informative |
| Expert | 501 / 499 | 2,770 | 96 | 70 | 8 |
| Evil | 369 / 631 | 3,331 | 206 | 124 | 26 |
| Total | 870 / 1,130 | 6,101 | 302 | 194 | 34 |

Among losses, 1,098 end in exact, positive-risk guesses and 32 in non-exact guesses. No recorded loss follows a decision claiming zero mine probability. This is an audit of this batch, not a universal safety guarantee. Exact risk does not imply an optimal long-horizon policy: information value and future forced guesses can still distinguish equally safe current choices. However, the current reveal-head update does not affect decisions that remain on the exact branch.

After board-level partitioning, training contains 158 counterfactual states, including 28 informative states; validation contains 36, including six informative states. The eight transformations yield 1,264 training examples. The validation Brier score decreases from 0.258125 to 0.253169, but the selected candidate wins in 13 of 36 validation states both before and after training. The best observed branch wins in 14 of 36. Validation selection performance therefore remains unchanged despite the lower probability error.

#### 5.3 Independent candidate comparison
The new candidate wins 2,191 of 5,000 boards versus 2,189 for frozen V44 (Table 4). Only 14 pairs disagree: six favor V44 and eight favor the candidate. The combined change is +0.04 pp, with exact p = 0.790527. The acceptance gate fails and V44 remains selected. This is absence of evidence for improvement at the tested scale, not proof that the two policies are identical or that future data cannot help.

Table 4. Candidate test using the same corrected graphical decision engine. This protocol differs from Table 2. Delta is candidate minus V44.

| Mode | Pairs | V44 wins | Candidate wins | b / c | Delta (pp) | Exact p |
| Expert | 2,500 | 1,253 (50.12%) | 1,253 (50.12%) | 3 / 3 | 0.00 | 1.000000 |
| Evil | 2,500 | 936 (37.44%) | 938 (37.52%) | 3 / 5 | +0.08 | 0.726563 |
| Combined | 5,000 | 2,189 (43.78%) | 2,191 (43.82%) | 6 / 8 | +0.04 | 0.790527 |

All 5,000 pairs are unique and complete. Collection and test seeds do not overlap, and both checkpoint hashes remain unchanged during the test. Eight-worker execution experienced a Python SystemError and a later worker-pool termination. Completed pairs were retained, incomplete pairs were not counted as losses, and missing indices were resumed without changing seeds, inference settings, or acceptance criteria. Four workers completed the remaining 882 pairs. The lower-level cause is unresolved; the default is now four workers. Wall-clock throughput across the interrupted comparison is not treated as a controlled performance result.

### 6 Deployment Integrity and Replay
A decision adapter can change a policy even when the checkpoint is unchanged. Before the new collection, the graphical adapter could replace an eligible neural choice by sorting the solver's raw probabilities again. The correction preserves the hybrid agent's selected cell, reason, and probability whenever it is eligible, while retaining safe-action priority and clickable-region restrictions. In the new collection, 272 decisions differ from what the old adapter would select. This counts changed choices, not additional wins. Both arms of Table 4 use the corrected adapter, so their difference cannot estimate the benefit of that correction [7].

The local application records complete wins and losses, seed and mine placement for reconstruction, action timestamps, effective and redundant actions, and pre-action model metadata. Saving a trajectory does not update model weights. A seekable timeline reconstructs public observations from the action sequence, stores sparse changes with periodic checkpoints, and renders earlier or later positions directly on the original board. An optional action marker, disabled by default, shows the action location alongside recorded probability and reason. Winning displays automatically flag remaining mines, while losing displays reveal unflagged mines. Rewinding before termination removes these terminal decorations. Replay pauses the current session and preserves it for return, rather than modifying its history or generating another training episode.

Input handling was also revised after rapid guessing exposed cancellation when a press and release landed on different cells. Unopened cells now reveal on the press; the subsequent release does not trigger another reveal or chord. Holding an already revealed number still previews only unopened, unflagged neighbors, and release on the same number permits a chord. A native integration test queues 200 cross-cell press-release pairs on a controlled 30 by 20 board and accepts all 200 reveals in approximately 0.17-0.19 seconds on the test machine. This is an event-queue test, not a human study or a latency guarantee. Board recognition remains compatible with the existing dark-theme adapter. The updated software suite contains 152 passing tests, including 17 native GUI checks.

#### 6.1 On demand replay action assessment
The replay interface also assesses a selected action using only the preceding public observation, total mine count, and action coordinates. It does not pass the hidden mine layout, seed, later observations, or terminal outcome into risk inference. Player flags are converted to unknown cells before constraint analysis, so an incorrect flag cannot manufacture a proof of safety. Exact probabilities refer to a uniform distribution over mine configurations consistent with the public constraints; incomplete enumeration is explicitly labeled as a solver approximation or a V44 blended estimate. The first protected opening is excluded from ordinary risk comparisons.

For a reveal, the panel reports the chosen cell's mine probability and identifies lower-risk unflagged alternatives, when available. Flag actions report the probability that the marked cell contains a mine. Chords are classified as proven safe only when every target is proven safe; otherwise the panel reports the largest target marginal probability, not a joint explosion probability. Effective actions and the observed number of newly revealed safe cells are displayed separately as retrospective effects. These counts do not estimate expected information gain, long-term win improvement, or action optimality.

Assessment is initiated by a button, executes in the existing background worker, and is cached per step within the loaded replay. Only one request can be pending, and results from another replay are discarded. The function performs no cloud inference or training. Six focused tests cover outcome independence, incorrect flags, safer alternatives, chord interpretation, approximate probabilities, and descriptive effects; a native GUI test checks asynchronous delivery, caching, and stale-result rejection. These are correctness checks, not evidence that the feature improves human playing skill.

### 7 Discussion and Limitations
The answer to whether V44 is strong is conditional. Against V32, the preserved report supports improved intermediate and expert results under its stated protocol, with its wins-only speed gate satisfied. The no-guess corpus shows broad successful coverage of available recorded boards. Neither result establishes superiority to the strongest external agent, general optimality, or production reliability across hardware and interfaces. A direct external comparison requires shared generation rules, openings, action policies, resource budgets, and independent seeds.

The new experiment explains why retaining more losses is useful but insufficient. Most sampled losses occur after exact risky guesses, while the updated head is consulted only on the non-exact path. Broad trajectory accumulation therefore creates many records outside the intervention point. Even within that path, only 34 of 194 sampled states distinguish the three tested branches. Correlated states from one board and geometric augmentations cannot substitute for additional independent decision opportunities. The small number of informative validation states makes model selection noisy.

Wins should be retained alongside losses. They can supply successful alternatives, maintain calibration coverage, and prevent a dataset consisting solely of terminal mistakes. Nonetheless, counterfactual supervision remains limited by candidate selection, the sampled hidden board, and the continuation policy. It can miss a useful action outside the three candidates, or favor an action only because later decisions are easier for the fixed agent. No causal claim is made that a single surviving branch is universally superior. Sampling multiple compatible boards per public state would better separate current luck from robust action value, at greater computational cost.

Several limits affect interpretation. The standard V32-to-V44 comparison combines changes in solver limits, learned components, and action strategy; it is not a factorial attribution of the gain to one mechanism. Earlier project experiments show that lower risk error and more outcome data can coexist with neutral or worse paired wins [7]. The latest candidate is one training run with one fixed evaluation cohort, not a distribution of outcomes across training seeds. Exact conditional tests can have low power when policies disagree rarely, and marginal confidence intervals do not measure paired uncertainty. An equivalence claim would require a prospectively defined margin and a suitable analysis.

The opening-rule difference is especially consequential: the standard expert result is 34.50%, while the no-flag, guaranteed-zero V44 reference in the new test reaches 50.12%. This difference must not be attributed to a weight update because V44 weights are unchanged and the protocols differ. Similarly, corpus success on selected no-guess boards cannot be compared directly with random-board win rates. Historical replay overlap, unmeasured external baselines, and the unresolved eight-worker runtime issue remain limits on generalization and deployment claims. A cleaner speed comparison would use paired timings on boards won by both agents. Those timings cannot be reconstructed for the historical standard test from the retained aggregates; a new run would be a separate experiment.

A useful next experiment would target more independent evil-board and non-exact candidate states, preserve both outcomes, freeze a new test before training, and measure actual policy changes as well as offline loss. Extending learned long-horizon selection into exact-risk ties is another hypothesis, not an established improvement. The present test set should remain excluded from subsequent training if it is to retain its role as an independent historical evaluation.

### 8 Reproducibility and Conclusion
Experiments are implemented in the local Python project. The new collection uses four CPU workers with one PyTorch thread per worker; training runs on an NVIDIA RTX 4070 Ti SUPER with 16 GB memory. The host has an Intel Core i9-12900KF and 64 GB RAM. Candidate training limits the allocator to half the GPU memory and uses automatic mixed precision. Python is 3.12.14. Exact dependency versions, checkpoint identifiers, and protocol artifacts accompany the manuscript in the evidence manifest.

The main source artifacts are the frozen standard report [6], the collection protocol and result JSON files [7], the no-guess scorecard [8], and implementation modules for the solver, hybrid agent, graphical adapter, replay model, and counterfactual training. They document more than aggregate win rates: deterministic seed assignment, complete trajectories, split checks, per-pair outcomes, and hash verification support inspection. The private project archive includes source, available checkpoints, reports, and retained research datasets with SHA-256 manifests. Public redistribution permissions for imported human replay data have not been established. Those data remain access-restricted; this manuscript grants no redistribution permission. Publication checks found that five of the six files frozen in the September 4 report still match their recorded hashes; the current solver does not. The exact earlier solver snapshot and original per-board rows of that older standard experiment were not available. The aggregate report is preserved unchanged, and current-source reruns must be identified separately. Complete per-pair records are retained for the September 6 candidate test. This manuscript does not assert an independent public replication.

The preserved standard report supports improved performance over the internal V32 baseline under its stated protocol, subject to the historical evidence and wins-only timing limitations. The deployment-aligned update shows the complementary lesson: data collection can work correctly, validation error can decrease, and an independent gameplay gain can still remain unproven. The selected model is therefore unchanged. Further progress should be judged by independent full-game outcomes and the number of informative, policy-relevant decisions, with replay and action provenance used to investigate failures rather than treating every loss as a training error.

### References
[1] R. Kaye. Minesweeper is NP-complete. The Mathematical Intelligencer, 22(2), 9-15, 2000. doi:10.1007/BF03025367. Author manuscript: https://academic.timwylie.com/17CSCI4341/minesweeper_kay.pdf

[2] Y. P. Sinha, P. Malviya, and R. K. Nayak. Fast constraint satisfaction problem and learning-based algorithm for solving Minesweeper. arXiv:2105.04120, 2021. https://arxiv.org/abs/2105.04120

[3] M. Hamza Sajjad. Neural Network Learner for Minesweeper. arXiv:2212.10446, 2022. https://arxiv.org/abs/2212.10446

[4] S. Ross, G. Gordon, and D. Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. Proceedings of AISTATS, PMLR 15, 627-635, 2011. https://proceedings.mlr.press/v15/ross11a.html

[5] M. W. Fagerland, S. Lydersen, and P. Laake. The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional. BMC Medical Research Methodology, 13, 91, 2013. https://doi.org/10.1186/1471-2288-13-91

[6] AI Minesweeper Project. V44 frozen blind standard test. Local protocol and result report, 4 September 2026. experiments/replay-agent-v44-blind-test.md. Frozen seed and six artifact hashes recorded in the report.

[7] AI Minesweeper Project. Local self-play and candidate-policy experiment. Local technical report, protocol, and results, 6 September 2026. experiments/local-selfplay-20260906.md; local-selfplay-20260906-protocol.json; local-selfplay-20260906-result.json. Historical context: replay-agent-v33-v34.md; replay-agent-v35-v39.md; replay-agent-v40-v41.md.

[8] AI Minesweeper Project. Replay agent V42-V44. Local technical report, 4 September 2026. experiments/replay-agent-v42-v43.md. Recorded no-guess scorecard and development comparisons.