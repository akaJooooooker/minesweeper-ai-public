# V44 frozen blind standard test

Pre-registered: 2026-09-04 Asia/Taipei, before running the test.

## Frozen subject

- Deployment: `models/replay-agent-v44.json`
- Reference: `models/replay-agent-v32.json`
- V44 deployment SHA-256: `6DCA016EB53DF7996F38313B3EF431E8DF7D48420D7047D65046EF344406939E`
- Checkpoint SHA-256: `F8F827AFA6C3A67D0B5909978B2F285D8430BABDCC1F8E274F2B8AE391851FD4`
- Agent SHA-256: `7A1915CD5C36FFF873E3B2BA6CBAF2EDAC749F4CB102BF21AB286D96A88F9539`
- Solver SHA-256: `F4896105E70F2142EDECC2A64D3613A11A52C66A38B0692DD5C066A2EC4C1DBD`
- Deployment loader SHA-256: `A4762DDDF9C0A98AE7B64CEC9586293281C8DA19A543B52AADD3E12352D60079`
- Paired evaluator SHA-256: `2673603FFF8D5D77143395586FA2FDB49C3AD5E7C7E62B68902FF0158EA7BAF3`

## Test contract

- One-time cryptographic random seed: `6739657020312379691`
- 2000 paired boards per standard mode; 6000 boards total.
- Both deployments receive the identical board seed in each pair.
- Standard first click is safe only, freely selected, and is not forced to be zero.
- Report win rate, discordant wins, exact McNemar p, guesses, 3BV/s, and IOE.
- No V44 code, weights, or settings may be changed in response to this test.

## Pre-registered acceptance gate

V44 passes only if all conditions hold:

1. No mode has a V44 win-rate regression worse than 0.5 percentage points.
2. V44 has more wins than V32 across intermediate and expert combined.
3. V44 retains at least 90% of V32 3BV/s in every mode.
4. The full unit-test suite remains green after the benchmark.

## Results

Passed.

| Mode | V32 wins | V44 wins | Delta | V32-only | V44-only | McNemar p | V32 3BV/s | V44 3BV/s | Speed retained | V44 IOE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Beginner | 1669 (83.45%) | 1668 (83.40%) | -0.05 pp | 1 | 0 | 1.000000 | 4064.14 | 4070.76 | 100.16% | 0.7234 |
| Intermediate | 1371 (68.55%) | 1388 (69.40%) | +0.85 pp | 5 | 22 | 0.001514 | 1960.18 | 1808.76 | 92.28% | 0.8171 |
| Expert | 657 (32.85%) | 690 (34.50%) | +1.65 pp | 22 | 55 | 0.000217 | 1103.78 | 1054.61 | 95.55% | 0.8699 |

- Total wins: V32 3697; V44 3746; V44 +49.
- Intermediate plus expert wins: V32 2028; V44 2078; V44 +50.
- Total guesses: V32 15335; V44 15205; V44 used 130 fewer guesses.
- Gate 1 passed: worst per-mode regression was 0.05 pp, within the 0.5 pp limit.
- Gate 2 passed: V44 gained 50 intermediate-plus-expert wins.
- Gate 3 passed: the lowest retained speed was 92.28%, above 90%.
- Gate 4 passed: 85/85 unit tests passed after the benchmark.
- All six frozen SHA-256 values matched after the run.

This blind test covers the local simulator under the correct standard safe-only first-click rule. It does not count as a live Minesweeper Online UI test; browser startup was blocked by the local browser sandbox before the site could be opened.
