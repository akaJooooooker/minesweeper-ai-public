# Minesweeper AI | Offline play, V44 and replay analysis

[繁體中文](README.md) · **English**

Play Minesweeper yourself, let V44 play, or revisit a game to inspect individual decisions. This project combines a Windows desktop application, a hybrid agent led by exact constraint inference, and a research release with recorded wins, losses and paired evaluation results.

**[Download Windows v0.1.0](https://github.com/akaJooooooker/minesweeper-ai-public/releases/tag/v0.1.0)** · [Research paper](output/research/minesweeper-research-paper.pdf) · [Architecture](docs/ARCHITECTURE.md) · [Reproduction guide](docs/REPRODUCIBILITY.md)

![Offline board, controls and game statistics](docs/assets/local-gui-normal.png)

> The current binary is a Windows x64 public preview with a Traditional Chinese interface. It includes Python, CPU-only PyTorch and the V44 model. Play and analysis run locally, without an account, API key or cloud AI service.

## What is included?

Players can download the application, play manually, watch the AI, and review whether a move had safer alternatives. Developers and researchers can inspect the solver, deployment settings, checkpoints and retained evaluation evidence.

| Component | Capabilities |
| --- | --- |
| Local game | Four presets, custom boards, protected openings, flags, chords and live statistics |
| V44 autoplay | Start, stop or take one step; resume manually; configure guessing and flag use |
| Replay and analysis | Seekable board timeline, stepping, playback, pre-action mine risk and safer cells |
| Game records | Wins and losses, action order, timestamps and available AI decision metadata |
| Research release | Paper, historical reports, local simulation data and complete newer paired outcomes |
| Source code | Environment, solver, model inference, training utilities, GUI and desktop observer |

## Download and run

### Windows application

1. Open the [v0.1.0 release](https://github.com/akaJooooooker/minesweeper-ai-public/releases/tag/v0.1.0) and download **`MinesweeperAI-v0.1.0-windows-x64.zip`**, approximately 173 MB.
2. **Extract the entire ZIP.** Keep the `_internal` directory beside `MinesweeperAI.exe`. The extracted package is approximately 495 MB.
3. Run `MinesweeperAI.exe`. Play manually or press **F8** to start the AI.

No separate Python, CUDA or NVIDIA GPU installation is required. The binary uses CPU inference; the first AI action needs to load the model.

This is an unsigned Windows x64 preview. Windows may identify the publisher as unknown. Use this project's release page and compare your download with `SHA256SUMS.txt`:

```powershell
Get-FileHash .\MinesweeperAI-v0.1.0-windows-x64.zip -Algorithm SHA256
```

To update, extract the new version into another directory. Binary-edition game records live in your Windows user profile, so replacing the program folder does not overwrite them.

### Which release files do I need?

| File | Purpose |
| --- | --- |
| `MinesweeperAI-v0.1.0-windows-x64.zip` | Everything needed to play, including V44 and dependency notices |
| `MinesweeperAI-v0.1.0-source.zip` | Source snapshot at v0.1.0; the main branch has the latest documentation |
| `minesweeper-research-paper.pdf` / `.docx` | Research manuscript in PDF and editable Word formats |
| `local-experiment-data-20260906.zip` | Approximately 12 MB of local experiment records; unnecessary for ordinary play |
| `PUBLICATION_NOTES.md` | Public scope, historical evidence limitations and packaging notes |
| `SHA256SUMS.txt` / `windows-binary-check.json` | Download checksums and the limited binary validation record |

## Controls and statistics

| Input | Action |
| --- | --- |
| Left-click a hidden cell | Reveal immediately on press |
| Right-click | Toggle a flag |
| Hold a revealed number | Preview adjacent hidden, unflagged cells |
| Release on the same number | Chord if the adjacent flag count matches; incorrect flags can still cause a loss |
| Middle-click or left/right combination | Chord |
| F2 | New game |
| F8 | Start AI |
| F9 / Esc | Stop AI; manual play can continue |

Presets are beginner **9×9 / 10 mines**, intermediate **16×16 / 40**, expert **30×16 / 99**, and evil **30×20 / 130**, with custom sizes available. Dimensions here are **columns × rows**.

The default first click opens a zero region. Disabling that option still guarantees a safe first click. The same seed, settings and first click reproduce a board. Random boards are **not guaranteed to be solvable without guessing**. Disabling AI guessing stops it when no provably safe action is available.

The statistics panel shows:

- **Time:** starts at the first effective reveal and freezes on completion; time spent in replay is excluded.
- **3BV:** a board measure based on connected zero openings and safe numbered cells not covered by those openings.
- **3BV/s:** completed 3BV divided by elapsed play time.
- **Clicks:** effective and redundant actions.
- **Efficiency:** completed 3BV divided by total clicks; chording can produce values above 100%.

These are explicitly defined local metrics, not a claim to reproduce another service's internal scoring. The [detailed controls guide](docs/local-game.md) is currently in Traditional Chinese.

## Inspect what was knowable before a move

![Replay timeline and on-demand action assessment](docs/assets/local-replay-analysis.png)

Select **重播本局** (replay current game), or load an exported JSON record, to seek directly on the board, step backward or forward, and play the timeline. **返回對局** returns to the preserved live session; press F8 to resume AI play.

Select a step and press **分析這一步** (analyze this step). Analysis runs only when requested and is cached within that replay.

| Pre-action assessment | Retrospective effects |
| --- | --- |
| Chosen cell's mine risk and exact/estimated source | Whether the action changed the board |
| Safer unflagged alternatives | How many safe cells actually opened |
| Probability that a flagged cell contains a mine | Recorded action and available decision metadata |
| Whether all chord targets are provably safe | Outcome is not treated as proof of decision quality |

Risk inference uses only the **preceding revealed observation and total mine count**, not hidden mines, seeds or future outcomes. Player flags are treated as unknown during analysis so that an incorrect flag cannot manufacture a false proof of safety. For a chord that is not entirely proven safe, the panel reports the largest target marginal risk, not a joint explosion probability.

“Exact” refers to compatible mine configurations under the stated model. Incomplete enumeration is labeled as an approximation or model estimate. Neither is a guarantee of optimal play, expected information gain or whole-game success. Losing a necessary guess does not automatically make it a bad decision.

## How V44 selects an action

V44 combines **constraint inference with selective neural estimates**. It does not query a large language model for each move.

1. Derive local constraints, subset deductions and proven safe/mined cells from revealed clues.
2. Analyze connected frontier components and account for the global remaining-mine count, increasing enumeration limits when needed.
3. Prioritize safe actions. When probabilities are exact, retain the exact-inference selection path.
4. Only when no safe action exists and inference remains non-exact, allow neural risk and a gated outcome policy to influence selection.

The V44 deployment uses the V40 checkpoint. A deployment version and a neural-weight version are different things; see the [selected configuration](models/replay-agent-v44.json).

The game environment owns hidden mines for gameplay and reconstruction, but the playing agent receives only public observations, the mine count and action options. The GUI and desktop observer share `LiveDecisionEngine`. Stale decisions are discarded after stopping, starting another game or manually changing the board. [Architecture](docs/ARCHITECTURE.md)

## Research results and their scope

The manuscript is **Exact Inference and Selective Learning for Minesweeper: Paired Evaluation and Deployment Aligned Counterfactual Adaptation**. [PDF](output/research/minesweeper-research-paper.pdf) · [Word](output/research/minesweeper-research-paper.docx) · [Source](output/research/minesweeper-study.md)

### Historical standard comparison

Each mode contains 2,000 pairs with safe-only openings and proven flags/chords:

| Mode | V32 wins | V44 wins | Win-rate difference |
| --- | ---: | ---: | ---: |
| Beginner | 1,669 / 2,000 | 1,668 / 2,000 | −0.05 percentage points |
| Intermediate | 1,371 / 2,000 | 1,388 / 2,000 | +0.85 percentage points |
| Expert | 657 / 2,000 | 690 / 2,000 | +1.65 percentage points |

These numbers come from the [preserved historical report](experiments/replay-agent-v44-blind-test.md). The original per-board rows and exact earlier solver snapshot are unavailable. A current-source rerun must not be described as complete recovery of that experiment. Historical speed summaries use each agent's own winning subset, rather than matched timings on identical won boards.

### New comparison under GUI rules

A separate collection retained **2,000 local games**: 870 wins and 1,130 losses. It produced 194 counterfactual states, only 34 of which distinguished candidate outcomes. After training, a new **5,000-pair** test compared the candidate with V44:

| Agent | Wins | Win rate |
| --- | ---: | ---: |
| V44 | 2,189 / 5,000 | 43.78% |
| New candidate | 2,191 / 5,000 | 43.82% |

The difference is **+0.04 percentage points**, with exact McNemar **p = 0.790527**. The candidate was not promoted. Validation Brier score improved from approximately 0.2581 to 0.2532, but the test did not establish a gameplay improvement. [Experiment report](experiments/local-selfplay-20260906.md)

This test uses guaranteed-zero openings, no flags and the GUI decision engine. Its win rates must not be directly compared with the historical standard protocol. Another result, **1,696 / 1,696** solved winning no-guess replays, measures coverage of a selected corpus—not a universal guarantee on random or all no-guess boards.

No matched evaluation against the strongest external solvers has been completed, and no state-of-the-art claim is made. [Public evidence notes](docs/PUBLICATION_NOTES.md)

## Records, data and privacy

| Edition | Automatically saved games |
| --- | --- |
| Windows binary | `%LOCALAPPDATA%\MinesweeperAI\local-feedback` |
| Source checkout | `data/local-feedback/` under the project |

Completed wins and losses are saved with reconstruction information and actions. Incomplete games can be exported manually. **Playing, replaying and saving never automatically train or update the model**, and records are not uploaded.

The public dataset contains only the specified local simulations and paired outcomes. Imported third-party human replays, personal GUI games, account data and original machine traceback logs are excluded. Checkpoint lineage includes human replay and simulator data; publishing weights does not make the complete historical training corpus available. [Data and licensing scope](NOTICE.md)

## Run from source

Use **Python 3.12 with Tk**, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/cpu.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\minesweeper-local.exe
```

After setup, `啟動本機踩地雷.cmd` launches the local game. Binary users do not need these steps.

The source-only screen observer is started by `啟動桌面版.cmd`. For the local board, select dark/standard, match the board settings and calibrate the corners. It is not a general plugin marketplace. Keep the board visible and its position/scale stable; avoid running it alongside built-in autoplay. [Observer compatibility](docs/desktop-bot.md)

Linux source use requires system Tk and `.venv/bin/` paths. No Linux or macOS binaries are provided in this release. Desktop mouse control is Windows-specific.

### Repository map

| Path | Contents |
| --- | --- |
| `src/minesweeper_ai/` | Game, solver, neural inference, GUI, replay and research tools |
| `models/` | V44, V32 and candidate deployments with corresponding weights |
| `experiments/` | Protocols, outcomes and historical reports |
| `output/research/` | Paper, editable document, manuscript source and statistics |
| `docs/` | Controls, architecture, reproduction and evidence boundaries |
| `packaging/` | Windows build configuration and instructions |
| `tests/` / `artifacts/` | Software tests and limited release validation records |

[Windows build](packaging/README.md) · [Restore data and reproduce experiments](docs/REPRODUCIBILITY.md)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# Native GUI checks require an interactive desktop.
$env:RUN_LOCAL_GUI_TESTS = '1'
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_local_gui.py -v
```

The actual Windows binary passed a small check from an unrelated working directory: manual flags, one completed V44 game, saving, replay assessment and return to the original session. This is not cross-machine validation or a new efficacy benchmark. [Record](artifacts/windows-binary-check.json)

## FAQ

**Does it learn as I play?**

No. Records can be analyzed or used in a separate training job, but a model update should still be evaluated on independent boards.

**Why can the AI lose?**

Some observations leave no certain safe move. Even an exact positive-risk guess can hit a mine. A single outcome is not enough to judge decision quality.

**Why does analysis not show the increase in whole-game win probability?**

The GUI reports pre-action risk, safer options and observed effects. Long-term win value and expected information gain require additional branch or compatible-board simulations and are not provided as GUI features.

**The application will not start, or the model does not load.**

Check that the entire ZIP was extracted and `_internal` remains next to the executable. If a startup log was produced, it is at `%LOCALAPPDATA%\MinesweeperAI\startup-error.log`. Model loading errors are also shown in the interface. Include your Windows version, application version, steps and error text when reporting a problem.

**Is there an English interface or adjustable scaling?**

The current GUI is Traditional Chinese at a fixed 150% scale, with scrolling for large boards. This English README does not change the application's interface language.

## Contributing and license

Use [Issues](https://github.com/akaJooooooker/minesweeper-ai-public/issues) for bugs and suggestions, or propose code and documentation improvements. Attaching a replay is optional; inspect it before sharing. Account credentials and private data are not needed.

Original code and documentation use [MIT](LICENSE). Project-authored weights and local simulation records are provided within the scope described in [NOTICE](NOTICE.md). Dependencies retain their own licenses, included in `THIRD_PARTY_LICENSES` in the Windows ZIP. For research citations, identify the release or commit used and see [CITATION.cff](CITATION.cff).
