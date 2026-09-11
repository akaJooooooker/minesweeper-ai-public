# Public evidence and reproduction

Install from README. Download `local-experiment-data-20260906.zip` from v0.1.0, check `SHA256SUMS.txt`, and extract into the repository root. It contains `data/selfplay-v44-batch1/`, `data/selfplay-v44-candidate-paired/` and a per-file hash manifest. The game does not require this download.

Data: 2,000 simulated games with complete episodes; 158/36 counterfactual train/validation states; 5,000 paired test rows. Imported human data, most historical checkpoints and machine-specific traceback logs are excluded. See [publication notes](PUBLICATION_NOTES.md).

A new GUI-aligned run on the saved cohort:

```powershell
.\.venv\Scripts\python.exe scripts/compare_local_deployments.py --reference models/replay-agent-v44.json --candidate models/replay-agent-local-selfplay-candidate.json --output tmp/new-comparison --seed 175985001720591491 --games 2500 --modes expert evil --workers 4
```

Do not tune on this cohort then call it a fresh independent test. This is first-zero, no-flag play.

Historical safe-only protocol with CURRENT source:

```powershell
.\.venv\Scripts\python.exe -m minesweeper_ai.deployment_compare --reference models/replay-agent-v32.json --candidate models/replay-agent-v44.json --games 2000 --seed 6739657020312379691 --device cpu
```

This is a new current-source run, not recovery of missing historical evidence.

Collection and training entry points: `local_selfplay.py` and `replay_counterfactual_training.py`; use `--help` and new output paths. Collection does not train. Original settings remain in the experiment report and checkpoint metadata. Full V44 training needs excluded historical data.

PDF/DOCX are delivered artifacts. Optional rebuild: install `requirements/paper.txt`, restore local data and run `scripts/build_research_paper.py`; export DOCX via Word/LibreOffice. A rebuild requires visual QA.
