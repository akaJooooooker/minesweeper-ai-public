"""Multiprocess self-play using the same public-board engine as the local GUI.

All finished wins and losses are retained. Optional counterfactual branches
label only non-exact decisions where the deployed neural model can intervene.
This collector never trains or edits a checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import secrets
import time

from .desktop_live import LiveDecisionEngine, _information_score
from .game import UNKNOWN, GameStatus, MinesweeperGame
from .local_feedback import build_feedback
from .local_game import LocalGameSession, TERMINAL
from .replay_counterfactual import CounterfactualCandidate, CounterfactualExample, _candidate_scores
from .replay_deployment import load_replay_deployment

SPECS = {"beginner": (9, 9, 10), "intermediate": (16, 16, 40),
         "expert": (30, 16, 99), "evil": (30, 20, 130)}
_WORKER = None


def board_seed(seed, mode, index):
    key = f"local-selfplay-v1:{seed}:{mode}:{index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _init_worker(config_path, cpu_threads):
    global _WORKER
    agent, config = load_replay_deployment(config_path, device="cpu", cpu_threads=cpu_threads)
    _WORKER = (agent, config)


class _DecisionProbe:
    def __init__(self, agent):
        self.agent = agent
        self.last = None

    def choose_move(self, *args, **kwargs):
        self.last = self.agent.choose_move(*args, **kwargs)
        return self.last


def _apply(game, action):
    if action.coord is None or action.action == "stuck":
        raise RuntimeError("Self-play stopped before a terminal outcome: " + action.reason)
    if action.action == "reveal":
        game.reveal(action.coord)
    elif action.action == "flag":
        game.toggle_flag(action.coord)
    elif action.action == "chord":
        game.chord(action.coord)
    else:
        raise RuntimeError("Unsupported live action")


def _continue(game, agent):
    engine = LiveDecisionEngine(agent, use_flags=False)
    guesses = 0
    for _ in range(game.width * game.height * 2):
        if game.status in TERMINAL:
            return game.status == GameStatus.WON, guesses
        action = engine.next_action(game.observation, game.mine_count, allow_guess=True)
        guesses += action.mine_probability not in (None, 0.0)
        _apply(game, action)
    raise RuntimeError("Counterfactual exceeded move limit")


def _counterfactual(game, agent, config, decision):
    scores = _candidate_scores(game.observation, game.mine_count, decision.analysis,
                               agent.risk_predictor, neural_blend=config.neural_blend,
                               neural_blend_mode=config.neural_blend_mode)
    ranked = sorted(scores, key=lambda cell: (scores[cell], cell))[:3]
    if decision.coord not in ranked:
        ranked = ranked[:2] + [decision.coord]
    candidates = []
    for coord in ranked:
        branch = game.clone()
        was_mine = game.is_mine(coord)
        branch.reveal(coord)
        won, guesses = _continue(branch, agent)
        candidates.append(CounterfactualCandidate(*coord, float(won), float(scores[coord]),
                                                 was_mine, float(guesses)))
    return candidates


def _play_one(task):
    mode, index, seed, first_zero, output, max_counterfactual = task
    agent, config = _WORKER
    width, height, mines = SPECS[mode]
    game_seed = board_seed(seed, mode, index)
    session = LocalGameSession(MinesweeperGame(width, height, mines, seed=game_seed,
                                              first_click_zero=first_zero))
    probe = _DecisionProbe(agent)
    engine = LiveDecisionEngine(probe, use_flags=False)
    counts = Counter()
    examples = []
    start = time.perf_counter()
    for _ in range(width * height * 2):
        if session.game.status in TERMINAL:
            break
        observation = session.game.observation
        action = engine.next_action(observation, mines, allow_guess=True)
        if action.coord is None or action.action == "stuck":
            raise RuntimeError(f"{mode}/{index}: {action.reason}")
        decision = probe.last
        guess = action.mine_probability not in (None, 0.0)
        if guess:
            counts["guesses"] += 1
            counts["exact_guesses" if action.exact else "nonexact_guesses"] += 1
            counts["neural_guesses"] += "neural" in action.reason
            counts["outcome_policy_guesses"] += "outcome policy" in action.reason
            analysis = decision.analysis
            if analysis is not None and not analysis.safe:
                candidates = [(r, c) for r in range(height) for c in range(width)
                              if observation[r][c] == UNKNOWN]
                old_choice = min(candidates, key=lambda cell: (
                    analysis.probabilities.get(cell, 1.0), -_information_score(cell, observation), cell))
                counts["old_adapter_would_override"] += old_choice != action.coord
            if not action.exact and len(examples) < max_counterfactual:
                candidates = _counterfactual(session.game, agent, config, decision)
                examples.append(CounterfactualExample("pending", len(session.history),
                    [list(row) for row in observation], mines, "simulator_local_selfplay", candidates))
        session.apply(action.action, action.coord, source="V44", decision=dict(
            reason=action.reason, mine_probability=action.mine_probability, exact=action.exact,
            allow_guess=True, use_flags=False))
        if session.game.status == GameStatus.LOST:
            counts["safe_claim_losses"] += action.mine_probability == 0.0
            counts["exact_guess_losses" if action.exact else "nonexact_guess_losses"] += 1
    if session.game.status not in TERMINAL:
        raise RuntimeError(f"{mode}/{index}: move limit")
    payload = build_feedback(session, seed=game_seed, origin="headless_selfplay",
                             model={"name": "V44", "config": asdict(config)})
    split = "validation" if index % 5 == 0 else "train"
    payload["collection"] = dict(run_seed=seed, mode=mode, index=index, split=split,
                                  use_flags=False, diagnostics=dict(counts))
    directory = Path(output) / "episodes"
    directory.mkdir(exist_ok=True)
    with (directory / f"{mode}-{index:06d}.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return dict(mode=mode, index=index, split=split, status=payload["status"],
                seconds=time.perf_counter()-start, counts=dict(counts),
                examples=[{**asdict(example), "game_id": payload["replay"]["game_id"]}
                          for example in examples])


def collect(config_path: Path, output: Path, *, modes, games_per_mode, workers=4,
            cpu_threads=1, seed=None, first_zero=True, counterfactual_states=0):
    if games_per_mode < 1 or workers < 1 or cpu_threads < 1 or counterfactual_states < 0:
        raise ValueError("Invalid self-play resource or game count")
    if not modes or len(set(modes)) != len(modes) or any(mode not in SPECS for mode in modes):
        raise ValueError("Modes must be unique supported board names")
    seed = secrets.randbits(63) if seed is None else seed
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config_path = Path(config_path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = config_path.parent / raw["checkpoint"]
    manifest = dict(seed=seed, games_per_mode=games_per_mode, modes=modes, workers=workers,
                    cpu_threads=cpu_threads, first_click_zero=first_zero, use_flags=False,
                    counterfactual_states=counterfactual_states, deployment=raw,
                    deployment_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    split="index divisible by 5 -> validation; remaining -> train",
                    completed=False, training_performed=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tasks = [(mode, index, seed, first_zero, str(output), counterfactual_states)
             for index in range(games_per_mode) for mode in modes]
    summary = {mode: Counter() for mode in modes}
    start = time.perf_counter()
    done = 0
    context = multiprocessing.get_context("spawn")
    with (output / "games.jsonl").open("w", encoding="utf-8") as games_file, \
         (output / "counterfactual-train.jsonl").open("w", encoding="utf-8") as train_file, \
         (output / "counterfactual-validation.jsonl").open("w", encoding="utf-8") as validation_file:
        with ProcessPoolExecutor(workers, mp_context=context, initializer=_init_worker,
                                 initargs=(str(config_path), cpu_threads)) as pool:
            futures = [pool.submit(_play_one, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                examples = result.pop("examples")
                for example in examples:
                    target = validation_file if result["split"] == "validation" else train_file
                    target.write(json.dumps(example, separators=(",", ":")) + "\n")
                    target.flush()
                counts = summary[result["mode"]]
                counts.update(result["counts"])
                counts["games"] += 1
                counts[result["status"]] += 1
                counts["counterfactual_states"] += len(examples)
                counts["informative_states"] += sum(len({c["outcome"] for c in ex["candidates"]}) > 1
                                                   for ex in examples)
                games_file.write(json.dumps(result, separators=(",", ":")) + "\n")
                games_file.flush()
                done += 1
                if done % max(1, len(tasks)//16) == 0 or done == len(tasks):
                    print(json.dumps(dict(completed=done, total=len(tasks),
                                          seconds=round(time.perf_counter()-start, 2))), flush=True)
    manifest.update(completed=True, summary={key:dict(value) for key,value in summary.items()},
                    elapsed_seconds=time.perf_counter()-start)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("models/replay-agent-v44.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=SPECS, default=["expert", "evil"])
    parser.add_argument("--games-per-mode", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--safe-first-only", action="store_true")
    parser.add_argument("--counterfactual-states", type=int, default=0)
    args = parser.parse_args()
    result = collect(args.config, args.output_dir, modes=args.modes, games_per_mode=args.games_per_mode,
                     workers=args.workers, cpu_threads=args.cpu_threads, seed=args.seed,
                     first_zero=not args.safe_first_only, counterfactual_states=args.counterfactual_states)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
