"""Generate Replay traces from the deployed solver-shielded neural policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Literal, Sequence

from .agent import HybridAgent
from .data import BoardSpec
from .game import UNKNOWN, GameStatus, MinesweeperGame
from .replay import ReplayGame, ReplayMove, write_replays
from .replay_inference import ReplayAgentPredictor
from .solver import ConstraintSolver


@dataclass(frozen=True)
class OnPolicyStats:
    replays: int
    wins: int
    losses: int
    moves: int
    guesses: int
    solver_cells: int
    checkpoint: str
    neural_blend: float
    neural_blend_mode: str
    device: str
    cpu_threads: int


def generate_on_policy_replays(
    checkpoint: str | Path,
    output: str | Path,
    *,
    games: int,
    seed: int,
    solver_cells: int = 24,
    neural_blend: float = 0.75,
    neural_blend_mode: Literal["probability", "rank"] = "probability",
    device: str = "cpu",
    cpu_threads: int = 2,
    spec: BoardSpec = BoardSpec(30, 16, 99, True),
) -> OnPolicyStats:
    if games < 1:
        raise ValueError("games must be positive")
    predictor = ReplayAgentPredictor(
        checkpoint, mode="standard", device=device, cpu_threads=cpu_threads
    )
    solver = ConstraintSolver(solver_cells)
    agent = HybridAgent(
        solver,
        risk_predictor=predictor,
        neural_blend=neural_blend,
        neural_blend_mode=neural_blend_mode,
    )
    rng = random.Random(seed)
    replays: list[ReplayGame] = []
    guess_count = 0
    for game_index in range(games):
        game_seed = rng.randrange(2**63)
        game = MinesweeperGame(
            spec.width,
            spec.height,
            spec.mines,
            seed=game_seed,
            first_click_zero=spec.first_click_zero,
        )
        replay, guesses = _record_game(
            game,
            agent,
            solver,
            game_id=f"sim-on-policy-{seed}-{game_index}",
        )
        replays.append(replay)
        guess_count += guesses
    write_replays(replays, output)
    wins = sum(replay.won for replay in replays)
    return OnPolicyStats(
        replays=len(replays),
        wins=wins,
        losses=len(replays) - wins,
        moves=sum(len(replay.moves) for replay in replays),
        guesses=guess_count,
        solver_cells=solver_cells,
        checkpoint=str(Path(checkpoint)),
        neural_blend=neural_blend,
        neural_blend_mode=neural_blend_mode,
        device=device,
        cpu_threads=cpu_threads,
    )

def _record_game(
    game: MinesweeperGame,
    agent: HybridAgent,
    solver: ConstraintSolver,
    *,
    game_id: str,
) -> tuple[ReplayGame, int]:
    moves: list[ReplayMove] = []
    guesses = 0
    elapsed = 0
    move_limit = game.width * game.height * 3
    while game.status not in (GameStatus.WON, GameStatus.LOST) and len(moves) < move_limit:
        if game.status == GameStatus.READY:
            row, col = game.height // 2, game.width // 2
            moves.append(ReplayMove("reveal", row, col, elapsed))
            game.reveal((row, col))
            elapsed += 100
            continue
        analysis = solver.analyse(game.observation, game.mine_count)
        if analysis.contradiction:
            break
        for row, col in sorted(analysis.mines):
            if game.observation[row][col] == UNKNOWN:
                moves.append(ReplayMove("flag", row, col, elapsed))
                game.toggle_flag((row, col))
                elapsed += 100
        decision = agent.choose_move(game.observation, game.mine_count, allow_guess=True)
        if decision.coord is None:
            break
        row, col = decision.coord
        moves.append(ReplayMove("reveal", row, col, elapsed))
        if decision.mine_probability not in (None, 0.0):
            guesses += 1
        game.reveal((row, col))
        elapsed += 100
    mines = [
        [row, col]
        for row in range(game.height)
        for col in range(game.width)
        if game.is_mine((row, col))
    ]
    return (
        ReplayGame(
            game_id=game_id,
            width=game.width,
            height=game.height,
            mines=game.mine_count,
            mode="standard",
            source="simulator_on_policy",
            won=game.status == GameStatus.WON,
            mine_positions=mines,
            moves=moves,
        ),
        guesses,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--solver-cells", type=int, default=24)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--neural-blend", type=float, default=0.75)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument(
        "--neural-blend-mode", choices=("probability", "rank"), default="probability"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stats = generate_on_policy_replays(
        args.checkpoint,
        args.output,
        games=args.games,
        seed=args.seed,
        solver_cells=args.solver_cells,
        neural_blend=args.neural_blend,
        neural_blend_mode=args.neural_blend_mode,
        device=args.device,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps(asdict(stats), separators=(",", ":")))


if __name__ == "__main__":
    main()
