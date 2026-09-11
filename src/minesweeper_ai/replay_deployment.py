"""Load the selected solver-shielded Replay deployment configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

from .agent import HybridAgent
from .replay_inference import ReplayAgentPredictor
from .solver import ConstraintSolver


@dataclass(frozen=True)
class ReplayDeploymentConfig:
    kind: str
    checkpoint: str
    mode: Literal["standard", "no_guess"]
    solver_cells: int
    neural_blend: float
    neural_blend_mode: Literal["probability", "rank"]
    fallback_solver_cells: int | None = None
    proof_solver_cells: int | None = None
    policy_top_k: int = 0
    policy_blend: float = 0.0
    policy_min_advantage: float = 0.0
    policy_min_board_cells: int = 0


def load_replay_deployment(
    config_path: str | Path,
    *,
    device: str = "cpu",
    cpu_threads: int = 2,
) -> tuple[HybridAgent, ReplayDeploymentConfig]:
    path = Path(config_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = ReplayDeploymentConfig(
        kind=raw["kind"],
        checkpoint=raw["checkpoint"],
        mode=raw["mode"],
        solver_cells=int(raw["solver_cells"]),
        neural_blend=float(raw["neural_blend"]),
        neural_blend_mode=raw["neural_blend_mode"],
        fallback_solver_cells=(
            int(raw["fallback_solver_cells"])
            if raw.get("fallback_solver_cells") is not None
            else None
        ),
        proof_solver_cells=(
            int(raw["proof_solver_cells"])
            if raw.get("proof_solver_cells") is not None
            else None
        ),
        policy_blend=float(raw.get("policy_blend", 0.0)),
        policy_top_k=int(raw.get("policy_top_k", 0)),
        policy_min_advantage=float(raw.get("policy_min_advantage", 0.0)),
        policy_min_board_cells=int(raw.get("policy_min_board_cells", 0)),
    )
    if config.kind not in {"replay-agent-deployment-v7", "replay-agent-deployment-v20"}:
        raise ValueError(f"unsupported deployment config kind: {config.kind}")
    checkpoint = Path(config.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = path.parent / checkpoint
    predictor = ReplayAgentPredictor(
        checkpoint,
        mode=config.mode,
        device=device,
        cpu_threads=cpu_threads,
    )
    agent = HybridAgent(
        ConstraintSolver(config.solver_cells),
        risk_predictor=predictor,
        fallback_solver=(
            ConstraintSolver(config.fallback_solver_cells)
            if config.fallback_solver_cells is not None
            else None
        ),
        neural_blend=config.neural_blend,
        proof_solver=(
            ConstraintSolver(config.proof_solver_cells)
            if config.proof_solver_cells is not None
            else None
        ),
        neural_blend_mode=config.neural_blend_mode,
        policy_blend=config.policy_blend,
        policy_top_k=config.policy_top_k,
        policy_min_advantage=config.policy_min_advantage,
        policy_min_board_cells=config.policy_min_board_cells,
    )
    return agent, config
