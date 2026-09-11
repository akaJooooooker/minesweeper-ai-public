"""Hybrid agent: exact deductions first, calibrated guesses only when needed."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Protocol, Sequence

from .game import UNKNOWN, Coord, GameStatus, MinesweeperGame, validate_observation
from .solver import Analysis, ConstraintSolver


@dataclass(frozen=True)
class Decision:
    action: Literal["reveal", "stuck"]
    coord: Coord | None
    mine_probability: float | None
    reason: str
    exact: bool
    analysis: Analysis | None = None


@dataclass(frozen=True)
class PlayResult:
    status: GameStatus
    won: bool
    reveals: int
    guesses: int
    flags: int
    chords: int
    escalations: int
    stuck: bool

    @property
    def clicks(self) -> int:
        return self.reveals + self.flags + self.chords


class RiskPredictor(Protocol):
    def predict(
        self,
        observation: Sequence[Sequence[int]],
        total_mines: int,
    ) -> Mapping[Coord, float]: ...


class HybridAgent:
    def __init__(
        self,
        solver: ConstraintSolver | None = None,
        risk_predictor: RiskPredictor | None = None,
        fallback_solver: ConstraintSolver | None = None,
        neural_blend: float = 1.0,
        neural_blend_mode: Literal["probability", "rank"] = "probability",
        policy_blend: float = 0.0,
        policy_top_k: int = 0,
        policy_min_advantage: float = 0.0,
        policy_min_board_cells: int = 0,
        proof_solver: ConstraintSolver | None = None,
    ) -> None:
        self.solver = solver or ConstraintSolver()
        self.risk_predictor = risk_predictor
        self.fallback_solver = fallback_solver
        self.proof_solver = proof_solver
        if not 0.0 <= neural_blend <= 1.0:
            raise ValueError("neural_blend must be between 0 and 1")
        self.neural_blend = neural_blend
        if neural_blend_mode not in {"probability", "rank"}:
            raise ValueError("neural_blend_mode must be probability or rank")
        self.neural_blend_mode = neural_blend_mode
        if not 0.0 <= policy_blend <= 1.0:
            raise ValueError("policy_blend must be between 0 and 1")
        self.policy_blend = policy_blend
        if policy_top_k < 0:
            raise ValueError("policy_top_k must be non-negative")
        self.policy_top_k = policy_top_k
        if not 0.0 <= policy_min_advantage <= 1.0:
            raise ValueError("policy_min_advantage must be between 0 and 1")
        self.policy_min_advantage = policy_min_advantage
        if policy_min_board_cells < 0:
            raise ValueError("policy_min_board_cells must be non-negative")
        self.policy_min_board_cells = policy_min_board_cells
        if (
            fallback_solver is not None
            and fallback_solver.max_component_cells <= self.solver.max_component_cells
        ):
            raise ValueError("fallback solver must have a larger component budget")
        if proof_solver is not None:
            if fallback_solver is None:
                raise ValueError("proof solver requires a fallback solver")
            if proof_solver.max_component_cells <= fallback_solver.max_component_cells:
                raise ValueError("proof solver must have a larger component budget")


    def choose_move(
        self,
        observation: Sequence[Sequence[int]],
        total_mines: int,
        *,
        allow_guess: bool = True,
    ) -> Decision:
        height, width = validate_observation(observation)
        unknown = [
            (row, col)
            for row in range(height)
            for col in range(width)
            if observation[row][col] == UNKNOWN
        ]
        if not unknown:
            return Decision("stuck", None, None, "no covered cells remain", True)

        revealed_count = sum(cell >= 0 for row in observation for cell in row)
        if revealed_count == 0:
            center = (height // 2, width // 2)
            coord = center if observation[center[0]][center[1]] == UNKNOWN else unknown[0]
            return Decision("reveal", coord, 0.0, "configured safe first click", True)

        direct_fallback = not allow_guess and self.fallback_solver is not None
        analysis_solver = self.fallback_solver if direct_fallback else self.solver
        assert analysis_solver is not None
        analysis = analysis_solver.analyse(observation, total_mines)
        if analysis.contradiction:
            return Decision(
                "stuck",
                None,
                None,
                f"invalid board: {analysis.contradiction}",
                analysis.exact,
                analysis,
            )

        used_fallback = direct_fallback
        if (
            self.fallback_solver is not None
            and not direct_fallback
            and not analysis.exact
            and not analysis.safe
        ):
            fallback = self.fallback_solver.analyse(observation, total_mines)
            if fallback.contradiction:
                return Decision(
                    "stuck",
                    None,
                    None,
                    f"fallback solver found invalid board: {fallback.contradiction}",
                    fallback.exact,
                    fallback,
                )
            if fallback.safe or fallback.exact:
                analysis = fallback
                used_fallback = True


        if (
            self.proof_solver is not None
            and not allow_guess
            and not analysis.exact
        ):
            proof = self.proof_solver.analyse(observation, total_mines)
            if proof.contradiction:
                return Decision(
                    "stuck",
                    None,
                    None,
                    f"proof solver found invalid board: {proof.contradiction}",
                    proof.exact,
                    proof,
                )
            if proof.safe or proof.exact:
                analysis = proof
                used_fallback = True
        safe = [coord for coord in analysis.safe if observation[coord[0]][coord[1]] == UNKNOWN]
        if safe:
            coord = max(safe, key=lambda item: (_information_score(item, observation), item))
            reason = (
                "logically forced safe after exact escalation"
                if used_fallback
                else "logically forced safe"
            )
            return Decision("reveal", coord, 0.0, reason, analysis.exact, analysis)

        if not allow_guess:
            return Decision("stuck", None, None, "no forced safe move", analysis.exact, analysis)

        probabilities = dict(analysis.probabilities)
        used_neural_model = False
        reveal_policy: dict[Coord, float] = {}
        if not analysis.exact and self.risk_predictor is not None:
            prediction = None
            if self.policy_blend > 0.0 and height * width >= self.policy_min_board_cells:
                predict_all = getattr(self.risk_predictor, "predict_all", None)
                if predict_all is not None:
                    prediction = predict_all(observation, total_mines)
            if prediction is None:
                learned = self.risk_predictor.predict(observation, total_mines)
            else:
                learned = prediction.risk
                reveal_policy = {
                    coord: score
                    for (action, coord), score in prediction.policy_logits.items()
                    if action == "reveal"
                }
            eligible_learned = {
                coord: probability
                for coord, probability in learned.items()
                if coord not in analysis.safe and coord not in analysis.mines
            }
            if self.neural_blend_mode == "rank":
                base_ranks = _normalized_ranks(
                    {
                        coord: probabilities.get(coord, probability)
                        for coord, probability in eligible_learned.items()
                    }
                )
                learned_ranks = _normalized_ranks(eligible_learned)
                probabilities.update(
                    {
                        coord: self.neural_blend * learned_ranks[coord]
                        + (1.0 - self.neural_blend) * base_ranks[coord]
                        for coord in eligible_learned
                    }
                )
            else:
                probabilities.update(
                    {
                        coord: self.neural_blend * probability
                        + (1.0 - self.neural_blend) * probabilities.get(coord, probability)
                        for coord, probability in eligible_learned.items()
                    }
                )
            used_neural_model = True
        candidates = [coord for coord in unknown if coord in probabilities]
        if not candidates:
            return Decision("stuck", None, None, "solver produced no candidate", False, analysis)
        selection_scores = probabilities
        used_policy_model = False
        eligible_policy = {coord: reveal_policy[coord] for coord in candidates if coord in reveal_policy}
        if self.policy_top_k and len(eligible_policy) > self.policy_top_k:
            top_risk = sorted(
                eligible_policy,
                key=lambda item: (probabilities[item], -_information_score(item, observation), item),
            )[: self.policy_top_k]
            eligible_policy = {coord: eligible_policy[coord] for coord in top_risk}

        if self.policy_min_advantage > 0.0 and len(eligible_policy) >= 2:
            risk_best = min(
                eligible_policy,
                key=lambda item: (
                    probabilities[item],
                    -_information_score(item, observation),
                    item,
                ),
            )
            policy_best = max(
                eligible_policy,
                key=lambda item: (eligible_policy[item], item),
            )
            policy_advantage = _sigmoid(eligible_policy[policy_best]) - _sigmoid(
                eligible_policy[risk_best]
            )
            if (
                not math.isfinite(policy_advantage)
                or policy_advantage < self.policy_min_advantage
            ):
                eligible_policy = {}

        if self.policy_blend > 0.0 and len(eligible_policy) >= 2:
            risk_ranks = _normalized_ranks({coord: probabilities[coord] for coord in eligible_policy})
            policy_ranks = _normalized_ranks(eligible_policy)
            selection_scores = {
                coord: (
                    (1.0 - self.policy_blend) * risk_ranks[coord]
                    + self.policy_blend * (1.0 - policy_ranks[coord])
                )
                for coord in eligible_policy
            }
            used_policy_model = True
        coord = min(
            selection_scores,
            key=lambda item: (
                selection_scores[item],
                -_information_score(item, observation),
                item,
            ),
        )
        probability = probabilities[coord]
        if analysis.exact:
            reason = "minimum exact mine probability"
        elif used_policy_model:
            reason = "minimum neural risk with outcome policy"
        elif used_neural_model:
            reason = "minimum neural risk after exact deductions"
        else:
            reason = "minimum estimated risk"
        return Decision("reveal", coord, probability, reason, analysis.exact, analysis)

    def play(self, game: MinesweeperGame, *, allow_guess: bool = True) -> PlayResult:
        reveals = 0
        guesses = 0
        flags = 0
        chords = 0
        escalations = 0
        stuck = False
        proven_flags: set[Coord] = set()
        move_limit = game.width * game.height * 2

        while (
            game.status not in (GameStatus.WON, GameStatus.LOST)
            and reveals + chords < move_limit
        ):
            if game.status != GameStatus.READY:
                direct_fallback = not allow_guess and self.fallback_solver is not None
                analysis_solver = self.fallback_solver if direct_fallback else self.solver
                assert analysis_solver is not None
                analysis = analysis_solver.analyse(game.observation, game.mine_count)
                if analysis.contradiction:
                    stuck = True
                    break
                if direct_fallback:
                    escalations += 1
                if (
                    self.fallback_solver is not None
                    and not direct_fallback
                    and not analysis.exact
                    and not analysis.safe
                ):
                    fallback = self.fallback_solver.analyse(
                        game.observation,
                        game.mine_count,
                    )
                    if fallback.contradiction:
                        stuck = True
                        break
                    analysis = fallback
                    escalations += 1
                if (
                    self.proof_solver is not None
                    and not allow_guess
                    and not analysis.exact
                ):
                    proof = self.proof_solver.analyse(
                        game.observation,
                        game.mine_count,
                    )
                    if proof.contradiction:
                        stuck = True
                        break
                    analysis = proof
                    escalations += 1
                for coord in sorted(analysis.mines):
                    if game.observation[coord[0]][coord[1]] == UNKNOWN:
                        if game.toggle_flag(coord):
                            flags += 1
                            proven_flags.add(coord)

                chord_coord = _best_proven_chord(game, proven_flags)
                if chord_coord is not None:
                    opened = game.chord(chord_coord)
                    if opened:
                        chords += 1
                        continue


            decision = self.choose_move(
                game.observation,
                game.mine_count,
                allow_guess=allow_guess,
            )
            if decision.action == "stuck" or decision.coord is None:
                stuck = True
                break
            if "escalation" in decision.reason:
                escalations += 1
            if decision.mine_probability not in (None, 0.0):
                guesses += 1
            game.reveal(decision.coord)
            reveals += 1

        return PlayResult(
            status=game.status,
            won=game.status == GameStatus.WON,
            reveals=reveals,
            guesses=guesses,
            flags=flags,
            chords=chords,
            escalations=escalations,
            stuck=stuck,
        )


def _information_score(coord: Coord, observation: Sequence[Sequence[int]]) -> int:
    height = len(observation)
    width = len(observation[0])
    row, col = coord
    score = 0
    for next_row in range(max(0, row - 1), min(height, row + 2)):
        for next_col in range(max(0, col - 1), min(width, col + 2)):
            if observation[next_row][next_col] >= 0:
                score += 1
    return score


def _best_proven_chord(
    game: MinesweeperGame,
    proven_flags: set[Coord],
) -> Coord | None:
    observation = game.observation
    best: Coord | None = None
    best_covered = 0
    for row in range(game.height):
        for col in range(game.width):
            clue = observation[row][col]
            if clue <= 0:
                continue
            neighbours = tuple(game.neighbours((row, col)))
            flagged = {
                coord
                for coord in neighbours
                if observation[coord[0]][coord[1]] == -2
            }
            if len(flagged) != clue or not flagged.issubset(proven_flags):
                continue
            covered = sum(
                observation[n_row][n_col] == UNKNOWN
                for n_row, n_col in neighbours
            )
            if covered > best_covered:
                best = (row, col)
                best_covered = covered
    return best


def _normalized_ranks(values: Mapping[Coord, float]) -> dict[Coord, float]:
    if not values:
        return {}
    ordered = sorted(set(values.values()))
    if len(ordered) == 1:
        return {coord: 0.0 for coord in values}
    rank = {value: index / (len(ordered) - 1) for index, value in enumerate(ordered)}
    return {coord: rank[value] for coord, value in values.items()}


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
