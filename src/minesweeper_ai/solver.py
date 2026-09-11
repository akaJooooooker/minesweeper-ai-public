"""Constraint propagation and exact frontier probabilities for Minesweeper."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import comb
from typing import Iterable, Mapping, Sequence

from .game import FLAGGED, UNKNOWN, Coord, validate_observation

ConstraintMap = dict[frozenset[Coord], int]


@dataclass(frozen=True)
class Analysis:
    """Result produced from public information only."""

    safe: frozenset[Coord]
    mines: frozenset[Coord]
    probabilities: Mapping[Coord, float]
    exact: bool
    model_count: int | None
    frontier_size: int
    contradiction: str | None = None


@dataclass
class _Enumeration:
    variables: tuple[Coord, ...]
    ways: dict[int, int]
    mine_ways: dict[Coord, dict[int, int]]


class ConstraintSolver:
    """Solve forced moves and enumerate small independent frontier components.

    The exact probability calculation combines every component with the global
    remaining-mine count. Components above ``max_component_cells`` fall back to
    a marked, non-exact local estimate so the agent can still make progress.
    """

    def __init__(self, max_component_cells: int = 24) -> None:
        if max_component_cells < 1:
            raise ValueError("max_component_cells must be positive")
        self.max_component_cells = max_component_cells

    def analyse(
        self,
        observation: Sequence[Sequence[int]],
        total_mines: int,
    ) -> Analysis:
        height, width = validate_observation(observation)
        if not 0 <= total_mines < height * width:
            raise ValueError("total_mines is invalid for this board")

        unknown = {
            (row, col)
            for row in range(height)
            for col in range(width)
            if observation[row][col] == UNKNOWN
        }
        flags = {
            (row, col)
            for row in range(height)
            for col in range(width)
            if observation[row][col] == FLAGGED
        }

        constraints, error = self._extract_constraints(observation)
        if error:
            return self._contradiction(error)

        constraints, safe, mines, error = self._propagate(constraints)
        if error:
            return self._contradiction(error)

        remaining_mines = total_mines - len(flags) - len(mines)
        unassigned = unknown - safe - mines
        if remaining_mines < 0 or remaining_mines > len(unassigned):
            return self._contradiction("global remaining-mine count is inconsistent")

        frontier = set().union(*constraints.keys()) if constraints else set()
        components = self._components(frontier, constraints)
        if any(len(variables) > self.max_component_cells for variables, _ in components):
            return self._heuristic_analysis(
                unassigned,
                constraints,
                safe,
                mines,
                remaining_mines,
                len(frontier),
            )

        enumerations: list[_Enumeration] = []
        for variables, component_constraints in components:
            enumeration = self._enumerate_component(variables, component_constraints)
            if not enumeration.ways:
                return self._contradiction("frontier constraints have no valid assignment")
            enumerations.append(enumeration)

        unconstrained = unassigned - frontier
        component_distribution: dict[int, int] = {0: 1}
        for enumeration in enumerations:
            component_distribution = _convolve(component_distribution, enumeration.ways)

        unconstrained_count = len(unconstrained)
        total_models = 0
        for component_mines, ways in component_distribution.items():
            free_mines = remaining_mines - component_mines
            if 0 <= free_mines <= unconstrained_count:
                total_models += ways * comb(unconstrained_count, free_mines)
        if total_models == 0:
            return self._contradiction("global mine count has no valid assignment")

        probabilities: dict[Coord, float] = {
            **{coord: 0.0 for coord in safe},
            **{coord: 1.0 for coord in mines},
        }
        exact_safe = set(safe)
        exact_mines = set(mines)

        for index, enumeration in enumerate(enumerations):
            other_distribution: dict[int, int] = {0: 1}
            for other_index, other in enumerate(enumerations):
                if other_index != index:
                    other_distribution = _convolve(other_distribution, other.ways)

            for coord in enumeration.variables:
                numerator = 0
                for local_mines, mine_assignments in enumeration.mine_ways[coord].items():
                    for other_mines, other_ways in other_distribution.items():
                        free_mines = remaining_mines - local_mines - other_mines
                        if 0 <= free_mines <= unconstrained_count:
                            numerator += (
                                mine_assignments
                                * other_ways
                                * comb(unconstrained_count, free_mines)
                            )
                probabilities[coord] = numerator / total_models
                if numerator == 0:
                    exact_safe.add(coord)
                elif numerator == total_models:
                    exact_mines.add(coord)

        if unconstrained_count:
            one_cell_mine_models = 0
            for component_mines, ways in component_distribution.items():
                free_mines = remaining_mines - component_mines
                if 1 <= free_mines <= unconstrained_count:
                    one_cell_mine_models += ways * comb(unconstrained_count - 1, free_mines - 1)
            probability = one_cell_mine_models / total_models
            for coord in unconstrained:
                probabilities[coord] = probability
            if one_cell_mine_models == 0:
                exact_safe.update(unconstrained)
            elif one_cell_mine_models == total_models:
                exact_mines.update(unconstrained)

        return Analysis(
            safe=frozenset(exact_safe),
            mines=frozenset(exact_mines),
            probabilities=probabilities,
            exact=True,
            model_count=total_models,
            frontier_size=len(frontier),
        )

    def _extract_constraints(
        self,
        observation: Sequence[Sequence[int]],
    ) -> tuple[ConstraintMap, str | None]:
        height = len(observation)
        width = len(observation[0])
        constraints: ConstraintMap = {}
        for row in range(height):
            for col in range(width):
                clue = observation[row][col]
                if clue < 0:
                    continue
                neighbours = tuple(_neighbours((row, col), height, width))
                variables = frozenset(
                    coord for coord in neighbours if observation[coord[0]][coord[1]] == UNKNOWN
                )
                flagged = sum(
                    observation[n_row][n_col] == FLAGGED for n_row, n_col in neighbours
                )
                required = clue - flagged
                if required < 0 or required > len(variables):
                    return {}, (
                        f"clue at {(row, col)} is inconsistent: "
                        f"clue={clue}, flags={flagged}, covered={len(variables)}"
                    )
                if not variables:
                    if required:
                        return {}, f"clue at {(row, col)} cannot be satisfied"
                    continue
                if variables in constraints and constraints[variables] != required:
                    return {}, "duplicate frontier constraints disagree"
                constraints[variables] = required
        return constraints, None

    def _propagate(
        self,
        initial: ConstraintMap,
    ) -> tuple[ConstraintMap, set[Coord], set[Coord], str | None]:
        constraints = dict(initial)
        safe: set[Coord] = set()
        mines: set[Coord] = set()

        while True:
            changed = False
            normalized: ConstraintMap = {}
            for variables, required in constraints.items():
                reduced = variables - safe - mines
                reduced_required = required - len(variables & mines)
                if reduced_required < 0 or reduced_required > len(reduced):
                    return {}, safe, mines, "constraint propagation found a contradiction"
                if not reduced:
                    if reduced_required:
                        return {}, safe, mines, "an empty constraint still requires a mine"
                    continue
                previous = normalized.get(reduced)
                if previous is not None and previous != reduced_required:
                    return {}, safe, mines, "equivalent constraints disagree"
                normalized[reduced] = reduced_required
            constraints = normalized

            for variables, required in list(constraints.items()):
                if required == 0:
                    additions = set(variables) - safe
                    if additions:
                        safe.update(additions)
                        changed = True
                elif required == len(variables):
                    additions = set(variables) - mines
                    if additions:
                        mines.update(additions)
                        changed = True
            if safe & mines:
                return {}, safe, mines, "a cell was inferred as both safe and mined"
            if changed:
                continue

            derived: ConstraintMap = {}
            items = list(constraints.items())
            for left_variables, left_required in items:
                for right_variables, right_required in items:
                    if left_variables == right_variables or not left_variables < right_variables:
                        continue
                    difference = right_variables - left_variables
                    difference_required = right_required - left_required
                    if difference_required < 0 or difference_required > len(difference):
                        return {}, safe, mines, "subset constraints contradict each other"
                    existing = constraints.get(difference, derived.get(difference))
                    if existing is not None and existing != difference_required:
                        return {}, safe, mines, "derived constraints disagree"
                    if difference not in constraints:
                        derived[difference] = difference_required
            if derived:
                constraints.update(derived)
                changed = True
            if not changed:
                return constraints, safe, mines, None

    def _components(
        self,
        frontier: set[Coord],
        constraints: ConstraintMap,
    ) -> list[tuple[set[Coord], ConstraintMap]]:
        if not frontier:
            return []
        constraint_items = list(constraints.items())
        by_variable: dict[Coord, list[int]] = defaultdict(list)
        for index, (variables, _) in enumerate(constraint_items):
            for variable in variables:
                by_variable[variable].append(index)

        unseen = set(frontier)
        result: list[tuple[set[Coord], ConstraintMap]] = []
        while unseen:
            start = next(iter(unseen))
            variables: set[Coord] = set()
            constraint_indexes: set[int] = set()
            queue: deque[Coord] = deque([start])
            while queue:
                variable = queue.popleft()
                if variable in variables:
                    continue
                variables.add(variable)
                unseen.discard(variable)
                for constraint_index in by_variable[variable]:
                    if constraint_index in constraint_indexes:
                        continue
                    constraint_indexes.add(constraint_index)
                    queue.extend(constraint_items[constraint_index][0] - variables)
            component_constraints = {
                constraint_items[index][0]: constraint_items[index][1]
                for index in constraint_indexes
            }
            result.append((variables, component_constraints))
        return result

    def _enumerate_component(
        self,
        variable_set: set[Coord],
        constraints: ConstraintMap,
    ) -> _Enumeration:
        degrees = defaultdict(int)
        for variables in constraints:
            for variable in variables:
                degrees[variable] += 1
        variables = tuple(sorted(variable_set, key=lambda coord: (-degrees[coord], coord)))
        index_by_coord = {coord: index for index, coord in enumerate(variables)}
        indexed_constraints = [
            (tuple(index_by_coord[coord] for coord in cells), required)
            for cells, required in constraints.items()
        ]
        by_variable: list[list[int]] = [[] for _ in variables]
        for constraint_index, (indexes, _) in enumerate(indexed_constraints):
            for variable_index in indexes:
                by_variable[variable_index].append(constraint_index)

        assigned = [-1] * len(variables)
        assigned_count = [0] * len(indexed_constraints)
        assigned_mines = [0] * len(indexed_constraints)
        ways: dict[int, int] = defaultdict(int)
        mine_ways: dict[Coord, dict[int, int]] = {
            coord: defaultdict(int) for coord in variables
        }

        def search(position: int, mine_count: int) -> None:
            if position == len(variables):
                ways[mine_count] += 1
                for variable_index, value in enumerate(assigned):
                    if value:
                        mine_ways[variables[variable_index]][mine_count] += 1
                return

            for value in (0, 1):
                assigned[position] = value
                valid = True
                touched = by_variable[position]
                for constraint_index in touched:
                    assigned_count[constraint_index] += 1
                    assigned_mines[constraint_index] += value
                    indexes, required = indexed_constraints[constraint_index]
                    remaining = len(indexes) - assigned_count[constraint_index]
                    if (
                        assigned_mines[constraint_index] > required
                        or assigned_mines[constraint_index] + remaining < required
                    ):
                        valid = False
                if valid:
                    search(position + 1, mine_count + value)
                for constraint_index in touched:
                    assigned_count[constraint_index] -= 1
                    assigned_mines[constraint_index] -= value
            assigned[position] = -1

        search(0, 0)
        return _Enumeration(variables, dict(ways), {k: dict(v) for k, v in mine_ways.items()})

    def _heuristic_analysis(
        self,
        unassigned: set[Coord],
        constraints: ConstraintMap,
        safe: set[Coord],
        mines: set[Coord],
        remaining_mines: int,
        frontier_size: int,
    ) -> Analysis:
        global_probability = remaining_mines / len(unassigned) if unassigned else 0.0
        local: dict[Coord, list[float]] = defaultdict(list)
        for variables, required in constraints.items():
            probability = required / len(variables)
            for variable in variables:
                local[variable].append(probability)
        probabilities = {
            coord: sum(local[coord]) / len(local[coord]) if local[coord] else global_probability
            for coord in unassigned
        }
        probabilities.update({coord: 0.0 for coord in safe})
        probabilities.update({coord: 1.0 for coord in mines})
        return Analysis(
            safe=frozenset(safe),
            mines=frozenset(mines),
            probabilities=probabilities,
            exact=False,
            model_count=None,
            frontier_size=frontier_size,
        )

    @staticmethod
    def _contradiction(message: str) -> Analysis:
        return Analysis(
            safe=frozenset(),
            mines=frozenset(),
            probabilities={},
            exact=False,
            model_count=0,
            frontier_size=0,
            contradiction=message,
        )


def _neighbours(coord: Coord, height: int, width: int) -> Iterable[Coord]:
    row, col = coord
    for next_row in range(max(0, row - 1), min(height, row + 2)):
        for next_col in range(max(0, col - 1), min(width, col + 2)):
            if (next_row, next_col) != coord:
                yield next_row, next_col


def _convolve(left: Mapping[int, int], right: Mapping[int, int]) -> dict[int, int]:
    result: dict[int, int] = defaultdict(int)
    for left_mines, left_ways in left.items():
        for right_mines, right_ways in right.items():
            result[left_mines + right_mines] += left_ways * right_ways
    return dict(result)

