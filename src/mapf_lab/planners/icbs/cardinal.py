from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mapf_lab.core.constraints import Constraint, split_conflict_to_constraints
from mapf_lab.core.conflicts import Conflict
from mapf_lab.core.paths import DiscretePath


Cardinality = Literal["cardinal", "semi-cardinal", "non-cardinal"]

@dataclass
class ClassifiedConflict:
    """
    Conflict annotated with cardinality and constraint impact information.

    Attributes:
        conflict: The original conflict detected in the solution.
        cardinality: The classification of the conflict's cardinality.
        increased_i: Whether the first agent's path cost increases under its constraint.
        increased_j: Whether the second agent's path cost increases under its constraint.
    """
    conflict: Conflict
    cardinality: Cardinality
    increased_i: bool
    increased_j: bool


def classify_conflict(
    *,
    conflict: Conflict,
    node_constraints: list[Constraint],
    node_paths: dict[int, DiscretePath],
    world,
    robots_by_id,
    low_level,
) -> ClassifiedConflict:
    """
    Determine the cardinality of a conflict by testing the impact of its constraints.
    For each agent involved in the conflict, we check if adding the corresponding
    constraint increases the agent's path cost.

    Args:
        conflict: The conflict to classify.
        node_constraints: The current set of constraints in the CT node.
        node_paths: The current paths for all agents in the CT node.
        world: The world model for low-level planning.
        robots_by_id: Mapping from agent id to robot model for low-level planning.
        low_level: The low-level planner instance to use for testing constraints.
    
    Returns:
        A ClassifiedConflict object containing the original conflict, its cardinality,
        and information about whether each agent's path cost increases under its constraint.
    """
    c1, c2 = split_conflict_to_constraints(conflict)

    old_cost_i = node_paths[conflict.agent_i].cost()
    old_cost_j = node_paths[conflict.agent_j].cost()

    increased_i = False
    increased_j = False

    # try agent i branch
    result_i = low_level.solve(
        world,
        robots_by_id[c1.agent],
        constraints=list(node_constraints) + [c1],
    )
    if result_i.success and result_i.cost > old_cost_i:
        increased_i = True

    # try agent i branch
    result_j = low_level.solve(
        world,
        robots_by_id[c2.agent],
        constraints=list(node_constraints) + [c2],
    )
    if result_j.success and result_j.cost > old_cost_j:
        increased_j = True

    if increased_i and increased_j:
        cardinality: Cardinality = "cardinal"
    elif increased_i or increased_j:
        cardinality: Cardinality = "semi-cardinal"
    else:
        cardinality: Cardinality = "non-cardinal"

    return ClassifiedConflict(
        conflict=conflict,
        cardinality=cardinality,
        increased_i=increased_i,
        increased_j=increased_j,
    )