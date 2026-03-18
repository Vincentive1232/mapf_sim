from __future__ import annotations

"""Point robot model with zero collision radius."""

from dataclasses import dataclass

import numpy as np

from mapf_lab.robots.base import BaseRobot


@dataclass
class PointRobot(BaseRobot):
    """Robot model represented by a position-only state.

    A point robot has no physical radius and does not require a heading term.
    """

    def __init__(self, id: int, start: np.ndarray, goal: np.ndarray) -> None:
        """Initialize a point robot from start and goal states.

        Args:
            id: Unique robot identifier.
            start: Initial state vector, typically ``[x, y]``.
            goal: Goal state vector, typically ``[x, y]``.
        """
        super().__init__(
            id=id,
            model="point",
            start=np.asarray(start, dtype=float),
            goal=np.asarray(goal, dtype=float),
            radius=0.0,
        )
        self.validate()