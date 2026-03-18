from __future__ import annotations

"""Disk robot model with circular collision geometry."""

from dataclasses import dataclass

import numpy as np

from mapf_lab.robots.base import BaseRobot


@dataclass
class DiskRobot(BaseRobot):
    """Robot model represented by a position state and circular body.

    The robot state is typically ``[x, y]`` and collision checking uses the
    configured radius.
    """

    def __init__(self, id: int, start: np.ndarray, goal: np.ndarray, radius: float) -> None:
        """Initialize a disk robot from start and goal states.

        Args:
            id: Unique robot identifier.
            start: Initial state vector, typically ``[x, y]``.
            goal: Goal state vector, typically ``[x, y]``.
            radius: Robot body radius used for collision checking.
        """
        super().__init__(
            id=id,
            model="disk",
            start=np.asarray(start, dtype=float),
            goal=np.asarray(goal, dtype=float),
            radius=float(radius),
        )
        self.validate()