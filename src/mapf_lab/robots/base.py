from __future__ import annotations

"""Base robot interfaces and shared robot data structures."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RobotLike(Protocol):
    """Structural protocol implemented by robot models.

    Attributes:
        id: Unique robot identifier.
        model: Robot model name.
        start: Initial state vector.
        goal: Goal state vector.
        radius: Robot body radius used for collision checking.
    """

    id: int
    model: str
    start: np.ndarray
    goal: np.ndarray
    radius: float

    def state_dim(self) -> int:
        """Return the dimensionality of the robot state vector."""
        ...

    def requires_heading(self) -> bool:
        """Return whether the robot state must include an orientation term."""
        ...


@dataclass
class BaseRobot:
    """Base robot definition shared by concrete robot models.

    Attributes:
        id: Unique robot identifier.
        model: Robot model name.
        start: Initial robot state.
        goal: Goal robot state.
        radius: Robot collision radius.
    """

    id: int
    model: str
    start: np.ndarray
    goal: np.ndarray
    radius: float = 0.0

    def state_dim(self) -> int:
        """Return the dimensionality of the start/goal state vectors."""
        return int(self.start.shape[0])

    def requires_heading(self) -> bool:
        """Return whether this robot model requires a heading state."""
        return False

    def validate(self) -> None:
        """Validate basic shape and radius constraints.

        Raises:
            ValueError: If start and goal shapes differ or radius is negative.
        """
        if self.start.shape != self.goal.shape:
            raise ValueError(
                f"Robot {self.id}: start shape {self.start.shape} != goal shape {self.goal.shape}"
            )
        if self.radius < 0.0:
            raise ValueError(f"Robot {self.id}: radius must be non-negative")