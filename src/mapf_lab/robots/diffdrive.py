from __future__ import annotations

"""Differential-drive robot model with heading and kinematic limits."""

from dataclasses import dataclass

import numpy as np

from mapf_lab.robots.base import BaseRobot


@dataclass
class DiffDriveRobot(BaseRobot):
    """Robot model with differential-drive kinematics.

    Attributes:
        wheelbase: Distance between the drive wheels.
        max_v: Maximum linear velocity.
        max_w: Maximum angular velocity.
    """

    wheelbase: float = 0.0
    max_v: float = 0.0
    max_w: float = 0.0

    def __init__(
        self,
        id: int,
        start: np.ndarray,
        goal: np.ndarray,
        radius: float,
        wheelbase: float,
        max_v: float,
        max_w: float,
    ) -> None:
        """Initialize a differential-drive robot.

        Args:
            id: Unique robot identifier.
            start: Initial state vector, at least ``[x, y, theta]``.
            goal: Goal state vector, at least ``[x, y, theta]``.
            radius: Robot body radius used for collision checking.
            wheelbase: Distance between the left and right wheels.
            max_v: Maximum linear speed.
            max_w: Maximum angular speed.

        Raises:
            ValueError: If the state dimension is less than 3 or any kinematic
                limit is non-positive.
        """
        super().__init__(
            id=id,
            model="diffdrive",
            start=np.asarray(start, dtype=float),
            goal=np.asarray(goal, dtype=float),
            radius=float(radius),
        )
        self.wheelbase = float(wheelbase)
        self.max_v = float(max_v)
        self.max_w = float(max_w)
        self.validate()

        if self.state_dim() < 3:
            raise ValueError("DiffDriveRobot requires at least [x, y, theta] state")
        if self.wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive")
        if self.max_v <= 0.0:
            raise ValueError("max_v must be positive")
        if self.max_w <= 0.0:
            raise ValueError("max_w must be positive")

    def requires_heading(self) -> bool:
        """Return ``True`` because diff-drive robots require orientation."""
        return True