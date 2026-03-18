from __future__ import annotations

"""Collision checking helpers for robot states and motion segments."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CollisionChecker:
    """Collision queries against a world SDF.

    Attributes:
        sample_step: Maximum spatial step used when discretizing an edge for
            collision checking.
    """

    sample_step: float = 0.25

    def state_in_collision(self, state: np.ndarray, robot_radius: float, world) -> bool:
        """Return whether a robot state is in collision.

        States outside the world bounds are treated as colliding.
        """
        if not world.in_bounds(state):
            return True
        return world.sdf(state) <= robot_radius

    def clearance(self, state: np.ndarray, robot_radius: float, world) -> float:
        """Return obstacle clearance after subtracting the robot radius.

        States outside the world bounds return ``-inf``.
        """
        if not world.in_bounds(state):
            return -float("inf")
        return float(world.sdf(state) - robot_radius)

    def edge_in_collision(
        self,
        s1: np.ndarray,
        s2: np.ndarray,
        robot_radius: float,
        world,
    ) -> bool:
        """Return whether the segment from ``s1`` to ``s2`` intersects obstacles.

        The edge is checked by uniform sampling with spacing controlled by
        ``sample_step``.
        """
        diff = s2[:2] - s1[:2]
        length = float(np.linalg.norm(diff))
        if length == 0.0:
            return self.state_in_collision(s1, robot_radius, world)

        n = max(2, int(np.ceil(length / self.sample_step)) + 1)
        for alpha in np.linspace(0.0, 1.0, n):
            p = (1.0 - alpha) * s1 + alpha * s2
            if self.state_in_collision(p, robot_radius, world):
                return True
        return False