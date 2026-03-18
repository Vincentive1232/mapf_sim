from __future__ import annotations

"""2D world representations for grid and geometric MAPF environments."""

from dataclasses import dataclass, field

import numpy as np

from mapf_lab.world.obstacles import BoxObstacle, CircleObstacle, ObstacleLike, PolygonObstacle
from mapf_lab.world.sdf import GridSDF, union_sdf


@dataclass
class GridWorld2D:
    """Discrete 2D occupancy grid world.

    Attributes:
        width: Grid width in cells.
        height: Grid height in cells.
        obstacles: Occupied cells as ``(x, y)`` integer coordinates.
        connectivity: Neighborhood connectivity, typically 4 or 8.
    """

    width: int
    height: int
    obstacles: list[tuple[int, int]] = field(default_factory=list)
    connectivity: int = 4

    def __post_init__(self) -> None:
        """Build occupancy and SDF grids from the obstacle list."""
        self.occupancy = np.zeros((self.height, self.width), dtype=bool)
        for x, y in self.obstacles:
            if not self.in_bounds_xy(x, y):
                raise ValueError(f"Obstacle {(x, y)} out of bounds")
            self.occupancy[y, x] = True
        self.grid_sdf = GridSDF(self.occupancy, resolution=1.0, origin=(0.0, 0.0))

    def in_bounds_xy(self, x: int, y: int) -> bool:
        """Return ``True`` if the integer cell coordinate lies inside the grid."""
        return 0 <= x < self.width and 0 <= y < self.height

    def in_bounds(self, p: np.ndarray) -> bool:
        """Return ``True`` if the point maps to a valid grid cell."""
        x, y = int(p[0]), int(p[1])
        return self.in_bounds_xy(x, y)

    def is_occupied_xy(self, x: int, y: int) -> bool:
        """Return whether the given cell is occupied.

        Cells outside the grid are treated as occupied.
        """
        if not self.in_bounds_xy(x, y):
            return True
        return bool(self.occupancy[y, x])

    def is_occupied(self, p: np.ndarray) -> bool:
        """Return whether the point maps to an occupied grid cell."""
        x, y = int(p[0]), int(p[1])
        return self.is_occupied_xy(x, y)

    def sdf(self, p: np.ndarray) -> float:
        """Query the grid-based signed distance field at point ``p``."""
        return self.grid_sdf.query(p)

    def add_obstacle(self, x: int, y: int) -> None:
        """Mark a grid cell as occupied and rebuild the SDF."""
        if not self.in_bounds_xy(x, y):
            raise ValueError(f"Obstacle {(x, y)} out of bounds")
        self.occupancy[y, x] = True
        self.grid_sdf = GridSDF(self.occupancy, resolution=1.0, origin=(0.0, 0.0))

    def remove_obstacle(self, x: int, y: int) -> None:
        """Mark a grid cell as free and rebuild the SDF."""
        if not self.in_bounds_xy(x, y):
            raise ValueError(f"Cell {(x, y)} out of bounds")
        self.occupancy[y, x] = False
        self.grid_sdf = GridSDF(self.occupancy, resolution=1.0, origin=(0.0, 0.0))


@dataclass
class GeometricWorld2D:
    """Continuous 2D world with analytic obstacle geometry.

    Attributes:
        bounds: World bounds as ``(xmin, xmax, ymin, ymax)``.
        obstacles: Obstacle objects implementing the ``ObstacleLike`` protocol.
    """

    bounds: tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
    obstacles: list[ObstacleLike] = field(default_factory=list)

    def in_bounds(self, p: np.ndarray) -> bool:
        """Return ``True`` if point ``p`` lies inside the world bounds."""
        xmin, xmax, ymin, ymax = self.bounds
        return xmin <= float(p[0]) <= xmax and ymin <= float(p[1]) <= ymax

    def sdf(self, p: np.ndarray) -> float:
        """Return the signed distance to the nearest obstacle.

        Points outside the world bounds return ``-inf``. Worlds with no
        obstacles return ``inf``.
        """
        if not self.in_bounds(p):
            return -float("inf")
        if not self.obstacles:
            return float("inf")
        return union_sdf(p, self.obstacles)

    def contains_obstacle(self, p: np.ndarray) -> bool:
        """Return ``True`` if point ``p`` is inside or on any obstacle."""
        return self.sdf(p) <= 0.0

    def add_obstacle(self, obs: ObstacleLike) -> None:
        """Append an obstacle to the world."""
        self.obstacles.append(obs)

    @staticmethod
    def obstacle_from_dict(data: dict) -> ObstacleLike:
        """Construct an obstacle instance from a config dictionary.

        Supported obstacle types are ``circle``, ``box``, and ``polygon``.
        """
        obs_type = data["type"]
        if obs_type == "circle":
            return CircleObstacle(
                center=tuple(data["center"]),
                radius=float(data["radius"]),
            )
        if obs_type == "box":
            return BoxObstacle(
                xmin=float(data["xmin"]),
                xmax=float(data["xmax"]),
                ymin=float(data["ymin"]),
                ymax=float(data["ymax"]),
            )
        if obs_type == "polygon":
            return PolygonObstacle(points=[tuple(p) for p in data["points"]])
        raise ValueError(f"Unsupported obstacle type: {obs_type}")

    @classmethod
    def from_config(cls, bounds: list[float], obstacles: list[dict]) -> "GeometricWorld2D":
        """Build a geometric world from config-style bounds and obstacle dicts."""
        obs = [cls.obstacle_from_dict(o) for o in obstacles]
        return cls(bounds=(bounds[0], bounds[1], bounds[2], bounds[3]), obstacles=obs)