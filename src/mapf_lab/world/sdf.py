from __future__ import annotations

"""Signed distance field utilities for 2D MAPF worlds."""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt

from mapf_lab.world.obstacles import ObstacleLike


def union_sdf(p: np.ndarray, obstacles: list[ObstacleLike]) -> float:
    """Return the minimum SDF value across all obstacles (union operation).

    Args:
        p: Query point as a numpy array with at least 2 elements.
        obstacles: List of obstacle objects satisfying the ``ObstacleLike`` protocol.

    Returns:
        Minimum signed distance to any obstacle surface. Returns ``inf`` when
        the obstacle list is empty.
    """
    if not obstacles:
        return float("inf")
    return min(obs.sdf(p) for obs in obstacles)


@dataclass
class GridSDF:
    """Precomputed signed distance field on a regular grid.

    Converts a binary occupancy grid into a continuous SDF by applying
    Euclidean distance transforms to both the free and occupied regions.

    Attributes:
        occupancy: 2-D boolean/integer array where ``True``/1 means occupied.
        resolution: Physical size of each grid cell in world units.
        origin: (x, y) world coordinates of the grid's bottom-left corner.
        sdf_grid: Precomputed SDF array, populated in ``__post_init__``.
    """

    occupancy: np.ndarray
    resolution: float = 1.0
    origin: tuple[float, float] = (0.0, 0.0)
    sdf_grid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Build the SDF grid from the occupancy map."""
        occ = self.occupancy.astype(bool)

        # 外部到障碍物距离
        outside = distance_transform_edt(~occ) * self.resolution
        # 内部到自由空间距离
        inside = distance_transform_edt(occ) * self.resolution

        self.sdf_grid = outside - inside

    def world_to_grid(self, p: np.ndarray) -> tuple[int, int]:
        """Convert a world-space point to integer grid indices.

        Args:
            p: World-space point as a numpy array with at least 2 elements.

        Returns:
            (col, row) integer grid indices corresponding to *p*.
        """
        x = int((float(p[0]) - self.origin[0]) / self.resolution)
        y = int((float(p[1]) - self.origin[1]) / self.resolution)
        return x, y

    def query(self, p: np.ndarray) -> float:
        """Look up the SDF value at a world-space point.

        Args:
            p: World-space query point as a numpy array with at least 2 elements.

        Returns:
            SDF value at the nearest grid cell. Returns ``-inf`` when *p* falls
            outside the grid bounds.
        """
        x, y = self.world_to_grid(p)
        h, w = self.sdf_grid.shape
        if not (0 <= y < h and 0 <= x < w):
            return -float("inf")
        return float(self.sdf_grid[y, x])