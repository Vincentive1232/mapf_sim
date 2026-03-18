from __future__ import annotations

"""Geometric obstacle primitives for 2D MAPF worlds."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from shapely.geometry import Point, Polygon, box


class ObstacleLike(Protocol):
    """Structural protocol shared by all obstacle types.

    Any class that implements ``sdf``, ``contains``, ``distance``, and
    ``to_shapely`` satisfies this protocol and can be used wherever an
    obstacle is expected.
    """

    def sdf(self, p: np.ndarray) -> float:
        """Signed distance from point *p* to the obstacle surface.

        Negative inside, zero on the boundary, positive outside.
        """
        ...

    def contains(self, p: np.ndarray) -> bool:
        """Return ``True`` if point *p* is inside or on the obstacle."""
        ...

    def distance(self, p: np.ndarray) -> float:
        """Non-negative Euclidean distance from *p* to the nearest obstacle surface."""
        ...

    def to_shapely(self):
        """Return the equivalent Shapely geometry object."""
        ...


@dataclass
class CircleObstacle:
    """Circular obstacle defined by a center point and radius.

    Attributes:
        center: (x, y) coordinates of the circle center.
        radius: Circle radius in world units.
    """

    center: tuple[float, float]
    radius: float

    def sdf(self, p: np.ndarray) -> float:
        """Signed distance from *p* to the circle boundary."""
        return float(np.linalg.norm(p[:2] - np.asarray(self.center, dtype=float)) - self.radius)

    def contains(self, p: np.ndarray) -> bool:
        """Return ``True`` if *p* is inside or on the circle."""
        return self.sdf(p) <= 0.0

    def distance(self, p: np.ndarray) -> float:
        """Non-negative distance from *p* to the circle boundary."""
        return max(0.0, self.sdf(p))

    def to_shapely(self):
        """Return a Shapely ``Polygon`` approximating this circle."""
        return Point(self.center).buffer(self.radius)


@dataclass
class BoxObstacle:
    """Axis-aligned rectangular obstacle.

    Attributes:
        xmin: Left boundary in world coordinates.
        xmax: Right boundary in world coordinates.
        ymin: Bottom boundary in world coordinates.
        ymax: Top boundary in world coordinates.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def sdf(self, p: np.ndarray) -> float:
        """Signed distance from *p* to the box boundary.

        Uses the standard box SDF formula: computes per-axis distances from
        the half-extents, then combines the outside (L2) and inside (Chebyshev)
        components.
        """
        x, y = float(p[0]), float(p[1])
        cx = 0.5 * (self.xmin + self.xmax)
        cy = 0.5 * (self.ymin + self.ymax)
        hx = 0.5 * (self.xmax - self.xmin)
        hy = 0.5 * (self.ymax - self.ymin)

        dx = abs(x - cx) - hx
        dy = abs(y - cy) - hy

        outside = np.linalg.norm([max(dx, 0.0), max(dy, 0.0)])
        inside = min(max(dx, dy), 0.0)
        return float(outside + inside)

    def contains(self, p: np.ndarray) -> bool:
        """Return ``True`` if *p* is inside or on the box."""
        return self.sdf(p) <= 0.0

    def distance(self, p: np.ndarray) -> float:
        """Non-negative distance from *p* to the box boundary."""
        return max(0.0, self.sdf(p))

    def to_shapely(self):
        """Return a Shapely ``Polygon`` representing this box."""
        return box(self.xmin, self.ymin, self.xmax, self.ymax)


@dataclass
class PolygonObstacle:
    """Convex or concave polygonal obstacle.

    Attributes:
        points: Ordered list of (x, y) vertices defining the polygon boundary.
    """

    points: list[tuple[float, float]]

    def contains(self, p: np.ndarray) -> bool:
        """Return ``True`` if *p* is strictly inside or on the polygon boundary."""
        poly = self.to_shapely()
        return poly.contains(Point(float(p[0]), float(p[1]))) or poly.touches(
            Point(float(p[0]), float(p[1]))
        )

    def distance(self, p: np.ndarray) -> float:
        """Non-negative distance from *p* to the nearest polygon edge."""
        poly = self.to_shapely()
        point = Point(float(p[0]), float(p[1]))
        return float(poly.distance(point))

    def sdf(self, p: np.ndarray) -> float:
        """Signed distance from *p* to the polygon boundary.

        Negative inside, positive outside.
        """
        d = self.distance(p)
        return -d if self.contains(p) else d

    def to_shapely(self):
        """Return a Shapely ``Polygon`` built from the vertex list."""
        return Polygon(self.points)