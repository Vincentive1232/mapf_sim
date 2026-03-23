from __future__ import annotations

"""Support for loading MovingAI .map (octile/grid) map files."""

from dataclasses import dataclass
from pathlib import Path


# Common blocked-cell markers used by MovingAI grid maps.
BLOCKED_TILES: set[str] = {"@", "O", "T", "W"}


@dataclass(frozen=True)
class OctileMap:
    """Parsed MovingAI octile map.

    Attributes:
        width: Map width in cells.
        height: Map height in cells.
        rows: Raw map rows as strings.
    """

    width: int
    height: int
    rows: tuple[str, ...]

    def is_blocked(self, x: int, y: int) -> bool:
        """Return whether the given cell is blocked."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Cell {(x, y)} out of bounds for map {self.width}x{self.height}")
        return self.rows[y][x] in BLOCKED_TILES

    def to_obstacles(self) -> list[tuple[int, int]]:
        """Convert blocked cells into the project's obstacle list format."""
        obstacles: list[tuple[int, int]] = []
        for y, row in enumerate(self.rows):
            for x, ch in enumerate(row):
                if ch in BLOCKED_TILES:
                    obstacles.append((x, y))
        return obstacles


def load_movingai_map(path: str | Path) -> OctileMap:
    """Load a MovingAI .map file.

    Supported header format:

    type octile
    height H
    width W
    map
    ....

    Args:
        path: Path to the .map file.

    Returns:
        Parsed :class:`OctileMap`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is malformed.
    """

    map_path = Path(path)
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    with map_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n\r") for line in f]

    if len(lines) < 4:
        raise ValueError(f"Invalid .map file (too short): {map_path}")

    header0 = lines[0].strip().lower()
    if not header0.startswith("type "):
        raise ValueError(f"Invalid .map file (missing type): {map_path}")

    try:
        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid .map file dimensions: {map_path}") from exc

    if lines[3].strip().lower() != "map":
        raise ValueError(f"Invalid .map file (missing 'map' line): {map_path}")

    rows = lines[4:]
    if len(rows) != height:
        raise ValueError(
            f"Invalid .map file {map_path}: expected {height} map rows, got {len(rows)}"
        )

    for idx, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f"Invalid .map file {map_path}: row {idx} has width {len(row)}, expected {width}"
            )

    return OctileMap(width=width, height=height, rows=tuple(rows))