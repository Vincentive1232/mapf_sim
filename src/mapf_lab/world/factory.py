from __future__ import annotations

"""Factory helpers for constructing runtime world objects from configs."""

from pathlib import Path

from mapf_lab.config.models import GeometricWorldConfig, GridWorldConfig
from mapf_lab.maps.octile_map import load_movingai_map
from mapf_lab.world.world2d import GeometricWorld2D, GridWorld2D


def build_world(
    world_cfg: GridWorldConfig | GeometricWorldConfig,
    *,
    base_dir: Path | None = None,
    ):
    """Build a runtime world object from a validated world config.

    Args:
        world_cfg: Parsed grid or geometric world configuration.
        base_dir: Optional based directory used to resolve relative "map_file" paths.

    Returns:
        A ``GridWorld2D`` or ``GeometricWorld2D`` instance.

    Raises:
        FileNotFoundError: If the config references a non-existent map file.
        TypeError: If the config type is unsupported.
        ValueError: If the config content is invalid (e.g. malformed map file).
    """

    if isinstance(world_cfg, GridWorldConfig):
        if world_cfg.map_file is not None:
            if world_cfg.map_format not in {None, "movingai"}:
                raise ValueError(f"Unsupported map format: {world_cfg.map_format}")

            map_path = Path(world_cfg.map_file)
            if not map_path.is_absolute():
                if base_dir is None:
                    raise FileNotFoundError(
                        "Relative map_file requires build_world(..., base_dir=...)"
                    )
                map_path = (base_dir / map_path).resolve()

            movingai_map = load_movingai_map(map_path)
            return GridWorld2D(
                width=movingai_map.width,
                height=movingai_map.height,
                obstacles=movingai_map.to_obstacles(),
                connectivity=world_cfg.connectivity,
            )

        if world_cfg.width is None or world_cfg.height is None:
            raise ValueError(
                "Grid world requires either map_file or both width and height"
            )

        return GridWorld2D(
            width=world_cfg.width,
            height=world_cfg.height,
            obstacles=[tuple(o) for o in world_cfg.obstacles],
            connectivity=world_cfg.connectivity,
        )
    
    if isinstance(world_cfg, GeometricWorldConfig):
        return GeometricWorld2D.from_config(
            bounds=world_cfg.bounds,
            obstacles=world_cfg.obstacles,
        )

    raise TypeError(f"Unsupported world config type: {type(world_cfg)}")