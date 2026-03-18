from __future__ import annotations

"""Factory helpers for constructing runtime world objects from configs."""

from mapf_lab.config.models import GeometricWorldConfig, GridWorldConfig
from mapf_lab.world.world2d import GeometricWorld2D, GridWorld2D


def build_world(world_cfg: GridWorldConfig | GeometricWorldConfig):
    """Build a runtime world object from a validated world config.

    Args:
        world_cfg: Parsed grid or geometric world configuration.

    Returns:
        A ``GridWorld2D`` or ``GeometricWorld2D`` instance.

    Raises:
        TypeError: If the config type is unsupported.
    """

    if isinstance(world_cfg, GridWorldConfig):
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