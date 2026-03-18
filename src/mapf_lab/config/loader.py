from __future__ import annotations

"""Load and compose YAML-based MAPF configuration objects."""

from pathlib import Path
from typing import Any

import yaml

from mapf_lab.config.models import (
    ExperimentConfig,
    GeometricWorldConfig,
    GridWorldConfig,
    PlannerConfig,
    RobotSpec,
    ScenarioConfig,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its top-level mapping.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML as a dictionary. Returns an empty dict for empty files.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If top-level YAML content is not a mapping.
    """

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML content must be a dict: {path}")
    return data


def load_world_config(path: Path) -> GridWorldConfig | GeometricWorldConfig:
    """Load and validate a world configuration file.

    Args:
        path: World config YAML path.

    Returns:
        A validated grid or geometric world config.

    Raises:
        ValueError: If the world type is unsupported.
    """

    data = _read_yaml(path)
    world_type = data.get("type")
    if world_type == "grid":
        return GridWorldConfig.model_validate(data)
    if world_type == "geometric":
        return GeometricWorldConfig.model_validate(data)
    raise ValueError(f"Unsupported world type in {path}: {world_type}")


def load_robot_config(path: Path) -> list[RobotSpec]:
    """Load and validate robot specifications from a YAML file.

    Args:
        path: Robot config YAML path.

    Returns:
        List of validated robot specifications.

    Raises:
        ValueError: If the robots section is missing or malformed.
    """

    data = _read_yaml(path)
    robots = data.get("robots")
    if not isinstance(robots, list):
        raise ValueError(f"'robots' must be a list in {path}")
    return [RobotSpec.model_validate(r) for r in robots]


def load_planner_config(path: Path) -> PlannerConfig:
    """Load and validate planner settings from a YAML file.

    Args:
        path: Planner config YAML path.

    Returns:
        Validated planner configuration.
    """

    data = _read_yaml(path)
    return PlannerConfig.model_validate(data)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate experiment entry configuration.

    Args:
        path: Experiment config YAML path.

    Returns:
        Validated experiment configuration.
    """

    data = _read_yaml(path)
    return ExperimentConfig.model_validate(data)


def build_scenario_from_experiment(project_root: Path, experiment_path: Path) -> ScenarioConfig:
    """Build a runtime scenario by resolving file-based experiment entries.

    Args:
        project_root: Repository root path containing the configs directory.
        experiment_path: Experiment YAML path under configs/experiment.

    Returns:
        Fully resolved and validated scenario configuration.
    """

    exp = load_experiment_config(experiment_path)

    world_path = project_root / "configs" / "worlds" / exp.world
    robots_path = project_root / "configs" / "robots" / exp.robots
    planner_path = project_root / "configs" / "planners" / exp.planner

    world_cfg = load_world_config(world_path)
    robot_cfg = load_robot_config(robots_path)
    planner_cfg = load_planner_config(planner_path)

    return ScenarioConfig(
        world=world_cfg,
        robots=robot_cfg,
        planner=planner_cfg,
    )