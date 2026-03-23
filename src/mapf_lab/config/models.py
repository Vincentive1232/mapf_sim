from __future__ import annotations

"""Pydantic models for MAPF world, robot, planner, and scenario configs."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GridWorldConfig(BaseModel):
    """Configuration for a 2D grid world.

    Attributes:
        type: World type discriminator, always "grid".
        width: Grid width in cells.
        height: Grid height in cells.
        obstacles: List of blocked grid coordinates in [x, y] format.
        connectivity: Neighborhood mode, either 4- or 8-connected.
    """

    type: Literal["grid"]
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    obstacles: list[list[int]] = Field(default_factory=list)

    map_file: str | None = None
    map_format: Literal["movingai"] | None = None

    connectivity: Literal[4, 8] = 4

    @model_validator(mode="after")
    def validate_obstacles(self) -> "GridWorldConfig":
        using_external_map = self.map_file is not None

        if using_external_map:
            if self.map_format is None:
                self.map_format = "movingai"
            return self

        if self.width is None or self.height is None:
            raise ValueError(
                "Grid world requires either map_file or both width and height"
            )

        for obs in self.obstacles:
            if len(obs) != 2:
                raise ValueError(f"Each obstacle must be [x, y], got {obs}")
            x, y = obs
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(
                    f"Obstacle {obs} out of bounds for grid {self.width}x{self.height}"
                )
        return self


class GeometricWorldConfig(BaseModel):
    """Configuration for a simple geometric 2D world.

    Attributes:
        type: World type discriminator, always "geometric".
        bounds: Rectangle bounds in [xmin, xmax, ymin, ymax] format.
        obstacles: List of geometric obstacles, usually polygon descriptors.
        resolution: Optional discretization resolution used by some planners.
    """

    type: Literal["geometric"]
    bounds: list[float]  # [xmin, xmax, ymin, ymax]
    obstacles: list[dict] = Field(default_factory=list)
    resolution: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "GeometricWorldConfig":
        if len(self.bounds) != 4:
            raise ValueError("bounds must be [xmin, xmax, ymin, ymax]")
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmax > xmin and ymax > ymin):
            raise ValueError("Invalid bounds ordering")
        return self


WorldConfig = GridWorldConfig | GeometricWorldConfig


class RobotSpec(BaseModel):
    """Robot definition used in a scenario.

    Attributes:
        id: Unique non-negative robot identifier.
        model: Robot dynamics model type.
        start: Start state vector.
        goal: Goal state vector.
        radius: Body radius for disk and diffdrive models.
        wheelbase: Wheelbase length for diffdrive model.
        max_v: Maximum linear velocity for diffdrive model.
        max_w: Maximum angular velocity for diffdrive model.
    """

    id: int = Field(..., ge=0)
    model: Literal["point", "disk", "diffdrive"]
    start: list[float]
    goal: list[float]
    radius: float | None = Field(default=None, gt=0)
    wheelbase: float | None = Field(default=None, gt=0)
    max_v: float | None = Field(default=None, gt=0)
    max_w: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_model_specific_fields(self) -> "RobotSpec":
        if self.model == "point":
            if len(self.start) < 2 or len(self.goal) < 2:
                raise ValueError("Point robot start/goal must have at least 2 values")
        elif self.model == "disk":
            if self.radius is None:
                raise ValueError("Disk robot requires radius")
        elif self.model == "diffdrive":
            if self.radius is None:
                raise ValueError("DiffDrive robot requires radius")
            if self.wheelbase is None:
                raise ValueError("DiffDrive robot requires wheelbase")
            if self.max_v is None:
                raise ValueError("DiffDrive robot requires max_v")
            if self.max_w is None:
                raise ValueError("DiffDrive robot requires max_w")
        return self


class PlannerConfig(BaseModel):
    """Planner configuration.

    Attributes:
        name: High-level planner algorithm.
        low_level: Low-level single-agent planner.
        objective: Optimization objective across all robots.
        time_step: Planning simulation time step.
        max_iter: Maximum planner iterations.
    """

    name: Literal["astar", "cbs", "dbcbs"]
    low_level: Literal["astar", "space_time_astar", "se2_astar"] = "astar"
    objective: Literal["soc", "makespan"] = "soc"
    time_step: float = Field(default=1.0, gt=0)
    max_iter: int = Field(default=10000, gt=0)


class ExperimentConfig(BaseModel):
    """File-based experiment entry in experiment config sets.

    Attributes:
        name: Experiment name.
        world: Path or key of world config.
        robots: Path or key of robot config.
        planner: Path or key of planner config.
    """

    name: str
    world: str
    robots: str
    planner: str


class ScenarioConfig(BaseModel):
    """Resolved scenario config used at runtime.

    Attributes:
        world: Parsed world configuration.
        robots: Robot specifications participating in this scenario.
        planner: Planner settings for this run.
    """

    world: WorldConfig
    robots: list[RobotSpec]
    planner: PlannerConfig

    @model_validator(mode="after")
    def validate_robot_ids_unique(self) -> "ScenarioConfig":
        ids = [r.id for r in self.robots]
        if len(ids) != len(set(ids)):
            raise ValueError("Robot ids must be unique")
        return self