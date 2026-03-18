from __future__ import annotations

"""Factory helpers for constructing robot objects from validated specs."""

from mapf_lab.config.models import RobotSpec
from mapf_lab.robots.base import BaseRobot
from mapf_lab.robots.diffdrive import DiffDriveRobot
from mapf_lab.robots.disk import DiskRobot
from mapf_lab.robots.point import PointRobot


def build_robot(spec: RobotSpec):
    """Build a concrete robot instance from a validated robot spec.

    Args:
        spec: Parsed robot specification.

    Returns:
        A ``PointRobot``, ``DiskRobot``, or ``DiffDriveRobot`` instance.

    Raises:
        ValueError: If the robot model is unsupported.
    """

    if spec.model == "point":
        return PointRobot(
            id=spec.id,
            start=spec.start,
            goal=spec.goal,
        )

    if spec.model == "disk":
        return DiskRobot(
            id=spec.id,
            start=spec.start,
            goal=spec.goal,
            radius=spec.radius,
        )

    if spec.model == "diffdrive":
        return DiffDriveRobot(
            id=spec.id,
            start=spec.start,
            goal=spec.goal,
            radius=spec.radius,
            wheelbase=spec.wheelbase,
            max_v=spec.max_v,
            max_w=spec.max_w,
        )

    raise ValueError(f"Unsupported robot model: {spec.model}")


def build_robots(specs: list[RobotSpec]) -> list[BaseRobot]:
    """Build a list of robot objects from validated robot specs."""
    return [build_robot(spec) for spec in specs]