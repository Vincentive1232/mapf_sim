from __future__ import annotations

"""Solution containers for multi-agent path planning results."""

from dataclasses import dataclass, field

from mapf_lab.core.paths import DiscretePath


@dataclass
class MultiAgentSolution:
    """Collection of paths for all agents in a MAPF solution.

    Attributes:
        paths: Mapping from agent id to its discrete path.
    """

    paths: dict[int, DiscretePath] = field(default_factory=dict)

    def cost_soc(self) -> float:
        """Return the sum-of-costs objective across all agent paths."""
        return float(sum(path.cost() for path in self.paths.values()))

    def cost_makespan(self) -> float:
        """Return the makespan objective across all agent paths.

        Returns:
            ``inf`` if there are no paths, otherwise the maximum single-agent
            path cost.
        """
        if not self.paths:
            return float("inf")
        return float(max(path.cost() for path in self.paths.values()))

    def horizon(self) -> int:
        """Return the maximum discrete path length across all agents."""
        if not self.paths:
            return 0
        return max(len(path) for path in self.paths.values())

    def to_dict(self) -> dict:
        """Serialize the solution to a dictionary of plain Python values."""
        return {
            "paths": {agent: path.to_list() for agent, path in self.paths.items()},
            "soc": self.cost_soc(),
            "makespan": self.cost_makespan(),
            "horizon": self.horizon(),
        }