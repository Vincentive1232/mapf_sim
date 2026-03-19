from __future__ import annotations

from pathlib import Path

from rich import print

from mapf_lab.config.loader import build_scenario_from_experiment
from mapf_lab.planners.cbs.planner import CBSPlanner
from mapf_lab.planners.low_level.astar import GridAStarPlanner
from mapf_lab.robots.factory import build_robots
from mapf_lab.world.factory import build_world


def run_benchmark(experiment_file: str, max_ct_nodes: int = 2000, timeout_sec: float = 10.0) -> dict:
    project_root = Path(__file__).resolve().parents[3]
    experiment_path = project_root / "configs" / "experiments" / experiment_file

    scenario = build_scenario_from_experiment(project_root, experiment_path)
    world = build_world(scenario.world)
    robots = build_robots(scenario.robots)

    low_level = GridAStarPlanner(heuristic="manhattan", max_time=256)
    cbs = CBSPlanner(
        low_level=low_level,
        max_ct_nodes=max_ct_nodes,
        timeout_sec=timeout_sec,
        debug=False,
    )
    result = cbs.solve(world, robots, objective=scenario.planner.objective)

    summary = {
        "experiment": experiment_file,
        "status": result.status,
        "wall_time_sec": result.wall_time_sec,
        "expanded_ct": result.expanded_ct,
        "generated_ct": result.generated_ct,
        "duplicate_skipped": result.duplicate_skipped,
        "replans": result.replans,
        "replan_failures": result.replan_failures,
        "best_cost_seen": result.best_cost_seen,
    }

    if result.solution is not None:
        summary["soc"] = result.solution.cost_soc()
        summary["makespan"] = result.solution.cost_makespan()

    return summary


def main() -> None:
    experiments = [
        "demo_empty.yaml",
        "demo_passing.yaml",
        "demo_corridor.yaml",
    ]

    print("[bold]CBS benchmark[/bold]")
    for exp in experiments:
        summary = run_benchmark(exp, max_ct_nodes=2000, timeout_sec=10.0)
        print(summary)


if __name__ == "__main__":
    main()