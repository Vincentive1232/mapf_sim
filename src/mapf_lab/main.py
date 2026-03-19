from __future__ import annotations

from pathlib import Path

from rich import print

from mapf_lab.config.loader import build_scenario_from_experiment
from mapf_lab.core.conflicts import detect_first_conflict
from mapf_lab.core.paths import DiscretePath
from mapf_lab.core.solution import MultiAgentSolution
from mapf_lab.planners.cbs.planner import CBSPlanner
from mapf_lab.planners.low_level.astar import GridAStarPlanner
from mapf_lab.robots.factory import build_robots
from mapf_lab.world.factory import build_world

from mapf_lab.viz.animate_grid import GridAnimator


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    experiment_path = project_root / "configs" / "experiments" / "demo_corridor.yaml"

    scenario = build_scenario_from_experiment(project_root, experiment_path)
    world = build_world(scenario.world)
    robots = build_robots(scenario.robots)

    print("[green]mapf_lab boot ok[/green]")
    print(f"Project root: {project_root}")

    low_level = GridAStarPlanner(heuristic="manhattan", max_time=128)

    # Independent planning
    indep_paths: dict[int, DiscretePath] = {}
    for robot in robots:
        result = low_level.solve(world, robot, constraints=[])
        if not result.success:
            print(f"[red]Independent planning failed for robot {robot.id}:[/red] {result.message}")
            return
        indep_paths[robot.id] = DiscretePath(states=result.path)

    indep_solution = MultiAgentSolution(paths=indep_paths)
    indep_conflict = detect_first_conflict(indep_solution.paths)

    print("\n[bold]Independent planning result:[/bold]")
    print(indep_solution.to_dict())
    print("\n[bold]First conflict in independent solution:[/bold]")
    print(indep_conflict)

    # CBS
    cbs = CBSPlanner(low_level=low_level, max_ct_nodes=100000)
    cbs_solution = cbs.solve(world, robots, objective=scenario.planner.objective)

    print("\n[bold]CBS result:[/bold]")
    if cbs_solution is None:
        print("[red]CBS failed to find a solution[/red]")
        return

    print(cbs_solution.to_dict())
    final_conflict = detect_first_conflict(cbs_solution.paths)
    print("\n[bold]Conflict after CBS:[/bold]")
    print(final_conflict)

    animator = GridAnimator(
        world=world,
        robots=robots,
        solution=cbs_solution,
        title="CBS Grid Solution",
        interval_ms=700,
        trail=True,
        show_conflict_text=True,
    )
    
    animator.save("outputs/cbs_demo.gif", fps=2)
    print("saved to outputs/cbs_demo.gif")


if __name__ == "__main__":
    main()