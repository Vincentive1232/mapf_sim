from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich import print

from mapf_lab.config.loader import build_scenario_from_experiment
from mapf_lab.core.conflicts import detect_first_conflict
from mapf_lab.core.paths import DiscretePath
from mapf_lab.core.solution import MultiAgentSolution
from mapf_lab.planners.cbs.planner import CBSPlanner
from mapf_lab.planners.icbs.planner import ICBSPlanner
from mapf_lab.planners.low_level.astar import GridAStarPlanner
from mapf_lab.robots.factory import build_robots
from mapf_lab.world.factory import build_world

from mapf_lab.viz.animate_grid import GridAnimator


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for map preview and video export settings."""
    parser = argparse.ArgumentParser(description="Run MAPF planning and export visualization.")
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Skip planning and export only a static map preview image.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=12,
        help="Output video frames per second.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help="Keep one frame every N planning steps during export.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on rendered frames for faster export.",
    )
    parser.add_argument(
        "--video-dpi",
        type=int,
        default=72,
        help="DPI used when rendering video frames.",
    )
    parser.add_argument(
        "--video-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable fast mp4 encoding settings.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[2]
    experiment_path = project_root / "configs" / "experiments" / "demo_Berlin_256.yaml"

    scenario = build_scenario_from_experiment(project_root, experiment_path)
    world = build_world(
        scenario.world,
        base_dir=project_root / "configs" / "worlds",
    )
    robots = build_robots(scenario.robots)

    print("[green]mapf_lab boot ok[/green]")
    print(f"Project root: {project_root}")

    map_only_env = os.getenv("MAPF_MAP_ONLY", "0") == "1"
    if args.map_only or map_only_env:
        # Map-only mode skips all planners and exports immediately.
        animator = GridAnimator(
            world=world,
            robots=robots,
            solution=MultiAgentSolution(paths={}),
            title="MAP Preview",
            interval_ms=100,
            trail=False,
            show_conflict_text=False,
        )
        animator.save_map("outputs/testmap_map.png", dpi=120)
        print("saved to outputs/testmap_map.png")
        return

    low_level = GridAStarPlanner(heuristic="manhattan", max_time=128000)

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
    # cbs = CBSPlanner(
    #     low_level=low_level, 
    #     max_ct_nodes=100000,
    #     timeout_sec=100.0,
    #     debug=False,
    # )
    # cbs_result = cbs.solve(world, robots, objective=scenario.planner.objective)

    # print("\n[bold]CBS result summary:[/bold]")
    # print(cbs_result.to_dict())

    # if not cbs_result.success():
    #     print(f"[red]CBS did not return a solution[/red]")
    #     return

    # cbs_solution = cbs_result.solution
    # final_conflict = detect_first_conflict(cbs_solution.paths)
    # print("\n[bold]Conflict after CBS:[/bold]")
    # print(final_conflict)

    # animator = GridAnimator(
    #     world=world,
    #     robots=robots,
    #     solution=cbs_solution,
    #     title="CBS Grid Solution",
    #     interval_ms=700,
    #     trail=True,
    #     show_conflict_text=True,
    # )
    
    # animator.save("outputs/cbs_demo.gif", fps=2)
    # print("saved to outputs/cbs_demo.gif")

    icbs = ICBSPlanner(
            low_level=low_level,
            max_ct_nodes=10000,
            timeout_sec=100.0,
            debug=False,
    )
    icbs_result = icbs.solve(world, robots, objective=scenario.planner.objective)

    print("\n[bold]ICBS result summary:[/bold]")
    print(icbs_result.to_dict())

    if not icbs_result.success():
        print(f"[red]ICBS did not return a solution[/red]")
        return  

    icbs_solution = icbs_result.solution
    final_conflict = detect_first_conflict(icbs_solution.paths)
    print("\n[bold]Conflict after ICBS:[/bold]")
    print(final_conflict)

    animator = GridAnimator(
        world=world,
        robots=robots,
        solution=icbs_solution,
        title="ICBS Grid Solution",
        interval_ms=100,
        trail=False,
        show_conflict_text=False,
    )

    video_fps = args.video_fps
    video_stride = args.frame_stride
    video_max_frames = args.max_frames
    video_dpi = args.video_dpi
    video_fast = args.video_fast

    print(
        "[cyan]Video export settings:[/cyan] "
        f"fps={video_fps}, stride={video_stride}, max_frames={video_max_frames}, "
        f"dpi={video_dpi}, fast={video_fast}"
    )
    animator.save(
        "outputs/testmap_demo.mp4",
        fps=video_fps,
        frame_stride=video_stride,
        max_frames=video_max_frames,
        dpi=video_dpi,
        fast=video_fast,
    )
    print("saved to outputs/testmap_demo.mp4")


if __name__ == "__main__":
    main()