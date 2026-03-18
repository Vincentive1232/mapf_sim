from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from mapf_lab.core.conflicts import detect_first_conflict
from mapf_lab.core.solution import MultiAgentSolution
from mapf_lab.viz.palette import default_colors


class GridAnimator:
    """Animate multi-agent paths on a 2D occupancy grid.

    Attributes:
        world: Runtime world object providing dimensions and occupancy queries.
        robots: Sequence of robot objects with ``id``, ``start``, and ``goal``.
        solution: Planned multi-agent path solution.
        title: Plot title displayed above the grid.
        interval_ms: Delay between animation frames in milliseconds.
        trail: Whether to draw trajectory prefixes for each robot.
        show_conflict_text: Whether to display per-frame conflict diagnostics.
    """

    def __init__(
        self,
        world,
        robots,
        solution: MultiAgentSolution,
        title: str = "MAPF Solution",
        interval_ms: int = 600,
        trail: bool = True,
        show_conflict_text: bool = True,
    ) -> None:
        """Initialize a grid animation helper for a solved MAPF instance.

        Args:
            world: Runtime world used for plotting occupied cells.
            robots: Ordered robot list to visualize.
            solution: Multi-agent solution containing one path per robot id.
            title: Plot title.
            interval_ms: Frame interval in milliseconds.
            trail: Whether to draw history trails up to the current frame.
            show_conflict_text: Whether to compute and show conflict status text.
        """
        self.world = world
        self.robots = robots
        self.solution = solution
        self.title = title
        self.interval_ms = interval_ms
        self.trail = trail
        self.show_conflict_text = show_conflict_text
        self.colors = default_colors()

        self.fig = None
        self.ax = None
        self.anim = None

    def _draw_grid(self) -> None:
        """Draw the background grid, axes, and occupied cells.

        Returns:
            ``None``. Updates matplotlib artists on ``self.ax`` in-place.
        """
        assert self.ax is not None

        self.ax.set_xlim(-0.5, self.world.width - 0.5)
        self.ax.set_ylim(-0.5, self.world.height - 0.5)
        self.ax.set_aspect("equal")
        self.ax.set_xticks(np.arange(-0.5, self.world.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.world.height, 1), minor=True)
        self.ax.grid(which="minor")
        self.ax.set_xticks(range(self.world.width))
        self.ax.set_yticks(range(self.world.height))
        self.ax.set_title(self.title)

        # Draw occupied cells as solid rectangles aligned to cell boundaries.
        for y in range(self.world.height):
            for x in range(self.world.width):
                if self.world.is_occupied_xy(x, y):
                    rect = plt.Rectangle((x - 0.5, y - 0.5), 1.0, 1.0)
                    self.ax.add_patch(rect)

    def _draw_static_markers(self) -> None:
        """Draw start/goal labels and legend entries for all robots.

        Returns:
            ``None``. Updates matplotlib artists on ``self.ax`` in-place.
        """
        assert self.ax is not None

        for i, robot in enumerate(self.robots):
            color = self.colors[i % len(self.colors)]

            sx, sy = int(robot.start[0]), int(robot.start[1])
            gx, gy = int(robot.goal[0]), int(robot.goal[1])

            self.ax.text(
                sx,
                sy,
                f"S{robot.id}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
            self.ax.text(
                gx,
                gy,
                f"G{robot.id}",
                ha="center",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.15", alpha=0.25),
            )

            self.ax.plot([], [], marker="o", label=f"robot {robot.id}")

        self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    def _path_xy(self, states: list[np.ndarray]) -> tuple[list[float], list[float]]:
        """Convert state vectors into x/y coordinates for plotting.

        Args:
            states: Sequence of robot states where ``state[0]`` and ``state[1]``
                are x/y positions.

        Returns:
            Pair ``(xs, ys)`` containing float coordinate lists.
        """
        xs = [float(s[0]) for s in states]
        ys = [float(s[1]) for s in states]
        return xs, ys

    def create_animation(self) -> animation.FuncAnimation:
        """Create and store a matplotlib animation for the current solution.

        Returns:
            Configured ``matplotlib.animation.FuncAnimation`` instance.
        """
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self._draw_grid()
        self._draw_static_markers()

        horizon = self.solution.horizon()
        robot_artists = {}
        trail_artists = {}
        text_artist = None

        for i, robot in enumerate(self.robots):
            color = self.colors[i % len(self.colors)]
            point_artist, = self.ax.plot([], [], marker="o", markersize=12)
            trail_artist, = self.ax.plot([], [], linewidth=2, alpha=0.7)
            robot_artists[robot.id] = (point_artist, color)
            trail_artists[robot.id] = trail_artist

        if self.show_conflict_text:
            text_artist = self.ax.text(
                0.02,
                1.02,
                "",
                transform=self.ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
            )

        def update(frame: int):
            """Update all robot artists for one animation frame.

            Args:
                frame: Current discrete time step.

            Returns:
                List of matplotlib artists that were updated.
            """
            for i, robot in enumerate(self.robots):
                path = self.solution.paths[robot.id]
                state = path[frame]
                x, y = float(state[0]), float(state[1])

                point_artist, color = robot_artists[robot.id]
                point_artist.set_data([x], [y])

                if self.trail:
                    # Show prefix of the path up to current frame (inclusive).
                    xs, ys = self._path_xy(path.states[: min(frame + 1, len(path.states))])
                    trail_artists[robot.id].set_data(xs, ys)
                else:
                    trail_artists[robot.id].set_data([], [])

            if text_artist is not None:
                # Build a truncated view of paths and run conflict check at this frame.
                partial_paths = {
                    rid: type(path)(states=path.states[: min(frame + 1, len(path.states))])
                    for rid, path in self.solution.paths.items()
                }
                conflict = detect_first_conflict(partial_paths)
                text_artist.set_text(f"t={frame}   conflict={conflict}")

            return [a[0] for a in robot_artists.values()] + list(trail_artists.values())

        self.anim = animation.FuncAnimation(
            self.fig,
            update,
            frames=horizon,
            interval=self.interval_ms,
            blit=False,
            repeat=True,
        )
        return self.anim

    def show(self) -> None:
        """Display the animation in an interactive matplotlib window."""
        if self.anim is None:
            self.create_animation()
        plt.show()

    def save(self, out_path: str | Path, fps: int = 2) -> None:
        """Render and save the animation to a file.

        Args:
            out_path: Output file path, such as ``.gif`` or ``.mp4``.
            fps: Output frames per second.

        Returns:
            ``None``.
        """
        if self.anim is None:
            self.create_animation()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.anim.save(str(out_path), fps=fps)