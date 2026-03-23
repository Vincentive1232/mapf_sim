# mapf_lab

A modular Python framework for testing multi-robot path planning algorithms,
including CBS, db-CBS, and future variants.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Run

```bash
source .venv/bin/activate
PYTHONPATH=src python -m mapf_lab.main                       # sanity check
PYTHONPATH=src python -m mapf_lab.experiments.benchmark      # benchmarking
PYTHONPATH=src python -m tests.cbs_test                      # CBS  Test
PYTHONPATH=src python -m tests.icbs_test                     # ICBS Test
```

## Visualization Export Options

Use `mapf_lab.main` CLI args to control export speed/quality.

Quick video export (recommended for fast preview):

```bash
PYTHONPATH=src python -m mapf_lab.main \
    --video-fps 10 \
    --frame-stride 8 \
    --max-frames 120 \
    --video-dpi 64 \
    --video-fast
```

High-quality video export:

```bash
PYTHONPATH=src python -m mapf_lab.main \
    --video-fps 24 \
    --frame-stride 1 \
    --video-dpi 140 \
    --no-video-fast
```

Map-only preview (skip planning and video, fastest):

```bash
PYTHONPATH=src python -m mapf_lab.main --map-only
```

## Add A New Map (Step By Step)
This project supports maps in [MovingAI format](https://movingai.com/benchmarks/mapf/index.html).

1. Download a `.map` file and place it in:
`configs/predef_map_files/`

    Example:
    `configs/predef_map_files/Berlin_256.map`

2. Create a world config in `configs/worlds/`.

    Example file: `configs/worlds/Berlin_256.yaml`

    ```yaml
    type: grid
    map_file: ../../configs/predef_map_files/Berlin_256.map
    map_format: movingai
    connectivity: 4
    ```

3. Define agent count and start/goal positions in `configs/robots/`.

    Example file: `configs/robots/Berlin_256_4agents.yaml`

    ```yaml
    robots:
      - id: 0
        model: point
        start: [10, 10]
        goal: [245, 245]

      - id: 1
        model: point
        start: [245, 10]
        goal: [20, 230]

      - id: 2
        model: point
        start: [230, 230]
        goal: [30, 30]

      - id: 3
        model: point
        start: [30, 220]
        goal: [220, 30]
    ```

4. Create an experiment file in `configs/experiments/`.

    Example file: `configs/experiments/demo_Berlin_256.yaml`

    ```yaml
    name: demo_Berlin_256
    world: Berlin_256.yaml
    robots: Berlin_256_4agents.yaml
    planner: cbs.yaml
    ```

5. Point `src/mapf_lab/main.py` to your experiment file.

    For example, in `src/mapf_lab/main.py`, update:

    ```python
    experiment_path = project_root / "configs" / "experiments" / "demo_Berlin_256.yaml"
    ```

6. Run.

    Quick map preview only (skip planning and video, fastest):

    ```bash
    PYTHONPATH=src python -m mapf_lab.main --map-only
    ```

    Fast video export (with planner):

    ```bash
    PYTHONPATH=src python -m mapf_lab.main --video-fast --frame-stride 8 --max-frames 120 --video-dpi 64 --video-fps 10
    ```

    High-quality video export (with planner):

    ```bash
    PYTHONPATH=src python -m mapf_lab.main --no-video-fast --frame-stride 1 --video-dpi 140 --video-fps 24
    ```

## Algorithm Status

| Algorithm | Level | Status | Entry / File | Notes |
| --- | --- | --- | --- | --- |
| A* (GridAStarPlanner) | Low-level | Implemented | `src/mapf_lab/planners/low_level/astar.py` | Used by CBS for single-agent replanning. |
| CBS (Conflict-Based Search) | High-level | Implemented | `src/mapf_lab/planners/cbs/planner.py` | CBS currently used in `tests/cbs_test.py`. |
| ICBS (Improved Conflict-Based Search) | High-level | Implemented | `src/mapf_lab/planners/icbs/planner.py` | ICBS currently tested in `tests/icbs_test.py`. |
| db-CBS | High-level | Planned (placeholder) | `src/mapf_lab/planners/dbcbs/__init__.py` | Package exists but no planner implementation yet. |
<!-- | _Add new algorithm here_ | _High-level / Low-level_ | _Planned / In Progress / Implemented_ | `src/mapf_lab/planners/...` | _Keep one algorithm per row for easy tracking._ | -->

## CBS Demo

CBS animation output is saved in `outputs` and shown below.

- File link: [`outputs/cbs_demo.gif`](outputs/cbs_demo.gif)

![CBS Demo](outputs/cbs_demo.gif)

## ICBS Demo

ICBS animation output is saved in `outputs` and shown below.

- File link: [`outputs/icbs_demo.gif`](outputs/icbs_demo.gif)

![CBS Demo](outputs/icbs_demo.gif)