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
PYTHONPATH=src python -m mapf_lab.main
```

## Algorithm Status

| Algorithm | Level | Status | Entry / File | Notes |
| --- | --- | --- | --- | --- |
| A* (GridAStarPlanner) | Low-level | Implemented | `src/mapf_lab/planners/low_level/astar.py` | Used by CBS for single-agent replanning. |
| CBS (Conflict-Based Search) | High-level | Implemented | `src/mapf_lab/planners/cbs/planner.py` | Main multi-agent solver currently used in `main.py`. |
| db-CBS | High-level | Planned (placeholder) | `src/mapf_lab/planners/dbcbs/__init__.py` | Package exists but no planner implementation yet. |
<!-- | _Add new algorithm here_ | _High-level / Low-level_ | _Planned / In Progress / Implemented_ | `src/mapf_lab/planners/...` | _Keep one algorithm per row for easy tracking._ | -->

## CBS Demo

CBS animation output is saved in `outputs` and shown below.

- File link: [`outputs/cbs_demo.gif`](outputs/cbs_demo.gif)

![CBS Demo](outputs/cbs_demo.gif)