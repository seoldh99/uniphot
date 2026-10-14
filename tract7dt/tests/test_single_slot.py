"""A one-slot final pass runs in a spawned worker, like every other slot (R17).

The final grid pass used to solve the grids inside the parent process when it
resolved to one slot (one GPU with one worker, or a single grid). The parent
imports JAX without x64, and JAX reads that setting only at import, so the
worker initializer's x64 check raised there and every such run died after the
island stage. Grid solves now always run in a worker that sets x64 before its
JAX import; the parent only hands tasks to worker processes.
"""
from __future__ import annotations

import ast
from pathlib import Path

import tract7dt.fit.grid_gpu.pipeline as gpu_pipeline

WORKER_ONLY = ("_worker_init", "run_one_grid", "run_one_grid_managed")


def test_the_parent_never_runs_a_grid_solve_itself():
    tree = ast.parse(Path(gpu_pipeline.__file__).read_text())
    direct = sorted(
        (node.func.id, node.lineno) for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in WORKER_ONLY)
    assert direct == [], (
        f"grid solves/worker init called in the parent: {direct}; hand them "
        "to a worker process (initializer=/submit) instead")
