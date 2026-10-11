"""Model evaluation: baselines, held-out sets, sweeps and multi-checkpoint evaluation."""

from __future__ import annotations

import csv
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..data.tasks import Task, TaskBatch, induction_solver, most_frequent_solver
from .metrics import run_batched, solver_accuracy, summarize


def baselines(task: Task, batch: TaskBatch) -> dict:
    """Context for model accuracy: exact solver, chance, and heuristic shortcuts."""
    return {
        "algorithmic": solver_accuracy(task.solve(batch), batch),
        "chance": task.chance_accuracy,
        "induction_rule": solver_accuracy(induction_solver(batch.tokens, batch.loss_mask), batch),
        "most_frequent_context_token": solver_accuracy(most_frequent_solver(batch.tokens, batch.loss_mask), batch),
        "n_examples": len(batch),
        "n_scored_positions": batch.n_scored,
    }


def evaluate_model(model, batch: TaskBatch, n_boot: int = 1000) -> dict:
    """Summary metrics (mean + 95% bootstrap CI over examples)."""
    model.eval()
    return summarize(run_batched(model, batch), n_boot=n_boot)


def sweep(model, task: Task, param: str, values, n: int = 256, seed: int = 12345, split: str = "train",
          group_by_meta: Optional[str] = None) -> list[dict]:
    """Evaluate across task variants, e.g. ``sweep(model, task, "evidence_pos", range(1, 28))``.
    ``param`` may join several parameters with ``+`` to set them to the same value.

    Every variant must keep the model's vocabulary; sequences longer than ``n_ctx`` are skipped
    and reported as such (never silently truncated).
    """
    rows = []
    names = param.split("+")  # "min_len+max_len" sets several parameters to the same value
    for v in values:
        variant = task.with_params(**{name: v for name in names})
        if variant.seq_len > model.cfg.n_ctx:
            rows.append({param: v, "skipped": f"seq_len {variant.seq_len} > n_ctx {model.cfg.n_ctx}"})
            continue
        # zlib.crc32 is stable across processes (Python's str hash is salted per process)
        rng = np.random.default_rng([seed, zlib.crc32(f"{param}={v}".encode())])
        batch = variant.sample(n, rng, split)
        s = summarize(run_batched(model, batch))
        rows.append({param: v, **{f"{k}_mean": s[k]["mean"] for k in s}, **{f"{k}_lo": s[k]["lo"] for k in s},
                     **{f"{k}_hi": s[k]["hi"] for k in s}, "n": len(batch),
                     "algorithmic": solver_accuracy(variant.solve(batch), batch)})
    return rows


def metric_by_meta(model, batch: TaskBatch, key: str, metric: str = "accuracy") -> list[dict]:
    """Group per-example metrics by a metadata field (e.g. ``distance``) — for position analysis."""
    m = run_batched(model, batch)[metric]
    vals = np.asarray(batch.meta[key])
    rows = []
    for v in np.unique(vals):
        sel = m[vals == v]
        rows.append({key: int(v), f"{metric}_mean": float(sel.mean()), "n": int(sel.size)})
    return rows


def write_csv(rows: list[dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
