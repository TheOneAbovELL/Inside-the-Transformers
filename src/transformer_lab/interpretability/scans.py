"""Reusable experiment scans: ablate every component, patch every head, random-component controls."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from ..data.tasks import Task, TaskBatch
from ..evaluation.metrics import run_batched
from .interventions import Reference, head_ablation_hooks, head_label, mlp_ablation_hooks, reference_names
from .patching import Patcher, head_specs, region_specs, task_regions
from .stats import bootstrap_ci, paired_difference


def _effect_row(label: str, treated: dict, base: dict, metrics: Sequence[str], **extra) -> dict:
    row = {"component": label, **extra}
    for m in metrics:
        d = paired_difference(treated[m], base[m])
        row[f"{m}"] = float(np.nanmean(treated[m]))
        row[f"d_{m}"] = d["mean"]
        row[f"d_{m}_lo"], row[f"d_{m}_hi"] = d["lo"], d["hi"]
    return row


@torch.no_grad()
def ablation_scan(model, batch: TaskBatch, ref_batch: TaskBatch, method: str = "mean",
                  metrics: Sequence[str] = ("logprob", "accuracy", "loss"), include_mlps: bool = True,
                  neurons_top_k: int = 0) -> dict:
    """Ablate each head (and each MLP block) individually; paired effects vs the unablated model.

    ``neurons_top_k > 0`` additionally ablates, per MLP layer, the k neurons with the largest mean
    |activation| on the reference batch (a crude but explicit neuron-level intervention).
    """
    model.eval()
    ref = Reference.from_batch(model, ref_batch, reference_names(model, heads=True, mlps=include_mlps,
                                                                  neurons=neurons_top_k > 0))
    base = run_batched(model, batch)
    rows = [{"component": "none (baseline)", **{m: float(np.nanmean(base[m])) for m in metrics}}]
    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            hooks = head_ablation_hooks([(l, h)], method, ref)
            treated = run_batched(model, batch, fwd_hooks_fn=lambda sl, hooks=hooks: hooks)
            rows.append(_effect_row(head_label(l, h), treated, base, metrics, kind="head", layer=l, head=h))
    if include_mlps and not model.cfg.attn_only:
        for l in range(model.cfg.n_layers):
            hooks = mlp_ablation_hooks([l], method, ref)
            treated = run_batched(model, batch, fwd_hooks_fn=lambda sl, hooks=hooks: hooks)
            rows.append(_effect_row(f"L{l}.mlp", treated, base, metrics, kind="mlp_block", layer=l))
            if neurons_top_k:
                act = ref.acts[f"blocks.{l}.mlp.hook_post"].abs().mean((0, 1))
                idx = torch.topk(act, neurons_top_k).indices.tolist()
                hooks = mlp_ablation_hooks([l], method, ref, neurons=idx)
                treated = run_batched(model, batch, fwd_hooks_fn=lambda sl, hooks=hooks: hooks)
                rows.append(_effect_row(f"L{l}.mlp.top{neurons_top_k}neurons", treated, base, metrics,
                                        kind="mlp_neurons", layer=l, neurons=idx))
    return {"method": method, "n": len(batch), "rows": rows, "baseline": base}


@torch.no_grad()
def random_head_control(model, batch: TaskBatch, ref: Reference, n_heads: int, n_samples: int,
                        exclude: Sequence[tuple[int, int]] = (), method: str = "mean", metric: str = "logprob",
                        seed: int = 0, base: Optional[dict] = None) -> tuple[np.ndarray, bool]:
    """Mean effect of ablating ``n_heads`` other heads (excluding ``exclude``), for many head sets.

    If there are at most ``n_samples`` possible sets they are **all** evaluated (exhaustive control);
    otherwise ``n_samples`` distinct random sets. Returns ``(effects, exhaustive)``. With an
    exhaustive control of K sets a permutation p-value cannot go below 1/(K+1) — callers must
    account for that instead of pretending to significance.
    """
    import itertools
    from math import comb

    pool = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads) if (l, h) not in set(exclude)]
    if len(pool) < n_heads:
        return np.array([]), True
    total = comb(len(pool), n_heads)
    if total <= n_samples:
        combos, exhaustive = list(itertools.combinations(pool, n_heads)), True
    else:
        rng = np.random.default_rng(seed)
        seen: set = set()
        while len(seen) < n_samples:
            seen.add(tuple(sorted(pool[i] for i in rng.choice(len(pool), size=n_heads, replace=False))))
        combos, exhaustive = sorted(seen), False
    base = base if base is not None else run_batched(model, batch)
    effects = []
    for choice in combos:
        hooks = head_ablation_hooks(list(choice), method, ref)
        treated = run_batched(model, batch, fwd_hooks_fn=lambda sl, hooks=hooks: hooks)
        effects.append(float(np.nanmean(treated[metric] - base[metric])))
    return np.array(effects), exhaustive


@torch.no_grad()
def head_patching_scan(model, task: Task, clean: TaskBatch, corrupt: TaskBatch, metric: str = "logprob",
                       positions: str = "scored", direction: str = "denoise", min_gap: float = 0.25) -> dict:
    """Patch every head's ``z`` (at ``positions``: 'scored' or 'all') from clean into corrupted."""
    p = Patcher(model, clean, corrupt, metric=metric, direction=direction, min_gap=min_gap,
                names=[f"blocks.{l}.attn.hook_z" for l in range(model.cfg.n_layers)])
    pos = task_regions(task, clean)["scored"] if positions == "scored" else None
    res = [p.run(s) for s in head_specs(model, pos)]
    return {"clean": float(p.m_clean.mean()), "corrupt": float(p.m_corrupt.mean()), "gap": p.gap,
            "informative": p.informative, "metric": metric, "direction": direction, "positions": positions,
            "n": len(clean), "rows": [r.row() for r in res]}


@torch.no_grad()
def region_patching_scan(model, task: Task, clean: TaskBatch, corrupt: TaskBatch, metric: str = "logprob",
                         kinds=("resid_pre", "attn_out", "mlp_out"), min_gap: float = 0.25) -> dict:
    """Patch residual-stream / sub-layer outputs over named regions (where does information live?)."""
    names = []
    for l in range(model.cfg.n_layers):
        names += [f"blocks.{l}.hook_resid_pre", f"blocks.{l}.hook_attn_out"]
        if not model.cfg.attn_only:
            names.append(f"blocks.{l}.hook_mlp_out")
    p = Patcher(model, clean, corrupt, metric=metric, min_gap=min_gap, names=names)
    regions = task_regions(task, clean)
    res = [p.run(s) for s in region_specs(model, regions, kinds)]
    return {"clean": float(p.m_clean.mean()), "corrupt": float(p.m_corrupt.mean()), "gap": p.gap,
            "informative": p.informative, "metric": metric, "n": len(clean), "regions": list(regions),
            "rows": [r.row() for r in res]}


def as_grid(rows: Sequence[dict], value: str, n_layers: int, n_heads: int) -> np.ndarray:
    g = np.full((n_layers, n_heads), np.nan)
    for r in rows:
        if r.get("kind", "head") == "head" and "layer" in r and "head" in r:
            g[r["layer"], r["head"]] = r[value]
    return g


def summarize_metric(values) -> dict:
    return bootstrap_ci(values)
