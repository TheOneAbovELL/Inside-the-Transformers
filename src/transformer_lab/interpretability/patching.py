"""Activation patching (a.k.a. causal tracing / interchange intervention).

Denoising direction (default): run the model on a **corrupted** input, but replace one activation
with its value from the **clean** run. If the clean behaviour comes back, that activation (at those
positions) carries information *sufficient*, given the rest of the corrupted computation, to
restore the behaviour. The noising direction (clean run, corrupted activation patched in) tests
whether the activation is *necessary*. The two are generally not complements (Heimersheim &
Nanda, 2024); both are supported via ``direction``.

Recovery metric::

    R = (M_patched - M_corrupt) / (M_clean - M_corrupt)

computed on batch means of a per-example metric ``M`` (default: mean log-prob of the clean answer,
or logit difference clean-answer vs corrupted-answer for symmetric corruptions). If
``|M_clean - M_corrupt|`` is below ``min_gap`` the denominator is noise: the experiment is reported
as **uninformative** and R is NaN rather than a large, meaningless number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from ..data.tasks import Task, TaskBatch
from ..evaluation.metrics import run_batched
from .stats import bootstrap_ci


@dataclass
class PatchSpec:
    """Which activation to patch: hook name, optional heads (for ``[b,s,h,...]`` activations) and an
    optional per-example position mask ``[b, s]`` (None = all positions)."""

    label: str
    hook: str
    heads: Optional[Sequence[int]] = None
    positions: Optional[torch.Tensor] = None
    meta: dict = field(default_factory=dict)


def _patch_fn(source: torch.Tensor, heads, positions):
    def fn(act, hook):
        src = source.to(act.device)
        if act.shape != src.shape:
            raise RuntimeError(f"patch source shape {tuple(src.shape)} != activation {tuple(act.shape)} at {hook.name}")
        if heads is None and positions is None:
            return src.clone()
        out = act.clone()
        if positions is None:
            pos = torch.ones(act.shape[:2], dtype=torch.bool, device=act.device)
        else:
            pos = positions.to(act.device)
        if heads is None:
            m = pos.view(*pos.shape, *([1] * (act.ndim - 2))).expand_as(act)
            out = torch.where(m, src, act)
        else:
            hs = list(heads)
            m = pos.view(*pos.shape, 1, *([1] * (act.ndim - 3))).expand_as(act[:, :, hs])
            out[:, :, hs] = torch.where(m, src[:, :, hs], act[:, :, hs])
        return out
    return fn


@dataclass
class PatchResult:
    label: str
    metric: str
    clean: float
    corrupt: float
    patched: float
    recovery: float
    recovery_ci: tuple
    gap: float
    informative: bool
    n: int
    patched_per_example: np.ndarray = field(repr=False, default=None)
    meta: dict = field(default_factory=dict)

    def row(self) -> dict:
        return {"label": self.label, "metric": self.metric, "clean": self.clean, "corrupt": self.corrupt,
                "patched": self.patched, "recovery": self.recovery, "recovery_lo": self.recovery_ci[0],
                "recovery_hi": self.recovery_ci[1], "gap": self.gap, "informative": self.informative,
                "n": self.n, **self.meta}


def _ratio_bootstrap(patched, clean, corrupt, n_boot=1000, seed=0, level: float = 0.95) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(patched)
    idx = rng.integers(0, n, size=(n_boot, n))
    num = patched[idx].mean(1) - corrupt[idx].mean(1)
    den = clean[idx].mean(1) - corrupt[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
    r = r[np.isfinite(r)]
    if r.size == 0:
        return (float("nan"), float("nan"))
    a = (1 - level) / 2
    return (float(np.quantile(r, a)), float(np.quantile(r, 1 - a)))


class Patcher:
    """Holds a clean/corrupted pair, their baseline metrics and the source caches.

    ``direction="denoise"``: corrupted run, clean activations patched in (default).
    ``direction="noise"``:   clean run, corrupted activations patched in.
    """

    def __init__(self, model, clean: TaskBatch, corrupt: TaskBatch, metric: str = "logprob",
                 direction: str = "denoise", min_gap: float = 0.25, names: Optional[Sequence[str]] = None) -> None:
        if clean.tokens.shape != corrupt.tokens.shape or not torch.equal(clean.loss_mask, corrupt.loss_mask):
            raise ValueError("clean and corrupted batches must be aligned (same shape and scored positions)")
        if direction not in ("denoise", "noise"):
            raise ValueError("direction must be denoise or noise")
        self.model, self.clean, self.corrupt = model, clean, corrupt
        self.metric, self.direction, self.min_gap = metric, direction, min_gap
        # metrics always use the CLEAN targets (and the corrupted run's alternative answer if any)
        self.alt = corrupt.alt_targets
        model.eval()
        self.m_clean = run_batched(model, clean, alt_targets=self.alt)[metric]
        self.m_corrupt = run_batched(model, self._with_clean_targets(corrupt), alt_targets=self.alt)[metric]
        self.src_batch, self.dst_batch = (clean, corrupt) if direction == "denoise" else (corrupt, clean)
        with torch.no_grad():
            _, self.src_cache = model.run_with_cache(self.src_batch.tokens, names_filter=names)

    def _with_clean_targets(self, batch: TaskBatch) -> TaskBatch:
        """The corrupted input, scored against the clean answer.

        Targets live in ``tokens[:, 1:]``; scored positions predict tokens that are themselves
        inputs only to *later* positions, so we score with a copy whose targets are the clean ones.
        For tasks whose corruption changes a scored target in-place this would matter; ours never do
        (asserted here).
        """
        m = batch.loss_mask
        if not torch.equal(batch.targets[m], self.clean.targets[m]):
            raise ValueError("corruption changed a scored target; metrics would be ill-defined")
        return batch

    @property
    def gap(self) -> float:
        return float(self.m_clean.mean() - self.m_corrupt.mean())

    @property
    def informative(self) -> bool:
        return abs(self.gap) >= self.min_gap

    def run(self, spec, extra_hooks: Sequence = ()) -> PatchResult:
        """Patch one :class:`PatchSpec`, or several simultaneously (pass a list)."""
        specs = list(spec) if isinstance(spec, (list, tuple)) else [spec]

        def hooks_for(sl: slice):
            hooks = list(extra_hooks)
            for sp in specs:
                pos = None if sp.positions is None else sp.positions[sl]
                hooks.append((sp.hook, _patch_fn(self.src_cache[sp.hook][sl], sp.heads, pos)))
            return hooks

        if len(specs) == 1:
            spec = specs[0]
        else:
            spec = PatchSpec("+".join(s.label for s in specs), "multiple", meta={"parts": [s.label for s in specs]})

        dst = self.dst_batch if self.direction == "noise" else self._with_clean_targets(self.dst_batch)
        patched = run_batched(self.model, dst, fwd_hooks_fn=hooks_for, alt_targets=self.alt)[self.metric]
        clean_m, cor_m = self.m_clean.mean(), self.m_corrupt.mean()
        if self.informative:
            rec = float((patched.mean() - cor_m) / (clean_m - cor_m))
            ci = _ratio_bootstrap(patched, self.m_clean, self.m_corrupt)
        else:
            rec, ci = float("nan"), (float("nan"), float("nan"))
        meta = dict(spec.meta)
        # Per-example diagnostics, so a ratio of means near 1 is not over-read: the median recovery
        # over examples whose own clean-corrupt gap exceeds min_gap, and the fraction that over-recover.
        gap_i = self.m_clean - self.m_corrupt
        ok = np.abs(gap_i) >= self.min_gap
        if ok.any():
            r_i = (patched[ok] - self.m_corrupt[ok]) / gap_i[ok]
            meta["median_per_example_recovery"] = float(np.median(r_i))
            meta["frac_examples_over_recovered"] = float((r_i > 1.1).mean())
            meta["n_examples_with_gap"] = int(ok.sum())
        return PatchResult(spec.label, self.metric, float(clean_m), float(cor_m), float(patched.mean()), rec, ci,
                           self.gap, self.informative, len(patched), patched, meta)


# --------------------------------------------------------------------------- common scans
def head_specs(model, positions: Optional[torch.Tensor] = None) -> list[PatchSpec]:
    return [PatchSpec(f"L{l}H{h}", f"blocks.{l}.attn.hook_z", heads=[h], positions=positions,
                      meta={"layer": l, "head": h})
            for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]


def region_specs(model, regions: dict, kinds=("resid_pre",)) -> list[PatchSpec]:
    """Patch whole residual-stream / sub-layer outputs over named position regions."""
    specs = []
    for kind in kinds:
        layers = range(model.cfg.n_layers)
        for l in layers:
            hook = {"resid_pre": f"blocks.{l}.hook_resid_pre", "attn_out": f"blocks.{l}.hook_attn_out",
                    "mlp_out": f"blocks.{l}.hook_mlp_out", "resid_post": f"blocks.{l}.hook_resid_post"}[kind]
            if kind == "mlp_out" and model.cfg.attn_only:
                continue
            for rname, mask in regions.items():
                specs.append(PatchSpec(f"{kind}@L{l}:{rname}", hook, positions=mask,
                                       meta={"component": kind, "layer": l, "region": rname}))
    return specs


def task_regions(task: Task, batch: TaskBatch) -> dict:
    """Named, per-example position masks ``[b, T]`` for region-level patching."""
    b, T = batch.tokens.shape
    regions: dict = {}

    def mask_from(fn):
        m = torch.zeros(b, T, dtype=torch.bool)
        for i in range(b):
            for j in fn(i):
                m[i, j] = True
        return m

    meta = batch.meta
    if task.name == "induction_repeat":
        regions["prefix"] = mask_from(lambda i: range(1, int(meta["first_start"][i])))
        regions["first_copy"] = mask_from(lambda i: range(int(meta["first_start"][i]), int(meta["second_start"][i])))
        regions["second_copy"] = mask_from(lambda i: range(int(meta["second_start"][i]),
                                                           int(meta["second_start"][i]) + int(meta["L"][i])))
    elif task.name == "induction_pair":
        q = T - 2
        regions["evidence_A"] = mask_from(lambda i: [int(meta["evidence_pos"][i])])
        regions["evidence_B"] = mask_from(lambda i: [int(meta["evidence_pos"][i]) + 1])
        regions["query_A"] = mask_from(lambda i: [q])
        regions["other"] = ~(regions["evidence_A"] | regions["evidence_B"] | regions["query_A"])
        regions["other"][:, q + 1:] = False
    elif task.name == "assoc_recall":
        regions["query_key"] = mask_from(lambda i: [int(meta["query_pos"][i])])
        regions["evidence_value"] = mask_from(lambda i: [int(meta["evidence_pos"][i])])
        regions["evidence_key"] = mask_from(lambda i: [int(meta["evidence_pos"][i]) - 1])
        regions["query_marker"] = mask_from(lambda i: [int(meta["query_pos"][i]) - 1])
    regions["scored"] = torch.cat([batch.loss_mask, torch.zeros(b, 1, dtype=torch.bool)], 1)
    return regions


def results_table(results: Sequence[PatchResult]) -> list[dict]:
    return [r.row() for r in results]
