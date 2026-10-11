"""Ablation interventions on attention heads and MLPs.

Component boundaries
--------------------
* **Head**: ``blocks.{l}.attn.hook_z[:, :, h]`` — the head's weighted values *before* ``W_O``.
  Because ``attn_out = sum_h z_h W_O[h] + b_O``, replacing ``z_h`` removes/replaces exactly head
  h's contribution to the residual stream and nothing else (unit-tested).
* **MLP block**: ``blocks.{l}.hook_mlp_out`` — the whole MLP's write to the residual stream.
* **MLP neurons**: ``blocks.{l}.mlp.hook_post[..., idx]`` — post-activation units; their effect
  passes through ``W_out``. Ablating neurons is a different intervention from ablating the block
  output and the two are reported separately.

Ablation methods
----------------
* ``zero``: set to 0. Simple but can push activations off-distribution (a head's typical output
  is not zero), so effects can be exaggerated.
* ``mean``: replace with the mean over a *separate reference batch from the same distribution*,
  per position. Removes example-specific information while staying near the typical activation.
* ``resample``: replace with the activation from a *different* reference example at the same
  position. Removes information about this example while keeping a realistic activation.

No method is "correct"; disagreements between them are themselves informative and reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from ..data.tasks import TaskBatch

METHODS = ("zero", "mean", "resample")


def parse_heads(spec: str) -> list[tuple[int, int]]:
    """``"1.2,0.3"`` -> ``[(1, 2), (0, 3)]``."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            l, h = part.split(".")
            out.append((int(l), int(h)))
    return out


def head_label(l: int, h: int) -> str:
    return f"L{l}H{h}"


@dataclass
class Reference:
    """Activations of a reference batch used by mean / resample ablation."""

    acts: dict  # hook name -> tensor [n_ref, s, ...]
    seed: int = 0
    _calls: int = 0  # advances per resample call so successive chunks get independent donors

    @classmethod
    @torch.no_grad()
    def from_batch(cls, model, batch: TaskBatch, names: Sequence[str], seed: int = 0) -> "Reference":
        _, cache = model.run_with_cache(batch.tokens, names_filter=list(names))
        return cls({n: cache[n] for n in names}, seed)

    def mean(self, name: str) -> torch.Tensor:
        return self.acts[name].mean(0)  # [s, ...]

    def sample(self, name: str, n: int, call_seed: int) -> torch.Tensor:
        """Donor activations for ``n`` examples, drawn with replacement from the reference batch.

        Donors come from a *different* batch (same distribution, different seed), so an example
        never receives its own activation. Each call advances an internal counter: previously every
        evaluation chunk reused identical donor indices (fixed 2026-10-11, regression-tested).
        Deterministic given the sequence of calls.
        """
        rng = np.random.default_rng([self.seed, call_seed, self._calls])
        self._calls += 1
        idx = rng.integers(0, self.acts[name].shape[0], size=n)
        return self.acts[name][torch.from_numpy(idx)]


def _replacement(method: str, act: torch.Tensor, name: str, ref: Optional[Reference], call_seed: int) -> torch.Tensor:
    if method == "zero":
        return torch.zeros_like(act)
    if ref is None:
        raise ValueError(f"{method} ablation needs a reference batch")
    s = act.shape[1]
    if method == "mean":
        r = ref.mean(name)
        if r.shape[0] < s:
            raise ValueError("reference sequences shorter than the ablated batch")
        return r[:s].unsqueeze(0).expand_as(act)
    if method == "resample":
        return ref.sample(name, act.shape[0], call_seed)[:, :s]
    raise ValueError(f"method must be one of {METHODS}")


def head_ablation_hooks(heads: Sequence[tuple[int, int]], method: str = "mean", ref: Optional[Reference] = None,
                        positions: Optional[torch.Tensor] = None, call_seed: int = 0) -> list:
    """Hooks that ablate the given heads' ``z``. ``positions``: optional bool mask ``[b, s]``."""
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    hooks = []
    for l, hs in by_layer.items():
        name = f"blocks.{l}.attn.hook_z"

        def fn(z, hook, hs=hs, name=name):
            rep = _replacement(method, z, name, ref, call_seed)
            z = z.clone()
            if positions is None:
                z[:, :, hs] = rep[:, :, hs]
            else:
                m = positions.to(z.device)[..., None, None].expand(-1, -1, len(hs), z.shape[-1])
                z[:, :, hs] = torch.where(m, rep[:, :, hs], z[:, :, hs])
            return z
        hooks.append((name, fn))
    return hooks


def mlp_ablation_hooks(layers: Sequence[int], method: str = "mean", ref: Optional[Reference] = None,
                       neurons: Optional[Sequence[int]] = None, call_seed: int = 0) -> list:
    """Ablate whole MLP outputs (``neurons is None``) or selected post-activation neurons."""
    hooks = []
    for l in layers:
        if neurons is None:
            name = f"blocks.{l}.hook_mlp_out"

            def fn(act, hook, name=name):
                return _replacement(method, act, name, ref, call_seed)
        else:
            name = f"blocks.{l}.mlp.hook_post"
            idx = list(neurons)

            def fn(act, hook, name=name, idx=idx):
                rep = _replacement(method, act, name, ref, call_seed)
                act = act.clone()
                act[..., idx] = rep[..., idx]
                return act
        hooks.append((name, fn))
    return hooks


def reference_names(model, heads: bool = True, mlps: bool = False, neurons: bool = False) -> list[str]:
    names = []
    for l in range(model.cfg.n_layers):
        if heads:
            names.append(f"blocks.{l}.attn.hook_z")
        if mlps and not model.cfg.attn_only:
            names.append(f"blocks.{l}.hook_mlp_out")
        if neurons and not model.cfg.attn_only:
            names.append(f"blocks.{l}.mlp.hook_post")
    return names
