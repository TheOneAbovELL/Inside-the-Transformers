"""Per-example task metrics computed from logits.

All metrics are computed per *example* (averaging over that example's scored positions) so that
uncertainty can be bootstrapped over independent task instances. Definitions:

* ``loss``         mean NLL ``-log p(target)`` over scored positions (nats)
* ``logprob``      ``-loss``
* ``prob``         mean ``p(target)``
* ``accuracy``     fraction of scored positions where ``argmax == target``
* ``seq_correct``  1 if every scored position is correct
* ``logit_diff``   mean ``logit[target] - logit[alt_target]`` where an alternative answer is defined
                   (symmetric corruptions); NaN otherwise
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from ..data.tasks import TaskBatch
from ..interpretability.stats import bootstrap_ci

METRIC_NAMES = ("loss", "logprob", "prob", "accuracy", "seq_correct", "logit_diff")
HIGHER_IS_BETTER = {"loss": False, "logprob": True, "prob": True, "accuracy": True, "seq_correct": True,
                    "logit_diff": True}


def per_example_metrics(logits: torch.Tensor, batch: TaskBatch, alt_targets: Optional[torch.Tensor] = None) -> dict:
    """``logits [n,T,V]`` + batch -> dict of ``np.ndarray [n]``."""
    logits = logits[:, :-1].float()
    targets = batch.targets.to(logits.device)
    mask = batch.loss_mask.to(logits.device)
    logp = torch.log_softmax(logits, dim=-1)
    tgt_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mf = mask.float()
    count = mf.sum(1).clamp_min(1)
    loss = -(tgt_logp * mf).sum(1) / count
    prob = (tgt_logp.exp() * mf).sum(1) / count
    correct = (logits.argmax(-1) == targets) & mask
    acc = correct.float().sum(1) / count
    seq_correct = (correct == mask).all(1).float()
    out = {
        "loss": loss, "logprob": -loss, "prob": prob, "accuracy": acc, "seq_correct": seq_correct,
    }
    alt = alt_targets if alt_targets is not None else batch.alt_targets
    if alt is not None:
        alt = alt.to(logits.device)
        valid = (alt >= 0) & mask
        tgt_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        alt_logit = logits.gather(-1, alt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        vf = valid.float()
        ld = ((tgt_logit - alt_logit) * vf).sum(1) / vf.sum(1).clamp_min(1)
        ld[vf.sum(1) == 0] = float("nan")
        out["logit_diff"] = ld
    else:
        out["logit_diff"] = torch.full_like(loss, float("nan"))
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


@torch.no_grad()
def run_batched(model, batch: TaskBatch, fwd_hooks_fn: Optional[Callable] = None, chunk: int = 256,
                alt_targets: Optional[torch.Tensor] = None) -> dict:
    """Evaluate per-example metrics in chunks (bounded memory).

    ``fwd_hooks_fn(chunk_slice) -> list of hooks`` lets interventions depend on which examples are
    in the chunk (e.g. patching in the matching clean activations).
    """
    device = next(model.parameters()).device
    parts: list[dict] = []
    for start in range(0, len(batch), chunk):
        sl = slice(start, min(start + chunk, len(batch)))
        sub = batch.subset(np.arange(sl.start, sl.stop))
        hooks = fwd_hooks_fn(sl) if fwd_hooks_fn else []
        logits = model.run_with_hooks(sub.tokens.to(device), fwd_hooks=hooks)
        alt = None if alt_targets is None else alt_targets[sl]
        parts.append(per_example_metrics(logits, sub, alt))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summarize(metrics: dict, n_boot: int = 1000) -> dict:
    """Mean + bootstrap CI for each metric."""
    return {k: bootstrap_ci(v, n_boot=n_boot) for k, v in metrics.items() if not np.all(np.isnan(v))}


def solver_accuracy(pred: torch.Tensor, batch: TaskBatch) -> float:
    """Position-level accuracy of a baseline solver's predictions on scored positions."""
    m = batch.loss_mask
    return float(((pred == batch.targets) & m).sum() / m.sum())
