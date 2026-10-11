"""Inspection tools: one example, several examples, correct vs incorrect, checkpoints, traces.

All functions return plain dicts / numpy arrays (no plotting) so the CLI, the UI and reports can
share them. Large activations are never retained beyond what each function needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..data.tasks import Task, TaskBatch
from ..evaluation.metrics import per_example_metrics, run_batched
from ..interpretability.attention_scores import head_scores


@torch.no_grad()
def inspect_example(model, task: Task, batch: TaskBatch, index: int, top_k: int = 3) -> dict:
    """Everything needed to look at one example: tokens, targets, predictions, attention, residual norms."""
    model.eval()
    ex = batch.subset([index])
    logits, cache = model.run_with_cache(ex.tokens)
    probs = torch.softmax(logits[0, :-1], -1)
    tokens = ex.tokens[0]
    mask = ex.loss_mask[0]
    rows = []
    for t in range(len(tokens) - 1):
        top = torch.topk(probs[t], top_k)
        tgt = int(tokens[t + 1])
        rows.append({
            "pos": t, "token": task.token_str(tokens[t]), "target": task.token_str(tgt), "scored": bool(mask[t]),
            "pred": task.token_str(int(top.indices[0])), "p_target": float(probs[t, tgt]),
            "correct": bool(int(top.indices[0]) == tgt),
            "top": [(task.token_str(int(i)), float(p)) for p, i in zip(top.values, top.indices)],
        })
    L = model.cfg.n_layers
    resid_norms = {f"resid_pre.{l}": cache["resid_pre", l][0].norm(dim=-1).numpy() for l in range(L)}
    resid_norms[f"resid_post.{L - 1}"] = cache["resid_post", L - 1][0].norm(dim=-1).numpy()
    out = {
        "index": index,
        "tokens": task.decode(tokens),
        "positions": rows,
        "patterns": np.stack([cache["pattern", l][0].numpy() for l in range(L)]),  # [L,H,q,k]
        "resid_norms": resid_norms,
        "attn_out_norms": np.stack([cache["attn_out", l][0].norm(dim=-1).numpy() for l in range(L)]),
        "meta": {k: (v[index].item() if hasattr(v[index], "item") else v[index]) for k, v in batch.meta.items()},
        "metrics": {k: float(v[0]) for k, v in per_example_metrics(logits, ex).items()},
    }
    if not model.cfg.attn_only:
        out["mlp_post"] = np.stack([cache["mlp_post", l][0].numpy() for l in range(L)])  # [L,s,m]
        out["mlp_out_norms"] = np.stack([cache["mlp_out", l][0].norm(dim=-1).numpy() for l in range(L)])
    return out


@torch.no_grad()
def correct_vs_incorrect(model, batch: TaskBatch, metric: str = "seq_correct") -> dict:
    """Split examples by whether the model got them right and compare head attention scores.

    A head whose induction score differs sharply between the groups is a *candidate* for being
    involved — still only observational evidence.
    """
    m = run_batched(model, batch)
    ok = m[metric] >= 0.5 if metric == "seq_correct" else m[metric] >= np.median(m[metric])
    out = {"n_correct": int(ok.sum()), "n_incorrect": int((~ok).sum())}
    for name, sel in (("correct", ok), ("incorrect", ~ok)):
        if sel.sum() == 0:
            out[name] = None
            continue
        sub = batch.subset(np.nonzero(sel)[0])
        out[name] = {k: v.tolist() for k, v in head_scores(model, sub).items()}
        if batch.meta:
            out[f"{name}_meta_means"] = {k: float(np.mean(np.asarray(v)[sel])) for k, v in batch.meta.items()
                                         if np.issubdtype(np.asarray(v).dtype, np.number)}
    return out


@torch.no_grad()
def compare_examples(model, task: Task, batch: TaskBatch, indices) -> list[dict]:
    """Compact per-example summary for several examples side by side."""
    m = run_batched(model, batch.subset(list(indices)))
    return [{"index": int(i), "text": " ".join(task.decode(batch.tokens[i])),
             **{k: float(v[j]) for k, v in m.items()}} for j, i in enumerate(indices)]


def list_checkpoints(run_dir) -> list[tuple[int, Path]]:
    ck = Path(run_dir) / "checkpoints"
    out = []
    for p in ck.glob("step_*.pt"):
        out.append((int(p.stem.split("_")[1]), p))
    return sorted(out)


@torch.no_grad()
def checkpoint_trajectory(run_dir, batch: TaskBatch, extra_fn=None) -> list[dict]:
    """Metrics + head scores at every saved checkpoint (e.g. to study how a mechanism forms)."""
    from ..training.checkpoint import load_model

    rows = []
    for step, path in list_checkpoints(run_dir):
        model, _ = load_model(path)
        m = run_batched(model, batch)
        sc = head_scores(model, batch)
        row = {"step": step, "loss": float(m["loss"].mean()), "accuracy": float(m["accuracy"].mean()),
               "max_prev_token_L0": float(sc["prev_token"][0].max()),
               "max_induction": float(np.nanmax(sc["induction"][1:])) if model.cfg.n_layers > 1 else float("nan"),
               "head_scores": {k: v.tolist() for k, v in sc.items()}}
        if extra_fn is not None:
            row.update(extra_fn(model, batch))
        rows.append(row)
    return rows


def export_npz(path, **arrays) -> Path:
    """Compact analysis artifact (compressed numpy archive)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


@torch.no_grad()
def trace_prediction(model, task: Task, batch: TaskBatch, index: int, position: Optional[int] = None) -> dict:
    """Trace one prediction back through the network.

    For the chosen scored position: the direct logit attribution of each component to the target
    logit, the logit lens across layers, and for every head the source positions it attends to most.
    """
    from ..interpretability.attribution import direct_logit_attribution, logit_lens

    ex = batch.subset([index])
    scored = torch.nonzero(ex.loss_mask[0]).flatten().tolist()
    pos = scored[-1] if position is None else position
    if pos not in scored:
        raise ValueError(f"position {pos} is not a scored position ({scored})")
    single = TaskBatch(ex.tokens, torch.zeros_like(ex.loss_mask), ex.meta, ex.alt_targets)
    single.loss_mask[0, pos] = True
    dla = direct_logit_attribution(model, single)
    lens = logit_lens(model, single)
    _, cache = model.run_with_cache(ex.tokens, names_filter=lambda n: n.endswith("hook_pattern"))
    toks = task.decode(ex.tokens[0])
    heads = []
    for l in range(model.cfg.n_layers):
        p = cache["pattern", l][0, :, pos]  # [H, k]
        for h in range(model.cfg.n_heads):
            top = torch.topk(p[h], min(3, pos + 1))
            heads.append({"head": f"L{l}H{h}", "top_sources": [(int(i), toks[int(i)], float(w))
                                                               for w, i in zip(top.values, top.indices)]})
    order = np.argsort(-np.abs(dla["contrib"]))
    return {
        "index": index, "position": pos, "token": toks[pos], "target": toks[pos + 1],
        "dla": [(dla["labels"][i], float(dla["contrib"][i])) for i in order],
        "dla_const": dla["const"], "target_logit": dla["total_logit"],
        "logit_lens": lens["stages"], "attention": heads,
    }
