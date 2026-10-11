"""Direct logit attribution (DLA) and the logit lens.

DLA decomposes a logit into additive contributions of residual-stream components (embeddings,
each head, each MLP, biases). For pre-LN models the final LayerNorm is *not* linear, so we
**freeze its scale** at the value observed on this input: with the scale fixed, LN is an affine map
and the decomposition is exact *for this forward pass* (verified by a test that the parts sum to
the actual logit). What DLA does **not** capture: indirect effects (a head that changes what later
heads do), and how the LN scale would change if a component were removed. It is an attribution
of the final linear readout, not a causal effect.

The logit lens applies the final LN + unembedding to intermediate residual streams. It is an
approximation (earlier layers were never trained to be read this way).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..data.tasks import TaskBatch


@torch.no_grad()
def direct_logit_attribution(model, batch: TaskBatch, use_alt: bool = False) -> dict:
    """Mean contribution of each component to the target logit (or target-minus-alt logit diff)
    at scored positions. Returns ``{"labels", "contrib" [n_comp], "per_example" [n_comp, b],
    "const", "total_logit"}``."""
    cfg = model.cfg
    if cfg.norm == "post":
        raise NotImplementedError("DLA requires an additive residual stream (pre-LN or no LN)")
    model.eval()
    logits, cache = model.run_with_cache(batch.tokens)
    labels, parts = cache.decompose_resid()  # [C, b, s, d]
    mask = batch.loss_mask  # [b, s-1]
    tgt = batch.targets
    W_U = model.W_U  # [d, V]
    if use_alt:
        if batch.alt_targets is None:
            raise ValueError("batch has no alt_targets")
        alt = batch.alt_targets.clamp_min(0)
        mask = mask & (batch.alt_targets >= 0)
        direction = W_U[:, tgt] - W_U[:, alt]  # [d, b, s-1]
        bias = model.b_U[tgt] - model.b_U[alt]
    else:
        direction = W_U[:, tgt]
        bias = model.b_U[tgt]
    direction = direction.permute(1, 2, 0)  # [b, s-1, d]
    parts = parts[:, :, :-1]  # drop last position (it predicts nothing)
    if cfg.norm == "pre":
        scale = cache["ln_final_scale"][:, :-1]  # [b, s-1, 1]
        w, b_ln = model.ln_final.w, model.ln_final.b
        centred = parts - parts.mean(-1, keepdim=True)  # LN centring is linear: apply per component
        contrib = ((centred / scale) * w * direction).sum(-1)  # [C, b, s-1]
        const = (b_ln * direction).sum(-1) + bias
    else:
        contrib = (parts * direction).sum(-1)
        const = bias
    mf = mask.float()
    denom = mf.sum(1).clamp_min(1)
    per_ex = (contrib * mf).sum(-1) / denom  # [C, b]
    const_ex = (const * mf).sum(-1) / denom
    final = logits[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    if use_alt:
        final = final - logits[:, :-1].gather(-1, alt.unsqueeze(-1)).squeeze(-1)
    total = (final * mf).sum(-1) / denom
    return {"labels": labels, "contrib": per_ex.mean(1).numpy(), "per_example": per_ex.numpy(),
            "const": float(const_ex.mean()), "total_logit": float(total.mean()),
            "reconstruction_error": float((per_ex.sum(0) + const_ex - total).abs().max())}


@torch.no_grad()
def logit_lens(model, batch: TaskBatch) -> dict:
    """Target log-prob and accuracy at scored positions when decoding each layer's residual stream."""
    model.eval()
    _, cache = model.run_with_cache(batch.tokens, names_filter=lambda n: "resid" in n or n == "hook_embed")
    mask = batch.loss_mask
    rows = []
    stages = [("resid_pre", 0)] + [("resid_post", l) for l in range(model.cfg.n_layers)]
    for kind, l in stages:
        resid = cache[kind, l]
        logits = model.ln_final(resid) @ model.W_U + model.b_U
        logp = torch.log_softmax(logits[:, :-1], -1).gather(-1, batch.targets.unsqueeze(-1)).squeeze(-1)
        acc = (logits[:, :-1].argmax(-1) == batch.targets)
        rows.append({"stage": f"{kind}.{l}", "logprob": float(logp[mask].mean()), "accuracy": float(acc[mask].float().mean())})
    return {"stages": rows}
