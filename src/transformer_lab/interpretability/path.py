"""Path-specific interventions between two heads (a restricted form of path patching).

Question answered: *does head S (sender) affect head R (receiver) through R's query, key, or value
input?* The intervention replaces S's contribution **only inside R's chosen input**, leaving S's
contribution to the residual stream (and therefore to every other head, R's other inputs, and the
logits) untouched.

Mechanics, in a single forward pass with ``split_qkv_input`` enabled::

    1. a hook on blocks.{l_S}.attn.hook_z records z_S and computes result_S = z_S W_O[S]   [b,s,d]
    2. a hook on blocks.{l_R}.hook_{q|k|v}_input (shape [b,s,h,d]) edits only head R's copy:
           x[:, :, R] <- x[:, :, R] - result_S + replacement_S

``replacement_S`` is the per-position mean of result_S over a reference batch ("mean") or zero
("zero"). Everything downstream of R's other inputs is computed normally. Because the residual
input of R is normalised by LayerNorm *after* the edit, the effect includes the change in LN scale
— an inherent property of intervening before LN that is documented, not hidden.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch

from ..data.tasks import TaskBatch
from ..evaluation.metrics import run_batched


@contextmanager
def split_inputs(model):
    prev = [b.split_qkv_input for b in model.blocks]
    model.set_split_qkv_input(True)
    try:
        yield model
    finally:
        for b, p in zip(model.blocks, prev):
            b.split_qkv_input = p


@torch.no_grad()
def mean_head_result(model, ref: TaskBatch, layer: int, head: int) -> torch.Tensor:
    """Per-position mean of a head's residual contribution over a reference batch: ``[s, d]``."""
    _, cache = model.run_with_cache(ref.tokens, names_filter=[f"blocks.{layer}.attn.hook_z"])
    return cache.head_results(layer)[:, :, head].mean(0)


def path_hooks(model, sender, receiver, channel: str, replacement=None) -> list:
    """Hooks implementing the sender(s) -> receiver(s)[channel] intervention (use inside ``split_inputs``).

    ``sender`` / ``receiver`` are ``(layer, head)`` tuples or lists of them; a sender may also be
    ``(layer, "mlp")`` for an MLP block's output. Every receiver head's ``channel`` input has the
    summed contribution of every sender replaced; nothing else changes (in particular the senders'
    effects on *other* components — e.g. a head's output read by the next MLP — are untouched, so a
    path that is mediated by an intermediate component is NOT removed).
    ``replacement``: ``None`` (zero), a ``[s, d]`` tensor (single sender), or a dict
    ``{sender: [s, d]}`` (one per sender).
    """
    senders = [sender] if isinstance(sender[0], int) else [tuple(s) for s in sender]
    receivers = [receiver] if isinstance(receiver[0], int) else list(receiver)
    if channel not in ("q", "k", "v"):
        raise ValueError("channel must be q, k or v")
    if max(l for l, _ in senders) >= min(l for l, _ in receivers):
        raise ValueError("every sender must be in an earlier layer than every receiver")
    if replacement is not None and not isinstance(replacement, dict):
        if len(senders) != 1:
            raise ValueError("pass a dict of replacements for multiple senders")
        replacement = {tuple(senders[0]): replacement}
    store: dict = {}
    hooks = []
    for l, h in senders:
        if h != "mlp":
            continue
        if model.cfg.attn_only:
            raise ValueError("attention-only model has no MLP sender")

        def record_mlp(out, hook, l=l):
            rep = torch.zeros_like(out) if replacement is None else \
                replacement[(l, "mlp")][: out.shape[1]].to(out).expand_as(out)
            store[(l, "mlp")] = rep - out
            return None
        hooks.append((f"blocks.{l}.hook_mlp_out", record_mlp))
    head_senders = [(l, h) for l, h in senders if h != "mlp"]
    for ls in sorted({l for l, _ in head_senders}):
        hs = [h for l, h in head_senders if l == ls]
        W_O = model.blocks[ls].attn.W_O

        def record(z, hook, ls=ls, hs=hs, W_O=W_O):
            for h in hs:
                res = torch.einsum("bse,ed->bsd", z[:, :, h], W_O[h])
                rep = torch.zeros_like(res) if replacement is None else \
                    replacement[(ls, h)][: res.shape[1]].to(res).expand_as(res)
                store[(ls, h)] = rep - res  # delta to add to receivers' inputs
            return None
        hooks.append((f"blocks.{ls}.attn.hook_z", record))
    for lr in sorted({l for l, _ in receivers}):
        hr = [h for l, h in receivers if l == lr]

        def edit(x, hook, hr=hr):
            delta = sum(store.values())
            x = x.clone()
            for h in hr:
                x[:, :, h] = x[:, :, h] + delta
            return x
        hooks.append((f"blocks.{lr}.hook_{channel}_input", edit))
    return hooks


@torch.no_grad()
def path_effect(model, batch: TaskBatch, sender, receiver, channel: str, replacement=None, metric: str = "logprob",
                extra_metric_hooks=()) -> dict:
    """Per-example metric under the path intervention (and the unmodified baseline)."""
    model.eval()
    base = run_batched(model, batch)[metric]
    with split_inputs(model):
        hooks = path_hooks(model, sender, receiver, channel, replacement)
        treated = run_batched(model, batch, fwd_hooks_fn=lambda sl: hooks)[metric]
    return {"baseline": base, "treated": treated}


@torch.no_grad()
def mean_head_results(model, ref: TaskBatch, heads) -> dict:
    """``{(layer, head): [s, d]}`` per-position mean residual contribution of each head."""
    out = {}
    layers = sorted({l for l, h in heads if h != "mlp"})
    mlps = sorted({l for l, h in heads if h == "mlp"})
    names = [f"blocks.{l}.attn.hook_z" for l in layers] + [f"blocks.{l}.hook_mlp_out" for l in mlps]
    _, cache = model.run_with_cache(ref.tokens, names_filter=names)
    for l in layers:
        res = cache.head_results(l)
        for ll, h in heads:
            if ll == l and h != "mlp":
                out[(l, h)] = res[:, :, h].mean(0)
    for l in mlps:
        out[(l, "mlp")] = cache["mlp_out", l].mean(0)
    return out


@torch.no_grad()
def pattern_under_path(model, batch: TaskBatch, sender, receiver, channel: str, replacement=None) -> torch.Tensor:
    """Receiver's attention pattern ``[b,q,k]`` under the path intervention (for mechanism checks)."""
    lr, hr = receiver
    with split_inputs(model):
        hooks = path_hooks(model, sender, receiver, channel, replacement)
        _, cache = model.run_with_cache(batch.tokens, names_filter=[f"blocks.{lr}.attn.hook_pattern"], fwd_hooks=hooks)
    return cache["pattern", lr][:, hr]
