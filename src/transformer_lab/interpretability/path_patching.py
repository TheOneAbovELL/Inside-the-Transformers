"""Path patching between two inputs, along explicitly named edges.

Definitions (this is the precise meaning of every number this module produces):

* **Sender** — a component whose output is written into the residual stream: an attention head
  ``(layer, head)`` (its write ``z_h W_O^h``) or an MLP block ``(layer, "mlp")`` (``hook_mlp_out``).
* **Receiver** — one specific *reader* of the residual stream:
    - ``("head", l, h, "q"|"k"|"v")``: the query / key / value input of head ``h`` in layer ``l``
      (requires per-head split inputs; the edit is applied before that head's LayerNorm);
    - ``("mlp", l)``: the input of MLP ``l`` (``hook_mlp_in``, before ``ln2``);
    - ``("logits",)``: the final residual stream read by the unembedding (the *direct* path).
* **Source / destination runs.** The model is run on the *destination* input. Along each listed
  edge sender → receiver, the receiver reads ``r_dst - out_dst(sender) + out_src(sender)``, where
  ``out_src`` is the sender's output on the *source* input. Nothing else changes: the sender's
  contribution along all other edges, every other component, and the residual stream itself keep
  their destination-run values. Effects that flow sender → X → receiver are **not** included
  unless X → receiver is itself listed (and X's own input is patched).
* ``direction="denoise"``: destination = corrupted, source = clean (does this path carry enough
  information to restore the behaviour?). ``"noise"``: destination = clean, source = corrupted (does
  corrupting this path break it?). Recovery ``R = (M_patched − M_corrupt) / (M_clean − M_corrupt)``
  on batch means, exactly as in :mod:`.patching` (uninformative if the gap < ``min_gap``).

Validated by an exact identity (patching a sender along ALL of its outgoing edges equals ordinary
activation patching of that sender) and on hand-wired models with known routes
(``tests/scientific/test_path_patching.py``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from ..data.tasks import TaskBatch
from ..evaluation.metrics import run_batched
from .patching import Patcher, _ratio_bootstrap


def _sender_outputs(model, tokens: torch.Tensor, senders) -> dict:
    names = sorted({f"blocks.{l}.attn.hook_z" for l, h in senders if h != "mlp"} |
                   {f"blocks.{l}.hook_mlp_out" for l, h in senders if h == "mlp"})
    _, cache = model.run_with_cache(tokens, names_filter=names)
    out = {}
    for l, h in senders:
        out[(l, h)] = cache["mlp_out", l] if h == "mlp" else cache.head_results(l)[:, :, h]
    return out


def _validate(model, senders, receivers) -> None:
    L = model.cfg.n_layers
    for r in receivers:
        if r[0] == "head":
            _, lr, hr, ch = r
            if ch not in ("q", "k", "v") or not (0 <= hr < model.cfg.n_heads):
                raise ValueError(f"bad head receiver {r}")
            if any(ls >= lr for ls, _ in senders):
                raise ValueError(f"sender must be in an earlier layer than receiver {r}")
        elif r[0] == "mlp":
            if model.cfg.attn_only:
                raise ValueError("attention-only model has no MLP receivers")
            lr = r[1]
            for ls, hs in senders:  # a head in the same layer precedes the MLP; an MLP does not
                if ls > lr or (ls == lr and hs == "mlp"):
                    raise ValueError(f"sender {(ls, hs)} does not precede receiver {r}")
        elif r[0] == "logits":
            if model.cfg.norm == "post":
                raise NotImplementedError("direct-path receiver needs pre-LN or no-LN models")
        else:
            raise ValueError(f"unknown receiver {r}")
        if r[0] in ("head", "mlp") and not (0 <= r[1] < L):
            raise ValueError(f"receiver layer out of range: {r}")


def path_patch_hooks(model, senders, receivers, src_outputs: dict) -> list:
    """Hooks that deliver ``src_outputs`` along sender -> receiver edges in the current run.
    Head receivers require ``split_qkv_input`` (use :func:`.path.split_inputs`)."""
    senders = [tuple(s) for s in senders]
    _validate(model, senders, receivers)
    store: dict = {}
    hooks = []
    for l, h in senders:
        if h == "mlp":
            def rec_mlp(out, hook, key=(l, h)):
                store[key] = src_outputs[key].to(out) - out
            hooks.append((f"blocks.{l}.hook_mlp_out", rec_mlp))
        else:
            W_O = model.blocks[l].attn.W_O

            def rec_head(z, hook, key=(l, h), W_O=W_O):
                store[key] = src_outputs[key].to(z) - torch.einsum("bse,ed->bsd", z[:, :, key[1]], W_O[key[1]])
            hooks.append((f"blocks.{l}.attn.hook_z", rec_head))

    def delta():
        return sum(store.values())

    by_head: dict = {}
    for r in receivers:
        if r[0] == "head":
            by_head.setdefault((r[1], r[3]), []).append(r[2])
        elif r[0] == "mlp":
            hooks.append((f"blocks.{r[1]}.hook_mlp_in", lambda x, hook: x + delta()))
        else:  # final residual: read only by ln_final -> unembedding
            hooks.append((f"blocks.{model.cfg.n_layers - 1}.hook_resid_post", lambda x, hook: x + delta()))
    for (l, ch), hs in by_head.items():
        def edit(x, hook, hs=hs):
            x = x.clone()
            for h in hs:
                x[:, :, h] = x[:, :, h] + delta()
            return x
        hooks.append((f"blocks.{l}.hook_{ch}_input", edit))
    return hooks


@torch.no_grad()
def path_patch(model, clean: TaskBatch, corrupt: TaskBatch, senders: Sequence, receivers: Sequence,
               metric: str = "logprob", direction: str = "denoise", min_gap: float = 0.25) -> dict:
    """Patch senders' outputs from the source run into the destination run along the listed edges."""
    from .path import split_inputs

    patcher = Patcher(model, clean, corrupt, metric=metric, direction=direction, min_gap=min_gap, names=[])
    src, dst = patcher.src_batch, patcher.dst_batch
    senders = [tuple(s) for s in senders]
    src_out = _sender_outputs(model, src.tokens, senders)
    needs_split = any(r[0] == "head" for r in receivers)

    def hooks_for(sl: slice):
        return path_patch_hooks(model, senders, receivers, {k: v[sl] for k, v in src_out.items()})

    if needs_split:
        with split_inputs(model):
            patched = run_batched(model, dst, fwd_hooks_fn=hooks_for, alt_targets=patcher.alt)[metric]
    else:
        patched = run_batched(model, dst, fwd_hooks_fn=hooks_for, alt_targets=patcher.alt)[metric]
    clean_m, cor_m = float(patcher.m_clean.mean()), float(patcher.m_corrupt.mean())
    if patcher.informative:
        rec = float((patched.mean() - cor_m) / (clean_m - cor_m))
        lo, hi = _ratio_bootstrap(patched, patcher.m_clean, patcher.m_corrupt)
    else:
        rec, lo, hi = float("nan"), float("nan"), float("nan")
    label = lambda c: f"L{c[0]}.mlp" if c[1] == "mlp" else f"L{c[0]}H{c[1]}"
    rlabel = lambda r: {"head": lambda: f"L{r[1]}H{r[2]}.{r[3]}", "mlp": lambda: f"L{r[1]}.mlp_in",
                        "logits": lambda: "logits(direct)"}[r[0]]()
    return {"senders": [label(s) for s in senders], "receivers": [rlabel(r) for r in receivers],
            "direction": direction, "metric": metric, "clean": clean_m, "corrupt": cor_m,
            "patched": float(patched.mean()), "recovery": rec, "recovery_lo": lo, "recovery_hi": hi,
            "gap": patcher.gap, "informative": patcher.informative, "n": len(patched),
            "patched_per_example": patched}


def all_outgoing_edges(model, sender) -> list:
    """Every receiver that reads ``sender``'s output (used for the completeness identity)."""
    l, h = sender
    out = []
    for lr in range(model.cfg.n_layers):
        if lr > l:
            out += [("head", lr, hr, ch) for hr in range(model.cfg.n_heads) for ch in "qkv"]
        if not model.cfg.attn_only and (lr > l or (lr == l and h != "mlp")):
            out.append(("mlp", lr))
    out.append(("logits",))
    return out
