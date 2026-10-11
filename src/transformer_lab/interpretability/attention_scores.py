"""Observational head diagnostics: attention-pattern scores and weight-based circuit scores.

These are *observational* evidence. A high induction score says where a head puts its attention
on these inputs; it does not say the head's output matters for the prediction (that needs an
intervention — see ``interventions.py`` / ``patching.py``). Every function documents what it
measures and the approximations it makes.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..data.tasks import TaskBatch


def query_mask_from_batch(batch: TaskBatch) -> torch.Tensor:
    """Scored prediction positions as a ``[b, T]`` query mask (position t predicts t+1)."""
    m = batch.loss_mask
    return torch.cat([m, torch.zeros(m.shape[0], 1, dtype=torch.bool)], dim=1)


def prev_token_score(pattern: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    """Mean attention from position q to q-1 over selected queries. ``pattern [b,h,q,k] -> [h]``."""
    b, h, s, _ = pattern.shape
    q = torch.arange(1, s)
    a = pattern[:, :, q, q - 1]  # [b,h,s-1]
    m = query_mask[:, 1:].float().unsqueeze(1)
    return (a * m).sum((0, 2)) / m.sum((0, 2)).clamp_min(1)


def _earlier_occurrence(tokens: torch.Tensor) -> torch.Tensor:
    """``prev[b, q]`` = index of the most recent earlier occurrence of tokens[b, q], or -1."""
    tok = tokens.cpu().numpy()
    prev = np.full(tok.shape, -1, dtype=np.int64)
    for i in range(tok.shape[0]):
        last: dict = {}
        for q, t in enumerate(tok[i]):
            prev[i, q] = last.get(t, -1)
            last[t] = q
    return torch.from_numpy(prev)


def induction_score_per_example(pattern: torch.Tensor, tokens: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    """Per-example induction score ``[b, h]`` (NaN for examples without qualifying queries)."""
    prev = _earlier_occurrence(tokens)
    valid = query_mask & (prev >= 0) & (prev + 1 < torch.arange(tokens.shape[1]))
    k_idx = (prev + 1).clamp(min=0, max=tokens.shape[1] - 1)  # [b, q]
    att = pattern.gather(3, k_idx[:, None, :, None].expand(-1, pattern.shape[1], -1, 1)).squeeze(-1)  # [b,h,q]
    m = valid.float().unsqueeze(1)
    cnt = m.sum(-1)
    out = (att * m).sum(-1) / cnt.clamp_min(1)
    out[cnt.expand_as(out) == 0] = float("nan")
    return out


def prev_token_score_per_example(pattern: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    s = pattern.shape[-1]
    q = torch.arange(1, s)
    a = pattern[:, :, q, q - 1]
    m = query_mask[:, 1:].float().unsqueeze(1)
    cnt = m.sum(-1)
    out = (a * m).sum(-1) / cnt.clamp_min(1)
    out[cnt.expand_as(out) == 0] = float("nan")
    return out


def induction_score(pattern: torch.Tensor, tokens: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    """Mean attention from query q to ``prev(q) + 1`` — the token *after* the previous occurrence of
    the current token — over selected queries that have such an occurrence. ``-> [h]``.

    An ideal induction head scores ~1. Note this is the *pattern* signature only; the copying
    half of the mechanism is measured separately (``ov_copying_score``) and causally (ablation).
    """
    prev = _earlier_occurrence(tokens)
    valid = query_mask & (prev >= 0) & (prev + 1 < torch.arange(tokens.shape[1]))
    b_idx, q_idx = torch.nonzero(valid, as_tuple=True)
    if b_idx.numel() == 0:
        return torch.full((pattern.shape[1],), float("nan"))
    k_idx = prev[b_idx, q_idx] + 1
    vals = pattern[b_idx, :, q_idx, k_idx]  # [n, h]
    return vals.mean(0)


def duplicate_token_score(pattern: torch.Tensor, tokens: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    """Mean attention mass on earlier positions holding the *same* token as the query. ``-> [h]``."""
    same = tokens.unsqueeze(2) == tokens.unsqueeze(1)  # [b,q,k]
    s = tokens.shape[1]
    same &= torch.tril(torch.ones(s, s, dtype=torch.bool), diagonal=-1)
    mass = (pattern * same.unsqueeze(1)).sum(-1)  # [b,h,q]
    valid = query_mask & same.any(-1)
    m = valid.float().unsqueeze(1)
    return (mass * m).sum((0, 2)) / m.sum((0, 2)).clamp_min(1)


def bos_score(pattern: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    """Mean attention to position 0 (often a 'resting' position for heads with nothing to do)."""
    m = query_mask.float().unsqueeze(1)
    return (pattern[..., 0] * m).sum((0, 2)) / m.sum((0, 2)).clamp_min(1)


@torch.no_grad()
def head_scores(model, batch: TaskBatch, query_mask: Optional[torch.Tensor] = None) -> dict:
    """All pattern scores for every head: dict of ``np.ndarray [n_layers, n_heads]``."""
    model.eval()
    qm = query_mask if query_mask is not None else query_mask_from_batch(batch)
    _, cache = model.run_with_cache(batch.tokens, names_filter=lambda n: n.endswith("hook_pattern"))
    out = {k: [] for k in ("prev_token", "induction", "duplicate_token", "bos")}
    for l in range(model.cfg.n_layers):
        p = cache["pattern", l]
        out["prev_token"].append(prev_token_score(p, qm))
        out["induction"].append(induction_score(p, batch.tokens, qm))
        out["duplicate_token"].append(duplicate_token_score(p, batch.tokens, qm))
        out["bos"].append(bos_score(p, qm))
    return {k: torch.stack(v).numpy() for k, v in out.items()}


# ----------------------------------------------------------------------- weight-based scores
def _qk(model, l, h):
    a = model.blocks[l].attn
    return (a.W_Q[h] @ a.W_K[h].T).detach()  # [d,d]  score = x W_QK y^T


def _ov(model, l, h):
    a = model.blocks[l].attn
    return (a.W_V[h] @ a.W_O[h]).detach()  # [d,d]  output = y W_OV


@torch.no_grad()
def ov_copying_score(model, tokens: Optional[torch.Tensor] = None) -> np.ndarray:
    """Fraction of tokens ``i`` for which the head's direct OV circuit ``W_E W_OV W_U`` maps ``i`` to
    a logit vector whose argmax is ``i`` itself (``[n_layers, n_heads]``).

    Approximation: ignores LayerNorm scales and everything except the direct embedding -> head ->
    unembedding path. For layer >0 heads the real value input also contains earlier heads' outputs.
    """
    toks = tokens if tokens is not None else torch.arange(2, model.cfg.vocab_size)
    E = model.W_E[toks]
    U = model.W_U[:, toks]
    out = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for l in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            M = E @ _ov(model, l, h) @ U  # [n, n]
            out[l, h] = (M.argmax(-1) == torch.arange(len(toks))).float().mean().item()
    return out


@torch.no_grad()
def composition_scores(model) -> dict:
    """Elhage et al. (2021) composition scores between every layer-l1 head and earlier layer-l0 head.

    ``||A B||_F / (||A||_F ||B||_F)`` with (row-vector convention, x -> x W):
    * K-composition: ``W_QK(receiver) W_OV(sender)^T``  (sender output feeds the receiver's keys)
    * Q-composition: ``W_OV(sender) W_QK(receiver)``    (sender output feeds the receiver's queries)
    * V-composition: ``W_OV(sender) W_OV(receiver)``    (sender output feeds the receiver's values)

    Returns dict ``kind -> np.ndarray [n_layers, n_heads, n_layers, n_heads]`` (receiver, sender);
    entries with sender layer >= receiver layer are NaN. Weight-only: ignores LayerNorm and the
    actual input distribution, so compare against the same scores at initialisation.
    """
    L, H = model.cfg.n_layers, model.cfg.n_heads
    out = {k: np.full((L, H, L, H), np.nan) for k in ("K", "Q", "V")}
    for l1 in range(L):
        for h1 in range(H):
            qk, ov1 = _qk(model, l1, h1), _ov(model, l1, h1)
            for l0 in range(l1):
                for h0 in range(H):
                    ov0 = _ov(model, l0, h0)
                    norm = lambda A, B: (torch.linalg.norm(A @ B) / (torch.linalg.norm(A) * torch.linalg.norm(B))).item()
                    out["K"][l1, h1, l0, h0] = norm(qk, ov0.T)
                    out["Q"][l1, h1, l0, h0] = norm(ov0, qk)
                    out["V"][l1, h1, l0, h0] = norm(ov0, ov1)
    return out
