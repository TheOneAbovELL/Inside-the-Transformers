"""Measurements for the context-matching investigation (preregistration
``configs/preregistration/context_matching_v1.yaml``).

* :func:`context_preference` — behavioural: on :class:`ContextCompetition` inputs, the logit
  difference between the successor of the context-matching earlier occurrence (A) and of the
  non-matching one (B). Single-token induction gives 0 by symmetry (validated on the hand-wired
  model: exactly 0.000); context-matching retrieval gives > 0 (hand-wired context model: +1.99 at
  context gain 2).
* :func:`context_sensitivity` — the earlier (ambiguous) measurement: change only the token two
  positions before the induction target and measure the change in log-prob. Context matching and
  non-specific interference in imperfect keys both predict a drop; only CP separates them.
* :func:`cp_under_path` — CP when layer-0 output is removed from ONLY the induction group's
  query (or key / value) inputs: locates where context enters the circuit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..data.tasks import ContextCompetition, InductionRepeat, TaskBatch
from ..evaluation.metrics import per_example_metrics, run_batched
from .attention_scores import head_scores
from .path import mean_head_results, path_hooks, split_inputs
from .stats import bootstrap_ci


def _logit_diff_and_choice(logits: torch.Tensor, batch: TaskBatch) -> dict:
    pos = batch.loss_mask.float().argmax(1)  # one scored position per example
    idx = torch.arange(len(batch))
    lg = logits[idx, pos].float()
    A = batch.targets[idx, pos]
    B = batch.alt_targets[idx, pos]
    ld = (lg[idx, A] - lg[idx, B]).numpy()
    arg = lg.argmax(-1)
    return {"logit_diff": ld, "chose_A": (arg == A).numpy(), "chose_B": (arg == B).numpy()}


@torch.no_grad()
def context_preference(model, n: int = 1024, seed: int = 606, context_len: int = 1, fwd_hooks=(),
                       split: bool = False) -> dict:
    """Context preference on a balanced ContextCompetition set (half the examples have the
    matching occurrence more recent). Returns CP (mean logit[A]-logit[B]) with a 95% bootstrap CI,
    CP separately for each recency configuration, the selection rate P(A | argmax in {A, B}) and
    the 'either' rate P(argmax in {A, B}) — the validity check that the model treats the input as
    an induction problem at all."""
    task = ContextCompetition(n_symbols=model.cfg.vocab_size - 2, seq_len=min(40, model.cfg.n_ctx),
                              context_len=context_len)
    batch = task.sample(n, np.random.default_rng(seed))
    model.eval()
    if split:
        with split_inputs(model):
            logits = model.run_with_hooks(batch.tokens, fwd_hooks=list(fwd_hooks))
    else:
        logits = model.run_with_hooks(batch.tokens, fwd_hooks=list(fwd_hooks))
    r = _logit_diff_and_choice(logits[:, :-1], batch)
    rec = batch.meta["match_recent"] == 1
    either = r["chose_A"] | r["chose_B"]
    return {
        "cp": bootstrap_ci(r["logit_diff"]),
        "cp_match_recent": bootstrap_ci(r["logit_diff"][rec]),
        "cp_match_earlier": bootstrap_ci(r["logit_diff"][~rec]),
        "either_rate": float(either.mean()),
        "selection_rate": float(r["chose_A"][either].mean()) if either.any() else float("nan"),
        "n": n, "seed": seed, "context_len": context_len,
        "per_example_logit_diff": r["logit_diff"],
    }


@torch.no_grad()
def context_sensitivity(model, n: int = 512, seed: int = 404) -> dict:
    """Δ log-prob at the query whose induction target's 2-back token was replaced ("affected"),
    and at a query 2 positions later ("control"), on standard contiguous repeats (segments >= 8)."""
    task = InductionRepeat(n_symbols=model.cfg.vocab_size - 2, seq_len=min(40, model.cfg.n_ctx), min_len=8)
    rng = np.random.default_rng(seed)
    clean = task.sample(n, rng)
    cor = clean.tokens.clone()
    aff, ctl = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    S = task.params["n_symbols"]
    for i in range(n):
        L, a, b = (int(clean.meta[k][i]) for k in ("L", "first_start", "second_start"))
        j = int(rng.integers(1, L - 4))
        present = set(cor[i].tolist())
        cor[i, a + j] = int(rng.choice([s + 2 for s in range(S) if s + 2 not in present]))
        aff[i], ctl[i] = b + j + 1, b + j + 3

    def lp(tokens, pos):
        logp = torch.log_softmax(model(tokens)[:, :-1].float(), -1)
        idx = torch.arange(n)
        return logp[idx, torch.from_numpy(pos), tokens[idx, torch.from_numpy(pos) + 1]].numpy()
    model.eval()
    return {"affected": bootstrap_ci(lp(cor, aff) - lp(clean.tokens, aff)),
            "control": bootstrap_ci(lp(cor, ctl) - lp(clean.tokens, ctl)), "n": n, "seed": seed}


@torch.no_grad()
def induction_group(model, n: int = 512, seed: int = 101, min_score: float = 0.25) -> list:
    """Last-layer heads with induction score >= min_score on standard contiguous repeats."""
    task = InductionRepeat(n_symbols=model.cfg.vocab_size - 2, seq_len=min(40, model.cfg.n_ctx))
    sc = head_scores(model, task.sample(n, np.random.default_rng(seed)))["induction"]
    L = model.cfg.n_layers - 1
    return [(L, h) for h in range(model.cfg.n_heads) if np.isfinite(sc[L, h]) and sc[L, h] >= min_score]


@torch.no_grad()
def cp_under_path(model, receivers, channel: str, n: int = 1024, seed: int = 606, ref_seed: int = 303,
                  senders: Optional[list] = None) -> dict:
    """CP when the senders' (default: all layer-0 heads, plus MLP0 if present) contribution is
    replaced by its per-position mean (from standard repeats, seed ``ref_seed``) in ONLY the
    receivers' ``channel`` input."""
    if senders is None:
        senders = [(0, h) for h in range(model.cfg.n_heads)]
        if not model.cfg.attn_only:
            senders.append((0, "mlp"))
    ref_task = InductionRepeat(n_symbols=model.cfg.vocab_size - 2, seq_len=min(40, model.cfg.n_ctx))
    rep = mean_head_results(model, ref_task.sample(512, np.random.default_rng(ref_seed)), senders)
    hooks = path_hooks(model, senders, receivers, channel, rep)
    out = context_preference(model, n=n, seed=seed, fwd_hooks=hooks, split=True)
    out["senders"] = [f"L{l}H{h}" if h != "mlp" else f"L{l}.mlp" for l, h in senders]
    out["receivers"] = [f"L{l}H{h}" for l, h in receivers]
    out["channel"] = channel
    return out


def bootstrap_difference(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """CI of mean(a) - mean(b) for two INDEPENDENT samples (different models' per-example values)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, a.size, (n_boot, a.size))].mean(1)
    db = b[rng.integers(0, b.size, (n_boot, b.size))].mean(1)
    d = da - db
    return {"mean": float(a.mean() - b.mean()), "lo": float(np.quantile(d, 0.025)), "hi": float(np.quantile(d, 0.975))}


@torch.no_grad()
def context_route(model, n: int = 512, seed: int = 909, min_gap: float = 0.5) -> dict:
    """Which edges carry the query's context? Denoising path patching on ContextCompetition with the
    ``swap_context`` corruption: deliver clean layer-0 outputs (heads, plus MLP0 if present) into the
    corrupted run along the induction group's q, k or v inputs only, and along all edges (identity
    check: equals activation patching of layer 0). Returns ``{edge_set: path_patch result}``."""
    from .path_patching import path_patch

    task = ContextCompetition(n_symbols=model.cfg.vocab_size - 2, seq_len=min(40, model.cfg.n_ctx))
    clean = task.sample(n, np.random.default_rng(seed))
    corrupt = task.corrupt(clean, np.random.default_rng(0))
    group = induction_group(model)
    senders = [(0, h) for h in range(model.cfg.n_heads)] + ([] if model.cfg.attn_only else [(0, "mlp")])
    out = {"group": [f"L{l}H{h}" for l, h in group]}
    for ch in "qkv":
        r = path_patch(model, clean, corrupt, senders, [("head", l, h, ch) for l, h in group],
                       metric="logit_diff", min_gap=min_gap)
        r.pop("patched_per_example")
        out[ch] = r
    out["all"] = all_layer0_edges_patch(model, clean, corrupt, senders, min_gap)
    return out


def all_layer0_edges_patch(model, clean, corrupt, senders, min_gap: float = 0.5) -> dict:
    """Patch every outgoing edge of a set of layer-0 senders at once (== activation patching of them).

    If MLP0 is among the senders, edges INTO MLP0 are dropped: MLP0's own output is replaced by its
    source-run value along all its outgoing edges, so what flows into it no longer matters, and an
    MLP cannot be both patched sender and receiver in one pass (tested identity:
    tests/scientific/test_path_patching.py::test_layer0_union_with_mlp_equals_activation_patching)."""
    from .path_patching import all_outgoing_edges, path_patch
    patched_mlps = {("mlp", l) for l, h in senders if h == "mlp"}
    edges = sorted({e for s in senders for e in all_outgoing_edges(model, s)} - patched_mlps)
    r = path_patch(model, clean, corrupt, senders, edges, metric="logit_diff", min_gap=min_gap)
    r.pop("patched_per_example", None)
    return r


@torch.no_grad()
def context_route_per_sender(model, n: int = 512, seed: int = 909, min_gap: float = 0.5,
                             channels: tuple = ("q", "k")) -> dict:
    """Which INDIVIDUAL layer-0 sender carries the query's context? Same design as
    :func:`context_route` (swap_context corruption, denoising path patching into the induction
    group), but one sender at a time: each layer-0 head, and MLP0 if present. Returns
    ``{"group": [...], "rows": [{"sender", "channel", recovery, CI, ...}]}``."""
    from .path_patching import path_patch

    task = ContextCompetition(n_symbols=model.cfg.vocab_size - 2, seq_len=min(40, model.cfg.n_ctx))
    clean = task.sample(n, np.random.default_rng(seed))
    corrupt = task.corrupt(clean, np.random.default_rng(0))
    group = induction_group(model)
    senders = [(0, h) for h in range(model.cfg.n_heads)] + ([] if model.cfg.attn_only else [(0, "mlp")])
    rows = []
    for s in senders:
        for ch in channels:
            r = path_patch(model, clean, corrupt, [s], [("head", l, h, ch) for l, h in group],
                           metric="logit_diff", min_gap=min_gap)
            rows.append({"sender": f"L0.mlp" if s[1] == "mlp" else f"L0H{s[1]}", "channel": ch,
                         "recovery": r["recovery"], "recovery_lo": r["recovery_lo"],
                         "recovery_hi": r["recovery_hi"], "clean": r["clean"], "corrupt": r["corrupt"],
                         "informative": r["informative"]})
    return {"group": [f"L{l}H{h}" for l, h in group], "rows": rows}


@torch.no_grad()
def competition_attention(model, n: int = 1024, seed: int = 606, context_len: int = 1) -> dict:
    """Where does each induction-group head look on context-competition inputs? Mean attention from
    the query position to the A-successor position (after the context-matching occurrence) and to
    the B-successor position, per head, overall and per recency configuration. Locates a context
    preference in the attention pattern (QK) rather than in what is copied (OV)."""
    task = ContextCompetition(n_symbols=model.cfg.vocab_size - 2, seq_len=min(40, model.cfg.n_ctx),
                              context_len=context_len)
    batch = task.sample(n, np.random.default_rng(seed))
    group = induction_group(model)
    layers = sorted({l for l, _ in group})
    _, cache = model.run_with_cache(batch.tokens, names_filter=[f"blocks.{l}.attn.hook_pattern" for l in layers])
    T, k = task.seq_len, context_len
    q = T - 2
    pa = torch.from_numpy(batch.meta["pos_match"] + k + 1)   # position of A
    pb = torch.from_numpy(batch.meta["pos_nonmatch"] + k + 1)  # position of B
    idx = torch.arange(n)
    rec = batch.meta["match_recent"] == 1
    rows = []
    for l, h in group:
        p = cache["pattern", l][:, h, q]  # [b, k]
        a, b = p[idx, pa].numpy(), p[idx, pb].numpy()
        rows.append({"head": f"L{l}H{h}", "attn_A": bootstrap_ci(a), "attn_B": bootstrap_ci(b),
                     "attn_A_minus_B": bootstrap_ci(a - b),
                     "A_minus_B_match_recent": float((a - b)[rec].mean()),
                     "A_minus_B_match_earlier": float((a - b)[~rec].mean())})
    return {"group": [f"L{l}H{h}" for l, h in group], "rows": rows, "n": n, "seed": seed}


@torch.no_grad()
def context_dose_response(model, ks=(1, 2, 3), n: int = 1024, seed: int = 606) -> list:
    """Context preference as a function of the number of matching preceding tokens."""
    out = []
    for k in ks:
        r = context_preference(model, n=n, seed=seed + k, context_len=k)
        out.append({"context_len": k, "cp": r["cp"], "either_rate": r["either_rate"],
                    "selection_rate": r["selection_rate"]})
    return out
