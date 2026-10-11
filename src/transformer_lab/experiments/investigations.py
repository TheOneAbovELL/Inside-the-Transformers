"""Named, rerunnable investigations (``itl investigate <name> --run ... --run ...``).

Session-1/2 follow-up analyses lived in ``scripts/`` and wrote records that ``itl rerun`` could not
re-execute. These are the library versions: each writes an experiment record whose ``config.json``
contains everything needed to re-run it (``itl rerun <id>`` diffs the result against the original).
The original scripts are kept unchanged because some are referenced by frozen preregistrations.

Every investigation here is EXPLORATORY unless a preregistration says otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.tasks import InductionRepeat
from ..interpretability.context import (competition_attention, context_dose_response, context_route,
                                        context_route_per_sender)
from ..interpretability.stats import bootstrap_ci
from ..models.handwired import build_induction_model
from ..reporting.report import build_report, table_md
from ..training import load_run
from .registry import Experiment


def _models(runs, ground_truth: bool):
    out = []
    if ground_truth:
        out.append(("handwired_context_gain2", build_induction_model(48, 40, context_gain=2.0)))
        out.append(("handwired_single_token", build_induction_model(48, 40)))
    for r in runs:
        out.append((Path(r).name, load_run(r)[0]))
    return out


def _context_heads(models, n, seed):
    rows = []
    for name, m in models:
        r = context_route_per_sender(m, n=n, seed=seed)
        for row in r["rows"]:
            rows.append({"model": name, "group": " ".join(r["group"]), **row})
    return rows


def _competition_attention(models, n, seed):
    rows = []
    for name, m in models:
        r = competition_attention(m, n=n, seed=seed)
        for row in r["rows"]:
            rows.append({"model": name, "head": row["head"], "attn_A": row["attn_A"]["mean"], "attn_B": row["attn_B"]["mean"],
                         "A_minus_B": row["attn_A_minus_B"]["mean"], "A_minus_B_lo": row["attn_A_minus_B"]["lo"],
                         "A_minus_B_hi": row["attn_A_minus_B"]["hi"],
                         "A_minus_B_match_recent": row["A_minus_B_match_recent"],
                         "A_minus_B_match_earlier": row["A_minus_B_match_earlier"]})
    return rows


def _context_dose(models, n, seed):
    rows = []
    for name, m in models:
        for d in context_dose_response(m, n=n, seed=seed):
            rows.append({"model": name, "context_len": d["context_len"], "CP": d["cp"]["mean"], "CP_lo": d["cp"]["lo"],
                         "CP_hi": d["cp"]["hi"], "either_rate": d["either_rate"], "selection_rate": d["selection_rate"]})
    return rows


def _context_route(models, n, seed):
    rows = []
    for name, m in models:
        r = context_route(m, n=n, seed=seed)
        rows.append({"model": name, "group": " ".join(r["group"]),
                     **{f"R_{e}": r[e]["recovery"] for e in ("q", "k", "v", "all")},
                     **{f"R_{e}_lo": r[e]["recovery_lo"] for e in ("q", "k", "v")},
                     **{f"R_{e}_hi": r[e]["recovery_hi"] for e in ("q", "k", "v")},
                     "clean": r["q"]["clean"], "corrupt": r["q"]["corrupt"]})
    return rows


@torch.no_grad()
def _mechanism_type(models, n, seed):
    """Behavioural classification on shuffled first copies (see scripts/mechanism_type_test.py)."""
    rows = []
    for name, m in models:
        task = InductionRepeat(n_symbols=m.cfg.vocab_size - 2, seq_len=min(40, m.cfg.n_ctx))
        rng = np.random.default_rng(seed)
        clean = task.sample(n, rng)
        cor = task.corrupt(clean, rng, "shuffle")
        pred = m(cor.tokens)[:, :-1].argmax(-1)
        mask = cor.loss_mask & (cor.alt_targets >= 0)
        content, positional, clean_ans = [], [], []
        for i in range(n):
            L = int(clean.meta["L"][i])
            ts = torch.nonzero(mask[i]).flatten()
            p = pred[i, ts]
            pos_ans = cor.tokens[i, ts + 1 - L]
            content.append((p == cor.alt_targets[i, ts]).float().mean().item())
            positional.append((p == pos_ans).float().mean().item())
            clean_ans.append((p == clean.targets[i, ts]).float().mean().item())
        rows.append({"model": name, "content_induction": float(np.mean(content)),
                     "positional_copy": float(np.mean(positional)), "clean_answer": float(np.mean(clean_ans))})
    return rows


INVESTIGATIONS = {
    "context-heads": (_context_heads, 512, 909,
                      "Which single layer-0 sender carries the query's context (path patching, swap_context)?"),
    "competition-attention": (_competition_attention, 1024, 606,
                              "Group-head attention to the context-matching (A) vs non-matching (B) successor."),
    "context-dose": (_context_dose, 1024, 606, "Context preference vs number of matching preceding tokens (1-3)."),
    "context-route": (_context_route, 512, 909, "Layer 0 -> group q / k / v path patching on swapped-context inputs."),
    "mechanism-type": (_mechanism_type, 512, 505, "Content induction vs positional copying on shuffled first copies."),
}


def run_investigation(name: str, runs, n=None, seed=None, ground_truth: bool = True) -> Path:
    if name not in INVESTIGATIONS:
        raise ValueError(f"unknown investigation {name!r}; available: {sorted(INVESTIGATIONS)}")
    fn, n_default, seed_default, question = INVESTIGATIONS[name]
    n = n_default if n is None else int(n)
    seed = seed_default if seed is None else int(seed)
    runs = [str(r) for r in runs]
    args = {"name": name, "runs": runs, "n": n, "seed": seed, "ground_truth": ground_truth}
    exp = Experiment("investigation", name.replace("-", "_"), {"run": runs[0] if runs else None, "args": args,
                                                                "status": "exploratory"})
    rows = fn(_models(runs, ground_truth), n, seed)
    exp.save_table(name.replace("-", "_"), rows)
    exp.save_results({"name": name, "question": question, "rows": rows, "status": "exploratory",
                      "headline": {"name": name, "n_models": len(runs) + (2 if ground_truth else 0)}})
    exp.write_report(build_report(f"Investigation: {name} (exploratory)",
                                  {"question": question, "runs": ", ".join(Path(r).name for r in runs) or "(ground truth only)",
                                   "examples": f"{n} (seed {seed})",
                                   "ground-truth models included": ground_truth},
                                  table_md(rows),
                                  interpretation=["Exploratory: read against the hand-wired ground-truth rows first."],
                                  limitations=["Not preregistered unless a preregistration file references this record."]))
    return exp.finalize(f"{name}: {len(rows)} rows")
