"""Test: do the trained induction circuits match ONLY the current token, or the preceding context too?

Construction (per example, induction_repeat with L >= 8): pick j in [1, L-5] and replace the
first-copy token r_j (position a+j) by a fresh symbol absent from the sequence. Then

* AFFECTED query  b+j+1 (current token r_{j+1}): its earlier occurrence a+j+1 is intact and still
  followed by the target r_{j+2}; only the token BEFORE that occurrence changed.
* CONTROL query   b+j+3 (current token r_{j+3}): its earlier occurrence is intact and the changed
  token is 3 positions further back.

Predictions:
* pure single-token induction (e.g. the hand-wired model): no change at the affected query;
* context (n-gram) matching: log-prob drops at the affected query, much less at the control.

Usage: python scripts/context_sensitivity.py <run_dir> [...]   (writes an experiment directory)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from transformer_lab.data import InductionRepeat
from transformer_lab.experiments.registry import Experiment
from transformer_lab.interpretability.stats import bootstrap_ci, paired_difference
from transformer_lab.models.handwired import build_induction_model
from transformer_lab.reporting.report import build_report, table_md
from transformer_lab.training import load_run

N, SEED = 512, 404


def make_pairs(task: InductionRepeat, n: int, seed: int):
    rng = np.random.default_rng(seed)
    clean = task.with_params(min_len=max(8, task.params["min_len"])).sample(n, rng)
    tok = clean.tokens.clone()
    aff, ctl = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    S = task.params["n_symbols"]
    for i in range(n):
        L, a, b = (int(clean.meta[k][i]) for k in ("L", "first_start", "second_start"))
        j = int(rng.integers(1, L - 4))
        present = set(tok[i].tolist())
        fresh = [s + 2 for s in range(S) if s + 2 not in present]
        tok[i, a + j] = int(rng.choice(fresh))
        aff[i], ctl[i] = b + j + 1, b + j + 3
    return clean, tok, aff, ctl


@torch.no_grad()
def logprob_at(model, tokens, pos):
    logp = torch.log_softmax(model(tokens)[:, :-1].float(), -1)
    idx = torch.arange(len(tokens))
    tgt = tokens[idx, torch.from_numpy(pos) + 1]
    return logp[idx, torch.from_numpy(pos), tgt].numpy()


def main(runs):
    exp = Experiment("investigation", "context_sensitivity", {"run": runs[0] if runs else None, "runs": runs,
                                                              "args": {"runs": runs, "n": N, "seed": SEED}})
    rows = []
    models = []
    for run in runs:
        model, task, _, _ = load_run(run)
        models.append((Path(run).name, model, task))
    # reference: hand-wired pure single-token induction on the same task family
    ref_task = models[0][2] if models else InductionRepeat()
    models.append(("handwired_single_token_induction",
                   build_induction_model(ref_task.params["n_symbols"], ref_task.seq_len), ref_task))
    for name, model, task in models:
        model.eval()
        clean, cor, aff, ctl = make_pairs(task, N, SEED)
        out = {"model": name}
        for label, pos in (("affected", aff), ("control", ctl)):
            c = logprob_at(model, clean.tokens, pos)
            k = logprob_at(model, cor, pos)
            d = paired_difference(k, c)
            out[f"{label}_clean_logprob"] = float(c.mean())
            out[f"{label}_d_logprob"], out[f"{label}_lo"], out[f"{label}_hi"] = d["mean"], d["lo"], d["hi"]
            out[f"{label}_acc_clean"] = float((np.exp(c) > 0.5).mean())
        rows.append(out)
        print(name, {k: round(v, 3) for k, v in out.items() if isinstance(v, float)}, flush=True)
    exp.save_table("context_sensitivity", rows)
    interp = []
    for r in rows:
        interp.append(f"{r['model']}: changing the token before the matched occurrence changes log-prob at the "
                      f"affected query by {r['affected_d_logprob']:.2f} [{r['affected_lo']:.2f}, {r['affected_hi']:.2f}] "
                      f"(control query: {r['control_d_logprob']:.2f}).")
    exp.save_results({"rows": rows, "headline": {r["model"]: r["affected_d_logprob"] for r in rows}})
    exp.write_report(build_report("Investigation: does induction match context beyond the current token?",
                                  {"runs": ", ".join(runs), "examples": f"{N} (seed {SEED}), segments L >= 8",
                                   "manipulation": "first-copy token r_j replaced by a fresh symbol"},
                                  table_md(rows), interpretation=interp,
                                  limitations=["The manipulation also removes r_j as a match for query b+j; that query "
                                               "is excluded from both measurements.",
                                               "Distances 1 (affected) vs 3 (control) are a coarse dose-response probe."]))
    print(exp.finalize("context sensitivity"))


if __name__ == "__main__":
    main(sys.argv[1:])
