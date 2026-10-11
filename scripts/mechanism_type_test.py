"""Behavioural mechanism classification on shuffled repeats.

For induction_repeat inputs whose FIRST copy is shuffled (successor relations deranged), different
strategies predict different tokens at each scored position t = b+i (current token r_i):

* content induction:   successor of r_i inside the shuffled first copy   (task's alt_targets)
* positional copying:  the token at the same offset, corrupted[t + 1 - L] (fixed-distance copy)
* clean answer:        r_{i+1} (not recoverable from the corrupted input by either strategy)

The fraction of argmax predictions matching each answer classifies the mechanism behaviourally.
(The two strategies coincide only when the shuffle happens to put r_{i+1}... never: the derangement
guarantees the content-induction answer differs from r_{i+1}; positional and content answers can
coincide by chance, which is reported.)

Usage: python scripts/mechanism_type_test.py <run_dir> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from transformer_lab.data import InductionRepeat
from transformer_lab.experiments.registry import Experiment
from transformer_lab.interpretability.stats import bootstrap_ci
from transformer_lab.models.handwired import build_induction_model
from transformer_lab.reporting.report import build_report, table_md
from transformer_lab.training import load_run

N, SEED = 512, 505


@torch.no_grad()
def classify(model, task: InductionRepeat, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    clean = task.sample(n, rng)
    cor = task.corrupt(clean, rng, "shuffle")
    pred = model(cor.tokens)[:, :-1].argmax(-1)
    m = cor.loss_mask & (cor.alt_targets >= 0)
    positional = torch.full_like(pred, -1)
    for i in range(n):
        L = int(clean.meta["L"][i])
        for t in torch.nonzero(m[i]).flatten().tolist():
            positional[i, t] = cor.tokens[i, t + 1 - L]
    per_ex = {"content_induction": [], "positional_copy": [], "clean_answer": [], "both_coincide": []}
    for i in range(n):
        mi = m[i]
        p = pred[i][mi]
        per_ex["content_induction"].append((p == cor.alt_targets[i][mi]).float().mean().item())
        per_ex["positional_copy"].append((p == positional[i][mi]).float().mean().item())
        per_ex["clean_answer"].append((p == clean.targets[i][mi]).float().mean().item())
        per_ex["both_coincide"].append((cor.alt_targets[i][mi] == positional[i][mi]).float().mean().item())
    return {k: bootstrap_ci(v) for k, v in per_ex.items()}


def main(runs, include_learned_variable_eval=True):
    exp = Experiment("investigation", "mechanism_type", {"run": runs[0] if runs else None, "runs": runs,
                                                         "args": {"runs": runs, "n": N, "seed": SEED}})
    rows = []
    entries = []
    for run in runs:
        model, task, _, cfg = load_run(run)
        entries.append((Path(run).name + f" [{cfg.model.positional}, own task]", model, task))
        if task.params.get("fixed_prefix_len") is not None or task.params["min_len"] == task.params["max_len"]:
            # also evaluate fixed-offset models on variable offsets (their OOD condition)
            var = task.with_params(min_len=6, max_len=16, fixed_prefix_len=None)
            entries.append((Path(run).name + f" [{cfg.model.positional}, variable offsets]", model, var))
    t0 = entries[0][2] if entries else InductionRepeat()
    entries.append(("handwired_single_token_induction", build_induction_model(t0.params["n_symbols"], t0.seq_len),
                    t0.with_params(min_len=6, max_len=16, fixed_prefix_len=None)))
    for name, model, task in entries:
        model.eval()
        r = classify(model, task, N, SEED)
        row = {"model": name}
        for k, v in r.items():
            row[k] = v["mean"]
            row[k + "_ci"] = f"[{v['lo']:.3f}, {v['hi']:.3f}]"
        rows.append(row)
        print(name, {k: round(v["mean"], 3) for k, v in r.items()}, flush=True)
    exp.save_table("mechanism_type", rows)
    interp = [f"{r['model']}: {r['content_induction']:.2f} of predictions follow the shuffled successor (content "
              f"induction), {r['positional_copy']:.2f} copy the fixed-offset token, {r['clean_answer']:.2f} give the "
              f"clean answer." for r in rows]
    exp.save_results({"rows": rows, "headline": {r["model"]: r["content_induction"] for r in rows}})
    exp.write_report(build_report("Behavioural mechanism classification (shuffled first copy)",
                                  {"runs": ", ".join(runs), "examples": f"{N} (seed {SEED})"}, table_md(rows),
                                  interpretation=interp,
                                  limitations=["Argmax agreement only; a model can mix strategies across positions.",
                                               "'both_coincide' is the fraction of positions where the two strategies "
                                               "predict the same token (agreement there is uninformative)."]))
    print(exp.finalize("mechanism type"))


if __name__ == "__main__":
    main(sys.argv[1:])
