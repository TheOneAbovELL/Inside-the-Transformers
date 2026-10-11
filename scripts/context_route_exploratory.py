"""EXPLORATORY follow-up (post hoc, NOT preregistered) to context_matching_v1's inconclusive H3.

H3 used mean-removal of layer-0 output from the induction group's queries; in 3 of 6 models that
also destroyed induction (either_rate < 0.5), so the frozen rule returned 'inconclusive'. Here the
same question is asked with path patching between two VALID inputs:

* clean   = context-competition input (query context y, prefers A);
* corrupt = identical except the query's preceding token is swapped to z (now matches B's context).

Only positions >= T-3 differ, so earlier keys/values are identical in both runs. Denoising path
patching delivers clean-run outputs of the senders along ONE edge set into the corrupted run:

* layer-0 heads -> group QUERY inputs      (does context reach the heads through the query?)
* layer-0 heads -> group KEY inputs         (keys at the changed positions)
* layer-0 heads -> group VALUE inputs
* layer-0 heads -> ALL edges                (= activation patching of layer 0)

Metric: logit difference A - B; recovery R as in path_patch. Hand-wired context model = ground truth
(context enters via its query by construction). Writes an 'investigation' experiment record.
Usage: python scripts/context_route_exploratory.py <run_dir> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from transformer_lab.data import ContextCompetition
from transformer_lab.experiments.registry import Experiment
from transformer_lab.interpretability.context import induction_group
from transformer_lab.interpretability.path_patching import all_outgoing_edges, path_patch
from transformer_lab.models.handwired import build_induction_model
from transformer_lab.reporting.report import build_report, table_md
from transformer_lab.training import load_run

N, SEED = 512, 808


def analyse(name, model):
    task = ContextCompetition(n_symbols=model.cfg.vocab_size - 2, seq_len=40)
    clean = task.sample(N, np.random.default_rng(SEED))
    corrupt = task.corrupt(clean, np.random.default_rng(0))
    group = induction_group(model)
    senders = [(0, h) for h in range(model.cfg.n_heads)] + ([] if model.cfg.attn_only else [(0, "mlp")])
    edges = {
        "L0->group.q": [("head", l, h, "q") for l, h in group],
        "L0->group.k": [("head", l, h, "k") for l, h in group],
        "L0->group.v": [("head", l, h, "v") for l, h in group],
    }
    rows = []
    for label, receivers in edges.items():
        r = path_patch(model, clean, corrupt, senders, receivers, metric="logit_diff", min_gap=0.5)
        rows.append({"model": name, "edges": label, "clean": r["clean"], "corrupt": r["corrupt"],
                     "patched": r["patched"], "R": r["recovery"], "R_lo": r["recovery_lo"], "R_hi": r["recovery_hi"],
                     "informative": r["informative"]})
    # all outgoing edges of every layer-0 sender == activation patching of layer 0
    all_edges = sorted({e for s in senders for e in all_outgoing_edges(model, s)})
    r = path_patch(model, clean, corrupt, senders, all_edges, metric="logit_diff", min_gap=0.5)
    rows.append({"model": name, "edges": "L0->ALL", "clean": r["clean"], "corrupt": r["corrupt"], "patched": r["patched"],
                 "R": r["recovery"], "R_lo": r["recovery_lo"], "R_hi": r["recovery_hi"], "informative": r["informative"]})
    for row in rows:
        print(f"{row['model'][:40]:40s} {row['edges']:12s} clean {row['clean']:+.2f} corrupt {row['corrupt']:+.2f} "
              f"patched {row['patched']:+.2f} R {row['R']:.2f} [{row['R_lo']:.2f}, {row['R_hi']:.2f}]", flush=True)
    return rows


def main(runs):
    exp = Experiment("investigation", "context_route_exploratory",
                     {"run": runs[0] if runs else None, "runs": runs,
                      "args": {"runs": runs, "n": N, "seed": SEED}, "status": "EXPLORATORY (post hoc)"})
    rows = analyse("handwired_context_gain2 (ground truth: query route)", build_induction_model(48, 40, context_gain=2.0))
    for run in runs:
        model, _, _, _ = load_run(run)
        rows += analyse(Path(run).name, model)
    exp.save_table("context_route", rows)
    exp.save_results({"rows": rows, "status": "exploratory", "headline": {"n_models": len(runs)}})
    exp.write_report(build_report(
        "EXPLORATORY: which path carries the query's context? (post hoc follow-up to H3)",
        {"status": "exploratory, not preregistered", "runs": ", ".join(Path(r).name for r in runs),
         "clean/corrupt": "context competition; corrupt = query's preceding token swapped to the non-matching context",
         "metric": "logit[A] - logit[B]; R = (patched - corrupt)/(clean - corrupt)"},
        table_md(rows),
        interpretation=["Read R per edge set: R near 1 for L0->group.q with R near 0 for keys/values would locate the "
                        "context signal on the query side; this analysis was designed after seeing H3."],
        limitations=["Post hoc: designed after the preregistered H3 was inconclusive; treat as hypothesis-generating.",
                     "Layer-0 heads are patched as a group; individual heads are not separated."]))
    print(exp.finalize("exploratory context route"))


if __name__ == "__main__":
    main(sys.argv[1:])
