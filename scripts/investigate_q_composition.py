"""Follow-up investigation: why does removing layer-0 output from the induction heads' QUERIES hurt?

Pre-registered hypotheses v1 (P6) and v2 (Q4) both predicted a small Q-path effect; both were
contradicted on all seeds. Candidate explanations, each with a test:

  A. Measurement artifact of mean replacement before LayerNorm  -> compare zero / mean / resample
     replacement; a genuine dependence should survive resample (realistic activations).
  B. Layer-0 heads attend substantially to the CURRENT token, so their output carries current-token
     identity that queries read                                  -> attention mass by relative offset.
  C. The Q-path effect acts through WHERE the heads look         -> induction score of the group
     under each path intervention (K / Q / V).

Usage: python scripts/investigate_q_composition.py <run_dir> [<run_dir> ...]
Writes an experiment directory under artifacts/experiments/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from transformer_lab.evaluation.metrics import per_example_metrics
from transformer_lab.experiments.registry import Experiment
from transformer_lab.interpretability.attention_scores import (head_scores, induction_score_per_example,
                                                               query_mask_from_batch)
from transformer_lab.interpretability.path import path_hooks, split_inputs
from transformer_lab.interpretability.stats import bootstrap_ci
from transformer_lab.reporting.report import build_report, table_md
from transformer_lab.training import load_run

N, SEED, REF_SEED = 256, 202, 303


@torch.no_grad()
def offsets_profile(model, batch, layer=0, max_off=4):
    """Mean attention mass at relative offset 0 (self), 1 (previous), ..., >=max_off, per head,
    over (a) scored queries and (b) first-copy positions (where keys are formed)."""
    _, cache = model.run_with_cache(batch.tokens, names_filter=[f"blocks.{layer}.attn.hook_pattern"])
    p = cache["pattern", layer]  # [b,h,q,k]
    b, h, s, _ = p.shape
    q = torch.arange(s)[:, None]
    k = torch.arange(s)[None, :]
    off = (q - k)  # [q,k]
    first = torch.zeros(b, s, dtype=torch.bool)
    for i in range(b):
        a, s2 = int(batch.meta["first_start"][i]), int(batch.meta["second_start"][i])
        first[i, a + 1:s2] = True
    regions = {"scored queries": query_mask_from_batch(batch), "first-copy positions": first}
    rows = []
    for rname, qm in regions.items():
        for hh in range(h):
            row = {"region": rname, "head": f"L{layer}H{hh}"}
            for o in range(max_off):
                m = (off == o).float()
                row[f"off{o}"] = float(((p[:, hh] * m).sum(-1) * qm).sum() / qm.sum())
            m = (off >= max_off).float()
            row[f"off>={max_off}"] = float(((p[:, hh] * m).sum(-1) * qm).sum() / qm.sum())
            rows.append(row)
    return rows


@torch.no_grad()
def group_heads(model, batch, thr=0.25):
    sc = head_scores(model, batch)["induction"]
    return [(1, h) for h in range(model.cfg.n_heads) if sc[1, h] >= thr]


@torch.no_grad()
def path_run(model, batch, senders, receivers, channel, replacement):
    """Per-example log-prob and group induction score under the path intervention."""
    with split_inputs(model):
        hooks = path_hooks(model, senders, receivers, channel, replacement)
        logits, cache = model.run_with_cache(batch.tokens, names_filter=["blocks.1.attn.hook_pattern"], fwd_hooks=hooks)
    lp = per_example_metrics(logits, batch)["logprob"]
    qm = query_mask_from_batch(batch)
    ind = induction_score_per_example(cache["pattern", 1], batch.tokens, qm)  # [b,h]
    return lp, ind[:, [h for _, h in receivers]].mean(1).numpy()


def main(runs):
    exp = Experiment("investigation", "q_composition", {"run": runs[0], "runs": runs,
                                                        "args": {"runs": runs, "n": N, "seed": SEED}})
    all_rows, offs_all = [], []
    for run in runs:
        model, task, _, cfg = load_run(run)
        batch = task.sample(N, np.random.default_rng(SEED))
        ref = task.sample(N, np.random.default_rng(REF_SEED))
        receivers = group_heads(model, task.sample(N, np.random.default_rng(101)))  # selection set
        senders = [(0, h) for h in range(model.cfg.n_heads)]
        offs = offsets_profile(model, batch)
        for r in offs:
            r["run"] = Path(run).name
        offs_all += offs
        # reference results for replacements
        _, rc = model.run_with_cache(ref.tokens, names_filter=["blocks.0.attn.hook_z"])
        ref_res = rc.head_results(0)  # [n, s, h, d]
        base_logits, bc = model.run_with_cache(batch.tokens, names_filter=["blocks.1.attn.hook_pattern"])
        base_lp = per_example_metrics(base_logits, batch)["logprob"]
        base_ind = induction_score_per_example(bc["pattern", 1], batch.tokens, query_mask_from_batch(batch))[
            :, [h for _, h in receivers]].mean(1).numpy()
        perm = torch.from_numpy(np.random.default_rng(7).permutation(N))
        reps = {
            "zero": None,
            "mean": {(0, h): ref_res[:, :, h].mean(0) for h in range(model.cfg.n_heads)},
            "resample": {(0, h): ref_res[perm][:, :, h] for h in range(model.cfg.n_heads)},
        }
        for rep_name, rep in reps.items():
            for ch in "kqv":
                if rep_name == "resample":
                    # path_hooks expands [s,d] replacements; resample needs per-example rows
                    hooks = _resample_hooks(model, senders, receivers, ch, rep)
                    with split_inputs(model):
                        logits, cache = model.run_with_cache(batch.tokens, names_filter=["blocks.1.attn.hook_pattern"],
                                                             fwd_hooks=hooks)
                    lp = per_example_metrics(logits, batch)["logprob"]
                    ind = induction_score_per_example(cache["pattern", 1], batch.tokens, query_mask_from_batch(batch))[
                        :, [h for _, h in receivers]].mean(1).numpy()
                else:
                    lp, ind = path_run(model, batch, senders, receivers, ch, rep)
                d = bootstrap_ci(lp - base_lp)
                di = bootstrap_ci(ind - base_ind)
                all_rows.append({"run": Path(run).name, "replacement": rep_name, "channel": ch,
                                 "receivers": "+".join(f"L{l}H{h}" for l, h in receivers),
                                 "d_logprob": d["mean"], "d_lo": d["lo"], "d_hi": d["hi"],
                                 "group_induction_score": float(np.nanmean(ind)), "d_induction_score": di["mean"],
                                 "d_ind_lo": di["lo"], "d_ind_hi": di["hi"]})
        print(f"done {Path(run).name}", flush=True)
    exp.save_table("path_effects", all_rows)
    exp.save_table("layer0_offsets", offs_all)
    obs = ("### Layer-0 attention mass by relative offset (0 = current token, 1 = previous)\n\n" + table_md(offs_all)
           + "\n### Layer-0 → induction-group path interventions, by replacement method\n\n" + table_md(all_rows))
    kq = {}
    for r in all_rows:
        kq.setdefault((r["run"], r["replacement"]), {})[r["channel"]] = r["d_logprob"]
    interp = []
    res_q = [v["q"] for (run, rep), v in kq.items() if rep == "resample"]
    res_k = [v["k"] for (run, rep), v in kq.items() if rep == "resample"]
    interp.append(f"Under RESAMPLE replacement (realistic activations) the Q-path effect is "
                  f"{np.round(res_q, 2).tolist()} nats vs K-path {np.round(res_k, 2).tolist()} across runs.")
    self_mass = [r["off0"] for r in offs_all if r["region"] == "scored queries"]
    interp.append(f"Layer-0 heads put {min(self_mass):.2f}–{max(self_mass):.2f} of their attention on the current "
                  "token at scored queries (offset 0).")
    exp.save_results({"path_effects": all_rows, "offsets": offs_all,
                      "headline": {"q_resample": res_q, "k_resample": res_k}})
    exp.write_report(build_report("Investigation: query-side composition", {"runs": ", ".join(Path(r).name for r in runs),
                                  "examples": f"{N} (seed {SEED}); reference seed {REF_SEED}; selection seed 101",
                                  "senders": "all layer-0 heads", "receivers": "layer-1 heads with induction score >= 0.25"},
                                  obs, interpretation=interp,
                                  limitations=["Resample replacement draws another example's layer-0 output at the "
                                               "same position; with variable-offset sequences positions are only "
                                               "approximately comparable."]))
    print(exp.finalize("q-composition investigation"))


def _resample_hooks(model, senders, receivers, channel, rep):
    """Like path_hooks, but replacement rows are per-example ([b,s,d])."""
    store = {}
    hooks = []
    W_O = model.blocks[0].attn.W_O

    def record(z, hook):
        for l, h in senders:
            res = torch.einsum("bse,ed->bsd", z[:, :, h], W_O[h])
            store[h] = rep[(l, h)][:, : res.shape[1]] - res
        return None

    def edit(x, hook):
        delta = sum(store.values())
        x = x.clone()
        for _, h in receivers:
            x[:, :, h] = x[:, :, h] + delta
        return x
    return [("blocks.0.attn.hook_z", record), (f"blocks.1.hook_{channel}_input", edit)]


if __name__ == "__main__":
    main(sys.argv[1:])
