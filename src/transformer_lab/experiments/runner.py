"""Analysis experiments. Each ``run_*`` function loads a trained run, performs one analysis, and writes
a complete experiment directory (config, results, tables, plots, report) via the registry.

All functions take only JSON-serialisable keyword arguments, recorded in ``config.json``, so any
experiment can be re-executed with :func:`rerun`.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..evaluation.evaluate import baselines, evaluate_model, sweep
from ..evaluation.metrics import run_batched
from ..interpretability.stats import bootstrap_ci
from ..reporting.report import build_report, ci_str, table_md
from ..training import load_run
from ..utils import environment_info, read_json
from ..visualization import plots
from .registry import Experiment, file_sha256, find_experiment


# ------------------------------------------------------------------------------ helpers
def _load(run: str, checkpoint: str = "final"):
    model, task, summary, cfg = load_run(run, checkpoint)
    ck = Path(run) / "checkpoints" / (checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt")
    return model, task, summary, cfg, ck


def _context(run, checkpoint, n, extra="") -> str:
    return f"run={Path(run).name} | checkpoint={checkpoint} | n={n} examples" + (f" | {extra}" if extra else "")


def _setup(run, checkpoint, cfg, task, n, seed, ck, **extra) -> dict:
    m = cfg.model
    return {"run": run, "checkpoint": f"{checkpoint} (sha256 {file_sha256(ck)})",
            "model": f"{m.n_layers} layers x {m.n_heads} heads, d_model={m.d_model}, attn_only={m.attn_only}, "
                     f"norm={m.norm}, positional={m.positional}",
            "task": f"{task.name} {json.dumps(task.params)}", "examples": f"{n} (seed {seed})", **extra}


def _exp(kind: str, run: Optional[str], args: dict) -> Experiment:
    name = Path(run).name if run else kind
    return Experiment(kind, name, {"run": run, "args": args})


def _batch(task, n, seed, split="train"):
    return task.sample(n, np.random.default_rng(seed), split)


def ood_variants(task) -> dict:
    """Out-of-distribution task variants that keep the vocabulary and sequence length."""
    p = task.params
    if task.name == "induction_repeat":
        out = {}
        if p["min_len"] > 2:
            out["shorter_segments"] = {"min_len": 2, "max_len": p["min_len"] - 1}
        longest = (p["seq_len"] - 1) // 2
        if p["max_len"] < longest:
            out["longer_segments"] = {"min_len": p["max_len"] + 1, "max_len": longest}
        return out
    if task.name == "induction_pair":
        return {"with_decoys": {"n_decoys": 3}, "more_distractors": {"n_distractor_pairs": p["n_distractor_pairs"] + 2}}
    if task.name == "assoc_recall":
        return {"max_pairs_only": {"min_pairs": p["max_pairs"]}}
    return {}


# ------------------------------------------------------------------------- environment
def env_report() -> dict:
    from ..utils import artifact_root
    info = environment_info()
    info["artifact_root"] = str(artifact_root())
    try:
        import streamlit  # noqa: F401
        info["streamlit"] = streamlit.__version__
    except Exception:
        info["streamlit"] = "not installed (UI unavailable; CLI works)"
    return info


def smoke(keep: bool = False) -> bool:
    """Install check: train a tiny model, reload it, evaluate, cache, ablate, patch, test a hypothesis."""
    from ..config import ExperimentConfig
    from ..interpretability.hypothesis import HypothesisSpec, run_hypothesis
    from ..training import train_from_config
    from ..utils import PROJECT_ROOT

    steps = []

    def step(name, fn):
        try:
            out = fn()
            steps.append((name, "ok", out))
            print(f"  [ok]   {name}: {out}")
            return True
        except Exception as e:  # report every failure explicitly
            steps.append((name, "FAILED", repr(e)))
            print(f"  [FAIL] {name}: {e!r}")
            return False

    cfg = ExperimentConfig.load(PROJECT_ROOT / "configs" / "experiments" / "smoke.yaml")
    tmp = None if keep else tempfile.TemporaryDirectory()
    run_dir = None if keep else Path(tmp.name) / "run"
    print("[smoke] training a tiny model ...")
    state = {}
    ok = step("train", lambda: state.setdefault("s", train_from_config(cfg, run_dir=run_dir, verbose=False))["status"])
    if ok:
        rd = state["s"]["run_dir"]
        ok &= step("reload checkpoint", lambda: load_run(rd)[0].n_params())
        model, task = load_run(rd)[0], load_run(rd)[1]
        b = _batch(task, 64, 0)
        ok &= step("evaluate", lambda: round(evaluate_model(model, b)["loss"]["mean"], 4))
        ok &= step("cache activations", lambda: len(model.run_with_cache(b.tokens)[1]))
        from ..interpretability.interventions import head_ablation_hooks
        ok &= step("ablate head", lambda: round(float(run_batched(model, b, fwd_hooks_fn=lambda sl: head_ablation_hooks(
            [(1, 0)], "zero"))["loss"].mean()), 4))
        from ..interpretability.patching import Patcher, PatchSpec
        def _patch():
            r = Patcher(model, b, task.corrupt(b, np.random.default_rng(0), "replace")).run(
                PatchSpec("resid0", "blocks.0.hook_resid_pre"))
            # the 40-step smoke model has not learned the task, so the clean/corrupt gap is expected
            # to be too small: the correct output is "uninformative", not a recovery number
            return f"R={r.recovery:.3f}" if r.informative else f"uninformative as expected (gap {r.gap:.3f})"
        ok &= step("patch", _patch)
        spec = HypothesisSpec.load(PROJECT_ROOT / "configs" / "hypotheses" / "induction_2l.yaml")
        spec.n_examples = 64
        for p in spec.predictions:
            if "control" in p:
                p["control"]["n_samples"] = 3
        ok &= step("hypothesis framework", lambda: run_hypothesis(model, task, spec)["verdict_counts"])
    if tmp:
        tmp.cleanup()
    print(f"[smoke] {'PASSED' if ok else 'FAILED'} ({sum(s[1] == 'ok' for s in steps)}/{len(steps)} steps)")
    return bool(ok)


# --------------------------------------------------------------------------------- eval
def run_eval(run, checkpoint="final", n=512, seed=1234, all_checkpoints=False, unseen_only=False) -> Path:
    """``unseen_only``: also report every set restricted to sequences that never occurred in the
    run's training stream (regenerated from its seed) — separates memorisation from generalisation
    for tasks with small sequence spaces."""
    args = dict(run=run, checkpoint=checkpoint, n=n, seed=seed, all_checkpoints=all_checkpoints, unseen_only=unseen_only)
    model, task, summary, cfg, ck = _load(run, checkpoint)
    exp = _exp("eval", run, args)
    rows = []
    sets = {"in-distribution (train split, fresh seed)": _batch(task, n, seed, "train"),
            "held-out split (eval)": _batch(task, n, seed + 1, "eval")}
    for name, ov in ood_variants(task).items():
        sets[f"OOD: {name} {ov}"] = _batch(task.with_params(**ov), n, seed + 2, "train")
    if unseen_only:
        from ..training.trainer import training_stream_hashes, unseen_mask
        seeds = read_json(Path(run) / "seeds.json")
        steps = (summary or {}).get("steps_completed", cfg.train.steps)
        seen = training_stream_hashes(task, seeds["train_data"], steps, cfg.train.batch_size)
        for name in list(sets):
            m = unseen_mask(sets[name], seen)
            if m.sum() < len(m):
                sets[f"{name} [UNSEEN ONLY: {int(m.sum())}/{len(m)}]"] = sets[name].subset(np.nonzero(m)[0])
    results = {"sets": {}}
    for name, b in sets.items():
        s = evaluate_model(model, b)
        bl = baselines(task, b)
        results["sets"][name] = {"metrics": s, "baselines": bl}
        rows.append({"set": name, "accuracy": ci_str(s["accuracy"]), "loss": ci_str(s["loss"]),
                     "seq_correct": ci_str(s["seq_correct"]), "algorithmic": bl["algorithmic"],
                     "chance": bl["chance"], "most_frequent_heuristic": bl["most_frequent_context_token"],
                     "n": bl["n_examples"]})
    exp.save_table("evaluation", rows)
    obs = table_md(rows)
    plot_names = []
    if all_checkpoints:
        from ..instrumentation.inspect import list_checkpoints
        from ..training.checkpoint import load_model
        b = sets["in-distribution (train split, fresh seed)"]
        ck_rows = []
        for step, path in list_checkpoints(run):
            m, _ = load_model(path)
            s = evaluate_model(m, b, n_boot=200)
            ck_rows.append({"step": step, "accuracy": s["accuracy"]["mean"], "acc_lo": s["accuracy"]["lo"],
                            "acc_hi": s["accuracy"]["hi"], "loss": s["loss"]["mean"]})
        results["checkpoints"] = ck_rows
        exp.save_table("checkpoints", ck_rows)
        obs += "\n### All checkpoints (in-distribution)\n\n" + table_md(ck_rows)
        if ck_rows:
            xs = [r["step"] for r in ck_rows]
            exp.save_plot(plots.curves(xs, {"accuracy": [r["accuracy"] for r in ck_rows]},
                                       "Accuracy across checkpoints", _context(run, "all", n), "training step",
                                       "accuracy", bands={"accuracy": ([r["acc_lo"] for r in ck_rows],
                                                                       [r["acc_hi"] for r in ck_rows])}), "checkpoints")
            plot_names.append("checkpoints")
    id_acc = results["sets"]["in-distribution (train split, fresh seed)"]["metrics"]["accuracy"]
    interp = [f"In-distribution accuracy {ci_str(id_acc)} vs chance {task.chance_accuracy:.3f} and exact solver 1.0."]
    if id_acc["lo"] > 5 * task.chance_accuracy:
        interp.append("The model has learned the task to a measurable degree (CI far above chance).")
    else:
        interp.append("The model has NOT clearly learned the task; interpretability results on it would not "
                      "be evidence of a learned mechanism.")
    leak = (summary or {}).get("leakage_audit")
    if leak:
        interp.append(f"Training-set leakage audit: {leak.get('set0_overlap')} / {leak.get('set0_size')} validation and "
                      f"{leak.get('set1_overlap')} / {leak.get('set1_size')} test sequences occurred in training.")
    results["headline"] = {"id_accuracy": id_acc["mean"], "id_acc_lo": id_acc["lo"], "id_acc_hi": id_acc["hi"]}
    exp.save_results(results)
    exp.write_report(build_report("Evaluation", _setup(run, checkpoint, cfg, task, n, seed, ck), obs,
                                  interpretation=interp, plots=plot_names,
                                  limitations=["OOD variants only vary parameters that keep the vocabulary and "
                                               "sequence length; positional extrapolation is not tested."]))
    return exp.finalize(f"acc={id_acc['mean']:.3f}")


# ------------------------------------------------------------------------ inspect/trace
def run_inspect(run, checkpoint="final", n=64, seed=1234, example=0) -> Path:
    from ..instrumentation.inspect import correct_vs_incorrect, export_npz, inspect_example
    args = dict(run=run, checkpoint=checkpoint, n=n, seed=seed, example=example)
    model, task, summary, cfg, ck = _load(run, checkpoint)
    b = _batch(task, n, seed)
    info = inspect_example(model, task, b, example)
    exp = _exp("inspect", run, args)
    export_npz(exp.dir / "activations.npz", patterns=info["patterns"], attn_out_norms=info["attn_out_norms"],
               **{k.replace(".", "_"): v for k, v in info["resid_norms"].items()})
    scored = [r for r in info["positions"] if r["scored"]]
    obs = (f"Example {example}: `{' '.join(info['tokens'])}`\n\nMeta: {info['meta']}\n\nMetrics: "
           f"{ {k: round(v, 3) for k, v in info['metrics'].items()} }\n\n### Scored positions\n\n"
           + table_md([{k: r[k] for k in ("pos", "token", "target", "pred", "p_target", "correct")} for r in scored]))
    cvi = correct_vs_incorrect(model, b)
    obs += f"\n### Correct vs incorrect examples (n={cvi['n_correct']} / {cvi['n_incorrect']})\n\n"
    for grp in ("correct", "incorrect"):
        if cvi[grp]:
            obs += f"- {grp}: max induction score per layer " + str([round(max(x), 3) for x in cvi[grp]["induction"]]) + "\n"
    plot_names = []
    L, H = model.cfg.n_layers, model.cfg.n_heads
    q = scored[-1]["pos"] if scored else None
    for l in range(L):
        for h in range(H):
            name = f"attention_L{l}H{h}"
            exp.save_plot(plots.attention_pattern(info["patterns"][l, h], info["tokens"], f"Attention L{l}H{h}",
                                                  _context(run, checkpoint, 1, f"example {example} (seed {seed})"),
                                                  highlight_query=q), name)
            plot_names.append(name)
    exp.save_results({"example": {k: v for k, v in info.items() if k not in ("patterns",)}, "correct_vs_incorrect": cvi,
                      "headline": {"example": example, **info["metrics"]}})
    exp.write_report(build_report(f"Inspection of example {example}", _setup(run, checkpoint, cfg, task, n, seed, ck), obs,
                                  interpretation=["Attention plots show where heads look on THIS example; they are "
                                                  "not evidence that the attended information is used."],
                                  plots=plot_names))
    return exp.finalize(f"example {example}")


def run_trace(run, checkpoint="final", n=64, seed=1234, example=0, position=None) -> Path:
    from ..instrumentation.inspect import trace_prediction
    args = dict(run=run, checkpoint=checkpoint, n=n, seed=seed, example=example, position=position)
    model, task, summary, cfg, ck = _load(run, checkpoint)
    b = _batch(task, n, seed)
    tr = trace_prediction(model, task, b, example, position)
    exp = _exp("trace", run, args)
    obs = (f"Prediction at position {tr['position']} (token `{tr['token']}`, target `{tr['target']}`); target logit "
           f"{tr['target_logit']:.3f}\n\n### Direct logit attribution (LN scale frozen)\n\n"
           + table_md([{"component": c, "contribution": v} for c, v in tr["dla"]] + [{"component": "constant (LN bias + b_U)",
                                                                                      "contribution": tr["dla_const"]}])
           + "\n### Logit lens\n\n" + table_md(tr["logit_lens"]) + "\n### Top attention sources\n\n"
           + table_md([{"head": a["head"], "top_sources": "; ".join(f"{p}:{t} ({w:.2f})" for p, t, w in a["top_sources"])}
                       for a in tr["attention"]]))
    exp.save_results({**tr, "headline": {"top_component": tr["dla"][0][0], "target_logit": tr["target_logit"]}})
    exp.write_report(build_report("Prediction trace", _setup(run, checkpoint, cfg, task, n, seed, ck), obs,
                                  interpretation=[f"The largest direct contribution comes from {tr['dla'][0][0]}."],
                                  limitations=["DLA measures only direct effects through the final readout with the LN "
                                               "scale frozen; indirect effects via later heads are not captured.",
                                               "Logit lens decodes intermediate layers with the final readout, which "
                                               "they were not trained for."]))
    return exp.finalize(f"top={tr['dla'][0][0]}")


# ------------------------------------------------------------------------------ scores
def run_scores(run, checkpoint="final", n=512, seed=1234) -> Path:
    from ..interpretability.attention_scores import composition_scores, head_scores, ov_copying_score
    from ..training.checkpoint import load_model
    args = dict(run=run, checkpoint=checkpoint, n=n, seed=seed)
    model, task, summary, cfg, ck = _load(run, checkpoint)
    b = _batch(task, n, seed)
    sc = head_scores(model, b)
    ov = ov_copying_score(model)
    comp = composition_scores(model)
    init_path = Path(run) / "checkpoints" / "step_000000.pt"
    init = None
    if init_path.exists():
        m0, _ = load_model(init_path)
        init = {"scores": {k: v.tolist() for k, v in head_scores(m0, b).items()}, "ov_copying": ov_copying_score(m0).tolist()}
        c0 = composition_scores(m0)
        init["K_comp_max"] = float(np.nanmax(c0["K"])) if model.cfg.n_layers > 1 else None
    exp = _exp("scores", run, args)
    ctx = _context(run, checkpoint, n, "observational")
    names = []
    for key in ("prev_token", "induction", "duplicate_token", "bos"):
        exp.save_plot(plots.head_grid(sc[key], f"{key} attention score", ctx, cbar_label="mean attention"), f"score_{key}")
        names.append(f"score_{key}")
    exp.save_plot(plots.head_grid(ov, "OV copying score (direct path, LN ignored)", ctx), "ov_copying")
    names.append("ov_copying")
    rows = [{"head": f"L{l}H{h}", **{k: float(sc[k][l, h]) for k in sc}, "ov_copying": float(ov[l, h])}
            for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    exp.save_table("head_scores", rows)
    obs = table_md(rows)
    if model.cfg.n_layers > 1:
        crow = []
        for kind in ("K", "Q", "V"):
            c = comp[kind]
            for h1 in range(model.cfg.n_heads):
                for h0 in range(model.cfg.n_heads):
                    crow.append({"type": kind, "receiver": f"L1H{h1}", "sender": f"L0H{h0}", "score": float(c[1, h1, 0, h0])})
        exp.save_table("composition_scores", crow)
        top = sorted(crow, key=lambda r: -r["score"])[:6]
        obs += "\n### Strongest weight-based composition scores (layer 0 -> layer 1)\n\n" + table_md(top)
        if init:
            obs += f"\nAt initialisation the maximum K-composition score was {init['K_comp_max']:.3f} (baseline).\n"
    if init:
        obs += ("\n### Same scores at initialisation (baseline)\n\n" +
                table_md([{"score": k, "max over heads": float(np.nanmax(np.array(v)))} for k, v in init["scores"].items()]))
    il, ih = np.unravel_index(np.nanargmax(np.nan_to_num(sc["induction"], nan=-1)), sc["induction"].shape)
    pl, ph = np.unravel_index(np.nanargmax(sc["prev_token"]), sc["prev_token"].shape)
    headline = {"top_induction_head": f"L{il}H{ih}", "top_induction_score": float(sc["induction"][il, ih]),
                "top_prev_head": f"L{pl}H{ph}", "top_prev_score": float(sc["prev_token"][pl, ph])}
    exp.save_results({"scores": {k: v.tolist() for k, v in sc.items()}, "ov_copying": ov.tolist(),
                      "composition": {k: v.tolist() for k, v in comp.items()}, "init_baseline": init, "headline": headline})
    exp.write_report(build_report("Observational head scores", _setup(run, checkpoint, cfg, task, n, seed, ck), obs,
                                  interpretation=[f"Highest induction-pattern score: {headline['top_induction_head']} "
                                                  f"({headline['top_induction_score']:.2f}); highest previous-token score: "
                                                  f"{headline['top_prev_head']} ({headline['top_prev_score']:.2f}).",
                                                  "These identify CANDIDATES only. Causal claims require the ablation, "
                                                  "patching and hypothesis experiments."],
                                  plots=names,
                                  limitations=["Attention scores are averages over scored positions; heads can behave "
                                               "differently on other positions.",
                                               "OV copying and composition scores ignore LayerNorm and the actual input "
                                               "distribution."]))
    return exp.finalize(f"ind={headline['top_induction_head']}:{headline['top_induction_score']:.2f}")


# ----------------------------------------------------------------------------- ablation
def run_ablation(run, checkpoint="final", n=512, seed=1234, method="mean", neurons_top_k=0) -> Path:
    from ..interpretability.scans import ablation_scan, as_grid
    args = dict(run=run, checkpoint=checkpoint, n=n, seed=seed, method=method, neurons_top_k=neurons_top_k)
    model, task, summary, cfg, ck = _load(run, checkpoint)
    b, ref = _batch(task, n, seed), _batch(task, n, seed + 7)
    res = ablation_scan(model, b, ref, method=method, neurons_top_k=neurons_top_k)
    exp = _exp("ablation", run, args)
    rows = res["rows"]
    exp.save_table("ablation", rows)
    ctx = _context(run, checkpoint, n, f"{method}-ablation, reference batch seed {seed + 7}")
    eff = [r for r in rows if "d_logprob" in r]
    exp.save_plot(plots.effect_bars([r["component"] for r in eff], [r["d_logprob"] for r in eff],
                                    [r["d_logprob_lo"] for r in eff], [r["d_logprob_hi"] for r in eff],
                                    f"Change in mean target log-prob when ablating each component ({method})", ctx,
                                    "Δ log-prob (ablated - baseline)"), "ablation_effects")
    g = as_grid(eff, "d_logprob", model.cfg.n_layers, model.cfg.n_heads)
    exp.save_plot(plots.head_grid(g, f"Δ log-prob under {method}-ablation of each head", ctx, center_zero=True), "ablation_grid")
    base = rows[0]
    obs = f"Baseline: log-prob {base['logprob']:.3f}, accuracy {base['accuracy']:.3f}.\n\n" + table_md(
        [{k: r.get(k) for k in ("component", "accuracy", "d_accuracy", "d_logprob", "d_logprob_lo", "d_logprob_hi")} for r in eff])
    worst = min(eff, key=lambda r: r["d_logprob"])
    nonzero = [r["component"] for r in eff if r["d_logprob_hi"] < -0.1]
    interp = [f"Largest single-component effect: {worst['component']} (Δ log-prob {worst['d_logprob']:.3f}, "
              f"CI [{worst['d_logprob_lo']:.3f}, {worst['d_logprob_hi']:.3f}]).",
              f"Components whose ablation lowers log-prob by >0.1 with the whole CI below: {nonzero or 'none'}.",
              "A small single-component effect does NOT show a component is unimportant: redundant components can "
              "compensate (test combinations / patching)."]
    exp.save_results({"method": method, "rows": rows, "headline": {"method": method, "worst": worst["component"],
                                                                     "worst_d_logprob": worst["d_logprob"]}})
    exp.write_report(build_report(f"Ablation scan ({method})", _setup(run, checkpoint, cfg, task, n, seed, ck,
                                                                       method=method), obs, interpretation=interp,
                                  plots=["ablation_effects", "ablation_grid"],
                                  limitations=[f"{method}-ablation replaces activations; zero-ablation in particular "
                                               "can move activations off-distribution and exaggerate effects."]))
    return exp.finalize(f"{method}: worst={worst['component']} {worst['d_logprob']:.2f}")


# ----------------------------------------------------------------------------- patching
def run_patching(run, checkpoint="final", n=512, seed=1234, corruption=None, metric="logprob", direction="denoise",
                 min_gap=0.25) -> Path:
    from ..interpretability.scans import as_grid, head_patching_scan, region_patching_scan
    args = dict(run=run, checkpoint=checkpoint, n=n, seed=seed, corruption=corruption, metric=metric,
                direction=direction, min_gap=min_gap)
    model, task, summary, cfg, ck = _load(run, checkpoint)
    clean = _batch(task, n, seed)
    corrupt = task.corrupt(clean, np.random.default_rng(seed + 11), corruption)
    if metric == "logit_diff" and corrupt.alt_targets is None:
        raise ValueError(f"corruption {corruption!r} defines no alternative answer; use metric=logprob")
    heads = head_patching_scan(model, task, clean, corrupt, metric=metric, direction=direction, min_gap=min_gap)
    regions = region_patching_scan(model, task, clean, corrupt, metric=metric, min_gap=min_gap) if direction == "denoise" else None
    exp = _exp("patching", run, args)
    ctx = _context(run, checkpoint, n, f"{direction}, corruption={corruption or 'default'}, metric={metric}")
    exp.save_table("head_patching", heads["rows"])
    names = []
    g = as_grid(heads["rows"], "recovery", model.cfg.n_layers, model.cfg.n_heads)
    exp.save_plot(plots.head_grid(g, "Recovery R when patching each head's z at scored positions", ctx, center_zero=True,
                                  cbar_label="R"), "head_recovery")
    names.append("head_recovery")
    obs = (f"Clean metric {heads['clean']:.3f}; corrupted {heads['corrupt']:.3f}; gap {heads['gap']:.3f} "
           f"({'informative' if heads['informative'] else 'UNINFORMATIVE — recovery not computed'}).\n\n"
           "### Head patching (z at scored positions)\n\n"
           + table_md([{k: r[k] for k in ("label", "patched", "recovery", "recovery_lo", "recovery_hi")} for r in heads["rows"]]))
    if regions:
        exp.save_table("region_patching", regions["rows"])
        exp.save_plot(plots.region_heatmap(regions["rows"], "recovery", "Recovery R by component and position region",
                                           ctx), "region_recovery")
        names.append("region_recovery")
        obs += "\n### Region patching\n\n" + table_md([{k: r[k] for k in ("label", "recovery", "recovery_lo", "recovery_hi")}
                                                       for r in regions["rows"]])
    interp = []
    if heads["informative"]:
        best = max(heads["rows"], key=lambda r: r["recovery"])
        interp.append(f"Patching {best['label']} alone restores R={best['recovery']:.2f} "
                      f"[{best['recovery_lo']:.2f}, {best['recovery_hi']:.2f}] of the clean-corrupted gap.")
        interp.append("Denoising shows SUFFICIENCY of the patched activation given the rest of the corrupted run; "
                      "a component can be necessary yet show R~0 here (e.g. one input to an AND-like circuit).")
    else:
        interp.append("Clean and corrupted runs are too similar: this patching experiment is uninformative.")
    exp.save_results({"heads": heads, "regions": regions,
                      "headline": {"gap": heads["gap"], "informative": heads["informative"],
                                   "best_head": max(heads["rows"], key=lambda r: np.nan_to_num(r["recovery"], nan=-9))["label"]}})
    exp.write_report(build_report("Activation patching", _setup(run, checkpoint, cfg, task, n, seed, ck,
                                                                 corruption=corruption or "task default",
                                                                 metric=metric, direction=direction), obs,
                                  interpretation=interp, plots=names,
                                  limitations=["Recovery is a ratio of batch means; per-example recoveries vary.",
                                               "Conclusions depend on the corruption: a different corruption tests a "
                                               "different counterfactual (Zhang & Nanda, 2024)."]))
    return exp.finalize(f"gap={heads['gap']:.2f}")


# --------------------------------------------------------------------------- hypothesis
def run_hypothesis_cmd(spec, run, checkpoint="final") -> Path:
    from ..interpretability.hypothesis import HypothesisSpec, run_hypothesis
    runs = run if isinstance(run, list) else [run]
    args = dict(spec=str(spec), run=runs, checkpoint=checkpoint)
    hs = HypothesisSpec.load(spec)
    exp = Experiment("hypothesis", hs.name, {"run": runs[0] if len(runs) == 1 else None, "runs": runs, "args": args,
                                              "spec_file": str(spec)})
    per_run = {}
    for r in runs:
        model, task, summary, cfg, ck = _load(r, checkpoint)
        res = run_hypothesis(model, task, hs)
        res["checkpoint_sha256"] = file_sha256(ck)
        res["run_config"] = {"layers": cfg.model.n_layers, "positional": cfg.model.positional, "seed": cfg.seed,
                             "attn_only": cfg.model.attn_only}
        per_run[r] = res
    # ---- tables
    pred_ids = [p["id"] for p in hs.predictions]
    matrix = []
    for r, res in per_run.items():
        row = {"run": Path(r).name, "baseline_acc": res["baseline"]["accuracy"],
               "candidates": "; ".join(f"{k}={','.join(v)}" for k, v in res["candidates"].items())}
        for p in res["predictions"]:
            row[p["id"]] = p["verdict"]
        matrix.append(row)
    exp.save_table("verdicts", matrix)
    detail_rows = []
    for r, res in per_run.items():
        for p in res["predictions"]:
            o = p.get("observed", {})
            detail_rows.append({"run": Path(r).name, "id": p["id"], "verdict": p["verdict"],
                                "observed": o.get("mean", o.get("recovery")), "lo": o.get("lo"), "hi": o.get("hi"),
                                "reason": p.get("reason", "")})
    exp.save_table("predictions", detail_rows)
    obs = f"**Statement.** {hs.statement}\n\n### Verdicts per run\n\n" + table_md(matrix, ["run", "baseline_acc", "candidates"] + pred_ids)
    obs += "\n### Prediction details\n\n" + "\n".join(
        f"- **{p['id']}** — {p['claim']}" for p in hs.predictions) + "\n\n" + table_md(detail_rows)
    # ---- plot: effects per prediction per run
    names = []
    eff_preds = [p for p in hs.predictions if p["test"]["type"] in ("ablate_heads", "path")]
    if eff_preds:
        labels, means, los, his = [], [], [], []
        for r, res in per_run.items():
            for p in res["predictions"]:
                if p["id"] in [e["id"] for e in eff_preds] and "observed" in p:
                    labels.append(f"{p['id'].split('_', 1)[0]}·{Path(r).name.split('_s')[1][:1] if '_s' in Path(r).name else ''}")
                    means.append(p["observed"]["mean"])
                    los.append(p["observed"]["lo"])
                    his.append(p["observed"]["hi"])
        exp.save_plot(plots.effect_bars(labels, means, los, his, "Intervention effects (Δ log-prob) per prediction · seed",
                                        f"hypothesis={hs.name} | runs={len(runs)} | n_test={hs.n_examples} per run",
                                        "Δ log-prob"), "prediction_effects")
        names.append("prediction_effects")
    counts = {v: sum(res["verdict_counts"][v] for res in per_run.values()) for v in ("supported", "contradicted", "inconclusive")}
    interp = [f"Across {len(runs)} run(s) and {len(pred_ids)} predictions: {counts}.",
              "A prediction is only 'supported' under its stated intervention and thresholds; see the reasons column."]
    for pid in pred_ids:
        vs = [p["verdict"] for res in per_run.values() for p in res["predictions"] if p["id"] == pid]
        if len(set(vs)) > 1:
            interp.append(f"{pid} is NOT consistent across runs ({vs}): the finding does not replicate uniformly.")
    interp.append("Alternative explanations recorded in advance: " + " | ".join(hs.alternatives))
    exp.save_results({"spec": hs.to_dict(), "per_run": per_run, "verdict_counts": counts,
                      "headline": {"runs": len(runs), **counts}})
    exp.write_report(build_report(f"Circuit hypothesis test: {hs.name}",
                                  {"spec": str(spec), "runs": ", ".join(Path(r).name for r in runs),
                                   "checkpoint": checkpoint, "selection / test / reference seeds":
                                       f"{hs.selection_seed} / {hs.test_seed} / {hs.reference_seed}",
                                   "examples per split": hs.n_examples, "ablation method": hs.ablation_method},
                                  obs, interpretation=interp, plots=names,
                                  limitations=["Candidate heads are chosen by a declared rule on a separate selection set; "
                                               "the rule itself (top-1) can miss distributed circuits.",
                                               "Thresholds (min_effect, recovery_min) are part of the hypothesis and were "
                                               "fixed before running it."]))
    return exp.finalize(f"{counts}")


# ----------------------------------------------------------------------------- formation
def run_formation(run, checkpoint="final", n=512, seed=1234, ablate=False) -> Path:
    from ..instrumentation.inspect import checkpoint_trajectory
    from ..interpretability.interventions import Reference, head_ablation_hooks, reference_names
    args = dict(run=run, checkpoint=checkpoint, n=n, seed=seed, ablate=ablate)
    model, task, summary, cfg, ck = _load(run, checkpoint)
    b, ref_b = _batch(task, n, seed), _batch(task, n, seed + 7)

    def extra(m, batch):
        if not ablate or m.cfg.n_layers < 2:
            return {}
        from ..interpretability.attention_scores import head_scores
        sc = head_scores(m, batch)
        h = int(np.nanargmax(np.nan_to_num(sc["induction"][-1], nan=-1)))
        ref = Reference.from_batch(m, ref_b, reference_names(m))
        base = run_batched(m, batch)["logprob"]
        abl = run_batched(m, batch, fwd_hooks_fn=lambda sl: head_ablation_hooks([(m.cfg.n_layers - 1, h)], "mean", ref))["logprob"]
        return {"top_induction_head": f"L{m.cfg.n_layers - 1}H{h}", "ablation_d_logprob": float((abl - base).mean())}

    rows = checkpoint_trajectory(run, b, extra_fn=extra)
    exp = _exp("formation", run, args)
    flat = [{k: v for k, v in r.items() if k != "head_scores"} for r in rows]
    exp.save_table("trajectory", flat)
    xs = [r["step"] for r in rows]
    series = {"accuracy": [r["accuracy"] for r in rows], "max prev-token score (L0)": [r["max_prev_token_L0"] for r in rows]}
    if model.cfg.n_layers > 1:
        series["max induction score (L>=1)"] = [r["max_induction"] for r in rows]
    exp.save_plot(plots.curves(xs, series, "Mechanism formation across training", _context(run, "all saved", n),
                               "training step", "score / accuracy", logx=True), "formation")
    exp.save_plot(plots.curves(xs, {"loss": [r["loss"] for r in rows]}, "Loss across checkpoints",
                               _context(run, "all saved", n), "training step", "loss (nats)", logx=True), "formation_loss")
    names = ["formation", "formation_loss"]
    if ablate and any("ablation_d_logprob" in r for r in rows):
        exp.save_plot(plots.curves(xs, {"Δ log-prob, ablate top induction head": [r.get("ablation_d_logprob", np.nan) for r in rows]},
                                   "Causal importance of the top induction head over training",
                                   _context(run, "all saved", n, "mean-ablation"), "training step", "Δ log-prob", logx=True),
                      "formation_ablation")
        names.append("formation_ablation")
    obs = table_md(flat)
    # find the steepest accuracy increase between consecutive checkpoints
    interp = []
    if len(rows) > 2:
        jumps = [(rows[i + 1]["accuracy"] - rows[i]["accuracy"]) / max(1, rows[i + 1]["step"] - rows[i]["step"]) for i in range(len(rows) - 1)]
        i = int(np.argmax(jumps))
        interp.append(f"Steepest accuracy increase per step between checkpoints {rows[i]['step']} and {rows[i + 1]['step']} "
                      f"({rows[i]['accuracy']:.2f} -> {rows[i + 1]['accuracy']:.2f}); induction score "
                      f"{rows[i]['max_induction']:.2f} -> {rows[i + 1]['max_induction']:.2f}.")
        interp.append("Co-occurrence of score and accuracy changes is correlational; checkpoint spacing limits "
                      "temporal resolution. One run cannot establish a universal phase transition.")
    exp.save_results({"rows": rows, "headline": {"n_checkpoints": len(rows), "final_accuracy": rows[-1]["accuracy"] if rows else None}})
    exp.write_report(build_report("Mechanism formation over training", _setup(run, "all saved checkpoints", cfg, task, n, seed, ck),
                                  obs, interpretation=interp, plots=names))
    return exp.finalize(f"{len(rows)} checkpoints")


# --------------------------------------------------------------------------------- sweep
def run_sweep(run, param, values, checkpoint="final", n=256, seed=1234, task_set=(), split="train",
              task_name=None) -> Path:
    """Sweep one task parameter. ``task_name`` evaluates the model ZERO-SHOT on a different task
    family (it must use the same vocabulary size and fit in the context) — a test of whether the
    learned mechanism transfers to differently structured inputs."""
    from ..config import parse_override
    from ..data.tasks import make_task
    args = dict(run=run, param=param, values=list(values), checkpoint=checkpoint, n=n, seed=seed, task_set=list(task_set),
                split=split, task_name=task_name)
    model, task, summary, cfg, ck = _load(run, checkpoint)
    fixed = dict(parse_override(x) for x in task_set)
    if task_name and task_name != task.name:
        base_task = make_task(task_name, fixed)
        if base_task.vocab_size != model.cfg.vocab_size or base_task.seq_len > model.cfg.n_ctx:
            raise ValueError(f"{task_name} {fixed} has vocab {base_task.vocab_size}/seq_len {base_task.seq_len}; "
                             f"model needs vocab {model.cfg.vocab_size} and seq_len <= {model.cfg.n_ctx}")
        task = base_task
    else:
        base_task = task.with_params(**fixed) if fixed else task
    rows = sweep(model, base_task, param, values, n=n, seed=seed, split=split)
    for r in rows:  # heuristic baseline per variant (shows whether a frequency shortcut would suffice)
        if "skipped" not in r:
            from ..data.tasks import most_frequent_solver
            from ..evaluation.metrics import solver_accuracy
            vb = base_task.with_params(**{p_: r[param] for p_ in param.split("+")}).sample(
                n, np.random.default_rng([seed, 99]), split)
            r["most_frequent_heuristic"] = solver_accuracy(most_frequent_solver(vb.tokens, vb.loss_mask), vb)
    exp = _exp("sweep", run, args)
    exp.save_table("sweep", rows)
    ok = [r for r in rows if "skipped" not in r]
    names = []
    if ok:
        xs = [r[param] for r in ok]
        exp.save_plot(plots.curves(xs, {"model accuracy": [r["accuracy_mean"] for r in ok],
                                        "exact solver": [r["algorithmic"] for r in ok],
                                        "most-frequent-token heuristic": [r["most_frequent_heuristic"] for r in ok]},
                                   f"Accuracy vs {param}", _context(run, checkpoint, n, f"fixed: {fixed or 'none'}, split={split}"),
                                   param, "accuracy", bands={"model accuracy": ([r["accuracy_lo"] for r in ok],
                                                                                [r["accuracy_hi"] for r in ok])}), "sweep")
        names.append("sweep")
    obs = table_md([{k: r.get(k) for k in (param, "accuracy_mean", "accuracy_lo", "accuracy_hi", "loss_mean", "algorithmic",
                                         "most_frequent_heuristic", "n", "skipped")}
                    for r in rows])
    interp = []
    if len(ok) > 1:
        accs = [r["accuracy_mean"] for r in ok]
        interp.append(f"Accuracy ranges from {min(accs):.3f} ({param}={ok[int(np.argmin(accs))][param]}) to "
                      f"{max(accs):.3f} ({param}={ok[int(np.argmax(accs))][param]}).")
    exp.save_results({"rows": rows, "fixed": fixed, "headline": {"param": param, "n_values": len(rows)}})
    exp.write_report(build_report(f"Sweep over {param}" + (f" (zero-shot on {task_name})" if task_name else ""),
                                  _setup(run, checkpoint, cfg, base_task, n, seed, ck, fixed_params=fixed),
                                  obs, interpretation=interp, plots=names,
                                  limitations=["Variants outside the training distribution test generalisation of the "
                                               "learned mechanism, not of 'reasoning'."]))
    return exp.finalize(f"{param}: {len(rows)} values")


# --------------------------------------------------------------------------------- rerun
def rerun(ref: str) -> Path:
    d = find_experiment(ref)
    cfg = read_json(d / "config.json")
    if cfg["kind"] == "prereg":
        from .context_experiment import run_context_experiment, run_heads_prereg, run_route_prereg
        name = cfg["name"]
        fn = {"context_matching_v1": run_context_experiment, "context_route_v1": run_route_prereg,
              "context_heads_v1": run_heads_prereg}[name]
        return fn(cfg["args"]["spec"])
    if cfg["kind"] == "investigation":
        if "name" in cfg.get("args", {}):
            from .investigations import run_investigation
            return run_investigation(**cfg["args"])
        raise ValueError("this investigation record was written by a script in scripts/ (see its config.json "
                         "'args'); re-run that script, or use the equivalent `itl investigate` command")
    fn = {"eval": run_eval, "inspect": run_inspect, "trace": run_trace, "scores": run_scores, "ablation": run_ablation,
          "patching": run_patching, "hypothesis": run_hypothesis_cmd, "formation": run_formation, "sweep": run_sweep}[cfg["kind"]]
    return fn(**cfg["args"])
