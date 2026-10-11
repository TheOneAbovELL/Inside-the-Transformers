"""Analysis for the preregistered context-matching experiment.

Implements EXACTLY the decision rules frozen in ``configs/preregistration/context_matching_v1.yaml``
(rules are hard-coded here from that spec; the spec's SHA-256 is checked against the freeze log so
an edited spec cannot be analysed silently). Writes an experiment record with every measurement,
the validity checks, the verdicts and the reasons.
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from ..interpretability.context import (bootstrap_difference, context_preference, context_sensitivity,
                                        cp_under_path, induction_group)
from ..models.handwired import build_induction_model
from ..reporting.report import build_report, table_md
from ..training import load_run
from ..utils import PROJECT_ROOT, read_json
from ..visualization import plots
from .registry import Experiment

PREREG_LOG = "preregistration_log.jsonl"


def _frozen_hash(spec_id: str, rel_path: str, artifacts: Path) -> Optional[str]:
    log = artifacts / PREREG_LOG
    if not log.exists():
        return None
    for line in log.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("id") == spec_id and e.get("event") == "frozen":
            return e["sha256"].get(rel_path)
    return None


def _find_run(config_name: str, seed: int, runs_root: Path) -> Optional[str]:
    cands = sorted(glob.glob(str(runs_root / f"{config_name}_s{seed}_*")))
    done = [c for c in cands if (Path(c) / "summary.json").exists()]
    return done[-1] if done else None


def _measure(model, label: str, own_val_acc: Optional[float], spec: dict) -> dict:
    p = spec["measurements"]["primary"]
    cp = context_preference(model, n=p["n"], seed=p["seed"], context_len=1)
    cp2 = context_preference(model, n=p["n"], seed=607, context_len=2)
    sens = context_sensitivity(model, n=512, seed=404)
    group = induction_group(model, n=512, seed=101, min_score=0.25)
    out = {"model": label, "own_val_accuracy": own_val_acc, "group": [f"L{l}H{h}" for l, h in group],
           "cp": cp["cp"], "cp_match_recent": cp["cp_match_recent"], "cp_match_earlier": cp["cp_match_earlier"],
           "either_rate": cp["either_rate"], "selection_rate": cp["selection_rate"],
           "cp_context_len2": cp2["cp"], "either_rate_context_len2": cp2["either_rate"],
           "sensitivity_affected": sens["affected"], "sensitivity_control": sens["control"],
           "_cp_values": cp["per_example_logit_diff"]}
    out["path"] = {}
    if group:
        for ch in ("q", "k", "v"):
            r = cp_under_path(model, group, ch, n=p["n"], seed=p["seed"])
            out["path"][ch] = {"cp": r["cp"], "either_rate": r["either_rate"], "senders": r["senders"]}
    return out


def _valid(m: dict) -> tuple[bool, str]:
    if m["own_val_accuracy"] is not None and m["own_val_accuracy"] < 0.80:
        return False, f"own-task validation accuracy {m['own_val_accuracy']:.3f} < 0.80"
    if m["either_rate"] < 0.50:
        return False, f"competition either_rate {m['either_rate']:.3f} < 0.50"
    return True, ""


def _within(ci: dict, lo: float, hi: float) -> bool:
    return ci["lo"] >= lo and ci["hi"] <= hi


def evaluate_rules(measured: dict) -> dict:
    """Apply the frozen decision rules. ``measured``: condition -> list of measurement dicts."""
    out = {}
    # ---- H1
    orig = [m for m in measured["original"] if m["valid"]]
    n_sup = sum(m["cp"]["lo"] > 0.5 for m in orig)
    n_neg = sum(m["cp"]["hi"] < 0 for m in orig)
    n_null = sum(_within(m["cp"], -0.5, 0.5) for m in orig)
    if n_sup >= 2 and n_neg == 0:
        v = "supported"
    elif n_null >= 2:
        v = "contradicted"
    else:
        v = "inconclusive"
    out["H1"] = {"verdict": v, "n_valid": len(orig), "n_cp_lower_ci_above_0.5": n_sup,
                 "n_cp_upper_ci_below_0": n_neg, "n_cp_ci_within_pm0.5": n_null}
    # ---- H2
    same = {m["seed"]: m for m in measured["same_order"] if m["valid"]}
    shuf = {m["seed"]: m for m in measured["shuffled"] if m["valid"]}
    pairs = sorted(set(same) & set(shuf))
    if len(same) < 2 or len(shuf) < 2 or len(pairs) < 2:
        out["H2"] = {"verdict": "inconclusive", "reason": "fewer than 2 valid models per condition / pairs"}
    else:
        diffs = {s: bootstrap_difference(same[s]["_cp_values"], shuf[s]["_cp_values"], seed=s) for s in pairs}
        mean_diff = float(np.mean([d["mean"] for d in diffs.values()]))
        n_above0 = sum(d["lo"] > 0 for d in diffs.values())
        n_equiv = sum(_within(d, -0.5, 0.5) for d in diffs.values())
        if mean_diff > 0.5 and n_above0 >= 2:
            v = "supported"
        elif n_equiv >= 2:
            v = "contradicted"
        else:
            v = "inconclusive"
        out["H2"] = {"verdict": v, "mean_paired_difference": mean_diff, "pairs": {str(k): d for k, d in diffs.items()},
                     "n_pairs_diff_ci_above_0": n_above0, "n_pairs_diff_ci_within_pm0.5": n_equiv}
    # ---- H3
    per = []
    for cond in ("original", "same_order", "shuffled"):
        for m in measured[cond]:
            if not m["valid"] or m["cp"]["lo"] <= 0.5:
                continue
            q = m["path"].get("q")
            if q is None:
                per.append({"model": m["model"], "verdict": "inconclusive", "reason": "no induction group"})
                continue
            base, qc = m["cp"], q["cp"]
            if q["either_rate"] < 0.5:
                pv, reason = "inconclusive", f"either_rate under query removal {q['either_rate']:.2f} < 0.5"
            elif qc["mean"] <= 0.5 * base["mean"] and qc["hi"] < base["lo"]:
                pv, reason = "supported", f"CP {base['mean']:.2f} -> {qc['mean']:.2f}"
            elif qc["lo"] > 0.75 * base["mean"]:
                pv, reason = "contradicted", f"CP {base['mean']:.2f} -> {qc['mean']:.2f} (largely survives)"
            else:
                pv, reason = "inconclusive", f"CP {base['mean']:.2f} -> {qc['mean']:.2f}"
            per.append({"model": m["model"], "verdict": pv, "reason": reason})
    if not per:
        out["H3"] = {"verdict": "not evaluated", "reason": "no valid model with CP lower CI > 0.5", "per_model": []}
    else:
        k = len(per)
        sup = sum(p["verdict"] == "supported" for p in per)
        con = sum(p["verdict"] == "contradicted" for p in per)
        v = "supported" if sup >= 2 * k / 3 else ("contradicted" if con >= 2 * k / 3 else "inconclusive")
        out["H3"] = {"verdict": v, "per_model": per, "n_evaluated": k}
    return out


def run_context_experiment(spec_path: str = "configs/preregistration/context_matching_v1.yaml") -> Path:
    from ..utils import artifact_root
    artifacts = artifact_root()
    spec_file = Path(spec_path)
    rel = spec_file.resolve().relative_to(PROJECT_ROOT).as_posix()
    spec = yaml.safe_load(spec_file.read_text(encoding="utf-8"))
    sha = hashlib.sha256(spec_file.read_bytes()).hexdigest()
    frozen = _frozen_hash(spec["id"], rel, artifacts)
    if frozen is None:
        raise RuntimeError(f"{spec['id']} has no 'frozen' entry in {artifacts / PREREG_LOG}; refusing to analyse")
    if frozen != sha:
        raise RuntimeError(f"{rel} changed since it was frozen (sha {sha[:12]} != {frozen[:12]}); refusing to analyse")

    exp = Experiment("prereg", spec["id"], {"run": None, "args": {"spec": rel}, "spec_sha256": sha})
    measured = {"calibration": [], "original": [], "same_order": [], "shuffled": []}
    # calibration (metric validity)
    for label, m in (("handwired_single_token", build_induction_model(48, 40)),
                     ("handwired_context_gain2", build_induction_model(48, 40, context_gain=2.0))):
        r = _measure(m, label, None, spec)
        r["valid"], r["exclusion"] = True, ""
        measured["calibration"].append(r)
    cal = {r["model"]: r for r in measured["calibration"]}
    metric_valid = abs(cal["handwired_single_token"]["cp"]["mean"]) < 0.05 and cal["handwired_context_gain2"]["cp"]["lo"] > 1.0

    runs_root = artifacts / "runs"
    entries = [("original", Path(p).name, str(PROJECT_ROOT / p) if not Path(p).is_absolute() else p, None)
               for p in spec["models"]["original"]]
    for cond in ("same_order", "shuffled"):
        cfg_name = Path(spec["models"][cond]["config"]).stem
        for seed in spec["models"][cond]["seeds"]:
            run = _find_run(cfg_name, seed, runs_root)
            entries.append((cond, f"{cfg_name}_s{seed}", run, seed))
    for cond, label, run, seed in entries:
        if run is None:
            measured[cond].append({"model": label, "seed": seed, "valid": False, "exclusion": "run missing / incomplete"})
            continue
        model, task, summary, cfg = load_run(run)
        r = _measure(model, Path(run).name, summary["final_eval"]["val_accuracy"], spec)
        r["seed"] = cfg.seed if seed is None else seed
        r["valid"], r["exclusion"] = _valid(r)
        measured[cond].append(r)
        print(f"  measured {Path(run).name}: CP {r['cp']['mean']:+.3f} [{r['cp']['lo']:+.3f}, {r['cp']['hi']:+.3f}]", flush=True)

    verdicts = evaluate_rules(measured) if metric_valid else {"invalid": "metric calibration failed"}
    # ---- tables
    rows = []
    for cond, ms in measured.items():
        for m in ms:
            if "cp" not in m:
                rows.append({"condition": cond, "model": m["model"], "valid": False, "exclusion": m["exclusion"]})
                continue
            rows.append({
                "condition": cond, "model": m["model"], "valid": m["valid"], "exclusion": m["exclusion"],
                "own_val_acc": m["own_val_accuracy"], "group": " ".join(m["group"]),
                "CP": m["cp"]["mean"], "CP_lo": m["cp"]["lo"], "CP_hi": m["cp"]["hi"],
                "CP_recent": m["cp_match_recent"]["mean"], "CP_earlier": m["cp_match_earlier"]["mean"],
                "selection_rate": m["selection_rate"], "either_rate": m["either_rate"],
                "CP_ctx2": m["cp_context_len2"]["mean"],
                "sens_affected": m["sensitivity_affected"]["mean"], "sens_control": m["sensitivity_control"]["mean"],
                "CP_q_removed": m["path"].get("q", {}).get("cp", {}).get("mean"),
                "CP_k_removed": m["path"].get("k", {}).get("cp", {}).get("mean"),
                "CP_v_removed": m["path"].get("v", {}).get("cp", {}).get("mean"),
                "either_q_removed": m["path"].get("q", {}).get("either_rate"),
                "either_k_removed": m["path"].get("k", {}).get("either_rate"),
            })
    exp.save_table("measurements", rows)
    clean = {c: [{k: v for k, v in m.items() if k != "_cp_values"} for m in ms] for c, ms in measured.items()}
    exp.save_results({"spec_id": spec["id"], "spec_sha256": sha, "metric_valid": metric_valid,
                      "measurements": clean, "verdicts": verdicts,
                      "headline": {k: v.get("verdict") for k, v in verdicts.items() if isinstance(v, dict)}})
    plot_rows = [r for r in rows if "CP" in r]
    if plot_rows:
        exp.save_plot(plots.effect_bars([f"{r['condition'][:5]}:{r['model'][-19:]}" for r in plot_rows],
                                        [r["CP"] for r in plot_rows], [r["CP_lo"] for r in plot_rows],
                                        [r["CP_hi"] for r in plot_rows],
                                        "Context preference CP = logit[A] - logit[B] (context-matching vs non-matching successor)",
                                        f"prereg {spec['id']} | n={spec['measurements']['primary']['n']} per model | seed 606",
                                        "CP (logits)"), "context_preference")
    obs = table_md([{k: r.get(k) for k in ("condition", "model", "valid", "own_val_acc", "CP", "CP_lo", "CP_hi",
                                           "CP_recent", "CP_earlier", "selection_rate", "either_rate", "CP_ctx2")}
                    for r in rows])
    obs += "\n### Secondary: old sensitivity measure and path removals (CP after removing layer-0 output from the group's q / k / v)\n\n"
    obs += table_md([{k: r.get(k) for k in ("model", "group", "sens_affected", "sens_control", "CP_q_removed",
                                            "either_q_removed", "CP_k_removed", "either_k_removed", "CP_v_removed")}
                     for r in rows if "CP" in r])
    obs += "\n### Verdicts (frozen rules)\n\n```\n" + json.dumps(verdicts, indent=1, default=str) + "\n```\n"
    interp = [f"Metric calibration {'PASSED' if metric_valid else 'FAILED'}: single-token CP "
              f"{cal['handwired_single_token']['cp']['mean']:.3f}, context-model CP {cal['handwired_context_gain2']['cp']['mean']:.3f}."]
    for h, v in verdicts.items():
        if isinstance(v, dict):
            interp.append(f"{h}: {v.get('verdict')}.")
    exp.write_report(build_report(f"Preregistered experiment {spec['id']}",
                                  {"spec": f"{rel} (sha256 {sha[:16]}, matches freeze log)",
                                   "primary measurement": "CP on ContextCompetition, n 1024, seed 606",
                                   "decision rules": "see spec; implemented in experiments/context_experiment.py"},
                                  obs, interpretation=interp, plots=["context_preference"] if plot_rows else [],
                                  limitations=["Competition inputs are out of distribution for every trained model "
                                               "(two occurrences of the query token never occur in training).",
                                               "CP is in logits; its scale depends on each model's confidence."]))
    with open(artifacts / PREREG_LOG, "a", encoding="utf-8") as f:
        code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        f.write(json.dumps({"id": spec["id"], "event": "analyzed", "time": datetime.datetime.now().isoformat(timespec="seconds"),
                            "experiment": exp.id, "analysis_code_sha256": code_sha,
                            "verdicts": {k: v.get("verdict") for k, v in verdicts.items() if isinstance(v, dict)}}) + "\n")
    return exp.finalize(json.dumps({k: v.get("verdict") for k, v in verdicts.items() if isinstance(v, dict)}))


def evaluate_route_rules(per_model: list) -> dict:
    """Frozen rules of context_route_v1 (see configs/preregistration/context_route_v1.yaml)."""
    valid = [m for m in per_model if m["valid"]]
    if len(valid) < 2:
        return {"verdict": "inconclusive", "reason": "fewer than 2 valid models"}
    sup = all(m["q"]["recovery_lo"] > 0.5 and m["k"]["recovery_hi"] < 0.5 and m["v"]["recovery_hi"] < 0.5 for m in valid)
    con = all(m["q"]["recovery_hi"] < 0.5 for m in valid)
    return {"verdict": "supported" if sup else ("contradicted" if con else "inconclusive"), "n_valid": len(valid)}


_KNOWN_YAML_DEFECTS = {
    # context_route_v1 (frozen 2026-10-11T22:58:12): an unquoted free-text value starting with '>'
    # is read by YAML as a block-scalar indicator, so the frozen file does not parse. The line only
    # restates a validity rule that the analysis code implements verbatim. The file is never edited
    # (its hash must keep matching); the value is quoted in memory and the deviation is logged.
    "  experiment: >= 2 valid models, else inconclusive":
        '  experiment: ">= 2 valid models, else inconclusive"',
}


def load_frozen_spec(spec_file: Path) -> tuple[dict, str]:
    """Parse a frozen spec; if it fails only because of a known, logged YAML defect, repair it in
    memory and return a description of the deviation (empty string if none)."""
    text = spec_file.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text), ""
    except yaml.YAMLError:
        fixed = text
        for bad, good in _KNOWN_YAML_DEFECTS.items():
            fixed = fixed.replace(bad, good)
        spec = yaml.safe_load(fixed)
        return spec, ("frozen spec has a YAML syntax error in a free-text validity line (unquoted value "
                      "starting with '>'); value quoted in memory for parsing only; file and hash unchanged; "
                      "the rule it restates is implemented verbatim in evaluate_route_rules")


def run_route_prereg(spec_path: str = "configs/preregistration/context_route_v1.yaml") -> Path:
    from ..interpretability.context import context_route
    from ..utils import artifact_root
    artifacts = artifact_root()
    spec_file = Path(spec_path)
    rel = spec_file.resolve().relative_to(PROJECT_ROOT).as_posix()
    sha = hashlib.sha256(spec_file.read_bytes()).hexdigest()
    spec, deviation = load_frozen_spec(spec_file)
    frozen = _frozen_hash(spec["id"], rel, artifacts)
    if frozen is None or frozen != sha:
        raise RuntimeError(f"{rel} is not frozen or changed since freezing; refusing to analyse")
    if deviation:
        with open(artifacts / PREREG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": spec["id"], "event": "deviation",
                                "time": datetime.datetime.now().isoformat(timespec="seconds"),
                                "description": deviation}) + "\n")
    exp = Experiment("prereg", spec["id"], {"run": None, "args": {"spec": rel}, "spec_sha256": sha})
    cfg_name = Path(spec["models"]["confirmation"]["config"]).stem
    per_model = []
    gt = context_route(build_induction_model(48, 40, context_gain=2.0))
    for seed in spec["models"]["confirmation"]["seeds"]:
        run = _find_run(cfg_name, seed, artifacts / "runs")
        if run is None:
            per_model.append({"model": f"{cfg_name}_s{seed}", "valid": False, "exclusion": "run missing"})
            continue
        model, _, summary, _ = load_run(run)
        r = context_route(model)
        acc = summary["final_eval"]["val_accuracy"]
        valid = acc >= 0.80 and r["q"]["informative"]
        per_model.append({"model": Path(run).name, "own_val_accuracy": acc, "valid": valid,
                          "exclusion": "" if valid else "accuracy < 0.80 or uninformative gap", **r})
        print(f"  {Path(run).name}: R_q {r['q']['recovery']:.2f} R_k {r['k']['recovery']:.2f} "
              f"R_v {r['v']['recovery']:.2f} R_all {r['all']['recovery']:.2f}", flush=True)
    verdict = evaluate_route_rules(per_model)
    rows = [{"model": m["model"], "valid": m["valid"], **{f"R_{e}": m[e]["recovery"] for e in ("q", "k", "v", "all") if e in m},
             **{f"R_{e}_lo": m[e]["recovery_lo"] for e in ("q", "k", "v") if e in m},
             **{f"R_{e}_hi": m[e]["recovery_hi"] for e in ("q", "k", "v") if e in m},
             "clean": m.get("q", {}).get("clean"), "corrupt": m.get("q", {}).get("corrupt")} for m in per_model]
    rows.insert(0, {"model": "handwired_context_gain2 (ground truth)", "valid": True,
                    **{f"R_{e}": gt[e]["recovery"] for e in ("q", "k", "v", "all")}})
    exp.save_table("route", rows)
    exp.save_results({"spec_id": spec["id"], "spec_sha256": sha, "ground_truth": gt, "per_model": per_model,
                      "verdict": verdict, "headline": {"H3b": verdict["verdict"]}})
    exp.write_report(build_report(f"Preregistered experiment {spec['id']}",
                                  {"spec": f"{rel} (sha256 {sha[:16]}, matches freeze log)",
                                   "models": "fresh seeds trained after freezing"},
                                  table_md(rows) + f"\n\nVerdict (frozen rule): **{verdict['verdict']}** {verdict}\n",
                                  interpretation=[f"H3b: {verdict['verdict']}."],
                                  limitations=["Layer-0 heads are patched as a group.",
                                               "Competition inputs are out of distribution for the trained models."]))
    with open(artifacts / PREREG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": spec["id"], "event": "analyzed", "time": datetime.datetime.now().isoformat(timespec="seconds"),
                            "experiment": exp.id, "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "verdicts": {"H3b": verdict["verdict"]}}) + "\n")
    return exp.finalize(f"H3b: {verdict['verdict']}")


def evaluate_heads_rules(per_model: list) -> dict:
    """Frozen rule of context_heads_v1 (configs/preregistration/context_heads_v1.yaml)."""
    def q(m, head):
        return next(r for r in m["rows"] if r["sender"] == head and r["channel"] == "q")

    def supported(m):
        h0 = q(m, "L0H0")
        others = [q(m, f"L0H{h}") for h in (1, 2, 3)]
        return (h0["recovery_lo"] > 0.8 and all(h0["recovery_lo"] > o["recovery_hi"] for o in others)
                and q(m, "L0H2")["recovery_hi"] < 0.3 and q(m, "L0H3")["recovery_hi"] < 0.3)

    if len(per_model) < 2:
        return {"verdict": "inconclusive", "reason": "fewer than 2 models"}
    if all(supported(m) for m in per_model):
        return {"verdict": "supported", "n_models": len(per_model)}
    if all(q(m, "L0H0")["recovery_hi"] < 0.5 for m in per_model):
        return {"verdict": "contradicted", "n_models": len(per_model)}
    return {"verdict": "inconclusive", "n_models": len(per_model),
            "per_model_supported": [supported(m) for m in per_model]}


def run_heads_prereg(spec_path: str = "configs/preregistration/context_heads_v1.yaml") -> Path:
    from ..interpretability.context import context_route_per_sender
    from ..utils import artifact_root
    artifacts = artifact_root()
    spec_file = Path(spec_path)
    rel = spec_file.resolve().relative_to(PROJECT_ROOT).as_posix()
    sha = hashlib.sha256(spec_file.read_bytes()).hexdigest()
    spec, deviation = load_frozen_spec(spec_file)
    frozen = _frozen_hash(spec["id"], rel, artifacts)
    if frozen is None or frozen != sha:
        raise RuntimeError(f"{rel} is not frozen or changed since freezing; refusing to analyse")
    exp = Experiment("prereg", spec["id"], {"run": None, "args": {"spec": rel}, "spec_sha256": sha})
    per_model, rows = [], []
    for run in spec["models"]:
        path = run if Path(run).is_absolute() else str(PROJECT_ROOT / run)
        model = load_run(path)[0]
        r = context_route_per_sender(model, n=512, seed=909)
        per_model.append({"model": Path(run).name, "group": r["group"], "rows": r["rows"]})
        for row in r["rows"]:
            rows.append({"model": Path(run).name, **row})
            print(f"  {Path(run).name} {row['sender']} {row['channel']}: R {row['recovery']:.2f} "
                  f"[{row['recovery_lo']:.2f}, {row['recovery_hi']:.2f}]", flush=True)
    verdict = evaluate_heads_rules(per_model)
    exp.save_table("per_sender", rows)
    exp.save_results({"spec_id": spec["id"], "spec_sha256": sha, "per_model": per_model, "verdict": verdict,
                      "headline": {"H4": verdict["verdict"]}})
    exp.write_report(build_report(f"Preregistered experiment {spec['id']}",
                                  {"spec": f"{rel} (sha256 {sha[:16]}, matches freeze log)",
                                   "models": ", ".join(Path(r).name for r in spec["models"])},
                                  table_md(rows) + f"\n\nVerdict (frozen rule): **{verdict['verdict']}** {verdict}\n",
                                  interpretation=[f"H4: {verdict['verdict']}."],
                                  limitations=["Single-sender patching cannot see redundancy between layer-0 heads.",
                                               "Competition inputs are out of distribution for the trained models."]))
    with open(artifacts / PREREG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": spec["id"], "event": "analyzed", "time": datetime.datetime.now().isoformat(timespec="seconds"),
                            "experiment": exp.id, "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "verdicts": {"H4": verdict["verdict"]}}) + "\n")
    return exp.finalize(f"H4: {verdict['verdict']}")
