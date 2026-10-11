"""Reproduction audit of a finished training run.

Checks, using only what the run directory recorded:

1. ``config.yaml`` rebuilds exactly the model config in ``resolved_model.json`` and the task in
   ``task_spec.json`` (vocabulary, length, parameters).
2. The final checkpoint loads, its step equals ``steps_completed``, and its stored model config
   matches.
3. Re-evaluating the final checkpoint on the validation and test sets — regenerated from
   ``seeds.json`` — reproduces ``summary.json``'s final metrics (tolerance ``tol``).
4. The run's recorded code fingerprint (if any) matches the current code.

A mismatch in (3) means the current code computes a different function for the same weights
(e.g. a non-persistent buffer such as the sinusoidal table changed) or the data generator changed:
the run's recorded numbers can no longer be regenerated, and it is reported as such.
"""

from __future__ import annotations

from pathlib import Path

from ..config import ExperimentConfig
from ..evaluation.evaluate import evaluate_model
from ..utils import code_fingerprint, read_json
from .checkpoint import load_model
from .trainer import build_task_and_model_config, eval_sets


def verify_run(run_dir, tol: float = 1e-5) -> dict:
    d = Path(run_dir)
    out = {"run": d.name, "checks": {}, "problems": []}

    def check(name, ok, detail=""):
        out["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            out["problems"].append(f"{name}: {detail}")

    if not (d / "summary.json").exists():
        check("summary_exists", False, "no summary.json (run incomplete or crashed)")
        out["status"] = "incomplete"
        return out
    summary = read_json(d / "summary.json")
    cfg = ExperimentConfig.load(d / "config.yaml")
    task, model_cfg, seeds = build_task_and_model_config(cfg)
    check("model_config_rebuilds", model_cfg.to_dict() == read_json(d / "resolved_model.json"),
          "config.yaml -> ModelConfig differs from resolved_model.json")
    spec = read_json(d / "task_spec.json")
    # Parameters added to a task after the run was made are acceptable only if they sit at their
    # defaults (the default must reproduce the old behaviour); anything else is a real difference.
    added = {k: v for k, v in task.params.items() if k not in spec["params"]}
    defaults = type(task).defaults
    changed = {k for k in spec["params"] if spec["params"][k] != task.params.get(k)}
    added_non_default = {k for k, v in added.items() if v != defaults.get(k)}
    check("task_spec_rebuilds", spec["vocab_size"] == task.vocab_size and spec["seq_len"] == task.seq_len
          and not changed and not added_non_default,
          f"changed={sorted(changed)} added_non_default={sorted(added_non_default)}")
    if added:
        out["schema_note"] = f"task parameters added since this run (at defaults): {sorted(added)}"
    check("seeds_rebuild", read_json(d / "seeds.json") == seeds, "derived seeds differ from seeds.json")
    model, ck = load_model(d / "checkpoints" / "final.pt")
    check("checkpoint_step", ck.get("step") == summary["steps_completed"],
          f"checkpoint step {ck.get('step')} vs summary {summary['steps_completed']}")
    check("checkpoint_config", ck["model_config"] == model_cfg.to_dict(), "checkpoint model_config differs")
    fe = summary.get("final_eval") or {}
    val, test = eval_sets(task, seeds, cfg.train.eval_size)
    rv, rt = evaluate_model(model, val, n_boot=50), evaluate_model(model, test, n_boot=50)
    diffs = {
        "val_loss": abs(rv["loss"]["mean"] - fe.get("val_loss", float("nan"))),
        "val_accuracy": abs(rv["accuracy"]["mean"] - fe.get("val_accuracy", float("nan"))),
        "test_loss": abs(rt["loss"]["mean"] - fe.get("test_loss", float("nan"))),
        "test_accuracy": abs(rt["accuracy"]["mean"] - fe.get("test_accuracy", float("nan"))),
    }
    out["recomputed"] = {"val_loss": rv["loss"]["mean"], "val_accuracy": rv["accuracy"]["mean"],
                         "test_loss": rt["loss"]["mean"], "test_accuracy": rt["accuracy"]["mean"]}
    out["recorded"] = {k: fe.get(k) for k in diffs}
    check("metrics_reproduce", all(v <= tol for v in diffs.values()),
          "max |recomputed - recorded| = " + f"{max(diffs.values()):.2e}")
    env = read_json(d / "env.json") if (d / "env.json").exists() else {}
    # The fingerprint covers ALL library source, so any later edit (even to the CLI) changes it. A
    # difference is therefore informational — it is a problem only if the metrics stop reproducing,
    # which check (3) already tests.
    if "code_fingerprint" in env:
        same = env["code_fingerprint"] == code_fingerprint()
        out["checks"]["code_fingerprint"] = {
            "ok": same, "detail": "identical source" if same else
            "source changed since this run (informational; metrics are checked separately)"}
        out["code_changed_since_run"] = not same
    else:
        out["checks"]["code_fingerprint"] = {"ok": None, "detail": "not recorded (run predates fingerprinting)"}
        out["code_changed_since_run"] = None
    out["status"] = "reproduced" if not out["problems"] else "MISMATCH"
    return out
