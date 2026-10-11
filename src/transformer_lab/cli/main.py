"""Command-line interface: ``itl <command>`` (or ``python -m transformer_lab.cli.main <command>``).

Training, analysis and reporting all run from here without the UI. Every analysis command creates
an experiment directory under ``artifacts/experiments/`` that can be re-run with ``itl rerun``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_run_args(p, n_default=512):
    p.add_argument("--run", required=True, help="training run directory (artifacts/runs/...)")
    p.add_argument("--checkpoint", default="final", help="checkpoint name in <run>/checkpoints (default: final)")
    p.add_argument("--n", type=int, default=n_default, help="number of evaluation examples")
    p.add_argument("--seed", type=int, default=1234, help="seed for the evaluation examples")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="itl", description="Inside the Transformer — interpretability lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="show environment, hardware and installation status")
    p = sub.add_parser("smoke", help="validate the installation with a minimal end-to-end experiment")
    p.add_argument("--keep", action="store_true", help="keep artifacts (default: temporary directory)")

    p = sub.add_parser("train", help="train a model from a config (or resume a run)")
    p.add_argument("--config", help="experiment YAML")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. train.steps=100")
    p.add_argument("--resume", metavar="RUN_DIR", help="resume an interrupted run")
    p.add_argument("--run-dir", help="explicit output directory")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-leakage-audit", action="store_true")

    sub.add_parser("runs", help="list training runs")
    sub.add_parser("tasks", help="list synthetic tasks, their parameters and defaults")
    sub.add_parser("configs", help="list experiment configs, hypotheses and preregistrations")
    p = sub.add_parser("verify", help="reproduction audit: re-evaluate a run's final checkpoint vs its records")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--all", action="store_true")
    p = sub.add_parser("investigate", help="run a named, rerunnable exploratory investigation")
    p.add_argument("name", help="context-heads | competition-attention | context-dose | context-route | mechanism-type")
    p.add_argument("--run", action="append", default=[], help="repeatable")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-ground-truth", action="store_true", help="omit the hand-wired reference models")
    p = sub.add_parser("prereg", help="analyse a FROZEN preregistered experiment (refuses edited specs)")
    p.add_argument("--spec", default="configs/preregistration/context_matching_v1.yaml")
    sub.add_parser("experiments", help="list analysis experiments (registry)")

    p = sub.add_parser("eval", help="evaluate a run: held-out metrics, baselines, optional all checkpoints")
    _add_run_args(p)
    p.add_argument("--all-checkpoints", action="store_true")
    p.add_argument("--unseen-only", action="store_true", help="also score only sequences never seen in training")

    p = sub.add_parser("inspect", help="inspect one example (tokens, predictions, attention) + plots")
    _add_run_args(p, n_default=64)
    p.add_argument("--example", type=int, default=0)

    p = sub.add_parser("trace", help="trace one prediction: DLA, logit lens, top attention sources")
    _add_run_args(p, n_default=64)
    p.add_argument("--example", type=int, default=0)
    p.add_argument("--position", type=int, default=None)

    p = sub.add_parser("scores", help="observational head scores: attention patterns, OV copying, composition")
    _add_run_args(p)

    p = sub.add_parser("ablate", help="ablate every head / MLP and measure effects (with CIs)")
    _add_run_args(p)
    p.add_argument("--method", default="mean", choices=["zero", "mean", "resample"])
    p.add_argument("--neurons-top-k", type=int, default=0)

    p = sub.add_parser("patch", help="activation patching (clean -> corrupted) over heads and regions")
    _add_run_args(p)
    p.add_argument("--corruption", default=None, help="task corruption kind (default: task default)")
    p.add_argument("--metric", default="logprob", choices=["logprob", "logit_diff", "prob", "accuracy"])
    p.add_argument("--direction", default="denoise", choices=["denoise", "noise"])
    p.add_argument("--min-gap", type=float, default=0.25)

    p = sub.add_parser("hypothesis", help="test a pre-registered circuit hypothesis on one or more runs")
    p.add_argument("--spec", required=True)
    p.add_argument("--run", action="append", required=True, help="repeatable: test on several seeds/models")
    p.add_argument("--checkpoint", default="final")

    p = sub.add_parser("formation", help="track mechanism formation across saved checkpoints")
    _add_run_args(p)
    p.add_argument("--ablate", action="store_true", help="also ablate the top induction head at each checkpoint")

    p = sub.add_parser("sweep", help="evaluate across task variants (position, distance, distractors, length)")
    _add_run_args(p, n_default=256)
    p.add_argument("--param", required=True)
    p.add_argument("--values", required=True, help="comma list or start:stop[:step]")
    p.add_argument("--task-set", action="append", default=[], metavar="KEY=VALUE",
                   help="fixed task-parameter overrides for the sweep (must keep the vocabulary)")
    p.add_argument("--split", default="train", choices=["train", "eval"])
    p.add_argument("--task-name", default=None, help="evaluate zero-shot on a different task family (same vocab)")

    p = sub.add_parser("compare", help="compare experiments or runs side by side")
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", help="write the comparison table to this CSV")

    p = sub.add_parser("rerun", help="re-run an analysis experiment from its recorded config")
    p.add_argument("experiment")

    p = sub.add_parser("ui", help="launch the local Streamlit interface")
    p.add_argument("--port", type=int, default=8501)
    return ap


def _parse_values(text: str):
    if ":" in text:
        parts = [int(x) for x in text.split(":")]
        return list(range(*parts))
    out = []
    for x in text.split(","):
        x = x.strip()
        try:
            out.append(int(x))
        except ValueError:
            out.append(float(x))
    return out


def main(argv=None) -> int:
    """Entry point. Expected user errors (missing files, invalid configs or values) print a one-line
    message and return 2; set ITL_DEBUG=1 to see the full traceback."""
    import os
    from ..config import ConfigError
    try:
        return _main(argv)
    except (FileNotFoundError, ConfigError, ValueError, KeyError) as e:
        if os.environ.get("ITL_DEBUG"):
            raise
        hint = ""
        if isinstance(e, FileNotFoundError):
            hint = " (list training runs with `itl runs`, experiments with `itl experiments`)"
        print(f"itl: error: {type(e).__name__}: {e}{hint}", file=sys.stderr)
        return 2


def _main(argv=None) -> int:
    # Windows consoles default to cp1252, which cannot print the Greek letters / dashes used in
    # reports; never let console encoding crash an experiment that already succeeded.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    cmd = args.cmd

    if cmd == "env":
        from ..experiments.runner import env_report
        print(json.dumps(env_report(), indent=2))
        return 0

    if cmd == "smoke":
        from ..experiments.runner import smoke
        ok = smoke(keep=args.keep)
        return 0 if ok else 1

    if cmd == "train":
        from ..config import ExperimentConfig, parse_override
        from ..training import resume_run, train_from_config
        if args.resume:
            s = resume_run(args.resume, verbose=not args.quiet)
        else:
            if not args.config:
                print("train needs --config or --resume", file=sys.stderr)
                return 2
            cfg = ExperimentConfig.load(args.config)
            if args.set:
                cfg = cfg.with_overrides(dict(parse_override(x) for x in args.set))
            s = train_from_config(cfg, run_dir=args.run_dir, verbose=not args.quiet,
                                  audit_leakage=not args.no_leakage_audit)
        fe = s.get("final_eval") or {}
        print(f"[{s['status']}] {s['run_dir']}  steps={s['steps_completed']}  {s['elapsed_s']}s  "
              f"val_acc={fe.get('val_accuracy', float('nan')):.3f}  test_acc={fe.get('test_accuracy', float('nan')):.3f}")
        return 0 if s["status"] in ("completed", "early_stopped") else 1

    if cmd == "tasks":
        from ..data.tasks import TASKS
        for name, cls in TASKS.items():
            print(f"{name}\n    {cls.description}\n    defaults: {cls.defaults}")
            if cls.corruption_kinds:
                print(f"    corruptions: {list(cls.corruption_kinds)}")
        return 0

    if cmd == "configs":
        from ..utils import PROJECT_ROOT
        for sub_dir in ("experiments", "hypotheses", "preregistration"):
            print(f"configs/{sub_dir}/")
            for f in sorted((PROJECT_ROOT / "configs" / sub_dir).glob("*.yaml")):
                first = next((l[2:].strip() for l in f.read_text(encoding="utf-8").splitlines() if l.startswith("# ")), "")
                print(f"    {f.name:40s} {first[:90]}")
        return 0

    if cmd == "verify":
        from ..experiments.registry import list_runs
        from ..training.verify import verify_run
        targets = [r["run_dir"] for r in list_runs()] if args.all else [args.run]
        bad = 0
        for t in targets:
            r = verify_run(t)
            bad += r["status"] == "MISMATCH"
            note = "  (source changed since run)" if r.get("code_changed_since_run") else ""
            print(f"{r['status']:11s} {r['run']}{note}" + (f"  <- {'; '.join(r['problems'])}" if r["problems"] else ""))
        return 1 if bad else 0

    if cmd == "investigate":
        from ..experiments.investigations import run_investigation
        exp = run_investigation(args.name, args.run, n=args.n, seed=args.seed, ground_truth=not args.no_ground_truth)
        print(f"experiment written to {exp}")
        print((Path(exp) / "report.md").read_text(encoding="utf-8")[:6000])
        return 0

    if cmd == "prereg":
        import re
        from ..experiments.context_experiment import run_context_experiment, run_route_prereg
        m = re.search(r"^id:\s*(\S+)", Path(args.spec).read_text(encoding="utf-8"), re.M)
        spec_id = m.group(1) if m else None
        from ..experiments.context_experiment import run_heads_prereg
        runners = {"context_matching_v1": run_context_experiment, "context_route_v1": run_route_prereg,
                   "context_heads_v1": run_heads_prereg}
        if spec_id not in runners:
            raise ValueError(f"no analysis registered for preregistration {spec_id!r}; known: {sorted(runners)}")
        exp = runners[spec_id](args.spec)
        print(f"experiment written to {exp}")
        print((Path(exp) / "report.md").read_text(encoding="utf-8")[:6000])
        return 0

    if cmd == "runs":
        from ..experiments.registry import list_runs
        for r in list_runs():
            print(f"{r['run_dir']}\n    status={r['status']} steps={r['steps']} val_acc={r['val_accuracy']} "
                  f"test_acc={r['test_accuracy']} task={r['task']} pos={r['positional']} layers={r['n_layers']}")
        return 0

    if cmd == "experiments":
        from ..experiments.registry import list_experiments
        for e in list_experiments():
            print(f"{e['id']}  kind={e['kind']}  run={e.get('run', '')}  {e.get('headline', '')}")
        return 0

    if cmd == "compare":
        from ..experiments.registry import compare
        rows = compare(args.paths, out=args.out)
        from ..reporting.report import table_md
        print(table_md(rows))
        return 0

    if cmd == "rerun":
        from ..experiments.runner import rerun
        from ..experiments.registry import compare_results, find_experiment
        original = find_experiment(args.experiment)
        exp = rerun(args.experiment)
        cmp = compare_results(original, exp)
        print(f"re-ran -> {exp}")
        print(f"compared with original: {cmp['n_leaves']} values, {cmp['n_differences']} differ "
              f"(max |diff| {cmp['max_abs_numeric_diff']:.2e}) -> "
              f"{'IDENTICAL' if cmp['identical_within_tol'] else 'DIFFERENT'}")
        for k, a, b in cmp["differences"][:10]:
            print(f"  {k}: {a} -> {b}")
        return 0 if cmp["identical_within_tol"] else 1

    if cmd == "ui":
        import subprocess
        app = Path(__file__).resolve().parents[3] / "app" / "streamlit_app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
                                "--server.address", "localhost", "--browser.gatherUsageStats", "false"])

    # ---------------------------------------------------------------- analysis commands
    from ..experiments import runner
    kwargs = {k: v for k, v in vars(args).items() if k != "cmd"}
    if cmd == "sweep":
        kwargs["values"] = _parse_values(args.values)
    fn = {"eval": runner.run_eval, "inspect": runner.run_inspect, "trace": runner.run_trace,
          "scores": runner.run_scores, "ablate": runner.run_ablation, "patch": runner.run_patching,
          "hypothesis": runner.run_hypothesis_cmd, "formation": runner.run_formation, "sweep": runner.run_sweep}[cmd]
    exp_dir = fn(**kwargs)
    print(f"experiment written to {exp_dir}")
    report = Path(exp_dir) / "report.md"
    if report.exists():
        print(report.read_text(encoding="utf-8")[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
