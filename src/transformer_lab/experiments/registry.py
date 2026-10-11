"""Experiment registry: every analysis gets a unique, self-describing local directory.

``artifacts/experiments/<timestamp>_<kind>_<name>/``::

    config.json    the command kind + arguments (enough to re-run: `itl rerun <dir>`)
    env.json       software / hardware snapshot
    results.json   machine-readable results (observations + statistics)
    *.csv          tabular results
    plots/*.png    figures (each titled with run, checkpoint, n, conditions)
    report.md      human-readable report (observations / statistics / interpretation / limitations)

``artifacts/experiments/registry.jsonl`` is an append-only index.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Optional

from ..utils import artifact_root, environment_info, read_json, timestamp_id, write_json


def file_sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


class Experiment:
    def __init__(self, kind: str, name: str, config: dict, root: Optional[Path] = None) -> None:
        root = Path(root) if root else artifact_root() / "experiments"
        base = f"{timestamp_id()}_{kind}_{name}"
        d = root / base
        k = 1
        while d.exists():
            d = root / f"{base}_{k}"
            k += 1
        d.mkdir(parents=True)
        (d / "plots").mkdir()
        self.dir, self.id, self.kind, self.root = d, d.name, kind, root
        self.config = {"kind": kind, "name": name, **config}
        write_json(d / "config.json", self.config)
        write_json(d / "env.json", environment_info())

    def save_results(self, results: dict) -> None:
        write_json(self.dir / "results.json", results)

    def save_table(self, name: str, rows: list[dict]) -> Path:
        path = self.dir / f"{name}.csv"
        keys: list = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
        return path

    def save_plot(self, fig, name: str) -> Path:
        path = self.dir / "plots" / f"{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path

    def write_report(self, markdown: str) -> Path:
        path = self.dir / "report.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    def finalize(self, headline: str = "", **extra) -> Path:
        entry = {"id": self.id, "kind": self.kind, "dir": str(self.dir), "headline": headline,
                 "run": self.config.get("run"), **extra}
        with open(self.root / "registry.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return self.dir


def list_experiments(root: Optional[Path] = None) -> list[dict]:
    root = Path(root) if root else artifact_root() / "experiments"
    reg = root / "registry.jsonl"
    if not reg.exists():
        return []
    out = []
    for line in reg.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            if Path(e["dir"]).exists():
                out.append(e)
    return out


def find_experiment(ref: str) -> Path:
    p = Path(ref)
    if p.exists():
        return p
    for e in list_experiments():
        if e["id"] == ref or e["id"].startswith(ref):
            return Path(e["dir"])
    raise FileNotFoundError(f"No experiment {ref!r}")


_VOLATILE_KEYS = {"timestamp", "time", "dir", "run_dir", "elapsed_s", "process_rss_mb", "system_available_mb"}


def _numeric_leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in _VOLATILE_KEYS:
                yield from _numeric_leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numeric_leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, bool) or obj is None:
        yield prefix, obj
    elif isinstance(obj, (int, float)):
        yield prefix, float(obj)
    elif isinstance(obj, str):
        yield prefix, obj


def compare_results(dir_a, dir_b, tol: float = 1e-6) -> dict:
    """Compare two experiments' ``results.json`` leaf by leaf (numbers within ``tol``; NaN == NaN)."""
    import math
    a = dict(_numeric_leaves(read_json(Path(dir_a) / "results.json")))
    b = dict(_numeric_leaves(read_json(Path(dir_b) / "results.json")))
    diffs, max_abs = [], 0.0
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(va, float) and isinstance(vb, float):
            if math.isnan(va) and math.isnan(vb):
                continue
            d = abs(va - vb)
            max_abs = max(max_abs, d)
            if d > tol:
                diffs.append((k, va, vb))
        elif va != vb:
            diffs.append((k, va, vb))
    return {"n_leaves": len(set(a) | set(b)), "n_differences": len(diffs), "max_abs_numeric_diff": max_abs,
            "differences": diffs[:20], "identical_within_tol": not diffs}


def list_runs(root: Optional[Path] = None) -> list[dict]:
    root = Path(root) if root else artifact_root() / "runs"
    out = []
    for d in sorted(root.glob("*")) if root.exists() else []:
        if not (d / "config.yaml").exists():
            continue
        s = read_json(d / "summary.json") if (d / "summary.json").exists() else {}
        fe = s.get("final_eval") or {}
        mc = s.get("model_config") or (read_json(d / "resolved_model.json") if (d / "resolved_model.json").exists() else {})
        out.append({
            "run_dir": str(d), "name": d.name, "status": s.get("status", "no summary (running or crashed)"),
            "steps": s.get("steps_completed"), "elapsed_s": s.get("elapsed_s"),
            "val_accuracy": None if not fe else round(fe["val_accuracy"], 4),
            "test_accuracy": None if not fe else round(fe["test_accuracy"], 4),
            "val_loss": None if not fe else round(fe["val_loss"], 4),
            "task": (s.get("task") or {}).get("name"), "positional": mc.get("positional"),
            "n_layers": mc.get("n_layers"), "attn_only": mc.get("attn_only"),
            "seed": (s.get("experiment") or {}).get("seed"),
        })
    return out


def compare(paths: list, out: Optional[str] = None) -> list[dict]:
    """Side-by-side table of runs (summary.json) and/or experiments (results.json headline numbers)."""
    rows = []
    for ref in paths:
        p = Path(ref)
        if (p / "summary.json").exists():  # a training run
            s = read_json(p / "summary.json")
            fe = s.get("final_eval") or {}
            rows.append({"item": p.name, "type": "run", "status": s["status"], "seed": s["experiment"]["seed"],
                         "positional": s["model_config"]["positional"], "layers": s["model_config"]["n_layers"],
                         "val_acc": fe.get("val_accuracy"), "test_acc": fe.get("test_accuracy"),
                         "val_loss": fe.get("val_loss"), "steps": s["steps_completed"], "time_s": s["elapsed_s"]})
            continue
        d = find_experiment(ref)
        res = read_json(d / "results.json")
        cfg = read_json(d / "config.json")
        row = {"item": d.name, "type": cfg["kind"], "run": Path(cfg.get("run", "") or "").name}
        row.update(res.get("headline", {}))
        rows.append(row)
    if out:
        from ..evaluation.evaluate import write_csv
        write_csv(rows, out)
    return rows
