"""Reproduction audit and analysis-record comparison."""

import json

import pytest

from transformer_lab.config import ExperimentConfig
from transformer_lab.experiments.registry import compare_results
from transformer_lab.training import train_from_config
from transformer_lab.training.verify import verify_run
from transformer_lab.utils import code_fingerprint


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("rep") / "run"
    cfg = ExperimentConfig.load("configs/experiments/smoke.yaml").with_overrides({"train.steps": 20})
    train_from_config(cfg, run_dir=d, verbose=False, audit_leakage=False)
    return d


def test_verify_reproduces_fresh_run(run_dir):
    r = verify_run(run_dir)
    assert r["status"] == "reproduced", r["problems"]
    assert r["checks"]["code_fingerprint"]["ok"] is True
    assert json.loads((run_dir / "env.json").read_text())["code_fingerprint"] == code_fingerprint()


def test_verify_detects_tampered_record(run_dir, tmp_path):
    import shutil
    d = tmp_path / "tampered"
    shutil.copytree(run_dir, d)
    s = json.loads((d / "summary.json").read_text())
    s["final_eval"]["val_accuracy"] += 0.1
    (d / "summary.json").write_text(json.dumps(s))
    r = verify_run(d)
    assert r["status"] == "MISMATCH" and any("metrics_reproduce" in p for p in r["problems"])


def test_compare_results_detects_differences(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "results.json").write_text(json.dumps({"x": 1.0, "y": [1, 2], "time": 5, "nan": float("nan")}))
    (b / "results.json").write_text(json.dumps({"x": 1.0, "y": [1, 2], "time": 9, "nan": float("nan")}))
    assert compare_results(a, b)["identical_within_tol"]  # volatile 'time' ignored, NaN == NaN
    (b / "results.json").write_text(json.dumps({"x": 1.5, "y": [1, 2]}))
    c = compare_results(a, b)
    assert not c["identical_within_tol"] and c["max_abs_numeric_diff"] == pytest.approx(0.5)


def test_investigation_records_rerun_identically():
    """`itl investigate` records are rerunnable and reproduce bit-for-bit (ground-truth models only)."""
    from transformer_lab.experiments.investigations import run_investigation
    from transformer_lab.experiments.runner import rerun
    exp = run_investigation("context-dose", [], n=64, seed=11)
    again = rerun(str(exp))
    assert compare_results(exp, again)["identical_within_tol"]
    with pytest.raises(ValueError):
        run_investigation("no-such-investigation", [])
