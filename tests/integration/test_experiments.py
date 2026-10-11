"""Every analysis command runs end-to-end on a tiny trained model and writes a complete experiment."""

import json
from pathlib import Path

import pytest

from transformer_lab.cli.main import main as cli
from transformer_lab.experiments import runner
from transformer_lab.experiments.registry import compare, list_experiments, list_runs

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    import os
    root = tmp_path_factory.mktemp("lab")
    os.environ["ITL_ARTIFACT_ROOT"] = str(root)
    rd = root / "runs" / "tiny"
    rc = cli(["train", "--config", "configs/experiments/smoke.yaml", "--set", "train.steps=30",
              "--set", "model.positional=alibi", "--run-dir", str(rd), "--quiet"])
    assert rc == 0
    return str(rd)


def _check(exp_dir):
    d = Path(exp_dir)
    for f in ("config.json", "env.json", "results.json", "report.md"):
        assert (d / f).exists(), f
    res = json.loads((d / "results.json").read_text())
    assert "headline" in res
    md = (d / "report.md").read_text(encoding="utf-8")
    for section in ("## Setup", "## Observations", "## Interpretation", "## Limitations"):
        assert section in md
    return res


def test_all_analysis_commands(run_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("ITL_ARTIFACT_ROOT", str(Path(run_dir).parents[1]))
    _check(runner.run_eval(run_dir, n=64, all_checkpoints=True))
    _check(runner.run_inspect(run_dir, n=8, example=1))
    _check(runner.run_trace(run_dir, n=8, example=0))
    _check(runner.run_scores(run_dir, n=64))
    _check(runner.run_ablation(run_dir, n=64, method="zero"))
    _check(runner.run_patching(run_dir, n=64, corruption="shuffle", metric="logit_diff"))
    _check(runner.run_formation(run_dir, n=32, ablate=True))
    _check(runner.run_sweep(run_dir, param="min_len", values=[4, 5], n=32))
    hyp = _check(runner.run_hypothesis_cmd("configs/hypotheses/induction_2l.yaml", [run_dir]))
    assert sum(hyp["verdict_counts"].values()) == 8

    exps = list_experiments()
    assert {e["kind"] for e in exps} >= {"eval", "inspect", "trace", "scores", "ablation", "patching", "formation",
                                         "sweep", "hypothesis"}
    # re-running from the recorded config reproduces the numbers exactly
    abl = next(e for e in exps if e["kind"] == "ablation")
    again = runner.rerun(abl["dir"])
    a = json.loads((Path(abl["dir"]) / "results.json").read_text())["rows"]
    b = json.loads((Path(again) / "results.json").read_text())["rows"]
    assert a == b
    rows = compare([run_dir, abl["dir"]])
    assert rows[0]["type"] == "run" and rows[1]["type"] == "ablation"
    assert list_runs()[0]["status"] == "completed"


def test_cli_smoke_command():
    assert cli(["smoke"]) == 0


def test_cli_rejects_bad_override_and_missing_run(capsys):
    assert cli(["train", "--config", "configs/experiments/smoke.yaml", "--set", "model.d_modle=3", "--quiet"]) == 2
    assert "ConfigError" in capsys.readouterr().err
    assert cli(["eval", "--run", "artifacts/runs/does_not_exist"]) == 2
    assert "itl runs" in capsys.readouterr().err


def test_cli_listing_commands(capsys):
    assert cli(["tasks"]) == 0 and "context_competition" in capsys.readouterr().out
    assert cli(["configs"]) == 0 and "context_matching_v1.yaml" in capsys.readouterr().out
