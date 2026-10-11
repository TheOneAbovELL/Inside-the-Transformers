"""Headless end-to-end test of every UI page against real (tiny) backend artifacts.

Uses Streamlit's official AppTest harness: the script runs in-process with the real backend, and
any exception raised by a page fails the test. Nothing here is mocked.
"""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.slow

APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    import os
    from transformer_lab.cli.main import main as cli
    from transformer_lab.experiments import runner
    root = tmp_path_factory.mktemp("ui_lab")
    old = os.environ.get("ITL_ARTIFACT_ROOT")
    os.environ["ITL_ARTIFACT_ROOT"] = str(root)
    rd = root / "runs" / "tiny"
    assert cli(["train", "--config", "configs/experiments/smoke.yaml", "--set", "train.steps=20",
                "--set", "model.positional=alibi", "--run-dir", str(rd), "--quiet"]) == 0
    runner.run_hypothesis_cmd("configs/hypotheses/induction_2l.yaml", [str(rd)])
    runner.run_scores(str(rd), n=32)
    # both preregistration record formats (structure as written by experiments/context_experiment.py)
    from transformer_lab.experiments.registry import Experiment
    e = Experiment("prereg", "context_route_v1", {"run": None, "args": {"spec": "x"}})
    e.save_results({"spec_id": "context_route_v1", "spec_sha256": "0" * 64, "verdict": {"verdict": "supported"},
                    "headline": {"H3b": "supported"}})
    e.save_table("route", [{"model": "m", "R_q": 1.2}])
    e.finalize("H3b: supported")
    e = Experiment("prereg", "context_matching_v1", {"run": None, "args": {"spec": "x"}})
    cp = {"mean": 1.0, "lo": 0.8, "hi": 1.2}
    e.save_results({"spec_id": "context_matching_v1", "spec_sha256": "0" * 64, "metric_valid": True,
                    "verdicts": {"H1": {"verdict": "supported"}},
                    "measurements": {"original": [{"model": "m", "valid": True, "cp": cp, "either_rate": 0.8}]},
                    "headline": {"H1": "supported"}})
    e.finalize("H1: supported")
    yield root
    if old is None:
        os.environ.pop("ITL_ARTIFACT_ROOT", None)
    else:
        os.environ["ITL_ARTIFACT_ROOT"] = old


def _app(lab, monkeypatch):
    monkeypatch.setenv("ITL_ARTIFACT_ROOT", str(lab))
    at = st_testing.AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


PAGES = ["Overview", "Training runs", "Dataset", "Predictions", "Attention", "Activations", "Ablation",
         "Hypotheses & evidence", "Compare", "Reports", "Documentation"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_error(lab, monkeypatch, page):
    at = _app(lab, monkeypatch)
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, (page, at.exception)


def test_patching_page_only_runs_on_button(lab, monkeypatch):
    at = _app(lab, monkeypatch)
    at.sidebar.radio[0].set_value("Activation patching").run()
    assert not at.exception
    assert not any("corrupted=" in m.value for m in at.markdown)  # nothing computed yet
    at.button[0].click().run()
    assert not at.exception
    assert any("corrupted=" in m.value for m in at.markdown)


def test_ablation_page_runs_real_intervention(lab, monkeypatch):
    at = _app(lab, monkeypatch)
    at.sidebar.radio[0].set_value("Ablation").run()
    at.multiselect[0].set_value(["1.0"]).run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.table) == 1  # baseline vs ablated table with CIs


@pytest.mark.parametrize("which", [0, 1, 2])
def test_hypotheses_page_shows_every_record_format(lab, monkeypatch, which):
    at = _app(lab, monkeypatch)
    at.sidebar.radio[0].set_value("Hypotheses & evidence").run()
    at.selectbox[0].select_index(which).run()
    assert not at.exception, at.exception
