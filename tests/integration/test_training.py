"""End-to-end training pipeline: artifacts, resume determinism, failure handling."""

import json

import pytest
import torch

from transformer_lab.config import ExperimentConfig
from transformer_lab.training import Trainer, TrainingDiverged, load_run, resume_run, train_from_config
from transformer_lab.training.checkpoint import load_checkpoint


def tiny_cfg(**train_overrides) -> ExperimentConfig:
    train = dict(steps=20, batch_size=8, warmup_steps=2, eval_every=10, eval_size=32, log_every=5,
                 checkpoint_every=10, checkpoint_steps=[0, 10], num_threads=1)
    train.update(train_overrides)
    return ExperimentConfig.from_dict({
        "name": "tiny", "seed": 3,
        "model": {"d_model": 16, "n_layers": 2, "n_heads": 2, "attn_only": True},
        "task": {"name": "induction_repeat", "params": {"n_symbols": 20, "seq_len": 20, "min_len": 3, "max_len": 6}},
        "train": train,
    })


def test_run_directory_contents(tmp_path):
    s = train_from_config(tiny_cfg(), run_dir=tmp_path / "run", verbose=False)
    run = tmp_path / "run"
    for f in ("config.yaml", "resolved_model.json", "env.json", "metrics.jsonl", "summary.json", "task_spec.json",
              "seeds.json", "checkpoints/latest.pt", "checkpoints/final.pt", "checkpoints/step_000000.pt",
              "checkpoints/step_000010.pt"):
        assert (run / f).exists(), f
    assert s["status"] == "completed" and s["steps_completed"] == 20
    assert s["initial_eval"]["step"] == 0 and s["final_eval"]["step"] == 20
    assert s["leakage_audit"]["set0_overlap"] == 0
    events = [json.loads(l) for l in open(run / "metrics.jsonl")]
    assert {e["kind"] for e in events} >= {"train", "eval"}
    assert all(torch.isfinite(torch.tensor(e["loss"])) for e in events if e["kind"] == "train")
    # config on disk reproduces the run
    assert ExperimentConfig.load(run / "config.yaml").to_dict() == tiny_cfg().to_dict()
    model, task, summary, cfg = load_run(run)
    assert summary["n_params"] == model.n_params()


def test_same_config_same_result(tmp_path):
    a = train_from_config(tiny_cfg(), run_dir=tmp_path / "a", verbose=False, audit_leakage=False)
    b = train_from_config(tiny_cfg(), run_dir=tmp_path / "b", verbose=False, audit_leakage=False)
    assert a["final_eval"]["val_loss"] == b["final_eval"]["val_loss"]
    sa = load_checkpoint(tmp_path / "a/checkpoints/final.pt")["state_dict"]
    sb = load_checkpoint(tmp_path / "b/checkpoints/final.pt")["state_dict"]
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


def test_interrupted_then_resumed_run_matches_uninterrupted(tmp_path):
    train_from_config(tiny_cfg(), run_dir=tmp_path / "full", verbose=False, audit_leakage=False)

    tr = Trainer(tiny_cfg(), run_dir=tmp_path / "split", verbose=False, audit_leakage=False)
    calls = {"n": 0}

    def interrupt(act, hook):
        if tr.model.training:
            calls["n"] += 1
            if calls["n"] == 14:  # mid-way through step 14 (after its batch was drawn)
                raise KeyboardInterrupt
    with tr.model.hooks([("hook_embed", interrupt)]):
        with pytest.raises(KeyboardInterrupt):
            tr.train()
    assert json.load(open(tmp_path / "split/summary.json"))["status"] == "interrupted"
    assert load_checkpoint(tmp_path / "split/checkpoints/latest.pt")["step"] == 13

    resume_run(tmp_path / "split", verbose=False)
    full = load_checkpoint(tmp_path / "full/checkpoints/final.pt")["state_dict"]
    resumed = load_checkpoint(tmp_path / "split/checkpoints/final.pt")["state_dict"]
    for k in full:
        assert torch.equal(full[k], resumed[k]), k


def test_resume_refuses_changed_config(tmp_path):
    train_from_config(tiny_cfg(steps=10), run_dir=tmp_path / "r", verbose=False, audit_leakage=False)
    tiny_cfg(steps=30).save(tmp_path / "r/config.yaml")
    with pytest.raises(ValueError):
        resume_run(tmp_path / "r", verbose=False)


def test_divergence_is_detected_and_recorded(tmp_path):
    tr = Trainer(tiny_cfg(), run_dir=tmp_path / "d", verbose=False)

    def poison(act, hook):
        return act * float("nan") if tr.model.training else None
    with tr.model.hooks([("blocks.0.hook_attn_out", poison)]):
        with pytest.raises(TrainingDiverged):
            tr.train()
    assert json.load(open(tmp_path / "d/summary.json"))["status"] == "diverged"


def test_early_stopping_on_target_accuracy(tmp_path):
    s = train_from_config(tiny_cfg(target_accuracy=0.0), run_dir=tmp_path / "e", verbose=False, audit_leakage=False)
    assert s["status"] == "early_stopped" and s["steps_completed"] == 10


def test_learning_rate_schedule():
    from transformer_lab.config import TrainConfig
    from transformer_lab.training.trainer import lr_factor
    tc = TrainConfig(steps=100, warmup_steps=10, schedule="cosine", min_lr_ratio=0.1)
    assert lr_factor(0, tc) == pytest.approx(0.1)
    assert lr_factor(9, tc) == pytest.approx(1.0)
    assert lr_factor(10, tc) == pytest.approx(1.0)
    assert lr_factor(100, tc) == pytest.approx(0.1)
    assert lr_factor(55, tc) == pytest.approx(0.55)
