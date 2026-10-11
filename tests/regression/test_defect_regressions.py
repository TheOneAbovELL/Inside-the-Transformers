"""One test per defect found so far, so none can silently return.

| defect (date found)                                   | test                                   |
|-------------------------------------------------------|----------------------------------------|
| sinusoidal table ~50x too large (2026-10-11)          | tests/unit/test_model.py::test_sinusoidal_scale_matches_token_embedding_scale |
| resume skipped the batch of an interrupted step       | tests/integration/test_training.py::test_interrupted_then_resumed_run_matches_uninterrupted |
| sweep seeds used Python's salted str hash             | test_sweep_is_identical_across_processes_with_different_hash_seeds |
| Patcher.run([single spec]) returned a list label      | test_patcher_accepts_single_spec_list |
| resample donors identical in every evaluation chunk   | test_resample_donors_differ_between_chunks |
| leakage audit found memorisable repeat_pattern evals  | test_leakage_audit_detects_small_sequence_spaces |
| analysis of an edited preregistration                 | tests/unit/test_prereg_rules.py::test_analysis_refuses_unfrozen_or_edited_spec |
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from conftest import make_model
from transformer_lab.data import InductionRepeat, make_task
from transformer_lab.interpretability.interventions import Reference

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SWEEP_SNIPPET = """
import json, torch
from transformer_lab.config import ModelConfig
from transformer_lab.models import Transformer
from transformer_lab.data import InductionRepeat
from transformer_lab.evaluation.evaluate import sweep
m = Transformer(ModelConfig(vocab_size=22, n_ctx=24, d_model=16, n_heads=2, n_layers=1, seed=3))
rows = sweep(m, InductionRepeat(n_symbols=20, seq_len=24, min_len=3, max_len=6), "min_len", [3, 4], n=16, seed=5)
print(json.dumps([[r["min_len"], round(r["loss_mean"], 10)] for r in rows]))
"""


def _run(code: str, hashseed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_sweep_is_identical_across_processes_with_different_hash_seeds():
    assert _run(_SWEEP_SNIPPET, "1") == _run(_SWEEP_SNIPPET, "987654")


_TRAIN_SNIPPET = """
import hashlib, sys, tempfile, torch
from transformer_lab.config import ExperimentConfig
from transformer_lab.training import train_from_config
from transformer_lab.training.checkpoint import load_checkpoint
cfg = ExperimentConfig.load("configs/experiments/smoke.yaml").with_overrides({"train.steps": 12, "train.num_threads": 1})
with tempfile.TemporaryDirectory() as d:
    train_from_config(cfg, run_dir=d + "/r", verbose=False, audit_leakage=False)
    sd = load_checkpoint(d + "/r/checkpoints/final.pt")["state_dict"]
    h = hashlib.sha256(b"".join(sd[k].numpy().tobytes() for k in sorted(sd))).hexdigest()
print(h)
"""


@pytest.mark.slow
def test_training_is_bit_identical_across_processes():
    assert _run(_TRAIN_SNIPPET, "0") == _run(_TRAIN_SNIPPET, "4242")


def test_patcher_accepts_single_spec_list():
    from transformer_lab.interpretability.patching import Patcher, PatchSpec
    m = make_model(vocab_size=22, n_ctx=24)
    t = InductionRepeat(n_symbols=20, seq_len=24, min_len=3, max_len=6)
    b = t.sample(8, np.random.default_rng(0))
    p = Patcher(m, b, t.corrupt(b, np.random.default_rng(1), "replace"), min_gap=0.0)
    r = p.run([PatchSpec("only", "blocks.0.hook_resid_pre")])
    assert r.label == "only" and r.recovery == pytest.approx(1.0, abs=1e-5)


def test_resample_donors_differ_between_chunks():
    ref = Reference({"x": torch.arange(100).float().view(100, 1, 1)}, seed=0)
    a = ref.sample("x", 50, call_seed=0).flatten()
    b = ref.sample("x", 50, call_seed=0).flatten()  # second chunk, same call_seed
    assert not torch.equal(a, b)
    ref2 = Reference({"x": torch.arange(100).float().view(100, 1, 1)}, seed=0)
    assert torch.equal(ref2.sample("x", 50, 0).flatten(), a)  # still deterministic


def test_leakage_audit_detects_small_sequence_spaces():
    from transformer_lab.training.trainer import leakage_audit
    t = make_task("repeat_pattern", {"n_symbols": 6, "min_period": 2, "max_period": 2, "seq_len": 10})
    ev = t.sample(64, np.random.default_rng(99))
    r = leakage_audit(t, train_seed=1, steps=20, batch_size=32, *[ev]) if False else \
        leakage_audit(t, 1, 20, 32, ev)
    assert r["set0_overlap"] == 64  # only 30 distinct sequences exist: all eval sequences were seen
