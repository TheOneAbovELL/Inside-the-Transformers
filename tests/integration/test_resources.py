"""Resource checks: the minimal configuration must be cheap on a laptop CPU."""

import time

import pytest

from transformer_lab.config import ExperimentConfig
from transformer_lab.training import train_from_config

psutil = pytest.importorskip("psutil")


def test_smoke_config_runs_within_cpu_and_memory_budget(tmp_path):
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    peak = rss0
    cfg = ExperimentConfig.load("configs/experiments/smoke.yaml")
    t0 = time.perf_counter()
    s = train_from_config(cfg, run_dir=tmp_path / "r", verbose=False)
    elapsed = time.perf_counter() - t0
    peak = max(peak, proc.memory_info().rss)
    assert s["status"] == "completed"
    # generous bounds: the reference laptop does this in a few seconds even under heavy load
    assert elapsed < 120, f"smoke training took {elapsed:.1f}s"
    assert (peak - rss0) / 2**20 < 500, f"RSS grew by {(peak - rss0) / 2**20:.0f} MB"


def test_activation_cache_size_is_bounded():
    """Caching everything for a principal-size model on a 64-example batch stays well under 100 MB."""
    import numpy as np
    from transformer_lab.config import ModelConfig
    from transformer_lab.data import make_task
    from transformer_lab.models import Transformer

    task = make_task("induction_repeat")
    m = Transformer(ModelConfig(vocab_size=task.vocab_size, n_ctx=task.seq_len, d_model=64, n_layers=2, n_heads=4,
                                attn_only=True, positional="alibi"))
    b = task.sample(64, np.random.default_rng(0))
    _, cache = m.run_with_cache(b.tokens)
    assert cache.nbytes() < 100 * 2**20
    _, small = m.run_with_cache(b.tokens, names_filter=lambda n: n.endswith("hook_pattern"))
    assert small.nbytes() < cache.nbytes() / 3
