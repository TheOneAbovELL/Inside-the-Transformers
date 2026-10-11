"""Regression fixtures: detect unintended changes to data generation and model numerics.

Fixtures live in ``tests/regression/fixtures.json``. If a change is *intended* (e.g. a new task
parameter that alters sampling), regenerate them with

    python tests/regression/test_regression.py --regenerate

and record why in the commit message. Numerical checks use tolerances (CPU float32 kernels may
differ in the last bits across PyTorch versions); token fingerprints are exact.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

FIXTURES = Path(__file__).with_name("fixtures.json")

TASKS = {
    "repeat_pattern": {},
    "assoc_recall": {"holdout_mod": 5, "max_noise": 3},
    "induction_repeat": {},
    "induction_pair": {"holdout_mod": 5, "n_distractor_pairs": 2, "n_decoys": 2},
}


@torch.no_grad()
def compute() -> dict:
    from transformer_lab.config import ModelConfig
    from transformer_lab.data import make_task
    from transformer_lab.models import Transformer

    out = {"tasks": {}, "model": {}}
    for name, params in TASKS.items():
        t = make_task(name, params)
        b = t.sample(32, np.random.default_rng(2024), "eval")
        out["tasks"][name] = {
            "tokens_sha": hashlib.sha256(b.tokens.numpy().tobytes()).hexdigest()[:20],
            "mask_sha": hashlib.sha256(b.loss_mask.numpy().tobytes()).hexdigest()[:20],
        }
    for pos in ("learned", "alibi", "sinusoidal"):
        torch.manual_seed(0)
        m = Transformer(ModelConfig(vocab_size=20, n_ctx=12, d_model=32, n_heads=4, n_layers=2, positional=pos, seed=7))
        x = torch.arange(24).reshape(2, 12) % 20
        logits = m(x)
        out["model"][pos] = {"logits_sum": float(logits.sum()), "logits_abs_mean": float(logits.abs().mean()),
                             "logit_0_0_0": float(logits[0, 0, 0]), "n_params": m.n_params()}
    return out


def test_regression_fixtures():
    expected = json.loads(FIXTURES.read_text())
    got = compute()
    assert got["tasks"] == expected["tasks"], "task generators changed (regenerate fixtures only if intended)"
    for pos, e in expected["model"].items():
        g = got["model"][pos]
        assert g["n_params"] == e["n_params"]
        for k in ("logits_sum", "logits_abs_mean", "logit_0_0_0"):
            assert abs(g[k] - e[k]) <= 1e-4 * max(1.0, abs(e[k])), (pos, k, g[k], e[k])


if __name__ == "__main__" and "--regenerate" in sys.argv:
    FIXTURES.write_text(json.dumps(compute(), indent=2))
    print(f"wrote {FIXTURES}")
