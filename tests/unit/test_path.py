"""Path interventions on a trained-shape model with MLPs (heads and MLPs as senders)."""

import numpy as np
import pytest
import torch

from conftest import make_model
from transformer_lab.data import InductionRepeat
from transformer_lab.interpretability.path import mean_head_results, path_effect, path_hooks, split_inputs

TASK = InductionRepeat(n_symbols=12, seq_len=16, min_len=3, max_len=5)


@pytest.fixture
def setup():
    m = make_model(vocab_size=TASK.vocab_size, n_ctx=16, attn_only=False)
    b = TASK.sample(16, np.random.default_rng(0))
    return m, b


@pytest.mark.parametrize("sender", [[(0, "mlp")], [(0, 1)], [(0, 0), (0, 3), (0, "mlp")]])
def test_identity_replacement_is_a_no_op(setup, sender):
    """Replacing a sender's contribution by itself (single example) must leave the output unchanged."""
    m, b = setup
    one = b.subset([0])
    rep = mean_head_results(m, one, sender)  # mean over ONE example == its own activation
    out = path_effect(m, one, sender, [(1, 2)], "k", replacement=rep, metric="logprob")
    np.testing.assert_allclose(out["treated"], out["baseline"], atol=1e-5)


def test_mlp_sender_changes_only_the_receivers_pattern(setup):
    m, b = setup
    _, clean = m.run_with_cache(b.tokens, names_filter=["blocks.1.attn.hook_pattern"])
    with split_inputs(m):
        hooks = path_hooks(m, [(0, "mlp")], [(1, 2)], "k", replacement=None)
        _, cache = m.run_with_cache(b.tokens, names_filter=["blocks.1.attn.hook_pattern"], fwd_hooks=hooks)
    p0, p1 = clean["pattern", 1], cache["pattern", 1]
    assert not torch.allclose(p0[:, 2], p1[:, 2])
    torch.testing.assert_close(p0[:, [0, 1, 3]], p1[:, [0, 1, 3]], rtol=1e-5, atol=1e-6)


def test_split_mode_is_restored_after_path_effect(setup):
    m, b = setup
    path_effect(m, b, [(0, 0)], [(1, 0)], "q")
    assert not any(blk.split_qkv_input for blk in m.blocks)


def test_invalid_path_rejected(setup):
    m, _ = setup
    with pytest.raises(ValueError):
        path_hooks(m, [(1, 0)], [(1, 2)], "k")
    with pytest.raises(ValueError):
        path_hooks(m, [(0, 0)], [(1, 2)], "x")
