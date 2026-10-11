"""Instrumentation must observe without perturbing, and intervene exactly where asked."""

import pytest
import torch

from conftest import make_model


def test_caching_does_not_change_outputs(model, tokens):
    ref = model(tokens)
    logits, cache = model.run_with_cache(tokens)
    torch.testing.assert_close(logits, ref, rtol=0, atol=0)
    # every hook point fires except the per-head q/k/v input copies, which exist only in split mode
    split_only = {n for n in model.hook_names() if n.endswith(("hook_q_input", "hook_k_input", "hook_v_input"))}
    assert set(cache.keys()) == set(model.hook_names()) - split_only
    model.set_split_qkv_input(True)
    _, cache_split = model.run_with_cache(tokens)
    model.set_split_qkv_input(False)
    assert set(cache_split.keys()) == set(model.hook_names())


def test_hooks_are_removed_after_context_and_after_exceptions(model, tokens):
    def boom(act, hook):
        raise RuntimeError("intentional")

    with pytest.raises(RuntimeError):
        model.run_with_hooks(tokens, fwd_hooks=[("blocks.0.hook_resid_post", boom)])
    assert not any(hp.has_hooks for hp in model.hook_dict.values())
    model(tokens)  # would raise if the hook leaked


def test_names_filter_limits_cache(model, tokens):
    _, cache = model.run_with_cache(tokens, names_filter=["blocks.1.attn.hook_pattern"])
    assert list(cache.keys()) == ["blocks.1.attn.hook_pattern"]
    with pytest.raises(KeyError):
        cache["pattern", 0]


def test_cached_activations_reproduce_forward_computation(model, tokens):
    """The residual stream identities hold exactly on recorded activations."""
    logits, c = model.run_with_cache(tokens)
    torch.testing.assert_close(c["resid_pre", 0], c["embed"] + c["pos_embed"])
    for l in range(model.cfg.n_layers):
        torch.testing.assert_close(c["resid_mid", l], c["resid_pre", l] + c["attn_out", l])
        torch.testing.assert_close(c["resid_post", l], c["resid_mid", l] + c["mlp_out", l])
        if l + 1 < model.cfg.n_layers:
            torch.testing.assert_close(c["resid_pre", l + 1], c["resid_post", l])
        mlp = model.blocks[l].mlp
        torch.testing.assert_close(c["mlp_post", l] @ mlp.W_out + mlp.b_out, c["mlp_out", l])
        # recompute attention pattern from cached q, k
        q, k = c["q", l], c["k", l]
        scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / model.cfg.d_head ** 0.5
        s = tokens.shape[1]
        scores = scores.masked_fill(~torch.tril(torch.ones(s, s, dtype=torch.bool)), float("-inf"))
        torch.testing.assert_close(torch.softmax(scores, -1), c["pattern", l])
    final = c["ln_final_out"] @ model.W_U + model.b_U
    torch.testing.assert_close(final, logits)


def test_residual_decomposition_sums_to_stream(model, tokens):
    _, c = model.run_with_cache(tokens)
    labels, parts = c.decompose_resid()
    assert len(labels) == parts.shape[0]
    torch.testing.assert_close(parts.sum(0), c["resid_post", model.cfg.n_layers - 1], rtol=1e-5, atol=1e-5)
    labels1, parts1 = c.decompose_resid(layer=1)
    torch.testing.assert_close(parts1.sum(0), c["resid_pre", 1], rtol=1e-5, atol=1e-5)


def test_zeroing_one_head_only_removes_that_head(model, tokens):
    """Zero-ablating z of head (1,2) changes attn_out by exactly that head's result vector."""
    _, clean = model.run_with_cache(tokens)

    def zero_head(z, hook):
        z = z.clone()
        z[:, :, 2] = 0
        return z

    _, abl = model.run_with_cache(tokens, fwd_hooks=[("blocks.1.attn.hook_z", zero_head)])
    # layer 0 untouched
    torch.testing.assert_close(abl["attn_out", 0], clean["attn_out", 0], rtol=0, atol=0)
    expected = clean["attn_out", 1] - clean.head_results(1)[:, :, 2]
    torch.testing.assert_close(abl["attn_out", 1], expected, rtol=1e-5, atol=1e-6)
    # other heads' z unchanged (layer-1 heads read resid_pre 1, which is unchanged)
    torch.testing.assert_close(abl["z", 1][:, :, [0, 1, 3]], clean["z", 1][:, :, [0, 1, 3]], rtol=0, atol=0)


def test_hook_shape_mismatch_is_an_error(model, tokens):
    with pytest.raises(RuntimeError):
        model.run_with_hooks(tokens, fwd_hooks=[("blocks.0.hook_attn_out", lambda a, h: a[:, :1])])


def test_predicate_hooks_attach_to_all_matching_points(model, tokens):
    seen = []
    model.run_with_hooks(tokens, fwd_hooks=[(lambda n: n.endswith("hook_z"), lambda a, h: seen.append(h.name))])
    assert seen == ["blocks.0.attn.hook_z", "blocks.1.attn.hook_z"]


def test_post_ln_decomposition_refused():
    m = make_model(norm="post")
    _, c = m.run_with_cache(torch.zeros(1, 4, dtype=torch.long))
    with pytest.raises(NotImplementedError):
        c.decompose_resid()
