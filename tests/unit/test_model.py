"""Correctness tests for the core Transformer (Phase 1)."""

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import make_model
from transformer_lab.config import ConfigError, ModelConfig
from transformer_lab.models import LayerNorm, Transformer, lm_loss, per_token_nll
from transformer_lab.training.checkpoint import load_model, save_checkpoint


# ----------------------------------------------------------------------------- shapes
def test_shapes_at_every_hook(model, tokens):
    b, s = tokens.shape
    cfg = model.cfg
    logits, cache = model.run_with_cache(tokens)
    assert logits.shape == (b, s, cfg.vocab_size)
    d, h, e, m = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp
    expected = {"embed": (b, s, d), "pos_embed": (b, s, d), "ln_final_out": (b, s, d), "ln_final_scale": (b, s, 1)}
    for l in range(cfg.n_layers):
        expected.update({
            ("resid_pre", l): (b, s, d), ("resid_mid", l): (b, s, d), ("resid_post", l): (b, s, d),
            ("attn_out", l): (b, s, d), ("mlp_out", l): (b, s, d),
            ("q", l): (b, s, h, e), ("k", l): (b, s, h, e), ("v", l): (b, s, h, e), ("z", l): (b, s, h, e),
            ("attn_scores", l): (b, h, s, s), ("pattern", l): (b, h, s, s),
            ("mlp_pre", l): (b, s, m), ("mlp_post", l): (b, s, m),
            ("ln1_out", l): (b, s, d), ("ln2_out", l): (b, s, d), ("ln1_scale", l): (b, s, 1),
        })
    for key, shape in expected.items():
        assert tuple(cache[key].shape) == shape, key


def test_parameter_count_matches_analytic():
    for kw in [dict(), dict(attn_only=True), dict(norm="post"), dict(norm="none"), dict(positional="sinusoidal"),
               dict(positional="none", n_layers=3)]:
        m = make_model(**kw)
        assert m.n_params() == m.cfg.n_params(), kw


def test_invalid_config_rejected():
    with pytest.raises(ConfigError):
        ModelConfig(vocab_size=10, n_ctx=8, d_model=30, n_heads=4)  # not divisible
    with pytest.raises(ConfigError):
        ModelConfig(vocab_size=10, n_ctx=8, norm="sandwich")
    with pytest.raises(ConfigError):
        ModelConfig.from_dict({"vocab_size": 10, "n_ctx": 8, "d_modle": 32})  # typo must not pass silently


def test_sequence_longer_than_context_rejected(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, model.cfg.n_ctx + 1, dtype=torch.long))


# ------------------------------------------------------------------------- masking
def test_causal_mask_future_tokens_do_not_affect_past(model, tokens):
    model.eval()
    logits = model(tokens)
    t = 5
    altered = tokens.clone()
    altered[:, t + 1:] = (altered[:, t + 1:] + 7) % model.cfg.vocab_size
    logits2 = model(altered)
    torch.testing.assert_close(logits[:, : t + 1], logits2[:, : t + 1], rtol=0, atol=1e-6)
    assert not torch.allclose(logits[:, t + 1:], logits2[:, t + 1:])


def test_attention_pattern_is_causal_and_normalised(model, tokens):
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_pattern"))
    s = tokens.shape[1]
    upper = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
    for l in range(model.cfg.n_layers):
        p = cache["pattern", l]
        assert torch.all(p[..., upper] == 0)
        torch.testing.assert_close(p.sum(-1), torch.ones_like(p.sum(-1)), rtol=0, atol=1e-6)
        assert torch.all(p[:, :, 0, 0] == 1.0)  # first position can only attend to itself


# ---------------------------------------------------- numerical equivalence checks
def test_attention_matches_torch_sdpa(model, tokens):
    """Our explicit per-head attention == PyTorch's fused scaled_dot_product_attention."""
    _, cache = model.run_with_cache(tokens)
    for l in range(model.cfg.n_layers):
        q, k, v = (cache[n, l].transpose(1, 2) for n in ("q", "k", "v"))  # -> [b,h,s,e]
        z_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
        torch.testing.assert_close(cache["z", l], z_ref, rtol=1e-5, atol=1e-6)


def test_per_head_output_equals_concat_then_project(model, tokens):
    """sum_i z_i W_O[i] == concat(z_1..z_h) @ W_O_fused (the textbook formulation)."""
    _, cache = model.run_with_cache(tokens)
    for l in range(model.cfg.n_layers):
        attn = model.blocks[l].attn
        z = cache["z", l]
        b, s, h, e = z.shape
        fused = z.reshape(b, s, h * e) @ attn.W_O.reshape(h * e, -1) + attn.b_O
        torch.testing.assert_close(fused, cache["attn_out", l], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(cache.head_results(l).sum(2) + attn.b_O, cache["attn_out", l], rtol=1e-5, atol=1e-6)


def test_layernorm_matches_torch():
    ln = LayerNorm(16, 1e-5)
    with torch.no_grad():
        ln.w.uniform_(0.5, 1.5)
        ln.b.uniform_(-0.5, 0.5)
    x = torch.randn(4, 7, 16) * 3 + 1
    ref = F.layer_norm(x, (16,), ln.w, ln.b, eps=1e-5)
    torch.testing.assert_close(ln(x), ref, rtol=1e-5, atol=1e-5)


def test_split_qkv_input_is_numerically_equivalent(tokens):
    for norm in ("pre", "post", "none"):
        m = make_model(norm=norm)
        ref = m(tokens)
        m.set_split_qkv_input(True)
        out = m(tokens)
        m.set_split_qkv_input(False)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_scaling_keeps_score_variance_order_one():
    """With unit-variance q,k components, q.k has variance d_head; dividing by sqrt(d_head) restores ~1."""
    g = torch.Generator().manual_seed(0)
    for e in (4, 16, 64, 256):
        q = torch.randn(20000, e, generator=g)
        k = torch.randn(20000, e, generator=g)
        dots = (q * k).sum(-1)
        assert abs(dots.var().item() / e - 1) < 0.1
        assert abs((dots / math.sqrt(e)).var().item() - 1) < 0.1


# ----------------------------------------------------------- variants & stability
@pytest.mark.parametrize("kw", [
    dict(norm="pre"), dict(norm="post"), dict(norm="none"), dict(attn_only=True),
    dict(positional="sinusoidal"), dict(positional="none"), dict(act_fn="relu"), dict(act_fn="gelu_tanh"),
    dict(n_layers=1), dict(d_head=4),
])
def test_variants_forward_backward_finite(kw, tokens):
    m = make_model(**kw)
    logits = m(tokens)
    assert torch.isfinite(logits).all()
    loss = lm_loss(logits, tokens)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in m.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_initial_loss_close_to_uniform(model, tokens):
    """At init (small weights) predictions are ~uniform, so loss ~ ln(V)."""
    loss = lm_loss(model(tokens), tokens).item()
    assert abs(loss - math.log(model.cfg.vocab_size)) < 0.1


def test_gradient_flows_through_attention_and_mlp_of_every_layer(model, tokens):
    lm_loss(model(tokens), tokens).backward()
    for l, block in enumerate(model.blocks):
        for p in (block.attn.W_Q, block.attn.W_K, block.attn.W_V, block.attn.W_O, block.mlp.W_in, block.mlp.W_out):
            assert p.grad.abs().sum() > 0, f"no gradient in layer {l}"
    assert model.W_E.grad.abs().sum() > 0 and model.W_pos.grad.abs().sum() > 0


def test_optimizer_step_changes_params_and_reduces_loss(model, tokens):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    loss0 = lm_loss(model(tokens), tokens)
    loss0.backward()
    opt.step()
    opt.zero_grad()
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert len(changed) == len(before)
    assert lm_loss(model(tokens), tokens).item() < loss0.item()


def test_small_batch_overfit():
    """A model that cannot memorise 4 sequences has a bug somewhere."""
    m = make_model(d_model=64, d_mlp=128)
    g = torch.Generator().manual_seed(3)
    toks = torch.randint(0, 23, (4, 12), generator=g)
    # Distinct first tokens: otherwise two sequences can share a prefix with different
    # continuations, which no model can memorise (this happened with seed 3: rows 0 and 3).
    toks[:, 0] = torch.arange(4)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(300):
        loss = lm_loss(m(toks), toks)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    acc = (m(toks)[:, :-1].argmax(-1) == toks[:, 1:]).float().mean().item()
    assert acc == 1.0


def test_masked_loss_uses_only_selected_positions(model, tokens):
    logits = model(tokens)
    nll = per_token_nll(logits, tokens)
    mask = torch.zeros_like(nll, dtype=torch.bool)
    mask[:, 3] = True
    torch.testing.assert_close(lm_loss(logits, tokens, mask), nll[:, 3].mean())
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
    torch.testing.assert_close(lm_loss(logits, tokens), ref, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        lm_loss(logits, tokens, torch.zeros_like(mask))


# ---------------------------------------------------------- determinism, ckpt, gen
def test_init_is_deterministic_and_seed_dependent(tokens):
    a, b, c = make_model(seed=5), make_model(seed=5), make_model(seed=6)
    torch.testing.assert_close(a(tokens), b(tokens), rtol=0, atol=0)
    assert not torch.allclose(a(tokens), c(tokens))


def test_init_independent_of_global_rng(tokens):
    torch.manual_seed(0)
    a = make_model(seed=1)
    torch.manual_seed(999)
    b = make_model(seed=1)
    torch.testing.assert_close(a(tokens), b(tokens), rtol=0, atol=0)


def test_checkpoint_round_trip(tmp_path, model, tokens):
    model.eval()
    path = save_checkpoint(tmp_path / "m.pt", model, step=7)
    loaded, ckpt = load_model(path)
    assert ckpt["step"] == 7
    assert loaded.cfg == model.cfg
    torch.testing.assert_close(loaded(tokens), model(tokens), rtol=0, atol=0)


def test_generation_shape_validity_and_greedy(model, tokens):
    prompt = tokens[:, :5]
    out = model.generate(prompt, max_new_tokens=6)
    assert out.shape == (3, 11)
    assert torch.equal(out[:, :5], prompt)
    assert out.min() >= 0 and out.max() < model.cfg.vocab_size
    # greedy step == argmax of the logits
    assert torch.equal(out[:, 5], model(prompt)[:, -1].argmax(-1))
    # sampling is reproducible given a generator and respects top_k=1 == greedy
    s1 = model.generate(prompt, 6, temperature=1.0, generator=torch.Generator().manual_seed(0))
    s2 = model.generate(prompt, 6, temperature=1.0, generator=torch.Generator().manual_seed(0))
    assert torch.equal(s1, s2)
    assert torch.equal(model.generate(prompt, 6, temperature=0.7, top_k=1), out)


def test_generation_beyond_context_uses_sliding_window(model):
    out = model.generate(torch.zeros(1, 4, dtype=torch.long), max_new_tokens=model.cfg.n_ctx + 3)
    assert out.shape == (1, 4 + model.cfg.n_ctx + 3)


def test_alibi_attention_matches_sdpa_with_bias(tokens):
    m = make_model(positional="alibi")
    _, cache = m.run_with_cache(tokens)
    s = tokens.shape[1]
    for l in range(m.cfg.n_layers):
        bias = m.blocks[l].attn.alibi_bias[:, :s, :s]
        causal = torch.zeros(s, s).masked_fill(~torch.tril(torch.ones(s, s, dtype=torch.bool)), float("-inf"))
        q, k, v = (cache[n, l].transpose(1, 2) for n in ("q", "k", "v"))
        z_ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bias + causal).transpose(1, 2)
        torch.testing.assert_close(cache["z", l], z_ref, rtol=1e-5, atol=1e-6)
    # ALiBi is recency-biased at init: attention to the previous token exceeds attention to distant ones
    p = cache["pattern", 0][:, 0]  # steepest head
    assert (p[:, 10, 9] > p[:, 10, 1]).all()


def test_sinusoidal_scale_matches_token_embedding_scale():
    m = make_model(positional="sinusoidal", init_std=0.02)
    rms = m.W_pos.pow(2).mean().sqrt().item()
    assert abs(rms - 0.02) < 0.004
