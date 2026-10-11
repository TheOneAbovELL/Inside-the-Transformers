# Mathematics guide — from equations to code

Every equation below is implemented in `src/transformer_lab/models/transformer.py` unless stated
otherwise; the right-hand column of each section names the exact code. Shapes use
`b` batch, `s` sequence (`q` query / `k` key positions), `d` = `d_model`, `h` heads,
`e` = `d_head`, `m` = `d_mlp`, `V` vocabulary.

---

## 1. Embedding the input

Tokens `x_1..x_s` (integers) become vectors in the **residual stream**:

$$ r^{(0)}_t = W_E[x_t] + p_t \qquad r^{(0)} \in \mathbb{R}^{s\times d} $$

| positional scheme (`ModelConfig.positional`) | `p_t` |
|---|---|
| `learned` | a trained row `W_pos[t]` |
| `sinusoidal` | fixed: `PE[t,2i] = sin(t / 10000^{2i/d})`, `PE[t,2i+1] = cos(...)` (Vaswani et al., 2017), **scaled** to the token-embedding init std — see §8 |
| `alibi` | `p_t = 0`; position enters only as a bias on attention scores (§2) |
| `none` | `p_t = 0` (only the causal mask carries order information) |

Code: `Transformer.embed` — hook points `hook_embed` (the `W_E[x_t]` term) and `hook_pos_embed`.

## 2. Causal scaled dot-product attention

For head `i` with input `X` (rows = positions):

$$ Q_i = XW_Q^{i} + b_Q^{i},\quad K_i = XW_K^{i}+b_K^{i},\quad V_i = XW_V^{i}+b_V^{i}\qquad (s\times e) $$

$$ A_i = \operatorname{softmax}\!\Big(\frac{Q_iK_i^\top}{\sqrt{e}} + B_i + M\Big) \qquad (s\times s),\qquad
   M_{qk} = \begin{cases}0 & k\le q\\ -\infty & k>q\end{cases} $$

$$ z_i = A_i V_i \quad (s\times e), \qquad \text{Attn}(X) = \sum_{i=1}^{h} z_i W_O^{i} + b_O \quad (s\times d) $$

`B_i` is the ALiBi bias `B_{i,qk} = -m_i (q-k)` with fixed slopes `m_i = 2^{-8(i+1)/h}` when
`positional: alibi`, else 0.

**Shapes through the computation** (code `Attention.forward`):

| step | code | shape |
|---|---|---|
| input | `x_q, x_k, x_v` | `[b,s,d]` (or `[b,s,h,d]` in split mode) |
| projections | `hook_q/k/v` | `[b,s,h,e]` |
| scores `QKᵀ/√e (+B)` masked | `hook_attn_scores` | `[b,h,q,k]` |
| softmax over keys | `hook_pattern` | `[b,h,q,k]`, rows sum to 1 |
| weighted values | `hook_z` | `[b,s,h,e]` |
| output | `Σ_i z_i W_O^i + b_O` | `[b,s,d]` |

**Why divide by √e.** If the components of `q` and `k` are independent with mean 0 and variance 1,
then `q·k = Σ_j q_j k_j` has variance `e`. Without scaling, scores grow like `√e`, the softmax
saturates (one weight ≈ 1, gradients ≈ 0). Dividing by `√e` keeps score variance ≈ 1 regardless
of head size. This is checked numerically in `tests/unit/test_model.py::test_scaling_keeps_score_variance_order_one`.

**Why mask the future.** The model is trained to predict token `t+1` from tokens `≤ t` at *every*
position in parallel (§6). If position `t` could attend to `t+1`, it could simply read the answer
and the learned function would be useless for generation. Adding `-∞` before the softmax gives those
positions exactly zero weight. Tests: `test_causal_mask_future_tokens_do_not_affect_past` (changing
future tokens leaves earlier logits bit-identical) and `test_attention_pattern_is_causal_and_normalised`.

**Multi-head = sum of per-head writes.** The textbook form concatenates heads and multiplies by
one `W_O ∈ R^{(h·e)×d}`. Splitting `W_O` into `h` row-blocks `W_O^i` shows
`concat(z_1..z_h) W_O = Σ_i z_i W_O^i`: each head independently **adds a vector to the residual
stream**. This is why replacing `z_i` removes exactly head `i`'s contribution (test
`test_zeroing_one_head_only_removes_that_head`). The implementation stores `W_Q, W_K, W_V: [h,d,e]`
and `W_O: [h,e,d]` and computes the projection as one reshaped matmul for speed; equivalence with
PyTorch's fused `scaled_dot_product_attention` is tested (`test_attention_matches_torch_sdpa`,
`test_alibi_attention_matches_sdpa_with_bias`).

**QK and OV circuits** (Elhage et al., 2021). Writing `W_{QK}^i = W_Q^i W_K^{i\top}` (`d×d`) and
`W_{OV}^i = W_V^i W_O^i` (`d×d`): the score between residual vectors `x_q, x_k` is
`x_q W_{QK}^i x_k^⊤/√e` (ignoring biases) — *where to look*; the vector written when attending to
`x_k` is `x_k W_{OV}^i` — *what to move*. Code: `interpretability/attention_scores.py::_qk, _ov`.

## 3. The residual stream and block structure

Each block reads the stream and **adds** its output (He et al., 2016 residual connections):

**Pre-LN (default, `norm: pre`)** — GPT-2 placement:

$$ r^{\text{mid}} = r + \text{Attn}(\text{LN}_1(r)),\qquad r^{\text{post}} = r^{\text{mid}} + \text{MLP}(\text{LN}_2(r^{\text{mid}})) $$

and finally `logits = LN_f(r^{(L)}) W_U + b_U`.

**Post-LN (`norm: post`)** — original 2017 placement:

$$ r^{\text{mid}} = \text{LN}_1(r + \text{Attn}(r)),\qquad r^{\text{post}} = \text{LN}_2(r^{\text{mid}} + \text{MLP}(r^{\text{mid}})) $$

with no final LN (the stream is already normalised).

**No LN (`norm: none`)** — used by the hand-wired ground-truth model.

Because pre-LN only ever *adds* to `r`, the final stream decomposes exactly:

$$ r^{(L)} = W_E[x] + p + \sum_{l}\Big(\sum_i z_{l,i}W_O^{l,i} + b_O^{l}\Big) + \sum_l \text{MLP}_l(\cdot) $$

Code: `ActivationCache.decompose_resid` (tested to sum to the stream). Under post-LN each block
re-normalises the stream, so no such decomposition exists and the method refuses.

**Choice of default.** Pre-LN keeps an identity path from embeddings to logits, has well-behaved
gradients at initialisation and does not need warm-up to be stable (Xiong et al., 2020); it also
makes the additive decomposition above exact, which interpretability relies on. The three variants
are selected explicitly by `norm` — never mixed within a model.

## 4. Layer normalisation

$$ \text{LN}(x) = \frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}\odot w + b,\qquad \mu,\sigma^2 \text{ over the } d \text{ features} $$

Code: `LayerNorm.forward`; the denominator is hooked as `hook_scale` and the output as
`hook_normalized`. Matches `torch.nn.functional.layer_norm` (`test_layernorm_matches_torch`).
Note: centring (`x - μ`) is linear; division by the input-dependent scale is not. Attribution
methods (§7) freeze the scale to make LN linear for one input.

## 5. MLP

$$ \text{MLP}(x) = \phi(xW_{\text{in}} + b_{\text{in}})\,W_{\text{out}} + b_{\text{out}},\qquad W_{\text{in}}\in\mathbb R^{d\times m} $$

`φ` is GELU `x·Φ(x)` (Hendrycks & Gimpel, 2016; exact erf form, `act_fn: gelu`), its tanh
approximation, ReLU or SiLU. Applied independently at each position. Hooks: `mlp.hook_pre`
(pre-activation), `mlp.hook_post` (post-activation "neurons"), `hook_mlp_out`.

## 6. Training objective: teacher forcing and next-token prediction

Logits `ℓ_t = f(x_{≤t}) ∈ R^V`, `p_t = softmax(ℓ_t)`. The loss over scored positions `S`:

$$ \mathcal L = -\frac{1}{|S|}\sum_{t\in S}\log p_t(x_{t+1}) $$

This is cross-entropy between the one-hot next token and `p_t`, i.e. the negative log-likelihood;
minimising it is maximum-likelihood estimation. **Teacher forcing**: during training the model
always conditions on the *true* previous tokens, and because of the causal mask one forward pass
yields predictions for every position at once. **Autoregressive generation** instead feeds the
model's own sampled token back as input (`Transformer.generate`: greedy argmax when
`temperature=0`, otherwise sampling from `softmax(ℓ/T)` optionally restricted to the top-k). The
two differ only in where the conditioning tokens come from; errors can compound in generation,
which teacher-forced training never sees.

Code: `per_token_nll`, `lm_loss` (masked mean; computed in float32). The task supplies
`loss_mask` (scored positions) and optionally `train_mask` (trained positions).

**Optimisation**: AdamW (Kingma & Ba, 2015; Loshchilov & Hutter, 2019) with weight decay on
matrices only, linear warm-up then cosine decay to `min_lr_ratio`, global gradient-norm clipping.
Code: `training/trainer.py::make_optimizer, lr_factor`.

**Initial loss sanity check.** At initialisation logits are ≈ 0, so `p_t ≈ uniform` and
`L ≈ ln V` (tested: `test_initial_loss_close_to_uniform`).

## 7. Interpretability quantities

**Direct logit attribution** (`interpretability/attribution.py`). With the final LN scale `σ`
frozen at its observed value, `LN_f` is affine, so the target logit decomposes over the residual
components `c_j` of §3:

$$ \ell_t[y] = \sum_j \frac{(c_j - \bar c_j)}{\sigma}\odot w_f \cdot W_U[:,y] \;+\; b_f\cdot W_U[:,y] + b_U[y] $$

Exact for this forward pass (reconstruction error is reported and tested); it measures *direct*
effects only.

**Activation patching recovery** (`interpretability/patching.py`):

$$ R = \frac{\bar M_{\text{patched}} - \bar M_{\text{corrupt}}}{\bar M_{\text{clean}} - \bar M_{\text{corrupt}}} $$

with `M` a per-example metric (mean log-prob of the clean answer, or logit difference clean vs
corrupted answer). Undefined (reported as uninformative) when the denominator is below `min_gap`.

**Composition scores** (Elhage et al., 2021; `attention_scores.py::composition_scores`), e.g. for
K-composition of sender head `S` into receiver `R`:

$$ C_K = \frac{\| W_{QK}^R\, (W_{OV}^S)^\top \|_F}{\|W_{QK}^R\|_F\,\|W_{OV}^S\|_F} $$

Weight-only (ignores LN and the data distribution) — compare against the same score at
initialisation.

**Induction score** of a head: mean attention from query `q` to `prev(q)+1`, where `prev(q)` is
the most recent earlier position holding the same token. Previous-token score: mean attention
from `q` to `q-1`. Both averaged over scored positions.

## 8. Initialisation and scale

All weight matrices start `N(0, init_std²)` (default 0.02); `W_O` and `W_out` are further scaled by
`1/√(2L)` so the residual stream's variance does not grow with depth (Radford et al., 2019 report a
`1/√N` scaling of residual weights). Initialisation uses a private `torch.Generator`, so it depends
only on `ModelConfig.seed` (tested).

**A scale bug found during this project.** Fixed sinusoidal positions have amplitude ≈ 1 per
dimension, while token embeddings start at std 0.02. After pre-LN normalisation the stream is then
almost purely positional and the model trained at chance (loss = ln 48). Fixed by scaling the
table to the embedding init scale (`Transformer.__init__`); regression test
`test_sinusoidal_scale_matches_token_embedding_scale`.
