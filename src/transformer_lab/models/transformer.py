"""A transparent decoder-only Transformer with named hook points everywhere.

Notation (used in all shape comments)::

    b  batch size            s  sequence length (query positions q / key positions k)
    d  d_model               h  n_heads           e  d_head
    m  d_mlp                 V  vocab_size

Data flow (pre-LN, the default)::

    tokens [b,s]
      -> W_E[tokens] + W_pos[:s]                                 resid_0      [b,s,d]
      for each block l:
          resid_mid  = resid_pre  + Attn(LN1(resid_pre))          [b,s,d]
          resid_post = resid_mid  + MLP(LN2(resid_mid))           [b,s,d]
      -> LN_final -> W_U, b_U                                     logits       [b,s,V]

Every intermediate is passed through a :class:`HookPoint`; the full list is available from
``model.hook_names()``. Weights are stored *per head* (``W_Q: [h,d,e]``, ``W_O: [h,e,d]``) rather than
fused, so each head's QK and OV circuits can be read directly from the parameters.

See ``docs/mathematics.md`` for derivations that map line-by-line onto this file.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..config import ModelConfig
from .hooks import HookedRootModule, HookPoint


def _activation(name: str):
    if name == "gelu":
        return lambda x: F.gelu(x)  # exact: x * Phi(x)
    if name == "gelu_tanh":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "relu":
        return F.relu
    if name == "silu":
        return F.silu
    raise ValueError(name)


class LayerNorm(nn.Module):
    """LayerNorm written out explicitly so its scale can be hooked (and frozen for attribution).

    ``LN(x) = (x - mean(x)) / sqrt(var(x) + eps) * w + b`` over the last dimension.

    Hook points: ``hook_scale`` (the denominator, ``[..., 1]``) and ``hook_normalized`` (the full
    output, after the affine transform). If ``enabled`` is false the module is the identity and has
    no parameters (``norm: none``); the hooks still exist so analysis code can be uniform.
    """

    def __init__(self, d_model: int, eps: float, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.eps = eps
        if enabled:
            self.w = nn.Parameter(torch.ones(d_model))
            self.b = nn.Parameter(torch.zeros(d_model))
        self.hook_scale = HookPoint()
        self.hook_normalized = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.hook_normalized(x)
        x = x - x.mean(dim=-1, keepdim=True)
        scale = self.hook_scale((x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt())
        return self.hook_normalized(x / scale * self.w + self.b)


class Attention(nn.Module):
    """Multi-head causal self-attention, computed explicitly so every stage can be hooked.

    For each head ``i``::

        q_i = x W_Q[i] + b_Q[i]                     [b,s,e]
        k_i = x W_K[i] + b_K[i]                     [b,s,e]
        v_i = x W_V[i] + b_V[i]                     [b,s,e]
        A_i = softmax(q_i k_i^T / sqrt(e) + M)      [b,q,k]   M_{qk} = 0 if k <= q else -inf
        z_i = A_i v_i                               [b,q,e]
        out = sum_i z_i W_O[i] + b_O                [b,q,d]

    Summing ``z_i W_O[i]`` over heads is exactly the usual "concatenate heads, then multiply by one
    big W_O" — the per-head form makes it explicit that each head *adds its own vector* to the
    residual stream. Zeroing ``z_i`` therefore removes exactly head ``i``'s contribution.

    Inputs may be ``[b,s,d]`` (shared by all heads) or ``[b,s,h,d]`` (a separate input per head, used
    for path-specific interventions on one head's query/key/value input).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h, d, e = cfg.n_heads, cfg.d_model, cfg.d_head
        self.n_heads, self.d_head = h, e
        self.W_Q = nn.Parameter(torch.empty(h, d, e))
        self.W_K = nn.Parameter(torch.empty(h, d, e))
        self.W_V = nn.Parameter(torch.empty(h, d, e))
        self.W_O = nn.Parameter(torch.empty(h, e, d))
        self.b_Q = nn.Parameter(torch.zeros(h, e))
        self.b_K = nn.Parameter(torch.zeros(h, e))
        self.b_V = nn.Parameter(torch.zeros(h, e))
        self.b_O = nn.Parameter(torch.zeros(d))
        mask = torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)
        self.alibi = cfg.positional == "alibi"
        if self.alibi:
            # ALiBi (Press et al., 2022): score_{qk} += -m_h * (q - k). Fixed, head-specific slopes
            # m_h = 2^(-8 (h+1) / n_heads). Steep heads are recency-biased from initialisation.
            slopes = torch.tensor([2.0 ** (-8.0 * (i + 1) / h) for i in range(h)])
            pos = torch.arange(cfg.n_ctx)
            dist = (pos[:, None] - pos[None, :]).clamp(min=0).float()  # [q, k]
            self.register_buffer("alibi_bias", -slopes[:, None, None] * dist[None], persistent=False)  # [h,q,k]

        self.hook_q = HookPoint()             # [b,s,h,e]
        self.hook_k = HookPoint()             # [b,s,h,e]
        self.hook_v = HookPoint()             # [b,s,h,e]
        self.hook_attn_scores = HookPoint()   # [b,h,q,k]  scaled (+ALiBi bias) + masked (-inf above diagonal)
        self.hook_pattern = HookPoint()       # [b,h,q,k]  softmax probabilities
        self.hook_z = HookPoint()             # [b,s,h,e]  per-head weighted values

    @staticmethod
    def _project(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            # Shared input: one [d, h*e] matmul is much faster on CPU than a batched einsum and is
            # the same linear map (W[h] stacked along the output dimension).
            h, d, e = W.shape
            out = x @ W.permute(1, 0, 2).reshape(d, h * e)
            return out.view(*x.shape[:2], h, e) + b
        return torch.einsum("bshd,hde->bshe", x, W) + b  # per-head input

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor) -> torch.Tensor:
        s = x_q.shape[1]
        q = self.hook_q(self._project(x_q, self.W_Q, self.b_Q))
        k = self.hook_k(self._project(x_k, self.W_K, self.b_K))
        v = self.hook_v(self._project(x_v, self.W_V, self.b_V))

        scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / math.sqrt(self.d_head)
        if self.alibi:
            scores = scores + self.alibi_bias[:, :s, :s]
        scores = scores.masked_fill(~self.causal_mask[:s, :s], float("-inf"))
        scores = self.hook_attn_scores(scores)
        pattern = self.hook_pattern(torch.softmax(scores, dim=-1))

        z = self.hook_z(torch.einsum("bhqk,bkhe->bqhe", pattern, v))
        b_, s_, h, e = z.shape
        # == sum_i z_i W_O[i]  (concatenate heads, multiply by the stacked W_O)
        return z.reshape(b_, s_, h * e) @ self.W_O.reshape(h * e, -1) + self.b_O


class MLP(nn.Module):
    """Position-wise MLP: ``W_out act(x W_in + b_in) + b_out`` applied independently at each position."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))
        self.act = _activation(cfg.act_fn)
        self.hook_pre = HookPoint()   # [b,s,m] pre-activation
        self.hook_post = HookPoint()  # [b,s,m] post-activation ("neurons")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.hook_pre(x @ self.W_in + self.b_in)
        post = self.hook_post(self.act(pre))
        return post @ self.W_out + self.b_out


class TransformerBlock(nn.Module):
    """One decoder block. ``norm`` selects pre-LN, post-LN or no normalisation explicitly.

    Pre-LN  : resid_mid = x + Attn(LN1(x));        resid_post = resid_mid + MLP(LN2(resid_mid))
    Post-LN : resid_mid = LN1(x + Attn(x));        resid_post = LN2(resid_mid + MLP(resid_mid))
    None    : resid_mid = x + Attn(x);             resid_post = resid_mid + MLP(resid_mid)

    Note that under post-LN the "residual stream" is re-normalised after each sub-layer, so the
    clean additive decomposition used by direct logit attribution only holds for pre-LN / none.
    """

    def __init__(self, cfg: ModelConfig, layer: int) -> None:
        super().__init__()
        self.cfg, self.layer = cfg, layer
        use_ln = cfg.norm != "none"
        self.ln1 = LayerNorm(cfg.d_model, cfg.ln_eps, enabled=use_ln)
        self.attn = Attention(cfg)
        if not cfg.attn_only:
            self.ln2 = LayerNorm(cfg.d_model, cfg.ln_eps, enabled=use_ln)
            self.mlp = MLP(cfg)
            self.hook_mlp_out = HookPoint()   # [b,s,d]
            # The residual stream as read by the MLP only (before ln2): editing it changes the MLP's
            # input without touching the stream itself (used by path patching).
            self.hook_mlp_in = HookPoint()    # [b,s,d]
        self.hook_resid_pre = HookPoint()     # [b,s,d]
        self.hook_attn_out = HookPoint()      # [b,s,d]
        self.hook_resid_mid = HookPoint()     # [b,s,d]
        self.hook_resid_post = HookPoint()    # [b,s,d]
        # Per-head copies of the attention input; only active when the model's
        # ``split_qkv_input`` flag is set (costs h x memory, so off by default).
        self.hook_q_input = HookPoint()       # [b,s,h,d]
        self.hook_k_input = HookPoint()
        self.hook_v_input = HookPoint()
        self.split_qkv_input = False

    def _attention(self, x: torch.Tensor, ln: Optional[LayerNorm]) -> torch.Tensor:
        if self.split_qkv_input:
            b, s, d = x.shape
            xh = x.unsqueeze(2).expand(b, s, self.cfg.n_heads, d)
            xq, xk, xv = self.hook_q_input(xh), self.hook_k_input(xh), self.hook_v_input(xh)
            if ln is not None:
                xq, xk, xv = ln(xq), ln(xk), ln(xv)
            return self.attn(xq, xk, xv)
        n = ln(x) if ln is not None else x
        return self.attn(n, n, n)

    def forward(self, resid: torch.Tensor) -> torch.Tensor:
        x = self.hook_resid_pre(resid)
        if self.cfg.norm == "post":
            attn_out = self.hook_attn_out(self._attention(x, None))
            mid = self.hook_resid_mid(self.ln1(x + attn_out))
            if self.cfg.attn_only:
                return self.hook_resid_post(mid)
            mlp_out = self.hook_mlp_out(self.mlp(self.hook_mlp_in(mid)))
            return self.hook_resid_post(self.ln2(mid + mlp_out))

        # pre-LN (and "none", where LayerNorm is the identity)
        attn_out = self.hook_attn_out(self._attention(x, self.ln1))
        mid = self.hook_resid_mid(x + attn_out)
        if self.cfg.attn_only:
            return self.hook_resid_post(mid)
        mlp_out = self.hook_mlp_out(self.mlp(self.ln2(self.hook_mlp_in(mid))))
        return self.hook_resid_post(mid + mlp_out)


def sinusoidal_table(n_ctx: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal positions (Vaswani et al., 2017): PE[p, 2i] = sin(p / 10000^(2i/d)), PE[p, 2i+1] = cos(...)."""
    pos = torch.arange(n_ctx, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    table = torch.zeros(n_ctx, d_model)
    table[:, 0::2] = torch.sin(pos * div)
    table[:, 1::2] = torch.cos(pos * div)
    return table


class Transformer(HookedRootModule):
    """Decoder-only Transformer language model with named hook points.

    ``forward(tokens)`` returns logits ``[b,s,V]`` where ``logits[:, t]`` is the model's prediction
    for ``tokens[:, t+1]`` given ``tokens[:, :t+1]`` (teacher forcing: every position is trained in
    parallel and the causal mask guarantees position ``t`` never sees ``t+1..``).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.require_complete()
        self.cfg = cfg
        d, V = cfg.d_model, cfg.vocab_size
        self.W_E = nn.Parameter(torch.empty(V, d))
        if cfg.positional == "learned":
            self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx, d))
        elif cfg.positional == "sinusoidal":
            # Scaled so its per-dimension RMS equals the token-embedding init std. Unscaled
            # (amplitude ~1 vs token std 0.02), LayerNorm divides by a norm dominated by position
            # and token identity is all but erased: observed experimentally as chance-level learning
            # (docs/findings.md, "sinusoidal scale bug"). Vaswani et al. instead multiply token
            # embeddings by sqrt(d_model); either fixes the mismatch.
            table = sinusoidal_table(cfg.n_ctx, d) * (cfg.init_std * math.sqrt(2.0))
            self.register_buffer("W_pos", table, persistent=False)
        else:
            self.W_pos = None
        self.blocks = nn.ModuleList([TransformerBlock(cfg, l) for l in range(cfg.n_layers)])
        self.ln_final = LayerNorm(d, cfg.ln_eps, enabled=(cfg.norm == "pre"))
        self.W_U = nn.Parameter(torch.empty(d, V))
        self.b_U = nn.Parameter(torch.zeros(V))

        self.hook_embed = HookPoint()      # [b,s,d] token embeddings
        self.hook_pos_embed = HookPoint()  # [b,s,d] positional embeddings (zeros if positional=none)
        self.setup_hooks()
        self.reset_parameters()

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Deterministic init from ``cfg.seed`` using a private generator.

        Weight matrices ~ N(0, init_std^2). The output projection of every residual branch
        (W_O, W_out) is scaled by 1/sqrt(2 * n_layers) so the residual stream's variance does not
        grow with depth at initialisation (GPT-2 style). Biases start at 0; LayerNorm at (1, 0).
        """
        g = torch.Generator().manual_seed(int(self.cfg.seed))
        std = self.cfg.init_std
        resid_scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)

        def normal_(p: torch.Tensor, s: float) -> None:
            p.copy_(torch.randn(p.shape, generator=g) * s)

        normal_(self.W_E, std)
        if isinstance(self.W_pos, nn.Parameter):
            normal_(self.W_pos, std)
        normal_(self.W_U, std)
        self.b_U.zero_()
        for block in self.blocks:
            a = block.attn
            for W in (a.W_Q, a.W_K, a.W_V):
                normal_(W, std)
            normal_(a.W_O, std * resid_scale)
            for b_ in (a.b_Q, a.b_K, a.b_V, a.b_O):
                b_.zero_()
            if not self.cfg.attn_only:
                normal_(block.mlp.W_in, std)
                normal_(block.mlp.W_out, std * resid_scale)
                block.mlp.b_in.zero_()
                block.mlp.b_out.zero_()
            for ln in (block.ln1, getattr(block, "ln2", None)):
                if ln is not None and ln.enabled:
                    ln.w.fill_(1.0)
                    ln.b.zero_()
        if self.ln_final.enabled:
            self.ln_final.w.fill_(1.0)
            self.ln_final.b.zero_()

    # --------------------------------------------------------------- forward
    def set_split_qkv_input(self, enabled: bool) -> None:
        """Give every head its own copy of the attention input (enables per-head path interventions)."""
        for block in self.blocks:
            block.split_qkv_input = enabled

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        b, s = tokens.shape
        if s > self.cfg.n_ctx:
            raise ValueError(f"Sequence length {s} exceeds n_ctx={self.cfg.n_ctx}")
        if tokens.dtype not in (torch.int64, torch.int32):
            raise TypeError("tokens must be an integer tensor")
        x = self.hook_embed(self.W_E[tokens])
        if self.W_pos is not None:
            pos = self.W_pos[:s].unsqueeze(0).expand(b, s, -1)
        else:
            pos = torch.zeros_like(x)
        pos = self.hook_pos_embed(pos)
        return x + pos

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """``tokens [b,s] -> logits [b,s,V]``."""
        resid = self.embed(tokens)
        for block in self.blocks:
            resid = block(resid)
        resid = self.ln_final(resid)
        return resid @ self.W_U + self.b_U

    # ------------------------------------------------------------- utilities
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Autoregressive generation.

        ``temperature == 0`` is greedy decoding (argmax). Otherwise sample from
        ``softmax(logits / temperature)``, optionally restricted to the ``top_k`` most likely
        tokens. If the sequence grows beyond ``n_ctx`` only the last ``n_ctx`` tokens are used
        (a sliding window: absolute positions then shift, which the model never saw in training).
        """
        was_training = self.training
        self.eval()
        out = tokens
        for _ in range(max_new_tokens):
            ctx = out[:, -self.cfg.n_ctx:]
            logits = self(ctx)[:, -1, :]
            if temperature == 0.0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            out = torch.cat([out, nxt], dim=1)
        self.train(was_training)
        return out


def per_token_nll(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of each next token: ``[b,s,V],[b,s] -> [b,s-1]``.

    ``nll[:, t] = -log softmax(logits[:, t])[tokens[:, t+1]]``. Computed in float32 for stability.
    """
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return -logp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)


def lm_loss(logits: torch.Tensor, tokens: torch.Tensor, loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean next-token cross-entropy, optionally restricted to ``loss_mask [b,s-1]`` positions.

    Cross-entropy ``-log p(target)`` is the negative log-likelihood under the model's softmax
    distribution; minimising it is maximum-likelihood estimation of the next-token distribution.
    """
    nll = per_token_nll(logits, tokens)
    if loss_mask is None:
        return nll.mean()
    m = loss_mask.to(nll.dtype)
    denom = m.sum()
    if denom == 0:
        raise ValueError("loss_mask selects no positions")
    return (nll * m).sum() / denom
