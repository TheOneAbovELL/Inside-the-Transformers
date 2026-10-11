"""Hand-constructed 2-layer transformers that implement induction *by design*.

Purpose: ground truth for validating interpretability tools. On these models we *know* which
component does what, so ablation / patching / path interventions / attention scores / attribution
must recover it. Tools that fail here cannot be trusted on trained models.

Residual stream layout (``norm="none"``, so the stream is exactly additive)::

    [ token (V) | previous token (V) | output (V) | position (n_ctx) | prev-prev token (V)* | relay (V)* ]
                                                        (* only in the variants that need them)

Base model (``build_induction_model``):

* L0H0 — previous-token head. QK: position p attends to p-1 (positional match). OV: copies the
  attended token's identity into the *previous-token* slot.
* L1H0 — single-token induction head. QK: query = current token, key = previous-token slot
  (K-composition with L0H0). OV: copies the attended token into the *output* slot.
* L0H1, L1H1 — dead heads (all weights zero): negative controls.
* The unembedding reads only the output slot.

Variants:

* ``context_gain > 0`` — **context-matching induction** (positive control for context
  preference). L0H1 becomes a *previous-previous* head writing the token two back into its own
  slot; L1H0's query additionally reads the previous-token slot (current position's predecessor)
  and its key the prev-prev slot, so a key whose *preceding context* also matches scores higher by
  ``context_gain``. This is Q-composition (query reads L0H0) plus K-composition (key reads L0H1).
* ``mlp_relay=True`` — L0H0's output reaches the induction keys **only through MLP0**: MLP0 copies
  the previous-token slot into a relay slot (ReLU is the identity on these non-negative one-hot
  values) and L1H0's key reads the relay slot. Ground truth for MLP-mediated composition.
"""

from __future__ import annotations

import torch

from ..config import ModelConfig
from .transformer import Transformer


@torch.no_grad()
def build_induction_model(n_symbols: int, n_ctx: int, gain: float = 100.0, out_gain: float = 20.0,
                          context_gain: float = 0.0, mlp_relay: bool = False) -> Transformer:
    V = n_symbols + 2  # matches the task vocabularies (<pad>, <bos>, symbols)
    use_ctx = context_gain > 0
    TOK, PREV, OUT, POS = 0, V, 2 * V, 3 * V
    nxt = 3 * V + n_ctx
    PREV2 = RELAY = None
    if use_ctx:
        PREV2, nxt = nxt, nxt + V
    if mlp_relay:
        RELAY, nxt = nxt, nxt + V
    d_model = nxt
    d_head = max(2 * V if use_ctx else V, n_ctx)
    cfg = ModelConfig(vocab_size=V, n_ctx=n_ctx, d_model=d_model, n_layers=2, n_heads=2, d_head=d_head,
                      attn_only=not mlp_relay, d_mlp=V if mlp_relay else None, act_fn="relu",
                      norm="none", positional="learned")
    m = Transformer(cfg)
    for p in m.parameters():
        p.zero_()
    for t in range(V):
        m.W_E[t, TOK + t] = 1.0
    for p in range(n_ctx):
        m.W_pos[p, POS + p] = 1.0

    a0 = m.blocks[0].attn  # L0H0: previous-token head
    for p in range(n_ctx):
        a0.W_Q[0, POS + p, p] = gain
    for j in range(n_ctx - 1):
        a0.W_K[0, POS + j, j + 1] = 1.0
    for t in range(V):
        a0.W_V[0, TOK + t, t] = 1.0
        a0.W_O[0, t, PREV + t] = 1.0
    if use_ctx:  # L0H1: previous-previous-token head
        for p in range(n_ctx):
            a0.W_Q[1, POS + p, p] = gain
        for j in range(n_ctx - 2):
            a0.W_K[1, POS + j, j + 2] = 1.0
        for t in range(V):
            a0.W_V[1, TOK + t, t] = 1.0
            a0.W_O[1, t, PREV2 + t] = 1.0
    if mlp_relay:  # MLP0 copies the previous-token slot into the relay slot
        mlp = m.blocks[0].mlp
        for t in range(V):
            mlp.W_in[PREV + t, t] = 1.0
            mlp.W_out[t, RELAY + t] = 1.0

    key_slot = RELAY if mlp_relay else PREV
    a1 = m.blocks[1].attn  # L1H0: induction head
    for t in range(V):
        a1.W_Q[0, TOK + t, t] = gain
        a1.W_K[0, key_slot + t, t] = 1.0
        a1.W_V[0, TOK + t, t] = 1.0
        a1.W_O[0, t, OUT + t] = 1.0
        if use_ctx:  # query's predecessor must match the key's prev-prev token
            a1.W_Q[0, PREV + t, V + t] = context_gain
            a1.W_K[0, PREV2 + t, V + t] = 1.0

    for t in range(V):
        m.W_U[OUT + t, t] = out_gain
    m.eval()
    return m
