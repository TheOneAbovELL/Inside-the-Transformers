"""Activation cache: a read-only mapping from hook name to recorded activation.

Short names make analysis code readable::

    cache["pattern", 1]        # == cache["blocks.1.attn.hook_pattern"]   [b,h,q,k]
    cache["resid_post", 0]     # == cache["blocks.0.hook_resid_post"]     [b,s,d]
    cache["embed"]             # == cache["hook_embed"]

The cache only contains what was requested via ``names_filter`` in ``run_with_cache`` — nothing is
recorded during ordinary forward passes or training.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch

_LAYER_NAMES = {
    "resid_pre": "blocks.{l}.hook_resid_pre",
    "resid_mid": "blocks.{l}.hook_resid_mid",
    "resid_post": "blocks.{l}.hook_resid_post",
    "attn_out": "blocks.{l}.hook_attn_out",
    "mlp_out": "blocks.{l}.hook_mlp_out",
    "q": "blocks.{l}.attn.hook_q",
    "k": "blocks.{l}.attn.hook_k",
    "v": "blocks.{l}.attn.hook_v",
    "z": "blocks.{l}.attn.hook_z",
    "attn_scores": "blocks.{l}.attn.hook_attn_scores",
    "pattern": "blocks.{l}.attn.hook_pattern",
    "mlp_pre": "blocks.{l}.mlp.hook_pre",
    "mlp_post": "blocks.{l}.mlp.hook_post",
    "mlp_in": "blocks.{l}.hook_mlp_in",
    "ln1_scale": "blocks.{l}.ln1.hook_scale",
    "ln1_out": "blocks.{l}.ln1.hook_normalized",
    "ln2_scale": "blocks.{l}.ln2.hook_scale",
    "ln2_out": "blocks.{l}.ln2.hook_normalized",
    "q_input": "blocks.{l}.hook_q_input",
    "k_input": "blocks.{l}.hook_k_input",
    "v_input": "blocks.{l}.hook_v_input",
}
_GLOBAL_NAMES = {
    "embed": "hook_embed",
    "pos_embed": "hook_pos_embed",
    "ln_final_scale": "ln_final.hook_scale",
    "ln_final_out": "ln_final.hook_normalized",
}


def act_name(short: str, layer: Optional[int] = None) -> str:
    """Translate a short activation name (+ layer) to the full hook-point name."""
    if short in _GLOBAL_NAMES:
        return _GLOBAL_NAMES[short]
    if short in _LAYER_NAMES:
        if layer is None:
            raise ValueError(f"Activation {short!r} needs a layer index")
        return _LAYER_NAMES[short].format(l=layer)
    return short  # assume it is already a full name


class ActivationCache:
    def __init__(self, store: dict[str, torch.Tensor], model) -> None:
        self._store = store
        self.model = model

    # mapping protocol ---------------------------------------------------------------
    def __getitem__(self, key) -> torch.Tensor:
        name = act_name(*key) if isinstance(key, tuple) else act_name(key)
        try:
            return self._store[name]
        except KeyError:
            raise KeyError(f"{name!r} not in cache (was it excluded by names_filter?)") from None

    def __contains__(self, key) -> bool:
        name = act_name(*key) if isinstance(key, tuple) else act_name(key)
        return name in self._store

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def keys(self):
        return self._store.keys()

    def items(self):
        return self._store.items()

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._store.values())

    # derived quantities ---------------------------------------------------------------
    def head_results(self, layer: int) -> torch.Tensor:
        """Each head's contribution to the residual stream, ``z_i W_O[i]``: ``[b,s,h,d]``.

        Computed from cached ``z`` and the model's ``W_O``; summing over heads and adding
        ``b_O`` reproduces ``attn_out`` exactly (tested).
        """
        z = self["z", layer]
        W_O = self.model.blocks[layer].attn.W_O.detach().to(z.device)
        return torch.einsum("bshe,hed->bshd", z, W_O)

    def decompose_resid(self, layer: Optional[int] = None) -> tuple[list[str], torch.Tensor]:
        """Split the residual stream into additive per-component contributions.

        Returns ``(labels, components)`` with ``components: [n_components, b, s, d]`` such that
        ``components.sum(0)`` equals ``resid_pre`` of block ``layer`` (or the final ``resid_post``
        when ``layer is None``). Components: token embedding, positional embedding, each head's
        output, each layer's ``b_O`` and each MLP's output.

        Only valid when the residual stream is purely additive (``norm`` = ``pre`` or ``none``).
        Under post-LN every block re-normalises the stream, so no such decomposition exists.
        """
        cfg = self.model.cfg
        if cfg.norm == "post":
            raise NotImplementedError("Residual decomposition is undefined for post-LN models")
        upto = cfg.n_layers if layer is None else layer
        labels = ["embed", "pos_embed"]
        parts = [self["embed"], self["pos_embed"]]
        for l in range(upto):
            res = self.head_results(l)
            for h in range(cfg.n_heads):
                labels.append(f"L{l}H{h}")
                parts.append(res[:, :, h])
            b_O = self.model.blocks[l].attn.b_O.detach().to(res.device)
            labels.append(f"L{l}.b_O")
            parts.append(b_O.expand_as(parts[0]))
            if not cfg.attn_only:
                labels.append(f"L{l}.mlp")
                parts.append(self["mlp_out", l])
        return labels, torch.stack(parts)
