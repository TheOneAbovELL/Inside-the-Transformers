"""Checkpoint save/load.

A checkpoint is a plain dict saved with ``torch.save`` and loaded with ``weights_only=True``
(no arbitrary code execution on load). It contains the model config, weights, and — for
resumable training — optimiser/scheduler state and every RNG state needed to continue the
exact same data stream.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch

from ..config import ModelConfig
from ..models import Transformer

FORMAT_VERSION = 1


def save_checkpoint(path: Union[str, Path], model: Transformer, **extra) -> Path:
    """Atomically write ``{model_config, state_dict, **extra}`` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_config": model.cfg.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # never leaves a half-written checkpoint behind
    return path


def load_checkpoint(path: Union[str, Path]) -> dict:
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format {ckpt.get('format_version')} in {path}")
    return ckpt


def load_model(path: Union[str, Path], device: Optional[Union[str, torch.device]] = "cpu") -> tuple[Transformer, dict]:
    """Rebuild a :class:`Transformer` from a checkpoint; returns ``(model, checkpoint_dict)``."""
    ckpt = load_checkpoint(path)
    model = Transformer(ModelConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
