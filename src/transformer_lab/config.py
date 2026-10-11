"""Typed, validated configuration objects.

Every run is fully described by an :class:`ExperimentConfig` (model + task + training),
which is saved next to its artifacts so a run can be reproduced from disk.

Design notes
------------
* Configs are plain dataclasses (no external dependency) with explicit validation in
  ``__post_init__``. Unknown keys in YAML are an *error*, not silently ignored, so a typo
  cannot quietly change an experiment.
* ``vocab_size`` and ``n_ctx`` of the model are usually *derived from the task* (see
  :func:`transformer_lab.experiments.runner.build_model_config`), so they are optional here.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

_ACT_FNS = ("gelu", "gelu_tanh", "relu", "silu")
_NORMS = ("pre", "post", "none")
_POSITIONALS = ("learned", "sinusoidal", "alibi", "none")
_SCHEDULES = ("constant", "cosine", "linear")


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration values."""


def _from_dict(cls, data: Optional[dict]):
    """Construct dataclass ``cls`` from ``data``, rejecting unknown keys."""
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"Unknown {cls.__name__} keys: {sorted(unknown)}; allowed: {sorted(known)}")
    return cls(**data)


@dataclass
class ModelConfig:
    """Architecture hyper-parameters for :class:`transformer_lab.models.Transformer`.

    Attributes
    ----------
    vocab_size, n_ctx:
        Vocabulary size and maximum context length. Usually filled in from the task.
    d_model:
        Width of the residual stream.
    n_layers, n_heads:
        Number of decoder blocks and attention heads per block.
    d_head:
        Per-head dimension. Defaults to ``d_model // n_heads``.
    d_mlp:
        Hidden width of the MLP. Defaults to ``4 * d_model``. Ignored when ``attn_only``.
    attn_only:
        If true, blocks contain attention only (no MLP). Standard for induction-head studies.
    act_fn:
        MLP non-linearity: ``gelu`` (exact erf form), ``gelu_tanh``, ``relu`` or ``silu``.
    norm:
        ``pre`` (default; LN before each sub-layer + final LN), ``post`` (original 2017 placement:
        LN after each residual addition, no final LN) or ``none`` (no normalisation at all).
    positional:
        ``learned`` absolute embeddings (default), fixed ``sinusoidal``, ``alibi`` (no embedding;
        a fixed linear distance penalty on attention scores, Press et al. 2022) or ``none``.
    init_std:
        Std of the normal initialisation of weight matrices. Output projections of each residual
        branch (``W_O``, ``W_out``) are additionally scaled by ``1/sqrt(2 * n_layers)``.
    seed:
        Seed for parameter initialisation (uses a private generator, so it does not depend on
        global RNG state).
    """

    vocab_size: Optional[int] = None
    n_ctx: Optional[int] = None
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_head: Optional[int] = None
    d_mlp: Optional[int] = None
    attn_only: bool = False
    act_fn: str = "gelu"
    norm: str = "pre"
    positional: str = "learned"
    ln_eps: float = 1e-5
    init_std: float = 0.02
    seed: int = 0

    def __post_init__(self) -> None:
        if self.d_head is None:
            if self.d_model % self.n_heads != 0:
                raise ConfigError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}; set d_head")
            self.d_head = self.d_model // self.n_heads
        if self.d_mlp is None:
            self.d_mlp = 4 * self.d_model
        for name in ("d_model", "n_layers", "n_heads", "d_head", "d_mlp"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("vocab_size", "n_ctx"):
            val = getattr(self, name)
            if val is not None and val <= 0:
                raise ConfigError(f"{name} must be positive, got {val}")
        if self.act_fn not in _ACT_FNS:
            raise ConfigError(f"act_fn must be one of {_ACT_FNS}, got {self.act_fn!r}")
        if self.norm not in _NORMS:
            raise ConfigError(f"norm must be one of {_NORMS}, got {self.norm!r}")
        if self.positional not in _POSITIONALS:
            raise ConfigError(f"positional must be one of {_POSITIONALS}, got {self.positional!r}")
        if self.positional == "sinusoidal" and self.d_model % 2:
            raise ConfigError("sinusoidal positions need an even d_model")
        if self.init_std <= 0:
            raise ConfigError("init_std must be positive")

    def require_complete(self) -> "ModelConfig":
        if self.vocab_size is None or self.n_ctx is None:
            raise ConfigError("ModelConfig needs vocab_size and n_ctx (normally derived from the task)")
        return self

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ModelConfig":
        return _from_dict(cls, data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def n_params(self) -> int:
        """Analytic parameter count (matches ``sum(p.numel())`` of the built model)."""
        self.require_complete()
        d, h, dh, L = self.d_model, self.n_heads, self.d_head, self.n_layers
        n = self.vocab_size * d + d * self.vocab_size + self.vocab_size  # W_E, W_U, b_U
        if self.positional == "learned":
            n += self.n_ctx * d
        attn = 4 * h * d * dh + 3 * h * dh + d  # W_Q,K,V,O + b_Q,K,V + b_O
        ln = 2 * d if self.norm != "none" else 0
        mlp = 0 if self.attn_only else (2 * d * self.d_mlp + self.d_mlp + d)
        n_ln_per_block = 1 if self.attn_only else 2
        n += L * (attn + mlp + n_ln_per_block * ln)
        if self.norm == "pre":
            n += 2 * d  # final LayerNorm
        return n


@dataclass
class TaskConfig:
    """Which synthetic task to generate, and its parameters (validated by the task class)."""

    name: str = "induction_repeat"
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TaskConfig":
        return _from_dict(cls, data)

    def to_dict(self) -> dict:
        return {"name": self.name, "params": dict(self.params)}


@dataclass
class TrainConfig:
    """Optimisation and logging settings.

    ``checkpoint_steps`` lists extra steps at which to keep a checkpoint permanently (useful for
    studying how mechanisms *form* during training); ``checkpoint_every`` controls the rolling
    "latest" checkpoint used for resuming.
    """

    steps: int = 2000
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    warmup_steps: int = 100
    schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    grad_clip: Optional[float] = 1.0
    eval_every: int = 250
    eval_size: int = 512
    log_every: int = 50
    checkpoint_every: int = 500
    checkpoint_steps: list = field(default_factory=list)
    early_stopping_patience: Optional[int] = None  # in evaluations; None disables
    early_stopping_metric: str = "val_loss"
    early_stopping_min_delta: float = 1e-4
    target_accuracy: Optional[float] = None  # stop once val accuracy >= this (None disables)
    device: str = "cpu"
    num_threads: Optional[int] = None

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)
        if self.steps < 0 or self.batch_size <= 0:
            raise ConfigError("steps must be >= 0 and batch_size > 0")
        if self.lr <= 0:
            raise ConfigError("lr must be positive")
        if self.schedule not in _SCHEDULES:
            raise ConfigError(f"schedule must be one of {_SCHEDULES}")
        if not (0.0 <= self.min_lr_ratio <= 1.0):
            raise ConfigError("min_lr_ratio must be in [0, 1]")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ConfigError("grad_clip must be positive or null")
        if self.eval_every <= 0 or self.log_every <= 0 or self.checkpoint_every <= 0:
            raise ConfigError("eval_every, log_every, checkpoint_every must be positive")
        if self.early_stopping_metric not in ("val_loss", "val_accuracy"):
            raise ConfigError("early_stopping_metric must be val_loss or val_accuracy")
        self.checkpoint_steps = sorted({int(s) for s in self.checkpoint_steps})

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "TrainConfig":
        return _from_dict(cls, data)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["betas"] = list(self.betas)
        return d


@dataclass
class ExperimentConfig:
    """A complete, reproducible training run: model + task + optimisation + seed."""

    name: str
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        data = dict(data)
        allowed = {"name", "seed", "model", "task", "train", "notes"}
        unknown = set(data) - allowed
        if unknown:
            raise ConfigError(f"Unknown ExperimentConfig keys: {sorted(unknown)}")
        if "name" not in data:
            raise ConfigError("ExperimentConfig needs a 'name'")
        return cls(
            name=str(data["name"]),
            seed=int(data.get("seed", 0)),
            model=ModelConfig.from_dict(data.get("model")),
            task=TaskConfig.from_dict(data.get("task")),
            train=TrainConfig.from_dict(data.get("train")),
            notes=str(data.get("notes", "")),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "model": self.model.to_dict(),
            "task": self.task.to_dict(),
            "train": self.train.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def with_overrides(self, overrides: dict[str, Any]) -> "ExperimentConfig":
        """Return a copy with dotted-key overrides applied, e.g. ``{"train.steps": 10, "seed": 3}``."""
        d = self.to_dict()
        for key, value in overrides.items():
            parts = key.split(".")
            node = d
            for p in parts[:-1]:
                if p not in node or not isinstance(node[p], dict):
                    raise ConfigError(f"Override path {key!r} not found")
                node = node[p]
            if parts[-1] not in node and parts[0] != "task":
                raise ConfigError(f"Override key {key!r} not found")
            node[parts[-1]] = value
        return ExperimentConfig.from_dict(d)


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key=value`` where value is YAML (so ``1e-3``, ``true``, ``[1,2]`` work)."""
    if "=" not in text:
        raise ConfigError(f"Override must look like key=value, got {text!r}")
    key, raw = text.split("=", 1)
    return key.strip(), yaml.safe_load(raw)
