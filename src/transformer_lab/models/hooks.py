"""Named hook points: the instrumentation interface of the model.

A :class:`HookPoint` is an identity ``nn.Module`` placed at every interesting activation in the
forward pass (``blocks.1.attn.hook_pattern``, ``blocks.0.hook_resid_post``, ...). Because it is
an identity, it does not change the computation; it only gives a stable, named place where code
can *read* (cache) or *replace* (intervene on) an activation without editing the model.

Hook functions have the signature ``fn(activation: Tensor, hook: HookPoint) -> Tensor | None``.
Returning ``None`` leaves the activation unchanged (pure observation); returning a tensor
replaces it for the rest of the forward pass (intervention).

Hooks are attached through :meth:`HookedRootModule.hooks`, a context manager that *always* removes
them, even if the forward pass raises. This prevents the classic bug where a leftover ablation
hook silently corrupts every later experiment.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, Optional, Sequence, Union

import torch
from torch import nn

HookFn = Callable[[torch.Tensor, "HookPoint"], Optional[torch.Tensor]]
NamesFilter = Union[None, str, Sequence[str], Callable[[str], bool]]


class HookPoint(nn.Module):
    """Identity module that exposes a named activation to hook functions."""

    def __init__(self) -> None:
        super().__init__()
        self.name: str = ""
        self._handles: list = []

    def add_hook(self, fn: HookFn):
        def _wrapper(module: nn.Module, inputs, output):
            result = fn(output, self)
            if result is not None and result.shape != output.shape:
                raise RuntimeError(
                    f"Hook on {self.name} returned shape {tuple(result.shape)}, expected {tuple(output.shape)}"
                )
            return result

        handle = self.register_forward_hook(_wrapper)
        self._handles.append(handle)
        return handle

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def has_hooks(self) -> bool:
        return bool(self._handles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - identity
        return x

    def extra_repr(self) -> str:
        return self.name


def _make_filter(names_filter: NamesFilter) -> Callable[[str], bool]:
    if names_filter is None:
        return lambda name: True
    if isinstance(names_filter, str):
        return lambda name: name == names_filter
    if callable(names_filter):
        return names_filter
    allowed = set(names_filter)
    return lambda name: name in allowed


class HookedRootModule(nn.Module):
    """Mixin giving a model named hook points, hook management and activation caching."""

    def setup_hooks(self) -> None:
        """Assign each HookPoint its fully-qualified module name. Call at the end of ``__init__``."""
        self.hook_dict: dict[str, HookPoint] = {}
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.name = name
                self.hook_dict[name] = module

    def hook_names(self) -> list[str]:
        return list(self.hook_dict)

    def reset_hooks(self) -> None:
        for hp in self.hook_dict.values():
            hp.remove_hooks()

    def _resolve(self, name: str) -> HookPoint:
        try:
            return self.hook_dict[name]
        except KeyError:
            raise KeyError(f"No hook point named {name!r}. Example names: {self.hook_names()[:6]} ...") from None

    @contextmanager
    def hooks(self, fwd_hooks: Iterable[tuple[Union[str, Callable[[str], bool]], HookFn]] = ()) -> Iterator["HookedRootModule"]:
        """Temporarily attach hooks; guaranteed removal on exit.

        ``fwd_hooks`` is a list of ``(name_or_predicate, fn)``. A predicate attaches ``fn`` to every
        hook point whose name satisfies it.
        """
        handles = []
        try:
            for target, fn in fwd_hooks:
                if callable(target):
                    points = [hp for n, hp in self.hook_dict.items() if target(n)]
                else:
                    points = [self._resolve(target)]
                for hp in points:
                    handles.append((hp, hp.add_hook(fn)))
            yield self
        finally:
            for hp, h in handles:
                h.remove()
                if h in hp._handles:
                    hp._handles.remove(h)

    def run_with_hooks(self, tokens: torch.Tensor, fwd_hooks=(), **kwargs) -> torch.Tensor:
        with self.hooks(fwd_hooks):
            return self(tokens, **kwargs)

    def run_with_cache(
        self,
        tokens: torch.Tensor,
        names_filter: NamesFilter = None,
        fwd_hooks=(),
        detach: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        """Run the model and record activations at the selected hook points.

        Parameters
        ----------
        names_filter:
            ``None`` (all hook points), a name, a list of names or a predicate. Restricting this
            keeps memory bounded — the cache is never populated during ordinary training.
        fwd_hooks:
            Additional intervention hooks applied during the same run. Cache hooks are added
            *after* them, so the cache records the post-intervention value at each hook point.
        detach:
            Detach cached tensors from the autograd graph (default). Set ``False`` to keep
            gradients flowing, e.g. for attribution methods.

        Returns ``(logits, ActivationCache)``.
        """
        from ..instrumentation.cache import ActivationCache  # local import avoids a cycle

        keep = _make_filter(names_filter)
        store: dict[str, torch.Tensor] = {}

        def _save(act: torch.Tensor, hook: HookPoint):
            t = act.detach() if detach else act
            if device is not None:
                t = t.to(device)
            store[hook.name] = t
            return None

        cache_hooks = [(n, _save) for n in self.hook_dict if keep(n)]
        with self.hooks(list(fwd_hooks) + cache_hooks):
            logits = self(tokens, **kwargs)
        return logits, ActivationCache(store, self)
