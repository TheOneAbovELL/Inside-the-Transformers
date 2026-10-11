"""Synthetic algorithmic tasks with known solutions.

Every task provides

* a seed-controlled generator ``sample(n, rng, split)`` returning a :class:`TaskBatch`;
* an exact **algorithmic solver** ``solve(batch)`` (the "direct lookup" baseline — if it is not
  100% accurate the generator is buggy, and tests check this);
* an analytic **chance** accuracy;
* a **corruption** ``corrupt(batch, rng, kind)`` for activation patching that keeps the clean
  targets and, where possible, defines an alternative answer (``alt_targets``) so logit
  differences can be measured (symmetric token replacement, cf. Zhang & Nanda, 2024).

Conventions
-----------
``tokens[i, t]`` is the input; ``logits[i, t]`` predicts ``tokens[i, t+1]``. ``loss_mask[i, t]``
(shape ``[n, T-1]``) marks the prediction positions that are *determined by the context* — the only
positions that are trained on and scored. Token 0 is ``<pad>`` and token 1 is ``<bos>`` in every task.

Train/eval separation
---------------------
Training data is an infinite stream from one seed; validation and test sets come from different
seeds (derived with ``numpy.random.SeedSequence``). Tasks with a ``holdout_mod`` parameter
additionally hold out a deterministic subset of (key, value) / (A, B) pairs: ``split="train"`` never
*contains* a held-out pair, ``split="eval"`` always *queries* one. Accuracy on such an eval set
cannot come from memorising specific pairs.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np
import torch

PAD, BOS = 0, 1


@dataclass
class TaskBatch:
    tokens: torch.Tensor  # [n, T] int64
    loss_mask: torch.Tensor  # [n, T-1] bool
    meta: dict = field(default_factory=dict)  # name -> np.ndarray [n]
    alt_targets: Optional[torch.Tensor] = None  # [n, T-1] int64, -1 where undefined
    # Positions trained on, if different from the scored ones (e.g. also unpredictable positions,
    # which penalises confident false copying). Evaluation always uses ``loss_mask``.
    train_mask: Optional[torch.Tensor] = None  # [n, T-1] bool

    def __len__(self) -> int:
        return self.tokens.shape[0]

    @property
    def targets(self) -> torch.Tensor:
        return self.tokens[:, 1:]

    @property
    def n_scored(self) -> int:
        return int(self.loss_mask.sum())

    @property
    def training_mask(self) -> torch.Tensor:
        return self.loss_mask if self.train_mask is None else self.train_mask

    def to(self, device) -> "TaskBatch":
        opt = lambda t: None if t is None else t.to(device)
        return TaskBatch(self.tokens.to(device), self.loss_mask.to(device), self.meta, opt(self.alt_targets),
                         opt(self.train_mask))

    def subset(self, idx) -> "TaskBatch":
        idx = torch.as_tensor(idx, dtype=torch.long)
        meta = {k: np.asarray(v)[idx.numpy()] for k, v in self.meta.items()}
        opt = lambda t: None if t is None else t[idx]
        return TaskBatch(self.tokens[idx], self.loss_mask[idx], meta, opt(self.alt_targets), opt(self.train_mask))

    @staticmethod
    def concat(batches: list["TaskBatch"]) -> "TaskBatch":
        T = max(b.tokens.shape[1] for b in batches)
        def pad(t, fill):
            return torch.nn.functional.pad(t, (0, T - t.shape[1]), value=fill) if t.shape[1] < T else t
        def padm(t, fill):
            return torch.nn.functional.pad(t, (0, T - 1 - t.shape[1]), value=fill) if t.shape[1] < T - 1 else t
        tokens = torch.cat([pad(b.tokens, PAD) for b in batches])
        mask = torch.cat([padm(b.loss_mask, False) for b in batches])
        keys = set.intersection(*(set(b.meta) for b in batches))
        meta = {k: np.concatenate([np.asarray(b.meta[k]) for b in batches]) for k in keys}
        alt = None
        if all(b.alt_targets is not None for b in batches):
            alt = torch.cat([padm(b.alt_targets, -1) for b in batches])
        tm = None
        if any(b.train_mask is not None for b in batches):
            tm = torch.cat([padm(b.training_mask, False) for b in batches])
        return TaskBatch(tokens, mask, meta, alt, tm)


def _heldout(a, b, mod: int):
    """Deterministic hash split of (a, b) index pairs; ~1/mod of pairs are held out."""
    if not mod:
        return np.zeros(np.broadcast(a, b).shape, dtype=bool) if np.ndim(a) or np.ndim(b) else False
    return (np.asarray(a) * 31 + np.asarray(b) * 17) % mod == 0


def induction_solver(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Generic 'find the previous occurrence of the current token and copy its successor' rule.

    Returns predictions ``[n, T-1]`` (-1 where not masked or no previous occurrence). This is the
    exact algorithm an ideal induction head implements; it solves ``induction_repeat``,
    ``induction_pair`` *and* ``assoc_recall`` (a key's earlier occurrence is followed by its value).
    """
    tok = tokens.cpu().numpy()
    m = mask.cpu().numpy()
    out = np.full(m.shape, -1, dtype=np.int64)
    for i, t in zip(*np.nonzero(m)):
        prev = np.nonzero(tok[i, :t] == tok[i, t])[0]
        if prev.size:
            out[i, t] = tok[i, prev[-1] + 1]
    return torch.from_numpy(out)


def most_frequent_solver(tokens: torch.Tensor, mask: torch.Tensor, exclude=(PAD, BOS)) -> torch.Tensor:
    """Heuristic baseline: predict the most frequent *other* context token (ties -> most recent).

    The current token is excluded (otherwise recency tie-breaking always predicts it, which is
    never the answer in these tasks). Useful as a *misleading-shortcut* baseline: decoy tokens in
    ``induction_pair`` are designed to fool it.
    """
    tok = tokens.cpu().numpy()
    m = mask.cpu().numpy()
    out = np.full(m.shape, -1, dtype=np.int64)
    for i, t in zip(*np.nonzero(m)):
        ctx = tok[i, :t]
        ctx = ctx[~np.isin(ctx, list(exclude) + [tok[i, t]])]
        if ctx.size:
            vals, counts = np.unique(ctx, return_counts=True)
            best = vals[counts == counts.max()]
            last = max(best, key=lambda v: np.nonzero(ctx == v)[0][-1])
            out[i, t] = last
    return torch.from_numpy(out)


class Task:
    """Base class. Subclasses define ``name``, ``defaults`` and the generation logic."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    defaults: ClassVar[dict] = {}
    corruption_kinds: ClassVar[tuple] = ()

    def __init__(self, **params) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}; allowed {sorted(self.defaults)}")
        self.params = {**copy.deepcopy(self.defaults), **params}
        self._validate()

    # -- interface -----------------------------------------------------------------------
    def _validate(self) -> None:  # pragma: no cover - overridden
        pass

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def seq_len(self) -> int:
        raise NotImplementedError

    @property
    def chance_accuracy(self) -> float:
        raise NotImplementedError

    def sample(self, n: int, rng: np.random.Generator, split: str = "train") -> TaskBatch:
        raise NotImplementedError

    def solve(self, batch: TaskBatch) -> torch.Tensor:
        """Exact algorithmic predictions ``[n, T-1]`` (-1 off-mask)."""
        return induction_solver(batch.tokens, batch.loss_mask)

    def corrupt(self, batch: TaskBatch, rng: np.random.Generator, kind: Optional[str] = None) -> TaskBatch:
        raise NotImplementedError(f"{self.name} defines no corruption")

    def token_str(self, tok: int) -> str:
        return {PAD: "<pad>", BOS: "<bos>"}.get(int(tok), f"t{int(tok)}")

    # -- helpers -------------------------------------------------------------------------
    def decode(self, tokens) -> list[str]:
        return [self.token_str(t) for t in np.asarray(tokens).tolist()]

    def with_params(self, **overrides) -> "Task":
        """A variant (e.g. for OOD evaluation) that must keep the same vocabulary."""
        new = type(self)(**{**self.params, **overrides})
        if new.vocab_size != self.vocab_size:
            raise ValueError("Variant changes the vocabulary; a trained model could not read it")
        return new

    def spec(self) -> dict:
        return {"name": self.name, "params": dict(self.params), "vocab_size": self.vocab_size,
                "seq_len": self.seq_len, "chance_accuracy": self.chance_accuracy, "description": self.description}

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in ("train", "eval"):
            raise ValueError("split must be 'train' or 'eval'")


# ============================================================================ Task 1
class RepeatPattern(Task):
    """``<bos> p0 p1 .. p_{P-1} p0 p1 ...`` — a random pattern of distinct symbols repeated.

    The period P is only identifiable once ``p0`` recurs, so the targets scored are content
    positions ``c >= P + 1``. Purpose: the simplest in-context next-token baseline.
    """

    name = "repeat_pattern"
    description = "Continue a periodic sequence of distinct symbols (period varies per example)."
    defaults = {"n_symbols": 16, "seq_len": 24, "min_period": 2, "max_period": 6}

    def _validate(self) -> None:
        p = self.params
        if not (2 <= p["min_period"] <= p["max_period"] <= p["n_symbols"]):
            raise ValueError("need 2 <= min_period <= max_period <= n_symbols")
        if p["seq_len"] - 1 < p["max_period"] + 2:
            raise ValueError("seq_len too short to score any position")

    @property
    def vocab_size(self) -> int:
        return 2 + self.params["n_symbols"]

    @property
    def seq_len(self) -> int:
        return self.params["seq_len"]

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / self.params["n_symbols"]

    def sample(self, n, rng, split="train") -> TaskBatch:
        self._check_split(split)
        p, T = self.params, self.seq_len
        tokens = np.empty((n, T), dtype=np.int64)
        mask = np.zeros((n, T - 1), dtype=bool)
        periods = rng.integers(p["min_period"], p["max_period"] + 1, size=n)
        for i, P in enumerate(periods):
            pattern = rng.choice(p["n_symbols"], size=P, replace=False) + 2
            tokens[i, 0] = BOS
            tokens[i, 1:] = np.resize(pattern, T - 1)
            mask[i, P + 1:] = True  # t indexes predictions; target content index == t
        return TaskBatch(torch.from_numpy(tokens), torch.from_numpy(mask), {"period": periods})

    def solve(self, batch):
        tok = batch.tokens.numpy()
        out = np.full(batch.loss_mask.shape, -1, dtype=np.int64)
        for i, t in zip(*np.nonzero(batch.loss_mask.numpy())):
            content = tok[i, 1: t + 1]  # content indices 0..t-1
            period = 1 + int(np.nonzero(content[1:] == content[0])[0][0])
            out[i, t] = content[t - period]
        return torch.from_numpy(out)

    def token_str(self, tok):
        return {PAD: "<pad>", BOS: "<bos>"}.get(int(tok), f"s{int(tok) - 2}")


# ============================================================================ Task 2
class AssocRecall(Task):
    """``<bos> k v k v ... [noise] <q> k* v*`` — retrieve the value bound to the queried key.

    Keys are distinct within a sequence; values are distinct too (so a symmetric corruption with a
    *different* answer always exists). Optional noise tokens are inserted between pairs
    (distractor robustness). ``query_index`` fixes which pair is queried (position studies).
    Scored position: the queried key; target: its value.
    """

    name = "assoc_recall"
    description = "Key-value retrieval from context (A 7 B 3 C 9 ? B -> 3)."
    defaults = {"n_keys": 16, "n_values": 16, "min_pairs": 2, "max_pairs": 6, "max_noise": 0,
                "min_noise": 0, "n_noise_symbols": 4, "holdout_mod": 0, "query_index": None}
    corruption_kinds = ("query_key",)

    def _validate(self) -> None:
        p = self.params
        if not (1 <= p["min_pairs"] <= p["max_pairs"] <= min(p["n_keys"], p["n_values"])):
            raise ValueError("need 1 <= min_pairs <= max_pairs <= min(n_keys, n_values)")
        if not (0 <= p["min_noise"] <= p["max_noise"]):
            raise ValueError("need 0 <= min_noise <= max_noise")
        m = p["holdout_mod"]
        if m:
            if m < 2 or m % 17 == 0 or p["n_values"] < m:
                raise ValueError("holdout_mod must be >= 2, not a multiple of 17, and <= n_values")
        if p["query_index"] is not None and not (0 <= p["query_index"] < p["min_pairs"]):
            raise ValueError("query_index must be < min_pairs")

    # token layout: 0 pad, 1 bos, 2 query, keys, values, noise
    QUERY = 2

    @property
    def key0(self):
        return 3

    @property
    def val0(self):
        return 3 + self.params["n_keys"]

    @property
    def noise0(self):
        return self.val0 + self.params["n_values"]

    @property
    def vocab_size(self) -> int:
        return self.noise0 + self.params["n_noise_symbols"]

    @property
    def seq_len(self) -> int:
        p = self.params
        return 1 + 2 * p["max_pairs"] + p["max_noise"] + 3

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / self.params["n_values"]

    def _sample_values(self, rng, keys, split, q):
        p = self.params
        m = p["holdout_mod"]
        used: set = set()
        vals = np.empty(len(keys), dtype=np.int64)
        order = [q] + [j for j in range(len(keys)) if j != q]  # choose queried value first
        for j in order:
            cand = np.array([v for v in range(p["n_values"]) if v not in used])
            held = _heldout(keys[j], cand, m) if m else np.zeros(len(cand), bool)
            if split == "eval" and j == q and m:
                cand = cand[held]
            elif m:
                cand = cand[~held]
            if cand.size == 0:
                return None
            vals[j] = rng.choice(cand)
            used.add(int(vals[j]))
        return vals

    def sample(self, n, rng, split="train") -> TaskBatch:
        self._check_split(split)
        p, T = self.params, self.seq_len
        tokens = np.full((n, T), PAD, dtype=np.int64)
        mask = np.zeros((n, T - 1), dtype=bool)
        meta = {k: np.zeros(n, dtype=np.int64) for k in
                ("n_pairs", "n_noise", "query_index", "query_pos", "evidence_pos", "distance", "answer")}
        for i in range(n):
            while True:
                k = int(rng.integers(p["min_pairs"], p["max_pairs"] + 1))
                keys = rng.choice(p["n_keys"], size=k, replace=False)
                q = p["query_index"] if p["query_index"] is not None else int(rng.integers(k))
                vals = self._sample_values(rng, keys, split, q)
                if vals is not None:
                    break
            n_noise = int(rng.integers(p["min_noise"], p["max_noise"] + 1))
            # units: pairs in order, noise tokens inserted at random boundaries
            units: list = [("pair", j) for j in range(k)]
            for _ in range(n_noise):
                units.insert(int(rng.integers(len(units) + 1)), ("noise", int(rng.integers(p["n_noise_symbols"]))))
            seq = [BOS]
            value_pos = {}
            for kind, j in units:
                if kind == "pair":
                    seq += [self.key0 + keys[j], self.val0 + vals[j]]
                    value_pos[j] = len(seq) - 1
                else:
                    seq.append(self.noise0 + j)
            seq += [self.QUERY, self.key0 + keys[q], self.val0 + vals[q]]
            tokens[i, : len(seq)] = seq
            qpos = len(seq) - 2
            mask[i, qpos] = True
            meta["n_pairs"][i], meta["n_noise"][i], meta["query_index"][i] = k, n_noise, q
            meta["query_pos"][i], meta["evidence_pos"][i] = qpos, value_pos[q]
            meta["distance"][i] = qpos - value_pos[q]
            meta["answer"][i] = self.val0 + vals[q]
        return TaskBatch(torch.from_numpy(tokens), torch.from_numpy(mask), meta)

    def corrupt(self, batch, rng, kind=None) -> TaskBatch:
        """Symmetric token replacement: query a *different* key that is present in context.

        The clean answer stays the target; ``alt_targets`` holds the corrupted query's answer.
        """
        kind = kind or "query_key"
        if kind != "query_key":
            raise ValueError(f"unknown corruption {kind!r}")
        tok = batch.tokens.clone().numpy()
        alt = np.full(batch.loss_mask.shape, -1, dtype=np.int64)
        for i in range(len(batch)):
            qpos = int(batch.meta["query_pos"][i])
            key = tok[i, qpos]
            ctx = tok[i, 1: qpos - 1]
            key_positions = [j + 1 for j, t in enumerate(ctx) if self.key0 <= t < self.val0 and t != key]
            j = int(rng.choice(key_positions))
            tok[i, qpos] = tok[i, j]
            alt[i, qpos] = tok[i, j + 1]
        meta = dict(batch.meta)
        return TaskBatch(torch.from_numpy(tok), batch.loss_mask.clone(), meta, torch.from_numpy(alt))

    def token_str(self, tok):
        tok = int(tok)
        if tok == self.QUERY:
            return "<q>"
        if self.key0 <= tok < self.val0:
            return f"k{tok - self.key0}"
        if self.val0 <= tok < self.noise0:
            return f"v{tok - self.val0}"
        if tok >= self.noise0:
            return f"n{tok - self.noise0}"
        return super().token_str(tok)


# ============================================================================ Task 3a
class InductionRepeat(Task):
    """``<bos> [random prefix] r_1..r_L r_1..r_L`` with L and the prefix length varying per example.

    The second copy is predictable only by finding the current token's earlier occurrence and
    copying its successor (induction). Varying L and the prefix makes a *fixed positional offset*
    useless — with a fixed L, a model could copy from position ``t - L + 1`` without any
    content-based matching, and induction-head claims would be confounded.

    Scored positions: second-copy tokens except its first (whose predecessor is r_L, unmatched).
    """

    name = "induction_repeat"
    description = "Random segment repeated after a random-length prefix; predict the second copy."
    defaults = {"n_symbols": 48, "seq_len": 40, "min_len": 6, "max_len": 16, "fixed_prefix_len": None,
                "train_on_all": False, "pair_mode": None}
    corruption_kinds = ("replace", "shuffle")
    PAIR_MODES = (None, "same_order", "shuffled")

    def _validate(self) -> None:
        p = self.params
        if not (2 <= p["min_len"] <= p["max_len"]):
            raise ValueError("need 2 <= min_len <= max_len")
        if 1 + 2 * p["max_len"] > p["seq_len"]:
            raise ValueError("seq_len must be >= 1 + 2 * max_len")
        if p["n_symbols"] < 2 * p["max_len"] + 1:
            raise ValueError("n_symbols must be >= 2 * max_len + 1 (prefix and corruptions need spare symbols)")
        if p["fixed_prefix_len"] is not None and p["fixed_prefix_len"] + 2 * p["max_len"] + 1 > p["seq_len"]:
            raise ValueError("fixed_prefix_len too long")
        if p["pair_mode"] not in self.PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {self.PAIR_MODES}")
        if p["pair_mode"] is not None:
            if p["train_on_all"] or p["fixed_prefix_len"] is not None:
                raise ValueError("pair_mode is incompatible with train_on_all / fixed_prefix_len")
            if (p["min_len"] + 1) // 2 > p["max_len"] // 2 or p["max_len"] < 4:
                raise ValueError("pair_mode needs an even segment length >= 4 within [min_len, max_len]")

    @property
    def vocab_size(self) -> int:
        return 2 + self.params["n_symbols"]

    @property
    def seq_len(self) -> int:
        return self.params["seq_len"]

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / self.params["n_symbols"]

    def sample(self, n, rng, split="train") -> TaskBatch:
        self._check_split(split)
        if self.params["pair_mode"] is not None:
            return self._sample_pairs(n, rng)
        p, T = self.params, self.seq_len
        S = p["n_symbols"]
        tokens = np.full((n, T), PAD, dtype=np.int64)
        mask = np.zeros((n, T - 1), dtype=bool)
        meta = {k: np.zeros(n, dtype=np.int64) for k in ("L", "first_start", "second_start", "prefix_len")}
        for i in range(n):
            L = int(rng.integers(p["min_len"], p["max_len"] + 1))
            P = T - 1 - 2 * L if p["fixed_prefix_len"] is None else p["fixed_prefix_len"]
            perm = rng.permutation(S)
            r, others = perm[:L] + 2, perm[L:] + 2
            prefix = rng.choice(others, size=P, replace=True)
            seq = np.concatenate([[BOS], prefix, r, r])
            tokens[i, : len(seq)] = seq
            a = 1 + P
            b = a + L
            mask[i, b: b + L - 1] = True  # targets r_2..r_L of the second copy
            meta["L"][i], meta["first_start"][i], meta["second_start"][i], meta["prefix_len"][i] = L, a, b, P
        train_mask = None
        if p["train_on_all"]:
            # every real (non-pad) next-token target, predictable or not
            train_mask = torch.from_numpy(tokens[:, 1:] != PAD)
        return TaskBatch(torch.from_numpy(tokens), torch.from_numpy(mask), meta, train_mask=train_mask)

    def _sample_pairs(self, n, rng) -> TaskBatch:
        """Pair-structured segments (``pair_mode``), used to test context-dependent retrieval.

        The segment is ``x1 y1 x2 y2 ... xP yP`` (all distinct). Only ``x_i -> y_i`` in the second
        segment is scored (the token after each ``x``), so both modes have identical targets,
        tokens and lengths. They differ only in the second segment's ORDER of pairs:

        * ``same_order``: second segment == first (a contiguous repeat); each ``x_i`` (i >= 2) is
          preceded by the same token at both occurrences.
        * ``shuffled``: pairs permuted so that EVERY ``x_i`` is preceded by a different token than
          at its first occurrence (rejection-sampled order). The preceding context is then
          uninformative, so single-token induction is the only consistent strategy.

        Single-token induction solves both modes perfectly; context matching helps nothing in
        ``same_order`` and can only hurt in ``shuffled``. (The default contiguous mode's random
        stream is untouched: this method is only called when ``pair_mode`` is set.)
        """
        p, T, S = self.params, self.seq_len, self.params["n_symbols"]
        tokens = np.full((n, T), PAD, dtype=np.int64)
        mask = np.zeros((n, T - 1), dtype=bool)
        meta = {k: np.zeros(n, dtype=np.int64) for k in ("L", "first_start", "second_start", "prefix_len", "n_pairs")}
        lo, hi = (p["min_len"] + 1) // 2, p["max_len"] // 2
        for i in range(n):
            P = int(rng.integers(lo, hi + 1))
            L = 2 * P
            pre = T - 1 - 2 * L
            perm = rng.permutation(S)
            seg, others = perm[:L] + 2, perm[L:] + 2
            prefix = rng.choice(others, size=pre, replace=True)
            order = np.arange(P)
            if p["pair_mode"] == "shuffled":
                for _ in range(10000):
                    order = rng.permutation(P)
                    # pair k is preceded by pair k-1 in the first segment; forbid that adjacency
                    if all(order[j] != order[j - 1] + 1 for j in range(1, P)):
                        break
                else:  # pragma: no cover - P >= 2 always admits a valid order
                    raise RuntimeError("no valid pair order")
            pairs = seg.reshape(P, 2)
            seq = np.concatenate([[BOS], prefix, seg, pairs[order].reshape(-1)])
            tokens[i, : len(seq)] = seq
            a, b = 1 + pre, 1 + pre + L
            mask[i, b: b + L: 2] = True  # every x in the second segment predicts its paired y
            meta["L"][i], meta["first_start"][i], meta["second_start"][i] = L, a, b
            meta["prefix_len"][i], meta["n_pairs"][i] = pre, P
        return TaskBatch(torch.from_numpy(tokens), torch.from_numpy(mask), meta)

    def corrupt(self, batch, rng, kind=None) -> TaskBatch:
        """``replace``: first copy -> fresh symbols (no earlier match; induction impossible).
        ``shuffle``: first copy permuted so every successor changes; an induction mechanism then
        predicts a *specific wrong* token, recorded in ``alt_targets`` (symmetric corruption).
        """
        kind = kind or ("shuffle" if self.params["pair_mode"] is None else "replace")
        if kind not in self.corruption_kinds:
            raise ValueError(f"unknown corruption {kind!r}")
        if kind == "shuffle" and self.params["pair_mode"] is not None:
            raise ValueError("'shuffle' corruption assumes a contiguous repeat; use 'replace' with pair_mode")
        tok = batch.tokens.clone().numpy()
        alt = np.full(batch.loss_mask.shape, -1, dtype=np.int64)
        S = self.params["n_symbols"]
        for i in range(len(batch)):
            L, a, b = (int(batch.meta[k][i]) for k in ("L", "first_start", "second_start"))
            r = tok[i, a: a + L].copy()
            if kind == "replace":
                used = set(tok[i, : b + L].tolist())
                fresh = np.array([s + 2 for s in range(S) if s + 2 not in used])
                if fresh.size < L:  # fall back to any symbol not in r
                    fresh = np.array([s + 2 for s in range(S) if s + 2 not in set(r.tolist())])
                tok[i, a: a + L] = rng.choice(fresh, size=L, replace=False)
            else:
                succ = {int(r[j]): int(r[j + 1]) for j in range(L - 1)}
                for _ in range(1000):
                    sh = rng.permutation(r)
                    if all(succ.get(int(sh[j])) != int(sh[j + 1]) for j in range(L - 1)):
                        break
                else:  # pragma: no cover - vanishingly unlikely for L >= 4
                    raise RuntimeError("could not derange successors")
                tok[i, a: a + L] = sh
                pos_in_sh = {int(x): j for j, x in enumerate(sh)}
                for t in range(b, b + L - 1):
                    j = pos_in_sh[int(tok[i, t])]
                    # successor in the corrupted first copy (the last element is followed by r_1 at b)
                    alt[i, t] = sh[j + 1] if j + 1 < L else tok[i, b]
        return TaskBatch(torch.from_numpy(tok), batch.loss_mask.clone(), dict(batch.meta),
                         torch.from_numpy(alt) if kind == "shuffle" else None)


# ============================================================================ Task 3b/4/5
class InductionPair(Task):
    """``<bos> ... A B ... A -> B`` with controllable evidence position and distractors.

    One evidence bigram (A, B) occurs in a filler sequence; the final token is A again and the
    model must output B. ``A`` occurs nowhere else, ``B`` nowhere else, so the answer is unique.

    * ``evidence_pos`` fixes A's index (position/distance studies, Task 4).
    * ``n_distractor_pairs`` adds other bigrams (C, D) that occur twice (competing induction
      opportunities); ``n_decoys`` adds copies of one decoy token so that "predict the most
      frequent token" fails (Task 5).
    """

    name = "induction_pair"
    description = "Single-pair induction: earlier 'A B', final 'A' -> predict 'B'."
    defaults = {"n_symbols": 32, "seq_len": 32, "holdout_mod": 0, "evidence_pos": None,
                "n_distractor_pairs": 0, "n_decoys": 0, "min_distance": 1}
    corruption_kinds = ("replace_answer", "remove_evidence")

    def _validate(self) -> None:
        p = self.params
        body = p["seq_len"] - 3
        need = 2 + 4 * p["n_distractor_pairs"] + p["n_decoys"]
        if body < need + 1:
            raise ValueError(f"seq_len too short: body {body} < {need + 1}")
        if p["n_symbols"] < 4 + 2 * p["n_distractor_pairs"] + 2:
            raise ValueError("not enough symbols")
        m = p["holdout_mod"]
        if m and (m < 2 or m % 17 == 0 or p["n_symbols"] < m):
            raise ValueError("bad holdout_mod")
        if p["evidence_pos"] is not None and not (1 <= p["evidence_pos"] <= p["seq_len"] - 4):
            raise ValueError("evidence_pos must be in [1, seq_len - 4]")

    @property
    def vocab_size(self) -> int:
        return 2 + self.params["n_symbols"]

    @property
    def seq_len(self) -> int:
        return self.params["seq_len"]

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / self.params["n_symbols"]

    def _pick_pair(self, rng, split):
        S, m = self.params["n_symbols"], self.params["holdout_mod"]
        while True:
            A, B = rng.choice(S, size=2, replace=False)
            h = bool(_heldout(A, B, m)) if m else False
            if not m or (split == "eval") == h:
                return int(A), int(B)

    def sample(self, n, rng, split="train") -> TaskBatch:
        self._check_split(split)
        p, T, S = self.params, self.seq_len, self.params["n_symbols"]
        qpos = T - 2
        tokens = np.full((n, T), PAD, dtype=np.int64)
        mask = np.zeros((n, T - 1), dtype=bool)
        meta = {k: np.zeros(n, dtype=np.int64) for k in ("evidence_pos", "distance", "answer", "decoy")}
        for i in range(n):
            A, B = self._pick_pair(rng, split)
            while True:
                occ = np.zeros(T, dtype=bool)
                occ[0] = occ[qpos] = occ[qpos + 1] = True
                seq = np.full(T, -1, dtype=np.int64)
                e = p["evidence_pos"] if p["evidence_pos"] is not None else int(
                    rng.integers(1, qpos - p["min_distance"]))
                if e + 1 >= qpos:
                    continue
                seq[e], seq[e + 1] = A, B
                occ[e] = occ[e + 1] = True
                pool = np.array([s for s in range(S) if s not in (A, B)])
                rng.shuffle(pool)
                ok = True
                for d in range(p["n_distractor_pairs"]):
                    C, D = pool[2 * d], pool[2 * d + 1]
                    for _ in range(2):
                        free = [j for j in range(1, qpos - 1) if not occ[j] and not occ[j + 1]]
                        if not free:
                            ok = False
                            break
                        j = int(rng.choice(free))
                        seq[j], seq[j + 1] = C, D
                        occ[j] = occ[j + 1] = True
                decoy = -1
                if ok and p["n_decoys"]:
                    decoy = int(pool[2 * p["n_distractor_pairs"]])
                    free = [j for j in range(1, qpos) if not occ[j]]
                    if len(free) < p["n_decoys"]:
                        ok = False
                    else:
                        for j in rng.choice(free, size=p["n_decoys"], replace=False):
                            seq[j] = decoy
                            occ[j] = True
                if ok:
                    break
            reserved = {A, B} | {int(x) for x in seq[occ] if x >= 0}
            filler_pool = np.array([s for s in range(S) if s not in reserved]) if p["n_distractor_pairs"] or p["n_decoys"] \
                else np.array([s for s in range(S) if s not in (A, B)])
            free = np.nonzero(~occ)[0]
            seq[free] = rng.choice(filler_pool, size=free.size, replace=True)
            seq[qpos], seq[qpos + 1] = A, B
            tokens[i] = seq + 2
            tokens[i, 0] = BOS
            mask[i, qpos] = True
            meta["evidence_pos"][i], meta["distance"][i] = e, qpos - (e + 1)
            meta["answer"][i], meta["decoy"][i] = B + 2, (decoy + 2) if decoy >= 0 else -1
        return TaskBatch(torch.from_numpy(tokens), torch.from_numpy(mask), meta)

    def corrupt(self, batch, rng, kind=None) -> TaskBatch:
        """``replace_answer`` (symmetric): earlier B -> C (absent from the sequence); an induction
        mechanism should now say C, recorded in ``alt_targets``.
        ``remove_evidence``: earlier A -> an absent symbol, so there is nothing to match.
        """
        kind = kind or "replace_answer"
        if kind not in self.corruption_kinds:
            raise ValueError(f"unknown corruption {kind!r}")
        tok = batch.tokens.clone().numpy()
        alt = np.full(batch.loss_mask.shape, -1, dtype=np.int64)
        S = self.params["n_symbols"]
        qpos = self.seq_len - 2
        for i in range(len(batch)):
            e = int(batch.meta["evidence_pos"][i])
            absent = np.array([s + 2 for s in range(S) if s + 2 not in set(tok[i].tolist())])
            c = int(rng.choice(absent))
            if kind == "replace_answer":
                tok[i, e + 1] = c
                alt[i, qpos] = c
            else:
                tok[i, e] = c
        return TaskBatch(torch.from_numpy(tok), batch.loss_mask.clone(), dict(batch.meta),
                         torch.from_numpy(alt) if kind == "replace_answer" else None)


# ============================================================================ evaluation-only
class ContextCompetition(Task):
    """Two earlier occurrences of the query token that differ only in their PRECEDING context.

    Layout (``context_len=1``)::

        <bos> filler ... [y x A] ... [z x B] ... filler [y x] A
                         match triple   non-match         query

    The query ``x`` (scored, second-to-last position) is preceded by ``y``. ``x`` occurred twice
    before: once preceded by ``y`` and followed by ``A`` (the *context-matching* candidate) and
    once preceded by ``z`` and followed by ``B``. Target = ``A``; ``alt_targets`` = ``B``. All of
    ``x, y, z, A, B`` and the filler symbols are distinct (filler drawn without replacement), so
    there are no other repeats.

    * Single-token induction (match on ``x`` only) has no reason to prefer ``A`` over ``B``.
    * Context-matching retrieval (``y`` must match as well) prefers ``A``.
    * Recency / position biases are cancelled by counterbalancing ``match_recent``: by default
      even-indexed examples put the matching triple LATER (more recent), odd-indexed EARLIER.

    Primary metric: logit difference ``logit[A] - logit[B]`` at the query ("context preference").
    ``context_len=k`` (1-3) makes k preceding tokens match (k=2: ``w y x A`` vs ``v z x B``, query ``w y x``).
    Evaluation-only: never used for training.
    """

    name = "context_competition"
    description = "Query token occurs twice earlier; only one occurrence shares the query's preceding context."
    defaults = {"n_symbols": 48, "seq_len": 40, "match_recent": None, "context_len": 1, "min_sep": 1}

    def _validate(self) -> None:
        p = self.params
        k = p["context_len"]
        if k not in (1, 2, 3):
            raise ValueError("context_len must be 1, 2 or 3")
        trip = k + 2
        body = p["seq_len"] - (k + 2) - 1
        if body < 2 * trip + p["min_sep"]:
            raise ValueError("seq_len too short")
        if p["n_symbols"] < 3 + 2 * k + (body - 2 * trip):
            raise ValueError("not enough symbols for distinct filler")
        if p["match_recent"] not in (None, True, False):
            raise ValueError("match_recent must be null, true or false")

    @property
    def vocab_size(self) -> int:
        return 2 + self.params["n_symbols"]

    @property
    def seq_len(self) -> int:
        return self.params["seq_len"]

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / self.params["n_symbols"]

    def sample(self, n, rng, split="train") -> TaskBatch:
        self._check_split(split)
        p, T, S, k = self.params, self.seq_len, self.params["n_symbols"], self.params["context_len"]
        trip = k + 2  # context tokens + x + successor
        q_start = T - (k + 2)  # query block: context tokens, x, then the answer A at T-1
        body = q_start - 1  # body positions 1 .. q_start-1
        tokens = np.zeros((n, T), dtype=np.int64)
        mask = np.zeros((n, T - 1), dtype=bool)
        alt = np.full((n, T - 1), -1, dtype=np.int64)
        meta = {key: np.zeros(n, dtype=np.int64) for key in
                ("match_recent", "pos_match", "pos_nonmatch", "dist_match", "dist_nonmatch", "answer", "alt")}
        for i in range(n):
            mr = (i % 2 == 0) if p["match_recent"] is None else bool(p["match_recent"])
            syms = rng.permutation(S) + 2
            ctx_m, ctx_n = syms[:k], syms[k:2 * k]
            x, A, B = syms[2 * k], syms[2 * k + 1], syms[2 * k + 2]
            filler = syms[2 * k + 3:]
            m_trip = np.concatenate([ctx_m, [x, A]])
            n_trip = np.concatenate([ctx_n, [x, B]])
            # start positions s1 < s2 inside the body, triples separated by >= min_sep filler tokens
            s1 = int(rng.integers(1, body - 2 * trip - p["min_sep"] + 2))
            s2 = int(rng.integers(s1 + trip + p["min_sep"], body - trip + 2))
            seq = np.zeros(T, dtype=np.int64)
            seq[0] = BOS
            first, second = (n_trip, m_trip) if mr else (m_trip, n_trip)
            seq[s1: s1 + trip], seq[s2: s2 + trip] = first, second
            free = [j for j in range(1, q_start) if not (s1 <= j < s1 + trip or s2 <= j < s2 + trip)]
            seq[free] = filler[: len(free)]
            seq[q_start: q_start + k] = ctx_m
            seq[T - 2], seq[T - 1] = x, A
            tokens[i] = seq
            mask[i, T - 2] = True
            alt[i, T - 2] = B
            pm, pn = (s2, s1) if mr else (s1, s2)
            meta["match_recent"][i], meta["pos_match"][i], meta["pos_nonmatch"][i] = int(mr), pm, pn
            meta["dist_match"][i] = (T - 2) - (pm + k)
            meta["dist_nonmatch"][i] = (T - 2) - (pn + k)
            meta["answer"][i], meta["alt"][i] = A, B
        return TaskBatch(torch.from_numpy(tokens), torch.from_numpy(mask), meta, torch.from_numpy(alt))

    corruption_kinds = ("swap_context",)

    def corrupt(self, batch, rng, kind=None) -> TaskBatch:
        """``swap_context``: replace the query's preceding context (``y``) by the NON-matching
        occurrence's context (``z``). Nothing else changes (all earlier positions are identical, so
        keys/values at earlier positions are unaffected under causal attention). A context matcher's
        preference flips from A to B; the targets (A) and alternatives (B) are kept, so the logit
        difference measures the flip."""
        kind = kind or "swap_context"
        if kind != "swap_context":
            raise ValueError(f"unknown corruption {kind!r}")
        k, T = self.params["context_len"], self.seq_len
        tok = batch.tokens.clone()
        for i in range(len(batch)):
            pn = int(batch.meta["pos_nonmatch"][i])
            tok[i, T - 2 - k: T - 2] = batch.tokens[i, pn: pn + k]
        return TaskBatch(tok, batch.loss_mask.clone(), dict(batch.meta),
                         None if batch.alt_targets is None else batch.alt_targets.clone())

    def solve(self, batch):
        """Exact *context-matching* rule: copy the successor of the earlier occurrence of the current
        token whose preceding ``context_len`` tokens also match (the single-token rule is ambiguous
        here by construction)."""
        k = self.params["context_len"]
        tok = batch.tokens.numpy()
        out = np.full(batch.loss_mask.shape, -1, dtype=np.int64)
        for i, t in zip(*np.nonzero(batch.loss_mask.numpy())):
            ctx = tok[i, t - k: t + 1]
            for j in range(k, t):
                if np.array_equal(tok[i, j - k: j + 1], ctx):
                    out[i, t] = tok[i, j + 1]
        return torch.from_numpy(out)


TASKS: dict[str, type[Task]] = {cls.name: cls for cls in
                                (RepeatPattern, AssocRecall, InductionRepeat, InductionPair, ContextCompetition)}



def make_task(name: str, params: Optional[dict] = None) -> Task:
    try:
        cls = TASKS[name]
    except KeyError:
        raise ValueError(f"Unknown task {name!r}; available: {sorted(TASKS)}") from None
    return cls(**(params or {}))
