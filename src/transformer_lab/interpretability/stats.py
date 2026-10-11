"""Uncertainty estimates.

We use the non-parametric percentile bootstrap (Efron & Tibshirani, 1993) over *examples* for
means and paired differences. Rationale: per-example effects of interventions are often skewed
and bounded (accuracies, probabilities), so normal-theory intervals can be misleading, and we
care about the population of task instances rather than a significance ritual.

Seed-level variation (different trained models) is reported separately as dispersion across
seeds — with 3 seeds a CI would be nearly meaningless, so we show every seed's value.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def bootstrap_ci(values, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Mean and percentile-bootstrap CI of ``values`` (1-D, NaNs dropped)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "std": float("nan")}
    if n == 1:
        v = float(x[0])
        return {"mean": v, "lo": v, "hi": v, "n": 1, "std": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return {
        "mean": float(x.mean()),
        "lo": float(np.quantile(means, alpha)),
        "hi": float(np.quantile(means, 1 - alpha)),
        "n": int(n),
        "std": float(x.std(ddof=1)),
    }


def paired_difference(treated, baseline, **kw) -> dict:
    """Bootstrap CI of the mean per-example difference ``treated - baseline``.

    Pairing (same examples under both conditions) removes between-example variance, which is
    why every intervention in this lab is evaluated on the *same* inputs as its baseline.
    """
    t = np.asarray(treated, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if t.shape != b.shape:
        raise ValueError("paired arrays must have identical shapes")
    return bootstrap_ci(t - b, **kw)


def fmt_ci(stat: dict, digits: int = 3) -> str:
    return f"{stat['mean']:.{digits}f} [{stat['lo']:.{digits}f}, {stat['hi']:.{digits}f}] (n={stat['n']})"


def empirical_quantile_rank(value: float, null_samples, higher_is_extreme: bool = True) -> Optional[float]:
    """Fraction of control samples at least as extreme as ``value`` (a permutation-style p-value
    with the +1 correction). Returns None if there are no control samples."""
    s = np.asarray(null_samples, dtype=np.float64)
    if s.size == 0:
        return None
    extreme = (s >= value) if higher_is_extreme else (s <= value)
    return float((extreme.sum() + 1) / (s.size + 1))
