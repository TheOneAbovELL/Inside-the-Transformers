"""Matplotlib figures. Every figure carries a context line identifying the run, checkpoint,
example(s) and experimental condition, so a saved PNG is traceable on its own."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless, works on any machine
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ATTN_CAVEAT = "Attention weights are observational: they do not by themselves show causal importance."


def _context(fig, context: str) -> None:
    fig.text(0.01, 0.005, context, fontsize=7, color="#555", ha="left", va="bottom", wrap=True)


def attention_pattern(pattern: np.ndarray, tokens: Sequence[str], title: str, context: str,
                      highlight_query: Optional[int] = None):
    """``pattern [q, k]`` for one head on one example."""
    s = pattern.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, s * 0.22), max(4.5, s * 0.2)))
    im = ax.imshow(pattern, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(s))
    ax.set_yticks(range(s))
    ax.set_xticklabels(tokens[:s], rotation=90, fontsize=6)
    ax.set_yticklabels([f"{i}:{t}" for i, t in enumerate(tokens[:s])], fontsize=6)
    ax.set_xlabel("key (attended-to) position")
    ax.set_ylabel("query position")
    if highlight_query is not None:
        ax.axhline(highlight_query, color="red", lw=0.6)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title + "\n" + ATTN_CAVEAT, fontsize=8)
    _context(fig, context)
    return fig


def head_grid(grid: np.ndarray, title: str, context: str, cmap: str = "viridis", center_zero: bool = False,
              fmt: str = "{:.2f}", cbar_label: str = ""):
    """``[n_layers, n_heads]`` heatmap with annotated values."""
    L, H = grid.shape
    fig, ax = plt.subplots(figsize=(1.2 + H * 0.9, 1.0 + L * 0.8))
    if center_zero:
        v = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
        v = v or 1.0
        im = ax.imshow(grid, cmap="RdBu", vmin=-v, vmax=v)
    else:
        im = ax.imshow(grid, cmap=cmap)
    for l in range(L):
        for h in range(H):
            if np.isfinite(grid[l, h]):
                ax.text(h, l, fmt.format(grid[l, h]), ha="center", va="center", fontsize=8,
                        color="black" if center_zero else "white")
    ax.set_xticks(range(H))
    ax.set_yticks(range(L))
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(title, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label(cbar_label, fontsize=7)
    _context(fig, context)
    return fig


def effect_bars(labels: Sequence[str], means, los, his, title: str, context: str, ylabel: str,
                control: Optional[Sequence[float]] = None):
    """Bar chart of intervention effects with 95% CIs (and optional control distribution)."""
    means, los, his = map(np.asarray, (means, los, his))
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.45), 3.6))
    x = np.arange(len(labels))
    ax.bar(x, means, color=["#c44" if m < 0 else "#48c" for m in means])
    ax.errorbar(x, means, yerr=[means - los, his - means], fmt="none", ecolor="black", capsize=2, lw=0.8)
    if control is not None and len(control):
        ax.axhspan(np.min(control), np.max(control), color="grey", alpha=0.2, label="range of control effects")
        ax.legend(fontsize=7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    _context(fig, context)
    return fig


def curves(xs, series: dict, title: str, context: str, xlabel: str, ylabel: str, logx: bool = False,
           bands: Optional[dict] = None):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", ms=3, label=name)
        if bands and name in bands:
            lo, hi = bands[name]
            ax.fill_between(xs, lo, hi, alpha=0.2)
    if logx:
        ax.set_xscale("symlog", linthresh=50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=9)
    _context(fig, context)
    return fig


def region_heatmap(rows: list[dict], value: str, title: str, context: str):
    """Rows of region patching -> heatmap (component@layer x region)."""
    comps = sorted({(r["component"], r["layer"]) for r in rows}, key=lambda c: (c[1], c[0]))
    regions = list(dict.fromkeys(r["region"] for r in rows))
    g = np.full((len(comps), len(regions)), np.nan)
    for r in rows:
        g[comps.index((r["component"], r["layer"])), regions.index(r["region"])] = r[value]
    fig, ax = plt.subplots(figsize=(1.5 + len(regions) * 1.1, 1.0 + len(comps) * 0.45))
    im = ax.imshow(g, cmap="RdBu", vmin=-1, vmax=1)
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if np.isfinite(g[i, j]):
                ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=30, fontsize=7)
    ax.set_yticks(range(len(comps)))
    ax.set_yticklabels([f"{c}@L{l}" for c, l in comps], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.04).set_label("recovery R", fontsize=7)
    ax.set_title(title, fontsize=9)
    _context(fig, context)
    return fig
