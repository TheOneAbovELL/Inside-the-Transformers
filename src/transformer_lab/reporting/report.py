"""Markdown report helpers.

Reports keep five things apart, always in this order:
**Setup** → **Observations** (raw numbers) → **Statistics** (CIs, controls) →
**Interpretation** (rule-based, explicitly labelled, conservative) → **Limitations**.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

STANDARD_LIMITATIONS = [
    "Small synthetic task and tiny model: findings need not transfer to large language models.",
    "Effects are measured under the stated intervention, examples and checkpoint only.",
    "95% intervals are percentile bootstraps over examples; they ignore variability across training seeds "
    "unless several runs are listed.",
]


def _fmt(v, digits=3):
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.{digits}f}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x, digits) for x in v)
    return str(v)


def table_md(rows: Sequence[dict], cols: Optional[Sequence[str]] = None, digits: int = 3) -> str:
    if not rows:
        return "_(no rows)_\n"
    if cols is None:
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
    out = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    for r in rows:
        out += "| " + " | ".join(_fmt(r.get(c, ""), digits) for c in cols) + " |\n"
    return out


def ci_str(stat: dict, digits: int = 3) -> str:
    return f"{stat['mean']:.{digits}f} [{stat['lo']:.{digits}f}, {stat['hi']:.{digits}f}]"


def build_report(title: str, setup: dict, observations: str, statistics: str = "", interpretation: Sequence[str] = (),
                 limitations: Sequence[str] = (), plots: Sequence[str] = ()) -> str:
    md = f"# {title}\n\n## Setup\n\n"
    for k, v in setup.items():
        md += f"- **{k}**: {v}\n"
    md += f"\n## Observations\n\n{observations}\n"
    if plots:
        md += "\n" + "\n".join(f"![{p}](plots/{p}.png)" for p in plots) + "\n"
    if statistics:
        md += f"\n## Statistical summary\n\n{statistics}\n"
    md += "\n## Interpretation (automated, rule-based — read with the limitations)\n\n"
    md += "\n".join(f"- {s}" for s in interpretation) if interpretation else "- No interpretation generated."
    md += "\n\n## Limitations\n\n" + "\n".join(f"- {s}" for s in list(limitations) + STANDARD_LIMITATIONS) + "\n"
    return md
