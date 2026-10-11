"""Circuit-hypothesis testing framework.

A hypothesis is written down *before* looking at the results (YAML, see
``configs/hypotheses/``) as:

1. a behavioural phenomenon and metric;
2. candidate components — either named explicitly or selected by a declared rule (e.g. "the
   layer-1 head with the highest induction score") on a **selection set** of examples;
3. falsifiable predictions, each an intervention/measurement with an expected outcome and an
   optional control distribution;
4. alternative explanations.

Predictions are evaluated on a **disjoint test set** (different seed) so that choosing the
candidates cannot inflate the evidence for them. Each prediction gets a verdict:

* ``supported``      — the 95% CI lies entirely on the predicted side of the threshold and, if a
                       control is defined, the effect is more extreme than the control
                       distribution (empirical p <= alpha);
* ``contradicted``   — the CI lies entirely on the other side of the threshold;
* ``inconclusive``   — anything else (CI straddles the threshold, control not exceeded, or the
                       patching experiment is uninformative).

The framework never assumes the hypothesis is true: it reports every prediction, including
failures, and the overall summary is a count of verdicts — not a single "circuit found" flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

from ..data.tasks import Task, TaskBatch
from ..evaluation.metrics import run_batched
from .attention_scores import (head_scores, induction_score_per_example, prev_token_score_per_example,
                               query_mask_from_batch)
from .interventions import Reference, head_ablation_hooks, mlp_ablation_hooks, reference_names
from .patching import Patcher, PatchSpec, _ratio_bootstrap, task_regions
from .path import mean_head_results, path_effect
from .scans import random_head_control
from .stats import bootstrap_ci, empirical_quantile_rank

VERDICTS = ("supported", "contradicted", "inconclusive")


@dataclass
class HypothesisSpec:
    name: str
    statement: str
    behavior: dict
    candidates: dict
    predictions: list
    alternatives: list = field(default_factory=list)
    n_examples: int = 512
    selection_seed: int = 101
    test_seed: int = 202
    reference_seed: int = 303
    ablation_method: str = "mean"
    alpha: float = 0.05

    @classmethod
    def load(cls, path) -> "HypothesisSpec":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        allowed = set(cls.__dataclass_fields__)
        unknown = set(d) - allowed
        if unknown:
            raise ValueError(f"Unknown hypothesis keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ------------------------------------------------------------------------- candidates
def select_candidates(model, batch: TaskBatch, spec: dict) -> tuple[dict, dict]:
    """Resolve candidate groups -> list of (layer, head). Returns (groups, selection_scores)."""
    scores = head_scores(model, batch)
    groups: dict[str, list] = {}
    for name, rule in spec.items():
        if "heads" in rule:
            groups[name] = [_parse_component(h) for h in rule["heads"]]
            continue
        sel = rule.get("select")
        if sel == "layer":  # every head of a layer
            groups[name] = [(int(rule["layer"]), h) for h in range(model.cfg.n_heads)]
            continue
        if sel == "threshold":  # every head (optionally in one layer) whose score >= min_score
            s = scores[rule["score"]]
            layers = [rule["layer"]] if rule.get("layer") is not None else range(s.shape[0])
            groups[name] = [(int(l), int(h)) for l in layers for h in range(s.shape[1])
                            if np.isfinite(s[l, h]) and s[l, h] >= float(rule["min_score"])]
            if not groups[name]:
                raise ValueError(f"candidate group {name!r}: no head reaches {rule['score']} >= {rule['min_score']}")
            continue
        if sel != "top":
            raise ValueError(f"candidate {name}: need 'heads' or select: top | threshold | layer")
        s = scores[rule["score"]].copy()
        layer = rule.get("layer")
        if layer is not None:
            mask = np.zeros_like(s, dtype=bool)
            mask[layer] = True
            s[~mask] = -np.inf
        k = int(rule.get("k", 1))
        flat = np.argsort(-np.nan_to_num(s, nan=-np.inf), axis=None)[:k]
        groups[name] = [tuple(int(v) for v in np.unravel_index(i, s.shape)) for i in flat]
    return groups, {k: v.tolist() for k, v in scores.items()}


def _parse_component(text: str) -> tuple:
    """``"1.2"`` -> (1, 2) (a head); ``"0.mlp"`` -> (0, "mlp") (an MLP block, path senders only)."""
    l, h = str(text).split(".")
    return (int(l), "mlp") if h == "mlp" else (int(l), int(h))


def _resolve_heads(ref, groups) -> list:
    if isinstance(ref, str):
        if ref in groups:
            return list(groups[ref])
        return [_parse_component(ref)]
    out = []
    for r in ref:
        out += _resolve_heads(r, groups)
    return out


# --------------------------------------------------------------------------- verdicts
def _effect_verdict(ci: dict, expect: dict, control_p: Optional[float], alpha: float,
                    control: Optional[dict] = None) -> tuple[str, str]:
    lo, hi = ci["lo"], ci["hi"]
    lab = ci.get("label", "95% CI")
    direction = expect.get("direction", "decrease")
    if direction == "none":
        tol = float(expect["max_effect"])
        if -tol <= lo and hi <= tol:
            return "supported", f"{lab} [{lo:.3f}, {hi:.3f}] within ±{tol}"
        if hi < -tol or lo > tol:
            return "contradicted", f"{lab} [{lo:.3f}, {hi:.3f}] entirely outside ±{tol}"
        return "inconclusive", f"{lab} [{lo:.3f}, {hi:.3f}] overlaps the ±{tol} boundary"
    thr = float(expect.get("min_effect", 0.0))
    if direction == "decrease":
        ok, bad = hi < -thr, lo > -thr
        txt = f"effect {lab} [{lo:.3f}, {hi:.3f}] vs required decrease > {thr}"
    elif direction == "increase":
        ok, bad = lo > thr, hi < thr
        txt = f"effect {lab} [{lo:.3f}, {hi:.3f}] vs required increase > {thr}"
    else:
        raise ValueError(f"unknown direction {direction}")
    if bad:
        return "contradicted", txt
    if not ok:
        return "inconclusive", txt
    if control_p is None:
        return "supported", txt
    if control_p <= alpha:
        return "supported", txt + f"; more extreme than controls (p={control_p:.3f})"
    if control and control.get("exhaustive") and control["n"] > 0:
        # every alternative component set was tested; p cannot reach alpha, so require the whole CI
        # to be beyond the most extreme control effect instead
        beyond = hi < control["min"] if direction == "decrease" else lo > control["max"]
        floor = 1 / (control["n"] + 1)
        if beyond:
            return "supported", txt + (f"; CI beyond ALL {control['n']} possible control sets "
                                       f"(exhaustive; p floor {floor:.3f} > alpha, so this criterion is used)")
        return "inconclusive", txt + f"; not beyond all {control['n']} exhaustive control sets"
    return "inconclusive", txt + f"; but not more extreme than controls (p={control_p:.3f})"


def _threshold_verdict(ci: dict, expect: dict) -> tuple[str, str]:
    op, v = expect["op"], float(expect["value"])
    lo, hi = ci["lo"], ci["hi"]
    if op == ">":
        res = "supported" if lo > v else ("contradicted" if hi < v else "inconclusive")
    elif op == "<":
        res = "supported" if hi < v else ("contradicted" if lo > v else "inconclusive")
    else:
        raise ValueError(op)
    return res, f"{ci.get('label', '95% CI')} [{lo:.3f}, {hi:.3f}] vs {op} {v}"


def _ci(values, level: float) -> dict:
    c = bootstrap_ci(values, ci=level)
    c["label"] = f"{100 * level:.1f}% CI"
    return c


# ------------------------------------------------------------------------------ runner
class HypothesisTest:
    def __init__(self, model, task: Task, spec: HypothesisSpec) -> None:
        self.model, self.task, self.spec = model, task, spec
        model.eval()
        n = spec.n_examples
        self.selection = task.sample(n, np.random.default_rng(spec.selection_seed), "train")
        self.test = task.sample(n, np.random.default_rng(spec.test_seed), "train")
        ref_batch = task.sample(n, np.random.default_rng(spec.reference_seed), "train")
        self.ref_batch = ref_batch
        self.ref = Reference.from_batch(model, ref_batch, reference_names(model, heads=True, mlps=True))
        self.metric = spec.behavior.get("metric", "logprob")
        self.base = run_batched(model, self.test)
        # Integrity: selection / test / reference sets must not share sequences (double dipping)
        rows = lambda b: {r.tobytes() for r in b.tokens.numpy()}
        sel, tst, ref = rows(self.selection), rows(self.test), rows(ref_batch)
        self.overlap = {"selection_test": len(sel & tst), "reference_test": len(ref & tst)}

    def run(self) -> dict:
        groups, sel_scores = select_candidates(self.model, self.selection, self.spec.candidates)
        results = []
        m = len(self.spec.predictions)
        fw_level = 1 - self.spec.alpha / m  # Bonferroni-adjusted interval level for the family
        for pred in self.spec.predictions:
            try:
                r = self._run_prediction(pred, groups)
                fw = r.pop("_recompute", None)
                if fw is not None:  # same rule at the family-wise interval: a sensitivity analysis
                    r["verdict_familywise"], r["reason_familywise"] = fw(fw_level)
                results.append(r)
            except Exception as e:  # a broken prediction is reported, not hidden
                results.append({"id": pred.get("id"), "claim": pred.get("claim"), "verdict": "inconclusive",
                                "reason": f"ERROR: {type(e).__name__}: {e}"})
        # Dependencies: a prediction listing ``requires: [ids]`` is only meaningful if those were
        # supported (e.g. "path Q has little effect" is vacuous when path K had no effect either).
        verdict_of = {r["id"]: r["verdict"] for r in results}
        for pred, r in zip(self.spec.predictions, results):
            unmet = [i for i in pred.get("requires", []) if verdict_of.get(i) != "supported"]
            if unmet and r["verdict"] != "inconclusive":
                r["reason"] = (f"prerequisite(s) {unmet} not supported, so this prediction is not interpretable "
                               f"(raw verdict was {r['verdict']}: {r.get('reason', '')})")
                r["raw_verdict"], r["verdict"] = r["verdict"], "inconclusive"
        for pred, r in zip(self.spec.predictions, results):  # dependencies apply family-wise too
            unmet = [i for i in pred.get("requires", []) if verdict_of.get(i) != "supported"]
            if unmet and r.get("verdict_familywise") not in (None, "inconclusive"):
                r["verdict_familywise"] = "inconclusive"
        counts = {v: sum(r["verdict"] == v for r in results) for v in VERDICTS}
        counts_fw = {v: sum(r.get("verdict_familywise", r["verdict"]) == v for r in results) for v in VERDICTS}
        base = bootstrap_ci(self.base[self.metric])
        return {
            "hypothesis": self.spec.to_dict(),
            "candidates": {k: [_label(c) for c in v] for k, v in groups.items()},
            "selection_scores": sel_scores,
            "baseline": {"metric": self.metric, **base, "accuracy": float(self.base["accuracy"].mean())},
            "predictions": results,
            "verdict_counts": counts,
            "verdict_counts_familywise": counts_fw,
            "familywise_interval_level": fw_level,
            "set_overlap": self.overlap,
            "n_test": len(self.test),
        }

    # ------------------------------------------------------------------ predictions
    def _run_prediction(self, pred: dict, groups: dict) -> dict:
        test = pred["test"]
        kind = test["type"]
        expect = pred.get("expect", {})
        metric = test.get("metric", self.metric)
        out = {"id": pred.get("id"), "claim": pred.get("claim", ""), "test": test, "expect": expect}

        if kind == "attention_score":
            heads = _resolve_heads(test["heads"], groups)
            qm = query_mask_from_batch(self.test)
            _, cache = self.model.run_with_cache(self.test.tokens, names_filter=lambda n: n.endswith("hook_pattern"))
            vals = []
            for l, h in heads:
                p = cache["pattern", l]
                fn = {"induction": lambda: induction_score_per_example(p, self.test.tokens, qm),
                      "prev_token": lambda: prev_token_score_per_example(p, qm)}[test["score"]]
                vals.append(fn()[:, h].numpy())
            v = np.mean(vals, 0)
            ci = bootstrap_ci(v)
            out["observed"] = ci
            out["verdict"], out["reason"] = _threshold_verdict(ci, expect)
            out["_recompute"] = lambda lvl: _threshold_verdict(_ci(v, lvl), expect)
            return out

        if kind in ("ablate_heads", "ablate_mlps"):
            if kind == "ablate_heads":
                heads = _resolve_heads(test["heads"], groups)
                hooks = head_ablation_hooks(heads, test.get("method", self.spec.ablation_method), self.ref)
                out["components"] = [f"L{l}H{h}" for l, h in heads]
            else:
                hooks = mlp_ablation_hooks(test["layers"], test.get("method", self.spec.ablation_method), self.ref)
                out["components"] = [f"L{l}.mlp" for l in test["layers"]]
            treated = run_batched(self.model, self.test, fwd_hooks_fn=lambda sl: hooks)
            eff = treated[metric] - self.base[metric]
            ci = bootstrap_ci(eff)
            out["observed"] = ci
            out["treated_accuracy"] = float(treated["accuracy"].mean())
            control_p = None
            ctrl = pred.get("control")
            if ctrl:
                excl = _resolve_heads(ctrl.get("exclude", []), groups)
                n_heads = len(_resolve_heads(test["heads"], groups)) if kind == "ablate_heads" else 1
                effects, exhaustive = random_head_control(
                    self.model, self.test, self.ref, n_heads, int(ctrl.get("n_samples", 20)), exclude=excl,
                    method=test.get("method", self.spec.ablation_method), metric=metric,
                    seed=int(ctrl.get("seed", 0)), base=self.base)
                direction = expect.get("direction", "decrease")
                control_p = empirical_quantile_rank(ci["mean"], effects, higher_is_extreme=(direction == "increase"))
                out["control"] = {"type": "random_heads", "n": int(effects.size), "exhaustive": exhaustive,
                                  "effects": effects.tolist(),
                                  "mean": float(effects.mean()) if effects.size else None,
                                  "min": float(effects.min()) if effects.size else None,
                                  "max": float(effects.max()) if effects.size else None, "p": control_p}
            out["verdict"], out["reason"] = _effect_verdict(ci, expect, control_p, self.spec.alpha,
                                                            out.get("control"))
            out["_recompute"] = lambda lvl: _effect_verdict(_ci(eff, lvl), expect, control_p, self.spec.alpha,
                                                            out.get("control"))
            return out

        if kind == "path":
            senders = _resolve_heads(test["sender"], groups)
            receivers = _resolve_heads(test["receiver"], groups)
            rep = mean_head_results(self.model, self.ref_batch, senders) if test.get("replacement", "mean") == "mean" else None
            r = path_effect(self.model, self.test, senders, receivers, test["channel"], replacement=rep, metric=metric)
            d = r["treated"] - r["baseline"]
            ci = bootstrap_ci(d)
            out["observed"] = ci
            out["_recompute"] = lambda lvl: _effect_verdict(_ci(d, lvl), expect, None, self.spec.alpha)
            out["components"] = [f"{'+'.join(_label(c) for c in senders)} -> "
                                 f"{'+'.join(f'L{l}H{h}' for l, h in receivers)}.{test['channel']}"]
            out["verdict"], out["reason"] = _effect_verdict(ci, expect, None, self.spec.alpha)
            return out

        if kind == "patch_heads":
            heads = _resolve_heads(test["heads"], groups)
            corrupt = self.task.corrupt(self.test, np.random.default_rng(int(test.get("seed", 0))), test.get("corruption"))
            patcher = Patcher(self.model, self.test, corrupt, metric=test.get("metric", "logprob"),
                              direction=test.get("direction", "denoise"), min_gap=float(test.get("min_gap", 0.25)),
                              names=sorted({f"blocks.{l}.attn.hook_z" for l, _ in heads}))
            positions = task_regions(self.task, self.test)[test.get("positions", "scored")] \
                if test.get("positions", "scored") != "all" else None
            # patch all listed heads simultaneously (possibly across layers)
            by_layer: dict = {}
            for l, h in heads:
                by_layer.setdefault(l, []).append(h)
            specs = [PatchSpec(f"L{l}", f"blocks.{l}.attn.hook_z", heads=hs, positions=positions)
                     for l, hs in by_layer.items()]
            res = patcher.run(specs)
            out["components"] = [f"L{l}H{h}" for l, h in heads]
            out["observed"] = {"clean": res.clean, "corrupt": res.corrupt, "patched": res.patched,
                               "recovery": res.recovery, "lo": res.recovery_ci[0], "hi": res.recovery_ci[1],
                               "gap": res.gap, "informative": res.informative, "n": res.n}
            if not res.informative:
                out["verdict"], out["reason"] = "inconclusive", f"uninformative: clean-corrupt gap {res.gap:.3f} < min_gap"
            else:
                thr = float(expect.get("recovery_min", 0.5))
                if res.recovery_ci[0] >= thr:
                    out["verdict"] = "supported"
                elif res.recovery_ci[1] < thr:
                    out["verdict"] = "contradicted"
                else:
                    out["verdict"] = "inconclusive"
                out["reason"] = f"recovery {res.recovery:.3f} CI [{res.recovery_ci[0]:.3f}, {res.recovery_ci[1]:.3f}] vs >= {thr}"

                def fw(lvl, res=res, thr=thr, patcher=patcher):
                    lo, hi = _ratio_bootstrap(res.patched_per_example, patcher.m_clean, patcher.m_corrupt, level=lvl)
                    v = "supported" if lo >= thr else ("contradicted" if hi < thr else "inconclusive")
                    return v, f"recovery {res.recovery:.3f} {100 * lvl:.1f}% CI [{lo:.3f}, {hi:.3f}] vs >= {thr}"
                out["_recompute"] = fw
            for k in ("median_per_example_recovery", "frac_examples_over_recovered", "n_examples_with_gap"):
                if k in res.meta:
                    out["observed"][k] = res.meta[k]
            return out

        raise ValueError(f"unknown test type {kind!r}")


def run_hypothesis(model, task: Task, spec_or_path) -> dict:
    spec = spec_or_path if isinstance(spec_or_path, HypothesisSpec) else HypothesisSpec.load(spec_or_path)
    return HypothesisTest(model, task, spec).run()


def _label(c) -> str:
    l, h = c
    return f"L{l}.mlp" if h == "mlp" else f"L{l}H{h}"
