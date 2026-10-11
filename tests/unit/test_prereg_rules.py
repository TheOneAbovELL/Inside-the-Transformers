"""The frozen decision rules of context_matching_v1, checked on synthetic measurements
(written before the experiment's real measurements were analysed)."""

import json

import numpy as np
import pytest

from transformer_lab.experiments.context_experiment import evaluate_rules, run_context_experiment


def m(name, cp, lo, hi, seed=0, valid=True, q=None, q_either=1.0, values=None):
    rng = np.random.default_rng(seed)
    d = {"model": name, "seed": seed, "valid": valid, "cp": {"mean": cp, "lo": lo, "hi": hi},
         "_cp_values": values if values is not None else rng.normal(cp, 0.5, 400), "path": {}}
    if q is not None:
        d["path"]["q"] = {"cp": {"mean": q[0], "lo": q[1], "hi": q[2]}, "either_rate": q_either}
    return d


def conds(orig, same, shuf):
    return {"original": orig, "same_order": same, "shuffled": shuf, "calibration": []}


def test_h1_supported_contradicted_inconclusive():
    sup = [m("a", 2, 1.5, 2.5), m("b", 1, 0.6, 1.4, 1), m("c", 0.1, -0.2, 0.4, 2)]
    assert evaluate_rules(conds(sup, [], []))["H1"]["verdict"] == "supported"
    null = [m("a", 0.1, -0.2, 0.4), m("b", 0, -0.3, 0.3, 1), m("c", 2, 1.5, 2.5, 2)]
    assert evaluate_rules(conds(null, [], []))["H1"]["verdict"] == "contradicted"
    # one seed with a significantly NEGATIVE CP blocks support
    neg = [m("a", 2, 1.5, 2.5), m("b", 1, 0.6, 1.4, 1), m("c", -1, -1.4, -0.6, 2)]
    assert evaluate_rules(conds(neg, [], []))["H1"]["verdict"] == "inconclusive"
    # invalid models do not count
    inv = [m("a", 2, 1.5, 2.5), m("b", 1, 0.6, 1.4, 1, valid=False), m("c", 0.1, -0.2, 0.4, 2)]
    assert evaluate_rules(conds(inv, [], []))["H1"]["verdict"] == "inconclusive"


def test_h2_rules():
    same = [m("s0", 2, 1.9, 2.1, 0), m("s1", 2, 1.9, 2.1, 1), m("s2", 2, 1.9, 2.1, 2)]
    shuf = [m("h0", 0, -0.1, 0.1, 0), m("h1", 0, -0.1, 0.1, 1), m("h2", 0, -0.1, 0.1, 2)]
    assert evaluate_rules(conds([], same, shuf))["H2"]["verdict"] == "supported"
    shuf_eq = [m("h0", 2, 1.9, 2.1, 0), m("h1", 2, 1.9, 2.1, 1), m("h2", 2, 1.9, 2.1, 2)]
    assert evaluate_rules(conds([], same, shuf_eq))["H2"]["verdict"] == "contradicted"
    one_valid = [m("h0", 0, -0.1, 0.1, 0), m("h1", 0, -0.1, 0.1, 1, valid=False), m("h2", 0, -0.1, 0.1, 2, valid=False)]
    assert evaluate_rules(conds([], same, one_valid))["H2"]["verdict"] == "inconclusive"


def test_h3_rules():
    models = [m("a", 2, 1.8, 2.2, q=(0.2, 0.0, 0.4)), m("b", 2, 1.8, 2.2, 1, q=(0.3, 0.1, 0.5)),
              m("c", 2, 1.8, 2.2, 2, q=(1.9, 1.7, 2.1))]
    r = evaluate_rules(conds(models, [], []))["H3"]
    assert r["verdict"] == "supported" and [p["verdict"] for p in r["per_model"]] == ["supported", "supported",
                                                                                       "contradicted"]
    # removal that destroys induction altogether is uninterpretable
    r = evaluate_rules(conds([m("a", 2, 1.8, 2.2, q=(0.0, -0.1, 0.1), q_either=0.2)], [], []))["H3"]
    assert r["per_model"][0]["verdict"] == "inconclusive"
    assert evaluate_rules(conds([m("a", 0.1, -0.1, 0.3)], [], []))["H3"]["verdict"] == "not evaluated"


def test_analysis_refuses_unfrozen_or_edited_spec(tmp_path, monkeypatch):
    import shutil
    from transformer_lab.utils import PROJECT_ROOT
    monkeypatch.setenv("ITL_ARTIFACT_ROOT", str(tmp_path / "art"))
    with pytest.raises(RuntimeError, match="no 'frozen' entry"):
        run_context_experiment("configs/preregistration/context_matching_v1.yaml")
    (tmp_path / "art").mkdir(exist_ok=True)
    (tmp_path / "art" / "preregistration_log.jsonl").write_text(json.dumps(
        {"id": "context_matching_v1", "event": "frozen",
         "sha256": {"configs/preregistration/context_matching_v1.yaml": "0" * 64}}) + "\n")
    with pytest.raises(RuntimeError, match="changed since it was frozen"):
        run_context_experiment(str(PROJECT_ROOT / "configs/preregistration/context_matching_v1.yaml"))


def test_route_rules():
    from transformer_lab.experiments.context_experiment import evaluate_route_rules
    r = lambda q, k, v, valid=True: {"valid": valid, "q": {"recovery_lo": q[0], "recovery_hi": q[1]},
                                     "k": {"recovery_lo": k[0], "recovery_hi": k[1]},
                                     "v": {"recovery_lo": v[0], "recovery_hi": v[1]}}
    good = [r((1, 1.3), (-.3, 0), (0, .1)), r((.9, 1.2), (-.2, .1), (-.1, .1))]
    assert evaluate_route_rules(good)["verdict"] == "supported"
    assert evaluate_route_rules([r((0, .3), (-.3, 0), (0, .1)), r((.1, .4), (0, .1), (0, .1))])["verdict"] == "contradicted"
    assert evaluate_route_rules([good[0], r((.9, 1.2), (.4, .7), (0, .1))])["verdict"] == "inconclusive"  # keys too
    assert evaluate_route_rules([good[0], r((0, 0), (0, 0), (0, 0), valid=False)])["verdict"] == "inconclusive"
