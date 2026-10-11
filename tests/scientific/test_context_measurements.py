"""Ground-truth calibration of the context-matching measurements (hand-wired models)."""

import pytest

from transformer_lab.interpretability.context import (context_preference, context_sensitivity, cp_under_path,
                                                      induction_group)
from transformer_lab.models.handwired import build_induction_model

pytestmark = pytest.mark.scientific


@pytest.fixture(scope="module")
def models():
    return {"single": build_induction_model(48, 40), "ctx": build_induction_model(48, 40, context_gain=2.0)}


def test_single_token_model_has_zero_context_preference(models):
    r = context_preference(models["single"], n=256)
    assert abs(r["cp"]["mean"]) < 0.01 and r["either_rate"] == 1.0
    assert abs(r["cp_match_recent"]["mean"]) < 0.01 and abs(r["cp_match_earlier"]["mean"]) < 0.01


def test_context_model_prefers_matching_context_in_both_recency_configurations(models):
    r = context_preference(models["ctx"], n=256)
    assert r["cp"]["lo"] > 1.5 and r["selection_rate"] == 1.0
    assert r["cp_match_recent"]["lo"] > 1.5 and r["cp_match_earlier"]["lo"] > 1.5


def test_context_sensitivity_cannot_detect_a_saturated_context_matching_mechanism(models):
    """Methodological finding (2026-10-11): the 'context sensitivity' measurement used in the first
    research session is BLIND to a genuine context-matching mechanism when the current-token match
    dominates — with a single earlier occurrence, the mismatched context does not change which key
    wins. It reads exactly 0 for both ground-truth models, while context preference (CP) separates
    them. So a non-zero sensitivity in trained models was never specific evidence for context
    matching (and a zero would not have excluded it)."""
    for name in ("single", "ctx"):
        r = context_sensitivity(models[name], n=128)
        assert abs(r["affected"]["mean"]) < 1e-3 and abs(r["control"]["mean"]) < 1e-3, name
    assert context_preference(models["ctx"], n=128)["cp"]["mean"] > 1.5


def test_query_path_removal_abolishes_context_preference_only_in_context_model(models):
    """In the context model, context enters through the QUERY (L0H0 -> L1H0.q): removing layer-0
    output from the query kills the preference; in the single-token model there is none to kill."""
    m = models["ctx"]
    group = induction_group(m)
    assert group == [(1, 0)]
    q = cp_under_path(m, group, "q", n=256)
    assert abs(q["cp"]["mean"]) < 0.05
    v = cp_under_path(m, group, "v", n=256)  # values carry no context in this construction
    assert v["cp"]["lo"] > 1.5


def test_per_sender_route_identifies_the_query_carrier(models):
    from transformer_lab.interpretability.context import context_route_per_sender
    rows = {(r["sender"], r["channel"]): r["recovery"] for r in context_route_per_sender(models["ctx"], n=128)["rows"]}
    assert rows[("L0H0", "q")] > 0.99  # by construction the query reads L0H0's previous-token slot
    assert abs(rows[("L0H1", "q")]) < 0.01 and abs(rows[("L0H0", "k")]) < 0.01


def test_competition_attention_locates_preference_in_the_pattern(models):
    from transformer_lab.interpretability.context import competition_attention
    ctx = competition_attention(models["ctx"], n=128)["rows"][0]
    single = competition_attention(models["single"], n=128)["rows"][0]
    assert ctx["attn_A_minus_B"]["lo"] > 0.05
    assert abs(single["attn_A_minus_B"]["mean"]) < 1e-3


def test_dose_response_reflects_how_many_tokens_are_matched(models):
    """The hand-wired context model matches exactly one preceding token: more matching context
    must not increase its preference (a flat curve is the correct ground truth)."""
    from transformer_lab.interpretability.context import context_dose_response
    cps = [d["cp"]["mean"] for d in context_dose_response(models["ctx"], n=128)]
    assert max(cps) - min(cps) < 0.05
