"""The hypothesis framework must confirm a true hypothesis and refuse to confirm a false one."""

import pytest

from transformer_lab.config import ModelConfig
from transformer_lab.data import InductionRepeat
from transformer_lab.interpretability.hypothesis import HypothesisSpec, run_hypothesis
from transformer_lab.models import Transformer
from transformer_lab.models.handwired import build_induction_model

pytestmark = pytest.mark.scientific
TASK = InductionRepeat(n_symbols=12, seq_len=16, min_len=3, max_len=5)


def spec(**kw):
    s = HypothesisSpec.load("configs/hypotheses/induction_2l.yaml")
    s.n_examples = 128
    for p in s.predictions:
        if "control" in p:
            p["control"]["n_samples"] = 5
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_true_hypothesis_is_supported_on_ground_truth_model():
    res = run_hypothesis(build_induction_model(12, 16), TASK, spec())
    assert res["candidates"] == {"induction": ["L1H0"], "prev_token": ["L0H0"]}
    verdicts = {p["id"]: p["verdict"] for p in res["predictions"]}
    assert all(v == "supported" for v in verdicts.values()), res["predictions"]


def test_hypothesis_not_supported_on_untrained_model():
    """Negative control: an untrained model must not 'have' the circuit."""
    m = Transformer(ModelConfig(vocab_size=TASK.vocab_size, n_ctx=16, d_model=32, n_layers=2, n_heads=2,
                                attn_only=True, seed=0))
    res = run_hypothesis(m, TASK, spec())
    v = {p["id"]: p["verdict"] for p in res["predictions"]}
    for pid in ("P1_pattern_induction", "P2_pattern_prev", "P3_ablate_induction", "P4_ablate_prev", "P5_path_K"):
        assert v[pid] == "contradicted", (pid, res["predictions"])
    # P8: untrained model has no clean/corrupt gap -> must be flagged, not scored
    p8 = next(p for p in res["predictions"] if p["id"] == "P8_patch_induction")
    assert p8["verdict"] == "inconclusive" and "uninformative" in p8["reason"]


def test_requires_makes_vacuous_specificity_inconclusive():
    """On an untrained model the Q-path 'little effect' prediction is trivially true; with
    requires: [P5_path_K] it must become inconclusive instead of supported."""
    s = spec()
    for p in s.predictions:
        if p["id"] in ("P6_path_Q_specificity", "P7_path_V_specificity"):
            p["requires"] = ["P5_path_K"]
    m = Transformer(ModelConfig(vocab_size=TASK.vocab_size, n_ctx=16, d_model=32, n_layers=2, n_heads=2,
                                attn_only=True, seed=0))
    res = run_hypothesis(m, TASK, s)
    p6 = next(p for p in res["predictions"] if p["id"] == "P6_path_Q_specificity")
    assert p6["raw_verdict"] == "supported" and p6["verdict"] == "inconclusive"


def test_selection_and_test_sets_are_disjoint_and_familywise_reported():
    res = run_hypothesis(build_induction_model(12, 16), TASK, spec())
    assert res["set_overlap"] == {"selection_test": 0, "reference_test": 0}
    assert res["familywise_interval_level"] == pytest.approx(1 - 0.05 / 8)
    assert all("verdict_familywise" in p for p in res["predictions"])
