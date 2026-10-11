"""Validate every interpretability tool against a model whose circuit is known by construction.

Known ground truth (see models/handwired.py): L0H0 = previous-token head, L1H0 = induction head
that K-composes with L0H0, L0H1 and L1H1 are dead. Each test states the prediction the ground
truth implies, including predictions that a naive reading would get wrong.
"""

import numpy as np
import pytest
import torch

from transformer_lab.data import InductionRepeat
from transformer_lab.evaluation.metrics import run_batched
from transformer_lab.interpretability.attention_scores import composition_scores, head_scores, ov_copying_score
from transformer_lab.interpretability.attribution import direct_logit_attribution
from transformer_lab.interpretability.interventions import Reference, head_ablation_hooks, reference_names
from transformer_lab.interpretability.patching import Patcher, PatchSpec, head_specs, task_regions
from transformer_lab.interpretability.path import path_effect, pattern_under_path
from transformer_lab.models.handwired import build_induction_model

pytestmark = pytest.mark.scientific

TASK = InductionRepeat(n_symbols=12, seq_len=16, min_len=3, max_len=5)


@pytest.fixture(scope="module")
def setup():
    model = build_induction_model(n_symbols=12, n_ctx=16)
    batch = TASK.sample(200, np.random.default_rng(0))
    return model, batch


def acc(model, batch, hooks=()):
    return run_batched(model, batch, fwd_hooks_fn=lambda sl: list(hooks))["accuracy"].mean()


def test_construction_solves_task(setup):
    model, batch = setup
    assert acc(model, batch) == 1.0


def test_attention_scores_identify_heads(setup):
    model, batch = setup
    s = head_scores(model, batch)
    assert s["prev_token"][0, 0] > 0.99 and s["induction"][1, 0] > 0.99
    assert s["prev_token"][0, 1] < 0.5 and s["induction"][1, 1] < 0.5  # dead heads attend ~uniformly


def test_ov_copying_score(setup):
    model, _ = setup
    ov = ov_copying_score(model)
    assert ov[1, 0] == 1.0  # induction head's OV copies tokens to their own logits
    assert ov[0, 0] < 0.2   # prev-token head writes to a slot the unembedding ignores


def test_composition_scores_find_k_composition(setup):
    model, _ = setup
    c = composition_scores(model)
    k = c["K"][1, 0, 0, 0]
    assert k > c["Q"][1, 0, 0, 0] + 0.1 and k > c["V"][1, 0, 0, 0] + 0.1
    assert np.isnan(c["K"][0, 0, 0, 0])


@pytest.mark.parametrize("method", ["zero", "mean", "resample"])
def test_ablation_effects_match_ground_truth(setup, method):
    model, batch = setup
    ref = Reference.from_batch(model, TASK.sample(200, np.random.default_rng(1)), reference_names(model))
    a_ind = acc(model, batch, head_ablation_hooks([(1, 0)], method, ref))
    a_prev = acc(model, batch, head_ablation_hooks([(0, 0)], method, ref))
    a_dead = acc(model, batch, head_ablation_hooks([(0, 1), (1, 1)], method, ref))
    assert a_ind < 0.35 and a_prev < 0.35  # both circuit heads are necessary
    assert a_dead == 1.0                   # negative control: dead heads do nothing


def test_path_interventions_isolate_k_composition(setup):
    """Removing L0H0's output from ONLY L1H0's key input destroys the behaviour and the induction
    pattern; removing it from L1H0's query or value input changes nothing."""
    model, batch = setup
    res = {ch: path_effect(model, batch, (0, 0), (1, 0), ch, metric="accuracy")["treated"].mean() for ch in "qkv"}
    assert res["k"] < 0.35
    assert res["q"] == 1.0 and res["v"] == 1.0
    pat = pattern_under_path(model, batch, (0, 0), (1, 0), "k")
    clean = model.run_with_cache(batch.tokens, names_filter=["blocks.1.attn.hook_pattern"])[1]["pattern", 1][:, 0]
    assert not torch.allclose(pat, clean)


def test_path_with_identity_replacement_is_a_no_op(setup):
    model, batch = setup
    one = batch.subset([0])
    _, cache = model.run_with_cache(one.tokens)
    own = cache.head_results(0)[0, :, 0]  # replacement == the sender's actual output
    out = path_effect(model, one, (0, 0), (1, 0), "k", replacement=own, metric="logprob")
    np.testing.assert_allclose(out["treated"], out["baseline"], atol=1e-5)


def test_patching_recovers_known_components(setup):
    model, batch = setup
    corrupt = TASK.corrupt(batch, np.random.default_rng(2), "replace")
    p = Patcher(model, batch, corrupt, metric="logprob")
    assert p.informative and p.gap > 5
    regions = task_regions(TASK, batch)
    results = {r.label: r for r in (p.run(s) for s in head_specs(model, regions["scored"]))}
    assert results["L1H0"].recovery > 0.95          # induction head output at the answer positions
    assert abs(results["L1H1"].recovery) < 0.02     # dead head
    # The prev-token head alone is NOT sufficient: in the corrupted run the induction head would
    # attend to the right place but copy the corrupted token. Recovery ~0 despite being necessary.
    assert abs(results["L0H0"].recovery) < 0.05


def test_patching_sanity_identities(setup):
    model, batch = setup
    corrupt = TASK.corrupt(batch, np.random.default_rng(3), "replace")
    p = Patcher(model, batch, corrupt)
    full = p.run(PatchSpec("all_resid0", "blocks.0.hook_resid_pre"))
    assert full.recovery == pytest.approx(1.0, abs=1e-5)
    none = p.run(PatchSpec("nothing", "blocks.0.hook_resid_pre", positions=torch.zeros_like(batch.tokens, dtype=torch.bool)))
    assert none.recovery == pytest.approx(0.0, abs=1e-6)
    # identical clean/corrupt -> flagged uninformative, never a fabricated recovery score
    same = Patcher(model, batch, batch)
    r = same.run(PatchSpec("x", "blocks.1.attn.hook_z", heads=[0]))
    assert not r.informative and np.isnan(r.recovery)


def test_dla_attributes_logit_to_induction_head(setup):
    model, batch = setup
    d = direct_logit_attribution(model, batch)
    assert d["reconstruction_error"] < 1e-3
    contrib = dict(zip(d["labels"], d["contrib"]))
    assert max(contrib, key=contrib.get) == "L1H0"
    assert abs(contrib["L0H0"]) < 1e-4  # writes only to a slot the unembedding ignores


def test_group_path_intervention_matches_single_when_extra_senders_are_dead(setup):
    """Adding a dead sender (L0H1) and a dead receiver (L1H1) to the groups must not change the effect."""
    model, batch = setup
    single = path_effect(model, batch, (0, 0), (1, 0), "k", metric="logprob")["treated"]
    group = path_effect(model, batch, [(0, 0), (0, 1)], [(1, 0), (1, 1)], "k", metric="logprob")["treated"]
    np.testing.assert_allclose(group, single, atol=1e-5)


def test_threshold_and_layer_candidate_selection(setup):
    from transformer_lab.interpretability.hypothesis import select_candidates
    model, batch = setup
    groups, _ = select_candidates(model, batch, {
        "ind": {"select": "threshold", "score": "induction", "layer": 1, "min_score": 0.5},
        "l0": {"select": "layer", "layer": 0}})
    assert groups["ind"] == [(1, 0)]
    assert groups["l0"] == [(0, 0), (0, 1)]


@pytest.mark.parametrize("params", [dict(min_len=2, max_len=2), dict(min_len=5, max_len=5, seq_len=16),
                                    dict(fixed_prefix_len=0, min_len=5, max_len=5, seq_len=11),
                                    dict(pair_mode="same_order", min_len=4, max_len=4),
                                    dict(pair_mode="shuffled", min_len=4, max_len=4)])
def test_construction_survives_input_variations(params):
    """The hand-wired circuit is position-independent: it must solve every variant within n_ctx."""
    from transformer_lab.data import InductionPair
    model = build_induction_model(n_symbols=12, n_ctx=16)
    t = InductionRepeat(**{"n_symbols": 12, "seq_len": 16, **params})
    assert acc(model, t.sample(100, np.random.default_rng(1))) == 1.0
    p = InductionPair(n_symbols=12, seq_len=16, n_distractor_pairs=1)
    assert acc(model, p.sample(100, np.random.default_rng(2))) == 1.0
