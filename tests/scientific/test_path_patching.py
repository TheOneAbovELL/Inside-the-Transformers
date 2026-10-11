"""Path patching: exact identities and hand-wired ground truth (including MLP-mediated routes)."""

import numpy as np
import pytest
import torch

from conftest import make_model
from transformer_lab.data import InductionRepeat
from transformer_lab.evaluation.metrics import run_batched
from transformer_lab.interpretability.patching import Patcher, PatchSpec
from transformer_lab.interpretability.path import mean_head_results, path_effect
from transformer_lab.interpretability.path_patching import all_outgoing_edges, path_patch
from transformer_lab.models.handwired import build_induction_model

pytestmark = pytest.mark.scientific
TASK = InductionRepeat(n_symbols=12, seq_len=16, min_len=3, max_len=5)


@pytest.mark.parametrize("attn_only,sender", [(True, (0, 1)), (True, (1, 2)), (False, (0, 1)),
                                              (False, (0, "mlp")), (False, (1, 2))])
def test_all_edges_equals_activation_patching(attn_only, sender):
    """Completeness identity: delivering a sender's source output along EVERY outgoing edge is the
    same intervention as replacing its output everywhere (ordinary activation patching)."""
    m = make_model(vocab_size=TASK.vocab_size, n_ctx=16, attn_only=attn_only)
    clean = TASK.sample(32, np.random.default_rng(0))
    corrupt = TASK.corrupt(clean, np.random.default_rng(1), "replace")
    pp = path_patch(m, clean, corrupt, [sender], all_outgoing_edges(m, sender), min_gap=0.0)
    hook = f"blocks.{sender[0]}.hook_mlp_out" if sender[1] == "mlp" else f"blocks.{sender[0]}.attn.hook_z"
    heads = None if sender[1] == "mlp" else [sender[1]]
    act = Patcher(m, clean, corrupt, min_gap=0.0, names=[hook]).run(PatchSpec("x", hook, heads=heads))
    np.testing.assert_allclose(pp["patched_per_example"], act.patched_per_example, rtol=1e-4, atol=1e-4)


def test_no_edges_is_a_no_op():
    m = make_model(vocab_size=TASK.vocab_size, n_ctx=16)
    clean = TASK.sample(16, np.random.default_rng(0))
    corrupt = TASK.corrupt(clean, np.random.default_rng(1), "replace")
    pp = path_patch(m, clean, corrupt, [(0, 0)], [], min_gap=0.0)
    assert pp["patched"] == pytest.approx(pp["corrupt"], abs=1e-6)


def test_invalid_edges_rejected():
    m = make_model(vocab_size=TASK.vocab_size, n_ctx=16)
    b = TASK.sample(4, np.random.default_rng(0))
    with pytest.raises(ValueError):
        path_patch(m, b, b, [(1, 0)], [("head", 1, 0, "k")])
    with pytest.raises(ValueError):
        path_patch(m, b, b, [(0, "mlp")], [("mlp", 0)])


@pytest.fixture(scope="module")
def relay():
    m = build_induction_model(12, 16, mlp_relay=True)
    clean = TASK.sample(200, np.random.default_rng(0))
    corrupt = TASK.corrupt(clean, np.random.default_rng(1), "replace")
    return m, clean, corrupt


def test_relay_model_solves_task(relay):
    m, clean, _ = relay
    assert run_batched(m, clean)["accuracy"].mean() == 1.0


def test_relay_ground_truth_routes_noising(relay):
    """Known route: L0H0 -> MLP0 -> L1H0 keys. Corrupting L0H0's output along the direct edge to the
    keys changes nothing (R~1); along the edge into MLP0 it destroys the behaviour (R~0)."""
    m, clean, corrupt = relay
    pp = lambda s, r, metric: path_patch(m, clean, corrupt, s, r, metric=metric, direction="noise")
    direct = pp([(0, 0)], [("head", 1, 0, "k")], "accuracy")
    via_mlp = pp([(0, 0)], [("mlp", 0)], "accuracy")
    mlp_to_keys = pp([(0, "mlp")], [("head", 1, 0, "k")], "accuracy")
    assert direct["recovery"] == pytest.approx(1.0) and via_mlp["recovery"] == 0.0 and mlp_to_keys["recovery"] == 0.0
    # Metric lesson: with log-prob the broken route still reads R ~ 0.33, because the induction head
    # then attends diffusely over a context that still contains the clean tokens, giving the right
    # token some probability. A 'partial recovery' in log-prob is not evidence of a partial route.
    lp = pp([(0, 0)], [("mlp", 0)], "logprob")
    assert 0.2 < lp["recovery"] < 0.5


def test_relay_ground_truth_mean_replacement_paths(relay):
    """The mean-replacement path tool used for hypothesis v3 recovers the same routing."""
    m, clean, _ = relay
    rep = mean_head_results(m, TASK.sample(200, np.random.default_rng(5)), [(0, 0), (0, "mlp")])
    direct = path_effect(m, clean, [(0, 0)], [(1, 0)], "k", replacement={(0, 0): rep[(0, 0)]}, metric="accuracy")
    via = path_effect(m, clean, [(0, "mlp")], [(1, 0)], "k", replacement={(0, "mlp"): rep[(0, "mlp")]}, metric="accuracy")
    assert direct["treated"].mean() == 1.0
    assert via["treated"].mean() < 0.35


def test_direct_path_of_output_head_is_everything():
    m = build_induction_model(12, 16)
    clean = TASK.sample(100, np.random.default_rng(0))
    corrupt = TASK.corrupt(clean, np.random.default_rng(1), "replace")
    r = path_patch(m, clean, corrupt, [(1, 0)], [("logits",)])
    assert r["recovery"] > 0.99


def test_layer0_union_with_mlp_equals_activation_patching():
    """Patching every outgoing edge of ALL layer-0 senders (heads + MLP0, edges into MLP0 dropped)
    equals activation patching of all of layer 0's outputs at once."""
    from transformer_lab.interpretability.context import all_layer0_edges_patch
    m = make_model(vocab_size=TASK.vocab_size, n_ctx=16, attn_only=False)
    clean = TASK.sample(32, np.random.default_rng(0))
    corrupt = TASK.corrupt(clean, np.random.default_rng(1), "replace")
    senders = [(0, h) for h in range(m.cfg.n_heads)] + [(0, "mlp")]
    p = Patcher(m, clean, corrupt, metric="logprob", min_gap=0.0,
                names=["blocks.0.attn.hook_z", "blocks.0.hook_mlp_out"])
    act = p.run([PatchSpec("z", "blocks.0.attn.hook_z"), PatchSpec("mlp", "blocks.0.hook_mlp_out")])
    from transformer_lab.interpretability.path_patching import all_outgoing_edges, path_patch
    edges = sorted({e for s in senders for e in all_outgoing_edges(m, s)} - {("mlp", 0)})
    pp = path_patch(m, clean, corrupt, senders, edges, metric="logprob", min_gap=0.0)
    np.testing.assert_allclose(pp["patched_per_example"], act.patched_per_example, rtol=1e-4, atol=1e-4)
    assert callable(all_layer0_edges_patch)
