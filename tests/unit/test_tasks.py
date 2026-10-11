"""Dataset generators: determinism, specification, solvers, splits and corruptions."""

import numpy as np
import pytest
import torch

from transformer_lab.data import (BOS, PAD, TASKS, InductionPair, InductionRepeat, TaskBatch, induction_solver,
                                  make_task)
from transformer_lab.data.tasks import _heldout
from transformer_lab.evaluation.metrics import solver_accuracy

TASK_PARAMS = {
    "repeat_pattern": {},
    "assoc_recall": {"holdout_mod": 5, "max_noise": 3},
    "induction_repeat": {},
    "induction_pair": {"holdout_mod": 5, "n_distractor_pairs": 2, "n_decoys": 2},
    "context_competition": {},
}


@pytest.mark.parametrize("name", sorted(TASKS))
def test_generator_is_seed_deterministic(name):
    t = make_task(name, TASK_PARAMS[name])
    a = t.sample(64, np.random.default_rng(7))
    b = t.sample(64, np.random.default_rng(7))
    c = t.sample(64, np.random.default_rng(8))
    assert torch.equal(a.tokens, b.tokens) and torch.equal(a.loss_mask, b.loss_mask)
    assert not torch.equal(a.tokens, c.tokens)


@pytest.mark.parametrize("name", sorted(TASKS))
@pytest.mark.parametrize("split", ["train", "eval"])
def test_shapes_vocab_and_exact_solver(name, split):
    t = make_task(name, TASK_PARAMS[name])
    b = t.sample(200, np.random.default_rng(0), split)
    assert b.tokens.shape == (200, t.seq_len)
    assert b.loss_mask.shape == (200, t.seq_len - 1)
    assert b.tokens.min() >= 0 and b.tokens.max() < t.vocab_size
    assert torch.all(b.tokens[:, 0] == BOS)
    assert torch.all(b.loss_mask.sum(1) >= 1), "every example must have a scored position"
    # never score a prediction whose target is padding
    assert not torch.any(b.loss_mask & (b.targets == PAD))
    # the exact algorithm must be perfect, otherwise the generator is wrong
    assert solver_accuracy(t.solve(b), b) == 1.0


def test_all_tasks_are_solvable_by_the_induction_rule():
    """Documented property: every task in this lab is solvable by 'copy the successor of the previous
    occurrence'. (Important when interpreting circuits: they may all share a mechanism.)"""
    for name in TASKS:
        if name == "context_competition":
            continue  # deliberately ambiguous for the single-token rule (tested below)
        t = make_task(name, TASK_PARAMS[name])
        b = t.sample(100, np.random.default_rng(1))
        assert solver_accuracy(induction_solver(b.tokens, b.loss_mask), b) == 1.0, name


def test_induction_repeat_structure():
    t = InductionRepeat(n_symbols=48, seq_len=40, min_len=6, max_len=16)
    b = t.sample(300, np.random.default_rng(0))
    for i in range(len(b)):
        L, a, s2 = (int(b.meta[k][i]) for k in ("L", "first_start", "second_start"))
        tok = b.tokens[i]
        assert torch.equal(tok[a: a + L], tok[s2: s2 + L])
        assert len(set(tok[a: a + L].tolist())) == L  # distinct within the segment
        assert not set(tok[1:a].tolist()) & set(tok[a: a + L].tolist())  # prefix avoids the segment
        assert b.loss_mask[i].sum() == L - 1
    # offsets genuinely vary (no fixed-offset shortcut)
    assert len(np.unique(b.meta["L"])) > 5 and len(np.unique(b.meta["prefix_len"])) > 5


def test_heldout_pairs_never_in_train_and_always_queried_in_eval():
    t = InductionPair(n_symbols=32, seq_len=32, holdout_mod=5)
    tr = t.sample(400, np.random.default_rng(0), "train")
    ev = t.sample(400, np.random.default_rng(1), "eval")
    def pair(b, i):
        q = t.seq_len - 2
        return int(b.tokens[i, q]) - 2, int(b.tokens[i, q + 1]) - 2
    assert not any(_heldout(*pair(tr, i), 5) for i in range(400))
    assert all(_heldout(*pair(ev, i), 5) for i in range(400))

    ar = make_task("assoc_recall", {"holdout_mod": 5})
    tr = ar.sample(300, np.random.default_rng(0), "train")
    ev = ar.sample(300, np.random.default_rng(1), "eval")
    def kv_pairs(b, i):
        tok = b.tokens[i].tolist()
        return [(tok[j] - ar.key0, tok[j + 1] - ar.val0) for j in range(1, len(tok) - 1)
                if ar.key0 <= tok[j] < ar.val0 and ar.val0 <= tok[j + 1] < ar.noise0]
    assert not any(_heldout(k, v, 5) for i in range(300) for k, v in kv_pairs(tr, i))
    for i in range(300):
        q = int(ev.meta["query_pos"][i])
        k, v = int(ev.tokens[i, q]) - ar.key0, int(ev.tokens[i, q + 1]) - ar.val0
        assert _heldout(k, v, 5)


def test_induction_pair_controls_position_and_distractors():
    t = InductionPair(n_symbols=32, seq_len=32, evidence_pos=5, n_distractor_pairs=2, n_decoys=3)
    b = t.sample(100, np.random.default_rng(0))
    assert np.all(b.meta["evidence_pos"] == 5)
    assert np.all(b.meta["distance"] == t.seq_len - 2 - 6)
    for i in range(100):
        tok = b.tokens[i].tolist()
        A, B = tok[-2], tok[-1]
        assert tok.count(A) == 2 and tok.count(B) == 2  # evidence + query/answer only
        assert tok.count(int(b.meta["decoy"][i])) == 3


def test_corruptions_keep_targets_and_define_alternatives():
    rng = np.random.default_rng(0)
    for name, kind in [("assoc_recall", "query_key"), ("induction_repeat", "shuffle"),
                       ("induction_repeat", "replace"), ("induction_pair", "replace_answer"),
                       ("induction_pair", "remove_evidence")]:
        t = make_task(name, TASK_PARAMS[name])
        clean = t.sample(100, rng)
        cor = t.corrupt(clean, rng, kind)
        assert cor.tokens.shape == clean.tokens.shape
        assert torch.equal(cor.loss_mask, clean.loss_mask)
        assert not torch.equal(cor.tokens, clean.tokens)
        # the clean answer is no longer what the induction rule predicts
        pred = induction_solver(cor.tokens, cor.loss_mask)
        assert solver_accuracy(pred, clean) < 0.05, (name, kind)
        if cor.alt_targets is not None:
            m = cor.loss_mask & (cor.alt_targets >= 0)
            # symmetric corruption: the induction rule now gives exactly the alternative answer
            assert torch.equal(pred[m], cor.alt_targets[m])
            assert not torch.any(cor.alt_targets[m] == clean.targets[m])


def test_unknown_params_and_bad_configs_rejected():
    with pytest.raises(ValueError):
        make_task("induction_repeat", {"n_symbol": 3})
    with pytest.raises(ValueError):
        make_task("nonexistent")
    with pytest.raises(ValueError):
        InductionRepeat(n_symbols=10, max_len=16)
    with pytest.raises(ValueError):
        InductionRepeat().with_params(n_symbols=60)  # changes vocab


def test_batch_subset_and_concat():
    t = make_task("assoc_recall", {"max_pairs": 3})
    b1 = t.sample(5, np.random.default_rng(0))
    b2 = t.sample(3, np.random.default_rng(1))
    cat = TaskBatch.concat([b1, b2])
    assert len(cat) == 8 and torch.equal(cat.tokens[:5], b1.tokens)
    sub = cat.subset([0, 6])
    assert torch.equal(sub.tokens[1], b2.tokens[1]) and sub.meta["n_pairs"][1] == b2.meta["n_pairs"][1]


# ------------------------------------------------------------------ context-matching experiment
def test_context_competition_is_ambiguous_for_single_token_rule_and_counterbalanced():
    from transformer_lab.data import ContextCompetition
    for k in (1, 2, 3):
        t = ContextCompetition(context_len=k)
        b = t.sample(400, np.random.default_rng(3))
        # the "most recent occurrence" single-token rule is right exactly when the match is recent: 50%
        assert solver_accuracy(induction_solver(b.tokens, b.loss_mask), b) == 0.5
        assert b.meta["match_recent"].mean() == 0.5
        # the context-matching rule is exact
        assert solver_accuracy(t.solve(b), b) == 1.0
        for i in range(400):
            tok = b.tokens[i].tolist()
            x, A, B = tok[-2], int(b.meta["answer"][i]), int(b.meta["alt"][i])
            assert tok.count(x) == 3 and tok.count(A) == 2 and tok.count(B) == 1
            # designed repeats only: x twice more, A once more, the k context tokens once more
            assert len(set(tok)) == len(tok) - (k + 3)
            pm, pn = int(b.meta["pos_match"][i]), int(b.meta["pos_nonmatch"][i])
            assert tok[pm: pm + k] == tok[-(k + 2): -2]  # matching context really matches
            assert tok[pn: pn + k] != tok[-(k + 2): -2]
            assert tok[pm + k] == x and tok[pn + k] == x
            assert tok[pm + k + 1] == A and tok[pn + k + 1] == B


def test_manual_context_competition_example():
    """Hand-check one tiny example end to end."""
    from transformer_lab.data import ContextCompetition
    t = ContextCompetition(n_symbols=12, seq_len=14, min_sep=1)
    b = t.sample(2, np.random.default_rng(0))
    toks = t.decode(b.tokens[0])
    assert toks[0] == "<bos>" and b.loss_mask[0].nonzero().flatten().tolist() == [12]
    # target is the last token; it follows the earlier "y x" whose y equals token 11
    y, x, A = toks[11], toks[12], toks[13]
    joined = " ".join(toks[1:11])
    assert f"{y} {x} {A}" in joined


@pytest.mark.parametrize("mode", ["same_order", "shuffled"])
def test_pair_mode_structure(mode):
    t = InductionRepeat(pair_mode=mode)
    b = t.sample(300, np.random.default_rng(0))
    assert solver_accuracy(t.solve(b), b) == 1.0
    assert all(int(L) % 2 == 0 for L in b.meta["L"])
    n_ctx_match = n_scored = 0
    for i in range(300):
        tok = b.tokens[i]
        a, s2, L = (int(b.meta[k][i]) for k in ("first_start", "second_start", "L"))
        first, second = tok[a: a + L].tolist(), tok[s2: s2 + L].tolist()
        assert sorted(first) == sorted(second)
        pairs1 = {(first[j], first[j + 1]) for j in range(0, L, 2)}
        pairs2 = {(second[j], second[j + 1]) for j in range(0, L, 2)}
        assert pairs1 == pairs2  # same (x, y) bindings
        for t_ in b.loss_mask[i].nonzero().flatten().tolist():
            x = int(tok[t_])
            j = first.index(x) + a
            n_scored += 1
            n_ctx_match += int(tok[j - 1] == tok[t_ - 1])
    if mode == "shuffled":
        assert n_ctx_match == 0  # preceding context never repeats
    else:
        assert n_ctx_match > 0.75 * n_scored  # contiguous repeat: context repeats except at the first pair


def test_pair_mode_does_not_change_default_stream():
    a = InductionRepeat().sample(20, np.random.default_rng(9))
    b = InductionRepeat(pair_mode=None).sample(20, np.random.default_rng(9))
    assert torch.equal(a.tokens, b.tokens)
