# Task audit

For each synthetic task: what it tests, how inputs are built, what is and is not available to the
model, known shortcuts, the meaning of its baselines, generalisation tests, corruptions, and what
it cannot establish. Every claim marked ✔ is checked by a test (named in brackets).
Generator code: `src/transformer_lab/data/tasks.py`.

Shared conventions: token 0 = `<pad>`, 1 = `<bos>`; `logits[t]` predicts `tokens[t+1]`;
`loss_mask[t]` marks scored predictions (only context-determined targets); validation and test sets
use different seeds from training (`numpy.random.SeedSequence`); a leakage audit regenerates the
whole training stream and counts exact overlaps with each evaluation set.

---

## repeat_pattern (Task 1)

1. **Tests**: continuing a periodic sequence — the simplest in-context copying.
2. **Input**: `<bos>` then a pattern of P ∈ [2, 6] *distinct* symbols repeated to fill the sequence.
3. **Correct output**: content position c ≥ P+1 equals content position c−P. ✔ exact solver = 1.0 [`test_shapes_vocab_and_exact_solver`]
4. **Available**: all earlier tokens. The period is identifiable once the first symbol recurs.
5. **Irrelevant**: absolute position.
6. **Shortcuts**: memorisation — with 16 symbols there are only 240 period-2 and 3,360 period-3 sequences, so most short-period evaluation sequences also occur in training. ✔ detected by the leakage audit (239/512 validation sequences seen) [`test_leakage_audit_detects_small_sequence_spaces`]. Positional copying with offset P works for any fixed P but P varies per example.
7. **Chance**: 1/16 (uniform over symbols). The "most frequent other token" heuristic scores 0 here because ties go to the current token's predecessor.
8. **Generalisation**: `itl eval --unseen-only` scores only never-seen sequences (0.74 vs 0.82 overall — the unseen subset is also harder, so not a pure memorisation estimate).
9. **Corruption**: none defined.
10. **Cannot establish**: whether the model detects the period or copies by recurrence; generalisation beyond the seen sequence space for short periods.

## assoc_recall (Task 2)

1. **Tests**: retrieving the value bound to a queried key from context.
2. **Input**: `<bos> k v k v … [noise] <q> k* v*`; distinct keys and distinct values per sequence; optional noise tokens between pairs.
3. **Correct output**: v* = the value that followed k* earlier. ✔ exact solver.
4. **Available**: keys, values, the query marker, noise tokens (separate vocabularies).
5. **Irrelevant**: noise tokens; the order of pairs; the query index.
6. **Shortcuts**: memorised (key, value) priors — blocked for evaluation by `holdout_mod`: held-out pairs never appear in training and are always queried in the test split ✔ [`test_heldout_pairs_never_in_train_and_always_queried_in_eval`]. Frequency: values are distinct, so frequency carries no information.
7. **Chance**: 1/16 over values (a model that knows the value vocabulary).
8. **Generalisation**: held-out pairs (test accuracy 1.000); query-position, noise and pair-count sweeps (all at ceiling).
9. **Corruption**: `query_key` — query a different key present in context (symmetric: the corrupted answer is defined).
10. **Cannot establish**: retrieval beyond 8 pairs (context length); every instance is also solvable by the single-token induction rule (`test_all_tasks_are_solvable_by_the_induction_rule`), so "associative recall" and "induction" need not use different mechanisms.

## induction_repeat (Task 3, principal)

1. **Tests**: copying the successor of the current token's earlier occurrence.
2. **Input**: `<bos>` + random prefix + segment r (L distinct symbols) + r again. L ∈ [6, 16] and the prefix length vary per example. ✔ structure [`test_induction_repeat_structure`].
3. **Correct output**: second-copy targets r₂…r_L.
4. **Available**: everything before the query; prefix symbols never occur in the segment, so each queried token has exactly one earlier occurrence.
5. **Irrelevant**: prefix content and absolute position.
6. **Shortcuts**: a *fixed-offset positional copy* — blocked by varying L and the prefix (offsets vary per example) ✔; yet a learned-position model still found a *smeared* positional strategy (findings §2), and a model trained on fixed offsets reaches 100% with zero content induction (findings §11). The `shuffle` corruption distinguishes content induction from positional copying behaviourally (`scripts/mechanism_type_test.py`).
7. **Chance**: 1/48.
8. **Generalisation**: segment lengths outside training (2–5, 17–19); zero-shot transfer to `induction_pair`.
9. **Corruptions**: `replace` (first copy → fresh symbols: no earlier match) and `shuffle` (first copy permuted so every successor changes; symmetric: the induction-rule answer is recorded). ✔ [`test_corruptions_keep_targets_and_define_alternatives`]
10. **Cannot establish**: whether context beyond the current token is used — single-token induction suffices, so matching on more context is neither needed nor detectable from accuracy (see the preregistered experiment).

### pair_mode variants (context-matching experiment)

`same_order`: segment `x1 y1 … xP yP` repeated; only `x→y` scored. `shuffled`: the second segment's pairs are reordered so no `x` is preceded by the same token twice ✔ [`test_pair_mode_structure`]. Identical tokens, lengths, scored targets and solvability; the default (contiguous) stream is unchanged by the option ✔ [`test_pair_mode_does_not_change_default_stream`, regression fixtures].

## induction_pair (Tasks 3b/4/5)

1. **Tests**: single-bigram induction under controlled position and interference.
2. **Input**: filler, one evidence bigram `A B`, optional repeated distractor bigrams and decoy copies, final `A`.
3. **Correct output**: `B`.
4. **Available**: everything before the final `A`; `A` and `B` occur only in the evidence and the query/answer ✔.
5. **Irrelevant**: filler, distractor bigrams, decoys.
6. **Shortcuts**: memorised (A, B) priors — blocked by held-out pairs; frequency — decoys make the most-frequent-token heuristic fail (it scores 0 throughout).
7. **Chance**: 1/32 (or 1/48 for zero-shot variants).
8. **Generalisation**: held-out pairs; evidence-position, distractor and decoy sweeps.
9. **Corruptions**: `replace_answer` (symmetric: earlier B → C), `remove_evidence`.
10. **Cannot establish**: whether the trained-on-it model learned induction — it did not (0/512 on held-out pairs: anti-generalising priors; findings §10). Useful as a zero-shot probe of models trained on `induction_repeat`.

## context_competition (evaluation only)

1. **Tests**: whether retrieval uses the *preceding context* of the query token, not only the token itself.
2. **Input**: `[y x A]` and `[z x B]` embedded in distinct filler, query `y x`; all other tokens distinct.
3. **Correct output (by convention)**: `A` (the context-matching successor). The single-token rule is ambiguous by design — the most-recent-occurrence rule scores exactly 0.5 ✔.
4. **Available**: two candidate successors that differ only in what preceded `x`.
5. **Irrelevant**: which candidate is more recent (counterbalanced 50/50 ✔).
6. **Shortcuts**: recency and distance — cancelled by counterbalancing and reported per configuration.
7. **Chance**: CP = 0 for any model without a context preference (hand-wired single-token model: 0.000).
8. **Generalisation**: out of distribution for every trained model (two occurrences of a token never occur in training); validity requires `either_rate` ≥ 0.5.
9. **Corruption**: none.
10. **Cannot establish**: *how* the preference is implemented (that needs the path removals), or a preference over contexts longer than 2 tokens.
