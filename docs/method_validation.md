# Method validation against ground truth

Every analysis method is checked on hand-wired models whose mechanism is known by construction
(`src/transformer_lab/models/handwired.py`). A method that fails here is not used for claims about
trained models. Each row: what the ground truth implies, what was observed, the tolerance, and what
the check does **not** establish.

## Ground-truth models

| model | mechanism (by construction) | solves |
|---|---|---|
| single-token induction | L0H0 previous-token head → (K-composition) → L1H0 induction head; L0H1, L1H1 dead | every `induction_repeat` / `pair_mode` / `induction_pair` variant within n_ctx: accuracy 1.000 [`test_construction_survives_input_variations`] |
| context-matching (`context_gain=2`) | as above + L0H1 previous-previous head; L1H0's **query** also reads the previous-token slot and its key the prev-prev slot | same; prefers the context-matching occurrence (CP +1.985) |
| MLP relay (`mlp_relay=True`) | L0H0 → **MLP0** → relay slot → L1H0 keys (no direct L0H0 → key path) | same |

## Results

| method | ground-truth expectation | observed | tolerance | does NOT establish | test |
|---|---|---|---|---|---|
| activation capture | cached tensors reproduce the forward pass (residual identities, logits, per-head sums) | exact | 1e-5–1e-6 | anything about meaning | `test_hooks.py::test_cached_activations_reproduce_forward_computation` |
| attention scores | L0H0 prev-token ≈ 1, L1H0 induction ≈ 1, dead heads low | > 0.99 / > 0.99 / < 0.5 | — | causal role | `test_attention_scores_identify_heads` |
| zero / mean / resample ablation | removing L0H0 or L1H0 breaks the task; dead heads do nothing | acc < 0.35 for both circuit heads under all three methods; dead heads 1.000 | — | sufficiency; behaviour under redundancy (the hand-wired model has none) | `test_ablation_effects_match_ground_truth[zero/mean/resample]` |
| activation patching | patching L1H0 at answer positions restores; dead head 0; **L0H0 alone ≈ 0 although necessary** (AND structure) | R > 0.95 / |R| < 0.02 / |R| < 0.05 | stated | necessity | `test_patching_recovers_known_components` |
| patching sanity identities | whole layer-0 residual → R = 1; nothing → R = 0; identical inputs → *uninformative*, R = NaN | as expected | 1e-5 | — | `test_patching_sanity_identities` |
| mean-replacement path intervention (Q/K/V) | L0H0 → L1H0 **keys** breaks; → queries / values no effect; identity replacement is a no-op | acc < 0.35 / 1.000 / 1.000; no-op exact | 1e-5 | indirect routes (see relay row) | `test_path_interventions_isolate_k_composition`, `test_path_with_identity_replacement_is_a_no_op` |
| path intervention with MLP senders | relay model: L0H0 → keys direct has no effect; MLP0 → keys breaks | acc 1.000 / < 0.35 | — | — | `test_relay_ground_truth_mean_replacement_paths` |
| path patching (clean ↔ corrupt) | **completeness**: a sender along ALL outgoing edges == activation patching of that sender | equal for heads and MLPs, with and without MLPs | 1e-4 | — | `test_all_edges_equals_activation_patching` (5 cases) |
| path patching routes | relay model, noising: direct edge R = 1.0; edge into MLP0 R = 0.0; MLP0 → keys R = 0.0 (accuracy metric). With log-prob the broken route reads R ≈ 0.33 — a metric floor, not a partial route | as expected | exact | — | `test_relay_ground_truth_routes_noising` |
| path patching (context route) | context model: context enters via **queries** → R_q = 1, R_k = R_v = 0 | 1.00 / 0.00 / 0.00 | — | — | `scripts/context_route_exploratory.py`, `context_route_v1` analysis |
| direct logit attribution | parts sum to the logit; L1H0 dominant; L0H0 contributes 0 directly | reconstruction error < 1e-3; as expected | 1e-3 | indirect effects | `test_dla_attributes_logit_to_induction_head` |
| OV copying / composition scores | L1H0 OV copies (1.0); K-composition L0H0→L1H0 exceeds Q- and V-composition | as expected | — | data-dependent behaviour (weights only, LN ignored) | `test_ov_copying_score`, `test_composition_scores_find_k_composition` |
| hypothesis framework | true hypothesis on ground truth: all supported; untrained model: P1–P5 contradicted, P8 uninformative; `requires:` blocks vacuous specificity; selection/test sets disjoint | as expected | — | correctness of hypotheses not tested | `test_hypothesis_framework.py` (4 tests) |
| context preference (CP) | single-token model 0; context model > 0 in both recency configurations | 0.000 / +1.985 (+1.94 recent, +2.03 earlier) | 0.05 | mechanism | `test_context_measurements.py` |
| per-sender context route (session 3) | context model: only L0H0 → queries carries context | R 1.00; L0H1 → q 0.00; keys 0.00 | 0.01 | redundancy among senders | `test_per_sender_route_identifies_the_query_carrier` |
| competition attention (session 3) | context model attends more to A than B; single-token model equally | +0.099 / 0.000 | 1e-3 | — | `test_competition_attention_locates_preference_in_the_pattern` |
| context dose-response (session 3) | a model matching exactly one token gives a FLAT curve over 1–3 matching tokens | +1.99 / +1.99 / +1.99 | 0.05 | — | `test_dose_response_reflects_how_many_tokens_are_matched` |
| path patching, MLP union (session 3) | all layer-0 heads + MLP0 along all edges (edges into the patched MLP dropped) == activation patching of layer 0 | equal | 1e-4 | — | `test_layer0_union_with_mlp_equals_activation_patching` |
| old "context sensitivity" | — | **0.000 for BOTH models**: blind to a saturated context-matching mechanism; it is not specific evidence either way | — | — | `test_context_sensitivity_cannot_detect_a_saturated_context_matching_mechanism` |

## What ground-truth validation cannot cover

* **Redundancy and self-repair.** Hand-wired circuits have no backup components; methods are
  additionally checked on trained models where redundancy is observed (assoc-recall group:
  single-head ablation −0.008 nats, group −5.08).
* **Superposition / distributed features.** One-hot residual slots are the easy case.
* **Soft attention.** Ground-truth heads are near one-hot (gain 100); trained heads are diffuse,
  which is exactly where metrics like log-prob recovery can mislead (see the relay row).
* **LayerNorm.** Ground-truth models use `norm: none`; LN-specific effects (scale changes under
  interventions before LN) are documented, not ground-truth-tested.
