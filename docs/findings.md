# Findings

Every number below is read from a saved artifact (run `summary.json` or experiment
`results.json` / CSV). Experiment IDs refer to `artifacts/experiments/<id>/`; run names to
`artifacts/runs/<name>/`. All runs and experiments are on the reference laptop (CPU, heavy
background load). 95% intervals are percentile bootstraps over examples unless stated.

Labels used: **Observation** (raw measurement) · **Supported** (passed a pre-registered test) ·
**Interpretation** (our reading; plausible, not proven) · **Negative result** · **Exploratory**
(hypothesis formed on the same data).

## Summary

| # | claim | status | evidence |
|---|---|---|---|
| 1 | 2-layer attention-only ALiBi models implement variable-offset induction with a **group** of layer-1 heads whose **keys** are built from layer-0 output | **Supported** on 2 held-out seeds (+1 exploratory) | §5 |
| 2 | Adding MLPs re-routes that key-side information **through MLP0** | **Supported** on 1 held-out seed (+1 exploratory) | §9 |
| 3 | 100% task accuracy can come from pure positional copying with **zero** induction (fixed-offset data) | **Supported** (behavioural classification) | §11 |
| 4 | Textbook two-head circuit (sharp previous-token head + K-only composition) | **Contradicted** on all 3 seeds | §4 |
| 5 | The trained heads prefer the earlier occurrence whose **preceding context** matches the query's (context-dependent retrieval) | **Supported (preregistered, 3/3 seeds)**; the session-1 evidence for it was ambiguous and is superseded | §6, §13 |
| 5b | That preference is **learned from training statistics**: present when context is consistent, *reversed* when it is systematically misleading | **Supported (preregistered, 3 seed pairs)**; the reversal was not predicted | §13 |
| 5c | The context signal reaches the induction group through its **queries** (layer-0 → query path) | Mean-removal test **inconclusive** (preregistered); path-patching test **supported** (preregistered on 2 fresh seeds, after an exploratory run on 6 models) | §13 |
| 5d | The steepest-slope ALiBi head **L0H0** is the dominant single carrier of that query context | **Supported (preregistered, 2 fresh seeds)** after exploration on 3 | §15.1 |
| 5e | The preference grows with the number of matching context tokens (≥ 3) and lives in the attention pattern | **Exploratory**, 9–11 models, ground-truth calibrated | §15.2–15.3 |
| 6 | Learned absolute positions → positional shortcut, no induction (3000–4500 steps) | **Negative result**; ALiBi prediction supported | §2 |
| 7 | Plateau, then joint rise of accuracy, induction pattern and causal importance (steps 150–300) | **Observation**, 3 seeds | §7 |
| 8 | Single-head ablation hides fully redundant heads (assoc. recall) | **Exploratory**, 1 model | §8 |
| 9 | Single-pair induction training learns anti-generalising pair priors | **Negative result** | §10 |

---

## 1. Training baselines

| run | task | arch | steps | val acc | held-out acc | chance | notes |
|---|---|---|---|---|---|---|---|
| `induction_2l_attn_s0_20261011-195451` | induction_repeat | 2L attn-only, ALiBi | 3000 | 0.904 | 0.913 | 0.021 | 484 s; leakage 0/1024 |
| `induction_2l_attn_s1_20261011-195451` | induction_repeat | same, seed 1 | 3000 | 0.936 | 0.929 | 0.021 | |
| `induction_2l_attn_s2_20261011-200343` | induction_repeat | same, seed 2 | 3000 | 0.959 | 0.954 | 0.021 | |
| `induction_1l_attn_s0_20261011-200354` | induction_repeat | **1L** attn-only, ALiBi | 3000 | 0.054 | 0.058 | 0.021 | architectural control |
| `induction_2l_attn_s0_20261011-191742` | induction_repeat | 2L attn-only, **learned pos.** | 3000 | 0.354 | 0.363 | 0.021 | config now `induction_2l_learnedpos.yaml` |
| `induction_2l_mlp_s0_20261011-201002` | induction_repeat | 2L **with MLPs**, ALiBi | 3000 | 0.897 | 0.897 | 0.021 | |
| `repeat_pattern_1l_s0_20261011-201853` | repeat_pattern | 1L attn-only | 1500 | 0.814 | 0.819 | 0.063 | Task 1 |
| `assoc_recall_2l_s0_20261011-201853` | assoc_recall | 2L attn-only | 4000 | 1.000 | 1.000 (held-out pairs) | 0.063 | Task 2 |
| `induction_pair_2l_s0_20261011-202054` | induction_pair | 2L attn-only | 4000 | 0.033 | **0.000** (held-out pairs) | 0.031 | **negative result**, §10 |
| `induction_fixed_offset_2l_s0_20261011-203742` | induction_repeat, fixed L=12 & prefix | 2L, learned pos. | 3000 | 1.000 | 1.000 | 0.021 | confound demo, §11 |
| `induction_2l_mlp_s1_20261011-202536` | induction_repeat | 2L with MLPs, seed 1 | 3000 | 0.902 | 0.902 | 0.021 | v3 confirmatory |

Every run's exact solver scores 1.000 on its evaluation sets (generator sanity).

**Leakage audit (regenerates each run's training stream from its seed):** 0 overlaps for every
induction run. **Exceptions:** `repeat_pattern` — 239/512 validation and 253/512 test sequences
occurred in training (periods 2–3 over 16 symbols allow only 240 and 3,360 distinct sequences); and
`assoc_recall` — 4/512 validation sequences (0/512 on the held-out-pair test set). Re-evaluated on
never-seen sequences only (`itl eval --unseen-only`, experiment `*_eval_repeat_pattern_*`):
`repeat_pattern` accuracy is 0.737 [0.730, 0.744] (val) / 0.740 [0.733, 0.748] (test) on unseen
sequences vs 0.815 / 0.819 overall, and whole-sequence accuracy on unseen sequences is 0.000. The
unseen subset is dominated by longer periods, so the gap mixes memorisation with difficulty.

## 2. Negative result: learned absolute positions → positional shortcut, no induction circuit

* **Observation.** The 2L attention-only model with learned positions plateaued at 35% (3000 steps);
  a longer diagnostic run at lr 3e-3 reached 42.5% at step 4500 and was **stopped manually** (not
  completed; `artifacts/diag_long.log`). At every checkpoint every layer-0 head had previous-token
  score 0.00; layer-0 heads attended to a window ~8–16 positions back (a smeared positional copy of
  the first copy), and per-position accuracy was U-shaped within the copy.
* **Tested and rejected explanation.** "Initialisation too small to escape the QK saddle": a sweep
  of init std 0.0625–0.125 did not help (0.149–0.190 at 1200 steps vs 0.287 for the original).
* **Bug found on the way.** Fixed sinusoidal positions were ~50× larger than token embeddings,
  so LayerNorm erased token identity → chance-level learning (loss = ln 48). Fixed and regression-tested.
  (The two pre-fix runs' recorded numbers can no longer be regenerated with current code — §14.)
* **Supported prediction.** If the barrier is that absolute positions make far-offset lookback as
  cheap as recency, ALiBi (recency-biased from initialisation) should enable induction: 0.837 at
  1500 steps vs 0.128–0.287 for learned / sinusoidal / one-head / train-on-all variants
  (sweep runs `sweep_*` in `artifacts/runs`). Full ALiBi runs: §1.
* **Interpretation.** In this setting the positional scheme determined *whether* the mechanism
  formed. This is one task and 3000-step budgets; with much longer training the learned-position
  model might still find induction.

## 3. Architectural control

**Observation.** Identical data and training, 1 layer instead of 2: 5.4% val accuracy (chance
2.1%) vs 90–95% for 2 layers. Consistent with the task requiring composition across layers.

## 4. Pre-registered hypothesis v1 (textbook two-head circuit) — partly contradicted

Experiment `20261011-201311_hypothesis_induction_circuit_2l_v1` (3 ALiBi seeds + learned-position model,
512 test examples each; candidates chosen on separate examples):

| prediction | seed 0 | seed 1 | seed 2 | learned-pos |
|---|---|---|---|---|
| P1 top L1 head has induction pattern > 0.4 | ✔ 0.56 | ✔ 0.57 | ✔ 0.60 | ✘ 0.06 |
| P2 top L0 head has prev-token pattern > 0.4 | ✘ 0.174 | ✘ 0.174 | ✘ 0.173 | ✘ |
| P3 ablating induction head: > 0.5 nats and beyond random controls | ? −2.19 | ? −2.06 | ? −1.64 | ✘ |
| P4 ablating prev-token candidate (L0H0) | ✔ −4.37 | ✔ −4.82 | ✔ −5.01 | ? |
| P5 L0H0 → induction head **keys** only | ✔ −4.23 | ✔ −4.33 | ✔ −3.97 | ✘ |
| P6 L0H0 → **queries** has little effect (< 0.3) | ✘ −3.21 | ✘ −3.09 | ✘ −2.44 | ✔ (vacuous) |
| P7 L0H0 → **values** has little effect (< 0.3) | ✘ −0.90 | ✘ −0.83 | ✘ −0.58 | ✔ (vacuous) |
| P8 patching induction head restores ≥ 50% | ✔ 0.78 | ✔ 0.78 | ✔ 0.78 | ✘ |

✔ supported · ✘ contradicted · ? inconclusive. P3 is inconclusive because other heads (also part of
the circuit, see §5) are as damaging as the candidate. P6/P7 on the learned-position model are
vacuously "supported" (no effect because there is no circuit) — a flaw in v1's design that the
`requires:` feature now addresses (v1 was not edited).

**Observation.** L0H0's previous-token score (0.174) equals its value at initialisation — ALiBi's
built-in recency — yet ablating it drops accuracy 0.91 → 0.03 (seed 0, experiment
`20261011-200547_ablation_*`).

## 5. Distributed circuit (v2): supported on held-out seeds

Exploratory analysis on seed 0 (scores `20261011-200427_*`, ablation `20261011-200547_*`, patching
`20261011-200651_*`) showed three layer-1 heads with induction patterns and OV copying, three
important layer-0 heads, and redundancy (L1H1/L1H3: small single ablation effects but 46%/31%
patching recovery each). Under the shuffle corruption the model's logit difference flips to −4.89
— it follows the shuffled successors.

Hypothesis v2 (`configs/hypotheses/induction_2l_distributed_v2.yaml`) was written from seed 0 and
tested on seeds 1–2 (`20261011-201458_*`; seed 0 exploratory in `20261011-201538_*`):

| prediction | seed 1 | seed 2 | (seed 0, exploratory) |
|---|---|---|---|
| Q1 ablate induction group (≥ 2 nats, beyond all 10 control sets) | ✔ −8.60 | ✔ −9.49 | ✔ −8.25 |
| Q2 patch group at answer positions (≥ 0.8) | ✔ 0.978 | ✔ 0.979 | ✔ 0.978 |
| Q3 layer-0 → group **keys** (≥ 2 nats) | ✔ −8.74 | ✔ −9.59 | ✔ −8.20 |
| Q4 layer-0 → group **queries** (< 1 nat) | ✘ −2.65 | ✘ −1.89 | ✘ −3.03 |
| Q5 layer-0 → group **values** (< 1 nat) | ✔ −0.65 | ✔ −0.56 | ✔ −0.69 |
| Q6 group induction pattern > 0.33 | ✔ 0.49 | ✔ 0.56 | ✔ 0.46 |

Caveat: v2 was written before, but run after, seeing v1's results on seeds 1–2.

**Strongest evidence-supported finding.** In these 2-layer attention-only ALiBi models, the
second copy is predicted by a group of layer-1 heads whose output at the answer positions carries
~98% of the task-relevant information, and whose keys depend on layer-0 output (removing it from
the keys alone costs 8–10 nats, from the values < 0.7). Replicated on two seeds not used to form
the hypothesis.

## 6. Follow-up: query-side composition and context matching

Experiment `20261011-201658_investigation_q_composition` (3 seeds):
* **Not an artifact.** The layer-0 → query effect persists under zero (−1.5 to −2.3), mean
  (−1.9 to −3.1) and resample (−3.9 to −4.6) replacement. Value-path interventions change the group's
  attention pattern by exactly 0.00 (consistency check).
* **Observation.** L0H0 attends 22% to the current token, 17% previous, 14% two back, 11% three
  back, 37% further — a recency-weighted context summary, not a previous-token head.
* **Observation.** The query-path intervention lowers the group's induction score by 0.18–0.31; the
  key-path intervention collapses it to ~0.05.

Experiment `20261011-201819_investigation_context_sensitivity` (512 examples per model): replacing
only the token *before* the matched earlier occurrence lowers log-prob at the affected query by
−1.20 [−1.36, −1.05], −0.83 [−0.98, −0.68], −0.59 [−0.73, −0.47] (seeds 0–2) vs −0.13/−0.11/−0.05
at a control query; the hand-wired pure single-token induction model: exactly 0.

**Session-1 interpretation (kept for the record).** "The trained heads match recent *context*
(current + preceding tokens, via layer-0 summaries reaching both queries and keys), i.e. a fuzzy
n-gram induction, rather than the textbook single-token induction head."

**Session-2 audit of that interpretation.** The evidence above did not establish it:
(i) the "context sensitivity" drop is also predicted by non-specific interference in imperfect,
blurry keys; (ii) on a hand-wired model that *does* match context, the same measurement reads
**exactly 0** whenever the current-token match dominates — it is neither sufficient nor necessary
evidence (`test_context_sensitivity_cannot_detect_a_saturated_context_matching_mechanism`);
(iii) query dependence on layer 0 could reflect a current-token readout. The claim was therefore
re-tested with a preregistered experiment designed to separate these explanations — see §13, where
context-dependent retrieval is **supported** and shown to be **learned from training statistics**.

## 7. Formation during training (Extension 1)

Experiments `20261011-201946_*`, `202018_*`, `202057_*` (17 checkpoints each, 256 examples):
in all three seeds, loss stays at ≈ ln 48 until step ~150; between steps 150 and 300 accuracy rises
from ≈ 0.04 to 0.49–0.54, the leading induction score from 0.04 to 0.30–0.35, and the effect of
ablating the leading induction head from ≈ 0 to −0.6 to −1.0 nats; layer-0 previous-token score
stays at its initial 0.17–0.19 throughout. The identity of the leading head changes during training
(L1H1 early → L1H2 or L1H3 later).
**Interpretation.** A plateau followed by a rapid joint rise of behaviour, attention signature and
causal importance — consistent with the phase change described by Olsson et al. (2022), here for
three seeds of one toy setup at 50-step resolution. Not evidence of universality.

## 8. Position, distance and interference (Tasks 4–5)

Seed 1 of the principal model:
* Segment length sweep (`20261011-203349_*`): accuracy 0.547 (L=2), 0.682 (3), 0.732 (4), 0.859 (6),
  0.908 (8), 0.962 (12), 0.955 (16), 0.866 (18), 0.752 (19); trained range 6–16.
* Zero-shot on single-pair induction (`20261011-203025_*`, n_symbols 48, length 40): accuracy 0.000
  with evidence at positions 1 and 4, 0.51–0.66 at positions 8–36. Training never placed the first
  copy before position 8. **Interpretation:** the mechanism did not generalise to early positions
  even with relative positions.
* Distractor pairs 0–4 (`20261011-203119_*`): 0.49–0.56, CIs overlapping; decoys 0–8
  (`20261011-203154_*`): 0.46–0.53. The most-frequent-token heuristic scores 0 throughout.
  **Observation:** no detectable degradation from these interference types.

Associative-recall model (`assoc_recall_2l_s0_20261011-201853`): accuracy ≥ 0.988 for every queried
pair index 0–7 with 8 pairs (`20261011-204347_*`), 1.000 for 0–4 noise tokens (`20261011-204414_*`)
and 1.000 for 2–8 pairs (`20261011-204445_*`). **Observation:** at ceiling within the trained range,
so these sweeps cannot reveal degradation; longer contexts would exceed the model's `n_ctx` (not
tested — variants beyond `n_ctx` are refused, never truncated).

### Redundancy in associative recall (exploratory)

Scans (`*_scores_assoc_*`, `*_ablation_assoc_*`, `*_patching_assoc_*`): three layer-1 heads have
induction scores 0.967–0.970; ablating any one changes log-prob by ≤ 0.011 nats, yet patching each
alone restores 21%/27%/40% (symmetric query-key corruption, logit difference); ablating L0H0 drops
accuracy 1.00 → 0.05. Hypothesis `assoc_recall_redundancy.yaml` (written after these scans, so
exploratory): single top head −0.008 nats; whole group −5.08; group patching 1.00; layer-0 → group
keys −4.95; layer-0 → group queries ≈ 0.000 — all 5 supported.
**Interpretation.** Textbook redundancy: single-component ablation would have called these heads
unimportant. And unlike `induction_repeat`, queries do not depend on layer 0 here — consistent with
the training-statistics account later supported in §13: in this format the token preceding a key
differs between its two occurrences (as in the `shuffled` condition), so context gives no benefit.

## 9. MLP models (Experiment B, Extension 5)

* Ablation (`20261011-202320_*`, seed 0): L0H0/L0H1/L0H3 −3.9/−3.1/−3.2 nats; L1H3 −3.19, L1H2
  −1.42; **L0 MLP block −2.10**, L1 MLP −0.42; ablating the 16 most active neurons of either MLP:
  ≤ 0.04 (the MLP's role is not concentrated in its most active units).
* v2 on the MLP model (`20261011-202220_*`): group ablation and patching supported, but **Q3
  contradicted**: removing layer-0 heads from the group's keys costs only −0.39 nats.
* v3 (MLP mediation, exploratory on seed 0, `20261011-202542_*`): MLP0 → keys −3.11; heads direct
  → keys −0.39; both −5.86; MLP0 → queries −0.40; group patching 0.98 — all 5 supported.
* **v3 confirmatory on MLP seed 1** (`20261011-205233_hypothesis_induction_mlp_mediation_v3`, a model
  trained after v3 was written): MLP0 → keys −3.41 [−3.50, −3.32]; heads direct → keys −0.30;
  both −6.69; MLP0 → queries −0.29; group patching 0.968 — **all 5 supported**.
* **Supported finding.** Adding MLP blocks re-routes the key-side composition: layer-0 head
  information reaches the layer-1 induction keys mainly through MLP0. The v2 prediction that held
  for attention-only models (direct layer-0 → keys) fails for MLP models — circuits found in one
  architecture should not be assumed in another.

## 10. Negative result: single-pair induction training

`induction_pair_2l_s0_20261011-202054` (4000 steps, 1 distractor pair, held-out (A,B) pairs):
validation accuracy 0.033 (chance 0.031); **held-out-pair accuracy 0.000** (initially 0.020). Under
chance guessing, 0 of 512 correct has probability (31/32)^512 ≈ 8.7 × 10⁻⁸, so the model actively
avoids held-out answers. **Interpretation:** with one scored position per sequence, training
learned pair-specific priors (which B tends to follow which A in training pairs) instead of
in-context retrieval — exactly the failure the held-out split exists to expose. Untested remedies:
more steps, denser supervision (several queries per sequence), curriculum from `induction_repeat`.
Note the `induction_repeat` models solve this task zero-shot at 0.5–0.66 (§8), so the mechanism is
learnable; the bottleneck is the training signal.

## 11. Behavioural mechanism classification and the fixed-offset confound

Experiment `20261011-205308_investigation_mechanism_type` (512 examples per model): on inputs whose
first copy is shuffled, the fraction of argmax predictions equal to the answer implied by each
strategy (positions where both strategies predict the same token: 0.109, the chance-coincidence
baseline):

| model | content induction (shuffled successor) | positional copy (fixed offset) | clean answer |
|---|---|---|---|
| hand-wired single-token induction | 1.000 | 0.109 | 0.000 |
| ALiBi 2L seed 0 / 1 / 2 | 0.568 / 0.660 / 0.756 | 0.091 / 0.101 / 0.095 | 0.03 / 0.03 / 0.02 |
| learned-pos 2L (35% accuracy) | 0.144 | 0.191 | 0.061 |
| **fixed-offset model, own task (100% accuracy)** | 0.088 | **1.000** | 0.081 |
| fixed-offset model, variable offsets | 0.071 | 0.094 | 0.065 |
| 1-layer ALiBi | 0.076 | 0.070 | 0.042 |

* **Supported finding.** A model can reach 100% on a repeated-sequence task with **no** content
  induction at all: the fixed-offset model copies the token at a fixed distance in 100% of cases and
  collapses on variable offsets (accuracy and both strategies near chance). Task accuracy alone does
  not identify the mechanism.
* **Observation.** The ALiBi models follow the shuffled successors 57–76% of the time (positional
  copying at the coincidence rate). That they are below the hand-wired model's 1.00 is consistent with
  context matching (§6): shuffling also destroys the matching preceding context.

## 12. Scope: what was not done

* Sparse autoencoders (Extension 4): deliberately **not** implemented — reasoning in
  `docs/research_log.md` (D7). The open questions here are component- and path-level, which the
  current tools answer; the models (d_model 64, near one-hot inputs) are too small for feature
  decomposition to add evidence; and SAE features would still need causal tests to be interpreted.
* Path patching covers named edges between heads/MLPs and head Q/K/V inputs, MLP inputs and the
  direct path to the logits (`interpretability/path_patching.py`, session 2). Multi-hop paths are
  expressed by patching several edges at once; there is no automatic circuit discovery.
* Larger models, natural-language data, more than 3–5 seeds per configuration.

## 13. Preregistered experiments (session 2)

### 13.1 context_matching_v1 — does retrieval use preceding context, and is that learned?

Preregistration: `configs/preregistration/context_matching_v1.yaml` + rationale
`docs/preregistration_context_matching_v1.md`, frozen 2026-10-11T22:33:44 (SHA-256 in
`artifacts/preregistration_log.jsonl`) **before** any context-preference value was computed on a
trained model. Pilot decision logged before analysis (both pilots ≥ 0.80 → 3000 steps).
Analysis: `20261011-225402_prereg_context_matching_v1` (re-run 20261011-231409: 805 values identical).

**Primary measurement** — context preference CP = mean logit[A] − logit[B] on `context_competition`
(the query token occurred twice earlier; only the A-occurrence shares the query's preceding token;
recency counterbalanced), n = 1024, seed 606.

| condition (training data) | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| calibration: hand-wired single-token / context model | −0.000 | +1.985 | |
| original (contiguous repeats, all positions scored) | **+2.31** [2.01, 2.62] | **+2.17** [1.85, 2.51] | **+1.91** [1.50, 2.30] |
| same_order (pairs, contiguous) | **+1.78** [1.46, 2.11] | **+1.61** [1.30, 1.93] | **+1.37** [1.04, 1.73] |
| shuffled (pairs, context never repeats) | **−0.54** [−0.96, −0.11] | **−0.68** [−1.13, −0.28] | **−0.49** [−0.91, −0.09] |

All 9 trained models valid (own-task validation accuracy 0.889–0.959; either-rate 0.71–0.92).

* **H1 supported** — 3/3 original seeds have CP lower CI > 0.5. Both recency configurations are
  positive (match-recent +2.42…+2.67, match-earlier +1.40…+1.97), so this is not a recency effect.
* **H2 supported** — paired difference CP_same − CP_shuffled = +2.32 [1.78, 2.85], +2.29 [1.74, 2.82],
  +1.86 [1.36, 2.40]; mean +2.16.
* **Unexpected (reported, not hidden):** the preregistered expectation for `shuffled` was CP ≈ 0; all
  three are significantly **negative** — models trained where context is systematically misleading
  learned to *disfavour* context-matching occurrences. **Interpretation (untested):** in shuffled
  data, layer-0 summaries make keys that share the query's preceding token look similar to the
  correct key while never being correct, so training pushes the context term negative.
* **Secondary (dose-response):** with two matching context tokens CP rises to +3.75/+4.06/+3.79
  (original) and +3.39/+3.60/+3.32 (same_order), and stays negative for shuffled (−0.82/−0.77/−0.57).
* **Secondary:** the session-1 sensitivity measure is −0.59…−1.20 for original/same_order models and
  ≈ 0 (+0.01…+0.12) for shuffled ones — in trained models it tracks the preference, although it is
  blind to it in the saturated hand-wired model.
* **H3 inconclusive** (frozen rule) — removing layer-0 output (mean replacement) from the group's
  queries drove CP to ≤ 0 in all 6 qualifying models (−0.13…−0.60), but in 3 of them it also broke
  induction itself (either-rate 0.34–0.46 < 0.5), so the rule correctly calls those uninterpretable
  (3/6 supported < 2/3). **Lesson:** mean-removal is too blunt to localise a signal carried by an
  input the circuit also needs for its main function.

### 13.2 context_route_v1 — does context enter through the queries? (confirmatory)

Exploratory first: `20261011-225623_investigation_context_route_exploratory` (6 models, post hoc).
Clean = competition input; corrupt = the query's preceding token swapped to the other occurrence's
context (only positions ≥ T−3 differ). Denoising **path patching** of all layer-0 heads into ONLY
the group's q / k / v inputs. Ground truth (hand-wired context model): R_q 1.00, R_k 0.00, R_v 0.00.

Then preregistered (`configs/preregistration/context_route_v1.yaml`, frozen 2026-10-11T22:58:12 with
no seed-3/4 run in existence) and tested on two fresh seeds of the original configuration
(validation accuracy 0.935, 0.953): `20261011-231249_prereg_context_route_v1`.

| model | clean CP → swapped | R_q | R_k | R_v | R_all (identity) |
|---|---|---|---|---|---|
| seed 3 | +1.80 → −2.14 | **1.27** [1.22, 1.33] | −0.15 [−0.20, −0.11] | 0.02 [−0.00, 0.05] | 1.03 |
| seed 4 | +2.03 → −2.32 | **1.24** [1.18, 1.29] | −0.15 [−0.19, −0.11] | −0.00 [−0.03, 0.02] | 1.00 |

* **H3b supported** — in both fresh models R_q lower CI > 0.5 and R_k, R_v upper CI < 0.5.
  Same pattern in all 6 exploratory models (R_q 1.24–1.42).
* **Observation:** R_q > 1 and R_k < 0 with R_all ≈ 1 — the query and key routes are not additive;
  the late-position keys push slightly against the preference.
* **Deviation (logged):** the frozen YAML had a syntax error in one free-text validity line (a value
  starting with `>`); the analysis verifies the unmodified file's hash, quotes that value in memory
  only, and implements the rule verbatim. Recorded as a `deviation` event in the preregistration log.

**Supported account (scope: 2-layer attention-only ALiBi models on these tasks).** The induction
group retrieves by *context*: its queries carry the identity of the query token's predecessor
(delivered by layer 0) in addition to the current token, so among several earlier occurrences the
one whose preceding context matches is preferred. Whether this preference appears — and its sign —
is set by the training statistics. Not established: which layer-0 heads carry it individually, and
whether larger models do anything similar.

## 14. Reproduction audit and methodological findings (session 2)

* **Runs.** `itl verify --all` re-evaluates every final checkpoint on validation/test sets
  regenerated from `seeds.json` and compares with `summary.json`: **26 of 29 runs reproduce
  exactly** (every principal run, and all 8 session-2 runs; output `artifacts/verify_all_final.txt`).
  One was stopped manually (diag run, documented). **Two cannot be reproduced:** `sweep_sinusoidal_*` and `sweep_sin_one_head_*` were trained before the
  sinusoidal-scale fix; the table is a non-persistent buffer, so current code computes a different
  function for the same weights. Their recorded "chance-level" numbers remain the evidence for that
  bug but cannot be regenerated. Runs now record a source-code fingerprint (`env.json`).
* **Analyses.** `itl rerun` re-executes an experiment from its record and diffs every value:
  hypothesis v2 (564 values) and v3 (279 values) and the context-matching prereg (805) are identical.
* **Defect fixed:** resample ablation reused identical donor indices in every evaluation chunk.
* **Metric lessons, each now a test:** (1) the session-1 context-sensitivity measure is blind to a
  saturated context-matching mechanism; (2) in path patching, log-prob recovery can read ≈ 0.33 for a
  completely broken route, because a head that cannot find its target attends diffusely over clean
  tokens — accuracy shows the true 0; (3) patching recoveries > 1 occur (R_q 1.24–1.42) and are
  reported, not clipped.
* **Hypothesis-framework audit:** selection / test / reference sets verified disjoint (0 shared
  sequences, now reported per run); every verdict also reported under Bonferroni-adjusted intervals.

## 15. Session 3 (2026-10-11/12): which component carries the context, and how much context

No new models were trained (CPU at 100% and ~1.2 GB free RAM from unrelated workloads); every
analysis uses existing checkpoints. All records are `itl investigate` / `itl prereg` records and can
be re-executed with `itl rerun`.

### 15.1 context_heads_v1 — the steepest ALiBi head carries the query's context (preregistered)

Exploratory first (`20261011-235726_investigation_context_heads`, seeds 0–2): path-patching ONE
layer-0 head at a time into the induction group's queries (swap-context design of §13.2):
R_q(L0H0) = 1.34 / 1.31 / 1.30; L0H1 0.09 / 0.36 / 0.69 (inconsistent); L0H2 ≈ 0 / 0.00 / −0.14;
L0H3 −0.05 / −0.23 / −0.28. Head indices are comparable across seeds because ALiBi fixes head h's
slope: L0H0 is always the steepest (most recency-biased) head.

Preregistered `configs/preregistration/context_heads_v1.yaml` (frozen 2026-10-11T23:59:04; the log
lists the only prior per-head record) and tested on seeds 3–4, never analysed per head before:
`20261011-235940_prereg_context_heads_v1`.

| seed | L0H0 → q | L0H1 → q | L0H2 → q | L0H3 → q |
|---|---|---|---|---|
| 3 | **1.32** [1.24, 1.42] | 0.42 [0.35, 0.49] | 0.07 [0.05, 0.08] | −0.29 [−0.32, −0.26] |
| 4 | **1.28** [1.21, 1.35] | 0.36 [0.30, 0.42] | −0.04 [−0.05, −0.03] | −0.21 [−0.23, −0.18] |

**H4 supported** (frozen rule): in both models L0H0's lower CI > 0.8 and above every other head's
upper CI; L0H2 and L0H3 upper CI < 0.3. Ground truth: in the hand-wired context model exactly the
designed head carries it (R 1.00 vs 0.00).

### 15.2 Where the preference lives: the attention pattern (exploratory)

`20261012-000102_investigation_competition_attention`: attention from the query to the A- vs
B-successor on `context_competition`. In every original and same_order model one group head is
strongly context-selective (A − B = +0.13 to +0.19; attention to B ≈ 0.05; e.g. L1H2 in original
seeds 0/1, L1H3 in seed 2). In every shuffled model all group heads prefer B (A − B = −0.03 to
−0.08). Ground truth: context model +0.099, single-token model 0.000. **Interpretation:** the
preference (and its reversal after shuffled training) is implemented in where the heads look (QK),
not in what they copy (OV) — consistent with H3b (the signal enters through the queries).

### 15.3 Dose-response: the models use at least three tokens of context (exploratory)

`20261012-000131_investigation_context_dose` and `20261012-000424_investigation_context_dose`
(CP for 1 / 2 / 3 matching preceding tokens, n = 1024 each):

| models | k = 1 | k = 2 | k = 3 |
|---|---|---|---|
| hand-wired context (matches exactly 1 token) | +1.99 | +1.99 | +1.99 |
| hand-wired single-token | 0.00 | 0.00 | 0.00 |
| original seeds 0–2 | +2.03 / +2.05 / +2.28 | +3.83 / +4.22 / +4.39 | +5.09 / +5.24 / +5.20 |
| same_order seeds 0–2 | +1.78 / +1.44 / +1.31 | +3.79 / +3.69 / +3.78 | +4.79 / +4.66 / +4.74 |
| shuffled seeds 0–2 | −0.47 / −0.32 / −0.52 | −0.87 / −0.60 / −0.31 | −1.04 / −0.67 / −0.52 |
| with MLPs seeds 0–1 | +2.94 / +2.80 | +4.86 / +4.87 | +5.38 / +6.06 |

(The k = 1 values differ slightly from §13.1 because this analysis uses seed 606 + k.) The flat
hand-wired curve shows the test can tell how many tokens a mechanism matches; the trained models'
curves rise with every extra matching token.

### 15.4 Models with MLPs (exploratory)

`20261012-000547_investigation_context_route`, `20261012-000634_investigation_context_heads`:
context also enters through the queries (R_q 1.04 / 1.11; R_k −0.01 / −0.03; R_v ≈ 0.01). Per sender:
L0H0 → q 1.07 / 1.22, MLP0 → q 0.30 / 0.22, L0H1 → q −0.27 / −0.25. **Contrast with v3 (§9):** the
key-side previous-token information in these same models travels mainly *through* MLP0, but the
query-side context arrives mainly *directly* from L0H0. Shuffled-first-copy classification
(`20261012-000807_investigation_mechanism_type`): MLP seed 0 follows content induction 0.49
(positional 0.09); the library version reproduces the session-1 script's numbers for seed 0 exactly
(0.568 / 0.091).

### 15.5 Engineering in session 3

* `itl investigate {context-heads, competition-attention, context-dose, context-route, mechanism-type}`
  writes rerunnable records (`experiments/investigations.py`); `itl rerun` re-executes them
  (tested to reproduce bit-for-bit).
* `context_competition` supports up to 3 matching context tokens.
* **Defect found and fixed:** the "all layer-0 edges" path patch failed for MLP models (MLP0 would have
  to be patched sender and receiver at once); fixed by dropping edges into a patched MLP, with an
  identity test against activation patching (`test_layer0_union_with_mlp_equals_activation_patching`).
* An incomplete record (`20261012-000209_investigation_context_route`, written before that failure)
  is preserved; three investigations exist twice (deterministic duplicates; verified identical).
