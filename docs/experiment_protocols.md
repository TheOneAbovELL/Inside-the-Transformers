# Experiment handbook

Each protocol lists: **hypothesis/question · setup · metrics · controls · failure modes ·
interpretation rules · reproduction command**. All commands write a self-describing experiment
directory under `artifacts/experiments/` (re-run with `itl rerun <dir>`). `<RUN>` is a training run
directory under `artifacts/runs/`.

---

## E0. Training baseline (prerequisite for everything else)

* **Question.** Has the model learned the task to a measurable degree on held-out data?
* **Setup.** `itl train --config configs/experiments/<cfg>.yaml [--set seed=N]`. Initial behaviour
  is evaluated at step 0; validation (same distribution, different seed) and test (`split="eval"`,
  held-out pairs where supported) are evaluated every `eval_every` steps.
* **Metrics.** Masked loss, per-position accuracy, whole-sequence accuracy, with 95% bootstrap CIs;
  baselines: exact solver (must be 1.0), chance, "most frequent other context token" heuristic.
* **Controls.** Leakage audit regenerates the training stream from its seed and counts eval
  sequences that occurred in training (expect 0).
* **Failure modes.** Training stuck on a shortcut (see findings §2); divergence (detected, recorded
  as `status: diverged`); OOD variants that change the vocabulary (refused).
* **Interpretation.** Interpretability experiments on a model whose CI is not far above chance are
  not evidence of a learned mechanism (`itl eval` states this explicitly).
* **Reproduce.** `itl eval --run <RUN> --all-checkpoints`.

## E1. Observational head scores (Experiment F, observational half)

* **Question.** Which heads *look like* previous-token / induction / duplicate-token heads, and
  which have copying OV circuits or strong composition?
* **Setup.** `itl scores --run <RUN>`; scores averaged over scored positions of 512 examples; the
  same scores at the step-0 checkpoint are reported as a baseline.
* **Failure modes.** Mistaking ALiBi's built-in recency (score ≈ 0.17 at init for the steepest
  head) for a learned previous-token head; ignoring that weight scores ignore LN.
* **Interpretation.** Candidates only.

## E2. Head and MLP ablation (Experiments A, B)

* **Question.** Which components are *necessary* for the behaviour?
* **Setup.** `itl ablate --run <RUN> --method {mean,zero,resample} [--neurons-top-k K]`. Head
  ablation at `hook_z` (pre-`W_O`); MLP ablation of the whole block (`hook_mlp_out`) or of the K
  most active neurons (`mlp.hook_post`) — reported separately. Mean/resample statistics come from
  a separate reference batch (seed + 7).
* **Metrics.** Paired Δ log-prob, Δ accuracy, Δ loss with 95% CIs.
* **Controls.** Random-head sets of equal size (in hypothesis tests), different ablation methods,
  matched components from other layers (the full scan shows every head).
* **Failure modes.** Redundancy (small single-head effects); off-distribution zero-ablation.
* **Interpretation.** "Ablating X lowers log-prob by Δ [CI] under mean-ablation" — never "X is
  unimportant" from a small effect.

## E3. Activation patching (Experiment C)

* **Question.** Which activations carry information *sufficient* to restore the clean behaviour?
* **Setup.** `itl patch --run <RUN> --corruption {shuffle,replace,...} --metric {logprob,logit_diff}
  [--direction noise]`. Clean and corrupted batches share lengths and scored targets (asserted).
  Head-level (`z` at scored positions) and region-level (residual stream / sub-layer outputs over
  named position regions) scans.
* **Metrics.** Clean, corrupted and patched metric; recovery R with bootstrap CI over examples;
  number of examples; gap.
* **Controls.** Sanity identities (tested): patching the whole layer-0 residual gives R = 1;
  patching nothing gives R = 0; identical clean/corrupted inputs are flagged uninformative.
* **Failure modes.** Uninformative gap; corruption that changes more than intended; reading
  sufficiency as necessity.
* **Interpretation.** "Restoring X at positions P recovers R [CI] of the gap under corruption K."

## E4. Position and information flow (Experiment D, Task 4)

* **Question.** Where is task-relevant information at each layer, and how does accuracy depend on
  evidence position, distance and context length?
* **Setup.** Region patching (E3) for flow; sweeps:
  `itl sweep --run <RUN> --param evidence_pos --values 1:28` (induction_pair),
  `itl sweep --run <RUN> --param min_len --values 2,4,8,12,16 --task-set max_len=16`
  (induction_repeat; varies segment length within a fixed context),
  `itl sweep --run <RUN> --param query_index --values 0,1 --task-set min_pairs=2 --task-set max_pairs=2` (assoc_recall).
* **Failure modes.** Variants beyond `n_ctx` are *skipped and reported*, never truncated.
* **Interpretation.** Correlations between position and accuracy are behavioural; mechanism claims
  need patching.

## E5. Distractor robustness (Task 5)

* **Setup.** Models trained on `induction_pair` (1 distractor pair) or `assoc_recall` (≤4 noise
  tokens), evaluated with more interference:
  `itl sweep --run <RUN> --param n_distractor_pairs --values 0,1,2,3,4`,
  `itl sweep --run <RUN> --param n_decoys --values 0,2,4,6` (decoys fool the frequency heuristic).
* **Interpretation.** Compare against the "most frequent token" baseline to show the model is not
  using frequency.

## E6. Circuit hypothesis test (Experiment E, F)

* **Setup.** `itl hypothesis --spec configs/hypotheses/<h>.yaml --run <RUN1> --run <RUN2> ...`.
  Candidates selected on `selection_seed` examples; predictions evaluated on disjoint
  `test_seed` examples; mean-ablation statistics from `reference_seed` examples.
* **Prediction types.** `attention_score` (threshold on a per-example score), `ablate_heads` /
  `ablate_mlps` (paired effect, optional random-head control), `path` (sender→receiver via Q, K or
  V), `patch_heads` (recovery).
* **Verdicts.** supported / contradicted / inconclusive — rules in
  `interpretability/hypothesis.py` and printed with each verdict.
* **Interpretation.** Count of verdicts per run; inconsistency across seeds is flagged.
* **Validation.** On the hand-wired model every v1 prediction is supported; on an untrained model
  P1–P5 are contradicted and P8 is uninformative (`tests/scientific/test_hypothesis_framework.py`).

## E7. Mechanism formation (Extension 1)

* **Setup.** `itl formation --run <RUN> --ablate` over all saved checkpoints (dense early).
* **Metrics.** Accuracy, loss, max previous-token and induction scores, effect of ablating the top
  induction head at each checkpoint.
* **Failure modes.** Checkpoint spacing limits temporal resolution; one run is one trajectory.
* **Interpretation.** Co-occurrence is correlational; replicate across seeds before calling it a
  phase transition.

## E8. Cross-model replication (Extension 5)

* Seeds: `python scripts/train_queue.py configs/experiments/induction_2l_attn.yaml:0,1,2`.
* Architecture controls: `induction_1l_attn` (cannot compose; predicted to fail),
  `induction_2l_mlp` (MLPs added), `induction_2l_learnedpos` (positional scheme).
* Run the same hypothesis spec on every run and compare verdict tables.

## Statistics used

Percentile bootstrap (2000 resamples by default) over examples for means and paired differences;
ratio-of-means bootstrap for recovery; empirical rank against control effects (p = (k+1)/(K+1)).
Across seeds we report every seed's value rather than a CI from 3 points.

## E9. Preregistered experiment workflow (session 2)

1. Write the spec (`configs/preregistration/<id>.yaml`) and rationale (`docs/preregistration_<id>.md`):
   hypotheses, competing explanations, measurements, validity criteria, contingencies, decision rules.
2. Calibrate any new measurement on hand-wired models (allowed before freezing — it tests the metric,
   not the hypothesis).
3. Freeze: append the SHA-256 of the spec and the code it depends on to
   `artifacts/preregistration_log.jsonl` (see the session-2 entries for the format).
4. Pilot only what the contingencies need (e.g. trainability); log the pilot decision before analysis.
5. Train the remaining models; analyse with `itl prereg --spec <spec>` (refuses edited specs; logs the
   analysis and its code hash).
6. Report every verdict, deviation and surprise in `docs/findings.md`.

## E10. Context preference and context route

* `itl prereg --spec configs/preregistration/context_matching_v1.yaml` — CP on `context_competition`
  for original / same_order / shuffled models, path-removal secondary measures, frozen H1–H3.
* `python scripts/context_route_exploratory.py <run> ...` — path patching of layer 0 into the group's
  q / k / v on swapped-context inputs (exploratory tool).
* `itl prereg --spec configs/preregistration/context_route_v1.yaml` — the confirmatory version.

## E11. Reproduction audit

* `itl verify --all` — every run's final checkpoint re-evaluated on regenerated eval sets vs its
  `summary.json`, plus config / seed / task / code-fingerprint checks. Session-2 results:
  `artifacts/verify_all_runs.json`, `docs/findings.md` §14.
* `itl rerun <experiment>` — re-executes an analysis from its record and diffs every value.
