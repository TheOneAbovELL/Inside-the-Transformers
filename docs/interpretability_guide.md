# Mechanistic interpretability guide

How to reason about evidence in this lab — and how it can mislead you. Examples refer to real
results recorded in `docs/findings.md`.

## 1. What "circuit" means here

A **circuit** is a claim of the form: *on inputs from distribution D, behaviour B is produced
mainly by components C₁…Cₖ interacting through specific paths P (e.g. head S's output feeding
head R's keys), and other components contribute little.* It is a causal claim about a trained
model, under stated conditions. Concretely, a hypothesis in `configs/hypotheses/*.yaml` names:
the behaviour and metric, the candidate components (or a rule to choose them), falsifiable
predictions with thresholds, controls and alternative explanations.

A circuit is **not**: a heatmap, a list of heads with high scores, or a story that fits one example.

## 2. Kinds of evidence and what each can establish

| evidence | tool | establishes | cannot establish |
|---|---|---|---|
| Attention pattern / scores | `itl scores`, attention explorer | where a head looks on these inputs | that what it reads is used; that it matters |
| Weight-based scores (OV copying, composition) | `itl scores` | what the parameters *could* compute | what they *do* compute on the data distribution |
| Correlation of activations with features | inspection tools | decodability | use (Belinkov, 2022) |
| Direct logit attribution | `itl trace` | the direct, linear contribution to one logit with LN frozen | indirect effects; what happens if the component is removed |
| Ablation (zero / mean / resample) | `itl ablate` | **necessity** of a component under that replacement | sufficiency; importance of redundant components |
| Activation patching (denoising) | `itl patch` | **sufficiency** of an activation to restore behaviour, given the rest of the corrupted run | necessity; it can miss AND-like components |
| Path intervention | hypothesis `type: path` | that S affects R *through a specific input* (Q, K or V) | that nothing else also carries the signal |
| Replication across seeds / sizes | `itl hypothesis --run A --run B ...` | robustness of a finding | universality beyond the tested family |

## 3. Why attention weights alone are insufficient

* A head can attend somewhere and write nothing useful (its OV output may be tiny, cancelled, or
  ignored downstream).
* Information can travel through heads with *diffuse* attention. In our ALiBi models the
  layer-0 heads that matter most do **not** have sharp previous-token patterns (best score 0.17,
  equal to its value at initialisation), yet ablating L0H0 drops accuracy from 0.91 to 0.03 on
  seed 0. Selecting components by pattern would have missed or mis-described them.
* Conversely a head with a beautiful induction pattern (L1H3 on seed 0, score 0.43) changes
  log-prob by only −0.27 when ablated — because other heads are redundant with it.
* Jain & Wallace (2019) show attention weights are frequently uncorrelated with gradient-based
  importance; Wiegreffe & Pinter (2019) argue the answer depends on what "explanation" means. This
  lab sidesteps the debate by **requiring interventions** for causal claims.

## 4. Ablation: what it can and cannot tell you

Replacing a component's output (at `hook_z` for a head — exactly that head's write) tests whether
the model *needs* it. Pitfalls:

* **Method matters.** Zero-ablation can push activations off-distribution and exaggerate effects;
  mean-ablation keeps a typical value; resample-ablation keeps a realistic value from another
  input. The ground-truth tests check all three agree on the hand-wired model; on trained models
  report the method.
* **Redundancy / self-repair.** If two heads compute the same thing, ablating one may do little.
  McGrath et al. (2023) report layers compensating for ablated attention layers; Wang et al.
  (2023) describe backup heads. Always test **groups** and use patching. Our seed-0 example:
  single-head ablations of L1H1/L1H3 are small, but patching each alone restores 46% / 31%.
* **One example proves nothing.** Every effect here is a paired difference over hundreds of
  examples with a bootstrap CI, compared with **random-head controls** (or, when few control
  sets exist, *all* of them — then a p-value cannot go below 1/(K+1), and the verdict requires the
  whole CI to lie beyond every control effect).

## 5. Activation patching

Run the corrupted input, but replace one activation with its value from the clean input; measure
how much of the clean behaviour returns:
`R = (M_patched − M_corrupt) / (M_clean − M_corrupt)`.

* **Design the corruption deliberately.** It should change exactly the information you are
  asking about and keep everything else (lengths, positions, scored targets) fixed. Our tasks
  provide *symmetric* corruptions where possible (e.g. `shuffle`: the first copy is permuted so an
  induction mechanism predicts a *specific wrong* token), enabling a logit-difference metric
  (recommended by Zhang & Nanda, 2024). A Gaussian-noise corruption would instead push activations
  off-distribution.
* **Check the gap.** If clean and corrupted runs behave similarly, R is a ratio of noise. The
  `Patcher` refuses to compute R below `min_gap` and labels the experiment *uninformative*.
* **Sufficiency ≠ necessity.** In an AND-like circuit (head A sets up where to look, head B
  copies), patching A alone into a corrupted run gives R ≈ 0 even though A is necessary. The
  hand-wired model demonstrates exactly this (`test_patching_recovers_known_components`), and our
  trained models show the same pattern for layer-0 heads.
* **Denoising vs noising** (Heimersheim & Nanda, 2024): patching clean→corrupted asks "is this
  enough?", corrupted→clean asks "is this needed?". `itl patch --direction noise` runs the latter.

## 6. Confounders and distribution shift

* **Positional shortcuts.** With a fixed repeat length, "copy the token L−1 positions back" solves
  the induction task without matching content. We vary the offset per example — and still found
  that learned absolute positions led the model into a smeared positional strategy instead of
  induction (findings §2). Any claim of "induction" must survive variable-offset data and the
  `shuffle` corruption (a content-matching mechanism follows the shuffled successors; a positional
  one does not).
* **Shared solutions across tasks.** Every task here is solvable by the same "copy the successor of
  the previous occurrence" rule (`test_all_tasks_are_solvable_by_the_induction_rule`). Do not
  assume different tasks use different mechanisms.
* **Selection bias (double dipping).** Choosing heads on the same examples you test them on
  inflates evidence. The hypothesis framework selects candidates on one seed of examples and tests
  on a disjoint one.
* **Hypotheses shaped by data.** A hypothesis revised after looking at a model must be tested on
  *other* models (seeds) — see `induction_2l_distributed_v2.yaml`.
* **Off-distribution interventions.** Mean-ablating a position-dependent activation with a
  per-position mean is only meaningful when positions are comparable across examples; with
  variable-offset sequences this is approximate.

## 7. From observation to supported explanation — a checklist

1. The model has learned the behaviour (held-out accuracy far above chance; leakage audit clean).
2. The hypothesis, thresholds and controls are written down before testing.
3. Candidates are chosen by a declared rule on separate examples.
4. Observational evidence (patterns, weights) agrees with …
5. … necessity (ablation, with controls and groups) and …
6. … sufficiency (patching with a well-designed corruption, informative gap) and …
7. … path-specific predictions (K vs Q vs V) that distinguish the hypothesis from alternatives.
8. Results replicate across seeds; failures are reported.
9. Claims are scoped: "on this task, these models, these interventions".

## 8. Path patching (session 2)

Path patching asks whether a *specific edge* carries the information: the model runs on the
destination input, and only along the listed sender → receiver edges does the receiver read the
sender's output from the source input (`interpretability/path_patching.py`). Receivers are a head's
q / k / v input, an MLP's input, or the direct path to the logits. Effects that flow through an
intermediate component are **not** included unless that edge is listed too — the MLP-relay ground
truth shows the direct L0H0 → key edge reading R = 1.0 (irrelevant) while the edge into MLP0 reads
R = 0.0 (essential). Sanity identity: a sender along **all** its outgoing edges equals ordinary
activation patching of that sender (tested to 1e-4).

Two lessons, both encoded as tests:

* **Metric floors.** With log-prob as the metric, a completely broken route can show "recovery"
  ≈ 0.33: a head that cannot find its target attends diffusely over a context that still contains
  the right token. Use accuracy or a logit difference against a specific alternative to decide
  whether a route is intact.
* **Recovery above 1 and below 0 are informative.** In the context-route experiment the query edge
  alone gives R ≈ 1.25 and the key edge R ≈ −0.15 with the union ≈ 1: routes interact; report them
  raw instead of clipping.

## 9. Behavioural tests that separate mechanisms

Accuracy cannot separate mechanisms that agree on the training distribution. Construct inputs on
which they disagree:

* **Shuffled first copy** (`scripts/mechanism_type_test.py`): content induction predicts the
  shuffled successor, positional copying the token at the fixed offset — the fixed-offset model
  scored 100% task accuracy with 0% content induction.
* **Context competition** (`context_competition`): two earlier occurrences of the query token,
  differing only in their preceding token, with recency counterbalanced. Single-token induction is
  indifferent (hand-wired: CP = 0.000); context matching prefers the matching one.

Always calibrate such a test on ground-truth models first. The session-1 "context sensitivity"
measurement looked reasonable but reads exactly 0 on a hand-wired model that *does* match context —
a measurement that cannot detect the effect it is meant to detect is not evidence either way.

## 10. Preregistration in practice

`configs/preregistration/*.yaml` hold the hypotheses, measurements, validity criteria and decision
rules; `artifacts/preregistration_log.jsonl` records their SHA-256 at freezing, pilot decisions,
deviations and analyses. The analysis code refuses to run if the spec on disk no longer matches its
frozen hash. When a preregistered test is inconclusive, a post-hoc follow-up is labelled
exploratory and confirmed only on models that did not exist when the follow-up was frozen
(`context_route_v1`).
