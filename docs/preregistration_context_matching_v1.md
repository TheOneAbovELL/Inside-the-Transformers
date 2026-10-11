# Preregistration: context matching in the induction circuit (v1)

**Status:** frozen 2026-10-11 before any context-preference value was computed on a trained model.
Machine-readable rules: `configs/preregistration/context_matching_v1.yaml` (its SHA-256 is in
`artifacts/preregistration_log.jsonl`). Results go in `docs/findings.md`, never in this file.

## 1. Why this experiment

The first research session reported that in the 2-layer attention-only ALiBi models
(`induction_2l_attn`, seeds 0–2) the induction group's **queries** depend on layer-0 output
(removing it costs 1.9–3.0 nats) and that replacing the token *two positions before the induction
target* lowers log-prob (−0.59 to −1.20 nats). It interpreted this as **context matching**: the
heads match the preceding token(s), not only the current token.

That evidence is ambiguous (audit, this session):

* the "context sensitivity" drop can also come from **non-specific interference** — layer-0
  output is a blurry recency-weighted summary, so changing a nearby token perturbs the key and an
  imperfect learned QK circuit may score it lower, without the query using the preceding token;
* on a hand-wired model that *does* match context, that measurement reads **exactly 0** whenever
  the current-token match dominates (`tests/scientific/test_context_measurements.py`). It is
  neither sufficient nor necessary evidence;
* query dependence on layer 0 could reflect the query reading the *current* token from layer 0's
  summary (L0H0 attends ~22% to its own position) rather than reading context.

## 2. Formal hypothesis (the mechanism under test)

In the trained 2-layer attention-only models:

* **Layers / components.** Layer 0: four heads whose outputs are recency-weighted summaries of
  recent tokens (no sharp previous-token head). Layer 1: an *induction group* (heads with induction
  score ≥ 0.25 on standard repeats).
* **Keys** of the group encode, at each position, the identity of the preceding token(s)
  (assembled from layer-0 outputs; K-composition).
* **Queries** encode the current token **and its predecessor** (Q-composition with layer 0).
* **Matching.** A key scores highest when its preceding tokens match the query's current token
  *and* the query's preceding token, i.e. the heads retrieve by a short context (≥ 2 tokens).
* **MLP.** Not applicable (attention-only); in the MLP models the key-side information is relayed
  through MLP0 (finding §9) and the same tests apply with MLP0 included among the senders.

## 3. Controlled manipulation and behavioural prediction

**Context competition** (`ContextCompetition`): the query `x`, preceded by `y`, occurred twice
earlier: as `y x A` and as `z x B`. All other tokens are distinct. The single-token rule cannot
choose; a context matcher chooses `A`. Which occurrence is more recent is counterbalanced
(50/50), so recency and distance biases cancel in the average and are reported separately.
Primary outcome: **CP = mean logit[A] − logit[B]**.

**Training-statistics manipulation.** Two new training conditions identical in every respect
(same tokens, lengths, scored targets, architecture, budget) except one:

* `same_order` — the second segment repeats the first (`x1 y1 x2 y2 …`), so each `x` is preceded by
  the same token both times (context is consistent, though never *needed*: tokens are unique);
* `shuffled` — the pairs are reordered so that **no** `x` is preceded by the same token at both
  occurrences (context is systematically misleading for a context matcher).

Shortcut audit of this design: (i) all segment tokens are distinct, so the answer is determined by
the single current token in both conditions — neither condition *requires* context; (ii) the
`shuffled` order is rejection-sampled per example, so the offset between the two occurrences varies
within an example — no fixed positional rule works (it already fails across examples because segment
and prefix lengths vary); (iii) parity of positions is identical across conditions; (iv) scored
targets and their counts are identical. Residual difference: in `shuffled`, offsets also vary
*within* a sequence, slightly penalising any residual positional strategy — this cannot produce a
context preference by itself.

## 4. Competing explanations and what each predicts

| explanation | CP (original models) | CP same vs shuffled | CP after removing layer 0 from queries |
|---|---|---|---|
| E1 learned context matching (the hypothesis) | > 0, both recency configs | same > shuffled ≈ 0 | drops |
| E2 single-token induction + imprecise ("fuzzy") keys | ≈ 0 | ≈ 0 both | n/a |
| E3 recency / distance preference only | ≈ 0 on average; recent and earlier configs have opposite signs | ≈ 0 both | n/a |
| E4 context preference as an architectural by-product (blurry query and key summaries matched by a token-identity QK) | > 0 | similar in both (shuffled ≫ 0) | drops |
| E5 context carried only by keys (query = current token) | ≈ 0 (a key's extra context cannot match anything in a context-free query) | ≈ 0 | n/a |

H1 separates {E1, E4} from {E2, E3, E5}; H2 separates E1 from E4; H3 tests whether the
preference is mediated by the query path.

## 5. Decision rules, validity and contingencies

Exactly as in the YAML (copied here for readability):

* **Metric validity** — hand-wired single-token model |CP| < 0.05 and hand-wired context model CP
  lower CI > 1.0, otherwise the experiment is invalid. (Pre-freeze calibration: 0.000 and +1.986.)
* **Model validity** — own-task validation accuracy ≥ 0.80 and competition either-rate ≥ 0.50.
* **Training contingency** — pilot = seed 0 of both new conditions at 3000 steps; if either is
  below 0.80 validation accuracy, all new runs use 6000 steps and the pilots are excluded.
* **H1** supported if ≥ 2 valid original seeds have CP lower CI > 0.5 and none has upper CI < 0;
  contradicted if ≥ 2 have CI within [−0.5, 0.5]; else inconclusive.
* **H2** supported if the mean paired difference CP_same − CP_shuffled > 0.5 and ≥ 2 pairs have a
  difference CI entirely above 0; contradicted if ≥ 2 pairs have the difference CI within
  [−0.5, 0.5]; else inconclusive. Requires ≥ 2 valid models per condition.
* **H3** per qualifying model (CP lower CI > 0.5): supported if CP after query removal ≤ 50% of
  baseline and its upper CI < baseline lower CI; contradicted if its lower CI > 75% of baseline
  mean; overall by a 2/3 majority of evaluated models.

**Sample sizes.** 1024 competition examples per model (CI half-width for CP is expected to be
≲ 0.2 logits given per-example SDs of ~1–3 observed on similar logit differences); 3 seeds per
condition (budget: ~8–10 min per run on this CPU under background load; 6 new runs).

**Inconclusive by design if:** fewer than 2 valid models in a condition (H2), the competition
format is not treated as induction (either-rate < 0.5), or query removal destroys induction
altogether (H3 validity).

## 6. What would change the research interpretation

* H1 contradicted → the first session's "context matching" interpretation is withdrawn; the query
  dependence on layer 0 needs another explanation (e.g. current-token readout from layer 0).
* H1 supported, H2 contradicted → preference is not shaped by these training statistics (E4).
* H1 and H2 supported → context matching is learned when context is consistent (E1).
