# Research log

A concise record of decisions, experiments and open tasks so work can resume without repeating
completed investigations. Newest session last. Detailed results: `docs/findings.md`.
Preregistrations and their freeze / analysis events: `artifacts/preregistration_log.jsonl`.

## Session 1 (2026-10-11, build)

Built the laboratory (model, tasks, training, interventions, hypothesis framework, CLI, UI, docs).
Key events: learned-position model stuck on a positional shortcut → ALiBi fixed it (prediction
supported); sinusoidal-scale bug; hypotheses v1 (partly contradicted), v2 (distributed circuit,
supported on held-out seeds), v3 (MLP mediation, supported on a held-out seed); "context matching"
proposed from an ambiguous measurement.

## Session 2 (2026-10-11, audit + preregistered investigation)

### Baseline
* 97/97 tests passed, 0 skipped; `itl smoke` 7/7. Repository: no commits (nothing pushed or uploaded).
* Machine: ~86% CPU load from unrelated processes (other projects' Python workers, OneDrive sync);
  ~3.7 GB RAM free. Not touched.

### Decisions
* **D1 — Reproduction audit before new research.** Added `itl verify` (checkpoint → regenerated
  eval sets → recorded metrics) and `itl rerun` diffing. 18/21 runs reproduce exactly; 2 pre-fix
  sinusoidal runs cannot (documented, not deleted).
* **D2 — The session-1 context-matching evidence is ambiguous** (interference vs matching; the
  measure is blind on ground truth). Designed `context_competition` (two candidate occurrences
  differing only in preceding context, recency counterbalanced) and validated it on a hand-wired
  single-token (CP 0.000) and context-matching (CP +1.985) model *before* preregistering.
* **D3 — Training-statistics manipulation.** `pair_mode` same_order vs shuffled: identical tokens,
  lengths, scored targets and architecture; only whether the preceding context repeats. Default task
  stream unchanged (regression fixtures).
* **D4 — Preregistered and frozen** `context_matching_v1` (hashes logged) before any CP measurement on
  trained models; pilot gate logged before analysis.
* **D5 — H3 inconclusive → exploratory path patching → separately preregistered `context_route_v1`**
  and confirmed on two seeds trained after freezing (log shows none existed at freeze time).
* **D6 — General path patching implemented** (named edges incl. MLP inputs and the direct path;
  clean/corrupt semantics; completeness identity tested) because mean-removal could not answer H3.
* **D7 — Sparse autoencoders not implemented.** Question they would answer: what features layer 0
  writes. Here: d_model 64 with near one-hot token inputs and four layer-0 heads — the relevant
  quantities (which token at which offset) are already directly measurable, so an SAE would add a
  learned basis requiring its own validation (reconstruction, sparsity, stability across seeds) and
  causal tests, at real methodological cost, without answering an open question. Revisit if models
  grow (d_model ≥ 256, MLP-heavy) or natural-language data is used.
* **D8 — Hypothesis framework**: added set-overlap reporting and Bonferroni-sensitivity verdicts
  rather than changing verdict rules retroactively.
* **D9 — UI**: patching made button-triggered (it previously launched a full scan on page load);
  added "Hypotheses & evidence"; headless `AppTest` tests for every page.

### Experiments run (session 2)
| id | purpose | outcome |
|---|---|---|
| ctx_pairs_{same_order,shuffled} seeds 0–2 | context_matching_v1 models | all valid (0.889–0.922) |
| 20261011-225402_prereg_context_matching_v1 | H1/H2/H3 | supported / supported / inconclusive |
| 20261011-225623_investigation_context_route_exploratory | route of the context signal (post hoc) | query route in 6/6 models |
| induction_2l_attn seeds 3–4 | context_route_v1 confirmation models | 0.935, 0.953 |
| 20261011-231249_prereg_context_route_v1 | H3b | supported |
| re-runs 222223, 222313, 231409 | reproducibility of v2, v3, context_matching_v1 analyses | identical |

### Open questions / next experiments (ranked)
1. **Which layer-0 heads carry the predecessor identity into the queries?** Path-patch individual
   layer-0 heads → group queries on the competition task (cheap; existing models).
2. **Why does shuffled training produce a negative preference?** Test the interference
   interpretation: measure attention of the group to "context-sharing but token-mismatched" keys in
   shuffled vs same_order models.
3. **Does the context preference extend beyond 2 tokens, and how does it scale with match length?**
   `context_len` 3+ needs a generator extension.
4. **MLP models**: repeat context_matching_v1 on `induction_2l_mlp` (is context routed through MLP0
   too?).
5. Learned-position model trained much longer (does induction eventually form?).

## Session 3 (2026-10-11/12, refinement + book)

### Decisions
* **D10 — No new training.** CPU at 100% and ~1.2 GB free RAM from other projects' workers; every
  open question that existing checkpoints can answer was answered with them. Deferred: the
  learned-position long run.
* **D11 — Per-head question preregistered with an exploration / confirmation split by seed**
  (seeds 0–2 explore, 3–4 confirm) — possible without training because seeds 3–4 had only been
  analysed at the layer-0-group level. YAML validated before freezing (lesson from context_route_v1).
* **D12 — Investigations became library code** (`itl investigate`) so their records are rerunnable;
  the session-1/2 scripts are kept unchanged (one is referenced by a frozen preregistration).
* **D13 — Path-patching edge semantics:** when an MLP is a patched sender, edges into it are dropped
  (identity-tested); documented in `interpretability/context.py::all_layer0_edges_patch`.

### Experiments (all on existing checkpoints)
| id | purpose | outcome |
|---|---|---|
| 20261011-235726_investigation_context_heads | per-head route, seeds 0–2 (exploratory) | L0H0 dominant |
| 20261011-235940_prereg_context_heads_v1 | H4 on seeds 3–4 | supported |
| 20261012-000102_investigation_competition_attention | attention to A vs B | preference lives in the pattern |
| 20261012-000131 / 000424 _investigation_context_dose | CP vs 1–3 matching tokens | rises with context length |
| 20261012-000547_investigation_context_route, 000634_..._context_heads | MLP models | query route; mostly direct from L0H0 |
| 20261012-000807_investigation_mechanism_type | library re-implementation check | matches the session-1 script |

### Open questions / next experiments (ranked, updated)
1. **Why does shuffled training reverse the preference?** Now known to live in the attention
   pattern; next: compare QK-circuit weights (query-predecessor ↔ key-2-back terms) between
   conditions.
2. **Confirm the dose-response and MLP findings on fresh seeds** (currently exploratory) — requires
   training (cheap once the machine is idle).
3. **Does L0H0's query role and the MLP0 key route coexist by design?** Path-patch L0H0 → MLP0 →
   keys vs L0H0 → queries in the MLP models with clean/corrupt inputs that change only the predecessor.
4. Learned-position model trained much longer.
