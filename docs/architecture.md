# Architecture guide

## 1. The system at a glance

```
configs/experiments/*.yaml ──► training/trainer.py ──► artifacts/runs/<run>/      (checkpoints, metrics, summary)
configs/hypotheses/*.yaml  ──┐                                │
                             ▼                                ▼
                  experiments/runner.py  ◄── interpretability/* ◄── instrumentation/* ◄── models/*
                             │
                             ▼
                artifacts/experiments/<id>/  (config, results.json, CSV, plots, report.md) ──► registry.jsonl
                             ▲
          cli/main.py (`itl ...`)        app/streamlit_app.py (`itl ui`)
```

The scientific core (`models`, `data`, `training`, `evaluation`, `instrumentation`,
`interpretability`) has **no dependency on the UI**. The CLI and the Streamlit app are thin layers
over `experiments/runner.py`.

## 2. Data flow: tokens → logits

For the principal model (`configs/experiments/induction_2l_attn.yaml`: 2 layers, 4 heads,
`d_model=64`, `d_head=16`, attention-only, pre-LN, ALiBi, vocabulary 50, context 40):

| stage | code | tensor | shape |
|---|---|---|---|
| token ids | input | `tokens` | `[b, 40]` int64 |
| token embedding | `W_E[tokens]` → `hook_embed` | | `[b, 40, 64]` |
| positional embedding | `hook_pos_embed` (zeros under ALiBi) | | `[b, 40, 64]` |
| residual stream 0 | sum | `blocks.0.hook_resid_pre` | `[b, 40, 64]` |
| LN₁ | `blocks.0.ln1.hook_normalized` | | `[b, 40, 64]` |
| Q, K, V per head | `blocks.0.attn.hook_q/k/v` | | `[b, 40, 4, 16]` |
| scores `QKᵀ/√16 + ALiBi + mask` | `blocks.0.attn.hook_attn_scores` | | `[b, 4, 40, 40]` |
| pattern (softmax) | `blocks.0.attn.hook_pattern` | | `[b, 4, 40, 40]` |
| weighted values | `blocks.0.attn.hook_z` | | `[b, 40, 4, 16]` |
| attention output `Σ z_i W_O^i + b_O` | `blocks.0.hook_attn_out` | | `[b, 40, 64]` |
| residual after attention | `blocks.0.hook_resid_mid` (= `resid_post` when attention-only) | | `[b, 40, 64]` |
| … layer 1 identically … | `blocks.1.*` | | |
| final LN | `ln_final.hook_normalized` | | `[b, 40, 64]` |
| unembedding `· W_U + b_U` | return value | logits | `[b, 40, 50]` |

With MLPs (`attn_only: false`) each block also has `hook_mlp_in` (the stream as read by the MLP
only, before `ln2`; editing it changes the MLP's input without changing the stream),
`mlp.hook_pre`/`hook_post` `[b,s,d_mlp]` and `hook_mlp_out` `[b,s,d]`. `model.hook_names()` lists every hook point.

## 3. Modules and responsibilities

| module | responsibility | key design decisions |
|---|---|---|
| `config.py` | Typed configs (`ModelConfig`, `TaskConfig`, `TrainConfig`, `ExperimentConfig`), YAML I/O, dotted overrides | Unknown keys are **errors**; `vocab_size`/`n_ctx` derive from the task |
| `models/hooks.py` | `HookPoint` (identity module), `HookedRootModule.hooks()` context manager, `run_with_hooks`, `run_with_cache` | Hooks are removed in `finally`, even on exceptions (tested); caching is opt-in and filterable |
| `models/transformer.py` | The transformer, written out explicitly | Per-head weight tensors; explicit softmax attention (needed to hook the pattern); pre/post/no-LN selectable; ALiBi/learned/sinusoidal/none positions; private-generator init |
| `models/handwired.py` | Hand-built induction transformers: single-token, context-matching (`context_gain`), MLP-relay (`mlp_relay`) | Ground truth for validating every tool (`docs/method_validation.md`) |
| `data/tasks.py` | Four task families + `TaskBatch` | Every task has an exact solver, chance baseline, seed-controlled generator; corruptions keep targets fixed |
| `training/trainer.py` | Training loop, evaluation during training, checkpoints, resume, early stopping, leakage audit | Validation (decisions) and test (reporting) sets are separate; RNG state saved for exact resume |
| `training/checkpoint.py` | Atomic save, `weights_only` load | Never leaves half-written files; no code execution on load |
| `evaluation/` | Per-example metrics, bootstrap summaries, baselines, sweeps | Metrics are per example so uncertainty can be resampled |
| `instrumentation/` | `ActivationCache` (short names, head results, residual decomposition) and inspection utilities | Derived quantities are computed from cached tensors + weights, never stored redundantly |
| `interpretability/attention_scores.py` | Prev-token / induction / duplicate / BOS scores; OV copying; composition scores | Labelled observational |
| `interpretability/interventions.py` | Head and MLP ablation hooks (zero / mean / resample) | Head boundary = `hook_z` (pre-`W_O`), so one head is removed exactly |
| `interpretability/patching.py` | Clean→corrupted activation patching, recovery with bootstrap CIs, region masks | Uninformative experiments return NaN, not a number |
| `interpretability/path.py` | Sender→receiver Q/K/V path interventions with mean/zero replacement (heads or MLPs as senders) | Uses per-head split inputs (`split_qkv_input`); numerically equivalent to the normal forward when no edit is made (tested) |
| `interpretability/path_patching.py` | Clean↔corrupt **path patching** along named edges: sender (head / MLP) → receiver (head q/k/v input, MLP input via `hook_mlp_in`, or the direct path to the logits) | Completeness identity tested: a sender along all its edges == activation patching of it |
| `interpretability/context.py` | Context preference (CP), context sensitivity, CP under path removal, context-route path patching | Calibrated on hand-wired single-token / context models |
| `interpretability/attribution.py` | Direct logit attribution, logit lens | Exactness check reported with every DLA |
| `interpretability/scans.py` | Ablate-everything, patch-everything, random-head controls | Controls enumerate exhaustively when few sets exist |
| `interpretability/hypothesis.py` | Pre-registered hypothesis testing with selection/test split and verdicts | Verdicts are supported / contradicted / inconclusive; errors become visible "inconclusive" entries |
| `experiments/registry.py` | Experiment directories, registry index, run listing, comparison, `compare_results` (value-by-value diff of two records) | Every experiment records the arguments needed to re-run it |
| `experiments/context_experiment.py` | Analyses of the preregistered experiments; frozen decision rules | Refuses to run if the spec's SHA-256 differs from the freeze log |
| `training/verify.py` | Reproduction audit of a run (config → model/task/seeds; checkpoint → regenerated eval sets → recorded metrics; code fingerprint) | `itl verify` |
| `experiments/runner.py` | One function per analysis (eval, inspect, trace, scores, ablation, patching, hypothesis, formation, sweep), `smoke`, `rerun` | Reports separate observations / statistics / interpretation / limitations |
| `visualization/plots.py` | Matplotlib figures | Every figure carries run/checkpoint/n/condition text; attention plots carry the causal caveat |
| `reporting/report.py` | Markdown report assembly | Standard limitations appended to every report |
| `cli/main.py` | `itl` command (train, eval, inspect, trace, scores, ablate, patch, hypothesis, formation, sweep, prereg, verify, tasks, configs, runs, experiments, compare, rerun, ui) | Expected user errors print one line and exit 2 (`ITL_DEBUG=1` for tracebacks) |
| `app/streamlit_app.py` | Local UI | Calls the same backend; binds to localhost; usage stats disabled |

## 4. Why these design decisions

* **Explicit attention instead of a fused kernel.** Fused kernels never materialise the pattern,
  which is exactly what we need to read and intervene on. The cost on CPU is irrelevant at this
  scale; equivalence with `F.scaled_dot_product_attention` is tested.
* **Per-head parameters `[h, d, e]`.** Each head's QK/OV circuit is a slice, not a reshape of a
  fused matrix — fewer opportunities for indexing mistakes in analysis code.
* **Hook points rather than editing the model per experiment.** All interventions are functions
  attached for one forward pass and guaranteed to be removed.
* **Head intervention boundary at `hook_z`.** Because `attn_out = Σ_i z_i W_O^i + b_O`,
  replacing `z_i` removes/replaces exactly head `i`'s write. An alternative boundary (the pattern)
  would change *where* the head looks but still let it write.
* **Separate scored and trained positions.** `loss_mask` marks predictable positions (metrics);
  `train_mask` (optional) marks trained positions.
* **Seeds via `SeedSequence`.** One experiment seed derives independent seeds for model init,
  training stream, validation and test sets.

## 5. Generated artifacts

```
artifacts/
  preregistration_log.jsonl          append-only: freeze (hashes), pilot decisions, deviations, analyses
  runs/<name>_s<seed>_<timestamp>/   config.yaml, resolved_model.json, env.json, seeds.json, task_spec.json,
                                     metrics.jsonl, summary.json, checkpoints/{step_*.pt, latest.pt, final.pt}
  experiments/<timestamp>_<kind>_<name>/  config.json, env.json, results.json, *.csv, plots/*.png, report.md
  experiments/registry.jsonl
```

`artifacts/` is git-ignored. Override its location with `ITL_ARTIFACT_ROOT`.
