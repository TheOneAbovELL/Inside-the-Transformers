# Status of claims and capabilities (end of session 3, 2026-10-12)

Legend: **Implemented** (code exists) · **Tested** (automated test passes) · **Demonstrated**
(experiment run, result recorded) · **Partial** · **Inconclusive** · **Not tested** · **Not done**.

## Capabilities

| capability | status | evidence |
|---|---|---|
| Transparent transformer (pre/post/no-LN; learned / sinusoidal / ALiBi / none positions; hooks) | Implemented · Tested | `tests/unit/test_model.py`, `test_hooks.py` (SDPA / LayerNorm equivalence, causal mask, overfit) |
| Synthetic tasks (5 incl. evaluation-only `context_competition`; `pair_mode`) | Implemented · Tested | `tests/unit/test_tasks.py`, `docs/task_audit.md`, regression fixtures |
| Resumable training, exact resume, leakage audit | Implemented · Tested | `tests/integration/test_training.py`, cross-process bit-identical training test |
| Reproduction audit (`itl verify`), analysis re-run diff (`itl rerun`) | Implemented · Tested · Demonstrated | 26/29 runs reproduce; key analyses identical |
| Zero / mean / resample ablation (heads, MLP blocks, neurons) | Implemented · Tested on ground truth | `docs/method_validation.md` |
| Activation patching (denoise / noise, CIs, uninformative-gap guard, per-example diagnostics) | Implemented · Tested on ground truth | same |
| Q/K/V path interventions (mean/zero; heads and MLPs as senders) | Implemented · Tested on ground truth (incl. MLP relay) | same |
| Path patching between inputs along named edges (head q/k/v, MLP input, direct path) | Implemented · Tested (completeness identity; relay and context ground truth) | `tests/scientific/test_path_patching.py` |
| Logit attribution (DLA with frozen LN scale), logit lens | Implemented · Tested (exact reconstruction) | `test_dla_attributes_logit_to_induction_head` |
| Hypothesis framework (selection/test split, controls, `requires`, Bonferroni sensitivity, overlap check) | Implemented · Tested (positive and negative controls) | `tests/scientific/test_hypothesis_framework.py` |
| Preregistration workflow (frozen hashes, refusal on edits, logged decisions/deviations) | Implemented · Tested · Demonstrated | `tests/unit/test_prereg_rules.py`, `artifacts/preregistration_log.jsonl` |
| CLI (`itl`) | Implemented · Tested | `tests/integration/test_experiments.py` |
| Rerunnable investigations (`itl investigate`) | Implemented · Tested (bit-identical rerun) | `tests/integration/test_reproducibility.py` |
| Local UI (Streamlit) | Implemented · Tested headlessly on every page | `tests/integration/test_ui.py` (16 tests) |
| Sparse autoencoders | **Not done** (deliberate; `docs/research_log.md` D7) | — |
| Automatic circuit discovery | **Not done** | — |

## Scientific claims (2-layer models on these synthetic tasks only)

| claim | status |
|---|---|
| Learned absolute positions → positional shortcut, no induction circuit (3000–4500 steps) | Demonstrated (negative result; one seed) |
| ALiBi enables induction (stated prediction) | Demonstrated (5 seeds ≥ 0.90) |
| 1-layer model cannot solve variable-offset induction | Demonstrated (one seed, 5.4%) |
| Distributed induction group whose keys come from layer 0 | Supported (hypothesis v2, 2 held-out seeds) |
| Textbook two-head circuit (sharp previous-token head, K-only) | Contradicted (3 seeds) |
| MLPs re-route key-side information through MLP0 | Supported (v3, 1 held-out seed) |
| Fixed-offset model reaches 100% with no content induction | Demonstrated (behavioural classification) |
| Context-dependent retrieval in the original models | **Supported (preregistered H1, 3/3 seeds)** |
| That preference is learned from training statistics | **Supported (preregistered H2)**; its *reversal* under shuffled training is unexplained |
| Context reaches the induction heads through their queries | Mean-removal test **Inconclusive** (H3); path-patching test **Supported** (H3b, 2 fresh seeds) |
| The steepest ALiBi head L0H0 is the dominant carrier of that query context | **Supported (preregistered H4, 2 fresh seeds)** |
| Preference grows with matching-context length (1→3 tokens) and is implemented in attention | Exploratory (11 models, ground-truth calibrated) |
| MLP models: query context mainly direct from L0H0, keys via MLP0 | Exploratory (2 models) |
| Formation: plateau then joint rise (steps 150–300) | Demonstrated (3 seeds; correlational) |
| Assoc-recall redundancy | Exploratory (1 model) |
| Single-pair induction training | Negative result (anti-generalising priors) |
| Why shuffled training reverses the preference (mechanism) | Located in attention; cause not tested |
| Any of this in large language models | Not tested |
