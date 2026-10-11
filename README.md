# Inside the Transformer — a mechanistic interpretability laboratory

A local, CPU-friendly research system for training small Transformers on controlled algorithmic
tasks, inspecting their internals, intervening on them, and **testing pre-registered circuit
hypotheses** with controls, held-out examples and multiple seeds.

Hypothesis → dataset → training → baseline → observation → intervention → measurement →
replication → conclusion. Every step is a command that writes a reproducible artifact.

> Everything runs locally. No telemetry, no remote logging, no network access needed after
> installation. Generated artifacts live in `./artifacts` (git-ignored); set `ITL_ARTIFACT_ROOT`
> to move them (note: this folder is inside a OneDrive-synced Desktop).

## What is in here

| capability | where |
|---|---|
| Transparent decoder-only Transformer (per-head weights, explicit attention, pre/post/no-LN, learned/sinusoidal/ALiBi/no positions, hook points everywhere) | `src/transformer_lab/models/` |
| Hand-wired induction model (known ground truth for validating tools) | `models/handwired.py` |
| Synthetic tasks with exact solvers, held-out pairs, corruptions: repeated pattern, associative recall, variable-offset induction, single-pair induction with distractors/decoys | `src/transformer_lab/data/tasks.py` |
| Resumable, config-driven training with leakage audit | `src/transformer_lab/training/` |
| Activation caching, inspection, prediction tracing (DLA, logit lens) | `src/transformer_lab/instrumentation/`, `interpretability/attribution.py` |
| Ablation (zero/mean/resample; heads, MLP blocks, neurons) | `interpretability/interventions.py`, `scans.py` |
| Activation patching (denoise/noise, recovery with CIs, uninformative-gap detection) | `interpretability/patching.py` |
| Path interventions (head/MLP → head via Q, K or V) | `interpretability/path.py` |
| Pre-registered hypothesis testing (selection/test split, controls, verdicts) | `interpretability/hypothesis.py`, `configs/hypotheses/` |
| Experiment registry, reports, comparison, re-run | `experiments/`, `reporting/` |
| CLI (`itl`) and local Streamlit UI (`itl ui`) | `cli/main.py`, `app/streamlit_app.py` |

## Install (Windows, CPU)

Tested with Python 3.13.3 and torch 2.10.0+cpu on Windows 11.

```bash
python -m venv .venv --system-site-packages
```

```bash
.venv/Scripts/python -m pip install -e . --no-build-isolation
```

(`--system-site-packages` reuses an existing PyTorch install; drop it to install fresh.
Optional extras: `pip install -e .[ui,diag,dev]` for Streamlit/pandas, psutil and pytest.)

## Quick start

Validate the installation (trains a tiny model and exercises every component, ~30 s):

```bash
.venv/Scripts/itl smoke
```

Run the test suite (unit, integration, scientific, regression):

```bash
.venv/Scripts/python -m pytest
```

Train the principal research model (2-layer attention-only, ALiBi; a few minutes on CPU):

```bash
.venv/Scripts/itl train --config configs/experiments/induction_2l_attn.yaml
```

Train three seeds plus the architectural controls, two at a time:

```bash
.venv/Scripts/python scripts/train_queue.py --parallel 2 configs/experiments/induction_2l_attn.yaml:0,1,2 configs/experiments/induction_1l_attn.yaml:0 configs/experiments/induction_2l_mlp.yaml:0
```

Analyse a run (`<RUN>` = a directory printed by `itl runs`):

```bash
.venv/Scripts/itl eval --run <RUN> --all-checkpoints
```

```bash
.venv/Scripts/itl scores --run <RUN>
```

```bash
.venv/Scripts/itl ablate --run <RUN> --method mean
```

```bash
.venv/Scripts/itl patch --run <RUN> --corruption shuffle --metric logit_diff
```

```bash
.venv/Scripts/itl formation --run <RUN> --ablate
```

```bash
.venv/Scripts/itl trace --run <RUN> --example 0
```

Test a pre-registered circuit hypothesis on several seeds:

```bash
.venv/Scripts/itl hypothesis --spec configs/hypotheses/induction_2l.yaml --run <RUN_SEED0> --run <RUN_SEED1> --run <RUN_SEED2>
```

Compare, list and re-run:

```bash
.venv/Scripts/itl runs
```

```bash
.venv/Scripts/itl experiments
```

```bash
.venv/Scripts/itl rerun <EXPERIMENT_ID>
```

(`rerun` re-executes an analysis from its saved record and diffs every value against the original.)

Discover what exists, and audit reproducibility:

```bash
.venv/Scripts/itl tasks
```

```bash
.venv/Scripts/itl configs
```

```bash
.venv/Scripts/itl verify --all
```

Preregistered experiments (the analysis refuses to run if the spec changed after freezing):

```bash
.venv/Scripts/python scripts/train_queue.py --parallel 2 configs/experiments/ctx_pairs_same_order.yaml:0,1,2 configs/experiments/ctx_pairs_shuffled.yaml:0,1,2
```

```bash
.venv/Scripts/itl prereg --spec configs/preregistration/context_matching_v1.yaml
```

```bash
.venv/Scripts/itl prereg --spec configs/preregistration/context_route_v1.yaml
```

```bash
.venv/Scripts/itl prereg --spec configs/preregistration/context_heads_v1.yaml
```

Exploratory, rerunnable investigations (each includes the hand-wired reference models):

```bash
.venv/Scripts/itl investigate context-dose --run <RUN>
```

Launch the local UI (binds to localhost, usage statistics disabled):

```bash
.venv/Scripts/itl ui
```

## Principal findings (details and evidence: `docs/findings.md`)

1. **Positional scheme decided whether an induction circuit formed.** With learned absolute
   positions a 2-layer attention-only model plateaued at ~35% (positional shortcut, no
   previous-token head); with ALiBi, three seeds reached 91–95%. A 1-layer model reached 5% (chance 2%).
2. **The learned circuit is distributed and K-composition-dominated.** A group of layer-1 heads
   (patching restores 98%) whose keys are built from layer-0 outputs (removing them from the keys
   only: −8 to −10 nats). Replicated on seeds not used to form the hypothesis.
3. **The pre-registered textbook hypothesis was partly wrong.** No sharp previous-token head exists,
   and queries also depend on layer 0.
4. **Retrieval is context-dependent, and that is learned (preregistered).** Given two earlier
   occurrences of the query token, the models prefer the one whose preceding token matches
   (+1.9 to +2.3 logits, 3/3 seeds). Models trained where preceding context repeats show the
   preference; models trained where it never repeats show a *reversed* one. Path patching locates
   the context signal on the induction heads' **queries** (confirmed on two seeds trained after the
   preregistration was frozen), and specifically through the steepest-slope layer-0 head L0H0
   (preregistered, confirmed on two further seeds). Exploratory: the preference grows with each
   extra matching context token (1 → 3).
5. **Adding MLPs re-routes the circuit**: layer-0 information reaches the keys mostly through MLP0
   (confirmed on a seed trained after the hypothesis was written).
6. **Accuracy does not identify mechanism**: a fixed-offset model scores 100% while copying purely by
   position (0% content induction on shuffled inputs).
7. **Negative results are kept**: single-pair induction training learned anti-generalising priors
   (0/512 on held-out pairs); the leakage audit exposed memorisable eval sequences in `repeat_pattern`.

## Documentation

* `docs/architecture.md` — data flow, shapes, module responsibilities, design decisions
* `docs/mathematics.md` — derivations mapped to code
* `docs/interpretability_guide.md` — what each kind of evidence can and cannot show
* `docs/experiment_protocols.md` — handbook: question, setup, metrics, controls, commands
* `docs/findings.md` — results of every experiment run, with artifact paths (summary table first)
* `docs/research_log.md` — decisions, experiments and ranked next steps (start here to continue)
* `docs/status.md` — every capability and claim, with its evidence status
* `docs/preregistration_context_matching_v1.md` — the preregistered context-matching design
* `docs/method_validation.md` — every analysis method checked against hand-wired ground truth
* `docs/task_audit.md` — per-task: what it tests, shortcuts, baselines, what it cannot establish
* `docs/limitations.md` — read before quoting anything
* `docs/interview_guide.md` — hard questions with answers and code pointers
* `docs/references.md` — verified bibliography
* `docs/environment_audit.md` — the machine this was built and measured on

## Project layout

```
configs/experiments/   training configs (model + task + optimisation + seed)
configs/hypotheses/    pre-registered circuit hypotheses
src/transformer_lab/   the library (models, data, training, evaluation, instrumentation,
                       interpretability, experiments, visualization, reporting, cli)
app/                   Streamlit UI
scripts/               training queue, follow-up investigations
tests/                 unit/ integration/ scientific/ regression/
docs/                  documentation
artifacts/             generated (git-ignored): runs/, experiments/
```
