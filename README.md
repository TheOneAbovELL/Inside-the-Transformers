# Inside the Transformer — A Mechanistic Interpretability Laboratory

<div align="center">

### Train Small Transformers. Inspect Their Circuits. Test What They Learn.

A local, CPU-friendly research laboratory for training small Transformers on controlled algorithmic tasks, inspecting their internal computations, intervening on their activations, and **testing preregistered circuit hypotheses using controls, held-out examples, and multiple random seeds**.

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU--Friendly-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Focus-Mechanistic%20Interpretability-7057FF?style=flat-square)](#research-methodology)
[![Execution](https://img.shields.io/badge/Execution-Local--First-2EA44F?style=flat-square)](#local-first-design)
[![Interface](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](#interfaces)
[![Reproducibility](https://img.shields.io/badge/Experiments-Reproducible-2EA44F?style=flat-square)](#reproducibility-and-experiment-management)

**Hypothesis → Dataset → Training → Baseline → Observation → Intervention → Measurement → Replication → Conclusion**

</div>

---

## Overview

**Inside the Transformer** is a local research system for studying how small Transformer models implement learned algorithmic behavior.

Rather than treating a model as a black box, the laboratory provides tools for examining its internal representations, tracing prediction computations, ablating components, patching activations, measuring causal effects, and evaluating explicit hypotheses about the circuits responsible for a behavior.

The project combines transparent Transformer implementations, controlled synthetic datasets, reproducible training runs, interpretability interventions, preregistered hypothesis testing, and experiment-level reporting into a single workflow.

The central objective is to move beyond observing that a model performs a task and investigate **how the model performs it**.

A high accuracy score alone does not establish that a model has learned the intended algorithm. A model may exploit positional shortcuts, memorize evaluation sequences, or rely on a mechanism different from the hypothesized circuit. This laboratory is designed to help detect such cases through held-out examples, controls, interventions, and replication across seeds.

### Core research question

> When a small Transformer learns an algorithmic task, what internal computational mechanism does it use, which components are causally important, and how reliably does the evidence support a specific circuit hypothesis?

### Local-first design

Everything is designed to run locally. No telemetry, remote logging, or network access is required after installation.

Generated artifacts are stored under `./artifacts` by default and are git-ignored. The artifact directory can be relocated using the `ITL_ARTIFACT_ROOT` environment variable.

**Important storage note:** the current Desktop location is inside a OneDrive-synced directory. Setting `ITL_ARTIFACT_ROOT` to another local directory can move generated artifacts outside that synchronized location.

---

## Research Methodology

The laboratory follows a structured experimental process:

1. **Formulate a hypothesis.** State a specific claim about a model's internal mechanism.
2. **Construct a controlled dataset.** Use algorithmic tasks with known solutions, held-out examples, and relevant corruptions.
3. **Train the model.** Use configuration-driven, resumable training with recorded seeds and settings.
4. **Establish a baseline.** Measure task performance before drawing conclusions about mechanism.
5. **Inspect internal behavior.** Cache activations, trace predictions, and examine candidate components.
6. **Intervene on the model.** Use ablation, activation patching, and path interventions to test causal claims.
7. **Measure the effect.** Quantify changes using task metrics, logit differences, recovery measurements, and appropriate controls.
8. **Replicate the result.** Evaluate across seeds and held-out examples, including runs not used to formulate the hypothesis.
9. **Record the conclusion.** Preserve the experiment configuration, artifacts, measurements, and verdict.

The goal is not merely to generate interpretability visualizations. It is to evaluate hypotheses against evidence, including evidence that contradicts the original hypothesis.

Each stage is represented by executable commands and reproducible artifacts.

## What Is Implemented

| Capability | Implementation and location |
|---|---|
| Transparent decoder-only Transformer | `src/transformer_lab/models/` |
| Hand-wired induction model with known ground truth | `models/handwired.py` |
| Synthetic algorithmic tasks, exact solvers, held-out pairs, and corruptions | `src/transformer_lab/data/tasks.py` |
| Resumable, configuration-driven training with leakage audit | `src/transformer_lab/training/` |
| Activation caching, inspection, and prediction tracing | `src/transformer_lab/instrumentation/` |
| Direct logit attribution (DLA) and logit lens | `interpretability/attribution.py` |
| Component and neuron ablation | `interpretability/interventions.py`, `scans.py` |
| Activation patching with denoising/noising, recovery confidence intervals, and uninformative-gap detection | `interpretability/patching.py` |
| Path interventions between heads and MLPs through Q, K, or V | `interpretability/path.py` |
| Preregistered hypothesis testing with selection/test splits, controls, and verdicts | `interpretability/hypothesis.py`, `configs/hypotheses/` |
| Experiment registry, reports, comparison, and reruns | `experiments/`, `reporting/` |
| Command-line interface | `itl` |
| Local Streamlit research interface | `itl ui`, `app/streamlit_app.py` |

### Transparent Transformer architecture

The Transformer implementation is designed to expose the internal computations needed for interpretability experiments.

Supported architectural choices include:

- Explicit per-head attention weights.
- Explicit attention computations.
- Pre-layer normalization, post-layer normalization, and no-layer-normalization configurations.
- Learned positional embeddings.
- Sinusoidal positional embeddings.
- ALiBi positional biases.
- No positional information.
- Hook points throughout the model for instrumentation and interventions.

These options make it possible to compare models that differ in architecture while keeping the experimental task and evaluation procedure controlled.

### Hand-wired reference models

The project includes a hand-wired induction model with known ground-truth behavior.

This is particularly important for validating interpretability tools. Before interpreting a learned model, an analysis method can be checked against a model whose relevant computational structure is already understood.

The purpose is to establish whether the analysis method can recover or detect a mechanism when its behavior is known, rather than assuming that every visualization or intervention is informative.

## Synthetic Tasks

The laboratory provides controlled algorithmic tasks with exact solvers, held-out pairs, and corruption mechanisms.

| Task | Research purpose |
|---|---|
| Repeated pattern | Investigates pattern repetition and the possibility of memorization or shortcuts. |
| Associative recall | Tests retrieval of information associated with a previously observed key or context. |
| Variable-offset induction | Investigates induction behavior when relevant tokens occur at varying offsets. |
| Single-pair induction with distractors and decoys | Examines induction behavior when the relevant association must be distinguished from competing examples. |

The task implementations are maintained in `src/transformer_lab/data/tasks.py`.

### Why controlled tasks matter

Synthetic tasks make it possible to define the intended computation, construct held-out examples, and introduce targeted corruptions.

This helps distinguish several different outcomes:

- Correct task performance through the intended mechanism.
- Correct task performance through a positional shortcut.
- Memorization of training or evaluation sequences.
- Failure to generalize to unseen pairs.
- Behavior that changes under controlled interventions.

The laboratory includes a task-level audit documenting what each task tests, which shortcuts may exist, what baselines are relevant, and what conclusions the task cannot establish.

See [`docs/task_audit.md`](docs/task_audit.md).

## Interpretability Methods

### 1. Activation Caching and Inspection

The instrumentation system supports activation caching and inspection of internal model computations.

This provides the basis for comparing intermediate activations across examples, identifying candidate components, and supplying activations to downstream attribution and intervention analyses.

Location: `src/transformer_lab/instrumentation/`

### 2. Prediction Tracing

The laboratory supports prediction tracing, direct logit attribution (DLA), and the logit lens.

These tools help investigate how internal representations and components contribute to model predictions.

The results should be interpreted according to the assumptions and limitations of each method; a contribution to a prediction is not automatically proof of a complete causal mechanism.

Location: `interpretability/attribution.py`

### 3. Ablation

Ablation evaluates how model behavior changes when selected internal components are modified or removed.

Supported intervention modes include:

- **Zero ablation:** replace the selected activation with zero.
- **Mean ablation:** replace it with a mean activation.
- **Resample ablation:** use a resampled activation.

The laboratory supports ablation of:

- Attention heads.
- MLP blocks.
- Individual neurons.

These interventions help test whether a component is necessary for a behavior under a specified experimental condition. Their effects must be interpreted in the context of the intervention and its controls.

Locations: `interpretability/interventions.py`, `scans.py`

### 4. Activation Patching

Activation patching investigates how replacing selected activations changes a model's behavior.

The implementation supports:

- Denoising and noising interventions.
- Recovery measurements.
- Confidence intervals for recovery.
- Detection of uninformative gaps.

The objective is to quantify whether a selected activation or component carries information relevant to the behavior being studied.

Location: `interpretability/patching.py`

### 5. Path Interventions

Path interventions investigate information flow between model components.

The laboratory supports interventions along head/MLP-to-head paths through:

- Query projections (Q).
- Key projections (K).
- Value projections (V).

These interventions are useful for testing hypotheses about where information enters a computation and which intermediate components transmit it.

Location: `interpretability/path.py`

### 6. Preregistered Circuit Hypothesis Testing

The hypothesis-testing system supports:

- Explicit hypothesis specifications.
- Frozen preregistration configurations.
- Selection and test splits.
- Controls.
- Quantitative verdicts.
- Replication across multiple trained runs.

The system refuses to run a preregistered analysis if the specification has changed after being frozen.

This is intended to reduce the risk of changing the hypothesis after inspecting the results and then presenting the revised analysis as if it had been specified in advance.

Locations: `interpretability/hypothesis.py`, `configs/hypotheses/`

### 7. Experiment Registry and Reproduction

The experiment system records analyses, supports comparisons between runs, and allows previously recorded analyses to be rerun.

The rerun workflow re-executes an analysis from its saved record and compares every value against the original.

This makes it possible to investigate whether a reported result is reproducible under the recorded setup.

## Principal Findings

**Detailed evidence and artifact paths:** [`docs/findings.md`](docs/findings.md)

The following findings summarize the investigations recorded in the project. They describe the reported experimental configurations and should not be generalized beyond the evidence documented in the repository.

### 1. Positional Scheme Influenced Induction-Circuit Formation

With learned absolute positional embeddings, a two-layer attention-only model plateaued at approximately 35% accuracy and exhibited a positional shortcut without a previous-token head.

With ALiBi, three seeds reached 91–95% accuracy.

A one-layer model reached 5% accuracy, compared with a reported chance level of 2%.

These results indicate that the positional scheme substantially affected the behavior learned in the tested setup.

### 2. The Learned Circuit Was Distributed and Dominated by Key Composition

The reported circuit involved a group of layer-1 heads whose patching restored 98% of the measured behavior.

Removing layer-0-derived information from the keys alone reduced the relevant logit difference by approximately 8–10 nats.

The finding was replicated on seeds that were not used to formulate the hypothesis.

This supports a distributed, key-composition-dominated account for the tested model and task, rather than a mechanism explained by a single isolated component.

### 3. The Preregistered Textbook Hypothesis Was Partly Wrong

The expected sharp previous-token head did not emerge in the tested setup.

Queries also depended on layer-0 information.

This is an important negative result: the investigation did not simply confirm the original circuit hypothesis. The observed mechanism required a revised account of how information contributes to induction.

### 4. Retrieval Was Context-Dependent, and the Preference Was Learned

In the preregistered investigation, when two earlier occurrences of the query token were available, models preferred the occurrence whose preceding token matched.

The reported effect was approximately +1.9 to +2.3 logits across all three seeds.

The effect depended on the training distribution:

- Models trained where the preceding context repeated showed the matching-context preference.
- Models trained where that context never repeated showed a reversed preference.

Path patching localized the context signal to the induction heads' **queries**. This result was confirmed on two seeds trained after the preregistration was frozen.

A further preregistered investigation identified layer-0 head `L0H0` as the head associated with the steepest slope in the tested context-matching analysis. This was confirmed on two additional seeds.

An exploratory investigation also found that the preference increased with each additional matching context token, from one to three.

The preregistered findings and exploratory observation are distinct and should be interpreted accordingly.

### 5. Adding MLPs Rerouted the Circuit

When MLPs were introduced, layer-0 information reached the keys primarily through `MLP0` in the reported experiment.

This finding was confirmed on a seed trained after the hypothesis had been written.

The result illustrates that changing the model architecture can change the computational route used to implement a behavior, even when the high-level task remains related.

### 6. Accuracy Did Not Identify the Underlying Mechanism

A fixed-offset model achieved 100% accuracy while copying purely by position.

Its content-induction score was 0% on shuffled inputs.

This demonstrates why task accuracy alone is insufficient to establish that the intended algorithm has been learned.

The model succeeded on the original evaluation setup but failed a control designed to distinguish positional behavior from content-based induction.

### 7. Negative Results Were Preserved

The project records failures and limitations rather than reporting only successful experiments.

Examples include:

- Single-pair induction training learned anti-generalizing priors, achieving 0/512 on held-out pairs.
- The leakage audit exposed memorisable evaluation sequences in `repeat_pattern`.

These findings motivated the use of held-out examples, leakage audits, controls, and explicit task limitations.

For the complete experiment history, evidence, and artifact paths, see [`docs/findings.md`](docs/findings.md).

## Quick Start

The project provides a one-click Windows launcher, direct CLI commands, training workflows, interpretability analyses, and a local web interface.

### Option 1 — One-Click Start

Double-click:

```text
start_lab.bat
```

Or run:

```bash
python start.py
```

On first use, the launcher:

1. Creates the `.venv` virtual environment.
2. Installs the project after asking before downloading anything.
3. Checks the required imports.
4. Displays an interactive menu.

The menu contains:

```text
1  Open the lab (web interface in your browser)   [Enter]
2  Quick health check (environment + end-to-end smoke test)
3  Run the full test suite (~3-4 minutes)
4  Reproduction audit of all trained runs
5  List trained runs and experiment records
6  Open a terminal with the `itl` command ready
0  Quit
```

Each action is also available directly:

```bash
python start.py ui
python start.py check
python start.py test
python start.py audit
python start.py runs
python start.py shell
python start.py env
```

The launcher also accepts the action names as a single command argument, such as `python start.py ui`.

The web interface binds to localhost only. Close it with `Ctrl+C` in the launcher terminal.

**Windows naming note:** the launcher is intentionally named `start_lab.bat` rather than `start.bat`, because `start` is a built-in Windows Command Prompt command.

### Option 2 — Manual Installation on Windows

The documented environment was tested with Python 3.13.3 and PyTorch 2.10.0+cpu on Windows 11.

Create the virtual environment:

```bash
python -m venv .venv --system-site-packages
```

Install the project:

```bash
.venv/Scripts/python -m pip install -e . --no-build-isolation
```

The `--system-site-packages` option allows the environment to reuse an existing PyTorch installation.

If you prefer a fresh, isolated dependency installation, omit `--system-site-packages` when creating the environment.

Optional dependency groups are available for the UI, diagnostics, and development:

```bash
pip install -e .[ui,diag,dev]
```

These extras provide the optional Streamlit/pandas, psutil, and pytest dependencies described by the project configuration.

## Quick Start: Training and Analysis

The commands below use the Windows virtual-environment executable paths documented by the project.

### 1. Validate the Installation

Run the smoke workflow:

```bash
.venv/Scripts/itl smoke
```

This trains a tiny model and exercises the project's components. The documented runtime is approximately 30 seconds, depending on the environment.

### 2. Run the Test Suite

```bash
.venv/Scripts/python -m pytest
```

The suite includes unit, integration, scientific, and regression tests.

### 3. Train the Principal Research Model

Train the two-layer attention-only model with ALiBi:

```bash
.venv/Scripts/itl train --config configs/experiments/induction_2l_attn.yaml
```

The documented runtime is a few minutes on CPU.

### 4. Train Multiple Seeds and Architectural Controls

Run three seeds of the principal model alongside the one-layer and MLP-containing architectural controls, with two jobs at a time:

```bash
.venv/Scripts/python scripts/train_queue.py --parallel 2 configs/experiments/induction_2l_attn.yaml:0,1,2 configs/experiments/induction_1l_attn.yaml:0 configs/experiments/induction_2l_mlp.yaml:0
```

This allows the principal model to be compared with architectural alternatives while preserving the selected training configurations and seed identifiers.

### 5. Evaluate a Training Run

Replace `<RUN>` with a run directory reported by `itl runs`.

Evaluate all checkpoints:

```bash
.venv/Scripts/itl eval --run <RUN> --all-checkpoints
```

Inspect scores:

```bash
.venv/Scripts/itl scores --run <RUN>
```

### 6. Run Interpretability Interventions

Run mean ablation:

```bash
.venv/Scripts/itl ablate --run <RUN> --method mean
```

Run activation patching with shuffled-input corruption and logit-difference measurement:

```bash
.venv/Scripts/itl patch --run <RUN> --corruption shuffle --metric logit_diff
```

Run formation analysis with ablation:

```bash
.venv/Scripts/itl formation --run <RUN> --ablate
```

Trace an individual example:

```bash
.venv/Scripts/itl trace --run <RUN> --example 0
```

### 7. Test a Preregistered Circuit Hypothesis

Evaluate the induction hypothesis across multiple trained seeds:

```bash
.venv/Scripts/itl hypothesis --spec configs/hypotheses/induction_2l.yaml --run <RUN_SEED0> --run <RUN_SEED1> --run <RUN_SEED2>
```

Replace the placeholders with the corresponding run identifiers.

The analysis uses the hypothesis specification, controls, and test procedure defined by the repository.

### 8. Inspect and Compare Experiments

List trained runs:

```bash
.venv/Scripts/itl runs
```

List experiment records:

```bash
.venv/Scripts/itl experiments
```

Rerun a recorded analysis:

```bash
.venv/Scripts/itl rerun <EXPERIMENT_ID>
```

The rerun command re-executes an analysis from its saved record and diffs every value against the original.

### 9. Discover Tasks, Configurations, and Reproducibility Status

List available tasks:

```bash
.venv/Scripts/itl tasks
```

List configurations:

```bash
.venv/Scripts/itl configs
```

Verify all recorded runs:

```bash
.venv/Scripts/itl verify --all
```

These commands help identify the available experimental workflows and audit the reproducibility of recorded runs.

## Preregistered Experiments

The repository includes preregistered experimental designs for investigating context matching, context routing, and the roles of attention heads.

The analysis refuses to run if a specification has changed after it was frozen.

### Train the Context-Matching Controls

Run the same-order and shuffled-order training configurations:

```bash
.venv/Scripts/python scripts/train_queue.py --parallel 2 configs/experiments/ctx_pairs_same_order.yaml:0,1,2 configs/experiments/ctx_pairs_shuffled.yaml:0,1,2
```

### Context-Matching Preregistration

```bash
.venv/Scripts/itl prereg --spec configs/preregistration/context_matching_v1.yaml
```

### Context-Routing Preregistration

```bash
.venv/Scripts/itl prereg --spec configs/preregistration/context_route_v1.yaml
```

### Context-Heads Preregistration

```bash
.venv/Scripts/itl prereg --spec configs/preregistration/context_heads_v1.yaml
```

The specifications define the corresponding experimental analyses and should be treated as frozen research records once preregistered.

## Exploratory Investigations

The laboratory also supports exploratory investigations that can be rerun and that include hand-wired reference models.

For example, investigate context-dose effects:

```bash
.venv/Scripts/itl investigate context-dose --run <RUN>
```

Replace `<RUN>` with a valid run identifier.

Exploratory results can generate or refine hypotheses, but they should not be represented as preregistered confirmations unless they were evaluated under a corresponding frozen specification.

## Launch the Local UI

Run:

```bash
.venv/Scripts/itl ui
```

The Streamlit interface is designed for local use and binds to localhost. Usage statistics are disabled in the documented configuration.

The UI provides an interactive entry point to the research system; the CLI remains available for scripted, reproducible workflows.

## Command Reference

| Command | Purpose |
|---|---|
| `itl smoke` | Validate the installation using a small end-to-end workflow. |
| `itl train --config <CONFIG>` | Train a model using a configuration file. |
| `itl eval --run <RUN> --all-checkpoints` | Evaluate all checkpoints for a run. |
| `itl scores --run <RUN>` | Inspect task scores. |
| `itl ablate --run <RUN> --method mean` | Run mean ablation. |
| `itl patch --run <RUN> --corruption shuffle --metric logit_diff` | Run activation patching with a selected corruption and metric. |
| `itl formation --run <RUN> --ablate` | Run formation analysis with ablation. |
| `itl trace --run <RUN> --example 0` | Trace an example's prediction. |
| `itl hypothesis --spec <SPEC> --run <RUN>` | Evaluate a specified circuit hypothesis on one or more runs. |
| `itl runs` | List trained runs. |
| `itl experiments` | List experiment records. |
| `itl rerun <EXPERIMENT_ID>` | Re-execute and compare a recorded analysis. |
| `itl tasks` | List available tasks. |
| `itl configs` | List available configurations. |
| `itl verify --all` | Audit recorded runs for reproducibility. |
| `itl prereg --spec <SPEC>` | Execute a preregistered analysis. |
| `itl investigate context-dose --run <RUN>` | Run the exploratory context-dose investigation. |
| `itl ui` | Launch the local Streamlit interface. |

The documented Windows commands use `.venv/Scripts/itl` and `.venv/Scripts/python`. The shorter `itl` command is available in a terminal configured by the launcher or in an appropriately activated environment.

## Reproducibility and Experiment Management

Reproducibility is a core design objective rather than an optional reporting feature.

### Configuration-driven training

Training configurations specify the model, task, optimization setup, and seed.

The training system supports resumable execution and includes a leakage audit.

This helps keep training settings explicit and makes it possible to compare experiments without relying on undocumented changes to the execution environment.

### Preregistered hypotheses

The hypothesis system records experimental specifications before the associated analysis is performed.

Frozen specifications are protected against post-freeze modification: the analysis refuses to run if the specification has changed after freezing.

Selection and test splits, controls, and verdicts are part of the documented hypothesis-testing capability.

### Saved experiments and reruns

Experiment records allow analyses to be compared and rerun.

The rerun workflow re-executes an analysis from its saved record and compares every value against the original, providing a direct way to investigate reproducibility.

### Held-out examples and multiple seeds

The research workflows use held-out pairs and multiple seeds to test whether an observed result persists beyond the specific examples or initialization used to formulate a hypothesis.

A result observed on the same examples used to select a mechanism is weaker evidence than a result replicated on held-out examples and independent trained runs.

### Ground-truth validation

Hand-wired reference models provide known mechanisms against which analysis methods can be checked.

This is important because an intervention or visualization can be misleading if its behavior is not understood. Validating tools against a known computational structure helps establish what conclusions the methods can reasonably support.

### Artifact storage

Generated artifacts are stored under `./artifacts` by default, including run and experiment outputs.

The directory is git-ignored.

Set `ITL_ARTIFACT_ROOT` to change the artifact root:

```text
ITL_ARTIFACT_ROOT=<your-local-artifact-directory>
```

Choose a persistent local directory if you want artifacts outside the current project or outside a synchronized Desktop folder.

## Project Structure

```text
.
├── configs/
│   ├── experiments/            # Model, task, optimization and seed configurations
│   ├── hypotheses/             # Preregistered circuit hypotheses
│   └── preregistration/        # Frozen experimental specifications
│
├── src/
│   └── transformer_lab/
│       ├── models/             # Transparent Transformer implementations
│       ├── data/               # Synthetic tasks and dataset generation
│       ├── training/           # Configuration-driven, resumable training
│       └── instrumentation/    # Activation caching and inspection
│
├── models/
│   └── handwired.py            # Hand-wired induction reference model
│
├── interpretability/
│   ├── attribution.py          # Direct logit attribution and logit lens
│   ├── interventions.py        # Ablation methods
│   ├── patching.py             # Activation patching and recovery
│   ├── path.py                 # Path interventions
│   ├── hypothesis.py           # Preregistered hypothesis testing
│   └── scans.py                # Component and neuron scans
│
├── experiments/                # Experiment registry and analysis records
├── reporting/                  # Reports, comparisons and reproduction support
├── app/
│   └── streamlit_app.py        # Local Streamlit UI
├── scripts/
│   ├── train_queue.py          # Multi-run training queue
│   └── ...                     # Follow-up investigations and utilities
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── scientific/             # Scientific and method-validation tests
│   └── regression/             # Regression tests
├── docs/                       # Research, architecture and methodology docs
├── artifacts/                  # Generated, git-ignored run and experiment artifacts
├── start.py                    # Interactive launcher
├── start_lab.bat               # Windows one-click launcher
└── ...
```

This is a conceptual overview of the supplied project layout. Consult the repository for the complete set of files and modules.

## Documentation

The documentation connects the mathematical foundations, implementation details, experimental protocols, findings, and limitations.

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Data flow, tensor shapes, module responsibilities, and design decisions. |
| [`docs/mathematics.md`](docs/mathematics.md) | Mathematical derivations mapped to the code. |
| [`docs/interpretability_guide.md`](docs/interpretability_guide.md) | What each type of interpretability evidence can and cannot establish. |
| [`docs/experiment_protocols.md`](docs/experiment_protocols.md) | Experimental handbook covering research questions, setup, metrics, controls, and commands. |
| [`docs/findings.md`](docs/findings.md) | Results from every recorded experiment, including artifact paths; the summary table appears first. |
| [`docs/research_log.md`](docs/research_log.md) | Research decisions, experiments, and ranked next steps. Recommended starting point when continuing the research. |
| [`docs/status.md`](docs/status.md) | Capabilities and claims with their evidence status. |
| [`docs/preregistration_context_matching_v1.md`](docs/preregistration_context_matching_v1.md) | The preregistered context-matching design. |
| [`docs/method_validation.md`](docs/method_validation.md) | Validation of analysis methods against hand-wired ground truth. |
| [`docs/task_audit.md`](docs/task_audit.md) | Per-task objectives, shortcuts, baselines, and limitations. |
| [`docs/limitations.md`](docs/limitations.md) | Limitations and caveats; read before quoting experimental conclusions. |
| [`docs/interview_guide.md`](docs/interview_guide.md) | Difficult technical questions with answers and code pointers. |
| [`docs/references.md`](docs/references.md) | Verified bibliography. |
| [`docs/environment_audit.md`](docs/environment_audit.md) | Machine and environment used for the documented experiments. |

### Recommended reading order

For someone new to the repository:

1. [`docs/research_log.md`](docs/research_log.md) — understand the research trajectory and what to investigate next.
2. [`docs/architecture.md`](docs/architecture.md) — understand the code structure and data flow.
3. [`docs/mathematics.md`](docs/mathematics.md) — connect the mathematical ideas to the implementation.
4. [`docs/interpretability_guide.md`](docs/interpretability_guide.md) — understand the strengths and limitations of the analysis methods.
5. [`docs/experiment_protocols.md`](docs/experiment_protocols.md) — understand how experiments are designed and evaluated.
6. [`docs/findings.md`](docs/findings.md) — examine the findings and their artifact paths.
7. [`docs/method_validation.md`](docs/method_validation.md) — review validation against known ground truth.
8. [`docs/limitations.md`](docs/limitations.md) — understand what the results do not establish.
9. [`docs/status.md`](docs/status.md) — inspect the evidence status of project capabilities and claims.

## Limitations and Interpretation

Mechanistic interpretability is a process of constructing and testing explanations, not simply producing visualizations.

The findings in this repository are tied to particular models, tasks, configurations, seeds, and experimental interventions.

Several distinctions are essential when interpreting the results.

### Task performance is not proof of mechanism

A model can solve a task through a shortcut or memorization instead of learning the intended algorithm.

The fixed-offset model that achieved 100% accuracy but failed content induction on shuffled inputs illustrates this distinction.

### Ablation and patching are intervention-specific

An ablation measures the consequences of a particular component replacement or removal. Activation patching measures the effect of a specified activation intervention.

Neither should automatically be interpreted as a complete explanation of the model's algorithm. Results must be considered alongside the intervention design, controls, and other supporting evidence.

### Preregistered and exploratory results are different

Preregistered experiments test hypotheses defined in advance. Exploratory investigations can discover patterns and generate new hypotheses.

Exploratory observations should not be presented as preregistered confirmations when they were not evaluated under the corresponding frozen specification.

### Generalization must be tested explicitly

Training performance, performance on familiar evaluation sequences, and generalization to held-out pairs are different outcomes.

The project's leakage audit and negative findings demonstrate why evaluation design is part of the scientific methodology.

### Findings are scoped to the tested setup

The results describe the tested small Transformers and synthetic algorithmic tasks. They do not automatically establish how larger language models behave or prove that the same circuits will appear under different architectures and training distributions.

For a fuller discussion, read [`docs/limitations.md`](docs/limitations.md), [`docs/interpretability_guide.md`](docs/interpretability_guide.md), and [`docs/task_audit.md`](docs/task_audit.md).

## Local-Only Execution and Privacy

The laboratory is designed to run locally after installation.

- No telemetry is required.
- No remote logging is required.
- No network access is required after installation for the documented local workflows.
- The Streamlit interface binds to localhost.
- Generated artifacts are stored locally by default.
- The artifact directory is git-ignored.

Initial installation may require downloading Python packages and their dependencies. The launcher asks before downloading anything on first use.

The default artifact location is under `./artifacts`. Because the current Desktop directory is inside a OneDrive-synced location, users who want artifacts stored outside that synchronization boundary should set `ITL_ARTIFACT_ROOT` to a different local directory.

## Research Principles

The project is built around the following principles:

- **Test hypotheses, not just accuracy.** A successful task score is the beginning of a mechanistic investigation, not its conclusion.
- **Use controls.** Corruptions, held-out examples, architectural comparisons, and negative controls help distinguish intended behavior from shortcuts.
- **Validate analysis methods.** Hand-wired models provide known ground truth against which interpretability tools can be tested.
- **Separate discovery from confirmation.** Exploratory observations and preregistered findings are labeled and interpreted differently.
- **Preserve negative results.** Failed generalization, incorrect hypotheses, and leakage discoveries remain part of the research record.
- **Replicate across seeds.** Independent runs help determine whether a finding persists beyond a particular initialization.
- **Keep artifacts.** Training configurations, experiment records, and analysis outputs support reproducibility and future investigation.
- **State limitations explicitly.** Conclusions should not exceed what the task, model, intervention, and evidence establish.

---

<div align="center">

### Inside the Transformer

**Observe the computation. Intervene on the mechanism. Test the hypothesis. Replicate the evidence.**

A local research laboratory for understanding how small Transformers learn algorithmic behavior—and what their internal circuits actually do.

</div>
