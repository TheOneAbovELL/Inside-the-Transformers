"""Inside the Transformer — local research interface (Streamlit).

Launch:  itl ui    (or: streamlit run app/streamlit_app.py)
Runs only on localhost; usage statistics are disabled by the launcher. Every number shown is
computed from saved runs/experiments or live by the backend — nothing is hard-coded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from transformer_lab.data.tasks import make_task
from transformer_lab.evaluation.metrics import run_batched
from transformer_lab.experiments.registry import list_experiments, list_runs
from transformer_lab.experiments.runner import env_report
from transformer_lab.instrumentation.inspect import inspect_example, list_checkpoints, trace_prediction
from transformer_lab.interpretability.attention_scores import head_scores
from transformer_lab.interpretability.interventions import Reference, head_ablation_hooks, mlp_ablation_hooks, reference_names
from transformer_lab.interpretability.patching import Patcher, PatchSpec, task_regions
from transformer_lab.interpretability.stats import bootstrap_ci
from transformer_lab.training import load_run
from transformer_lab.utils import PROJECT_ROOT
from transformer_lab.visualization import plots

st.set_page_config(page_title="Inside the Transformer", layout="wide")
CAVEAT = ("⚠️ Attention weights show where a head looks on these inputs. They do **not** by themselves "
          "establish causal importance, faithful explanation or semantic understanding — use the ablation "
          "and patching pages for causal evidence.")

PAGES = ["Overview", "Training runs", "Dataset", "Predictions", "Attention", "Activations", "Ablation",
         "Activation patching", "Hypotheses & evidence", "Compare", "Reports", "Documentation"]
page = st.sidebar.radio("Section", PAGES)


@st.cache_resource(show_spinner="Loading checkpoint ...")
def _load(run: str, ckpt: str):
    model, task, summary, cfg = load_run(run, ckpt)
    return model, task, summary, cfg


def _ci(s):
    return f"{s['mean']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]"


def pick_run(key="run"):
    runs = [r for r in list_runs() if r["status"] in ("completed", "early_stopped", "interrupted")]
    if not runs:
        st.warning("No trained runs yet. Train one with `itl train --config configs/experiments/induction_2l_attn.yaml`.")
        st.stop()
    labels = {f"{Path(r['run_dir']).name}  (acc={r['val_accuracy']}, {r['positional']})": r["run_dir"] for r in runs}
    run = labels[st.sidebar.selectbox("Run", list(labels), key=key)]
    cks = ["final"] + [p.stem for _, p in list_checkpoints(run)]
    ckpt = st.sidebar.selectbox("Checkpoint", cks, key=key + "_ck")
    n = st.sidebar.slider("Examples", 16, 512, 128, step=16, key=key + "_n")
    seed = st.sidebar.number_input("Example seed", value=1234, key=key + "_seed")
    model, task, summary, cfg = _load(run, ckpt)
    batch = task.sample(n, np.random.default_rng(int(seed)))
    st.sidebar.caption(f"{cfg.model.n_layers}L × {cfg.model.n_heads}H, d={cfg.model.d_model}, "
                       f"{cfg.model.positional}, attn_only={cfg.model.attn_only}")
    return run, ckpt, model, task, summary, cfg, batch, int(seed)


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ============================================================================= pages
if page == "Overview":
    st.title("Inside the Transformer")
    st.markdown("A local laboratory for **hypothesis → training → observation → intervention → measurement → "
                "replication** on small transformers.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Environment")
        st.json(env_report())
    with c2:
        st.subheader("Artifacts")
        runs = list_runs()
        st.metric("Training runs", len(runs))
        st.metric("Analysis experiments", len(list_experiments()))
        if runs:
            st.dataframe(pd.DataFrame(runs)[["name", "status", "steps", "val_accuracy", "test_accuracy", "positional", "n_layers"]])

elif page == "Training runs":
    runs = list_runs()
    if not runs:
        st.info("No runs yet.")
        st.stop()
    run = st.selectbox("Run", [r["run_dir"] for r in runs], format_func=lambda p: Path(p).name)
    d = Path(run)
    events = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    tr = pd.DataFrame([e for e in events if e["kind"] == "train"])
    ev = pd.DataFrame([e for e in events if e["kind"] == "eval"])
    c1, c2 = st.columns(2)
    if not tr.empty:
        c1.subheader("Training loss / grad norm")
        c1.line_chart(tr.set_index("step")[["loss", "grad_norm"]])
    if not ev.empty:
        c2.subheader("Validation / held-out accuracy")
        c2.line_chart(ev.set_index("step")[["val_accuracy", "test_accuracy"]])
    if (d / "summary.json").exists():
        s = json.loads((d / "summary.json").read_text())
        st.subheader("Summary")
        st.write({k: s.get(k) for k in ("status", "steps_completed", "elapsed_s", "steps_per_s", "n_params")})
        st.write("Baselines (test set):", s.get("baselines_test"))
        st.write("Leakage audit:", s.get("leakage_audit"))
    st.subheader("Config")
    st.code((d / "config.yaml").read_text(), language="yaml")

elif page == "Dataset":
    st.title("Dataset inspection")
    from transformer_lab.data.tasks import TASKS
    name = st.selectbox("Task", sorted(TASKS))
    defaults = TASKS[name].defaults
    params = json.loads(st.text_area("Parameters (JSON)", json.dumps(defaults)))
    task = make_task(name, params)
    split = st.radio("Split", ["train", "eval"], horizontal=True)
    seed = st.number_input("Seed", value=0)
    b = task.sample(8, np.random.default_rng(int(seed)), split)
    st.caption(task.description + f"  |  vocab={task.vocab_size}, seq_len={task.seq_len}, chance={task.chance_accuracy:.3f}")
    for i in range(len(b)):
        toks = task.decode(b.tokens[i])
        mask = b.loss_mask[i].tolist() + [False]
        txt = " ".join(f"**[{t}]**" if (j > 0 and mask[j - 1]) else t for j, t in enumerate(toks))
        st.markdown(f"{i}: {txt}")
    st.caption("Bold tokens are scored targets (predicted from the token before them).")
    st.write("Metadata:", {k: np.asarray(v).tolist() for k, v in b.meta.items()})

elif page == "Predictions":
    run, ckpt, model, task, summary, cfg, batch, seed = pick_run()
    st.title("Prediction explorer")
    m = run_batched(model, batch)
    st.write({k: _ci(bootstrap_ci(v)) for k, v in m.items() if not np.all(np.isnan(v))})
    ok = m["seq_correct"] >= 0.5
    st.caption(f"{ok.sum()} fully correct / {(~ok).sum()} with errors")
    idx = st.number_input("Example", 0, len(batch) - 1, 0)
    info = inspect_example(model, task, batch, int(idx))
    df = pd.DataFrame([{k: r[k] for k in ("pos", "token", "target", "pred", "p_target", "correct", "scored")}
                       for r in info["positions"]])
    st.dataframe(df[df["scored"]] if st.checkbox("Scored positions only", True) else df)
    st.write("Meta:", info["meta"])

elif page == "Attention":
    run, ckpt, model, task, summary, cfg, batch, seed = pick_run()
    st.title("Attention explorer")
    st.warning(CAVEAT)
    sc = head_scores(model, batch)
    kind = st.selectbox("Head score", list(sc))
    show_fig(plots.head_grid(sc[kind], f"{kind} score", f"run={Path(run).name} | ckpt={ckpt} | n={len(batch)}"))
    idx = st.number_input("Example", 0, len(batch) - 1, 0)
    l = st.selectbox("Layer", range(model.cfg.n_layers))
    h = st.selectbox("Head", range(model.cfg.n_heads))
    info = inspect_example(model, task, batch, int(idx))
    show_fig(plots.attention_pattern(info["patterns"][l, h], info["tokens"], f"L{l}H{h}",
                                     f"run={Path(run).name} | ckpt={ckpt} | example {idx} (seed {seed})"))

elif page == "Activations":
    run, ckpt, model, task, summary, cfg, batch, seed = pick_run()
    st.title("Activation explorer")
    idx = int(st.number_input("Example", 0, len(batch) - 1, 0))
    info = inspect_example(model, task, batch, idx)
    st.subheader("Residual stream norm by position")
    st.line_chart(pd.DataFrame(info["resid_norms"]))
    st.subheader("Attention-output norm by position and layer")
    st.line_chart(pd.DataFrame(info["attn_out_norms"].T, columns=[f"L{l}" for l in range(model.cfg.n_layers)]))
    if "mlp_post" in info:
        l = st.selectbox("MLP layer", range(model.cfg.n_layers))
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(info["mlp_post"][l].T, aspect="auto", cmap="magma")
        ax.set_xlabel("position")
        ax.set_ylabel("neuron")
        show_fig(fig)
    st.subheader("Trace one prediction (DLA + logit lens)")
    scored = [r["pos"] for r in info["positions"] if r["scored"]]
    pos = st.selectbox("Scored position", scored)
    tr = trace_prediction(model, task, batch, idx, int(pos))
    st.caption("DLA freezes the final LayerNorm scale; it measures direct effects only.")
    st.dataframe(pd.DataFrame(tr["dla"], columns=["component", "contribution"]))
    st.dataframe(pd.DataFrame(tr["logit_lens"]))

elif page == "Ablation":
    run, ckpt, model, task, summary, cfg, batch, seed = pick_run()
    st.title("Ablation experiments")
    heads = [f"{l}.{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    chosen = st.multiselect("Heads to ablate", heads)
    mlps = [] if model.cfg.attn_only else st.multiselect("MLP blocks to ablate", list(range(model.cfg.n_layers)))
    method = st.radio("Method", ["mean", "zero", "resample"], horizontal=True)
    if st.button("Run ablation") and (chosen or mlps):
        ref_b = task.sample(len(batch), np.random.default_rng(seed + 7))
        ref = Reference.from_batch(model, ref_b, reference_names(model, mlps=True))
        hooks = head_ablation_hooks([tuple(map(int, x.split("."))) for x in chosen], method, ref) if chosen else []
        hooks += mlp_ablation_hooks(mlps, method, ref) if mlps else []
        base = run_batched(model, batch)
        abl = run_batched(model, batch, fwd_hooks_fn=lambda sl: hooks)
        rows = [{"metric": k, "baseline": _ci(bootstrap_ci(base[k])), "ablated": _ci(bootstrap_ci(abl[k])),
                 "paired Δ": _ci(bootstrap_ci(abl[k] - base[k]))} for k in ("logprob", "accuracy", "loss")]
        st.table(pd.DataFrame(rows))
        st.caption(f"n={len(batch)} examples; 95% bootstrap CIs; reference batch seed {seed + 7}. "
                   "A small effect does not prove a component is unimportant (redundancy).")

elif page == "Activation patching":
    run, ckpt, model, task, summary, cfg, batch, seed = pick_run()
    st.title("Activation patching (denoising: clean → corrupted)")
    if not task.corruption_kinds:
        st.info("This task defines no corruption.")
        st.stop()
    kind = st.selectbox("Corruption", task.corruption_kinds)
    corrupt = task.corrupt(batch, np.random.default_rng(seed + 11), kind)
    metric = st.selectbox("Metric", ["logprob", "logit_diff"] if corrupt.alt_targets is not None else ["logprob"])
    comp = st.selectbox("Component", ["head z"] + [f"resid_pre L{l}" for l in range(model.cfg.n_layers)])
    regions = task_regions(task, batch)
    region = st.selectbox("Positions", list(regions) + ["all"])
    pos = None if region == "all" else regions[region]
    n_runs = model.cfg.n_layers * model.cfg.n_heads if comp == "head z" else 1
    st.caption(f"Will run {n_runs + 2} forward passes over {len(batch)} examples (clean, corrupted, "
               f"{n_runs} patched). Nothing runs until you press the button.")
    if not st.button("Run patching"):
        st.stop()
    p = Patcher(model, batch, corrupt, metric=metric)
    st.write(f"clean={p.m_clean.mean():.3f}  corrupted={p.m_corrupt.mean():.3f}  gap={p.gap:.3f}  "
             f"({'informative' if p.informative else 'UNINFORMATIVE — recovery not computed'})")
    st.caption("R = (patched − corrupted) / (clean − corrupted) on batch means. Denoising shows sufficiency "
               "given the rest of the corrupted run, not necessity; an AND-like component can be necessary and "
               "still show R ≈ 0. Check the per-example median before reading R ≈ 1 as complete recovery.")
    if comp == "head z":
        rows = []
        for l in range(model.cfg.n_layers):
            for h in range(model.cfg.n_heads):
                r = p.run(PatchSpec(f"L{l}H{h}", f"blocks.{l}.attn.hook_z", heads=[h], positions=pos))
                rows.append(r.row())
        cols = [c for c in ["label", "patched", "recovery", "recovery_lo", "recovery_hi",
                            "median_per_example_recovery", "frac_examples_over_recovered", "informative"]
                if c in pd.DataFrame(rows).columns]
        st.dataframe(pd.DataFrame(rows)[cols])
    else:
        l = int(comp.split("L")[1])
        r = p.run(PatchSpec(comp, f"blocks.{l}.hook_resid_pre", positions=pos))
        st.write(r.row())

elif page == "Hypotheses & evidence":
    st.title("Hypotheses & evidence")
    st.markdown("Every verdict below comes from a saved experiment record. **supported / contradicted / "
                "inconclusive** follow the rules written in each hypothesis file *before* it was run; a "
                "'supported' prediction holds only under its stated intervention, threshold and examples. "
                "The family-wise column applies the same rule with Bonferroni-adjusted intervals.")
    exps = [e for e in list_experiments() if e["kind"] in ("hypothesis", "prereg")][::-1]
    if not exps:
        st.info("No hypothesis experiments yet (`itl hypothesis --spec ... --run ...`).")
        st.stop()
    labels = [f"{x['id']} — {x['headline']}" for x in exps]  # plain strings: robust across Streamlit versions
    e = exps[labels.index(st.selectbox("Experiment", labels))]
    res = json.loads((Path(e["dir"]) / "results.json").read_text(encoding="utf-8"))
    if e["kind"] == "prereg" and "measurements" not in res:  # e.g. context_route_v1
        st.subheader(f"Preregistered: {res['spec_id']}  (spec sha256 {res['spec_sha256'][:12]})")
        st.json(res.get("verdict", res.get("headline", {})))
        tables = [p for p in Path(e["dir"]).glob("*.csv")]
        for t in tables:
            st.caption(t.name)
            st.dataframe(pd.read_csv(t))
    elif e["kind"] == "prereg":
        st.subheader(f"Preregistered: {res['spec_id']}  (spec sha256 {res['spec_sha256'][:12]})")
        st.write("Metric calibration valid:", res["metric_valid"])
        st.json(res["verdicts"])
        rows = []
        for cond, ms in res["measurements"].items():
            for m in ms:
                if "cp" in m:
                    rows.append({"condition": cond, "model": m["model"], "valid": m["valid"],
                                 "CP": m["cp"]["mean"], "CP_lo": m["cp"]["lo"], "CP_hi": m["cp"]["hi"],
                                 "either_rate": m["either_rate"], "exclusion": m.get("exclusion", "")})
        st.dataframe(pd.DataFrame(rows))
    else:
        for run, r in res["per_run"].items():
            st.subheader(Path(run).name)
            st.caption(f"candidates: {r['candidates']} | baseline accuracy {r['baseline']['accuracy']:.3f} | "
                       f"selection/test overlap: {r.get('set_overlap', 'not recorded')}")
            st.dataframe(pd.DataFrame([{"id": p["id"], "verdict": p["verdict"],
                                        "family-wise": p.get("verdict_familywise", "n/a"),
                                        "reason": p.get("reason", ""), "claim": p.get("claim", "")}
                                       for p in r["predictions"]]))
    for png in sorted((Path(e["dir"]) / "plots").glob("*.png")):
        st.image(str(png), caption=png.name)

elif page == "Compare":
    st.title("Compare runs")
    runs = list_runs()
    sel = st.multiselect("Runs", [r["run_dir"] for r in runs], format_func=lambda p: Path(p).name)
    if sel:
        from transformer_lab.experiments.registry import compare
        st.dataframe(pd.DataFrame(compare(sel)))

elif page == "Reports":
    st.title("Experiment reports")
    exps = list_experiments()[::-1]
    if not exps:
        st.info("No experiments yet (e.g. `itl scores --run <run>`).")
        st.stop()
    labels = [f"{x['id']} — {x['headline']}" for x in exps]  # plain strings: robust across Streamlit versions
    e = exps[labels.index(st.selectbox("Experiment", labels))]
    d = Path(e["dir"])
    report = d / "report.md"
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8").split("![")[0])
    else:
        st.warning("This record has no report.md (incomplete or interrupted experiment); showing raw results.")
        if (d / "results.json").exists():
            st.json(json.loads((d / "results.json").read_text(encoding="utf-8")))
    for png in sorted((d / "plots").glob("*.png")):
        st.image(str(png), caption=png.name)

elif page == "Documentation":
    docs = sorted((PROJECT_ROOT / "docs").glob("*.md"))
    doc = st.selectbox("Document", docs, format_func=lambda p: p.name)
    st.markdown(doc.read_text(encoding="utf-8"))
