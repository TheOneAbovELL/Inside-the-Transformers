# Technical interview guide

Hard questions about this project, with answers you should be able to defend and pointers to the
code and evidence. Numbers quoted come from saved experiments listed in `docs/findings.md`.

---

## A. Architecture

**1. Walk me through the shapes from tokens to logits.**
`[b,s]` ids → `W_E[ids]` `[b,s,d]` (+ positions unless ALiBi) → per block: LN → Q,K,V `[b,s,h,e]` →
scores `[b,h,s,s]` (scaled, +ALiBi bias, causal −∞) → softmax → `z` `[b,s,h,e]` → `Σ_h z_h W_O^h + b_O`
`[b,s,d]` added to the stream (→ MLP if present) → final LN → `W_U` → `[b,s,V]`.
Code: `models/transformer.py`; table in `docs/architecture.md` §2.

**2. Why divide by √d_head? What happens if you don't?**
For unit-variance independent components, `q·k` has variance `d_head`; scores grow like `√d_head`,
the softmax saturates and its gradient vanishes. Scaling keeps score variance ≈ 1. Verified
numerically in `test_scaling_keeps_score_variance_order_one`.

**3. How do you know your causal mask is right?**
Two tests: changing tokens after position t leaves logits ≤ t *bit-identical*
(`test_causal_mask_future_tokens_do_not_affect_past`), and the attention pattern is exactly zero above
the diagonal with rows summing to 1. Plus equivalence with PyTorch's fused `scaled_dot_product_attention(is_causal=True)`.

**4. Why store W_O per head instead of one big matrix?**
`concat(z_1..z_h) W_O = Σ_i z_i W_O^i`. The per-head form makes "each head writes its own vector into
the residual stream" literal, so zeroing `z_i` removes exactly head i — tested in
`test_zeroing_one_head_only_removes_that_head`. Computation still uses one reshaped matmul (fast).

**5. Pre-LN vs post-LN — which did you use and why?**
Pre-LN by default: an identity path from embedding to logits, stable gradients at init without
warm-up (Xiong et al., 2020), and an exactly additive residual stream, which direct logit
attribution and residual decomposition require. Post-LN is implemented and selectable; the
decomposition refuses to run on it (`test_post_ln_decomposition_refused`).

**6. What is the residual stream, mechanistically?**
A shared `d_model`-dimensional channel that every component reads from (through LN) and writes to
additively. The final stream is a sum of embeddings, every head's output, biases and MLP outputs —
`ActivationCache.decompose_resid` returns those parts and a test checks they sum to the stream.

**7. Teacher forcing vs generation?**
Training conditions on true previous tokens and, thanks to the mask, predicts all positions in one
pass; generation feeds sampled tokens back one at a time (`Transformer.generate`). Same function,
different conditioning; generation can compound errors that training never sees.

## B. Training and data

**8. Why can't you just use a fixed-length "random sequence then repeat it" task?**
A fixed offset lets the model copy from a fixed distance back — a positional shortcut that is not
induction. We vary segment length and prefix length per example (`InductionRepeat`). Even so,
**with learned absolute positions the model learned a smeared positional strategy** (35% accuracy,
no previous-token head; layer-0 heads looked 8–16 tokens back). Content matching is confirmed only
by the `shuffle` corruption: the ALiBi models follow the shuffled successors (logit difference
flips to −4.89).

**9. How did you debug the failed training run?**
(1) Measured accuracy by position within the copy: U-shaped, consistent with "guess an unused
earlier token". (2) Checked head scores across checkpoints: identical-looking, near-uniform; no
previous-token head at any checkpoint. (3) Hypothesis "init too small" → tested with a sweep → *not
supported* (larger init was worse). (4) Inspected patterns: layer 0 attended a window 8–16 back
(positional shortcut). (5) Hypothesis "absolute positions make far-offset lookback as cheap as
recency" → predicted ALiBi would fix it → 84% after 1500 steps vs 22–29% for alternatives. Along the
way found and fixed a real scale bug in sinusoidal positions (chance-level learning).

**10. How do you prevent train/test leakage on a synthetic generator?**
Separate seeds derived with `SeedSequence`; optional held-out (key,value)/(A,B) pairs never present in
training; and a leakage audit that regenerates the exact training stream from its seed and counts
overlaps (0 / 1024 for the principal runs).

**11. How is resume made exact?**
Checkpoints store optimizer, scheduler and the data generator's bit-generator state; an interrupt
mid-step rolls the RNG back to the last completed step. A test interrupts at step 14 and asserts the
resumed run's final weights are bit-identical to an uninterrupted run.

## C. Interpretability method

**12. What does an attention pattern tell you, and what doesn't it?**
Where a head looks on these inputs — not whether what it reads matters. Example from this project:
L0H0's previous-token score (0.17) equals its value at initialisation, yet ablating it drops
accuracy 0.91 → 0.03. And L1H3 (seed 0) has an induction score of 0.43 but its single-head ablation
costs only 0.27 nats.

**13. Zero vs mean vs resample ablation?**
Zero can be off-distribution; mean keeps a typical value; resample keeps a realistic one from another
input (but injects misleading information — e.g. the value-path effect grows from −0.6 to −2.2 nats
under resample). All three agree on the hand-wired ground truth; on trained models we report the method.

**14. Explain activation patching and its pitfalls.**
Corrupted run + one clean activation → `R = (patched − corrupt)/(clean − corrupt)`. Pitfalls: a
tiny clean–corrupt gap (we refuse to compute R), corruptions that change more than intended,
off-distribution noise corruptions, and reading sufficiency as necessity. In an AND-like circuit,
patching one input head gives R ≈ 0 although it is necessary — demonstrated on the hand-wired model
and observed for layer-0 heads in trained models.

**15. What's a path intervention and why did you need one?**
Ablation tells you a head matters, not *how*. Removing sender S's output from only receiver R's key
input (via per-head split inputs) distinguishes K-, Q- and V-composition. Caveat: it only removes
the *direct* route; in the MLP model, layer-0 heads feed the keys via MLP0, which the direct path
misses (−0.39 nats direct vs −3.1 via MLP0).

**16. How did you avoid double dipping?**
Candidate heads are selected on one set of examples (seed 101); predictions are evaluated on a
disjoint set (seed 202); mean-ablation statistics come from a third (seed 303). A hypothesis revised
after looking at a model (v2, v3) is confirmatory only on *other* seeds.

**17. How do your random-head controls work when there are only 8 heads?**
If the number of possible control sets is ≤ the requested samples, all are evaluated. Then a
permutation p-value cannot go below 1/(K+1); rather than pretend to significance, the framework
requires the candidate's whole CI to lie beyond *every* control effect and says so in the verdict.

**18. Why might a specificity prediction be vacuous?**
"Q-path has little effect" is trivially true if the model has no circuit at all (it was "supported"
on the learned-position model). The framework supports `requires: [ids]` to make such predictions
inconclusive unless their prerequisite was supported.

## D. Findings

**19. What is your strongest finding?**
In 2-layer attention-only ALiBi models (3 seeds, all ≥ 90% accuracy), induction is implemented by a
*group* of layer-1 heads whose keys are built from layer-0 outputs: group ablation −8.3 to −9.5 nats
(beyond all control sets), group patching restores 98%, removing layer-0 output from only the group's
keys costs −8.2 to −9.6 nats, from values ≤ 0.7. Confirmed on seeds not used to form the hypothesis.

**20. What did you get wrong?**
The pre-registered v1 hypothesis: there is no sharp previous-token head (P2 contradicted on all
seeds), and the composition is not purely K-side (P6 contradicted; the query path costs 1.5–4.6 nats
depending on the replacement method). I also over-interpreted my own follow-up in session 1 (the
"context sensitivity" measure turned out to be ambiguous — see Q26); context-dependent retrieval was
only established later by a preregistered test.

**21. Is that a phase transition?**
In all three seeds the induction score of the leading head, its causal importance and accuracy rise
together between steps 150 and 300 after a plateau at chance. That co-occurrence is consistent with
Olsson et al.'s phase-change description, but it is three runs of one toy setup at 50-step resolution.

**22. Does any of this transfer to GPT-scale models?**
Not by itself. It shows the methods work on known ground truth and produces model-specific
existence results. The positional scheme alone changed whether the mechanism formed; MLPs changed
how information was routed.

**23. Why is a 1-layer model a good control?**
It cannot compose heads. Under the hypothesis it should fail on variable-offset induction; it
reached 5.4% (chance 2.1%) versus ≥ 90% for 2 layers, same data and training.

## E. Engineering

**24. How do you guarantee hooks don't leak between experiments?**
`HookedRootModule.hooks()` removes handles in `finally`; a test raises inside a hook and asserts no
hook point still has handles.

**25. How would you scale this up?**
Swap the task generator for a tokenised corpus, keep the hook/intervention/hypothesis layers;
add batched path patching, gradient-based attribution patching for screening, and run hypotheses
over many seeds on GPU. The hypothesis files and the registry are model-agnostic.

## F. Session 2: audit and preregistered follow-up

**26. Your first session claimed the heads "match context". Was that justified?**
Not fully. The measurement (changing the token before the matched occurrence lowers log-prob) is also
predicted by interference in blurry keys, and on a hand-wired model that *does* match context it reads
exactly 0. I designed a test that separates the explanations — two earlier occurrences of the query
token differing only in their preceding token, recency counterbalanced — calibrated it on ground truth
(0.000 vs +1.985), preregistered it, and only then measured trained models: +1.9 to +2.3 logits, 3/3 seeds.

**27. How do you know the preference is learned, not architectural?**
A matched training manipulation: identical tokens, lengths, targets and architecture, differing only
in whether the predecessor of each queried token repeats. Consistent context gives +1.4 to +1.8;
context that never repeats gives −0.5 to −0.7 (unexpectedly reversed). Paired difference +1.9 to +2.3,
CIs far from 0.

**28. One preregistered prediction was inconclusive. What did you do?**
H3 (context enters via queries) used mean-removal, which also broke induction in half the models, so
the frozen rule returned inconclusive. I kept that verdict, built clean-to-corrupt path patching, ran it
as an explicitly exploratory analysis (query route in 6/6 models), then preregistered that design and
confirmed it on two models trained after the freeze (R_q 1.27 and 1.24; keys −0.15; values ≈ 0).

**29. How do you know old results still hold after you changed the code?**
`itl verify --all` re-evaluates every checkpoint against its recorded metrics (18/21 identical; two
pre-bug-fix runs cannot be regenerated, documented), and `itl rerun` re-executes analyses and diffs
every value (0 differences on the key hypothesis records).
