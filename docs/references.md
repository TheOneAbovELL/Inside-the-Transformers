# Research references

Bibliographic details were checked on 2026-10-11 against primary sources (arXiv abstract pages,
proceedings sites, transformer-circuits.pub, ACL Anthology, LessWrong/Alignment Forum, the
TransformerLens repository). Entries marked **(partially verified)** could not be fully confirmed
from the primary document; treat those details with care.

Each entry says *how this project uses it*. A citation here means "this idea comes from / is
discussed in this source", not "this project reproduces the paper's result". Results of this
project are reported only in `docs/findings.md`, from saved artifacts.

## Architecture and training

1. **Vaswani, A., Shazeer, N., Parmar, N., et al. (2017).** *Attention Is All You Need.* NeurIPS 2017. arXiv:1706.03762.
   Scaled dot-product attention `softmax(QKᵀ/√d_k)V`; motivates `1/√d_k` because dot products "grow large in magnitude" for large `d_k` (variance `d_k` for unit-variance components). Sinusoidal positions. Post-LN placement. → `models/transformer.py`, `docs/mathematics.md` §2.
2. **Ba, J. L., Kiros, J. R., Hinton, G. E. (2016).** *Layer Normalization.* arXiv:1607.06450. → `LayerNorm`.
3. **Xiong, R., Yang, Y., He, D., et al. (2020).** *On Layer Normalization in the Transformer Architecture.* ICML 2020. arXiv:2002.04745.
   Post-LN has large gradients near the output at initialisation (hence warm-up); Pre-LN is well-behaved and can train without warm-up. → why `norm: pre` is the default.
4. **Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019).** *Language Models are Unsupervised Multitask Learners.* OpenAI technical report. **(partially verified: author list not read from the PDF)**
   GPT-2: LayerNorm moved to the input of each sub-block, extra LayerNorm after the final block, residual-layer weights scaled by `1/√N` (released code uses `1/√(2·n_layer)`). → our init of `W_O`, `W_out`.
5. **He, K., Zhang, X., Ren, S., Sun, J. (2016).** *Deep Residual Learning for Image Recognition.* CVPR 2016, pp. 770–778. arXiv:1512.03385.
6. **Hendrycks, D., Gimpel, K. (2016).** *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
7. **Kingma, D. P., Ba, J. (2015).** *Adam: A Method for Stochastic Optimization.* ICLR 2015. arXiv:1412.6980.
8. **Loshchilov, I., Hutter, F. (2019).** *Decoupled Weight Decay Regularization.* ICLR 2019. arXiv:1711.05101. → AdamW optimiser.
9. **Press, O., Smith, N. A., Lewis, M. (2022).** *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation.* ICLR 2022. arXiv:2108.12409.
   Sinusoidal embeddings extrapolate very weakly beyond training length; (footnote) learned absolute embeddings cannot represent positions beyond those trained. → limitation on length-OOD evaluation.

## Synthetic tasks

10. **Graves, A., Wayne, G., Danihelka, I. (2014).** *Neural Turing Machines.* arXiv:1410.5401. Copy and associative-recall tasks.
11. **Ba, J., Hinton, G. E., Mnih, V., Leibo, J. Z., Ionescu, C. (2016).** *Using Fast Weights to Attend to the Recent Past.* NeurIPS 2016. arXiv:1610.06258. Associative retrieval task (§4.1). → `assoc_recall`.
12. **Fu, D. Y., Dao, T., Saab, K. K., Thomas, A. W., Rudra, A., Ré, C. (2023).** *Hungry Hungry Hippos: Towards Language Modeling with State Space Models.* ICLR 2023. arXiv:2212.14052. Uses induction-head and associative-recall synthetic tasks as diagnostics. → `induction_pair`, `assoc_recall`.

## Mechanistic interpretability

13. **Elhage, N., Nanda, N., Olsson, C., et al. (2021).** *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread. https://transformer-circuits.pub/2021/framework/index.html
    Residual stream as a communication channel; QK and OV circuits; Q-/K-/V-composition; induction heads in 2-layer attention-only models. → the circuit hypothesis tested in `configs/hypotheses/induction_2l.yaml`, composition scores, OV copying scores.
14. **Olsson, C., Elhage, N., Nanda, N., et al. (2022).** *In-context Learning and Induction Heads.* Transformer Circuits Thread. arXiv:2209.11895.
    Induction heads form in a phase change early in training, visible as a bump in the training loss, co-occurring with in-context learning ability. → formation experiment (we test, not assume, this on our toy task).
15. **Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022).** *Locating and Editing Factual Associations in GPT.* NeurIPS 2022. arXiv:2202.05262.
    Causal tracing: restoring clean hidden states into a corrupted run (main contribution is the ROME editing method). → activation patching (denoising direction).
16. **Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., Shieber, S. (2020).** *Investigating Gender Bias in Language Models Using Causal Mediation Analysis.* NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html
    (Note: arXiv:2004.12265 is a different, expanded version with a different title and author list.) Causal mediation analysis of neurons and heads.
17. **Geiger, A., Lu, H., Icard, T., Potts, C. (2021).** *Causal Abstractions of Neural Networks.* NeurIPS 2021. arXiv:2106.02997. Interchange interventions.
18. **Wang, K. R., Variengien, A., Conmy, A., Shlegeris, B., Steinhardt, J. (2023).** *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small.* ICLR 2023. arXiv:2211.00593.
    Path patching; backup heads that compensate under ablation. → `interpretability/path.py`, redundancy caveats.
19. **Zhang, F., Nanda, N. (2024).** *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.* ICLR 2024. arXiv:2309.16042.
    Corruption method (Gaussian noising vs symmetric token replacement) and metric (probability vs logit difference vs KL) change conclusions; recommends symmetric token replacement and logit difference. → corruption design, metric choice in `interpretability/patching.py`.
20. **Heimersheim, S., Nanda, N. (2024).** *How to use and interpret activation patching.* arXiv:2404.15255.
    Denoising (clean→corrupt) tests sufficiency; noising (corrupt→clean) tests necessity; the two are usually not complements.
21. **Chan, L., Garriga-Alonso, A., Goldowsky-Dill, N., Greenblatt, R., Nitishinskaya, J., Radhakrishnan, A., Shlegeris, B., Thomas, N. (2022).** *Causal Scrubbing: a method for rigorously testing interpretability hypotheses.* Redwood Research, Alignment Forum / LessWrong. https://www.lesswrong.com/posts/JvZhhzycHu2Yd57RN **(partially verified: one author's full name inferred from a handle)**
22. **Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., Garriga-Alonso, A. (2023).** *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023. arXiv:2304.14997.
23. **McGrath, T., Rahtz, M., Kramár, J., Mikulik, V., Legg, S. (2023).** *The Hydra Effect: Emergent Self-repair in Language Model Computations.* arXiv:2307.15771.
    Ablating one *attention* layer can cause another layer to compensate; late MLP layers can down-regulate the maximum-likelihood token. → why single-component ablations can underestimate importance.
24. **Makelov, A., Lange, G., Geiger, A., Nanda, N. (2024).** *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* ICLR 2024. arXiv:2311.17030.
25. **nostalgebraist (2020).** *interpreting GPT: the logit lens.* LessWrong. https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens → `logit_lens`.
26. **Bricken, T., Templeton, A., Batson, J., et al. (2023).** *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits Thread. Sparse autoencoders on a one-layer transformer's MLP activations. → future-work extension only.
27. **Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. (2023).** *Progress measures for grokking via mechanistic interpretability.* ICLR 2023. arXiv:2301.05217.

## Attention as explanation

28. **Jain, S., Wallace, B. C. (2019).** *Attention is not Explanation.* NAACL 2019. arXiv:1902.10186. Attention weights are frequently uncorrelated with gradient-based importance; different attention distributions can yield the same prediction.
29. **Wiegreffe, S., Pinter, Y. (2019).** *Attention is not not Explanation.* EMNLP-IJCNLP 2019. arXiv:1908.04626. Whether attention is "explanation" depends on the definition; prior work does not disprove its usefulness.
30. **Belinkov, Y. (2022).** *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics 48(1):207–219. arXiv:2102.12452. Decodability ≠ use.

## Statistics and software

31. **Efron, B., Tibshirani, R. J. (1993).** *An Introduction to the Bootstrap.* Chapman & Hall. → percentile bootstrap CIs in `interpretability/stats.py`.
32. **Nanda, N., Bloom, J. (2022).** *TransformerLens.* https://github.com/TransformerLensOrg/TransformerLens
    Our hook names (`blocks.{l}.attn.hook_z`, `hook_pattern`, `blocks.{l}.hook_resid_pre`, ...) deliberately follow TransformerLens's HookedTransformer convention so readers can transfer intuition. **This project does not depend on TransformerLens**; the implementation is independent.
