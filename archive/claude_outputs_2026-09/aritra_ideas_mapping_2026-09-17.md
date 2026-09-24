# Aritra's two papers against the four papers — what transfers, where, and how

*17 Sep 2026. Sources read in full: Bal, Binder, Klute, Maier, Spannowsky, "QUIVER: Quantum-Informed Views for Enhanced Representations in Large ML Models", arXiv:2606.02785 (1 Jun 2026); Bal, Klute, Maier, Spannowsky, "From Information Geometry to Jet Substructure: A Triality of Cumulant Tensors, Energy Correlators, and Hypergraphs", arXiv:2605.03063v2 (7 May 2026). Repo state used: VERITAS @ 778eb83 (two-claim revision, 16 Sep), agent-checkpoint-intervention @ 493682f (instrument-first spine, 15 Sep).*


**Decision, 17 Sep 2026 (later the same day):** the agent paper will *not* be upgraded with these two papers (§5 is kept as a record of the assessment only). The Paper 1 items of §4 are worked out in the manuscript repo at `plan/ARITRA_FISHER_CUMULANT_NOTE_2026-09-17.md`.
---

## 0. Short verdict

The honest ranking of who benefits is **Paper 3 (ORACLE/VERITAS) ≫ Paper 1 (MLST) > agent paper > governance-harness upgrade (nothing)**.

The reason is structural. Both of Aritra's papers are about the **Fisher information of a model, and what lies just beyond it**. Paper 1 and Paper 3 are built on exactly that object: Paper 1's Bridge Theorem is the Gaussian Fisher metric Σ⁻¹, and ORACLE's `M_recon = J̃_gᵀ J̃_g` is literally "the Gaussian Fisher information under the assumed Σ̂" (`docs/PREREGISTRATION.md` §2, `paper3_proposal.tex` l. 239). The triality paper tells you precisely what your Gaussian framework is a truncation *of*, and gives you the first correction term with a physical reading. That is a direct fit.

The agent paper is about a horizon-one decision process, causal identification and replay. Neither of Aritra's papers touches agents, causality or sequential decisions. There are two defensible hooks (one concrete, one a remark), but **they do not address the depth problem you named on 15 Sep** (causal inference alone feels thin; you want the Bellman/RL framing to carry rigour). Information geometry is not the missing rigour there; the occupancy-shift bound in `notes/mdp_framing/` is. I would not let these two papers pull the agent paper's revision.

---

## 1. What the two papers actually contribute, stripped of their domain dressing

Read past the quantum circuits and the jets. Five ideas survive the translation; every mapping below refers to them by number.

**I1 — The KL expansion is a tower of Fisher tensors, exactly equal to connected cumulants in natural exponential-family coordinates.** (Triality, Thm 1, Cor 2.) For p_θ(x) = h(x) exp[θ·φ(x) − ψ(θ)], the n-th derivative of ψ is simultaneously the n-th KL coefficient and the n-th connected cumulant of the sufficient statistics: I⁽²⁾ = Cov(φ), I⁽³⁾ = ⟨s s s⟩, and D_KL(p_θ‖p_{θ+δ}) = Σ_{n≥2} I⁽ⁿ⁾δⁿ/n!. The quadratic truncation *is* the Fisher-metric approximation; the cubic term is the first controlled non-Gaussian correction. **Caveat they state carefully (Remark 1, App. A):** the identity is exact only in the natural chart. In a generic parametrisation the cubic KL coefficient picks up Hessian terms ⟨s_a ∂_bc ℓ⟩ and is *not* a cumulant. A Gaussian model with sufficient statistics (x, xxᵀ) is an exponential family, so the chart is natural for your residuals; a learned latent z is not automatically one.

**I2 — Whitening by I⁽²⁾ isolates irreducible higher-order structure.** (Triality, Eq. 6.6, Fig. 3–4.) Set z = I⁽²⁾^{-1/2}(φ − ⟨φ⟩); then Ĩ⁽²⁾ = 1 and whatever survives in Ĩ⁽³⁾ is pure third-order information "decoupled from second-order content". They use it to show a pairwise Fisher graph saturated at ±1 while the hypergraph still ranks nodes. This is your Whitening Lemma, continued one order further.

**I3 — Fisher hypergraphs and higher-order Laplacians with weights fixed from measured cumulants.** (Triality, §4–5, §6.5.) Vertices label observables; the all-distinct entries of I⁽ⁿ⁾ are signed n-hyperedge weights (use |w| or w², carry the sign as an attribute); L⁽ʳ⁾ is the normalised hypergraph Laplacian, L_multi = Σ α_r L⁽ʳ⁾. In §6.5 they freeze the order-2 and order-3 propagators from the reference sample and train only a ~500-parameter readout; the hypergraph beats the graph at 3000–9000 labelled events (0.8258±0.0029 vs 0.8184±0.0039 AUC at 6000; ~1.5σ over five splits — modest, and they say so). The point is not "hypergraphs are better" but "an architecture whose topology and weights are physics objects, with a tiny trainable head".

**I4 — Task-aligned cubic scores and a compression criterion beyond pairwise.** (Triality, §6.3–6.4.) Given a task direction u (a class-mean difference) or a deformation direction Δ, score observables by s⁽²⁾_a = u_a I⁽²⁾_ab u_b and s⁽³⁾_a = u_a I⁽³⁾_abc u_b u_c; grow a basis greedily by redundancy-suppressed quadratic gain plus aligned cubic completion; evaluate a retained basis by the retained local divergence fraction R⁽³⁾(S) (Eq. 6.34) and aligned triplet coverage C₃(S). At 12 of 33 observables: R⁽³⁾ = 0.937 vs 0.871 for the pairwise selector.

**I5 — A model's own Fisher information as a complementary "view", injected without adding capacity.** (QUIVER.) The QFIM of an auxiliary model trained on the same task is treated as a second modality: appended as tokens to Particle Transformer (+7 % params) or, more cleanly, as a **residual multiplicative gate x̃ = (1 + α·Θ(Q)) x with α initialised to zero**, so the augmented and baseline networks are identical at step 0 (+0.27 % params). Evidence that the view carries *information rather than capacity* is a paired-seed design: identical splits and init RNG, 10 seeds, paired t-test (t₉ = 5.78). The quantum part is irrelevant to you; the abstraction — "the Fisher geometry of a model is a feature, and here is how to add it so that any gain is attributable" — is not.

Things I would explicitly **not** import: variational quantum circuits and the QFIM specifically (no physics reason in your detectors, and simulation caps them at 10 qubits); the energy-correlator observables; the jet benchmarks.

---

## 2. The mapping at a glance

| Idea | Paper 1 (MLST) | Paper 3 / ORACLE (VERITAS) | Agent paper | Gov-harness ICLR |
|---|---|---|---|---|
| I1 KL tower / cubic Fisher tensor | **Yes — one Remark + one zero-cost diagnostic**: names exactly what the Gaussian framework truncates | **Yes — supporting statistic**: the declared "non-Gaussian-tail cells" get a principled object | Remark only (trajectory audit) | No |
| I2 whitening isolates I⁽³⁾ | **Yes — a corollary of the Whitening Lemma**, no new class | **Yes — the N_cov vs N_struct signature** (hypothesis, Tier 1 testable) | No | No |
| I3 Fisher hypergraph / L⁽³⁾ propagators, frozen | Outlook paragraph only (noise-Laplacian PE → noise hypergraph); rule "nothing new built for Paper 1" holds | **Yes — the readable non-linear NFPA counterpart you asked for on 11 Sep** | No | No |
| I4 task-aligned cubic score, R⁽³⁾ compression | No | **Yes — cubic companion to the Claim-2 task length; hook-selection criterion for Claim 1** | No | No |
| I5 model-Fisher as a registered view; zero-init gate; paired-seed attribution | Methodology precedent for the ≥10-seed nonlinear arm | **Yes — methodological precedent for Claim 1's capacity-matched control; gate pattern for the structured AE** | **Yes — one concrete richer representation Z′ for the Thm 6 audit** | No |

---

## 3. Paper 3 / ORACLE — the main beneficiary

### 3.1 Why the fit is exact

ORACLE already commits to three objects that sit on the quadratic rung of I1: `M_recon` (the Gaussian Fisher information under Σ̂), the Mahalanobis reference distances in the reference-cell metric, and the Claim-2 task length ‖z − z̄‖_{M_task} with M_task = J_yᵀ W_y J_y. The pre-registration then says, in §3: *"non-Gaussian-tail cells — the whitened quadratic form is a diagnostic, not a likelihood."* And the Tier-1 simulator already has a non-Gaussian family (`EXPERIMENT_DESIGN.md` l. 1014: "non-Gaussian sparse bursts"). So the proposal has a hole shaped exactly like I⁽³⁾: you currently have no object that says *how far* a cell is from the Gaussian rung, only a sentence that says the quadratic form stops being a likelihood there.

### 3.2 Integration A — the whitened third cumulant as the N_cov / N_struct separator (I1 + I2)

*What:* On the whitened residual r̃ = Σ̂^{-1/2}(x − g(z)) (a natural chart: Gaussian reference, sufficient statistics known), compute the connected third cumulant tensor Ĩ⁽³⁾ per window, summarised as a scalar (‖Ĩ⁽³⁾‖ or, better, their Table-1 quantity: the ratio of the quadratic to quadratic-plus-cubic KL truncation error along the window's displacement direction).

*Why it is more than a kurtosis check:* the signature table in `EXPERIMENT_DESIGN.md` §III distinguishes N_cov (covariance change) from N_struct (bursts, glitches, line pickup). A pure covariance shift changes I⁽²⁾ and, **after re-whitening with the realised covariance**, leaves Ĩ⁽³⁾ ≈ 0; a structural shift does not. That is a conditional signature of the kind the protocol already registers, and it is testable on Tier 1 with paired replay where both Σ̂ and the realised Σ are known. State it as a hypothesis; the proposal's own rule (13 match / 1 documented / 0 mismatch table) is the right place for it to land.

*Where in the protocol:* as a **supporting analysis**, not a third claim. If you want it inside Claim 1, the natural home is the intermediate-hook family — the hooks already carry "reference distances of pooled mean and second moment" per channel/token; a pooled **third** moment is the obvious next entry. Be aware of the asymmetry: putting it in `generic_rich` raises the bar Claim 1 must clear; putting it in `full_intermediate` is what Claim 1 is *about*. Decide before freeze; the pre-registration is unfrozen, so this week is the moment, and §8 of that file is where the decision is recorded.

*Cost:* numpy only; a third-cumulant estimator on a d-dimensional whitened residual is O(n·d³) per window and d is small on Tier 1. Two afternoons including tests. The estimator variance at n_ref ≲ 100 will be worse than for distances (you already measured that ordering instability at that n); report it in the Phase-B precision study.

### 3.3 Integration B — the cubic companion to the Claim-2 task score (I4)

Claim 2's task-sensitive score is quadratic: the length of the window's deviation in M_task. I4 gives the cubic term along the same direction: S⁽³⁾ = Δ_a Δ_b Δ_c I⁽³⁾_abc with Δ the window's displacement in the reference-cell chart. The reading is "task-aligned skewness": a cell whose harm is carried by an asymmetric, non-Gaussian excursion in the task direction. This should enter only as the **supporting intermediate score** the pre-registration already reserves ("the supporting intermediate score" in §1), never as a change to the primary estimand ΔAUROC_harm.

The second use of I4 is more valuable and cheaper: **hook selection**. Claim 1 has hook-drop controls but no principled rule for which hooks to keep. The greedy selector of Triality §6.4 (redundancy-suppressed quadratic gain + aligned cubic completion, evaluated by retained R⁽³⁾) is a registered, label-free way to choose the intermediate hook set from `reference_fit` data alone — it never sees evaluation labels, so it respects the information contract of §3. That replaces "we hooked these layers" with "we retained the hooks that keep R⁽³⁾ ≥ x of the reference-cell local divergence".

### 3.4 Integration C — the readable non-linear counterpart to NFPA (I3 + I5)

On 11 Sep you asked for a non-linear architecture whose latent stays readable as physical representations, and your stated worry about the rung-4 transformer was that diagnostics, representation choice and fine-tuning would eat the time. Triality §6.5 is the smallest architecture that answers this, and it comes with its own physics semantics:

- Vertices = detector channels (TES/QP channels on arm B, band-frames on TIDMAD).
- L⁽²⁾ built from the measured noise covariance — this **is** Paper 1's noise-Laplacian PE, which makes the two papers' vocabulary coincide for free.
- L⁽³⁾ built from the measured third cumulant of the noise (or of the whitened signal-plus-noise), sign carried as an edge attribute.
- Propagators P₂, P₃ frozen from the reference cell; only a small readout trained (their ~500 params per order).

The latent stays readable because the propagators are fixed physics objects; a monitor can then ask the ORACLE question ("what does the frozen model report under N vs S?") on an object whose every weight has a provenance. This is a Tier-2/Tier-3 subject, alongside the transformer, not instead of it — but it is the one you could diagnose in a week rather than a quarter. If you do add a trainable correction on top of the linear class, use the **QUIVER zero-initialised residual gate** (I5): at α = 0 the model is exactly NFPA, so the whole Paper-1 theory applies at initialisation and any departure is measurable as α grows. That is the cleanest possible version of "does the metric carry into a non-linear reconstruction".

### 3.5 Methodology to cite

QUIVER's paired-seed, identical-split, zero-init design is the same argument form as Claim 1's capacity-matched (`generic_rich_matched`) control: any gain must be attributable to information, not parameters. Cite it as precedent in the "designed controls" text; it costs one sentence and pre-empts a referee's "you just added capacity". Your ≥10-seed plan for the nonlinear arm matches their N = 10.

### 3.6 What not to do

Do not put a VQC anywhere near ORACLE. Do not apply I1 to a learned latent z and call the cubic term a cumulant — the exactness needs the natural chart (their App. A). Compute cumulants on whitened residuals or on channel data, where the Gaussian reference makes the chart natural; on z, report the cubic KL coefficient with its Hessian terms or do not report it.

---

## 4. Paper 1 (MLST) — a Remark, a diagnostic, an outlook; nothing built

Paper 1 is in its editorial/freeze phase (T18–T23 done, MLST compression deferred, arXiv v1 after the 29 Sep R/PS decision, rule: no new model class). So the integration is text plus one zero-cost check.

**Remark (I1, after the Bridge Theorem or in the Whitening Lemma section).** The unified Gaussian maximum-likelihood framework is the quadratic truncation of the KL expansion in the natural coordinates of the Gaussian family; Σ⁻¹ is I⁽²⁾; the first term the framework cannot see is I⁽³⁾, the connected third cumulant of the whitened residual. This gives the Bridge Theorem an explicit **domain of validity** stated as a measurable quantity rather than a modelling assumption: the framework is adequate exactly where Ĩ⁽³⁾ of the whitened residual is negligible relative to the quadratic term. That is a better limitations sentence than "we assume Gaussian noise", and it is one paragraph.

**Zero-cost diagnostic (I2, T23-style).** For each arm you already have (QP simulator, CRESST, TIDMAD held-out), compute the whitened residual's cubic KL truncation-error ratio along the planted-subspace direction (their Table 1 quantity). Report one number per arm in the claim table. Prediction worth registering: the TIDMAD arm, where recovery failed and p/n = 187, will show the largest non-Gaussian residual; if it does not, that strengthens the sample-covariance-error explanation you already prefer. Either outcome is informative and the computation is minutes.

**Outlook (I3, one sentence).** The noise-Laplacian PE is the r = 2 member of a hypergraph-Laplacian family whose r = 3 member is built from the noise's third cumulant; the deferred PE-uniqueness question (P1-E8) has a natural higher-order form there. Say it, cite Triality, do not run it — that work belongs in ORACLE (§3.4 above), which is consistent with the "same levers, opposite side of the freeze" boundary you will explain to Junjie.

---

## 5. The agent paper — what these papers can and cannot do for it

You asked this question first, so here is the plain answer. **Neither paper strengthens the agent paper's core.** The paper's contribution is a measurement contract (applicability recorded before treatment, isolated restored state, per-layer restoration audit) and what finite replay can decide under an unaudited discrepancy p; its theorems are conditional-mean identities on a horizon-one MDP. Fisher tensors, cumulants and hypergraphs have nothing to say about that, and a referee at ICML would read citations to them as decoration. Two hooks are real, in decreasing order of value.

### 5.1 A concrete registered richer representation Z′ for the Thm 6 audit (I5)

Thm 6 (`thm:refinement`) says Ref_j ≥ V^{*Z′}_j − V^{*Z}_j ≥ 0 for any richer registered Z′, and `02_contract.tex` says the representation audit "returns a number rather than a verdict" and is the estimand of the representation experiment. That needs an actual Z′, chosen before data. QUIVER's thesis — the Fisher geometry of a model is a genuinely complementary view of the same example — gives you one that is not ad hoc: **the agent policy's own local Fisher information at the checkpoint**, i.e. the empirical Fisher (diagonal, or block over the last layers) of the LLM's token log-likelihood evaluated on the checkpoint context. Your E0 candidates (Qwen3-8B, DeepSeek) are open-weight, so this is computable; it is pre-treatment (a function of H_i only), so it satisfies the S_i exclusions; and it is exactly the kind of thing the registered risk score does *not* carry. It also fits the primary positive condition, which is a randomized risk-blind feature-engineering × configuration interaction — "with vs without the Fisher view" is a legitimate feature-engineering arm. Register it as one Z′, not as the method. If the audit returns ≈ 0, that is a result about the risk score's sufficiency, which is your Thm 5 question from the other side.

Cost: a forward-and-backward pass per checkpoint on the open-weight model; no new theory; one paragraph in `08_study_logic.tex` and one row in the protocol.

### 5.2 A remark on the trajectory audit (I1)

`notes/mdp_framing/` L0–L3 ladder: the one-step certificate transfers to repeated deployment only up to an occupancy-shift term, and the single unobservable ingredient is the span of the one-step advantage over post-intervention states. If you write the controller's deviation from ρ as a KL between the two checkpoint-kernel families, Cor 2 of Triality says that KL is Fisher-quadratic to leading order with the cubic coefficient given by the third cumulant of the score — exact if the kernel family is written in exponential coordinates, otherwise with Hessian terms. That connects your bound to the TRPO-style monotonic-improvement bounds (max-KL × advantage span), which RL readers will recognise, and it names *which* order of the expansion the depth-one instrument can and cannot measure. This is one remark in `sec:bellman`; it does not remove the unobservable ingredient and you should not write it as if it did.

### 5.3 What I would skip

The α = 0 residual gate as an "architectural analogue of the pessimistic gated rule" is cute and empty. The hypergraph machinery has no object in the agent paper to attach to.

### 5.4 Bottom line for the 15 Sep worry

The depth you want comes from finishing the occupancy-shift bound in `notes/mdp_framing/` and promoting it into `06a_theory.tex` as a fifth result, not from information geometry. If you want an outside source of rigour for that, it is the safe-policy-improvement and offline-RL pessimism literature you already cite, plus Kakade–Langford / TRPO for the KL form. Aritra's papers can supply §5.1 and a footnote; they should not steer the revision.

---

## 6. Governance-harness ICLR upgrade (deadline 24 Sep)

No. One week out and no fit; the upgrade's added value is causality + RL, which these papers do not touch.

---

## 7. Suggested order of work

1. **This week (VERITAS, before freeze):** decide whether the pooled third-moment hook enters `full_intermediate` (§3.2); record it in `PREREGISTRATION.md` §8. Add the whitened-Ĩ⁽³⁾ N_cov/N_struct signature to the `EXPERIMENT_DESIGN.md` §III conditional-signature table as a hypothesis with a Tier-1 test.
2. **Next (VERITAS, Tier 1):** implement the cumulant estimator in `src/latent_monitor/statistics.py` beside the existing moment statistics; run it in the dev smoke; add the R⁽³⁾ hook selector to `protocol/` as a reference-fit-only step.
3. **Paper 1, editorial pass:** the Remark, the one-number-per-arm diagnostic, the outlook sentence. Nothing else.
4. **Agent paper:** register the policy-Fisher Z′ as one feature-engineering arm; one remark in `sec:bellman`. Then go back to the occupancy-shift bound.
5. **ORACLE Tier 2/3, later:** the frozen-propagator hypergraph subject (§3.4) as the readable non-linear arm, with the zero-init gate if a trainable correction is wanted.

Aritra is at ETP. The cumulant-tensor and hypergraph-Laplacian code behind Triality §6 almost certainly exists; asking for it, and for a read of §3.2–3.4 here, is cheaper than reimplementing and is a natural co-authorship on the ORACLE non-linear arm if it goes that way.

---

## 8. Citation details

- Bal A., Binder M., Klute M., Maier B., Spannowsky M. *QUIVER: QUantum-Informed Views for Enhanced Representations in Large Machine Learning Models.* arXiv:2606.02785 [cs.LG], 1 Jun 2026. Key numbers: ParT kin. 5M: AUC 0.97832±0.00004 → 0.98070±0.00003, 1/ε_B 176 → 240; QDimeNet++ MAE 72.42±1.52 → 67.92±1.98 meV, ΔMAE 4.50±2.46, t₉ = 5.78, +0.27 % params.
- Bal A., Klute M., Maier B., Spannowsky M. *From Information Geometry to Jet Substructure: A Triality of Cumulant Tensors, Energy Correlators, and Hypergraphs.* arXiv:2605.03063v2 [hep-ph], 7 May 2026. Key objects: Thm 1 (Fisher tensors as cumulants), Cor 2 (exact KL expansion), Eq. 6.6 (whitened basis), Eq. 4.3–4.5 (hypergraph Laplacian, multi-order layer), Eq. 6.27/6.32/6.34 (aligned score, triplet coverage, retained R⁽³⁾), App. A (generic-coordinate caveat). Key numbers: ~30× KL truncation-error reduction at t = 0.15; R⁽³⁾ 0.937 vs 0.871 at 12/33 observables; AUC 0.8258±0.0029 vs 0.8184±0.0039 at 6000 events.
