# Reference upgrades for Paper 3, revised after a coverage review

**Date:** 8 October 2026. **Status:** planning document, adopted by the author on 8 Oct.
Nothing here is frozen, run or claimed. `PREREGISTRATION.md` stays an unfrozen draft; the
proposal (`latex/paper3_proposal.tex`) is unchanged, because its text needs Junjie's agreement
(draft wording is in Appendix B).

**Sources.**
- **[BPWM]** Buchanan, Pai, Wang, Ma, *Principles and Practice of Deep Representation Learning*
  (arXiv 2606.06624v2).
- **[TMS]** Elhage et al., *Toy Models of Superposition* (arXiv 2209.10652).
- **[TKV]** Tian, Kong, Valiant, *Learning Populations of Parameters* (arXiv 1709.02707).
- The Paper 1 routing of these three references, TR01–TR30:
  `arXiv-v1-polished-extended-preprint/plan/onthefly/UPGRADE_ROUTING_THREE_REFERENCES_2026-10-06.md`,
  revised in that repo's `plan/MLST_OPEN_ITEMS.md` §9. That tracker says the Paper 3 items
  belong here; this file is now their home.
- A second agent's coverage review of the same references (8 Oct), reproduced verbatim in
  Appendix A, and the author's agreement with the verdicts below.

**Baseline checked.** This checkout at `5b8dfac` (branch `dev`): `latex/paper3_proposal.tex`
§3 "Two metrics" and §4 "Two claims", `PREREGISTRATION.md`, `TWO_CLAIM_REVISION_PLAN.md`
(M1–M5), `src/latent_monitor/{tier1,reference,support,task_metric,cumulant}.py`.
The VERITAS plan v3 (`UPGRADE_PLAN_COVARIANCE_AWARE_NONLINEAR_2026-10-04.md`, items C1–C12,
D1–D9) and the uncommitted `src/whitebox_subjects/` are **not** in this checkout; where an item
below may overlap them it says so.

---

## 1. Summary

| ID | What | Serves | Kind | When | Cost |
|---|---|---|---|---|---|
| **R1** | Co-occurrence and correlation-shift S family (TR15, extended) | S contract, Claim 1 and 2 cells | Tier-1 family + declared prediction | declare before `core` freeze; run after Phase B | 2–3 days |
| **R2** | First-order N-harm prediction from noise-only records (TR24, promoted to supporting) | Claim 2, supporting row | analysis | linear subject now (exact); trained subject after Phase C | ~1 day + `jac_input` |
| **R3** | Privileged-basis admissibility rule for hook statistics (replaces TR25 as a rule) | Claim 1, M4 | protocol rule + unit test | now (no run) | ½ day |
| **R4** | Abstention reason codes: unresolvable vs outside support (M2 refinement) | abstention, Claim 1 supporting | reporting rule | now (no run) | ½ day |

No item adds a claim, an arm or a subject. The shape rule from the Paper 1 tracker stands: one
main nonlinear subject, the linear control and one transfer arm; Phase B before anything new
runs.

---

## 2. Adopted items

### R1. Co-occurrence and correlation-shift S family (TR15, extended)

**Source.** TMS, "Correlated and Anticorrelated Features": with too little capacity, correlated
features collapse onto their principal component; local almost-orthogonal bases hold only on the
training sub-distribution. Rarely co-occurring features are packed with interference.

**Linear limit (for the text, not load-bearing; M5).** Paper 1's excited-support proposition is
the linear case: when the training latent covariance K is singular on the observable latent
subspace, the population projector recovers only B(range K), i.e. the principal direction of the
correlated factors. TMS's collapse is the same effect reached through capacity in a nonlinear
model, without K being singular.

**Arm (a), co-occurrence (pile-up).** Tier 1 already has `double_pulse` as a held-out S family
with a fixed second-pulse offset of +0.2 record fractions. Turn its offset into a severity axis:
Δt ∈ {0.05, 0.10, 0.20, 0.40} (values PENDING Phase B). Training sees one pulse per window, as now.
- *Declared prediction (direction only).* Amplitude harm K rises as Δt falls (the two pulses'
  whitened templates overlap more).
- *Mechanism statistic (supporting).* The Σ̂⁻¹ cosine between the two pulses' whitened templates,
  computed from the generator; harm should be monotone in it.

**Arm (b), correlation shift.** Tier-1 latents (amplitude, t0, τ_t) are drawn independently
today (`EventFamily.draw`). Add a copula option that couples amplitude and τ_t with correlation ρ.
- **Linear subject, now (CPU).** Sweep ρ_train ∈ {0, 0.5, 0.8, 0.95}; evaluate at ρ_test = 0 and
  −ρ_train. This is the linear-limit control.
- **Trained subject.** Needs one decision (D-R1 below): either the reference family itself carries
  ρ_ref = 0.8 (no extra training; the ρ_test = 0 and −0.8 cells become S cells), or arm (b)
  stays on the linear subject only. A second trained copy is not added (shape rule).
- *Declared prediction (direction only).* Amplitude harm grows with |ρ_ref − ρ_test| and with the
  Σ̂⁻¹ cosine between ∂g/∂amplitude and ∂g/∂τ_t. Magnitudes are descriptive.
- *Support labels.* With uniform marginals, the ρ_test = 0 window lies inside the joint support but
  in its low-density corners: a known-truth **in-support-but-rare** case, which tests the M2 support
  estimator. The ρ_test = −ρ_ref cell is the **outside-estimated-support** case.

**Where.** `tier1.py` (`EventFamily` copula option; `event_cells` severity axis), the family
table in `EXPERIMENT_DESIGN.md`, `PREREGISTRATION.md` §4. Declare before `core` is frozen.

### R2. First-order N-harm prediction (TR24, promoted to supporting)

**Source.** TMS, "Relationship to Adversarial Robustness": the end-to-end map of a feature picks
up off-diagonal interference terms, and vulnerability tracks features per dimension.
**Do not import the ">3×" figure.** It is a worst-case L2 attack in a sparse toy; covariance-only N
is a random perturbation.

**Content.** For a covariance-only N cell (Σ̂ held, realised Σ_r ≠ Σ_ref; signature (i)), let
J = ∂y/∂x̃ be the Jacobian of the declared outputs with respect to the Σ̂-whitened input at a
reference event, and Σ̃ = Σ̂^{-1/2} Σ Σ̂^{-1/2} for the realised noise of a cell. The predicted
excess output covariance relative to the reference cell is

    ΔC ≈ E_ref[ J (Σ̃_r − Σ̃_ref) Jᵀ ],      predicted K-variance excess  v_K = e_Kᵀ ΔC e_K.

- It is **exact** for the linear subject (affine readout) and first order for a trained subject;
  report the linearisation check.
- Σ̃_r and Σ̃_ref are estimated from random-trigger noise-only records, which are alarm-time
  information (`PREREGISTRATION.md` §3). The prediction uses no evaluation label.
- v_K is a variance and K is a mean absolute error, so only rank agreement is read.

**Interference split (the TMS reading).** Let r = e_Kᵀ J (one readout row) and
T = Σ̂^{-1/2} ∂g/∂θ the whitened physical tangents (simulator-known on Tier 1).
- *Cross-talk.* c = rT is how output K responds to each physical factor; the ideal is e_K. Its
  off-diagonal entries are the analogue of TMS's interference terms in (WᵀW)_K.
- *Noise geometry.* ‖r‖² = ‖r_in‖² + ‖r_out‖², split by projection onto span(T). The in-span part
  equals c(TᵀT)⁻¹cᵀ; for an unbiased readout (c = e_K) it is the Cramér–Rao floor
  [(TᵀT)⁻¹]_KK, which grows with tangent collinearity (report the variance-inflation factor
  [(TᵀT)⁻¹]_KK·[TᵀT]_KK). The out-of-span part is pure noise pickup that no physical factor needs.
- Report the cross-talk norm, the variance-inflation factor and the in-span / out-of-span shares
  per subject; these are geometry, fixed before any N cell is seen.

**Rows (supporting, never primary).**
1. Rank correlation across covariance-only N cells between v_K and realised K. Tier 1 has four
   such cell types today (`corr_up`, `corr_down`, `bandwidth_down`, `line_pickup`); with
   severities the count is set by Phase B. Below 10 cells the row is descriptive only.
2. Per cell, v_K split into its in-span, cross and out-of-span terms
   (r_in Δ r_inᵀ, 2 r_in Δ r_outᵀ, r_out Δ r_outᵀ with Δ = Σ̃_r − Σ̃_ref).

**Scope.** Covariance-only N families (`sigma_cov:*`, including held-out `line_pickup`). Structural
N (gain, channel loss, jitter) changes g, not Σ, and is out of scope. The Claim-2 primary estimand
and its committed scores are unchanged.

**Overlap.** VERITAS plan v3 WP-C has the worst-case direction (top generalised eigenvector of the
pullback). R2 is its average-case counterpart. If WP-C already computes ΔC, merge rather than
duplicate.

**Needs.** A `jac_input` method on `Subject` (composition of the encoder Jacobian and
`jac_output`; autograd on `TransformerSubject`).

### R3. Privileged-basis admissibility rule (replaces TR25 as a rule)

**Source.** TMS, "Privileged vs non-privileged bases": only an elementwise nonlinearity (or
sparsity) privileges the coordinate basis; elsewhere the basis is a rotation gauge.

**Rule (adds to the M4 invariance paragraph).** Each hook is declared privileged or
non-privileged.
- **Non-privileged hooks** (no elementwise nonlinearity before the next linear map: transformer
  residual stream and attention outputs, pooled stage means, the linear subject's z) admit only
  statistics that are invariant under O(d), or covariant with a data-defined chart:
  - Ledoit–Wolf Mahalanobis with a scaled-identity target (`support.py`): admissible.
  - Diagonal moments in a reference-PCA chart (`cumulant.project_diagonal_third`): admissible only
    when the retained PCs are separated by a declared eigengap. Report the gap; below it, use the
    Frobenius-norm statistic only.
  - Per-unit (neuron-basis) statistics: not admissible.
- **Privileged hooks** (post-activation MLP hidden units) additionally admit per-unit statistics.

**Unit test.** Rotate a non-privileged hook by a random orthogonal Q and compensate in the next
linear map, so that the subject's function is unchanged. Every admissible statistic must be
unchanged. This is stronger than the existing orthogonal control, which rotates without
preserving the function.

TR25's per-latent physical-tangent shares stay **Later** as a diagnostic.

### R4. Abstention reason codes (M2 refinement)

**Source.** Paper 1's pullback and excited-support propositions (linear case, cited without
load; M5). The second review attributed the distinction to BPWM §7.5; there it appears only as a
remark about the measurement null space, so Paper 1 and `kay1993` are the sources to cite.

**Rule.** Every abstention records why.
- **U, unresolvable:** the window's representation deviation lies mainly in `P_weak` of M_recon.
  The measurement cannot resolve it under Σ̂, and no retraining fixes it.
- **O, outside estimated support:** the support score exceeds its threshold while the deviation lies
  in `P_resolved`. Retraining on a wider support could fix it.
- **B, both.**

The abstention table reports counts per code. No threshold, no new endpoint. The `P_weak` energy
share uses the existing `reference.py` projectors; its cut-off is declared with the abstention
threshold.

---

## 3. Considered and not adopted

Numbering follows the review in Appendix A.

| # | Item | Verdict | Reason |
|---|---|---|---|
| 1 | BPWM §7.5 measurement self-consistency, pullback (7.5.8/7.5.17), Remarks 7.2/7.3 | **covered**; optional citation only | M1 is already fixed: M_recon = J̃_gᵀJ̃_g (Fisher, Σ̂-weighted) and M_task = J_yᵀW_yJ_y. (7.5.17) pulls back one score field through a Euclidean pseudo-inverse in order to learn a generative prior from y-only samples; it does not compare two pullbacks and ends with an open question. Remark 7.3 (Erlangen) adds nothing beyond the exact invariance groups of M4 |
| 2 | Low-dimensional support and measurement null space | **covered**; refined as R4 | M2 already separates resolvability (rank of M_recon) from estimated support (`support.py`). BPWM's nonlinear case uses the local Jacobian A(x_t), so it is no stronger than a local Fisher-rank statement |
| 3 | MCR², R_ε and ΔR_ε as monitor scores | **No** | Claim 1's baseline must be a strong established monitor; a log-det batch statistic would read as a weak one. ΔR_ε needs a class partition, and the K targets are regression outputs |
| 4 | §3.3 memorisation–generalisation phase transition | **No** | Concerns generative diffusion models. The TR23 caveat stands (the 1/3 threshold is not portable) |
| 5 | §6.3 continual learning and forgetting | **No** | The subjects are frozen; forgetting needs updates. An adaptation arm would add the complexity the shape rule removes |
| 6 | §6.2.1 closed-loop self-consistency ΔR_ε(Z, Ẑ) | **conditional** | Only for a subject with a decoder (the TR21 CRATE-MAE path). If that path is taken, a re-encoding residual ‖z − f(g(z))‖ in the subject's own metric is one hook statistic; no coding rate |
| 7 | §2.2.3 ICA and kurtosis | **covered** | TR18 (revised) pairs the SURE gate with a cumulant check on noise-only records; D7 adds the third cumulant |
| 8 | Remark 7.1, Bayesian vs end-to-end inference | optional citation | — |
| 9 | TMS adversarial robustness | **adopted as R2** | as average-case linear response, without the >3× figure |
| 10 | Feature dimensionality and interference sums | **merged into R2** | the interference split of R2 is the per-output version |
| 11 | Correlated features collapse to their principal component | **adopted as R1(b)** | — |
| 12 | Asymmetric superposition, inhibition motif | **No** | low value |
| 13 | Privileged basis as a methodological threshold | **adopted as R3** | — |
| 14 | TMS open questions; compressed-sensing bound | **No** | Open questions are not methods. The bound counts sparse features storable in m dimensions; the physical factors here are few and known |
| 15 | TKV population/histogram estimation as a support estimator | **No** (TR26/27 stay Later) | In simulated arms the training support is known by construction. On real data it becomes Gaussian deconvolution, where TKV's O(1/t) binomial rate does not transfer. `support.py` already estimates support and validates it on controls |
| 16 | TKV Proposition 1, 1/(4t), as a per-entity floor | **No** | Misreading: the bound is on W1 error of the *population histogram*, not on one entity. The per-event floor is the Cramér–Rao bound already in M_recon |
| 17 | TKV Theorem 2, multivariate populations | **No** | low value; same objection as #15 |

---

## 4. Status of every Paper 3 item from the Paper 1 routing (TR14–TR30)

| ID | Item | Status after 8 Oct |
|---|---|---|
| TR14 | Framing: simulated detectors are toy models with known features; a physics dictionary is the sparse-coding route | P3 text, unchanged (proposal §1, after Junjie agrees) |
| TR15 | Co-occurrence / pile-up S family | **extended as R1** |
| TR16 | Linear-first training-dynamics logging | only if a primary comparison needs it; rides on Phase C |
| TR17 | Frequency-resolved whitener (D5) | done in VERITAS code |
| TR18 | SURE gate: necessary, not sufficient; paired with a cumulant check | P3 protocol rule, unchanged |
| TR19 | Shot noise as its own stochastic quantity | P3 bookkeeping, unchanged |
| TR20 | GLS-FISTA N/S control | only if a primary comparison needs it |
| TR21 | White-box LUCiD subject | only if LUCiD is the transfer arm; replaces, never adds, a subject |
| TR22 | Tangent/normal excess | only if needed; after TR21 |
| TR23 | Memorisation ratio | only if needed; no 1/3 threshold |
| TR24 | Interference ↔ N-fragility | **promoted to R2 (supporting)** |
| TR25 | Privileged basis | **rule R3 now**; the per-latent diagnostic stays Later |
| TR26/27 | TKV populations | Later; support-estimation use declined (#15) |
| TR28/29 | Mixture-as-attention; CRATE code | reference only |
| TR30 | Not pursued | unchanged; MCR² reason now also covers the monitor use (#3) |

---

## 5. Decisions required

- **D-R1.** Does the reference event family carry a correlated (amplitude, τ_t) with ρ_ref = 0.8?
  Decide before the Phase C training run, since it changes the reference for every cell. If no,
  R1(b) runs on the linear subject only.
- **D10 conflict (open since 6 Oct).** `TODO.md` still says the hypergraph subject is "one of two
  nonlinear subjects, both reported". The shape rule says one. Resolve before Phase C.
- **Junjie.** R1–R4 change the family table and the mechanism text he has seen. Show him Appendix B
  before the proposal is edited.

## 6. Order

1. Now: R3 and R4 into `PREREGISTRATION.md` (done in draft, 8 Oct); their unit tests.
2. Now, CPU: R1(b) linear sweep and R2 on the linear subject, as development (non-citable) checks.
3. Phase B precision study, unchanged and first among runs.
4. Before `core` freeze: R1 families, severities and directional predictions declared; D-R1 decided.
5. Phase C: R2 on the trained subject, with its linearisation check.

---

## Appendix A. The coverage review (verbatim, 8 Oct)

```
未被 cover 的增强点
BPWM（教材）

§7.5 measurement self-consistency + 测量映射下的 pullback（Eq. 7.5.8/7.5.17，Remark 7.2 并行/分布式 sensing，Remark 7.3 Klein Erlangen 不变量）。这正是 P3 的 M1（J_recon vs J_y 两种 pullback）和 M4（不变量群）的现成推导：x̂ = x + s·A(x)†∇log p。TR22 只引 plan C6/C7，没引这个。强候选，未 cover。
§1.2.2 / §2.1 / §7.5.1 低维 support 与测量零空间不可辨识（"null space 的分量永远无法从观测恢复"，h(Sx)⊆Sy）。这是 M2（training support vs resolvability）与 abstention 的理论依据，比现有"Fisher rank 只能给局部可辨识性"更直接。强候选，未 cover。
§4.2.3 MCR² / 编码率 Rε(Z)、ΔRε(Z)。P3 需要一个强的 generic monitor score 和一个 task-sensitive score；Rε 是无标签 generic 统计量，ΔRε 是 label-light 的 task-sensitive 目标。目前只在 TR30 作为"P1 不做 MCR²"被否掉，P3 侧的用途没讨论。中，未 cover。
§3.3 lossy coding rate 的 ε 与 memorization→generalization 相变（Eq. 3.3.8、Fig. 3.12、模型/数据规模比相变）。TR23 只收了 memorization ratio（且 caveat 1/3 阈值），没收"相变 + 泛化分数"作为冻结模型脆性/support 诊断。中，部分未 cover。
§6.3 continuous learning / catastrophic forgetting / stability-plasticity（子空间保持约束）。给 P3 一个"冻结模型下 support shift 不可逆伤害"的机制与判据。中，未 cover。
§6.2.1 closed-loop transcription 的自洽残差 ΔRε(Z,Ẑ)——可作为 intermediate-hook 的表示自洽统计量。中，未 cover。
§2.2.3 ICA/kurtosis（FastICA，benign landscape）：非高斯方向的 N-signature；目前只在 TR30 被列为"detector 上不做 ICA"。低/中。
§7.1.1 / Remark 7.1 Bayesian vs end-to-end inference：支撑"resolvability ≠ training support"的引用框架。低。
TMS 9. "Relationship to Adversarial Robustness"：superposition 形成后脆弱性 >3×，且随 features-per-dimension（interference）增长。给 Claim 2 一个可预注册的 harm 机制（harm ∝ interference）。最接近的 TR24 只在 Later，且只是 hypothesis。强候选，未 cover。 10. Geometry 的 feature dimensionality D_i 与 interference Σ_{j≠i}(W_i·W_j)²：具体、可计算的 per-direction intermediate-hook 统计量。TR25（Later）只谈 physical-tangent shares。中，未 cover。 11. Collapse of correlated features → principal component / McGrath 的 "confused feature" W1≈W2≈½：容量/密度不足时不同物理特征合并成一个方向——一个具体的 S/harm signature。TR15 有 pile-up，但没收"collapse"机制。中，未 cover。 12. Asymmetric superposition + inhibition motif：可解释的 2-neuron 机制诊断。低。 13. Privileged basis 作为方法论门槛（决定哪些 hook 的 neuron-level 统计有意义）。TR25 只作 Later。中。 14. Open questions：superposition 的统计检验、feature importance/sparsity curve 估计、compressed-sensing bound m=Ω(−n(1−S)log(1−S))：直接对应 M2 的 support 估计与上限。中，未 cover。

TKV 15. population/histogram 估计 + empirical-Bayes/Stein shrinkage（Thm 1）：从每个实体很少的观测估计总体分布并作为先验——这是 M2 reference-latent density / training-support 估计器的现成方法。tracker 只把 TKV 放到 Later（TR26/27），并以"binomial O(1/t) 不迁移到高斯反卷积"否掉，但support 估计这一用途从未被考虑。中，未 cover。 16. Prop 1 下界 1/(4t)：per-entity resolvability 的信息论地板，可用来校准 abstention/阈值。中，未 cover。 17. Thm 2 多参数 population：多参数物理（amplitude+timing、多顶点/pile-up）的联合 support 估计。低。

建议
最值得补进 P3 的是 #1、#2（BPWM §7.5/低维 support，直接服务 M1/M2/M4） 和 #9（TMS adversarial/interference harm，直接服务 Claim 2）；其次是 #3 MCR²/编码率 和 #15 TKV 作为 support 估计器。其余多为 tracker 已 deliberate deferred（Later/No），需要在 VERITAS 侧重新判定是否升级，而不是 P1 仓库的 tracker。
```

## Appendix B. Draft proposal wording (for after Junjie agrees; not applied)

For §3, end of "Invariance, stated exactly":

```latex
Each hook is also declared privileged or not: only an elementwise
nonlinearity makes its coordinate basis meaningful~\cite{elhage2022superposition},
so non-privileged hooks (residual streams, pooled means, a linear latent) use
only statistics invariant under orthogonal maps, or diagonal statistics in a
reference principal chart above a declared eigengap.
```

For §3, end of "Training support is estimated, not inferred from rank":

```latex
Each abstention records whether the window's deviation lies mainly in the
weakly resolved directions of $M_{\mathrm{recon}}$, which no retraining can
recover, or outside estimated support within the resolved directions, which
wider training could cover.
```

For §5, Tier-1 families: one sentence naming the pile-up separation axis and the
correlation-shift family, with TMS's correlated-feature collapse as the mechanism and the
excited-support statement of the companion preprint as its linear case.

`references.bib` would need the `elhage2022superposition` entry (fields in the Paper 1
`references.bib`).
