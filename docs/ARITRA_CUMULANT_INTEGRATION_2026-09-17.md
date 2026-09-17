# ORACLE — integrating the Fisher-cumulant tower into the two-claim study

*17 September 2026. Design note for the two-claim proposal and its protocol. Sources: Bal, Klute, Maier, Spannowsky, "From Information Geometry to Jet Substructure: A Triality of Cumulant Tensors, Energy Correlators, and Hypergraphs", arXiv:2605.03063v2 (Thm 1, Cor 2, Remark 1, App. A, Eq. 6.6, Eqs. 4.3–4.5, §6.4–6.5); Bal, Binder, Klute, Maier, Spannowsky, "QUIVER", arXiv:2606.02785 (§3.2.1, Eq. 8, §4.2). Repository state: VERITAS @ 778eb83, `dev`, worktree dirty; the design is `docs/EXPERIMENT_DESIGN.md` (two-claim, Part V), the protocol draft `docs/PREREGISTRATION.md` (unfrozen). The companion assessment across all four papers is `Claude outputs/aritra_ideas_mapping_2026-09-17.md`; the agent prompt that implements this note is `docs/IMPLEMENTATION_PROMPT_CUMULANT.md`.*

*Status: a proposal. Nothing here is frozen, nothing changes the two claims or their primary estimands, and no result exists.*

---

## 1. Why this fits, in one paragraph

The proposal's mechanism section defines two pullback metrics on the monitored representation and says of the first: "the Fisher information of r is I(r) = J_gᵀ Σ̂⁻¹ J_g — the standard Gaussian-model result." The pre-registration then declares, for non-Gaussian-tail cells, that "the whitened quadratic form is a diagnostic, not a likelihood" (`PREREGISTRATION.md` §3), and the Tier-1 simulator already generates a non-Gaussian family ("non-Gaussian sparse bursts", `EXPERIMENT_DESIGN.md` §III.7). Bal et al. supply the object that sentence is missing. In the natural coordinates of an exponential family the KL divergence between nearby members is D_KL = Σ_{n≥2} I⁽ⁿ⁾δⁿ/n!, the n-th Fisher tensor I⁽ⁿ⁾ = ∂ⁿψ is the n-th connected cumulant of the sufficient statistics, and the quadratic truncation — M_recon — is exact exactly when the higher cumulants vanish. So "how far is this cell from the Gaussian rung" is a number, not a caveat: the first non-vanishing connected cumulant of the whitened residual in the reference chart. Everything below is that number used in the three places the protocol already has slots for, plus one architecture decision the author asked for on 11 September.

## 2. The four integrations

### 2.1 Supporting statistic: the whitened-residual cumulant as the N_cov / N_struct separator

**Object.** For a window, take the whitened residual r̃ = Σ̂^{-1/2}(x − g(z)) (`protocol.features.whitened_residual`), and compute along the reference chart the connected third cumulant (real time-domain traces; the arms are real) and, as the circular-convention companion, the fourth. Report per window (a) the per-coordinate third central moment in the reference PCA basis (n_pcs coordinates) and (b) the Frobenius norm of the whitened third-cumulant tensor after subtracting the reference-cell tensor — a scalar. Bal et al.'s Table-1 quantity (the ratio of the quadratic to the quadratic-plus-cubic KL truncation error along the window's displacement direction) is the interpretable version of (b) and should be reported beside it in the development table.

**Chart caveat (Bal et al. Remark 1, App. A).** The cumulant reading is exact only in the natural chart. The whitened residual under the Gaussian reference is such a chart; the pooled latent z on the *linear* subject is too (z is linear in x); z on a nonlinear subject is not, and there the same statistic is a *deviation from the reference cell's own third cumulant*, reported as such. This is consistent with §III.2.3: prove it on the linear subject, then see whether it survives on the transformer.

**Signature (conditional; a hypothesis for the §III.1 table).** A covariance-type N (Σ̂ ≠ Σ: correlation, bandwidth, line, alias fold) changes I⁽²⁾ and, after re-whitening with the realised Σ (an evaluation-only quantity), leaves the whitened third cumulant at reference; a structural N (sparse bursts, glitches, gain drift with clipping) does not. In-span S (double pulse, oscillation) moves z but leaves the residual cumulants at reference; out-of-support S (glitch family) moves them. This row is testable on Tier 1 under paired replay where both Σ̂ and the realised Σ are known, with the same 13/1/0-style match table as the 6 Sep signatures. It is a supporting analysis. It is not a third claim.

**Where in the arms — the symmetry rule.** The arm definitions (`protocol/arms.py`) rest on one principle: `generic_rich` receives "the same reference-distance transforms the intermediate arm enjoys, applied to generic quantities." A pooled third moment is a new *transform*, so it must be added on both sides or neither: `im_{hook}_third_maha` for the channel and token hooks (beside `im_{hook}_second_maha`, with its own `hook_third_nulls`), **and** the residual cumulant scalars in `generic_rich`, **and** the quadratic capacity control re-truncated to the new intermediate count. Adding it only to the intermediate hooks would let Claim 1 win by a transform, which the refutation clause of Claim 1 already names as a failure. The pre-registration is unfrozen; this is a §8 decision to record now, before freeze.

### 2.2 Claim-2 supporting score: the cubic companion to the task length

The task-sensitive score is quadratic: ‖z − z̄‖_{M_task}. Bal et al.'s aligned score (Eq. 6.27) gives the cubic term along the same displacement: S⁽³⁾ = Δ_a Δ_b Δ_c Î⁽³⁾_abc with Δ the window's displacement in the reference chart and Î⁽³⁾ the third cumulant estimated on `reference_fit` z. Its reading is task-aligned skewness — harm carried by an asymmetric excursion. It enters **only** as the "supporting intermediate score" that `PREREGISTRATION.md` §1 already reserves, calibrated on clean calibration windows like every other score, and never touches the primary estimand ΔAUROC_harm or the committed generic score (`GENERIC_COMMITTED` stays exactly as declared). The linear subject is again the chart where the reading is exact.

### 2.3 Hook and feature selection: a registered, label-free rule (secondary)

Claim 1 has hook-drop controls but no principled rule for *which* hooks to keep. Bal et al. §6.4 gives one: greedy forward selection by redundancy-suppressed quadratic gain plus aligned cubic completion, scored by the retained local divergence fraction R⁽³⁾(S) (Eq. 6.34) and aligned triplet coverage (Eq. 6.32), on reference data alone. It needs a declared direction family Δ; the designed families of §7 (`output_aligned`, `task_aligned`, built from the frozen head at the reference mean, linear subject only) are exactly that and are label-free. So the rule respects the information contract (`reference_fit` only; no evaluation label) and can be registered before freeze. Because the designed families are linear-only, the selector is linear-only for now. Treat this as a secondary item: implement it as a reference-fit diagnostic that *reports* R⁽³⁾ per candidate hook set; do not let it choose the arms in this pass.

### 2.4 Architecture decision: a readable nonlinear subject with frozen cumulant propagators

On 11 September the author asked for a nonlinear counterpart to NFPA whose latent stays readable as physical representations, and named the worry that the rung-4 transformer's diagnostics, representation choice and fine-tuning would eat the time. Bal et al. §6.5 is the smallest architecture that answers this and carries its own physics semantics:

- vertices = channels (TES/QP channels on arm B; band-frames on TIDMAD);
- the order-2 propagator P₂ from the measured noise covariance — the noise-Laplacian PE of Paper 1, so the vocabulary of the two papers coincides;
- the order-3 propagator P₃ from the measured third cumulant of the noise (sign carried as an edge attribute; |w| or w² in the Laplacian, Eq. 5.6);
- P₂, P₃ **frozen from the reference cell**; only a small readout trained (their ~500 parameters per order, Eq. 6.36–6.37).

The latent stays readable because every propagator weight has a provenance, and the ORACLE question ("what does the frozen model report under N vs S?") can be asked of it at a fraction of the transformer's diagnostic cost. For Gaussian noise P₃ vanishes identically, so the subject is non-trivial exactly on the non-Gaussian cells — which is where the study's residual-cumulant statistic (§2.1) is also non-trivial; the two integrations are one mechanism seen twice. If a trainable correction on top of the linear class is wanted, use QUIVER's zero-initialised residual multiplicative gate (Eq. 8: x̃ = (1 + α·Θ) x, α = 0 at init) so the model is exactly the tied linear AE at step 0 and any departure is measurable as α grows.

**This is an addition of an architecture, which the two-claim prompt forbids "to rescue the story."** It is not proposed as a rescue and it is not proposed for this pass: it is a *candidate third subject* alongside the transformer for the Tier-2/Tier-3 nonlinear arm, recorded in `EXPERIMENT_DESIGN.md` §I.6 and `TESTBEDS.md` §2 as a design with a `Subject`-conforming interface, built only after the trained Tier-1 run exists and only if the author opts in (`IMPLEMENTATION_PROMPT_CUMULANT.md`, Block D). Both nonlinear subjects would be reported; neither is chosen by result.

## 3. Methodological precedent to cite

QUIVER's evidence design — identical splits and initialisation RNG, N = 10 paired seeds, zero-initialised gate so baseline and augmented models coincide at step 0, a paired t-test as the argument that a gain is information rather than capacity — is the same argument form as Claim 1's capacity-matched control (`generic_rich_matched`). One sentence in the proposal's designed-controls text, citing it as precedent, pre-empts the "you added capacity" objection. The quantum content plays no role.

## 4. What this does not change

The two claims, their primary estimands, the committed generic score, the arms' names, the information contract, the splits, the hard-matching rule, κ_m (still pending, never invented), the 16 September development artifacts (preserved, non-citable). No VQC. No cumulant computed on a nonlinear latent is called a cumulant. No hypergraph model is built in this pass.

## 5. Decisions the author records before freeze (`PREREGISTRATION.md` §8, new items)

7. Whether the pooled third-moment transform enters the arms (both sides, per §2.1) — yes/no, with the compression declared (n_pcs coordinates + one norm scalar).
8. Whether the cubic task score is the registered supporting intermediate score — yes/no.
9. Whether the R⁽³⁾ hook-selection diagnostic is registered as reference-fit-only reporting — yes/no.
10. Whether the frozen-propagator hypergraph subject is a candidate third subject for the nonlinear arm — yes/no, gated on the trained Tier-1 run.

## 6. Acceptance tests the implementation must add

- Dimensional and chart checks: the third-cumulant estimator returns zero (to sampling error, with a declared tolerance and n) on Gaussian reference data; is invariant under orthogonal reparameterisation of the whitened chart; and is not invariant under shear (recorded, like the existing invariance table in `tests/test_spine.py`).
- Symmetry of the arms: the feature count of `generic_rich_matched` equals that of `full_intermediate` after the new transform; a test that fails if a third-moment feature exists on one side only.
- Information contract: the new features carry `{raw_window, reference_fit}` tags and no evaluation-only source; the adversarial leakage test covers them.
- Signature test on the linear subject (development, paired replay): the §2.1 hypothesis row evaluated on the existing Tier-1 families plus the sparse-burst family, reported in the 13/1/0 format; a negative outcome is reportable.
- Claim-2 supporting score: calibrated on `calibration` windows, reported beside the task length; never combined into the committed generic score.
- New dated development smoke in a new directory; JSON strict; every result file says NOT CITABLE.

## 7. Cost

Estimator and tests: numpy only; O(N d³) per window with d = n_pcs ≤ 10 on Tier 1 — minutes. Arm and calibration plumbing: one day. Design text and pre-registration items: half a day. The hypergraph subject (if opted in later): three to five days for a `Subject`-conforming implementation with frozen propagators and a readout, excluding training.

## 8. Bibliography entries (verify against arXiv before use)

```bibtex
@article{bal2026triality,
  title   = {From Information Geometry to Jet Substructure: A Triality of Cumulant Tensors, Energy Correlators, and Hypergraphs},
  author  = {Bal, Aritra and Klute, Markus and Maier, Benedikt and Spannowsky, Michael},
  journal = {arXiv preprint arXiv:2605.03063},
  year    = {2026}, note = {v2, 7 May 2026}
}
@article{bal2026quiver,
  title   = {{QUIVER}: Quantum-Informed Views for Enhanced Representations in Large Machine Learning Models},
  author  = {Bal, Aritra and Binder, Michael and Klute, Markus and Maier, Benedikt and Spannowsky, Michael},
  journal = {arXiv preprint arXiv:2606.02785},
  year    = {2026}, note = {1 June 2026}
}
```
