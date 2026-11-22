# READOUT_SPEC v1 — frozen readouts for r2-tier1 (runbook 2026-11-22 Appendix A, P5)

**Frozen 2026-11-22, before the first read of any sealed pack.** No readout, feature, cut or probe hyper-parameter changes after the first sealed read (§T2.3). A later bug fix is made on validation evidence only and every re-read is logged in `SEALED_ACCESS_LOG.md`.

| Pinned file | SHA256 at freeze |
|---|---|
| `src/jepa_noise/r2_probe_repair.py` (feature construction, committed MLP) | `a29c65c7de5708acaf34e1dcdc5286146a7e9f388c84048316783bd598865d4a` |
| `scripts/tier1_eval.py` (entry point, attentive probe, metrics) | `39860c09eaafaf98b1846421f9b7a78cb60583fe8302f54a1fba0ce1ce1478ca` |
| Runbook (Appendix A text below) | `43c8635c18a1067182e5bbf642dc2d3ab27a964fa815e0cc8da4c2a8d77236f3` |

Splits: probes are fitted on `dev_probe_train`, selected (early stopping) on `dev_probe_test` (the validation split) and applied to `sealed_test`.

## Appendix A (verbatim from the runbook)

Token features x_p ∈ ℝ⁹ per PMT: scaled position r_p (indices 0–2), direction (3–5), ℓ_p = ln(max(q_p, 10⁻⁶ PE) / 1 PE) for hit PMTs (index 6), τ_p = t_p / T_scale (index 7), hit flag h_p (index 8). z_p ∈ ℝ¹²⁸ is the token embedding from `encode_tokens` of the full, unmasked event. N is the number of PMTs.

**A.1 All-mean.** z̄ = (1/N) Σ_p z_p.

**A.2 Spatial-moment latent (768 features), the primary readout.** Let H = {p : h_p = 1} (all tokens if H is empty) and P = {p ∈ H : τ_p ≥ 0}.
- m₁ = mean of z_p over H.
- m₂ = Σ_H w^q_p z_p with w^q_p ∝ exp(3 ℓ_p), normalised over H.
- m₃ = Σ_P w^e_p z_p with w^e_p ∝ exp(3 ℓ_p − 4 (τ_p − min_P τ)), normalised over P; if P is empty, m₃ = m₂.
- C = (1/|H|) Σ_H (r_p − r̄) z_pᵀ ∈ ℝ³ˣ¹²⁸, with r̄ the mean of r_p over H.
- Feature vector: [m₁, m₂, m₃, vec(C)].

**A.3 Raw-feature baseline (33 features).** The construction of A.2 with z_p replaced by (ℓ_p, τ_p, h_p), plus 15 summaries: |H|/N; ln max(|H|, 1); mean and SD of ℓ over H; mean and SD of τ over P; r̄; Σ w^q_p r_p; Σ w^e_p r_p. `spatial_moment_full` is A.2 plus the same 15 summaries.

**Ridge protocol.** Features standardised on the training split; one Ridge per target coordinate with intercept; α ∈ {10⁻⁶, 10⁻⁴, 10⁻², 1, 100} chosen by five-fold cross-validation with folds grouped by run; vertex error = mean Euclidean distance on the test split.

**A.4 Attentive probe (new; frozen here).** Per event, at most 512 hit tokens (keyed subsample by event id). Eight heads: score s_{h,p} = q_hᵀ W z_p / 8 with W: 128 → 64 and learned q_h; a_{h,p} = softmax over the event's tokens. Each head outputs [Σ_p a_{h,p} V z_p, Σ_p a_{h,p} r_p] with V: 128 → 32. The 8 × 35 values pass through LayerNorm and one linear layer to the standardised target. AdamW (lr 10⁻³, weight decay 10⁻²), batch 256, cosine schedule, ≤ 200 epochs, early stopping on the validation split with patience 25, 3 seeds.

**MLP probe.** As committed (`r2_probe_repair.fit_mlp`), unchanged, with 3 probe seeds.

---

## Implementation map (file:line at freeze)

- Token embeddings: `arm.encode_tokens(features)` of the full, unmasked event (`within` masker with `n_min = 0` used only for tokenisation), `scripts/tier1_eval.py:184`.
- A.1 all-mean: `lucid_arms._mean_tokens(z, None)` (mean over all N tokens).
- A.2 spatial-moment latent (768): `r2_probe_repair.moment_features(z, raw)[0]` (`r2_probe_repair.py:61`, weights `_weights` at `:11`: m₂ ∝ exp(3ℓ) over H, m₃ ∝ exp(3ℓ − 4(τ − min_P τ)) over P, falls back to m₂ when P is empty; C = hit-weighted centred cross-moment).
- A.3 raw baseline (33): `moment_features(raw[..., 6:9], raw)[1]`; `spatial_moment_full` (783) = `moment_features(z, raw)[1]`.
- Ridge: `lucid_arms.fit_probe` → `probes.grouped_ridge`, α ∈ `probes.ALPHAS` = (10⁻⁶, 10⁻⁴, 10⁻², 1, 100), five folds grouped by run (`lucid_arms._fold_labels`), features standardised on train, intercept, one fit per target.
- MLP: `r2_probe_repair.fit_mlp` unchanged (`:114`), on `spatial_moment_full` and on `spatial_moment_latent`, probe seeds 0, 1, 2 (its internal early stopping holds out 10 % of training runs, as committed).
- Attentive (A.4): `tier1_eval.fit_attentive` (`:278`), token subsample `attentive_pick` (`:145`, keyed by event id, `keyed_rng(0, "r2-tier1/attentive", event_id)`), probe seeds 0, 1, 2.

## Choices fixed here where Appendix A is silent

1. **Attentive tokens** are stored as float16 (cast to float32 for training); an event with no hit token uses token 0 as its only token (photon bombs always have hits in practice). Early stopping uses the validation split, so attentive numbers on `dev_probe_test` are optimistic; only sealed numbers count.
2. **Vertex error** = mean Euclidean 3-D error (m) on photon bombs; median, 68 % quantile and error / R_fid also reported.
3. **Direction (μ, e separately):** Ridge on the spatial-moment latent (A.2) features to the unit direction, prediction renormalised; median and 68 % angle. The committed MLP direction probe on all-mean features (`ext_followup._direction_tasks`, as in `ext_r2.py:1117`) is reported for continuity.
4. **Energy (μ, e separately):** events with LUCiD `contained = True` and `deposited_MeV > 0` (the committed suite's target and flag); Ridge on ln E from the A.2 features; Ê = exp(prediction); 68 % half-width = (q₈₄ − q₁₆)/2 of (Ê − E)/E, RMS, median bias; R² on ln E and on E. Not evaluated with fewer than 50 contained events in train or test.
5. **PID:** `ext_followup._logistic_auc` (L2 logistic, standardised A.2 features, C ∈ {0.01, 0.1, 1, 10} by run-grouped 5-fold CV) on μ versus e; score = p(μ). μ mis-ID at 99 % e efficiency = fraction of μ with score ≤ the 99 % quantile of e scores; e mis-ID at 99 % μ efficiency = fraction of e with score ≥ the 1 % quantile of μ scores, thresholds set on the evaluated sample. AUC also reported.
6. **Collapse:** f_event, f_layout from `ext_readout.layout_fractions` over photon-bomb tokens of the test cohort.
7. **Bootstrap:** per-event values are saved; 2,000 resamples with indices that depend only on the cohort name and event count (`boot_indices`, seed 20261122 + cohort hash), so every run shares them (paired bootstrap of contrasts).
8. **Matched init** = each run's own `snapshot_step_00000.pt`; **legacy init** = `build_r2_arm("init", …)` with `EncoderConfig()` defaults (window 64).
9. **Time-reference cell** (`__trmedian`, T5.1): the encoder input uses the run's own median-referenced times (`r2_store.apply_time_reference`); the Appendix A weights and summaries use the trigger-relative τ as for every other cell.

## Validation of the implementation (before freeze)

`tier1_eval.py` on `wcte__mae_repaired__s9000` step 10000, `dev_probe_test`, Deepthought CPU: spatial-moment Ridge 0.0521 m, all-mean 0.1410 m, raw baseline 0.0752 m, f_event 0.1852, f_layout 0.548, committed MLP direction e 8.34° / μ 10.19° — identical to the committed r2-lucid `final.json` values (μ direction 10.20°, MLP run-to-run noise). New metrics: spatial MLP (latent) 0.0482 m, attentive 0.0624 m (3-seed means), μ energy 68 % half-width 0.033 (310 contained), μ mis-ID at 99 % e efficiency 0.055.

**Pin update, 2026-11-22, still before the first sealed read:** `tier1_eval.py` gained `--log-only` (batch route), staged-pack paths and a multi-step loop; no readout, feature or probe code changed. The pinned SHA256 above is the updated file (39860c09eaafaf98b1846421f9b7a78cb60583fe8302f54a1fba0ce1ce1478ca).
