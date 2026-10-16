# Proposed A3 — E1 cohorts, calibration and estimator information

**Status: UNAPPROVED; author requested modification on2026-10-16. Specific revision instructions are pending.** No dependent outcome generation is authorized by this document. A1 and A2 remain approved; A3 does not change their laws, any R1–R5 threshold, seed population, main grid, depth, learning rate or horizon. Implementation review and the complete E1 budget projection remain separate gates.

## Why this clarification is needed

Runbook §1.3, lines 48–55, fixes evaluation seeds, five grouped ridge folds, alpha selection, per-target errors and undefined targets, but gives no E1 probe sample sizes or cohort manifest. WP3, lines 204–225, fixes representation sample sizes and asks for nuisance probes and physical errors; those representation samples do not automatically define independent probe cohorts. WP5's later E2 cohort table is not an E1 assignment.

Runbook §2.3, lines 95–107, already fixes **4096 event and 4096 noise records for E1/E2 Criterion M**. That count is not missing. What remains unspecified is E1's M development setting and scope, and the separate full-dimensional noise-calibration sample count for sample GEVP/finite-beta controls. Runbook §2.2 P4, line 88, requires matched noise-only calibration; WP2, line 198, explicitly names **exact noise whitening** for the whitened PCA control. These information sources must be labelled separately.

## Proposed amendment A3

### 1. Subject, probe and calibration cohorts

Keep all registered subject sample sizes and seeds. For E1a, generate a distinct subject batch for every `(theta_scale, eta_scale, gamma_load, n, seed)`; the loading geometry is fixed by the evaluation seed across sample sizes and sharing levels. Subject data draw namespaces include n. Main model sharing remains `share="target"`; `share="both"` stays the Appendix A reproduction only.

For each E1a `(theta_scale, eta_scale, gamma_load, seed)`, generate **10000 independent probe-training and 10000 independent probe-test events**. Reuse these same probe cohorts across methods and the three subject sample sizes so convergence comparisons use paired evaluation events. Probe namespaces omit subject n but include the physical cell and seed. Their random streams are independent of subject and noise-calibration streams; probe labels never fit representation maps or choose scientific coefficients.

For each of the **nine unique A1 E1b cells** and each evaluation seed, likewise generate 10000 probe-training and 10000 probe-test raw waveform events. Reuse the same raw cohorts across all four prescribed splits and all methods. This pairing does not claim that the four splits are independent experiments. No representation or probe cohort is shared with E1a, E1c, E2 or the M development cohort.

Freeze explicit role namespaces `E1a_subject`, `E1a_probe_train`, `E1a_probe_test`, `E1a_noise_cal`, and their E1b analogues. Within each role, use event IDs `0..count-1`; distinct namespaces, not integer offsets alone, establish independence. Store namespace keys, seeds, event IDs, source/config hashes and RNG states. Within a cell, all methods receive identical subject/probe/calibration inputs. A1's shared intersection `(line=0,g=.45)` is computed once and referenced by both sweep displays, never treated as two independent cells.

For sample noise-covariance controls, use **n_cal = n_subject** independent noise-only records per E1a cell/n/seed, and **n_cal=100000** per E1b cell/seed, shared across that cell's methods and raw splits. Center calibration records using their own means and form unbiased sample covariance with divisor `n_cal-1`. These noise records are independent of every subject/probe/M record. Do not replace failed sample covariance by exact covariance or add undeclared jitter. This is separate from Criterion M's existing 4096+4096 count.

### 2. Probe fitting and grouping

Assign probe-training IDs deterministically to five blocks of 2000 events: `0..1999`, `2000..3999`, `4000..5999`, `6000..7999`, `8000..9999`. Each fold holds out one whole block. These are independent synthetic-event blocks, not five independent detector acquisitions. All methods share their fold assignment.

Keep the registered per-output alpha grid `{1e-6,1e-4,1e-2,1,100}`, mean held-out MSE selection and ties favoring larger alpha. Fit feature standardization on each fold's training partition only; fit the final feature scaler and selected-alpha probe on all probe-training events. Use a fitted intercept; feature scaling cannot use subject, calibration or test records. Score once on the independent probe-test cohort. The linear embedding is its frozen context row map applied to the complete declared context view, without stochastic masks or per-event normalization. Oracle embeddings include their frozen context preprocessing/pullback; they never use target or gap observations at inference.

E1a physics targets are the three theta coordinates; nuisance is the single eta coefficient. E1b physics targets are the three centered delta-theta coordinates of the registered linearized fixture; nuisance targets are the two line and two common-mode coefficients declared by A1. Exact constant targets are undefined and excluded only according to the registered common valid-target rule. Rank-ineligible physics metrics remain explicitly not evaluable when their required Fisher support is absent; no pseudoinverse supplies a substitute CRLB.

All three finite-beta coefficients are reported as prespecified E1 arms. A3 adds no beta selection, development tuning or best-test checkpoint.

### 3. Ten E1a/E1b method/settings rows and information labels

Use the same ten rows for every E1a and E1b view dataset:

| Row | Estimator | Noise information |
|---|---|---|
| 1 | CCA | Subject sample covariance; registered CCA ridge rule |
| 2 | Hard-isotropy prediction | Subject sample covariance; no noise input |
| 3 | Context PCA | Subject sample covariance; no noise input |
| 4 | Noise-whitened context PCA | **Exact population context noise covariance; oracle-noise control**, as runbook line 198 requires |
| 5 | Pure noise GEVP | Independent **sample calibration** covariance |
| 6–8 | Finite-beta prediction, beta .1/1/10 | The same independent **sample calibration** covariance as row 5 |
| 9 | Conditional joint-whitening oracle | **Exact retained population joint noise covariance**, with population rank/canonical eligibility recorded separately from sample angular accuracy |
| 10 | Context OF/GLS physics baseline | **Exact population context noise covariance and known signal loading; oracle-information physics estimator** |

Population theorem checks and the noise-weighted reference use exact generating matrices. They are separate from sample control estimates. Exact covariance is not silently supplied to sample GEVP/finite-beta, and no sample-control failure is repaired by exchanging those inputs.

Keep the existing CCA contract: unregularized covariance for exact identities/R3; otherwise the fixed `1e-6*trace(S)/d` marginal ridge is recorded. Other sample controls use the unregularized empirical covariance matrices defining their stated objectives; numerical failure is recorded, not repaired through a new ridge. This clarifies covariance inputs without modifying the reviewed eigensolver's tolerances. R3's additional transformed/unregularized checks are diagnostic calculations, not extra method rows; their execution count must be included in the projection.

For row 10, jointly estimate all three physics coordinates from **the same context coordinates available to the corresponding representation** using `(A_c.T N_cc^-1 A_c)^-1 A_c.T N_cc^-1 x_c`. E1a uses A_c/theta; E1b uses J0's context rows/delta-theta. Full-rank Fisher support is required. Use the matching context Fisher inverse for MSE/CRLB denominators on the same probe-test events. Where support fails, preserve not-evaluable status. No full-record or target observation is granted to OF. The existing known-other-parameters amplitude variance `1/I_aa` remains a separate analytic sanity check, not the multi-parameter OF score. OF needs no fitted ridge probe and has no nuisance-R² claim.

### 4. Criterion M in E1

Perform the registered M heuristic once for **E1b**, on a separate linearized development setting **line multiplier=1.5, common strength=.45, development seed=100**. This setting is proposed before feature/probe outcomes; it is not selected from measured gamma_n or scientific scores. Draw **4096 event and 4096 independent noise records**, with separate `E1b_M_event` and `E1b_M_noise` namespaces, disjoint from every subject/probe/calibration cohort.

Use the unchanged seed-2718 QR projections, k=3, regularization, candidate order and tie rule of §2.3. Freeze and publish all four scores/projections and the winning split. In E1b this is a **mask-selection diagnostic annotation**: retain all four registered split rows and do not introduce a new method, remove a losing split, replace an ineligible split or change any R1–R5 criterion. E1a and E1c have their fixed two-view model and no alternative raw-split search, so no M selection is added there. Degenerate/invalid M inputs still trigger the scoped G6 rule.

### 5. Scope of E1c

A2's ten n100000 finite subject batches, depth pairing, initialization pairing, loss and 100000-step schedules remain unchanged. E1c's required endpoint is the four absolute mode trajectories and R5 eligibility/censoring. A3 does **not** add optional E1c physics/nuisance probe campaigns or a new hyperparameter search. The separately labelled diagonal-covariance engineering fixture and its gradient/stop-gradient review still precede E1c launch.

## Counts and budget impact if A3 is approved

| Component | Required logical count |
|---|---:|
| E1a subject datasets | 3 physics scales ×2 nuisance scales ×7 sharing levels ×3 n ×5 seeds = **630** |
| E1a method/settings records | 630×10 = **6300**, including **5670** representation maps and 630 OF rows |
| E1a subject observations | 210×(1000+10000+100000) = **23,310,000** |
| E1a independent noise-calibration observations | **23,310,000** under n_cal=n_subject |
| E1a probe cohorts | 42 physical cells ×5 seeds =210 paired cohorts; **4,200,000** total train+test events, reused across n/methods |
| E1b unique raw cells | 5 line-sweep +5 common-sweep −1 shared intersection = **9** |
| E1b raw subject datasets | 9×5 = **45**, each n100000; **4,500,000** raw events |
| E1b view datasets | 45×4 splits = **180** |
| E1b method/settings records | 180×10 = **1800**, including **1620** representation maps and 180 OF rows |
| E1b independent calibration noise | **4,500,000** raw records, reused across the four views |
| E1b probe cohorts | 45 paired raw cohorts; **900000** train+test events, reused across views/methods |
| E1b M | **4096+4096** independent raw records, four projected candidate scores |
| E1c | 2 sharing levels ×5 seeds = **10** n100000 batches; ×3 depths = **30** runs; **3,000,000** GD updates |
| Extra checks | Diagonal dynamics fixture, population eligibility/recovery, R3 transformed fits, annihilation and numerical/gradient fixtures, all separately charged |

The two E1b sweep plots contain ten displayed cell positions but only nine independent unique cells. Reusing the intersection avoids doubling its inferential weight; it does not omit a registered point.

A conservative ridge-work count is **26 fits per valid output** (five alphas ×five folds plus one final refit). With nine representation methods, E1a has 5670×4 outputs×26 = **589680** fits. Across E1b's nine unique cells, valid output counts sum to `4×7 +4×5 +1×3 =51` (three physics outputs plus nonconstant nuisance outputs); therefore 9 methods ×4 splits ×5 seeds ×51 ×26 = **238680** fits. Combined upper planning count: **828360 small ridge fits**, before accounting for genuine not-evaluable Fisher/target cases. Shared design matrices/factorizations may reduce runtime without changing folds, objectives or counts. No savings from favorable outcomes may be assumed.

Full projection must additionally include raw generation, covariance accumulation, independent calibration/M, eigensolves and target completion, whitening/oracle operations, probe embedding extraction, exact CRLB/OF, trajectory logging, failed attempts, artifact transfer/hashing, figures/tables and **25% reserve**. A single E1b n100000 float64 raw array occupies about **1.64 GB decimal**, before simultaneous views, calibration/probe arrays or intermediates. Allocate workers from measured memory and host load; do not multiply nominal CPU capacity by a per-job timing while ignoring RAM.

E1's cap remains **24 wall-hours at the recorded worker allocation**, with aggregate worker-hours divided by that allocation plus serial preparation/evaluation and an additional active-wall cap (§3.2 lines 146–155). E1 has no automatic grid, seed, sample or depth reduction. Benchmark representative generation/solvers, probe work and A2 dynamics under the existing ten-CPU-minute per-benchmark ceiling, then project the complete campaign. If over cap, G2 applies; A3 does not authorize a CPU/GPU substitution or an unlisted fallback.

## Author decision requested

Approve A3 as a whole, or identify the exact clause to revise before affected outcomes. This asks for the missing scientific/data-contract choices together, not renewed push/GPU permission. Registration of approved A3, independent implementation review and resource acceptance are still required before launch.
