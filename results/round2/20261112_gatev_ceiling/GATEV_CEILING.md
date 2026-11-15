# Gate V photon-statistics ceiling (addendum §2; diagnostic, never gates anything)

Perfect-model references: the Λ incident photons of each committed Gate V row redistributed as multinomial(Λ, μ/Σμ), detected charge n*·w̄, K = 500 draws (seed 20261112 keyed by detector, event, draw), scored with the `compare_event` arithmetic. c = median of draws per event; C = median over the 10 holdout events; ρ̃ = observed / c.

| Detector | Gate V (absolute) | Module obs. median | Module C | ρ̃ module median | Log-charge obs. median | Log C | ρ̃ log median | Photons / hit PMT | Split-half √rel. (module) |
|---|---|---|---|---|---|---|---|---|---|
| WCTE | passed | 0.902 | 0.9993 | 0.902 | 0.815 | 0.979 | 0.833 | 19.6 | 0.9988 |
| SK | failed | 0.829 | 0.9972 | 0.831 | 0.692 | 0.911 | 0.761 | 11.1 | 0.9983 |
| JUNO | failed | 0.888 | 0.9989 | 0.889 | 0.761 | 0.971 | 0.784 | 16.0 | 0.9992 |
| HK | failed | 0.756 | 0.9925² | 0.762 | 0.563 | 0.730² | 0.768 | 6.7 | –² |

μ reproduces the committed `expected_detected` on every event (max relative difference WCTE 0.0e+00, SK 0.0e+00, JUNO 0.0e+00).

**Reading (reported, not decided).** The module-pooled ceiling is ≥ 0.997 on every detector, so photon statistics cap the module metric only marginally: the SK and JUNO shortfalls against 0.85 are not finite-photon noise in the reference. The log-charge ceiling falls on SK (0.91) and so explains part, not all, of the log-charge shortfall. The analytic model is therefore not validated against PhotonSim + LUCiD at these scales, and every round-2 result carries this caveat.

Review notes (non-blocking): the μ check against the committed rows is enforced at run time (the unit test uses a synthetic event); rows and events are paired by position (the μ check partly guards this); the split-half estimate keeps PMTs hit in both halves (not the harness `keep` rule); the event cache is not tied to a code version.

¹ (superseded by ²) HK ceiling first skipped: a second HK transport (> 15 CPU-h at the observed speed) exceeds the addendum's 3 wall-h ceiling allowance (1.30 h used). HK Gate V (informational): yield ratio 2.07 (outside 0.5–2), module median 0.756, log-charge median 0.563, time 0.80 ns.

² HK ceiling computed in phase r2-report (16 Nov 2026) without a second transport (`src/jepa_noise/ext_r2_gatev_scaled.py`, `hk.json`): in the `compare_event` arithmetic the transport enters μ only through one uniform QE scalar, so μ's shape follows from the PhotonSim truth and its scale from Σμ = the committed `expected_detected`. Validated against the exact path on the cached SK and JUNO events (max relative μ difference 1.7e-15; C identical; `validate_scaled_*.json`). For HK the 1e-6 check is therefore true by construction; the split-half reliability and photons per hit module are not computed. Fresh-context review PASS. Reading (reported, not decided): the HK module ceiling (0.993) is far above the observed 0.756, so the module shortfall is not finite-photon noise; the HK log-charge ceiling (0.730) is itself near the 0.70 threshold (6.7 photons per hit PMT), so photon statistics explain a large part of the HK log-charge shortfall. HK's yield ratio 2.07 is outside 0.5–2 regardless.
