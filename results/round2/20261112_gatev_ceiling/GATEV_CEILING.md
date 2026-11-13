# Gate V photon-statistics ceiling (addendum §2; diagnostic, never gates anything)

Perfect-model references: the Λ incident photons of each committed Gate V row redistributed as multinomial(Λ, μ/Σμ), detected charge n*·w̄, K = 500 draws (seed 20261112 keyed by detector, event, draw), scored with the `compare_event` arithmetic. c = median of draws per event; C = median over the 10 holdout events; ρ̃ = observed / c.

| Detector | Gate V (absolute) | Module obs. median | Module C | ρ̃ module median | Log-charge obs. median | Log C | ρ̃ log median | Photons / hit PMT | Split-half √rel. (module) |
|---|---|---|---|---|---|---|---|---|---|
| WCTE | passed | 0.902 | 0.9993 | 0.902 | 0.815 | 0.979 | 0.833 | 19.6 | 0.9988 |
| SK | failed | 0.829 | 0.9972 | 0.831 | 0.692 | 0.911 | 0.761 | 11.1 | 0.9983 |
| JUNO | failed | 0.888 | 0.9989 | 0.889 | 0.761 | 0.971 | 0.784 | 16.0 | 0.9992 |
| HK | failed | 0.756 | skipped¹ | – | 0.563 | skipped¹ | – | – | – |

μ reproduces the committed `expected_detected` on every event (max relative difference WCTE 0.0e+00, SK 0.0e+00, JUNO 0.0e+00).

**Reading (reported, not decided).** The module-pooled ceiling is ≥ 0.997 on every detector, so photon statistics cap the module metric only marginally: the SK and JUNO shortfalls against 0.85 are not finite-photon noise in the reference. The log-charge ceiling falls on SK (0.91) and so explains part, not all, of the log-charge shortfall. The analytic model is therefore not validated against PhotonSim + LUCiD at these scales, and every round-2 result carries this caveat.

Review notes (non-blocking): the μ check against the committed rows is enforced at run time (the unit test uses a synthetic event); rows and events are paired by position (the μ check partly guards this); the split-half estimate keeps PMTs hit in both halves (not the harness `keep` rule); the event cache is not tied to a code version.

¹ HK ceiling skipped: a second HK transport (> 15 CPU-h at the observed speed) exceeds the addendum's 3 wall-h ceiling allowance (1.30 h used). HK Gate V (informational): yield ratio 2.07 (outside 0.5–2), module median 0.756, log-charge median 0.563, time 0.80 ns.
