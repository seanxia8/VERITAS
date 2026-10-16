# G5 — author decision required for the waveform nuisance law

Status: RESOLVED by explicit author approval in this chat on 2026-10-16: “批准 GATE_G5.md 中的完整方案”. Amendment A1 below is approved before any affected outcomes. Implementation and G6 validation remain pending. The original registered runbook remains byte-identical.

The missing source-contract fields are documented in GATE_G6.md and THEORY.md, “Source/configuration gate.” Runbook §1.2 permits only deterministic derivations; the existing source does not determine these fields. The final v3 execution review explicitly preserves this pre-outcome gate. This is why the launch authorization alone cannot resolve it.

## Approved amendment A1

Adopt THEORY.md's proposed independent Gaussian I/Q sinusoid at the registered template-selected FFT bin, zero bandwidth, identical unit channel loading, exact additive covariance, and reference-white variance after composite normalization. Adopt its proposed two leading reference-temporal covariance basis vectors, deterministic degenerate-eigenspace handling, mean-square-one normalization, and independent coefficients of variance `g*v_white/2`. Interpret g as added variance, not correlation.

Resolve the remaining sweep choices together: E1b line sweep holds g=.45; E1b common-mode sweep holds line multiplier=0. E2 fixes g=.45 throughout development/evaluation and selects its three line multipliers by the registered gamma_n rule. All other runbook values, masks, endpoints, budgets and eligibility thresholds stay as registered. Nonzero common mode makes WP7's x2 amplitude fault an actual perturbation. The base colored noise and physical white addition remain present in every cell; neither is rescaled by the added components.

This approved amendment defines the generating law; it is not a claim that it will satisfy rank, distinct gamma_n levels, trainability or budgets. G6 checks can still stop affected work after approval. Any later change requires a new explicit amendment before affected outcomes. Implementation may now adopt A1, with exact source/config binding and pre-outcome validation.

Affected work: E1b/E2 and their dependent E3/WP7 work. Exact-model population analysis is independent and continues. Author decision is recorded above; no further approval of this same law is required.

## Approved amendment A2 — E1c sample and loss binding

The author explicitly approved E1C_CONFIG_PROPOSAL.md on 2026-10-16 before any E1c outcomes. Its fixed n100000 full-batch sample, paired depth data/initialization and primary-source half-squared-norm reduction are now part of the contract. Original runbook thresholds, learning rate and horizon remain unchanged. Numerical gradient review and campaign budgeting are still required.
