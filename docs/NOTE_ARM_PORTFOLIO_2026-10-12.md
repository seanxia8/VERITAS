# Note for Junjie — re-scoping N and the arm portfolio (12 Oct 2026)

One page. Full runbook: `docs/RUNBOOK_RESCOPE_N_2026-10-12.md` (the comment that
triggered it is Appendix A there). Nothing below changes either claim, an estimand,
`GENERIC_COMMITTED`, or a registered threshold; `PREREGISTRATION.md` stays unfrozen.

## What changed

| | before | now |
|---|---|---|
| LUCiD N | front-end electronics (`noise_module_lucid` clock, common mode, digitiser, cable) | **detector response**: efficiency loss, gain drift, timing drift, threshold walk (time-walk), module loss — `noise_module_lucid/response.py` |
| LUCiD electronics | building towards a long-window preset | **frozen** — removed by hit-finding thresholds before reconstruction: harm ≈ 0, generic-visible |
| what counts as N | implicit | **a change after training**; a static spread at training time is the reference; angular acceptance, scattering, absorption, medium = **M** (mis-modelling), excluded from both claims |
| TIDMAD | real recorded noise only | + **controlled N** on the 208 noise-only files (`tidmad_transformer/n_injection.py`: added pink noise, in-band line, gain, drift, glitches, jitter, dead time) |
| training on the realism arms | not written down | baseline only, **≥ 3 (target 5) seeds**, variants at test time; no per-variant retraining for N families |
| arms | — | **no new arm** |

## Two corrections to the comment

1. DELight is not without N machinery: HeST → `qp_simulator` → `noise_module` (with the
   Al2O3 athermal budget) exists; it was demoted on 18 Sep because it stops at
   quasiparticle arrival times. It comes back only if Phase B shows we need a second
   sub-noise arm.
2. The "signal at the noise floor, no threshold removes the noise" regime is already
   carried by Tier 1 (controlled covariance injection, the primary substrate) and by the
   provisional transfer arm B, NuRadioMC (Askaryan pulses near the thermal floor,
   coherent Galactic noise, reconstruction on noisy traces).

## What I need from you

1. **The transfer arm (D-T).** NuRadioMC stays provisional. If you prefer LArTPC via
   JAXTPC — coherent wire-plane noise, filtering as a real reconstruction step, and your
   group's code — it would **replace** NuRadioMC, not join it. Either way we need its K
   and the κ_m source.
2. **LUCiD's role (D-L).** With electronics noise frozen and gate A0 still open, arm A is
   at most a null-expected control (a generic monitor may still catch gain drift through
   the charge histogram). Keep it, or drop it from the paper?
3. **Category M (D-M).** Is "mis-modelling is not N, keep it out of every cell" enough, or
   do you want M as its own origin label so it can be reported as an abstention target?

## Not being done

No LIGO arm (future work), no JAXTPC build before D-T, no HeST revival before Phase B,
no retraining per LUCiD variant, no training runs in this pass.
