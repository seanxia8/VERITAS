# Note for Junjie — re-scoping N and the arm portfolio (12 Oct 2026)

One page. Full runbook: `docs/RUNBOOK_RESCOPE_N_2026-10-12.md` (the comment that
triggered it is Appendix A there). Nothing below changes either claim, an estimand,
`GENERIC_COMMITTED`, or a registered threshold; `PREREGISTRATION.md` stays unfrozen.

## What changed

| | before | now |
|---|---|---|
| LUCiD N | front-end electronics (`noise_module_lucid` clock, common mode, digitiser, cable) | **detector response**: efficiency loss, gain drift, timing drift, threshold walk (time-walk), module loss — `noise_module_lucid/response.py` |
| LUCiD electronics | building towards a long-window preset | **frozen** — removed by hit-finding thresholds before reconstruction: harm ≈ 0, generic-visible |
| what counts as N | implicit | **a change after training**; a static spread at training time is the reference; a different *simulation model* of acceptance or medium = **M**, excluded from both claims; a *physical* drift of transparency or acceptance after training is N |
| TIDMAD | real recorded noise only | + **controlled N** on the 208 noise-only files (`tidmad_transformer/n_injection.py`: added pink noise, in-band line, gain, drift, jitter, dead time; glitches as the undeclared `unknown` family, as on every arm) |
| training on the realism arms | not written down | baseline only, **≥ 3 (target 5) seeds**, variants at test time; no per-variant retraining for N families |
| arms | — | **no new arm** |

## Two corrections to the comment

1. "DELight adds none" is half right. There is no DELight simulation chain, but a
   sub-noise TES chain with noise does exist — HeST → `qp_simulator` → `noise_module`
   with the HeRALD TES placeholder budget — demoted on 18 Sep because it stops at
   quasiparticle arrival times; `noise_module` also holds a DELight-like Al2O3 athermal
   budget that no chain uses yet. Either comes back only if Phase B shows we need a
   second sub-noise arm.
2. The "signal at the noise floor, no threshold removes the noise" regime is the
   design's primary substrate already — Tier 1, controlled covariance injection, built —
   and the planned transfer arm B, NuRadioMC (Askaryan pulses near the thermal floor,
   coherent Galactic noise, reconstruction on noisy traces), is in that regime too; its
   pilot has not been run.

## What I need from you

1. **The transfer arm (D-T).** NuRadioMC stays provisional. If you prefer LArTPC via
   JAXTPC — coherent wire-plane noise, filtering as a real reconstruction step, and your
   group's code — it would **replace** NuRadioMC, not join it. Either way we need its K
   and the κ_m source.
2. **LUCiD's role (D-L).** The response families harm reconstruction, so Claim 2 has
   something to rank there; Claim 1 may still be null because gain drift and module loss
   are visible to the generic arm. With gate A0 still open — keep arm A as a Claim-1
   null control and Claim-2 replication, or drop it from the paper?
3. **Category M (D-M).** Is "mis-modelling is not N, keep it out of every cell" enough, or
   do you want M as its own origin label so it can be reported as an abstention target?

## Not being done

No LIGO arm (future work), no JAXTPC build before D-T, no HeST revival before Phase B,
no retraining per LUCiD variant, no training runs in this pass.
