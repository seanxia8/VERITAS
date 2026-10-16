# WP0 local CPU preflight — v3, 16 October 2026

Registration: `f36c179ad5efca3305379ed3143ddd545b02c760`, branch `JEPA`, base `0cae6b061e488e319643cfb0ba820318b8923578`. Both independent v3 reviews end in ACCEPT with no remaining P1. This local check does not clear later numerical/data/resource gates or any remote host.

The separate environment is `/private/tmp/jepa-venv-cpu311`, created with `uv venv --python 3.11`. It follows README's isolated CPU recipe with the runbook-required Python 3.11 and torch 2.5.1 pins, without bare `uv sync` or dependency changes to protected packages. Imports use `PYTHONPATH=src`; editable installation is unnecessary for this source-path preflight. Temporary environment/cache locations avoid modifying other studies' environments. Recreate from `results/jepa/20261016_wp0_local/requirements-frozen.txt` if the temporary directory is cleaned.

Initial sandbox package acquisition failed with DNS resolution failure; the authorized network escalation succeeded. No outcome-producing script ran in that failed attempt. Python 3.11.11 and torch 2.5.1 are the prescribed CPU versions. Full resolved inventory was frozen before reproduction. `RECONSTRUCTION_DISABLE_TORCH_COMPILE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` are explicit. No CUDA/MPS training is performed.

Reproduction command:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RECONSTRUCTION_DISABLE_TORCH_COMPILE=1 /private/tmp/jepa-venv-cpu311/bin/python results/jepa/20261016_wp0_local/reproduce.py
```

The wrapper runs the registered Appendix A and P5 scripts unchanged, verifies the Appendix A correlation/angle tolerances and classifies the annihilation example as `oracle_rank_ineligible`. This is reproduction of disclosed development evidence, not prospective grid evidence. The wrapper records source and environment hashes, git status, host, RNG final state, explicit seed, timings and exit status in `provenance.json`. A complete execution is indicated only by that file's `exit_status=0` and passing checks. Source scripts were committed before execution; newly added package scaffolding and the reproduction wrapper are explicitly captured as untracked implementation inputs.

The package modules are placeholders, not completed WP1–WP6 implementations. Root changes are limited to the workspace member and README index entry. No dependency lock regeneration is needed for this isolated environment; root CUDA workspace resolution was not invoked.

## Observed acceptance evidence

All 12 required/dependency imports passed on the local CPU host. Appendix A passed correlation tolerance .01 and angle tolerance .5 degrees; printed canonical spectra agree after whitening. The P5 fixture is SPD and correctly classified `oracle_rank_ineligible`. Exit status 0. Total measured execution 110.956832 seconds (0.030821342 one-worker CPU hours), including imports. Per-script timings are in provenance. Environment installation and setup wall time are not included in this run timer and remain coordinator-accounted preparation. Matplotlib used an automatically created writable temporary cache after warning that its default home cache was unwritable; this did not affect numerical checks.

No local G1 fired. Remote imports, campaign projection and later implementation/data gates remain independent acceptance requirements.
