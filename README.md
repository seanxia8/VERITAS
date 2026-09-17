# Consequence-Aware Failure Diagnostics for Particle Reconstruction

This repository is a research concept-note and feasibility package for studying
failure diagnostics in learned particle-reconstruction systems. It asks two
questions that ordinary distribution-shift detection does not answer:

1. Can a frozen model distinguish corrupted acquisition from clean physics that
   is under-represented in its training data?
2. Does the magnitude of a monitoring alarm track the downstream scientific
   damage caused by that shift?

The project is currently a **two-claim proposal with a protocol prototype and
development-mode pilots**, not a completed study or a production monitoring
library: no protocol is frozen, no result is confirmatory, there is no
trained-model or transfer evidence, and no scientific claim is validated by the
code passing its tests. The main deliverable is the concept note in
[`latex/paper3_proposal.pdf`](latex/paper3_proposal.pdf) (two-claim revision,
16 Sep 2026; `docs/REVISION_REPORT_2026-09-16_pass2.md` says what changed and
what remains). A bounded 17 Sep pass added the first non-Gaussian rung of the
Fisher-cumulant tower (the connected third cumulant of the whitened residual and
the pooled hooks, Bal et al. `arXiv:2605.03063v2`) as **supporting** machinery,
with a cubic companion of the task length, a Tier-1 signature row and a
design-only candidate nonlinear subject; it changes no claim or estimand
(`docs/REVISION_REPORT_2026-09-17_cumulant.md`).

## Study design

The proposal defines three failure contracts:

- **N — acquisition/noise shift:** corrupted measurements such as module loss,
  hit thinning, jitter, drift, glitches, or covariance changes.
- **S — training-support shift:** clean, physically valid events from sparsely
  represented parts of the training distribution.
- **E — evaluation-contract fault:** deterministic pipeline errors such as
  incorrect units, coordinates, output semantics, or metrics.

N versus S is the main attribution problem; E is handled as a deterministic
validation gate. The proposed experiments evaluate detection, attribution with
abstention, monitoring cost, and the relationship between alarm magnitude and
scientific consequence. They compare layerwise representation monitoring with
input, output, uncertainty, MMD, classifier two-sample, and embedding baselines
under the same false-alert budget.

The paper makes exactly two claims (`docs/PREREGISTRATION.md`, unfrozen
draft): **Claim 1** — do *intermediate* representations add N-versus-S
attribution information beyond a capacity-matched strong generic monitor with
identical alarm-time side information (paired ΔF1 on a hard, signature-matched
evaluation set, with capacity-matched and hook-drop controls and abstention on
undeclared families evaluated empirically); **Claim 2** — does a predeclared
task-sensitive representation score (the length of a window's deviation in the
task metric J_yᵀ W_y J_y) rank scientific harm on held-out physical
intervention cells better than a committed generic score (paired ΔAUROC for
K ≥ κ_m with the cell as the resampling unit; κ_m is pending on every arm).
Detection, cost, probes, designed controls, resolvability and patching are
supporting analyses. Origin (N/S/G/mixture/unknown) and harm (benign/harmful)
are separate labels; the proposal's "E" is an evaluation-contract fault (EF),
the controlled-variable table's "E" is event variation (EV).

Three tiers, one direction of travel (`docs/EXPERIMENT_DESIGN.md`):

- **Tier 1 — controlled waveforms** (`src/noise_module/`, `src/latent_monitor/`):
  assumed and realized covariance both known; the mechanism tier.
- **Tier 2 — realism arms**: HeST → `qp_simulator` → `noise_module` (built),
  LUCiD + `noise_module` (licence-gated), Prometheus/DynEdge (frozen public
  model; fallback).
- **Tier 3 — TIDMAD real data**: one compact transformer trained under MSE and
  inverse-PSD objectives on identical real electronics noise.

## What has been run

The checked-in NuBench work is a feasibility test on the Hexagon Ice LE DynEdge
direction model:

- Rescoring 2,989,339 released predictions comes close to the published Table 6
  values, but the largest discrepancy (0.055 degrees or 0.051 percentage point)
  exceeds the predeclared strict 0.01-unit identity tolerance.
- Restoring the released 1,358,099-parameter checkpoint exposes a
  128-dimensional backbone representation, but CPU re-inference on 256 events
  differs from the released directions by 0.944 degrees at the median (7.708
  degrees at the 95th percentile).
- An exploratory module-dropout pilot shows the expected monotone response from
  0% to 50% dropout: median angular error rises from 15.28 to 25.12 degrees,
  standardized embedding displacement rises from 0.000 to 0.149, and 10-nearest
  neighbour retention falls from 1.000 to 0.393.

On Tier 1 the linear-subject table (`results/latent_monitor_tier1/`, 6 Sep) shows
the predicted signatures under paired replay — development evidence of the
signatures, not alarm-time attribution — and two protocol smokes
(`results/latent_monitor_smoke_dev/`, pass 1, whose layerwise attribution delta
was negative/inconclusive and whose layerwise harm AUROC was weak;
`results/latent_monitor_smoke_dev_2026-09-16_pass2/`, the two-claim chain at toy
size; `results/latent_monitor_smoke_dev_2026-09-17_cumulant/`, the same chain
with the third-cumulant features and the supporting cubic score) exercise the
protocol end to end, and `results/latent_monitor_sig_c3_2026-09-17/` is the
development Tier-1 signature row for the residual third cumulant (a development
negative for the glitch / sparse-burst separation at the tested size). All are
labelled development output and none is evidence for either claim.

These results establish that the perturbation and embedding hook are useful for
protocol development. They **do not validate the proposal's scientific claims**:
the 256-event sample is deliberately enriched for high-multiplicity events, and
exact checkpoint/released-prediction parity remains unresolved. The full
interpretation and blocker are kept with the NuBench feasibility work, outside
this repository.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/noise_module/` | Validated stationary/nonstationary/multichannel noise and PSD simulation (numpy/scipy only) |
| `src/latent_monitor/` | Controlled-variable latent monitoring: `Subject` protocol, reference-cell projectors (`P_resolved/P_weak`; legacy `P_exc/P_unexc`), per-event Δz statistics, the development lookup, adjustments; linear and transformer subjects (`docs/EXPERIMENT_DESIGN.md` §§III.2–III.4); `task_metric.py` (M_recon vs M_task), `support.py` (validated training-support estimator), `designed.py` (positive controls, linear-only); `cumulant.py` (connected third/fourth cumulant estimators, D7) and `hypergraph_subject.py` (the design-only frozen-propagator candidate nonlinear subject, D10); `estimators/` holds the four linear classes of Paper 1 (future-work probes); `protocol/` is the two-claim protocol layer (Part V): typed alarm-time feature builder with source-tagged batches, five event-group partitions, primary/control/diagnostic arms, abstention chain, hard matching and the joint decision table, Claim-2 scores with paired ΔAUROC and cell-outer resampling, dev/confirmatory gates with run-dependency verification, and the CPU smoke |
| `results/latent_monitor_smoke_dev/` | The pass-1 protocol smoke (16 Sep) — development output, not citable, preserved as recorded |
| `results/latent_monitor_smoke_dev_2026-09-16_pass2/` | The two-claim protocol smoke — development output, not citable; regenerate with `PYTHONPATH=src python -m latent_monitor.protocol.smoke --out <new dated dir>` |
| `results/latent_monitor_smoke_dev_2026-09-17_cumulant/` | The two-claim smoke with the D7/D8 features — development output, not citable |
| `results/latent_monitor_sig_c3_2026-09-17/` | The Tier-1 signature row for the residual third cumulant — development output, not citable; run with `PYTHONPATH=src python -m latent_monitor.run_cumulant_signature --out <dir>` |
| `docs/TWO_CLAIM_REVISION_PLAN.md` | The two-claim revision and implementation-repair plan (M1–M5, I1–I13, D1–D4) |
| `docs/PREREGISTRATION.md` | The `core` protocol part — **unfrozen draft**; `protocol/frozen/` does not exist, so confirmatory mode refuses |
| `docs/REVISION_REPORT_2026-09-16.md`, `docs/REVISION_REPORT_2026-09-16_pass2.md`, `docs/REVISION_REPORT_2026-09-17_cumulant.md` | The pass-1 and pass-2 revision reports and the 17 Sep Fisher-cumulant integration report: what changed, what was tested, which blocks are ticked or deferred, the next gate |
| `src/herald_simulation/` | HeST → `qp_simulator` → `noise_module`: the paired superfluid-helium dark-matter arm (plan §7); HeST pinned and unpatched via `fetch_hest.sh` |
| `results/latent_monitor_tier1/` | The §1 table on the linear subject: 13 match / 1 documented / 0 mismatch, plus re-whitening, patching and stage-refit outcomes |
| `src/qp_simulator/` | Minimal standalone quasi-particle (QP) trace simulator (numpy only) |
| `src/reconstruction_model/` | DELight transformer reconstruction model + architecture catalog |
| `src/tidmad_transformer/` | TIDMAD band-frame STFT denoising arm (backbone from `reconstruction_model`, vendored Paper-1 benchmark helpers) |
| `notebooks/` | Smoke/inference notebooks, the noise-module tutorials, and the two executed HeRALD/LUCiD walk-throughs (`one_event_herald_lucid.ipynb`, `noise_models_herald_lucid.ipynb`) |
| `scripts/` | Local/Condor training helpers and smoke tests |
| `containers/` | Runtime container image definition |
| `docs/EXPERIMENT_DESIGN.md` | **The canonical design** (10 Sep 2026, consolidated 16 Sep as a two-claim study): Part I the study and the evidence ladder; Part II the arms; Part III the controlled-variable protocol with conditional signatures; Part IV the linear subject classes (future-work probes); Part V the two-claim protocol — one current definition throughout, no precedence rules |
| `docs/TESTBEDS.md` | Canonical testbed inventory: simulations and real datasets that serve and do not, what is implemented, and the structured autoencoder target |
| `docs/IMPLEMENTATION_PLAN.md` | Work packages, interfaces, acceptance criteria, gates |
| `docs/REVIEW_PROMPTS.md` | Reviewer prompts (§A before implementation, §B per milestone), synchronised with the amended C4 endpoint; reviews land in `docs/reviews/` |
| `docs/archive/` | Superseded documents, indexed in `docs/archive/README.md`: the 5 Sep arms and latent-monitoring plans and testbed survey (merged into `EXPERIMENT_DESIGN.md` / `TESTBEDS.md`), the 3 Sep theme/novelty/dev notes, audit, open decisions, revision plan, dataset-production plan, novelty review, package docs |
| `reference/papers/` | Prior-art and testbed PDFs, with the novelty analysis (`papers.tsv` is the manifest; the README has the fetch loop) |
| `scripts/nubench/` | NuBench feasibility scripts (migrated 2026-08-17, post-audit) |
| `results/` | Checked-in feasibility results with audit caveats |
| `docs/archive/PAPER3_AUDIT.md` | Adversarial audit of the proposal, repo and plan (2026-08-17; archived) |
| `docs/archive/OPEN_DECISIONS.md` | Researched resolutions of the open technical decisions (archived; superseded by `EXPERIMENT_DESIGN.md`) |
| `docs/archive/REVISION_PLAN.md` | Earlier shared execution plan (migrated 2026-08-23; archived, superseded by `IMPLEMENTATION_PLAN.md`) |
| `docs/archive/PERSONAL_RESEARCH_GUIDE.md` | Private working record and review log — **not for the shared view** |
| `archive/` | Code, scripts and notes off the active path (`archive/README.md` says what and why) |
| `latex/paper3_proposal.tex` | Source for the collaboration concept note (five pages since the 2026-09-02 mechanism section) |
| `latex/paper3_proposal.pdf` | Compiled proposal |
| `latex/figures/` | Proposal figures |
| `references.bib` | Working bibliography; the current proposal uses a self-contained bibliography in the TeX source |

## Python packages

This is a [`uv`](https://docs.astral.sh/uv/) workspace with four members sharing
one lockfile:

- `src/noise_module/` — `modular-noise-simulator` (numpy/scipy only)
- `src/qp_simulator/` — `qp-simulator` (numpy only)
- `src/reconstruction_model/` — `delight-reconstruction` (PyTorch CUDA build)
- `src/tidmad_transformer/` — `tidmad_transformer` (PyTorch CUDA build, plus h5py/scipy/PyYAML)

Install everything with:

```bash
uv sync
```

`reconstruction_model` and `tidmad` pin `torch==2.5.1+cu124` (Linux GPU nodes).
**That wheel does not exist for macOS**, so a plain `uv sync` cannot resolve
those two packages on a Mac. Either install just the noise package, which has no
PyTorch dependency:

```bash
uv sync --package modular-noise-simulator
```

or build a separate CPU environment — see "Running the tests" below.

### Running the tests

The noise package needs nothing external and runs anywhere:

```bash
uv run pytest src/noise_module/tests
```

`latent_monitor` (numpy/scipy only, torch optional) and its protocol layer:

```bash
PYTHONPATH=src python -m pytest src/latent_monitor/tests -q        # 94 passed, 1 skipped without torch, ~15 s
PYTHONPATH=src python -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev_<date>   # ~10 s CPU, dev only, NOT CITABLE
PYTHONPATH=src python -m latent_monitor.run_cumulant_signature --out results/latent_monitor_sig_c3_<date>   # Tier-1 D7 signature row, dev only, NOT CITABLE
PYTHONPATH=src python -m latent_monitor.protocol.smoke --mode confirmatory   # refuses: no freeze, κ_m provisional, dirty tree, no data/model hash
```

The `tidmad` suite needs torch, and on a Linux GPU node `uv run pytest
src/tidmad_transformer/tests` is enough. On macOS the cu124 pin blocks that, so give it its
own CPU environment; `--no-deps` on the two workspace packages is what keeps the
pin out of the way:

```bash
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu --no-config \
    torch numpy jaxtyping h5py scipy pyyaml pytest
uv pip install --python .venv-cpu --no-config --no-deps \
    -e src/reconstruction_model -e src/tidmad_transformer
.venv-cpu/bin/python -m pytest src/tidmad_transformer/tests
```

`--no-config` keeps `[tool.uv.sources]` in this file from redirecting torch back
at the CUDA index; `--no-deps` keeps the cu124 pin out of the resolution when
the two workspace packages go in.

**Invoke the interpreter by path.** Neither of the obvious shortcuts works here:

- bare `pytest` runs whatever is first on `PATH` — on a Mac with MacPorts or
  Homebrew Python that is the system interpreter, which has no torch, and you
  get five collection errors that look like code failures.
- `uv run pytest` re-syncs the project first, which means resolving
  `torch==2.5.1+cu124`, which is the thing that cannot resolve on macOS.

So `.venv-cpu/bin/python -m pytest ...`, or `source .venv-cpu/bin/activate`
first.

A related trap: a `uv sync` that failed on torch still leaves a `.venv` behind
holding only the `dev` group — numpy, pytest, jupyter. The directory exists and
looks like a working environment, but importing `tidmad` from it fails exactly
as the system Python does. `.venv-cpu` is deliberately separate so a later
`uv sync` cannot half-repair it.

One test is skipped without the external `docs/tidmad_data_contract.json`.
`tests/conftest.py` already sets `RECONSTRUCTION_DISABLE_TORCH_COMPILE=1`,
because Muon's Newton-Schulz `torch.compile` path needs a C++/OpenMP toolchain
that macOS does not supply by default; the numerics are identical either way.

The noise-module tutorials live in
[`notebooks/noise_module_tutorial.ipynb`](notebooks/noise_module_tutorial.ipynb) and
[`notebooks/noise_psd_1mhz_resampling_tutorial.ipynb`](notebooks/noise_psd_1mhz_resampling_tutorial.ipynb);
they import the installed package directly (no repository-path probing).

Two further notebooks, checked in **with their outputs**, drive the real
simulators end to end:

* [`notebooks/one_event_herald_lucid.ipynb`](notebooks/one_event_herald_lucid.ipynb) —
  one HeRALD event through HeST → `qp_simulator` → `noise_module` and one
  water-Cherenkov event through LUCiD, then changes one factor at a time
  (physics, geometry, noise, structural) and shows what each does.
* [`notebooks/noise_models_herald_lucid.ipynb`](notebooks/noise_models_herald_lucid.ipynb) —
  what noise each arm simulates, term by term (TES thermal-fluctuation,
  Johnson, SQUID, lines; PMT counting statistics, front-end electronics,
  crate coherence, the alias fold), what it looks like and why.

Re-running them needs the two external simulators, neither of which is
vendored: `src/herald_simulation/fetch_hest.sh` fetches the pinned HeST
commit into `src/herald_simulation/external/`, and a LUCiD clone is found
through `LUCID_PATH` (default `external/LUCiD` at the repository root).
Run from `notebooks/` with `JAX_PLATFORMS=cpu`; the builders
`notebooks/_build_nb1.py` and `_build_nb2.py` regenerate the notebook
sources.

## Build the proposal

A LaTeX installation providing `extarticle`, `tcolorbox`, `booktabs`,
`tabularx`, `microtype`, and the other packages imported by the source is
required. Compile from `latex/` so the figure paths resolve:

```bash
cd latex
pdflatex -interaction=nonstopmode -halt-on-error paper3_proposal.tex
pdflatex -interaction=nonstopmode -halt-on-error paper3_proposal.tex
```

The second pass resolves internal references and the self-contained
bibliography.

## Reproduce the NuBench checks

The NuBench pilot scripts now live in [`scripts/nubench/`](scripts/nubench/)
and the feasibility results (with post-audit caveats) in
[`results/nubench_hexagon_ice_le_dynedge/`](results/nubench_hexagon_ice_le_dynedge/RESULT.md);
both were migrated from the external research folder on 17 August 2026 with
the audit fixes applied (`docs/archive/PAPER3_AUDIT.md`, C11–C17).

The large NuBench database, released prediction Parquet file, and model
checkpoint are external artifacts and are not included in this repository.
The pilot script is designed for the official GraphNeT 1.8.0 CPU environment
and additionally imports PyTorch, PyTorch Geometric, NumPy, pandas, PyArrow,
scikit-learn, and Matplotlib. The metric-only script requires Polars.

Recompute the released-prediction metrics (post-audit: computes BOTH the
`is_track` and `interaction` groupings against Table 6):

```bash
python scripts/nubench/nubench_reference_metrics.py \
  --predictions /path/to/DynEdge_predictions.parquet \
  --output /path/to/reference_metrics.json
```

Run the paired module-dropout pilot (post-audit: uniform-random sampling,
nested severities, alignment assert, paired bootstrap CIs):

```bash
python scripts/nubench/nubench_smoke_pilot.py \
  --database /path/to/hexagon_ice_le.db \
  --checkpoint /path/to/DynEdge_checkpoint.pth \
  --released-predictions /path/to/DynEdge_predictions.parquet \
  --output-dir /path/to/pilot-output
```

The SQLite database must contain the NuBench `pulses_no_noise` and `mc_truth`
tables expected by the script. The pilot defaults to 256 balanced track/cascade
events and dropout fractions of 0, 0.1, 0.25, and 0.5. It writes
`pilot_metrics.csv`, `smoke_test.json`, and PNG/PDF plots. Because GraphNeT's
loader deserializes the released model, only use a checkpoint from a trusted
source.

## Current blocker and next step

Confirm the exact NuBench commit, Python/PyTorch/PyG versions, detector class,
graph-construction backend, and inference command used to produce the released
Hexagon Ice LE DynEdge predictions. Confirmatory experiments should remain
gated until clean re-inference either matches those predictions within an
agreed backend tolerance or the deterministic variation is explained and the
downstream metric passes a predeclared tolerance.

The parallel TIDMAD code path is locally validated on synthetic HDF5 fixtures.
Its remaining blocking gate is the public-data smoke run because the large
TIDMAD files are intentionally external; follow
[`docs/archive/tidmad.md`](docs/archive/tidmad.md).
