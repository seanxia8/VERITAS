# LUCiD install and smoke tests (r2-lucid L1)

## Release
- Repository: https://github.com/CIDeR-ML/LUCiD, branch `main`, cloned 21 Nov 2026 01:43 UTC.
- Commit `e651e836b420f541b02672d9866e6219f2c6d66d` (2026-11-20 02:31 +0100, "Change PhotonSim's build in the image ...").
- Clone: `/ceph/dwong/jepa/external/LUCiD_main_e651e836` (the pinned clone `/ceph/dwong/jepa/external/LUCiD` at `eaf2e89` is untouched).
- Licence statement in the release: code under Apache-2.0 (`LICENSE`, `pyproject.toml: license = "Apache-2.0"`), data release assets under CC BY 4.0 (`README.md`). The runbook records the licence as not finalised for our purposes; LUCiD code and LUCiD-generated data stay on group storage (internal research only).

## Environment
- Conda prefix `/ceph/dwong/envs/lucid_e651e836` (visible from portal1, Deepthought and deepthought2): Python 3.11, `pip install -e ".[all]" -c tests/constraints.txt`, then `pip install "jax[cuda12]==0.6.2" -c tests/constraints.txt`.
- Versions (from `pip freeze`, `/work/dwong/r2_lucid/pip_freeze.txt`): jax 0.6.2, jaxlib 0.6.2, jax-cuda12-plugin/pjrt 0.6.2, CUDA runtime 12.9.79, cuDNN 9.28.1.4, flax 0.10.6, optax 0.2.4, numpy 2.1.3, scipy 1.15.2, h5py 3.13.0.
- Batch: the same versions as a wheelhouse (`/ceph/srv/dwong/r2/lucid/wheels_lucid_cp311.tar`, 3.36 GB, 65 wheels, sha256 `6b59af98…`) installed with `pip --target` inside `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`; the checkout is staged as `lucid/LUCiD_e651e836.tar` (148 MB, sha256 `56473f2a…`).
- Data: `python scripts/download_data.py` (the release ships a Python downloader, not `download_data.sh`): `lucid-data-v1`, 136 MB installed under `data/` (SIREN muon/electron Cherenkov and dE/dx emitters, two GEANT4 muon ROOT files, sensor-overlap tables), verified against `data/MANIFEST.sha256`.

## Smoke tests
| Test | Host | Result | Time |
|---|---|---|---|
| `examples/hello_simulate.py` | Deepthought, 4 CPU cores | `expected charge: 9472 pe over 10764 PMTs`, display written | 64.0 s |
| `examples/hello_simulate.py` | Deepthought Titan X (GPU 0) | fails: `CUDA_ERROR_ILLEGAL_ADDRESS` (driver 525 / Maxwell with the CUDA 12.9 runtime) — Titans unusable | – |
| `jax.devices()` in the batch image | HTCondor A100 (GridKa) | `[CudaDevice(id=0)]` | – |
| LUCiD test suite (`pytest tests`, default `-m 'not slow'`) | Deepthought, 4 CPU cores | 1368 passed, 3 failed, 4 skipped, 180 deselected (slow) | 527 s |
| Production-kernel smoke (`scripts/lucid_generate.py run`, WCTE, 50 events) | HTCondor A100 | 50 events, 5.31 M photons, all triggered | 35.2 s incl. JIT |

The three failures are `tests/fitting/test_source_device_placement.py` (forced host-device count): the test run set `XLA_FLAGS` to cap CPU threads, which overrides the test's own `--xla_force_host_platform_device_count`. They test multi-device placement of the fitting code, which this campaign does not use.

The quickstart logic (one muon through `setup_event_simulator(..., source='particle')`, expected charge and display) is what `hello_simulate.py` runs; the campaign's own generation path is exercised end to end by the production-kernel smoke and by `src/jepa_noise/tests/test_lucid_mc_generator.py` (LUCiD environment: 6 passed, 1 skipped (needs torch); native environment: 6 passed, 1 skipped (needs JAX)).

## Throughput
Production kernel (`source='photons'`, sampled transport and detection, `segment_truth`, K = 12):

| Host | Detector | Photons/s |
|---|---|---|
| Deepthought, 8 CPU cores | SK (32,768-photon buffer, isotropic from centre) | 5.99 × 10⁴ |
| Deepthought, 8 CPU cores | WCTE / SK / JUNO / HK (20,000 photons in a 32,768 buffer) | 3.49 / 3.50 / 3.78 / – × 10⁴ |
| HTCondor A100 | WCTE pilot | see `VALIDATION.md` (filled from the pilot receipts) |

At ≈ 1.5 × 10⁵ photons per event and 1.2 × 10⁵ simulated events per detector, CPU generation would take ≈ 90 h per detector on 8 cores, so mass generation runs on GPUs.
