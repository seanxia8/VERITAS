# Round-2 preflight: WCTE regression and SK X0 (12 Nov 2026)

## WCTE regression (§4.2): PASS
Run on the deepthought2 L40S through `jepa_run.sh`, reading through the new HDF5 reader from a WCTE pack of the two round-1 probe cohorts. That pack was verified byte for byte against 400 runs.

| Model | Full-input vertex | Committed | \|Δ\| |
|---|---:|---:|---:|
| init9000 | 0.16007166 m | 0.16007169 m | 3e-8 m |
| mae_within_trigger9000 (checkpoint `a2b5384f`) | 0.10491558 m | 0.10491560 m | 2e-8 m |

Tolerance is 1e-3 m. File: `wcte_regression.json`.

## SK X0 (`sk.json`): **S2, SK stopped**

**§2.2 values**, recomputed from `sk_geometry.npz`: no difference from the table.
- 11,096 PMTs, pseudo-modules of k = 19 (584 modules).
- L = 49.74 m, t_cross = 220.7 ns, prompt window (−20, 270) ns, time scale 450 ns.
- N = 6, W_rf = 384 / 192.
- The readout window contains prompt ± 50 ns.

**Yield range** (round-1 procedure, 1,000 events, converged):
- Y_lo = 24,904 photons (median hit fraction 0.1001).
- Y_hi = 1,881,272 photons, bound by the 0.95 hit fraction; the 300 PE module-charge bound would be 3.27e6.

**Trigger check:**
- Efficiency 100 % for photon bombs at Y_lo, muons and electrons.
- 1 of 10,000 dark-only 1 µs windows triggered at N = 6. This is recorded, not a stop, because §9 defines S3 by efficiency only. The JSON was written before that code fix and still carries `"stop": "S3"`; the record that governs is `EXECUTION_sk.json`.

**Gate V** (10 holdout PhotonSim μ⁻, LUCiD transport, all 11,096 sensors mapped): **FAIL**.

| Holdout median | Value | Gate | Result |
|---|---:|---|---|
| Yield ratio | 1.186 | 0.5–2 | pass |
| Module pattern | **0.829** | ≥ 0.85 | fail |
| Log-charge pattern | **0.692** | ≥ 0.70 | fail |
| Time residual | 0.78 ns | ≤ 5 ns | pass |

The same harness reproduces the round-1 WCTE holdout exactly. The yield per MeV is not refitted. Under §9 this is **S2: SK is stopped and JUNO and HK continue**.

**Benchmark and projections** (not used, because SK stops):
- 6.44 events/s per process, 3,062 hits/event.
- CPU 0.44 wall-h on 16 cores (remaining cap 29.57 h).
- Disk about 4.6 GB, against 176 GB free on /work and ample space on /ceph; Ceph free entries 39,265.
- GPU: SK main and rf runs now cost 0 h. JUNO and HK keep the design projection (§8) until their smokes are measured.
