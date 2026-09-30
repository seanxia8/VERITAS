## Introduction

`noise_module_lucid` adds the PMT readout electronics to LUCiD, which stops at the anode: LUCiD gives QE, SPE charge smearing, TTS, dark counts and TDC jitter, all independent per photon and per PMT, with no electronic noise and no inter-channel covariance. The package supplies the front end as a declared, replaceable model — amplifier floor and bandwidth, per-crate clock coherence, cable, digitiser, and event-level pulse processes — and depends on `noise_module` for the spectral and multichannel machinery without forking it. It is MIT (ours); LUCiD's licence is a separate gate.

The three layers are kept apart, since conflating them breaks the acquisition-versus-physics attribution:

| layer | owns | enters `Sigma`? |
|---|---|---|
| signal / background physics | event generators (LUCiD, PhotonSim): DSNB, atmospheric tail, QE vs DIS, radon | no |
| LUCiD event layer | `digitizer.py`: dark counts, TTS, SPE charge, pre/after-pulse | no |
| front-end acquisition | this package: amplifier, clock coherence, cable, digitiser, alias fold | yes (and structural N) |

Module folder: [`src/noise_module_lucid/`](https://github.com/seanxia8/VERITAS/tree/dev/src/noise_module_lucid)
Demo notebook: [`notebooks/noise_module_lucid_tutorial.ipynb`](https://github.com/seanxia8/VERITAS/blob/dev/notebooks/noise_module_lucid_tutorial.ipynb)
