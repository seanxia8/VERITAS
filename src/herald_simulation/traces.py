# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Per-sensor arrival times → clean multichannel traces, via ``qp_simulator.QPSimulator``.

Units: HeST reports µs; ``QPSimulator`` takes ns at 2.5e5 Hz × 16384 samples
(65.5 ms). HeST's evaporation window is of order 5 ms, so one conversion and no
resampling. ``t_offset_us`` places the event inside the record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from qp_simulator import QPSimulator

from .events import HeraldEvent


@dataclass(frozen=True)
class TraceConfig:
    sampling_frequency: float = 2.5e5
    trace_samples: int = 16_384
    tau_rise_ns: float = 50e3          # QPSimulator's single-QP template: 50 µs rise, 3 ms decay
    tau_decay_ns: float = 3e6
    t_offset_us: float = 6000.0        # event start inside the record (QPSimulator trigger convention)
    gain_QE: float = 15.0
    E_to_ADC: float = 2.0
    meV_per_QP: float = 1.0
    amplitude_scale: float = 1.0       # e.g. 1/qp_fraction to undo population thinning

    def simulator(self) -> QPSimulator:
        return QPSimulator(sampling_frequency=self.sampling_frequency, trace_samples=self.trace_samples,
                           tau_rise=self.tau_rise_ns, tau_decay=self.tau_decay_ns,
                           gain_QE=self.gain_QE, E_to_ADC=self.E_to_ADC, meV_per_QP=self.meV_per_QP)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def clean_traces(event: HeraldEvent, cfg: TraceConfig = TraceConfig(), sim: QPSimulator | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """(C, N) clean ADC traces, one per sensor, plus per-sensor metadata."""
    sim = cfg.simulator() if sim is None else sim
    C, N = event.n_sensors, cfg.trace_samples
    X = np.zeros((C, N), dtype=float)
    meta = {"n_inside": [], "amplitude_adc": [], "peak_adc": []}
    for c, t_us in enumerate(event.arrival_times_us):
        t_ns = (np.asarray(t_us, dtype=float) + cfg.t_offset_us) * 1e3
        trace, amp, m = sim.generate(t_ns, return_amplitude=True, return_metadata=True)
        X[c] = trace * cfg.amplitude_scale
        meta["n_inside"].append(int(m["n_QP_inside"]))
        meta["amplitude_adc"].append(float(amp * cfg.amplitude_scale))
        meta["peak_adc"].append(float(m["peak_ADC"] * cfg.amplitude_scale))
    return X, meta
