# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Declared readout registry: which photosensor + electronics the preset models.

The front-end preset is **readout-specific**: the SPE template, the sampling
rate, the front-end corner and the clock lines are properties of the sensor and
its electronics, not of the physics. The study's default is a generic 1 GHz PMT
front end (LUCiD's convention, WCTE/SK-like). Other readouts — IceCube DOMs
(ATWD/FADC at 300/40 MS/s), SiPMs, TES calorimeters — need their own measured
reference; this module makes that a declared, replaceable object rather than an
implicit assumption.

Every readout carries ``calibrated`` and ``reference``: a placeholder readout
has ``calibrated=False`` and no reference, and the dataset provenance records
it, so a result can never be silently attributed to a validated detector model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapter import crate_preset
from .presets import (
    FRONT_END_CORNER_HZ,
    FS_L,
    GROUP,
    PMT_PRIVATE,
    PMT_SHARED,
    RMS_MV,
    SPE_MV_PER_PE,
    SPE_TAU_FALL_NS,
    SPE_TAU_RISE_NS,
    spe_bandwidth_hz,
)


@dataclass(frozen=True)
class Readout:
    """A photosensor + front end, with its own SPE, band, clock and calibration state."""

    name: str
    sampling_frequency_hz: float
    spe_tau_rise_ns: float
    spe_tau_fall_ns: float
    mv_per_pe: float
    rms_mv: float
    frontend_corner_hz: float
    frontend_order: float = 4.0
    ringing_hz: float = 1.5e8
    clock_lines_hz: tuple = (6.25e7, 1.25e8, 1.875e8)
    clock_scales: tuple = (3.7, 0.95, 0.35)
    clock_width_hz: float = 2.0e6
    deterministic_clock: bool = False
    group: int = GROUP
    calibrated: bool = False
    reference: str | None = None
    note: str = ""

    # -- construction ---------------------------------------------------------
    def spe(self, length_ns: float = 60.0) -> np.ndarray:
        """The SPE voltage template for this readout."""
        from .units import spe_template

        return spe_template(self.sampling_frequency_hz, self.spe_tau_rise_ns,
                            self.spe_tau_fall_ns, self.mv_per_pe, length_ns)

    def spe_bandwidth_hz(self) -> float:
        return spe_bandwidth_hz(self.sampling_frequency_hz, self.spe_tau_rise_ns,
                                self.spe_tau_fall_ns)

    def private_components(self) -> list[dict]:
        front_end = [
            {"type": "rolloff", "corner_hz": self.frontend_corner_hz, "order": self.frontend_order,
             "kind": "lowpass", "name": "frontend_bandwidth"},
            {"type": "peaking", "center_hz": self.ringing_hz, "half_width_hz": 2.0e7, "gain": 0.5,
             "name": "base_ringing"},
        ]
        return [
            {"type": "filtered", "name": "amplifier_floor",
             "source": {"type": "white", "scale": 1.0}, "filters": front_end},
            {"type": "filtered", "name": "flicker",
             "source": {"type": "powerlaw", "scale": 0.05, "exponent": -1.0, "reference_hz": 1.0e7},
             "filters": front_end},
        ]

    def shared_components(self) -> list[dict]:
        return [{"type": "line", "scale": s, "frequency_hz": f, "width_hz": self.clock_width_hz,
                 "name": f"clock_{f/1e6:.1f}MHz"} for f, s in zip(self.clock_lines_hz, self.clock_scales)]

    def base_crate(self, clock: str | None = None) -> tuple[dict, dict]:
        """``(base_config, crate_config)`` for this readout."""
        clock = ("deterministic" if self.deterministic_clock else "gaussian") if clock is None else clock
        return crate_preset(shared=self.shared_components(), private=self.private_components(),
                            sampling_frequency=self.sampling_frequency_hz,
                            noise_power=self.rms_mv**2, clock=clock)

    def provenance(self) -> dict[str, Any]:
        return {
            "name": self.name, "sampling_frequency_hz": self.sampling_frequency_hz,
            "spe_tau_rise_ns": self.spe_tau_rise_ns, "spe_tau_fall_ns": self.spe_tau_fall_ns,
            "mv_per_pe": self.mv_per_pe, "rms_mv": self.rms_mv,
            "frontend_corner_hz": self.frontend_corner_hz, "frontend_order": self.frontend_order,
            "clock_lines_hz": list(self.clock_lines_hz), "group": self.group,
            "calibrated": self.calibrated, "reference": self.reference, "note": self.note,
        }


#: The default: a generic 1 GHz PMT front end (LUCiD's convention, WCTE/SK-like).
#: All levels are placeholders — ``calibrated=False``.
PMT_1GHZ = Readout(
    name="pmt_1ghz_wcte_placeholder",
    sampling_frequency_hz=FS_L,
    spe_tau_rise_ns=SPE_TAU_RISE_NS,
    spe_tau_fall_ns=SPE_TAU_FALL_NS,
    mv_per_pe=SPE_MV_PER_PE,
    rms_mv=RMS_MV,
    frontend_corner_hz=FRONT_END_CORNER_HZ,
    frontend_order=4.0,
    calibrated=False,
    reference=None,
    note="LUCiD 1 GHz convention; placeholder levels, not a validated PMT front end",
)

#: Known readouts. Only the 1 GHz PMT is fully specified; the others are
#: declared so the arm can say what it does *not* cover.
READOUTS: dict[str, Readout] = {PMT_1GHZ.name: PMT_1GHZ}

#: Readouts named but not modelled here; each needs a measured reference.
UNMODELLED = {
    "icecube_dom": "ATWD/FADC 300/40 MS/s, 10\" PMT; needs the real DOM electronics reference",
    "sipm": "no dynode SPE, different gain/noise; needs a SiPM front-end reference",
    "tes": "calorimeter, not a PMT; see noise_module.tes_budget / herald_simulation",
}


def get_readout(name: str) -> Readout:
    if name not in READOUTS:
        raise KeyError(f"unknown readout {name!r}; known: {sorted(READOUTS)}; unmodelled: {sorted(UNMODELLED)}")
    return READOUTS[name]


def custom_readout(name: str, *, sampling_frequency_hz: float, spe_tau_rise_ns: float,
                   spe_tau_fall_ns: float, mv_per_pe: float, rms_mv: float,
                   bandwidth_ratio: float = 2.5, reference: str | None = None,
                   **kwargs) -> Readout:
    """Build a readout whose front-end corner is derived from its own SPE band.

    Use for a different photosensor once its SPE shape, gain and rms are known.
    ``calibrated`` is set from whether ``reference`` is given.
    """
    from .presets import spe_bandwidth_hz as _bw

    corner = bandwidth_ratio * _bw(sampling_frequency_hz, spe_tau_rise_ns, spe_tau_fall_ns)
    ro = Readout(name=name, sampling_frequency_hz=sampling_frequency_hz,
                 spe_tau_rise_ns=spe_tau_rise_ns, spe_tau_fall_ns=spe_tau_fall_ns,
                 mv_per_pe=mv_per_pe, rms_mv=rms_mv, frontend_corner_hz=corner,
                 calibrated=reference is not None, reference=reference, **kwargs)
    return ro


def register(readout: Readout) -> Readout:
    READOUTS[readout.name] = readout
    return readout
