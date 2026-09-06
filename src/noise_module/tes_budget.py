# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Closed-form noise budget for a transition-edge-sensor (TES) calorimeter channel.

This is the HeRALD-shaped sibling of :class:`~noise_module.reference_budget.AthermalNoiseBudget`.
That budget is a *magnetic* calorimeter (Al2O3:Er with a SQUID readout); HeRALD
(arXiv:2307.11877) reads its quantum-evaporation signal with a TES on Si, so
the terms change while the spectral machinery does not. Every term below is a
textbook form from Irwin & Hilton, *Transition-Edge Sensors* (2005), built
from the existing :mod:`noise_module.spectral_models` components.

Spectral forms (one-sided current-noise PSD referred to the SQUID input, in
readout-units^2 / Hz):

* Thermal-fluctuation noise (TFN) between TES and bath, shaped by the
  responsivity: ``S(f) = S_tfn / (1 + (f / f_eff)^2)``, with
  ``f_eff = 1 / (2 pi tau_eff)``. Dominant in band for a well-designed TES.
* TES Johnson noise with electrothermal-feedback suppression at low frequency,
  the standard two-parameter form
  ``S(f) = S_tes * (1/L_I^2 + (f/f_el)^2) / (1 + (f/f_el)^2)``:
  suppressed by the loop gain ``L_I`` below the electrical corner ``f_el``,
  white above it.
* Shunt (load) Johnson noise: white.
* SQUID amplifier: white floor plus 1/f, crossing at ``squid_knee_hz`` — the
  same two-parameter form the athermal budget uses.
* Narrow lines: mains (fundamental plus harmonics) and vibration /
  microphonics. At the DELight/HeST sampling (2.5e5 Hz, 16384 samples) the
  frequency resolution is 15.3 Hz, so these ARE in band — the opposite of a
  1 GHz PMT digitiser, where a 512 ns record cannot represent 50 Hz at all.
  A line needs to sit at least two bins above DC to be a line rather than a
  drift: the ~1.4 Hz pulse-tube fundamental is therefore *not* a line on this
  grid but slow drift, which belongs to :mod:`noise_module.temporal_noise`.
* No paramagnetic-spin term: that is a magnetic-calorimeter effect.

Provenance is a first-class field. Every constant carries one of three
states — ``placeholder`` (a physically plausible number nobody has measured
for this detector), ``design`` (a design value), ``from_paper`` (read from a
cited source) — and :meth:`TESNoiseBudget.to_dict` records them, so a dataset
generated from a placeholder budget says so.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .spectral_models import CompositeSpectrum, Line, PowerLaw, RollOff, White

__all__ = ["TESNoiseBudget", "HERALD_V1_PLACEHOLDER", "PROVENANCE_STATES"]

PROVENANCE_STATES = ("placeholder", "design", "from_paper")


@dataclass(frozen=True)
class TESNoiseBudget:
    """Closed-form noise budget for one TES calorimeter readout channel."""

    name: str

    # Thermal-fluctuation noise, shaped by the effective thermal time constant.
    tfn_psd: float
    tau_eff_s: float

    # TES Johnson noise: white level, loop gain, electrical corner.
    tes_johnson_psd: float
    loop_gain: float
    tau_el_s: float

    # Shunt / load resistor Johnson noise: white.
    shunt_johnson_psd: float

    # SQUID readout: white floor plus 1/f, crossing at ``squid_knee_hz``.
    squid_white_psd: float
    squid_knee_hz: float

    # Narrow lines, as (frequency_hz, psd_peak, width_hz) triples.
    mains_hz: float = 50.0
    mains_harmonics: int = 3
    mains_line_psd: float = 0.0
    mains_line_width_hz: float = 0.5
    vibration_lines: tuple[tuple[float, float, float], ...] = ()

    # Two-pole signal response (rise = electrical, decay = thermal).
    signal_amplitude: float = 1.0
    signal_rise_s: float | None = None
    signal_decay_s: float | None = None

    #: Provenance state per constant; unspecified constants are ``placeholder``.
    provenance: dict[str, str] = field(default_factory=dict)
    citation: str = "Irwin & Hilton, Transition-Edge Sensors, in Cryogenic Particle Detection (2005)"

    def __post_init__(self) -> None:
        for key in ("tfn_psd", "tes_johnson_psd", "shunt_johnson_psd", "squid_white_psd", "mains_line_psd"):
            if getattr(self, key) < 0.0:
                raise ValueError(f"{key} must be non-negative.")
        for key in ("tau_eff_s", "tau_el_s", "squid_knee_hz", "mains_hz", "mains_line_width_hz"):
            if getattr(self, key) <= 0.0:
                raise ValueError(f"{key} must be positive.")
        if self.loop_gain < 1.0:
            raise ValueError("loop_gain must be >= 1 (no suppression at 1).")
        for freq, _, width in self.vibration_lines:
            if freq <= 0.0 or width <= 0.0:
                raise ValueError("vibration_lines entries need positive frequency and width.")
        for state in self.provenance.values():
            if state not in PROVENANCE_STATES:
                raise ValueError(f"provenance states must be one of {PROVENANCE_STATES}, got {state!r}.")

    # -- derived quantities ---------------------------------------------

    @property
    def f_eff_hz(self) -> float:
        return 1.0 / (2.0 * np.pi * self.tau_eff_s)

    @property
    def f_el_hz(self) -> float:
        return 1.0 / (2.0 * np.pi * self.tau_el_s)

    @property
    def squid_one_over_f_psd(self) -> float:
        return self.squid_white_psd * self.squid_knee_hz

    @property
    def rise_s(self) -> float:
        return self.tau_el_s if self.signal_rise_s is None else self.signal_rise_s

    @property
    def decay_s(self) -> float:
        return self.tau_eff_s if self.signal_decay_s is None else self.signal_decay_s

    def constant_names(self) -> list[str]:
        return [
            "tfn_psd", "tau_eff_s", "tes_johnson_psd", "loop_gain", "tau_el_s",
            "shunt_johnson_psd", "squid_white_psd", "squid_knee_hz",
            "mains_hz", "mains_harmonics", "mains_line_psd", "mains_line_width_hz",
            "vibration_lines", "signal_amplitude", "signal_rise_s", "signal_decay_s",
        ]

    def provenance_record(self) -> dict[str, str]:
        """Every constant with its state; unspecified ones are ``placeholder``."""
        return {k: self.provenance.get(k, "placeholder") for k in self.constant_names()}

    @property
    def has_placeholders(self) -> bool:
        return any(v == "placeholder" for v in self.provenance_record().values())

    # -- spectra ---------------------------------------------------------

    def component_psds(self, frequencies: np.ndarray) -> dict[str, np.ndarray]:
        """One-sided PSD per named term, plus their quadrature sum."""
        f = np.asarray(frequencies, dtype=float)
        r_eff = (f / self.f_eff_hz) ** 2
        r_el = (f / self.f_el_hz) ** 2
        tfn = self.tfn_psd / (1.0 + r_eff)
        tes = self.tes_johnson_psd * (1.0 / self.loop_gain**2 + r_el) / (1.0 + r_el)
        shunt = self.shunt_johnson_psd * np.ones_like(f)
        with np.errstate(divide="ignore"):
            squid = self.squid_white_psd + np.where(f > 0, self.squid_one_over_f_psd / np.where(f > 0, f, 1.0), 0.0)
        lines = np.zeros_like(f)
        for freq, peak, width in self.line_triples():
            lines += peak * np.exp(-0.5 * ((f - freq) / width) ** 2)
        out = {"TFN": tfn, "TES_Johnson": tes, "shunt_Johnson": shunt, "SQUID": squid, "lines": lines}
        out["total"] = tfn + tes + shunt + squid + lines
        return out

    def line_triples(self) -> list[tuple[float, float, float]]:
        triples: list[tuple[float, float, float]] = []
        if self.mains_line_psd > 0.0:
            for k in range(1, int(self.mains_harmonics) + 1):
                triples.append((k * self.mains_hz, self.mains_line_psd / k**2, self.mains_line_width_hz))
        triples.extend((float(a), float(b), float(c)) for a, b, c in self.vibration_lines)
        return triples

    def signal_magnitude(self, frequencies: np.ndarray) -> np.ndarray:
        f = np.asarray(frequencies, dtype=float)
        f_decay = 1.0 / (2.0 * np.pi * self.decay_s)
        f_rise = 1.0 / (2.0 * np.pi * self.rise_s)
        return self.signal_amplitude / np.sqrt((1.0 + (f / f_decay) ** 2) * (1.0 + (f / f_rise) ** 2))

    def to_composite(self) -> CompositeSpectrum:
        """The budget as a :class:`CompositeSpectrum` (exact on any grid)."""
        L2 = self.loop_gain**2
        components: list = [
            RollOff(scale=self.tfn_psd, corner_hz=self.f_eff_hz, order=2.0, kind="lowpass", name="tfn"),
            White(scale=self.tes_johnson_psd / L2, name="tes_johnson_floor"),
            RollOff(scale=self.tes_johnson_psd * (1.0 - 1.0 / L2), corner_hz=self.f_el_hz,
                    order=2.0, kind="highpass", name="tes_johnson_etf"),
            White(scale=self.shunt_johnson_psd, name="shunt_johnson"),
            White(scale=self.squid_white_psd, name="squid_white"),
            PowerLaw(scale=self.squid_one_over_f_psd, exponent=-1.0, reference_hz=1.0, name="squid_1_f"),
        ]
        for i, (freq, peak, width) in enumerate(self.line_triples()):
            components.append(Line(scale=peak, frequency_hz=freq, width_hz=width, name=f"line_{i}_{freq:g}hz"))
        return CompositeSpectrum(components)

    def to_component_dicts(self) -> list[dict[str, Any]]:
        """Serialisable ``components`` list for :class:`~noise_module.config.NoiseConfig`."""
        return [
            {"type": item.__class__.__name__.lower(), **asdict(item)}
            for item in self.to_composite().components
        ]

    def on_grid(self, sampling_frequency: float, n_samples: int) -> "TESNoiseBudget":
        """Return a copy whose lines are representable on the given rFFT grid.

        A Gaussian line narrower than the bin spacing ``df`` can fall between
        two bins and vanish (a 0.5 Hz-wide 50 Hz line on a 15.3 Hz grid does
        exactly that). This widens every line to ``max(width, df)`` while
        holding its *integrated* power fixed, so the budget's total variance is
        grid-independent to first order. Lines below two bins are refused:
        they are drift, not lines.
        """
        from dataclasses import replace

        df = float(sampling_frequency) / int(n_samples)
        for freq, _, _ in self.line_triples():
            if freq < 2.0 * df:
                raise ValueError(
                    f"line at {freq:g} Hz is below two bins (df = {df:g} Hz): model it as drift, not a line."
                )

        def widen(width: float, peak: float) -> tuple[float, float]:
            new_width = max(float(width), df)
            return new_width, float(peak) * float(width) / new_width  # keep peak*width (∝ power) fixed

        mains_w, mains_p = widen(self.mains_line_width_hz, self.mains_line_psd)
        vib = tuple((float(fq), *reversed(widen(w, pk))) for fq, pk, w in self.vibration_lines)
        vib = tuple((fq, pk, w) for fq, w, pk in vib)
        return replace(self, mains_line_width_hz=mains_w, mains_line_psd=mains_p, vibration_lines=vib)

    def to_noise_config(self, sampling_frequency: float, n_samples: int) -> dict[str, Any]:
        """A ``NoiseConfig`` mapping in absolute physical units, lines resolved on the grid.

        ``noise_power`` is set to the budget's own in-band variance on the
        requested grid so the value is informative; with
        ``composite_psd_scaling='absolute'`` it is not used to rescale.
        """
        from scipy.fft import rfftfreq

        budget = self.on_grid(sampling_frequency, n_samples)
        f = rfftfreq(int(n_samples), d=1.0 / float(sampling_frequency))
        df = float(sampling_frequency) / int(n_samples)
        total = budget.component_psds(f)["total"]
        total[0] = 0.0
        return {
            "noise_type": "composite",
            "components": budget.to_component_dicts(),
            "composite_psd_scaling": "absolute",
            "sampling_frequency": float(sampling_frequency),
            "noise_power": float(np.sum(total) * df),
            "power_definition": "variance",
        }

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["provenance"] = self.provenance_record()
        payload["has_placeholders"] = self.has_placeholders
        payload["derived"] = {
            "f_eff_hz": self.f_eff_hz,
            "f_el_hz": self.f_el_hz,
            "squid_one_over_f_psd_at_1hz": self.squid_one_over_f_psd,
            "signal_rise_s": self.rise_s,
            "signal_decay_s": self.decay_s,
            "lines": self.line_triples(),
        }
        return payload

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=list) + "\n")
        return path


#: HeRALD-shaped placeholder budget for a TES on Si reading quantum evaporation.
#:
#: The *structure* is the claim; the *numbers* are not ours to guess. Every
#: constant below is ``placeholder`` except the two time constants, which are
#: ``design`` values matching ``qp_simulator.QPSimulator``'s single-QP template
#: (50 us rise, 3 ms decay) so the noise and the signal share one response.
#: Replace with ``from_paper`` values once read from arXiv:2307.11877 and the
#: HeRALD group's noise characterisation. Units: (ADC)^2 / Hz.
HERALD_V1_PLACEHOLDER = TESNoiseBudget(
    name="HERALD_V1_PLACEHOLDER",
    tfn_psd=4.0e-3,
    tau_eff_s=3.0e-3,
    tes_johnson_psd=1.0e-3,
    loop_gain=10.0,
    tau_el_s=5.0e-5,
    shunt_johnson_psd=2.0e-4,
    squid_white_psd=1.0e-4,
    squid_knee_hz=100.0,
    mains_hz=50.0,
    mains_harmonics=3,
    mains_line_psd=5.0e-3,
    mains_line_width_hz=0.5,
    vibration_lines=((31.0, 2.0e-3, 1.0), (73.0, 1.0e-3, 1.5)),
    signal_amplitude=1.0,
    signal_rise_s=5.0e-5,
    signal_decay_s=3.0e-3,
    provenance={"tau_eff_s": "design", "tau_el_s": "design",
                "signal_rise_s": "design", "signal_decay_s": "design",
                "mains_hz": "design"},
)
