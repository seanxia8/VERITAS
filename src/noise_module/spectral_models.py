# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the modular noise simulator written for the ORACLE study.
# If you use this module in published work, please cite it: see CITATION.cff
# at the repository root.
"""Composable one-sided PSD components with explicit physical normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class SpectralComponent:
    """Base PSD component.

    ``normalization='density'`` interprets ``scale`` as a density multiplier.
    ``normalization='power'`` normalizes the component's discrete integral to
    ``scale`` on the requested grid.
    """

    scale: float = 1.0
    normalization: str = "density"
    name: str | None = None

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(
        self, frequencies: np.ndarray, df: float, *, zero_dc: bool = False
    ) -> np.ndarray:
        values = np.clip(np.asarray(self.shape(frequencies), dtype=float), 0.0, None)
        if zero_dc and values.size:
            values[0] = 0.0
        if self.normalization == "density":
            return self.scale * values
        if self.normalization == "power":
            integral = float(np.sum(values) * df)
            if self.scale == 0.0:
                return np.zeros_like(values)
            if integral <= 0.0:
                raise ValueError(f"Component {self.label!r} has zero spectral support.")
            return values * (self.scale / integral)
        raise ValueError("normalization must be 'density' or 'power'.")

    @property
    def label(self) -> str:
        return self.name or self.__class__.__name__.lower()

    def __add__(self, other: "SpectralComponent | CompositeSpectrum") -> "CompositeSpectrum":
        if isinstance(other, CompositeSpectrum):
            return CompositeSpectrum([self, *other.components])
        return CompositeSpectrum([self, other])


@dataclass
class White(SpectralComponent):
    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        return np.ones_like(frequencies, dtype=float)


@dataclass
class PowerLaw(SpectralComponent):
    exponent: float = -1.0
    reference_hz: float = 1.0
    low_cutoff_hz: float | None = None
    high_cutoff_hz: float | None = None

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.exponent) or self.reference_hz <= 0.0:
            raise ValueError("PowerLaw exponent must be finite and reference_hz positive.")
        f = np.asarray(frequencies, dtype=float)
        output = np.zeros_like(f)
        active = f > 0.0
        if self.low_cutoff_hz is not None:
            active &= f >= self.low_cutoff_hz
        if self.high_cutoff_hz is not None:
            active &= f <= self.high_cutoff_hz
        output[active] = (f[active] / self.reference_hz) ** self.exponent
        return output


@dataclass
class Lorentzian(SpectralComponent):
    center_hz: float = 0.0
    half_width_hz: float = 1.0

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        if self.center_hz < 0.0 or self.half_width_hz <= 0.0:
            raise ValueError("Lorentzian center must be non-negative and width positive.")
        return 1.0 / (
            1.0 + ((np.asarray(frequencies) - self.center_hz) / self.half_width_hz) ** 2
        )


@dataclass
class Resonance(Lorentzian):
    """Named Lorentzian resonance component."""


@dataclass
class BandLimited(SpectralComponent):
    low_hz: float = 0.0
    high_hz: float = 1.0

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        if self.low_hz < 0.0 or self.high_hz < self.low_hz:
            raise ValueError("BandLimited requires 0 <= low_hz <= high_hz.")
        f = np.asarray(frequencies)
        return ((f >= self.low_hz) & (f <= self.high_hz)).astype(float)


@dataclass
class RollOff(SpectralComponent):
    corner_hz: float = 1.0
    order: float = 2.0
    kind: str = "lowpass"

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        if self.corner_hz <= 0.0 or self.order <= 0.0:
            raise ValueError("RollOff corner_hz and order must be positive.")
        ratio = np.asarray(frequencies, dtype=float) / self.corner_hz
        if self.kind == "lowpass":
            return 1.0 / (1.0 + ratio**self.order)
        if self.kind == "highpass":
            response = ratio**self.order / (1.0 + ratio**self.order)
            response[0] = 0.0
            return response
        raise ValueError("RollOff kind must be 'lowpass' or 'highpass'.")


@dataclass
class Line(SpectralComponent):
    frequency_hz: float = 1.0
    width_hz: float = 0.0

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        f = np.asarray(frequencies, dtype=float)
        if self.frequency_hz < 0.0 or self.frequency_hz > f[-1]:
            raise ValueError("Line frequency must lie on the one-sided frequency band.")
        if self.width_hz < 0.0:
            raise ValueError("Line width_hz must be non-negative.")
        if self.width_hz == 0.0:
            output = np.zeros_like(f)
            output[int(np.argmin(np.abs(f - self.frequency_hz)))] = 1.0
            return output
        return np.exp(-0.5 * ((f - self.frequency_hz) / self.width_hz) ** 2)


@dataclass
class Peaking(SpectralComponent):
    """Resonant *transfer function* ``|H|^2 = 1 + gain * L(f)``, a unit-gain
    response with a Lorentzian bump at ``center_hz`` (ringing of a base,
    connector or amplifier stage).

    Meant to be used as a filter inside :class:`Filtered`; as a standalone
    additive component it is a white term with a bump, which is rarely what a
    budget means.
    """

    center_hz: float = 1.0
    half_width_hz: float = 1.0
    gain: float = 1.0

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        if self.center_hz < 0.0 or self.half_width_hz <= 0.0 or self.gain < 0.0:
            raise ValueError("Peaking needs center_hz >= 0, half_width_hz > 0, gain >= 0.")
        f = np.asarray(frequencies, dtype=float)
        return 1.0 + self.gain / (1.0 + ((f - self.center_hz) / self.half_width_hz) ** 2)


@dataclass
class Reflection(SpectralComponent):
    """Cable-reflection *transfer function* ``|1 + r exp(-2 pi i f tau)|^2``:
    a comb of period ``1 / delay_s`` (round-trip delay ``tau``), amplitude set by
    the reflection coefficient ``r``. Use inside :class:`Filtered`.
    """

    delay_s: float = 1.0
    reflection: float = 0.1

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        if self.delay_s <= 0.0 or not 0.0 <= self.reflection < 1.0:
            raise ValueError("Reflection needs delay_s > 0 and 0 <= reflection < 1.")
        f = np.asarray(frequencies, dtype=float)
        r = self.reflection
        return 1.0 + r * r + 2.0 * r * np.cos(2.0 * np.pi * f * self.delay_s)


@dataclass
class Filtered(SpectralComponent):
    """A source spectrum shaped by one or more transfer functions.

    ``density(f) = scale * source.scale * source.shape(f) * prod_k filter_k.scale * filter_k.shape(f)``

    ``source`` and every entry of ``filters`` are component configs (or
    instances). Their ``shape`` is read as ``|H_k(f)|^2``; ``RollOff``
    (lowpass / highpass), :class:`Peaking` and :class:`Reflection` are the
    intended filters, but any component works. This is the *multiplicative*
    operator the additive :class:`CompositeSpectrum` lacks: a front-end
    bandwidth belongs here, as a filter on the amplifier floor, not as a
    second additive term (which would leave the floor flat to Nyquist).
    """

    source: Any = None
    filters: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.source is None:
            raise ValueError("Filtered requires a source component.")
        self.source = self.source if isinstance(self.source, SpectralComponent) else component_from_config(self.source)
        raw = self.filters or []
        if not isinstance(raw, (list, tuple)):
            raise ValueError("Filtered.filters must be a list of component configs.")
        self.filters = [c if isinstance(c, SpectralComponent) else component_from_config(c) for c in raw]
        if isinstance(self.source, Filtered) or any(isinstance(c, Filtered) for c in self.filters):
            raise ValueError("Filtered components do not nest; flatten the filter list instead.")

    def shape(self, frequencies: np.ndarray) -> np.ndarray:
        f = np.asarray(frequencies, dtype=float)
        out = self.source.scale * np.asarray(self.source.shape(f), dtype=float)
        for component in self.filters:
            out = out * (component.scale * np.asarray(component.shape(f), dtype=float))
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "source": {"name": self.source.label, "type": self.source.__class__.__name__, "scale": float(self.source.scale)},
            "filters": [
                {"name": c.label, "type": c.__class__.__name__, "scale": float(c.scale)} for c in self.filters
            ],
        }


@dataclass
class CompositeSpectrum:
    components: list[SpectralComponent]

    def __init__(self, components: Iterable[SpectralComponent]):
        self.components = list(components)
        if not self.components:
            raise ValueError("CompositeSpectrum requires at least one component.")

    def __add__(self, other: SpectralComponent | "CompositeSpectrum") -> "CompositeSpectrum":
        if isinstance(other, CompositeSpectrum):
            return CompositeSpectrum([*self.components, *other.components])
        return CompositeSpectrum([*self.components, other])

    def evaluate(
        self, frequencies: np.ndarray, df: float, *, zero_dc: bool = False
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        total = np.zeros_like(frequencies, dtype=float)
        metadata = []
        for component in self.components:
            density = component.evaluate(frequencies, df, zero_dc=zero_dc)
            total += density
            entry = {
                "name": component.label,
                "type": component.__class__.__name__,
                "integrated_power": float(np.sum(density) * df),
                "normalization": component.normalization,
                "scale": float(component.scale),
            }
            describe = getattr(component, "describe", None)
            if callable(describe):
                entry["detail"] = describe()
            metadata.append(entry)
        return total, metadata


_COMPONENT_TYPES = {
    "white": White,
    "powerlaw": PowerLaw,
    "power_law": PowerLaw,
    "lorentzian": Lorentzian,
    "resonance": Resonance,
    "bandlimited": BandLimited,
    "band_limited": BandLimited,
    "rolloff": RollOff,
    "roll_off": RollOff,
    "line": Line,
    "peaking": Peaking,
    "reflection": Reflection,
    "filtered": Filtered,
}


def component_from_config(config: dict[str, Any]) -> SpectralComponent:
    data = dict(config)
    kind = str(data.pop("type")).lower()
    try:
        cls = _COMPONENT_TYPES[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported spectral component type: {kind}") from exc
    return cls(**data)
