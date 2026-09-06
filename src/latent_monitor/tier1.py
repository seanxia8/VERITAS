# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Tier-1 (ORACLE-Cov) paired cells: one factor moved at a time, event-paired.

A *cell* is ``(geometry, noise model, event family)``. The reference cell is
``(G₀, Σ̂₀ = Σ₀, E₀)``. Every other cell moves exactly one of the three and is
paired to the reference by ``event_id``: the planted signal parameters are a
deterministic function of ``event_id``, and — where the noise structure allows
— the underlying white draws are too (common random numbers), so a per-event
Δz isolates the factor that moved.

Every cell also carries *noise-only* records (a detector's random triggers):
the whitened-variance statistic and the re-whitening adjustment use them, and
they are the only honest way to measure a covariance change with a tied
decoder, whose in-span residual is zero by construction.

Planted signal: a shared two-component pulse ``templates.pulse_template_2``
seen by every channel with a geometry-tied gain ``g(pos)``. Targets are the
planted ``(amplitude, t0, tau_t)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from noise_module import MultiChannelNoiseGenerator, NoiseGenerator
from noise_module.templates import pulse_template_2

from .subject import Geometry
from .whitening import KroneckerWhitener

TARGET_NAMES = ("amplitude", "t0", "tau_t")


def grid_geometry(n_channels: int, box: float = 1.0, name: str = "grid") -> Geometry:
    """``n_channels`` sensors on a square grid filling the same ``box`` — the granularity axis."""
    side = int(np.ceil(np.sqrt(n_channels)))
    xs = np.linspace(-box / 2, box / 2, side)
    pts = np.array([[x, y, 0.0] for y in xs for x in xs])[:n_channels]
    groups = np.arange(n_channels) // max(1, n_channels // 4)
    return Geometry(positions=pts, groups=groups, name=f"{name}_{n_channels}")


def geometry_gain(geometry: Geometry) -> np.ndarray:
    """Smooth, geometry-tied per-channel signal gain in (0.8, 1.2)."""
    p = geometry.positions
    return 1.0 + 0.2 * np.sin(2.0 * np.pi * p[:, 0]) * np.cos(np.pi * p[:, 1])


@dataclass(frozen=True)
class NoiseModel:
    """A noise model = base PSD config + multichannel structure."""

    base: dict[str, Any]
    corr_strength: float = 0.4
    mode: str = "shared_private"
    n_latent: int = 2
    channel_gain_jitter: float = 0.05
    label: str = "reference"

    def generator(self, C: int, seed: int) -> MultiChannelNoiseGenerator:
        gen = MultiChannelNoiseGenerator(
            self.base,
            {"mode": self.mode, "n_channels": C, "corr_strength": self.corr_strength,
             "n_latent": self.n_latent, "channel_gain_jitter": self.channel_gain_jitter,
             "freeze_channel_structure": True, "normalize_channel_variance": False},
            seed=seed,
        )
        return gen

    def implied(self, C: int, N: int, structure_seed: int) -> tuple[np.ndarray, np.ndarray]:
        """(Σ_c with unit mean diagonal, one-sided PSD) — the Kronecker factors."""
        gen = self.generator(C, structure_seed)
        _, meta = gen.generate(N, return_metadata=True)
        Sc = meta["implied_covariance"]
        Sc = Sc / np.mean(np.diag(Sc))
        _, psd = NoiseGenerator(self.base).build_psd_density(N)
        return Sc, psd

    def whitener(self, C: int, N: int, structure_seed: int) -> KroneckerWhitener:
        Sc, psd = self.implied(C, N, structure_seed)
        return KroneckerWhitener(Sc, psd, float(self.base["sampling_frequency"]), N)


@dataclass(frozen=True)
class EventFamily:
    """The planted-signal family. ``extra`` adds an out-of-support component."""

    amplitude_range: tuple[float, float] = (3.0, 6.0)
    t0_range: tuple[float, float] = (0.30, 0.40)       # fraction of the record
    tau_t_range: tuple[float, float] = (0.06, 0.10)    # fraction of the record
    tau_n: float = 0.006
    tau_in: float = 0.002
    An_frac: float = 0.3
    extra: str | None = None                            # None | "glitch" | "oscillation" | "double_pulse"
    extra_scale: float = 1.0
    label: str = "reference"

    def draw(self, event_id: int) -> dict[str, float]:
        rng = np.random.default_rng([event_id, 0xE7])
        return {
            "amplitude": float(rng.uniform(*self.amplitude_range)),
            "t0": float(rng.uniform(*self.t0_range)),
            "tau_t": float(rng.uniform(*self.tau_t_range)),
        }

    def waveform(self, event_id: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """(shared waveform (N,), targets (3,)) in record-fraction time units."""
        p = self.draw(event_id)
        t = np.arange(n_samples) / n_samples
        w = pulse_template_2(t, p["t0"], self.An_frac * p["amplitude"], p["amplitude"],
                             self.tau_n, self.tau_in, p["tau_t"])
        if self.extra == "oscillation":
            rng = np.random.default_rng([event_id, 0x05C])
            f0 = rng.uniform(8.0, 14.0)
            w = w + self.extra_scale * p["amplitude"] * 0.5 * np.exp(-(t - p["t0"]) / 0.15) * (t >= p["t0"]) * np.sin(2 * np.pi * f0 * (t - p["t0"]))
        elif self.extra == "glitch":
            # a fast transient (3-sample damped impulse) at a random time: broadband, so after
            # whitening by a low-pass noise model its energy sits where the noise is weak — OUTSIDE
            # the smooth pulse subspace the encoder spans. This is the out-of-support family.
            rng = np.random.default_rng([event_id, 0x611])
            i0 = int(rng.integers(int(0.1 * n_samples), int(0.9 * n_samples)))
            g = np.zeros(n_samples); g[i0] = 1.0
            if i0 + 1 < n_samples: g[i0 + 1] = -0.6
            if i0 + 2 < n_samples: g[i0 + 2] = 0.2
            w = w + self.extra_scale * 2.0 * p["amplitude"] * g
        elif self.extra == "double_pulse":
            w = w + self.extra_scale * pulse_template_2(t, min(p["t0"] + 0.2, 0.9), self.An_frac * p["amplitude"], p["amplitude"],
                                                        self.tau_n, self.tau_in, p["tau_t"])
        return w, np.array([p["amplitude"], p["t0"], p["tau_t"]])


@dataclass(frozen=True)
class Cell:
    geometry: Geometry
    noise: NoiseModel
    family: EventFamily
    n_samples: int
    label: str
    structure_seed: int = 11
    #: per-channel multiplicative gain drift (structural N); None = none
    gain_drift: np.ndarray | None = None
    #: boolean mask of live channels (structural N: channel loss); None = all live
    live_mask: np.ndarray | None = None
    #: per-channel timing jitter in samples (structural N); None = none
    timing_jitter: np.ndarray | None = None
    moved: str = "none"                                # which factor moved: none|sigma_cov|sigma_struct|geometry|event

    def _apply_structural(self, X: np.ndarray) -> np.ndarray:
        if self.timing_jitter is not None:
            X = np.stack([np.roll(X[c], int(self.timing_jitter[c])) for c in range(X.shape[0])])
        if self.gain_drift is not None:
            X = X * self.gain_drift[:, None]
        if self.live_mask is not None:
            X = X * self.live_mask[:, None]
        return X

    def event(self, event_id: int, replicate: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """One event ``(X (C, N), targets (3,))`` — signal + noise, paired by ``event_id``.

        ``replicate`` selects an independent noise realisation of the *same*
        event; replicate 0 is the pairing partner across cells (common random
        numbers), replicates ≥ 1 give the reference-vs-reference null.
        """
        C = self.geometry.n_channels
        w, targets = self.family.waveform(event_id, self.n_samples)
        signal = geometry_gain(self.geometry)[:, None] * w[None, :]
        gen = self.noise.generator(C, self.structure_seed)
        gen.generate(8, return_metadata=False)                  # pin the structure at this seed
        gen.rng = np.random.default_rng([event_id, 0x01, int(replicate)])
        noise = gen.generate(self.n_samples)
        return self._apply_structural(signal + noise), targets

    def noise_record(self, record_id: int) -> np.ndarray:
        C = self.geometry.n_channels
        gen = self.noise.generator(C, self.structure_seed)
        gen.generate(8, return_metadata=False)
        gen.rng = np.random.default_rng([record_id, 0x02])
        return self._apply_structural(gen.generate(self.n_samples))

    def batch(self, event_ids: np.ndarray, replicate: int = 0) -> tuple[np.ndarray, np.ndarray]:
        Xs, Ts = zip(*(self.event(int(i), replicate) for i in event_ids))
        return np.stack(Xs), np.stack(Ts)

    def noise_batch(self, record_ids: np.ndarray) -> np.ndarray:
        return np.stack([self.noise_record(int(i)) for i in record_ids])

    def implied_whitener(self) -> KroneckerWhitener:
        return self.noise.whitener(self.geometry.n_channels, self.n_samples, self.structure_seed)


# ----------------------------------------------------------------------- presets

def default_noise_base(sampling_frequency: float = 1.0e4, noise_power: float = 1.0) -> dict[str, Any]:
    return {
        "noise_type": "composite", "sampling_frequency": sampling_frequency, "noise_power": noise_power,
        "power_definition": "variance", "composite_psd_scaling": "normalize",
        "components": [
            {"type": "white", "scale": 0.3, "name": "floor"},
            {"type": "rolloff", "scale": 1.0, "corner_hz": 0.15 * sampling_frequency, "order": 2.0, "kind": "lowpass", "name": "bandwidth"},
            {"type": "powerlaw", "scale": 0.05, "exponent": -1.0, "reference_hz": 0.01 * sampling_frequency, "name": "flicker"},
        ],
    }


def reference_cell(n_channels: int = 8, n_samples: int = 256, corr_strength: float = 0.4) -> Cell:
    return Cell(
        geometry=grid_geometry(n_channels, name="ref"),
        noise=NoiseModel(base=default_noise_base(), corr_strength=corr_strength, label="reference"),
        family=EventFamily(),
        n_samples=n_samples,
        label="reference",
    )


def sigma_covariance_cells(ref: Cell) -> list[Cell]:
    """Σ̂ ≠ Σ, covariance-type: the *noise model* moves, nothing else."""
    base = ref.noise.base
    stronger = replace(ref.noise, corr_strength=min(ref.noise.corr_strength + 0.35, 0.9), label="corr_up")
    weaker = replace(ref.noise, corr_strength=max(ref.noise.corr_strength - 0.35, 0.0), label="corr_down")
    comps = [dict(c) for c in base["components"]]
    comps[1] = {**comps[1], "corner_hz": comps[1]["corner_hz"] * 0.4}
    bandwidth = replace(ref.noise, base={**base, "components": comps}, label="bandwidth_down")
    line = replace(ref.noise, base={**base, "components": base["components"] + [
        # density scale 12 over ~2.5 bins ≈ a quarter of the record's noise power in one narrow line
        {"type": "line", "scale": 12.0, "frequency_hz": 0.08 * base["sampling_frequency"], "width_hz": 0.004 * base["sampling_frequency"], "name": "pickup"}]},
        label="line_pickup")
    return [replace(ref, noise=nm, label=f"sigma_cov:{nm.label}", moved="sigma_cov") for nm in (stronger, weaker, bandwidth, line)]


def sigma_structural_cells(ref: Cell, seed: int = 3) -> list[Cell]:
    """N by contract, mean-shift in the latent: gain drift, channel loss, timing jitter."""
    rng = np.random.default_rng(seed)
    C = ref.geometry.n_channels
    drift = 1.0 + rng.normal(0.0, 0.15, size=C)
    mask = np.ones(C, dtype=bool); mask[rng.choice(C, size=max(1, C // 4), replace=False)] = False
    jitter = rng.integers(-6, 7, size=C)
    return [
        replace(ref, gain_drift=drift, label="sigma_struct:gain_drift", moved="sigma_struct"),
        replace(ref, live_mask=mask, label="sigma_struct:channel_loss", moved="sigma_struct"),
        replace(ref, timing_jitter=jitter, label="sigma_struct:timing_jitter", moved="sigma_struct"),
    ]


def geometry_cells(ref: Cell, counts: tuple[int, ...] | None = None) -> list[Cell]:
    """Granularity at fixed active volume: same box, half and double the sensor count."""
    if counts is None:
        c0 = ref.geometry.n_channels
        counts = (max(2, c0 // 2), 2 * c0)
    return [replace(ref, geometry=grid_geometry(c, name="geom"), label=f"geometry:{c}ch", moved="geometry") for c in counts]


def event_cells(ref: Cell) -> list[Cell]:
    """Event-type cells.

    ``event`` is *outside* the excited support (a broadband glitch the smooth
    pulse basis cannot represent). ``event_in_span`` families are supported by
    the representation but not by the output head — the "supported-but-rare
    physics" (S) case the 2026-09-02 Tier-1 review said must be told apart from
    N: they move z, leave the residual and every noise-only statistic alone,
    and raise the consequence.
    """
    return [
        replace(ref, family=replace(ref.family, extra="glitch", label="glitch"), label="event:glitch", moved="event"),
        replace(ref, family=replace(ref.family, extra="oscillation", label="oscillation"), label="event_in_span:oscillation", moved="event_in_span"),
        replace(ref, family=replace(ref.family, extra="double_pulse", label="double_pulse"), label="event_in_span:double_pulse", moved="event_in_span"),
    ]


def all_cells(ref: Cell) -> list[Cell]:
    return sigma_covariance_cells(ref) + sigma_structural_cells(ref) + geometry_cells(ref) + event_cells(ref)
