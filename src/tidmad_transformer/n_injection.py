# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Controlled N injection on TIDMAD noise-only records (12 Oct 2026 re-scope, R-5).

Why
---
TIDMAD is a regime where the N–S question has content: the signal sits at or
below the noise floor, overlaps it spectrally and temporally, and no threshold
step removes the noise. Until now the arm used only the real recorded noise, so
it had no *declared* N family. This module corrupts the 208 no-injection
science files (the noise-only records) with declared acquisition families, so
each window carries a known origin and severity. The arm's two-Σ̂ contrast
(WP10) is unchanged; this adds controlled N on top of the unchosen real noise.

Conventions
-----------
* A record is the **float** SQUID readout after the release's ``int8 + 128``
  shift (``data.read_channel_pair``); ``ZERO_LEVEL = 128`` is the analogue zero.
  Multiplicative families (gain) act about the analogue zero, not the mean.
* :func:`requantise_int8` brings an injected record back to the release's
  integer convention and counts saturation (the C1 int8-saturation lesson), so
  an injected window goes through the same reader path as a real one.
* Severities are **relative** to the record's own rms where that is the natural
  scale, so the same cell means the same thing on every file.
* No torch here, and h5py only inside :func:`inject_file`.

Origins follow ``latent_monitor.protocol.labels.ORIGIN``: covariance-type
families are ``N_cov``; structural ones (gain, drift, glitches, jitter, dead
time) are ``N_struct``. The default severity grids are development choices,
not registered (``PREREGISTRATION.md`` §8 item 3 principles: each grid ends
with a point beyond the range a monitor would be calibrated on).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

FS_TIDMAD = 10_000_000.0
ZERO_LEVEL = 128.0


# --------------------------------------------------------------------------- helpers


def _rng(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _record(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError("a TIDMAD record is one-dimensional (one SQUID channel).")
    if a.size < 16:
        raise ValueError("record too short.")
    return a


def _rms(x: np.ndarray) -> float:
    return max(float(np.std(x)), 1e-12)


def _noise(noise_type: str, power: float, n: int, fs: float, rng: np.random.Generator,
           components: list[dict] | None = None) -> tuple[np.ndarray, float]:
    """Zero-mean Gaussian noise of expected variance ``power`` from ``noise_module``.

    Returns the draw and the PSD integral over the exact rFFT grid (the
    declared added variance).
    """
    from noise_module import NoiseGenerator

    cfg: dict[str, Any] = {"sampling_frequency": fs, "noise_type": noise_type,
                           "noise_power": float(power), "power_definition": "variance"}
    if components is not None:
        cfg.update(components=components, composite_psd_scaling="normalize")
    gen = NoiseGenerator(cfg, rng=rng)
    f, s = gen.build_psd_density(n)
    declared = float(np.sum(s[1:]) * fs / n)
    return np.asarray(gen.generate_noise(n), dtype=float), declared


# --------------------------------------------------------------------------- families


def added_coloured_noise(x, rms_ratio: float, fs: float = FS_TIDMAD, rng=None, *,
                         noise_type: str = "pink") -> tuple[np.ndarray, dict]:
    """N_cov: add Gaussian noise of a declared spectrum with std ``rms_ratio`` × record std.

    The realised covariance becomes Σ + Σ_add with Σ_add known exactly.
    """
    x = _record(x)
    if rms_ratio < 0:
        raise ValueError("rms_ratio must be non-negative.")
    power = (rms_ratio * _rms(x)) ** 2
    if power == 0.0:
        return x.copy(), {"declared_added_variance": 0.0, "noise_type": noise_type}
    add, declared = _noise(noise_type, power, x.size, fs, _rng(rng))
    return x + add, {"declared_added_variance": declared, "realised_added_variance": float(np.var(add)),
                     "noise_type": noise_type}


def band_line(x, rms_ratio: float, fs: float = FS_TIDMAD, rng=None, *,
              frequency_hz: float = 2.0e5, width_hz: float = 2.0e3) -> tuple[np.ndarray, dict]:
    """N_cov: a narrow Gaussian line (pickup) inside the analysed band (< ~800 kHz).

    A stationary-Gaussian line overlaps a narrow-band signal in frequency —
    the case where physics-level features cannot easily separate N from S.
    """
    x = _record(x)
    if rms_ratio < 0:
        raise ValueError("rms_ratio must be non-negative.")
    if not 0.0 < frequency_hz < fs / 2:
        raise ValueError("frequency_hz must lie in (0, fs/2).")
    power = (rms_ratio * _rms(x)) ** 2
    if power == 0.0:
        return x.copy(), {"declared_added_variance": 0.0}
    comps = [{"type": "line", "scale": 1.0, "frequency_hz": float(frequency_hz),
              "width_hz": float(width_hz), "name": "injected_line"}]
    add, declared = _noise("composite", power, x.size, fs, _rng(rng), components=comps)
    return x + add, {"declared_added_variance": declared, "realised_added_variance": float(np.var(add)),
                     "frequency_hz": float(frequency_hz), "width_hz": float(width_hz)}


def gain_change(x, gain: float, fs: float = FS_TIDMAD, rng=None, *,
                zero_level: float = ZERO_LEVEL) -> tuple[np.ndarray, dict]:
    """N_struct: the readout gain changes; scales about the analogue zero."""
    x = _record(x)
    if gain < 0:
        raise ValueError("gain must be non-negative.")
    return zero_level + gain * (x - zero_level), {"gain": float(gain), "zero_level": float(zero_level)}


def baseline_drift(x, pp_ratio: float, fs: float = FS_TIDMAD, rng=None, *,
                   kind: str = "random_walk") -> tuple[np.ndarray, dict]:
    """N_struct: a slow baseline drift with peak-to-peak ``pp_ratio`` × record std.

    ``random_walk``: an integrated white sequence, smoothed and scaled (power
    concentrated at the lowest frequencies); ``linear``: a ramp.
    """
    x = _record(x)
    if pp_ratio < 0:
        raise ValueError("pp_ratio must be non-negative.")
    n = x.size
    if kind == "linear":
        d = np.linspace(0.0, 1.0, n)
    elif kind == "random_walk":
        d = np.cumsum(_rng(rng).normal(size=n))
        k = max(n // 64, 1)                                  # smooth away sample-scale steps
        d = np.convolve(np.pad(d, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]
    else:
        raise ValueError("kind must be 'random_walk' or 'linear'.")
    d = d - d[0]
    span = float(np.ptp(d))
    target = pp_ratio * _rms(x)
    d = d * (target / span) if span > 0 else d
    return x + d, {"kind": kind, "peak_to_peak": float(np.ptp(d))}


def glitches(x, rate_hz: float, fs: float = FS_TIDMAD, rng=None, *,
             amp_rms: float = 5.0, duration_samples: tuple[int, int] = (32, 256)) -> tuple[np.ndarray, dict]:
    """N_struct: transient glitches at ``rate_hz`` with amplitude ``amp_rms`` × baseline rms.

    Uses ``noise_module.ArtifactInjector`` (impulse / exponential decay / damped
    sine templates, homogeneous Poisson starts, truncation at the edge).
    """
    from noise_module import ArtifactInjector

    x = _record(x)
    if rate_hz < 0:
        raise ValueError("rate_hz must be non-negative.")
    cfg = {"sampling_frequency": fs, "enable_glitches": True, "glitch_rate": float(rate_hz),
           "glitch_amp_range": [float(amp_rms), float(amp_rms)],
           "glitch_duration_samples": [int(duration_samples[0]), int(duration_samples[1])],
           "amplitude_unit": "baseline_rms"}
    y, meta = ArtifactInjector(cfg, rng=_rng(rng)).apply(x, return_metadata=True)
    g = meta["glitches"]
    return y, {"rate_hz": float(rate_hz), "amp_rms": float(amp_rms), "count": int(g["count"]),
               "expected_count": float(g["expected_count"]),
               "starts": [int(e["start"]) for e in g["glitches"]]}


def sampling_jitter(x, sigma_samples: float, fs: float = FS_TIDMAD, rng=None) -> tuple[np.ndarray, dict]:
    """N_struct: random sampling-instant error, first order: ``y = x + x'(t) δt``.

    ``δt ~ N(0, sigma_samples)`` per sample (aperture/slew-rate form, valid for
    any bandwidth; it adds noise proportional to the local slope).
    """
    x = _record(x)
    if sigma_samples < 0:
        raise ValueError("sigma_samples must be non-negative.")
    slope = np.gradient(x)                                    # per sample
    dt = _rng(rng).normal(0.0, sigma_samples, size=x.size)
    return x + slope * dt, {"sigma_samples": float(sigma_samples), "sigma_s": float(sigma_samples / fs)}


def dropout(x, fraction: float, fs: float = FS_TIDMAD, rng=None, *, n_segments: int = 4,
            zero_level: float = ZERO_LEVEL) -> tuple[np.ndarray, dict]:
    """N_struct: dead time — ``n_segments`` non-overlapping gaps covering ``fraction`` of the record.

    Gaps read the analogue zero (the DAQ delivers no signal and no noise).
    """
    x = _record(x)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must lie in [0, 1).")
    n = x.size
    total = int(round(fraction * n))
    y = x.copy()
    if total == 0:
        return y, {"fraction": 0.0, "segments": []}
    n_segments = max(1, min(n_segments, total))
    lengths = np.full(n_segments, total // n_segments)
    lengths[: total % n_segments] += 1
    free = n - total                                          # samples outside the gaps
    cuts = np.sort(_rng(rng).choice(free + 1, size=n_segments, replace=True))
    segments, offset = [], 0
    for c, L in zip(cuts, lengths):
        start = int(c + offset)
        y[start:start + int(L)] = zero_level
        segments.append([start, int(L)])
        offset += int(L)
    return y, {"fraction": float(total / n), "segments": segments}


# --------------------------------------------------------------------------- registry

#: name -> (origin, function, severity parameter, development grid)
FAMILIES: dict[str, dict[str, Any]] = {
    "added_pink_noise": {"origin": "N_cov", "fn": added_coloured_noise, "parameter": "rms_ratio",
                         "grid": [0.1, 0.25, 0.5, 1.0]},
    "band_line": {"origin": "N_cov", "fn": band_line, "parameter": "rms_ratio",
                  "grid": [0.05, 0.1, 0.25, 0.5]},
    "gain_change": {"origin": "N_struct", "fn": gain_change, "parameter": "gain",
                    "grid": [0.8, 0.9, 1.1, 1.25]},
    "baseline_drift": {"origin": "N_struct", "fn": baseline_drift, "parameter": "pp_ratio",
                       "grid": [0.5, 1.0, 2.0, 4.0]},
    "glitches": {"origin": "N_struct", "fn": glitches, "parameter": "rate_hz",
                 "grid": [10.0, 50.0, 200.0, 1000.0]},
    "sampling_jitter": {"origin": "N_struct", "fn": sampling_jitter, "parameter": "sigma_samples",
                        "grid": [0.01, 0.05, 0.1, 0.2]},
    "dropout": {"origin": "N_struct", "fn": dropout, "parameter": "fraction",
                "grid": [0.01, 0.05, 0.1, 0.2]},
}


@dataclass(frozen=True)
class NCell:
    """One injected cell: a family at a severity, with a seed and optional extra parameters."""

    family: str
    severity: float
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{FAMILIES[self.family]['origin']}:{self.family}@{self.severity:g}"


def default_cells(seed: int = 0) -> list[NCell]:
    """Every family at every development severity (unregistered)."""
    return [NCell(name, float(s), seed) for name, spec in FAMILIES.items() for s in spec["grid"]]


# --------------------------------------------------------------------------- quantisation and drivers


def requantise_int8(x, *, zero_level: float = ZERO_LEVEL) -> tuple[np.ndarray, dict]:
    """Back to the release convention: ``round(x - 128)`` clipped to int8, then ``+128``.

    Returns the float record as the reader would deliver it and the
    saturation fraction (samples clipped at -128 or +127).
    """
    a = np.asarray(x, dtype=float)
    r = np.round(a - zero_level)
    clipped = (r < -128) | (r > 127)
    q = np.clip(r, -128, 127).astype(np.int8)
    return q.astype(float) + zero_level, {"saturation_fraction": float(clipped.mean()),
                                          "n_saturated": int(clipped.sum())}


def inject(record, cell: NCell, fs: float = FS_TIDMAD, *, requantise: bool = True) -> tuple[np.ndarray, dict]:
    """Apply one cell to a record; return the corrupted record and its provenance."""
    if cell.family not in FAMILIES:
        raise KeyError(f"unknown family {cell.family!r}; declared: {sorted(FAMILIES)}")
    spec = FAMILIES[cell.family]
    x = _record(record)
    rng = np.random.default_rng(cell.seed)
    y, meta = spec["fn"](x, cell.severity, fs, rng, **cell.params)
    prov: dict[str, Any] = {"cell": cell.label, "family": cell.family, "origin": spec["origin"],
                            "parameter": spec["parameter"], "severity": float(cell.severity),
                            "seed": int(cell.seed), "params": dict(cell.params), "fs": float(fs),
                            "input_rms": _rms(x), "family_meta": meta}
    if requantise:
        y, qmeta = requantise_int8(y)
        prov["requantised"] = True
        prov.update(qmeta)
    else:
        prov["requantised"] = False
    from noise_module import __version__ as nm_version

    prov["noise_module_version"] = nm_version
    return y, prov


def read_noise_only(path, dataset: str, start: int, length: int, *, shift: float = ZERO_LEVEL) -> np.ndarray:
    """Read one window of a noise-only file's SQUID channel, shifted as upstream (``int8 + 128``)."""
    import h5py

    with h5py.File(path, "r") as f:
        raw = np.asarray(f[dataset][start:start + length], dtype=np.float64)
    if raw.size != length:
        raise ValueError(f"requested {length} samples from {start}, file holds only {raw.size}.")
    return raw + shift


def inject_file(path, dataset: str, start: int, length: int, cell: NCell, fs: float = FS_TIDMAD, *,
                requantise: bool = True) -> tuple[np.ndarray, dict]:
    """Read a noise-only window and inject one cell; provenance names the file and window."""
    x = read_noise_only(path, dataset, start, length)
    y, prov = inject(x, cell, fs, requantise=requantise)
    prov.update({"file": str(path), "dataset": dataset, "start": int(start), "length": int(length)})
    return y, prov


def describe() -> dict[str, str]:
    """Family name -> origin."""
    return {name: spec["origin"] for name, spec in FAMILIES.items()}


__all__ = [
    "FS_TIDMAD", "ZERO_LEVEL", "FAMILIES", "NCell", "default_cells", "describe",
    "added_coloured_noise", "band_line", "gain_change", "baseline_drift", "glitches",
    "sampling_jitter", "dropout", "requantise_int8", "inject", "read_noise_only", "inject_file",
]
