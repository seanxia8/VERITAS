# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""The feature-availability manifest and the alarm-time information contract (R2.1–R2.2; I2).

Every quantity a monitor could consume is declared once with the phase at
which it becomes available:

    reference_fit    fitted on clean reference windows before deployment
                     (Σ̂, projectors, null quantiles, the classifier itself)
    alarm_time       measurable on the observed window when the alarm is raised
    delayed_label    arrives later (task metrics on labelled events, replay)
    evaluation_only  never available in deployment (truth, realised Σ, the
                     paired clean twin, the intervention label)

**Two layers of enforcement (I2).** The *manifest* is declarative: names carry
a phase, and :class:`AlarmTimeContract` refuses an arm that names a feature
outside its allowed phases. That cannot catch an implementation that computes
an allowed name from forbidden data. The *data-flow* layer therefore makes the
forbidden data unreachable: alarm-time features are built only through
:class:`AlarmTimeInputs`, a typed input object that carries the raw window, its
random-trigger records and the geometry — and nothing else — and every value
that enters a :class:`FeatureBatch` must carry a source tag drawn from that
object's tags. A feature tagged with anything else raises at insertion, and an
arm is scored only from a locked batch whose tags all lie in the allowed set.

**Residual limitation, stated honestly.** A builder that *lies* about a source
tag is not caught by either layer; the tags are as trustworthy as the code that
sets them. What the layers buy is that truth, the twin and the realised
covariance are not *arguments* of any alarm-time builder, so smuggling them in
requires a visible violation of the builder signature that code review will
see (``tests/test_protocol.py::test_adversarial_builder_cannot_smuggle_truth``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

PHASES = ("reference_fit", "alarm_time", "delayed_label", "evaluation_only")

#: source tags an alarm-time builder may use
ALARM_TIME_SOURCES = frozenset({"raw_window", "noise_record", "reference_fit"})
#: source tags that are evaluation-side by definition
FORBIDDEN_AT_ALARM_TIME = frozenset({"truth", "twin", "sigma_realized", "intervention_label", "delayed_label"})


class LeakageError(RuntimeError):
    """A feature is consumed at a phase before it is available, or built from a forbidden source."""


@dataclass(frozen=True)
class Feature:
    name: str
    availability: str                 # one of PHASES
    source: str                       # human description
    operational: bool = True          # False ⇒ the acquisition does not supply it; consuming it is privileged
    group: str = "generic"            # feature group the arms refer to
    note: str = ""

    def __post_init__(self) -> None:
        if self.availability not in PHASES:
            raise ValueError(f"availability must be one of {PHASES}, got {self.availability!r}")


@dataclass
class FeatureManifest:
    arm_name: str
    acquisition: dict = field(default_factory=dict)      # e.g. {"supplies_noise_only_records": True}
    features: dict[str, Feature] = field(default_factory=dict)

    def add(self, f: Feature) -> "FeatureManifest":
        if f.name in self.features:
            raise ValueError(f"feature {f.name!r} declared twice")
        self.features[f.name] = f
        return self

    def group(self, name: str) -> list[str]:
        return [f.name for f in self.features.values() if f.group == name]

    def phase_of(self, name: str) -> str:
        if name not in self.features:
            raise KeyError(f"feature {name!r} is not in the manifest — undeclared features are leakage by default")
        return self.features[name].availability

    def to_dict(self) -> dict:
        return {"arm": self.arm_name, "acquisition": dict(self.acquisition),
                "features": {k: vars(v) for k, v in self.features.items()}}


# --------------------------------------------------------------------------- data-flow layer

@dataclass(frozen=True)
class AlarmTimeInputs:
    """The only inputs an alarm-time feature builder receives. No truth, twin, label or realised Σ field exists."""

    X: np.ndarray                          # (n, C, N) observed windows
    geometry: object                       # latent_monitor.subject.Geometry
    noise_records: np.ndarray | None = None  # (n, m, C, N) random-trigger records supplied with each window, or None

    @property
    def tags(self) -> frozenset:
        t = {"raw_window", "reference_fit"}
        if self.noise_records is not None:
            t.add("noise_record")
        return frozenset(t)

    @property
    def n(self) -> int:
        return int(np.asarray(self.X).shape[0])


class FeatureBatch:
    """Feature arrays with a source tag per feature; insertion refuses forbidden sources."""

    def __init__(self, allowed_sources: Iterable[str] = ALARM_TIME_SOURCES, n: int | None = None):
        self.allowed = frozenset(allowed_sources)
        self._values: dict[str, np.ndarray] = {}
        self._sources: dict[str, frozenset] = {}
        self._n = n
        self._locked = False

    def add(self, name: str, values: np.ndarray, sources: Iterable[str] | str) -> "FeatureBatch":
        if self._locked:
            raise RuntimeError("feature batch is locked")
        src = frozenset([sources] if isinstance(sources, str) else sources)
        if not src:
            raise LeakageError(f"feature {name!r} has no source tag")
        bad = src - self.allowed
        if bad:
            raise LeakageError(f"feature {name!r} is built from forbidden source(s) {sorted(bad)}; allowed: {sorted(self.allowed)}")
        v = np.asarray(values, dtype=float).reshape(-1)
        if self._n is None:
            self._n = len(v)
        elif len(v) != self._n:
            raise ValueError(f"feature {name!r} has {len(v)} rows, batch has {self._n}")
        if name in self._values:
            raise ValueError(f"feature {name!r} added twice")
        self._values[name] = v
        self._sources[name] = src
        return self

    def lock(self) -> "FeatureBatch":
        self._locked = True
        return self

    @property
    def locked(self) -> bool:
        return self._locked

    def names(self) -> list[str]:
        return list(self._values)

    def sources(self, name: str) -> frozenset:
        return self._sources[name]

    def __getitem__(self, name: str) -> np.ndarray:
        return self._values[name]

    def __contains__(self, name: str) -> bool:
        return name in self._values

    def subset(self, rows: np.ndarray) -> "FeatureBatch":
        b = FeatureBatch(self.allowed)
        for k, v in self._values.items():
            b.add(k, v[rows], self._sources[k])
        return b.lock() if self._locked else b

    def concat(self, other: "FeatureBatch") -> "FeatureBatch":
        b = FeatureBatch(self.allowed | other.allowed)
        for k in self._values:
            b.add(k, np.concatenate([self._values[k], other._values[k]]), self._sources[k] | other._sources[k])
        return b

    def assert_sources_within(self, allowed: Iterable[str], names: Iterable[str] | None = None) -> None:
        allowed = frozenset(allowed)
        for k in (names if names is not None else self._values):
            bad = self._sources[k] - allowed
            if bad:
                raise LeakageError(f"feature {k!r} carries source(s) {sorted(bad)} outside {sorted(allowed)}")

    def matrix(self, names: Iterable[str]) -> np.ndarray:
        return np.column_stack([self._values[n] for n in names])

    def to_dict(self) -> dict[str, np.ndarray]:
        return dict(self._values)


# --------------------------------------------------------------------------- declarative layer

@dataclass(frozen=True)
class AlarmTimeContract:
    """What an alarm-time arm may consume: ``alarm_time`` features only, built from alarm-time sources only."""

    manifest: FeatureManifest
    allowed: tuple[str, ...] = ("alarm_time",)
    allowed_sources: frozenset = ALARM_TIME_SOURCES

    def check(self, arm: str, feature_names: Iterable[str], batch: FeatureBatch | None = None) -> dict:
        """Raise :class:`LeakageError` on any feature outside the allowed phases or sources; return the arm's status."""
        names = list(feature_names)
        bad = []
        privileged = []
        for n in names:
            ph = self.manifest.phase_of(n)
            if ph not in self.allowed:
                bad.append((n, ph))
            if not self.manifest.features[n].operational:
                privileged.append(n)
        if bad:
            raise LeakageError(
                f"arm {arm!r} consumes {len(bad)} feature(s) outside {self.allowed}: "
                + ", ".join(f"{n} [{ph}]" for n, ph in bad)
            )
        if batch is not None:
            if not batch.locked:
                raise LeakageError(f"arm {arm!r}: the feature batch must be locked before scoring")
            missing = [n for n in names if n not in batch]
            if missing:
                raise KeyError(f"arm {arm!r}: features {missing} are not in the batch")
            batch.assert_sources_within(self.allowed_sources, names)
        return {"arm": arm, "n_features": len(names), "status": "privileged" if privileged else "operational",
                "privileged_features": privileged, "dataflow_checked": batch is not None}

    def check_arms(self, arms: Mapping[str, Iterable[str]], batch: FeatureBatch | None = None) -> dict[str, dict]:
        return {a: self.check(a, fs, batch) for a, fs in arms.items()}


def tier1_manifest(*, supplies_noise_only_records: bool = True, latent_dim: int = 6, n_targets: int = 3,
                   n_pcs: int = 3) -> FeatureManifest:
    """The Tier-1 (ORACLE-Cov) manifest. Groups match ``protocol.arms.ARMS`` and ``protocol.features``.

    The controlled simulator emits random-trigger records, so noise-only
    features are operational here. On an arm whose acquisition has none
    (Prometheus hit-level output), pass ``supplies_noise_only_records=False``
    and the noise-only features become privileged.
    """
    m = FeatureManifest("tier1", acquisition={"supplies_noise_only_records": bool(supplies_noise_only_records),
                                              "noise_only_source": "random-trigger records from the same generator, structural N applied"})
    for n in ("in_rms", "in_kurtosis", "in_spectral_centroid", "in_spectral_flatness", "in_channel_rms_spread", "in_peak_abs"):
        m.add(Feature(n, "alarm_time", "raw trace", group="input_quality"))
    for i in range(n_targets):
        m.add(Feature(f"out_{i}", "alarm_time", "frozen model output", group="output"))
    m.add(Feature("unc_recon_energy", "alarm_time", "raw-domain reconstruction residual energy",
                  group="uncertainty", note="the linear subject has no predictive uncertainty; this is the declared proxy"))
    for i in range(latent_dim):
        m.add(Feature(f"z_{i}", "alarm_time", "pooled representation z", group="final_embedding"))
    m.add(Feature("z_mahalanobis", "alarm_time", "z against the reference-fitted mean/shrunk cov", group="final_embedding"))
    for n in ("no_var_ratio", "no_psd_dev_smooth", "no_psd_dev_line", "no_chan_corr_shift"):
        m.add(Feature(n, "alarm_time", "noise-only record", operational=supplies_noise_only_records, group="noise_only"))
    # generic_rich: reference-distance transforms of generic quantities (input, output, pre-output, final z, residual)
    m.add(Feature("gr_input_maha", "alarm_time", "pooled whitened input vs reference null (shrunk, calibrated)", group="generic_rich"))
    m.add(Feature("gr_output_maha", "alarm_time", "output vector vs reference null", group="generic_rich"))
    m.add(Feature("gr_pre_output_maha", "alarm_time", "pre-output vs reference null", group="generic_rich"))
    for p in ("out", "null", "resolved", "weak"):
        m.add(Feature(f"gr_z_energy_{p}", "alarm_time", "energy split of z − z̄ in the reference projectors", group="generic_rich"))
    m.add(Feature("gr_out_of_span", "alarm_time", "residual energy / (residual + in-span) of the observed window", group="generic_rich"))
    m.add(Feature("gr_support_novelty", "alarm_time", "training-support novelty of z (support.SupportEstimator)", group="generic_rich"))
    # intermediate_only: strictly internal hooks (channel, token) — no input, no final z, no output
    for h in ("channel", "token"):
        m.add(Feature(f"im_{h}_mean_maha", "alarm_time", f"pooled {h} mean vs reference null (shrunk, calibrated)", group="intermediate"))
        m.add(Feature(f"im_{h}_second_maha", "alarm_time", f"pooled {h} channel second moment vs reference null", group="intermediate"))
        for i in range(n_pcs):
            m.add(Feature(f"im_{h}_pc_{i}", "alarm_time", f"pooled {h} mean projected on reference PC {i}", group="intermediate"))
    # capacity control: quadratic expansion of generic_rich scalars, truncated to the intermediate feature count
    n_im = 2 * (2 + n_pcs)
    for i in range(n_im):
        m.add(Feature(f"gq_{i}", "alarm_time", "quadratic expansion of generic_rich scalars (capacity control)", group="generic_quadratic"))
    # legacy layerwise (pass-1 development diagnostic; contaminated with input/output duplicates by design of the old arm)
    for h in ("whitened", "channel", "token", "z", "pre_output", "output"):
        m.add(Feature(f"lw_{h}_maha", "alarm_time", "pass-1 per-hook Mahalanobis (legacy diagnostic)", group="layerwise_legacy"))
    # evaluation-side quantities — declared so that consuming them is caught
    m.add(Feature("twin_dz", "evaluation_only", "paired clean twin (replay of the same event)", group="replay",
                  note="every Δz statistic of statistics.cell_statistics; a development diagnostic, not an alarm-time feature"))
    m.add(Feature("truth_targets", "evaluation_only", "planted (amplitude, t0, tau_t)", group="truth"))
    m.add(Feature("sigma_realized", "evaluation_only", "generator implied/realised covariance", group="truth",
                  note="operational monitors use sigma_hat (reference_fit); κ_cond is an evaluation quantity"))
    m.add(Feature("intervention_label", "evaluation_only", "cell origin/family/severity", group="truth"))
    m.add(Feature("task_metric_delayed", "delayed_label", "labelled task metric arriving after the alarm", group="delayed"))
    m.add(Feature("sigma_hat", "reference_fit", "assumed covariance in the whitening layer", group="reference"))
    return m
