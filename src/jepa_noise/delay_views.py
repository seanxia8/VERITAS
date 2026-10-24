# SPDX-License-Identifier: MIT
"""Tier-1 waveform views with channel-ordered arrival delays (extension, registered: false).

Author request, 24 Oct 2026: "在 toy model 的 traceset 里面，我们要是加上各组 arriving
time 逐组变晚。加一个这样的 causal 关系呢？"

Channel c (0..7) sees the same pulse as before, but its onset is delayed by
``c * delta`` (record-fraction units), so channels arrive in order and the
channel split (context 0-3, target 4-7) becomes causal: the context always
fires before the target.  Two modes:

* ``fixed``: ``delta = delay_step`` for every event.  The physics parameters are
  unchanged (amplitude, t0, tau); the delay is a known causal structure.
* ``per_event``: ``delta`` is a fourth physics parameter drawn per event,
  uniform in ``delay_range`` (like a source position or propagation speed that
  sets the arrival gradient across the array).  It is probed like the others.

Everything else (noise law, splits, nuisance bases, epsilon_dc) is the
registered A4 law of :class:`jepa_noise.views.Tier1Views`; only the signal
template changes.  Nothing here edits a registered file or verdict.
"""
from __future__ import annotations

import numpy as np

from .views import Tier1Views

DELAY_MODES = ("fixed", "per_event")


class DelayedTier1Views(Tier1Views):
    def __init__(self, *args, delay_mode="fixed", delay_step=0.01,
                 delay_range=(0.005, 0.015), **kwargs):
        if delay_mode not in DELAY_MODES:
            raise ValueError(f"delay_mode must be one of {DELAY_MODES}")
        lo, hi = (float(x) for x in delay_range)
        if not (0.0 <= lo < hi) or delay_step < 0:
            raise ValueError("delays must be non-negative with lo < hi")
        if 0.40 + 7 * max(hi, delay_step) > 0.70:
            raise ValueError("the last channel's onset must stay before 70% of the record")
        # set before the parent constructor, which already calls self.waveform
        self.delay_mode, self.delay_step, self.delay_range = delay_mode, float(delay_step), (lo, hi)
        super().__init__(*args, **kwargs)
        if delay_mode == "per_event":
            from .crlb import central_jacobian
            self.theta0 = np.r_[self.theta0, 0.5 * (lo + hi)]
            self.ranges = np.r_[self.ranges, hi - lo]
            self.theta_sd = np.r_[self.theta_sd, (hi - lo) / np.sqrt(12.0)]
            self.j0 = central_jacobian(lambda p: self.waveform(p).ravel(), self.theta0, self.ranges)

    @property
    def n_physics(self):
        return 4 if self.delay_mode == "per_event" else 3

    def channel_delays(self, theta):
        delta = float(theta[3]) if len(theta) == 4 else self.delay_step
        return np.arange(8) * delta

    def waveform(self, theta):
        from noise_module import pulse_template_2
        theta = np.asarray(theta, dtype=float)
        a, onset, tau = theta[:3]
        f = self.family
        t = np.arange(256) / 256
        rows = [pulse_template_2(t, onset + d, f.An_frac * a, a, f.tau_n, f.tau_in, tau)
                for d in self.channel_delays(theta)]
        return self.gains[:, None] * np.asarray(rows)

    def sample(self, event_ids, role='subject_train', return_components=False):
        ids = np.asarray(event_ids, dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids < 0):
            raise ValueError('unique nonnegative event IDs required')
        raw, theta, latent, components = [], [], [], []
        lo, hi = self.delay_range
        for event_id in ids:
            rng = self._rng(role, event_id, 'physics')
            if self.signal_mode == 'linearized':
                physical = rng.normal(size=self.n_physics) * self.theta_sd
                signal = (self.j0 @ physical).reshape(8, 256)
            else:
                low = [3 * self.amplitude_scale, .30, .06] + ([lo] if self.n_physics == 4 else [])
                high = [6 * self.amplitude_scale, .40, .10] + ([hi] if self.n_physics == 4 else [])
                physical = rng.uniform(low, high)
                signal = self.waveform(physical)
            noise, target, parts = self.noise_record(event_id, role=role + '/event_noise',
                                                     return_components=True)
            raw.append(signal + noise); theta.append(physical); latent.append(list(target.values()))
            if return_components:
                components.append(parts)
        raw = np.asarray(raw).reshape(len(ids), 8, 256)
        flat = raw.reshape(len(ids), 2048)
        result = {'raw': raw, 'context': flat[:, self.split.context], 'target': flat[:, self.split.target],
                  'theta': np.asarray(theta), 'nuisance': np.asarray(latent),
                  'nuisance_names': ['line_I', 'line_Q', 'common_0', 'common_1'],
                  'nuisance_valid': [self.line_multiplier > 0] * 2 + [self.common_strength > 0] * 2,
                  'event_ids': ids, 'role': role, 'cell': self.cell, 'split_hash': self.split.digest}
        if return_components:
            result['components'] = components
        return result

    def manifest(self):
        out = super().manifest()
        out.update(delay_mode=self.delay_mode, delay_step=self.delay_step,
                   delay_range=list(self.delay_range), registered=False,
                   extension='toy_channel_delays_2026-10-24',
                   theta_semantics=(out['theta_semantics'] if self.n_physics == 3
                                    else out['theta_semantics'] + ',delay_per_channel'))
        return out
