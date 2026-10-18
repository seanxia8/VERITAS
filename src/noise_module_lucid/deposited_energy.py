"""D-E development endpoint: primary-muon kinetic energy deposited inside the instrumented volume.

Author definition (18 Oct 2026, docs/PMT_DECISIONS_LOG.md): E_dep = T(entry, or at the vertex)
- T(exit), with T(exit) = 0 for a stopped track; kinetic energies, primary muon only, from
PhotonSim's per-segment beta. Development default, pending Junjie.

Conventions: T at a segment start is m(gamma(beta_start) - 1); T falls linearly in path length
along a segment to the next segment's start value, and to 0 at the end of the last segment
(PhotonSim records the primary down to beta = 0). At a vertex inside the volume T is the
generator kinetic energy ``t0_mev``. Only the first exit after the first entry counts;
re-entries are counted, not added.
"""
from __future__ import annotations

import numpy as np

M_MU_MEV = 105.6583755


def kinetic_from_beta(beta, mass=M_MU_MEV):
    b = np.asarray(beta, dtype=float)
    if np.any(~np.isfinite(b)) or np.any((b < 0) | (b >= 1)):
        raise ValueError("beta must be finite and in [0, 1)")
    return mass*(1/np.sqrt(1 - b*b) - 1)


def _inside(p, radius, height):
    return (p[..., 0]**2 + p[..., 1]**2 < radius**2) & (np.abs(p[..., 2]) < height/2)


def _crossing(p0, p1, radius, height, entering):
    """Fraction along p0->p1 where the segment first enters (or leaves) the cylinder (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi)/2
        if bool(_inside(p0 + mid*(p1 - p0), radius, height)) == entering:
            hi = mid
        else:
            lo = mid
    return hi


def deposited_energy(starts_m, ends_m, beta_start, t0_mev, radius_m, height_m, mass=M_MU_MEV):
    """E_dep of one primary track whose ordered, continuous segments are given (metres)."""
    s = np.asarray(starts_m, dtype=float)
    e = np.asarray(ends_m, dtype=float)
    if s.ndim != 2 or s.shape[1] != 3 or s.shape != e.shape or not len(s):
        raise ValueError("aligned non-empty (n, 3) segment starts and ends required")
    if np.max(np.linalg.norm(s[1:] - e[:-1], axis=1), initial=0.0) > 1e-6:
        raise ValueError("primary segments must be ordered and continuous")
    t_start = kinetic_from_beta(beta_start, mass)
    if t_start.shape != (len(s),):
        raise ValueError("one beta per segment required")
    t_start = t_start.copy()
    if _inside(s[0], radius_m, height_m):
        t_start[0] = float(t0_mev)  # generator kinetic energy at the vertex
    t_end = np.r_[t_start[1:], 0.0]
    inside_start, inside_end = _inside(s, radius_m, height_m), _inside(e, radius_m, height_m)
    t_entry = t_exit = None
    entries = 0
    state = False
    for i in range(len(s)):
        if i == 0 and inside_start[0]:
            state, t_entry, entries = True, t_start[0], 1
        if not state and inside_end[i] and not inside_start[i]:
            f = _crossing(s[i], e[i], radius_m, height_m, entering=True)
            entries += 1
            if t_entry is None:
                t_entry = t_start[i] + f*(t_end[i] - t_start[i])
            state = True
        elif state and not inside_end[i]:
            f = _crossing(s[i], e[i], radius_m, height_m, entering=False)
            if t_exit is None:
                t_exit = t_start[i] + f*(t_end[i] - t_start[i])
            state = False
    if t_entry is None:
        return dict(e_dep_mev=0.0, t_entry_mev=None, t_exit_mev=None, exited=False, stopped_inside=False, entries=0)
    stopped = t_exit is None
    return dict(e_dep_mev=float(t_entry - (0.0 if stopped else t_exit)), t_entry_mev=float(t_entry),
                t_exit_mev=0.0 if stopped else float(t_exit), exited=not stopped, stopped_inside=stopped,
                entries=int(entries))
