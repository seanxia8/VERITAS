"""Unregistered predictability extension: keyed detector-noise resampling."""
from copy import deepcopy
from dataclasses import replace

import numpy as np

from jepa_noise.lucid_bombs import DEFAULT_SEED, dark_photoelectrons, keyed_rng


def resample_noise(draws, geometry, cfg, r, *, seed=DEFAULT_SEED):
    """Keep event physics fixed and draw a fresh detector-noise bank.

    Dark-time array length follows the redrawn count, as in ``draw_event``.
    """
    if r < 0:
        raise ValueError("replicate must be nonnegative")
    if r == 0:
        return deepcopy(draws)

    def rng(role):
        return keyed_rng(seed, "ext_predictability", draws.run_id, draws.event_id, r, role)

    counts, times = dark_photoelectrons(geometry, cfg, rng("dark"), rng("dark_time"))
    return replace(
        draws,
        poisson_u=np.clip(rng("poisson").random(draws.poisson_u.shape),
                          np.finfo(float).tiny, 1.0),
        charge_z=rng("charge").standard_normal(draws.charge_z.shape),
        transit_z=rng("transit").standard_normal(draws.transit_z.shape),
        dark_counts=counts,
        dark_times=times,
    )


def scheme_b_state(geometry, spread, run_id, event_id, r, *, seed=DEFAULT_SEED):
    """Draw an independent F1 + F3 state from the registered cell law."""
    from jepa_noise.lucid_bombs import state_for_run

    state_seed = int(keyed_rng(seed, "ext_predictability_state", run_id, event_id, r)
                     .integers(0, 2**32, dtype=np.uint32))
    return state_for_run("F1", spread, run_id, geometry, seed=state_seed)


def predictability(values):
    """Estimate one minus within-event variance divided by total variance.

    The last axis is the independent replicate dimension. The unbiased within
    variance removes the finite-R downward bias for a pure-noise feature.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] < 2:
        raise ValueError("at least two realisations per item required")
    variance = np.var(values)
    if variance == 0:
        raise ValueError("predictability is undefined for a constant feature")
    return float(1.0 - np.var(values, axis=-1, ddof=1).mean() / variance)
