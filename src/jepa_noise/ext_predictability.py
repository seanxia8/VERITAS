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
                     .integers(0, 2**64, dtype=np.uint64))
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


def tokenise_realisations(draws, geometry, cfg, state, scheme, tmp_root, *,
                         spread, seed=DEFAULT_SEED, replicates=16, mask_seed=0, scratch_dir=None):
    """Read out one event repeatedly using the registered tokenisation path.

    The temporary sparse run is removed before return. Synthetic row IDs are
    used only within the temporary file: the within-view partition is fixed by
    PMT parity and does not depend on event ID. In scheme B, noise indices
    1..R ensure every freshly drawn state also receives new noise.
    """
    from pathlib import Path
    from contextlib import nullcontext
    from tempfile import TemporaryDirectory

    from jepa_noise import lucid_bombs as lb
    from jepa_noise import point_encoder as pe
    from jepa_noise.lucid_views import ViewMasker

    if scheme not in ("A", "B") or replicates < 2:
        raise ValueError("expected scheme A or B and at least two replicates")
    root = Path(tmp_root)
    if not root.is_dir():
        raise ValueError("temporary root must exist")
    readout = lb.make_readout(geometry, cfg)
    hits = []
    for r in range(replicates):
        noise_index = r if scheme == "A" else r + 1
        sample = resample_noise(draws, geometry, cfg, noise_index, seed=seed)
        state_r = state if scheme == "A" else scheme_b_state(
            geometry, spread, draws.run_id, draws.event_id, r, seed=seed)
        hits.append(readout(state_r, sample))
    scratch = TemporaryDirectory(prefix="event_", dir=root) if scratch_dir is None else nullcontext(str(scratch_dir))
    with scratch as temp:
        if not Path(temp).resolve().is_relative_to(root.resolve()):
            raise ValueError("scratch directory must be within the temporary root")
        ids = np.arange(replicates, dtype=np.int64)
        lb.write_sparse_hits(Path(temp) / "hits.npz", lb.sparse_from_readouts(ids, hits))
        batch = pe.tokenise(temp, geometry, split="within", seed=mask_seed,
                           scaling=pe.FeatureScaling.from_geometry(geometry),
                           masker=ViewMasker(geometry, "within", seed=mask_seed, n_min=0))
    if len(batch.event_ids) != replicates:
        raise RuntimeError("within-view N_min excluded a resampled realisation")
    return batch.features.detach().cpu().numpy(), batch.target_index[0].cpu().numpy()


def event_moments(features, target_index):
    """Per-target sufficient statistics without retaining raw realisations."""
    features = np.asarray(features)
    target_index = np.asarray(target_index, dtype=np.int64)
    if features.ndim != 3 or features.shape[0] < 2 or features.shape[2] < 9:
        raise ValueError("expected R x PMT x feature input with R >= 2")
    x = features[:, target_index, 6:9].astype(np.float64)
    mask = x[:, :, 2] > 0
    hit_values = np.where(mask[:, :, None], x, 0.0)
    return {
        "mean": x.mean(axis=0).astype(np.float32),
        "variance": x.var(axis=0, ddof=1).astype(np.float32),
        "hit_count": mask.sum(axis=0).astype(np.uint8),
        "hit_sum": hit_values.sum(axis=0).astype(np.float32),
        "hit_sum_squares": np.square(hit_values).sum(axis=0).astype(np.float32),
    }


def moments_predictability(mean, variance, *, replicates=16):
    """Estimate predictability from unbiased per-pair moments only."""
    mean, variance = np.asarray(mean, float), np.asarray(variance, float)
    if mean.shape != variance.shape or mean.size == 0 or replicates < 2:
        raise ValueError("paired nonempty moments and at least two replicates required")
    within_population = variance * ((replicates - 1) / replicates)
    total = np.mean(within_population + mean * mean) - np.mean(mean) ** 2
    if total <= 0:
        return None
    return float(1 - variance.mean() / total)


def hit_moments_predictability(count, total, squares):
    """Conditional hit-only predictability and contributing sample counts."""
    n, s, q = (np.asarray(x, dtype=np.float64) for x in (count, total, squares))
    if not (n.shape == s.shape == q.shape):
        raise ValueError("hit-only sufficient statistics must have identical shapes")
    used = n >= 2
    observations = int(n.sum())
    pairs = int(used.sum())
    if observations == 0 or pairs == 0:
        return {"rho": None, "pairs": pairs, "observations": observations}
    total_var = q.sum() / observations - (s.sum() / observations) ** 2
    if total_var <= np.finfo(float).eps * max(1.0, q.sum() / observations):
        return {"rho": None, "pairs": pairs, "observations": observations}
    within = np.maximum(0, q[used] - s[used] ** 2 / n[used]) / (n[used] - 1)
    return {"rho": float(1 - within.mean() / total_var),
            "pairs": pairs, "observations": observations}


def restore_cached_truth(draws, cached, *, rtol=1e-12, atol=1e-12):
    """Restore the frozen event truth after checking keyed reconstruction.

    The cached event is authoritative. Tiny platform-dependent roundoff in the
    keyed reconstruction is tolerated only for comparison; the returned draw
    uses the saved values exactly. Noise and QE arrays are retained unchanged.
    """
    def close(name, actual, expected):
        if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False):
            raise ValueError(f'cached truth mismatch: {name}')

    if str(cached['domain']) != draws.domain:
        raise ValueError('cached truth mismatch: domain')
    close('vertex_m', draws.vertex_m, cached['vertex_m'])
    close('yield_photons', draws.yield_photons, cached['yield_photons'])
    close('emission_offset_ns', draws.emission_offset_ns, cached['emission_offset_ns'])
    truth = draws.truth
    if truth is not None:
        for name, cache_name in (('direction', 'direction_m'), ('energy_MeV', 'energy_MeV'),
                                 ('deposited_MeV', 'deposited_MeV'), ('track_length_m', 'track_length_m')):
            close(name, getattr(truth, name), cached[cache_name])
        if truth.contained != bool(cached['contained']):
            raise ValueError('cached truth mismatch: contained')
        truth = replace(truth, vertex_m=np.array(cached['vertex_m'], copy=True),
                        direction=np.array(cached['direction_m'], copy=True),
                        energy_MeV=float(cached['energy_MeV']),
                        deposited_MeV=float(cached['deposited_MeV']),
                        track_length_m=float(cached['track_length_m']),
                        contained=bool(cached['contained']))
    return replace(draws, vertex_m=np.array(cached['vertex_m'], copy=True),
                   yield_photons=float(cached['yield_photons']), truth=truth,
                   emission_offset_ns=float(cached['emission_offset_ns']))
