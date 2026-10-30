"""Unregistered de-jitter extension helpers (runbook 2026-10-30, reviewed).

Everything here is an extension (``registered: false``).  Two pieces:

* the ``median_time`` token transform (and its process-level ``tokenise``
  wrapper), which subtracts each event's hit-token time median from the hit
  token times;
* the counterfactual inputs of §2.2: trigger-shifted readouts, noise redraws
  and a keyed 0.20 m vertex displacement with its fiducial/capacity redraws.

Pure numpy functions are importable without torch; the token transform handles a
torch tensor if one is given.
"""
import dataclasses

import numpy as np

LOG_CHARGE_INDEX = 6
TIME_INDEX = 7
HIT_INDEX = 8
RAW_FEATURE_BASE = 6
TIME_SCALE_NS = 100.0
VERTEX_DELTA_M = 0.20
VERTEX_NAMESPACE = "ext_dejitter/vertex"


def _as_numpy(features):
    if hasattr(features, "detach"):
        return features.detach().cpu().numpy()
    return np.asarray(features)


def median_hit_time(features):
    """Per-event median of the hit-token times (time_100ns units); 0 if no hits."""
    values = _as_numpy(features).astype(np.float64)
    hit = values[..., HIT_INDEX] > 0
    times = values[..., TIME_INDEX]
    result = np.zeros(len(values))
    for event in range(len(values)):
        selected = times[event][hit[event]]
        if selected.size:
            ordered = np.sort(selected)
            count = ordered.size
            result[event] = 0.5 * (ordered[(count - 1) // 2] + ordered[count // 2])
    return result


def median_time_features(features):
    """Return ``features`` with every hit token time reduced by the event median.

    Unhit token times stay 0; an event with no hit token is left unchanged.
    """
    values = _as_numpy(features).astype(np.float64)
    hit = values[..., HIT_INDEX] > 0
    medians = median_hit_time(values)
    shifted = values.copy()
    shifted[..., TIME_INDEX] = np.where(hit,
                                        values[..., TIME_INDEX] - medians[:, None],
                                        values[..., TIME_INDEX])
    if hasattr(features, "detach"):
        import torch
        return torch.as_tensor(shifted, dtype=features.dtype, device=features.device)
    return shifted


def apply_median_time(batch):
    """Apply the transform to a ``LucidBatch`` (frozen; ``target`` carries no time)."""
    return dataclasses.replace(batch, features=median_time_features(batch.features))


def install_median_time_wrapper(module):
    """Replace ``module.tokenise`` with the median_time transform; return the original.

    The wrapper calls the original tokeniser and then transforms the hit-token
    times.  Restore the attribute (in a ``finally``) with the returned function.
    """
    original = module.tokenise

    def wrapped(*args, **kwargs):
        return apply_median_time(original(*args, **kwargs))

    module.tokenise = wrapped
    return original


def remove_median_time_wrapper(module, original):
    module.tokenise = original


def fiducial_ok(geometry, vertex, fraction=0.8):
    """Inside the 80 % fiducial cylinder used by ``fiducial_vertex``."""
    vertex = np.asarray(vertex, float)
    radius = float(fraction) * float(geometry.radius_m)
    half_height = float(fraction) * float(geometry.half_height_m)
    return bool(np.hypot(vertex[0], vertex[1]) <= radius and abs(vertex[2]) <= half_height)


def _unit_direction(rng):
    direction = rng.standard_normal(3)
    norm = float(np.linalg.norm(direction))
    while norm == 0.0:
        direction = rng.standard_normal(3)
        norm = float(np.linalg.norm(direction))
    return direction / norm


def sample_physical_vertex(draws, geometry, cfg, *, seed, run_id, event_id,
                           delta_m=VERTEX_DELTA_M, max_tries=100):
    """§2.2 ``x_phys``: displace the vertex by ``delta_m`` along a keyed direction.

    Redraws the direction when the candidate leaves the 80 % fiducial cylinder or
    when the Poisson count capacity computed at the candidate vertex would fall
    below the event's capacity.  Returns ``(new_draws, info)``.
    """
    from jepa_noise import lucid_bombs as lb
    rng = lb.keyed_rng(seed, VERTEX_NAMESPACE, run_id, event_id)
    vertex = np.asarray(draws.vertex_m, float)
    fiducial_redraws = capacity_failures = 0
    for attempt in range(int(max_tries)):
        direction = _unit_direction(rng)
        candidate = vertex + float(delta_m) * direction
        if not fiducial_ok(geometry, candidate):
            fiducial_redraws += 1
            continue
        bound = lb.expected_charge(geometry, candidate, draws.yield_photons,
                                   draws.qe_eff, lb.UNATTENUATED_M, cfg)
        if lb.count_capacity(float(bound.max())) > int(draws.capacity):
            capacity_failures += 1
            continue
        info = {"attempts": attempt + 1, "fiducial_redraws": fiducial_redraws,
                "capacity_failures": capacity_failures,
                "direction": direction.tolist(), "displacement_m": float(delta_m)}
        return dataclasses.replace(draws, vertex_m=candidate), info
    raise RuntimeError(f"no valid displaced vertex after {max_tries} tries")


def propagation_time_change(geometry, cfg, vertex_from, vertex_to):
    """Per-PMT source-time change for the displaced vertex (ns)."""
    from jepa_noise import lucid_bombs as lb
    return (lb.propagation_time_ns(geometry, vertex_to, cfg)
            - lb.propagation_time_ns(geometry, vertex_from, cfg))


def dark_moved(record):
    """True when a dark coincidence moved the readout trigger."""
    return bool(record.source_trigger_ns is not None
                and (record.trigger_ns is None
                     or abs(record.trigger_ns - record.source_trigger_ns) > 1e-9))


def sensitivity_terms(z, delta):
    """E||delta||^2 and V = trace Cov(z) over events; z and delta are (n, d)."""
    z = np.asarray(z, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    covariance = np.cov(z, rowvar=False, ddof=1) if z.shape[1] > 1 else np.var(z, ddof=1)[None, None]
    return {"mean_delta_squared": float(np.mean(np.sum(delta * delta, axis=1))),
            "trace_covariance": float(np.trace(np.atleast_2d(covariance)))}


def whitened_sensitivity_terms(z, deltas, threshold=1e-6):
    """Whitened E||W delta||^2 / rank with W from the truncated pseudoinverse.

    Eigenvalues of the event covariance at or below ``threshold`` (absolute) are
    dropped; the retained rank is reported.
    """
    z = np.asarray(z, dtype=np.float64)
    covariance = np.cov(z, rowvar=False, ddof=1)
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    keep = values > float(threshold)
    whitening = (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T
    out = {"rank": int(keep.sum()),
           "trace_covariance": float(values[keep].sum())}
    for name, delta in deltas.items():
        transformed = np.asarray(delta, dtype=np.float64) @ whitening.T
        out[name] = float(np.mean(np.sum(transformed * transformed, axis=1)))
    return out
