"""Endpoint identifiability preflight for consequence pilots.

Pure-NumPy diagnostics that separate "the subject is undertrained" from "the
declared observable does not carry the target".  They were added after the
retained multi-energy muon pilot showed that the primary in-cylinder path
length is effectively constant across 500/1000/1500 MeV: initial kinetic
energy is then weakly observable, while an explicitly defined
visible/deposited-energy target on the same support stays well posed.

Nothing here fits, rescales or selects an evaluation metric.  Every statistic
is a diagnostic, and every acceptance criterion is supplied by the caller so
that thresholds are declared before outcomes are inspected.  These routines
must not be used to re-score the already-inspected pilot.

Status ``dev_unreviewed`` (author, 18 Oct 2026): found untracked after ``56179eb``
(written 16 Oct 21:48, no run output uses it); committed for the D-E containment
preflight. Not reviewed for physics.
"""
from __future__ import annotations

import numpy as np

STATUS = "dev_unreviewed"


def _inside_cylinder(points, radius, height):
    """Strict interior test, identical convention to multienergy_data.inside_cylinder."""
    p = np.asarray(points, float)
    return (p[..., 0] ** 2 + p[..., 1] ** 2 < radius ** 2) & (np.abs(p[..., 2]) < height / 2)


def clip_segment_length(p0, p1, radius, height):
    """Exact length of the intersection of the segment ``p0 -> p1`` with a cylinder.

    The cylinder is ``x^2 + y^2 < radius^2`` and ``|z| < height/2``, centred on
    the origin, matching the transport fixture's water volume.
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    if p0.shape != (3,) or p1.shape != (3,):
        raise ValueError("segment endpoints must be 3-vectors")
    if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
        raise ValueError("finite endpoints required")
    if not radius > 0 or not height > 0:
        raise ValueError("positive radius and height required")
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length == 0.0:
        return float(length) if _inside_cylinder(p0, radius, height) else 0.0
    enter, leave = 0.0, 1.0
    # Radial bound: quadratic in t for |x(t)|^2 + |y(t)|^2 = radius^2.
    a = delta[0] ** 2 + delta[1] ** 2
    b = 2.0 * (p0[0] * delta[0] + p0[1] * delta[1])
    c = p0[0] ** 2 + p0[1] ** 2 - radius ** 2
    if a == 0.0:
        if c >= 0.0:
            return 0.0
    else:
        disc = b * b - 4.0 * a * c
        if disc <= 0.0:
            return 0.0
        root = np.sqrt(disc)
        t_lo, t_hi = (-b - root) / (2.0 * a), (-b + root) / (2.0 * a)
        enter, leave = max(enter, t_lo), min(leave, t_hi)
    # Axial bound.
    if delta[2] == 0.0:
        if abs(p0[2]) >= height / 2:
            return 0.0
    else:
        t_lo, t_hi = (-height / 2 - p0[2]) / delta[2], (height / 2 - p0[2]) / delta[2]
        if t_lo > t_hi:
            t_lo, t_hi = t_hi, t_lo
        enter, leave = max(enter, t_lo), min(leave, t_hi)
    if leave <= enter:
        return 0.0
    return float(length * (leave - enter))


def primary_path_lengths(starts, ends, radius, height):
    """Total and in-cylinder length of a polyline of primary track segments.

    ``contained`` follows the retained pilot's convention exactly: every segment
    endpoint (start and end) lies strictly inside the cylinder.
    """
    starts = np.atleast_2d(np.asarray(starts, float))
    ends = np.atleast_2d(np.asarray(ends, float))
    if starts.shape != ends.shape or starts.shape[1] != 3:
        raise ValueError("aligned (n, 3) segment endpoints required")
    total = float(np.sum(np.linalg.norm(ends - starts, axis=1)))
    inside = float(sum(clip_segment_length(s, e, radius, height) for s, e in zip(starts, ends)))
    contained = bool(_inside_cylinder(starts, radius, height).all()
                     and _inside_cylinder(ends, radius, height).all())
    return dict(total_m=total, inside_m=inside, n_segments=int(len(starts)), contained=contained)


def light_yield_saturation(energies_mev, observable):
    """Per-energy summary of one observable plus its label-separation strength.

    Reports medians, ratios against the lowest declared energy, the squared
    Pearson correlation with the nominal energy label, within-energy
    coefficients of variation, and a separation effect size
    ``(max median - min median) / pooled within-energy sigma``.
    """
    energies = np.asarray(energies_mev, float)
    values = np.asarray(observable, float)
    if energies.shape != values.shape or energies.ndim != 1:
        raise ValueError("aligned 1-D energy and observable arrays required")
    if not np.isfinite(values).all():
        raise ValueError("finite observable required")
    levels = np.unique(energies)
    if len(levels) < 2:
        raise ValueError("at least two energy levels required")
    medians, within_sd, within_cv = {}, {}, {}
    for level in levels:
        sel = values[energies == level]
        medians[float(level)] = float(np.median(sel))
        within_sd[float(level)] = float(np.std(sel, ddof=1)) if len(sel) > 1 else 0.0
        within_cv[float(level)] = (float(within_sd[float(level)] / np.mean(sel))
                                   if np.mean(sel) != 0 else float("nan"))
    base = float(levels[0])
    ratios = {float(l): (medians[float(l)] / medians[base] if medians[base] != 0 else float("nan"))
              for l in levels}
    spread = max(medians.values()) - min(medians.values())
    pooled = float(np.sqrt(np.mean([s ** 2 for s in within_sd.values()])))
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = float(np.corrcoef(energies, values)[0, 1] ** 2)
    return dict(
        levels=[float(l) for l in levels],
        medians=medians,
        ratios_vs_lowest=ratios,
        r2_energy_label=r2,
        within_cv=within_cv,
        separation=(float(spread / pooled) if pooled > 0 else float("inf")),
        marginal_span_fraction=float(spread / np.mean(list(medians.values()))),
    )


def endpoint_identifiability(predicted_energy_mev, true_energy_mev):
    """Predictor skill against a constant equal to the mean true energy of the evaluated set.

    The constant is ``mean(true_energy_mev)`` in MeV over the evaluated rows: not the
    training-split mean of log energy used by the R-3 gates, and not equal-weight across
    energies, so this is a diagnostic only, not comparable to an R-3 ratio (review, 18 Oct).
    ``improvement_over_constant`` is ``1 - MAE_model / MAE_constant``; it is the
    headroom a subject actually captures on this support.  Per-energy relative
    MAE and signed bias are returned so a single scale error is distinguishable
    from a genuine lack of information.
    """
    pred = np.asarray(predicted_energy_mev, float)
    true = np.asarray(true_energy_mev, float)
    if pred.shape != true.shape or pred.ndim != 1 or len(pred) == 0:
        raise ValueError("aligned non-empty 1-D energy arrays required")
    if not (np.isfinite(pred).all() and np.isfinite(true).all()):
        raise ValueError("finite energies required")
    model_mae = float(np.mean(np.abs(pred / true - 1.0)))
    constant_mae = float(np.mean(np.abs(np.mean(true) / true - 1.0)))
    per_energy = {}
    for level in np.unique(true):
        sel = true == level
        per_energy[float(level)] = dict(
            n=int(sel.sum()),
            relative_mae=float(np.mean(np.abs(pred[sel] / true[sel] - 1.0))),
            bias_fraction=float(np.mean(pred[sel] / true[sel] - 1.0)),
            mean_prediction_mev=float(np.mean(pred[sel])),
        )
    return dict(
        n=int(len(pred)),
        model_relative_mae=model_mae,
        constant_relative_mae=constant_mae,
        improvement_over_constant=float(1.0 - model_mae / constant_mae) if constant_mae > 0 else float("nan"),
        prediction_span_fraction=float((np.max(pred) - np.min(pred)) / np.mean(true)),
        by_energy=per_energy,
    )


def preflight_verdict(statistics, criteria):
    """Evaluate caller-declared criteria against diagnostics; never invents one.

    ``criteria`` maps a dotted path into ``statistics`` to a ``{"op", "value"}``
    rule with ``op`` in ``{"ge", "gt", "le", "lt"}``.  Returns per-criterion
    values plus an overall flag, so an unidentifiable support is reported as
    such before training rather than as a trainability failure after it.
    """
    def lookup(path):
        node = statistics
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"criterion path not found in diagnostics: {path}")
            node = node[part]
        if not np.isscalar(node):
            raise ValueError(f"criterion path is not scalar: {path}")
        return float(node)

    ops = {"ge": np.greater_equal, "gt": np.greater, "le": np.less_equal, "lt": np.less}
    results, reasons = {}, []
    for path, rule in criteria.items():
        if rule["op"] not in ops:
            raise ValueError(f"unsupported criterion op: {rule['op']}")
        value = lookup(path)
        ok = bool(ops[rule["op"]](value, float(rule["value"])))
        results[path] = dict(value=value, op=rule["op"], threshold=float(rule["value"]), passed=ok)
        if not ok:
            reasons.append(f"{path}={value:.6g} fails {rule['op']} {rule['value']:.6g}")
    return dict(passed=all(r["passed"] for r in results.values()), criteria=results, reasons=reasons)
