"""Unregistered timing extension: feature attribution and trigger de-jitter.

Runbook: ``docs/jepa/RUNBOOK_2026-10-30_TIMING.md`` (30 Oct 2026).
Everything here is an extension (``registered: false``); no registered code or
config is edited.  The Criterion-M covariance definitions are imported from
``jepa_noise.ext_m_dose`` (never reimplemented).

The module holds only numpy-level helpers so the §3.4 unit tests can drive them
without loading a checkpoint.
"""
import numpy as np

FEATURE_NAMES = ("log_charge", "time_100ns", "hit")
N_FEATURES = 3
TIME_SCALE_NS = 100.0
#: Time feature coordinates in the module-pooled layout: group * 3 + 1.
TIME_OFFSET = 1
HIT_OFFSET = 2
#: The tokenised full feature block starts at index 6 (log charge, time, hit).
RAW_FEATURE_BASE = 6


def regularise(matrix):
    """The runbook epsilon: 1e-3 * trace / d * I (same as ext_m_dose)."""
    return matrix + 1e-3 * np.trace(matrix) / len(matrix) * np.eye(len(matrix))


def blocks_from_joint(S, A, B, context_dim, target_dim):
    """Split joint covariances into the six blocks per source (S, N_A, N_B)."""
    def parts(matrix):
        return (matrix[:context_dim, :context_dim],
                matrix[context_dim:, context_dim:],
                matrix[:context_dim, context_dim:])
    scc, stt, sct = parts(S)
    acc, att, act = parts(A)
    bcc, btt, bct = parts(B)
    expected = {"scc": (context_dim, context_dim),
                "stt": (target_dim, target_dim),
                "sct": (context_dim, target_dim),
                "acc": (context_dim, context_dim),
                "att": (target_dim, target_dim),
                "act": (context_dim, target_dim),
                "bcc": (context_dim, context_dim),
                "btt": (target_dim, target_dim),
                "bct": (context_dim, target_dim)}
    blocks = dict(scc=scc, stt=stt, sct=sct, acc=acc, att=att, act=act,
                  bcc=bcc, btt=btt, bct=bct)
    for name, block in blocks.items():
        if np.asarray(block).shape != expected[name]:
            raise ValueError(f"block {name} has shape {np.shape(block)}, "
                             f"expected {expected[name]}")
    return blocks


def _apply(Tc, Tt, M, view):
    if view == "c":
        return Tc @ M @ Tc.T
    if view == "t":
        return Tt @ M @ Tt.T
    return Tc @ M @ Tt.T


BLOCK_VIEWS = {"scc": "c", "acc": "c", "bcc": "c",
               "stt": "t", "att": "t", "btt": "t",
               "sct": "ct", "act": "ct", "bct": "ct"}


def transform_blocks(blocks, Tc, Tt):
    """Apply one linear map per view to every block: T_c X T_c^T etc."""
    return {name: _apply(Tc, Tt, block, BLOCK_VIEWS[name])
            for name, block in blocks.items()}


def selection_map(n_groups, feature):
    """Keep only one pooled feature (0 log charge, 1 time, 2 hit) per group."""
    if feature not in (0, 1, 2):
        raise ValueError("feature must be 0, 1 or 2")
    matrix = np.zeros((n_groups, N_FEATURES * n_groups))
    for group in range(n_groups):
        matrix[group, group * N_FEATURES + feature] = 1.0
    return matrix


def drop_feature_map(n_groups, feature):
    """Drop one pooled feature per group (kept coordinates stay group-major)."""
    keep = [index for index in range(N_FEATURES) if index != feature]
    matrix = np.zeros((len(keep) * n_groups, N_FEATURES * n_groups))
    row = 0
    for group in range(n_groups):
        for index in keep:
            matrix[row, group * N_FEATURES + index] = 1.0
            row += 1
    return matrix


def common_time_map(n_groups):
    """Project the time coordinates of one view off the all-ones vector."""
    dim = N_FEATURES * n_groups
    matrix = np.eye(dim)
    time = np.arange(TIME_OFFSET, dim, N_FEATURES)
    projector = np.eye(n_groups) - np.ones((n_groups, n_groups)) / n_groups
    matrix[np.ix_(time, time)] = projector
    return matrix, dim


def hit_energy_eigenvectors(S_vv, means_v, n_eig=3):
    """Top eigenvalues of E[h h^T] = hit block of S + m_h m_h^T (per view)."""
    hit = np.arange(HIT_OFFSET, S_vv.shape[0], N_FEATURES)
    energy = S_vv[np.ix_(hit, hit)] + np.outer(means_v[hit], means_v[hit])
    eigenvalues, eigenvectors = np.linalg.eigh((energy + energy.T) / 2)
    order = np.argsort(eigenvalues)[::-1][:n_eig]
    return eigenvectors[:, order], eigenvalues[order]


def time_hit_map(S_vv, means_v, n_eig=3):
    """Project the event-varying jitter directions h (top n_eig) out of the time coords."""
    n_groups = S_vv.shape[0] // N_FEATURES
    directions, _ = hit_energy_eigenvectors(S_vv, means_v, n_eig)
    dim = N_FEATURES * n_groups
    matrix = np.eye(dim)
    time = np.arange(TIME_OFFSET, dim, N_FEATURES)
    projector = np.eye(n_groups) - directions @ directions.T
    matrix[np.ix_(time, time)] = projector
    return matrix, dim


def variant_blocks(blocks, variant, *, S, means, context_dim, target_dim):
    """Return (transformed blocks, Tc, Tt) for a §2.3 variant."""
    n_groups_c, n_groups_t = context_dim // N_FEATURES, target_dim // N_FEATURES
    means_c = means[:, :context_dim].mean(axis=0)
    means_t = means[:, context_dim:].mean(axis=0)
    S_cc = S[:context_dim, :context_dim]
    S_tt = S[context_dim:, context_dim:]
    if variant == "all":
        return dict(blocks), np.eye(context_dim), np.eye(target_dim)
    if variant in ("time_only", "charge_only", "hit_only"):
        feature = {"charge_only": 0, "time_only": 1, "hit_only": 2}[variant]
        Tc = selection_map(n_groups_c, feature)
        Tt = selection_map(n_groups_t, feature)
        return transform_blocks(blocks, Tc, Tt), Tc, Tt
    if variant == "no_time":
        Tc = drop_feature_map(n_groups_c, 1)
        Tt = drop_feature_map(n_groups_t, 1)
        return transform_blocks(blocks, Tc, Tt), Tc, Tt
    if variant == "time_common_removed":
        Tc, _ = common_time_map(n_groups_c)
        Tt, _ = common_time_map(n_groups_t)
        return transform_blocks(blocks, Tc, Tt), Tc, Tt
    if variant == "time_hit_removed":
        Tc, _ = time_hit_map(S_cc, means_c)
        Tt, _ = time_hit_map(S_tt, means_t)
        return transform_blocks(blocks, Tc, Tt), Tc, Tt
    raise ValueError(f"unknown variant {variant!r}")


def pair_metrics(blocks, k=4, extra_cross=()):
    """Run the Criterion-M pair construction on transformed blocks."""
    from jepa_noise.ext_m_dose import pair_diagnostics
    diagnostics = pair_diagnostics(blocks["scc"], blocks["stt"], blocks["sct"],
                                   blocks["bcc"], blocks["btt"], blocks["bct"], k,
                                   *extra_cross)
    shares = diagnostics["shares"]
    result = {
        "nu_B": shares[0], "nu_A": shares[1], "nu_gain": shares[2],
        "nu_star_B": float(np.max(shares[0])),
        "nu_star_A": float(np.max(shares[1])),
        "nu_star_gain": float(np.max(shares[2])),
        "argmax_B": int(np.argmax(shares[0])),
        "argmax_A": int(np.argmax(shares[1])),
        "argmax_gain": int(np.argmax(shares[2])),
        "sigma_phys": diagnostics["sigma_phys"],
        "sigma_noise": diagnostics["sigma_noise"],
        "r": diagnostics["ratio"],
        "r_s": diagnostics["ratio_s"],
        "identity_error": diagnostics["identity_error"],
    }
    return result


def trigger_deltas(triggers):
    """delta_r = trigger_r - mean_r trigger, per event."""
    triggers = np.asarray(triggers, dtype=float)
    if triggers.ndim != 1:
        raise ValueError("triggers must be one-dimensional per event")
    if not np.isfinite(triggers).all():
        raise ValueError("triggers must be finite")
    return triggers - triggers.mean()


def dejitter_pooled(pooled, deltas):
    """Add h_r * delta_r / 100 to every pooled time coordinate.

    ``pooled`` is (R, 3 * n_groups) in the M-dose pooled layout (group * 3 + f,
    f = 0 log charge, 1 time, 2 hit); ``deltas`` is (R,).  ``h_r`` is the hit
    fraction of the group in that realisation (the pooled hit coordinate).
    """
    pooled = np.asarray(pooled, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    if pooled.ndim != 2 or pooled.shape[1] % N_FEATURES:
        raise ValueError("pooled must be (R, 3 * n_groups)")
    if deltas.shape != (pooled.shape[0],):
        raise ValueError("one trigger delta per realisation required")
    corrected = pooled.copy()
    for group in range(pooled.shape[1] // N_FEATURES):
        hit = pooled[:, group * N_FEATURES + HIT_OFFSET]
        corrected[:, group * N_FEATURES + TIME_OFFSET] += (
            hit * deltas / TIME_SCALE_NS)
    return corrected


def cosine(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / denominator) if denominator else None


def time_indicator(n_groups):
    """Vector that is one on every time coordinate and zero elsewhere."""
    vector = np.zeros(N_FEATURES * n_groups)
    vector[TIME_OFFSET::N_FEATURES] = 1.0
    return vector


def hit_time_pattern(n_groups, mean_hit):
    """The jitter direction h in feature space: mean hit fraction at time coords."""
    vector = np.zeros(N_FEATURES * n_groups)
    vector[TIME_OFFSET::N_FEATURES] = np.asarray(mean_hit, float)
    return vector


def sqrtm_psd(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    if values.min() <= 0:
        raise ValueError("positive definite matrix required")
    return (vectors * np.sqrt(values)) @ vectors.T


def alignment_metrics(blocks, means_context):
    """§2.5 alignment of the A-block noise pattern with the time directions."""
    from jepa_noise.views import invsqrt
    wc = regularise(blocks["acc"])
    wt = regularise(blocks["att"])
    h_noise = invsqrt(wc) @ blocks["act"] @ invsqrt(wt)
    left = np.linalg.svd(h_noise, full_matrices=False)[0][:, 0]
    pattern = sqrtm_psd(wc) @ left
    n_groups = blocks["acc"].shape[0] // N_FEATURES
    mean_hit = means_context[:, HIT_OFFSET::N_FEATURES].mean(axis=0)
    return {
        "cos_time_ones": cosine(pattern, time_indicator(n_groups)),
        "cos_time_hit": cosine(pattern, hit_time_pattern(n_groups, mean_hit)),
        "pattern_norm": float(np.linalg.norm(pattern)),
        "mean_hit_fraction": float(np.mean(mean_hit)),
    }


def shifted_time_rule(rule, shift_ns):
    """Runtime stub equal to a time rule with its trigger shifted by ``shift_ns``.

    Installed by a test into ``lucid_bombs._PMT_TIME_RULE`` and restored in a
    ``finally`` block; this is the documented test stub, not a code edit.
    """
    real_trigger, mask, n, window_ns, prompt = rule

    def shifted(digit_times, *, n=n, window_ns=window_ns):
        value = real_trigger(digit_times, n=n, window_ns=window_ns)
        return None if value is None else float(value) + float(shift_ns)

    return (shifted, mask, n, window_ns, prompt)
