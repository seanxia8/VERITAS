"""Bounded QE tables and identity-keyed streams; parameters require provenance."""
from dataclasses import dataclass
import hashlib
import numpy as np


def probabilities(x):
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return x


@dataclass(frozen=True)
class QECurve:
    wavelength_nm: tuple[float, ...]
    efficiency: tuple[float, ...]
    source: str = "synthetic_dev"

    def __post_init__(self):
        w = np.asarray(self.wavelength_nm)
        if w.ndim != 1 or len(w) < 3 or len(w) != len(self.efficiency) or not np.all(np.isfinite(w)) or np.any(np.diff(w) <= 0) or np.any(w <= 0):
            raise ValueError("QE needs >=3 ordered positive wavelength anchors")
        probabilities(self.efficiency)
        if not self.source:
            raise ValueError("QE provenance required")

    def __call__(self, wavelength_nm):
        w = np.asarray(wavelength_nm, dtype=float)
        if np.any(~np.isfinite(w)) or np.any((w < self.wavelength_nm[0]) | (w > self.wavelength_nm[-1])):
            raise ValueError("wavelength outside calibrated domain")
        return np.interp(w, self.wavelength_nm, self.efficiency)

    def drift(self, amplitude, shape):
        """Anchor-defined shape: h(lambda)=interp(QE_anchor*h_anchor)/QE_ref(lambda).

        Thus returned linear interpolation equals QE_ref*(1+a*h) exactly;
        h uses QE-weighted interpolation, not plain interpolation of h anchors.
        Where QE_ref is zero, its drift is also zero. Knot validation bounds
        every segment since the resulting QE table is piecewise linear.
        """
        h = np.asarray(shape, dtype=float)
        if h.shape != (len(self.efficiency),) or not np.isfinite(amplitude) or np.any(~np.isfinite(h)):
            raise ValueError("shape must match QE anchors and be finite")
        return QECurve(self.wavelength_nm, tuple(probabilities(np.asarray(self.efficiency) * (1 + amplitude*h))), self.source)

    def integrated_probability(self, wavelengths, weights, ce=1.0):
        weights = np.asarray(weights, dtype=float)
        p = self(wavelengths) * probabilities(ce)
        if weights.shape != p.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("invalid spectrum weights")
        return float(np.average(p, weights=weights))


def keyed_uniform(seed, event_id, identities, stage):
    """Stable per-identity BLAKE2b uniforms; deletion/reordering never shifts draws."""
    ids = np.asarray(identities)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("persistent IDs must be unique nonnegative integers")
    prefix = f"pmt-v1:{int(seed)}:{int(event_id)}:{stage}:"
    # Top 53 bits plus half-bin avoids both inverse-CDF endpoints.
    return np.array([((int.from_bytes(hashlib.blake2b((prefix+str(int(i))).encode(), digest_size=8).digest(), 'big') >> 11)+0.5)/2**53 for i in ids])
