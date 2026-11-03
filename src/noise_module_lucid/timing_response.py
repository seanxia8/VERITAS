"""Forward-time digit-width models and the D-TQ walk.

``InverseSqrtWalk`` is the D-TQ walk (runbook v2 §3); the polynomial
``WalkCurve`` is retained as the legacy option.

Digit width, two declared kernels
---------------------------------
* :class:`WidthCurve` — the v2 symmetric Gaussian (D-J approximation): one
  ``sigma`` per charge, applied as ``sigma * z``.
* :class:`SplitNormalWidth` — R-7 (author message, 23 Oct 2026): the split-normal
  of thesis Eq. 6.3, with an early-side ``sigma_e`` and a late-side ``sigma_l``.
  The mode is the photon arrival plus the walk; the mean sits at the mode plus
  ``sqrt(2/pi) * (sigma_l - sigma_e)``. Values are literature proxies for the
  20-inch Hyper-Kamiokande Box&Line PMT, not WCTE's 3-inch PMT.

* :class:`WCSimTTSWidth` — R-7 v2 (author message, 23 Oct 2026; decisions 3 Nov
  2026), the default of ``pmt-response-v3``: WCSim's
  ``PMT3inchR14374_WCTE::HitTimeSmearing`` for WCTE's 3-inch Hamamatsu R14374,
  a symmetric Gaussian with ``sigma(Q) = sqrt(1.5**2 + R(min(Q, 4))**2) / 2.355``
  ns. ``R`` is WCSim's 14-point table, interpolated as ROOT's
  ``TGraph::Eval(Q, 0, "S")`` does (``TSpline3`` with no end conditions, i.e. de
  Boor's not-a-knot cubic spline).

All expose ``jitter(q, u)`` (inverse-CDF draw from one uniform ``u`` per digit)
and ``mean_shift(q)``, so the readout chain treats them uniformly.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import ndtr, ndtri

SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))


def domain(q, interval):
    q = np.asarray(q, dtype=float)
    if len(interval) != 2 or not np.all(np.isfinite(interval)) or interval[0] < 0 or interval[0] >= interval[1]:
        raise ValueError("invalid charge domain")
    if np.any(~np.isfinite(q)) or np.any((q < interval[0]) | (q > interval[1])):
        raise ValueError("charge outside calibrated domain; extrapolation forbidden")
    return q


@dataclass(frozen=True)
class WalkCurve:
    coefficients_ns: tuple[float, ...] = (0.0,)
    domain_pe: tuple[float, float] = (0.25, 100.0)

    def __post_init__(self):
        domain([], self.domain_pe)
        if not 1 <= len(self.coefficients_ns) <= 4 or not np.all(np.isfinite(self.coefficients_ns)):
            raise ValueError("finite polynomial degree 0..3 required")

    def __call__(self, q):
        q = domain(q, self.domain_pe)
        u = (q-self.domain_pe[0])/(self.domain_pe[1]-self.domain_pe[0])
        return np.polynomial.polynomial.polyval(u, self.coefficients_ns)

    @classmethod
    def fit(cls, q, location_ns, *, degree, domain_pe):
        q = domain(q, domain_pe)
        t = np.asarray(location_ns, dtype=float)
        if q.shape != t.shape or not 0 <= degree <= 3 or len(np.unique(q)) <= degree or not np.all(np.isfinite(t)):
            raise ValueError("invalid reference calibration samples")
        u = (q-domain_pe[0])/(domain_pe[1]-domain_pe[0])
        return cls(tuple(np.polynomial.polynomial.polyfit(u, t, degree)), domain_pe)


@dataclass(frozen=True)
class InverseSqrtWalk:
    """D-TQ walk ``a + b/sqrt(q)`` on [domain low, q_sat]; constant above q_sat."""
    a_ns: float = 0.0
    b_ns: float = 2.0
    q_sat_pe: float = 50.0
    domain_pe: tuple[float, float] = (0.25, 1000.0)

    def __post_init__(self):
        domain([], self.domain_pe)
        if self.domain_pe[0] <= 0 or not np.all(np.isfinite([self.a_ns, self.b_ns, self.q_sat_pe])) or not self.domain_pe[0] < self.q_sat_pe <= self.domain_pe[1]:
            raise ValueError("finite a, b and q_sat inside a positive charge domain required")

    def __call__(self, q):
        return self.a_ns + self.b_ns/np.sqrt(np.minimum(domain(q, self.domain_pe), self.q_sat_pe))

    def scaled(self, slope_factor=1.0, offset_ns=0.0):
        """Same form with slope ``b*slope_factor`` and offset ``a+offset_ns``."""
        return InverseSqrtWalk(self.a_ns+offset_ns, self.b_ns*slope_factor, self.q_sat_pe, self.domain_pe)


@dataclass(frozen=True)
class WidthCurve:
    charge_pe: tuple[float, ...] = (0.25, 1.0, 5.0, 100.0)
    sigma_ns: tuple[float, ...] = (2.5, 1.5, 0.8, 0.5)

    def __post_init__(self):
        if len(self.charge_pe) < 2 or len(self.charge_pe) != len(self.sigma_ns) or not np.all(np.isfinite(self.charge_pe)) or np.any(np.diff(self.charge_pe) <= 0) or self.charge_pe[0] < 0 or not np.all(np.isfinite(self.sigma_ns)) or np.any(np.asarray(self.sigma_ns) < 0):
            raise ValueError("invalid charge/width table")

    @property
    def domain_pe(self) -> tuple[float, float]:
        return (float(self.charge_pe[0]), float(self.charge_pe[-1]))

    def __call__(self, q):
        return np.interp(domain(q, self.domain_pe), self.charge_pe, self.sigma_ns)

    def jitter(self, q, u):
        """Symmetric-Gaussian inverse-CDF jitter: ``sigma(q) * ndtri(u)``."""
        u = _uniforms(u, np.shape(q))
        return self(q) * ndtri(u)

    def mean_shift(self, q):
        """Symmetric Gaussian: the mode and the mean coincide (shift 0)."""
        domain(q, self.domain_pe)
        return np.zeros(np.shape(q), dtype=float)


@dataclass(frozen=True)
class SplitNormalWidth:
    """Thesis Eq. 6.3 split-normal digit width (R-7, author message, 23 Oct 2026).

    Density ``sqrt(2/pi)/(sigma_e + sigma_l) * exp(-(t - t0)^2 / (2 sigma^2))``,
    with ``sigma = sigma_e`` for ``t < t0`` (early side) and ``sigma = sigma_l``
    for ``t >= t0`` (late side). ``t0`` is the mode: photon arrival plus walk.
    The sampled mean is therefore ``t0 + sqrt(2/pi) * (sigma_l - sigma_e)``.

    ``sigma_e`` and ``sigma_l`` are tabulated in measured charge and interpolated
    linearly in ``log10(q)``, as the author's R-7 message requires. The knot at
    2 PE sits above its neighbours; that bump is in the measured table and is
    kept. Values are literature proxies (Hyper-Kamiokande 20-inch Box&Line PMT,
    thesis Fig. 6.3 / Fig. 6.4(b)), not WCTE measurements.
    """

    charge_pe: tuple[float, ...]
    sigma_e_ns: tuple[float, ...]
    sigma_l_ns: tuple[float, ...]

    def __post_init__(self):
        q = np.asarray(self.charge_pe, dtype=float)
        e = np.asarray(self.sigma_e_ns, dtype=float)
        l = np.asarray(self.sigma_l_ns, dtype=float)
        if q.ndim != 1 or len(q) < 2 or q.shape != e.shape or q.shape != l.shape:
            raise ValueError("charge/sigma_e/sigma_l tables must be equal-length (>= 2)")
        if not np.all(np.isfinite(q)) or np.any(np.diff(q) <= 0) or q[0] <= 0:
            raise ValueError("charge table must be finite, strictly increasing and positive")
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(l)) or np.any(e <= 0) or np.any(l <= 0):
            raise ValueError("split-normal sigmas must be finite and positive on both sides")

    @property
    def domain_pe(self) -> tuple[float, float]:
        return (float(self.charge_pe[0]), float(self.charge_pe[-1]))

    def _interp(self, values, q):
        q = domain(q, self.domain_pe)
        return np.interp(np.log10(q), np.log10(np.asarray(self.charge_pe, dtype=float)), np.asarray(values, dtype=float))

    def sigma_e(self, q):
        """Early-side standard deviation (ns) at measured charge ``q`` (PE)."""
        return self._interp(self.sigma_e_ns, q)

    def sigma_l(self, q):
        """Late-side standard deviation (ns) at measured charge ``q`` (PE)."""
        return self._interp(self.sigma_l_ns, q)

    def pdf(self, x, q):
        """Density at offset ``x = t - t0`` (ns), per unit time, at charge ``q``."""
        q = np.asarray(q, dtype=float)
        x = np.asarray(x, dtype=float)
        e, l = self.sigma_e(q), self.sigma_l(q)
        s = np.where(x < 0, e, l)
        return SQRT_2_OVER_PI / (e + l) * np.exp(-x ** 2 / (2.0 * s ** 2))

    def cdf(self, x, q):
        """Cumulative probability of an offset ``x = t - t0`` (ns), at charge ``q``."""
        q = np.asarray(q, dtype=float)
        x = np.asarray(x, dtype=float)
        e, l = self.sigma_e(q), self.sigma_l(q)
        out = np.where(x < 0,
                       2.0 * (e / (e + l)) * ndtr(x / e),
                       e / (e + l) + 2.0 * (l / (e + l)) * (ndtr(x / l) - 0.5))
        return out

    def mean_shift(self, q):
        """Mean minus mode: ``sqrt(2/pi) * (sigma_l - sigma_e)`` (R-7 records this shift)."""
        s = self.sigma_l(q) - self.sigma_e(q)
        return SQRT_2_OVER_PI * s

    def jitter(self, q, u):
        """Inverse-CDF draw from the split normal, one uniform ``u`` per digit.

        ``sqrt(2/pi)/(sigma_e + sigma_l)`` normalises the density over both
        halves, so the early half carries probability ``sigma_e/(sigma_e +
        sigma_l)``. Inverting each half with ``ndtri`` keeps the draw a
        deterministic function of ``(q, u)``, which is what pairs the response
        cells: scaling both sigmas by a per-module factor scales the draw by the
        same factor (the ratio, and so the half probabilities, is unchanged).
        """
        q = np.asarray(q, dtype=float)
        u = _uniforms(u, q.shape)
        e, l = self.sigma_e(q), self.sigma_l(q)
        w = e / (e + l)
        lo, hi = np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)
        # Early half: CDF(x) = 2 w Phi(x/sigma_e), so Phi = u/(2w).
        early = e * ndtri(np.clip(u / (2.0 * w), lo, hi))
        # Late half: CDF(x) = w + 2(1-w)(Phi(x/sigma_l) - 1/2).
        late = l * ndtri(np.clip(0.5 + (u - w) / (2.0 * (1.0 - w)), lo, hi))
        return np.where(u < w, early, late)


@dataclass(frozen=True)
class WCSimTTSWidth:
    """WCSim ``PMT3inchR14374_WCTE::HitTimeSmearing`` (R-7 v2, the ``pmt-response-v3`` default).

    ``sigma(Q) = sqrt(tts_fwhm**2 + R(min(Q, q_clamp))**2) / fwhm_to_sigma`` ns, a
    symmetric Gaussian about the photon arrival plus walk (mean shift 0). ``R`` is
    the FWHM charge-resolution table (WCSim ``c5ed0b1``, ``src/WCSimPMTObject.cc``
    lines 3809-3843), interpolated with a not-a-knot cubic spline: ROOT's
    ``TGraph::Eval(Q, 0, "S")`` builds a ``TSpline3`` with no end conditions, which
    is de Boor's CUBSPL not-a-knot case (``scipy.interpolate.CubicSpline``,
    ``bc_type="not-a-knot"``). Above ``q_clamp`` (4 PE) the width is constant, as
    in WCSim. Below the first knot (0.2 PE) WCSim's spline would extrapolate; here
    the declared charge domain must start at or above that knot, so no charge
    ever reaches an extrapolated width. The per-module width factor multiplies
    ``sigma``, as WCSim's ``TTSFF`` does.
    """

    charge_pe: tuple[float, ...]
    resolution_fwhm_ns: tuple[float, ...]
    tts_fwhm_ns: float = 1.5
    q_clamp_pe: float = 4.0
    domain_pe: tuple[float, float] = (0.25, 1000.0)
    fwhm_to_sigma: float = 2.355

    def __post_init__(self):
        q = np.asarray(self.charge_pe, dtype=float)
        r = np.asarray(self.resolution_fwhm_ns, dtype=float)
        if q.ndim != 1 or len(q) < 4 or q.shape != r.shape:
            raise ValueError("charge/resolution tables must be equal-length (>= 4 knots for a not-a-knot spline)")
        if not np.all(np.isfinite(q)) or np.any(np.diff(q) <= 0) or q[0] <= 0:
            raise ValueError("charge table must be finite, strictly increasing and positive")
        if not np.all(np.isfinite(r)) or np.any(r < 0):
            raise ValueError("resolution table must be finite and non-negative")
        if not all(np.isfinite([self.tts_fwhm_ns, self.q_clamp_pe, self.fwhm_to_sigma])) or self.tts_fwhm_ns < 0 or self.fwhm_to_sigma <= 0:
            raise ValueError("finite non-negative TTS FWHM and positive FWHM-to-sigma factor required")
        if not q[0] < self.q_clamp_pe <= q[-1]:
            raise ValueError("charge clamp must lie inside the resolution table")
        domain([], self.domain_pe)
        if self.domain_pe[0] < q[0]:
            raise ValueError("charge domain starts below the first resolution knot; extrapolation forbidden")
        object.__setattr__(self, "_spline", CubicSpline(q, r, bc_type="not-a-knot"))

    def resolution_fwhm(self, q):
        """``R(min(q, q_clamp))`` (ns FWHM) at measured charge ``q`` (PE)."""
        q = domain(q, self.domain_pe)
        return self._spline(np.minimum(q, self.q_clamp_pe))

    def __call__(self, q):
        """Gaussian ``sigma`` (ns) at measured charge ``q`` (PE)."""
        return np.sqrt(self.tts_fwhm_ns ** 2 + self.resolution_fwhm(q) ** 2) / self.fwhm_to_sigma

    def pdf(self, x, q):
        """Gaussian density at offset ``x = t - t0`` (ns) at charge ``q``."""
        s = self(q)
        return np.exp(-np.asarray(x, dtype=float) ** 2 / (2.0 * s ** 2)) / (np.sqrt(2.0 * np.pi) * s)

    def cdf(self, x, q):
        """Cumulative probability of an offset ``x = t - t0`` (ns) at charge ``q``."""
        return ndtr(np.asarray(x, dtype=float) / self(q))

    def jitter(self, q, u):
        """Symmetric-Gaussian inverse-CDF jitter: ``sigma(q) * ndtri(u)``."""
        u = _uniforms(u, np.shape(q))
        return self(q) * ndtri(u)

    def mean_shift(self, q):
        """Symmetric Gaussian: mode and mean coincide (shift 0)."""
        domain(q, self.domain_pe)
        return np.zeros(np.shape(q), dtype=float)


def _uniforms(u, shape):
    """Validate one strictly-interior uniform (0, 1) per entry of ``shape``."""
    u = np.asarray(u, dtype=float)
    if u.shape != tuple(shape):
        raise ValueError(f"one uniform per charge required: {u.shape} != {tuple(shape)}")
    if np.any(~np.isfinite(u)) or np.any((u <= 0.0) | (u >= 1.0)):
        raise ValueError("uniforms must lie strictly inside (0, 1)")
    return u


@dataclass(frozen=True)
class ChannelWalkCurve:
    curves: tuple[WalkCurve, ...]

    def evaluate(self, q, sensor):
        sensor = np.asarray(sensor)
        if sensor.shape != np.asarray(q).shape or np.any((sensor < 0) | (sensor >= len(self.curves))):
            raise ValueError("missing channel T-Q curve")
        return np.array([self.curves[int(s)](x) for s,x in zip(sensor,q)])


def evaluate_walk(curve, q, sensor):
    return curve.evaluate(q, sensor) if isinstance(curve, ChannelWalkCurve) else curve(q)
