"""Forward-time Gaussian engineering model, not a reproduction of thesis Eq.6.3."""
from dataclasses import dataclass
import numpy as np


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
class WidthCurve:
    charge_pe: tuple[float, ...] = (0.25, 1.0, 5.0, 100.0)
    sigma_ns: tuple[float, ...] = (2.5, 1.5, 0.8, 0.5)

    def __post_init__(self):
        if len(self.charge_pe) < 2 or len(self.charge_pe) != len(self.sigma_ns) or not np.all(np.isfinite(self.charge_pe)) or np.any(np.diff(self.charge_pe) <= 0) or self.charge_pe[0] < 0 or not np.all(np.isfinite(self.sigma_ns)) or np.any(np.asarray(self.sigma_ns) < 0):
            raise ValueError("invalid charge/width table")

    def __call__(self, q):
        return np.interp(domain(q, (self.charge_pe[0], self.charge_pe[-1])), self.charge_pe, self.sigma_ns)


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
