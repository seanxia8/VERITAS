"""Versioned photon-stage and operational-observation boundary."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PhotonBatch:
    event_id: int
    photon_id: np.ndarray
    sensor_id: np.ndarray
    arrival_time_ns: np.ndarray
    wavelength_nm: np.ndarray | None
    stage: str = "incident_photons"
    applied_effects: tuple[str, ...] = ()
    schema: str = "pmt-photons-v1"
    units: tuple[str, str, str] = ("m", "ns", "nm")

    def validate(self, n_channels):
        if self.schema != "pmt-photons-v1" or self.units != ("m", "ns", "nm"):
            raise ValueError("wrong schema or units")
        if self.stage not in ("incident_photons", "detected_pe"):
            raise ValueError("unknown photon stage")
        pid, sid, t = map(np.asarray, (self.photon_id, self.sensor_id, self.arrival_time_ns))
        if pid.ndim != 1 or sid.shape != pid.shape or t.shape != pid.shape:
            raise ValueError("photon fields must be aligned 1-D arrays")
        if not np.issubdtype(pid.dtype, np.integer) or np.any(pid < 0) or len(np.unique(pid)) != len(pid):
            raise ValueError("persistent photon IDs must be unique nonnegative integers")
        if not np.issubdtype(sid.dtype, np.integer) or np.any((sid < 0) | (sid >= n_channels)) or not np.all(np.isfinite(t)):
            raise ValueError("invalid sensor/time")
        if self.wavelength_nm is not None:
            w = np.asarray(self.wavelength_nm)
            if w.shape != pid.shape or not np.all(np.isfinite(w)) or np.any(w <= 0):
                raise ValueError("invalid wavelength")
        if len(set(self.applied_effects)) != len(self.applied_effects) or set(self.applied_effects) - {"QE", "CE", "TTS"}:
            raise ValueError("unknown or repeated upstream effect")
        if self.stage == "incident_photons" and set(self.applied_effects) & {"QE", "CE"}:
            raise ValueError("incident photons cannot already have QE/CE")
        if self.stage == "detected_pe" and not {"QE", "CE"} <= set(self.applied_effects):
            raise ValueError("detected PE must declare QE and CE ownership")
        return self


@dataclass(frozen=True)
class OperationalDigits:
    sensor: np.ndarray
    charge: np.ndarray
    time: np.ndarray
    positions: np.ndarray
    group_id: np.ndarray


def operational_digits(digits, positions, group_id):
    """Allowlist projection: no truth counts, wavelengths, twins or cell labels."""
    return OperationalDigits(*(np.array(x, copy=True) for x in
        (digits.sensor, digits.charge, digits.time, np.asarray(positions)[digits.sensor], np.asarray(group_id)[digits.sensor])))


def monitor_features(observed):
    if type(observed) is not OperationalDigits:
        raise TypeError("monitor requires OperationalDigits")
    q, t = observed.charge, observed.time
    return np.array([q.sum(), len(q), q.std() if len(q) else 0,
                     t.mean() if len(t) else 0, t.std() if len(t) else 0])
