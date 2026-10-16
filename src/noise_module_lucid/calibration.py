"""Frozen reference-fit artifact, independent of evaluation events and twins."""
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
from .readout import readout_chain
from .consequence import energy_calibration, FitConfig


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def event_digits(event, n_channels, params):
    return readout_chain(event.photon_sensor, event.photon_time, n_channels, params,
                         seed=int(event.event_id), photon_batch=getattr(event, "photon_batch", None))


@dataclass(frozen=True)
class ReferenceCalibration:
    event_ids: tuple[int, ...]
    energy_scale: float
    channel_share: tuple[float, ...]
    positions_hash: str
    response_hash: str
    input_hash: str
    fit_config_hash: str
    schema: str = "pmt-reference-calibration-v1"

    def __post_init__(self):
        a = np.asarray(self.channel_share)
        if not self.event_ids or len(set(self.event_ids)) != len(self.event_ids) or not np.isfinite(self.energy_scale) or self.energy_scale <= 0 or a.ndim != 1 or np.any(~np.isfinite(a)) or np.any(a < 0) or not np.isclose(a.sum(), 1):
            raise ValueError("invalid calibration artifact")
        if self.schema != "pmt-reference-calibration-v1":
            raise ValueError("unknown calibration schema")

    @property
    def sha256(self):
        return digest(asdict(self))

    def save(self, path):
        from pathlib import Path
        Path(path).write_text(json.dumps({"payload": asdict(self), "sha256": self.sha256}, indent=2))

    @classmethod
    def load(cls, path):
        from pathlib import Path
        d = json.loads(Path(path).read_text())
        if digest(d["payload"]) != d["sha256"]:
            raise ValueError("calibration hash mismatch")
        v = d["payload"]
        v["event_ids"] = tuple(v["event_ids"])
        v["channel_share"] = tuple(v["channel_share"])
        return cls(**v)

    def validate_evaluation(self, events, positions, reference, fit_config=None):
        if self.fit_config_hash != digest(asdict(fit_config or FitConfig())):
            raise ValueError("frozen reconstruction configuration mismatch")
        ids = [int(e.event_id) for e in events]
        if not ids or len(set(ids)) != len(ids) or set(ids) & set(self.event_ids):
            raise ValueError("calibration/evaluation event IDs must be disjoint and unique")
        if self.positions_hash != digest(np.asarray(positions).tolist()) or self.response_hash != digest(reference.to_dict()) or len(self.channel_share) != len(positions):
            raise ValueError("calibration detector/reference mismatch")


def fit_reference_calibration(events, positions, reference, *, fit_config=None):
    if not events:
        raise ValueError("separate reference-fit events required")
    nch = len(positions)
    ds = [event_digits(e, nch, reference) for e in events]
    totals = [float(d.charge.sum()) for d in ds]
    shares = np.zeros(nch)
    for d, total in zip(ds, totals):
        if total <= 0:
            raise ValueError("zero reference charge")
        shares[d.sensor] += d.charge / total
    inputs = [{"event_id": int(e.event_id), "energy": float(e.true_energy),
               "sensor": np.asarray(e.photon_sensor).tolist(), "time": np.asarray(e.photon_time).tolist(),
               "batch": None if getattr(e, "photon_batch", None) is None else
                 {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in asdict(e.photon_batch).items()}}
              for e in events]
    return ReferenceCalibration(tuple(int(e.event_id) for e in events),
        energy_calibration(totals, [e.true_energy for e in events]), tuple(shares/len(events)),
        digest(np.asarray(positions).tolist()), digest(reference.to_dict()), digest(inputs), digest(asdict(fit_config or FitConfig())))
