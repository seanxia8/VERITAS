"""Explicit group gain/width proxy; no conversion to magnetic field strength."""
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class GroupMap:
    channel_group: tuple[int, ...]
    provenance: str
    source: str

    def __post_init__(self):
        a = np.asarray(self.channel_group)
        if a.ndim != 1 or not a.size or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
            raise ValueError("one nonnegative group ID per channel required")
        object.__setattr__(self, "channel_group", tuple(int(v) for v in a))
        if self.provenance not in ("physical", "synthetic_dev") or not self.source:
            raise ValueError("group provenance/source required")


def group_intervention(ref, mapping, affected, *, gain_factor=1.0, width_factor=1.0):
    if not np.all(np.isfinite([gain_factor, width_factor])) or min(gain_factor, width_factor) < 0:
        raise ValueError("finite nonnegative group factors required")
    a = np.asarray(mapping.channel_group)
    if not set(affected) <= set(a):
        raise ValueError("unmapped group")
    hit = np.isin(a, affected)
    gain = np.broadcast_to(ref.gain, a.shape).copy().astype(float)
    width = np.broadcast_to(ref.timing_width_factor, a.shape).copy().astype(float)
    gain[hit] *= gain_factor
    width[hit] *= width_factor
    return replace(ref, gain=gain, timing_width_factor=width)


def validate_cell(cell, ref):
    moved = set(cell.params.moved_from(ref))
    if moved & {"reference_walk", "reference_offsets_ns", "walk_calibration_pe"}:
        raise ValueError("reference calibration is frozen across cells")
    if cell.family == "fixed_tq" and moved - {"walk_curve"}:
        raise ValueError("pure T-Q may change only walk_curve")
    if len(moved) <= 1:
        return
    if cell.family != "group_response" or moved != {"gain", "timing_width_factor"} or set(cell.changed_fields) != moved or not cell.causal_control or not cell.affected_groups:
        raise ValueError("multi-field cell requires explicit coupled group response declaration")
