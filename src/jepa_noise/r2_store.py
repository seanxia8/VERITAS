# SPDX-License-Identifier: MIT
"""Round-2 storage: one HDF5 file per detector, its byte-for-byte verification, and the
sparse-cache reader that serves training batches and probe sets (runbook §3.2, §3.3).

Layout of ``<det>_r2.h5`` (one group per cohort)::

    /<cohort>/run_id            (n_runs,) bytes      run ids, in run order
    /<cohort>/run_events        (n_runs + 1,) int64  event offsets of each run
    /<cohort>/run_state_sha256  (n_runs,) bytes      the per-run detector-state hash
    /<cohort>/event_ids         (n_events,) int64
    /<cohort>/offsets           (n_events + 1,) int64  CSR hit offsets (global)
    /<cohort>/pmt_id            (n_hits,) int32
    /<cohort>/charge_pe         (n_hits,) float32
    /<cohort>/time_ns           (n_hits,) float32
    /<cohort>/twin_offsets, twin_pmt_id, twin_charge_pe, twin_time_ns   (probe cohorts)
    /<cohort>/truth/<field>     the round-1 ``events.npz`` fields (+ ``expected_pe_pmt``)
    attrs: per-run manifests (JSON), geometry and config hashes, the §2.2 derived values

Every run is re-read from the file and compared with its shard (``hits.npz``,
``twin_hits.npz``, ``events.npz``, ``manifest.json``) array for array, dtype and bytes.

Reader: sparse runs stay in RAM; dense tokens are built per batch (the round-1 dense
cache would need 65-125 GB here).  :func:`tokenise_sparse` is ``point_encoder.tokenise``
on an in-memory :class:`~jepa_noise.lucid_bombs.SparseHitStore`; :class:`R2TrainingLoader`
keeps the round-1 sampling (4 events from each of 8 runs, keyed by (seed, step)).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import lucid_bombs as lb
from . import lucid_run as lr
from . import point_encoder as pe
from .lucid_views import ViewMasker

__all__ = ["HIT_FIELDS", "TRUTH_FIELDS", "pack_detector", "verify_pack", "R2Cohort",
           "open_cohort", "tokenise_sparse", "R2TrainingLoader", "scaling_for", "sha256_file"]

HIT_FIELDS = ("pmt_id", "charge_pe", "time_ns")
TRUTH_FIELDS = ("vertex_m", "yield_photons", "expected_pe", "emission_offset_ns", "domain",
                "particle", "direction_m", "energy_MeV", "deposited_MeV", "contained",
                "track_length_m", "expected_pe_pmt")
COMPRESSION = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
SCHEMA = "jepa-r2-detector-pack-v1"


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def scaling_for(geometry, time_scale_ns: float) -> pe.FeatureScaling:
    """Round-1 fixed scaling with the §2.2 time scale: (R, R, H/2) [sphere (R, R, R)], T."""
    return pe.FeatureScaling(radial_scale_m=float(geometry.radius_m),
                             axial_scale_m=float(geometry.half_height_m),
                             time_scale_ns=float(time_scale_ns))


# ======================================================================= pack
def _run_dirs(shard_root: Path, cohort: str, runs) -> list:
    return [shard_root / cohort / run.run_id.replace("/", "__") for run in runs]


def _chunk(n: int, per_run: int) -> tuple | None:
    if n <= 0:
        return None
    return (int(max(1, min(n, per_run))),)


def pack_detector(h5_path, shard_root, cohorts: dict, *, attrs: dict) -> dict:
    """Pack per-run shards into one HDF5 file (written to ``<path>.tmp`` then renamed).

    ``cohorts`` maps cohort name -> run specs (``r2_generator.cohort_runs``).
    """
    import h5py
    h5_path, shard_root = Path(h5_path), Path(shard_root)
    tmp = h5_path.with_name(h5_path.name + ".tmp")
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with h5py.File(tmp, "w") as f:
        f.attrs["schema"] = SCHEMA
        for key, value in attrs.items():
            f.attrs[key] = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        for cohort, runs in cohorts.items():
            dirs = _run_dirs(shard_root, cohort, runs)
            stores = [lb.read_sparse_hits(d / "hits.npz") for d in dirs]
            twin_present = (dirs[0] / "twin_hits.npz").is_file()
            twins = [lb.read_sparse_hits(d / "twin_hits.npz") for d in dirs] if twin_present else None
            g = f.create_group(cohort)
            n_ev = np.array([s.event_ids.size for s in stores], np.int64)
            run_events = np.concatenate([[0], np.cumsum(n_ev)]).astype(np.int64)
            g.create_dataset("run_id", data=np.array([r.run_id.encode() for r in runs]))
            g.create_dataset("run_events", data=run_events)
            manifests = [(d / "manifest.json").read_text() for d in dirs]
            g.create_dataset("run_state_sha256", data=np.array(
                [json.loads(m)["runs"][0]["state_sha256"].encode() for m in manifests]))
            g.create_dataset("run_manifest", data=np.array([m.encode() for m in manifests]))
            per_run_events = int(n_ev.max())
            g.create_dataset("event_ids", data=np.concatenate([s.event_ids for s in stores]),
                             chunks=_chunk(int(n_ev.sum()), per_run_events), **COMPRESSION)
            for prefix, group in (("", stores), ("twin_", twins)):
                if group is None:
                    continue
                n_hits = np.array([s.pmt_id.size for s in group], np.int64)
                base = np.concatenate([[0], np.cumsum(n_hits)])[:-1]
                offsets = np.concatenate([s.offsets[:-1] + b for s, b in zip(group, base)]
                                         + [[int(n_hits.sum())]]).astype(np.int64)
                g.create_dataset(prefix + "offsets", data=offsets,
                                 chunks=_chunk(offsets.size, per_run_events + 1), **COMPRESSION)
                per_run_hits = int(n_hits.max()) if n_hits.size else 1
                for name in HIT_FIELDS:
                    data = np.concatenate([getattr(s, name) for s in group])
                    g.create_dataset(prefix + name, data=data,
                                     chunks=_chunk(data.size, per_run_hits), **COMPRESSION)
            truth_group = g.create_group("truth")
            loaded = []
            for d in dirs:
                with np.load(d / "events.npz", allow_pickle=False) as data:
                    loaded.append({k: data[k] for k in data.files})
            if not all(np.array_equal(t["event_ids"], s.event_ids) for t, s in zip(loaded, stores)):
                raise ValueError(f"{cohort}: events.npz and hits.npz event ids disagree")
            for name in TRUTH_FIELDS:
                if name not in loaded[0]:
                    continue
                data = np.concatenate([t[name] for t in loaded])
                if data.dtype.kind == "U":
                    data = data.astype("S")
                chunks = (min(data.shape[0], per_run_events),) + tuple(data.shape[1:])
                truth_group.create_dataset(name, data=data, chunks=chunks, **COMPRESSION)
            g.attrs["twin_present"] = bool(twin_present)
            g.attrs["n_runs"] = len(runs)
            g.attrs["n_events"] = int(n_ev.sum())
            summary[cohort] = {"n_runs": len(runs), "n_events": int(n_ev.sum()),
                               "n_hits": int(sum(s.pmt_id.size for s in stores)),
                               "twin_present": bool(twin_present)}
    tmp.replace(h5_path)
    return summary


def verify_pack(h5_path, shard_root, cohorts: dict) -> dict:
    """Re-read every run from the HDF5 file and compare it with its shard, byte for byte."""
    import h5py
    shard_root = Path(shard_root)
    checked = {}
    with h5py.File(h5_path, "r") as f:
        for cohort, runs in cohorts.items():
            g = f[cohort]
            dirs = _run_dirs(shard_root, cohort, runs)
            run_events = g["run_events"][:]
            offsets = g["offsets"][:]
            twin_offsets = g["twin_offsets"][:] if "twin_offsets" in g else None
            for i, (run, d) in enumerate(zip(runs, dirs)):
                if g["run_id"][i].decode() != run.run_id:
                    raise ValueError(f"{cohort} run {i}: run id mismatch")
                if g["run_manifest"][i].decode() != (d / "manifest.json").read_text():
                    raise ValueError(f"{cohort} {run.run_id}: manifest text differs")
                a, b = int(run_events[i]), int(run_events[i + 1])
                for prefix, name, offs in (("", "hits.npz", offsets),
                                           ("twin_", "twin_hits.npz", twin_offsets)):
                    shard = d / name
                    if offs is None:
                        if shard.is_file():
                            raise ValueError(f"{cohort} {run.run_id}: twin shard not packed")
                        continue
                    ref = lb.read_sparse_hits(shard)
                    got_off = offs[a:b + 1] - offs[a]
                    lo, hi = int(offs[a]), int(offs[b])
                    pairs = [("event_ids", g["event_ids"][a:b], ref.event_ids),
                             ("offsets", got_off.astype(ref.offsets.dtype), ref.offsets)]
                    pairs += [(f, g[prefix + f][lo:hi], getattr(ref, f)) for f in HIT_FIELDS]
                    for label, got, want in pairs:
                        if got.dtype != want.dtype or got.tobytes() != want.tobytes():
                            raise ValueError(f"{cohort} {run.run_id} {prefix}{label}: bytes differ")
                with np.load(d / "events.npz", allow_pickle=False) as data:
                    for name in data.files:
                        want = data[name]
                        got = g["event_ids"][a:b] if name == "event_ids" else g["truth"][name][a:b]
                        if want.dtype.kind == "U":
                            got = got.astype(want.dtype)
                        if got.dtype != want.dtype or got.tobytes() != want.tobytes():
                            raise ValueError(f"{cohort} {run.run_id} truth {name}: bytes differ")
            checked[cohort] = len(runs)
    return {"verified_runs": checked, "sha256": sha256_file(h5_path)}


# ======================================================================= reader
@dataclass
class R2Cohort:
    """One cohort held sparse in RAM, with per-run views."""

    name: str
    run_ids: list
    run_events: np.ndarray
    event_ids: np.ndarray
    offsets: np.ndarray
    pmt_id: np.ndarray
    charge_pe: np.ndarray
    time_ns: np.ndarray
    truth: dict
    state_sha256: list
    manifests: list

    @property
    def n_runs(self) -> int:
        return len(self.run_ids)

    def run_slice(self, i: int) -> slice:
        return slice(int(self.run_events[i]), int(self.run_events[i + 1]))

    def run_store(self, i: int) -> lb.SparseHitStore:
        """The round-1 per-run sparse store of run ``i`` (identical to its ``hits.npz``)."""
        s = self.run_slice(i)
        offs = self.offsets[s.start:s.stop + 1]
        lo, hi = int(offs[0]), int(offs[-1])
        return lb.SparseHitStore(self.event_ids[s], (offs - offs[0]).astype(np.int64),
                                 self.pmt_id[lo:hi], self.charge_pe[lo:hi], self.time_ns[lo:hi])

    def run_truth(self, i: int) -> dict:
        s = self.run_slice(i)
        out = {"event_ids": self.event_ids[s]}
        for name, values in self.truth.items():
            out[name] = values[s]
        return out

    def run_rows(self) -> list:
        """Round-1 ``Dataset.runs``-like rows (run id, start, length, setting)."""
        rows = []
        for i, run_id in enumerate(self.run_ids):
            manifest = json.loads(self.manifests[i])["runs"][0]
            rows.append({"run_id": run_id, "index": i,
                         "event_id_start": int(manifest["event_id_start"]),
                         "n_events": int(manifest["n_events"]),
                         "setting": float(manifest["setting"])})
        return rows


def open_cohort(h5_path, cohort: str, *, twin: bool = False, truth_fields=None) -> R2Cohort:
    """Load one cohort into RAM (sparse); ``twin=True`` serves the twin hits instead."""
    import h5py
    prefix = "twin_" if twin else ""
    with h5py.File(h5_path, "r") as f:
        g = f[cohort]
        truth = {}
        for name in (truth_fields if truth_fields is not None else g["truth"].keys()):
            if name == "expected_pe_pmt" and truth_fields is None:
                continue  # large; load explicitly when needed
            data = g["truth"][name][:]
            truth[name] = data.astype(str) if data.dtype.kind == "S" else data
        return R2Cohort(name=cohort, run_ids=[x.decode() for x in g["run_id"][:]],
                        run_events=g["run_events"][:], event_ids=g["event_ids"][:],
                        offsets=g[prefix + "offsets"][:], pmt_id=g[prefix + "pmt_id"][:],
                        charge_pe=g[prefix + "charge_pe"][:], time_ns=g[prefix + "time_ns"][:],
                        truth=truth, state_sha256=[x.decode() for x in g["run_state_sha256"][:]],
                        manifests=[x.decode() for x in g["run_manifest"][:]])


def tokenise_sparse(store: lb.SparseHitStore, geometry, *, event_ids=None, limit=None,
                    split: str = "within", seed: int = 0, occupancy: str = "natural",
                    scaling: pe.FeatureScaling | None = None, masker: ViewMasker | None = None,
                    n_min: int | None = None, device="cpu") -> pe.LucidBatch:
    """``point_encoder.tokenise`` on an in-memory sparse store (statement for statement)."""
    scaling = scaling or pe.FeatureScaling.from_geometry(geometry)
    base = np.zeros((geometry.n_pmt, pe.N_FEATURES), dtype=np.float32)
    base[:, :6] = pe.geometry_features(geometry, scaling)
    if masker is None:
        kw = {} if n_min is None else {"n_min": int(n_min)}
        masker = ViewMasker(geometry, split, seed=seed, occupancy=occupancy, **kw)
    wanted = None if event_ids is None else {int(x) for x in np.asarray(event_ids).ravel()}
    features, targets, hidden, contexts, target_indices = [], [], [], [], []
    hits, event_kept, keys = [], [], []
    for index, event_id in enumerate(store.event_ids):
        if wanted is not None and int(event_id) not in wanted:
            continue
        if limit is not None and len(event_kept) >= int(limit):
            break
        event, hit = pe._event_features(store, index, geometry, scaling, base)
        mask = masker.mask(int(event_id), hit)
        if not mask.accepted:
            continue
        features.append(event)
        targets.append(mask.target)
        hidden.append(mask.hidden)
        contexts.append(mask.context)
        target_indices.append(mask.target)
        hits.append(hit)
        event_kept.append(int(event_id))
        keys.append(mask.key)
    if not features:
        raise RuntimeError("tokenise: every event was excluded by the N_min rule")
    if len({t.size for t in targets}) != 1 or len({c.size for c in contexts}) != 1:
        raise ValueError("tokenise: ragged views; one split per batch only")
    target_np = np.zeros((len(features), geometry.n_pmt), dtype=bool)
    hidden_np = np.zeros_like(target_np)
    for row, (target, hide) in enumerate(zip(targets, hidden)):
        target_np[row, target] = True
        hidden_np[row, hide] = True
    features_np = np.stack(features)
    return pe.LucidBatch(
        features=torch.as_tensor(features_np, dtype=torch.float32, device=device),
        target=torch.as_tensor(target_np, dtype=torch.bool, device=device),
        hidden=torch.as_tensor(hidden_np, dtype=torch.bool, device=device),
        context_index=torch.as_tensor(np.stack(contexts), dtype=torch.long, device=device),
        target_index=torch.as_tensor(np.stack(target_indices), dtype=torch.long, device=device),
        hit_flags=np.stack(hits), log_charge=features_np[..., pe.LOG_CHARGE_INDEX].copy(),
        event_ids=np.asarray(event_kept, dtype=np.int64), keys=tuple(keys),
        report=masker.report(), scaling=scaling, split=masker.split,
        occupancy=masker.occupancy)


class R2TrainingLoader:
    """Round-1 ``TrainingLoader.batch_at`` on the sparse cache: 4 events from each of 8 runs.

    The run choice and the per-run event draw use the round-1 keyed stream
    ``keyed_rng(seed, "lucid/batches", step)`` unchanged; dense tokens are built only for
    the drawn events (the within view is per-event deterministic, so the batch equals the
    round-1 one built from the whole tokenised run).
    """

    def __init__(self, cohort: R2Cohort, geometry, *, seed: int, scaling, split: str = "within",
                 n_min: int = lb.N_MIN, batch: int = 32, mask_seed: int = lb.DEFAULT_SEED,
                 occupancy: str = "natural"):
        self.cohort, self.geometry, self.seed = cohort, geometry, int(seed)
        self.scaling, self.split, self.n_min, self.batch = scaling, split, int(n_min), int(batch)
        self.mask_seed, self.occupancy = int(mask_seed), occupancy
        self.rows = cohort.run_rows()
        self._stores = {}

    def _store(self, index: int) -> lb.SparseHitStore:
        if index not in self._stores:
            self._stores[index] = self.cohort.run_store(index)
        return self._stores[index]

    def batch_at(self, step):
        rng = lb.keyed_rng(self.seed, "lucid/batches", int(step))
        chosen = np.sort(rng.choice(len(self.rows), min(8, len(self.rows)), replace=False))
        per = max(1, self.batch // chosen.size)
        batches, rows = [], []
        for index in chosen:
            row = self.rows[int(index)]
            take = np.sort(int(row["event_id_start"]) + rng.permutation(int(row["n_events"]))[:per])
            masker = ViewMasker(self.geometry, self.split, seed=self.mask_seed,
                                occupancy=self.occupancy, n_min=self.n_min)
            try:
                b = tokenise_sparse(self._store(int(index)), self.geometry, event_ids=take,
                                    scaling=self.scaling, masker=masker)
            except RuntimeError:
                continue
            batches.append(b)
            rows.append(np.arange(b.event_ids.size))
        return lr._stack(batches, rows), None
