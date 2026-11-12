# SPDX-License-Identifier: MIT
"""Round-2 store: test 6 (pack -> read round trip, tokens, sampling protocol)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from jepa_noise import lucid_bombs as lb
from jepa_noise import lucid_run as lr
from jepa_noise import point_encoder as pe
from jepa_noise import r2_generator as gen
from jepa_noise import r2_store as rs
from jepa_noise.lucid_views import ViewMasker
from jepa_noise.tests.r2_fixtures import small_cylinder


def _runs(name, start, n_runs, n_events):
    return tuple(lb.RunSpec(f"lucid/{name}/{i:04d}", f"lucid/{name}", "F1", 0.05,
                            start + i * n_events, n_events, lb.DEFAULT_SEED) for i in range(n_runs))


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    root = tmp_path_factory.mktemp("r2store")
    geometry = small_cylinder()
    cfg = gen.load_r2_config(4000.0, 40000.0)
    rule = gen.TimeRule(4, 20.0, (-20.0, 60.0), 100.0)
    cohorts = {"dev_train": _runs("dev_train", 1_000_000, 4, 15),
               "dev_probe_train": _runs("dev_probe_train", 1_200_000, 2, 9)}
    for cohort, runs in cohorts.items():
        twin = cohort != "dev_train"
        for run in runs:
            r = gen.generate_run(geometry, cfg, run, rule, with_twin=twin, store_expected_pmt=twin)
            gen.write_generated_run(root / "shards" / cohort / run.run_id.replace("/", "__"), r,
                                    geometry, cfg, rule)
    h5 = root / "det_r2.h5"
    summary = rs.pack_detector(h5, root / "shards", cohorts, attrs={"detector": "synthetic"})
    return SimpleNamespace(root=root, geometry=geometry, cohorts=cohorts, h5=h5, summary=summary)


def test_pack_verify_and_round_trip(packed):
    out = rs.verify_pack(packed.h5, packed.root / "shards", packed.cohorts)
    assert out["verified_runs"] == {"dev_train": 4, "dev_probe_train": 2}
    assert packed.summary["dev_probe_train"]["twin_present"]
    for cohort, runs in packed.cohorts.items():
        c = rs.open_cohort(packed.h5, cohort)
        assert c.run_ids == [r.run_id for r in runs]
        for i, run in enumerate(runs):
            d = packed.root / "shards" / cohort / run.run_id.replace("/", "__")
            ref, got = lb.read_sparse_hits(d / "hits.npz"), c.run_store(i)
            for name in ("event_ids", "offsets", "pmt_id", "charge_pe", "time_ns"):
                assert getattr(got, name).tobytes() == getattr(ref, name).tobytes(), name
            assert c.state_sha256[i] == run.state_sha256(packed.geometry)
    twin = rs.open_cohort(packed.h5, "dev_probe_train", twin=True)
    ref = lb.read_sparse_hits(packed.root / "shards" / "dev_probe_train"
                              / "lucid__dev_probe_train__0000" / "twin_hits.npz")
    assert twin.run_store(0).pmt_id.tobytes() == ref.pmt_id.tobytes()


def test_tamper_is_detected(packed, tmp_path):
    import shutil
    h5 = tmp_path / "copy.h5"
    shutil.copy(packed.h5, h5)
    with h5py.File(h5, "r+") as f:
        f["dev_train/charge_pe"][0] = f["dev_train/charge_pe"][0] + 1.0
    with pytest.raises(ValueError, match="bytes differ"):
        rs.verify_pack(h5, packed.root / "shards", packed.cohorts)


def test_hdf5_tokens_equal_per_run_tokens(packed):
    g = packed.geometry
    scaling = rs.scaling_for(g, 450.0)
    c = rs.open_cohort(packed.h5, "dev_probe_train")
    for i, run in enumerate(packed.cohorts["dev_probe_train"]):
        d = packed.root / "shards" / "dev_probe_train" / run.run_id.replace("/", "__")
        a = pe.tokenise(d, g, split="within", scaling=scaling,
                        masker=ViewMasker(g, "within", seed=0, n_min=0))
        b = rs.tokenise_sparse(c.run_store(i), g, scaling=scaling,
                               masker=ViewMasker(g, "within", seed=0, n_min=0))
        assert torch.equal(a.features, b.features) and torch.equal(a.hidden, b.hidden)
        assert torch.equal(a.target_index, b.target_index)
        assert np.array_equal(a.event_ids, b.event_ids) and a.keys == b.keys
    assert float(a.features[..., pe.TIME_INDEX].abs().max()) < 1.0  # scaled by 450 ns


def test_training_sampling_protocol_unchanged(packed, tmp_path):
    g = packed.geometry
    (tmp_path / "ds").mkdir()
    (tmp_path / "ds" / "runs").symlink_to(packed.root / "shards" / "dev_train")
    runs = packed.cohorts["dev_train"]
    dataset = SimpleNamespace(directory=tmp_path / "ds", seed=lb.DEFAULT_SEED, runs=tuple(
        {"run_id": r.run_id, "run_dir": r.run_id.replace("/", "__"), "event_id_start": r.event_start,
         "n_events": r.n_events} for r in runs))
    scaling = rs.scaling_for(g, 100.0)
    old = lr.TrainingLoader(dataset, g, split="within", seed=9000, n_min=3, batch=32,
                            scaling=scaling)
    new = rs.R2TrainingLoader(rs.open_cohort(packed.h5, "dev_train"), g, seed=9000,
                              scaling=scaling, n_min=3, batch=32)
    for step in (0, 1, 7, 9999):
        a, _ = old.batch_at(step)
        b, _ = new.batch_at(step)
        assert np.array_equal(a.event_ids, b.event_ids)
        assert torch.equal(a.features, b.features) and torch.equal(a.hidden, b.hidden)
        assert torch.equal(a.context_index, b.context_index) and a.keys == b.keys
