# SPDX-License-Identifier: MIT
"""Round-2 loss-share logger: test 10."""
from __future__ import annotations

import json
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from jepa_noise import ext_r2 as x


def test_shares_sum_to_one_and_match_terms():
    rng = np.random.default_rng(0)
    lines = []
    for step in range(1, 301):
        c, h, t = rng.uniform(0.1, 1.0, 3)
        lines.append({"step": step, "metrics": {"charge": c, "hit": h, "time": t,
                                                "loss": c + h + t}})
    out = x.loss_shares(lines)
    for window, rows in (("start", lines[:100]), ("end", lines[-100:])):
        s = out[window]["shares"]
        assert abs(sum(s.values()) - 1.0) < 1e-12
        means = {k: np.mean([r["metrics"][k] for r in rows]) for k in x.LOSS_TERMS}
        for k in x.LOSS_TERMS:
            assert abs(s[k] - means[k] / sum(means.values())) < 1e-12
        assert out[window]["n_steps"] == 100
    assert out["start"]["steps"] == [1, 100] and out["end"]["steps"] == [201, 300]


def test_shares_from_a_real_training_log(tmp_path):
    torch = pytest.importorskip("torch")
    from jepa_noise import lucid_bombs as lb
    from jepa_noise import lucid_run as lr
    from jepa_noise.tests.test_r2_arms import _batch
    from jepa_noise.tests.r2_fixtures import small_cylinder
    g = small_cylinder()
    batch = _batch(g)

    class Loader:
        def batch_at(self, step):
            return batch, None
    model = x.build_r2_arm("mae_mse", g, 9000)
    opts, _info, sched = lr.build_optimizers(model, x.run_config())
    log = tmp_path / "train_log.jsonl"
    lr.train_loop(arm=model, loader=Loader(), steps=4, optimizers=opts,
                  checkpoint_path=tmp_path / "ck.pt",
                  identity={"run_id": "t", "cell_id": "t", "arm": "mae_mse", "seed": 9000,
                            "beta": None, "steps_scheduled": 4, "gradient_clip": None},
                  config_sha="t", checkpoint_minutes=0, log_path=log, accum_steps=4,
                  log_every=1, scheduler=sched)
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert [l["step"] for l in lines] == [1, 2, 3, 4]
    for l in lines:
        m = l["metrics"]
        assert abs(m["charge"] + m["hit"] + m["time"] - m["loss"]) < 1e-5
    out = x.loss_shares(lines, first=2, last=2)
    assert abs(sum(out["start"]["shares"].values()) - 1.0) < 1e-12
