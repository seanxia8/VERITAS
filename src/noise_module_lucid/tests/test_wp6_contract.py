"""WP6 frozen contract v1: required cells present, D-E-dependent entries pending, modules partitioned."""
import json
from pathlib import Path
import numpy as np

PATH = Path(__file__).resolve().parents[3] / "configs/lucid_pmt/wp6_contract_v1.json"


def contract():
    return json.loads(PATH.read_text())


def test_frozen_with_required_cells():
    c = contract()
    assert c["status"] == "frozen_v1"
    ids = {x["id"] for x in c["cells"]}
    # runbook v2 §5: QE shape, T-Q slope, width, coupled mPMT gain + width, offset and module-loss
    # controls, identity and exact-degeneracy controls.
    assert {"qe_shape", "tq_slope", "timing_width", "group_coupled", "tq_offset_control", "module_loss",
            "identity", "exact_qe_ce_product"} <= ids
    n_cells = [x for x in c["cells"] if x["role"] == "N"]
    assert all(x.get("severity") for x in n_cells)
    assert c["held_out"]["family"] in {x["id"] for x in n_cells}


def test_de_dependent_entries_are_pending():
    c = contract()
    assert c["consequence"]["K_energy"]["status"] == "pending_D-E" and c["consequence"]["K"]["status"] == "pending_D-E"
    assert c["consequence"]["K_vertex"]["status"] == "frozen" and c["event_support"]["status"] == "pending_D-E"
    assert c["consequence"]["kappa_m"]["status"] == "provisional_dev"


def test_module_sets_partition_the_105_modules():
    idx = np.arange(105)
    permitted, held = idx[idx % 3 != 2], idx[idx % 3 == 2]
    assert len(permitted) == 70 and len(held) == 35 and not set(permitted) & set(held)
    assert "mod 3 == 2" in contract()["modules"]["held_out"]
