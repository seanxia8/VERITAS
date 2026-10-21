"""The open-decision register: schema, non-breaking defaults, and the claim gate."""
import json
from pathlib import Path

import pytest

from noise_module_lucid import decision_registry as dr

REPO = Path(dr.__file__).resolve().parents[2]


def test_register_loads_and_validates():
    register = dr.load()
    assert register["schema"] == dr.SCHEMA
    assert len(dr.decisions(register)) == len({e["id"] for e in dr.decisions(register)})


def test_every_decision_names_an_owner_status_default_and_switch():
    for entry in dr.decisions():
        assert entry["owner"] in dr.OWNERS
        assert entry["status"] in dr.STATUSES
        assert entry["default"]
        assert entry["switch"]
        assert entry["config_refs"], f"{entry['id']}: no place to apply the switch"


def test_config_refs_exist():
    for entry in dr.decisions():
        for ref in entry["config_refs"]:
            assert (REPO / ref).exists(), f"{entry['id']}: missing config ref {ref}"


def test_pending_decisions_are_named_owners_not_the_author():
    for entry in dr.decisions():
        if entry["status"] == "pending":
            assert entry["owner"] in ("Junjie", "author"), entry["id"]


def test_defaults_preserve_current_behaviour():
    assert dr.default_of("D-E")["energy_endpoint"] == "deposited_energy_in_volume"
    assert dr.default_of("D-QESPREAD")["per_pmt_qe_sigma"] == 0.0
    assert dr.default_of("D-CH")["per_channel_walk_slope_sigma"] == 0.0
    assert dr.default_of("D-CR")["grouping"] == "mpmt_modules"
    assert dr.default_of("D-TQ")["form"] == "inverse_sqrt"
    assert dr.default_of("D-WP7")["launch"] is False


def test_claim_gate_blocks_the_expected_decisions():
    blockers = {entry["id"] for entry in dr.claim_blockers()}
    assert {"D-E", "D-TQ", "D-QE", "D-WP7"} <= blockers
    with pytest.raises(dr.DecisionGateError) as exc:
        dr.assert_claim_ready()
    assert "D-E" in str(exc.value)


def test_claim_gate_clears_only_when_all_claim_gate_entries_are_confirmed(tmp_path):
    register = json.loads(json.dumps(dr.load()))
    for entry in register["decisions"]:
        if entry["claim_gate"]:
            entry["status"] = "confirmed"
    path = tmp_path / "register.json"
    path.write_text(json.dumps(register))
    dr.assert_claim_ready(dr.load(path))


def test_unknown_decision_is_an_error():
    with pytest.raises(KeyError):
        dr.get("D-NOT-A-DECISION")


def test_provenance_records_the_register_hash():
    provenance = dr.provenance()
    assert len(provenance["sha256"]) == 64
    assert provenance["schema"] == dr.SCHEMA
