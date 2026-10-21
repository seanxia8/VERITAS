"""Machine-readable register of open decisions and their development defaults.

One place records every decision that is still open (or only defaulted), the
default in force, who owns the switch, and what must change before a claim run.
A default is a *development* setting: it lets engineering continue, but the
claim gate refuses a Claim-1/Claim-2 comparison while any ``claim_gate`` entry
is not ``confirmed``.

The register itself is data (``configs/decisions/open_decisions_v1.json``); this
module only loads, validates and queries it. Changing a default is therefore a
reviewable data change plus a dated amendment, not a code edit.

CLI::

    PYTHONPATH=src .venv/bin/python -m noise_module_lucid.decision_registry
    PYTHONPATH=src .venv/bin/python -m noise_module_lucid.decision_registry --claim-gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA = "veritas-decisions-v1"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "decisions" / "open_decisions_v1.json"
STATUSES = ("pending", "provisional_default", "adopted", "confirmed", "retired")
OWNERS = ("Junjie", "author", "Junjie+author")
REQUIRED = ("id", "title", "question", "owner", "status", "claim_gate", "default",
            "alternatives", "blocks", "config_refs", "switch")


class DecisionGateError(RuntimeError):
    """A claim run was requested while a claim-gate decision is unresolved."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(register):
    """Validate the register and return it unchanged (raises ``ValueError``)."""
    if not isinstance(register, dict):
        raise ValueError("register must be a JSON object")
    if register.get("schema") != SCHEMA:
        raise ValueError(f"register schema {register.get('schema')!r} is not {SCHEMA}")
    _require(isinstance(register.get("created"), str), "register needs a created date")
    _require(isinstance(register.get("claim_gate_rule"), str) and register["claim_gate_rule"], "register needs the claim-gate rule")
    decisions = register.get("decisions")
    _require(isinstance(decisions, list) and decisions, "register needs a non-empty decisions list")
    seen = set()
    for entry in decisions:
        _require(isinstance(entry, dict), "each decision must be an object")
        missing = [k for k in REQUIRED if k not in entry]
        _require(not missing, f"decision {entry.get('id')!r} lacks {missing}")
        _require(isinstance(entry["id"], str) and entry["id"], f"invalid decision id {entry['id']!r}")
        _require(entry["id"] not in seen, f"duplicate decision id {entry['id']!r}")
        seen.add(entry["id"])
        _require(isinstance(entry["title"], str) and entry["title"], f"{entry['id']}: title required")
        _require(isinstance(entry["question"], str) and entry["question"], f"{entry['id']}: question required")
        _require(entry["owner"] in OWNERS, f"{entry['id']}: owner {entry['owner']!r} not in {OWNERS}")
        _require(entry["status"] in STATUSES, f"{entry['id']}: status {entry['status']!r} not in {STATUSES}")
        _require(isinstance(entry["claim_gate"], bool), f"{entry['id']}: claim_gate must be bool")
        _require(isinstance(entry["default"], dict) and entry["default"], f"{entry['id']}: default must be a non-empty object")
        _require(isinstance(entry["alternatives"], list), f"{entry['id']}: alternatives must be a list")
        _require(isinstance(entry["blocks"], list), f"{entry['id']}: blocks must be a list")
        _require(isinstance(entry["config_refs"], list), f"{entry['id']}: config_refs must be a list")
        _require(isinstance(entry["switch"], str) and entry["switch"], f"{entry['id']}: switch required")
    return register


def load(path=None):
    """Load and validate the register (default: ``configs/decisions/open_decisions_v1.json``)."""
    path = Path(path or DEFAULT_PATH)
    return validate(json.loads(path.read_text()))


def decisions(register=None):
    return list((register or load())["decisions"])


def by_id(register=None):
    return {entry["id"]: entry for entry in decisions(register)}


def get(decision_id, register=None):
    table = by_id(register)
    if decision_id not in table:
        raise KeyError(f"unknown decision {decision_id!r}; known: {sorted(table)}")
    return table[decision_id]


def default_of(decision_id, register=None):
    return dict(get(decision_id, register)["default"])


def claim_blockers(register=None):
    """Claim-gate entries that are not ``confirmed`` (an author waiver does not clear this)."""
    return [entry for entry in decisions(register) if entry["claim_gate"] and entry["status"] != "confirmed"]


def assert_claim_ready(register=None):
    """Raise ``DecisionGateError`` unless every claim-gate decision is confirmed."""
    blockers = claim_blockers(register)
    if blockers:
        detail = ", ".join(f"{e['id']}({e['status']}, owner={e['owner']})" for e in blockers)
        raise DecisionGateError(f"claim run blocked by unresolved decisions: {detail}")


def summary(register=None):
    return [{"id": e["id"], "status": e["status"], "owner": e["owner"], "claim_gate": e["claim_gate"],
             "title": e["title"]} for e in decisions(register)]


def provenance(path=None):
    path = Path(path or DEFAULT_PATH)
    load(path)  # fail before reporting provenance for an invalid register
    return {"path": str(path), "schema": SCHEMA, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", default=None, help="register path (default: configs/decisions/open_decisions_v1.json)")
    parser.add_argument("--claim-gate", action="store_true", help="exit nonzero if any claim-gate decision is unresolved")
    args = parser.parse_args(argv)
    register = load(args.path)
    if args.claim_gate:
        try:
            assert_claim_ready(register)
        except DecisionGateError as exc:
            print(exc)
            return 1
        print("claim gate clear")
        return 0
    for row in summary(register):
        print(f"{row['id']:<12} {row['status']:<20} owner={row['owner']:<12} claim_gate={row['claim_gate']!s:<5} {row['title']}")
    print(f"\n{len(summary(register))} decisions; {len(claim_blockers(register))} claim-gate blockers")
    print("provenance:", provenance(args.path))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
