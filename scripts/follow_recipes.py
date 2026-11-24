# SPDX-License-Identifier: MIT
"""r2-follow T0: rebuild a training recipe from its submission CLI and diff it against a reference run.

Run with ``PYTHONPATH=<code tree>/src`` so that the recipe is built by the code that will train
(the staged tarball, extracted). Nothing is trained and nothing is written except ``--out``.

    follow_recipes.py --cli "sk mae_repaired --seed 9002 --micro-batch 4 --patch-size 43 ..." \
        --ref <reference run dir> --out diff.json [--weights-prefix jepa.encoder.]

Reports: the rebuilt recipe and its hash; every field (dotted path) that differs from the
reference ``recipe.json``; a SHA-256 of the initial model state (seed-matched init check); and,
when ``--weights-prefix`` is given, whether the initial weights under that prefix equal the
reference run's ``snapshot_step_00000.pt`` tensor by tensor (same-seed references only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import torch


def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def state_sha(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", required=True, help="ext_r2_repair_train arguments exactly as submitted")
    ap.add_argument("--ref", type=Path, required=True, help="reference run directory (read-only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights-prefix", action="append", default=[],
                    help="compare initial weights under this state-dict prefix with the reference step 0")
    a = ap.parse_args()
    from jepa_noise import data_roots as dr
    from jepa_noise import ext_r2_repair_train as t
    captured = {}
    t.train = lambda args: captured.setdefault("args", args) and 0
    sys.argv = ["ext_r2_repair_train"] + shlex.split(a.cli)
    t.main()
    args = captured["args"]
    args.data_kind = dr.check(getattr(args, "data_kind", None))
    args.seed = int(getattr(args, "seed", t.SEED) or t.SEED)
    args.device = "cpu"
    run_id = t.run_id(args.detector, args.arm, args.seed) + t.run_suffix(args)
    p = t.prepare(args)
    recipe, state = p["recipe"], p["arm"].state_dict()
    ref = json.loads((a.ref / "recipe.json").read_text())
    # JSON round trip: tuples in the in-memory recipe are lists in recipe.json (same hash)
    new_f, ref_f = flatten(json.loads(json.dumps(recipe))), flatten(ref["recipe"])
    diff = {k: {"new": new_f.get(k, "<absent>"), "ref": ref_f.get(k, "<absent>")}
            for k in sorted(set(new_f) | set(ref_f)) if new_f.get(k, "<absent>") != ref_f.get(k, "<absent>")}
    out = {"cli": a.cli, "run_id": run_id, "code_tree": str(Path(t.__file__).resolve().parents[2]),
           "config_sha256": p["config_sha"], "ref": str(a.ref), "ref_config_sha256": ref["config_sha256"],
           "ref_code_sha": ref.get("code_sha"), "diff": diff, "recipe": recipe,
           "init_state_sha256": state_sha(state)}
    if a.weights_prefix:
        snap = torch.load(a.ref / "snapshot_step_00000.pt", map_location="cpu", weights_only=False)["model"]
        res = {}
        for prefix in a.weights_prefix:
            keys_new = {k for k in state if k.startswith(prefix)}
            keys_ref = {k for k in snap if k.startswith(prefix)}
            equal = sorted(k for k in keys_new & keys_ref if torch.equal(state[k].cpu(), snap[k].cpu()))
            res[prefix] = {"n_new": len(keys_new), "n_ref": len(keys_ref), "n_equal": len(equal),
                           "only_new": sorted(keys_new - keys_ref), "only_ref": sorted(keys_ref - keys_new),
                           "all_equal": keys_new == keys_ref == set(equal) and bool(equal)}
        out["weights_vs_ref_step0"] = res
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"run_id": run_id, "n_diff": len(diff), "diff_keys": list(diff),
                      "init": out["init_state_sha256"][:12],
                      "weights": {k: v["all_equal"] for k, v in out.get("weights_vs_ref_step0", {}).items()}}))


if __name__ == "__main__":
    main()
