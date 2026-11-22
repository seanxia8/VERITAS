"""V0: collect, per checkpoint, the encoder architecture and token scaling from the run's
own provenance (run_record.json / recipe.json next to the checkpoint). Config keys only;
no metric fields are read or written."""
import json, os, sys, torch
KEEP = ("arm", "detector", "time_scale_ns", "encoder", "scaling", "seed", "steps", "code_sha")
def find_cfg(ckpt):
    d = os.path.dirname(ckpt); out = {}
    for name in ("run_record.json", "recipe.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            j = json.load(open(p)); j = {**j, **j.get("recipe", {})}
            out[name] = {k: j[k] for k in KEEP if k in j}
            out[name]["posfree_keys"] = {k: j[k] for k in j if "posfree" in k.lower() or "position" in k.lower()}
    return out
res = {}
for line in open(sys.argv[1]):
    sha, p = line.split()
    c = torch.load(p, map_location="cpu", weights_only=False)
    res[p] = {"identity": c.get("identity"), "step": c.get("step"), "sha256": sha,
              "model_keys_nonstd": sorted(k for k in c["model"] if not k.startswith(("jepa.", "target_encoder.", "head.", "hit_head.", "obs_head."))),
              "provenance": find_cfg(p)}
    pr = res[p]["provenance"]; e = {k: v for n in pr.values() for k, v in n.items()}
    print(res[p]["identity"]["run_id"], e.get("arm"), (e.get("encoder") or {}).get("window"), e.get("time_scale_ns") or (e.get("scaling") or {}).get("time_scale_ns"), e.get("posfree_keys"), res[p]["model_keys_nonstd"], flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1, default=str)
