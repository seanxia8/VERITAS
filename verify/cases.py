"""K1 case table, parsed from the metric-free INPUTS.txt (runbook §V0.4)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_INPUTS = os.path.join(HERE, "..", "results", "round2", "20261121_verify", "INPUTS.txt")
INPUTS = os.environ.get("R2V_INPUTS", _REPO_INPUTS if os.path.exists(_REPO_INPUTS) else os.path.join(HERE, "INPUTS.txt"))
ARMS = ("init", "mae_mse", "sigreg_jepa", "target_posfree")
WANT = {("init", 9000), ("init", 9001), ("mae_mse", 9000), ("mae_mse", 9001),
        ("sigreg_jepa", 9000), ("target_posfree", 9000), ("target_posfree", 9001)}
# encoder (window, shift) and readout convention from the runs' own provenance (V0)
REPAIRED = {"wcte__jepa_repaired__s9000": (64, 32, "layernorm"),
            "wcte__mae_repaired__s9000": (64, 32, "raw")}


def cases():
    out, section = [], None
    for line in open(INPUTS):
        if line.startswith("# cmd1"):
            section = 1; continue
        if line.startswith("# cmd2"):
            section = 2; continue
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        if section == 1:
            det, arm, seed, step, ckpt, sha = f[0], f[1], int(f[2]), f[3], f[4], f[5]
            if (arm, seed) in WANT:
                out.append(dict(case=f"{det}__{arm}__s{seed}", det=det, arm=arm, seed=seed,
                                ckpt=None if ckpt == "None" else ckpt, sha256=None if sha == "None" else sha,
                                window=64, shift=32, readout="raw"))
        elif section == 2 and f[0] in REPAIRED:
            w, s, ro = REPAIRED[f[0]]
            det, arm, seed = f[0].split("__")
            out.append(dict(case=f[0], det=det, arm=arm, seed=int(seed[1:]), ckpt=f[1], sha256=f[2],
                            window=w, shift=s, readout=ro))
    return out


if __name__ == "__main__":
    for c in cases():
        print(c["case"], c["ckpt"], c["window"], c["readout"])
    print(len(cases()))
