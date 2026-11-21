"""K3: independent Epps-Pulley SIGReg statistic (LeJEPA, Balestriero & LeCun, arXiv 2511.08544).

For embeddings X (N, D) and M unit directions a_m: x_nm = <X_n, a_m>.  For each slice
    EP_m = N * sum_k w_k |phi_N(t_k) - phi(t_k)|^2,   phi_N(t) = (1/N) sum_n exp(i t x_nm),
phi(t) = exp(-t^2/2) the N(0,1) characteristic function, t_k a uniform grid on [0, t_max]
and w_k = trapezoid weights (doubled: the integrand is even in t, so the integral over the
real line is twice the half-line one) times the Gaussian window exp(-t^2/2).
SIGReg = mean_m EP_m.  Expected value for a sample with characteristic function psi:
    E[EP] = sum_k w_k [ N |psi - phi|^2 + (1 - |psi|^2) ]   (linear in N).

Usage: python k3_sigreg.py OUT.json [official_src_dir]
"""
import json
import sys

import numpy as np
import torch

from common import DET, Cohort, Geometry, REPO, build_features, load_encoder, sha256

T_MAX, N_T, M_DIR = 3.0, 17, 1024
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def grid():
    t = torch.linspace(0, T_MAX, N_T)
    dt = T_MAX / (N_T - 1)
    w = torch.full((N_T,), 2 * dt)
    w[0] = w[-1] = dt
    phi = torch.exp(-0.5 * t ** 2)
    return t, phi, w * phi


T, PHI, W = grid()


def directions(d, m=M_DIR, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, m, generator=g)
    return a / a.norm(dim=0, keepdim=True)


def ep_stat(x, a):
    """SIGReg value of embeddings x (N, D) on directions a (D, M)."""
    p = x @ a                                  # (N, M)
    arg = p[..., None] * T                     # (N, M, K)
    c, s = torch.cos(arg).mean(0), torch.sin(arg).mean(0)
    err = (c - PHI) ** 2 + s ** 2              # (M, K)
    return x.shape[0] * (err @ W).mean()


def expected_ep(n, psi):
    psi = torch.as_tensor(psi)
    return float((W * (n * (psi - PHI) ** 2 + 1 - psi ** 2)).sum())


def scaling_block(x, a, ns=(4, 8, 32, 256)):
    return {int(n): float(ep_stat(x[:n], a)) for n in ns if n <= x.shape[0]}


def accumulation_block(x, a):
    """8 disjoint micro-batches of 4 vs the union of 32: values and embedding-gradient cosine."""
    x = x[:32].clone().requires_grad_(True)
    acc = sum(ep_stat(x[4 * b:4 * b + 4], a) for b in range(8)) / 8
    g_acc, = torch.autograd.grad(acc, x)
    full = ep_stat(x, a)
    g_full, = torch.autograd.grad(full, x)
    cos = torch.nn.functional.cosine_similarity(g_acc.flatten(), g_full.flatten(), dim=0)
    return {"mean_8x4": float(acc), "union_1x32": float(full), "ratio_8x4_over_1x32": float(acc / full),
            "grad_cosine": float(cos), "grad_norm_ratio_8x4_over_1x32": float(g_acc.norm() / g_full.norm())}


def context_embeddings(n_events=256, seed=9000):
    """SK sigreg_jepa s9000: per-event mean of context-pass tokens over context PMTs, using the
    repository's training view masker (split 'within', mask seed = run seed) as a black box."""
    sys.path.insert(0, f"{REPO}/src")
    from jepa_noise.lucid_views import ViewMasker
    ck = "/ceph/dwong/jepa/returns/r2_multidetector_2026-11-11/runs/sk__sigreg_jepa__s9000/checkpoint.pt"
    assert sha256(ck) == "9797b2a65d70f388dd8ffd7bb7fab2954f8a0215364645ad04143701ed3fa54c"
    torch.set_default_dtype(torch.float32)
    model, rgeo = load_encoder("sk", ck, None)
    torch.set_default_dtype(torch.float64)
    geo = Geometry("sk")
    coh = Cohort.__new__(Cohort)
    import h5py
    with h5py.File(DET["sk"]["h5"], "r") as f:
        g = f["dev_train"]
        off = g["offsets"][:n_events + 1]
        coh.offsets, coh.idx = off, np.arange(n_events)
        coh.pmt_id, coh.charge, coh.time = (g[k][:off[-1]] for k in ("pmt_id", "charge_pe", "time_ns"))
        ev_ids = g["event_ids"][:n_events]
    masker = ViewMasker(rgeo, "within", seed=seed)
    embs = []
    for k in range(n_events):
        x = torch.from_numpy(build_features(geo, coh, [k])).float()
        m = masker.mask(int(ev_ids[k]), x[0, :, 8].numpy() > 0.5)
        hidden = torch.zeros(1, geo.n, dtype=torch.bool)
        hidden[0, torch.as_tensor(m.hidden)] = True
        with torch.inference_mode():
            z = model.encode(x, hidden_mask=hidden)[0]
        embs.append(z[torch.as_tensor(m.context)].double().mean(0))
    return torch.stack(embs)


def main():
    out = {"definition": {"t_max": T_MAX, "n_t": N_T, "n_directions": M_DIR, "weights": "trapezoid x2 x exp(-t^2/2)",
                          "scaling": "N * mean over slices"}}
    d = 128
    a = directions(d)
    rng = torch.Generator().manual_seed(1)
    # synthetic: (i) N(0, I) -> psi = phi; (ii) N(0, 4 I) -> every slice N(0, 4), psi = exp(-2 t^2)
    syn = {}
    for name, scale in (("gauss_unit", 1.0), ("gauss_var4", 2.0)):
        psi = torch.exp(-0.5 * (scale * T) ** 2)
        reps = {n: [] for n in (4, 8, 32, 256)}
        for r in range(64):
            x = scale * torch.randn(256, d, generator=rng)
            for n in reps:
                reps[n].append(float(ep_stat(x[:n], a)))
        syn[name] = {int(n): {"mean_over_64": float(np.mean(v)), "sd": float(np.std(v)), "expected": expected_ep(n, psi)}
                     for n, v in reps.items()}
        x = scale * torch.randn(256, d, generator=rng)
        syn[name]["single_draw_scaling"] = scaling_block(x, a)
        syn[name]["accumulation"] = accumulation_block(x, a)
    out["synthetic"] = syn
    # real
    emb = context_embeddings()
    out["real_sk_sigreg_s9000"] = {"n_events": int(emb.shape[0]), "embedding": "mean of context tokens (within view)",
                                   "scaling": scaling_block(emb, a), "accumulation": accumulation_block(emb, a),
                                   "emb_mean_norm": float(emb.mean(0).norm()), "emb_mean_var": float(emb.var(0).mean())}
    # official implementation, if available
    if len(sys.argv) > 2:
        sys.path.insert(0, sys.argv[2])
        try:
            out["official"] = official_compare(emb, a, syn)
        except Exception as e:  # recorded, not fatal
            out["official"] = {"error": repr(e)}
    # invariants
    s = out["synthetic"]["gauss_var4"]
    lin = {n: s[n]["mean_over_64"] / s[4]["mean_over_64"] for n in (8, 32, 256)}
    r = out["real_sk_sigreg_s9000"]["scaling"]
    out["invariants"] = {
        "I9": {"synthetic_ratio_to_N4": lin, "ideal_N_over_4": {8: 2, 32: 8, 256: 64},
               "synthetic_vs_expected_rel": {n: s[n]["mean_over_64"] / s[n]["expected"] - 1 for n in (4, 8, 32, 256)},
               "real_ratio_to_N4": {n: r[n] / r[4] for n in r},
               "operationalisation": "E[EP_N] = A N + B exactly; pass if the synthetic means match A N + B within 10% at every N and the real statistic is within 10% of proportional (ratio to N=4 vs N/4)",
               "pass": bool(all(abs(s[n]["mean_over_64"] / s[n]["expected"] - 1) < 0.1 for n in (4, 8, 32, 256))
                            and all(abs((r[n] / r[4]) / (int(n) / 4) - 1) < 0.1 for n in r))},
        "I10": {"real": out["real_sk_sigreg_s9000"]["accumulation"], "synthetic_var4": s["accumulation"],
                "pass": bool(abs(out["real_sk_sigreg_s9000"]["accumulation"]["ratio_8x4_over_1x32"] - 1) > 0.25
                             and out["real_sk_sigreg_s9000"]["accumulation"]["grad_cosine"] < 0.9)},
    }
    if "official" in out and "max_rel_diff" in out["official"]:
        out["invariants"]["I11"] = {"max_rel_diff": out["official"]["max_rel_diff"], "pass": out["official"]["max_rel_diff"] <= 1e-4}
    else:
        out["invariants"]["I11"] = {"pass": None, "note": "official implementation unavailable"}
    print(json.dumps(out["invariants"], indent=1, default=str))
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)


def official_compare(emb, a, syn):
    """Run the official lejepa Epps-Pulley univariate test on the same projections."""
    from lejepa.univariate import EppsPulley
    test = EppsPulley(t_max=T_MAX, n_points=N_T).double()
    rows = []
    rng = torch.Generator().manual_seed(7)
    for name, x in (("real", emb), ("gauss_var4", 2 * torch.randn(256, 128, generator=rng))):
        for n in (4, 8, 32, 256):
            mine = float(ep_stat(x[:n], a))
            theirs = float(test(x[:n] @ a).mean())
            rows.append({"input": name, "N": n, "mine": mine, "official": theirs, "rel": abs(mine - theirs) / abs(theirs)})
    return {"rows": rows, "max_rel_diff": max(r["rel"] for r in rows)}


if __name__ == "__main__":
    main()
