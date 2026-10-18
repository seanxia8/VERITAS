"""R-4 step 1: generation-only, range-based muon containment in WCTE (author option 1, 18 Oct 2026).

Run from the repository root: PYTHONPATH=src python -B results/pmt_r4_dev_2026-10-18/containment/containment_check.py
"""
from pathlib import Path
import json
import numpy as np
from noise_module_lucid import containment as c

R, H = 1.5771308488246254, 2.77697099384   # pinned WCTE cylinder via PromptTransport (Deepthought, 18 Oct)
PDG = {100.0: 31.24, 200.0: 78.94, 300.0: 129.1, 400.0: 179.6}
rows = [c.contained_fraction(t, R, H) for t in range(100, 401, 50)]
orig = c.csda_range_m
sens = {}
for s in (0.95, 1.05):
    c.csda_range_m = lambda t, s=s: s*orig(t)
    sens[f"range_x{s:g}"] = {int(t): c.contained_fraction(t, R, H)["contained_fraction"] for t in range(100, 401, 50)}
c.csda_range_m = orig
pilot = {int(t): c.contained_fraction(t, R, H)["contained_fraction"] for t in (500, 1000, 1500)}
out = dict(
    question="does any mu- kinetic energy in 100-400 MeV reach >= 50% containment in WCTE?",
    answer=[r["kinetic_mev"] for r in rows if r["contained_fraction"] >= 0.5],
    cylinder=dict(radius_m=R, height_m=H, source="pinned WCTE geometry, upstream eaf2e89, PromptTransport"),
    sampling="vertex uniform in [-0.35, 0.35]^3 m; uniform SO(3) rotation of +z (isotropic); 200000 tracks per energy, seed 20261018",
    containment="straight track of CSDA length; end point strictly inside x^2+y^2<R^2, |z|<H/2",
    range_check=[dict(kinetic_mev=t, pdg_csda_g_cm2=r, this_g_cm2=orig(t)*100, relative=orig(t)*100/r - 1) for t, r in PDG.items()],
    pdg_source="PDG Atomic and Nuclear Properties, Muon energy loss in water (liquid), https://pdg.lbl.gov/2024/AtomicNuclearProperties/MUE/muE_water_liquid.txt (sha256 2cebc6a422ad9f8fa90b0e06a2e3c6e9ae68c2ecbe49d0717f85514b698599c2)",
    rows=rows, range_sensitivity=sens,
    pilot_energies_straight_track=pilot, pilot_geant4_contained="500 MeV: 3/160 placements (56179eb); 1000, 1500 MeV: 0/160",
    caveat="Multiple scattering and range straggling are ignored. Scattering shortens the net displacement, so the straight-track contained fraction is approximately a lower bound; straggling (a few % of the range) moves it either way (see range_sensitivity). Not a Geant4 result.",
    evidence_stage="development", non_citable=True)
Path(__file__).with_name("containment.json").write_text(json.dumps(out, indent=2) + "\n")
for r in rows: print(int(r["kinetic_mev"]), f"{r['csda_range_m']:.3f} m", f"{r['contained_fraction']:.4f} +- {r['binomial_se']:.4f}")
print("energies >= 50%:", out["answer"])
