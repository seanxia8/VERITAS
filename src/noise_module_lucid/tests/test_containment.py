"""Range-based containment: CSDA range against the PDG muon table in liquid water.

Reference values: PDG Atomic and Nuclear Properties, "Muon energy loss in water (liquid)",
https://pdg.lbl.gov/2024/AtomicNuclearProperties/MUE/muE_water_liquid.txt (CSDA range column,
g/cm^2; density 1.000). Tolerance 0.5%, declared before the first comparison.
"""
import numpy as np
import pytest
from noise_module_lucid.containment import contained_fraction, csda_range_m, dedx_mev_cm2_g

PDG_CSDA_G_CM2 = {100.0: 31.24, 200.0: 78.94, 300.0: 129.1, 400.0: 179.6}
PDG_DEDX = {100.0: 2.256, 300.0: 1.981}  # total dE/dx column; radiative part < 0.1% here


@pytest.mark.parametrize("t,r", PDG_CSDA_G_CM2.items())
def test_csda_range_matches_pdg(t, r):
    assert csda_range_m(t)*100 == pytest.approx(r, rel=5e-3)


@pytest.mark.parametrize("t,s", PDG_DEDX.items())
def test_stopping_power_matches_pdg(t, s):
    assert dedx_mev_cm2_g(t) == pytest.approx(s, rel=5e-3)


def test_containment_limits_and_geometry():
    huge = contained_fraction(100, 10.0, 20.0, n=2000)
    assert huge["contained_fraction"] == 1.0
    tiny = contained_fraction(400, 0.5, 0.8, n=2000)
    assert tiny["contained_fraction"] == 0.0
    a, b = contained_fraction(300, 1.577, 2.777, n=5000), contained_fraction(300, 1.577, 2.777, n=5000)
    assert a == b
    assert contained_fraction(350, 1.577, 2.777, n=5000)["contained_fraction"] > contained_fraction(400, 1.577, 2.777, n=5000)["contained_fraction"]
    with pytest.raises(ValueError): csda_range_m(0.5)
