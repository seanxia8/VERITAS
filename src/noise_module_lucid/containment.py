"""Range-based muon containment in the WCTE water cylinder (D-E preflight, development).

The range is the CSDA range from a numerically integrated Bethe-Bloch mean ionisation loss
with the Sternheimer density correction, in water (PDG constants below). Radiative losses
are omitted (below 1 GeV they are < 0.5% of dE/dx in water). Tracks are straight lines of
length equal to the CSDA range. Multiple scattering and range straggling are ignored:
scattering makes the net displacement shorter than the path length, so a straight track
overstates how far the muon gets and the contained fraction here is approximately a lower
bound. Straggling (a few % of the range) moves it either way. Not a Geant4 result.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.spatial.transform import Rotation

M_MU_MEV = 105.6583755
M_E_MEV = 0.51099895
K_MEV_CM2_G = 0.307075
# Water (liquid), PDG Atomic and Nuclear Properties, muE_water_liquid table header.
WATER = dict(z_over_a=0.55509, density_g_cm3=1.0, i_ev=79.7, a=0.0912, k=3.4773, x0=0.2400, x1=2.8004, cbar=3.5017, delta0=0.0)
# PDG CSDA range at 1 MeV kinetic, the integration start (g/cm^2, same table).
R_1MEV_G_CM2 = 1.863e-3


def _delta(bg, w=WATER):
    x = np.log10(bg)
    if x >= w["x1"]:
        return 2*np.log(10)*x - w["cbar"]
    if x >= w["x0"]:
        return 2*np.log(10)*x - w["cbar"] + w["a"]*(w["x1"] - x)**w["k"]
    return w["delta0"]*10**(2*(x - w["x0"]))


def dedx_mev_cm2_g(t_mev, mass=M_MU_MEV, w=WATER):
    """Mean ionisation stopping power (Bethe-Bloch, density-corrected), MeV cm^2/g."""
    gamma = 1 + t_mev/mass
    beta2 = 1 - 1/gamma**2
    bg = np.sqrt(beta2)*gamma
    tmax = 2*M_E_MEV*bg**2 / (1 + 2*gamma*M_E_MEV/mass + (M_E_MEV/mass)**2)
    i_mev = w["i_ev"]*1e-6
    return K_MEV_CM2_G*w["z_over_a"]/beta2*(0.5*np.log(2*M_E_MEV*bg**2*tmax/i_mev**2) - beta2 - _delta(bg, w)/2)


def csda_range_m(t_mev, mass=M_MU_MEV, w=WATER):
    """CSDA range in metres for kinetic energy ``t_mev`` (>= 1 MeV)."""
    if not t_mev >= 1.0:
        raise ValueError("kinetic energy >= 1 MeV required")
    grams = R_1MEV_G_CM2 + quad(lambda t: 1/dedx_mev_cm2_g(t, mass, w), 1.0, t_mev, limit=200)[0]
    return grams / w["density_g_cm3"] / 100.0


def contained_fraction(t_mev, radius_m, height_m, *, n=200_000, seed=20261018, vertex_half_width_m=0.35):
    """Fraction of straight CSDA-range tracks whose end point lies strictly inside the cylinder.

    Sampling as the multi-energy pilot: vertex uniform in [-0.35, 0.35]^3 m and a uniform
    SO(3) rotation of the +z primary direction (an isotropic direction).
    """
    rng = np.random.default_rng(seed)
    vertex = rng.uniform(-vertex_half_width_m, vertex_half_width_m, (n, 3))
    direction = Rotation.random(n, random_state=rng).apply([0.0, 0.0, 1.0])
    end = vertex + csda_range_m(t_mev)*direction
    inside = (end[:, 0]**2 + end[:, 1]**2 < radius_m**2) & (np.abs(end[:, 2]) < height_m/2)
    f = float(inside.mean())
    return dict(kinetic_mev=float(t_mev), csda_range_m=csda_range_m(t_mev), n=n, contained_fraction=f,
                binomial_se=float(np.sqrt(f*(1-f)/n)))
