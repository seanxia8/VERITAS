import numpy as np

from jepa_noise.p2a_ceiling_repair import fit_block_ridge


def test_infinite_side_penalty_recovers_base_fit():
    rng = np.random.default_rng(2117)
    z = rng.normal(size=(300, 6))
    u = rng.normal(size=(300, 105))
    y = z[:, :3] + rng.normal(scale=0.03, size=(300, 3))
    groups = np.repeat(np.arange(30), 10)
    fit = fit_block_ridge(z, u, y, groups, gammas=(np.inf,))
    bare = fit_block_ridge(z, u[:, :0], y, groups, gammas=(np.inf,))
    np.testing.assert_allclose(fit.predict(z, u), bare.predict(z, u[:, :0]), atol=1e-12)
    assert fit.alpha == bare.alpha
    np.testing.assert_array_equal(fit.coef[6:], 0.0)


def test_split_probe_uses_informative_side_block():
    rng = np.random.default_rng(31)
    z = rng.normal(size=(600, 8))
    u = rng.normal(size=(600, 105))
    y = z[:, :3] + u[:, :3] + rng.normal(scale=0.02, size=(600, 3))
    groups = np.repeat(np.arange(30), 16)
    fit = fit_block_ridge(z[:480], u[:480], y[:480], groups)
    prediction = fit.predict(z[480:], u[480:])
    assert np.isfinite(fit.gamma)
    assert np.linalg.norm(prediction-y[480:], axis=1).mean() < 0.05
