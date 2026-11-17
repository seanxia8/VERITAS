import numpy as np

from jepa_noise.lucid_arms import _mean_tokens
from jepa_noise.r2_probe_repair import fit_ridge, moment_features, pool_tokens


def test_all_mean_matches_legacy():
    rng = np.random.default_rng(23)
    tokens = rng.normal(size=(3, 20, 6)).astype(np.float32)
    features = np.zeros((3, 20, 9), dtype=np.float32)
    active = rng.random((3, 20)) > 0.5
    for mask in (None, active):
        np.testing.assert_array_equal(pool_tokens(tokens, features, "all_mean", mask),
                                      _mean_tokens(tokens, mask))


def test_hit_mean_ignores_unhit_and_obeys_active_mask():
    tokens = np.arange(12.0).reshape(1, 4, 3)
    features = np.zeros((1, 4, 9))
    features[0, [1, 3], 8] = 1
    active = np.array([[True, True, True, False]])
    want = tokens[:, 1]
    np.testing.assert_array_equal(pool_tokens(tokens, features, "hit_mean", active), want)
    tokens[:, [0, 2, 3]] = 1e6
    np.testing.assert_array_equal(pool_tokens(tokens, features, "hit_mean", active), want)


def test_weighted_pooling_dark_and_empty_hit_fallbacks():
    tokens = np.arange(4.0).reshape(1, 4, 1)
    features = np.zeros((1, 4, 9))
    features[0, :, 6] = [1000, 0, np.log(2) / 3, 0]
    features[0, :, 7] = [0, -1, 0.5, 1]
    features[0, 1:3, 8] = 1
    np.testing.assert_allclose(pool_tokens(tokens, features, "charge_weighted"), [[5 / 3]])
    np.testing.assert_allclose(pool_tokens(tokens, features, "early_time_weighted"), [[2]])
    features[..., 8] = 0
    for mode in ("hit_mean", "charge_weighted", "early_time_weighted"):
        np.testing.assert_allclose(pool_tokens(tokens, features, mode), [[1.5]])


def test_spatial_moment_recovers_spherical_wavefront():
    rng = np.random.default_rng(20261117)
    direction = rng.normal(size=(100, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    pos = np.concatenate([direction, -direction])
    vertex = rng.uniform(-0.2, 0.2, size=(240, 3))
    arrival = np.linalg.norm(pos[None, :, :] - vertex[:, None, :], axis=-1)
    raw = np.zeros((len(vertex), len(pos), 9))
    raw[..., :3] = pos
    raw[..., 7] = arrival
    raw[..., 8] = 1
    tokens = arrival[..., None]
    latent, full = moment_features(tokens, raw)
    assert latent.shape == (len(vertex), 6)
    assert full.shape == (len(vertex), 21)
    groups = np.repeat(np.arange(16), 12)

    def error(features):
        prediction, _ = fit_ridge(features[:192], vertex[:192], groups, features[192:],
                                  form="test", target_names=("x", "y", "z"))
        return np.linalg.norm(prediction - vertex[192:], axis=1).mean()

    assert error(latent) * 5 < error(pool_tokens(tokens, raw, "all_mean"))


def test_checkpoint_selection_skips_incomplete_copy(tmp_path):
    import torch

    from jepa_noise.ext_r2_repair_probes import final_checkpoint

    identity = {"arm": "target_posfree", "seed": 9000}
    partial = tmp_path / "partial.pt"
    final = tmp_path / "final.pt"
    torch.save({"step": 3168, "identity": identity}, partial)
    torch.save({"step": 10000, "identity": identity}, final)
    selected, payload, digest = final_checkpoint([partial, final], "target_posfree", 9000)
    assert selected == final
    assert payload["step"] == 10000
    assert len(digest) == 64
