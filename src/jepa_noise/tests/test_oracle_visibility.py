"""Raw dependency and retained-marginal oracle checks; no waveform outcomes."""
import numpy as np
import pytest
from jepa_noise.masks import Split, raw_split, retained_oracle_transform, oracle_preprocess


def fixture():
    c, t = np.array([0, 1]), np.array([256, 257])
    split = Split(c, t, np.setdiff1d(np.arange(2048), np.r_[c, t]))
    cov = np.eye(2048)
    retained = np.r_[c, t]
    lower = np.array([[1., 0, 0, 0], [.2, 1., 0, 0], [.4, .1, 1., 0], [0, .3, .2, 1.]])
    cov[np.ix_(retained, retained)] = lower @ lower.T
    return split, cov


def test_exact_retained_marginal_not_conditioned_on_gap():
    split, cov = fixture()
    original, indices = retained_oracle_transform(cov, split)
    # Change correlations to a discarded gap coordinate; retained marginal is
    # identical. Conditioning on the gap would change the transform.
    cov[0, 20] = cov[20, 0] = .2
    changed, new_indices = retained_oracle_transform(cov, split)
    np.testing.assert_array_equal(original, changed)
    np.testing.assert_array_equal(indices, new_indices)
    marginal = cov[np.ix_(indices, indices)]
    np.testing.assert_allclose(changed @ marginal @ changed.T, np.eye(4), atol=1e-12)


def test_exact_jacobian_and_finite_perturbation_visibility():
    split, cov = fixture(); transform, indices = retained_oracle_transform(cov, split)
    # Function is linear: evaluating coordinate basis computes its exact
    # Jacobian, independently of the internal indexing implementation.
    jacobian = oracle_preprocess(np.eye(2048).reshape(2048, 8, 256), split, transform, indices).reshape(2048, 2048).T
    np.testing.assert_array_equal(jacobian[np.ix_(split.context, np.r_[split.target, split.unused])], 0)
    np.testing.assert_array_equal(jacobian[:, split.unused], 0)
    np.testing.assert_array_equal(jacobian[split.unused, :], 0)
    assert np.linalg.norm(jacobian[np.ix_(split.target, split.context)]) > 0
    raw = np.random.default_rng(3).normal(size=(2, 8, 256))
    before = oracle_preprocess(raw, split, transform, indices).reshape(2, 2048)
    perturb = raw.copy().reshape(2, 2048)
    perturb[:, np.r_[split.target, split.unused]] += 100.
    after = oracle_preprocess(perturb.reshape(raw.shape), split, transform, indices).reshape(2, 2048)
    np.testing.assert_array_equal(before[:, split.context], after[:, split.context])
    np.testing.assert_array_equal(after[:, split.unused], 0)
    perturb = raw.copy().reshape(2, 2048); perturb[:, split.context] += 1.
    after_context = oracle_preprocess(perturb.reshape(raw.shape), split, transform, indices).reshape(2, 2048)
    assert not np.array_equal(before[:, split.target], after_context[:, split.target])


@pytest.mark.parametrize('name', ['time', 'channel', 'interleaved', 'block'])
def test_registered_positions_reembedded_with_zero_unused(name):
    split = raw_split(name); indices = np.r_[split.context, split.target]
    raw = np.arange(2048., dtype=float).reshape(8, 256)
    output = oracle_preprocess(raw, split, np.eye(len(indices)), indices).ravel()
    np.testing.assert_array_equal(output[indices], raw.ravel()[indices])
    np.testing.assert_array_equal(output[split.unused], 0)


def test_reject_wrong_order_shape_symmetric_or_invalid_partition():
    split, cov = fixture(); matrix, indices = retained_oracle_transform(cov, split)
    raw = np.zeros((8, 256))
    with pytest.raises(ValueError, match='index order'):
        oracle_preprocess(raw, split, matrix, indices[::-1])
    with pytest.raises(ValueError, match='exact shape'):
        oracle_preprocess(raw, split, np.eye(2048), indices)
    with pytest.raises(ValueError, match='inverse lower'):
        oracle_preprocess(raw, split, matrix + matrix.T, indices)
    broken = Split(split.context, split.context, split.unused)
    with pytest.raises(ValueError, match='partition'):
        retained_oracle_transform(cov, broken)


def test_joint_singularity_rejected_before_factorization():
    split, cov = fixture(); retained = np.r_[split.context, split.target]
    cov[np.ix_(retained, retained)] = np.ones((4, 4))
    with pytest.raises(ValueError, match='not positive definite'):
        retained_oracle_transform(cov, split)
