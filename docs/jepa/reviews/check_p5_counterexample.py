"""Population counterexample to unconditional P5; review-only, no training.
Run: .venv/bin/python docs/jepa/reviews/check_p5_counterexample.py
"""
import numpy as np

# Independent theta, eta and private noises, each variance 1.
# Context: [theta + eta + e_c1, e_c2].
# Target: [theta/4 + eta/2 + e_t1, e_t2].
a = np.array([[1.0], [0.0], [0.25], [0.0]])
nc = np.diag([2.0, 1.0])
nt = np.diag([1.25, 1.0])
nct = np.diag([0.5, 0.0])
noise = np.block([[nc, nct], [nct.T, nt]])
whitener = np.linalg.inv(np.linalg.cholesky(noise))
a_white = whitener @ a
cov_white = whitener @ (a @ a.T + noise) @ whitener.T
print('noise eigenvalues:', np.linalg.eigvalsh(noise))
print('original target physics rank:', np.linalg.matrix_rank(a[2:]))
print('whitened target physics loading:', a_white[2:].ravel())
print('whitened cross-view covariance norm:', np.linalg.norm(cov_white[:2, 2:]))
print('context noise-weighted physics direction:', np.linalg.solve(nc, a[:2]).ravel())
assert np.linalg.eigvalsh(noise).min() > 0
assert np.linalg.matrix_rank(a[2:]) == 1
assert np.linalg.norm(a_white[2:]) < 1e-14
assert np.linalg.norm(cov_white[:2, 2:]) < 1e-14
print('P5 requires an additional non-annihilation/rank condition after innovation.')
