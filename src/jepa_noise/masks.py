"""Registered E1b/E2 raw-index masks; no data or labels enter selection here."""
from dataclasses import dataclass
import hashlib
import numpy as np

@dataclass(frozen=True)
class Split:
    context: np.ndarray
    target: np.ndarray
    unused: np.ndarray

    @property
    def digest(self):
        return hashlib.sha256(b''.join(x.astype('<i8').tobytes() for x in (self.context,self.target,self.unused))).hexdigest()


def raw_split(name):
    indices=np.arange(8*256).reshape(8,256)
    if name=='time': c,t=indices[:,:96],indices[:,160:]
    elif name=='channel': c,t=indices[:4],indices[4:]
    elif name=='interleaved': c,t=indices[::2],indices[1::2]
    elif name=='block': c,t=indices[:4,:96],indices[4:,160:]
    else: raise ValueError('unknown registered split')
    c,t=c.ravel(),t.ravel()
    return Split(c,t,np.setdiff1d(indices.ravel(),np.r_[c,t]))


def namespace_rng(seed,role,cell='',model_seed=0):
    """Stable independent namespaces, identical across paired methods."""
    import json
    key=json.dumps([int(seed),str(role),str(cell),int(model_seed)],separators=(',',':')).encode()
    entropy=np.frombuffer(hashlib.sha256(key).digest(),dtype='<u4').tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _validated_retained(split):
    """Return context-then-target indices; validate the complete raw partition."""
    parts = [np.asarray(x) for x in (split.context, split.target, split.unused)]
    if any(x.ndim != 1 or x.dtype.kind not in 'iu' for x in parts):
        raise ValueError('raw indices must be one-dimensional integer arrays')
    all_indices = np.concatenate(parts)
    if (not len(parts[0]) or not len(parts[1]) or len(all_indices) != 2048
            or not np.array_equal(np.sort(all_indices), np.arange(2048))):
        raise ValueError('split must partition exactly 8x256 raw coordinates')
    return np.concatenate(parts[:2])


def retained_oracle_transform(noise_covariance, split):
    """Inverse lower Cholesky of the retained *marginal*, never conditioned gap.

    The full raw covariance supplies coordinate provenance only. Gap/unused
    rows and columns are discarded before any factorization or whitening.
    """
    from scipy.linalg import eigvalsh, solve_triangular
    retained = _validated_retained(split)
    covariance = np.asarray(noise_covariance, dtype=float)
    if covariance.shape != (2048, 2048):
        raise ValueError('full raw 8x256 covariance required for marginal indexing')
    marginal = covariance[np.ix_(retained, retained)]
    if not np.isfinite(marginal).all() or not np.allclose(marginal, marginal.T, rtol=1e-12, atol=1e-12):
        raise ValueError('retained covariance must be finite symmetric')
    if eigvalsh(marginal, subset_by_index=[0, 0])[0] <= 0:
        raise ValueError('retained joint covariance is not positive definite; no jitter')
    lower = np.linalg.cholesky(marginal)
    transform = solve_triangular(lower, np.eye(len(retained)), lower=True)
    return transform, retained.copy()


def oracle_preprocess(raw, split, transform, retained_indices):
    """Transform retained (context,target), then zero-unused embed as 8x256.

    Context cannot depend on raw target/gap. Target innovations may depend on
    raw context. `retained_indices` must equal the exact context-then-target
    ordering used to construct the transform; reordered covariance is rejected.
    """
    retained = _validated_retained(split)
    indices = np.asarray(retained_indices)
    if indices.dtype.kind not in 'iu' or not np.array_equal(indices, retained):
        raise ValueError('transform marginal index order does not match split')
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (len(retained), len(retained)) or not np.isfinite(matrix).all():
        raise ValueError('finite retained joint transform with exact shape required')
    if np.any(np.triu(matrix, 1) != 0) or np.any(np.diag(matrix) <= 0):
        raise ValueError('inverse lower Cholesky required; symmetric/full-record whitening prohibited')
    raw = np.asarray(raw, dtype=float)
    if raw.shape[-2:] != (8, 256) or not np.isfinite(raw).all():
        raise ValueError('finite raw (...,8,256) measurements required')
    flat = raw.reshape(*raw.shape[:-2], 2048)
    transformed = flat[..., retained] @ matrix.T
    embedded = np.zeros_like(flat)
    embedded[..., retained] = transformed
    return embedded.reshape(raw.shape)
