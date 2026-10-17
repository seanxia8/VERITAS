"""Exact Gaussian model; waveform laws v4-A4 (default) and v3-A1 (reproduction).

Runbook v4 §3.2 (A4) changes the Tier-1 base noise to channel-private
(``corr_strength = 0``), keeps the temporal PSD and the A1 line/common laws and
normalisations, recomputes the §4.1 white addition from that base, and uses the
E2-matched θ perturbation SD for E1b.  The v3 law stays selectable for
reproduction; the registered grids are module constants and mirrored in
``configs/waveform_v2_a4.json``.
"""
import numpy as np
from .linear import invsqrt

# --- Registered law parameters (runbook v4 §3.2 A4; v3 §5 WP3) ---------------
V3_LAW = 'v3-A1-waveform-v1'
A4_LAW = 'v4-A4-waveform-v1'

V3_BASE_RHO = .4   # reference-cell corr_strength inherited by A1
A4_BASE_RHO = 0.   # A4: channel-private base noise, temporal PSD unchanged
BASE_RHOS = (A4_BASE_RHO, V3_BASE_RHO)

# E1b θ perturbation SDs.  A4's are the SDs of E2's uniform ranges:
# (3, .1, .04)/sqrt(12); the v3 values stay selectable.  The registered values
# are also written at config precision (7 decimals), so membership is checked
# at 1e-7.
V3_THETA_SD = np.array([.15, .005, .002])
A4_THETA_SD = np.array([3., .1, .04])/np.sqrt(12.)
THETA_SDS = (A4_THETA_SD, V3_THETA_SD)
THETA_SD_TOL = 1e-7

# Named registered laws and their pinned (base_rho, θ SD).  ``LAW`` is the
# default law; ``SUPPORTED_LAWS`` is the machine-readable law record.
SUPPORTED_LAWS = (A4_LAW, V3_LAW)
LAW = A4_LAW
_LAW_PARAMS = {A4_LAW: (A4_BASE_RHO, A4_THETA_SD),
               V3_LAW: (V3_BASE_RHO, V3_THETA_SD)}

# E1b common-mode / line grids.  A4 adds 0.675-free G and keeps every v3 value.
V3_COMMON_STRENGTHS = (0., .225, .45, .675, .9)
A4_COMMON_STRENGTHS = (0., .005, .01, .02, .05, .1, .225, .45, .9)
A4_LINE_MULTIPLIERS = (.75, 1.5, 2.25, 3.)   # only at g = 0
LINE_MULTIPLIERS = (0.,) + A4_LINE_MULTIPLIERS
COMMON_STRENGTHS = tuple(sorted(set(A4_COMMON_STRENGTHS) | set(V3_COMMON_STRENGTHS)))

# A4.3: channel is primary for E1b/E2; interleaved is M's alternative; the
# time/block sample splits are diagnostic only.  A4.4 fixes the 13 unique cells
# (9 common-mode cells at line 0, 4 line cells at g = 0); (0, 0) is the clean
# reference cell.
PRIMARY_SPLIT = 'channel'
SPLIT_ROLES = {'channel': 'primary', 'interleaved': 'alternative',
               'time': 'diagnostic', 'block': 'diagnostic'}
A4_CANDIDATE_ORDER = ('channel', 'interleaved', 'time', 'block')
A4_CELL_GRID = (tuple((float(g), 0.) for g in A4_COMMON_STRENGTHS)
                + tuple((0., float(line)) for line in A4_LINE_MULTIPLIERS))


def _is_theta_sd(values, registered):
    """Registered θ SD comparison at the config's 7-decimal precision."""
    values = np.asarray(values, dtype=float)
    return (values.shape == (3,)
            and np.allclose(values, registered, rtol=0., atol=THETA_SD_TOL))


def validate_a4_cells(cells):
    """Return the registered A4 13-cell grid, rejecting any other cell set.

    The grid is the two one-factor sweeps of runbook v4 §3.2.4, not their
    Cartesian product: nine common-mode strengths at line 0 plus four line
    multipliers at g = 0, with (0, 0) the clean reference cell.
    """
    grid = tuple((float(g), float(line)) for g, line in cells)
    if grid != A4_CELL_GRID or len(set(grid)) != 13:
        raise ValueError('unregistered A4 cell grid (13 one-factor cells required)')
    return grid


class ExactModel:
    """Toy-source loading construction with explicit sharing convention.

    For share='both', one instance and sequential sample(gamma=0),
    sample(gamma=1) preserve Appendix A RNG order, including calibration draws.
    All factors are independent Gaussians. Dimensions are per view.
    """
    def __init__(self, d=40, k=3, seed=0, theta_scale=.4, eta_scale=3., share='target'):
        if share not in ('target','both'): raise ValueError('invalid sharing convention')
        if d<k or k<1 or theta_scale<=0 or eta_scale<0: raise ValueError('invalid model dimensions/scales')
        self.d,self.k,self.share=d,k,share
        self.theta_scale,self.eta_scale=theta_scale,eta_scale
        self.rng=np.random.default_rng(seed)
        self.ac=self.rng.normal(size=(d,k)); self.at=self.rng.normal(size=(d,k))
        l=self.rng.normal(size=(d,d))/np.sqrt(d)
        self.private=l@l.T+.3*np.eye(d); self.chol=np.linalg.cholesky(self.private)
        self.bc=self.rng.normal(size=(d,1)); self.bt=self.rng.normal(size=(d,1))

    def population(self,gamma_load):
        if not np.isfinite(gamma_load) or gamma_load<0: raise ValueError('invalid gamma_load')
        bc=self.bc*(gamma_load if self.share=='both' else 1.)
        bt=self.bt*gamma_load
        ncc=self.private+self.eta_scale**2*bc@bc.T
        ntt=self.private+self.eta_scale**2*bt@bt.T
        nct=self.eta_scale**2*bc@bt.T
        scc=ncc+self.theta_scale**2*self.ac@self.ac.T
        stt=ntt+self.theta_scale**2*self.at@self.at.T
        sct=nct+self.theta_scale**2*self.ac@self.at.T
        gamma_n=float(np.linalg.svd(invsqrt(ncc)@nct@invsqrt(ntt),compute_uv=False)[0])
        return dict(ac=self.ac,at=self.at,bc=bc,bt=bt,ncc=ncc,ntt=ntt,nct=nct,
                    scc=scc,stt=stt,sct=sct,reference=np.linalg.solve(ncc,self.ac),
                    gamma_n=gamma_n,gamma_load=gamma_load)

    def sample(self,n,gamma_load=0.):
        if n<2: raise ValueError('at least two samples required')
        p=self.population(gamma_load); rng=self.rng
        theta=rng.normal(size=(n,self.k))*self.theta_scale
        eta=rng.normal(size=(n,1))*self.eta_scale
        nc=rng.normal(size=(n,self.d))@self.chol.T
        nt=rng.normal(size=(n,self.d))@self.chol.T
        noise_c=eta@p['bc'].T+nc; noise_t=eta@p['bt'].T+nt
        xc=theta@self.ac.T+noise_c; xt=theta@self.at.T+noise_t
        # Keep multiplication ordering identical to the disclosed toy.
        scale=gamma_load if self.share=='both' else 1.
        cal_c=scale*rng.normal(size=(n,1))*self.eta_scale@self.bc.T+rng.normal(size=(n,self.d))@self.chol.T
        return dict(context=xc,target=xt,noise_context=noise_c,noise_target=noise_t,
                    calibration_noise_context=cal_c,theta=theta,eta=eta,
                    population=p,rng_state=rng.bit_generator.state)


def temporal_covariance(psd, fs, n):
    """Exact finite-grid physical covariance, retaining zero DC (no floor)."""
    from latent_monitor.whitening import psd_to_bin_variance
    power = psd_to_bin_variance(psd, fs, n)
    lag = np.fft.irfft(power, n=n) / n
    indices = np.arange(n)
    return lag[(indices[:, None]-indices[None, :]) % n], power / n


def leading_temporal_basis(eigenvalues, count=2):
    """Real circulant eigenspaces; project coordinate axes, including exact ties.

    Interior cosine/sine modes share the same supplied eigenvalue exactly.
    This avoids tolerance-based rotations of merely nearby eigenvalues.
    """
    from .linear import _basis
    n = 2*(len(eigenvalues)-1)
    t = np.arange(n)
    groups = {}
    for k, value in enumerate(eigenvalues):
        if k in (0, n//2):
            columns = [np.cos(2*np.pi*k*t/n)/np.sqrt(n)]
        else:
            columns = [np.sqrt(2/n)*np.cos(2*np.pi*k*t/n),
                       np.sqrt(2/n)*np.sin(2*np.pi*k*t/n)]
        groups.setdefault(float(value), []).extend(columns)
    basis, spectrum, multiplicities = [], [], []
    for value in sorted(groups, reverse=True):
        block = np.array(groups[value]).T
        vectors = _basis(block@block.T, block.shape[1])
        multiplicities.append(len(vectors))
        for vector in vectors:
            spectrum.append(value)
            if len(basis) < count:
                basis.append(vector*np.sqrt(n))
    return np.array(basis), np.array(spectrum), multiplicities


class Tier1Views:
    """Registered Tier-1 waveform noise (default v4-A4) with fixed-J0/nonlinear signals.

    Components and covariance are additive. Flattening is channel-major.
    Constructing this object derives source constants only; population() and
    sampling must obey the independent pre-outcome configuration review gate.

    ``base_rho`` selects the base-noise channel structure: 0 (default) is the
    A4 channel-private law, ``V3_BASE_RHO`` reproduces the v3/A1 reference cell.
    ``theta_sd`` selects the E1b θ perturbation SD; by default it is paired with
    the law (E2-matched for A4, (.15,.005,.002) for v3).  ``law`` selects one of
    the named registered laws (``SUPPORTED_LAWS`` / ``LAW``) and pins its base
    and θ fixture; the law tag is recorded in the manifest.  ``channel_covariance``
    keeps its v3 meaning: the reference-cell channel factor H_ref that fixes the
    A1 white/line/common normalisation.  The law's own base factor is
    ``base_channel_covariance``; ``raw_noise_covariance`` and ``noise_covariance``
    are built from it.
    """
    def __init__(self, signal_mode='linearized', split=PRIMARY_SPLIT, line_multiplier=0.,
                 common_strength=.45, seed=0, cell='development', model_seed=0,
                 amplitude_scale=1., base_rho=A4_BASE_RHO, theta_sd=None, law=None):
        from latent_monitor.tier1 import reference_cell, geometry_gain
        from noise_module import NoiseGenerator
        from .masks import raw_split
        from .crlb import central_jacobian
        if signal_mode not in ('linearized', 'nonlinear'):
            raise ValueError('unknown signal mode')
        if line_multiplier not in LINE_MULTIPLIERS:
            raise ValueError('unregistered line multiplier')
        if common_strength not in COMMON_STRENGTHS:
            raise ValueError('unregistered common strength')
        if amplitude_scale not in (1., .25):
            raise ValueError('unregistered SNR amplitude scale')
        if signal_mode == 'linearized' and amplitude_scale != 1.:
            raise ValueError('E1b has no amplitude-scale sweep')
        if law is not None:
            if law not in _LAW_PARAMS:
                raise ValueError(f'unregistered law {law!r}')
            law_rho, law_theta = _LAW_PARAMS[law]
            # A named law pins its base and θ fixture.  base_rho is 0 by
            # default (the A4 value), so a named law may move it only when it
            # was left at that default; contradicting explicit values fail.
            if base_rho == A4_BASE_RHO and law_rho != base_rho:
                base_rho = law_rho
            elif base_rho != law_rho:
                raise ValueError('named law and base_rho disagree')
            if theta_sd is None:
                theta_sd = law_theta
            elif not _is_theta_sd(theta_sd, law_theta):
                raise ValueError('named law and theta_sd disagree')
        if base_rho not in BASE_RHOS:
            raise ValueError('unregistered base rho (registered: 0 v4-A4, .4 v3-A1)')
        self.base_rho = float(base_rho)
        if theta_sd is None:
            theta_sd = A4_THETA_SD if self.base_rho == A4_BASE_RHO else V3_THETA_SD
        if not _is_theta_sd(theta_sd, A4_THETA_SD) and not _is_theta_sd(theta_sd, V3_THETA_SD):
            raise ValueError('unregistered theta perturbation SD')
        theta_sd = np.asarray(theta_sd, dtype=float)
        if not np.isfinite(theta_sd).all() or np.any(theta_sd <= 0):
            raise ValueError('invalid theta perturbation SD')
        self.theta_sd = theta_sd
        self.theta_fixture = ('v4-a4-e1b' if _is_theta_sd(theta_sd, A4_THETA_SD)
                              else 'v3-a1')
        # Law tag follows the registered base-noise law; the θ fixture is
        # recorded separately in the manifest (see class docstring).
        self.law = law if law is not None else (
            A4_LAW if self.base_rho == A4_BASE_RHO else V3_LAW)
        self.signal_mode, self.split_name = signal_mode, split
        self.seed, self.cell, self.model_seed = seed, cell, model_seed
        self.line_multiplier, self.common_strength = line_multiplier, common_strength
        self.amplitude_scale = amplitude_scale
        self.split = raw_split(split)
        if split not in SPLIT_ROLES:
            raise ValueError('unknown registered split')
        self.split_role = SPLIT_ROLES[split]
        self.reference_cell = reference_cell()
        self.gains = geometry_gain(self.reference_cell.geometry)
        self.family = self.reference_cell.family
        self.theta0 = np.array([4.5, .35, .08])
        self.ranges = np.array([3., .1, .04])
        self.fs, self.n, self.channels = 10000., 256, 8
        # Match Cell.event's structure initialization exactly (N=8, seed=11).
        generator = self.reference_cell.noise.generator(8, 11)
        _, structure = generator.generate(8, return_metadata=True)
        # Reference-cell channel factor: unchanged A1 normalisation.
        rho = self.reference_cell.noise.corr_strength
        reference_shared = np.sqrt(rho)*structure['gains']
        reference_private = np.sqrt(1-rho)*structure['private_strengths']
        self.reference_channel_covariance = (np.outer(reference_shared, reference_shared)
                                             + np.diag(reference_private**2))
        self.channel_covariance = self.reference_channel_covariance
        # A4 law: the same channel structure at base_rho (0 = channel-private).
        self.shared_weight = np.sqrt(self.base_rho)*structure['gains']
        self.private_weight = np.sqrt(1-self.base_rho)*structure['private_strengths']
        self.base_channel_covariance = (np.outer(self.shared_weight, self.shared_weight)
                                        + np.diag(self.private_weight**2))
        _, self.psd, psd_meta = NoiseGenerator(self.reference_cell.noise.base).build_psd_density(256, return_metadata=True)
        floor = next(x for x in psd_meta['component_contributions'] if x['name'] == 'floor')
        self.white_temporal_variance = floor['integrated_power_after_global_scaling']
        white_psd = np.zeros_like(self.psd)
        white_psd[1:] = self.white_temporal_variance / ((len(white_psd)-1)*self.fs/self.n)
        self.white_psd = white_psd
        self.temporal_covariance, self.temporal_eigenvalues = temporal_covariance(self.psd, self.fs, self.n)
        _, self.white_eigenvalues = temporal_covariance(white_psd, self.fs, self.n)
        residual = self.temporal_eigenvalues-self.white_eigenvalues
        if residual.min() < -1e-12:
            raise ValueError('reference white component exceeds total spectrum')
        self.colored_eigenvalues = np.maximum(residual, 0.)
        # A1 normalisation unchanged: reference-cell mean channel variance.
        self.v_white = float(np.trace(self.reference_channel_covariance)/8*self.white_temporal_variance)
        # §4.1 white addition recomputed from the A4 base (runbook v4 §3.2.1).
        self.epsilon_dc = float(.01*np.trace(self.base_channel_covariance)/8*np.trace(self.temporal_covariance)/256)
        self.common_basis, self.temporal_spectrum, self.temporal_multiplicities = leading_temporal_basis(self.temporal_eigenvalues)
        waveform = self.waveform(self.theta0)[0]
        powers = np.abs(np.fft.rfft(waveform))**2
        self.line_bin = int(np.argmax(powers[1:])+1)
        phase = 2*np.pi*self.line_bin*np.arange(256)/256
        self.line_basis = np.array([np.cos(phase), np.sin(phase)])
        self.j0 = central_jacobian(lambda p: self.waveform(p).ravel(), self.theta0, self.ranges)
        self.raw_noise_covariance = np.kron(self.base_channel_covariance, self.temporal_covariance)
        self.line_covariance = np.kron(np.ones((8,8)), self.line_multiplier*self.v_white*self.line_basis.T@self.line_basis)
        self.common_covariance = np.kron(np.ones((8,8)), self.common_strength*self.v_white/2*self.common_basis.T@self.common_basis)
        self.noise_covariance = (self.raw_noise_covariance+self.epsilon_dc*np.eye(2048)
                                 +self.line_covariance+self.common_covariance)

    def waveform(self, theta):
        from noise_module import pulse_template_2
        a, onset, tau = np.asarray(theta)
        f = self.family
        pulse = pulse_template_2(np.arange(256)/256, onset, f.An_frac*a, a,
                                 f.tau_n, f.tau_in, tau)
        return self.gains[:, None]*pulse[None, :]

    def jacobian(self, theta, half=False):
        from .crlb import central_jacobian
        ranges = self.ranges.copy()
        ranges[0] *= self.amplitude_scale
        return central_jacobian(lambda p: self.waveform(p).ravel(), theta, ranges, half=half)

    def _rng(self, role, event_id, component):
        from .masks import namespace_rng
        return namespace_rng(self.seed, f'{role}/{int(event_id)}/{component}', self.cell, self.model_seed)

    def noise_record(self, event_id, role='noise_cal', return_components=False):
        components = {}
        for label, eigen in (('base_white', self.white_eigenvalues), ('base_colored', self.colored_eigenvalues)):
            rng = self._rng(role, event_id, label)
            records = np.fft.irfft(np.fft.rfft(rng.normal(size=(9, 256)), axis=-1)*np.sqrt(eigen), n=256, axis=-1)
            components[label+'_shared'] = self.shared_weight[:, None]*records[0]
            components[label+'_private'] = self.private_weight[:, None]*records[1:]
        components['epsilon_private'] = self._rng(role, event_id, 'epsilon').normal(size=(8,256))*np.sqrt(self.epsilon_dc)
        line = self._rng(role, event_id, 'line').normal(size=2)*np.sqrt(self.line_multiplier*self.v_white)
        common = self._rng(role, event_id, 'common').normal(size=2)*np.sqrt(self.common_strength*self.v_white/2)
        components['line'] = np.broadcast_to(line@self.line_basis, (8,256)).copy()
        components['common'] = np.broadcast_to(common@self.common_basis, (8,256)).copy()
        noise = sum(components.values())
        targets = {'line_I': line[0], 'line_Q': line[1], 'common_0': common[0], 'common_1': common[1]}
        return (noise, targets, components) if return_components else (noise, targets)

    def sample(self, event_ids, role='subject_train', return_components=False):
        ids = np.asarray(event_ids, dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids < 0):
            raise ValueError('unique nonnegative event IDs required')
        raw, theta, latent, components = [], [], [], []
        for event_id in ids:
            rng = self._rng(role, event_id, 'physics')
            if self.signal_mode == 'linearized':
                physical = rng.normal(size=3)*self.theta_sd
                signal = (self.j0@physical).reshape(8,256)
            else:
                physical = rng.uniform([3*self.amplitude_scale,.30,.06], [6*self.amplitude_scale,.40,.10])
                signal = self.waveform(physical)
            noise, target, parts = self.noise_record(event_id, role=role+'/event_noise', return_components=True)
            raw.append(signal+noise); theta.append(physical); latent.append(list(target.values()))
            if return_components: components.append(parts)
        raw = np.asarray(raw).reshape(len(ids),8,256)
        flat = raw.reshape(len(ids),2048)
        result = {'raw':raw, 'context':flat[:,self.split.context], 'target':flat[:,self.split.target],
                  'theta':np.asarray(theta), 'nuisance':np.asarray(latent),
                  'nuisance_names':['line_I','line_Q','common_0','common_1'],
                  'nuisance_valid':[self.line_multiplier>0]*2+[self.common_strength>0]*2,
                  'event_ids':ids, 'role':role, 'cell':self.cell, 'split_hash':self.split.digest}
        if return_components: result['components'] = components
        return result

    def population(self):
        """Exact linearized populations; nonlinear signal covariance is not invented."""
        c, t = self.split.context, self.split.target
        ncc = self.noise_covariance[np.ix_(c,c)]
        ntt = self.noise_covariance[np.ix_(t,t)]
        nct = self.noise_covariance[np.ix_(c,t)]
        gamma_n = float(np.linalg.svd(invsqrt(ncc)@nct@invsqrt(ntt), compute_uv=False)[0])
        out = dict(ncc=ncc, ntt=ntt, nct=nct, gamma_n=gamma_n,
                   context_indices=c.copy(), target_indices=t.copy(), unused_indices=self.split.unused.copy())
        if self.signal_mode == 'linearized':
            ac, at = self.j0[c], self.j0[t]
            sigma = np.diag(self.theta_sd**2)
            out.update(ac=ac, at=at, scc=ncc+ac@sigma@ac.T,
                       stt=ntt+at@sigma@at.T, sct=nct+ac@sigma@at.T,
                       reference=np.linalg.solve(ncc, ac), sigma_theta=sigma)
        return out

    def displacement(self, k=3):
        """A4.5 population displacement D of this fixture/split, in degrees."""
        return displacement(self.population(), k=k)

    def manifest(self):
        import hashlib
        digest = lambda a: hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
        return {'law':self.law, 'base_rho':self.base_rho,
                'theta_sd':self.theta_sd.tolist(), 'theta_fixture':self.theta_fixture,
                'signal_mode':self.signal_mode, 'split':self.split_name,
                'split_role':self.split_role, 'primary_split':PRIMARY_SPLIT,
                'split_hash':self.split.digest, 'line_multiplier':self.line_multiplier,
                'common_strength':self.common_strength, 'line_bin':self.line_bin,
                'line_bandwidth_hz':0., 'v_white':self.v_white, 'epsilon_dc':self.epsilon_dc,
                'common_basis':self.common_basis.tolist(), 'common_basis_sha256':digest(self.common_basis),
                'temporal_top_eigenvalues':self.temporal_spectrum[:4].tolist(),
                'temporal_top_gap':float(self.temporal_spectrum[1]-self.temporal_spectrum[2]),
                'temporal_multiplicities':self.temporal_multiplicities,
                'noise_covariance_sha256':digest(self.noise_covariance), 'j0_sha256':digest(self.j0),
                'channel_covariance':self.channel_covariance.tolist(),
                'base_channel_covariance':self.base_channel_covariance.tolist(),
                'base_channel_covariance_sha256':digest(self.base_channel_covariance),
                'seed':self.seed, 'cell':self.cell, 'model_seed':self.model_seed,
                'amplitude_scale':self.amplitude_scale,
                'theta_semantics':'delta_theta' if self.signal_mode=='linearized' else 'amplitude,t0,tau_t'}


def displacement(population, k=3):
    """A4.5 population displacement D (degrees) from a ``Tier1Views.population()``.

    ``D`` is the maximum principal angle between the top-k context canonical
    directions of the population CCA (linearised fixture, A4 covariances, given
    split) and the Paper 1 reference ``R_c``: the top-k generalised eigenvectors
    of ``(A_c Sigma_theta A_c^T, N_cc)``.  The covariance code is not duplicated;
    this only reuses the population blocks.  ``gamma_n`` stays a separate,
    reported quantity of that mapping.

    Returns ``{'k', 'angle_deg', 'angles_deg', 'canonical_correlations'}`` where
    ``angles_deg`` holds the maximum principal angle for every rank 1..k (the
    §A k = 1 / 2 / 3 diagnostics) and ``canonical_correlations`` the top-k
    population canonical correlations of the context/target pair.
    """
    if not isinstance(k, int) or not 1 <= k <= 3:
        raise ValueError('displacement rank must be 1, 2, or 3')
    try:
        scc = np.asarray(population['scc'], dtype=float)
        stt = np.asarray(population['stt'], dtype=float)
        sct = np.asarray(population['sct'], dtype=float)
        ncc = np.asarray(population['ncc'], dtype=float)
        ac = np.asarray(population['ac'], dtype=float)
        sigma = np.asarray(population['sigma_theta'], dtype=float)
    except KeyError as error:
        raise ValueError('displacement requires a linearised population mapping') from error
    from scipy.linalg import eigh
    from .metrics import principal_angles
    hc, ht = invsqrt(scc), invsqrt(stt)
    whitened = hc@sct@ht
    left, canonical, _ = np.linalg.svd(whitened, full_matrices=False)
    context = hc@left[:, :k]
    # Paper 1 reference R_c: top-k GEVP of (A_c Sigma_theta A_c^T, N_cc).
    eigenvalues, vectors = eigh(ac@sigma@ac.T, ncc)
    reference = vectors[:, np.argsort(eigenvalues)[::-1][:k]]
    angles = {}
    for rank in range(1, k+1):
        angles[rank] = float(np.max(principal_angles(
            context[:, :rank].T, reference[:, :rank].T, degrees=True)))
    return {'k':k, 'angle_deg':angles[k], 'angles_deg':angles,
            'canonical_correlations':canonical[:k].tolist()}
