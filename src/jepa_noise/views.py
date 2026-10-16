"""Exact Gaussian model; waveform law remains gated pending author resolution."""
import numpy as np
from .linear import invsqrt


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
    """Approved v3/A1 Gaussian noise with fixed-J0 or nonlinear pulse signals.

    Components and covariance are additive. Flattening is channel-major.
    Constructing this object derives source constants only; population() and
    sampling must obey the independent pre-outcome configuration review gate.
    """
    def __init__(self, signal_mode='linearized', split='time', line_multiplier=0.,
                 common_strength=.45, seed=0, cell='development', model_seed=0,
                 amplitude_scale=1.):
        from latent_monitor.tier1 import reference_cell, geometry_gain
        from noise_module import NoiseGenerator
        from .masks import raw_split
        from .crlb import central_jacobian
        if signal_mode not in ('linearized', 'nonlinear'):
            raise ValueError('unknown signal mode')
        if line_multiplier not in (0., .75, 1.5, 2.25, 3.):
            raise ValueError('unregistered line multiplier')
        if common_strength not in (0., .225, .45, .675, .9):
            raise ValueError('unregistered common strength')
        if amplitude_scale not in (1., .25):
            raise ValueError('unregistered SNR amplitude scale')
        if signal_mode == 'linearized' and amplitude_scale != 1.:
            raise ValueError('E1b has no amplitude-scale sweep')
        self.signal_mode, self.split_name = signal_mode, split
        self.seed, self.cell, self.model_seed = seed, cell, model_seed
        self.line_multiplier, self.common_strength = line_multiplier, common_strength
        self.amplitude_scale = amplitude_scale
        self.split = raw_split(split)
        self.reference_cell = reference_cell()
        self.gains = geometry_gain(self.reference_cell.geometry)
        self.family = self.reference_cell.family
        self.theta0 = np.array([4.5, .35, .08])
        self.ranges = np.array([3., .1, .04])
        self.theta_sd = np.array([.15, .005, .002])
        self.fs, self.n, self.channels = 10000., 256, 8
        # Match Cell.event's structure initialization exactly (N=8, seed=11).
        generator = self.reference_cell.noise.generator(8, 11)
        _, structure = generator.generate(8, return_metadata=True)
        rho = self.reference_cell.noise.corr_strength
        self.shared_weight = np.sqrt(rho)*structure['gains']
        self.private_weight = np.sqrt(1-rho)*structure['private_strengths']
        self.channel_covariance = (np.outer(self.shared_weight, self.shared_weight)
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
        self.v_white = float(np.trace(self.channel_covariance)/8*self.white_temporal_variance)
        self.epsilon_dc = float(.01*np.trace(self.channel_covariance)/8*np.trace(self.temporal_covariance)/256)
        self.common_basis, self.temporal_spectrum, self.temporal_multiplicities = leading_temporal_basis(self.temporal_eigenvalues)
        waveform = self.waveform(self.theta0)[0]
        powers = np.abs(np.fft.rfft(waveform))**2
        self.line_bin = int(np.argmax(powers[1:])+1)
        phase = 2*np.pi*self.line_bin*np.arange(256)/256
        self.line_basis = np.array([np.cos(phase), np.sin(phase)])
        self.j0 = central_jacobian(lambda p: self.waveform(p).ravel(), self.theta0, self.ranges)
        self.raw_noise_covariance = np.kron(self.channel_covariance, self.temporal_covariance)
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

    def manifest(self):
        import hashlib
        digest = lambda a: hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()
        return {'law':'v3-A1-waveform-v1', 'signal_mode':self.signal_mode, 'split':self.split_name,
                'split_hash':self.split.digest, 'line_multiplier':self.line_multiplier,
                'common_strength':self.common_strength, 'line_bin':self.line_bin,
                'line_bandwidth_hz':0., 'v_white':self.v_white, 'epsilon_dc':self.epsilon_dc,
                'common_basis':self.common_basis.tolist(), 'common_basis_sha256':digest(self.common_basis),
                'temporal_top_eigenvalues':self.temporal_spectrum[:4].tolist(),
                'temporal_top_gap':float(self.temporal_spectrum[1]-self.temporal_spectrum[2]),
                'temporal_multiplicities':self.temporal_multiplicities,
                'noise_covariance_sha256':digest(self.noise_covariance), 'j0_sha256':digest(self.j0),
                'channel_covariance':self.channel_covariance.tolist(),
                'seed':self.seed, 'cell':self.cell, 'model_seed':self.model_seed,
                'amplitude_scale':self.amplitude_scale,
                'theta_semantics':'delta_theta' if self.signal_mode=='linearized' else 'amplitude,t0,tau_t'}
