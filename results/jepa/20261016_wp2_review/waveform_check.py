"""Independent bounded waveform identities. No calibration or eligibility grid."""
import hashlib,json,time,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from jepa_noise.views import Tier1Views
from jepa_noise import crlb
from noise_module import NoiseGenerator
from latent_monitor.tier1 import reference_cell

out=Path(__file__).resolve().parent; start=time.perf_counter()
files=['src/jepa_noise/views.py','src/jepa_noise/configs/waveform_v1.json','src/jepa_noise/tests/test_waveform.py','docs/jepa/WAVEFORM_DERIVATION.md','src/noise_module/core/generator.py','src/noise_module/multichannel/generator.py','src/latent_monitor/tier1.py','src/latent_monitor/whitening.py','src/jepa_noise/crlb.py']
hashes=lambda:{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
before=hashes(); r={'start_utc':datetime.now(timezone.utc).isoformat(),'source_sha256_before':before}
assert before[files[0]]=='692ab80833bd5440f4426b358bd3996ad9259781e52fe5aaa701b76fac3aa277'
assert before[files[1]]=='efa2c03fc40a875c0a0d1410bfca10685f6d32f74def74dac55364e87136f712'
m=Tier1Views(line_multiplier=1.5,common_strength=.45,seed=456,cell='review_identity_only')
ref=reference_cell(); gen=ref.noise.generator(8,11); _,meta=gen.generate(8,return_metadata=True)
h=.4*np.outer(meta['gains'],meta['gains'])+.6*np.diag(meta['private_strengths']**2)
_,density,pmeta=NoiseGenerator(ref.noise.base).build_psd_density(256,return_metadata=True)
# Derive real covariance directly from independent cosine/sine Fourier synthesis.
n=256; fs=10000; phase=2*np.pi*np.outer(np.arange(n),np.arange(n//2+1))/n
power=density*fs*n; power[1:-1]/=2
temporal=(2*(np.cos(phase[:,1:-1])*power[1:-1])@np.cos(phase[:,1:-1]).T+2*(np.sin(phase[:,1:-1])*power[1:-1])@np.sin(phase[:,1:-1]).T+power[-1]*np.outer(np.cos(phase[:,-1]),np.cos(phase[:,-1]))+power[0])/n**2
r['source_covariance_max_error']=float(np.max(abs(temporal-m.temporal_covariance)))
assert r['source_covariance_max_error']<1e-12
assert np.max(abs(h-m.channel_covariance))<1e-12
floor=next(x for x in pmeta['component_contributions'] if x['name']=='floor')['integrated_power_after_global_scaling']
vwhite=np.trace(h)/8*floor; eps=.01*np.trace(h)/8*np.trace(temporal)/n
assert abs(vwhite-m.v_white)<1e-12 and abs(eps-m.epsilon_dc)<1e-12
# Filter basis vectors rather than random draws: F F.T must equal exact T.
f=np.fft.irfft(np.fft.rfft(np.eye(n),axis=0)*np.sqrt(power/n)[:,None],n=n,axis=0)
r['filtered_white_identity_error']=float(np.max(abs(f@f.T-temporal)))
assert r['filtered_white_identity_error']<1e-12
expected=np.kron(h,temporal)+eps*np.eye(2048)+np.kron(np.ones((8,8)),1.5*vwhite*m.line_basis.T@m.line_basis+.45*vwhite/2*m.common_basis.T@m.common_basis)
r['total_covariance_error']=float(np.max(abs(expected-m.noise_covariance)))
assert r['total_covariance_error']<1e-12
# Direct decomposition into physics + exact named independent noise streams.
b=m.sample([11,3],return_components=True); reverse=m.sample([3,11])
assert np.array_equal(b['raw'],reverse['raw'][::-1])
for i in range(2):
 assert np.max(abs((b['raw'][i]-sum(b['components'][i].values())).ravel()-m.j0@b['theta'][i]))<1e-12
assert not np.array_equal(m.sample([11],role='dev')['raw'],m.sample([11],role='eval')['raw'])
r['namespaces_and_fixed_j0']='pass'
# Full modified covariance Fisher, including actual physical epsilon.
j=m.jacobian(m.theta0); half=m.jacobian(m.theta0,half=True)
r['half_step_relative_error']=float(np.linalg.norm(j-half)/np.linalg.norm(half)); assert r['half_step_relative_error']<1e-5
factor=cho_factor(expected); info=j.T@cho_solve(factor,j); bound=np.linalg.inv(info)
# Exercise production CRLB, not just independent formula.
actual=crlb.crlb(j,expected)
r['production_crlb_relative_error']=float(np.linalg.norm(actual-bound)/np.linalg.norm(bound)); assert r['production_crlb_relative_error']<1e-8
amp=m.waveform(m.theta0).ravel()/m.theta0[0]
of=crlb.of_amplitude(amp,expected)['variance']
r['of_vs_information_relative_error']=float(abs(of*info[0,0]-1)); assert r['of_vs_information_relative_error']<.01
r['source_sha256_after']=hashes(); r['sources_unchanged']=before==r['source_sha256_after']; r['elapsed_seconds']=time.perf_counter()-start
r['end_utc']=datetime.now(timezone.utc).isoformat()
(out/'waveform_independent_results.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
