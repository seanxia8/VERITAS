"""Recompute saved full covariance without repeating source generation."""
import hashlib,json,time,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from jepa_noise.views import Tier1Views
start=time.perf_counter(); out=Path(__file__).resolve().parent
src=out.parent/'20261016_wp2_waveform_covariance'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inventory={name:digest for digest,name in (line.split() for line in (src/'SHA256SUMS').read_text().splitlines())}
assert all(sha(src/name)==digest for name,digest in inventory.items())
cfg=json.loads((src/'PREDECLARATION.json').read_text()); result=json.loads((src/'RESULT.json').read_text()); provenance=json.loads((src/'PROVENANCE.json').read_text()); projection=json.loads((src/'PROJECTION.json').read_text())
assert cfg['n_records']==result['n']==200000
assert provenance['predeclaration_sha256']==sha(src/'PREDECLARATION.json')
assert provenance['runner_sha256']==sha(src/'check_full_covariance.py')
assert not result['source_changed_during_run']
assert provenance['source_sha256']['src/jepa_noise/views.py']=='692ab80833bd5440f4426b358bd3996ad9259781e52fe5aaa701b76fac3aa277'
assert provenance['source_sha256']['src/jepa_noise/configs/waveform_v1.json']=='efa2c03fc40a875c0a0d1410bfca10685f6d32f74def74dac55364e87136f712'
m=Tier1Views(seed=cfg['seed'],cell=cfg['cell'],model_seed=cfg['model_seed'],line_multiplier=cfg['line_multiplier'],common_strength=cfg['common_strength'],signal_mode=cfg['signal_mode'],split=cfg['split'])
assert m.manifest()==result['model_manifest']
n=result['n']; sums=np.load(src/'SUM_X.npy'); xx=np.load(src/'SUM_XX.npy'); saved=np.load(src/'EMPIRICAL_COVARIANCE.npy')
assert sums.shape==(2048,) and xx.shape==(2048,2048)
centered=xx-np.outer(sums,sums)/n
c0=xx/n-np.outer(sums/n,sums/n)
c1=centered/(n-1)
assert np.array_equal(c0,saved)
truth=m.noise_covariance; norm=np.linalg.norm(truth)
e0=float(np.linalg.norm(c0-truth)/norm); e1=float(np.linalg.norm(c1-truth)/norm)
assert abs(e0-result['relative_frobenius_error'])<1e-12 and max(e0,e1)<.02
tr=np.trace(truth); fro2=np.sum(truth**2)
rms=float(np.sqrt(((n-1)*(tr**2+fro2)+fro2)/(n*n*fro2)))
assert abs(rms-projection['gaussian_expected_relative_rms'])<1e-15
r={'status':'ACCEPT_full_covariance_fixture','source_directory':str(src.resolve()),'producer_host':provenance['host'],'n':n,'shape':[2048,2048],'all_producer_hashes_verified':True,'producer_inventory':inventory,'empirical_byte_equality':True,'ddof0_relative_frobenius':e0,'ddof1_relative_frobenius':e1,'normalization_factor_n_over_nminus1':n/(n-1),'expected_ddof0_relative_rms':rms,'producer_active_seconds':result['active_seconds'],'review_elapsed_seconds':time.perf_counter()-start,'review_end_utc':datetime.now(timezone.utc).isoformat(),'review_host':platform.node(),'limitations':'Recomputed complete sufficient statistics; did not regenerate all noise draws or verify a saved prefix because no prefix hash was stored.'}
(out/'full_covariance_recheck.json').write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,indent=2))
