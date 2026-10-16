"""Bounded checkpoint review: independent direct-sample and augmented-LS checks."""
import hashlib, json, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
from jepa_noise.dynamics import empirical_moments, encoder, loss_and_gradients
from jepa_noise.probes import grouped_ridge, ALPHAS

resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
out = Path(__file__).resolve().parent
started = time.monotonic()
paths = [Path('src/jepa_noise')/p for p in ['dynamics.py','probes.py','metrics.py','tests/test_dynamics.py','tests/test_probes.py']]
hashes = lambda: {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
record = {'before': hashes(), 'environment': {k:os.environ.get(k) for k in ['PYTHONPATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}, 'python':sys.version}
rng = np.random.default_rng(81725)
c = rng.normal(size=(29,4)); t = c@rng.normal(size=(4,4)) + rng.normal(size=(29,4))
c -= c.mean(0); t -= t.mean(0)
errors=[]
for depth in [1,2,3]:
    layers=[rng.normal(size=(4,4))*.3 for _ in range(depth)]; p=rng.normal(size=(4,4))*.3
    frozen=encoder(layers).copy()
    direct=lambda: .5*np.mean(np.sum((c@(p@encoder(layers)).T-t@frozen.T)**2,axis=1))
    loss,gs,gp=loss_and_gradients(layers,p,empirical_moments(c,t))
    assert abs(loss-direct())<1e-12
    for param,grad in [*zip(layers,gs),(p,gp)]:
        for index in np.ndindex(param.shape):
            v=param[index]; h=1e-6
            param[index]=v+h; plus=direct()
            param[index]=v-h; minus=direct();param[index]=v
            errors.append(abs((plus-minus)/(2*h)-grad[index]))
assert max(errors)<1e-8
record['asymmetric_direct_sample_gradient_max_error']=max(errors)
x=rng.normal(size=(35,5)); y=np.c_[x@rng.normal(size=5)+2, rng.normal(size=35)]
groups=np.repeat(np.arange(5),7)
fits=grouped_ridge(x,y,groups,fit_intercept=True); maxerr=0.
for j,f in enumerate(fits):
    scores={}
    for alpha in ALPHAS:
        total=0.
        for g in range(5):
            train=groups!=g; held=~train
            mean=x[train].mean(0); scale=x[train].std(0); z=(x[train]-mean)/scale
            offset=y[train,j].mean()
            coef=np.linalg.lstsq(np.vstack([z,np.sqrt(alpha)*np.eye(5)]),np.r_[y[train,j]-offset,np.zeros(5)],rcond=None)[0]
            pred=(x[held]-mean)/scale@coef+offset
            total+=np.sum((pred-y[held,j])**2)
        scores[alpha]=total/len(x)
        maxerr=max(maxerr,abs(scores[alpha]-f['validation_mse'][alpha]))
    assert min(ALPHAS,key=lambda a:(scores[a],-a))==f['alpha']
assert maxerr<1e-10
record['augmented_svd_cv_mse_max_error']=maxerr
command=[sys.executable,'-m','pytest','src/jepa_noise/tests/test_dynamics.py','src/jepa_noise/tests/test_probes.py','-q']
run=subprocess.run(command,capture_output=True,text=True,timeout=15)
(out/'pytest.log').write_text(run.stdout+run.stderr)
record.update(command=command,exit_status=run.returncode,after=hashes(),elapsed_seconds=time.monotonic()-started)
assert record['before']==record['after'] and run.returncode==0
(out/'independent_results.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
