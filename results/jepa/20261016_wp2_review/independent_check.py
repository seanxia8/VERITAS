"""Reviewer-owned bounded checks; no experiment grids or learned outcomes."""
import hashlib, json, platform, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.linalg import subspace_angles, eigh
from jepa_noise import linear, metrics, crlb
from jepa_noise.views import ExactModel
from jepa_noise.masks import raw_split

root=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent
paths=['docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md','docs/jepa/GATE_G5.md','docs/jepa/THEORY.md','docs/jepa/WP2_IMPLEMENTATION.md','src/jepa_noise/configs/linear_solver_v1.json']
paths += ['src/jepa_noise/'+p+'.py' for p in ['linear','masks','metrics','crlb','views','report']]
paths += ['src/jepa_noise/tests/test_'+p+'.py' for p in ['linear','theory']]
paths += ['src/latent_monitor/estimators/cw_pca.py','docs/jepa/toy_check_2026-10-16.py']
hashes=lambda:{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
start=time.perf_counter(); before=hashes()
record={'start_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'host':platform.node(),'source_sha256_before':before,'checks':{}}
check=record['checks']; rng=np.random.default_rng(1901)

# Independent population construction and generalized eigensolver reference.
dc,dt,k=5,4,2
ac=rng.normal(size=(dc,k)); at=rng.normal(size=(dt,k))
nc=np.diag([1.,2.,.7,1.3,.9]); nt=np.diag([.9,1.2,2.,.8]); v=np.diag([.4,.8])
sc=nc+ac@v@ac.T; st=nt+at@v@at.T; cross=ac@v@at.T
ref=np.linalg.solve(nc,ac)
angles={}
for name in ['cca','hard_isotropy','gevp','whitened_pca']:
    f=linear.fit(name,sc,st,cross,k=k,ncc=nc)
    angles[name]=float(max(subspace_angles(f.context.T,ref)))
    assert angles[name]<1e-6
check['population_recovery_radians']=angles

# Direct unreduced predictor objective: independently verify isotropies,
# stationarity under tangent rotations and dominance over random feasible maps.
hc=np.linalg.inv(np.linalg.cholesky(sc)); ht=np.linalg.inv(np.linalg.cholesky(st))
def objective(wc,wt,beta):
    p=wt@cross.T@wc.T
    return float(np.trace(wt@st@wt.T)-2*np.trace(p@wc@cross@wt.T)+np.trace(p@wc@sc@wc.T@p.T)+beta*np.trace(wc@nc@wc.T))
checks=[]
for beta in [0.,.1,1.,10.]:
    fit=linear.finite_beta(sc,st,cross,nc,k,beta)
    value=objective(fit.context,fit.target,beta)
    independent=k-eigh(cross@np.linalg.solve(st,cross.T)-beta*nc,sc,eigvals_only=True)[-k:].sum()
    assert abs(value-independent)<1e-8
    random_values=[]
    for _ in range(100):
        uc=np.linalg.qr(rng.normal(size=(dc,k)))[0].T
        ut=np.linalg.qr(rng.normal(size=(dt,k)))[0].T
        random_values.append(objective(uc@hc,ut@ht,beta))
    assert min(random_values)>=value-1e-8
    # Rotate whitened fitted rowspace in ambient context coordinates.
    uc=fit.context@np.linalg.cholesky(sc)
    skew=rng.normal(size=(dc,dc)); skew=skew-skew.T
    from scipy.linalg import expm
    h=1e-5
    derivative=(objective(uc@expm(h*skew)@hc,fit.target,beta)-objective(uc@expm(-h*skew)@hc,fit.target,beta))/(2*h)
    assert abs(derivative)<1e-7
    checks.append({'beta':beta,'objective':value,'independent_error':float(abs(value-independent)),'best_random_excess':min(random_values)-value,'tangent_fd':derivative})
check['unreduced_objective']=checks

# Known-parameter OF versus joint inverse Fisher, and analytical Jacobian.
j=np.array([[1.,1.],[0.,1.],[2.,-.2]])
n=np.diag([.7,1.3,2.]); information=j.T@np.linalg.solve(n,j)
assert np.max(abs(crlb.crlb(j,n)-np.linalg.inv(information)))<1e-12
assert abs(crlb.of_amplitude(j[:,0],n)['variance']-1/information[0,0])<1e-12
check['joint_vs_known_amplitude']=[float(np.linalg.inv(information)[0,0]),float(1/information[0,0])]
f=lambda t: np.array([np.exp(t[0]),t[0]*np.sin(t[1]),t[1]**3])
truth=np.array([[np.exp(.3),0],[np.sin(.4),.3*np.cos(.4)],[0,3*.4**2]])
err=[float(np.max(abs(crlb.central_jacobian(f,[.3,.4],[1.,2.],half=half)-truth))) for half in [False,True]]
assert max(err)<5e-8; check['analytic_jacobian_errors']=err

# Exact registered coordinate sets for every mask, not just coverage.
raw=np.arange(2048).reshape(8,256)
expected={'time':(raw[:,:96],raw[:,160:]),'channel':(raw[:4],raw[4:]),'interleaved':(raw[::2],raw[1::2]),'block':(raw[:4,:96],raw[4:,160:])}
for name,(c,t) in expected.items():
    split=raw_split(name)
    assert np.array_equal(split.context,c.ravel()) and np.array_equal(split.target,t.ravel())
check['mask_hashes']={name:raw_split(name).digest for name in expected}

# Adversarial constant-target input. Preserve finding without hiding failure.
a=np.full((100,1),.1); r=metrics.nuisance_r2(a,np.zeros_like(a))
check['constant_decimal_target']={'expected_valid':False,'actual_valid':bool(r['valid'][0]),'score':float(r['per_target'][0]),'passes':not bool(r['valid'][0])}
record['source_sha256_after']=hashes()
record['sources_unchanged']=before==record['source_sha256_after']
record['elapsed_seconds']=time.perf_counter()-start
record['end_utc']=datetime.now(timezone.utc).isoformat()
(out/'independent_results.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'checks':check,'sources_unchanged':record['sources_unchanged'],'elapsed_seconds':record['elapsed_seconds']},indent=2))
