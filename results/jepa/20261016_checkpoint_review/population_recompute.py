"""Independent raw-matrix population check; no production solvers imported."""
import json,hashlib,time,signal
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky,solve_triangular,eigh,svdvals,subspace_angles
from scipy.sparse.linalg import svds
out=Path(__file__).resolve().parent; started=time.perf_counter()
prior=out/'population_recompute.json'
record=json.loads(prior.read_text()) if prior.exists() else {'cells':[],'numeric_seconds':0.}
remaining=120-record['numeric_seconds']
if remaining<=1: raise RuntimeError('review allocation exhausted')
signal.signal(signal.SIGALRM,lambda *args: (_ for _ in ()).throw(TimeoutError('review 120-second allocation')))
signal.alarm(max(1,int(remaining)))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ih(a):
 w,v=eigh(a); return (v/np.sqrt(w))@v.T
raw=np.arange(2048).reshape(8,256)
splits={'time':(raw[:,:96].ravel(),raw[:,160:].ravel()),'channel':(raw[:4].ravel(),raw[4:].ravel()),'interleaved':(raw[::2].ravel(),raw[1::2].ravel()),'block':(raw[:4,:96].ravel(),raw[4:,160:].ravel())}
try:
 for path in sorted(out.parent.glob('20261016_wp2_population_*/provenance.json')):
  if path.parent.name in [x['cell'] for x in record['cells']]:continue
  provenance=json.loads(path.read_text())
  if provenance['status']!='complete':continue
  assert provenance['exit_status']==0 and provenance['changed_scientific_inputs']==[]
  assert provenance['code_sha']==provenance['final_code_sha']=='338326c58a8a2ae6194a41056a46dac60e68f0eb'
  for line in (path.parent/'SHA256SUMS').read_text().splitlines():
   h,name=line.split(); assert sha(path.parent/name)==h
  for name,h in provenance['raw_sha256'].items():assert sha(Path(name))==h
  rawdir=Path(provenance['raw_directory'].split(':',1)[1]); noise=np.load(rawdir/'noise_covariance.npy'); j=np.load(rawdir/'J0.npy')
  cell={'cell':path.parent.name,'line':provenance['line_multiplier'],'common':provenance['common_strength'],'raw_sha256':provenance['raw_sha256'],'splits':[]}
  for name,(c,t) in splits.items():
   observed=json.loads((path.parent/(name+'.json')).read_text()); nc=noise[np.ix_(c,c)]; nt=noise[np.ix_(t,t)]; ct=noise[np.ix_(c,t)]; ac=j[c]; at=j[t]
   lc=cholesky(nc,lower=True); lt=cholesky(nt,lower=True)
   wc=solve_triangular(lc,ct,lower=True); noise_cross=solve_triangular(lt,wc.T,lower=True).T
   gamma=float(svds(noise_cross,k=1,return_singular_vectors=False,v0=np.ones(min(noise_cross.shape)),tol=1e-12)[0])
   nc_inv_ac=solve_triangular(lc.T,solve_triangular(lc,ac,lower=True),lower=False)
   residual=at-ct.T@nc_inv_ac
   singular={key:svdvals(a) for key,a in [('context',ac),('target',at),('residual_target',residual)]}
   ratios={key:float(s[-1]/s[0]) for key,s in singular.items()}
   schur=nt-wc.T@wc; ls=cholesky(schur,lower=True)
   hc=solve_triangular(lc,ac,lower=True)*[.15,.005,.002]; ht=solve_triangular(ls,residual,lower=True)*[.15,.005,.002]
   qc,rc=np.linalg.qr(hc,mode='reduced'); qt,rt=np.linalg.qr(ht,mode='reduced')
   small=ih(np.eye(3)+rc@rc.T)@rc@rt.T@ih(np.eye(3)+rt@rt.T)
   u,rho,_=np.linalg.svd(small); kth=float(rho[-1]); eligible=min(ratios['context'],ratios['residual_target'])>1e-8 and kth>1e-8
   status='eligible' if eligible else ('oracle_rank_ineligible' if min(ratios['context'],ratios['residual_target'])<=1e-8 else 'oracle_canonical_ineligible')
   directions=solve_triangular(lc.T,qc@ih(np.eye(3)+rc@rc.T)@u,lower=False)
   angle=float(max(subspace_angles(directions,nc_inv_ac))) if eligible else None
   assert status==observed['status']
   assert abs(gamma-observed['gamma_n'])<1e-8
   assert abs(kth-observed['kth_correlation'])<1e-8
   for key in singular: assert np.allclose(singular[key],observed['singular_values'][key],atol=1e-8,rtol=1e-8)
   assert angle is None or angle<=1e-6
   cell['splits'].append({'split':name,'gamma_n':gamma,'status':status,'ratios':ratios,'kth_correlation':kth,'canonical_gap':kth,'recovery_angle_rad':angle,'producer_angle_rad':observed['population_recovery_angle_rad'],'gamma_abs_error':abs(gamma-observed['gamma_n']),'kth_abs_error':abs(kth-observed['kth_correlation'])})
  record['cells'].append(cell)
 linecells=sorted([x for x in record['cells'] if x['common']==.45],key=lambda x:x['line'])
 if len(linecells)==5:
  gammas=[next(s['gamma_n'] for s in x['splits'] if s['split']=='time') for x in linecells]
  record['E2_time_selection']={'targets':[.1,.5,.9],'gammas':gammas,'lines':[x['line'] for x in linecells],'selected_lines':[linecells[int(np.argmin(np.abs(np.array(gammas)-target)))]['line'] for target in [.1,.5,.9]],'distinct':len(set(int(np.argmin(np.abs(np.array(gammas)-target))) for target in [.1,.5,.9]))==3}
finally:
 signal.alarm(0); record['numeric_seconds']+=time.perf_counter()-started
 prior.write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({'cells':len(record['cells']),'seconds':record['numeric_seconds'],'E2':record.get('E2_time_selection')},indent=2))
