import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from noise_module_lucid.lucid_transport import PromptTransport
from noise_module_lucid.multienergy_data import transform_event, inside_cylinder
from noise_module_lucid.lucid_integration import analytic_first

root=Path('/work/dwong/pmt_multienergy_20261016')
up=Path('/work/dwong/pmt_lucid_20261016')
raw=np.load(root/'source_balanced/e1000_018.npz')
results=[]
for candidates in (16,64):
 t=PromptTransport(up/'upstream/config/WCTE_geom_config.json',up/'code/configs/lucid_pmt/upstream_manifest.json',max_candidates_per_ray=candidates);g=t.geometry
 rng=np.random.default_rng(202610170+1000*10000+18*100);rot=Rotation.random(random_state=rng).as_matrix();vertex=rng.uniform(-.35,.35,3)
 o,d,_,_=transform_event(raw,rot,vertex);pos=np.asarray(g.sensor_points)
 keep=inside_cylinder(o,float(g.detector.r),float(g.detector.H))&(raw['wavelength_nm']>=200)&(raw['wavelength_nm']<=800)
 keep &= cKDTree(pos).query(o,workers=4)[0]>float(g.sensor_radius)
 o=o[keep];d=d[keep];ix=np.linspace(0,len(o)-1,256,dtype=int);d=d/np.linalg.norm(d,axis=1,keepdims=True)
 r=g.propagator(o[ix],d[ix]);dist=np.asarray(r['times']).reshape(-1,256);sid=np.asarray(r['sensor_indices']);valid=np.asarray(r['inside_sensor'])&(dist>0)&(sid>=0)
 first=np.argmin(np.where(valid,dist,np.inf),axis=0);obs=sid[first,np.arange(256)];obs[~valid.any(0)]=-1
 exp=analytic_first(o[ix],d[ix],pos,float(g.sensor_radius),g.detector)
 mismatches=[dict(sample=int(j),expected=int(exp[j]),observed=int(obs[j]),candidates=sid[:,j].tolist(),expected_in_candidates=bool(exp[j] in sid[:,j])) for j in np.flatnonzero(obs!=exp)]
 row=dict(candidate_limit=candidates,agreement=float(np.mean(exp==obs)),mismatches=mismatches);results.append(row);print(json.dumps(row),flush=True)
(root/'evidence/diagnosis_02.json').write_text(json.dumps(results,indent=2))
assert results[1]['agreement']==1
