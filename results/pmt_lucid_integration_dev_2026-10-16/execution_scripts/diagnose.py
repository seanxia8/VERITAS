import numpy as np,json
from scipy.spatial.transform import Rotation
from noise_module_lucid.lucid_transport import PromptTransport
from noise_module_lucid.lucid_integration import analytic_first
root='/work/dwong/pmt_lucid_20261016/'
t=PromptTransport(root+'upstream/config/WCTE_geom_config.json',root+'code/configs/lucid_pmt/upstream_manifest.json');g=t.geometry
raw=np.load(root+'source_subset/event_018.npz');o=raw['origins_m'];d=raw['directions'];rot=Rotation.random(random_state=np.random.default_rng(20261016+1800)).as_matrix();o=(o-(o.min(0)+o.max(0))/2)@rot.T;d=d@rot.T
sel=(o[:,0]**2+o[:,1]**2<float(g.detector.r)**2)&(np.abs(o[:,2])<float(g.detector.H)/2)&(raw['wavelength_nm']>=200)&(raw['wavelength_nm']<=800)
o=o[sel];d=d[sel];ix=np.linspace(0,len(o)-1,256,dtype=int)
r=g.propagator(o[ix],d[ix]);dist=np.asarray(r['times']).reshape(-1,256);valid=np.asarray(r['inside_sensor']);sid=np.asarray(r['sensor_indices']);first=np.argmin(np.where(valid,dist,np.inf),axis=0);obs=sid[first,np.arange(256)];obs[~valid.any(0)]=-1
exp=analytic_first(o[ix],d[ix],np.asarray(g.sensor_points),float(g.sensor_radius),g.detector)
for j in np.flatnonzero(obs!=exp):
 c=np.asarray(g.sensor_points)[exp[j]];oc=o[ix[j]]-c;b=oc@d[ix[j]];disc=b*b-oc@oc+float(g.sensor_radius)**2
 print('mismatch',j,'expected',exp[j],'observed',obs[j],'candidates',sid[:,j],'disc',disc,'norm',np.linalg.norm(d[ix[j]]),'inside',valid[:,j],flush=True)
