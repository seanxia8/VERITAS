"""Paired PMT invariants on real Cherenkov photons; supporting development checks."""
from pathlib import Path
from dataclasses import replace
import argparse,json
import numpy as np
from scipy.spatial.transform import Rotation
from .lucid_transport import PromptTransport
from .lucid_integration import response
from .readout import readout_chain
from .quantum_efficiency import QECurve
from .timing_response import WalkCurve,ChannelWalkCurve


def run(upstream,source,manifest,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    transport=PromptTransport(Path(upstream)/'config/WCTE_geom_config.json',manifest);g=transport.geometry;ref=response(g.num_sensors)
    exact=replace(ref,qe_curve=QECurve((200,300,400,600,800),(0,.5,1,.5,0)),collection_efficiency=.4)
    shifted=replace(ref,walk_curve=ChannelWalkCurve(tuple(WalkCurve((c.coefficients_ns[0]+1,*c.coefficients_ns[1:]),c.domain_pe) for c in ref.walk_curve.curves)))
    rows=[]
    for eid in range(24):
        raw=np.load(Path(source)/f'event_{eid:03d}.npz');o=raw['origins_m'];d=np.asarray(raw['directions'],dtype=float);d/=np.linalg.norm(d,axis=1,keepdims=True)
        rot=Rotation.random(random_state=np.random.default_rng(20261016+eid*100)).as_matrix();o=(o-(o.min(0)+o.max(0))/2)@rot.T;d=d@rot.T
        keep=(o[:,0]**2+o[:,1]**2<float(g.detector.r)**2)&(np.abs(o[:,2])<float(g.detector.H)/2)&(raw['wavelength_nm']>=200)&(raw['wavelength_nm']<=800)
        batch=transport.export(eid*100,raw['photon_id'][keep],o[keep],d[keep],raw['emission_ns'][keep],raw['wavelength_nm'][keep])
        def digit(p,b=batch):return readout_chain(b.sensor_id,b.arrival_time_ns,g.num_sensors,p,seed=eid*100,photon_batch=b)
        a=digit(ref);b=digit(exact);c=digit(shifted);identity=digit(ref)
        row=dict(physical_event_id=eid,exact_product_equal=all(np.array_equal(getattr(a,k),getattr(b,k)) for k in ('sensor','charge','time')),identity_equal=all(np.array_equal(getattr(a,k),getattr(identity,k)) for k in ('sensor','charge','time')),fixed_tq_membership_equal=np.array_equal(a.sensor,c.sensor),fixed_tq_charge_equal=np.array_equal(a.charge,c.charge),fixed_tq_time_plus_1ns=bool(np.allclose(c.time-a.time,1,atol=1e-12,rtol=0)))
        reverse=replace(batch,**{k:getattr(batch,k)[::-1] for k in ('photon_id','sensor_id','arrival_time_ns','wavelength_nm')});r=digit(ref,reverse)
        row['photon_order_equal']=all(np.array_equal(getattr(a,k),getattr(r,k)) for k in ('sensor','charge','time'));rows.append(row)
    passed=all(all(v for k,v in row.items() if k!='physical_event_id') for row in rows)
    (out/'controls.json').write_text(json.dumps(dict(passed=passed,events=rows,evidence_stage='development',non_citable=True),indent=2));print('controls',passed,flush=True)
    if not passed:raise RuntimeError('paired invariant failure')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('upstream','source','manifest','out'):p.add_argument('--'+name,required=True)
    run(**vars(p.parse_args()))
