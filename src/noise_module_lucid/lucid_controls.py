"""Paired PMT invariants on real Cherenkov photons; supporting development checks.

The T-Q control changes the D-TQ slope ``b`` (runbook v2 R-2), so its time shift
must be charge-dependent; a separate offset-only control moves ``a`` alone.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json
import numpy as np
from scipy.spatial.transform import Rotation
from .lucid_transport import PromptTransport
from .lucid_integration import response
from .readout import readout_chain
from . import response_manifest

SLOPE_FACTORS=(1.2,0.8)
OFFSET_NS=1.0


def _same(a,b,keys=('sensor','charge','time')):
    return all(np.array_equal(getattr(a,k),getattr(b,k)) for k in keys)


def paired_checks(digit,ref,batch):
    """Invariants for one event; ``digit(params, batch)`` runs the readout with fixed draws."""
    exact=response_manifest.exact_product_control(ref)
    a=digit(ref,batch)
    row=dict(n_digits=len(a),exact_product_equal=_same(a,digit(exact,batch)),identity_equal=_same(a,digit(ref,batch)))
    curve=ref.walk_curve
    q=np.minimum(a.charge,curve.q_sat_pe)
    for factor in SLOPE_FACTORS:
        c=digit(replace(ref,walk_curve=curve.scaled(factor)),batch);tag=f'tq_slope_{factor:g}'
        expected=(factor-1)*curve.b_ns/np.sqrt(q)
        row[tag+'_membership_equal']=bool(np.array_equal(a.sensor,c.sensor))
        row[tag+'_charge_equal']=bool(np.array_equal(a.charge,c.charge))
        row[tag+'_shift_matches_curve']=bool(len(a)==len(c) and np.allclose(c.time-a.time,expected,atol=1e-9,rtol=0))
        # A slope change must not reduce to a global offset whenever charges differ.
        row[tag+'_charge_dependent']=bool(len(np.unique(q))<2 or np.ptp(c.time-a.time)>1e-9)
    o=digit(replace(ref,walk_curve=curve.scaled(offset_ns=OFFSET_NS)),batch)
    row['tq_offset_control_membership_charge_equal']=_same(a,o,('sensor','charge'))
    row['tq_offset_control_time_plus_offset']=bool(np.allclose(o.time-a.time,OFFSET_NS,atol=1e-12,rtol=0))
    reverse=replace(batch,**{k:getattr(batch,k)[::-1] for k in ('photon_id','sensor_id','arrival_time_ns','wavelength_nm')})
    row['photon_order_equal']=_same(a,digit(ref,reverse))
    row['overflow_clamped']=int(a.counts.get('overflow_clamped',0))
    row['digits_before_arrival']=int(a.counts['digits_before_arrival'])
    return row


def passed(rows):
    return all(all(v for k,v in row.items() if isinstance(v,bool)) for row in rows)


def run(upstream,source,manifest,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    transport=PromptTransport(Path(upstream)/'config/WCTE_geom_config.json',manifest);g=transport.geometry;ref=response(g)
    rows=[]
    for eid in range(24):
        raw=np.load(Path(source)/f'event_{eid:03d}.npz');o=raw['origins_m'];d=np.asarray(raw['directions'],dtype=float);d/=np.linalg.norm(d,axis=1,keepdims=True)
        rot=Rotation.random(random_state=np.random.default_rng(20261016+eid*100)).as_matrix();o=(o-(o.min(0)+o.max(0))/2)@rot.T;d=d@rot.T
        keep=(o[:,0]**2+o[:,1]**2<float(g.detector.r)**2)&(np.abs(o[:,2])<float(g.detector.H)/2)&(raw['wavelength_nm']>=200)&(raw['wavelength_nm']<=800)
        batch=transport.export(eid*100,raw['photon_id'][keep],o[keep],d[keep],raw['emission_ns'][keep],raw['wavelength_nm'][keep])
        def digit(p,b,eid=eid):return readout_chain(b.sensor_id,b.arrival_time_ns,g.num_sensors,p,seed=eid*100,photon_batch=b)
        rows.append(dict(physical_event_id=eid,**paired_checks(digit,ref,batch)))
    ok=passed(rows)
    (out/'controls.json').write_text(json.dumps(dict(passed=ok,events=rows,slope_factors=SLOPE_FACTORS,offset_ns=OFFSET_NS,response_manifest=response_manifest.provenance(),evidence_stage='development',non_citable=True),indent=2));print('controls',ok,flush=True)
    if not ok:raise RuntimeError('paired invariant failure')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('upstream','source','manifest','out'):p.add_argument('--'+name,required=True)
    run(**vars(p.parse_args()))
