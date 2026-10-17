"""Real PhotonSim/LUCiD development closure and bounded digit dataset.

This is NOT a primary scientific evaluation or a target-PMT calibration.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .lucid_transport import PromptTransport
from .readout import readout_chain
from . import response_manifest


def response(geometry):
    """Manifest reference response on the pinned geometry's physical mPMT modules (D-G)."""
    module=np.asarray(geometry.detector.mpmt_id,dtype=int)
    if module.shape!=(len(geometry.sensor_points),):raise ValueError('mPMT map not aligned to sensors')
    return response_manifest.reference_response(len(module),response_manifest.module_groups(module))


def analytic_first(origins,directions,centres,radius,detector):
    """Independent float64 all-sensor intersection, no candidate grid."""
    oc=origins[:,None,:]-centres[None,:,:]
    b=np.einsum('nsc,nc->ns',oc,directions)
    disc=b*b-(oc*oc).sum(axis=-1)+radius*radius
    dist=-b-np.sqrt(np.maximum(disc,0))
    point=origins[:,None,:]+dist[:,:,None]*directions[:,None,:]
    # WCTE cylinder bounds, independent of the upstream JAX function.
    inside=(point[:,:,0]**2+point[:,:,1]**2 <= float(detector.r)**2) & (np.abs(point[:,:,2]) <= float(detector.H)/2)
    valid=(disc>0)&(dist>0)&inside
    times=np.where(valid,dist,np.inf)
    sid=times.argmin(axis=1); sid[~valid.any(axis=1)]=-1
    return sid


def run(upstream, source, config, manifest, out):
    start=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(config).read_text());src=json.loads((Path(source)/'manifest.json').read_text())
    transport=PromptTransport(Path(upstream)/'config/WCTE_geom_config.json',manifest)
    geom=transport.geometry;pos=np.asarray(geom.sensor_points,dtype=float);n=len(pos)
    ref=response(geom); arrays=[];labels=[];groups=[];rows=[];cone=[];checks=[]
    for rec in src['events']:
        path=Path(source)/f"event_{rec['event_id']:03d}.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['sha256']:raise ValueError('source hash mismatch')
        raw=np.load(path);eid=rec['event_id'];d=np.asarray(raw['directions'],dtype=float);d=d/np.linalg.norm(d,axis=1,keepdims=True);o=raw['origins_m']
        # Supporting local-emission cone check uses segment direction/beta, never a point vertex.
        beta=raw['segment_beta'];cos=np.sum(d*raw['segment_direction'],axis=1)
        valid=(beta*float(geom.medium.refractive_index)>1)&(raw['wavelength_nm']>=300)&(raw['wavelength_nm']<=600)
        angle=np.degrees(np.arccos(np.clip(cos[valid],-1,1)))
        theory=np.degrees(np.arccos(1/(beta[valid]*float(geom.medium.refractive_index))))
        cone.extend(np.abs(angle-theory).tolist())
        # Fixed centring preserves the complete source shower, then selects in-volume emissions.
        midpoint=(o.min(axis=0)+o.max(axis=0))/2
        for rotation in range(cfg['rotations_per_event']):
            if time.monotonic()-start>cfg['generation_wall_seconds']:raise RuntimeError('generation wall cap')
            rot=Rotation.random(random_state=np.random.default_rng(20261016+eid*100+rotation)).as_matrix()
            origins=(o-midpoint)@rot.T;directions=d@rot.T
            inside=(origins[:,0]**2+origins[:,1]**2<float(geom.detector.r)**2)&(np.abs(origins[:,2])<float(geom.detector.H)/2)
            # Spectral support declared before response; retain counts of all exclusions.
            band=(raw['wavelength_nm']>=200)&(raw['wavelength_nm']<=800)
            selected=inside&band
            pid=raw['photon_id'][selected]; oo=origins[selected];dd=directions[selected]
            event_id=eid*100+rotation
            batch=transport.export(event_id,pid,oo,dd,raw['emission_ns'][selected],raw['wavelength_nm'][selected])
            if rotation==0:
                ix=np.searchsorted(pid,batch.photon_id)
                travel=(batch.arrival_time_ns-raw['emission_ns'][selected][ix])*float(geom.speed_of_light)
                hit=oo[ix]+dd[ix]*travel[:,None]
                radius_error=np.max(np.abs(np.linalg.norm(hit-pos[batch.sensor_id],axis=1)-float(geom.sensor_radius))) if len(ix) else float('inf')
                ixcheck=np.linspace(0,len(pid)-1,cfg['geometry_check_rays'],dtype=int)
                expected=analytic_first(oo[ixcheck],dd[ixcheck],pos,float(geom.sensor_radius),geom.detector)
                observed=np.full(len(pid),-1,dtype=int);observed[ix]=batch.sensor_id
                agreement=float(np.mean(expected==observed[ixcheck]))
                # Independent quadratic entry time for exported selected sensors.
                oc=oo[ix]-pos[batch.sensor_id];bb=np.sum(oc*dd[ix],axis=1)
                direct=-bb-np.sqrt(np.maximum(bb*bb-np.sum(oc*oc,axis=1)+float(geom.sensor_radius)**2,0))
                terr=float(np.max(np.abs(direct-travel))/float(geom.speed_of_light))
                checks.append(dict(event_id=eid,analytic_sensor_agreement=agreement,max_sphere_error_m=float(radius_error),max_time_error_ns=terr))
            digits=readout_chain(batch.sensor_id,batch.arrival_time_ns,n,ref,seed=event_id,photon_batch=batch)
            x=np.zeros((n,6),np.float32);x[:,2:5]=pos/2
            x[digits.sensor,0]=np.log1p(digits.charge)
            x[digits.sensor,1]=(digits.time-digits.time.min())/20 if len(digits) else 0
            x[digits.sensor,5]=1
            arrays.append(x);labels.append(raw['primary_direction']@rot.T);groups.append(eid)
            rows.append(dict(event_id=event_id,physical_event_id=eid,rotation=rotation,source_photons=len(pid),outside_volume=int((~inside).sum()),outside_spectrum=int((~band).sum()),incident_hits=len(batch.photon_id),digits=len(digits),overflow_clamped=int(digits.counts['overflow_clamped']),digits_before_arrival=int(digits.counts['digits_before_arrival'])))
        print('physical_event',eid,'samples',len(arrays),flush=True)
    cone=np.asarray(cone);limits=cfg['closure']
    passed=all(c['analytic_sensor_agreement']>=limits['analytic_sensor_agreement'] and c['max_sphere_error_m']<limits['max_ray_sphere_distance_error_m'] and c['max_time_error_ns']<limits['max_flight_time_residual_ns'] for c in checks)
    cone_summary=dict(n=len(cone),median_abs_error_deg=float(np.median(cone)),p95_abs_error_deg=float(np.quantile(cone,.95)),interpretation='constant-index approximation versus PhotonSim local segment emission; supporting only')
    passed &= cone_summary['median_abs_error_deg']<limits['cone_median_abs_error_deg_max'] and cone_summary['p95_abs_error_deg']<limits['cone_p95_abs_error_deg_max']
    report=dict(passed=bool(passed),transport=transport.provenance(),geometry_checks=checks,cone=cone_summary,events=rows,response=ref.to_dict(),response_manifest=response_manifest.provenance(),readout_counts=dict(overflow_clamped=sum(r['overflow_clamped'] for r in rows),digits_before_arrival_fraction=sum(r['digits_before_arrival'] for r in rows)/max(1,sum(r['digits'] for r in rows))),source_manifest=src,config=cfg,elapsed_seconds=time.monotonic()-start)
    (out/'closure.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/'digits.npz',x=np.asarray(arrays),y=np.asarray(labels,dtype=np.float32),physical_event_id=np.asarray(groups))
    print(json.dumps(dict(passed=bool(passed),cone=cone_summary,elapsed_seconds=report['elapsed_seconds'])),flush=True)
    if not passed:raise RuntimeError('predeclared closure failed: training prohibited')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('upstream','source','config','manifest','out'):p.add_argument('--'+name,required=True)
    run(**vars(p.parse_args()))
