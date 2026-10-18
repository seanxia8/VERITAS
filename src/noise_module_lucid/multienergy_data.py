"""Multi-energy prompt-track development dataset, explicit primary targets."""
import argparse, hashlib, json, time
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from .lucid_transport import PromptTransport
from .lucid_integration import analytic_first,response,time_feature
from .readout import nhits_trigger_time, readout_chain
from . import response_manifest
from .deposited_energy import deposited_energy


def inside_cylinder(points,radius,height):
    p=np.asarray(points)
    return (p[...,0]**2+p[...,1]**2<radius**2)&(np.abs(p[...,2])<height/2)


def transform_event(raw,rotation,vertex):
    """The primary origin maps to vertex; no truth-centering of the shower."""
    origin=raw['vertex_m']
    p=(raw['origins_m']-origin)@rotation.T+vertex
    d=np.asarray(raw['directions'],float);d=d/np.linalg.norm(d,axis=1,keepdims=True)
    direction=np.asarray(raw['primary_direction'],float);direction/=np.linalg.norm(direction)
    track=(raw['primary_path_m']-origin)@rotation.T+vertex
    return p,d@rotation.T,track,direction@rotation.T


def run(config,source,upstream,manifest,out,segments=None):
    cfg=json.loads(Path(config).read_text());source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();receipt=json.loads((source/'manifest.json').read_text());transport=PromptTransport(Path(upstream)/'config/WCTE_geom_config.json',manifest,max_candidates_per_ray=cfg['max_candidates_per_ray'])
    g=transport.geometry;pos=np.asarray(g.sensor_points,float);n=len(pos);ref=response(g)
    # R-4b switches. Absent keys reproduce r4_pilot.json (R-4): dark counts on, 1 us (window-length) time unit.
    if not cfg.get('dark_counts',True):ref=replace(ref,dark_rate_hz=0.0)
    unit=cfg.get('time_feature_unit_ns');time_origin=cfg.get('time_origin','window_start');offsets=list(cfg.get('readout_seed_offsets',[0]));only=set(cfg.get('generate_roles',cfg['splits']))
    if not offsets or len(set(offsets))!=len(offsets) or not all(isinstance(o,int) and o>=0 for o in offsets):raise ValueError('distinct nonnegative integer readout_seed_offsets required')
    module=np.asarray(g.detector.mpmt_id,dtype=int)
    mapping=dict(source='pinned WCTE npz metadata reordered by upstream Cylinder.from_file',channel_mpmt_id=module.tolist(),channel_pmt_id=np.asarray(g.detector.pmt_id).tolist(),physical_mpmt_groups=int(len(np.unique(module))),crate_map=None,response_factors='synthetic; not measured module calibration',snap_to_wall=bool(g.detector.snap_to_wall),max_position_snap_m=float(np.max(np.linalg.norm(pos-np.asarray(g.detector.raw_positions),axis=1))))
    (out/'physical_module_map.json').write_text(json.dumps(mapping,indent=2))
    sensor_tree=cKDTree(pos)
    xlist=[];ylist=[];rows=[];geometry=[];cone=[]
    for rec in receipt['events']:
        energy,eid=rec['energy_MeV'],rec['event_id'];p=source/f'e{energy}_{eid:03d}.npz'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=rec['sha256']:raise ValueError('source hash mismatch')
        raw=np.load(p);beta=raw['segment_beta'];d=np.asarray(raw['directions'],float);d/=np.linalg.norm(d,axis=1,keepdims=True)
        eligible=(beta*float(g.medium.refractive_index)>1)&(raw['wavelength_nm']>=300)&(raw['wavelength_nm']<=600)
        angle=np.degrees(np.arccos(np.clip(np.sum(d*raw['segment_direction'],axis=1)[eligible],-1,1)));theory=np.degrees(np.arccos(1/(beta[eligible]*float(g.medium.refractive_index))))
        cone.append(dict(energy_MeV=energy,event_id=eid,n=len(angle),median_abs_error_deg=float(np.median(np.abs(angle-theory))),p95_abs_error_deg=float(np.quantile(np.abs(angle-theory),.95))))
        role=next(k for k,v in cfg['splits'].items() if eid in v)
        if role not in only:continue
        target=cfg.get('energy_target','initial_energy')
        if target=='deposited_energy_in_volume':
            if segments is None:raise ValueError('deposited-energy target requires --segments')
            sp=np.load(Path(segments)/f'e{energy}_{eid:03d}_primary.npz')
            if not np.allclose(sp['vertex_m'],raw['vertex_m'],atol=1e-9,rtol=0) or float(sp['t0_mev'])!=float(raw['energy_MeV']):raise ValueError('segment file does not match photon source event')
        elif target!='initial_energy':raise ValueError(f'unknown energy target {target!r}')
        for aug in range(cfg['rotations_per_event']):
            if time.monotonic()-start>cfg['generation_wall_seconds']:raise RuntimeError('generation wall cap')
            rng=np.random.default_rng(cfg['augmentation_seed']+energy*10000+eid*100+aug);rot=Rotation.random(random_state=rng).as_matrix();vertex=rng.uniform(-.35,.35,3)
            o,dirs,track,axis=transform_event(raw,rot,vertex);inside=inside_cylinder(o,float(g.detector.r),float(g.detector.H));band=(raw['wavelength_nm']>=200)&(raw['wavelength_nm']<=800);in_sensor=np.zeros(len(o),dtype=bool)
            in_sensor[inside]=sensor_tree.query(o[inside],workers=4)[0]<=float(g.sensor_radius)
            keep=inside&band&~in_sensor
            pid=raw['photon_id'][keep];event_id=energy*100000+eid*100+aug
            batch=transport.export(event_id,pid,o[keep],dirs[keep],raw['emission_ns'][keep],raw['wavelength_nm'][keep])
            if aug==0:
                ix=np.searchsorted(pid,batch.photon_id);travel=(batch.arrival_time_ns-raw['emission_ns'][keep][ix])*float(g.speed_of_light);oc=o[keep][ix]-pos[batch.sensor_id];dd=dirs[keep][ix];bb=np.sum(oc*dd,axis=1);direct=-bb-np.sqrt(np.maximum(bb*bb-np.sum(oc*oc,axis=1)+float(g.sensor_radius)**2,0));hit=o[keep][ix]+dd*travel[:,None]
                sampled=np.linspace(0,len(pid)-1,cfg['geometry_check_rays'],dtype=int);expected=analytic_first(o[keep][sampled],dirs[keep][sampled],pos,float(g.sensor_radius),g.detector);observed=np.full(len(pid),-1,int);observed[ix]=batch.sensor_id
                geometry.append(dict(energy_MeV=energy,event_id=eid,agreement=float(np.mean(expected==observed[sampled])),max_time_error_ns=float(np.max(np.abs(direct-travel))/float(g.speed_of_light)),max_sphere_error_m=float(np.max(np.abs(np.linalg.norm(hit-pos[batch.sensor_id],axis=1)-float(g.sensor_radius))))))
            # Readout re-draws (precision study): the keyed seed changes, transport and placement do not.
            xs=[]
            for off in offsets:
                d_=readout_chain(batch.sensor_id,batch.arrival_time_ns,n,ref,seed=event_id+off*1_000_000_000,photon_batch=batch)
                x_=np.zeros((n,6),np.float32);x_[:,2:5]=pos/2;x_[d_.sensor,0]=np.log1p(d_.charge);x_[d_.sensor,1]=time_feature(d_.time,ref.readout_window_ns,unit,time_origin);x_[d_.sensor,5]=1
                xs.append(x_)
                if off==offsets[0]:digit=d_
            x=xs[0] if len(xs)==1 else np.stack(xs)
            dep=None
            if target=='deposited_energy_in_volume':
                origin=raw['vertex_m'];ss=(sp['starts_m']-origin)@rot.T+vertex;ee=(sp['ends_m']-origin)@rot.T+vertex
                dep=deposited_energy(ss,ee,sp['beta_start'],float(sp['t0_mev']),float(g.detector.r),float(g.detector.H))
                mid=(ss+ee)/2;dep['segment_edep_inside_mev']=float(sp['edep_mev'][inside_cylinder(mid,float(g.detector.r),float(g.detector.H))].sum())  # supporting cross-check only
            e_target=dep['e_dep_mev'] if dep else float(raw['energy_MeV'])
            xlist.append(x);ylist.append(np.r_[axis,vertex,np.log(e_target/1000)])
            rows.append(dict(event_id=event_id,physical_event_id=f'{energy}:{eid}',source_event_id=eid,energy_MeV=energy,role=role,rotation=aug,primary_track_contained=bool(inside_cylinder(track,float(g.detector.r),float(g.detector.H)).all()),source_photons=len(o),outside_volume=int((~inside).sum()),inside_sensor_volume=int(in_sensor.sum()),selected_photons=int(keep.sum()),incident_hits=len(batch.photon_id),digits=len(digit),charge_sum=float(digit.charge.sum()),overflow_clamped=int(digit.counts['overflow_clamped']),digits_before_arrival=int(digit.counts['digits_before_arrival']),nhits_fallback=bool(time_origin=='nhits' and nhits_trigger_time(digit) is None),energy_target=target,target_MeV=e_target,**({f'dep_{k}':v for k,v in dep.items()} if dep else {})))
        print(energy,eid,len(rows),flush=True)
    passed=all(c['agreement']==cfg['geometry_agreement_required'] and c['max_time_error_ns']<cfg['max_time_residual_ns'] and c['max_sphere_error_m']<cfg['max_sphere_residual_m'] for c in geometry)
    report=dict(passed=passed,config=cfg,readout_settings=dict(dark_rate_hz=ref.dark_rate_hz,time_feature_unit_ns=unit if unit is not None else 'window_length',time_origin=time_origin,readout_seed_offsets=offsets,generate_roles=sorted(only)),source=receipt,transport=transport.provenance(),geometry=geometry,cone=cone,events=rows,elapsed_seconds=time.monotonic()-start,response=ref.to_dict(),response_manifest=response_manifest.provenance(),readout_counts=dict(overflow_clamped=sum(r['overflow_clamped'] for r in rows),digits_before_arrival_fraction=sum(r['digits_before_arrival'] for r in rows)/max(1,sum(r['digits'] for r in rows)),nhits_fallback=sum(r['nhits_fallback'] for r in rows),nhits_fallback_by_cell={f'{k}:{e}':sum(r['nhits_fallback'] for r in rows if r['role']==k and r['energy_MeV']==e) for k in sorted(only) for e in cfg['energies_MeV']}),evidence_stage='development',non_citable=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'digits.npz',x=np.asarray(xlist),y=np.asarray(ylist,dtype=np.float32),role=np.array([r['role'] for r in rows]),physical_event_id=np.array([r['physical_event_id'] for r in rows]),energy_MeV=np.array([r['energy_MeV'] for r in rows]),contained=np.array([r['primary_track_contained'] for r in rows]),charge=np.array([r['charge_sum'] for r in rows]),target_MeV=np.array([r['target_MeV'] for r in rows]),energy_target=np.array([r['energy_target'] for r in rows]))
    print('transport_gate',passed,flush=True)
    if not passed:raise RuntimeError('transport closure failed; no training')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('config','source','upstream','manifest','out'):p.add_argument('--'+a,required=True)
    p.add_argument('--segments')
    run(**vars(p.parse_args()))
