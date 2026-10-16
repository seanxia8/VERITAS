"""Bounded multi-energy PhotonSim subset; chunks retain global photon identity."""
import argparse, hashlib, io, json, time
from pathlib import Path
import numpy as np
import requests
import uproot

BASE='https://cernbox.cern.ch/remote.php/dav/public-files/vOhH8P78hnSQKNZ/ROOT_files/water/mu-/'
SIZES={500:428491135,1000:965067614,1500:1495166847}

class Ranges(io.RawIOBase):
    def __init__(self, energy, budget):
        self.url=BASE+f'{energy}MeV_100events.root';self.size=SIZES[energy];self.pos=0;self.budget=budget
        self.session=requests.Session();self.session.auth=('vOhH8P78hnSQKNZ','')
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        if not n:return b''
        if self.budget['used']+n>self.budget['cap']:raise RuntimeError('aggregate source read cap')
        if time.monotonic()>self.budget['deadline']:raise RuntimeError('source wall cap')
        with self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},stream=True,timeout=40) as r:
            if r.status_code!=206 or int(r.headers.get('Content-Length',-1))!=n:raise RuntimeError('range response rejected')
            result=r.content
        if len(result)!=n:raise RuntimeError('short range')
        self.pos+=n;self.budget['used']+=n;return result


def extract(config,out,old_500):
    cfg=json.loads(Path(config).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=False)
    budget={'used':0,'cap':cfg['source_read_cap_bytes'],'deadline':time.monotonic()+cfg['source_wall_seconds']};rows=[]
    keys=['EventID','PrimaryEnergy','NOpticalPhotons','Photon_SegmentIndex','Segment_NCherenkov','Segment_TrackID','Segment_BetaStart']+[f'Segment_{q}{a}' for q in ('Start','End','Dir') for a in 'XYZ']+['TrackInfo_TrackID','TrackInfo_ParentTrackID','TrackInfo_Energy']+[f'TrackInfo_{q}{a}' for q in ('Pos','Dir') for a in 'XYZ']
    for energy in cfg['energies_MeV']:
        source=Ranges(energy,budget)
        with uproot.open(source) as f:
            raw=f['OpticalPhotonsRaw'];meta=f['OpticalPhotons']
            eids=raw['EventID'].array(library='np');starts=raw['ChunkStartID'].array(library='np')
            for event in range(cfg['events_per_energy']):
                b=meta.arrays(keys,entry_start=event,entry_stop=event+1,library='np')
                if energy==500:
                    previous=Path(old_500)/f'event_{event:03d}.npz'
                    receipt=json.loads((Path(old_500)/'manifest.json').read_text())['events'][event]
                    if hashlib.sha256(previous.read_bytes()).hexdigest()!=receipt['sha256']:raise ValueError('old source hash')
                    old=np.load(previous);o=old['origins_m'];d=old['directions'];times=old['emission_ns'];w=old['wavelength_nm'];pid=old['photon_id']
                else:
                    ix=np.flatnonzero(eids==event);ix=ix[np.argsort(starts[ix])];parts=[];offset=0
                    for chunk in ix:
                        a=raw.arrays([f'Photon{q}{k}' for q in ('Pos','Dir') for k in 'XYZ']+['PhotonTime','PhotonWavelength'],entry_start=int(chunk),entry_stop=int(chunk)+1,library='np')
                        if int(starts[chunk])!=offset:raise ValueError('photon chunk gap or overlap')
                        parts.append({k:v[0] for k,v in a.items()});offset+=len(a['PhotonTime'][0])
                    a={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
                    o=np.column_stack([a['PhotonPos'+k] for k in 'XYZ'])/1000;d=np.column_stack([a['PhotonDir'+k] for k in 'XYZ']);times=a['PhotonTime'];w=a['PhotonWavelength'];pid=np.arange(len(o),dtype=np.int64)
                seg=np.asarray(b['Photon_SegmentIndex'][0],int);primary=np.flatnonzero(b['TrackInfo_ParentTrackID'][0]==0)
                if len(primary)!=1:raise ValueError('one primary required')
                j=primary[0];track_id=b['TrackInfo_TrackID'][0][j];pmask=b['Segment_TrackID'][0]==track_id
                if int(b['EventID'][0])!=event or len(seg)!=len(o) or int(b['NOpticalPhotons'][0])!=len(o) or int(np.sum(b['Segment_NCherenkov'][0]))!=len(o):raise ValueError('identity/count mismatch')
                sd=np.column_stack([b['Segment_Dir'+k][0] for k in 'XYZ']);beta=b['Segment_BetaStart'][0]
                vals=dict(origins_m=o,directions=d,emission_ns=times,wavelength_nm=w,photon_id=pid,segment_direction=sd[seg],segment_beta=beta[seg],primary_direction=np.array([b['TrackInfo_Dir'+k][0][j] for k in 'XYZ']),vertex_m=np.array([b['TrackInfo_Pos'+k][0][j] for k in 'XYZ'])/1000,primary_path_m=np.concatenate([np.column_stack([b['Segment_'+q+k][0][pmask] for k in 'XYZ'])/1000 for q in ('Start','End')]),energy_MeV=float(b['PrimaryEnergy'][0]),track_initial_energy_MeV=float(b['TrackInfo_Energy'][0][j]))
                dest=out/f'e{energy}_{event:03d}.npz';np.savez_compressed(dest,**vals)
                rows.append(dict(energy_MeV=energy,event_id=event,photons=len(o),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),primary_energy=vals['energy_MeV'],track_initial_energy=vals['track_initial_energy_MeV']))
                (out/'progress.json').write_text(json.dumps({'events':rows,'bytes_read':budget['used']},indent=2))
                print(energy,event,len(o),budget['used'],flush=True)
    (out/'manifest.json').write_text(json.dumps({'events':rows,'bytes_read':budget['used'],'source_cap_bytes':budget['cap'],'source_base':BASE,'source_sizes':SIZES,'old_500':str(old_500),'units':['m','ns','nm'],'evidence_stage':'development'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('config','out','old_500'):p.add_argument('--'+arg.replace('_','-'),required=True)
    extract(**vars(p.parse_args()))
