"""Bounded public PhotonSim example extraction with persistent physical IDs."""
import io, requests, uproot, json
class Ranges(io.RawIOBase):
 def __init__(self):
  self.url='https://cernbox.cern.ch/remote.php/dav/public-files/vOhH8P78hnSQKNZ/ROOT_files/water/mu-/500MeV_100events.root'; self.session=requests.Session(); self.session.auth=('vOhH8P78hnSQKNZ',''); self.size=428491135; self.pos=0; self.used=0
 def readable(self): return True
 def seekable(self): return True
 def tell(self): return self.pos
 def seek(self,n,whence=0):
  self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
  return self.pos
 def read(self,n=-1):
  if n<0: n=self.size-self.pos
  if self.used+n>128*1024**2: raise RuntimeError('128 MiB network cap')
  r=self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=30,stream=True)
  if r.status_code!=206 or int(r.headers.get('Content-Length',-1))!=n:
   r.close(); raise RuntimeError(f'Range failure {r.status_code}')
  content=r.content; r.close()
  if len(content)!=n: raise RuntimeError('short range response')
  self.used+=n; self.pos+=n; return content

if __name__ == "__main__":
    import numpy as np, pathlib, hashlib

    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',required=True)
    out=pathlib.Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    s=Ranges(); rows=[]
    with uproot.open(s) as f:
     raw=f['OpticalPhotonsRaw']; meta=f['OpticalPhotons']
     eids=raw['EventID'].array(library='np'); starts=raw['ChunkStartID'].array(library='np')
     keys=['EventID','NOpticalPhotons','Segment_NCherenkov','PrimaryEnergy','Photon_SegmentIndex','Segment_DirX','Segment_DirY','Segment_DirZ','Segment_BetaStart','Segment_TrackID','TrackInfo_TrackID','TrackInfo_ParentTrackID','TrackInfo_DirX','TrackInfo_DirY','TrackInfo_DirZ','TrackInfo_PosX','TrackInfo_PosY','TrackInfo_PosZ']
     for event in range(24):
      ix=np.flatnonzero(eids==event);ix=ix[np.argsort(starts[ix])]
      if len(ix)!=1 or starts[ix[0]]!=0: raise ValueError('bounded fixture expects one complete chunk per event')
      a=raw.arrays(['PhotonPosX','PhotonPosY','PhotonPosZ','PhotonDirX','PhotonDirY','PhotonDirZ','PhotonTime','PhotonWavelength'],entry_start=int(ix[0]),entry_stop=int(ix[0])+1,library='np')
      b=meta.arrays(keys,entry_start=event,entry_stop=event+1,library='np')
      o=np.column_stack([a['PhotonPos'+k][0] for k in 'XYZ'])/1000
      d=np.column_stack([a['PhotonDir'+k][0] for k in 'XYZ'])
      seg=np.asarray(b['Photon_SegmentIndex'][0],dtype=int)
      sd=np.column_stack([b['Segment_Dir'+k][0] for k in 'XYZ'])
      beta=np.asarray(b['Segment_BetaStart'][0]); primary=np.flatnonzero(b['TrackInfo_ParentTrackID'][0]==0)
      if len(primary)!=1: raise ValueError('one primary required')
      j=primary[0]; axis=np.array([b['TrackInfo_Dir'+k][0][j] for k in 'XYZ'])
      v=np.array([b['TrackInfo_Pos'+k][0][j] for k in 'XYZ'])/1000
      if len(seg)!=len(o): raise ValueError('photon-segment alignment failure')
      if int(b['EventID'][0])!=event or int(b['NOpticalPhotons'][0])!=len(o) or int(np.sum(b['Segment_NCherenkov'][0]))!=len(o): raise ValueError('source identity or Cherenkov count mismatch')
      vals=dict(origins_m=o,directions=d,emission_ns=a['PhotonTime'][0],wavelength_nm=a['PhotonWavelength'][0],photon_id=np.arange(len(o),dtype=np.int64),segment_direction=sd[seg],segment_beta=beta[seg],primary_direction=axis,vertex_m=v,energy_MeV=b['PrimaryEnergy'][0])
      dest=out/f'event_{event:03d}.npz';np.savez_compressed(dest,**vals)
      rows.append(dict(event_id=event,photons=len(o),sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
      print(event,len(o),s.used,flush=True)
    (out/'manifest.json').write_text(json.dumps(dict(url=s.url,remote_bytes=s.size,read_bytes=s.used,cap_bytes=128*1024**2,source_units='PhotonSim mm converted to m; ns; nm',events=rows,evidence_stage='development',non_citable=True),indent=2))
