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
  r=self.session.get(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'},timeout=30)
  if r.status_code!=206 or len(r.content)!=n: raise RuntimeError(f'Range failure {r.status_code}')
  self.used+=n; self.pos+=n; return r.content
s=Ranges()
with uproot.open(s) as f:
 print(f.keys())
 for k in f.keys():
  if hasattr(f[k],'keys'): print(k, f[k].num_entries, f[k].keys())
print('bytes',s.used)
