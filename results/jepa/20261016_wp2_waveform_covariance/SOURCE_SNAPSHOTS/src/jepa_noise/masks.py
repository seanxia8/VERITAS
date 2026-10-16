"""Registered E1b/E2 raw-index masks; no data or labels enter selection here."""
from dataclasses import dataclass
import hashlib
import numpy as np

@dataclass(frozen=True)
class Split:
    context: np.ndarray
    target: np.ndarray
    unused: np.ndarray

    @property
    def digest(self):
        return hashlib.sha256(b''.join(x.astype('<i8').tobytes() for x in (self.context,self.target,self.unused))).hexdigest()


def raw_split(name):
    indices=np.arange(8*256).reshape(8,256)
    if name=='time': c,t=indices[:,:96],indices[:,160:]
    elif name=='channel': c,t=indices[:4],indices[4:]
    elif name=='interleaved': c,t=indices[::2],indices[1::2]
    elif name=='block': c,t=indices[:4,:96],indices[4:,160:]
    else: raise ValueError('unknown registered split')
    c,t=c.ravel(),t.ravel()
    return Split(c,t,np.setdiff1d(indices.ravel(),np.r_[c,t]))


def namespace_rng(seed,role,cell='',model_seed=0):
    """Stable independent namespaces, identical across paired methods."""
    import json
    key=json.dumps([int(seed),str(role),str(cell),int(model_seed)],separators=(',',':')).encode()
    entropy=np.frombuffer(hashlib.sha256(key).digest(),dtype='<u4').tolist()
    return np.random.default_rng(np.random.SeedSequence(entropy))
