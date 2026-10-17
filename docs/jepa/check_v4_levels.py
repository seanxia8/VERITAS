# v4 design check (16 Oct 2026): population displacement of linear JEPA (CCA) from the Paper 1
# noise-weighted subspace, A4 Tier-1 design vs v3 design. Run from repo root:
#   PYTHONPATH=src python docs/jepa/check_v4_levels.py
# Uses jepa_noise.views.Tier1Views (JEPA f021d3c) for J0, temporal covariance, line/common bases.
import numpy as np, scipy.linalg as sl, sys
sys.path.insert(0,'src')
from jepa_noise.views import Tier1Views
from jepa_noise.masks import raw_split
def isqrt(M):
    w,V=np.linalg.eigh(M); return V@np.diag(w**-0.5)@V.T
def angle(U,V):
    qu,_=np.linalg.qr(U);qv,_=np.linalg.qr(V);s=np.linalg.svd(qu.T@qv,compute_uv=False);return np.degrees(np.arccos(np.clip(s.min(),0,1)))
v=Tier1Views(split='channel',line_multiplier=0.,common_strength=0.)
sd_e2=np.array([3/np.sqrt(12),0.1/np.sqrt(12),0.04/np.sqrt(12)])
def build(rho, g, line, theta_sd, split):
    sh=np.sqrt(rho/0.4)*v.shared_weight; pr=np.sqrt((1-rho)/0.6)*v.private_weight
    C=np.outer(sh,sh)+np.diag(pr**2)
    N=np.kron(C,v.temporal_covariance)+v.epsilon_dc*np.eye(2048)
    N+=np.kron(np.ones((8,8)),line*v.v_white*v.line_basis.T@v.line_basis)
    N+=np.kron(np.ones((8,8)),g*v.v_white/2*v.common_basis.T@v.common_basis)
    s=raw_split(split); c,t=s.context,s.target
    ncc,ntt,nct=N[np.ix_(c,c)],N[np.ix_(t,t)],N[np.ix_(c,t)]
    ac,at=v.j0[c],v.j0[t]; S=np.diag(theta_sd**2)
    scc=ncc+ac@S@ac.T; stt=ntt+at@S@at.T; sct=nct+ac@S@at.T
    H=isqrt(scc)@sct@isqrt(stt); U,sv,Vt=np.linalg.svd(H)
    gam=np.linalg.svd(isqrt(ncc)@nct@isqrt(ntt),compute_uv=False)[0]
    Hs=isqrt(scc)@(ac@S@at.T)@isqrt(stt); ssv=np.linalg.svd(Hs,compute_uv=False)
    ref=np.linalg.solve(ncc,ac)
    out={}
    for k in (1,2,3):
        W=isqrt(scc)@U[:,:k]   # context canonical directions as functionals (columns)
        # compare functional spans: CCA functionals vs reference functionals span(N^-1 A) top-k GEVP
        w,R=sl.eigh(ac@S@ac.T,ncc); Rk=R[:,np.argsort(w)[::-1][:k]]
        out[k]=angle(W,Rk)
    return gam, sv[:4], ssv[:3], out
for split in ['channel','interleaved']:
  for rho in [0.4,0.0]:
    for sdname,sd in [('v3sd',v.theta_sd),('e2sd',sd_e2)]:
        print(f"== split={split} base_rho={rho} theta_sd={sdname}")
        for g in [0,0.005,0.01,0.02,0.05,0.1,0.225,0.45,0.9]:
            gam,sv,ssv,ang=build(rho,g,0.,sd,split)
            print(f"  g={g:<6} gamma_n={gam:.3f} cc={np.round(sv,3)} signalcc={np.round(ssv,3)} angle k1/2/3={ang[1]:.1f}/{ang[2]:.1f}/{ang[3]:.1f}")
