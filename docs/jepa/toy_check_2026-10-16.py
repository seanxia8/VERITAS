# Toy check for RUNBOOK_JEPA_NOISE_2026-10-16 §A.1 — linear JEPA (CCA) vs Paper 1 noise-weighted subspace.
# Run: python docs/jepa/toy_check_2026-10-16.py   (numpy, scipy; ~1 min)
import numpy as np, scipy.linalg as sl
rng=np.random.default_rng(0)
d=40; k=3; N=200000
def angles(U,V):
    qu,_=np.linalg.qr(U); qv,_=np.linalg.qr(V)
    s=np.linalg.svd(qu.T@qv,compute_uv=False); return np.degrees(np.arccos(np.clip(s,0,1))).max()
def cca(Xc,Xt,k):
    Scc=np.cov(Xc.T); Stt=np.cov(Xt.T); Stc=(Xt-Xt.mean(0)).T@(Xc-Xc.mean(0))/(len(Xc)-1)
    M=Stc.T@np.linalg.solve(Stt,Stc); w,V=sl.eigh(M,Scc); o=np.argsort(w)[::-1]
    return np.sqrt(np.clip(w[o],0,None)),V[:,o[:k]]
# physics: shared latent theta (k dims), templates differ per view; low SNR
Ac=rng.normal(size=(d,k)); At=rng.normal(size=(d,k))
# coloured noise in each view
L=rng.normal(size=(d,d))/np.sqrt(d); Sn=L@L.T+0.3*np.eye(d)
Ln=np.linalg.cholesky(Sn)
# coherent nuisance eta (e.g. line amplitude) shared across views, strong
Bc=rng.normal(size=(d,1)); Bt=rng.normal(size=(d,1))
for gam in [0.0,1.0]:
    th=rng.normal(size=(N,k))*0.4
    eta=rng.normal(size=(N,1))*3.0
    nc=rng.normal(size=(N,d))@Ln.T; nt=rng.normal(size=(N,d))@Ln.T
    Xc=th@Ac.T+gam*eta@Bc.T+nc; Xt=th@At.T+gam*eta@Bt.T+nt
    # Paper-1 noise-weighted subspace for context view: GEVP(Ac Sth Ac^T, Sn_total_noise_c)
    Snc=Sn+gam**2*9*Bc@Bc.T
    w,V=sl.eigh(Ac@Ac.T*0.16,Snc); Vnw=V[:,np.argsort(w)[::-1][:k]]
    r,Vj=cca(Xc,Xt,k)
    print(f"gamma_load={gam}: top canon corr {np.round(r[:4],3)}  angle(JEPA/CCA top-{k}, noise-weighted) = {angles(Vj,Vnw):.1f} deg")
    # whitening invariance
    Wh=np.linalg.inv(np.linalg.cholesky(Snc)).T
    r2,_=cca(Xc@Wh,Xt@Wh,k); print("   canon corr after whitening:",np.round(r2[:4],3))
    # noise-anchored: SIGReg-style whitening on events + min variance on noise-only records
    nrec_c=gam*rng.normal(size=(N,1))*3.0@Bc.T+rng.normal(size=(N,d))@Ln.T
    Sxx=np.cov(Xc.T); Snn=np.cov(nrec_c.T)
    w3,V3=sl.eigh(Snn,Sxx); Va=V3[:,np.argsort(w3)[:k]]
    print(f"   noise-anchored fixed point angle to noise-weighted = {angles(Va,Vnw):.1f} deg")
