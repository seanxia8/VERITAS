"""Bounded fixtures, not registered empirical outcome grids."""
import numpy as np
import pytest
from jepa_noise import linear, metrics, crlb
from jepa_noise.views import ExactModel
from jepa_noise.masks import raw_split, namespace_rng
from latent_monitor.estimators.cw_pca import principal_angles as reference_angles


def model(): return ExactModel(d=6,k=2,seed=42,theta_scale=.7,eta_scale=1.)


def test_no_shared_noise_recovery_and_controls():
    p=model().population(0.)
    for name in ['cca','hard_isotropy','gevp','whitened_pca']:
        result=linear.fit(name,p['scc'],p['stt'],p['sct'],k=2,ncc=p['ncc'])
        assert max(metrics.principal_angles(result.context,p['reference'].T))<1e-6
    assert linear.fit('pca',p['scc'],k=2).context.shape==(2,6)


def test_angles_existing_utility_agree():
    rng=np.random.default_rng(4); a,b=rng.normal(size=(2,6)),rng.normal(size=(2,6))
    np.testing.assert_allclose(np.sort(metrics.principal_angles(a,b)),np.sort(reference_angles(a,b,degrees=False)),atol=1e-8)


def test_finite_beta_negative_and_rank_deficient_target_completion():
    s=np.eye(2); n=np.diag([.99,1]); cross=np.diag([.01,.99])
    for beta in [.1,1.,10.]:
        r=linear.finite_beta(s,s,cross,n,1,beta)
        assert abs(r.context[0,1])>1-1e-8
        assert r.diagnostics['status']=='converged'
    r=linear.finite_beta(np.eye(3),np.eye(3),np.diag([.8,0,0]),np.eye(3),2,1.)
    assert r.diagnostics['target_cross_rank']==1
    np.testing.assert_allclose(r.target@r.target.T,np.eye(2),atol=1e-8)
    r=linear.finite_beta(np.eye(3),np.eye(3),np.zeros((3,3)),np.eye(3),2,1.)
    assert r.diagnostics['target_cross_rank']==0
    np.testing.assert_allclose(r.target@r.target.T,np.eye(2),atol=1e-8)


def test_reduced_gradient_and_analytic_optimum():
    p=model().population(.7); hc,ht=linear.invsqrt(p['scc']),linear.invsqrt(p['stt'])
    t=hc@p['sct']@ht; a=t@t.T-.7*hc@p['ncc']@hc
    rng=np.random.default_rng(5); u=rng.normal(size=(2,6)); h=1e-6
    objective=lambda x:2-np.trace(x@a@x.T)
    gradient=-2*u@a
    for i in range(2):
        for j in range(6):
            e=np.zeros_like(u); e[i,j]=h
            np.testing.assert_allclose((objective(u+e)-objective(u-e))/(2*h),gradient[i,j],rtol=1e-6,atol=1e-8)
    r=linear.finite_beta(p['scc'],p['stt'],p['sct'],p['ncc'],2,.7)
    expected=2-np.linalg.eigvalsh(a)[-2:].sum()
    assert abs(r.diagnostics['objective']-expected)<1e-8


def test_invertible_cca_pullback():
    p=model().population(.7); rng=np.random.default_rng(8)
    tc=rng.normal(size=(6,6))+5*np.eye(6); tt=rng.normal(size=(6,6))+5*np.eye(6)
    a=linear.cca(p['scc'],p['stt'],p['sct'],2)
    b=linear.cca(tc@p['scc']@tc.T,tt@p['stt']@tt.T,tc@p['sct']@tt.T,2)
    np.testing.assert_allclose(a.spectrum,b.spectrum,atol=1e-8)
    assert max(metrics.principal_angles(a.context,b.context@tc))<1e-6


def test_oracle_eligible_and_raw_dependency():
    p=model().population(.7)
    r=linear.oracle(*(p[x] for x in ['scc','stt','sct','ncc','ntt','nct','ac','at']),2)
    assert r['status']=='eligible'
    assert max(metrics.principal_angles(r['context'],p['reference'].T))<1e-6
    np.testing.assert_allclose(r['transform'][:6,6:],0,atol=0)
    n=np.block([[p['ncc'],p['nct']],[p['nct'].T,p['ntt']]])
    np.testing.assert_allclose(r['transform']@n@r['transform'].T,np.eye(12),atol=1e-8)


def test_oracle_annihilation_and_singular_joint_rejection():
    ac=np.array([[1.],[0.]]); at=.25*ac
    nc=np.diag([2.,1.]); nt=np.diag([1.25,1.]); cross=np.diag([.5,0.])
    r=linear.oracle(nc+ac@ac.T,nt+at@at.T,cross+ac@at.T,nc,nt,cross,ac,at,1)
    assert r['status']=='oracle_rank_ineligible' and r['context'] is None
    with pytest.raises(ValueError):
        linear.oracle(np.eye(2),np.eye(2),np.eye(2),np.eye(2),np.eye(2),np.eye(2),ac,at,1)


def test_small_fixture_covariance_within_two_percent():
    m=ExactModel(d=4,k=2,seed=56); sample=m.sample(200000,.5); p=sample['population']
    observed=np.cov(np.c_[sample['context'],sample['target']].T)
    expected=np.block([[p['scc'],p['sct']],[p['sct'].T,p['stt']]])
    assert np.linalg.norm(observed-expected)/np.linalg.norm(expected)<.02
    noise=np.cov(np.c_[sample['noise_context'],sample['noise_target']].T)
    expected_n=np.block([[p['ncc'],p['nct']],[p['nct'].T,p['ntt']]])
    assert np.linalg.norm(noise-expected_n)/np.linalg.norm(expected_n)<.02
    assert np.linalg.norm(np.cov(sample['calibration_noise_context'].T)-p['ncc'])/np.linalg.norm(p['ncc'])<.02


def test_exactmodel_draw_order_matches_independent_toy_transcription():
    # Small bounded fixture preserves every original toy draw in both gammas.
    rng=np.random.default_rng(0); d,k,n=5,2,31
    ac=rng.normal(size=(d,k)); at=rng.normal(size=(d,k)); l=rng.normal(size=(d,d))/np.sqrt(d)
    chol=np.linalg.cholesky(l@l.T+.3*np.eye(d)); bc=rng.normal(size=(d,1)); bt=rng.normal(size=(d,1))
    m=ExactModel(d=d,k=k,share='both')
    for gamma in [0.,1.]:
        theta=rng.normal(size=(n,k))*.4; eta=rng.normal(size=(n,1))*3
        nc=rng.normal(size=(n,d))@chol.T; nt=rng.normal(size=(n,d))@chol.T
        xc=theta@ac.T+gamma*eta@bc.T+nc; xt=theta@at.T+gamma*eta@bt.T+nt
        calibration=gamma*rng.normal(size=(n,1))*3@bc.T+rng.normal(size=(n,d))@chol.T
        s=m.sample(n,gamma)
        np.testing.assert_allclose(s['context'],xc,atol=1e-14)
        np.testing.assert_allclose(s['target'],xt,atol=1e-14)
        np.testing.assert_array_equal(s['calibration_noise_context'],calibration)
        assert s['rng_state']==rng.bit_generator.state


def test_masks_namespaces_and_pairing():
    for name in ['time','channel','interleaved','block']:
        s=raw_split(name)
        assert len(np.unique(np.r_[s.context,s.target,s.unused]))==2048
        assert len(s.context)+len(s.target)+len(s.unused)==2048
        assert s.digest==raw_split(name).digest
    s=raw_split('time'); raw=np.arange(2048).reshape(8,256)
    np.testing.assert_array_equal(s.context,raw[:,:96].ravel())
    np.testing.assert_array_equal(s.target,raw[:,160:].ravel())
    a=namespace_rng(0,'development','mid',100).normal(size=100)
    np.testing.assert_array_equal(a,namespace_rng(0,'development','mid',100).normal(size=100))
    assert not np.array_equal(a,namespace_rng(0,'evaluation','mid',0).normal(size=100))


def test_independent_nuisance_and_constant_target():
    rng=np.random.default_rng(22); train=rng.normal(size=(10000,3)); test=rng.normal(size=(10000,3))
    y=rng.normal(size=(10000,1)); yt=rng.normal(size=(10000,1))
    coef=np.linalg.lstsq(train,y,rcond=None)[0]
    r=metrics.nuisance_r2(np.c_[yt,np.ones(10000)],np.c_[test@coef,np.ones(10000)])
    assert abs(r['per_target'][0])<.02
    assert not r['valid'][1] and np.isnan(r['per_target'][1])


def test_fisher_of_and_step_stability():
    j=np.array([[1.,.2],[2.,-.3],[.5,1.]])
    n=np.diag([1.,2.,3.]); info=crlb.fisher(j,n)
    np.testing.assert_allclose(crlb.crlb(j,n)@info,np.eye(2),atol=1e-8)
    assert abs(crlb.of_amplitude(j[:,0],n)['variance']*info[0,0]-1)<.01
    with pytest.raises(ValueError): crlb.crlb(np.c_[j[:,0],j[:,0]],n)
    fun=lambda t:np.array([t[0]**2,np.sin(t[1]),t[0]*t[1]])
    a=crlb.central_jacobian(fun,[1.,.2],[2.,1.]); b=crlb.central_jacobian(fun,[1.,.2],[2.,1.],half=True)
    np.testing.assert_allclose(a,b,atol=1e-8)


def test_bad_support_rejected_and_fixed_ridge():
    with pytest.raises(ValueError): linear.invsqrt(np.diag([1.,0.]))
    rng=np.random.default_rng(1); x=rng.normal(size=(20,3)); y=rng.normal(size=(20,4))
    cc,tt,ct=linear.sample_covariances(x,y)
    rcc,rtt,rct=linear.sample_covariances(x,y,1e-6)
    np.testing.assert_allclose(rcc-cc,np.eye(3)*1e-6*np.trace(cc)/3,atol=1e-15)
    np.testing.assert_array_equal(ct,rct)
