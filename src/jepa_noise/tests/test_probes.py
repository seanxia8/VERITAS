"""Bounded synthetic ridge fixtures; no E1 cohort defaults or outcomes."""
import numpy as np
import pytest
from jepa_noise.probes import ALPHAS, fit_ridge, select_alpha, grouped_ridge, evaluate


def fixture():
    rng=np.random.default_rng(18)
    x=rng.normal(size=(25,3)); y=np.c_[2*x[:,0]+3, -x[:,1]+.2*rng.normal(size=25)]
    groups=np.repeat(np.arange(5),5)
    return x,y,groups


def test_direct_ridge_reference_and_explicit_intercept():
    x,y,_=fixture()
    for intercept in [False,True]:
        f=fit_ridge(x,y[:,0],alpha=.01,fit_intercept=intercept)
        z=(x-x.mean(0))/x.std(0)
        offset=y[:,0].mean() if intercept else 0.
        expected=np.linalg.solve(z.T@z+.01*np.eye(3),z.T@(y[:,0]-offset))
        np.testing.assert_allclose(f.coefficients,expected)
        assert f.intercept==offset
    with pytest.raises(TypeError): grouped_ridge(x,y,np.repeat(np.arange(5),5))


def test_exact_tie_favors_larger_alpha_only():
    assert select_alpha({alpha:1. for alpha in ALPHAS})==100.
    scores={alpha:1. for alpha in ALPHAS}; scores[1e-6]=np.nextafter(1.,0.)
    assert select_alpha(scores)==1e-6


def test_heldout_perturbation_cannot_change_its_training_fits():
    x,y,g=fixture(); original=grouped_ridge(x,y,g,fit_intercept=True)
    changed_x=x.copy(); changed_y=y.copy()
    changed_x[g==0]+=10000; changed_y[g==0]-=20000
    changed=grouped_ridge(changed_x,changed_y,g,fit_intercept=True)
    for j in range(2):
        a,b=original[j]['folds'][0],changed[j]['folds'][0]
        assert a['held_group']==b['held_group']==0
        for alpha in ALPHAS:
            fa,fb=a['fits'][alpha],b['fits'][alpha]
            np.testing.assert_array_equal(fa.mean,fb.mean)
            np.testing.assert_array_equal(fa.scale,fb.scale)
            np.testing.assert_array_equal(fa.coefficients,fb.coefficients)
            assert fa.intercept==fb.intercept
        assert original[j]['validation_mse'] != changed[j]['validation_mse']


def test_test_perturbation_cannot_modify_final_fit_or_alpha():
    x,y,g=fixture(); fitted=grouped_ridge(x,y,g,fit_intercept=True)
    snapshots=[(p['alpha'],p['fit'].mean.copy(),p['fit'].scale.copy(),p['fit'].coefficients.copy(),p['fit'].intercept) for p in fitted]
    evaluate(fitted,x+100,y-100)
    evaluate(fitted,x-500,y+50)
    for p,s in zip(fitted,snapshots):
        assert p['alpha']==s[0] and p['fit'].intercept==s[4]
        for a,b in zip([p['fit'].mean,p['fit'].scale,p['fit'].coefficients],s[1:4]): np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(p['fit'].mean,x.mean(0))
        np.testing.assert_array_equal(p['fit'].scale,x.std(0))


def test_group_integrity_and_constant_target_status():
    x,y,g=fixture(); y=np.c_[y,np.full(25,.1)]
    result=grouped_ridge(x,y,g,fit_intercept=False)
    assert result[-1]['status']=='not_evaluable' and result[-1]['fit'] is None
    for fold in result[0]['folds']:
        assert not set(g[fold['train_indices']]) & set(g[fold['held_indices']])
    with pytest.raises(ValueError,match='exactly five'):
        grouped_ridge(x,y,np.zeros(25),fit_intercept=False)


def test_feature_constant_and_nonfinite_rejection():
    x,y,g=fixture(); x[:,2]=.1
    result=grouped_ridge(x,y,g,fit_intercept=True)
    assert result[0]['fit'].scale[2]==1.
    assert result[0]['fit'].mean[2]==.1
    assert result[0]['fit'].coefficients[2]==0.
    y[0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        grouped_ridge(x,y,g,fit_intercept=True)


def test_validation_loss_pools_events_without_group_reweighting():
    x,y,_=fixture(); g=np.r_[np.zeros(1),np.ones(2),np.full(3,2),np.full(4,3),np.full(15,4)]
    result=grouped_ridge(x,y,g,fit_intercept=True)
    for output,p in enumerate(result):
        for alpha in ALPHAS:
            total=0.
            for fold in p['folds']:
                ids=fold['held_indices']; pred=fold['fits'][alpha].predict(x[ids])
                total+=np.sum((pred-y[ids,output])**2)
            assert p['validation_mse'][alpha]==pytest.approx(total/25)
