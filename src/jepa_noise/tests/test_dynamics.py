"""Bounded E1c A2 engineering identities; no campaign outcomes."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from jepa_noise.dynamics import (empirical_moments, diagonal_fixture, initialize,
                                encoder, loss_and_gradients, mode_contributions, run)


def test_empirical_moments_divisor_n_and_direct_sample_loss():
    rng=np.random.default_rng(719)
    c=rng.normal(size=(11,4))+3
    t=.4*c+rng.normal(size=(11,4))-2
    m=empirical_moments(c,t)
    cc,tt=c-c.mean(0),t-t.mean(0)
    assert_allclose(m.cc,cc.T@cc/11,atol=1e-13)
    assert_allclose(m.ct,cc.T@tt/11,atol=1e-13)
    layers=[rng.normal(size=(4,4))*.2 for _ in range(2)]
    p=rng.normal(size=(4,4))*.3
    w=encoder(layers)
    target=w.copy()
    loss,gl,gp=loss_and_gradients(layers,p,m,fixed_target=target)
    residual=cc@(p@w).T-tt@target.T
    assert_allclose(loss,.5*np.mean(np.sum(residual**2,axis=1)),atol=1e-13)
    assert_allclose(gp,residual.T@(cc@w.T)/11,atol=1e-13)


@pytest.mark.parametrize('depth',[1,2,3])
def test_finite_difference_stop_gradient_target_held_fixed(depth):
    rng=np.random.default_rng(62+depth)
    layers=[rng.normal(size=(4,4))*.3 for _ in range(depth)]
    p=rng.normal(size=(4,4))*.4
    m=diagonal_fixture()
    target=encoder(layers).copy()
    _,gradients,gp=loss_and_gradients(layers,p,m,fixed_target=target)
    h=1e-6
    for parameter,gradient in [*zip(layers,gradients),(p,gp)]:
        numerical=np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            saved=parameter[index]
            parameter[index]=saved+h
            plus=loss_and_gradients(layers,p,m,fixed_target=target)[0]
            parameter[index]=saved-h
            minus=loss_and_gradients(layers,p,m,fixed_target=target)[0]
            parameter[index]=saved
            numerical[index]=(plus-minus)/(2*h)
        assert_allclose(gradient,numerical,atol=1e-8,rtol=1e-6)
    # Re-evaluating the target under perturbations differentiates a different loss.
    original=layers[0][0,0]
    layers[0][0,0]=original+h
    plus=loss_and_gradients(layers,p,m)[0]
    layers[0][0,0]=original-h
    minus=loss_and_gradients(layers,p,m)[0]
    layers[0][0,0]=original
    assert abs((plus-minus)/(2*h)-gradients[0][0,0])>1e-5


def test_init_is_paired_across_depth_and_independent_by_layer_predictor():
    sets=[initialize(3,depth) for depth in (1,2,3)]
    assert np.array_equal(sets[0][0][0],sets[2][0][0])
    assert np.array_equal(sets[1][0][1],sets[2][0][1])
    assert np.array_equal(sets[0][1],sets[2][1])
    assert not np.array_equal(sets[2][0][0],sets[2][0][1])
    assert not np.array_equal(sets[2][0][0],sets[2][1])
    assert not np.array_equal(sets[2][0][0],initialize(4,3)[0][0])


def test_diagonal_fixture_and_simultaneous_update_preserve_coordinate_decoupling():
    m=diagonal_fixture()
    layers=[np.diag([.001,.002,.003,.004]) for _ in range(2)]
    p=np.diag([.002,.001,.004,.003])
    ac=np.eye(4)[:,:3];bc=np.eye(4)[:,3:]
    loss,gl,gp=loss_and_gradients(layers,p,m)
    result=run(m,layers,p,ac,bc,np.ones(3),1.,budget_seconds=1.,engineering_steps=1)
    assert result['status']=='engineering_incomplete'
    assert result['completed_steps']==1
    assert result['R5']['status']=='not_evaluable'
    for actual,old,gradient in zip(result['layers'],layers,gl):
        assert_allclose(actual,old-.001*gradient,atol=1e-15)
        assert_allclose(actual,np.diag(np.diag(actual)),atol=1e-15)
    assert_allclose(result['predictor'],p-.001*gp,atol=1e-15)
    assert_allclose(result['absolute'][0],np.r_[np.linalg.norm(encoder(layers)@ac,axis=0),
                                             np.linalg.norm(encoder(layers)@bc)])
    assert result['absolute'].shape==(2,4)


def test_modes_use_separate_true_loadings_and_scaled_absolute_contribution():
    layers=[np.diag([2.,3.,4.,5.])]
    ac=np.eye(4)[:,:3];bc=np.ones((4,1))
    q,normalized,norm=mode_contributions(layers,ac,bc,np.array([.1,.2,.3]),2.)
    assert_allclose(q[:3],[.2,.6,1.2])
    assert_allclose(q[3],2*np.sqrt(54))
    smaller=mode_contributions([layers[0]*.01],ac,bc,np.array([.1,.2,.3]),2.)
    assert_allclose(smaller[0],q*.01)
    assert_allclose(smaller[1],normalized)


def test_resource_and_numerical_stops_are_preserved_without_learning():
    m=diagonal_fixture();layers,p=initialize(4,1,d=4)
    ac=np.eye(4)[:,:3];bc=np.eye(4)[:,3:]
    stopped=run(m,layers,p,ac,bc,np.ones(3),1.,budget_seconds=1e-300,engineering_steps=3)
    assert stopped['status']=='stopped' and stopped['reason']=='resource_cap'
    assert stopped['completed_steps']==0 and stopped['R5']['status']=='not_evaluable'
    failed=run(m,[np.ones((4,4))*1e250],p,ac,bc,np.ones(3),1.,budget_seconds=1.,engineering_steps=3)
    assert failed['status']=='failed' and failed['reason']=='numerical_instability'
    assert failed['R5']['status']=='not_evaluable'
    with pytest.raises(ValueError,match='campaign requires'):
        run(m,layers,p,ac,bc,np.ones(3),1.,budget_seconds=1.)
