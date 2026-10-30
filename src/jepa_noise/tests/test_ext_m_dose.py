"""Acceptance tests for the unregistered M-dose extension."""
import numpy as np
import pytest

from jepa_noise import lucid_bombs as lb, run, views
from jepa_noise.ext_m_dose import m_terms, pair_diagnostics, project
from jepa_noise.ext_m_dose import f1x_state
from jepa_noise.tests.test_run_e1 import CONFIG


def blocks(s, n):
    return s[:6,:6],s[6:,6:],s[:6,6:],n[:6,:6],n[6:,6:],n[:6,6:]


def test_gaussian_population_estimation():
    rng = np.random.default_rng(481)
    p = np.block([[np.eye(6),np.diag([.65,.55,.45,.35,.25,.15])],
                  [np.diag([.65,.55,.45,.35,.25,.15]),np.eye(6)]])
    n = np.block([[np.eye(6)*.5,np.diag([.15,.1,.08,.05,.03,.01])],
                  [np.diag([.15,.1,.08,.05,.03,.01]),np.eye(6)*.5]])
    ps = rng.multivariate_normal(np.zeros(12), p, size=20000)
    ns = rng.multivariate_normal(np.zeros(12), n, size=20000)
    estimated = pair_diagnostics(*blocks(np.cov(ps+ns,rowvar=False),np.cov(ns,rowvar=False)),3)
    exact = pair_diagnostics(*blocks(p+n,n),3)
    np.testing.assert_allclose(estimated["shares"][0],exact["shares"][0],atol=.02,rtol=0)
    assert abs(estimated["ratio"]-exact["ratio"]) < .02


def test_zero_noise_and_zero_physics():
    i = np.eye(6); cross = np.diag([.6,.5,.4,.3,.2,.1]); zero = np.zeros((6,6))
    p = pair_diagnostics(i,i,cross,i,i,zero,3)
    np.testing.assert_array_equal(p["shares"][0], np.zeros(3))
    assert p["ratio"] == 0
    q = pair_diagnostics(i,i,cross,i,i,cross,3)
    np.testing.assert_allclose(q["shares"][0],np.ones(3),atol=1e-12)


def test_pairing_identity_and_no_linear_cca(monkeypatch):
    from jepa_noise import linear
    monkeypatch.setattr(linear, "cca", lambda *a,**k: pytest.fail("linear.cca called"))
    i=np.eye(6); c=np.diag([.9,.8,.7,.6,.5,.4])
    x=pair_diagnostics(i,i,c,i,i,c/3,3)
    np.testing.assert_allclose(x["shares"][0],1/3,atol=1e-12)


def test_f1x_gain_and_registered_state_unchanged():
    geo=lb.from_synthetic(); rid="ext_m_dose/evt/0001"; seed=13
    before=lb.state_for_run("F1",.3,rid,geo,seed=seed)
    x=f1x_state(geo,1.5,rid,seed=seed)
    u=lb.keyed_rng(seed,lb.NS_F1,rid).standard_normal(geo.n_modules)
    np.testing.assert_array_equal(x.module_gain,np.exp(1.5*u))
    np.testing.assert_array_equal(x.gains,x.module_gain[geo.module_of_pmt]*x.pmt_gain)
    after=lb.state_for_run("F1",.3,rid,geo,seed=seed)
    assert before.digest == after.digest


def test_m_terms_registered_e1b_development():
    cfg=run.load_config(CONFIG)[0]; q=cfg["E1b"]["M_diagnostic"]
    gen=run.make_tier1(cfg,q["candidate_order"][0],q["common_strength"],
                       q["line_multiplier"],q["development_seed"],"E1b_M_development")
    ev=gen.sample(np.arange(q["events"],dtype=np.int64),role="E1b_M_event")["raw"].reshape(q["events"],2048)
    noise=np.stack([gen.noise_record(i,role="E1b_M_noise")[0] for i in range(q["noise_records"])]).reshape(q["noise_records"],2048)
    from jepa_noise.masks import raw_split
    idx=raw_split("channel"); d=min(q["projection_dim"],len(idx.context),len(idx.target))
    rng=np.random.default_rng(np.random.SeedSequence(run._seed_entropy(q["projection_seed"],"channel")))
    qc,_=np.linalg.qr(rng.normal(size=(len(idx.context),d)))
    qt,_=np.linalg.qr(rng.normal(size=(len(idx.target),d)))
    def cov(x):
        a=x[:,idx.context]@qc; b=x[:,idx.target]@qt
        z=np.cov(a,b,rowvar=False,ddof=1)
        return z[:d,:d],z[d:,d:],z[:d,d:]
    assert abs(m_terms(*cov(ev),*cov(noise),q["k"])[0]-.013519688568) < 1e-9


@pytest.mark.parametrize("g,nu,ratio,angle,mfull,mreg",[
    (0,0,0,0,.682,.0099),(.005,.043,.530,11.27,.317,-.0025),
    (.02,.525,1.036,47.96,-.024,-.038),(.05,.918,1.281,76.95,-.186,-.102)])
def test_toy_acceptance(g,nu,ratio,angle,mfull,mreg):
    v=views.Tier1Views(split="channel",line_multiplier=0,common_strength=g)
    p=v.population(); b=tuple(p[k] for k in ("scc","stt","sct","ncc","ntt","nct"))
    r=pair_diagnostics(*b,3)
    assert abs(r["shares"][0][2]-nu) < (.000001 if g==0 else .005)
    assert abs(r["ratio"]-ratio)<.005
    assert abs(v.displacement(3)["angle_deg"]-angle)<.5
    assert abs(r["M_full"]-mfull)<.005
    assert abs(m_terms(*project(b,"channel"),3)[0]-mreg)<.005
