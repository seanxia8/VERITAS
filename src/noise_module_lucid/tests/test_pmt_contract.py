"""Engineering acceptance: all stochastic tolerances declared before first run.

Seeds: 4 for rate/identity, 71 for timing. Rate tolerance = 6 binomial
standard deviations; Gaussian moments = 6 SE; width relative tolerance 3%.
These are synthetic validation thresholds, never detector harm thresholds.
"""
from dataclasses import replace
import json
import numpy as np
import pytest
from noise_module_lucid.photon_contract import PhotonBatch, operational_digits, monitor_features
from noise_module_lucid.quantum_efficiency import QECurve, keyed_uniform
from noise_module_lucid.timing_response import WalkCurve, WidthCurve
from noise_module_lucid.readout import ReadoutParams, readout_chain
from noise_module_lucid.group_response import GroupMap, group_intervention, validate_cell
from noise_module_lucid.dataset import ResponseCellSpec, run_response_matrix
from noise_module_lucid.calibration import fit_reference_calibration, ReferenceCalibration
from noise_module_lucid.consequence import cell_consequence, aggregate_endpoints, FitConfig, fit_vertex
from noise_module_lucid.toy import cylinder_detector, prompt_source_event
from noise_module_lucid.lucid_adapter import incident_record


def batch(n=10000, wavelength=400, q=1):
    return PhotonBatch(71, np.arange(n*q), np.repeat(np.arange(n), q), np.zeros(n*q), np.full(n*q, wavelength))


def read(b, p, seed=4):
    return readout_chain(b.sensor_id, b.arrival_time_ns, int(max(b.sensor_id, default=-1))+1, p, seed, photon_batch=b)


def reference():
    curve = WalkCurve((2, -1), (.25, 100))
    return ReadoutParams(qe_curve=QECurve((300,400,600),(.25,.5,.25)), walk_curve=curve,
        reference_walk=curve, timing_width=WidthCurve(), spe_sigma=0)


def test_qe_domain_interpolation_and_drift():
    c = reference().qe_curve
    np.testing.assert_allclose(c([300,350,400,500,600]), [.25,.375,.5,.375,.25])
    assert c.drift(0, [1,0,-1]) == c
    assert c.integrated_probability([300,400,600],[1,2,1],.5) == .1875
    for w in ([299], [601], [np.nan]):
        with pytest.raises(ValueError): c(w)
    with pytest.raises(ValueError): c.drift(4, [1,1,1])
    with pytest.raises(ValueError): QECurve((400,300,600),(.2,.3,.4))


@pytest.mark.parametrize("w,p", [(300,.25),(350,.375),(400,.5),(500,.375),(600,.25)])
def test_analytic_rate(w,p):
    b = batch(30000,w)
    d = read(b, reference())
    assert abs(d.counts["photons_detected"]-30000*p) <= 6*np.sqrt(30000*p*(1-p))


def test_exact_degeneracy_and_persistent_pairing():
    b = batch()
    a = reference()
    c = replace(a, qe_curve=QECurve((300,400,600),(.5,1,.5)), collection_efficiency=.5)
    da, dc = read(b,a), read(b,c)
    for k in ("sensor","time","charge"): np.testing.assert_array_equal(getattr(da,k),getattr(dc,k))
    ids = np.array([9,2,7])
    u = keyed_uniform(4,71,ids,"detection")
    np.testing.assert_array_equal(u[[2,0]],keyed_uniform(4,71,ids[[2,0]],"detection"))
    keep = np.arange(len(b.sensor_id)) % 2 == 0
    small = replace(b, **{k:getattr(b,k)[keep] for k in ("photon_id","sensor_id","arrival_time_ns","wavelength_nm")})
    ds = read(small,a)
    np.testing.assert_array_equal(ds.sensor, da.sensor[da.sensor%2==0])
    np.testing.assert_array_equal(ds.charge, da.charge[da.sensor%2==0])
    np.testing.assert_array_equal(ds.time, da.time[da.sensor%2==0])


def test_timing_sign_identity_width_and_membership():
    b = batch(20000,q=2)
    p = replace(reference(), qe_curve=QECurve((300,400,600),(1,1,1)))
    a = read(b,p,71)
    identity = read(b,replace(p,walk_curve=WalkCurve(p.walk_curve.coefficients_ns,p.walk_curve.domain_pe)),71)
    np.testing.assert_array_equal(a.time,identity.time)
    moved = replace(p,walk_curve=WalkCurve((4,-1),(.25,100)))
    d = read(b,moved,71)
    np.testing.assert_array_equal(a.charge,d.charge)
    np.testing.assert_array_equal(a.sensor,d.sensor)
    np.testing.assert_allclose(d.time-a.time,2,atol=1e-14)
    sigma = float(p.timing_width(2))
    assert abs(a.time.mean()) < 6*sigma/np.sqrt(len(a))
    assert a.time.std() == pytest.approx(sigma,rel=.03)
    assert sigma > 0 and d.counts["lost_time_window"] == 0
    wide = read(b,replace(p,timing_width_factor=2),71)
    np.testing.assert_allclose(wide.time,2*a.time)
    assert p.timing_width(.5) > p.timing_width(5)


def test_polynomial_reference_fit_and_bounds():
    q=np.linspace(.25,10,30); u=(q-.25)/9.75
    f=WalkCurve.fit(q,3-2*u+u*u,degree=2,domain_pe=(.25,10))
    np.testing.assert_allclose(f(q),3-2*u+u*u,atol=1e-13)
    np.testing.assert_allclose(f([.25,10]),[3,2])
    with pytest.raises(ValueError): f([10.01])
    with pytest.raises(ValueError): WidthCurve()([100.01])


def test_group_coupling_controls_and_one_field_rule():
    b=batch(80,q=2); p=replace(reference(),qe_curve=QECurve((300,400,600),(1,1,1)))
    mapping=GroupMap(tuple(np.arange(80)%2),"synthetic_dev","test alternating groups")
    coupled=group_intervention(p,mapping,[1],gain_factor=.5,width_factor=2)
    width=group_intervention(p,mapping,[1],width_factor=2)
    gain=group_intervention(p,mapping,[1],gain_factor=.5)
    a,d,w,g=[read(b,x) for x in (p,coupled,width,gain)]
    np.testing.assert_array_equal(d.time[::2],a.time[::2])
    np.testing.assert_array_equal(d.charge[::2],a.charge[::2])
    np.testing.assert_allclose(d.charge[1::2],a.charge[1::2]/2)
    np.testing.assert_array_equal(w.charge,a.charge)
    np.testing.assert_allclose(w.time[1::2],2*a.time[1::2])
    # Gain recomputes the timing width at measured Q, even in gain-only control.
    np.testing.assert_allclose(g.time[1::2],a.time[1::2]*p.timing_width(1)/p.timing_width(2))
    c=ResponseCellSpec("coupled","group_response",1,coupled,affected_groups=(1,),changed_fields=("gain","timing_width_factor"),causal_control="proxy-strength")
    validate_cell(c,p)
    with pytest.raises(ValueError): validate_cell(replace(c,causal_control=None),p)
    low=read(b,group_intervention(p,mapping,[1],gain_factor=.1))
    assert np.all(low.sensor%2==0) and low.counts["lost_threshold"] == 40


@pytest.mark.parametrize("change", [dict(units=("mm","ns","nm")),dict(photon_id=np.zeros(10,dtype=int)),dict(applied_effects=("QE",)),dict(wavelength_nm=None)])
def test_bad_photon_contract(change):
    with pytest.raises(ValueError): read(replace(batch(10),**change),reference())


def test_double_qe_tts_and_observation_firewall():
    b=batch(20);p=reference()
    with pytest.raises(ValueError): read(replace(b,stage="detected_pe",applied_effects=("QE","CE")),p)
    with pytest.raises(ValueError): read(replace(b,applied_effects=("TTS",)),p)
    d=read(b,p); positions=np.zeros((20,3)); groups=np.arange(20)%2
    o=operational_digits(d,positions,groups); f=monitor_features(o)
    d.n_pe_true[:]=999999; d.counts["secret"]=1
    np.testing.assert_array_equal(f,monitor_features(operational_digits(d,positions,groups)))
    assert set(vars(o)) == {"sensor","charge","time","positions","group_id"}
    with pytest.raises(TypeError): monitor_features(b)
    with pytest.raises(TypeError): monitor_features(d)


def test_adapter_requires_actual_fields():
    b=batch(10)
    record={**vars(b),"upstream_revision":"eaf2e89","prompt_direct":True,"scattering":False,"reflection":False,"angular_acceptance":"uniform","concentrator":False,"absorption":"off","dispersion":"off"}
    assert incident_record(record,n_channels=10).stage == "incident_photons"
    for key in ("photon_id","wavelength_nm","applied_effects","units"):
        with pytest.raises(ValueError): incident_record({k:v for k,v in record.items() if k!=key},n_channels=10)
    with pytest.raises(ValueError): incident_record({**record,"reflection":True},n_channels=10)


def test_calibration_disjoint_hash_binding_and_endpoint_failure(tmp_path):
    det=cylinder_detector(4,12,2);p=reference()
    fit=[prompt_source_event(det,1000+i,n_photons=2000) for i in range(6)]
    ev=[prompt_source_event(det,2000+i,n_photons=2000) for i in range(4)]
    cal=fit_reference_calibration(fit,det.positions,p,fit_config=FitConfig(grid=5));path=tmp_path/"cal.json";cal.save(path)
    assert ReferenceCalibration.load(path).sha256 == cal.sha256
    cal.validate_evaluation(ev,det.positions,p,FitConfig(grid=5))
    with pytest.raises(ValueError): cal.validate_evaluation(fit,det.positions,p,FitConfig(grid=5))
    with pytest.raises(ValueError): cal.validate_evaluation(ev,det.positions,replace(p,gain=2))
    z=json.loads(path.read_text());z["payload"]["energy_scale"]*=2;path.write_text(json.dumps(z))
    with pytest.raises(ValueError): ReferenceCalibration.load(path)
    cells=[ResponseCellSpec("reference","reference",None,p,origin="clean"),ResponseCellSpec("dead","module_loss",1,replace(p,dead_channels=tuple(range(det.n_channels))))]
    s=run_response_matrix(cells,ev,det.positions,n_boot=0,calibration=cal,fit_config=FitConfig(grid=5))
    assert s["non_citable"] and s["evidence_stage"] == "development"
    dead=s["cells"][1]
    assert np.isnan(dead["K"]) and dead["harmful_dev"] is None
    assert dead["K_vertex"]["failure_rate_cell"] == 1
    with pytest.raises(ValueError): run_response_matrix(cells,ev,det.positions)
    with pytest.raises(ValueError): run_response_matrix(cells,[replace(ev[0],endpoint_kind="track")],det.positions,calibration=cal)
    assert np.isnan(aggregate_endpoints(1,np.nan))
    result=cell_consequence([2,np.nan],[1,1])
    assert np.isnan(result["K"]) and result["K_complete_case"] == 2
    assert result["status"] == "unevaluable_fit_failure"


def test_point_source_exact_tof_closure():
    det=cylinder_detector(4,12,2);v=np.array([.2,-.3,.4]);t=10+np.linalg.norm(det.positions-v,axis=1)/FitConfig().v_group
    f=fit_vertex(det.positions,np.arange(det.n_channels),t,np.ones(det.n_channels),FitConfig(grid=5))
    assert f.ok and np.linalg.norm(f.vertex-v)<.002


def test_per_channel_walk_and_frozen_reference():
    from noise_module_lucid.timing_response import ChannelWalkCurve
    b=batch(2,q=2);p=replace(reference(),qe_curve=QECurve((300,400,600),(1,1,1)))
    curves=ChannelWalkCurve((p.walk_curve,WalkCurve((4,-1),(.25,100))))
    a=read(b,p);d=read(b,replace(p,walk_curve=curves))
    np.testing.assert_allclose(d.time-a.time,[0,2])
    with pytest.raises(ValueError): validate_cell(ResponseCellSpec("bad","fixed_tq",1,replace(p,threshold_pe=.5)),p)
    with pytest.raises(ValueError): validate_cell(ResponseCellSpec("bad","other",1,replace(p,reference_walk=WalkCurve())),p)


def test_polynomial_discriminator_uses_declared_pe_peak():
    p=replace(reference(),qe_curve=QECurve((300,400,600),(1,1,1)),gain=.245)
    assert len(read(batch(10),p)) == 0
    assert len(read(batch(10),replace(p,gain=.25))) == 10


@pytest.mark.parametrize("charge",[1,2,5])
def test_charge_conditioned_gaussian_widths(charge):
    p=replace(reference(),qe_curve=QECurve((300,400,600),(1,1,1)))
    d=read(batch(10000,q=charge),p,71)
    sigma=float(p.timing_width(charge))
    assert abs(d.time.mean()) < 6*sigma/np.sqrt(len(d))
    assert d.time.std() == pytest.approx(sigma,rel=.03)


def test_lucid_name_does_not_make_development_citable():
    det=cylinder_detector(4,12,2);p=reference()
    fit=[prompt_source_event(det,800+i,n_photons=1000) for i in range(4)]
    ev=[prompt_source_event(det,900+i,n_photons=1000) for i in range(3)]
    cal=fit_reference_calibration(fit,det.positions,p)
    s=run_response_matrix([ResponseCellSpec("reference","reference",None,p)],ev,det.positions,source="lucid",n_boot=0,calibration=cal)
    assert s["non_citable"] is True and s["evidence_stage"] == "development"
    assert s["protocol_freeze_id"] is None


def test_duplicate_dead_channels_cannot_double_count_correction():
    with pytest.raises(ValueError, match="dead channels"):
        read(batch(10),replace(reference(),dead_channels=(1,1)))
