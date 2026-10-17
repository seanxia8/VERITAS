import importlib.util
from pathlib import Path
import numpy as np
import pytest
from noise_module_lucid.trainability_gates import load_gates,constant_predictor,evaluate,DEFAULT

ROOT=Path(__file__).resolve().parents[3]


DE='deposited_energy_in_volume'

def rows(vertex_error,groups,dir_deg=0.):
    """Truth vertex +-1 m on x; prediction offset on y; constant vertex 0 (RMSE 1 m everywhere)."""
    n=len(groups);t=np.zeros((n,7));t[:,2]=1;t[:,3]=np.where(np.arange(n)%2,1.,-1.);t[:,6]=np.log(2)
    p=t.copy();p[:,4]=vertex_error;a=np.radians(dir_deg);p[:,:3]=[np.sin(a),0,np.cos(a)]
    c=np.zeros(7);c[2]=1;return p,t,c


def test_config_declares_runbook_thresholds():
    g=load_gates()
    assert DEFAULT.name=='trainability_gates.json' and g['source']=='runbook v2 §4 R-3' and g['status']=='development'
    assert (g['vertex_rmse_ratio_max_pooled'],g['vertex_rmse_ratio_max_subgroup'],g['direction_mean_deg_max'],g['energy_relative_mae_ratio_max_pooled'])==(.8,1.,20,.8)


def test_vertex_ratio_exactly_point_eight_passes_and_above_fails():
    g=load_gates();s=np.repeat([500,1000],2)
    p,t,c=rows(np.full(4,.8),s);r=evaluate(p,t,c,s,g,energy_endpoint=DE)
    assert r['table']['pooled']['vertex_rmse_m']['ratio']==.8 and r['gates']['vertex_pooled']['passed'] and r['passed']
    p,t,c=rows(np.full(4,np.nextafter(.8,1)*1.0000001),s);r=evaluate(p,t,c,s,g,energy_endpoint=DE)
    assert not r['gates']['vertex_pooled']['passed'] and not r['passed']


def test_one_subgroup_above_constant_fails_while_pooled_passes():
    g=load_gates();s=np.repeat([500,1000],2);p,t,c=rows(np.r_[.4,.4,1.05,1.05],s);r=evaluate(p,t,c,s,g,energy_endpoint=DE)
    assert r['gates']['vertex_pooled']['passed'] and r['gates']['vertex_subgroup_500']['passed']
    assert not r['gates']['vertex_subgroup_1000']['passed'] and not r['passed']


def test_direction_gate_is_strict():
    s=np.repeat([500,1000],2);p,t,c=rows(np.zeros(4),s,dir_deg=20.);g=dict(load_gates())
    deg=evaluate(p,t,c,s,g,energy_endpoint=DE)['table']['pooled']['direction_mean_deg']['model'];assert deg==pytest.approx(20,abs=1e-9)
    g['direction_mean_deg_max']=deg;r=evaluate(p,t,c,s,g,energy_endpoint=DE);assert not r['gates']['direction_pooled']['passed'] and not r['passed']
    g['direction_mean_deg_max']=np.nextafter(deg,np.inf);assert evaluate(p,t,c,s,g,energy_endpoint=DE)['passed']


def test_undefined_denominator_and_empty_subgroup_fail_with_reason():
    g=load_gates();s=np.repeat([500,1000],2);p,t,c=rows(np.zeros(4),s);c[6]=t[0,6]
    r=evaluate(p,t,c,s,g,energy_endpoint=DE);assert not r['passed'] and 'undefined denominator' in r['gates']['energy_pooled']['reason']
    p,t,c=rows(np.zeros(4),s);r=evaluate(p,t,c,s,g,energy_endpoint=DE,expected_subgroups=[500,1000,1500])
    assert not r['gates']['vertex_subgroup_1500']['passed'] and 'empty subgroup' in r['gates']['vertex_subgroup_1500']['reason']


def test_energy_endpoint_is_generic_and_named():
    g=load_gates();s=np.repeat([500,1000],2);p,t,c=rows(np.zeros(4),s)
    p=np.column_stack((p,np.full(4,np.log(1.5))));t=np.column_stack((t,np.full(4,np.log(1.5))));c=np.r_[c,0.]
    r=evaluate(p,t,c,s,g,energy_column=7,energy_endpoint=DE);assert r['energy_endpoint']==DE and r['table']['pooled']['energy_relative_mae']['model']==0 and r['passed']


def test_constant_uses_train_rows_only():
    rng=np.random.default_rng(1);y=rng.normal(size=(20,7));train=np.arange(10);before=constant_predictor(y,train)
    y[10:]+=100;np.testing.assert_array_equal(constant_predictor(y,train),before)


def test_past_run_constant_metrics_reproduce():
    path=ROOT/'results/pmt_multienergy_dev_2026-10-16/constant_baseline_table.py'
    if not path.exists():pytest.skip('results/ not shipped (remote code bundle)')
    spec=importlib.util.spec_from_file_location('constant_baseline_table',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rep=mod.build()
    for k in ('vertex_rmse_m','direction_mean_deg','energy_relative_mae','vertex_mse','direction_mse','log_energy_mse'):
        v=rep['reconstruction_check'][k];assert v['reconstructed']==pytest.approx(v['recorded'],rel=1e-5)
    assert rep['reconstruction_check']['vertex_rmse_m']['recorded']==pytest.approx(.36972,abs=1e-5)
    assert all(not s['recorded_passed'] and not s['recorded_gates']['energy_absolute'] for s in rep['seeds'])


def test_energy_gate_fails_closed_off_the_de_endpoint():
    g=load_gates();s=np.repeat([500,1000],2);p,t,c=rows(np.zeros(4),s)
    assert evaluate(p,t,c,s,g,energy_endpoint=DE)['passed']
    r=evaluate(p,t,c,s,g)
    assert not r['passed'] and 'not the required D-E endpoint' in r['gates']['energy_pooled']['reason']
