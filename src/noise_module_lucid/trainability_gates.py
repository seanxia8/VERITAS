"""Constant-relative, per-subgroup trainability gates (runbook v2 §4 R-3)."""
import json
from pathlib import Path
import numpy as np
from .multienergy_baseline import metrics

DEFAULT=Path(__file__).resolve().parents[2]/'configs/lucid_pmt/trainability_gates.json'
KEYS=('vertex_rmse_m','direction_mean_deg','energy_relative_mae')


def load_gates(path=DEFAULT):
    g=json.loads(Path(path).read_text())
    if g.get('source')!='runbook v2 §4 R-3':raise ValueError('undeclared gate source')
    return g


def constant_predictor(y,train_index):
    """Pooled training-split mean; never a subgroup-specific constant."""
    return np.asarray(y)[np.asarray(train_index)].mean(0)


def _ratio(model,constant):
    if not (np.isfinite(model) and np.isfinite(constant)):return None,'non-finite metric'
    if constant<=0:return None,'undefined denominator: constant metric is zero'
    return float(model/constant),None


def _gate(value,limit,strict,reason,label):
    if reason:return dict(passed=False,value=value,limit=limit,reason=f'{label}: {reason}')
    ok=bool(value<limit) if strict else bool(value<=limit)
    return dict(passed=ok,value=value,limit=limit,reason=None if ok else f'{label}: {value:.6g} {"not <" if strict else ">"} {limit}')


def compare(p,t,c):
    """Model, constant and model/constant ratio for each gated metric."""
    m=metrics(p,t);k=metrics(np.tile(c,(len(t),1)),t);row=dict(n=m['n'])
    for name in KEYS:r,why=_ratio(m[name],k[name]);row[name]=dict(model=m[name],constant=k[name],ratio=r,undefined=why)
    return row


def evaluate(pred,truth,constant,subgroups,gates,energy_column=6,energy_endpoint='initial_energy',expected_subgroups=None):
    """Model vs constant pooled and per subgroup; energy column must be log(E/1000 MeV)."""
    cols=[0,1,2,3,4,5,energy_column];p=np.asarray(pred)[:,cols];t=np.asarray(truth)[:,cols];c=np.asarray(constant)[cols];s=np.asarray(subgroups)
    labels=sorted(set(s.tolist())|set(expected_subgroups or []),key=str);table=dict(pooled=compare(p,t,c) if len(t) else None,subgroups={})
    for g in labels:ix=s==g;table['subgroups'][str(g)]=compare(p[ix],t[ix],c) if ix.any() else None
    out={}
    def ratio_gate(name,limit,where,row):
        if row is None:return _gate(None,limit,False,'empty subgroup',where)
        e=row[name];return _gate(e['ratio'],limit,False,e['undefined'],where)
    def abs_gate(name,limit,where,row):
        if row is None:return _gate(None,limit,True,'empty subgroup',where)
        v=row[name]['model'];return _gate(v,limit,True,None if np.isfinite(v) else 'non-finite metric',where)
    out['vertex_pooled']=ratio_gate('vertex_rmse_m',gates['vertex_rmse_ratio_max_pooled'],'pooled vertex RMSE / constant',table['pooled'])
    out['direction_pooled']=abs_gate('direction_mean_deg',gates['direction_mean_deg_max'],'pooled direction mean deg',table['pooled'])
    out['energy_pooled']=ratio_gate('energy_relative_mae',gates['energy_relative_mae_ratio_max_pooled'],f'pooled {energy_endpoint} relative MAE / constant',table['pooled'])
    for g,row in table['subgroups'].items():
        out[f'vertex_subgroup_{g}']=ratio_gate('vertex_rmse_m',gates['vertex_rmse_ratio_max_subgroup'],f'subgroup {g} vertex RMSE / constant',row)
        out[f'direction_subgroup_{g}']=abs_gate('direction_mean_deg',gates['direction_mean_deg_max'],f'subgroup {g} direction mean deg',row)
        out[f'energy_subgroup_{g}']=ratio_gate('energy_relative_mae',gates['energy_relative_mae_ratio_max_subgroup'],f'subgroup {g} {energy_endpoint} relative MAE / constant',row)
    required=gates['energy_endpoint_id_required']
    if energy_endpoint!=required:  # fail closed: R-3 gates energy only on the D-E endpoint
        for k in [k for k in out if k.startswith('energy_')]:
            out[k]=dict(out[k],passed=False,reason=f'energy endpoint {energy_endpoint!r} is not the required D-E endpoint {required!r}')
    if not table['subgroups']:out['subgroups_present']=dict(passed=False,value=None,limit=None,reason='no subgroups supplied')
    return dict(energy_endpoint=energy_endpoint,energy_column=int(energy_column),constant=np.asarray(constant).tolist(),table=table,gates=out,passed=all(v['passed'] for v in out.values()),failures=[v['reason'] for v in out.values() if not v['passed']],gate_config=gates)
