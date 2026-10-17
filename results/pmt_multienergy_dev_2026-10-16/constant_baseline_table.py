"""Reconstruct the training-mean constant of training_01 and tabulate model vs constant (runbook v2 R-3).

Run from the repository root: PYTHONPATH=src python -B results/pmt_multienergy_dev_2026-10-16/constant_baseline_table.py
Reads committed files only; never modifies existing results.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from noise_module_lucid.multienergy_baseline import metrics
from noise_module_lucid.trainability_gates import KEYS,compare

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NOTE='Recorded verdict (unchanged): energy_absolute gate failed for all seeds; R-3 gates apply to future runs only.'
ENDPOINT='initial kinetic energy (recorded target, log(E/1000 MeV)); the D-E deposited-energy endpoint was not trained'


def reconstruct_targets(cfg,manifest):
    """Targets in digits.npz row order: source primary direction +z, as multienergy_data.run."""
    roles=[];y=[]
    for rec in manifest['events']:
        e,eid=rec['energy_MeV'],rec['event_id'];role=next(k for k,v in cfg['splits'].items() if eid in v)
        for aug in range(cfg['rotations_per_event']):
            rng=np.random.default_rng(cfg['augmentation_seed']+e*10000+eid*100+aug);rot=Rotation.random(random_state=rng).as_matrix();vertex=rng.uniform(-.35,.35,3)
            roles.append(role);y.append(np.r_[np.array([0.,0,1])@rot.T,vertex,np.log(float(rec['primary_energy'])/1000)])
    return np.array(roles),np.asarray(y,dtype=np.float32)


def build(rtol=1e-5):
    cfg=json.loads((ROOT/'configs/lucid_pmt/multienergy_pilot.json').read_text());training=json.loads((HERE/'training_01/training.json').read_text())
    if training['config']!=cfg:raise ValueError('config differs from recorded run')
    roles,y=reconstruct_targets(cfg,json.loads((HERE/'source_balanced/manifest.json').read_text()))
    constant=y[roles=='train'].mean(0);test=y[roles=='test'];recorded=training['constant_metrics'];got=metrics(np.tile(constant,(len(test),1)),test)
    check={k:dict(recorded=recorded[k],reconstructed=got[k]) for k in recorded if isinstance(recorded[k],float)}
    for k,v in check.items():
        if not np.isclose(v['reconstructed'],v['recorded'],rtol=rtol,atol=0):raise AssertionError(f'constant metric {k} not reproduced: {v}')
    seeds=[]
    for r in training['results']:
        a=np.load(HERE/f'training_01/seed_{r["seed"]}_test.npz');p,t,e=a['prediction'],a['truth'],a['energy_MeV']
        if np.abs(t-test).max()>1e-6:raise AssertionError('saved test truth differs from reconstruction')
        rows=dict(pooled=compare(p,t,constant),**{str(g):compare(p[e==g],t[e==g],constant) for g in cfg['energies_MeV']})
        for name in KEYS:
            if not np.isclose(rows['pooled'][name]['model'],r['metrics'][name],rtol=rtol):raise AssertionError('model metric mismatch')
            for g in cfg['energies_MeV']:
                if not np.isclose(rows[str(g)][name]['model'],r['by_energy'][str(g)][name],rtol=rtol):raise AssertionError('subgroup model metric mismatch')
        seeds.append(dict(seed=r['seed'],recorded_gates=r['gates'],recorded_passed=r['passed'],rows=rows))
    return dict(source='runbook v2 §4 R-3',run='results/pmt_multienergy_dev_2026-10-16/training_01',constant_predictor='pooled training-split mean of the 7-target vector, applied unchanged to every energy subgroup',constant_vector=constant.tolist(),target_order=['dir_x','dir_y','dir_z','vertex_x_m','vertex_y_m','vertex_z_m','log(E/1000 MeV)'],energy_endpoint=ENDPOINT,reconstruction_check=check,reconstruction_rtol=rtol,charge_benchmark=training['charge_benchmark'],seeds=seeds,note=NOTE,evidence_stage='development',non_citable=True)


def markdown(rep):
    f=lambda x:'n/a' if x is None else f'{x:.4f}'
    lines=['| Seed | Subgroup | n | Vertex RMSE model / constant (m) | Ratio | Direction mean model / constant (deg) | Ratio | Initial-energy relative MAE model / constant | Ratio |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in rep['seeds']:
        for g,row in s['rows'].items():
            cells=[f"{f(row[k]['model'])} / {f(row[k]['constant'])} | {f(row[k]['ratio'])}" for k in KEYS]
            lines.append(f"| {s['seed']} | {'pooled' if g=='pooled' else g+' MeV'} | {row['n']} | "+' | '.join(cells)+' |')
    cb=rep['charge_benchmark'];lines+=['',f"Charge-calibration benchmark (response-fit least squares, recorded in training.json): pooled relative energy MAE {cb['relative_mae']:.4f}; "+', '.join(f'{k} MeV {v:.4f}' for k,v in cb['by_energy'].items())+f"; negative predictions {cb['negative_predictions']}.",'',f"Energy endpoint: {rep['energy_endpoint']}. The constant is the pooled training-split mean, evaluated unchanged in every subgroup; ratio = model / constant.",'',rep['note']]
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    rep=build();(HERE/'constant_baseline_comparison.json').write_text(json.dumps(rep,indent=2));(HERE/'constant_baseline_comparison.md').write_text('# Constant-predictor comparison, training_01 (runbook v2 R-3)\n\n'+markdown(rep))
    print(json.dumps(rep['reconstruction_check'],indent=1));print(markdown(rep))
