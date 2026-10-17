"""Baseline-only multi-target development pilot with independent charge calibration."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np


def build_subject():
    import torch
    from torch import nn
    from torch.nn import functional as F
    class Subject(nn.Module):
        def __init__(self):
            super().__init__();self.sensor=nn.Sequential(nn.Linear(6,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU());self.head=nn.Sequential(nn.Linear(258,256),nn.ReLU(),nn.Linear(256,7))
        def forward(self,x):
            mask=x[:,:,5:6]>0;h=self.sensor(x);count=mask.sum(1).clamp_min(1)
            avg=(h*mask).sum(1)/count;mx=h.masked_fill(~mask,-1e6).max(1).values;mx=torch.where(mask.any(1),mx,torch.zeros_like(mx))
            q=torch.expm1(x[:,:,0:1].masked_fill(~mask,0)).sum(1)/1000;occupancy=mask.sum(1)/1995.0
            raw=self.head(torch.cat((avg,mx,q,occupancy),dim=1))
            return torch.cat((F.normalize(raw[:,:3],dim=1),raw[:,3:]),dim=1)
    return Subject()


def metrics(pred,truth):
    p=np.asarray(pred);t=np.asarray(truth);vertex=np.linalg.norm(p[:,3:6]-t[:,3:6],axis=1);angle=np.degrees(np.arccos(np.clip(np.sum(p[:,:3]*t[:,:3],axis=1)/np.maximum(np.linalg.norm(p[:,:3],axis=1)*np.linalg.norm(t[:,:3],axis=1),1e-12),-1,1)));energy=1000*np.exp(p[:,6]);true_energy=1000*np.exp(t[:,6])
    return dict(n=len(p),vertex_rmse_m=float(np.sqrt(np.mean(vertex**2))),vertex_median_m=float(np.median(vertex)),direction_mean_deg=float(np.mean(angle)),direction_median_deg=float(np.median(angle)),energy_relative_mae=float(np.mean(np.abs(energy/true_energy-1))),energy_bias_fraction=float(np.mean(energy/true_energy-1)),direction_mse=float(np.mean((p[:,:3]-t[:,:3])**2)),vertex_mse=float(np.mean((p[:,3:6]-t[:,3:6])**2)),log_energy_mse=float(np.mean((p[:,6]-t[:,6])**2)),finite=bool(np.isfinite(p).all()))


def run(config,data,out,gates=None):
    import torch
    from .trainability_gates import DEFAULT,load_gates,constant_predictor,evaluate
    from torch.nn import functional as F
    cfg=json.loads(Path(config).read_text());data=Path(data);out=Path(out);gate_cfg=load_gates(gates or DEFAULT)
    if not json.loads((data.parent/'validation.json').read_text())['passed']:raise ValueError('transport gate failed')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(cfg['cpu_threads']);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for authorized pilot')
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    a=np.load(data);x=torch.from_numpy(a['x']);y=torch.from_numpy(a['y']);roles=a['role'];groups=a['physical_event_id'];indices={k:np.flatnonzero(roles==k) for k in cfg['splits']}
    for ka,ia in indices.items():
        if not len(ia):raise ValueError('empty split')
        for kb,ib in indices.items():
            if ka!=kb and set(groups[ia])&set(groups[ib]):raise ValueError('physical event split leakage')
    fit=indices['response_fit'];design=np.column_stack((a['charge'][fit],np.ones(len(fit))));coef=np.linalg.lstsq(design,a['energy_MeV'][fit],rcond=None)[0]
    cal=dict(method='least squares charge-to-primary-energy, intercept included',fit_physical_event_ids=sorted(set(groups[fit])),coefficients=coef.tolist(),source_dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest());cal['sha256']=hashlib.sha256(json.dumps(cal,sort_keys=True).encode()).hexdigest();(out/'charge_calibration.json').write_text(json.dumps(cal,indent=2))
    test=indices['test'];constant_vector=constant_predictor(a['y'],indices['train']);constant=np.tile(constant_vector,(len(test),1));constant_metrics=metrics(constant,a['y'][test])
    q_pred=coef[0]*a['charge'][test]+coef[1];charge_benchmark=dict(relative_mae=float(np.mean(np.abs(q_pred/a['energy_MeV'][test]-1))),negative_predictions=int((q_pred<=0).sum()),by_energy={str(e):float(np.mean(np.abs(q_pred[a['energy_MeV'][test]==e]/e-1))) for e in cfg['energies_MeV']})
    def loss(p,t):return F.mse_loss(p[:,:3],t[:,:3])+F.mse_loss(p[:,3:6]/.35,t[:,3:6]/.35)+F.mse_loss(p[:,6],t[:,6])
    def predict(model,ix):
        model.eval();p=[]
        with torch.no_grad():
            for lo in range(0,len(ix),cfg['batch_size']):p.append(model(x[ix[lo:lo+cfg['batch_size']]].cuda()).cpu())
        return torch.cat(p)
    start=time.monotonic();results=[]
    for seed in cfg['model_seeds']:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=build_subject().cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.0001);history=[];best=float('inf');checkpoint=out/f'seed_{seed}.pt'
        for epoch in range(cfg['epochs']):
            if time.monotonic()-start>cfg['training_wall_seconds']:raise RuntimeError('total training wall cap')
            order=rng.permutation(indices['train']);model.train();train=[]
            for lo in range(0,len(order),cfg['batch_size']):
                ix=order[lo:lo+cfg['batch_size']];opt.zero_grad(set_to_none=True);v=loss(model(x[ix].cuda()),y[ix].cuda());v.backward();opt.step();train.append(float(v.detach().cpu()))
            val=float(loss(predict(model,indices['validation']),y[indices['validation']]));history.append(dict(epoch=epoch,train_loss=float(np.mean(train)),validation_loss=val))
            if val<best:best=val;best_epoch=epoch;torch.save(model.state_dict(),checkpoint)
            if epoch%20==0:print(seed,epoch,history[-1],flush=True)
        model.load_state_dict(torch.load(checkpoint,weights_only=True));p=predict(model,test).numpy();truth=a['y'][test];m=metrics(p,truth);relative=evaluate(p,truth,constant_vector,a['energy_MeV'][test],gate_cfg,energy_column=cfg.get('energy_column',6),energy_endpoint=cfg.get('energy_endpoint','initial_energy'),expected_subgroups=cfg['energies_MeV'])
        endpoint_gates={name:bool(m[name+'_mse']<constant_metrics[name+'_mse']) for name in ('direction','vertex','log_energy')}
        endpoint_gates.update({'r3_'+k:v['passed'] for k,v in relative['gates'].items()})
        result=dict(seed=seed,metrics=m,gates=endpoint_gates,relative_gates=relative,passed=all(endpoint_gates.values()),best_epoch=best_epoch,parameters=sum(t.numel() for t in model.parameters()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),by_energy={str(e):metrics(p[a['energy_MeV'][test]==e],truth[a['energy_MeV'][test]==e]) for e in cfg['energies_MeV']},by_containment={str(v):metrics(p[a['contained'][test]==v],truth[a['contained'][test]==v]) for v in (False,True) if np.any(a['contained'][test]==v)})
        np.savez_compressed(out/f'seed_{seed}_test.npz',prediction=p,truth=truth,physical_event_id=groups[test],energy_MeV=a['energy_MeV'][test],contained=a['contained'][test]);(out/f'seed_{seed}_history.json').write_text(json.dumps(history,indent=2));results.append(result);print(json.dumps(result),flush=True)
    report=dict(results=results,passed=all(r['passed'] for r in results),constant_metrics=constant_metrics,constant_vector=constant_vector.tolist(),trainability_gates=gate_cfg,charge_benchmark=charge_benchmark,config=cfg,elapsed_seconds=time.monotonic()-start,dataset_sha256=cal['source_dataset_sha256'],peak_gpu_bytes=torch.cuda.max_memory_allocated(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),evidence_stage='development',non_citable=True,primary_claims_evaluated=False)
    (out/'training.json').write_text(json.dumps(report,indent=2));print('all_trainability_gates',report['passed'],flush=True)
    if not report['passed']:raise RuntimeError('trainability gate failure retained; no retuning')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for a in ('config','data','out'):p.add_argument('--'+a,required=True)
    p.add_argument('--gates')
    run(**vars(p.parse_args()))
