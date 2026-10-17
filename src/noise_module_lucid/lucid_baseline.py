"""Bounded baseline-only GPU direction pilot; all outputs development-only."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np


def build_subject():
    import torch
    from torch import nn
    from torch.nn import functional as F
    class Subject(nn.Module):
        def __init__(self):
            super().__init__()
            self.sensor=nn.Sequential(nn.Linear(6,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
            self.head=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,3))
        def forward(self,z):
            mask=z[:,:,5:6]>0;h=self.sensor(z)
            mean=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
            maximum=h.masked_fill(~mask,-1e6).max(1).values
            maximum=torch.where(mask.any(1),maximum,torch.zeros_like(maximum))
            return F.normalize(self.head(torch.cat((mean,maximum),dim=1)),dim=1)
    return Subject()


def run(data, config, out):
    import torch
    from torch import nn
    from torch.nn import functional as F
    cfg=json.loads(Path(config).read_text());data=Path(data);out=Path(out)
    if not json.loads((data.parent/'closure.json').read_text())['passed']:
        raise ValueError('prompt closure must pass before training')
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(cfg['cpu_threads'])
    if not torch.cuda.is_available():raise RuntimeError('authorized GPU pilot requires CUDA')
    torch.cuda.set_per_process_memory_fraction(cfg['gpu_memory_fraction'])
    torch.use_deterministic_algorithms(True)
    a=np.load(data);x=torch.from_numpy(a['x']);y=torch.from_numpy(a['y']);groups=a['physical_event_id']
    indices={name:np.flatnonzero(np.isin(groups,ids)) for name,ids in cfg['splits'].items()}
    if any(set(cfg['splits'][a])&set(cfg['splits'][b]) for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('physical event overlap')
    if any(len(v)==0 for v in indices.values()):raise ValueError('empty partition')
    def evaluate(model,ix):
        model.eval();pred=[]
        with torch.no_grad():
            for chunk in np.array_split(ix,max(1,int(np.ceil(len(ix)/cfg['batch_size'])))):
                pred.append(model(x[chunk].cuda()).cpu())
        p=torch.cat(pred);target=y[ix]
        return float(F.mse_loss(p,target)),float(torch.rad2deg(torch.acos((p*target).sum(1).clamp(-1,1))).mean()),p
    constant=y[indices['train']].mean(0)
    constant_mse=float(((y[indices['test']]-constant)**2).mean())
    start=time.monotonic();results=[]
    for seed in cfg['source_seeds']:
        torch.manual_seed(seed);np.random.seed(seed);model=build_subject().cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.0001)
        best=float('inf');history=[];checkpoint=out/f'seed_{seed}.pt'
        for epoch in range(cfg['epochs']):
            if time.monotonic()-start>cfg['training_wall_seconds']:raise RuntimeError('total training wall cap')
            model.train();order=np.random.permutation(indices['train']);losses=[]
            for lo in range(0,len(order),cfg['batch_size']):
                ix=order[lo:lo+cfg['batch_size']];opt.zero_grad(set_to_none=True)
                loss=F.mse_loss(model(x[ix].cuda()),y[ix].cuda());loss.backward();opt.step();losses.append(loss.item())
            val,angle,_=evaluate(model,indices['validation']);history.append(dict(epoch=epoch,train_mse=float(np.mean(losses)),validation_mse=val,validation_angle_deg=angle))
            if val<best:
                best=val;torch.save(model.state_dict(),checkpoint);best_epoch=epoch
            if epoch%10==0:print(seed,epoch,history[-1],flush=True)
        model.load_state_dict(torch.load(checkpoint,weights_only=True));mse,angle,pred=evaluate(model,indices['test'])
        np.savez_compressed(out/f'seed_{seed}_test.npz',prediction=pred.numpy(),truth=y[indices['test']].numpy(),physical_event_id=groups[indices['test']])
        result=dict(seed=seed,test_mse=mse,constant_test_mse=constant_mse,mean_angle_deg=angle,best_epoch=best_epoch,parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),passed=mse<constant_mse*cfg['trainability']['all_seed_test_mse_below_constant_fraction'] and angle<cfg['trainability']['all_seed_mean_angle_deg_below'])
        (out/f'seed_{seed}_history.json').write_text(json.dumps(history,indent=2));results.append(result);print(result,flush=True)
    report=dict(results=results,passed=all(r['passed'] for r in results),config=cfg,torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(),peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start,dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),evidence_stage='development',non_citable=True,primary_claims_evaluated=False,energy_validated=False,independent_test_physical_events=len(cfg['splits']['test']))
    (out/'training.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('trainability gate failed; no retuning authorized by this pilot')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('data','config','out'):p.add_argument('--'+name,required=True)
    run(**vars(p.parse_args()))
