import datetime, hashlib, importlib, importlib.metadata, json, os, pathlib, platform, socket, subprocess, time
start=time.time()
root=pathlib.Path('/ceph/dwong/jepa/returns/20261016_wp0_remote_env_retry1')
sha='72add60e0e92ad4ec344bb309adbfc023b486455'
repo=pathlib.Path('/ceph/dwong/jepa/code')/sha
os.chdir(repo)
import sys
sys.path.insert(0,str(repo/'src'))
mods=['numpy','scipy','pandas','matplotlib','h5py','pytest','sklearn','jaxtyping','torch','reconstruction_model.model','tidmad_transformer.model','jepa_noise']
results={}
for name in mods:
    mod=importlib.import_module(name)
    results[name]={'status':'pass','path':getattr(mod,'__file__',None),'version':getattr(mod,'__version__',None)}
import torch
assert torch.__version__=='2.5.1+cpu'
assert torch.get_num_threads()==1
assert not torch.cuda.is_available()
status=subprocess.check_output(['git','status','--porcelain=v1'],text=True)
assert not status
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==sha
(root/'environment_freeze.txt').write_text(''.join(sorted(f'{d.metadata["Name"]}=={d.version}\n' for d in importlib.metadata.distributions())))
out={'status':'pass','code_sha':sha,'code_host_path':socket.gethostname()+':'+str(repo),'return_host_path':socket.gethostname()+':'+str(root),'python':sys.version,'platform':platform.platform(),'imports':results,'torch_threads':torch.get_num_threads(),'cuda_available':torch.cuda.is_available(),'git_status_porcelain':status,'environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RECONSTRUCTION_DISABLE_TORCH_COMPILE','TORCHDYNAMO_DISABLE']},'start_utc':datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.time()-start,'gpu_hours':0,'scientific_outcomes_generated':False,'exit_status':0}
(root/'import_provenance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
