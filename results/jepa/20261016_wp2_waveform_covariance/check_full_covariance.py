"""Single predeclared engineering attempt; stdout and files preserve STOP."""
import hashlib, json, os, platform, signal, subprocess, sys, time
import importlib.metadata
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import scipy
from jepa_noise.views import Tier1Views

OUT = Path(__file__).resolve().parent
CFG = json.loads((OUT/'PREDECLARATION.json').read_text())
START = time.monotonic()
UTC = lambda: datetime.now(timezone.utc).isoformat()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
root = Path.cwd()
sources = sorted(p for directory in ['src/jepa_noise','src/latent_monitor','src/noise_module']
                 for p in (root/directory).rglob('*.py') if '__pycache__' not in str(p))
inventory = {str(p.relative_to(root)):sha(p) for p in sources}
inventory['src/jepa_noise/configs/waveform_v1.json']=sha(root/'src/jepa_noise/configs/waveform_v1.json')
provenance = {'started_utc':UTC(),'host':platform.node(),'platform':platform.platform(),
 'python':sys.version,'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,
 'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'git_status':subprocess.check_output(['git','status','--porcelain=v1'],text=True),
 'source_sha256':inventory,'predeclaration_sha256':sha(OUT/'PREDECLARATION.json'),
 'runner_sha256':sha(__file__), 'env_threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
 'packages':sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())}
(OUT/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
sums=np.zeros(2048); cross=np.zeros((2048,2048)); count=0
result={'status':'started','n':0}

def timeout(signum, frame):
    raise TimeoutError('600-second single-attempt CPU active cap')
signal.signal(signal.SIGALRM,timeout)
signal.setitimer(signal.ITIMER_REAL,max(.01,600-(time.monotonic()-START)))
try:
    model=Tier1Views(seed=CFG['seed'],cell=CFG['cell'],model_seed=0,
                     line_multiplier=.75,common_strength=.45,signal_mode='linearized',split='time')
    covariance=model.noise_covariance
    fro2=float(np.sum(covariance**2)); tr=float(np.trace(covariance)); n=CFG['n_records']
    expected_rms=float(np.sqrt(((n-1)*(tr*tr+fro2)+fro2)/(n*n*fro2)))
    t=time.monotonic()
    # Prefix is actual fresh run stream, retained and counted if launch accepted.
    prefix=np.array([model.noise_record(i,role=CFG['role'])[0].ravel() for i in range(512)])
    generation_seconds=time.monotonic()-t
    t=time.monotonic(); prefix_cross=prefix.T@prefix; multiplication_seconds=time.monotonic()-t
    projected=(generation_seconds+multiplication_seconds)*n/512*1.25+(time.monotonic()-START)
    projection={'gaussian_expected_relative_rms':expected_rms,'trace':tr,'frobenius_squared':fro2,
                'effective_rank_trace2_over_fro2':tr*tr/fro2,'generation_512_seconds':generation_seconds,
                'covariance_product_512_seconds':multiplication_seconds,'projected_active_seconds_with_25pct_reserve':projected,
                'scope':'fixed prefix engineering timing; no model outcome or calibration'}
    (OUT/'PROJECTION.json').write_text(json.dumps(projection,indent=2)+'\n')
    print(json.dumps(projection),flush=True)
    if projected>=600:
        result={'status':'STOP_projection_exceeds_cap','n':0}
    else:
        sums=prefix.sum(axis=0); cross=prefix_cross; count=512
        while count<n:
            batch=min(512,n-count)
            x=np.array([model.noise_record(i,role=CFG['role'])[0].ravel() for i in range(count,count+batch)])
            sums+=x.sum(axis=0); cross+=x.T@x; count+=batch
            if count%10240==0 or count==n:
                (OUT/'PROGRESS.json').write_text(json.dumps({'records':count,'active_seconds':time.monotonic()-START})+'\n')
                print('records',count,'seconds',round(time.monotonic()-START,3),flush=True)
        empirical=cross/n-np.outer(sums/n,sums/n)
        error=float(np.linalg.norm(empirical-covariance)/np.sqrt(fro2))
        np.save(OUT/'EMPIRICAL_COVARIANCE.npy',empirical)
        result={'status':'pass' if error<.02 else 'fail','n':count,'relative_frobenius_error':error,
                'criterion':.02,'expected_relative_rms':expected_rms,'model_manifest':model.manifest()}
except TimeoutError as exc:
    result={'status':'STOP_timeout','n':count,'reason':str(exc)}
except Exception as exc:
    result={'status':'error','n':count,'reason':repr(exc)}
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
    np.save(OUT/'SUM_X.npy',sums); np.save(OUT/'SUM_XX.npy',cross)
    result.update(active_seconds=time.monotonic()-START,ended_utc=UTC(),cpu_threads=1)
    changed=[p for p,h in inventory.items() if sha(root/p)!=h]
    result['source_changed_during_run']=changed
    if changed: result['status']='STOP_source_changed'
    (OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
    print(json.dumps({k:v for k,v in result.items() if k!='model_manifest'}),flush=True)
