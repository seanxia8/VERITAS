"""Reproduce registered prior evidence and import preflight, not an outcome grid."""
import contextlib, datetime, hashlib, importlib, io, json, os, pathlib, platform, runpy, subprocess, sys, time
ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=pathlib.Path(__file__).resolve().parent
os.chdir(ROOT)
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def git(*a): return subprocess.check_output(['git',*a],text=True).strip()
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
sources=['docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md','docs/jepa/toy_check_2026-10-16.py','docs/jepa/reviews/check_p5_counterexample.py']
meta={'status':'development output, not reviewed','purpose':'WP0 reproduction of known prior evidence', 'code_sha':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'working_tree':git('status','--porcelain=v1'),'source_sha256':{p:sha(p) for p in sources},'python':sys.version,'executable':sys.executable,'host':platform.node(),'platform':platform.platform(),'start_utc':now(),'cpu_count':os.cpu_count(),'load_average':os.getloadavg(),'workers':1,'environment':{k:os.environ.get(k) for k in ['PYTHONPATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','RECONSTRUCTION_DISABLE_TORCH_COMPILE']},'config_hash':sha(__file__),'environment_hash':sha(OUT/'requirements-frozen.txt'),'data_hash':'synthetic; registered toy source defines data generation','split_hash':'not applicable: historical two-view toy','rng':'Appendix A numpy.default_rng(0), original draw order; P5 deterministic','runs':[]}
start=time.monotonic()
import numpy as np
for name in ['torch','numpy','scipy','pandas','matplotlib','h5py','pytest','sklearn','jaxtyping','reconstruction_model.model','tidmad_transformer.model','jepa_noise']:
 m=importlib.import_module(name)
 meta.setdefault('imports',{})[name]={'status':'pass','version':getattr(m,'__version__',None),'file':m.__file__}
import torch
meta['torch']={'version':torch.__version__,'threads':torch.get_num_threads(),'cuda_available':torch.cuda.is_available(),'device_used':'cpu'}
for source,name in [(sources[1],'appendix_a'),(sources[2],'p5_annihilation')]:
 t=time.monotonic(); buf=io.StringIO()
 with contextlib.redirect_stdout(buf): state=runpy.run_path(source)
 output=buf.getvalue(); (OUT/f'{name}.txt').write_text(output)
 meta['runs'].append({'name':name,'elapsed_seconds':time.monotonic()-t,'exit_status':0})
 if name=='appendix_a':
  import re
  expected=[([.922,.889,.880,.026],.9,1.),([.998,.920,.888,.880],89.3,.9)]
  lines=output.splitlines()
  for i,(corr,angle,gevp) in enumerate(expected):
   line=lines[3*i]; values=np.fromstring(line.split('[')[1].split(']')[0],sep=' ')
   assert np.max(abs(values-corr))<=.01
   assert abs(float(line.split('=')[-1].split()[0])-angle)<=.5
   assert abs(float(lines[3*i+2].split('=')[-1].split()[0])-gevp)<=.5
   whitened=np.fromstring(lines[3*i+1].split('[')[1].split(']')[0],sep=' ')
   assert np.array_equal(values,whitened)
  meta['appendix_a_tolerance']='pass: correlations .01, angles .5 degrees, printed whitening spectra identical'
  meta['rng_final_state']=state['rng'].bit_generator.state
 else:
  assert np.linalg.matrix_rank(state['a_white'][2:],tol=1e-8)==0
  meta['p5_status']='oracle_rank_ineligible: valid SPD annihilation negative control; not recovery failure'
meta.update(end_utc=now(),elapsed_seconds=time.monotonic()-start,exit_status=0)
meta['cumulative_cpu_worker_hours']=meta['elapsed_seconds']/3600
(OUT/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'imports':'pass','appendix_a':meta['appendix_a_tolerance'],'p5':meta['p5_status'],'seconds':meta['elapsed_seconds']},indent=2))
