"""Read-only artifact and accounting checks; no scientific data generation."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from jepa_noise.report import render
start=time.monotonic(); out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checked=[]
for folder in ['20261016_storage_recovery','20261016_storage_recovery/remote_env_retry1','20261016_wp2_waveform_covariance']:
    base=Path('results/jepa')/folder
    for line in (base/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split(maxsplit=1); p=base/name
        assert sha(p)==expected,str(p)
        checked.append(str(p))
base=Path('results/jepa/20261016_wp2_waveform_covariance')
result=json.loads((base/'RESULT.json').read_text()); n=result['n']
mean=np.load(base/'SUM_X.npy')/n
emp=np.load(base/'SUM_XX.npy')/n-np.outer(mean,mean)
assert np.array_equal(emp,np.load(base/'EMPIRICAL_COVARIANCE.npy'))
pop=json.loads(Path('results/jepa/20261016_wp2_population_line075_common045/provenance.json').read_text())
noise=np.load(Path(pop['raw_directory'].split(':',1)[1])/'noise_covariance.npy')
error=float(np.linalg.norm(emp-noise)/np.linalg.norm(noise))
assert abs(error-result['relative_frobenius_error'])<1e-12 and error<.02
ledger=json.loads(Path('docs/jepa/RESOURCE_LEDGER.json').read_text())
charge=sum(x['cpu_budget_hours'] for x in ledger['preflight_records'] if x['charge_to']=='E1')+sum(x['cpu_budget_hours'] for x in ledger['jobs'] if x['campaign']=='E1')
assert abs(charge-ledger['campaigns']['E1']['charged_cpu_hours'])<1e-12
rows=json.loads(Path('docs/jepa/PREDICTION_STATUS.json').read_text())['predictions']
assert render(rows)==Path('docs/jepa/PREDICTIONS.md').read_text()
binding={str(p):sha(p) for p in [Path('docs/jepa')/name for name in ['REPORT.md','COMPLETION.md','RESOURCE_LEDGER.json','GATE_G6.md','REGISTRATION.json','GATE_G5.md','E1C_CONFIG_PROPOSAL.md','E1_PROBE_CONFIG_PROPOSAL.md','RUNBOOK_JEPA_NOISE_2026-10-16.md']]}
registration=json.loads(Path('docs/jepa/REGISTRATION.json').read_text())
assert binding['docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md']==registration['source_sha256']['docs/jepa/RUNBOOK_JEPA_NOISE_2026-10-16.md']
data={'checked_inventory_files':checked,'full_covariance_error':error,'ledger_E1_hours':charge,'report_hashes':binding,'prediction_render_identical':True,'elapsed_seconds':time.monotonic()-start}
(out/'artifact_results.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k not in ['checked_inventory_files','report_hashes']},indent=2))
