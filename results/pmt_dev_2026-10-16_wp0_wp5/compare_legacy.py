import types,sys,subprocess,json
from pathlib import Path
import numpy as np
from noise_module_lucid import toy
from noise_module_lucid.readout import readout_chain,ReadoutParams
source=subprocess.check_output(['git','show','84f65348a438e1b8a4594246afd3d60785518d76:src/noise_module_lucid/readout.py'],text=True)
m=types.ModuleType('noise_module_lucid._baseline_readout');m.__package__='noise_module_lucid';sys.modules[m.__name__]=m;exec(source,m.__dict__)
d=toy.cylinder_detector();checks=[]
for i in range(12):
 e=toy.point_source_event(d,i,n_photons=800)
 for change in ({},{'survival':.7},{'gain':.8},{'threshold_pe':.75},{'timing_offsets_ns':2},{'dead_channels':tuple(range(64))}):
  old=m.readout_chain(e.photon_sensor,e.photon_time,d.n_channels,m.ReadoutParams(**change),seed=i)
  new=readout_chain(e.photon_sensor,e.photon_time,d.n_channels,ReadoutParams(**change),seed=i)
  checks.append(all(np.array_equal(getattr(old,k),getattr(new,k)) for k in ('sensor','charge','time','n_pe_true')))
assert all(checks)
Path('results/pmt_dev_2026-10-16_wp0_wp5/legacy_comparison.json').write_text(json.dumps({'base':'84f65348a438e1b8a4594246afd3d60785518d76','events':12,'settings_per_event':6,'all_72_digit_comparisons_exact':all(checks),'intentional_differences':['lost_time_window counter added','energy calibration now independent','point fitter initial simplex fixed','failed/missing endpoints unevaluable']},indent=2))
print('72 legacy digit comparisons exact')
