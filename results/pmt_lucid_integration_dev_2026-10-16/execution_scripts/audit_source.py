exec(open('/work/dwong/pmt_lucid_20261016/source_probe.py').read().split('s=Ranges()')[0])
import numpy as np, pathlib
root=pathlib.Path('/work/dwong/pmt_lucid_20261016');s=Ranges();m=json.load(open(root/'source_subset/manifest.json'));s.used=m['read_bytes']+43865
with uproot.open(s) as f:
 a=f['OpticalPhotons'].arrays(['EventID','NOpticalPhotons','Segment_NCherenkov'],entry_start=0,entry_stop=24,library='np')
 rows=[dict(event_id=int(a['EventID'][i]),total=int(a['NOpticalPhotons'][i]),cherenkov=int(np.sum(a['Segment_NCherenkov'][i])),saved=m['events'][i]['photons']) for i in range(24)]
 passed=all(r['event_id']==i and r['total']==r['cherenkov']==r['saved'] for i,r in enumerate(rows))
report=dict(passed=passed,events=rows,aggregate_source_bytes_including_probe=s.used)
(root/'evidence/source_audit.json').write_text(json.dumps(report,indent=2));print(report)
if not passed:raise RuntimeError('source audit failed')
