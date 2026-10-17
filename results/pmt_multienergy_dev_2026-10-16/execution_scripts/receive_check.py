"""Independent metric and hash receiving check; run from the worktree root."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path.cwd()
out = root/'results/pmt_multienergy_dev_2026-10-16'
receipt = json.loads((out/'runtime_receipt.json').read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
metadata = json.loads((out/'metadata_receipt.json').read_text())['appledouble_sidecars']
for p, h in receipt['code_sha256'].items():
    if p in metadata:
        assert Path(p).name.startswith('._') and metadata[p]['magic_hex'] == '00051607'
        assert metadata[p]['sha256'] == h
    else:
        assert sha(root/p) == h, p
assert sha(root/'configs/lucid_pmt/multienergy_pilot.json') == receipt['config_sha256']
report = json.loads((out/'training_01/training.json').read_text())
assert report['dataset_sha256'] == receipt['dataset_sha256']
checks = []
for row in report['results']:
    seed = row['seed']
    assert row['checkpoint_sha256'] == receipt['weights_sha256'][f'seed_{seed}.pt']
    a = np.load(out/f'training_01/seed_{seed}_test.npz')
    p, t = a['prediction'].astype(float), a['truth'].astype(float)
    cos = (p[:, :3]*t[:, :3]).sum(1)/(np.linalg.norm(p[:, :3], axis=1)*np.linalg.norm(t[:, :3], axis=1))
    values = {
        'direction_mean_deg': float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()),
        'vertex_rmse_m': float(np.sqrt(np.mean(np.sum((p[:, 3:6]-t[:, 3:6])**2, axis=1)))),
        'energy_relative_mae': float(np.abs(np.exp(p[:, 6]-t[:, 6])-1).mean()),
    }
    assert np.isfinite(p).all()
    for key, value in values.items():
        assert np.isclose(value, row['metrics'][key], atol=2e-4, rtol=1e-5), (seed, key)
    checks.append({'seed': seed, 'independent_metrics': values, 'matches': True})
result = {'runtime_code_config_match': True, 'verified_non_source_appledouble_sidecars': len(metadata), 'dataset_and_checkpoint_hashes_match': True,
          'independent_metric_checks': checks, 'primary_claims_evaluated': False}
(out/'receiving_checks.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
