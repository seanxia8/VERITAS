"""Read-only provenance audit after the bounded multi-energy run."""
import hashlib
import sys
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np

root = Path('/work/dwong/pmt_multienergy_20261016')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((root/'source_balanced/manifest.json').read_text())
mapping = json.loads((root/'evidence/track_closure_02/physical_module_map.json').read_text())
assert len(set(mapping['channel_pmt_id'])) == len(mapping['channel_pmt_id'])
module_sizes = Counter(mapping['channel_mpmt_id'])
source_ok = True
labels_ok = True
source_arrays_ok = True
max_ray_norm_error = 0.0
emission_ranges = {}
for row in manifest['events']:
    p = root/'source_balanced'/f"e{row['energy_MeV']}_{row['event_id']:03d}.npz"
    source_ok &= sha(p) == row['sha256']
    raw = np.load(p)
    key = str(row['energy_MeV'])
    lo, hi = float(raw['emission_ns'].min()), float(raw['emission_ns'].max())
    old = emission_ranges.get(key, {'minimum_ns': lo, 'maximum_ns': hi})
    emission_ranges[key] = {'minimum_ns': min(lo, old['minimum_ns']), 'maximum_ns': max(hi, old['maximum_ns'])}
    source_arrays_ok &= all(np.isfinite(raw[k]).all() for k in raw.files)
    source_arrays_ok &= np.array_equal(raw['photon_id'], np.arange(row['photons']))
    source_arrays_ok &= len(raw['primary_path_m']) > 0
    max_ray_norm_error = max(max_ray_norm_error, float(np.abs(np.linalg.norm(raw['directions'], axis=1)-1).max()))
    labels_ok &= row['primary_energy'] == row['track_initial_energy'] == row['energy_MeV']
a = np.load(root/'evidence/track_closure_02/digits.npz')
a0 = np.load(root/'evidence/track_closure_01/digits.npz')
assert all(np.array_equal(a[k], a0[k]) for k in ('y', 'role', 'physical_event_id', 'energy_MeV', 'contained'))
roles = sorted(set(a['role']))
groups = {r: set(a['physical_event_id'][a['role'] == r]) for r in roles}
report = {
    'python': sys.version,
    'packages': {k: version(k) for k in ('numpy', 'scipy', 'jax', 'jaxlib', 'torch', 'uproot')},
    'source_hashes_match': bool(source_ok),
    'all_energy_labels_match': bool(labels_ok),
    'source_arrays_finite_contiguous_ids_nonempty_tracks': bool(source_arrays_ok),
    'max_source_ray_norm_error': max_ray_norm_error,
    'source_emission_time_ranges': emission_ranges,
    'physical_events': len(manifest['events']),
    'physical_module_count': len(module_sizes),
    'physical_module_size_counts': {str(k): v for k, v in Counter(module_sizes.values()).items()},
    'unique_pmt_ids': len(set(mapping['channel_pmt_id'])),
    'samples': len(a['x']),
    'finite_inputs_targets': bool(np.isfinite(a['x']).all() and np.isfinite(a['y']).all()),
    'splits_disjoint': all(not groups[r] & groups[s] for r in roles for s in roles if r != s),
    'split_physical_counts': {r: len(g) for r, g in groups.items()},
    'split_sample_counts': {r: int((a['role'] == r).sum()) for r in roles},
    'containment_by_energy': {str(e): {'n': int((a['energy_MeV'] == e).sum()), 'contained': int(a['contained'][a['energy_MeV'] == e].sum())} for e in sorted(set(a['energy_MeV']))},
    'code_sha256': {str(p.relative_to(root/'code')): sha(p) for p in sorted((root/'code/src').rglob('*.py'))},
    'config_sha256': sha(root/'multienergy_pilot_repaired.json'),
    'dataset_sha256': sha(root/'evidence/track_closure_02/digits.npz'),
    'failed_initial_dataset_sha256': sha(root/'evidence/track_closure_01/digits.npz'),
    'repair_preserves_targets_splits_containment': True,
    'digit_samples_changed_by_candidate_repair': int(np.any(a['x'] != a0['x'], axis=(1, 2)).sum()),
    'weights_sha256': {p.name: sha(p) for p in sorted((root/'evidence/training_01').glob('*.pt'))},
    'evidence_stage': 'development', 'non_citable': True,
}
assert source_ok and labels_ok and source_arrays_ok and report['finite_inputs_targets'] and report['splits_disjoint']
(root/'evidence/runtime_receipt.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'code_sha256'}, indent=2))
