"""Like-for-like reference (E40): the 56179eb checkpoints scored on the validation split (CPU inference only).

The 0.73-0.88 vertex range of 56179eb was measured on test; arm 0 is evaluated on validation.
Usage: python -B reference_56179eb.py DATA.npz OUT.json CKPT [CKPT ...]
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
from noise_module_lucid.multienergy_baseline import metrics
from noise_module_lucid.precision_study import predict_checkpoints
from noise_module_lucid.trainability_gates import constant_predictor

data, out, ckpts = sys.argv[1], sys.argv[2], sys.argv[3:]
a = np.load(data); role = a['role']; val = np.flatnonzero(role == 'validation'); train = np.flatnonzero(role == 'train')
const = constant_predictor(a['y'], train); truth = a['y'][val]
cm = metrics(np.tile(const, (len(val), 1)), truth)
def heads(p, t):
    return dict(direction=float(np.mean((p[:, :3]-t[:, :3])**2)), vertex=float(np.mean(((p[:, 3:6]-t[:, 3:6])/.35)**2)), energy=float(np.mean((p[:, 6]-t[:, 6])**2)))
preds = predict_checkpoints(a['x'][val], ckpts)
rows = []
for path, p in zip(ckpts, preds):
    m = metrics(p, truth)
    rows.append(dict(checkpoint=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), vertex_ratio=m['vertex_rmse_m']/cm['vertex_rmse_m'],
                     direction_deg=m['direction_mean_deg'], energy_ratio=m['energy_relative_mae']/cm['energy_relative_mae'], validation_heads=heads(p, truth), metrics=m))
Path(out).write_text(json.dumps(dict(dataset=data, dataset_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(), split='validation', n=int(len(val)),
                                     constant_metrics=cm, constant_heads=heads(np.tile(const, (len(val), 1)), truth), results=rows,
                                     caveat='the validation split also selected these checkpoints'), indent=2))
print(json.dumps([(r['checkpoint'].split('/')[-1], round(r['vertex_ratio'], 4), round(r['energy_ratio'], 4)) for r in rows]))
