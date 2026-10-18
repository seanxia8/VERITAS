"""Bit-for-bit comparison of a data set with the 56179eb data set (E40). Usage: python -B compare_inputs.py OLD.npz NEW.npz OUT.json"""
import json, sys
import numpy as np

old, new = np.load(sys.argv[1]), np.load(sys.argv[2]); out = {}
for k in old.files:
    a, b = old[k], new[k] if k in new.files else None
    if b is None: out[k] = 'missing in new'; continue
    same = a.shape == b.shape and bool(np.array_equal(a, b))
    row = dict(shape_old=list(a.shape), shape_new=list(b.shape), identical=same)
    if not same and a.shape == b.shape and a.dtype.kind == 'f':
        d = np.abs(a.astype(float)-b.astype(float)); row.update(max_abs_diff=float(d.max()), n_diff=int((d > 0).sum()))
        if a.ndim == 3:
            row['columns_identical'] = [bool(np.array_equal(a[..., c], b[..., c])) for c in range(a.shape[-1])]
        if a.ndim == 2:
            row['columns_identical'] = [bool(np.array_equal(a[:, c], b[:, c])) for c in range(a.shape[-1])]
    out[k] = row
out['new_only'] = sorted(set(new.files)-set(old.files))
open(sys.argv[3], 'w').write(json.dumps(out, indent=2)); print(json.dumps(out))
