import json, torch, collections
C = json.load(open('/work/dwong/r2_tier1v/b/cases_b.json'))['cases']
def enc(p):
    s = torch.load(p, map_location='cpu', weights_only=False); s = s['model'] if 'model' in s else s
    return {k: v for k, v in s.items() if k.startswith('jepa.') and torch.is_tensor(v)}
out = {}; ref = {}
for c in C:
    if c['ckpt_step'] != 0: continue
    key = (c['det'], c['seed']); w = enc(c['ckpt'])
    if key not in ref: ref[key] = (c['case'], w); out[c['case']] = {'ref': c['case'], 'equal': True}; continue
    r = ref[key][1]
    eq = set(r) == set(w) and all(torch.equal(r[k], w[k]) for k in r)
    out[c['case']] = {'ref': ref[key][0], 'equal': bool(eq)}
json.dump(out, open('/work/dwong/r2_tier1v/b/step0_equality.json', 'w'), indent=1)
print(collections.Counter(v['equal'] for v in out.values()), {k: v['ref'] for k, v in out.items() if v['ref'] == k})
