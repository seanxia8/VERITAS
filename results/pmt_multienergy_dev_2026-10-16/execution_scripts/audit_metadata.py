"""Identify macOS AppleDouble sidecars without discarding any source hashes."""
import hashlib
import json
from pathlib import Path
root = Path('/work/dwong/pmt_multienergy_20261016')
r = json.loads((root/'evidence/runtime_receipt.json').read_text())
sidecars = {}
for name, expected in r['code_sha256'].items():
    if not Path(name).name.startswith('._'):
        continue
    data = (root/'code'/name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    assert data[:4] == bytes.fromhex('00051607'), name
    sidecars[name] = {'sha256': expected, 'magic_hex': data[:4].hex(), 'bytes': len(data)}
(root/'evidence/metadata_receipt.json').write_text(json.dumps({'appledouble_sidecars': sidecars}, indent=2))
print('verified AppleDouble sidecars', len(sidecars))
