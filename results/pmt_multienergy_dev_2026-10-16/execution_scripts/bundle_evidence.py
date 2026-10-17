"""Package small validation receipts; keep source arrays/digits/weights remote."""
from pathlib import Path
import tarfile

root = Path('/work/dwong/pmt_multienergy_20261016')
with tarfile.open(root/'evidence/receiving_bundle.tar.gz', 'w:gz') as bundle:
    for p in sorted((root/'evidence').rglob('*')):
        if p.is_file() and (p.suffix in {'.json', '.log', '.py'} or p.name.endswith('_test.npz')):
            bundle.add(p, arcname=str(p.relative_to(root/'evidence')))
    for directory in ('source', 'source_02', 'source_03', 'source_balanced'):
        for p in sorted((root/directory).glob('*.json')):
            bundle.add(p, arcname=str(p.relative_to(root)))
    p = root/'code/noise_module_lucid/multienergy_source.py'
    bundle.add(p, arcname='execution_scripts/extractor_attempt01.py')
print(root/'evidence/receiving_bundle.tar.gz')
