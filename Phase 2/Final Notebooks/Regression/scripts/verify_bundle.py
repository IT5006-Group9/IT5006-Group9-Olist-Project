"""Check the portable snapshot; private historical artifacts are not needed."""
from pathlib import Path
import hashlib
import json


def verify_bundle(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'bundle_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    print('Verified portable snapshot:', len(manifest['files']), 'immutable files')
    return {'files_checked': len(manifest['files']), 'all_hashes_match': True}


if __name__ == '__main__':
    verify_bundle()
