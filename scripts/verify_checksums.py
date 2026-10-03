"""Verify deposited file hashes against the release checksums."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]

def main():
    count = 0
    manifests = [ROOT / 'SHA256SUMS.txt']
    catalog_manifest = ROOT / 'CATALOG_SHA256SUMS.txt'
    if catalog_manifest.exists():
        manifests.append(catalog_manifest)
    for manifest in manifests:
        for line in manifest.read_text(encoding='utf-8').splitlines():
            expected, relative = line.split('  ', 1)
            path = ROOT / relative
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise RuntimeError(f'Checksum mismatch: {relative}')
            count += 1
    print(f'Verified {count} files.')

if __name__ == '__main__':
    main()
