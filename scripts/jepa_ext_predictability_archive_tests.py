#!/usr/bin/env python3
"""Archive, verify and remove only the author's three approved fixture trees."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile

BASE = Path('/ceph/dwong/jepa/tmp')
EXT = Path('/ceph/dwong/jepa/returns/ext_predictability_20261029')
APPROVED = {
    'pytest_predictability_9ef9f19_retry1': 1748,
    'pytest_predictability_5dc283b_full': 1755,
    'pytest_predictability_ad6ca1b_full': 1760,
}


def file_hash(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def inventory(source):
    source = Path(source)
    found = {}
    for parent, dirs, files in os.walk(source, followlinks=False):
        dirs.sort(); files.sort()
        for name in dirs + files:
            path = Path(parent) / name
            rel = path.relative_to(source).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                found[rel] = {'type': 'symlink', 'target': os.readlink(path)}
            elif stat.S_ISDIR(mode):
                found[rel] = {'type': 'directory'}
            elif stat.S_ISREG(mode):
                with path.open('rb') as stream:
                    found[rel] = {'type': 'file', 'sha256': file_hash(stream)}
            else:
                raise ValueError(f'unsupported filesystem entry: {path}')
    return found


def archive_inventory(path, label):
    found = {}
    with tarfile.open(path, 'r:') as tar:
        for member in tar:
            name = member.name.rstrip('/')
            if name == label:
                continue
            if not name.startswith(label + '/'):
                raise ValueError(f'archive member outside root: {name}')
            rel = name[len(label) + 1:]
            if rel in found or not rel or '..' in Path(rel).parts:
                raise ValueError(f'duplicate or unsafe archive member: {name}')
            if member.isdir():
                found[rel] = {'type': 'directory'}
            elif member.issym():
                found[rel] = {'type': 'symlink', 'target': member.linkname}
            elif member.isfile() or member.islnk():
                stream = tar.extractfile(member)
                if stream is None:
                    raise ValueError(f'unreadable archived file: {name}')
                with stream:
                    found[rel] = {'type': 'file', 'sha256': file_hash(stream)}
            else:
                raise ValueError(f'unsupported archive member: {name}')
    return found


def check_quota():
    p = '/ceph/dwong'
    free = (int(os.getxattr(p, 'ceph.quota.max_files')) -
            int(os.getxattr(p, 'ceph.dir.rfiles')) - int(os.getxattr(p, 'ceph.dir.rsubdirs')))
    if free < 30000:
        raise RuntimeError('S1: fewer than 30000 free entries')


def main():
    destination = EXT / 'fixture_archives'
    check_quota()
    if destination.exists():
        raise FileExistsError('archive destination already exists; do not overwrite')
    destination.mkdir()
    records = []
    for label, expected in APPROVED.items():
        source = BASE / label
        if source.is_symlink() or not source.is_dir() or source.resolve() != source:
            raise ValueError(f'unsafe source directory: {source}')
        before = inventory(source)
        if len(before) != expected:
            raise ValueError(f'{source}: expected {expected} entries, got {len(before)}')
        archive = destination / (label + '.tar')
        # The known sources total ~618 MiB; fail before writing if 2 GB cannot be honoured.
        current = sum(p.stat().st_size for p in EXT.rglob('*') if not p.is_symlink() and p.is_file())
        size_upper = sum(p.stat().st_size for p in source.rglob('*') if not p.is_symlink() and p.is_file())
        if current + size_upper + 50_000_000 > 2_000_000_000:
            raise RuntimeError('S1: archive could exceed 2 GB extension cap')
        check_quota()
        with tarfile.open(archive, mode='x:', format=tarfile.PAX_FORMAT, dereference=False) as tar:
            tar.add(source, arcname=label, recursive=True)
        after = archive_inventory(archive, label)
        if before != after or before != inventory(source):
            raise ValueError(f'{source}: archive or source inventory mismatch')
        with archive.open('rb') as stream:
            digest = file_hash(stream)
        record = {'source': str(source), 'entries': len(before),
                  'archive': str(archive), 'archive_bytes': archive.stat().st_size,
                  'archive_sha256': digest, 'files': before}
        check_quota()
        with (destination / (label + '.manifest.json')).open('x') as stream:
            json.dump(record, stream, sort_keys=True, indent=1)
            stream.write('\n')
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != 'files'}), flush=True)
    # All archives are verified before any source is removed; recheck each source.
    for record in records:
        source = Path(record['source'])
        if inventory(source) != record['files']:
            raise ValueError(f'{source}: source changed before removal')
    for record in records:
        check_quota()
        shutil.rmtree(record['source'])
        print('removed ' + record['source'], flush=True)
    check_quota()
    receipt = [{k: v for k, v in record.items() if k != 'files'} for record in records]
    with (destination / 'verified_cleanup.json').open('x') as stream:
        json.dump({'registered': False, 'archives': receipt}, stream, indent=2)
        stream.write('\n')
    print('verified_cleanup_complete', flush=True)


if __name__ == '__main__':
    main()
