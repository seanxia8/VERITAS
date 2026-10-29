"""Integrity checks for the approved fixture archive workflow."""
import os
import tarfile

from scripts.jepa_ext_predictability_archive_tests import archive_inventory, inventory


def test_archive_checks_each_file_and_symlink_without_following(tmp_path):
    source = tmp_path / 'fixture'
    source.mkdir()
    (source / 'nested').mkdir()
    (source / 'nested' / 'data').write_bytes(b'\x00fixture\xff')
    (source / 'empty').write_bytes(b'')
    os.link(source / 'nested' / 'data', source / 'hardlink')
    os.symlink('../outside', source / 'link')
    original = inventory(source)
    archive = tmp_path / 'fixture.tar'
    with tarfile.open(archive, mode='x:', format=tarfile.PAX_FORMAT, dereference=False) as tar:
        tar.add(source, arcname='fixture')
    assert archive_inventory(archive, 'fixture') == original
    (source / 'nested' / 'data').write_bytes(b'changed')
    assert inventory(source) != original
