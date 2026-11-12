"""Proposed fix for concurrent local_dir downloads, applied in-process for validation (not a change to the checkout).

Rewrites the two lock-held regions of `file_download._hf_hub_download_to_local_dir` (main @ 2c61ae8, L1485-1489 and
L1503-1519) from the function's own source:
  1. re-check under the lock whether D/<file> already holds this etag (metadata read without re-taking the lock),
  2. never unlink the current file first: download / copy to a unique tmp file and os.replace() it in place,
  3. write the metadata before releasing the lock.
This is the `Fixed = TRUE` variant of LocalDir.tla.
"""

import inspect
import os
import shutil
import textwrap
import time
import uuid


def _local_copy_is_current(paths, etag):
    """Same test as read_download_metadata (+ etag), for a caller that already holds paths.lock_path."""
    try:
        with paths.metadata_path.open() as f:
            f.readline()
            meta_etag = f.readline().strip()
            ts = float(f.readline().strip())
        return meta_etag == etag and paths.file_path.stat().st_mtime - 1 <= ts
    except (OSError, ValueError):
        return False


def _write_download_metadata_locked(paths, commit_hash, etag):
    tmp = paths.metadata_path.with_name(f"{paths.metadata_path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(f"{commit_hash}\n{etag}\n{time.time()}\n")
    os.replace(tmp, paths.metadata_path)


def _copy_atomically(src, paths, etag):
    paths.file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = paths.incomplete_path(etag).with_name(f"{uuid.uuid4().hex[:8]}.copy.incomplete")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, paths.file_path)
    finally:
        tmp.unlink(missing_ok=True)


def apply():
    from huggingface_hub import file_download as fd

    src = textwrap.dedent(inspect.getsource(fd._hf_hub_download_to_local_dir))
    old_copy = """            with WeakFileLock(paths.lock_path):
                paths.file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached_path, paths.file_path)
            write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=commit_hash, etag=etag)
            return str(paths.file_path)"""
    new_copy = """            with WeakFileLock(paths.lock_path):
                if not _local_copy_is_current(paths, etag):
                    _copy_atomically(cached_path, paths, etag)
                _write_download_metadata_locked(paths, commit_hash, etag)
            return str(paths.file_path)"""
    old_dl_head = """    with WeakFileLock(paths.lock_path):
        paths.file_path.unlink(missing_ok=True)  # delete outdated file first
        _download_to_tmp_and_move("""
    new_dl_head = """    with WeakFileLock(paths.lock_path):
      if force_download or not _local_copy_is_current(paths, etag):
        _download_to_tmp_and_move("""
    old_dl_tail = """            force_download=force_download,
            etag=etag,
            xet_file_data=xet_file_data,
            tqdm_class=tqdm_class,
        )

    write_download_metadata(local_dir=local_dir, filename=filename, commit_hash=commit_hash, etag=etag)
    return str(paths.file_path)"""
    new_dl_tail = """            force_download=True,
            etag=etag,
            xet_file_data=xet_file_data,
            tqdm_class=tqdm_class,
        )
      _write_download_metadata_locked(paths, commit_hash, etag)
    return str(paths.file_path)"""
    for old, new in ((old_copy, new_copy), (old_dl_head, new_dl_head), (old_dl_tail, new_dl_tail)):
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = dict(vars(fd))
    ns.update(
        _local_copy_is_current=_local_copy_is_current,
        _write_download_metadata_locked=_write_download_metadata_locked,
        _copy_atomically=_copy_atomically,
    )
    exec(compile(src, fd.__file__ + " (fixed)", "exec"), ns)
    fd._hf_hub_download_to_local_dir = ns["_hf_hub_download_to_local_dir"]
