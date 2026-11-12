#!/bin/bash
# Reproducers (CPU only). Outputs of the recorded runs: out_*.txt.
# Stubbed in every script: only the Hub HTTP endpoint (fakehub.py / fakehub_commit.py / a 403 preupload server);
# huggingface_hub itself is the unpatched checkout (main @ 2c61ae8) unless APPLY_FIX=1 (validates the proposed fix).
set -u
H=$(cd "$(dirname "$0")" && pwd)   # this folder
PY=${PY:-python}            # env with huggingface_hub installed from main (2c61ae8)
cd /tmp

export HF_HUB_DISABLE_PROGRESS_BARS=1

# 1. concurrent local_dir downloads (single file, then snapshot_download in 4 "ranks")
OFFSETS=0,0.05 $PY $H/repro_localdir_concurrent.py download
ETAG_KIND=regular COPY_MB=9 OFFSETS=0,0.05 $PY $H/repro_localdir_concurrent.py copy
SEED=0 $PY $H/repro_snapshot_localdir_ddp.py
APPLY_FIX=1 SEED=0 $PY $H/repro_snapshot_localdir_ddp.py

# 2. failing upload_folder aborts a concurrent xet download (needs huggingface.co, anonymous, 268 MB)
$PY $H/repro_upload_failure_aborts_downloads.py with_failing_upload
$PY $H/repro_upload_failure_aborts_downloads.py control
APPLY_FIX=1 $PY $H/repro_upload_failure_aborts_downloads.py with_failing_upload

# 3. cache without symlinks: duplicate download + truncated snapshot file returned
$PY $H/repro_nosymlink_partial.py

# 4. CommitScheduler on a filesystem with 1 s timestamps (ext2, 128-byte inodes) vs ext4
IMG=/tmp/hfhub-repro/coarse.img; MNT=/tmp/hfhub-repro/coarsefs
mkdir -p $MNT && dd if=/dev/zero of=$IMG bs=1M count=64 status=none && mkfs.ext2 -q -I 128 -F $IMG && mount -o loop $IMG $MNT
$PY $H/repro_commit_scheduler_lost_write.py $MNT/watched /tmp/hfhub-repro/finefs_watched
umount $MNT; rm -f $IMG
