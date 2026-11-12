------------------------------ MODULE LocalDir ------------------------------
(***************************************************************************)
(* hf_hub_download(..., local_dir=D) for ONE repo file, several processes. *)
(* Code: src/huggingface_hub/file_download.py (main @ 2c61ae8)             *)
(*   _hf_hub_download_to_local_dir  L1313-1519                             *)
(*   _download_to_tmp_and_move      L1969-2046 (unique tmp + atomic move)  *)
(* and src/huggingface_hub/_local_folder.py                                *)
(*   read_download_metadata  L296-344 (under lock, mtime check L335-339)   *)
(*   write_download_metadata L432-442 (under lock)                         *)
(* The lock is WeakFileLock(<metadata>.lock), used by these three regions. *)
(*                                                                         *)
(* Versions 1..2 stand for (commit, etag, content) triples: a process      *)
(* requesting version v resolves commit_v/etag_v and downloads content_v.  *)
(* mtime: logical clock `now` (incremented by every file/metadata write);  *)
(* the code's 1s slack (st_mtime - 1 <= ts) is ignored (downloads >1s).    *)
(* Copy-from-cache path (L1467-1489) is modelled when CacheHit = TRUE:     *)
(* shutil.copyfile writes the final path in place (truncate + write).      *)
(* Each lock-held region whose body has no other shared effect is one step *)
(* (read_download_metadata, write_download_metadata).                      *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Procs,      \* processes calling hf_hub_download(local_dir=D)
          Want,       \* Want[p] = version (commit/etag) process p resolves to
          LFS,        \* TRUE: etag is a sha256 (LFS/xet file) -> hash check at L1448
          CacheHit,   \* TRUE: file also in the HF cache -> copy path L1467-1489
          Fixed       \* TRUE: proposed fix (see FIX comments)

NONE == "none"
SameVersion == \A p, q \in Procs : Want[p] = Want[q]
(* file content: 0 = absent, v > 0 = complete content of version v,
                 v < 0 = truncated/partial content of version -v *)
(* --algorithm LocalDir
variables
  file = 0, fmtime = 0,               \* D/<filename> and its mtime
  meta = [v |-> 0, ts |-> 0],         \* .cache/huggingface/download/<f>.metadata (v=0: absent)
  lock = NONE,                        \* WeakFileLock(<f>.lock)
  now = 1,
  gets = [v \in 1..2 |-> 0],          \* GET requests per version (bandwidth)
  ret = [p \in Procs |-> 0];          \* version hf_hub_download returned to p (0 = not returned yet)

define
  MetaValid == meta.v # 0 /\ file # 0 /\ fmtime <= meta.ts
  \* (1) once some call returned D/<f> for v, D/<f> keeps holding v (all calls want the same v)
  ReturnedStaysValid == SameVersion => \A p \in Procs : ret[p] # 0 => file = ret[p]
  \* (2) a same-version run downloads the file at most once
  NoRedundantGet == SameVersion => \A v \in 1..2 : gets[v] <= 1
  \* (3) metadata that passes the mtime check describes the content on disk
  MetaSound == MetaValid => meta.v = file
  \* (3') ... also once every call has returned (a lasting, not transient, mismatch)
  TerminalMetaSound == (\A p \in Procs : pc[p] = "Done") => MetaSound
end define;

fair process proc \in Procs
variables lm = 0, myv = 0;
begin
Start:
  myv := Want[self];
ReadMeta:                                  \* L1340 read_download_metadata (lock-held region)
  await lock = NONE;
  if MetaValid then lm := meta.v else lm := 0 end if;
Shortcut:                                  \* L1343-1360 commit-hash shortcut (snapshot_download passes the commit)
  if file > 0 /\ lm = myv then ret[self] := myv; goto Done; end if;
CheckExisting:                             \* L1429-1462 (lm = stale value read at L1340)
  if file # 0 /\ (lm = myv \/ (lm = 0 /\ LFS /\ file = myv)) then
    goto WriteMeta;                        \* etag match (L1431) or sha256(file) == etag (L1448-1451)
  end if;
Acquire:                                   \* L1485 / L1503 WeakFileLock(paths.lock_path)
  await lock = NONE;
  lock := self;
Recheck:
  if Fixed /\ file = myv then              \* FIX 1: double-check under the lock (as the cache_dir path does)
    goto WriteMetaLocked;
  elsif CacheHit then
    goto CopyTrunc;
  end if;
Unlink:                                    \* L1504 paths.file_path.unlink(missing_ok=True)
  if ~Fixed then file := 0; end if;        \* FIX 2: no unlink, the move below replaces atomically
Download:                                  \* L1505 -> _download_to_tmp_and_move -> http_get/xet_get into a unique tmp
  gets[myv] := gets[myv] + 1;
Move:                                      \* L2042 _chmod_and_move (os.replace / rename: atomic)
  file := myv; fmtime := now; now := now + 1;
  goto Release;
CopyTrunc:                                 \* L1487 shutil.copyfile(cached_path, file_path): open(dst,'wb') truncates in place
  if ~Fixed then file := -myv; fmtime := now; now := now + 1; end if;   \* FIX 3: copy to tmp + os.replace
CopyDone:
  file := myv; fmtime := now; now := now + 1;
Release:
  if Fixed then goto WriteMetaLocked; else lock := NONE; end if;  \* FIX 4: write metadata before releasing the lock
WriteMeta:                                 \* L1432/1452/1488/1518 write_download_metadata (lock-held region)
  await lock = NONE;
  if Fixed /\ file # myv then goto Acquire;           \* FIX 4': re-validate under the lock
  else meta := [v |-> myv, ts |-> now]; now := now + 1; ret[self] := myv; goto Done;
  end if;
WriteMetaLocked:
  meta := [v |-> myv, ts |-> now]; now := now + 1;
  ret[self] := myv;
  lock := NONE;
end process;
end algorithm; *)
\* BEGIN TRANSLATION
VARIABLES pc, file, fmtime, meta, lock, now, gets, ret

(* define statement *)
MetaValid == meta.v # 0 /\ file # 0 /\ fmtime <= meta.ts

ReturnedStaysValid == SameVersion => \A p \in Procs : ret[p] # 0 => file = ret[p]

NoRedundantGet == SameVersion => \A v \in 1..2 : gets[v] <= 1

MetaSound == MetaValid => meta.v = file

TerminalMetaSound == (\A p \in Procs : pc[p] = "Done") => MetaSound

VARIABLES lm, myv

vars == << pc, file, fmtime, meta, lock, now, gets, ret, lm, myv >>

ProcSet == (Procs)

Init == (* Global variables *)
        /\ file = 0
        /\ fmtime = 0
        /\ meta = [v |-> 0, ts |-> 0]
        /\ lock = NONE
        /\ now = 1
        /\ gets = [v \in 1..2 |-> 0]
        /\ ret = [p \in Procs |-> 0]
        (* Process proc *)
        /\ lm = [self \in Procs |-> 0]
        /\ myv = [self \in Procs |-> 0]
        /\ pc = [self \in ProcSet |-> "Start"]

Start(self) == /\ pc[self] = "Start"
               /\ myv' = [myv EXCEPT ![self] = Want[self]]
               /\ pc' = [pc EXCEPT ![self] = "ReadMeta"]
               /\ UNCHANGED << file, fmtime, meta, lock, now, gets, ret, lm >>

ReadMeta(self) == /\ pc[self] = "ReadMeta"
                  /\ lock = NONE
                  /\ IF MetaValid
                        THEN /\ lm' = [lm EXCEPT ![self] = meta.v]
                        ELSE /\ lm' = [lm EXCEPT ![self] = 0]
                  /\ pc' = [pc EXCEPT ![self] = "Shortcut"]
                  /\ UNCHANGED << file, fmtime, meta, lock, now, gets, ret, 
                                  myv >>

Shortcut(self) == /\ pc[self] = "Shortcut"
                  /\ IF file > 0 /\ lm[self] = myv[self]
                        THEN /\ ret' = [ret EXCEPT ![self] = myv[self]]
                             /\ pc' = [pc EXCEPT ![self] = "Done"]
                        ELSE /\ pc' = [pc EXCEPT ![self] = "CheckExisting"]
                             /\ ret' = ret
                  /\ UNCHANGED << file, fmtime, meta, lock, now, gets, lm, myv >>

CheckExisting(self) == /\ pc[self] = "CheckExisting"
                       /\ IF file # 0 /\ (lm[self] = myv[self] \/ (lm[self] = 0 /\ LFS /\ file = myv[self]))
                             THEN /\ pc' = [pc EXCEPT ![self] = "WriteMeta"]
                             ELSE /\ pc' = [pc EXCEPT ![self] = "Acquire"]
                       /\ UNCHANGED << file, fmtime, meta, lock, now, gets, 
                                       ret, lm, myv >>

Acquire(self) == /\ pc[self] = "Acquire"
                 /\ lock = NONE
                 /\ lock' = self
                 /\ pc' = [pc EXCEPT ![self] = "Recheck"]
                 /\ UNCHANGED << file, fmtime, meta, now, gets, ret, lm, myv >>

Recheck(self) == /\ pc[self] = "Recheck"
                 /\ IF Fixed /\ file = myv[self]
                       THEN /\ pc' = [pc EXCEPT ![self] = "WriteMetaLocked"]
                       ELSE /\ IF CacheHit
                                  THEN /\ pc' = [pc EXCEPT ![self] = "CopyTrunc"]
                                  ELSE /\ pc' = [pc EXCEPT ![self] = "Unlink"]
                 /\ UNCHANGED << file, fmtime, meta, lock, now, gets, ret, lm, 
                                 myv >>

Unlink(self) == /\ pc[self] = "Unlink"
                /\ IF ~Fixed
                      THEN /\ file' = 0
                      ELSE /\ TRUE
                           /\ file' = file
                /\ pc' = [pc EXCEPT ![self] = "Download"]
                /\ UNCHANGED << fmtime, meta, lock, now, gets, ret, lm, myv >>

Download(self) == /\ pc[self] = "Download"
                  /\ gets' = [gets EXCEPT ![myv[self]] = gets[myv[self]] + 1]
                  /\ pc' = [pc EXCEPT ![self] = "Move"]
                  /\ UNCHANGED << file, fmtime, meta, lock, now, ret, lm, myv >>

Move(self) == /\ pc[self] = "Move"
              /\ file' = myv[self]
              /\ fmtime' = now
              /\ now' = now + 1
              /\ pc' = [pc EXCEPT ![self] = "Release"]
              /\ UNCHANGED << meta, lock, gets, ret, lm, myv >>

CopyTrunc(self) == /\ pc[self] = "CopyTrunc"
                   /\ IF ~Fixed
                         THEN /\ file' = -myv[self]
                              /\ fmtime' = now
                              /\ now' = now + 1
                         ELSE /\ TRUE
                              /\ UNCHANGED << file, fmtime, now >>
                   /\ pc' = [pc EXCEPT ![self] = "CopyDone"]
                   /\ UNCHANGED << meta, lock, gets, ret, lm, myv >>

CopyDone(self) == /\ pc[self] = "CopyDone"
                  /\ file' = myv[self]
                  /\ fmtime' = now
                  /\ now' = now + 1
                  /\ pc' = [pc EXCEPT ![self] = "Release"]
                  /\ UNCHANGED << meta, lock, gets, ret, lm, myv >>

Release(self) == /\ pc[self] = "Release"
                 /\ IF Fixed
                       THEN /\ pc' = [pc EXCEPT ![self] = "WriteMetaLocked"]
                            /\ lock' = lock
                       ELSE /\ lock' = NONE
                            /\ pc' = [pc EXCEPT ![self] = "WriteMeta"]
                 /\ UNCHANGED << file, fmtime, meta, now, gets, ret, lm, myv >>

WriteMeta(self) == /\ pc[self] = "WriteMeta"
                   /\ lock = NONE
                   /\ IF Fixed /\ file # myv[self]
                         THEN /\ pc' = [pc EXCEPT ![self] = "Acquire"]
                              /\ UNCHANGED << meta, now, ret >>
                         ELSE /\ meta' = [v |-> myv[self], ts |-> now]
                              /\ now' = now + 1
                              /\ ret' = [ret EXCEPT ![self] = myv[self]]
                              /\ pc' = [pc EXCEPT ![self] = "Done"]
                   /\ UNCHANGED << file, fmtime, lock, gets, lm, myv >>

WriteMetaLocked(self) == /\ pc[self] = "WriteMetaLocked"
                         /\ meta' = [v |-> myv[self], ts |-> now]
                         /\ now' = now + 1
                         /\ ret' = [ret EXCEPT ![self] = myv[self]]
                         /\ lock' = NONE
                         /\ pc' = [pc EXCEPT ![self] = "Done"]
                         /\ UNCHANGED << file, fmtime, gets, lm, myv >>

proc(self) == Start(self) \/ ReadMeta(self) \/ Shortcut(self)
                 \/ CheckExisting(self) \/ Acquire(self) \/ Recheck(self)
                 \/ Unlink(self) \/ Download(self) \/ Move(self)
                 \/ CopyTrunc(self) \/ CopyDone(self) \/ Release(self)
                 \/ WriteMeta(self) \/ WriteMetaLocked(self)

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == (\E self \in Procs: proc(self))
           \/ Terminating

Spec == /\ Init /\ [][Next]_vars
        /\ \A self \in Procs : WF_vars(proc(self))

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION
=============================================================================
