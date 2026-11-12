------------------------------- MODULE Cache -------------------------------
(***************************************************************************)
(* hf_hub_download into the shared cache (default path of from_pretrained),*)
(* several processes, one file whose etag E is the same at commits C1, C2  *)
(* (file unchanged between the two commits).                               *)
(* Code: src/huggingface_hub/file_download.py (main @ 2c61ae8)             *)
(*   _hf_hub_download_to_cache_dir L1068-1310, _create_symlink L633-724,   *)
(*   _download_to_tmp_and_move L1969-2046 (unique tmp, atomic move)        *)
(*   utils/_shared_blobs.py try_link_from_shared_store L386-418,           *)
(*   publish_blob_to_shared_store L432-484 (store lock held, atomic steps) *)
(*                                                                         *)
(* Symlinks = FALSE models are_symlinks_supported() == False (Windows      *)
(* without developer mode, HF_HUB_DISABLE_SYMLINKS=1): _create_symlink     *)
(* then moves the blob (new_blob=True, L721) or copies it (new_blob=False, *)
(* L724: shutil.copyfile straight into the snapshot path).                 *)
(* Shared = TRUE models the cache-wide shared blob store (xet files,       *)
(* default on, needs symlinks).                                            *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Procs, Commit, Symlinks, Shared

NONE == "none"
Commits == {"C1", "C2"}
(* blob: "absent" | "file" (complete regular file) | "link" (symlink into the store)
   store: "absent" | "file"
   ptr[c]: "absent" | "link" (symlink to blobs/E) | "partial" | "file" (complete copy) *)
(* --algorithm Cache
variables
  blob = "absent", store = "absent",
  ptr = [c \in Commits |-> "absent"],
  elock = NONE,                 \* WeakFileLock(.locks/<repo>/<etag>.lock)
  gets = 0,                     \* GET requests (downloads) of E
  got = [p \in Procs |-> NONE]; \* what the caller of hf_hub_download read at the returned path

define
  Resolve(c) == CASE ptr[c] = "link" -> (IF blob = "file" THEN "ok"
                                          ELSE IF blob = "link" /\ store = "file" THEN "ok" ELSE "dangling")
                  [] ptr[c] = "file" -> "ok"
                  [] ptr[c] = "partial" -> "partial"
                  [] OTHER -> "absent"
  Exists(c) == ptr[c] \in {"file", "partial"} \/ (ptr[c] = "link" /\ Resolve(c) = "ok")   \* os.path.exists
  \* a returned pointer path always yields the complete content
  ReturnedOk == \A p \in Procs : got[p] \in {NONE, "ok"}
  \* the lock makes concurrent calls download E once (comment at L1244-1248)
  DownloadOnce == gets <= 1
end define;

fair process proc \in Procs
variables mycommit = "C1", willdl = FALSE, reused = FALSE, shared_blob = FALSE;
begin
Start:
  mycommit := Commit[self];
Shortcut:                                  \* L1100-1113 (snapshot_download passes the commit hash) / L1226-1231
  if Exists(mycommit) then goto Use; end if;
BlobExists:                                \* L1257-1261
  if blob # "absent" then
    await elock = NONE; elock := self;
PtrFromBlob:
    if ~Exists(mycommit) then
      if Symlinks then ptr[mycommit] := "link";
      else ptr[mycommit] := "partial";            \* L724 shutil.copyfile(blob, pointer) (new_blob=False)
CopyEnd1: ptr[mycommit] := "file";
      end if;
    end if;
Rel1:
    elock := NONE; goto Use;
  end if;
Lock:                                      \* L1273
  await elock = NONE; elock := self;
TryLink:                                   \* L1274-1281 (store lock held inside: one step)
  if Shared /\ Symlinks /\ blob = "absent" /\ store = "file" then blob := "link"; reused := TRUE; end if;
Dl:                                        \* L1283-1296 -> _download_to_tmp_and_move (returns early if blob exists)
  if ~reused then
    willdl := blob = "absent";
    if blob = "absent" then gets := gets + 1; blob := "file"; end if;
Publish:                                   \* L1297-1305 publish_blob_to_shared_store, store lock held
    if Shared /\ Symlinks /\ willdl then
      if store = "absent" then store := "file"; end if;   \* os.replace(blob, store) (or keep usable entry)
      blob := "link";                                      \* os.replace(tmp_link, blob)
      shared_blob := TRUE;
    end if;
  else shared_blob := TRUE;
  end if;
MkPtr:                                     \* L1306-1308
  if ~Exists(mycommit) then
    if Symlinks then ptr[mycommit] := "link";
    elsif ~shared_blob then ptr[mycommit] := "file"; blob := "absent";   \* L721 shutil.move(blob, pointer): rename, atomic
    else ptr[mycommit] := "partial";
CopyEnd2: ptr[mycommit] := "file";
    end if;
  end if;
Rel2:
  elock := NONE;
Use:                                       \* the caller opens the returned pointer path
  got[self] := Resolve(mycommit);
end process;
end algorithm; *)
\* BEGIN TRANSLATION
VARIABLES pc, blob, store, ptr, elock, gets, got

(* define statement *)
Resolve(c) == CASE ptr[c] = "link" -> (IF blob = "file" THEN "ok"
                                        ELSE IF blob = "link" /\ store = "file" THEN "ok" ELSE "dangling")
                [] ptr[c] = "file" -> "ok"
                [] ptr[c] = "partial" -> "partial"
                [] OTHER -> "absent"
Exists(c) == ptr[c] \in {"file", "partial"} \/ (ptr[c] = "link" /\ Resolve(c) = "ok")

ReturnedOk == \A p \in Procs : got[p] \in {NONE, "ok"}

DownloadOnce == gets <= 1

VARIABLES mycommit, willdl, reused, shared_blob

vars == << pc, blob, store, ptr, elock, gets, got, mycommit, willdl, reused, 
           shared_blob >>

ProcSet == (Procs)

Init == (* Global variables *)
        /\ blob = "absent"
        /\ store = "absent"
        /\ ptr = [c \in Commits |-> "absent"]
        /\ elock = NONE
        /\ gets = 0
        /\ got = [p \in Procs |-> NONE]
        (* Process proc *)
        /\ mycommit = [self \in Procs |-> "C1"]
        /\ willdl = [self \in Procs |-> FALSE]
        /\ reused = [self \in Procs |-> FALSE]
        /\ shared_blob = [self \in Procs |-> FALSE]
        /\ pc = [self \in ProcSet |-> "Start"]

Start(self) == /\ pc[self] = "Start"
               /\ mycommit' = [mycommit EXCEPT ![self] = Commit[self]]
               /\ pc' = [pc EXCEPT ![self] = "Shortcut"]
               /\ UNCHANGED << blob, store, ptr, elock, gets, got, willdl, 
                               reused, shared_blob >>

Shortcut(self) == /\ pc[self] = "Shortcut"
                  /\ IF Exists(mycommit[self])
                        THEN /\ pc' = [pc EXCEPT ![self] = "Use"]
                        ELSE /\ pc' = [pc EXCEPT ![self] = "BlobExists"]
                  /\ UNCHANGED << blob, store, ptr, elock, gets, got, mycommit, 
                                  willdl, reused, shared_blob >>

BlobExists(self) == /\ pc[self] = "BlobExists"
                    /\ IF blob # "absent"
                          THEN /\ elock = NONE
                               /\ elock' = self
                               /\ pc' = [pc EXCEPT ![self] = "PtrFromBlob"]
                          ELSE /\ pc' = [pc EXCEPT ![self] = "Lock"]
                               /\ elock' = elock
                    /\ UNCHANGED << blob, store, ptr, gets, got, mycommit, 
                                    willdl, reused, shared_blob >>

PtrFromBlob(self) == /\ pc[self] = "PtrFromBlob"
                     /\ IF ~Exists(mycommit[self])
                           THEN /\ IF Symlinks
                                      THEN /\ ptr' = [ptr EXCEPT ![mycommit[self]] = "link"]
                                           /\ pc' = [pc EXCEPT ![self] = "Rel1"]
                                      ELSE /\ ptr' = [ptr EXCEPT ![mycommit[self]] = "partial"]
                                           /\ pc' = [pc EXCEPT ![self] = "CopyEnd1"]
                           ELSE /\ pc' = [pc EXCEPT ![self] = "Rel1"]
                                /\ ptr' = ptr
                     /\ UNCHANGED << blob, store, elock, gets, got, mycommit, 
                                     willdl, reused, shared_blob >>

CopyEnd1(self) == /\ pc[self] = "CopyEnd1"
                  /\ ptr' = [ptr EXCEPT ![mycommit[self]] = "file"]
                  /\ pc' = [pc EXCEPT ![self] = "Rel1"]
                  /\ UNCHANGED << blob, store, elock, gets, got, mycommit, 
                                  willdl, reused, shared_blob >>

Rel1(self) == /\ pc[self] = "Rel1"
              /\ elock' = NONE
              /\ pc' = [pc EXCEPT ![self] = "Use"]
              /\ UNCHANGED << blob, store, ptr, gets, got, mycommit, willdl, 
                              reused, shared_blob >>

Lock(self) == /\ pc[self] = "Lock"
              /\ elock = NONE
              /\ elock' = self
              /\ pc' = [pc EXCEPT ![self] = "TryLink"]
              /\ UNCHANGED << blob, store, ptr, gets, got, mycommit, willdl, 
                              reused, shared_blob >>

TryLink(self) == /\ pc[self] = "TryLink"
                 /\ IF Shared /\ Symlinks /\ blob = "absent" /\ store = "file"
                       THEN /\ blob' = "link"
                            /\ reused' = [reused EXCEPT ![self] = TRUE]
                       ELSE /\ TRUE
                            /\ UNCHANGED << blob, reused >>
                 /\ pc' = [pc EXCEPT ![self] = "Dl"]
                 /\ UNCHANGED << store, ptr, elock, gets, got, mycommit, 
                                 willdl, shared_blob >>

Dl(self) == /\ pc[self] = "Dl"
            /\ IF ~reused[self]
                  THEN /\ willdl' = [willdl EXCEPT ![self] = blob = "absent"]
                       /\ IF blob = "absent"
                             THEN /\ gets' = gets + 1
                                  /\ blob' = "file"
                             ELSE /\ TRUE
                                  /\ UNCHANGED << blob, gets >>
                       /\ pc' = [pc EXCEPT ![self] = "Publish"]
                       /\ UNCHANGED shared_blob
                  ELSE /\ shared_blob' = [shared_blob EXCEPT ![self] = TRUE]
                       /\ pc' = [pc EXCEPT ![self] = "MkPtr"]
                       /\ UNCHANGED << blob, gets, willdl >>
            /\ UNCHANGED << store, ptr, elock, got, mycommit, reused >>

Publish(self) == /\ pc[self] = "Publish"
                 /\ IF Shared /\ Symlinks /\ willdl[self]
                       THEN /\ IF store = "absent"
                                  THEN /\ store' = "file"
                                  ELSE /\ TRUE
                                       /\ store' = store
                            /\ blob' = "link"
                            /\ shared_blob' = [shared_blob EXCEPT ![self] = TRUE]
                       ELSE /\ TRUE
                            /\ UNCHANGED << blob, store, shared_blob >>
                 /\ pc' = [pc EXCEPT ![self] = "MkPtr"]
                 /\ UNCHANGED << ptr, elock, gets, got, mycommit, willdl, 
                                 reused >>

MkPtr(self) == /\ pc[self] = "MkPtr"
               /\ IF ~Exists(mycommit[self])
                     THEN /\ IF Symlinks
                                THEN /\ ptr' = [ptr EXCEPT ![mycommit[self]] = "link"]
                                     /\ pc' = [pc EXCEPT ![self] = "Rel2"]
                                     /\ blob' = blob
                                ELSE /\ IF ~shared_blob[self]
                                           THEN /\ ptr' = [ptr EXCEPT ![mycommit[self]] = "file"]
                                                /\ blob' = "absent"
                                                /\ pc' = [pc EXCEPT ![self] = "Rel2"]
                                           ELSE /\ ptr' = [ptr EXCEPT ![mycommit[self]] = "partial"]
                                                /\ pc' = [pc EXCEPT ![self] = "CopyEnd2"]
                                                /\ blob' = blob
                     ELSE /\ pc' = [pc EXCEPT ![self] = "Rel2"]
                          /\ UNCHANGED << blob, ptr >>
               /\ UNCHANGED << store, elock, gets, got, mycommit, willdl, 
                               reused, shared_blob >>

CopyEnd2(self) == /\ pc[self] = "CopyEnd2"
                  /\ ptr' = [ptr EXCEPT ![mycommit[self]] = "file"]
                  /\ pc' = [pc EXCEPT ![self] = "Rel2"]
                  /\ UNCHANGED << blob, store, elock, gets, got, mycommit, 
                                  willdl, reused, shared_blob >>

Rel2(self) == /\ pc[self] = "Rel2"
              /\ elock' = NONE
              /\ pc' = [pc EXCEPT ![self] = "Use"]
              /\ UNCHANGED << blob, store, ptr, gets, got, mycommit, willdl, 
                              reused, shared_blob >>

Use(self) == /\ pc[self] = "Use"
             /\ got' = [got EXCEPT ![self] = Resolve(mycommit[self])]
             /\ pc' = [pc EXCEPT ![self] = "Done"]
             /\ UNCHANGED << blob, store, ptr, elock, gets, mycommit, willdl, 
                             reused, shared_blob >>

proc(self) == Start(self) \/ Shortcut(self) \/ BlobExists(self)
                 \/ PtrFromBlob(self) \/ CopyEnd1(self) \/ Rel1(self)
                 \/ Lock(self) \/ TryLink(self) \/ Dl(self)
                 \/ Publish(self) \/ MkPtr(self) \/ CopyEnd2(self)
                 \/ Rel2(self) \/ Use(self)

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
