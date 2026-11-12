------------------------------ MODULE Extract ------------------------------
(***************************************************************************)
(* Concurrent extraction of the same archive into the shared cache         *)
(* (datasets main @ efc8247, src/datasets/utils/extract.py).               *)
(*                                                                         *)
(*  ExtractManager.extract (lines 47-54): _do_extract (lines 42-45) checks *)
(*    `not isfile(out) and not (isdir(out) and listdir(out))` WITHOUT the  *)
(*    lock; if false the path is returned and the caller reads it.         *)
(*  Extractor.extract (lines 448-462): under FileLock(out.lock):           *)
(*    rmtree(out) (unconditionally, no re-check), then the extractor       *)
(*    writes in place: tar/zip create the directory then add members       *)
(*    (TarExtractor 132-136), gzip opens out with "wb" then copies         *)
(*    (GzipExtractor 143-146).                                             *)
(* Readers (the dataset builder generating examples from the extracted     *)
(* files) do not take the lock.                                            *)
(*                                                                         *)
(* Members = number of extracted members (or gzip chunks).  `ver` counts   *)
(* rewrites so that a reader can detect that what it is reading changed.   *)
(* Invariant NoTornRead: a reader only ever reads a complete extraction    *)
(* that is not deleted/rewritten while it reads.  Fixed = TRUE: re-check   *)
(* under the lock and extract to a temp path + rename.                     *)
(***************************************************************************)
EXTENDS Naturals, TLC

CONSTANTS Procs, Members, Fixed

(* --algorithm Extract
variables
    out = 0,             \* members present at the output path (0 = absent / empty dir)
    complete = FALSE,    \* extraction of the output path finished
    ver = 0,             \* incremented by every rmtree/rewrite of the output path
    lock = "free",
    torn = FALSE;

define
    NoTornRead == ~torn
    Visible == complete \/ out > 0      \* isfile(out) or a non-empty dir (_do_extract is False)
end define;

process P \in Procs
variables seen = 0, k = 0;
begin
Check:
    if Visible then goto ReadStart; end if;          \* _do_extract False: return out (lines 52-54)
Acquire:
    await lock = "free"; lock := self;              \* FileLock(lock_path) (line 458)
Recheck:
    if Fixed /\ complete then lock := "free"; goto ReadStart; end if;
Rm:
    if ~Fixed then
        out := 0; complete := FALSE; ver := ver + 1;  \* rmtree(output_path) (lines 459-461)
    end if;
    k := 0;
Write:
    while k < Members do
        k := k + 1;
        if ~Fixed then out := k; end if;              \* in-place: members appear one by one
    end while;
Finish:
    if Fixed then out := Members; ver := ver + 1; end if;  \* fix: tmp dir + atomic rename
    complete := TRUE; lock := "free";
ReadStart:
    if ~complete then torn := TRUE; end if;          \* reads a partially extracted output
    seen := ver;
ReadEnd:
    if ver # seen \/ ~complete then torn := TRUE; end if;  \* output deleted/rewritten under the reader
end process;
end algorithm; *)
\* BEGIN TRANSLATION (chksum(pcal) = "7c787bc5" /\ chksum(tla) = "f1048f08")
VARIABLES pc, out, complete, ver, lock, torn

(* define statement *)
NoTornRead == ~torn
Visible == complete \/ out > 0

VARIABLES seen, k

vars == << pc, out, complete, ver, lock, torn, seen, k >>

ProcSet == (Procs)

Init == (* Global variables *)
        /\ out = 0
        /\ complete = FALSE
        /\ ver = 0
        /\ lock = "free"
        /\ torn = FALSE
        (* Process P *)
        /\ seen = [self \in Procs |-> 0]
        /\ k = [self \in Procs |-> 0]
        /\ pc = [self \in ProcSet |-> "Check"]

Check(self) == /\ pc[self] = "Check"
               /\ IF Visible
                     THEN /\ pc' = [pc EXCEPT ![self] = "ReadStart"]
                     ELSE /\ pc' = [pc EXCEPT ![self] = "Acquire"]
               /\ UNCHANGED << out, complete, ver, lock, torn, seen, k >>

Acquire(self) == /\ pc[self] = "Acquire"
                 /\ lock = "free"
                 /\ lock' = self
                 /\ pc' = [pc EXCEPT ![self] = "Recheck"]
                 /\ UNCHANGED << out, complete, ver, torn, seen, k >>

Recheck(self) == /\ pc[self] = "Recheck"
                 /\ IF Fixed /\ complete
                       THEN /\ lock' = "free"
                            /\ pc' = [pc EXCEPT ![self] = "ReadStart"]
                       ELSE /\ pc' = [pc EXCEPT ![self] = "Rm"]
                            /\ lock' = lock
                 /\ UNCHANGED << out, complete, ver, torn, seen, k >>

Rm(self) == /\ pc[self] = "Rm"
            /\ IF ~Fixed
                  THEN /\ out' = 0
                       /\ complete' = FALSE
                       /\ ver' = ver + 1
                  ELSE /\ TRUE
                       /\ UNCHANGED << out, complete, ver >>
            /\ k' = [k EXCEPT ![self] = 0]
            /\ pc' = [pc EXCEPT ![self] = "Write"]
            /\ UNCHANGED << lock, torn, seen >>

Write(self) == /\ pc[self] = "Write"
               /\ IF k[self] < Members
                     THEN /\ k' = [k EXCEPT ![self] = k[self] + 1]
                          /\ IF ~Fixed
                                THEN /\ out' = k'[self]
                                ELSE /\ TRUE
                                     /\ out' = out
                          /\ pc' = [pc EXCEPT ![self] = "Write"]
                     ELSE /\ pc' = [pc EXCEPT ![self] = "Finish"]
                          /\ UNCHANGED << out, k >>
               /\ UNCHANGED << complete, ver, lock, torn, seen >>

Finish(self) == /\ pc[self] = "Finish"
                /\ IF Fixed
                      THEN /\ out' = Members
                           /\ ver' = ver + 1
                      ELSE /\ TRUE
                           /\ UNCHANGED << out, ver >>
                /\ complete' = TRUE
                /\ lock' = "free"
                /\ pc' = [pc EXCEPT ![self] = "ReadStart"]
                /\ UNCHANGED << torn, seen, k >>

ReadStart(self) == /\ pc[self] = "ReadStart"
                   /\ IF ~complete
                         THEN /\ torn' = TRUE
                         ELSE /\ TRUE
                              /\ torn' = torn
                   /\ seen' = [seen EXCEPT ![self] = ver]
                   /\ pc' = [pc EXCEPT ![self] = "ReadEnd"]
                   /\ UNCHANGED << out, complete, ver, lock, k >>

ReadEnd(self) == /\ pc[self] = "ReadEnd"
                 /\ IF ver # seen[self] \/ ~complete
                       THEN /\ torn' = TRUE
                       ELSE /\ TRUE
                            /\ torn' = torn
                 /\ pc' = [pc EXCEPT ![self] = "Done"]
                 /\ UNCHANGED << out, complete, ver, lock, seen, k >>

P(self) == Check(self) \/ Acquire(self) \/ Recheck(self) \/ Rm(self)
              \/ Write(self) \/ Finish(self) \/ ReadStart(self)
              \/ ReadEnd(self)

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == (\E self \in Procs: P(self))
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION 
===============================================================================
