--------------------------- MODULE UploadPipeline ---------------------------
(***************************************************************************)
(* upload_folder with hf_xet installed (the default): the streamed multi- *)
(* commit pipeline that replaced upload_large_folder.                      *)
(* Code: src/huggingface_hub/_upload_pipeline.py (main @ 2c61ae8)          *)
(*   run L447-476, _coordinator_loop L480-538, _enqueue L540-550,          *)
(*   _committer_loop L561-579, _process_batch/_commit_with_split L581-622  *)
(*   utils/_xet.py: XetSessionHolder / abort_xet_session L254-331          *)
(* Actors: coordinator (caller thread), committer thread, and Other: an    *)
(* unrelated xet transfer in the same process (hf_hub_download,            *)
(* snapshot_download, HfFileSystem.get_file, another upload...) using the  *)
(* process-wide XetSession.                                                *)
(* The xet session is a generation counter: abort_xet_session() cancels    *)
(* every in-flight operation of the current session (sigint_abort) and the *)
(* next get_xet_session() creates a new one.                               *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS NB,          \* number of batches the coordinator builds
          CoordFail,   \* the coordinator may raise (preupload HTTP error, KeyboardInterrupt, ...)
          CommitFail,  \* a commit may fail (committer error path)
          Fixed        \* TRUE: abort only this pipeline's xet commits (batch.xet_commit.abort())

SENTINEL == 0
(* --algorithm Pipeline
variables
  q = <<>>,                          \* batch_queue (maxsize=1)
  abort = FALSE,                     \* abort_event
  errors = 0,                        \* len(self.errors)
  committed = [b \in 1..NB |-> 0],   \* number of git commits containing batch b
  xetAborted = [b \in 1..NB |-> FALSE],
  sessionGen = 0,                    \* XetSession generation (bumped by abort_xet_session)
  committerDone = FALSE,
  raised = FALSE,                    \* run() raised to the caller
  otherResult = "running";           \* outcome of the unrelated xet transfer

define
  NoDoubleCommit == \A b \in 1..NB : committed[b] <= 1
  \* an unrelated transfer only fails for its own reasons (none here)
  OtherUnaffected == otherResult # "cancelled"
  \* run() returning normally means every batch was committed
  CompleteOnSuccess == (pc["coord"] = "Done" /\ ~raised) => \A b \in 1..NB : committed[b] = 1
  QueueBound == Len(q) <= 1
end define;

fair process coordinator = "coord"
variables i = 1;
begin
Loop:
  while i <= NB do
    if abort then
      xetAborted[i] := TRUE;           \* _abort_batch(batch); return  (L485-487)
      goto Finally;
    end if;
Preupload:                             \* L489-504 _fetch_upload_modes, can raise
    either
      skip;
    or
      await CoordFail;
      goto Except;
    end either;
Enqueue:                               \* L536 -> _enqueue L544-550: put(timeout=1) while not abort
    await Len(q) = 0 \/ abort;
    if abort then xetAborted[i] := TRUE; else q := Append(q, i); end if;
    i := i + 1;
  end while;
  goto Finally;
Except:                                \* L454-457 except BaseException
  abort := TRUE;
  if Fixed then
    xetAborted[i] := TRUE;             \* FIX: abort only this pipeline's xet commit(s)
  else
    sessionGen := sessionGen + 1;      \* abort_xet_session(): process-wide
  end if;
  raised := TRUE;
Finally:                               \* L458-467
  if abort then
    either await committerDone; or skip; end either;   \* committer.join(timeout=10)
  else
PutSentinel:
    await Len(q) = 0;                  \* batch_queue.put(_SENTINEL): blocking, no timeout
    q := Append(q, SENTINEL);
Join:
    await committerDone;               \* committer.join()
  end if;
CheckErrors:                           \* L474-475
  if errors > 0 then raised := TRUE; end if;
end process;

fair process committer = "committer"
variables kb = 0;
begin
Get:
  while TRUE do
    await Len(q) > 0 \/ abort;         \* get(timeout=0.5); on Empty: return if aborted
    if Len(q) = 0 then
      committerDone := TRUE; goto Done;
    elsif Head(q) = SENTINEL then
      q := Tail(q); committerDone := TRUE; goto Done;
    else
      kb := Head(q); q := Tail(q);
    end if;
Process:
    if abort then
      xetAborted[kb] := TRUE;           \* _abort_batch
    else
      either
        committed[kb] := committed[kb] + 1;          \* wait_to_finish + _send_commit OK
      or
        await CommitFail;
        xetAborted[kb] := TRUE; errors := errors + 1; abort := TRUE;   \* L576-579
      end either;
    end if;
  end while;
end process;

fair process other = "other"
variables gen = 0;
begin
OStart:
  gen := sessionGen;                   \* get_xet_session() + start the transfer
OFinish:
  if gen = sessionGen then otherResult := "ok" else otherResult := "cancelled" end if;
end process;
end algorithm; *)
\* BEGIN TRANSLATION
VARIABLES pc, q, abort, errors, committed, xetAborted, sessionGen, 
          committerDone, raised, otherResult

(* define statement *)
NoDoubleCommit == \A b \in 1..NB : committed[b] <= 1

OtherUnaffected == otherResult # "cancelled"

CompleteOnSuccess == (pc["coord"] = "Done" /\ ~raised) => \A b \in 1..NB : committed[b] = 1
QueueBound == Len(q) <= 1

VARIABLES i, kb, gen

vars == << pc, q, abort, errors, committed, xetAborted, sessionGen, 
           committerDone, raised, otherResult, i, kb, gen >>

ProcSet == {"coord"} \cup {"committer"} \cup {"other"}

Init == (* Global variables *)
        /\ q = <<>>
        /\ abort = FALSE
        /\ errors = 0
        /\ committed = [b \in 1..NB |-> 0]
        /\ xetAborted = [b \in 1..NB |-> FALSE]
        /\ sessionGen = 0
        /\ committerDone = FALSE
        /\ raised = FALSE
        /\ otherResult = "running"
        (* Process coordinator *)
        /\ i = 1
        (* Process committer *)
        /\ kb = 0
        (* Process other *)
        /\ gen = 0
        /\ pc = [self \in ProcSet |-> CASE self = "coord" -> "Loop"
                                        [] self = "committer" -> "Get"
                                        [] self = "other" -> "OStart"]

Loop == /\ pc["coord"] = "Loop"
        /\ IF i <= NB
              THEN /\ IF abort
                         THEN /\ xetAborted' = [xetAborted EXCEPT ![i] = TRUE]
                              /\ pc' = [pc EXCEPT !["coord"] = "Finally"]
                         ELSE /\ pc' = [pc EXCEPT !["coord"] = "Preupload"]
                              /\ UNCHANGED xetAborted
              ELSE /\ pc' = [pc EXCEPT !["coord"] = "Finally"]
                   /\ UNCHANGED xetAborted
        /\ UNCHANGED << q, abort, errors, committed, sessionGen, committerDone, 
                        raised, otherResult, i, kb, gen >>

Preupload == /\ pc["coord"] = "Preupload"
             /\ \/ /\ TRUE
                   /\ pc' = [pc EXCEPT !["coord"] = "Enqueue"]
                \/ /\ CoordFail
                   /\ pc' = [pc EXCEPT !["coord"] = "Except"]
             /\ UNCHANGED << q, abort, errors, committed, xetAborted, 
                             sessionGen, committerDone, raised, otherResult, i, 
                             kb, gen >>

Enqueue == /\ pc["coord"] = "Enqueue"
           /\ Len(q) = 0 \/ abort
           /\ IF abort
                 THEN /\ xetAborted' = [xetAborted EXCEPT ![i] = TRUE]
                      /\ q' = q
                 ELSE /\ q' = Append(q, i)
                      /\ UNCHANGED xetAborted
           /\ i' = i + 1
           /\ pc' = [pc EXCEPT !["coord"] = "Loop"]
           /\ UNCHANGED << abort, errors, committed, sessionGen, committerDone, 
                           raised, otherResult, kb, gen >>

Except == /\ pc["coord"] = "Except"
          /\ abort' = TRUE
          /\ IF Fixed
                THEN /\ xetAborted' = [xetAborted EXCEPT ![i] = TRUE]
                     /\ UNCHANGED sessionGen
                ELSE /\ sessionGen' = sessionGen + 1
                     /\ UNCHANGED xetAborted
          /\ raised' = TRUE
          /\ pc' = [pc EXCEPT !["coord"] = "Finally"]
          /\ UNCHANGED << q, errors, committed, committerDone, otherResult, i, 
                          kb, gen >>

Finally == /\ pc["coord"] = "Finally"
           /\ IF abort
                 THEN /\ \/ /\ committerDone
                         \/ /\ TRUE
                      /\ pc' = [pc EXCEPT !["coord"] = "CheckErrors"]
                 ELSE /\ pc' = [pc EXCEPT !["coord"] = "PutSentinel"]
           /\ UNCHANGED << q, abort, errors, committed, xetAborted, sessionGen, 
                           committerDone, raised, otherResult, i, kb, gen >>

PutSentinel == /\ pc["coord"] = "PutSentinel"
               /\ Len(q) = 0
               /\ q' = Append(q, SENTINEL)
               /\ pc' = [pc EXCEPT !["coord"] = "Join"]
               /\ UNCHANGED << abort, errors, committed, xetAborted, 
                               sessionGen, committerDone, raised, otherResult, 
                               i, kb, gen >>

Join == /\ pc["coord"] = "Join"
        /\ committerDone
        /\ pc' = [pc EXCEPT !["coord"] = "CheckErrors"]
        /\ UNCHANGED << q, abort, errors, committed, xetAborted, sessionGen, 
                        committerDone, raised, otherResult, i, kb, gen >>

CheckErrors == /\ pc["coord"] = "CheckErrors"
               /\ IF errors > 0
                     THEN /\ raised' = TRUE
                     ELSE /\ TRUE
                          /\ UNCHANGED raised
               /\ pc' = [pc EXCEPT !["coord"] = "Done"]
               /\ UNCHANGED << q, abort, errors, committed, xetAborted, 
                               sessionGen, committerDone, otherResult, i, kb, 
                               gen >>

coordinator == Loop \/ Preupload \/ Enqueue \/ Except \/ Finally
                  \/ PutSentinel \/ Join \/ CheckErrors

Get == /\ pc["committer"] = "Get"
       /\ Len(q) > 0 \/ abort
       /\ IF Len(q) = 0
             THEN /\ committerDone' = TRUE
                  /\ pc' = [pc EXCEPT !["committer"] = "Done"]
                  /\ UNCHANGED << q, kb >>
             ELSE /\ IF Head(q) = SENTINEL
                        THEN /\ q' = Tail(q)
                             /\ committerDone' = TRUE
                             /\ pc' = [pc EXCEPT !["committer"] = "Done"]
                             /\ kb' = kb
                        ELSE /\ kb' = Head(q)
                             /\ q' = Tail(q)
                             /\ pc' = [pc EXCEPT !["committer"] = "Process"]
                             /\ UNCHANGED committerDone
       /\ UNCHANGED << abort, errors, committed, xetAborted, sessionGen, 
                       raised, otherResult, i, gen >>

Process == /\ pc["committer"] = "Process"
           /\ IF abort
                 THEN /\ xetAborted' = [xetAborted EXCEPT ![kb] = TRUE]
                      /\ UNCHANGED << abort, errors, committed >>
                 ELSE /\ \/ /\ committed' = [committed EXCEPT ![kb] = committed[kb] + 1]
                            /\ UNCHANGED <<abort, errors, xetAborted>>
                         \/ /\ CommitFail
                            /\ xetAborted' = [xetAborted EXCEPT ![kb] = TRUE]
                            /\ errors' = errors + 1
                            /\ abort' = TRUE
                            /\ UNCHANGED committed
           /\ pc' = [pc EXCEPT !["committer"] = "Get"]
           /\ UNCHANGED << q, sessionGen, committerDone, raised, otherResult, 
                           i, kb, gen >>

committer == Get \/ Process

OStart == /\ pc["other"] = "OStart"
          /\ gen' = sessionGen
          /\ pc' = [pc EXCEPT !["other"] = "OFinish"]
          /\ UNCHANGED << q, abort, errors, committed, xetAborted, sessionGen, 
                          committerDone, raised, otherResult, i, kb >>

OFinish == /\ pc["other"] = "OFinish"
           /\ IF gen = sessionGen
                 THEN /\ otherResult' = "ok"
                 ELSE /\ otherResult' = "cancelled"
           /\ pc' = [pc EXCEPT !["other"] = "Done"]
           /\ UNCHANGED << q, abort, errors, committed, xetAborted, sessionGen, 
                           committerDone, raised, i, kb, gen >>

other == OStart \/ OFinish

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == coordinator \/ committer \/ other
           \/ Terminating

Spec == /\ Init /\ [][Next]_vars
        /\ WF_vars(coordinator)
        /\ WF_vars(committer)
        /\ WF_vars(other)

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION
CoordTerminates == <>(pc["coord"] = "Done" /\ pc["other"] = "Done")
\* without an abort the committer always finishes too
CommitterTerminates == <>(abort \/ pc["committer"] = "Done")
=============================================================================
