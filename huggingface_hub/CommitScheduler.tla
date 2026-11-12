--------------------------- MODULE CommitScheduler ---------------------------
(***************************************************************************)
(* CommitScheduler (src/huggingface_hub/_commit_scheduler.py, main@2c61ae8)*)
(* used as documented: `with CommitScheduler(...) as s:` and appends to a  *)
(* file in the watched folder under `with s.lock:`.                        *)
(*   trigger L181-187 -> HfApi.run_as_future (hf_api.py:2218-2221): one    *)
(*   lazily-created ThreadPoolExecutor(max_workers=1) => FIFO, serialized. *)
(*   _push_to_hub L189-202, push_to_hub L204-277 (scan under self.lock,    *)
(*   change test `last_uploaded[path] != st_mtime` L238, upload capped to  *)
(*   the scanned size, last_uploaded updated after the commit L275-276),   *)
(*   __exit__ L167-171, _run_scheduler L173-179.                           *)
(* One appended file. Content = number of appended records; remote = the   *)
(* number of records in the committed file.                                *)
(* Mtime: "fine" = every write gets a new timestamp; "coarse" = writes in  *)
(* the same clock tick share it (jiffy clock: Linux < 6.13 local fs; 1-2 s *)
(* on FAT/HFS+/some network mounts); "multigrain" = Linux >= 6.13: a write *)
(* after a stat of the inode gets a fresh fine-grained timestamp.          *)
(* LazyRace: run_as_future's `if self._thread_pool is None` check-then-set *)
(* (hf_api.py:2218-2219) is not atomic: two threads calling it first can   *)
(* create two executors (two workers).                                     *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS K,          \* records the user appends
          T,          \* scheduled triggers (bounded)
          Mtime,      \* "fine" | "coarse" | "multigrain"
          LazyRace    \* allow two executors (lazy-init race)

NONE == "none"
Workers == {"w1", "w2"}
(* --algorithm Sched
variables
  size = 0, mtime = 0, now = 0, statted = FALSE,   \* local file
  remote = 0,                                      \* committed content (records)
  lastUp = -1,                                     \* last_uploaded[path] (-1 = absent)
  stopped = FALSE,
  pools = 0,                                       \* number of executors created
  queues = [w \in Workers |-> <<>>],               \* work queues of executor 1 / 2
  doneJobs = {},
  nextJob = 1,
  ticks = 0,
  regressed = FALSE;                               \* a commit replaced the remote file by an older (shorter) one

define
  \* everything written before leaving the `with` block is on the Hub once __exit__ returned
  FinalFlush == pc["user"] = "Done" => remote = size
  \* the committed file never goes backwards
  RemoteNeverRegresses == ~regressed
end define;

macro Write() begin
  size := size + 1;
  if Mtime = "fine" \/ (Mtime = "multigrain" /\ statted) then now := now + 1; end if;
  mtime := now; statted := FALSE;
end macro;

\* HfApi.run_as_future: lazily create the executor (hf_api.py:2218-2219), then submit to
\* `self._thread_pool` read again (L2221): with the race, the second creator's executor wins.
procedure Submit()
begin
CheckPool:
  if pools = 0 then
    if ~LazyRace then pools := 1; goto DoSubmit; end if;   \* check-then-set as one step
CreatePool:
    pools := pools + 1;
  end if;
DoSubmit:
  if pools = 2 then queues["w2"] := Append(queues["w2"], nextJob);
  else queues["w1"] := Append(queues["w1"], nextJob);
  end if;
  nextJob := nextJob + 1;
  return;
end procedure;

fair process user = "user"
variables n = 0, myjob = 0;
begin
UWrite:
  while n < K do
    Write();                            \* `with s.lock: f.write(...)` (atomic w.r.t. the scan)
    n := n + 1;
  end while;
UExit:                                  \* __exit__: self.trigger().result()
  myjob := nextJob;
  call Submit();
UWait:
  await myjob \in doneJobs;
UStop:
  stopped := TRUE;
end process;

fair process sched = "sched"
variables t = 0;
begin
STrigger:                               \* _run_scheduler: first trigger right away, then every `every` minutes
  while t < T do
    t := t + 1;
    call Submit();
SSleep:
    if stopped then goto Done; end if;
  end while;
end process;

fair process clock = "clock"
begin
Tick:
  while ticks < K + T + 2 do
    ticks := ticks + 1; now := now + 1;
  end while;
end process;

fair process worker \in Workers
variables job = 0, limit = 0, scanned = 0, upload = FALSE;
begin
WGet:
  while TRUE do
    await queues[self] # <<>>;
    job := Head(queues[self]); queues[self] := Tail(queues[self]);
WPush:                                  \* _push_to_hub
    if stopped then doneJobs := doneJobs \union {job}; goto WGet; end if;
WScan:                                  \* push_to_hub: under self.lock
    statted := TRUE;
    upload := lastUp # mtime;
    limit := size; scanned := mtime;
WCommit:                                \* create_commit with PartialFileIO(size_limit=limit)
    if upload then
      if limit < remote then regressed := TRUE; end if;
      remote := limit;
    end if;
WRecord:
    if upload then lastUp := scanned; end if;
    doneJobs := doneJobs \union {job};
  end while;
end process;
end algorithm; *)
\* BEGIN TRANSLATION
VARIABLES pc, size, mtime, now, statted, remote, lastUp, stopped, pools, 
          queues, doneJobs, nextJob, ticks, regressed, stack

(* define statement *)
FinalFlush == pc["user"] = "Done" => remote = size

RemoteNeverRegresses == ~regressed

VARIABLES n, myjob, t, job, limit, scanned, upload

vars == << pc, size, mtime, now, statted, remote, lastUp, stopped, pools, 
           queues, doneJobs, nextJob, ticks, regressed, stack, n, myjob, t, 
           job, limit, scanned, upload >>

ProcSet == {"user"} \cup {"sched"} \cup {"clock"} \cup (Workers)

Init == (* Global variables *)
        /\ size = 0
        /\ mtime = 0
        /\ now = 0
        /\ statted = FALSE
        /\ remote = 0
        /\ lastUp = -1
        /\ stopped = FALSE
        /\ pools = 0
        /\ queues = [w \in Workers |-> <<>>]
        /\ doneJobs = {}
        /\ nextJob = 1
        /\ ticks = 0
        /\ regressed = FALSE
        (* Process user *)
        /\ n = 0
        /\ myjob = 0
        (* Process sched *)
        /\ t = 0
        (* Process worker *)
        /\ job = [self \in Workers |-> 0]
        /\ limit = [self \in Workers |-> 0]
        /\ scanned = [self \in Workers |-> 0]
        /\ upload = [self \in Workers |-> FALSE]
        /\ stack = [self \in ProcSet |-> << >>]
        /\ pc = [self \in ProcSet |-> CASE self = "user" -> "UWrite"
                                        [] self = "sched" -> "STrigger"
                                        [] self = "clock" -> "Tick"
                                        [] self \in Workers -> "WGet"]

CheckPool(self) == /\ pc[self] = "CheckPool"
                   /\ IF pools = 0
                         THEN /\ IF ~LazyRace
                                    THEN /\ pools' = 1
                                         /\ pc' = [pc EXCEPT ![self] = "DoSubmit"]
                                    ELSE /\ pc' = [pc EXCEPT ![self] = "CreatePool"]
                                         /\ pools' = pools
                         ELSE /\ pc' = [pc EXCEPT ![self] = "DoSubmit"]
                              /\ pools' = pools
                   /\ UNCHANGED << size, mtime, now, statted, remote, lastUp, 
                                   stopped, queues, doneJobs, nextJob, ticks, 
                                   regressed, stack, n, myjob, t, job, limit, 
                                   scanned, upload >>

CreatePool(self) == /\ pc[self] = "CreatePool"
                    /\ pools' = pools + 1
                    /\ pc' = [pc EXCEPT ![self] = "DoSubmit"]
                    /\ UNCHANGED << size, mtime, now, statted, remote, lastUp, 
                                    stopped, queues, doneJobs, nextJob, ticks, 
                                    regressed, stack, n, myjob, t, job, limit, 
                                    scanned, upload >>

DoSubmit(self) == /\ pc[self] = "DoSubmit"
                  /\ IF pools = 2
                        THEN /\ queues' = [queues EXCEPT !["w2"] = Append(queues["w2"], nextJob)]
                        ELSE /\ queues' = [queues EXCEPT !["w1"] = Append(queues["w1"], nextJob)]
                  /\ nextJob' = nextJob + 1
                  /\ pc' = [pc EXCEPT ![self] = Head(stack[self]).pc]
                  /\ stack' = [stack EXCEPT ![self] = Tail(stack[self])]
                  /\ UNCHANGED << size, mtime, now, statted, remote, lastUp, 
                                  stopped, pools, doneJobs, ticks, regressed, 
                                  n, myjob, t, job, limit, scanned, upload >>

Submit(self) == CheckPool(self) \/ CreatePool(self) \/ DoSubmit(self)

UWrite == /\ pc["user"] = "UWrite"
          /\ IF n < K
                THEN /\ size' = size + 1
                     /\ IF Mtime = "fine" \/ (Mtime = "multigrain" /\ statted)
                           THEN /\ now' = now + 1
                           ELSE /\ TRUE
                                /\ now' = now
                     /\ mtime' = now'
                     /\ statted' = FALSE
                     /\ n' = n + 1
                     /\ pc' = [pc EXCEPT !["user"] = "UWrite"]
                ELSE /\ pc' = [pc EXCEPT !["user"] = "UExit"]
                     /\ UNCHANGED << size, mtime, now, statted, n >>
          /\ UNCHANGED << remote, lastUp, stopped, pools, queues, doneJobs, 
                          nextJob, ticks, regressed, stack, myjob, t, job, 
                          limit, scanned, upload >>

UExit == /\ pc["user"] = "UExit"
         /\ myjob' = nextJob
         /\ stack' = [stack EXCEPT !["user"] = << [ procedure |->  "Submit",
                                                    pc        |->  "UWait" ] >>
                                                \o stack["user"]]
         /\ pc' = [pc EXCEPT !["user"] = "CheckPool"]
         /\ UNCHANGED << size, mtime, now, statted, remote, lastUp, stopped, 
                         pools, queues, doneJobs, nextJob, ticks, regressed, n, 
                         t, job, limit, scanned, upload >>

UWait == /\ pc["user"] = "UWait"
         /\ myjob \in doneJobs
         /\ pc' = [pc EXCEPT !["user"] = "UStop"]
         /\ UNCHANGED << size, mtime, now, statted, remote, lastUp, stopped, 
                         pools, queues, doneJobs, nextJob, ticks, regressed, 
                         stack, n, myjob, t, job, limit, scanned, upload >>

UStop == /\ pc["user"] = "UStop"
         /\ stopped' = TRUE
         /\ pc' = [pc EXCEPT !["user"] = "Done"]
         /\ UNCHANGED << size, mtime, now, statted, remote, lastUp, pools, 
                         queues, doneJobs, nextJob, ticks, regressed, stack, n, 
                         myjob, t, job, limit, scanned, upload >>

user == UWrite \/ UExit \/ UWait \/ UStop

STrigger == /\ pc["sched"] = "STrigger"
            /\ IF t < T
                  THEN /\ t' = t + 1
                       /\ stack' = [stack EXCEPT !["sched"] = << [ procedure |->  "Submit",
                                                                   pc        |->  "SSleep" ] >>
                                                               \o stack["sched"]]
                       /\ pc' = [pc EXCEPT !["sched"] = "CheckPool"]
                  ELSE /\ pc' = [pc EXCEPT !["sched"] = "Done"]
                       /\ UNCHANGED << stack, t >>
            /\ UNCHANGED << size, mtime, now, statted, remote, lastUp, stopped, 
                            pools, queues, doneJobs, nextJob, ticks, regressed, 
                            n, myjob, job, limit, scanned, upload >>

SSleep == /\ pc["sched"] = "SSleep"
          /\ IF stopped
                THEN /\ pc' = [pc EXCEPT !["sched"] = "Done"]
                ELSE /\ pc' = [pc EXCEPT !["sched"] = "STrigger"]
          /\ UNCHANGED << size, mtime, now, statted, remote, lastUp, stopped, 
                          pools, queues, doneJobs, nextJob, ticks, regressed, 
                          stack, n, myjob, t, job, limit, scanned, upload >>

sched == STrigger \/ SSleep

Tick == /\ pc["clock"] = "Tick"
        /\ IF ticks < K + T + 2
              THEN /\ ticks' = ticks + 1
                   /\ now' = now + 1
                   /\ pc' = [pc EXCEPT !["clock"] = "Tick"]
              ELSE /\ pc' = [pc EXCEPT !["clock"] = "Done"]
                   /\ UNCHANGED << now, ticks >>
        /\ UNCHANGED << size, mtime, statted, remote, lastUp, stopped, pools, 
                        queues, doneJobs, nextJob, regressed, stack, n, myjob, 
                        t, job, limit, scanned, upload >>

clock == Tick

WGet(self) == /\ pc[self] = "WGet"
              /\ queues[self] # <<>>
              /\ job' = [job EXCEPT ![self] = Head(queues[self])]
              /\ queues' = [queues EXCEPT ![self] = Tail(queues[self])]
              /\ pc' = [pc EXCEPT ![self] = "WPush"]
              /\ UNCHANGED << size, mtime, now, statted, remote, lastUp, 
                              stopped, pools, doneJobs, nextJob, ticks, 
                              regressed, stack, n, myjob, t, limit, scanned, 
                              upload >>

WPush(self) == /\ pc[self] = "WPush"
               /\ IF stopped
                     THEN /\ doneJobs' = (doneJobs \union {job[self]})
                          /\ pc' = [pc EXCEPT ![self] = "WGet"]
                     ELSE /\ pc' = [pc EXCEPT ![self] = "WScan"]
                          /\ UNCHANGED doneJobs
               /\ UNCHANGED << size, mtime, now, statted, remote, lastUp, 
                               stopped, pools, queues, nextJob, ticks, 
                               regressed, stack, n, myjob, t, job, limit, 
                               scanned, upload >>

WScan(self) == /\ pc[self] = "WScan"
               /\ statted' = TRUE
               /\ upload' = [upload EXCEPT ![self] = lastUp # mtime]
               /\ limit' = [limit EXCEPT ![self] = size]
               /\ scanned' = [scanned EXCEPT ![self] = mtime]
               /\ pc' = [pc EXCEPT ![self] = "WCommit"]
               /\ UNCHANGED << size, mtime, now, remote, lastUp, stopped, 
                               pools, queues, doneJobs, nextJob, ticks, 
                               regressed, stack, n, myjob, t, job >>

WCommit(self) == /\ pc[self] = "WCommit"
                 /\ IF upload[self]
                       THEN /\ IF limit[self] < remote
                                  THEN /\ regressed' = TRUE
                                  ELSE /\ TRUE
                                       /\ UNCHANGED regressed
                            /\ remote' = limit[self]
                       ELSE /\ TRUE
                            /\ UNCHANGED << remote, regressed >>
                 /\ pc' = [pc EXCEPT ![self] = "WRecord"]
                 /\ UNCHANGED << size, mtime, now, statted, lastUp, stopped, 
                                 pools, queues, doneJobs, nextJob, ticks, 
                                 stack, n, myjob, t, job, limit, scanned, 
                                 upload >>

WRecord(self) == /\ pc[self] = "WRecord"
                 /\ IF upload[self]
                       THEN /\ lastUp' = scanned[self]
                       ELSE /\ TRUE
                            /\ UNCHANGED lastUp
                 /\ doneJobs' = (doneJobs \union {job[self]})
                 /\ pc' = [pc EXCEPT ![self] = "WGet"]
                 /\ UNCHANGED << size, mtime, now, statted, remote, stopped, 
                                 pools, queues, nextJob, ticks, regressed, 
                                 stack, n, myjob, t, job, limit, scanned, 
                                 upload >>

worker(self) == WGet(self) \/ WPush(self) \/ WScan(self) \/ WCommit(self)
                   \/ WRecord(self)

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == user \/ sched \/ clock
           \/ (\E self \in ProcSet: Submit(self))
           \/ (\E self \in Workers: worker(self))
           \/ Terminating

Spec == /\ Init /\ [][Next]_vars
        /\ WF_vars(user) /\ WF_vars(Submit("user"))
        /\ WF_vars(sched) /\ WF_vars(Submit("sched"))
        /\ WF_vars(clock)
        /\ \A self \in Workers : WF_vars(worker(self))

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION
UserTerminates == <>(pc["user"] = "Done")
=============================================================================
