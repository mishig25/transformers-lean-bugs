------------------------------ MODULE LoaderResume ------------------------------
(***************************************************************************)
(* torchdata StatefulDataLoader checkpoint/resume of a datasets            *)
(* IterableDataset (datasets main @ efc8247), over several epochs.          *)
(*                                                                         *)
(* Actors: the dataloader driver; one dataset copy per DataLoader worker   *)
(* (num_workers > 0) or the main-process dataset (num_workers = 0).        *)
(*  - IterableDataset.load_state_dict (iterable_dataset.py:2819-2867)      *)
(*    only stores _starting_state_dict; it is never cleared.               *)
(*  - every iteration start: _prepare_ex_iterable_for_iteration            *)
(*    (lines 3047-3054) loads _starting_state_dict when its epoch matches; *)
(*    in a worker, _iter_pytorch (lines 2908-2947) first calls it on the   *)
(*    *un-sharded* tree (line 2909), then shards the tree by worker and    *)
(*    loads the same state again into the worker tree (lines 2936-2947).   *)
(*  - _inner_load_state_dict (lines 300-313) walks the *target* list and   *)
(*    indexes new_state[i]: it raises IndexError when the saved list is    *)
(*    shorter than the target's.                                           *)
(*  - list lengths in the state: shuffle() -> InterleavedShardsExamples-   *)
(*    Iterable = Cycling over min(local shards, max_buffer_input_shards)   *)
(*    children (lines 1971-1985); concatenate_datasets -> Vertically-      *)
(*    Concatenated over the datasets, or over the single shards of a       *)
(*    worker after shard_data_sources (lines 1070-1121).                   *)
(*  - torchdata worker loop: worker_state is applied once, when workers    *)
(*    start after load_state_dict; persistent workers keep their dataset   *)
(*    copy across epochs, non-persistent workers get a fresh copy of the   *)
(*    main-process dataset each epoch; num_workers=0 iterates the main     *)
(*    dataset object itself.                                               *)
(*                                                                         *)
(* Properties: NoCrash (resuming does not raise) and LaterEpochsComplete   *)
(* (after the resumed epoch, every epoch delivers each worker's full       *)
(* share).  Fixed = TRUE models the proposed fix (no load into the         *)
(* un-sharded tree in workers; _starting_state_dict consumed once).        *)
(***************************************************************************)
EXTENDS Integers, FiniteSets, TLC

CONSTANTS Pipelines,     \* subset of {"plain", "shuffle", "concat"}
          ShardsRange,   \* number of shards of the (first) dataset
          ShardsBRange,  \* number of shards of the second dataset (concat only)
          MaxInRange,    \* shuffle(max_buffer_input_shards=...), default 10
          MaxNW,         \* num_workers ranges over 0..MaxNW (0 = main process)
          Epochs,        \* epochs run after resuming
          Len,           \* examples per worker share (abstract)
          Fixed

Min(a, b) == IF a < b THEN a ELSE b
AllProcs == 0..Min(MaxNW, 3)
NoCkpt == [ep |-> -1, pos |-> 0, list |-> 0]
Configs == [pipeline : Pipelines, ns : ShardsRange, nsb : ShardsBRange, maxIn : MaxInRange,
            nw : 0..MaxNW, persistent : BOOLEAN, setEpoch : BOOLEAN]

(* --algorithm LoaderResume
variables
    cfg \in Configs,
    ckptPos \in [AllProcs -> 0..Len],  \* per-worker position at checkpoint time (any point of epoch 0)
    ckpt = [q \in AllProcs |-> NoCkpt],
    mainStart = NoCkpt,                \* main-process dataset's _starting_state_dict
    wStart = [q \in AllProcs |-> NoCkpt],  \* worker copies' _starting_state_dict
    dsEpoch = 0,                       \* dataset epoch (shared tensor, set_epoch)
    epoch = 0,
    crashed = FALSE,
    reported = [q \in AllProcs |-> NoCkpt],  \* worker state StatefulDataLoader holds right after resuming
    delivered = [e \in 0..(Epochs-1) |-> [q \in AllProcs |-> 0]];

define
    Procs == IF cfg.nw = 0 THEN {0} ELSE 0..(cfg.nw-1)
    TotalShards == IF cfg.pipeline = "concat" THEN cfg.ns + cfg.nsb ELSE cfg.ns
    \* shards of worker w: split_shard_indices_by_worker(contiguous=False) (lines 283-291)
    WShards(w) == IF cfg.nw = 0 THEN TotalShards ELSE Cardinality({i \in 0..(TotalShards-1) : i % cfg.nw = w})
    \* length of the list inside the state dict (0 = no list)
    FullList == CASE cfg.pipeline = "shuffle" -> Min(cfg.ns, cfg.maxIn)
                []   cfg.pipeline = "concat"  -> 2
                []   OTHER                    -> 0
    WorkerList(w) == CASE cfg.pipeline = "shuffle" -> Min(WShards(w), cfg.maxIn)
                     []   cfg.pipeline = "concat"  -> IF cfg.nw = 0 THEN 2 ELSE WShards(w)
                     []   OTHER                    -> 0
    \* _inner_load_state_dict(state, new_state) (lines 300-313) raises iff the target list is longer
    LoadRaises(targetLen, savedLen) == targetLen > savedLen
    \* torchdata stops workers that get no shard (iterable_dataset.py:2932, 2961-2964): they deliver nothing
    Share(w) == IF WShards(w) = 0 THEN 0 ELSE Len

    NoCrash == ~crashed
    \* every epoch after the resumed one delivers the full share of every worker
    LaterEpochsComplete == \A e \in 1..(Epochs-1) : e < epoch => \A q \in Procs : delivered[e][q] = Share(q)
    \* the resumed epoch delivers exactly what was left at the checkpoint
    \* a checkpoint taken after resuming, before worker q yields a batch, must still hold q's restored position:
    \* torchdata reports dataset.state_dict() of the freshly loaded dataset as the worker's initial state
    \* (stateful_dataloader/worker.py: try_to_deserialize then _make_state_dict); IterableDataset.state_dict()
    \* returns self._state_dict (line 2817), which load_state_dict (line 2867) does not update.
    ReportedAfterLoad == epoch > 0 => \A q \in Procs : reported[q].pos = ckpt[q].pos
    ResumedEpochExact == (epoch > 0 /\ ~crashed) => \A q \in Procs : delivered[0][q] = Share(q) - (IF Share(q) = 0 THEN 0 ELSE ckptPos[q])
end define;

process Driver = "driver"
variables p = 0, st = NoCkpt, startPos = 0;
begin
Ckpt:
    \* dataloader.state_dict() during epoch 0: each worker reports its *worker tree* state
    ckpt := [q \in AllProcs |-> [ep |-> 0, pos |-> ckptPos[q], list |-> WorkerList(q)]];
Restore:
    \* new StatefulDataLoader(ds); load_state_dict(sd): with num_workers=0 the main dataset object is loaded,
    \* otherwise each worker's dataset copy is loaded when the workers start for the first iterator
    if cfg.nw = 0 then mainStart := ckpt[0];
    else wStart := ckpt;
    end if;
    \* the state each worker reports before producing anything: the stale start-of-epoch state, or the fix
    reported := [q \in AllProcs |-> IF Fixed THEN ckpt[q] ELSE [ep |-> 0, pos |-> 0, list |-> WorkerList(q)]];
Epoch:
    while epoch < Epochs do
        if cfg.setEpoch then dsEpoch := epoch; end if;
        p := 0;
Workers:
        while p \in Procs do
            st := IF cfg.nw = 0 THEN mainStart
                  ELSE IF cfg.persistent \/ epoch = 0 THEN wStart[p] ELSE NoCkpt;  \* fresh copy of main
            \* worker: _iter_pytorch line 2909 -> _prepare_ex_iterable_for_iteration line 3052 loads the worker's
            \* state into the *un-sharded* tree, even for a worker that then gets no shard (line 2932)
            if st.ep = dsEpoch /\ cfg.nw > 0 /\ ~Fixed /\ LoadRaises(FullList, st.list) then
                crashed := TRUE; goto Finish;
            elsif st.ep = dsEpoch /\ WShards(p) > 0 then
                startPos := st.pos;                       \* lines 2944-2947 (worker) / 3051-3054 (main)
            else
                startPos := 0;
            end if;
Iterate:
            delivered[epoch][p] := Share(p) - (IF Share(p) = 0 THEN 0 ELSE startPos);
            if Fixed then
                \* proposed fix: the iteration that used _starting_state_dict consumes it
                if cfg.nw = 0 then mainStart := NoCkpt; else wStart[p] := NoCkpt; end if;
            end if;
            p := p + 1;
        end while;
NextEpoch:
        epoch := epoch + 1;
    end while;
Finish:
    skip;
end process;
end algorithm; *)
\* BEGIN TRANSLATION (chksum(pcal) = "4ea3b0b5" /\ chksum(tla) = "8e89a86d")
VARIABLES pc, cfg, ckptPos, ckpt, mainStart, wStart, dsEpoch, epoch, crashed, 
          reported, delivered

(* define statement *)
Procs == IF cfg.nw = 0 THEN {0} ELSE 0..(cfg.nw-1)
TotalShards == IF cfg.pipeline = "concat" THEN cfg.ns + cfg.nsb ELSE cfg.ns

WShards(w) == IF cfg.nw = 0 THEN TotalShards ELSE Cardinality({i \in 0..(TotalShards-1) : i % cfg.nw = w})

FullList == CASE cfg.pipeline = "shuffle" -> Min(cfg.ns, cfg.maxIn)
            []   cfg.pipeline = "concat"  -> 2
            []   OTHER                    -> 0
WorkerList(w) == CASE cfg.pipeline = "shuffle" -> Min(WShards(w), cfg.maxIn)
                 []   cfg.pipeline = "concat"  -> IF cfg.nw = 0 THEN 2 ELSE WShards(w)
                 []   OTHER                    -> 0

LoadRaises(targetLen, savedLen) == targetLen > savedLen

Share(w) == IF WShards(w) = 0 THEN 0 ELSE Len

NoCrash == ~crashed

LaterEpochsComplete == \A e \in 1..(Epochs-1) : e < epoch => \A q \in Procs : delivered[e][q] = Share(q)





ReportedAfterLoad == epoch > 0 => \A q \in Procs : reported[q].pos = ckpt[q].pos
ResumedEpochExact == (epoch > 0 /\ ~crashed) => \A q \in Procs : delivered[0][q] = Share(q) - (IF Share(q) = 0 THEN 0 ELSE ckptPos[q])

VARIABLES p, st, startPos

vars == << pc, cfg, ckptPos, ckpt, mainStart, wStart, dsEpoch, epoch, crashed, 
           reported, delivered, p, st, startPos >>

ProcSet == {"driver"}

Init == (* Global variables *)
        /\ cfg \in Configs
        /\ ckptPos \in [AllProcs -> 0..Len]
        /\ ckpt = [q \in AllProcs |-> NoCkpt]
        /\ mainStart = NoCkpt
        /\ wStart = [q \in AllProcs |-> NoCkpt]
        /\ dsEpoch = 0
        /\ epoch = 0
        /\ crashed = FALSE
        /\ reported = [q \in AllProcs |-> NoCkpt]
        /\ delivered = [e \in 0..(Epochs-1) |-> [q \in AllProcs |-> 0]]
        (* Process Driver *)
        /\ p = 0
        /\ st = NoCkpt
        /\ startPos = 0
        /\ pc = [self \in ProcSet |-> "Ckpt"]

Ckpt == /\ pc["driver"] = "Ckpt"
        /\ ckpt' = [q \in AllProcs |-> [ep |-> 0, pos |-> ckptPos[q], list |-> WorkerList(q)]]
        /\ pc' = [pc EXCEPT !["driver"] = "Restore"]
        /\ UNCHANGED << cfg, ckptPos, mainStart, wStart, dsEpoch, epoch, 
                        crashed, reported, delivered, p, st, startPos >>

Restore == /\ pc["driver"] = "Restore"
           /\ IF cfg.nw = 0
                 THEN /\ mainStart' = ckpt[0]
                      /\ UNCHANGED wStart
                 ELSE /\ wStart' = ckpt
                      /\ UNCHANGED mainStart
           /\ reported' = [q \in AllProcs |-> IF Fixed THEN ckpt[q] ELSE [ep |-> 0, pos |-> 0, list |-> WorkerList(q)]]
           /\ pc' = [pc EXCEPT !["driver"] = "Epoch"]
           /\ UNCHANGED << cfg, ckptPos, ckpt, dsEpoch, epoch, crashed, 
                           delivered, p, st, startPos >>

Epoch == /\ pc["driver"] = "Epoch"
         /\ IF epoch < Epochs
               THEN /\ IF cfg.setEpoch
                          THEN /\ dsEpoch' = epoch
                          ELSE /\ TRUE
                               /\ UNCHANGED dsEpoch
                    /\ p' = 0
                    /\ pc' = [pc EXCEPT !["driver"] = "Workers"]
               ELSE /\ pc' = [pc EXCEPT !["driver"] = "Finish"]
                    /\ UNCHANGED << dsEpoch, p >>
         /\ UNCHANGED << cfg, ckptPos, ckpt, mainStart, wStart, epoch, crashed, 
                         reported, delivered, st, startPos >>

Workers == /\ pc["driver"] = "Workers"
           /\ IF p \in Procs
                 THEN /\ st' = (IF cfg.nw = 0 THEN mainStart
                                ELSE IF cfg.persistent \/ epoch = 0 THEN wStart[p] ELSE NoCkpt)
                      /\ IF st'.ep = dsEpoch /\ cfg.nw > 0 /\ ~Fixed /\ LoadRaises(FullList, st'.list)
                            THEN /\ crashed' = TRUE
                                 /\ pc' = [pc EXCEPT !["driver"] = "Finish"]
                                 /\ UNCHANGED startPos
                            ELSE /\ IF st'.ep = dsEpoch /\ WShards(p) > 0
                                       THEN /\ startPos' = st'.pos
                                       ELSE /\ startPos' = 0
                                 /\ pc' = [pc EXCEPT !["driver"] = "Iterate"]
                                 /\ UNCHANGED crashed
                 ELSE /\ pc' = [pc EXCEPT !["driver"] = "NextEpoch"]
                      /\ UNCHANGED << crashed, st, startPos >>
           /\ UNCHANGED << cfg, ckptPos, ckpt, mainStart, wStart, dsEpoch, 
                           epoch, reported, delivered, p >>

Iterate == /\ pc["driver"] = "Iterate"
           /\ delivered' = [delivered EXCEPT ![epoch][p] = Share(p) - (IF Share(p) = 0 THEN 0 ELSE startPos)]
           /\ IF Fixed
                 THEN /\ IF cfg.nw = 0
                            THEN /\ mainStart' = NoCkpt
                                 /\ UNCHANGED wStart
                            ELSE /\ wStart' = [wStart EXCEPT ![p] = NoCkpt]
                                 /\ UNCHANGED mainStart
                 ELSE /\ TRUE
                      /\ UNCHANGED << mainStart, wStart >>
           /\ p' = p + 1
           /\ pc' = [pc EXCEPT !["driver"] = "Workers"]
           /\ UNCHANGED << cfg, ckptPos, ckpt, dsEpoch, epoch, crashed, 
                           reported, st, startPos >>

NextEpoch == /\ pc["driver"] = "NextEpoch"
             /\ epoch' = epoch + 1
             /\ pc' = [pc EXCEPT !["driver"] = "Epoch"]
             /\ UNCHANGED << cfg, ckptPos, ckpt, mainStart, wStart, dsEpoch, 
                             crashed, reported, delivered, p, st, startPos >>

Finish == /\ pc["driver"] = "Finish"
          /\ TRUE
          /\ pc' = [pc EXCEPT !["driver"] = "Done"]
          /\ UNCHANGED << cfg, ckptPos, ckpt, mainStart, wStart, dsEpoch, 
                          epoch, crashed, reported, delivered, p, st, startPos >>

Driver == Ckpt \/ Restore \/ Epoch \/ Workers \/ Iterate \/ NextEpoch
             \/ Finish

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == Driver
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION 
===============================================================================
