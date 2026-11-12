------------------------------ MODULE DistShards ------------------------------
(***************************************************************************)
(* Exactly-once delivery of a streaming dataset across ranks x DataLoader   *)
(* workers (datasets main @ efc8247).                                      *)
(*                                                                         *)
(*  split_dataset_by_node (distributed.py:10-55,                           *)
(*    iterable_dataset.py:5738-5790): nested splits compose as             *)
(*    rank = W2*r1 + r2, world = W1*W2.                                    *)
(*  _prepare_ex_iterable_for_iteration (iterable_dataset.py:2992-3028):    *)
(*    "shards" (or "auto" with num_shards % world == 0): rank keeps shards *)
(*    at positions k % world == rank (contiguous=False, lines 283-291);    *)
(*    "examples": StepExamplesIterable(step=world, offset=rank).           *)
(*  _iter_pytorch (lines 2908-2964): worker w keeps the local shards       *)
(*    k % num_workers == w; StepExamplesIterable.shard_data_sources        *)
(*    forwards the worker split *below* the step (lines 776-781).          *)
(*  epoch > 0 / shuffle(seed): the shard order is permuted with the same   *)
(*    seed on every rank (lines 2985-2990); the shuffle-buffer order of a  *)
(*    worker stream depends on the worker id only (shift_ex_examples_rngs, *)
(*    line 2939).  Both are chosen nondeterministically from a few         *)
(*    permutations (identity / reverse / rotation) shared by all ranks.    *)
(*    SameSeedAcrossRanks = FALSE models an unseeded shuffle (the docs     *)
(*    warn against it): odd ranks then see another shard order.            *)
(*                                                                         *)
(* The (rank, worker) streams are delivered one example at a time; their   *)
(* interleaving is irrelevant for the property, so slots run in sequence.  *)
(* Invariants: NoDup always, AllDelivered at the end.                      *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS ShardsRange, PerShard, WorldRange, NWRange, SameSeedAcrossRanks

Strategies == {"auto", "shards", "examples"}
Kinds == {"id", "rev", "rot"}
Configs == [ns : ShardsRange, w1 : WorldRange, w2 : WorldRange, nw : NWRange, strategy : Strategies,
            shardOrder : Kinds, bufOrder : [0..3 -> Kinds]]
MaxNS == CHOOSE x \in ShardsRange : \A y \in ShardsRange : y <= x
AllExamples == {<<s, j>> : s \in 0..(MaxNS-1), j \in 1..PerShard}

RECURSIVE Concat(_)
Concat(ss) == IF ss = <<>> THEN <<>> ELSE Head(ss) \o Concat(Tail(ss))
RECURSIVE FilterPos(_, _, _)
\* keep the elements at 1-based positions k with (k-1) % m = r
FilterPos(seq, m, r) == IF seq = <<>> THEN <<>>
                        ELSE (IF (Len(seq) - 1) % m = r THEN FilterPos(SubSeq(seq, 1, Len(seq)-1), m, r) \o <<seq[Len(seq)]>>
                              ELSE FilterPos(SubSeq(seq, 1, Len(seq)-1), m, r))
Permute(seq, kind) == LET n == Len(seq) IN
    CASE kind = "id"  -> seq
    []   kind = "rev" -> [k \in 1..n |-> seq[n + 1 - k]]
    []   kind = "rot" -> IF n = 0 THEN seq ELSE [k \in 1..n |-> seq[(k % n) + 1]]

(* --algorithm DistShards
variables
    cfg \in Configs,
    delivered = [e \in AllExamples |-> 0],
    slot = 0, pos = 1;

define
    World == cfg.w1 * cfg.w2
    NS == cfg.ns
    NW == IF cfg.nw = 0 THEN 1 ELSE cfg.nw
    UseShards == cfg.strategy = "shards" \/ (cfg.strategy = "auto" /\ NS % World = 0)
    Valid == ~(cfg.strategy = "shards" /\ NS < World)          \* ValueError otherwise (lines 5774-5779, 2996-3003)
    \* nested split: slot -> (r1, r2, w); rank = W2*r1 + r2 (lines 5766-5768)
    R1(s) == s \div (cfg.w2 * NW)
    R2(s) == (s \div NW) % cfg.w2
    Rank(s) == cfg.w2 * R1(s) + R2(s)
    Wk(s) == s % NW
    Slots == World * NW
    ShardOrder(rank) == Permute([k \in 1..NS |-> k - 1],
                                IF ~SameSeedAcrossRanks /\ rank % 2 = 1 THEN "rev" ELSE cfg.shardOrder)
    RankShards(rank) == IF UseShards THEN FilterPos(ShardOrder(rank), World, rank) ELSE ShardOrder(rank)
    WorkerShards(rank, w) == FilterPos(RankShards(rank), NW, w)
    RawStream(rank, w) == Concat([k \in 1..Len(WorkerShards(rank, w)) |->
                                  [j \in 1..PerShard |-> <<WorkerShards(rank, w)[k], j>>]])
    Shuffled(rank, w) == Permute(RawStream(rank, w), cfg.bufOrder[w])
    Stream(rank, w) == IF UseShards THEN Shuffled(rank, w) ELSE FilterPos(Shuffled(rank, w), World, rank)
    CurStream == Stream(Rank(slot), Wk(slot))

    NoDup == \A e \in AllExamples : delivered[e] <= 1
    Finished == ~Valid \/ slot >= Slots
    AllDelivered == (Finished /\ Valid) =>
                    \A e \in AllExamples : delivered[e] = IF e[1] < NS THEN 1 ELSE 0
end define;

process Loader = "loader"
begin
Deliver:
    while Valid /\ slot < Slots do
        if pos <= Len(CurStream) then
            delivered[CurStream[pos]] := delivered[CurStream[pos]] + 1;
            pos := pos + 1;
        else
            slot := slot + 1; pos := 1;
        end if;
    end while;
end process;
end algorithm; *)
\* BEGIN TRANSLATION (chksum(pcal) = "c4825715" /\ chksum(tla) = "1aacdefc")
VARIABLES pc, cfg, delivered, slot, pos

(* define statement *)
World == cfg.w1 * cfg.w2
NS == cfg.ns
NW == IF cfg.nw = 0 THEN 1 ELSE cfg.nw
UseShards == cfg.strategy = "shards" \/ (cfg.strategy = "auto" /\ NS % World = 0)
Valid == ~(cfg.strategy = "shards" /\ NS < World)

R1(s) == s \div (cfg.w2 * NW)
R2(s) == (s \div NW) % cfg.w2
Rank(s) == cfg.w2 * R1(s) + R2(s)
Wk(s) == s % NW
Slots == World * NW
ShardOrder(rank) == Permute([k \in 1..NS |-> k - 1],
                            IF ~SameSeedAcrossRanks /\ rank % 2 = 1 THEN "rev" ELSE cfg.shardOrder)
RankShards(rank) == IF UseShards THEN FilterPos(ShardOrder(rank), World, rank) ELSE ShardOrder(rank)
WorkerShards(rank, w) == FilterPos(RankShards(rank), NW, w)
RawStream(rank, w) == Concat([k \in 1..Len(WorkerShards(rank, w)) |->
                              [j \in 1..PerShard |-> <<WorkerShards(rank, w)[k], j>>]])
Shuffled(rank, w) == Permute(RawStream(rank, w), cfg.bufOrder[w])
Stream(rank, w) == IF UseShards THEN Shuffled(rank, w) ELSE FilterPos(Shuffled(rank, w), World, rank)
CurStream == Stream(Rank(slot), Wk(slot))

NoDup == \A e \in AllExamples : delivered[e] <= 1
Finished == ~Valid \/ slot >= Slots
AllDelivered == (Finished /\ Valid) =>
                \A e \in AllExamples : delivered[e] = IF e[1] < NS THEN 1 ELSE 0


vars == << pc, cfg, delivered, slot, pos >>

ProcSet == {"loader"}

Init == (* Global variables *)
        /\ cfg \in Configs
        /\ delivered = [e \in AllExamples |-> 0]
        /\ slot = 0
        /\ pos = 1
        /\ pc = [self \in ProcSet |-> "Deliver"]

Deliver == /\ pc["loader"] = "Deliver"
           /\ IF Valid /\ slot < Slots
                 THEN /\ IF pos <= Len(CurStream)
                            THEN /\ delivered' = [delivered EXCEPT ![CurStream[pos]] = delivered[CurStream[pos]] + 1]
                                 /\ pos' = pos + 1
                                 /\ slot' = slot
                            ELSE /\ slot' = slot + 1
                                 /\ pos' = 1
                                 /\ UNCHANGED delivered
                      /\ pc' = [pc EXCEPT !["loader"] = "Deliver"]
                 ELSE /\ pc' = [pc EXCEPT !["loader"] = "Done"]
                      /\ UNCHANGED << delivered, slot, pos >>
           /\ cfg' = cfg

Loader == Deliver

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == Loader
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION 
===============================================================================
