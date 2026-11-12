------------------------------ MODULE StepResume ------------------------------
(***************************************************************************)
(* split_dataset_by_node "examples" strategy (StepExamplesIterable) over a *)
(* shuffled stream, checkpointed by every DDP rank after the same number   *)
(* K of examples (synchronous training), datasets main @ efc8247.          *)
(*                                                                         *)
(* All ranks see the same BufferShuffled output (same seed).  After a      *)
(* resume the buffer is lost and refilled from the saved source position   *)
(* Q (iterable_dataset.py:2058), so ranks only keep partitioning the same  *)
(* stream if they all resume from the same Q and the same step phase.      *)
(* Table sizes of BufferShuffled._iter_arrow (lines 2100-2135) do not      *)
(* depend on the permutation, so the model is deterministic in the data:   *)
(* it tracks *positions* in the shuffled stream, not values.               *)
(*                                                                         *)
(* Path = "python": StepExamplesIterable.__iter__ (lines 751-758) consumes *)
(*   exactly `step` elements per yield; state = live source position.     *)
(* Path = "arrow" : StepExamplesIterable._iter_arrow (760-769) consumes    *)
(*   whole shuffled tables; the top RebatchedArrowExamplesIterable saves   *)
(*   (state before the current table, chunks) (lines 574-634).             *)
(* Path = "arrow_tablefix": arrow path with the per-table fix of           *)
(*   ShuffleResume (state after the current table, 0 chunks).             *)
(* Invariant RanksAligned: once every rank has yielded K examples, all     *)
(* saved states are equal.                                                 *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS N, Bin, S, World, K, Path

Keep == S \div 2
Min(a, b) == IF a < b THEN a ELSE b
\* BufferShuffled output blocks as (rows yielded, source position after the pulls), from source position 0
RECURSIVE Blocks(_, _)
Blocks(src, held) ==
    IF src >= N THEN (IF held > 0 THEN <<[size |-> held, q |-> N]>> ELSE <<>>)
    ELSE LET k == CHOOSE k \in 1..N : (held + Min(k*Bin, N - src) >= S \/ src + k*Bin >= N)
                                      /\ \A j \in 1..(k-1) : ~(held + Min(j*Bin, N - src) >= S \/ src + j*Bin >= N)
             cur == held + Min(k*Bin, N - src)
             nsrc == Min(src + k*Bin, N)
         IN IF cur >= S THEN <<[size |-> cur - Keep, q |-> nsrc]>> \o Blocks(nsrc, Keep)
            ELSE <<[size |-> cur, q |-> nsrc]>>
B == Blocks(0, 0)
RECURSIVE Prefix(_)
Prefix(j) == IF j = 0 THEN 0 ELSE Prefix(j-1) + B[j].size
\* first 1-based global stream position covered by block j
First(j) == Prefix(j-1) + 1
BlockOf(pos) == CHOOSE j \in 1..Len(B) : First(j) <= pos /\ pos < First(j) + B[j].size
TotalRows == Prefix(Len(B))

(* --algorithm StepResume
variables saved = [r \in 0..(World-1) |-> <<>>], yielded = [r \in 0..(World-1) |-> 0];

define
    \* global stream position of rank r's n-th example (Step offset r)
    Pos(r, n) == (n - 1) * World + r + 1
    Q(j) == B[j].q
    QBefore(j) == IF j = 1 THEN 0 ELSE B[j-1].q
    \* rows of block j that go to rank r (Step keeps positions = r+1 mod World)
    RowsOf(r, j) == Cardinality({p \in First(j)..(First(j) + B[j].size - 1) : (p - 1) % World = r})
    ChunksDone(r, n) == Cardinality({p \in First(BlockOf(Pos(r, n)))..Pos(r, n) : (p - 1) % World = r})
    SavedState(r, n) ==
        CASE Path = "python" ->
               \* Step has consumed n*World elements; BufferShuffled is suspended in the block holding element n*World
               <<Q(BlockOf(Min(n * World, TotalRows))), 0>>
        []   Path = "arrow" -> <<QBefore(BlockOf(Pos(r, n))), ChunksDone(r, n)>>
        []   Path = "arrow_tablefix" -> <<Q(BlockOf(Pos(r, n))), 0>>
    Enough == \A r \in 0..(World-1) : Pos(r, K) <= TotalRows
    RanksAligned == (\A r \in 0..(World-1) : yielded[r] = K) =>
                        \A r1, r2 \in 0..(World-1) : saved[r1][1] = saved[r2][1]
end define;

process Rank \in 0..(World-1)
begin
Run:
    while Enough /\ yielded[self] < K do
        yielded[self] := yielded[self] + 1;
        saved[self] := SavedState(self, yielded[self]);
    end while;
end process;
end algorithm; *)
\* BEGIN TRANSLATION (chksum(pcal) = "deda82dc" /\ chksum(tla) = "a457a337")
VARIABLES pc, saved, yielded

(* define statement *)
Pos(r, n) == (n - 1) * World + r + 1
Q(j) == B[j].q
QBefore(j) == IF j = 1 THEN 0 ELSE B[j-1].q

RowsOf(r, j) == Cardinality({p \in First(j)..(First(j) + B[j].size - 1) : (p - 1) % World = r})
ChunksDone(r, n) == Cardinality({p \in First(BlockOf(Pos(r, n)))..Pos(r, n) : (p - 1) % World = r})
SavedState(r, n) ==
    CASE Path = "python" ->

           <<Q(BlockOf(Min(n * World, TotalRows))), 0>>
    []   Path = "arrow" -> <<QBefore(BlockOf(Pos(r, n))), ChunksDone(r, n)>>
    []   Path = "arrow_tablefix" -> <<Q(BlockOf(Pos(r, n))), 0>>
Enough == \A r \in 0..(World-1) : Pos(r, K) <= TotalRows
RanksAligned == (\A r \in 0..(World-1) : yielded[r] = K) =>
                    \A r1, r2 \in 0..(World-1) : saved[r1][1] = saved[r2][1]


vars == << pc, saved, yielded >>

ProcSet == (0..(World-1))

Init == (* Global variables *)
        /\ saved = [r \in 0..(World-1) |-> <<>>]
        /\ yielded = [r \in 0..(World-1) |-> 0]
        /\ pc = [self \in ProcSet |-> "Run"]

Run(self) == /\ pc[self] = "Run"
             /\ IF Enough /\ yielded[self] < K
                   THEN /\ yielded' = [yielded EXCEPT ![self] = yielded[self] + 1]
                        /\ saved' = [saved EXCEPT ![self] = SavedState(self, yielded'[self])]
                        /\ pc' = [pc EXCEPT ![self] = "Run"]
                   ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]
                        /\ UNCHANGED << saved, yielded >>

Rank(self) == Run(self)

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == (\E self \in 0..(World-1): Rank(self))
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION 
===============================================================================
