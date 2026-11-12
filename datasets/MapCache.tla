------------------------------ MODULE MapCache ------------------------------
(***************************************************************************)
(* Dataset.map(batched=True, num_proc=NP) run concurrently by every DDP    *)
(* rank into the same cache directory (datasets main @ efc8247).           *)
(*                                                                         *)
(*  arrow_dataset.py:3475-3505  glob existing shard cache files, pick      *)
(*                              num_shards                                 *)
(*  arrow_dataset.py:3558-3576  contiguous shards (Dataset.shard), one     *)
(*                              cache file per shard                       *)
(*  arrow_dataset.py:3578-3586  load already-cached shards                 *)
(*  arrow_dataset.py:3622-3637  process the others in a pool (any order)   *)
(*  _map_single 3970-4070: batches of B rows (drop_last_batch drops the    *)
(*      last partial batch of each shard);   the writer is created on the   *)
(*      first batch; data is written to a NamedTemporaryFile and           *)
(*      shutil.move'd (atomic rename) to the cache file; when no batch was *)
(*      processed (update_data stays None, line 3758) the input shard is   *)
(*      returned unchanged (line 4070) and nothing is cached.              *)
(*  3646-3657   concatenate shards in rank order.                          *)
(*                                                                         *)
(* Rows are 1..N; a mapped row r is represented as -r.  Invariants:        *)
(*  RanksAgree       every rank ends with the same, concatenated-in-order *)
(*                   result (shard i = cached or recomputed shard i), for  *)
(*                   any interleaving of ranks and pool jobs               *)
(*  OutputAllMapped  every row of the result went through the function    *)
(* A cache name only ever holds a complete file (tmp + atomic rename), so  *)
(* a torn read cannot be expressed; the model checks that loading another  *)
(* rank's file instead of recomputing never changes the result.            *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS N, B, NP, Ranks, DropLast

Div == N \div NP
Mod == N % NP
\* Dataset.shard(num_shards, index, contiguous=True)
Start(i) == Div * i + (IF i < Mod THEN i ELSE Mod)
Stop(i) == Start(i) + Div + (IF i < Mod THEN 1 ELSE 0)
ShardRows(i) == [k \in 1..(Stop(i) - Start(i)) |-> Start(i) + k]
\* rows covered by the batches _map_single processes (lines 3976-3980)
Kept(i) == IF DropLast THEN ((Stop(i) - Start(i)) \div B) * B ELSE Stop(i) - Start(i)
Mapped(i) == [k \in 1..Kept(i) |-> -(Start(i) + k)]
\* what _map_single returns for shard i: mapped rows, or the raw shard when no batch ran (line 4070)
Output(i) == IF Kept(i) = 0 THEN ShardRows(i) ELSE Mapped(i)
RECURSIVE Cat(_, _)
Cat(f, k) == IF k = 0 THEN <<>> ELSE Cat(f, k - 1) \o f[k - 1]
AllMapped(seq) == \A k \in 1..Len(seq) : seq[k] < 0

(* --algorithm MapCache
variables
    cache = [i \in 0..(NP-1) |-> "absent"],  \* cache-<fp>_<i>_of_<NP>.arrow: "absent" | "complete"
    result = [r \in Ranks |-> <<>>],
    done = [r \in Ranks |-> FALSE];

define
    RanksAgree == \A r \in Ranks : done[r] => (result[r] = Cat([i \in 0..(NP-1) |-> Output(i)], NP))
    OutputAllMapped == \A r \in Ranks : done[r] => AllMapped(result[r])
end define;

process Rank \in Ranks
variables shards = [i \in 0..(NP-1) |-> <<>>], todo = {}, cur = 0, written = 0;
begin
LoadCached:
    \* lines 3578-3586: every shard whose cache file exists (complete: it only appears by rename) is loaded
    todo := {i \in 0..(NP-1) : cache[i] = "absent"};
    shards := [i \in 0..(NP-1) |-> IF cache[i] = "complete" THEN Mapped(i) ELSE <<>>];
Process:
    while todo # {} do
        with i \in todo do cur := i; todo := todo \ {i}; end with;
        written := 0;
Write:
        \* _map_single writes batches to a private tmp file (not visible under the cache name)
        while written < Kept(cur) do
            written := written + B;
        end while;
Publish:
        if Kept(cur) > 0 then
            cache[cur] := "complete";            \* shutil.move(tmp, cache_file_name) (line 4056): atomic rename
            shards[cur] := Mapped(cur);          \* Dataset.from_file(cache_file_name) (line 4066)
        else
            shards[cur] := ShardRows(cur);       \* update_data is None: the input shard is returned (line 4070)
        end if;
    end while;
Concat:
    result[self] := Cat(shards, NP);
    done[self] := TRUE;
end process;
end algorithm; *)
\* BEGIN TRANSLATION (chksum(pcal) = "f61a510d" /\ chksum(tla) = "a27a7cc3")
VARIABLES pc, cache, result, done

(* define statement *)
RanksAgree == \A r \in Ranks : done[r] => (result[r] = Cat([i \in 0..(NP-1) |-> Output(i)], NP))
OutputAllMapped == \A r \in Ranks : done[r] => AllMapped(result[r])

VARIABLES shards, todo, cur, written

vars == << pc, cache, result, done, shards, todo, cur, written >>

ProcSet == (Ranks)

Init == (* Global variables *)
        /\ cache = [i \in 0..(NP-1) |-> "absent"]
        /\ result = [r \in Ranks |-> <<>>]
        /\ done = [r \in Ranks |-> FALSE]
        (* Process Rank *)
        /\ shards = [self \in Ranks |-> [i \in 0..(NP-1) |-> <<>>]]
        /\ todo = [self \in Ranks |-> {}]
        /\ cur = [self \in Ranks |-> 0]
        /\ written = [self \in Ranks |-> 0]
        /\ pc = [self \in ProcSet |-> "LoadCached"]

LoadCached(self) == /\ pc[self] = "LoadCached"
                    /\ todo' = [todo EXCEPT ![self] = {i \in 0..(NP-1) : cache[i] = "absent"}]
                    /\ shards' = [shards EXCEPT ![self] = [i \in 0..(NP-1) |-> IF cache[i] = "complete" THEN Mapped(i) ELSE <<>>]]
                    /\ pc' = [pc EXCEPT ![self] = "Process"]
                    /\ UNCHANGED << cache, result, done, cur, written >>

Process(self) == /\ pc[self] = "Process"
                 /\ IF todo[self] # {}
                       THEN /\ \E i \in todo[self]:
                                 /\ cur' = [cur EXCEPT ![self] = i]
                                 /\ todo' = [todo EXCEPT ![self] = todo[self] \ {i}]
                            /\ written' = [written EXCEPT ![self] = 0]
                            /\ pc' = [pc EXCEPT ![self] = "Write"]
                       ELSE /\ pc' = [pc EXCEPT ![self] = "Concat"]
                            /\ UNCHANGED << todo, cur, written >>
                 /\ UNCHANGED << cache, result, done, shards >>

Write(self) == /\ pc[self] = "Write"
               /\ IF written[self] < Kept(cur[self])
                     THEN /\ written' = [written EXCEPT ![self] = written[self] + B]
                          /\ pc' = [pc EXCEPT ![self] = "Write"]
                     ELSE /\ pc' = [pc EXCEPT ![self] = "Publish"]
                          /\ UNCHANGED written
               /\ UNCHANGED << cache, result, done, shards, todo, cur >>

Publish(self) == /\ pc[self] = "Publish"
                 /\ IF Kept(cur[self]) > 0
                       THEN /\ cache' = [cache EXCEPT ![cur[self]] = "complete"]
                            /\ shards' = [shards EXCEPT ![self][cur[self]] = Mapped(cur[self])]
                       ELSE /\ shards' = [shards EXCEPT ![self][cur[self]] = ShardRows(cur[self])]
                            /\ cache' = cache
                 /\ pc' = [pc EXCEPT ![self] = "Process"]
                 /\ UNCHANGED << result, done, todo, cur, written >>

Concat(self) == /\ pc[self] = "Concat"
                /\ result' = [result EXCEPT ![self] = Cat(shards[self], NP)]
                /\ done' = [done EXCEPT ![self] = TRUE]
                /\ pc' = [pc EXCEPT ![self] = "Done"]
                /\ UNCHANGED << cache, shards, todo, cur, written >>

Rank(self) == LoadCached(self) \/ Process(self) \/ Write(self)
                 \/ Publish(self) \/ Concat(self)

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == (\E self \in Ranks: Rank(self))
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION 
===============================================================================
