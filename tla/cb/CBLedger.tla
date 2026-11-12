------------------------------ MODULE CBLedger ------------------------------
(***************************************************************************)
(* Block ledger of a full-attention CacheAllocator with prefix sharing:    *)
(* cache_allocators/cache_allocator.py BlockLedger (:34-108) and           *)
(* CacheAllocator.free_blocks / fork_blocks / match_prefix_blocks /       *)
(* acquire_prefix_blocks / mark_complete_blocks (:233-311), the pool       *)
(* (cache_pool.py), PagedAttentionCache.evict_cached_blocks (cache.py:359). *)
(* The content of block i of request r is the prefix Cont[r][1..i]; its    *)
(* hash is that prefix (compute_block_hash chains the parent hash).        *)
(* Every operation is one atomic step (all run on the generation thread). *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Blocks, Req, Cont, Child, Parent   \* Child forks from Parent (num_return_sequences = 2)

VARIABLES tbl,      \* [Req -> Seq(Blocks)] block_table[r]
          hasT,     \* [Req -> BOOLEAN] r in block_table
          ref,      \* shared_ref_counts: [Blocks -> Nat], 0 = no entry (the code only keeps entries >= 2)
          b2h,      \* block_to_hash: [Blocks -> hash or "none"]
          h2b,      \* hash_to_block: set of <<hash, block>>
          cached,   \* cached_blocks (unreferenced hashed blocks)
          free      \* free blocks in the pool
vars == <<tbl, hasT, ref, b2h, h2b, cached, free>>

NoH == <<>>
Hash(r, i) == SubSeq(Cont[r], 1, i)                     \* block i (1-based) of r
Tbl(r) == IF hasT[r] THEN tbl[r] ELSE <<>>
H2B(h) == IF \E p \in h2b : p[1] = h THEN (CHOOSE p \in h2b : p[1] = h)[2] ELSE 0

Init == /\ tbl = [r \in Req |-> <<>>] /\ hasT = [r \in Req |-> FALSE] /\ ref = [b \in Blocks |-> 0] /\ b2h = [b \in Blocks |-> NoH]
        /\ h2b = {} /\ cached = {} /\ free = Blocks

\* ---- ledger primitives (return records of the updated variables) ----
Acquire(b, R, C) ==                                           \* BlockLedger.acquire :58-64
  IF b \in C THEN [ref |-> R, cached |-> C \ {b}]
  ELSE [ref |-> [R EXCEPT ![b] = (IF R[b] = 0 THEN 1 ELSE R[b]) + 1], cached |-> C]

\* BlockLedger.release :66-87 ; returns [ref, cached, b2h, h2b, toPool]
Release(b, nocache, R, C, B2H, H2Bs) ==
  LET nrc == (IF R[b] = 0 THEN 1 ELSE R[b]) - 1
      R1 == [R EXCEPT ![b] = IF nrc > 1 THEN nrc ELSE 0]
  IN IF nrc > 0 THEN [ref |-> R1, cached |-> C, b2h |-> B2H, h2b |-> H2Bs, toPool |-> FALSE]
     ELSE IF B2H[b] # NoH
          THEN IF nocache
               THEN [ref |-> R1, cached |-> C, b2h |-> [B2H EXCEPT ![b] = NoH],
                     h2b |-> H2Bs \ {<<B2H[b], b>>}, toPool |-> TRUE]
               ELSE [ref |-> R1, cached |-> C \cup {b}, b2h |-> B2H, h2b |-> H2Bs, toPool |-> FALSE]
          ELSE [ref |-> R1, cached |-> C, b2h |-> B2H, h2b |-> H2Bs, toPool |-> TRUE]

\* ---- operations ----
\* allocate_cache_to_request: one more fresh block (request grows by a page)
Alloc(r) == /\ Len(Tbl(r)) < Len(Cont[r]) /\ free # {} /\ (r = Child => hasT[r])
            /\ \E b \in free : /\ tbl' = [tbl EXCEPT ![r] = Append(Tbl(r), b)] /\ free' = free \ {b}
            /\ hasT' = [hasT EXCEPT ![r] = TRUE]
            /\ UNCHANGED <<ref, b2h, h2b, cached>>

\* free_blocks(r, no_cache): finish / cancel (nocache=FALSE) or offload (nocache=TRUE)
RECURSIVE FreeAll(_, _, _, _, _, _, _, _)
FreeAll(s, i, nc, R, C, B2H, H2Bs, F) ==
  IF i > Len(s) THEN [ref |-> R, cached |-> C, b2h |-> B2H, h2b |-> H2Bs, free |-> F]
  ELSE LET x == Release(s[i], nc, R, C, B2H, H2Bs)
       IN FreeAll(s, i + 1, nc, x.ref, x.cached, x.b2h, x.h2b, IF x.toPool THEN F \cup {s[i]} ELSE F)
Free(r, nc) == /\ hasT[r]
               /\ LET x == FreeAll(tbl[r], 1, nc, ref, cached, b2h, h2b, free)
                  IN /\ ref' = x.ref /\ cached' = x.cached /\ b2h' = x.b2h /\ h2b' = x.h2b /\ free' = x.free
               /\ tbl' = [tbl EXCEPT ![r] = <<>>] /\ hasT' = [hasT EXCEPT ![r] = FALSE]

\* mark_complete_blocks for the first unhashed block i (all previous ones hashed), cache_allocator.py:290-311
MarkComplete(r) ==
  /\ hasT[r] /\ UNCHANGED hasT
  /\ \E i \in 1..Len(tbl[r]) :
       /\ \A j \in 1..(i - 1) : b2h[tbl[r][j]] # NoH
       /\ b2h[tbl[r][i]] = NoH
       /\ LET b == tbl[r][i]
              h == Hash(r, i)
              idb == H2B(h)
          IN IF idb = 0
             THEN /\ b2h' = [b2h EXCEPT ![b] = h] /\ h2b' = h2b \cup {<<h, b>>}
                  /\ UNCHANGED <<tbl, ref, cached, free>>
             ELSE LET a == Acquire(idb, ref, cached)
                      x == Release(b, FALSE, a.ref, a.cached, b2h, h2b)
                  IN /\ tbl' = [tbl EXCEPT ![r][i] = idb]
                     /\ ref' = x.ref /\ cached' = x.cached /\ b2h' = x.b2h /\ h2b' = x.h2b
                     /\ free' = IF x.toPool THEN free \cup {b} ELSE free

\* search_prefix_match + acquire_prefix_blocks for a PENDING request without table (scheduler.py:119-129)
Match(r) ==
  /\ ~hasT[r] /\ r # Child /\ hasT' = [hasT EXCEPT ![r] = TRUE]
  /\ \E k \in 1..(Len(Cont[r]) - 1) :                   \* never the whole prompt
       /\ \A i \in 1..k : H2B(Hash(r, i)) # 0
       /\ LET blocks == [i \in 1..k |-> H2B(Hash(r, i))]
              RECURSIVE Acq(_, _, _)
              Acq(i, R, C) == IF i > k THEN [ref |-> R, cached |-> C]
                              ELSE LET a == Acquire(blocks[i], R, C) IN Acq(i + 1, a.ref, a.cached)
              x == Acq(1, ref, cached)
          IN /\ tbl' = [tbl EXCEPT ![r] = blocks] /\ ref' = x.ref /\ cached' = x.cached
             /\ UNCHANGED <<b2h, h2b, free>>

\* fork_blocks (cache_allocator.py:252-266): share the first n blocks, fresh blocks for the others
Fork == /\ hasT[Parent] /\ ~hasT[Child] /\ hasT' = [hasT EXCEPT ![Child] = TRUE]
        /\ \E n \in 0..Len(tbl[Parent]) :
             LET src == tbl[Parent]
                 RECURSIVE Acq(_, _, _)
                 Acq(i, R, C) == IF i > n THEN [ref |-> R, cached |-> C]
                                 ELSE LET a == Acquire(src[i], R, C) IN Acq(i + 1, a.ref, a.cached)
                 x == Acq(1, ref, cached)
                 m == Len(src) - n
             IN \E fb \in SUBSET free :
                  /\ Cardinality(fb) = m
                  /\ \E f \in [1..m -> fb] : /\ \A i, j \in 1..m : i # j => f[i] # f[j]
                                             /\ tbl' = [tbl EXCEPT ![Child] = SubSeq(src, 1, n) \o f]
                  /\ free' = free \ fb /\ ref' = x.ref /\ cached' = x.cached
                  /\ UNCHANGED <<b2h, h2b>>

\* evict_cached_blocks (cache_allocator.py:100-108, cache.py:359-369)
Evict == /\ cached # {}
         /\ b2h' = [b \in Blocks |-> IF b \in cached THEN NoH ELSE b2h[b]]
         /\ h2b' = {p \in h2b : p[2] \notin cached}
         /\ free' = free \cup cached /\ cached' = {}
         /\ UNCHANGED <<tbl, hasT, ref>>

Next == \/ \E r \in Req : Alloc(r) \/ Free(r, FALSE) \/ Free(r, TRUE) \/ MarkComplete(r) \/ Match(r)
        \/ Fork \/ Evict
Spec == Init /\ [][Next]_vars

\* ---- invariants ----
Occ(b) == Cardinality({<<r, i>> \in Req \X (1..4) : hasT[r] /\ i <= Len(tbl[r]) /\ tbl[r][i] = b})
Referenced == {b \in Blocks : Occ(b) > 0}
\* each block is in exactly one of: pool, cached, referenced
Partition == /\ free \cap cached = {} /\ free \cap Referenced = {} /\ cached \cap Referenced = {}
             /\ free \cup cached \cup Referenced = Blocks
\* the implicit/explicit reference count equals the number of block-table occurrences
RefCountOK == \A b \in Referenced : (IF ref[b] = 0 THEN 1 ELSE ref[b]) = Occ(b)
NoStaleRef == \A b \in Blocks : ref[b] # 1 /\ (ref[b] > 0 => b \in Referenced)
\* hash tables are consistent and only describe live (referenced or cached) blocks
HashOK == /\ \A p \in h2b : b2h[p[2]] = p[1]
          /\ \A b \in Blocks : b2h[b] # NoH => <<b2h[b], b>> \in h2b
          /\ \A b \in Blocks : b2h[b] # NoH => b \notin free
          /\ \A b \in cached : b2h[b] # NoH
\* a block holds the content its hash says, for every request referencing it at that position
ContentOK == \A r \in Req : hasT[r] =>
               \A i \in 1..Len(tbl[r]) : b2h[tbl[r][i]] # NoH => b2h[tbl[r][i]] = Hash(r, i)
NoDupInTable == \A r \in Req : hasT[r] =>
                  \A i, j \in 1..Len(tbl[r]) : i # j => tbl[r][i] # tbl[r][j]
=============================================================================
