------------------------------- MODULE CBLoop -------------------------------
(***************************************************************************)
(* Model of the transformers continuous-batching generation loop (main @    *)
(* 9b6e5e2), src/transformers/generation/continuous_batching/.              *)
(*                                                                         *)
(* Actors                                                                  *)
(*  - user threads: add_request (check + put, continuous_api.py:942-969),   *)
(*    cancel_request (:1007-1012), stop (:814-852)                          *)
(*  - generation loop thread: _run_generation_loop (:1106-1148) whose body  *)
(*    is _generation_loop_body (:1077-1104) = prepare_next_batch (:482-536)  *)
(*    + _generation_step + update_batch (:538-604).                         *)
(*                                                                         *)
(* Abstractions                                                            *)
(*  - The model forward is one atomic step producing one token per request *)
(*    with has_new_token.  EOS is optionally non-deterministic.             *)
(*  - One full-attention allocator; 1 block per sector (true for the tiny   *)
(*    Llama used in the reproducers), so "free" = free blocks + free sectors.*)
(*  - Prompts are pairwise distinct, so prefix matching never hits; with    *)
(*    SHARING we still track hashed blocks (hb) / cached blocks, and the    *)
(*    mark_complete_blocks block_table lookup (cache_allocator.py:294).     *)
(*  - CPU offloading disabled (cpu_offload_space default 0.0): offloading  *)
(*    = soft reset (offloading_manager.py:230-234).                         *)
(*  - cache_budget / decode fast path / max_requests_per_batch ignored     *)
(*    (no sharing => sum of reads <= capacity, see scheduler.py:226-235).   *)
(*  - Within one iteration the loop is atomic except that the input queue  *)
(*    and the cancel queue are drained at two different instants           *)
(*    (continuous_api.py:422, drain_queue twice).                           *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Req,          \* request ids
          Prompt,       \* [Req -> Nat] prompt length
          MaxNew,       \* [Req -> Nat] max_new_tokens
          PS,           \* page_size (tokens per block)
          NB,           \* number of blocks in the cache
          TB,           \* max_batch_tokens
          ASYNC,        \* use_async_batching
          SHARING,      \* use_prefix_sharing (full-attention model, allow_block_sharing=True)
          ALLOW_EOS,    \* model may emit EOS early
          STREAM,       \* streaming=True requests
          MN, MD,       \* safety_margin = MN/MD (FIFO default 0.15)
          ALLOW_CANCEL, \* users may call cancel_request
          MAX_STOP,     \* 0: never stop, 1: stop() (flush), 2: stop(hard_stop=True)
          ATOMIC_ADD,   \* TRUE: pretend check+put of add_request are atomic (hides the add/stop TOCTOU)
          FIX_BN,       \* proposed fix 1: a cancelled *active* request also resets block_new_requests
          FIX_CANCEL,   \* proposed fix 2: clear_cancelled_requests marks the cancelled state FINISHED
          FIX_HARD      \* proposed fix 3: on hard stop, drained requests are kept (and failed at exit)

PENDING == 0
PREFILLING == 1
DECODING == 2
FINISHED == 3

VARIABLES w,        \* loop-thread / scheduler / cache state (record)
          us,       \* user-side state of each request
          ucancel,  \* requests the user called cancel_request on
          stopReq   \* BackgroundThreadStatus._local_status requested by users (0,1,2)

vars == <<w, us, ucancel, stopReq>>

CeilDiv(a, b) == (a + b - 1) \div b
Remove(s, x) == SelectSeq(s, LAMBDA y : y # x)
InSeq(s, x) == \E i \in 1..Len(s) : s[i] = x
Min(a, b) == IF a < b THEN a ELSE b
Max(a, b) == IF a > b THEN a ELSE b

\* RequestState (requests.py:89-148).  nb = len(block_table[r]) or -1 if r has no block table.
\* hb = number of blocks of r that carry a content hash (BlockLedger.block_to_hash).
\* ep = identity of the RequestState object (soft reset creates a new object, offloading_manager.py:231).
\* plen = len(initial_tokens), tq = len(tokens_to_process).
InitSt(r) == [status |-> PENDING, pos |-> 0, rem |-> Prompt[r], gen |-> 0, M |-> MaxNew[r],
              nb |-> -1, hb |-> 0, ep |-> 0, plen |-> Prompt[r], tq |-> 0]

NoOut == [fin |-> 0, part |-> 0, err |-> 0, afterCancel |-> FALSE, tok |-> 0]

Init ==
  /\ w = [inQ |-> <<>>, pendIn |-> <<>>, cq |-> {}, tc |-> {}, wt |-> <<>>, act |-> <<>>,
          st |-> [r \in Req |-> InitSt(r)], free |-> NB, cached |-> 0, bn |-> FALSE,
          infl |-> <<>>, crashed |-> "none", out |-> [r \in Req |-> NoOut],
          tpStop |-> 0, loop |-> IF ASYNC THEN "boot" ELSE "run", phase |-> "drain",
          cancelled |-> {}, starved |-> <<>>, lost |-> {}]
  /\ us = [r \in Req |-> "new"]
  /\ ucancel = {}
  /\ stopReq = 0

(* ---------------------------------------------------------------------- *)
(* Cache allocator                                                         *)
(* ---------------------------------------------------------------------- *)

\* CacheAllocator.free_blocks (cache_allocator.py:233-239) + BlockLedger.release (:66-87):
\* hashed blocks are kept cached unless no_cache, others go back to the pool.
FreeBlocks(W, r, nocache) ==
  LET s == W.st[r]
      n == Max(s.nb, 0)
      h == IF SHARING /\ ~nocache THEN Min(s.hb, n) ELSE 0
  IN [W EXCEPT !.st[r].nb = -1, !.st[r].hb = 0, !.free = W.free + (n - h), !.cached = W.cached + h]

\* FullAttentionCacheAllocator._compute_blocks_needed (full_attention.py:118-122)
Need(s, len) == Max(CeilDiv(s.pos + len, PS) - Max(s.nb, 0), 0)

\* Scheduler._try_to_meet_safety_margin (scheduler.py:184-199)
InMargin(W) == W.free * MD >= MN * NB
TryMargin(W) == IF ~InMargin(W) /\ W.cached > 0
                THEN [W EXCEPT !.free = W.free + W.cached, !.cached = 0] ELSE W

(* ---------------------------------------------------------------------- *)
(* Scheduler (FIFOScheduler, scheduler.py:308-350)                         *)
(* ---------------------------------------------------------------------- *)
Candidates(W) ==
  LET dec == SelectSeq(W.act, LAMBDA r : W.st[r].status = DECODING)
      pre == SelectSeq(W.act, LAMBDA r : W.st[r].status = PREFILLING)
  IN dec \o pre \o (IF W.bn THEN <<>> ELSE W.wt)          \* scheduler.py:331-332

ReqTokLen(s) == IF s.status = DECODING THEN 1 ELSE s.rem  \* _infer_request_tokens, scheduler.py:131-137

\* Scheduler._process_candidates (scheduler.py:201-280) + _schedule_request (:139-182)
RECURSIVE PC(_, _, _, _, _, _, _)
PC(W, c, i, tb, sched, failed, inM) ==
  IF i > Len(c) THEN [w |-> W, sched |-> sched, failed |-> failed]
  ELSE
  LET r == c[i]
      s == W.st[r]
  IN IF ~inM /\ sched # <<>> /\ s.status # DECODING              \* :217-218
     THEN [w |-> W, sched |-> sched, failed |-> failed]
     ELSE
     LET len == ReqTokLen(s)
         rl == Min(len, tb)                                        \* :222
         need == Need(s, rl)
     IN IF need > W.free                                           \* :238-245 can_store_request_tokens fails
        THEN PC([W EXCEPT !.starved = IF InSeq(W.act, r)
                                      THEN Append(W.starved, [r |-> r, len |-> rl])
                                      ELSE W.starved],
                c, i + 1, tb, sched, TRUE, inM)
        ELSE
        LET s1 == [s EXCEPT !.nb = Max(s.nb, 0) + need]            \* allocate_cache_to_request
            wasPending == s.status = PENDING
            s2 == IF len <= tb
                  THEN IF s1.status <= PREFILLING
                       THEN [s1 EXCEPT !.status = DECODING, !.tq = s1.rem, !.rem = 0]
                       ELSE [s1 EXCEPT !.tq = 1]
                  ELSE [s1 EXCEPT !.status = PREFILLING, !.rem = len - tb, !.tq = tb]
            q == s2.tq
            cb == IF SHARING THEN ((s.pos % PS) + q) \div PS ELSE 0  \* count_new_complete_blocks, cache.py:335-346
            e == [r |-> r, ep |-> s.ep, q |-> q, nt |-> (s2.rem = 0), cb |-> cb]
            W2 == [W EXCEPT !.st[r] = s2, !.free = W.free - need,
                            !.act = IF wasPending THEN Append(W.act, r) ELSE W.act,
                            !.wt = Remove(W.wt, r)]
            tb2 == tb - q
        IN IF tb2 = 0 THEN [w |-> W2, sched |-> Append(sched, e), failed |-> failed]  \* :278
           ELSE PC(W2, c, i + 1, tb2, Append(sched, e), failed, inM)

ScheduleBatch(W) ==
  LET c == Candidates(W)
      W0 == TryMargin([W EXCEPT !.starved = <<>>])
      res == PC(W0, c, 1, TB, <<>>, FALSE, InMargin(W0))
  IN [w |-> res.w, sched |-> res.sched, none |-> (res.sched = <<>> /\ res.failed)]  \* :346-348

(* ---------------------------------------------------------------------- *)
(* OffloadingManager.offload_requests (offloading_manager.py:156-242)      *)
(* ---------------------------------------------------------------------- *)
RECURSIVE CS(_, _, _, _)
CS(W, lst, i, fr) == IF i > Len(lst) THEN Len(lst)
                     ELSE LET n == Need(W.st[lst[i].r], lst[i].len)
                          IN IF n > fr THEN i - 1 ELSE CS(W, lst, i + 1, fr - n)

RECURSIVE OffLoop(_, _, _)
OffLoop(W, off, numActive) ==
  IF W.starved = <<>> \/ numActive - Len(off) <= 1 THEN [w |-> W, off |-> off]          \* :183
  ELSE IF CS(W, W.starved, 1, W.free) = Len(W.starved) THEN [w |-> W, off |-> off]      \* :185-187
  ELSE LET last == W.starved[Len(W.starved)]                                            \* :190
           W1 == [W EXCEPT !.starved = SubSeq(W.starved, 1, Len(W.starved) - 1)]
       IN OffLoop(FreeBlocks(W1, last.r, TRUE), Append(off, last.r), numActive)        \* :198

\* soft reset of each victim, oldest first (:209-234): new RequestState object, old one -> FINISHED
RECURSIVE RQ(_, _, _)
RQ(W, off, i) ==
  IF i < 1 THEN W ELSE
  LET r == off[i]
      s == W.st[r]
      ns == [status |-> PENDING, pos |-> 0, rem |-> s.plen + s.gen, gen |-> 0, M |-> s.M - s.gen,
             nb |-> -1, hb |-> 0, ep |-> s.ep + 1, plen |-> s.plen + s.gen, tq |-> 0]
  IN RQ([W EXCEPT !.st[r] = ns, !.act = Remove(W.act, r), !.wt = Append(W.wt, r)], off, i - 1)

Offload(W) ==
  LET ev == W.cached > 0                                         \* evict_cached_blocks, :167
      W0 == [W EXCEPT !.free = W.free + W.cached, !.cached = 0]
  IN IF W0.starved = <<>> THEN [w |-> W0, ok |-> ev]
     ELSE LET res == OffLoop(W0, <<>>, Len(W0.act))
          IN IF res.off = <<>> THEN [w |-> res.w, ok |-> ev]
             ELSE [w |-> [RQ(res.w, res.off, Len(res.off)) EXCEPT !.bn = TRUE], ok |-> TRUE]  \* :236

(* ---------------------------------------------------------------------- *)
(* prepare_next_batch (continuous_api.py:482-536)                          *)
(* ---------------------------------------------------------------------- *)
\* Scheduler.clear_cancelled_requests (scheduler.py:96-113)
RECURSIVE CC(_, _)
CC(W, S) == IF S = {} THEN W ELSE
  LET r == CHOOSE x \in S : TRUE
      present == InSeq(W.act, r) \/ InSeq(W.wt, r)
      W1 == [W EXCEPT !.act = Remove(W.act, r), !.wt = Remove(W.wt, r),
                      !.cancelled = IF present THEN W.cancelled \cup {r} ELSE W.cancelled,
                      !.bn = IF FIX_BN /\ InSeq(W.act, r) THEN FALSE ELSE W.bn,
                      !.st[r].status = IF FIX_CANCEL /\ present THEN FINISHED ELSE W.st[r].status]
  IN CC(FreeBlocks(W1, r, FALSE), S \ {r})

RECURSIVE Retry(_, _)
Retry(sb, k) ==
  IF ~sb.none THEN [w |-> sb.w, sched |-> sb.sched, crash |-> "none"]
  ELSE IF k = 0 THEN [w |-> sb.w, sched |-> <<>>, crash |-> "RetryBound"]
  ELSE LET o == Offload(sb.w)
       IN IF ~o.ok THEN [w |-> o.w, sched |-> <<>>, crash |-> "NoRequestCanBeOffloaded"]  \* :508-509
          ELSE Retry(ScheduleBatch(o.w), k - 1)

\* position_offset += query_length (input_outputs.py:390)
RECURSIVE Adv(_, _, _)
Adv(W, b, i) == IF i > Len(b) THEN W
                ELSE Adv([W EXCEPT !.st[b[i].r].pos = @ + b[i].q,
                                   !.st[b[i].r].tq = IF b[i].nt THEN 1 ELSE @], b, i + 1)

\* Returns [w, ok, sched, crash].  Called with W after the input queue was drained into W.pendIn.
Prepare(W) ==
  LET tp == Max(W.tpStop, stopReq)                                 \* all-reduce of local_status, :427-434
  IN IF tp = 2                                                     \* :436-437 hard stop: payload dropped!
     THEN [w |-> IF FIX_HARD
                 THEN [W EXCEPT !.tpStop = tp, !.cq = {}, !.pendIn = <<>>, !.wt = W.wt \o W.pendIn]
                 ELSE [W EXCEPT !.tpStop = tp, !.cq = {}, !.pendIn = <<>>,
                           !.lost = W.lost \cup {W.pendIn[i] : i \in 1..Len(W.pendIn)}],
           ok |-> FALSE, sched |-> <<>>, crash |-> "none"]
     ELSE
     LET W1 == [W EXCEPT !.tpStop = tp, !.wt = W.wt \o W.pendIn, !.pendIn = <<>>,
                         !.tc = W.tc \cup W.cq, !.cq = {}]        \* :460-469
         W2 == [CC(W1, W1.tc) EXCEPT !.tc = {}]                    \* :491
     IN IF Len(W2.act) = 0 /\ Len(W2.wt) = 0                       \* :495-496
        THEN [w |-> W2, ok |-> FALSE, sched |-> <<>>, crash |-> "none"]
        ELSE
        LET r1 == Retry(ScheduleBatch(W2), 2 * Cardinality(Req) + 2)   \* :499-513
        IN IF r1.crash # "none" THEN [w |-> r1.w, ok |-> FALSE, sched |-> <<>>, crash |-> r1.crash]
           ELSE IF r1.sched = <<>> THEN [w |-> r1.w, ok |-> FALSE, sched |-> <<>>, crash |-> "none"]  \* :516-517
           ELSE
           LET W3 == IF r1.w.starved # <<>> THEN Offload(r1.w).w ELSE r1.w   \* :519-520
           IN [w |-> Adv(W3, r1.sched, 1), ok |-> TRUE, sched |-> r1.sched, crash |-> "none"]

(* ---------------------------------------------------------------------- *)
(* update_batch (continuous_api.py:538-577)                                *)
(* ---------------------------------------------------------------------- *)
RECURSIVE Upd(_, _, _, _)
Upd(W, b, i, eos) ==
  IF i > Len(b) \/ W.crashed # "none" THEN W ELSE
  LET e == b[i]
      r == e.r
      s == W.st[r]
      status == IF e.ep # s.ep THEN FINISHED ELSE s.status        \* stale (soft-reset) object is FINISHED
  IN IF status \in {FINISHED, PENDING}                             \* :546-552
     THEN IF ASYNC THEN Upd(W, b, i + 1, eos)
          ELSE [W EXCEPT !.crashed = "SyncUpdateOfFinishedRequest"]
     ELSE IF e.nt
     THEN
       LET s1 == IF s.gen = 0 THEN [s EXCEPT !.status = DECODING] ELSE s   \* :556-557
           isE == r \in eos
           g2 == IF isE \/ s1.gen < s1.M THEN s1.gen + 1 ELSE s1.gen      \* requests.py:223-228
           finished == isE \/ g2 >= s1.M                                    \* requests.py:232-234
           s2 == [s1 EXCEPT !.gen = g2, !.status = IF finished THEN FINISHED ELSE s1.status]
           W1 == [W EXCEPT !.st[r] = s2]
           mcb == SHARING /\ e.cb > 0 /\ ~finished                          \* cache.py:353
       IN IF mcb /\ s2.nb < 0
          THEN [W1 EXCEPT !.crashed = "KeyErrorInMarkCompleteBlocks"]       \* cache_allocator.py:294
          ELSE
          LET W2 == IF mcb THEN [W1 EXCEPT !.st[r].hb = Min(@ + e.cb, W1.st[r].nb)] ELSE W1
              W3 == IF finished                                             \* :567-569
                    THEN [FreeBlocks(W2, r, FALSE) EXCEPT !.act = Remove(W2.act, r), !.bn = FALSE]
                    ELSE W2
              W4 == IF STREAM \/ finished                                   \* :570-571
                    THEN [W3 EXCEPT !.out[r].fin = @ + (IF finished THEN 1 ELSE 0),
                                    !.out[r].part = @ + (IF finished THEN 0 ELSE 1),
                                    !.out[r].afterCancel = @ \/ (r \in W3.cancelled),
                                    !.out[r].tok = (s2.plen - Prompt[r]) + s2.gen]
                    ELSE W3
          IN Upd(W4, b, i + 1, eos)
     ELSE IF s.status = PREFILLING /\ SHARING /\ e.cb > 0                   \* :573-574
          THEN IF s.nb < 0 THEN [W EXCEPT !.crashed = "KeyErrorInMarkCompleteBlocks"]
               ELSE Upd([W EXCEPT !.st[r].hb = Min(@ + e.cb, s.nb)], b, i + 1, eos)
          ELSE Upd(W, b, i + 1, eos)

(* ---------------------------------------------------------------------- *)
(* exit paths: _fail_all_remaining_requests (:1232-1246), fail_all_requests *)
(* ---------------------------------------------------------------------- *)
RECURSIVE FailSeq(_, _, _)
FailSeq(W, sq, i) == IF i > Len(sq) THEN W
                     ELSE FailSeq([W EXCEPT !.out[sq[i]].err = @ + 1], sq, i + 1)

FailAll(W) ==
  LET W1 == FailSeq(W, W.inQ, 1)
      W2 == FailSeq(W1, W1.act, 1)
      W3 == FailSeq(W2, W2.wt, 1)
      W4 == [r \in Req |-> r]
  IN [W3 EXCEPT !.inQ = <<>>, !.act = <<>>, !.wt = <<>>, !.loop = "exited",
                !.free = NB, !.cached = 0,  \* free_all at reset (not needed for checking)
                !.st = [r \in Req |-> [W3.st[r] EXCEPT !.nb = -1, !.hb = 0]]]

Exit(W) ==
  LET W1 == IF ASYNC THEN Upd(W, W.infl, 1, {}) ELSE W              \* :1132-1134
  IN FailAll([W1 EXCEPT !.infl = <<>>])

Crash(W, why) == FailAll([W EXCEPT !.crashed = why])                 \* _handle_critical_error

(* ---------------------------------------------------------------------- *)
(* Actions                                                                 *)
(* ---------------------------------------------------------------------- *)
LoopDrainInput ==
  /\ w.loop # "exited" /\ w.crashed = "none" /\ w.phase = "drain"
  /\ w' = [w EXCEPT !.pendIn = w.inQ, !.inQ = <<>>, !.phase = "main"]
  /\ UNCHANGED <<us, ucancel, stopReq>>

LoopMain ==
  /\ w.loop # "exited" /\ w.crashed = "none" /\ w.phase = "main"
  /\ \E eos \in (IF ALLOW_EOS THEN SUBSET Req ELSE {{}}) :
     LET p == Prepare(w)
         W == [p.w EXCEPT !.phase = "drain"]
     IN IF p.crash # "none" THEN w' = Crash(W, p.crash)
        ELSE IF p.ok
        THEN IF ~ASYNC
             THEN LET W1 == Upd(W, p.sched, 1, eos)
                  IN w' = IF W1.crashed # "none" THEN Crash(W1, W1.crashed) ELSE W1
             ELSE IF W.loop = "boot"                               \* :1086-1087 no update when bootstrapping
                  THEN w' = [W EXCEPT !.infl = p.sched, !.loop = "run"]
                  ELSE LET W1 == Upd(W, W.infl, 1, eos)            \* update of the PREVIOUS batch
                       IN w' = IF W1.crashed # "none" THEN Crash(W1, W1.crashed)
                               ELSE [W1 EXCEPT !.infl = p.sched]
        ELSE IF W.tpStop = 2 THEN w' = Exit(W)                     \* :1092-1093
        ELSE IF W.tpStop = 1 /\ Len(W.act) = 0 /\ Len(W.wt) = 0     \* :1095-1099
             THEN w' = Exit(W)
        ELSE w' = W                                                \* idle: wait(0.1) and retry
  /\ UNCHANGED <<us, ucancel, stopReq>>

\* add_request: can_accept_new_requests check (:942-946) then input_queue.put (:968)
CanAccept == stopReq = 0 /\ w.crashed = "none" /\ w.loop # "exited"
UserAddCheck(r) ==
  /\ us[r] = "new"
  /\ IF ATOMIC_ADD /\ CanAccept
     THEN us' = [us EXCEPT ![r] = "put"] /\ w' = [w EXCEPT !.inQ = Append(w.inQ, r)]
     ELSE us' = [us EXCEPT ![r] = IF CanAccept THEN "checked" ELSE "dropped"] /\ UNCHANGED w
  /\ UNCHANGED <<ucancel, stopReq>>

UserAddPut(r) ==
  /\ us[r] = "checked"
  /\ us' = [us EXCEPT ![r] = "put"]
  /\ w' = [w EXCEPT !.inQ = Append(w.inQ, r)]
  /\ UNCHANGED <<ucancel, stopReq>>

UserCancel(r) ==
  /\ ALLOW_CANCEL /\ us[r] = "put" /\ r \notin ucancel
  /\ ucancel' = ucancel \cup {r}
  /\ w' = [w EXCEPT !.cq = w.cq \cup {r}]
  /\ UNCHANGED <<us, stopReq>>

UserStop(k) ==
  /\ k \in 1..MAX_STOP /\ k > stopReq
  /\ stopReq' = k
  /\ UNCHANGED <<w, us, ucancel>>

Next ==
  \/ LoopDrainInput \/ LoopMain
  \/ \E r \in Req : UserAddCheck(r) \/ UserAddPut(r) \/ UserCancel(r)
  \/ \E k \in 1..2 : UserStop(k)

Fairness == /\ WF_vars(LoopDrainInput) /\ WF_vars(LoopMain)
            /\ \A r \in Req : WF_vars(UserAddPut(r))

Spec == Init /\ [][Next]_vars /\ Fairness

(* ---------------------------------------------------------------------- *)
(* Properties                                                              *)
(* ---------------------------------------------------------------------- *)
TypeOK == w.free >= 0 /\ w.cached >= 0

\* every block is free, cached, or in exactly one block table (no sharing between live requests here)
RECURSIVE SumNb(_, _)
SumNb(W, X) == IF X = {} THEN 0 ELSE LET r == CHOOSE x \in X : TRUE IN Max(W.st[r].nb, 0) + SumNb(W, X \ {r})
BlockConservation == w.loop = "exited" \/ w.free + w.cached + SumNb(w, Req) = NB

NoCrash == w.crashed = "none"
NoOutputAfterCancel == \A r \in Req : ~w.out[r].afterCancel
AtMostOneFinal == \A r \in Req : w.out[r].fin + w.out[r].err <= 1
NoLostOnHardStop == w.lost = {}
\* without EOS, a request that finished produced exactly max_new_tokens tokens (soft resets included)
TokenCount == ~ALLOW_EOS => \A r \in Req : w.out[r].fin = 1 => w.out[r].tok = MaxNew[r]

Answered(r) == w.out[r].fin + w.out[r].err >= 1
\* every accepted (put) request that is not cancelled is eventually answered (finished or failed)
EveryRequestAnswered == \A r \in Req : (us[r] = "put") ~> (Answered(r) \/ r \in ucancel)
\* stop() (block=True, timeout=None) terminates
StopTerminates == (stopReq >= 1) ~> (w.loop = "exited")
=============================================================================
