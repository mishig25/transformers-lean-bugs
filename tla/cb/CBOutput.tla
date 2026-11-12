------------------------------ MODULE CBOutput ------------------------------
(***************************************************************************)
(* Output delivery of the CB manager (continuous_api.py) and its consumers *)
(*  - producer: update_batch -> OutputRouter.deliver_batch (:110-131): if a *)
(*    handler is registered for the request, the output goes to the asyncio *)
(*    loop (call_soon_threadsafe), else to the shared output_queue.        *)
(*  - get_result(request_id) (:1016-1031): pops the head, and if it is for  *)
(*    another request, puts it back at the TAIL and returns None.          *)
(*  - request_id_iter (:1040-1051): loops while the thread is alive,        *)
(*    returns after yielding a finished output.                           *)
(*  - __iter__ (:1033-1038): loops while the generation thread is alive.    *)
(*  - transformers serve streaming (cli/serving/utils.py:884-937):          *)
(*    add_request(...) THEN register_result_handler(...).                  *)
(* Each request r produces outputs 1..K[r] in order (K[r] is the finished  *)
(* one; streaming requests produce one output per token).                  *)
(***************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS Req,        \* requests
          K,          \* [Req -> Nat] number of outputs of each request (last one is finished)
          ServeReq,   \* requests submitted by `transformers serve` generate_streaming (handler path)
          RidReq,     \* requests consumed by their own request_id_iter thread (queue path)
          DocIter,    \* TRUE: one thread runs the documented `for result in manager:` loop inside the
                      \*       continuous_batching_context_manager block (stop only after the loop ends)
          SERVE_FIXED \* TRUE: register_result_handler before add_request (proposed fix)

VARIABLES added,      \* requests whose add_request put has happened
          produced,   \* [Req -> Nat] outputs produced so far
          handler,    \* requests with a registered result handler
          q,          \* shared output_queue: Seq of [r, i]
          ev,         \* [Req -> Seq(Nat)] outputs scheduled on the asyncio loop for the handler
          got,        \* [Req -> Seq(Nat)] outputs seen by the consumer of each request (in order)
          held,       \* [Req -> [r, i] or none]: output popped by request_id_iter(r) but not for r (being put back)
          ridDone,    \* request_id_iter(r) returned
          docSeen,    \* outputs seen by the documented `for result in manager` loop
          docDone,    \* the `for` loop terminated
          alive       \* generation thread alive

vars == <<added, produced, handler, q, ev, got, held, ridDone, docSeen, docDone, alive>>
None == [r |-> "none", i |-> 0]

Init == /\ added = {} /\ produced = [r \in Req |-> 0] /\ handler = {} /\ q = <<>>
        /\ ev = [r \in Req |-> <<>>] /\ got = [r \in Req |-> <<>>] /\ held = [r \in Req |-> None]
        /\ ridDone = {} /\ docSeen = 0 /\ docDone = FALSE /\ alive = TRUE

\* ---- user side: submission ----
Add(r) == /\ r \notin added
          /\ (r \in ServeReq /\ SERVE_FIXED) => r \in handler
          /\ added' = added \cup {r}
          /\ UNCHANGED <<produced, handler, q, ev, got, held, ridDone, docSeen, docDone, alive>>
Register(r) == /\ r \in ServeReq /\ r \notin handler
               /\ SERVE_FIXED \/ r \in added                 \* serve: add_request first (utils.py:902-936)
               /\ handler' = handler \cup {r}
               /\ UNCHANGED <<added, produced, q, ev, got, held, ridDone, docSeen, docDone, alive>>

\* ---- generation loop: produce and route the next output of r ----
Produce(r) == /\ alive /\ r \in added /\ produced[r] < K[r]
              /\ produced' = [produced EXCEPT ![r] = @ + 1]
              /\ IF r \in handler
                 THEN ev' = [ev EXCEPT ![r] = Append(@, produced[r] + 1)] /\ UNCHANGED q
                 ELSE q' = Append(q, [r |-> r, i |-> produced[r] + 1]) /\ UNCHANGED ev
              /\ UNCHANGED <<added, handler, got, held, ridDone, docSeen, docDone, alive>>

\* ---- asyncio loop runs the handler callbacks ----
Handle(r) == /\ ev[r] # <<>>
             /\ got' = [got EXCEPT ![r] = Append(@, Head(ev[r]))]
             /\ ev' = [ev EXCEPT ![r] = Tail(@)]
             /\ UNCHANGED <<added, produced, handler, q, held, ridDone, docSeen, docDone, alive>>

\* ---- request_id_iter(r) thread ----
RidGet(r) == /\ r \in RidReq /\ r \in added /\ r \notin ridDone /\ held[r] = None
             /\ alive /\ q # <<>>                              \* while thread alive: get_result(r, timeout)
             /\ LET x == Head(q) IN
                  /\ q' = Tail(q)
                  /\ IF x.r = r
                     THEN /\ got' = [got EXCEPT ![r] = Append(@, x.i)]
                          /\ ridDone' = IF x.i = K[r] THEN ridDone \cup {r} ELSE ridDone  \* :1050-1051
                          /\ UNCHANGED held
                     ELSE /\ held' = [held EXCEPT ![r] = x] /\ UNCHANGED <<got, ridDone>>
             /\ UNCHANGED <<added, produced, handler, ev, docSeen, docDone, alive>>
RidPutBack(r) == /\ held[r] # None                              \* :1027 output_queue.put(result)
                 /\ q' = Append(q, held[r]) /\ held' = [held EXCEPT ![r] = None]
                 /\ UNCHANGED <<added, produced, handler, ev, got, ridDone, docSeen, docDone, alive>>

\* ---- documented `for result in manager:` loop ----
DocGet == /\ DocIter /\ ~docDone /\ alive /\ q # <<>>
          /\ q' = Tail(q) /\ docSeen' = docSeen + 1
          /\ UNCHANGED <<added, produced, handler, ev, got, held, ridDone, docDone, alive>>
DocLoopExit == /\ DocIter /\ ~docDone /\ ~alive                  \* `while ... is_alive()` fails
               /\ docDone' = TRUE
               /\ UNCHANGED <<added, produced, handler, q, ev, got, held, ridDone, docSeen, alive>>
\* the context manager calls stop() only once the body (the for loop) is done; without DocIter, some
\* other thread may stop the manager at any time after all requests were produced
Stop == /\ alive /\ (DocIter => docDone) /\ (\A r \in Req : produced[r] = K[r])
        /\ alive' = FALSE
        /\ UNCHANGED <<added, produced, handler, q, ev, got, held, ridDone, docSeen, docDone>>

Next == \/ \E r \in Req : Add(r) \/ Register(r) \/ Produce(r) \/ Handle(r) \/ RidGet(r) \/ RidPutBack(r)
        \/ DocGet \/ DocLoopExit \/ Stop

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
        /\ \A r \in Req : WF_vars(Add(r)) /\ WF_vars(Register(r)) /\ WF_vars(Produce(r)) /\ WF_vars(Handle(r))
                          /\ SF_vars(RidGet(r)) /\ WF_vars(RidPutBack(r))
        /\ WF_vars(DocGet) /\ WF_vars(DocLoopExit)

\* ---- properties ----
IsOrdered(s) == \A i \in 1..(Len(s) - 1) : s[i] < s[i + 1]
\* every consumer sees the outputs of its request in production order (streaming deltas are well defined)
InOrder == \A r \in Req : IsOrdered(got[r])
\* a serve streaming request eventually reaches its handler with the finished output (the SSE stream ends)
ServeStreamEnds == \A r \in ServeReq : <>(\E i \in 1..Len(got[r]) : got[r][i] = K[r])
\* request_id_iter(r) eventually yields the finished output of r
RidGetsFinal == \A r \in RidReq : <>(r \in ridDone)
\* the documented example terminates
DocTerminates == DocIter => <>(docDone)
\* no stale output is left in the shared queue forever
QueueDrained == <>[](\A j \in 1..Len(q) : ~(q[j].r \in RidReq))
=============================================================================
