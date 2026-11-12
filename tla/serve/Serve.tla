------------------------------- MODULE Serve -------------------------------
(* transformers `transformers serve` (src/transformers/cli/serving/): ModelManager + TimedModel idle unload,
   GenerationState/CBGenerateManager (continuous batching), CB OutputRouter.

   One model id K. Copies of the model are numbered 1, 2, ... (each from_pretrained makes a new copy).
   - Timer (threading.Timer in TimedModel): may fire whenever armed (abstracts "timeout_seconds without a reset").
     Its callback delete_model runs WITHOUT the per-model lock: model := None, gc/empty_cache, timer.cancel(),
     on_unload -> loaded_models.pop(K).  reset_timer() = cancel() + new armed timer; cancel() cannot stop a
     callback that is already running.
   - Request handler (async, on the event loop): load_model_and_processor under the per-model lock
     (cached branch: reset_timer, then read .model / .processor), then get_manager(K, use_cb) + init_cb(model)
     (no-op once the CB manager has a model), then streaming CB generation:
     cb.add_request(...) ; [CBStreamer(...) built] ; cb.register_result_handler(rid, _on_output)
     The SSE stream ends only when _on_output sees a finished (or failed) output.
   - CB background thread: produces the (single, finishing: max_new_tokens = 1) output of each request and routes
     it with OutputRouter.deliver: handler registered -> call_soon_threadsafe(callback) else output_queue.
*)
EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS Requests, UseCB, MaxCopies, Fixed

(* --algorithm serve
variables
  cached = 0,              \* generation (= model copy id) of the TimedModel in loaded_models[K]; 0 = absent
  tmM = [g \in 1..MaxCopies |-> 0],   \* TimedModel g's .model (0 = None)
  nextCopy = 1,
  armed = {},              \* TimedModels with a pending threading.Timer
  lockHolder = "none",     \* per-model threading.Lock
  cbModel = 0,             \* model held by the CB manager (init_cb), 0 = not initialised
  cbQueue = <<>>,          \* requests added to CB, not yet processed by the background thread
  handlers = {},           \* OutputRouter.result_handlers keys
  outputQueue = {},        \* OutputRouter.output_queue (serve never reads it)
  scheduled = {},          \* outputs routed to a registered handler (callback queued on the event loop)
  streamEnded = [r \in Requests |-> FALSE],
  local = [r \in Requests |-> 0],   \* model copy held by the handler's local variable
  active = 0;              \* (fix) requests currently using the model

define
  LiveCopies == ({tmM[g] : g \in 1..MaxCopies} \cup {cbModel} \cup {local[r] : r \in Requests}) \ {0}
  OneCopy == Cardinality(LiveCopies) <= 1                      \* never two copies of the model in memory
  CBUsesCached == (UseCB /\ cbModel /= 0 /\ cached /= 0 /\ tmM[cached] /= 0) => cbModel = tmM[cached]
  NoHang == \A r \in Requests : pc[r] /= "Hang"
  NoFail == \A r \in Requests : pc[r] /= "Fail"
end define;

fair process timer = "timer"
variables tg = 0;
begin
T0: while TRUE do
      await armed /= {};
      with x \in armed do tg := x; armed := armed \ {x}; end with;   \* that Timer fires (idle timeout)
T1:   if Fixed then await lockHolder = "none"; lockHolder := "timer"; end if;
T1b:  if tmM[tg] /= 0 /\ ~(Fixed /\ active > 0) then        \* delete_model guard (+ fix: not while in use)
        tmM[tg] := 0;                                      \* del self.model; self.model = None
        if Fixed /\ cbModel = tg then cbModel := 0; end if;  \* fix: stop the CB manager holding this model
T2:     skip;                                             \* gc.collect(); reset_torch_cache(); self._timer.cancel()
T3:     cached := 0;                                      \* on_unload: loaded_models.pop(K)
      end if;
T4:   if Fixed then lockHolder := "none"; end if;
    end while;
end process;

fair process cbThread = "cb"
begin
B0: while TRUE do
      await cbQueue /= <<>>;
      with r = Head(cbQueue) do
        if r \in handlers then scheduled := scheduled \cup {r};   \* deliver -> call_soon_threadsafe(callback)
        else outputQueue := outputQueue \cup {r};         \* no handler registered yet -> output_queue
        end if;
        cbQueue := Tail(cbQueue);
      end with;
    end while;
end process;

fair process loopCallbacks = "loop"                      \* event loop runs _on_output for delivered outputs
begin
E0: while TRUE do
      with r \in scheduled do
        streamEnded[r] := TRUE;                           \* finished output -> streamer.end()
        handlers := handlers \ {r}; scheduled := scheduled \ {r};
      end with;
    end while;
end process;

fair process request \in Requests
begin
L0: await lockHolder = "none"; lockHolder := self; active := active + 1;  \* with lock:
L1: if cached = 0 then                                    \* key not in loaded_models
      local[self] := nextCopy; tmM[nextCopy] := nextCopy; cached := nextCopy;
      armed := armed \cup {nextCopy}; nextCopy := nextCopy + 1;          \* from_pretrained + TimedModel
      goto L3;
    else
      armed := armed \cup {cached};                      \* reset_timer(): cancel + new Timer (cannot stop a running one)
    end if;
L2: if cached = 0 then goto Fail;                         \* loaded_models[key] -> KeyError
    else local[self] := tmM[cached];                      \* .model (None while delete_model runs)
    end if;
L3: lockHolder := "none";
    if local[self] = 0 then goto Fail; end if;            \* handler would use model=None
Gen: if UseCB then
      if cbModel = 0 then cbModel := local[self]; end if; \* init_cb: no-op once initialised
      if Fixed then handlers := handlers \cup {self}; end if;   \* fix: register BEFORE add_request
Add:  cbQueue := Append(cbQueue, self);                  \* cb.add_request(...)
Build: skip;                                              \* CBStreamer(...) constructed
Reg:  handlers := handlers \cup {self};                   \* cb.register_result_handler(rid, _on_output)
Wait: await streamEnded[self] \/ self \in outputQueue;
      if self \in outputQueue then goto Hang; end if;     \* the finished output went to output_queue
    end if;
Done1: local[self] := 0; active := active - 1;           \* handler done, drops its reference
    goto Done;
Fail: local[self] := 0; active := active - 1; goto Done;
Hang: await FALSE;                                        \* SSE stream never ends
end process;
end algorithm; *)
\* BEGIN TRANSLATION
VARIABLES pc, cached, tmM, nextCopy, armed, lockHolder, cbModel, cbQueue, 
          handlers, outputQueue, scheduled, streamEnded, local, active

(* define statement *)
LiveCopies == ({tmM[g] : g \in 1..MaxCopies} \cup {cbModel} \cup {local[r] : r \in Requests}) \ {0}
OneCopy == Cardinality(LiveCopies) <= 1
CBUsesCached == (UseCB /\ cbModel /= 0 /\ cached /= 0 /\ tmM[cached] /= 0) => cbModel = tmM[cached]
NoHang == \A r \in Requests : pc[r] /= "Hang"
NoFail == \A r \in Requests : pc[r] /= "Fail"

VARIABLE tg

vars == << pc, cached, tmM, nextCopy, armed, lockHolder, cbModel, cbQueue, 
           handlers, outputQueue, scheduled, streamEnded, local, active, tg
        >>

ProcSet == {"timer"} \cup {"cb"} \cup {"loop"} \cup (Requests)

Init == (* Global variables *)
        /\ cached = 0
        /\ tmM = [g \in 1..MaxCopies |-> 0]
        /\ nextCopy = 1
        /\ armed = {}
        /\ lockHolder = "none"
        /\ cbModel = 0
        /\ cbQueue = <<>>
        /\ handlers = {}
        /\ outputQueue = {}
        /\ scheduled = {}
        /\ streamEnded = [r \in Requests |-> FALSE]
        /\ local = [r \in Requests |-> 0]
        /\ active = 0
        (* Process timer *)
        /\ tg = 0
        /\ pc = [self \in ProcSet |-> CASE self = "timer" -> "T0"
                                        [] self = "cb" -> "B0"
                                        [] self = "loop" -> "E0"
                                        [] self \in Requests -> "L0"]

T0 == /\ pc["timer"] = "T0"
      /\ armed /= {}
      /\ \E x \in armed:
           /\ tg' = x
           /\ armed' = armed \ {x}
      /\ pc' = [pc EXCEPT !["timer"] = "T1"]
      /\ UNCHANGED << cached, tmM, nextCopy, lockHolder, cbModel, cbQueue, 
                      handlers, outputQueue, scheduled, streamEnded, local, 
                      active >>

T1 == /\ pc["timer"] = "T1"
      /\ IF Fixed
            THEN /\ lockHolder = "none"
                 /\ lockHolder' = "timer"
            ELSE /\ TRUE
                 /\ UNCHANGED lockHolder
      /\ pc' = [pc EXCEPT !["timer"] = "T1b"]
      /\ UNCHANGED << cached, tmM, nextCopy, armed, cbModel, cbQueue, handlers, 
                      outputQueue, scheduled, streamEnded, local, active, tg >>

T1b == /\ pc["timer"] = "T1b"
       /\ IF tmM[tg] /= 0 /\ ~(Fixed /\ active > 0)
             THEN /\ tmM' = [tmM EXCEPT ![tg] = 0]
                  /\ IF Fixed /\ cbModel = tg
                        THEN /\ cbModel' = 0
                        ELSE /\ TRUE
                             /\ UNCHANGED cbModel
                  /\ pc' = [pc EXCEPT !["timer"] = "T2"]
             ELSE /\ pc' = [pc EXCEPT !["timer"] = "T4"]
                  /\ UNCHANGED << tmM, cbModel >>
       /\ UNCHANGED << cached, nextCopy, armed, lockHolder, cbQueue, handlers, 
                       outputQueue, scheduled, streamEnded, local, active, tg >>

T2 == /\ pc["timer"] = "T2"
      /\ TRUE
      /\ pc' = [pc EXCEPT !["timer"] = "T3"]
      /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, cbModel, 
                      cbQueue, handlers, outputQueue, scheduled, streamEnded, 
                      local, active, tg >>

T3 == /\ pc["timer"] = "T3"
      /\ cached' = 0
      /\ pc' = [pc EXCEPT !["timer"] = "T4"]
      /\ UNCHANGED << tmM, nextCopy, armed, lockHolder, cbModel, cbQueue, 
                      handlers, outputQueue, scheduled, streamEnded, local, 
                      active, tg >>

T4 == /\ pc["timer"] = "T4"
      /\ IF Fixed
            THEN /\ lockHolder' = "none"
            ELSE /\ TRUE
                 /\ UNCHANGED lockHolder
      /\ pc' = [pc EXCEPT !["timer"] = "T0"]
      /\ UNCHANGED << cached, tmM, nextCopy, armed, cbModel, cbQueue, handlers, 
                      outputQueue, scheduled, streamEnded, local, active, tg >>

timer == T0 \/ T1 \/ T1b \/ T2 \/ T3 \/ T4

B0 == /\ pc["cb"] = "B0"
      /\ cbQueue /= <<>>
      /\ LET r == Head(cbQueue) IN
           /\ IF r \in handlers
                 THEN /\ scheduled' = (scheduled \cup {r})
                      /\ UNCHANGED outputQueue
                 ELSE /\ outputQueue' = (outputQueue \cup {r})
                      /\ UNCHANGED scheduled
           /\ cbQueue' = Tail(cbQueue)
      /\ pc' = [pc EXCEPT !["cb"] = "B0"]
      /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, cbModel, 
                      handlers, streamEnded, local, active, tg >>

cbThread == B0

E0 == /\ pc["loop"] = "E0"
      /\ \E r \in scheduled:
           /\ streamEnded' = [streamEnded EXCEPT ![r] = TRUE]
           /\ handlers' = handlers \ {r}
           /\ scheduled' = scheduled \ {r}
      /\ pc' = [pc EXCEPT !["loop"] = "E0"]
      /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, cbModel, 
                      cbQueue, outputQueue, local, active, tg >>

loopCallbacks == E0

L0(self) == /\ pc[self] = "L0"
            /\ lockHolder = "none"
            /\ lockHolder' = self
            /\ active' = active + 1
            /\ pc' = [pc EXCEPT ![self] = "L1"]
            /\ UNCHANGED << cached, tmM, nextCopy, armed, cbModel, cbQueue, 
                            handlers, outputQueue, scheduled, streamEnded, 
                            local, tg >>

L1(self) == /\ pc[self] = "L1"
            /\ IF cached = 0
                  THEN /\ local' = [local EXCEPT ![self] = nextCopy]
                       /\ tmM' = [tmM EXCEPT ![nextCopy] = nextCopy]
                       /\ cached' = nextCopy
                       /\ armed' = (armed \cup {nextCopy})
                       /\ nextCopy' = nextCopy + 1
                       /\ pc' = [pc EXCEPT ![self] = "L3"]
                  ELSE /\ armed' = (armed \cup {cached})
                       /\ pc' = [pc EXCEPT ![self] = "L2"]
                       /\ UNCHANGED << cached, tmM, nextCopy, local >>
            /\ UNCHANGED << lockHolder, cbModel, cbQueue, handlers, 
                            outputQueue, scheduled, streamEnded, active, tg >>

L2(self) == /\ pc[self] = "L2"
            /\ IF cached = 0
                  THEN /\ pc' = [pc EXCEPT ![self] = "Fail"]
                       /\ local' = local
                  ELSE /\ local' = [local EXCEPT ![self] = tmM[cached]]
                       /\ pc' = [pc EXCEPT ![self] = "L3"]
            /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, cbModel, 
                            cbQueue, handlers, outputQueue, scheduled, 
                            streamEnded, active, tg >>

L3(self) == /\ pc[self] = "L3"
            /\ lockHolder' = "none"
            /\ IF local[self] = 0
                  THEN /\ pc' = [pc EXCEPT ![self] = "Fail"]
                  ELSE /\ pc' = [pc EXCEPT ![self] = "Gen"]
            /\ UNCHANGED << cached, tmM, nextCopy, armed, cbModel, cbQueue, 
                            handlers, outputQueue, scheduled, streamEnded, 
                            local, active, tg >>

Gen(self) == /\ pc[self] = "Gen"
             /\ IF UseCB
                   THEN /\ IF cbModel = 0
                              THEN /\ cbModel' = local[self]
                              ELSE /\ TRUE
                                   /\ UNCHANGED cbModel
                        /\ IF Fixed
                              THEN /\ handlers' = (handlers \cup {self})
                              ELSE /\ TRUE
                                   /\ UNCHANGED handlers
                        /\ pc' = [pc EXCEPT ![self] = "Add"]
                   ELSE /\ pc' = [pc EXCEPT ![self] = "Done1"]
                        /\ UNCHANGED << cbModel, handlers >>
             /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, cbQueue, 
                             outputQueue, scheduled, streamEnded, local, 
                             active, tg >>

Add(self) == /\ pc[self] = "Add"
             /\ cbQueue' = Append(cbQueue, self)
             /\ pc' = [pc EXCEPT ![self] = "Build"]
             /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, cbModel, 
                             handlers, outputQueue, scheduled, streamEnded, 
                             local, active, tg >>

Build(self) == /\ pc[self] = "Build"
               /\ TRUE
               /\ pc' = [pc EXCEPT ![self] = "Reg"]
               /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, 
                               cbModel, cbQueue, handlers, outputQueue, 
                               scheduled, streamEnded, local, active, tg >>

Reg(self) == /\ pc[self] = "Reg"
             /\ handlers' = (handlers \cup {self})
             /\ pc' = [pc EXCEPT ![self] = "Wait"]
             /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, cbModel, 
                             cbQueue, outputQueue, scheduled, streamEnded, 
                             local, active, tg >>

Wait(self) == /\ pc[self] = "Wait"
              /\ streamEnded[self] \/ self \in outputQueue
              /\ IF self \in outputQueue
                    THEN /\ pc' = [pc EXCEPT ![self] = "Hang"]
                    ELSE /\ pc' = [pc EXCEPT ![self] = "Done1"]
              /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, 
                              cbModel, cbQueue, handlers, outputQueue, 
                              scheduled, streamEnded, local, active, tg >>

Done1(self) == /\ pc[self] = "Done1"
               /\ local' = [local EXCEPT ![self] = 0]
               /\ active' = active - 1
               /\ pc' = [pc EXCEPT ![self] = "Done"]
               /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, 
                               cbModel, cbQueue, handlers, outputQueue, 
                               scheduled, streamEnded, tg >>

Fail(self) == /\ pc[self] = "Fail"
              /\ local' = [local EXCEPT ![self] = 0]
              /\ active' = active - 1
              /\ pc' = [pc EXCEPT ![self] = "Done"]
              /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, 
                              cbModel, cbQueue, handlers, outputQueue, 
                              scheduled, streamEnded, tg >>

Hang(self) == /\ pc[self] = "Hang"
              /\ FALSE
              /\ pc' = [pc EXCEPT ![self] = "Done"]
              /\ UNCHANGED << cached, tmM, nextCopy, armed, lockHolder, 
                              cbModel, cbQueue, handlers, outputQueue, 
                              scheduled, streamEnded, local, active, tg >>

request(self) == L0(self) \/ L1(self) \/ L2(self) \/ L3(self) \/ Gen(self)
                    \/ Add(self) \/ Build(self) \/ Reg(self) \/ Wait(self)
                    \/ Done1(self) \/ Fail(self) \/ Hang(self)

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == timer \/ cbThread \/ loopCallbacks
           \/ (\E self \in Requests: request(self))
           \/ Terminating

Spec == /\ Init /\ [][Next]_vars
        /\ WF_vars(timer)
        /\ WF_vars(cbThread)
        /\ WF_vars(loopCallbacks)
        /\ \A self \in Requests : WF_vars(request(self))

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION
=============================================================================
