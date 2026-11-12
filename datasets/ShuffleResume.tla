------------------------------ MODULE ShuffleResume ------------------------------
(***************************************************************************)
(* Checkpoint/resume of a shuffled streaming dataset (datasets main @ efc8247). *)
(*                                                                         *)
(* Pipeline built by IterableDataset.shuffle() + __iter__:                 *)
(*   source (ArrowExamplesIterable, tables of Bin rows)                    *)
(*   -> BufferShuffledExamplesIterable  (iterable_dataset.py:2025-2160)    *)
(*   -> RebatchedArrowExamplesIterable(batch_size=1), added by             *)
(*      _prepare_ex_iterable_for_iteration (iterable_dataset.py:3030-3033) *)
(*                                                                         *)
(* Mode = "arrow"  : current code, arrow-backed source (parquet/json/      *)
(*                   to_iterable_dataset): top Rebatched checkpoints       *)
(*                   (previous_state, num_chunks_since_previous_state)     *)
(*                   and resumes by replaying the child (lines 550-634).   *)
(* Mode = "python" : generator-backed source (no iter_arrow): the state is *)
(*                   the live source position (BufferShuffled.__iter__).  *)
(* Mode = "fixed"  : proposed fix: snapshot the child state *after* the    *)
(*                   shuffled table is produced, with 0 chunks to skip.    *)
(*                                                                         *)
(* The RNG permutation is modelled as a nondeterministic choice, which     *)
(* over-approximates every seed.  The buffer content is lost on resume     *)
(* (BufferShuffledExamplesIterable.load_state_dict, line 2058: documented). *)
(* Property NoReplay: nothing yielded before the checkpoint is yielded     *)
(* again after resuming (the docs only allow buffered examples to be lost).*)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS N, Bin, S, Mode

ASSUME N \in Nat /\ Bin \in Nat \ {0} /\ S \in Nat \ {0} /\ Mode \in {"arrow", "python", "fixed"}

Keep == S \div 2
Min(a, b) == IF a < b THEN a ELSE b
Range(f) == {f[i] : i \in DOMAIN f}
\* injective sequences of length k over set X (all orders the RNG permutation may produce)
Arr(X, k) == {f \in [1..k -> X] : \A i, j \in 1..k : i # j => f[i] # f[j]}
Pulled(s, b, k) == b \cup (s+1..Min(s + k*Bin, N))
StopPull(s, b, k) == Cardinality(Pulled(s, b, k)) >= S \/ s + k*Bin >= N
\* number of source tables BufferShuffled pulls before it yields (the for-loop of line 2111)
PullCount(s, b) == CHOOSE k \in 1..N : StopPull(s, b, k) /\ \A j \in 1..(k-1) : ~StopPull(s, b, j)

(* --algorithm ShuffleResume
variables
    phase = "run1",       \* run1: before the checkpoint; run2: after load_state_dict; done
    src = 0,              \* source rows pulled by BufferShuffled (its child's state_dict)
    buf = {},             \* rows held in the shuffle buffer (kept half)
    tbl = <<>>,           \* shuffled table yielded by BufferShuffled._iter_arrow, being re-chunked
    ci = 0,               \* chunks of tbl already consumed by the top Rebatched
    prev = 0,             \* Rebatched local `previous_state` (lines 574-576, 633-634)
    sdPrev = 0,           \* Rebatched _state_dict["previous_state"]
    sdChunks = 0,         \* Rebatched _state_dict["num_chunks_since_previous_state"]
    toSkip = 0,           \* num_chunks_to_skip after resume (line 571)
    srcEnd = FALSE,       \* BufferShuffled flushed its last buffer (lines 2132-2135)
    y1 = {},              \* yielded before the checkpoint
    y2 = {},              \* yielded after resuming
    dup = FALSE;          \* some row yielded twice after resuming (sanity)

define
    NoReplay == y1 \cap y2 = {}
    TypeOK == src \in 0..N /\ buf \subseteq 1..N /\ ci \in 0..N
end define;

\* BufferShuffledExamplesIterable._iter_arrow (lines 2100-2135), executed atomically between two yields of the
\* generator: pull source tables of Bin rows until current_len >= buffer_size (or the source ends), then yield a
\* permutation of current_len - keep rows and keep the rest; at the end of the source, flush the whole buffer.
macro Produce() begin
    if src < N then
        with k = PullCount(src, buf), nb = buf \cup (src+1..Min(src + k*Bin, N)) do
            src := Min(src + k*Bin, N);
            if Cardinality(nb) >= S then
                with t \in Arr(nb, Cardinality(nb) - Keep) do
                    tbl := t; buf := nb \ Range(t);
                end with;
            else
                with t \in Arr(nb, Cardinality(nb)) do     \* source ended before the buffer filled: flush
                    tbl := t; buf := {}; srcEnd := TRUE;
                end with;
            end if;
        end with;
    elsif buf # {} then
        with t \in Arr(buf, Cardinality(buf)) do            \* final flush (lines 2132-2135)
            tbl := t; buf := {}; srcEnd := TRUE;
        end with;
    else
        tbl := <<>>; srcEnd := TRUE;
    end if;
    ci := 0;
end macro;

macro Emit(r) begin
    if phase = "run1" then y1 := y1 \cup {r};
    else dup := dup \/ r \in y2; y2 := y2 \cup {r};
    end if;
end macro;

process It = "it"
begin
Start:
    sdPrev := src; prev := src;              \* lines 574-576
Loop:
    while phase # "done" do
        either
            \* the consumer asks for the next example
            if ci < Len(tbl) then
                if toSkip > 0 then
                    toSkip := toSkip - 1; ci := ci + 1;          \* lines 582-587: skip replayed chunks
                else
                    Emit(tbl[ci+1]);
                    ci := ci + 1;
                    if Mode = "python" then
                        \* BufferShuffled.__iter__: state_dict is the live child state (rows pulled so far)
                        sdPrev := src; sdChunks := 0;
                    elsif Mode = "arrow" then
                        sdPrev := prev; sdChunks := ci;      \* lines 600-615 (batch_size=1)
                    else
                        sdPrev := src; sdChunks := 0;        \* fix: remainder of tbl counts as lost buffer
                    end if;
                end if;
            elsif ~srcEnd then
                prev := src;                                 \* lines 633-634: child state after the table
                Produce();
            else
                phase := "done";
            end if;
        or
            \* checkpoint + restart: state_dict() then load_state_dict() into a fresh pipeline (run once)
            await phase = "run1";
            phase := "run2";
            src := sdPrev;                                   \* line 551: child.load_state_dict(previous_state)
            buf := {}; tbl := <<>>; ci := 0; srcEnd := FALSE; \* shuffle buffer lost, rng restarted
            toSkip := sdChunks;                               \* line 571
            prev := sdPrev;                                  \* line 575
        end either;
    end while;
end process;
end algorithm; *)
\* BEGIN TRANSLATION (chksum(pcal) = "97030724" /\ chksum(tla) = "ce18e75d")
VARIABLES pc, phase, src, buf, tbl, ci, prev, sdPrev, sdChunks, toSkip, 
          srcEnd, y1, y2, dup

(* define statement *)
NoReplay == y1 \cap y2 = {}
TypeOK == src \in 0..N /\ buf \subseteq 1..N /\ ci \in 0..N


vars == << pc, phase, src, buf, tbl, ci, prev, sdPrev, sdChunks, toSkip, 
           srcEnd, y1, y2, dup >>

ProcSet == {"it"}

Init == (* Global variables *)
        /\ phase = "run1"
        /\ src = 0
        /\ buf = {}
        /\ tbl = <<>>
        /\ ci = 0
        /\ prev = 0
        /\ sdPrev = 0
        /\ sdChunks = 0
        /\ toSkip = 0
        /\ srcEnd = FALSE
        /\ y1 = {}
        /\ y2 = {}
        /\ dup = FALSE
        /\ pc = [self \in ProcSet |-> "Start"]

Start == /\ pc["it"] = "Start"
         /\ sdPrev' = src
         /\ prev' = src
         /\ pc' = [pc EXCEPT !["it"] = "Loop"]
         /\ UNCHANGED << phase, src, buf, tbl, ci, sdChunks, toSkip, srcEnd, 
                         y1, y2, dup >>

Loop == /\ pc["it"] = "Loop"
        /\ IF phase # "done"
              THEN /\ \/ /\ IF ci < Len(tbl)
                               THEN /\ IF toSkip > 0
                                          THEN /\ toSkip' = toSkip - 1
                                               /\ ci' = ci + 1
                                               /\ UNCHANGED << sdPrev, 
                                                               sdChunks, y1, 
                                                               y2, dup >>
                                          ELSE /\ IF phase = "run1"
                                                     THEN /\ y1' = (y1 \cup {(tbl[ci+1])})
                                                          /\ UNCHANGED << y2, 
                                                                          dup >>
                                                     ELSE /\ dup' = (dup \/ (tbl[ci+1]) \in y2)
                                                          /\ y2' = (y2 \cup {(tbl[ci+1])})
                                                          /\ y1' = y1
                                               /\ ci' = ci + 1
                                               /\ IF Mode = "python"
                                                     THEN /\ sdPrev' = src
                                                          /\ sdChunks' = 0
                                                     ELSE /\ IF Mode = "arrow"
                                                                THEN /\ sdPrev' = prev
                                                                     /\ sdChunks' = ci'
                                                                ELSE /\ sdPrev' = src
                                                                     /\ sdChunks' = 0
                                               /\ UNCHANGED toSkip
                                    /\ UNCHANGED << phase, src, buf, tbl, prev, 
                                                    srcEnd >>
                               ELSE /\ IF ~srcEnd
                                          THEN /\ prev' = src
                                               /\ IF src < N
                                                     THEN /\ LET k == PullCount(src, buf) IN
                                                               LET nb == buf \cup (src+1..Min(src + k*Bin, N)) IN
                                                                 /\ src' = Min(src + k*Bin, N)
                                                                 /\ IF Cardinality(nb) >= S
                                                                       THEN /\ \E t \in Arr(nb, Cardinality(nb) - Keep):
                                                                                 /\ tbl' = t
                                                                                 /\ buf' = nb \ Range(t)
                                                                            /\ UNCHANGED srcEnd
                                                                       ELSE /\ \E t \in Arr(nb, Cardinality(nb)):
                                                                                 /\ tbl' = t
                                                                                 /\ buf' = {}
                                                                                 /\ srcEnd' = TRUE
                                                     ELSE /\ IF buf # {}
                                                                THEN /\ \E t \in Arr(buf, Cardinality(buf)):
                                                                          /\ tbl' = t
                                                                          /\ buf' = {}
                                                                          /\ srcEnd' = TRUE
                                                                ELSE /\ tbl' = <<>>
                                                                     /\ srcEnd' = TRUE
                                                                     /\ buf' = buf
                                                          /\ src' = src
                                               /\ ci' = 0
                                               /\ phase' = phase
                                          ELSE /\ phase' = "done"
                                               /\ UNCHANGED << src, buf, tbl, 
                                                               ci, prev, 
                                                               srcEnd >>
                                    /\ UNCHANGED << sdPrev, sdChunks, toSkip, 
                                                    y1, y2, dup >>
                      \/ /\ phase = "run1"
                         /\ phase' = "run2"
                         /\ src' = sdPrev
                         /\ buf' = {}
                         /\ tbl' = <<>>
                         /\ ci' = 0
                         /\ srcEnd' = FALSE
                         /\ toSkip' = sdChunks
                         /\ prev' = sdPrev
                         /\ UNCHANGED <<sdPrev, sdChunks, y1, y2, dup>>
                   /\ pc' = [pc EXCEPT !["it"] = "Loop"]
              ELSE /\ pc' = [pc EXCEPT !["it"] = "Done"]
                   /\ UNCHANGED << phase, src, buf, tbl, ci, prev, sdPrev, 
                                   sdChunks, toSkip, srcEnd, y1, y2, dup >>

It == Start \/ Loop

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == It
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION 
===============================================================================
