/-
WhisperTimeStampLogitsProcessor.__call__ (src/transformers/generation/logits_process.py:2041-2096), mask part
(the final "sum of timestamp probs > max text prob" rule only removes text tokens, so it can only shrink the
allowed set and is not needed for the safety invariants below).

Token layout of the model (small instance of Whisper's):
  text tokens      0 .. T-1
  eos              T          (Whisper: 50257, which is < no_timestamps; text tokens are those < eos)
  other specials   T+1 .. NT-1 (lang/task etc.; allowed unless masked — kept to check `[: eos_token_id]` masking)
  no_timestamps    NT
  timestamps       NT+1 .. NT+K     (timestamp_begin = NT+1)
-/

structure Lay where
  T : Nat     -- eos id
  NT : Nat    -- no_timestamps id (> T)
  K : Nat     -- number of timestamp tokens
  maxInit : Option Nat

def Lay.tb (l : Lay) := l.NT + 1
def Lay.V (l : Lay) := l.NT + 1 + l.K

/-- allowed[v] after processing generated sequence `seq` (= input_ids[begin_index:]) -/
def allowed (l : Lay) (seq : List Nat) (v : Nat) : Bool := Id.run do
  let tb := l.tb
  let mut ok := v != l.NT
  let n := seq.length
  let lastTs := n ≥ 1 && seq.getLast! ≥ tb
  let penTs := n < 2 || seq[n - 2]! ≥ tb
  if lastTs && penTs && v ≥ tb then ok := false
  if lastTs && !penTs && v < l.T then ok := false
  let tss := seq.filter (· ≥ tb)
  let tsLast := match tss.getLast? with
    | none => tb
    | some t => if lastTs && !penTs then t else t + 1
  if v ≥ tb && v < tsLast then ok := false
  if n == 0 then
    if v < tb then ok := false
    match l.maxInit with
    | some m => if v > tb + m then ok := false
    | none => pure ()
  return ok

/-- all sequences reachable by decoding (stopping at eos) up to length `d` -/
partial def reach (l : Lay) (d : Nat) (seq : List Nat) : List (List Nat) :=
  if d == 0 || (seq.getLast? == some l.T) then [seq]
  else
    let nexts := (List.range l.V).filter (allowed l seq)
    seq :: nexts.flatMap (fun v => reach l (d - 1) (seq ++ [v]))

/-- Invariants (OpenAI ApplyTimestampRules semantics):
 I1  timestamps never decrease
 I2  the first token is a timestamp ≤ tb + maxInit
 I3  timestamps come in pairs: a lone timestamp after text is followed only by a timestamp or eos
 I4  each segment <ts_a> text+ <ts_b> has b > a (non-zero length; prevents the seek loop from stalling)
 I5  no_timestamps never appears
 I6  never three timestamps in a row -/
def inv (l : Lay) (s : List Nat) : Bool := Id.run do
  let tb := l.tb
  let isTs := fun (x : Nat) => decide (x ≥ tb)
  let tss := s.filter isTs
  let i1 := (tss.zip tss.tail).all fun (a, b) => a ≤ b
  let i2 := match s.head? with
    | none => true
    | some x => isTs x && (match l.maxInit with | some m => x ≤ tb + m | none => true)
  let mut i3 := true
  let mut i4 := true
  let mut i6 := true
  let mut segStart : Option Nat := none
  for i in List.range s.length do
    let x := s[i]!
    if i ≥ 2 && isTs s[i-1]! && !isTs s[i-2]! && x < l.T then i3 := false  -- (specials in (eos, no_timestamps) are not masked here, same as OpenAI `logits[k, : tokenizer.eot] = -inf`)
    if i ≥ 2 && isTs x && isTs s[i-1]! && isTs s[i-2]! then i6 := false
    if isTs x then
      if i ≥ 1 && !isTs s[i-1]! then
        -- x closes a segment
        match segStart with
        | some a => if x ≤ a then i4 := false
        | none => pure ()
        segStart := none
      else segStart := some x
  let i5 := !(s.contains l.NT)
  return i1 && i2 && i3 && i4 && i5 && i6

def checkLay (l : Lay) (d : Nat) : Nat × List (List Nat) :=
  let all := reach l d []
  (all.length, (all.filter (fun s => !inv l s)).take 5)

-- 2 text tokens, eos=2, one other special (3), no_timestamps=4, 4 timestamps (5..8)
#eval checkLay ⟨2, 4, 4, some 1⟩ 7
#eval checkLay ⟨2, 4, 4, none⟩ 7
#eval checkLay ⟨1, 3, 5, some 0⟩ 8
