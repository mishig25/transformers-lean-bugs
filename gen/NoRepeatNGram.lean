/-
NoRepeatNGramLogitsProcessor.__call__ (src/transformers/generation/logits_process.py:1150-1170), vectorised version:

  cur_len = L
  if L < n: return scores
  prefix  = ids[L+1-n :]                       (length n-1)
  windows = ids.unfold(1, n, 1)                (windows i = ids[i : i+n],  0 ≤ i ≤ L-n)
  matches = windows[..., :-1] == prefix        (all over last dim)
  banned  = { windows[i][-1] | matches[i] }

Spec (Fairseq semantics, the old `_get_ngrams`/`_get_generated_ngrams` implementation):
  token t is banned  ⇔  the n-gram  ids[L-n+1 : L] ++ [t]  already occurs somewhere in ids.
-/

def slice (xs : List Nat) (a b : Nat) : List Nat := (xs.drop a).take (b - a)

/-- vectorised implementation -/
def bannedVec (ids : List Nat) (n : Nat) : List Nat :=
  let L := ids.length
  if L < n then []
  else
    let pfx := ids.drop (L + 1 - n)
    (List.range (L - n + 1)).filterMap fun i =>
      let w := slice ids i (i + n)
      if w.take (n - 1) == pfx then w.getLast? else none

/-- reference: old dict-based implementation (`_get_ngrams` + `_get_generated_ngrams`) -/
def bannedRef (ids : List Nat) (n : Nat) : List Nat :=
  let L := ids.length
  -- ngrams = zip(*[ids[i:] for i in range(n)])  -> all windows of length n
  let ngrams := (List.range (L + 1 - n)).filterMap fun i =>
    let w := slice ids i (i + n)
    if w.length == n then some w else none
  let key := slice ids (L + 1 - n) L
  ngrams.filterMap fun w => if w.take (n - 1) == key then w.getLast? else none

/-- semantic spec: t banned iff suffix ++ [t] occurs as a window -/
def bannedSpec (ids : List Nat) (n : Nat) (t : Nat) : Bool :=
  let L := ids.length
  if L < n then false
  else
    let target := ids.drop (L + 1 - n) ++ [t]
    (List.range (L + 1)).any fun i => i + n ≤ L && slice ids i (i + n) == target

def allLists : Nat → Nat → List (List Nat)
  | 0, _ => [[]]
  | k + 1, v => (allLists k v) ++ ((allLists k v).filter (·.length == k)).flatMap fun l => (List.range v).map (l ++ [·])

def check (maxLen vocab : Nat) : List (List Nat × Nat × Nat) := Id.run do
  let mut bad := []
  for ids in (allLists maxLen vocab) do
    for n in [1, 2, 3, 4] do
      for t in List.range vocab do
        let a := (bannedVec ids n).contains t
        let b := (bannedRef ids n).contains t
        let c := bannedSpec ids n t
        if a != b || a != c then bad := bad ++ [(ids, n, t)]
  return bad

-- all lists of length ≤ 7 over a 3-token vocab, n ∈ {1..4}: vectorised == reference == spec
#eval (allLists 7 3).length
#eval check 7 3
