/-
Model of `PromptLookupCandidateGenerator.get_candidates` (src/transformers/generation/candidate_generator.py:1057-1150)
and of the acceptance/crop step of `_assisted_decoding` (src/transformers/generation/utils.py:4139-4151).

Variables (all Nat):
  L  = input_length (current sequence length, = input_ids.shape[1])
  M  = max_length  (generation_config.max_length, passed as `max_length=`)
  K  = num_output_tokens (= prompt_lookup_num_tokens)
  idx, n = match index of the n-gram window, ngram size

Python:
  if M == L + 1: return input_ids                       # no candidates
  start_idx = idx + n
  end_idx   = min(start_idx + K, L, M)
  chosen    = input_ids[start_idx:end_idx]               # (end - start) candidates if start < end
-/

def numCandidates (L M K idx n : Nat) : Nat :=
  if M = L + 1 then 0
  else
    let s := idx + n
    let e := min (min (s + K) L) M
    if s < e then e - s else 0

/-- Length of the sequence fed to the target model's verification forward. -/
def verifyLen (L M K idx n : Nat) : Nat := L + numCandidates L M K idx n

/-- Intended spec (comment at line 1070: "Don't generate more than `max_length - 1` candidates since the target model
generates one extra token"): the verification forward never needs positions beyond `max_length - 1`, i.e.
`L + #candidates ≤ M - 1` whenever generation is still running (`L < M`) and the window is a real match
(`idx + n ≤ L`).  This is FALSE; see the counterexample search below. -/
def specHolds (L M K idx n : Nat) : Bool := verifyLen L M K idx n + 1 ≤ M

/-- Exhaustive counterexample search on a small grid of *reachable* states:
  1 ≤ n ≤ 2 (default max_matching_ngram_size = 2), idx + n ≤ L - 1 (a window that is not the suffix itself),
  L < M (MaxLengthCriteria has not fired). -/
def counterexamples : List (Nat × Nat × Nat × Nat × Nat) := Id.run do
  let mut out := []
  for L in List.range 13 do
    for M in List.range 14 do
      for K in [1, 3, 10] do
        for n in [1, 2] do
          for idx in List.range 12 do
            if L < M && idx + n + 1 ≤ L && !(specHolds L M K idx n) then
              out := out ++ [(L, M, K, idx, n)]
  return out

#eval counterexamples.length
#eval counterexamples.take 8
-- concrete instance matching the reproducer: prompt of 8 tokens, max_new_tokens = 2 (M = 10), K = 10, match at 0
#eval (numCandidates 8 10 10 0 2, verifyLen 8 10 10 0 2)   -- (6, 14): forward over 14 positions with max_length 10

/-- The cap actually used is only `end ≤ L`, which bounds candidates by L - (idx+n), not by M - L - 1. -/
theorem candidates_le_prompt_tail (L M K idx n : Nat) :
    numCandidates L M K idx n ≤ L - (idx + n) := by
  unfold numCandidates
  split
  · omega
  · simp only
    split <;> omega

/-- The `min(.., M)` term in `end_idx` never binds while generation is running (L < M): it is dead code. -/
theorem max_length_term_is_dead (L M K s : Nat) (h : L < M) :
    min (min (s + K) L) M = min (s + K) L := by
  omega

/-- Proposed fix: cap by the remaining budget, `end = min(s + K, L, s + (M - L - 1))`. -/
def numCandidatesFixed (L M K idx n : Nat) : Nat :=
  let s := idx + n
  let e := min (min (s + K) L) (s + (M - L - 1))
  if s < e then e - s else 0

theorem fixed_spec (L M K idx n : Nat) (h : L < M) :
    L + numCandidatesFixed L M K idx n + 1 ≤ M := by
  unfold numCandidatesFixed
  simp only
  split <;> omega

/-- Acceptance step (utils.py:4139-4151): valid_tokens has n_matches+1 ≤ cand+1 tokens, then is cropped to
`tokens_budget = M - L`.  So the *returned* sequence never exceeds max_length (this part is correct). -/
def acceptedLen (L M cand nMatches : Nat) : Nat :=
  let valid := min nMatches cand + 1
  L + min valid (M - L)

theorem accepted_le_max (L M cand nMatches : Nat) (h : L < M) : acceptedLen L M cand nMatches ≤ M := by
  unfold acceptedLen
  simp only
  omega
