set_option linter.unusedVariables false
/-
Length budgets in assisted generation (src/transformers/generation/candidate_generator.py, generation/utils.py).

AssistedCandidateGenerator._calculate_new_tokens (candidate_generator.py:281-286):
    new_cur_len    = L
    max_new_tokens = min(num_assistant_tokens, main_max_length - L - 1)
    min_new_tokens = max(min(max_new_tokens, main_min_length - L), 0)
(Python ints: main_max_length - L - 1 may be negative only when L ≥ M, which the loop never reaches.)
-/

def calcNew (A M Mn L : Int) : Int × Int :=
  let mx := min A (M - L - 1)
  let mn := max (min mx (Mn - L)) 0
  (mn, mx)

/-- Candidates + the bonus token never exceed max_length. -/
theorem assisted_budget (A M Mn L c : Int) (hA : 0 ≤ A) (hL : L < M) (hc : 0 ≤ c)
    (hcand : c ≤ (calcNew A M Mn L).2) : L + c + 1 ≤ M := by
  unfold calcNew at hcand; simp only at hcand; omega

/-- min ≤ max and both non-negative, so the assistant's own generate() gets a consistent request. -/
theorem assisted_min_le_max (A M Mn L : Int) (hA : 0 ≤ A) (hL : L < M) :
    0 ≤ (calcNew A M Mn L).1 ∧ (calcNew A M Mn L).1 ≤ (calcNew A M Mn L).2 := by
  unfold calcNew; simp only; omega

/-- EOS suppression is consistent with the main model's MinLengthLogitsProcessor: the assistant suppresses EOS on
its first `mn` tokens, i.e. at every position p (absolute index of the drafted token) with p < main_min_length,
unless the draft budget is too small to reach it. -/
theorem assisted_min_length_consistent (A M Mn L p : Int) (hA : 0 ≤ A) (hL : L < M)
    (hp : L ≤ p) (hpm : p < Mn) (hdraft : p < L + (calcNew A M Mn L).2) :
    p < L + (calcNew A M Mn L).1 := by
  unfold calcNew at *; simp only at *; omega

/-
Whisper `_set_max_new_tokens_and_length` (models/whisper/generation_whisper.py:1935-1961):

    if max_new_tokens + dec_len > max_target_positions: raise ValueError(...)
    ...
    elif max_new_tokens is not None and max_new_tokens + dec_len > max_target_positions:
        max_new_tokens = max_target_positions - dec_len          # clamp
The clamp branch is unreachable: its guard is exactly the raise guard.
-/
def whisperSet (mnt : Option Nat) (decLen P : Nat) : Option (Option Nat) :=   -- none = raised
  let m := mnt.getD 0
  if m + decLen > P then none
  else match mnt with
    | some k => if k + decLen > P then some (some (P - decLen)) else some (some k)
    | none => some none

theorem whisper_clamp_branch_dead (k decLen P : Nat) :
    whisperSet (some k) decLen P = none ∨ whisperSet (some k) decLen P = some (some k) := by
  unfold whisperSet; simp only [Option.getD_some]
  by_cases h : k + decLen > P
  · left; simp [h]
  · right; simp [h]

/- With condition_on_prev_tokens the decoder prompt is up to 1 + (P/2 - 1) previous tokens + 3 init tokens,
so any user `max_new_tokens > P/2 - 3` (= 221 for Whisper's P = 448) eventually raises mid long-form. -/
#eval (List.range 449).filter (fun k => whisperSet (some k) (1 + (448 / 2 - 1) + 3) 448 == none) |>.head?
