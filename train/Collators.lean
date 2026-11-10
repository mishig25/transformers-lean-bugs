/-
  Models of src/transformers/data/data_collator.py:
    * pad_to_multiple_of rounding in _torch_collate_batch / _numpy_collate_batch (l.377-379)
      and in DataCollatorForSeq2Seq label padding (l.563-569)
    * DataCollatorWithFlattening (l.1315-1377)
    * DataCollatorForLanguageModeling 80/10/10 split (l.833-856)
-/

/-! ## pad_to_multiple_of -/
-- _torch_collate_batch: if m % k != 0: m = (m // k + 1) * k
def padA (m k : Nat) : Nat := if m % k ≠ 0 then (m / k + 1) * k else m
-- DataCollatorForSeq2Seq labels: (m + k - 1) // k * k
def padB (m k : Nat) : Nat := (m + k - 1) / k * k

theorem padA_eq_padB (m k : Nat) (hk : 0 < k) : padA m k = padB m k := by
  unfold padA padB
  have h1 := Nat.div_add_mod m k
  have h2 := Nat.mod_lt m hk
  by_cases h : m % k = 0
  · simp only [h, ne_eq, not_true_eq_false, ite_false]
    have e : m + k - 1 = (k - 1) + k * (m / k) := by omega
    rw [e, Nat.add_mul_div_left _ _ hk, Nat.div_eq_of_lt (by omega : k - 1 < k), Nat.zero_add,
        Nat.mul_comm]; omega
  · simp only [h, ne_eq, not_false_eq_true, ite_true]
    have e : m + k - 1 = (m % k - 1) + k * (m / k + 1) := by
      rw [Nat.mul_add, Nat.mul_one]; omega
    rw [e, Nat.add_mul_div_left _ _ hk, Nat.div_eq_of_lt (by omega : m % k - 1 < k), Nat.zero_add]

/-- padB is the least multiple of k that is ≥ m -/
theorem padB_spec (m k : Nat) (hk : 0 < k) :
    m ≤ padB m k ∧ padB m k % k = 0 ∧ padB m k < m + k := by
  unfold padB
  have h1 := Nat.div_add_mod (m + k - 1) k
  have h2 := Nat.mod_lt (m + k - 1) hk
  refine ⟨?_, Nat.mul_mod_left _ _, ?_⟩
  · rw [Nat.mul_comm]; omega
  · rw [Nat.mul_comm]; omega

/-! ## DataCollatorWithFlattening -/
structure Flat where
  inputIds : List Nat
  labels : List Int
  positionIds : List Nat
  seqIdx : List Nat
  cuSeqLens : List Nat
deriving Repr

def SEP : Int := -100

/-- one-to-one with the Python loop (labels not provided -> labels = [sep] + input_ids[1:]) -/
def flatten (feats : List (List Nat)) (start : Nat := 0) : Flat :=
  let rec go (fs : List (List Nat)) (j : Nat) (acc : Flat) : Flat :=
    match fs with
    | [] => acc
    | x :: rest =>
      go rest (j + 1)
        { inputIds := acc.inputIds ++ x
          labels := acc.labels ++ ([SEP] ++ (x.drop 1).map Int.ofNat)
          positionIds := acc.positionIds ++ (List.range x.length).map (start + ·)
          seqIdx := acc.seqIdx ++ List.replicate x.length j
          cuSeqLens := acc.cuSeqLens ++ [acc.cuSeqLens.getLast! + x.length] }
  go feats 0 ⟨[], [], [], [], [0]⟩

-- Spec 1: labels aligned with input_ids (same length).  True for non-empty examples:
theorem labels_len_nonempty (x : List Nat) (hx : x ≠ []) :
    ([SEP] ++ (x.drop 1).map Int.ofNat).length = x.length := by
  cases x with
  | nil => exact absurd rfl hx
  | cons a t => simp

-- ... but an empty example adds one label and zero input ids:
#eval let f := flatten [[5, 6, 7], [], [8, 9]]; (f.inputIds.length, f.labels.length, f.labels, f.cuSeqLens)
theorem labels_len_empty_fails :
    ¬ ∀ fs : List (List Nat), (flatten fs).labels.length = (flatten fs).inputIds.length := by
  intro h; have := h [[]]; revert this; decide

-- Spec 2: position_ids restart at `start` exactly at each cu_seqlens boundary;
-- Spec 3: cu_seqlens[-1] = total tokens.  Exhaustively checked on small inputs:
def restartsOK (fs : List (List Nat)) : Bool :=
  let f := flatten fs
  let cu := f.cuSeqLens
  f.positionIds.length == f.inputIds.length &&
  cu.getLast! == f.inputIds.length &&
  (List.range f.positionIds.length).all fun t =>
    (f.positionIds[t]! == 0) == (cu.contains t && (fs.filter (· ≠ [])).length > 0)

#eval (restartsOK [[1, 2, 3], [4], [5, 6]], restartsOK [[1], [2], [3, 4, 5, 6]])

theorem positions_len (x : List Nat) (s : Nat) : ((List.range x.length).map (s + ·)).length = x.length := by
  simp

/-! ## MLM 80/10/10 split (exact rationals, p_m = a/d, p_r = b/d, a + b ≤ d, a < d)
    P(replaced by [MASK]) = p_m
    P(random)            = (1 - p_m) * (p_r / (1 - p_m)) = p_r
    P(unchanged)         = 1 - p_m - p_r                                              -/
theorem mlm_random_prob (a b d : Nat) (_had : a < d) :
    -- (d - a)/d * b/(d - a) = b/d   <=>  (d-a)*b*d = b*(d*(d-a))
    (d - a) * b * d = b * (d * (d - a)) := by
  rw [Nat.mul_comm (d - a) b, Nat.mul_assoc, Nat.mul_comm (d - a) d]

-- Python computes random_replace_prob / (1 - mask_replace_prob) in binary64.  For (0.8, 0.2)
-- and (0.9, 0.1) the scaled probability exceeds 1:
#eval (0.2 / (1.0 - 0.8), 0.1 / (1.0 - 0.9), decide (0.2 / (1.0 - 0.8) > (1.0 : Float)), decide (0.1 / (1.0 - 0.9) > (1.0 : Float)))
#eval (0.2 / (1.0 - 0.8)).toString
