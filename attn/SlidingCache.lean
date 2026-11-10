set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-- reduce projections, turn max/min into ifs, split every if, finish with omega -/
macro "crush" : tactic => `(tactic| (
  (try dsimp only at *) <;>
  (try simp only [Int.max_def, Int.min_def, ite_not] at *) <;>
  (repeat' split) <;> (try dsimp only at *) <;>
  (try simp only [eq_self_iff_true, true_or, or_true, true_and, and_true, Bool.true_eq_false,
    Bool.false_eq_true, false_or, or_false, not_false_eq_true, Int.add_zero] at *) <;> omega))
/-
  DynamicSlidingWindowLayer / StaticSlidingWindowLayer (src/transformers/cache_utils.py) vs. the mask
  sizes they advertise to masking_utils (`get_mask_sizes`, `get_query_offset` = cumulative length).

  Stored keys are always a contiguous run of absolute positions, so a key buffer is modelled as the
  half-open interval [s, e). Python slicing is modelled exactly on that interval.
-/

/-- Python's normalisation of a slice bound `i` for a sequence of length `n`. -/
def pyIdx (n i : Int) : Int := if i < 0 then max 0 (n + i) else min i n

/-- `x[a:b]` on the interval [s, e) (bounds `none` = omitted). -/
def pySlice (s e : Int) (a b : Option Int) : Int × Int :=
  let n := e - s
  let a' := match a with | none => 0 | some i => pyIdx n i
  let b' := match b with | none => n | some i => pyIdx n i
  (s + a', s + max a' b')

/-! ## DynamicSlidingWindowLayer -/

structure DS where
  s : Int        -- first cached absolute position
  e : Int        -- one past the last cached position
  c : Int        -- cumulative_length
  record : Bool  -- record_past

/-- `get_mask_sizes(query_length)` → (kv_length, kv_offset); uses the pre-update cumulative length. -/
def dsMaskSizes (W : Int) (st : DS) (q : Int) : Int × Int :=
  let off := max (st.c - W + 1) 0
  let len := if st.c ≥ W then W - 1 + q else st.c + q
  (len, off)

/-- `update(key_states)` with `q` new tokens at positions [c, c+q). Returns (new state, returned
    interval). The `cat` is only an interval if the buffer ends at `c` (checked by the invariant). -/
def dsUpdate (W : Int) (st : DS) (q : Int) : DS × (Int × Int) :=
  let full := (st.s, st.c + q)
  if !st.record then
    let k := pySlice full.1 full.2 (some (-W + 1)) none
    ({ s := k.1, e := k.2, c := st.c + q, record := false }, full)
  else
    let ret := pySlice full.1 full.2 (some (-(W - 1 + q))) none
    ({ s := full.1, e := full.2, c := st.c + q, record := true }, ret)

/-- `crop(-r)` (r ≥ 0), both branches of DynamicSlidingWindowLayer.crop (record_past = True). -/
def dsCrop (W : Int) (st : DS) (r : Int) : DS :=
  if st.c ≥ W then
    if r = 0 then let k := pySlice st.s st.e (some (-W + 1)) none; { st with s := k.1, e := k.2 }
    else let k := pySlice st.s st.e (some (-W + 1 - r)) (some (-r)); { st with s := k.1, e := k.2, c := st.c - r }
  else
    if r = 0 then st
    else let k := pySlice st.s st.e none (some (-r)); { st with s := k.1, e := k.2, c := k.2 - k.1 }

/-- Invariant needed by the mask: buffer ends at c; holds everything while c < W; otherwise holds at
    least the last W-1 tokens (exactly those when not recording). -/
def DSInv (W : Int) (st : DS) : Prop :=
  st.e = st.c ∧ 0 ≤ st.s ∧ st.s ≤ st.c ∧
  (st.c ≥ W ∨ st.s = 0) ∧ (st.c < W ∨ st.s ≤ st.c - W + 1) ∧
  (st.record = true ∨ st.c < W - 1 ∨ st.s = st.c - W + 1) ∧
  (st.record = true ∨ st.c ≥ W - 1 ∨ st.s = 0)

theorem ds_empty (W : Int) (hW : 2 ≤ W) (rec : Bool) : DSInv W { s := 0, e := 0, c := 0, record := rec } := by
  unfold DSInv; cases rec <;> simp <;> omega

/-- Contract: the returned keys are exactly positions [kv_offset, kv_offset + kv_length). -/
theorem ds_update_matches_mask (W : Int) (hW : 2 ≤ W) (st : DS) (q : Int) (hq : 1 ≤ q)
    (h : DSInv W st) :
    (dsUpdate W st q).2 = ((dsMaskSizes W st q).2, (dsMaskSizes W st q).2 + (dsMaskSizes W st q).1) := by
  unfold DSInv at h; unfold dsUpdate dsMaskSizes pySlice pyIdx
  cases hr : st.record <;> simp only [hr, Bool.not_false, Bool.not_true, ite_true, ite_false, Prod.mk.injEq,
    Bool.false_eq_true, Bool.true_eq_false, false_or, true_or, and_true] at * <;> crush

/-- Window coverage: every key a query at p ∈ [c, c+q) must see (p-W < k ≤ p, k ≥ 0) is returned. -/
theorem ds_update_covers (W : Int) (hW : 2 ≤ W) (st : DS) (q p k : Int) (hq : 1 ≤ q)
    (h : DSInv W st) (hp : st.c ≤ p ∧ p < st.c + q) (hk : 0 ≤ k ∧ p - W < k ∧ k ≤ p) :
    (dsUpdate W st q).2.1 ≤ k ∧ k < (dsUpdate W st q).2.2 := by
  have := ds_update_matches_mask W hW st q hq h
  rw [this]; unfold dsMaskSizes; unfold DSInv at h; crush

theorem ds_update_inv (W : Int) (hW : 2 ≤ W) (st : DS) (q : Int) (hq : 1 ≤ q) (h : DSInv W st) :
    DSInv W (dsUpdate W st q).1 := by
  unfold DSInv at *; unfold dsUpdate pySlice pyIdx
  cases hr : st.record <;> simp only [hr, Bool.not_false, Bool.not_true, ite_true, ite_false] at * <;>
    simp only [Bool.true_eq_false, Bool.false_eq_true, false_implies, true_implies, and_true, true_and, forall_const] at * <;> crush

/-- Crop contract. `crop(-r)` (record_past = True) preserves the invariant provided the buffer still holds
    the r tokens being removed plus the W-1 tokens before them, i.e. at least r tokens were appended since
    the last truncation (last crop or non-recording update). In `generate` this holds: the target model is
    cropped by `candidate_length - n_matches` < tokens fed in that forward, the assistant by
    `current - target_cache_size` < tokens it generated since its last crop, and DeferredStopCheck crops 1
    right after a 1-token forward. -/
theorem ds_crop_inv (W : Int) (hW : 2 ≤ W) (st : DS) (r : Int) (hr0 : 0 ≤ r) (hrec : st.record = true)
    (h : DSInv W st) (hr : r ≤ st.c - st.s) (hfresh : st.c < W ∨ r ≤ st.c - st.s - (W - 1)) :
    DSInv W (dsCrop W st r) := by
  unfold DSInv at *; unfold dsCrop pySlice pyIdx
  simp only [hrec, Bool.true_eq_false, false_implies, and_true] at * <;> crush

/-- Without the precondition the invariant really breaks: the cache claims (via
    get_mask_sizes, kv_offset = 0) to hold positions from 0 but has lost [0, s). -/
def badState : DS := { s := 3, e := 10, c := 10, record := true }   -- W = 8: holds [3,10)
#eval let W := 8; let st := dsCrop W badState 5; (st.s, st.e, st.c, dsMaskSizes W st 1)
-- (3, 5, 5, (6, 0)): mask says 6 slots from position 0, buffer+1 new token has 3 slots from 3.

/-! ## StaticSlidingWindowLayer (buffer of M = min(W, max_cache_len) slots). -/

structure SS where
  lo : Int      -- buffer slot i holds absolute position lo + i for i < valid
  valid : Int   -- number of meaningful slots (rest are zeros)
  c : Int       -- cumulative_length_int

def ssMaskSizes (M : Int) (st : SS) (q : Int) : Int × Int :=
  let off := max (st.c - M + 1) 0
  let len := if st.c ≥ M then M + q - 1 else if st.c + q > M then st.c + q else M
  (len, off)

/-- Returns (new state, returned first position, number of returned slots, number of meaningful slots). -/
def ssUpdate (M : Int) (st : SS) (q : Int) : SS × (Int × Int × Int) :=
  if st.c ≥ M then
    if q = 1 then ({ lo := st.c - M + 1, valid := M, c := st.c + 1 }, (st.c - M + 1, M, M))
    else -- cat(keys[1:], new) : positions [c-M+1, c+q)
      ({ lo := st.c + q - M, valid := M, c := st.c + q }, (st.c - M + 1, M - 1 + q, M - 1 + q))
  else if st.c + q > M then -- cat(keys[:c], new) : positions [0, c+q)
    ({ lo := st.c + q - M, valid := M, c := st.c + q }, (0, st.c + q, st.c + q))
  else -- index_copy_ into slots [c, c+q); return the whole M-slot buffer
    ({ lo := 0, valid := st.c + q, c := st.c + q }, (0, M, st.c + q))

def SSInv (M : Int) (st : SS) : Prop :=
  0 ≤ st.c ∧ (st.c < M ∨ (st.lo = st.c - M ∧ st.valid = M)) ∧ (st.c ≥ M ∨ (st.lo = 0 ∧ st.valid = st.c))

theorem ss_update_matches_mask (M : Int) (hM : 1 ≤ M) (st : SS) (q : Int) (hq : 1 ≤ q) (h : SSInv M st) :
    let r := (ssUpdate M st q).2
    r.1 = (ssMaskSizes M st q).2 ∧ r.2.1 = (ssMaskSizes M st q).1 := by
  unfold SSInv at h; unfold ssUpdate ssMaskSizes; crush

/-- Slots past the meaningful ones (zeros) start at a position ≥ c+q, i.e. after every query position:
    the causal mask hides them (returned slot j ↦ position first + j, increasing in j). -/
theorem ss_garbage_is_future (M : Int) (st : SS) (q : Int) (hq : 1 ≤ q) (h : SSInv M st) :
    let r := (ssUpdate M st q).2
    r.1 + r.2.2 ≥ st.c + q := by
  unfold SSInv at h; unfold ssUpdate; crush

theorem ss_update_inv (M : Int) (hM : 1 ≤ M) (st : SS) (q : Int) (hq : 1 ≤ q) (h : SSInv M st) :
    SSInv M (ssUpdate M st q).1 := by
  unfold SSInv at *; unfold ssUpdate; crush

/-- Coverage holds iff the buffer is at least the configured window (M = W, i.e. max_cache_len ≥ W). -/
theorem ss_covers (W M : Int) (hMW : M = W) (hM : 1 ≤ M) (st : SS) (q p k : Int) (hq : 1 ≤ q)
    (h : SSInv M st) (hp : st.c ≤ p ∧ p < st.c + q) (hk : 0 ≤ k ∧ p - W < k ∧ k ≤ p) :
    let r := (ssUpdate M st q).2
    r.1 ≤ k ∧ k < r.1 + r.2.2 := by
  unfold SSInv at h; unfold ssUpdate; crush
