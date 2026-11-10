set_option linter.unusedVariables false
/-
  Soundness of `_ignore_causal_mask_sdpa` (masking_utils.py) + the `is_causal` handling of
  `sdpa_attention_forward` (integrations/sdpa_attention.py).

  When the mask is skipped (None), SDPA computes:
    is_causal := q > 1;  if is_causal and kv_actual > q: keys sliced to the first q;
    is_causal = True  -> slot j visible from query i iff j ≤ i   (upper-left alignment)
    is_causal = False -> every returned slot visible.
  The reference ("true") mask is the one sdpa_mask would materialise:
    pad(kv_off + j) ∧ kv_off + j ≤ q_off + i ∧ (local → q_off + i - W < kv_off + j).
-/

/-- What the cache hands to attention (proved per layer type in SlidingCache.lean):
    `exact`  : returned slots are exactly [kv_off, kv_off + kv_len) and end at the last query
               (DynamicLayer, DynamicSlidingWindowLayer, StaticSlidingWindowLayer once full / becoming full);
    `padded` : static buffer, kv_off = 0, slots ≥ q_off + q are zero "future" garbage
               (StaticLayer, StaticSlidingWindowLayer not yet full). -/
inductive Layout | exact | padded

structure Call where
  q : Int
  kvLen : Int
  qOff : Int
  kvOff : Int
  local? : Option Int          -- sliding window / chunk size passed as `local_size`
  compileable : Bool           -- `past_key_values.is_compileable`
  layout : Layout

def Call.wf (c : Call) : Prop :=
  1 ≤ c.q ∧ 0 ≤ c.qOff ∧ 0 ≤ c.kvOff ∧
  match c.layout with
  | .exact  => c.kvOff + c.kvLen = c.qOff + c.q ∧ c.kvOff ≤ c.qOff
  | .padded => c.kvOff = 0 ∧ c.qOff + c.q ≤ c.kvLen ∧ c.compileable = true

/-- `_ignore_causal_mask_sdpa` (non-tracing), with the decode guard of create_*_mask folded in:
    `allow_is_causal_skip = not (is_compileable and q_length == 1)`.
    `allValid`: no padding among returned slots holding real tokens; `prefixPad`: the
    `padding_mask[:, :q]` all True / `padding_mask[:, q:]` all False condition (with the
    no-padding-mask case included). -/
def skip (c : Call) (allValid prefixPad : Prop) : Prop :=
  ¬ (c.compileable = true ∧ c.q = 1) ∧
  (∀ W, c.local? = some W → c.kvLen < W) ∧
  (((c.q = 1 ∨ c.kvLen = c.q) ∧ allValid) ∨ (c.qOff = 0 ∧ prefixPad))

/-- Effective SDPA visibility of slot j from query i when the mask is None. -/
def sdpaVisible (c : Call) (i j : Int) : Prop :=
  if c.q > 1 then j ≤ i ∧ j < c.q else j < c.kvLen

/-- Reference visibility (no padding inside the visible region, as guaranteed by the skip conditions). -/
def trueVisible (c : Call) (i j : Int) : Prop :=
  c.kvOff + j ≤ c.qOff + i ∧ (∀ W, c.local? = some W → c.qOff + i - W < c.kvOff + j)

theorem sdpa_skip_sound (c : Call) (allValid prefixPad : Prop) (hwf : c.wf)
    (hs : skip c allValid prefixPad) (i j : Int) (hi : 0 ≤ i ∧ i < c.q) (hj : 0 ≤ j ∧ j < c.kvLen) :
    sdpaVisible c i j ↔ trueVisible c i j := by
  obtain ⟨g, hloc, hcase⟩ := hs
  obtain ⟨hq, hqo, hko, hlay⟩ := hwf
  unfold sdpaVisible trueVisible
  cases hl : c.layout <;> simp only [hl] at hlay
  · -- exact layout
    obtain ⟨h1, h2⟩ := hlay
    cases h : c.local? with
    | none =>
      simp only [h, reduceCtorEq, false_implies, implies_true, and_true]
      split <;> rcases hcase with ⟨hc | hc, _⟩ | ⟨hc, _⟩ <;> omega
    | some W =>
      have hW := hloc W h
      simp only [h, Option.some.injEq, forall_eq']
      split <;> rcases hcase with ⟨hc | hc, _⟩ | ⟨hc, _⟩ <;> omega
  · -- padded (static) layout: compileable, so q = 1 never skips
    obtain ⟨h1, h2, h3⟩ := hlay
    simp only [h3, true_and] at g
    cases h : c.local? with
    | none =>
      simp only [h, reduceCtorEq, false_implies, implies_true, and_true]
      split <;> rcases hcase with ⟨hc | hc, _⟩ | ⟨hc, _⟩ <;> omega
    | some W =>
      have hW := hloc W h
      simp only [h, Option.some.injEq, forall_eq']
      split <;> rcases hcase with ⟨hc | hc, _⟩ | ⟨hc, _⟩ <;> omega

/-- The decode guard is load-bearing: a non-compileable cache holding a static buffer would
    see a 1-token step attend to every (zero-filled) slot. -/
theorem guard_needed :
    let c : Call := { q := 1, kvLen := 8, qOff := 0, kvOff := 0, local? := none,
                      compileable := false, layout := .padded }
    ¬ (sdpaVisible c 0 5 ↔ trueVisible c 0 5) := by
  simp [sdpaVisible, trueVisible]
