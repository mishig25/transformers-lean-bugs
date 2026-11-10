set_option linter.unusedVariables false
/-
  Bidirectional sliding-window attention: cross-backend + reference consistency.

  Each predicate says "query at position q may attend to key at position k" for a
  sequence of length L (positions 0..L-1), as implemented by one code path.
-/

/-- Original T5Gemma encoder mask (transformers <= 5.2.0, `sliding_window_bidirectional_mask_function`
    in models/t5gemma/modeling_t5gemma.py): `(q_idx - W < kv_idx) & (kv_idx < q_idx + W)`. -/
def t5gemmaOrig (W q k : Int) : Prop := q - W < k ∧ k < q + W

/-- main: masking_utils.sliding_window_bidirectional_overlay: `abs(q_idx - kv_idx) <= W`.
    Used by `create_bidirectional_sliding_window_mask`, i.e. the eager / sdpa / flex mask. -/
def hfOverlay (W q k : Int) : Prop := q - k ≤ W ∧ k - q ≤ W

/-- main: modeling_flash_attention_utils._process_flash_attention_kwargs, non-causal:
    `window_size = (sw - 1, sw - 1)` applied only when `key_length > sw`
    (flash-attn window bounds are inclusive). -/
def faBidir (sw L q k : Int) : Prop :=
  (L > sw → q - (sw - 1) ≤ k ∧ k ≤ q + (sw - 1))

/-- Gemma3 with `use_bidirectional_attention` (EmbeddingGemma): sliding-causal OR `abs(q-k) < W`. -/
def gemma3Bidir (W q k : Int) : Prop :=
  (k ≤ q ∧ q - W < k) ∨ (q - k < W ∧ k - q < W)

def inRange (L q k : Int) : Prop := 0 ≤ q ∧ q < L ∧ 0 ≤ k ∧ k < L

instance (W q k : Int) : Decidable (t5gemmaOrig W q k) := by unfold t5gemmaOrig; infer_instance
instance (W q k : Int) : Decidable (hfOverlay W q k) := by unfold hfOverlay; infer_instance
instance (sw L q k : Int) : Decidable (faBidir sw L q k) := by unfold faBidir; infer_instance

/-! ### T5Gemma (main): attention module passes `sliding_window = config.sliding_window` to FA,
    while the mask for eager/sdpa/flex is `hfOverlay config.sliding_window`. -/

/-- FA path still implements the original T5Gemma semantics. -/
theorem t5gemma_fa_eq_orig (W L q k : Int) (hW : 0 < W) (h : inRange L q k) :
    faBidir W L q k ↔ t5gemmaOrig W q k := by
  unfold faBidir t5gemmaOrig inRange at *; omega

/-- The eager/sdpa/flex mask on main differs from the original (and from FA) exactly on the two
    diagonals |q - k| = W. -/
theorem t5gemma_overlay_diff_iff (W q k : Int) (hW : 0 ≤ W) :
    (hfOverlay W q k ∧ ¬ t5gemmaOrig W q k) ↔ (q - k = W ∨ k - q = W) := by
  unfold hfOverlay t5gemmaOrig; omega

/-- ...and those diagonals exist as soon as the encoder input is longer than W tokens. -/
theorem t5gemma_mismatch_exists (W L : Int) (hW : 0 < W) (hL : W < L) :
    ∃ q k, inRange L q k ∧ hfOverlay W q k ∧ ¬ faBidir W L q k :=
  ⟨W, 0, by unfold inRange; omega, by unfold hfOverlay; omega, by unfold faBidir; omega⟩

/-- Short inputs (L <= W) are unaffected: every pair is visible on every path. -/
theorem t5gemma_short_inputs_ok (W L q k : Int) (hL : L ≤ W) (h : inRange L q k) :
    hfOverlay W q k ↔ t5gemmaOrig W q k := by
  unfold hfOverlay t5gemmaOrig inRange at *; omega

/-! ### Models that pass `config.sliding_window + 1` to FA (ModernBERT, EmbeddingGemma2, UltraBERT,
    NeoMME, OpenAI privacy filter): FA and the eager/sdpa mask agree for every length. -/
theorem plus_one_convention_consistent (W L q k : Int) (hW : 0 ≤ W) (h : inRange L q k) :
    faBidir (W + 1) L q k ↔ hfOverlay W q k := by
  unfold faBidir hfOverlay inRange at *; omega

/-! ### Gemma3 bidirectional (EmbeddingGemma): matches the Gemma reference and its FA path. -/
theorem gemma3_bidir_eq_orig (W q k : Int) (hW : 0 < W) :
    gemma3Bidir W q k ↔ t5gemmaOrig W q k := by
  unfold gemma3Bidir t5gemmaOrig; omega

theorem gemma3_bidir_fa_consistent (W L q k : Int) (hW : 0 < W) (h : inRange L q k) :
    faBidir W L q k ↔ gemma3Bidir W q k := by
  rw [gemma3_bidir_eq_orig W q k hW]; exact t5gemma_fa_eq_orig W L q k hW h

/-! ### sdpa skip (`_ignore_bidirectional_mask_sdpa`): with no padding and kv_length < W the mask is
    dropped (full attention). Sound for the overlay convention. -/
theorem skip_sound_overlay (W L q k : Int) (hL : L < W) (h : inRange L q k) : hfOverlay W q k := by
  unfold hfOverlay inRange at *; omega

/-! Concrete counterexample (W = 4, L = 6): cells where eager/sdpa (main) and FA disagree. -/
#eval ((List.range 6).map Int.ofNat).flatMap (fun q => ((List.range 6).map Int.ofNat).filterMap (fun k =>
  if decide (hfOverlay 4 q k) != decide (faBidir 4 6 q k) then some (q, k) else none))

/-! ### DiffusionGemma decoder (models/diffusion_gemma/modeling_diffusion_gemma.py)
    Sliding layers: keys = [W-1 cached encoder keys ; C canvas keys], queries = the C canvas tokens.
    Intended (create_diffusion_decoder_attention_mask + tests): every canvas query sees all W-1+C keys.
    FA path: `sliding_window=W`, is_causal=False → window_size=(W-1, W-1) when key_length > W, and
    flash-attn aligns queries bottom-right: query i sits at key index i + klen - qlen. -/
def faBR (sw qlen klen i j : Int) : Prop :=
  klen > sw → (i + klen - qlen - (sw - 1) ≤ j ∧ j ≤ i + klen - qlen + (sw - 1))

instance (sw qlen klen i j : Int) : Decidable (faBR sw qlen klen i j) := by unfold faBR; infer_instance

/-- With a full sliding cache (klen = W-1+C, C ≥ 2), canvas query i sees key j iff j ≥ i:
    it silently drops the i oldest cached keys (up to C-1 of them). -/
theorem diffusion_fa_visible_iff (W C i j : Int) (hW : 1 ≤ W) (hC : 2 ≤ C) (hCW : C ≤ W)
    (hi : 0 ≤ i ∧ i < C) (hj : 0 ≤ j ∧ j < W - 1 + C) :
    faBR W C (W - 1 + C) i j ↔ i ≤ j := by
  unfold faBR; omega

theorem diffusion_fa_ne_intended (W C : Int) (hW : 1 ≤ W) (hC : 2 ≤ C) :
    ¬ faBR W C (W - 1 + C) (C - 1) 0 := by
  unfold faBR; omega

/-- Short contexts are fine: if cached + canvas ≤ W the window is not applied at all. -/
theorem diffusion_fa_short_ok (W C c i j : Int) (h : c + C ≤ W) : faBR W C (c + C) i j := by
  unfold faBR; omega

-- published config: W = 1024, C = 256 → number of cached keys missed by each canvas query
#eval ((List.range 256).map Int.ofNat).map (fun i =>
  ((List.range 1279).map Int.ofNat).countP (fun j => !decide (faBR 1024 256 1279 i j))) |>.take 6
