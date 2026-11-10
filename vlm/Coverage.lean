/-
Families where processor placeholder count = model feature count (proved, under the stated hypotheses
that the corresponding resize code guarantees).
-/

/-! ## Shared lemma: grids that are multiples of a merge factor -/

theorem merge_count (a b m : Nat) (hm : 0 < m) :
    (a * m) * (b * m) / (m * m) = ((a * m) / m) * ((b * m) / m) := by
  rw [Nat.mul_div_cancel a hm, Nat.mul_div_cancel b hm]
  have : (a * m) * (b * m) = (a * b) * (m * m) := by
    simp only [Nat.mul_comm, Nat.mul_left_comm]
  rw [this, Nat.mul_div_cancel _ (Nat.mul_pos hm hm)]

/-! ## Qwen2-VL / Qwen2.5-VL / Qwen3-VL / GLM-4V / PaddleOCR-VL / MiniMax-M3 (smart_resize)
Every branch of `smart_resize` returns `k * factor` with `factor = patch * merge`
(`round(h/f)*f`, `max(f, floor(..)*f)`, `ceil(..)*f`). Processor: grid_t*grid_h*grid_w // merge²;
model: `split_sizes = grid_thw.prod(-1) // merge²`, and the vision merger needs grid_h, grid_w % merge = 0. -/

def qwenProc (t kh kw p m : Nat) : Nat := t * ((kh * (p * m)) / p) * ((kw * (p * m)) / p) / (m * m)
def qwenModel (t kh kw p m : Nat) : Nat := t * (((kh * (p * m)) / p) / m) * (((kw * (p * m)) / p) / m)

theorem qwen_grid_divisible (kh p m : Nat) (hp : 0 < p) : ((kh * (p * m)) / p) % m = 0 := by
  have : kh * (p * m) = (kh * m) * p := by simp only [Nat.mul_comm, Nat.mul_left_comm]
  rw [this, Nat.mul_div_cancel _ hp]; simp

theorem qwen_agree (t kh kw p m : Nat) (hp : 0 < p) (hm : 0 < m) :
    qwenProc t kh kw p m = qwenModel t kh kw p m := by
  unfold qwenProc qwenModel
  have e1 : kh * (p * m) / p = kh * m := by
    rw [show kh * (p * m) = (kh * m) * p by simp only [Nat.mul_comm, Nat.mul_left_comm]]
    exact Nat.mul_div_cancel _ hp
  have e2 : kw * (p * m) / p = kw * m := by
    rw [show kw * (p * m) = (kw * m) * p by simp only [Nat.mul_comm, Nat.mul_left_comm]]
    exact Nat.mul_div_cancel _ hp
  rw [e1, e2, Nat.mul_div_cancel _ hm, Nat.mul_div_cancel _ hm]
  rw [show t * (kh * m) * (kw * m) = (t * kh * kw) * (m * m) by
        simp only [Nat.mul_comm, Nat.mul_left_comm]]
  rw [Nat.mul_div_cancel _ (Nat.mul_pos hm hm)]

/-! ## Pixtral / Mistral3 / LightOn-OCR
`get_resize_output_image_size` returns `((h-1)//(p*s)+1) * (p*s)`; processor: H // (p*s) per row;
`Mistral3PatchMerger`: (H // p) unfolded with stride s -> (H//p)//s. -/

def pixtralResized (h p s : Nat) : Nat := ((h - 1) / (p * s) + 1) * (p * s)

theorem pixtral_agree (h p s : Nat) :
    pixtralResized h p s / (p * s) = (pixtralResized h p s / p) / s := by
  unfold pixtralResized
  rw [Nat.div_div_eq_div_mul]

/-! ## LFM2-VL: smart_resize gives multiples of encoder_patch*downsample (16*2). Processor uses
`ceil((H//16)/2)`, model `pixel_unshuffle` reshape needs (H//16) % 2 = 0 and yields (H//16)//2. -/

theorem lfm2_agree (k : Nat) :
    ((k * 32) / 16 + 2 - 1) / 2 = ((k * 32) / 16) / 2 ∧ ((k * 32) / 16) % 2 = 0 := by
  constructor <;> omega

/-! ## Gemma4: target side = k*(pool*patch); processor `patches // pool²`, pooler emits
(W_p/pool)*(H_p/pool) distinct kernel indices. -/

theorem gemma4_agree (a b k p : Nat) (hk : 0 < k) (hp : 0 < p) :
    ((a * (k * p)) / p) * ((b * (k * p)) / p) / (k * k)
      = (((a * (k * p)) / p) / k) * (((b * (k * p)) / p) / k) := by
  have e1 : a * (k * p) / p = a * k := by
    rw [show a * (k * p) = (a * k) * p by simp only [Nat.mul_comm, Nat.mul_left_comm]]
    exact Nat.mul_div_cancel _ hp
  have e2 : b * (k * p) / p = b * k := by
    rw [show b * (k * p) = (b * k) * p by simp only [Nat.mul_comm, Nat.mul_left_comm]]
    exact Nat.mul_div_cancel _ hp
  rw [e1, e2]; exact merge_count a b k hk

/-! ## Phi-4-multimodal: image processor counts `256 + 1 + sum(ds_mask) + sum(ds_mask[:,0]) + 16`
where the HD mask is a row-mask × column-mask product (padding rows/cols zeroed), so
sum(ds_mask) = uh*uw and sum(ds_mask[:,0]) = uh. Model: global 16*(16+1), separator 1,
sub-image uh*(uw+1). -/

theorem phi4_agree (uh uw : Nat) :
    256 + 1 + uh * uw + uh + 16 = uh * (uw + 1) + 1 + 16 * (16 + 1) := by
  rw [Nat.mul_succ]; omega

/-- the `[0::2]` downsampling of a length-R valid prefix keeps ceil(R/2) entries -/
def evensBelow (R : Nat) : Nat := ((List.range R).filter (fun i => i % 2 == 0)).length
example : (List.range 40).all (fun R => evensBelow R == (R + 1) / 2) = true := by decide
#eval (List.range 2000).all (fun R => evensBelow R == (R + 1) / 2)

/-! ## LLaVA-NeXT / LLaVA-NeXT-Video / LLaVA-OneVision / Granite-vision: `unpad_image` vs `_get_unpadded_features`.
Both compute the same `new_h` with the identical float expression; the model then slices
`tensor[:, pad : cur - pad]` (Python slice length = max(0, cur - 2*pad)) while the processor does
`cur - 2*pad`. They agree whenever `new_h ≤ cur`, which holds in the branch taken. -/

def sliceLen (start stop : Int) : Int := max 0 (stop - start)

theorem llava_unpad_agree (cur nh : Int) (h0 : 0 ≤ nh) (h1 : nh ≤ cur) :
    sliceLen ((cur - nh) / 2) (cur - (cur - nh) / 2) = cur - 2 * ((cur - nh) / 2) := by
  unfold sliceLen; omega

/-- exact-arithmetic witness that the branch condition implies `new_h ≤ cur` (floor and round-half-up). -/
theorem llava_branch_bound (oh ow cw ch : Nat) (how : 0 < ow) (hcond : oh * cw < ch * ow) :
    oh * cw / ow ≤ ch ∧ (2 * oh * cw + ow) / (2 * ow) ≤ ch := by
  constructor
  · exact Nat.le_of_lt (by
      have := (Nat.div_lt_iff_lt_mul how).2 hcond; exact this)
  · have h2 : 2 * oh * cw + ow < (ch + 1) * (2 * ow) := by
      have : 2 * oh * cw = 2 * (oh * cw) := by simp [Nat.mul_assoc]
      rw [this, Nat.add_mul, Nat.one_mul]
      have : ch * (2 * ow) = 2 * (ch * ow) := by rw [Nat.mul_left_comm]
      rw [this]; omega
    have := (Nat.div_lt_iff_lt_mul (by omega : 0 < 2 * ow)).2 h2
    omega

/-! ## Nemotron-H-Omni `get_patch_grid_size`: the remainder fix always returns a multiple of merge_size,
so `pixel_shuffle` (needs even sides) and `grid_h*grid_w // merge²` agree. -/

def remFix (g m : Nat) (fits : Bool) : Nat :=
  if g % m = 0 then g else if fits then g + (m - g % m) else max m (g - g % m)

theorem nemotron_fix_multiple (g m : Nat) (hm : 0 < m) (fits : Bool) : remFix g m fits % m = 0 := by
  unfold remFix
  have hdm := Nat.div_add_mod g m
  have hlt := Nat.mod_lt g hm
  split
  · assumption
  · split
    · -- g + (m - g%m) = (g/m + 1) * m
      have : g + (m - g % m) = (g / m + 1) * m := by
        rw [Nat.add_mul, Nat.one_mul]
        have : m * (g / m) = g / m * m := Nat.mul_comm _ _
        omega
      rw [this]; simp
    · have e : g - g % m = (g / m) * m := by
        have : m * (g / m) = g / m * m := Nat.mul_comm _ _
        omega
      rw [e]
      by_cases hc : m ≤ g / m * m
      · rw [Nat.max_eq_right hc]; simp
      · rw [Nat.max_eq_left (by omega)]; simp
