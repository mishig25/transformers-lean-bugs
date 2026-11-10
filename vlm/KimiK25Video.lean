/-
Kimi-K2.5 video path: processor placeholder count vs. model features.

Processor side (video_processing_kimi_k25.py `_preprocess`, processing_kimi_k25.py `replace_video_token`):
  * a video with F sampled frames is cut into chunks of `T = temporal_patch_size` (=4) frames,
    the last chunk holding F - T*(ceil(F/T)-1) frames;
  * each chunk gets `video_grid_thw = [t_c, gh, gw]` with t_c = #frames in the chunk;
  * placeholder tokens per chunk = gh*gw / (kh*kw)    (one frame's worth).

Model side (modeling_kimi_k25.py):
  * `get_vision_temporal_merge_index` builds, per chunk, a block of shape (gh/kh * gw/kw, t_c, kh*kw)
    and `torch.cat`s them on dim 0  -> requires every chunk to have the same t_c;
  * pooled output rows per chunk = (gh/kh)*(gw/kw)  (mean over the t_c axis);
  * `get_image_features` then splits with `split_sizes = t_c*gh*gw // (kh*kw)` per chunk.
-/

/-- frame counts of the chunks of a video with `F` frames and temporal chunk size `T` -/
def chunkFrames (F T : Nat) : List Nat :=
  (List.range ((F + T - 1) / T)).map (fun c => min T (F - c * T))

/-- processor: placeholder tokens for the whole video -/
def procTokens (F T gh gw k : Nat) : Nat :=
  ((chunkFrames F T).map (fun _ => gh * gw / (k * k))).foldl (· + ·) 0

/-- model: number of pooled feature rows actually produced -/
def modelRows (F T gh gw k : Nat) : Nat :=
  ((chunkFrames F T).map (fun _ => (gh / k) * (gw / k))).foldl (· + ·) 0

/-- model: sum of `split_sizes` used in `torch.split` in `get_image_features` -/
def modelSplitSum (F T gh gw k : Nat) : Nat :=
  ((chunkFrames F T).map (fun t => t * gh * gw / (k * k))).foldl (· + ·) 0

/-- `torch.cat` in `get_vision_temporal_merge_index` needs all chunks to share t_c -/
def catOk (F T : Nat) : Bool :=
  match chunkFrames F T with
  | [] => true
  | t :: ts => ts.all (· == t)

-- Processor count and real model rows agree whenever the grid is divisible by k (always true:
-- navit_resize pads to multiples of merge_size*patch_size).
theorem perChunk_proc_eq_rows (a b k : Nat) (hk : 0 < k) :
    (a * k) * (b * k) / (k * k) = ((a * k) / k) * ((b * k) / k) := by
  have h1 : (a * k) / k = a := Nat.mul_div_cancel a hk
  have h2 : (b * k) / k = b := Nat.mul_div_cancel b hk
  rw [h1, h2]
  have : (a * k) * (b * k) = (a * b) * (k * k) := by
    simp only [Nat.mul_comm, Nat.mul_left_comm]
  rw [this, Nat.mul_div_cancel _ (Nat.mul_pos hk hk)]

-- But the split sizes are t_c times too large: a single chunk with t frames.
theorem split_overcounts (t a b : Nat) :
    t * (a * 2) * (b * 2) / (2 * 2) = t * (a * b) := by
  have : t * (a * 2) * (b * 2) = t * (a * b) * 4 := by
    simp only [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
  rw [this]; omega

-- Bounded search: frames F in 1..64, grid gh=gw=36 (a 504x504 frame), k=2, T=4
#eval (List.range 64).filterMap (fun i =>
  let F := i + 1
  let p := procTokens F 4 36 36 2
  let r := modelRows F 4 36 36 2
  let s := modelSplitSum F 4 36 36 2
  if p != s || !catOk F 4 then some (F, p, r, s, catOk F 4) else none) |>.take 12

-- The only frame count for which split sizes are right is F = 1 (single-frame "video").
#eval (List.range 400).filter (fun i =>
  let F := i + 1
  modelSplitSum F 4 36 36 2 == modelRows F 4 36 36 2)

-- catOk fails exactly when F > T and F % T ≠ 0
#eval (List.range 40).filter (fun i => !catOk (i + 1) 4) |>.map (· + 1)

example : modelSplitSum 8 4 36 36 2 = 4 * modelRows 8 4 36 36 2 := by decide
example : catOk 6 4 = false := by decide
