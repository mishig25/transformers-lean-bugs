/-
DeepSeek-OCR-2: processor placeholder count vs. model features.

Processor (processing_deepseek_ocr2.py `replace_image_token`):
  ceil(size.height / patch / ds)^2 + ceil(tile / patch / ds)^2 * n_local + 1
  -- always uses `size.height` (1024) for the global view.
Image processor (image_processing_deepseek_ocr2.py `_preprocess`):
  global_target_size = size.height if crop_to_patches else tile_size     (1024 vs 768)
  n_local = 0 unless crop_to_patches ∧ max(h,w) > tile_size
Model (modeling_deepseek_ocr2.py): per view, SAM (patch 16) + 4x neck -> (side/64)^2 patches;
  queries = 144 if patches ≤ 144 else 256; features = local*144 + global_queries + 1 (view_separator).
-/

def ceilDiv (a b : Nat) : Nat := (a + b - 1) / b

def procTokens (sizeH tile patch ds nLocal : Nat) : Nat :=
  let g := ceilDiv sizeH (patch * ds)
  let l := ceilDiv tile (patch * ds)
  g * g + l * l * nLocal + 1

def modelQueries (side : Nat) : Nat :=
  let p := (side / 64) * (side / 64)
  if p ≤ 144 then 144 else 256

def nLocal (crop : Bool) (h w tile nTiles : Nat) : Nat :=
  if crop && decide (max h w > tile) then nTiles else 0

def modelFeatures (crop : Bool) (sizeH tile : Nat) (nl : Nat) : Nat :=
  let globalSide := if crop then sizeH else tile
  modelQueries tile * nl + modelQueries globalSide + 1

-- default checkpoint config: size 1024, tile 768, patch 16, downsample 4
theorem crop_true_agrees (nl : Nat) :
    procTokens 1024 768 16 4 nl = modelFeatures true 1024 768 nl := by
  simp [procTokens, modelFeatures, modelQueries, ceilDiv]; omega

theorem crop_false_disagrees :
    procTokens 1024 768 16 4 0 ≠ modelFeatures false 1024 768 0 := by
  decide

#eval (procTokens 1024 768 16 4 0, modelFeatures false 1024 768 0)  -- (257, 145)
